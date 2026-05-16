"""Train a Transolver surrogate on TandemFoilSet.

Four validation tracks with one pinned test track each:
  val_single_in_dist      / test_single_in_dist      — in-distribution sanity
  val_geom_camber_rc      / test_geom_camber_rc      — unseen front foil (raceCar)
  val_geom_camber_cruise  / test_geom_camber_cruise  — unseen front foil (cruise)
  val_re_rand             / test_re_rand             — stratified Re holdout

Primary ranking metric is ``avg/mae_surf_p`` — the equal-weight mean surface
pressure MAE across the four splits, computed in the original (denormalized)
target space. Train/val/test MAE all flow through ``data.scoring`` so the
numbers are produced identically.

Usage:
  python train.py [--debug] [--epochs 50] [--agent <name>] [--experiment_name <name>]
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)

# ---------------------------------------------------------------------------
# Transolver model
# ---------------------------------------------------------------------------

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 layer_scale_init: float = 0.0):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        self.layer_scale_init = layer_scale_init
        if layer_scale_init > 0:
            self.gamma_attn = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
            self.gamma_mlp = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
        else:
            self.register_parameter("gamma_attn", None)
            self.register_parameter("gamma_mlp", None)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        attn_out = self.attn(self.ln_1(fx))
        if self.gamma_attn is not None:
            attn_out = attn_out * self.gamma_attn
        fx = fx + attn_out
        mlp_out = self.mlp(self.ln_2(fx))
        if self.gamma_mlp is not None:
            mlp_out = mlp_out * self.gamma_mlp
        fx = fx + mlp_out
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class FourierPosFeatures(nn.Module):
    """Concat sin/cos of 2D position at multiple frequencies.

    Output dims for input pos of shape [B, N, 2] are 2 + 4 * n_freqs:
    the raw position plus sin/cos at K frequencies in each of 2 spatial axes.
    Frequencies are geometric: [base^0, base^1, ..., base^(K-1)] scaled by pi.
    """

    def __init__(self, n_freqs: int = 6, base: float = 2.0):
        super().__init__()
        freqs = base ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs.view(1, 1, 1, -1))
        self.n_freqs = n_freqs
        self.base = base

    def forward(self, pos):  # pos: [B, N, 2]  (normalized x, z)
        xz = pos.unsqueeze(-1) * self.freqs * math.pi  # [B, N, 2, K]
        sin = xz.sin().flatten(-2)  # [B, N, 2K]
        cos = xz.cos().flatten(-2)  # [B, N, 2K]
        return torch.cat([pos, sin, cos], dim=-1)  # [B, N, 2 + 4K]


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 n_freqs: int = 0, fourier_base: float = 2.0,
                 layer_scale_init: float = 0.0,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.n_freqs = n_freqs
        self.fourier = FourierPosFeatures(n_freqs=n_freqs, base=fourier_base) if n_freqs > 0 else None

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.layer_scale_init = layer_scale_init
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                layer_scale_init=layer_scale_init,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        if self.fourier is not None:
            pos = x[..., :2]
            feat = x[..., 2:]
            pos_enc = self.fourier(pos)
            x = torch.cat([pos_enc, feat], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, huber_delta: float) -> dict[str, float]:
    """Evaluate a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]

            # Drop samples whose GT has any non-finite value (e.g. corrupted CFD output).
            # Without this guard, Inf in y_norm makes sq_err Inf, and Inf*mask=NaN even
            # where mask is False — propagating NaN into the loss/metric accumulators.
            sample_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            safe = mask & sample_finite.unsqueeze(-1)
            y_norm = torch.where(
                sample_finite.view(-1, 1, 1).expand_as(y_norm),
                y_norm, torch.zeros_like(y_norm),
            )

            sq_err = F.huber_loss(pred, y_norm, reduction="none", delta=huber_delta)
            vol_mask = safe & ~is_surface
            surf_mask = safe & is_surface
            vol_loss_sum += (
                (sq_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            # Pass `safe` (which masks out the entire bad sample) so scoring sees the
            # bad sample as padding; also replace its y with zeros so |pred-y| stays finite.
            y_for_scoring = torch.where(
                sample_finite.view(-1, 1, 1).expand_as(y),
                y, torch.zeros_like(y),
            )
            ds, dv = accumulate_batch(pred_orig, y_for_scoring, is_surface, safe, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def _sanitize_path_token(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in s)
    return out.strip("-_.") or "experiment"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_metrics_jsonl(metrics_path: Path, record: dict) -> None:
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def gamma_stats(model: nn.Module) -> dict[str, float]:
    """Return per-block mean and max |gamma| for LayerScale parameters.

    Empty dict if the model has no LayerScale (gamma_attn/gamma_mlp are None).
    """
    stats: dict[str, float] = {}
    for i, block in enumerate(model.blocks):
        if block.gamma_attn is not None:
            g = block.gamma_attn.detach().abs()
            stats[f"gamma_attn/block{i}/mean"] = g.mean().item()
            stats[f"gamma_attn/block{i}/max"] = g.max().item()
        if block.gamma_mlp is not None:
            g = block.gamma_mlp.detach().abs()
            stats[f"gamma_mlp/block{i}/mean"] = g.mean().item()
            stats[f"gamma_mlp/block{i}/max"] = g.max().item()
    return stats


def param_norms(model: nn.Module) -> dict[str, float]:
    """L2 norms of LayerScale gamma vectors and a representative MLP weight matrix.

    Tracks how weight_decay shapes parameter magnitudes — per-block ‖gamma_attn‖
    and ‖gamma_mlp‖ (no-decay group with LayerScale on), plus the per-block
    MLP linear_pre weight (decay group) so we can see WD's effect on a normal
    parameter for comparison.
    """
    norms: dict[str, float] = {}
    for i, block in enumerate(model.blocks):
        if block.gamma_attn is not None:
            norms[f"param_norm/gamma_attn/block{i}"] = block.gamma_attn.detach().norm().item()
        if block.gamma_mlp is not None:
            norms[f"param_norm/gamma_mlp/block{i}"] = block.gamma_mlp.detach().norm().item()
        if hasattr(block, "mlp") and hasattr(block.mlp, "linear_pre"):
            w = block.mlp.linear_pre[0].weight
            norms[f"param_norm/mlp_linear_pre_w/block{i}"] = w.detach().norm().item()
    return norms


def write_experiment_summary(
    model_path: Path,
    model_dir: Path,
    cfg: "Config",
    best_metrics: dict,
    best_avg_surf_p: float,
    test_metrics: dict | None,
    test_avg: dict | None,
    n_params: int,
    model_config: dict,
) -> None:
    """Write a local summary next to the best checkpoint."""
    summary: dict = {
        "agent": cfg.agent,
        "experiment_name": cfg.experiment_name,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "model_config": model_config,
        "checkpoint": str(model_path),
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "epochs_configured": cfg.epochs,
        "huber_delta": cfg.huber_delta,
        "n_freqs": cfg.n_freqs,
        "fourier_base": cfg.fourier_base,
        "lr_t_max": cfg.lr_t_max,
        "layer_scale_init": cfg.layer_scale_init,
        "ema_decay": cfg.ema_decay,
    }
    if "raw_val_avg/mae_surf_p" in best_metrics:
        summary["best_raw_val_avg/mae_surf_p"] = best_metrics["raw_val_avg/mae_surf_p"]

    for split_name, m in best_metrics["per_split"].items():
        for k, v in m.items():
            summary[f"best_val/{split_name}/{k}"] = v
    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        summary["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    summary[f"test/{split_name}/{k}"] = v

    summary_path = model_dir / "metrics.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"\nSaved experiment summary to {summary_path}")


def print_split_metrics(split_name: str, m: dict[str, float]) -> None:
    print(
        f"    {split_name:<26s} "
        f"loss={m['loss']:.4f}  "
        f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
        f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_MIN = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    huber_delta: float = 1.0  # threshold (in normalized units) where Huber switches from quadratic to linear; matches MSE in the limit delta -> inf
    n_freqs: int = 0  # 0 disables Fourier positional features (raw x,z); >0 enables sin/cos at base^k * pi
    fourier_base: float = 2.0
    lr_t_max: int | None = None  # override cosine T_max; defaults to MAX_EPOCHS if None
    layer_scale_init: float = 0.0  # CaiT-style per-channel residual gain; 0.0 disables (baseline behavior)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    grad_clip_max_norm: float | None = None  # None disables clipping
    ema_decay: float | None = None  # None disables EMA; typical values 0.999 / 0.9995


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2 + 4 * cfg.n_freqs if cfg.n_freqs > 0 else 2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    n_freqs=cfg.n_freqs,
    fourier_base=cfg.fourier_base,
    layer_scale_init=cfg.layer_scale_init,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

if cfg.layer_scale_init > 0:
    # Exclude LayerScale gamma params from weight decay (standard ViT recipe).
    # Otherwise AdamW would slowly pull tiny gamma values back toward zero, defeating
    # the point of LayerScale at small init.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("gamma_attn") or name.endswith("gamma_mlp"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    print(f"Optimizer: AdamW with {len(no_decay_params)} no-decay gamma params, "
          f"{len(decay_params)} decay params")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

ema_model: AveragedModel | None = None
if cfg.ema_decay is not None:
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
    ema_model.to(device)
    print(f"EMA enabled (decay={cfg.ema_decay})")
cosine_t_max = cfg.lr_t_max if cfg.lr_t_max is not None else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)

experiment_label = cfg.experiment_name or cfg.agent or "tandemfoil"
experiment_stamp = time.strftime("%Y%m%d-%H%M%S")
model_dir = Path("models") / f"model-{_sanitize_path_token(experiment_label)}-{experiment_stamp}"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
metrics_jsonl_path = model_dir / "metrics.jsonl"
with open(model_dir / "config.yaml", "w") as f:
    yaml.safe_dump({
        **asdict(cfg),
        "model_config": model_config,
        "n_params": n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    }, f, sort_keys=True)

best_avg_surf_p = float("inf")
best_metrics: dict = {}
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    n_clipped = 0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        sq_err = F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip_max_norm is not None:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
            if gnorm.item() > cfg.grad_clip_max_norm:
                n_clipped += 1
        else:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        gnorm_val = gnorm.item()
        grad_norm_sum += gnorm_val
        if gnorm_val > grad_norm_max:
            grad_norm_max = gnorm_val
        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    eval_model = model
    if ema_model is not None:
        ema_model.eval()
        eval_model = ema_model.module
    split_metrics = {
        name: evaluate_split(eval_model, loader, stats, cfg.surf_weight, device, cfg.huber_delta)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]

    # Also evaluate the raw (non-EMA) model so we can report the EMA contribution.
    raw_split_metrics = None
    raw_avg_surf_p = None
    if ema_model is not None:
        raw_split_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.huber_delta)
            for name, loader in val_loaders.items()
        }
        raw_avg_surf_p = aggregate_splits(raw_split_metrics)["avg/mae_surf_p"]

    dt = time.time() - t0

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        if ema_model is not None:
            best_metrics["raw_val_avg/mae_surf_p"] = raw_avg_surf_p
            torch.save(ema_model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    grad_norm_mean = grad_norm_sum / max(n_batches, 1)
    clip_frac = n_clipped / max(n_batches, 1) if cfg.grad_clip_max_norm is not None else 0.0
    epoch_record = {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "current_lr": current_lr,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/grad_norm_mean": grad_norm_mean,
        "train/grad_norm_max": grad_norm_max,
        "train/clip_frac": clip_frac,
        "grad_clip_max_norm": cfg.grad_clip_max_norm,
        "ema_decay": cfg.ema_decay,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    }
    gstats = gamma_stats(model)
    if gstats:
        epoch_record.update(gstats)
    epoch_record.update(param_norms(model))
    if ema_model is not None:
        epoch_record["raw_val_avg/mae_surf_p"] = raw_avg_surf_p
        epoch_record["raw_val_splits"] = raw_split_metrics
    append_metrics_jsonl(metrics_jsonl_path, epoch_record)
    ema_str = ""
    if ema_model is not None:
        ema_str = f"  raw_val={raw_avg_surf_p:.4f}"
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}{ema_str}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.huber_delta)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        test_record = {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "test_avg": test_avg,
            "test_splits": test_metrics,
        }
        best_ckpt_gstats = gamma_stats(model)
        if best_ckpt_gstats:
            test_record["best_ckpt_gamma_stats"] = best_ckpt_gstats
        append_metrics_jsonl(metrics_jsonl_path, test_record)

    write_experiment_summary(
        model_path=model_path,
        model_dir=model_dir,
        cfg=cfg,
        best_metrics=best_metrics,
        best_avg_surf_p=best_avg_surf_p,
        test_metrics=test_metrics,
        test_avg=test_avg,
        n_params=n_params,
        model_config=model_config,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")
