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

import copy
import json
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


class GEGLU(nn.Module):
    """Gated Linear Unit (Shazeer 2020): out = value * gate_act(gate).

    Single Linear(dim_in -> 2*dim_out) chunked into (value, gate).
    gate_act in {"gelu" -> GEGLU, "silu" -> SwiGLU}.
    """

    def __init__(self, dim_in, dim_out, gate_act="gelu"):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out)
        self.gate_act = nn.GELU() if gate_act == "gelu" else nn.SiLU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gate_act(gate)


class GatedMLP(nn.Module):
    """FFN with gated activation: GEGLU(n_input -> n_hidden) -> Linear(n_hidden -> n_output).

    Drop-in replacement for MLP with n_layers=0. The GEGLU gate IS the
    nonlinearity, so no additional gated block is stacked.
    """

    def __init__(self, n_input, n_hidden, n_output, n_layers=0, gate_act="gelu", res=True):
        super().__init__()
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = GEGLU(n_input, n_hidden, gate_act=gate_act)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [GEGLU(n_hidden, n_hidden, gate_act=gate_act) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class ConditionMLP(nn.Module):
    """Produces (gamma, beta) FiLM parameters for a hidden_dim layer.

    Identity at init: final linear is zero-init except gamma bias = 1, so the
    block computes 1*fx + 0 = fx until training shapes the conditioning.
    """

    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[:hidden_dim] = 1.0

    def forward(self, cond):
        out = self.net(cond)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


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
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32, cond_dim=0,
                 ffn_act="gelu"):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if ffn_act == "geglu":
            self.mlp = GatedMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                                n_layers=0, gate_act="gelu", res=False)
        elif ffn_act == "swiglu":
            self.mlp = GatedMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                                n_layers=0, gate_act="silu", res=False)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.film = ConditionMLP(cond_dim, hidden_dim)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, cond=None):
        fx = self.attn(self.ln_1(fx)) + fx
        if self.cond_dim > 0 and cond is not None:
            gamma, beta = self.film(cond)
            fx = gamma.unsqueeze(1) * fx + beta.unsqueeze(1)
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False, cond_dim=0,
                 ffn_act="gelu",
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.cond_dim = cond_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                cond_dim=cond_dim, ffn_act=ffn_act,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)
        # Re-apply FiLM identity init since apply() above overwrites the
        # zero-init final layer of every ConditionMLP via trunc_normal_.
        for block in self.blocks:
            if hasattr(block, "film"):
                nn.init.zeros_(block.film.net[-1].weight)
                nn.init.zeros_(block.film.net[-1].bias)
                block.film.net[-1].bias.data[:n_hidden] = 1.0

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
        # Condition values (Re, AoA, NACA, gap, stagger) are sample-level
        # constants stored per-node; node 0 is always real (padding appended
        # at the end by pad_collate) so we read the global condition from it.
        cond = x[:, 0, 13:] if self.cond_dim > 0 else None
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, cond=cond)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device) -> dict[str, float]:
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

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
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
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
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
    test_metrics_raw: dict | None = None,
    test_avg_raw: dict | None = None,
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
        "ema_decay": cfg.ema_decay,
    }
    if "val_avg/mae_surf_p_raw" in best_metrics:
        summary["best_val_avg/mae_surf_p_raw"] = best_metrics["val_avg/mae_surf_p_raw"]

    for split_name, m in best_metrics["per_split"].items():
        for k, v in m.items():
            summary[f"best_val/{split_name}/{k}"] = v
    for split_name, m in best_metrics.get("per_split_raw", {}).items():
        for k, v in m.items():
            summary[f"best_val_raw/{split_name}/{k}"] = v
    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        summary["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    summary[f"test/{split_name}/{k}"] = v
    if test_avg_raw is not None and "avg/mae_surf_p" in test_avg_raw:
        summary["test_avg_raw/mae_surf_p"] = test_avg_raw["avg/mae_surf_p"]
        if test_metrics_raw is not None:
            for split_name, m in test_metrics_raw.items():
                for k, v in m.items():
                    summary[f"test_raw/{split_name}/{k}"] = v

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
    huber_delta: float = 1.0   # Huber loss transition threshold (normalized space)
    huber_delta_vel: float = 0.5  # Huber delta for Ux, Uy channels (per-channel)
    huber_delta_p: float = 0.25   # Huber delta for pressure channel p (per-channel)
    cond_dim: int = 11         # FiLM conditioning dim; 0 disables FiLM
    clip_grad_norm: float = 0.0  # Gradient clip max_norm; 0 disables
    ffn_act: str = "gelu"   # FFN activation: 'gelu' (default MLP), 'geglu', 'swiglu'
    eta_min: float = 0.0   # CosineAnnealingLR floor; 0 = anneal to zero (default)
    n_head: int = 4   # Transolver attention heads; head_dim = n_hidden // n_head
    ema_decay: float = 0.0  # EMA decay for weight averaging; 0 disables
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation


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
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=cfg.n_head,
    slice_num=64,
    mlp_ratio=2,
    cond_dim=cfg.cond_dim,  # log(Re), AoA1, NACA1(3), AoA2, NACA2(3), gap, stagger = 11; 0 disables FiLM
    ffn_act=cfg.ffn_act,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

ema_model = None
if cfg.ema_decay > 0:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    print(f"EMA enabled: decay={cfg.ema_decay}")


def _ema_update(ema_m, m, decay):
    with torch.no_grad():
        for ep, p in zip(ema_m.parameters(), m.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for eb, b in zip(ema_m.buffers(), m.buffers()):
            eb.data.copy_(b.data)


def _ema_l2_distance(ema_m, m) -> float:
    sq_sum = 0.0
    with torch.no_grad():
        for ep, p in zip(ema_m.parameters(), m.parameters()):
            sq_sum += (ep.data - p.data).pow(2).sum().item()
    return sq_sum ** 0.5

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15, eta_min=cfg.eta_min
)

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
    epoch_grad_norm_pre = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        abs_err = (pred - y_norm).abs()  # [B, N, 3]
        # Per-channel Huber delta: [Ux, Uy, p]
        delta = torch.tensor(
            [cfg.huber_delta_vel, cfg.huber_delta_vel, cfg.huber_delta_p],
            device=pred.device, dtype=pred.dtype,
        ).view(1, 1, 3)
        sq_err = torch.where(
            abs_err < delta,
            0.5 * abs_err ** 2,
            delta * (abs_err - 0.5 * delta),
        )

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.clip_grad_norm > 0:
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.clip_grad_norm
            )
            epoch_grad_norm_pre += float(pre_clip_norm)
        optimizer.step()

        if ema_model is not None:
            _ema_update(ema_model, model, cfg.ema_decay)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_grad_norm_pre /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics_raw = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
        for name, loader in val_loaders.items()
    }
    val_avg_raw = aggregate_splits(split_metrics_raw)
    avg_surf_p_raw = val_avg_raw["avg/mae_surf_p"]

    if ema_model is not None:
        split_metrics_ema = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device)
            for name, loader in val_loaders.items()
        }
        val_avg_ema = aggregate_splits(split_metrics_ema)
        avg_surf_p_ema = val_avg_ema["avg/mae_surf_p"]
        # Use EMA metrics as primary for checkpoint selection.
        split_metrics = split_metrics_ema
        avg_surf_p = avg_surf_p_ema
    else:
        split_metrics_ema = None
        avg_surf_p_ema = None
        split_metrics = split_metrics_raw
        avg_surf_p = avg_surf_p_raw

    dt = time.time() - t0

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
            "val_avg/mae_surf_p_raw": avg_surf_p_raw,
            "per_split_raw": split_metrics_raw,
        }
        # Save the model used for evaluation (EMA if enabled, else raw).
        save_model = ema_model if ema_model is not None else model
        torch.save(save_model.state_dict(), model_path)
        # When EMA is enabled, also snapshot the raw model at the same epoch
        # so we can run apples-to-apples test eval (raw vs EMA at best epoch).
        if ema_model is not None:
            raw_model_path = model_dir / "checkpoint_raw.pt"
            torch.save(model.state_dict(), raw_model_path)
        tag = " *"

    # Log EMA L2 distance to raw weights at epochs 1, 7, 13 (per PR request)
    ema_l2 = None
    if ema_model is not None and (epoch + 1) in (1, 7, 13):
        ema_l2 = _ema_l2_distance(ema_model, model)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    log_record = {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/grad_norm_pre_clip": epoch_grad_norm_pre,
        "clip_grad_norm": cfg.clip_grad_norm,
        "ema_decay": cfg.ema_decay,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_avg/mae_surf_p_raw": avg_surf_p_raw,
        "val_splits": split_metrics,
        "val_splits_raw": split_metrics_raw,
        "is_best": tag == " *",
    }
    if avg_surf_p_ema is not None:
        log_record["val_avg/mae_surf_p_ema"] = avg_surf_p_ema
        log_record["val_splits_ema"] = split_metrics_ema
    if ema_l2 is not None:
        log_record["ema_l2_distance_to_raw"] = ema_l2
    append_metrics_jsonl(metrics_jsonl_path, log_record)

    ema_extra = ""
    if avg_surf_p_ema is not None:
        ema_extra = f"  ema_surf_p={avg_surf_p_ema:.4f}"
        if ema_l2 is not None:
            ema_extra += f"  ema_l2={ema_l2:.3f}"
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p_raw:.4f}{ema_extra}{tag}"
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
    test_metrics_raw = None
    test_avg_raw = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])

        # When EMA is enabled, also evaluate the raw checkpoint (saved at the
        # same best epoch) so we can report raw-vs-EMA test metrics.
        if cfg.ema_decay > 0:
            raw_model_path = model_dir / "checkpoint_raw.pt"
            if raw_model_path.exists():
                print("\nEvaluating raw (non-EMA) model on test splits for comparison...")
                model.load_state_dict(
                    torch.load(raw_model_path, map_location=device, weights_only=True)
                )
                model.eval()
                test_metrics_raw = {
                    name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
                    for name, loader in test_loaders.items()
                }
                test_avg_raw = aggregate_splits(test_metrics_raw)
                print(f"\n  TEST (raw)  avg_surf_p={test_avg_raw['avg/mae_surf_p']:.4f}")
                for name in TEST_SPLIT_NAMES:
                    print_split_metrics(name, test_metrics_raw[name])

        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "test_avg": test_avg,
            "test_splits": test_metrics,
            "test_avg_raw": test_avg_raw,
            "test_splits_raw": test_metrics_raw,
        })

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
        test_metrics_raw=test_metrics_raw,
        test_avg_raw=test_avg_raw,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")
