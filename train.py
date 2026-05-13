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


class SwiGLUFFN(nn.Module):
    """Gated linear unit FFN (Shazeer 2020). Three projections; gate via SiLU."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim)
        self.w_value = nn.Linear(in_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return self.dropout(self.w_out(gate * value))


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
                 use_swiglu=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                                 dropout=dropout)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 pos_freq_bands: int = 0,
                 pos_freq_surface_only: bool = False,
                 use_swiglu=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.pos_freq_bands = pos_freq_bands
        self.pos_freq_surface_only = pos_freq_surface_only
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        # Fourier features per node: 2*L per spatial dim (sin + cos for each band).
        # Keep raw coords concatenated alongside, so effective input width is
        # space_dim + 2*space_dim*L + fun_dim.
        fourier_dim = 2 * space_dim * pos_freq_bands if pos_freq_bands > 0 else 0

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim + fourier_dim,
                                  n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        if pos_freq_bands > 0:
            freqs = 2.0 ** torch.arange(pos_freq_bands, dtype=torch.float32)
            self.register_buffer("_fourier_freqs", freqs)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                use_swiglu=use_swiglu,
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

    def fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """NeRF-style sinusoidal positional encoding.

        coords: (..., space_dim) — node coordinates (here, normalized x_norm[:, :2])
        returns: (..., 2 * space_dim * L) — [sin(2π·2ᵏ·d), cos(2π·2ᵏ·d)] features
        """
        proj = coords.unsqueeze(-1) * (2.0 * math.pi * self._fourier_freqs)
        return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(start_dim=-2)

    def forward(self, data, **kwargs):
        x = data["x"]
        if self.pos_freq_bands > 0:
            coords = x[..., :self.space_dim]
            fun = x[..., self.space_dim:]
            fourier = self.fourier_encode(coords)
            if self.pos_freq_surface_only:
                # Gate Fourier features by binary is_surface mask so only
                # surface nodes receive the high-frequency basis expansion.
                is_surface = data["is_surface"].to(fourier.dtype).unsqueeze(-1)
                fourier = fourier * is_surface
            x = torch.cat([coords, fourier, fun], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
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
            pred = model({"x": x_norm, "is_surface": is_surface})["preds"]

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
        "surf_weight_warmup_epochs": cfg.surf_weight_warmup_epochs,
        "surf_weight_init": cfg.surf_weight_init,
        "epochs_configured": cfg.epochs,
        "grad_clip": cfg.grad_clip,
        "use_onecycle": cfg.use_onecycle,
        "onecycle_max_lr_mult": cfg.onecycle_max_lr_mult,
        "onecycle_pct_start": cfg.onecycle_pct_start,
        "onecycle_div_factor": cfg.onecycle_div_factor,
        "onecycle_final_div_factor": cfg.onecycle_final_div_factor,
        "ema_decay": cfg.ema_decay,
        "pos_freq_bands": cfg.pos_freq_bands,
        "pos_freq_surface_only": cfg.pos_freq_surface_only,
        "use_swiglu": cfg.use_swiglu,
    }

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
    weight_decay: float = 1e-3
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    grad_clip: float = 1.0
    use_onecycle: bool = True  # OneCycleLR instead of CosineAnnealingLR
    onecycle_max_lr_mult: float = 10.0  # peak_lr = lr * this
    onecycle_pct_start: float = 0.05  # fraction of total steps used for warmup
    onecycle_div_factor: float = 10.0  # initial_lr = max_lr / this
    onecycle_final_div_factor: float = 100.0  # min_lr = initial_lr / this
    ema_decay: float = 0.999  # EMA of model weights for eval (0.0 disables)
    huber_delta: float = 1.0  # Huber loss threshold in normalized space
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    augment: bool = True
    aoa_jitter_rad: float = 0.00873  # std of Gaussian noise on AoA features (~ 0.5 deg)
    naca_jitter: float = 0.002       # std of Gaussian noise on normalized NACA camber feature
    surf_weight_warmup_epochs: int = 0  # epochs of linear ramp from surf_weight_init to surf_weight; 0 disables
    surf_weight_init: float = 1.0       # starting surf_weight at epoch 0 when warmup is enabled
    pos_freq_bands: int = 0          # Fourier positional encoding bands (0 = disabled, NeRF-style γ(x))
    pos_freq_surface_only: bool = False  # Gate Fourier features by is_surface mask (surface-only PE)
    use_swiglu: bool = False         # replace per-block GELU MLP with SwiGLU FFN


def augment_geometry(x: torch.Tensor, cfg: "Config") -> torch.Tensor:
    """AoA + NACA-camber jitter applied in raw (pre-normalization) space.

    Per-sample Gaussian noise broadcast across all nodes. Foil 2 jitter is
    masked to tandem samples — single-foil samples have all of dims 18-23
    exactly zero and we preserve that indicator (otherwise we induce a
    train/val distribution mismatch on the single-foil split).
    """
    x = x.clone()
    B = x.shape[0]
    aoa_noise = torch.randn(B, 1, 1, device=x.device) * cfg.aoa_jitter_rad
    naca_noise = torch.randn(B, 1, 1, device=x.device) * cfg.naca_jitter
    is_tandem = (x[:, :, 18:24].abs().sum(dim=(1, 2), keepdim=True) > 0).to(x.dtype)
    x[:, :, 14:15] += aoa_noise
    x[:, :, 15:16] += naca_noise
    x[:, :, 18:19] += aoa_noise * is_tandem
    x[:, :, 19:20] += naca_noise * is_tandem
    return x


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
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    pos_freq_bands=cfg.pos_freq_bands,
    pos_freq_surface_only=cfg.pos_freq_surface_only,
    use_swiglu=cfg.use_swiglu,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

steps_per_epoch = (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size
if cfg.use_onecycle:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr * cfg.onecycle_max_lr_mult,
        epochs=MAX_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.onecycle_pct_start,
        anneal_strategy="cos",
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )
    scheduler_step_per_batch = True
    print(
        f"Scheduler: OneCycleLR peak={cfg.lr * cfg.onecycle_max_lr_mult:.2e} "
        f"warmup_steps={int(steps_per_epoch * MAX_EPOCHS * cfg.onecycle_pct_start)} "
        f"total_steps={steps_per_epoch * MAX_EPOCHS}"
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scheduler_step_per_batch = False
    print(f"Scheduler: CosineAnnealingLR T_max={MAX_EPOCHS}")

ema_model = None
if cfg.ema_decay > 0.0:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print(f"EMA: enabled (decay={cfg.ema_decay})")


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

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
    epoch_grad_norm_sum = 0.0
    epoch_grad_clip_fires = 0
    n_batches = 0

    if cfg.surf_weight_warmup_epochs > 0 and epoch < cfg.surf_weight_warmup_epochs:
        progress = epoch / cfg.surf_weight_warmup_epochs
        current_surf_weight = cfg.surf_weight_init + progress * (
            cfg.surf_weight - cfg.surf_weight_init
        )
    else:
        current_surf_weight = cfg.surf_weight

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if cfg.augment:
            x = augment_geometry(x, cfg)
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
        residual = pred - y_norm
        sq_err = torch.where(
            residual.abs() <= cfg.huber_delta,
            0.5 * residual ** 2,
            cfg.huber_delta * (residual.abs() - 0.5 * cfg.huber_delta),
        )

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + current_surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            gn = grad_norm.item() if torch.isfinite(grad_norm) else float("nan")
            epoch_grad_norm_sum += gn
            if gn > cfg.grad_clip:
                epoch_grad_clip_fires += 1
        optimizer.step()
        if scheduler_step_per_batch:
            scheduler.step()
        if ema_model is not None:
            update_ema(model, ema_model, cfg.ema_decay)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    if not scheduler_step_per_batch:
        scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_grad_norm_mean = epoch_grad_norm_sum / max(n_batches, 1)
    epoch_grad_clip_fire_rate = epoch_grad_clip_fires / max(n_batches, 1)

    # --- Validate (use EMA weights for evaluation if available) ---
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    split_metrics = {
        name: evaluate_split(eval_model, loader, stats, cfg.surf_weight, device)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    dt = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        save_state = ema_model.state_dict() if ema_model is not None else model.state_dict()
        torch.save(save_state, model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": current_lr,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/grad_norm_mean": epoch_grad_norm_mean,
        "train/grad_clip_fire_rate": epoch_grad_clip_fire_rate,
        "train/current_surf_weight": current_surf_weight,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "ema_decay": cfg.ema_decay,
        "pos_freq_bands": cfg.pos_freq_bands,
        "pos_freq_surface_only": cfg.pos_freq_surface_only,
        "scheduler": "onecycle" if cfg.use_onecycle else "cosine",
        "surf_weight_warmup_epochs": cfg.surf_weight_warmup_epochs,
        "surf_weight_init": cfg.surf_weight_init,
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "test_avg": test_avg,
            "test_splits": test_metrics,
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
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")
