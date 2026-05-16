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


def signed_log1p(y: torch.Tensor) -> torch.Tensor:
    """Smooth, invertible magnitude compression with slope=1 at zero."""
    return torch.sign(y) * torch.log1p(torch.abs(y))


def signed_expm1(z: torch.Tensor) -> torch.Tensor:
    return torch.sign(z) * torch.expm1(torch.abs(z))


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


class SwiGLUMLP(nn.Module):
    """Gated FFN: linear_in -> SiLU(gate) * value -> dropout -> linear_out.

    The input projection produces 2*n_hidden activations, split into a gate
    stream (passed through SiLU) and a value stream. Output is gate*value
    projected back to n_output. Replaces the standard linear -> GELU -> linear
    pattern in TransolverBlock.mlp.
    """

    def __init__(self, n_input, n_hidden, n_output, dropout=0.0):
        super().__init__()
        self.fc_in = nn.Linear(n_input, 2 * n_hidden)
        self.fc_out = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.fc_in(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.dropout(x)
        return self.fc_out(x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True,
                 dropout=0.0):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(
            nn.Linear(n_input, n_hidden), act_fn(), nn.Dropout(dropout)
        )
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn(), nn.Dropout(dropout))
             for _ in range(n_layers)]
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


class LayerScale(nn.Module):
    """Per-channel learnable diagonal scaling for residual branches (CaIT, Touvron et al. 2021).

    Init at 1e-6 so the residual contribution starts near-zero and each block is
    activated gradually during training. Adds ``dim`` scalars per instance.
    """

    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                             dropout=0.1)
        self.ls1 = LayerScale(hidden_dim)
        self.ls2 = LayerScale(hidden_dim)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.ls1(self.attn(self.ln_1(fx))) + fx
        fx = self.ls2(self.mlp(self.ln_2(fx))) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class RFFEncoding(nn.Module):
    """Fixed Random Fourier Feature encoding for 2D coordinates.

    Lifts (x, z) into a 2*n_freq-dimensional Fourier basis using a fixed
    Gaussian projection matrix B ~ N(0, sigma^2). Per Tancik et al. (NeurIPS
    2020), the projection is computed as gamma(v) = [sin(2*pi*B*v), cos(2*pi*B*v)]
    and B is registered as a non-learnable buffer so the spectral guarantee holds.
    """

    def __init__(self, n_freq: int = 32, sigma: float = 1.0, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        B = torch.randn(2, n_freq, generator=gen) * sigma  # [2, n_freq]
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq

    def forward(self, xy):  # xy: [..., 2]
        proj = xy @ self.B  # [..., n_freq]
        return torch.cat([torch.sin(2 * torch.pi * proj),
                          torch.cos(2 * torch.pi * proj)], dim=-1)


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 rff_n_freq: int = 32, rff_sigma: float = 1.0,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 geom_ctx_dim: int = 11, geom_ctx_start: int = 13):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.geom_ctx_dim = geom_ctx_dim
        self.geom_ctx_start = geom_ctx_start

        self.rff = RFFEncoding(n_freq=rff_n_freq, sigma=rff_sigma)
        rff_dim = self.rff.out_dim

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(rff_dim + fun_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        self.geom_proj = MLP(geom_ctx_dim, n_hidden * 2, n_hidden,
                             n_layers=0, res=False, act=act)
        # Gates init at 0 so the injection starts as a no-op and recovers baseline.
        self.geom_gates = nn.Parameter(torch.zeros(n_layers))

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
        pos = x[..., :2]
        rest = x[..., 2:]
        pos_rff = self.rff(pos)
        x_rff = torch.cat([pos_rff, rest], dim=-1)
        fx = self.preprocess(x_rff) + self.placeholder[None, None, :]
        geom_ctx = x[:, 0, self.geom_ctx_start:self.geom_ctx_start + self.geom_ctx_dim]
        g = self.geom_proj(geom_ctx).unsqueeze(1)
        for i, block in enumerate(self.blocks):
            fx = fx + self.geom_gates[i] * g
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

            # Workaround for non-finite GT propagating through masked accumulation
            # (NaN*0 == NaN). Exclude any sample whose y contains non-finite values
            # entirely via mask, and zero-fill those positions so the multiplication
            # in loss/accumulate_batch is finite. Matches scoring.py's intent.
            y_finite_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            mask = mask & y_finite_sample.unsqueeze(-1)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

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
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    use_onecycle: bool = False  # OneCycleLR (Smith&Topin) instead of CosineAnnealingLR
    onecycle_pct_start: float = 0.3  # fraction of training for rising LR phase


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
    # Re-stratify the balanced sampler: upweight high-Re samples (Re>1e6) by 2x.
    # x dim 13 is the RAW log(Re) in the .pt files — normalization is applied at
    # use time inside the train/eval loops, not in the stored tensors. All mesh
    # nodes share the same value, so read node 0.
    log_re_threshold = math.log(1e6)
    re_multiplier = torch.ones(len(train_ds), dtype=torch.float64)
    re_count_high = 0
    re_loop_t0 = time.time()
    for i in range(len(train_ds)):
        x_i, _, _ = train_ds[i]
        log_re = float(x_i[0, 13].item())
        if log_re > log_re_threshold:
            re_multiplier[i] = 2.0
            re_count_high += 1
    print(
        f"Re-stratified sampler: {re_count_high}/{len(train_ds)} samples have Re>1e6 (x2 weight) "
        f"[loop took {time.time() - re_loop_t0:.1f}s]"
    )

    adjusted_weights = sample_weights.to(torch.float64) * re_multiplier
    sampler = WeightedRandomSampler(adjusted_weights, num_samples=len(train_ds), replacement=True)
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
    rff_n_freq=32,
    rff_sigma=1.0,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
if cfg.use_onecycle:
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        total_steps=total_steps,
        pct_start=cfg.onecycle_pct_start,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    onecycle_mode = True
    rising_epochs = int(total_steps * cfg.onecycle_pct_start / max(len(train_loader), 1))
    print(
        f"Scheduler: OneCycleLR  max_lr={optimizer.param_groups[0]['lr']:.2e}  "
        f"div_factor=25  final_div_factor=1e4  pct_start={cfg.onecycle_pct_start}  "
        f"total_steps={total_steps}  steps_per_epoch={len(train_loader)}\n"
        f"  Rising phase: epochs 1-{rising_epochs}; falling: epochs {rising_epochs+1}-end  "
        f"(cycle_momentum=False)"
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    onecycle_mode = False
    print("Scheduler: CosineAnnealingLR(T_max=15)")

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
slog1p_diag_printed = False

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if epoch == 0 and n_batches == 0:
            geom = x[:, :, 13:24]
            geom0 = geom[:, 0:1, :]
            diff = ((geom - geom0).abs() * mask.unsqueeze(-1).float()).max().item()
            assert diff < 1e-5, f"global features (dims 13-23) not per-sample constant (max diff {diff})"

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]

        # H11: signed-log1p target transform applied on the loss side only.
        # pred stays in linear (normalized) space for the metric/eval path;
        # both pred and y_norm are passed through slog1p before MSE so per-sample
        # gradients are magnitude-comparable across the Re range.
        y_log = signed_log1p(y_norm)
        pred_log = signed_log1p(pred)
        sq_err = (pred_log - y_log) ** 2

        if not slog1p_diag_printed:
            with torch.no_grad():
                m = mask.unsqueeze(-1)
                y_raw_p = y[..., 2][mask]
                y_norm_p = y_norm[..., 2][mask]
                y_log_p = y_log[..., 2][mask]
                print(
                    f"slog1p diag (pressure ch, batch 0, valid nodes):\n"
                    f"  y raw     : min={y_raw_p.min().item():.2f} "
                    f"max={y_raw_p.max().item():.2f} "
                    f"std={y_raw_p.std().item():.2f}\n"
                    f"  y_norm    : min={y_norm_p.min().item():.4f} "
                    f"max={y_norm_p.max().item():.4f} "
                    f"std={y_norm_p.std().item():.4f}\n"
                    f"  slog1p(y_norm): min={y_log_p.min().item():.4f} "
                    f"max={y_log_p.max().item():.4f} "
                    f"std={y_log_p.std().item():.4f}"
                )
            slog1p_diag_printed = True

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if onecycle_mode:
            scheduler.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    if not onecycle_mode:
        scheduler.step()
    epoch_lr = scheduler.get_last_lr()[0]
    if epoch == 0:
        print(f"  scheduler lr after epoch 1: {epoch_lr:.6e}")
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    dt = time.time() - t0

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": epoch_lr,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
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

    gate_values = model.geom_gates.detach().cpu().tolist()
    print(f"Final geom_gates (from best checkpoint): {gate_values}")
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "geom_gates",
        "best_epoch": best_metrics["epoch"],
        "geom_gates": gate_values,
    })

    ls1_norms: list[float] = []
    ls2_norms: list[float] = []
    ls1_means: list[float] = []
    ls2_means: list[float] = []
    print("LayerScale gamma diagnostics (from best checkpoint):")
    for i, block in enumerate(model.blocks):
        g1 = block.ls1.gamma.detach()
        g2 = block.ls2.gamma.detach()
        n1, n2 = g1.norm().item(), g2.norm().item()
        m1, m2 = g1.mean().item(), g2.mean().item()
        ls1_norms.append(n1)
        ls2_norms.append(n2)
        ls1_means.append(m1)
        ls2_means.append(m2)
        print(
            f"  Block {i}: ls1[norm={n1:.4f} mean={m1:.4e}] "
            f"ls2[norm={n2:.4f} mean={m2:.4e}]"
        )
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "layerscale",
        "best_epoch": best_metrics["epoch"],
        "ls1_norms": ls1_norms,
        "ls2_norms": ls2_norms,
        "ls1_means": ls1_means,
        "ls2_means": ls2_means,
    })

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
