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


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block (Shazeer 2020).

    Replaces a standard 2-matrix MLP `W2 · GELU(W1 · x)` with the gated
    variant `W_down · (SiLU(W_gate · x) ⊙ W_up · x)`.

    `hidden_dim` is the original MLP inner dim; we use the full
    `hidden_dim` here (rounded up to a multiple of 8) to give the
    gate/up/down projections their natural per-token routing capacity
    (was 2/3 × hidden_dim for param-matching with the GELU 2-matrix MLP).
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        inner_dim = 288  # H46: bisect capacity (was hidden_dim=256; 320 was compute-bound)
        inner_dim = ((inner_dim + 7) // 8) * 8  # stays 288 (multiple of 8)
        self.inner_dim = inner_dim
        self.w_gate = nn.Linear(in_dim, inner_dim, bias=False)
        self.w_up = nn.Linear(in_dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, in_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.relu(self.w_gate(x)) * self.w_up(x))   # H39: ReGLU gate (ReLU = max(0,x))


class FourierCoordEnc(nn.Module):
    """Replace the 2 normalized coord dims with 2*2*n_freqs Fourier features.

    Input  shape: [B, N, in_dim]  where in_dim = X_DIM (raw 24-dim feature vector,
                                  already normalized by stats["x_mean"], stats["x_std"]).
    Output shape: [B, N, in_dim + (4*n_freqs - 2)]  -- 2 coord dims replaced by 4*n_freqs
                                                       Fourier features.

    Freqs are learnable (dyadic init), trained with their own optimizer group
    (10x lr, no weight decay) and clamped to [0.1, 100] after each step.
    """

    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.freqs = nn.Parameter(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :2]
        angles = coords.unsqueeze(-1) * self.freqs[None, None, None, :] * torch.pi
        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)
        fourier = torch.cat([sin_feats, cos_feats], dim=-1).reshape(
            *x.shape[:-1], 4 * self.n_freqs
        )
        return torch.cat([fourier, x[..., 2:]], dim=-1)


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
                 stoch_depth_prob: float = 0.0,
                 layer_scale_init: float = 0.1):
        super().__init__()
        self.last_layer = last_layer
        self.stoch_depth_prob = stoch_depth_prob
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim, hidden_dim * mlp_ratio)
        self.layer_scale_attn = nn.Parameter(torch.ones(hidden_dim) * layer_scale_init)
        self.layer_scale_mlp = nn.Parameter(torch.ones(hidden_dim) * layer_scale_init)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        if self.training and self.stoch_depth_prob > 0.0:
            if torch.rand(1, device=fx.device).item() < self.stoch_depth_prob:
                if self.last_layer:
                    return self.mlp2(self.ln_3(fx))
                return fx
        fx = self.layer_scale_attn * self.attn(self.ln_1(fx)) + fx
        fx = self.layer_scale_mlp * self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
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
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                stoch_depth_prob=0.1 * (i / max(n_layers - 1, 1)),
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
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

# H63 reflection-TTA: which raw-x feature dims flip sign under z (vertical) reflection.
# Strategy B per researcher recommendation: negate only geometric per-node features.
# AoA (14, 18), NACA camber M (15, 19), and gap (22) are *physically* antisymmetric
# under reflection too, but those features are one-sided in training (NACA_M >= 0;
# raceCar AoA <= 0), so negating them creates OOD inputs that poison the average.
# pos_z (1) and saf (2, 3) are signed in training and stay in-distribution.
TTA_REFLECT_DIMS = (1, 2, 3)


def make_reflection_mask(device, x_dim: int = X_DIM) -> torch.Tensor:
    """Per-feature sign mask for y/z-axis reflection in raw-x space."""
    mask = torch.ones(x_dim, device=device)
    for d in TTA_REFLECT_DIMS:
        mask[d] = -1.0
    return mask


def predict_with_tta(model, x, stats, fourier_enc, refl_mask):
    """H63: y-axis reflection TTA — average prediction over the original and the
    z-reflected input. Reflection is applied to raw x (before normalization) so
    nonzero per-feature means do not break the mapping. Averaging is done in
    physical (denormalized) target space so the nonzero Uy mean does not bias
    the average; the returned prediction is in the normalized-target space the
    downstream loss/scoring code expects.
    """
    # Forward 1: original geometry.
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    x_norm = fourier_enc(x_norm)
    pred_orig_norm = model({"x": x_norm})["preds"]

    # Forward 2: reflected geometry (negate z-flipping raw features, then normalize).
    x_refl = x * refl_mask
    x_refl_norm = (x_refl - stats["x_mean"]) / stats["x_std"]
    x_refl_norm = fourier_enc(x_refl_norm)
    pred_refl_norm = model({"x": x_refl_norm})["preds"]

    # Denormalize, un-reflect Uy in physical space, then average in physical space.
    y_std = stats["y_std"]
    y_mean = stats["y_mean"]
    pred_orig_phys = pred_orig_norm * y_std + y_mean
    pred_refl_phys = pred_refl_norm * y_std + y_mean
    pred_refl_phys = pred_refl_phys.clone()
    pred_refl_phys[..., 1] = -pred_refl_phys[..., 1]  # Uy flips under reflection
    pred_avg_phys = 0.5 * (pred_orig_phys + pred_refl_phys)
    return (pred_avg_phys - y_mean) / y_std


def evaluate_split(model, loader, stats, surf_weight, device, tta: bool = False,
                   refl_mask: torch.Tensor | None = None) -> dict[str, float]:
    """Evaluate a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    If ``tta=True``, predictions are averaged over the y-axis (vertical)
    reflection of the input — see ``predict_with_tta``.
    """
    if tta and refl_mask is None:
        raise ValueError("refl_mask must be provided when tta=True")
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

            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            if tta:
                pred = predict_with_tta(model, x, stats, fourier_enc, refl_mask)
            else:
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                x_norm = fourier_enc(x_norm)
                pred = model({"x": x_norm})["preds"]

            abs_err = (pred - y_norm).abs()
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (abs_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            # H18: per-channel surf-loss weighting (mirrors training loop).
            surf_ch_weights = abs_err.new_tensor([0.5, 0.5, 2.0])
            surf_loss_sum += (
                ((abs_err * surf_ch_weights) * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            B = y.shape[0]
            finite_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if finite_sample.any():
                idx = finite_sample.nonzero(as_tuple=True)[0]
                ds, dv = accumulate_batch(
                    pred_orig[idx], y[idx], is_surface[idx], mask[idx],
                    mae_surf, mae_vol,
                )
            else:
                ds, dv = 0, 0
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
    val_metrics_tta: dict | None = None,
    val_avg_tta: dict | None = None,
    test_metrics_no_tta: dict | None = None,
    test_avg_no_tta: dict | None = None,
) -> None:
    """Write a local summary next to the best checkpoint.

    H63: when ``val_metrics_tta`` / ``test_metrics_no_tta`` are supplied,
    the summary surfaces both TTA-on and TTA-off val + test numbers at the
    same best checkpoint so the TTA delta is directly readable from
    metrics.yaml without parsing metrics.jsonl.
    """
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

    # H63: `best_val/...` is the TTA-off val at the best checkpoint (matches the
    # baseline trajectory; this is what was used for checkpoint selection).
    for split_name, m in best_metrics["per_split"].items():
        for k, v in m.items():
            summary[f"best_val/{split_name}/{k}"] = v
    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        # H63: `test_avg/mae_surf_p` and `test/...` are the TTA-on numbers — the
        # headline numbers for the paper-facing comparison.
        summary["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    summary[f"test/{split_name}/{k}"] = v

    # H63: TTA-on val + TTA-off test at the same best checkpoint.
    if val_avg_tta is not None and "avg/mae_surf_p" in val_avg_tta:
        summary["val_avg_tta/mae_surf_p"] = val_avg_tta["avg/mae_surf_p"]
        if val_metrics_tta is not None:
            for split_name, m in val_metrics_tta.items():
                for k, v in m.items():
                    summary[f"val_tta/{split_name}/{k}"] = v
    if test_avg_no_tta is not None and "avg/mae_surf_p" in test_avg_no_tta:
        summary["test_avg_no_tta/mae_surf_p"] = test_avg_no_tta["avg/mae_surf_p"]
        if test_metrics_no_tta is not None:
            for split_name, m in test_metrics_no_tta.items():
                for k, v in m.items():
                    summary[f"test_no_tta/{split_name}/{k}"] = v

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


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

# H63: y-axis reflection TTA mask. Built once and reused for both val and test passes.
refl_mask = make_reflection_mask(device)
_refl_dim_labels = [
    "pos_x", "pos_z", "saf_0", "saf_1",
    "dsdf_0", "dsdf_1", "dsdf_2", "dsdf_3", "dsdf_4", "dsdf_5", "dsdf_6", "dsdf_7",
    "is_surf", "log_Re",
    "AoA1", "NACA1_M", "NACA1_P", "NACA1_T",
    "AoA2", "NACA2_M", "NACA2_P", "NACA2_T",
    "gap", "stagger",
]
print("[H63] Reflection mask (raw-x dims, -1.0 = flipped under z-reflection):")
print("  " + ", ".join(
    f"{_refl_dim_labels[d]}={refl_mask[d].item():+.0f}" for d in range(X_DIM)
))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

# H63: sanity-check the reflection. Print pos_z, saf_0/1 before/after for a train sample.
_sanity_sample = train_ds[0]
_sanity_x = _sanity_sample[0].to(device).unsqueeze(0)  # [1, N, 24]
_sanity_x_refl = _sanity_x * refl_mask
print("[H63] Sanity check — first train sample, first 5 nodes:")
print(f"  pos_z       orig: {_sanity_x[0, :5, 1].cpu().tolist()}")
print(f"  pos_z       refl: {_sanity_x_refl[0, :5, 1].cpu().tolist()}")
print(f"  saf_0       orig: {_sanity_x[0, :5, 2].cpu().tolist()}")
print(f"  saf_0       refl: {_sanity_x_refl[0, :5, 2].cpu().tolist()}")
print(f"  pos_x  (unchanged) orig: {_sanity_x[0, :5, 0].cpu().tolist()}")
print(f"  pos_x  (unchanged) refl: {_sanity_x_refl[0, :5, 0].cpu().tolist()}")
print(f"  AoA1   (unchanged) orig={_sanity_x[0, 0, 14].item():+.4f} refl={_sanity_x_refl[0, 0, 14].item():+.4f}")
print(f"  NACA_M (unchanged) orig={_sanity_x[0, 0, 15].item():+.4f} refl={_sanity_x_refl[0, 0, 15].item():+.4f}")
del _sanity_sample, _sanity_x, _sanity_x_refl

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

N_FREQS = 6
fourier_enc = FourierCoordEnc(n_freqs=N_FREQS).to(device)

model_config = dict(
    space_dim=2,
    fun_dim=4 * N_FREQS + (X_DIM - 2) - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_model_params = sum(p.numel() for p in model.parameters())
n_fourier_params = sum(p.numel() for p in fourier_enc.parameters())
n_params = n_model_params + n_fourier_params  # includes 6 learnable freqs
print(f"Model: Transolver ({n_model_params/1e6:.2f}M params) + FourierCoordEnc ({n_fourier_params} freqs)")
print(f"n_params (total trainable): {n_params}")
swiglu_inner_dim = model.blocks[0].mlp.inner_dim
print(f"SwiGLU inner_dim: {swiglu_inner_dim}, total_params: {n_params}")
# H39: ReGLU gate sanity check (ReLU = max(0,x))
_h39_test_x = torch.tensor([-1.0, 0.0, 1.0])
print(f"[H39] ReGLU gate at x=-1: {F.relu(_h39_test_x[0]).item():.4f} (expected 0.0000)")
print(f"[H39] ReGLU gate at x= 0: {F.relu(_h39_test_x[1]).item():.4f} (expected 0.0000)")
print(f"[H39] ReGLU gate at x=+1: {F.relu(_h39_test_x[2]).item():.4f} (expected 1.0000)")
print(f"[H39] SwiGLU inner_dim: {model.blocks[0].mlp.inner_dim}, n_params: {n_params}")
print(f"[H46] SwiGLU inner_dim: {model.blocks[0].mlp.inner_dim}")  # should print 288
print(f"[H46] n_params: {n_params}")  # should print ~892,631
for i, b in enumerate(model.blocks):
    print(
        f"block {i}: layer_scale_attn init avg={b.layer_scale_attn.mean().item():.4f}, "
        f"layer_scale_mlp init avg={b.layer_scale_mlp.mean().item():.4f}"
    )

# H40: learned Fourier freqs in a separate param group with 10x lr and no weight decay.
# H54: LayerScale params (layer_scale_attn + layer_scale_mlp across 5 blocks = 10 tensors x 128
# channels = 1280 params) also moved to a 10x lr, no-WD group. Same insight as H40: WD pulls
# scaling factors toward zero (init=0.025), fighting the model's learned per-channel
# diversification, and default lr=5e-4 under-trains them.
freq_params = [p for n, p in fourier_enc.named_parameters() if n == "freqs"]
layer_scale_params = [p for n, p in model.named_parameters() if "layer_scale" in n]
other_params = [p for n, p in model.named_parameters() if "layer_scale" not in n]
assert len(freq_params) == 1 and freq_params[0].numel() == N_FREQS, (
    f"Expected 1 freq tensor with {N_FREQS} elems, "
    f"got {len(freq_params)} with {[p.numel() for p in freq_params]}"
)
# H54: param-group consistency check — every model param ends up in exactly one of
# (layer_scale_params, other_params); freq_params come from fourier_enc separately.
all_model_p = {id(p): p for p in model.parameters()}
all_assigned_p = {id(p): p for group in [layer_scale_params, other_params] for p in group}
assert set(all_model_p.keys()) == set(all_assigned_p.keys()), (
    f"Mismatch: {len(all_model_p)} model params, {len(all_assigned_p)} assigned"
)
n_layer_scale = sum(p.numel() for p in layer_scale_params)
expected_layer_scale = 2 * len(model.blocks) * model_config["n_hidden"]
assert n_layer_scale == expected_layer_scale, (
    f"Expected {expected_layer_scale} LayerScale params "
    f"({len(model.blocks)} blocks x 2 paths x {model_config['n_hidden']} channels), got {n_layer_scale}"
)
assert sum(p.numel() for p in freq_params) + sum(p.numel() for p in layer_scale_params) + sum(p.numel() for p in other_params) == (
    sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in fourier_enc.parameters())
)
optimizer = torch.optim.AdamW(
    [
        {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": freq_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0},
        {"params": layer_scale_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0},
    ],
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
print(
    f"[H54] Optimizer param groups: "
    f"other lr={optimizer.param_groups[0]['lr']:.2e} wd={optimizer.param_groups[0]['weight_decay']:.0e} "
    f"({sum(p.numel() for p in other_params)} params), "
    f"freqs lr={optimizer.param_groups[1]['lr']:.2e} wd={optimizer.param_groups[1]['weight_decay']:.0e} "
    f"({sum(p.numel() for p in freq_params)} params), "
    f"layerscale lr={optimizer.param_groups[2]['lr']:.2e} wd={optimizer.param_groups[2]['weight_decay']:.0e} "
    f"({n_layer_scale} params, expected {expected_layer_scale})"
)
print(f"[H40] Initial freqs: {fourier_enc.freqs.detach().cpu().tolist()}")
# H19: linear warm-up over the first epoch (batches), then cosine annealing for the remaining 14.
# scheduler.step() is called once per BATCH below, so total_iters and T_max are expressed in batches.
batches_per_epoch = len(train_loader)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=batches_per_epoch
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=14 * batches_per_epoch
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[batches_per_epoch],
)
print(f"LR schedule: linear warmup over {batches_per_epoch} batches (1 epoch), then cosine T_max={14 * batches_per_epoch} batches (14 epochs)")

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
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        x_norm = fourier_enc(x_norm)
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        abs_err = (pred - y_norm).abs()

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        # H18: per-channel surf-loss weighting. Mass-preserving (sum = 3.0).
        # Upweights pressure (channel 2 = p) which defines the primary metric val_avg/mae_surf_p.
        surf_ch_weights = abs_err.new_tensor([0.5, 0.5, 2.0])
        surf_loss = ((abs_err * surf_ch_weights) * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        # H40: clip both model and fourier_enc params together (freqs included).
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(fourier_enc.parameters()), max_norm=25.0
        )
        optimizer.step()
        # H40: keep freqs bounded after the update. Safety net for the higher freqs lr.
        with torch.no_grad():
            fourier_enc.freqs.clamp_(0.1, 100.0)
        scheduler.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate (TTA-off during training; TTA-on is inference-only, applied
    # at the end on the best checkpoint). This keeps the training-time val
    # trajectory identical to baseline so checkpoint selection isn't
    # confounded by the TTA recipe, and frees up wall-clock budget for the
    # same number of training epochs as baseline. ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, tta=False)
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
    current_lr = optimizer.param_groups[0]["lr"]
    freqs_lr = optimizer.param_groups[1]["lr"]
    layer_scale_lr = optimizer.param_groups[2]["lr"]
    freqs_now = fourier_enc.freqs.detach().cpu().tolist()
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": current_lr,
        "train/current_lr": current_lr,
        "train/freqs_lr": freqs_lr,
        "train/layer_scale_lr": layer_scale_lr,
        "train/freqs": freqs_now,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/last_grad_norm": float(total_norm),
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    print(
        f"    freqs (lr={freqs_lr:.2e}): "
        f"[{', '.join(f'{v:.4f}' for v in freqs_now)}]"
    )
    print(f"    layer_scale (lr={layer_scale_lr:.2e})")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Log final per-block LayerScale stats (end-of-training state) ---
final_layer_scale_stats: dict[str, float] = {}
for i, b in enumerate(model.blocks):
    final_layer_scale_stats[f"final/layer_scale_attn_l{i}_mean"] = float(b.layer_scale_attn.mean().item())
    final_layer_scale_stats[f"final/layer_scale_attn_l{i}_std"] = float(b.layer_scale_attn.std().item())
    final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_mean"] = float(b.layer_scale_mlp.mean().item())
    final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_std"] = float(b.layer_scale_mlp.std().item())
# H40: final learned freqs values (compare to dyadic init 1, 2, 4, 8, 16, 32).
final_freqs = fourier_enc.freqs.detach().cpu().tolist()
init_freqs = [2.0**k for k in range(N_FREQS)]
freq_drift = [f - i for f, i in zip(final_freqs, init_freqs)]
freq_rel_drift = [(f - i) / i for f, i in zip(final_freqs, init_freqs)]
freqs_summary = {
    "final/freqs": final_freqs,
    "final/freqs_init": init_freqs,
    "final/freqs_abs_drift": freq_drift,
    "final/freqs_rel_drift": freq_rel_drift,
}
append_metrics_jsonl(
    metrics_jsonl_path,
    {"event": "final", **final_layer_scale_stats, **freqs_summary},
)
print("Final LayerScale stats (end-of-training):")
for i in range(len(model.blocks)):
    print(
        f"  block {i}: attn mean={final_layer_scale_stats[f'final/layer_scale_attn_l{i}_mean']:.4f} "
        f"std={final_layer_scale_stats[f'final/layer_scale_attn_l{i}_std']:.4f}  "
        f"mlp mean={final_layer_scale_stats[f'final/layer_scale_mlp_l{i}_mean']:.4f} "
        f"std={final_layer_scale_stats[f'final/layer_scale_mlp_l{i}_std']:.4f}"
    )
print("[H40] Final learned freqs vs dyadic init:")
for k in range(N_FREQS):
    print(
        f"  freq[{k}]: init={init_freqs[k]:.4f} -> final={final_freqs[k]:.4f} "
        f"(drift {freq_drift[k]:+.4f}, {freq_rel_drift[k]*100:+.2f}%)"
    )

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits (with H63 reflection TTA)...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        # H63: TTA-on test pass (the headline result).
        _tta_t0 = time.time()
        test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                                 tta=True, refl_mask=refl_mask)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        tta_test_seconds = time.time() - _tta_t0
        print(f"\n  TEST [TTA-on]  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}  ({tta_test_seconds:.1f}s)")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "tta": True,
            "test_avg": test_avg,
            "test_splits": test_metrics,
            "tta_test_seconds": tta_test_seconds,
        })

        # H63 mechanism diagnostic: re-run val + test with both TTA on and off on the
        # best checkpoint, so the per-channel TTA delta compares the same weights.
        # The training-loop `split_metrics` belongs to the last epoch, which may not
        # be the best one — do not use it for the delta.
        print("\n[H63] Diagnostic: re-evaluating val + test with TTA OFF (same checkpoint)...")
        _no_t0 = time.time()
        val_metrics_no_tta = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, tta=False)
            for name, loader in val_loaders.items()
        }
        val_avg_no_tta = aggregate_splits(val_metrics_no_tta)
        no_tta_val_seconds = time.time() - _no_t0
        print(f"  VAL  [TTA-off] avg_surf_p={val_avg_no_tta['avg/mae_surf_p']:.4f}  ({no_tta_val_seconds:.1f}s)")
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(name, val_metrics_no_tta[name])

        _no_t0 = time.time()
        test_metrics_no_tta = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, tta=False)
            for name, loader in test_loaders.items()
        }
        test_avg_no_tta = aggregate_splits(test_metrics_no_tta)
        no_tta_test_seconds = time.time() - _no_t0
        print(f"  TEST [TTA-off] avg_surf_p={test_avg_no_tta['avg/mae_surf_p']:.4f}  ({no_tta_test_seconds:.1f}s)")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics_no_tta[name])

        # Re-run TTA-on val on the best checkpoint so the val delta is checkpoint-
        # consistent (the training-loop split_metrics is from the *last* epoch).
        _tta_val_t0 = time.time()
        val_metrics_tta = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                                 tta=True, refl_mask=refl_mask)
            for name, loader in val_loaders.items()
        }
        val_avg_tta = aggregate_splits(val_metrics_tta)
        tta_val_seconds = time.time() - _tta_val_t0
        print(f"  VAL  [TTA-on ] avg_surf_p={val_avg_tta['avg/mae_surf_p']:.4f}  ({tta_val_seconds:.1f}s)")
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(name, val_metrics_tta[name])

        # Print per-channel TTA delta (Ux, Uy, p) to surface the mechanism.
        print("\n[H63] Per-channel TTA delta = TTA_on - TTA_off (negative = improvement):")
        for split_pool, names in (("VAL ", VAL_SPLIT_NAMES), ("TEST", TEST_SPLIT_NAMES)):
            on_pool = test_metrics if split_pool == "TEST" else val_metrics_tta
            off_pool = test_metrics_no_tta if split_pool == "TEST" else val_metrics_no_tta
            for name in names:
                d_p = on_pool[name]["mae_surf_p"] - off_pool[name]["mae_surf_p"]
                d_ux = on_pool[name]["mae_surf_Ux"] - off_pool[name]["mae_surf_Ux"]
                d_uy = on_pool[name]["mae_surf_Uy"] - off_pool[name]["mae_surf_Uy"]
                print(
                    f"  {split_pool} {name:<26s} dp={d_p:+.4f}  dUx={d_ux:+.4f}  dUy={d_uy:+.4f}"
                )

        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "diagnostic_no_tta",
            "best_epoch": best_metrics["epoch"],
            "tta": False,
            "val_avg_no_tta": val_avg_no_tta,
            "val_splits_no_tta": val_metrics_no_tta,
            "test_avg_no_tta": test_avg_no_tta,
            "test_splits_no_tta": test_metrics_no_tta,
            "no_tta_val_seconds": no_tta_val_seconds,
            "no_tta_test_seconds": no_tta_test_seconds,
        })
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "diagnostic_tta_val_best_ckpt",
            "best_epoch": best_metrics["epoch"],
            "tta": True,
            "val_avg_tta_best_ckpt": val_avg_tta,
            "val_splits_tta_best_ckpt": val_metrics_tta,
            "tta_val_seconds": tta_val_seconds,
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
        val_metrics_tta=val_metrics_tta if not cfg.skip_test else None,
        val_avg_tta=val_avg_tta if not cfg.skip_test else None,
        test_metrics_no_tta=test_metrics_no_tta if not cfg.skip_test else None,
        test_avg_no_tta=test_avg_no_tta if not cfg.skip_test else None,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")
