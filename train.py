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
from torch.amp import GradScaler, autocast
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
from soap import SOAP

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
        # Diagnostic capture (uncompiled-model only). When True, computes and
        # stores per-head slice entropy mean for one batch into _last_slice_entropy.
        self._diag_capture = False
        self._last_slice_entropy: torch.Tensor | None = None

    def forward(self, x, gamma=None, beta=None):
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
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        if gamma is not None and beta is not None:
            slice_logits = slice_logits * (1.0 + gamma) + beta
        slice_weights = self.softmax(slice_logits)
        if self._diag_capture:
            with torch.no_grad():
                # entropy per (B, H, N): -sum_S w * log(w + eps); mean over (B, N) per head
                ent = -(slice_weights * torch.log(slice_weights.clamp(min=1e-8))).sum(dim=-1)
                self._last_slice_entropy = ent.mean(dim=(0, 2)).detach().to(torch.float64)
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
                 re_conditional_layernorm: bool = False,
                 ensemble_heads: int = 1):
        super().__init__()
        self.last_layer = last_layer
        self.re_conditional_layernorm = re_conditional_layernorm
        # When re_conditional_layernorm is True, the LN affine is provided by
        # shared ReConditionalLayerNorm modules owned by the parent Transolver;
        # this block contributes no LN parameters. Otherwise, plain nn.LayerNorm.
        if not re_conditional_layernorm:
            self.ln_1 = nn.LayerNorm(hidden_dim)
            self.ln_2 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if self.last_layer:
            if not re_conditional_layernorm:
                self.ln_3 = nn.LayerNorm(hidden_dim)
            # ensemble_heads > 1: K parallel output projections averaged at output
            # (deep-ensemble variance reduction). ensemble_heads == 1 preserves
            # the baseline single nn.Linear projection bit-for-bit.
            if ensemble_heads > 1:
                out_proj = EnsembleOutputHead(hidden_dim, out_dim, n_heads=ensemble_heads)
            else:
                out_proj = nn.Linear(hidden_dim, out_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                out_proj,
            )

    def forward(self, fx, gamma=None, beta=None, log_re=None,
                rcln_1=None, rcln_2=None, rcln_3=None):
        if self.re_conditional_layernorm:
            normed_1 = rcln_1(fx, log_re)
        else:
            normed_1 = self.ln_1(fx)
        fx = self.attn(normed_1, gamma, beta) + fx
        if self.re_conditional_layernorm:
            normed_2 = rcln_2(fx, log_re)
        else:
            normed_2 = self.ln_2(fx)
        fx = self.mlp(normed_2) + fx
        if self.last_layer:
            if self.re_conditional_layernorm:
                normed_3 = rcln_3(fx, log_re)
            else:
                normed_3 = self.ln_3(fx)
            return self.mlp2(normed_3)
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 re_conditional_layernorm: bool = False,
                 re_ln_hidden_film: int = 8,
                 ensemble_heads: int = 1):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.re_conditional_layernorm = re_conditional_layernorm
        self.ensemble_heads = ensemble_heads

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
                re_conditional_layernorm=re_conditional_layernorm,
                ensemble_heads=ensemble_heads,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)
        # FiLM Re-conditioning of slice logits. Shared single instance — gamma/beta
        # are computed once and passed to every block. Zero-init makes (gamma, beta)
        # = (0, 0) at step 0 → identical to baseline (no FiLM) at init.
        self.re_film = ReFiLM(heads=n_head, slice_num=slice_num, hidden=8)
        # Re-conditional LayerNorm γ/β at LN-affine injection point (CIN-style;
        # Dumoulin et al. 2017, DiT adaLN-Zero). Three shared instances — one per
        # LN role (pre-attn, pre-FFN, pre-out-mlp) — referenced by every block.
        # Created AFTER self.apply so the zero-init of the FiLM MLP's final layer
        # is preserved (apply would overwrite it with trunc_normal otherwise).
        if re_conditional_layernorm:
            self.re_ln_1 = ReConditionalLayerNorm(n_hidden, hidden_film=re_ln_hidden_film)
            self.re_ln_2 = ReConditionalLayerNorm(n_hidden, hidden_film=re_ln_hidden_film)
            self.re_ln_3 = ReConditionalLayerNorm(n_hidden, hidden_film=re_ln_hidden_film)
        # Re-init EnsembleOutputHead modules after self.apply (which overwrote
        # them with trunc_normal_). Keeps kaiming_normal + per-head symmetry
        # breaking noise intact.
        for m in self.modules():
            if isinstance(m, EnsembleOutputHead):
                m.reinit()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            # Skip param-free LayerNorm (elementwise_affine=False has weight/bias None).
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        # x is already in normalized space (caller normalises with stats), so x[:, 0, 13:14]
        # is the per-sample normalized log(Re). Same channel ReScaleHead uses.
        log_re = x[:, 0, 13:14]
        gamma, beta = self.re_film(log_re)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        if self.re_conditional_layernorm:
            for block in self.blocks:
                fx = block(fx, gamma, beta, log_re=log_re,
                           rcln_1=self.re_ln_1, rcln_2=self.re_ln_2,
                           rcln_3=self.re_ln_3)
        else:
            for block in self.blocks:
                fx = block(fx, gamma, beta)
        return {"preds": fx}


class ReScaleHead(nn.Module):
    """Per-sample, Re-conditioned output scale (DimINO-style redimensionalization).

    Takes normalized log(Re) as a scalar per sample and emits a positive scale
    factor per output channel, broadcast over mesh nodes. Initialized so the
    output is the identity (scale ≈ 1.0) at step 0, so the model starts where
    plain Transolver would and only deviates if the gradient signal supports it.
    """

    def __init__(self, hidden: int = 32, out_channels: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.fill_(0.541)  # softplus(0.541) ≈ 1.0

    def forward(self, log_re: torch.Tensor) -> torch.Tensor:
        # log_re: [B, 1]  →  [B, 1, C] scale, broadcast over N nodes.
        raw = self.mlp(log_re)
        scale = F.softplus(raw)
        return scale.unsqueeze(1)


class ReConditionalLayerNorm(nn.Module):
    """Re-conditional LayerNorm affine (Conditional Instance Norm / adaLN-Zero).

    Wraps a param-free LayerNorm plus a small FiLM MLP from log(Re) to
    (γ_residual, β). Applied as:  out = (1 + γ_residual) · LN(x) + β.

    Zero-init of the final layer makes (γ_r, β) = (0, 0) at step 0, so the
    module reproduces nn.LayerNorm(weight=1, bias=0) exactly at init — the
    conditioning gate must be actively opened by the optimiser.

    Designed to be SHARED across all TransolverBlocks for a given LN role
    (e.g. one shared instance for pre-attn LN across every block). The
    per-token LN is local (different x per block) but the affine (γ, β)
    depends only on log_re, so the same affine is applied at every block
    per sample.

    References: Dumoulin et al. 2017 (1610.07629, CIN); Peebles & Xie 2022
    (2212.09748, adaLN-Zero); Perez et al. 2018 (1709.07871, FiLM).
    """

    def __init__(self, n_hidden: int, hidden_film: int = 8):
        super().__init__()
        self.n_hidden = n_hidden
        self.ln = nn.LayerNorm(n_hidden, elementwise_affine=False)
        self.film_net = nn.Sequential(
            nn.Linear(1, hidden_film),
            nn.GELU(),
            nn.Linear(hidden_film, 2 * n_hidden),
        )
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, x: torch.Tensor, log_re: torch.Tensor) -> torch.Tensor:
        # x: [B, N, n_hidden], log_re: [B, 1]
        normed = self.ln(x)
        film = self.film_net(log_re)  # [B, 2*n_hidden]
        gamma_residual, beta = film.chunk(2, dim=-1)  # [B, n_hidden] each
        gamma = (1.0 + gamma_residual).unsqueeze(1)  # [B, 1, n_hidden]
        beta = beta.unsqueeze(1)  # [B, 1, n_hidden]
        return gamma * normed + beta


class EnsembleOutputHead(nn.Module):
    """K parallel output projections; final prediction is mean of K predictions.

    Shared-trunk deep-ensemble approximation (Lakshminarayanan et al. 2017):
    each head is an independent Linear with kaiming_normal init plus a small
    additive per-head noise to break symmetry and encourage divergent training
    trajectories. With K heads averaged at the output, the per-head gradient is
    1/K of the single-head gradient — heads still train but with weaker signal.

    Diagnostic: pairwise cosine similarity between flattened weight matrices
    quantifies divergence (cos < 0.95 ≈ healthy, > 0.99 ≈ collapsed).
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(n_heads)
        ])
        self.reinit()

    def reinit(self) -> None:
        for i, h in enumerate(self.heads):
            nn.init.kaiming_normal_(h.weight, mode="fan_out", nonlinearity="linear")
            with torch.no_grad():
                h.weight.add_(torch.randn_like(h.weight) * 0.02 * (i + 1))
                nn.init.zeros_(h.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(x) for h in self.heads], dim=0).mean(dim=0)


class ReFiLM(nn.Module):
    """FiLM Re-conditioner for PhysicsAttention slice logits.

    Maps log(Re) → per-head per-slice (gamma, beta) used to modulate slice
    logits before the softmax: slice_logits' = slice_logits * (1 + gamma) + beta.
    Zero-init final layer makes (gamma, beta) = (0, 0) at step 0, so the model
    starts identical to baseline; the optimiser must actively open the gate.

    Shared across all TransolverBlocks (one instance lives on the parent
    Transolver and is called once per forward). Outputs broadcast over N nodes.

    References: Perez et al. 2018 (1709.07871), Peebles & Xie 2022 (2212.09748,
    DiT adaLN-Zero), gMLP (2105.08050) on zero-init gating stability.
    """

    def __init__(self, heads: int, slice_num: int, hidden: int = 8):
        super().__init__()
        self.heads = heads
        self.slice_num = slice_num
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * heads * slice_num),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_re: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # log_re: [B, 1]  →  gamma, beta each [B, H, 1, S]
        B = log_re.shape[0]
        out = self.net(log_re)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma.reshape(B, self.heads, 1, self.slice_num)
        beta = beta.reshape(B, self.heads, 1, self.slice_num)
        return gamma, beta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, rescale_head, loader, stats, surf_weight, device) -> dict[str, float]:
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

            # Skip whole samples whose GT contains non-finite values (corrupt data:
            # test_geom_camber_cruise has one sample with +/-Inf in p). Without this,
            # Inf in y_norm propagates through (pred - y_norm)**2 (→ Inf loss) and
            # err * mask (Inf * 0 = NaN) into the MAE accumulator.
            y_sample_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            mask = mask & y_sample_finite.unsqueeze(-1)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            log_re_norm = x_norm[:, 0, 13:14]
            scale = rescale_head(log_re_norm)
            pred = model({"x": x_norm})["preds"] * scale

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
            B_now = y.shape[0]
            y_good = torch.isfinite(y.reshape(B_now, -1)).all(dim=-1)
            y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            mask_safe = mask & y_good.unsqueeze(-1)
            ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask_safe, mae_surf, mae_vol)
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
        "p_channel_weight": cfg.p_channel_weight,
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
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    grad_clip: float = 1.0
    precondition_frequency: int = 10
    max_precond_dim: int = 256
    # Per-channel loss weight for pressure (channels: 0=Ux, 1=Uy, 2=p).
    # Applied linearly to per-element Huber output (after the Huber transform),
    # so gradient on the p channel scales by exactly p_channel_weight regardless
    # of Huber regime. Numerator-only weighting; ||y||^2 denominator unchanged.
    p_channel_weight: float = 5.0
    # Re-conditional LayerNorm affine (γ/β) at every block's LN injection point.
    # Adds 3 shared FiLM(log Re) → (γ_residual, β) modules (~250 params each).
    # Zero-init reproduces baseline LN at step 0 (γ=1, β=0).
    re_conditional_layernorm: bool = False
    re_ln_hidden_film: int = 8
    # K=ensemble_heads parallel output projections, averaged at output for
    # deep-ensemble-style variance reduction (Lakshminarayanan 2017). 1 = baseline
    # single head. Heads share the trunk so per-head extra cost is negligible.
    ensemble_heads: int = 1


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
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    re_conditional_layernorm=cfg.re_conditional_layernorm,
    re_ln_hidden_film=cfg.re_ln_hidden_film,
    ensemble_heads=cfg.ensemble_heads,
)

model = Transolver(**model_config).to(device)
rescale_head = ReScaleHead(hidden=32, out_channels=3).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_params_head = sum(p.numel() for p in rescale_head.parameters())
n_params_film = sum(p.numel() for p in model.re_film.parameters())
n_params_re_ln = (
    sum(p.numel() for p in model.re_ln_1.parameters())
    + sum(p.numel() for p in model.re_ln_2.parameters())
    + sum(p.numel() for p in model.re_ln_3.parameters())
) if cfg.re_conditional_layernorm else 0
# Extract ensemble-head parameter count for visibility (last block's mlp2 final element).
_last_proj = model.blocks[-1].mlp2[-1]
n_params_ensemble = sum(p.numel() for p in _last_proj.parameters()) if isinstance(_last_proj, EnsembleOutputHead) else 0
print(
    f"Model: Transolver ({n_params/1e6:.2f}M params, incl. ReFiLM {n_params_film}, "
    f"ReCondLN {n_params_re_ln}, EnsembleHead(K={cfg.ensemble_heads}) {n_params_ensemble}) "
    f"+ ReScaleHead ({n_params_head} params)"
)

# Keep a reference to the uncompiled module for diagnostic forward passes
# (slice entropy capture). Diagnostics toggle a Python flag that would otherwise
# force torch.compile to recompile.
uncompiled_model = model

# torch.compile for forward/backward throughput. mode="default" + dynamic=True:
# pad_collate yields variable per-batch shapes (mesh nodes 74K-242K, B=4), so
# CUDA Graph modes ("reduce-overhead"/"max-autotune") would recompile per shape.
# A single dynamic-shape graph is the only viable option here.
torch_compile_mode = "default"
torch_compile_dynamic = True
print(f"Compiling model: mode={torch_compile_mode!r}, dynamic={torch_compile_dynamic}")
model = torch.compile(model, mode=torch_compile_mode, dynamic=torch_compile_dynamic)


def capture_slice_entropy(uncompiled, val_loader, stats_, device_):
    """Capture per-block per-head mean slice entropy on one val batch.

    Returns: list[list[float]] — outer = blocks, inner = heads (length H per block).
    Uses the uncompiled model to avoid torch.compile recompilation from the
    Python flag flip. Also returns the gamma/beta running stats observed on that batch.
    """
    attns = [b.attn for b in uncompiled.blocks]
    for a in attns:
        a._diag_capture = True
        a._last_slice_entropy = None
    was_training = uncompiled.training
    uncompiled.eval()
    captured: list | None = None
    gamma_stats = beta_stats = None
    try:
        with torch.no_grad():
            for x, _y, _is_surface, mask in val_loader:
                x = x.to(device_, non_blocking=True)
                mask = mask.to(device_, non_blocking=True)
                x_norm = (x - stats_["x_mean"]) / stats_["x_std"]
                log_re = x_norm[:, 0, 13:14]
                gamma, beta = uncompiled.re_film(log_re)
                gamma_stats = {
                    "mean": gamma.mean().item(),
                    "std": gamma.std().item(),
                    "absmax": gamma.abs().max().item(),
                }
                beta_stats = {
                    "mean": beta.mean().item(),
                    "std": beta.std().item(),
                    "absmax": beta.abs().max().item(),
                }
                _ = uncompiled({"x": x_norm})
                captured = [
                    [float(v) for v in a._last_slice_entropy.tolist()]
                    if a._last_slice_entropy is not None else []
                    for a in attns
                ]
                break
    finally:
        for a in attns:
            a._diag_capture = False
            a._last_slice_entropy = None
        if was_training:
            uncompiled.train()
    return captured or [], gamma_stats, beta_stats

optimizer = SOAP(
    list(model.parameters()) + list(rescale_head.parameters()),
    lr=cfg.lr,
    betas=(0.95, 0.95),
    weight_decay=cfg.weight_decay,
    precondition_frequency=cfg.precondition_frequency,
    max_precond_dim=cfg.max_precond_dim,
)
SCHEDULER_T_MAX = 28  # epoch 1 measured at 73s (compile + train + val) vs 108s baseline (~32% speedup); steady-state ~60-65s/epoch projects ~27-28 epochs in 30 min
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=1e-5)
scaler = GradScaler()

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

ch_weights = torch.tensor([1.0, 1.0, cfg.p_channel_weight], device=device).view(1, 1, 3)
print(f"Per-channel loss weights (Ux, Uy, p): {ch_weights.flatten().tolist()}")

best_avg_surf_p = float("inf")
best_metrics: dict = {}
slice_entropy_ep1: list[list[float]] | None = None
slice_entropy_film_stats_ep1: dict | None = None
slice_entropy_last: list[list[float]] | None = None
slice_entropy_film_stats_last: dict | None = None
last_completed_epoch: int = 0
diag_split_name = next(iter(val_loaders.keys()))  # use one val split as the diag batch source
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    rescale_head.train()
    epoch_vol = epoch_surf = 0.0
    epoch_grad_norm_sum = 0.0
    epoch_grad_norm_max = 0.0
    epoch_grad_clipped = 0
    epoch_l2_frac = 0.0
    n_batches = 0
    # Unweighted per-channel Huber loss diagnostics (sum / count, batch-summed).
    surf_huber_per_ch_sum = torch.zeros(3, dtype=torch.float64, device=device)
    vol_huber_per_ch_sum = torch.zeros(3, dtype=torch.float64, device=device)
    surf_count_total = torch.zeros(1, dtype=torch.float64, device=device)
    vol_count_total = torch.zeros(1, dtype=torch.float64, device=device)
    scale_sum = torch.zeros(3, dtype=torch.float64, device=device)
    scale_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
    log_re_sum = 0.0
    log_re_sq_sum = 0.0
    scale_log_re_cross = torch.zeros(3, dtype=torch.float64, device=device)
    scale_n = 0
    # ReConditionalLayerNorm γ/β diagnostics (per-role: 'attn', 'ffn', 'out').
    # Track absmax per role, plus per-sample |γ| mean / |β| mean accumulators
    # for correlation with log_Re.
    re_ln_diag = None
    if cfg.re_conditional_layernorm:
        re_ln_diag = {role: {
            "gamma_absmax": 0.0,
            "beta_absmax": 0.0,
            "g_sum": 0.0,
            "b_sum": 0.0,
            "g_sq_sum": 0.0,
            "b_sq_sum": 0.0,
            "g_logre_cross": 0.0,
            "b_logre_cross": 0.0,
            "n": 0,
        } for role in ("attn", "ffn", "out")}

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            log_re_norm = x_norm[:, 0, 13:14]
            scale = rescale_head(log_re_norm)
            pred = model({"x": x_norm})["preds"] * scale

            # Huber-shaped per-node residuals in normalized space.
            # Replaces the squared error in the relative-L2 numerator, combining
            # per-sample relative scaling with per-node outlier capping.
            HUBER_DELTA = 0.1
            residual = pred - y_norm
            abs_residual = residual.abs()
            sq_err = torch.where(
                abs_residual <= HUBER_DELTA,
                0.5 * residual ** 2,
                HUBER_DELTA * (abs_residual - 0.5 * HUBER_DELTA),
            )
            vol_mask = (mask & ~is_surface).unsqueeze(-1)
            surf_mask = (mask & is_surface).unsqueeze(-1)

            # Apply per-channel weights linearly to the per-element Huber output
            # (after Huber transform). This gives exact w× gradient amplification
            # on the pressure channel regardless of Huber regime — applying weights
            # to the raw residual would distort the δ threshold per-channel.
            sq_err_weighted = sq_err * ch_weights

            # Per-sample relative Huber-L2 (weighted Huber numerator, unweighted
            # ||y||^2 denominator). Each sample contributes Σ_ch w_ch · huber(pred - y)
            # / ||y||^2 — keeping the denominator unweighted preserves the
            # cross-sample relative scaling while amplifying the p gradient.
            vol_sq = (sq_err_weighted * vol_mask).sum(dim=(1, 2))
            vol_denom = (y_norm ** 2 * vol_mask).sum(dim=(1, 2)).clamp(min=1e-6)
            surf_sq = (sq_err_weighted * surf_mask).sum(dim=(1, 2))
            surf_denom = (y_norm ** 2 * surf_mask).sum(dim=(1, 2)).clamp(min=1e-6)
            vol_loss = (vol_sq / vol_denom).mean()
            surf_loss = (surf_sq / surf_denom).mean()
            loss = vol_loss + cfg.surf_weight * surf_loss

        # Diagnostic: track fraction of residuals in L2 (quadratic) regime
        # and per-channel unweighted Huber loss (so we can compare the raw
        # pressure-channel error signal against velocity channels).
        with torch.no_grad():
            valid = mask.unsqueeze(-1).expand_as(abs_residual)
            n_valid = valid.sum().clamp(min=1)
            n_l2 = ((abs_residual <= HUBER_DELTA) & valid).sum()
            l2_frac_batch = (n_l2.float() / n_valid.float()).item()
            sq_err_f64 = sq_err.detach().to(torch.float64)
            surf_huber_per_ch_sum += (sq_err_f64 * surf_mask).sum(dim=(0, 1))
            vol_huber_per_ch_sum += (sq_err_f64 * vol_mask).sum(dim=(0, 1))
            surf_count_total += surf_mask.sum().to(torch.float64)
            vol_count_total += vol_mask.sum().to(torch.float64)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(rescale_head.parameters()),
                cfg.grad_clip,
            )
            grad_norm_val = float(grad_norm)
            epoch_grad_norm_sum += grad_norm_val
            if grad_norm_val > epoch_grad_norm_max:
                epoch_grad_norm_max = grad_norm_val
            if grad_norm_val > cfg.grad_clip:
                epoch_grad_clipped += 1
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            sc = scale.squeeze(1).to(torch.float64)              # [B, 3]
            lr = log_re_norm.squeeze(1).to(torch.float64)        # [B]
            scale_sum += sc.sum(dim=0)
            scale_sq_sum += (sc ** 2).sum(dim=0)
            log_re_sum += lr.sum().item()
            log_re_sq_sum += (lr ** 2).sum().item()
            scale_log_re_cross += (sc * lr.unsqueeze(1)).sum(dim=0)
            scale_n += sc.shape[0]

            # ReConditionalLayerNorm γ/β diagnostics (call the FiLM MLP directly
            # — same operation as inside the model forward; tiny extra compute).
            if re_ln_diag is not None:
                role_to_rcln = (
                    ("attn", uncompiled_model.re_ln_1),
                    ("ffn", uncompiled_model.re_ln_2),
                    ("out", uncompiled_model.re_ln_3),
                )
                for role, rcln in role_to_rcln:
                    film = rcln.film_net(log_re_norm)
                    g_r, b_v = film.chunk(2, dim=-1)  # [B, n_hidden] each
                    g_per = g_r.abs().mean(dim=-1).to(torch.float64)  # [B]
                    b_per = b_v.abs().mean(dim=-1).to(torch.float64)  # [B]
                    d = re_ln_diag[role]
                    d["gamma_absmax"] = max(d["gamma_absmax"], g_r.abs().max().item())
                    d["beta_absmax"] = max(d["beta_absmax"], b_v.abs().max().item())
                    d["g_sum"] += g_per.sum().item()
                    d["b_sum"] += b_per.sum().item()
                    d["g_sq_sum"] += (g_per ** 2).sum().item()
                    d["b_sq_sum"] += (b_per ** 2).sum().item()
                    d["g_logre_cross"] += (g_per * lr).sum().item()
                    d["b_logre_cross"] += (b_per * lr).sum().item()
                    d["n"] += g_per.shape[0]

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        epoch_l2_frac += l2_frac_batch
        n_batches += 1

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_l2_frac /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    rescale_head.eval()
    split_metrics = {
        name: evaluate_split(model, rescale_head, loader, stats, cfg.surf_weight, device)
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
        torch.save(
            {"model": model.state_dict(), "rescale_head": rescale_head.state_dict()},
            model_path,
        )
        tag = " *"

    # Re-conditioned scale diagnostics (per-channel mean/std across epoch + corr w/ log(Re)).
    if scale_n > 0:
        scale_mean_t = scale_sum / scale_n
        scale_var_t = (scale_sq_sum / scale_n - scale_mean_t ** 2).clamp(min=0)
        scale_std_t = scale_var_t.sqrt()
        scale_mean = scale_mean_t.tolist()
        scale_std = scale_std_t.tolist()
        lr_mean = log_re_sum / scale_n
        lr_var = max(log_re_sq_sum / scale_n - lr_mean ** 2, 0.0)
        lr_std = lr_var ** 0.5
        if lr_std > 1e-8:
            cov = scale_log_re_cross / scale_n - scale_mean_t * lr_mean
            corr = (cov / scale_std_t.clamp(min=1e-8) / lr_std).tolist()
        else:
            corr = [0.0, 0.0, 0.0]
    else:
        scale_mean = scale_std = corr = [0.0, 0.0, 0.0]

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    grad_norm_mean = epoch_grad_norm_sum / max(n_batches, 1) if cfg.grad_clip > 0 else 0.0
    grad_clip_frac = epoch_grad_clipped / max(n_batches, 1) if cfg.grad_clip > 0 else 0.0
    surf_count_safe = surf_count_total.clamp(min=1)
    vol_count_safe = vol_count_total.clamp(min=1)
    surf_huber_per_ch = (surf_huber_per_ch_sum / surf_count_safe).tolist()
    vol_huber_per_ch = (vol_huber_per_ch_sum / vol_count_safe).tolist()

    # ReConditionalLayerNorm γ/β summary — absmax + correlation with log(Re).
    re_ln_summary: dict | None = None
    if re_ln_diag is not None:
        re_ln_summary = {}
        for role, d in re_ln_diag.items():
            n = max(d["n"], 1)
            g_mean = d["g_sum"] / n
            b_mean = d["b_sum"] / n
            g_var = max(d["g_sq_sum"] / n - g_mean ** 2, 0.0)
            b_var = max(d["b_sq_sum"] / n - b_mean ** 2, 0.0)
            g_std = g_var ** 0.5
            b_std = b_var ** 0.5
            if scale_n > 0 and lr_std > 1e-8 and g_std > 1e-12:
                g_cov = d["g_logre_cross"] / n - g_mean * lr_mean
                g_corr = g_cov / (g_std * lr_std)
            else:
                g_corr = 0.0
            if scale_n > 0 and lr_std > 1e-8 and b_std > 1e-12:
                b_cov = d["b_logre_cross"] / n - b_mean * lr_mean
                b_corr = b_cov / (b_std * lr_std)
            else:
                b_corr = 0.0
            re_ln_summary[role] = {
                "gamma_residual_absmax": d["gamma_absmax"],
                "beta_absmax": d["beta_absmax"],
                "gamma_residual_mean_abs": g_mean,
                "beta_mean_abs": b_mean,
                "corr_gamma_logre": float(g_corr),
                "corr_beta_logre": float(b_corr),
            }

    # Slice entropy diagnostic — capture at epoch 1 and on every later epoch
    # (the loop overwrites `last`, so after training ends, last reflects the
    # final completed epoch even if the run is cut short by timeout).
    epoch_slice_entropy: list[list[float]] | None = None
    epoch_gamma_stats: dict | None = None
    epoch_beta_stats: dict | None = None
    if (epoch + 1) == 1 or (epoch + 1) >= max(SCHEDULER_T_MAX - 2, 1):
        epoch_slice_entropy, epoch_gamma_stats, epoch_beta_stats = capture_slice_entropy(
            uncompiled_model, val_loaders[diag_split_name], stats, device,
        )
        if (epoch + 1) == 1:
            slice_entropy_ep1 = epoch_slice_entropy
            slice_entropy_film_stats_ep1 = {"gamma": epoch_gamma_stats, "beta": epoch_beta_stats}
        slice_entropy_last = epoch_slice_entropy
        slice_entropy_film_stats_last = {"gamma": epoch_gamma_stats, "beta": epoch_beta_stats}
    last_completed_epoch = epoch + 1

    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "train/lr": current_lr,
        "scheduler": "cosine_annealing_lr",
        "scheduler_T_max": SCHEDULER_T_MAX,
        "scheduler_eta_min": 1e-5,
        "amp_dtype": "bfloat16",
        "torch_compile_mode": torch_compile_mode,
        "torch_compile_dynamic": torch_compile_dynamic,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/grad_norm_mean": grad_norm_mean,
        "train/grad_norm_max": epoch_grad_norm_max,
        "train/grad_clip_frac": grad_clip_frac,
        "train/huber_l2_frac": epoch_l2_frac,
        "train/surf_huber_per_ch": surf_huber_per_ch,
        "train/vol_huber_per_ch": vol_huber_per_ch,
        "p_channel_weight": cfg.p_channel_weight,
        "huber_delta": 0.1,
        "loss_type": "huber_relative_l2_channel_weighted",
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "rescale/scale_mean": scale_mean,
        "rescale/scale_std": scale_std,
        "rescale/scale_logre_corr": corr,
        "film/slice_entropy_per_head": epoch_slice_entropy,
        "film/gamma_stats": epoch_gamma_stats,
        "film/beta_stats": epoch_beta_stats,
        "re_ln/per_role": re_ln_summary,
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  lr={current_lr:.6f}  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f} l2_frac={epoch_l2_frac:.3f}]  "
        f"grad[mean={grad_norm_mean:.3f} max={epoch_grad_norm_max:.3f} clipped={grad_clip_frac:.2f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    print(
        f"    HuberPerCh (unweighted)  surf[Ux={surf_huber_per_ch[0]:.5f} "
        f"Uy={surf_huber_per_ch[1]:.5f} p={surf_huber_per_ch[2]:.5f}]  "
        f"vol[Ux={vol_huber_per_ch[0]:.5f} "
        f"Uy={vol_huber_per_ch[1]:.5f} p={vol_huber_per_ch[2]:.5f}]"
    )
    print(
        f"    ReScaleHead  mean={[f'{v:.3f}' for v in scale_mean]}  "
        f"std={[f'{v:.3f}' for v in scale_std]}  "
        f"corr_logRe={[f'{v:+.3f}' for v in corr]}"
    )
    if epoch_slice_entropy is not None:
        flat_ent = [v for block_ent in epoch_slice_entropy for v in block_ent]
        if flat_ent:
            mean_ent = sum(flat_ent) / len(flat_ent)
            print(
                f"    SliceEntropy(diag)  mean={mean_ent:.3f}  "
                f"gamma[abs_max={epoch_gamma_stats['absmax']:.3f}, std={epoch_gamma_stats['std']:.3f}]  "
                f"beta[abs_max={epoch_beta_stats['absmax']:.3f}, std={epoch_beta_stats['std']:.3f}]"
            )
    if re_ln_summary is not None:
        for role, s in re_ln_summary.items():
            print(
                f"    ReCondLN[{role:<4s}]  "
                f"gamma_res[absmax={s['gamma_residual_absmax']:.4f}, mean|.|={s['gamma_residual_mean_abs']:.4f}]  "
                f"beta[absmax={s['beta_absmax']:.4f}, mean|.|={s['beta_mean_abs']:.4f}]  "
                f"corr(|γ|,logRe)={s['corr_gamma_logre']:+.3f}  corr(|β|,logRe)={s['corr_beta_logre']:+.3f}"
            )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# Always capture slice entropy on the actually-completed final epoch, so the
# "last" snapshot reflects reality even if training was cut short by the
# timeout before the SCHEDULER_T_MAX-2 trigger range was reached.
if last_completed_epoch > 0 and last_completed_epoch != 1:
    _ent, _g, _b = capture_slice_entropy(
        uncompiled_model, val_loaders[diag_split_name], stats, device,
    )
    slice_entropy_last = _ent
    slice_entropy_film_stats_last = {"gamma": _g, "beta": _b}

# Persist epoch-1 vs last-epoch slice-entropy comparison as a single record so it's
# easy to surface in the results comment.
append_metrics_jsonl(metrics_jsonl_path, {
    "event": "film_slice_entropy_summary",
    "n_params_film": n_params_film,
    "ep1": {
        "slice_entropy_per_head": slice_entropy_ep1,
        "film_stats": slice_entropy_film_stats_ep1,
    },
    "last": {
        "epoch": last_completed_epoch,
        "slice_entropy_per_head": slice_entropy_last,
        "film_stats": slice_entropy_film_stats_last,
    },
})

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    rescale_head.load_state_dict(ckpt["rescale_head"])
    model.eval()
    rescale_head.eval()

    # Ensemble-head divergence diagnostic on the best checkpoint state. The
    # hypothesis predicts heads diverge during training (init noise breaks
    # symmetry; per-head Linear gets independent direction). cos < 0.95 ≈
    # healthy divergence, > 0.99 ≈ collapsed (no ensemble effect).
    if cfg.ensemble_heads > 1:
        ensemble_head_module = uncompiled_model.blocks[-1].mlp2[-1]
        if isinstance(ensemble_head_module, EnsembleOutputHead):
            with torch.no_grad():
                weights = [h.weight.detach().flatten() for h in ensemble_head_module.heads]
                biases = [h.bias.detach() for h in ensemble_head_module.heads]
            n_heads = len(weights)
            pairwise_cos = []
            for i in range(n_heads):
                for j in range(i + 1, n_heads):
                    cos = F.cosine_similarity(
                        weights[i].unsqueeze(0), weights[j].unsqueeze(0)
                    ).item()
                    pairwise_cos.append({"head_i": i, "head_j": j, "cosine_similarity": cos})
            per_head_w_norm = [w.norm().item() for w in weights]
            per_head_b_norm = [b.norm().item() for b in biases]
            ensemble_diag = {
                "event": "ensemble_head_diversity",
                "n_heads": n_heads,
                "pairwise_cosine": pairwise_cos,
                "per_head_weight_l2_norm": per_head_w_norm,
                "per_head_bias_l2_norm": per_head_b_norm,
                "best_epoch": best_metrics["epoch"],
            }
            append_metrics_jsonl(metrics_jsonl_path, ensemble_diag)
            print(json.dumps(ensemble_diag))
            cos_vals = [r["cosine_similarity"] for r in pairwise_cos]
            print(
                f"  EnsembleHeads(K={n_heads})  cos_range=[{min(cos_vals):+.3f}, "
                f"{max(cos_vals):+.3f}]  weight_norms={[f'{v:.3f}' for v in per_head_w_norm]}"
            )

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
            name: evaluate_split(model, rescale_head, loader, stats, cfg.surf_weight, device)
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
