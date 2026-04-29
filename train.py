"""Train a Transolver / Transolver++ surrogate on TandemFoilSet.

Four validation tracks with one pinned test track each:
  val_single_in_dist      / test_single_in_dist      — in-distribution sanity
  val_geom_camber_rc      / test_geom_camber_rc      — unseen front foil (raceCar)
  val_geom_camber_cruise  / test_geom_camber_cruise  — unseen front foil (cruise)
  val_re_rand             / test_re_rand             — stratified Re holdout

Primary ranking metric is ``val_avg/mae_surf_p`` — equal-weight mean surface
pressure MAE across the four validation splits, in the original (denormalized)
target space. Train/val/test MAE all flow through ``data.scoring`` so the
numbers are produced identically.

Usage:
  python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]

This version exposes model and training hyperparameters and adds two
Transolver++ ablation flags (Ada-Temp, Rep-Slice) — see
arxiv 2502.02414 (Transolver++).
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
# Transolver / Transolver++ model
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
    """Physics-aware attention for irregular meshes.

    Modes:
      base     — vanilla Transolver (softmax over slice logits with shared temp)
      ada_temp — Transolver++ Ada-Temp: per-point temperature τ = τ₀ + Linear(x_mid)
      rep_slice — Transolver++ Rep-Slice: Gumbel-Softmax reparameterization
                  (in train mode only; eval falls back to soft argmax for stability).

    Default behavior (`use_ada_temp=False, use_rep_slice=False`) is bit-identical
    to the original Transolver baseline.
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        use_ada_temp: bool = False,
        use_rep_slice: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.use_ada_temp = use_ada_temp
        self.use_rep_slice = use_rep_slice

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        if use_ada_temp:
            # Per-point scalar correction: τ = τ_global + Linear(x_mid). Initialised to ~0.
            self.ada_temp = nn.Linear(dim_head, 1)
            nn.init.zeros_(self.ada_temp.weight)
            nn.init.zeros_(self.ada_temp.bias)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # x: [B, N, dim]
        # mask: [B, N] bool, True for real nodes; padding will be zeroed out
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

        slice_logits = self.in_project_slice(x_mid)  # [B, H, N, S]
        # Effective temperature. Clamped to a positive minimum (>=0.05) so that
        # division-by-zero / sign flips can't produce NaN — this is the failure
        # mode that broke our first Transolver++ ablations.
        tau = self.temperature
        if self.use_ada_temp:
            tau = tau + self.ada_temp(x_mid)  # broadcast: [B,H,N,1]
        tau = tau.clamp(min=0.05)

        # Gumbel-Softmax reparameterization (Rep-Slice). Train-only stochasticity.
        # Clip the Gumbel tail to bound noise magnitude (numerical stability;
        # standard practice for Gumbel-Softmax with low τ).
        if self.use_rep_slice and self.training:
            eps = torch.rand_like(slice_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(eps))
            gumbel = gumbel.clamp(min=-5.0, max=5.0)
            slice_logits = slice_logits + gumbel

        slice_weights = self.softmax(slice_logits / tau)

        if mask is not None:
            # Zero out padding contributions to slice tokens (they never had real x signal anyway)
            m = mask[:, None, :, None].to(slice_weights.dtype)  # [B,1,N,1]
            slice_weights = slice_weights * m

        slice_norm = slice_weights.sum(2)  # [B, H, S]
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_ada_temp: bool = False,
        use_rep_slice: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_ada_temp=use_ada_temp,
            use_rep_slice=use_rep_slice,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, mask=None):
        fx = self.attn(self.ln_1(fx), mask=mask) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class FourierFeatures(nn.Module):
    """Random-Fourier positional encoding for 2D (x, z) coordinates.

    γ(p) = [sin(2π Bp), cos(2π Bp)] with B ∈ R^{F×2} fixed.

    Output dim = 2 * num_frequencies. Used to inject high-frequency spatial
    information that helps capture sharp boundary-layer pressure gradients
    (cf. MARIO, arxiv 2505.14704).
    """

    def __init__(self, num_frequencies: int = 16, scale: float = 5.0, seed: int = 0):
        super().__init__()
        # Deterministic frequency matrix. Stored as buffer so it's part of state_dict.
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(num_frequencies, 2, generator=g) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num_frequencies

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [..., 2] in any space (we use *normalized* x_norm[:, :, :2]).
        proj = pos @ self.B.t() * 2 * math.pi  # [..., F]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        use_ada_temp: bool = False,
        use_rep_slice: bool = False,
        fourier_freq: int = 0,
        fourier_scale: float = 5.0,
        output_fields: list[str] | None = None,
        output_dims: list[int] | None = None,
    ):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.fourier_freq = fourier_freq
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        # Optional Fourier positional encoding from x[:, :, :2] (node coordinates).
        if fourier_freq > 0:
            self.fourier = FourierFeatures(num_frequencies=fourier_freq, scale=fourier_scale)
            extra_dim = self.fourier.out_dim
        else:
            self.fourier = None
            extra_dim = 0

        in_dim = fun_dim + space_dim + extra_dim
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1),
                use_ada_temp=use_ada_temp,
                use_rep_slice=use_rep_slice,
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

    def forward(self, data, mask: torch.Tensor | None = None, **kwargs):
        x = data["x"]
        if self.fourier is not None:
            # x[:, :, :2] are the (already-normalized) coordinates
            pos = x[:, :, :2]
            x = torch.cat([x, self.fourier(pos)], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            if isinstance(block, TransolverBlock) and not block.last_layer:
                fx = block(fx, mask=mask)
            else:
                fx = block(fx, mask=mask)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_loss(
    pred: torch.Tensor,
    y_norm: torch.Tensor,
    mask: torch.Tensor,
    is_surface: torch.Tensor,
    surf_weight: float,
    surf_p_weight: float | None = None,
):
    """Squared-error loss split between volume and surface, with optional
    extra weight on the pressure (channel 2) on surface nodes.

    Returns (total_loss, vol_loss_scalar, surf_loss_scalar).
    """
    sq_err = (pred - y_norm) ** 2
    vol_mask = mask & ~is_surface
    surf_mask = mask & is_surface

    vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
    if surf_p_weight is not None:
        # Channel-weighted surface loss: extra weight on pressure (channel 2).
        ch_w = torch.tensor(
            [1.0, 1.0, surf_p_weight], device=pred.device, dtype=pred.dtype
        )
        surf_loss = (sq_err * ch_w * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    else:
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    total = vol_loss + surf_weight * surf_loss
    return total, vol_loss, surf_loss


def evaluate_split(model, loader, stats, surf_weight, device, amp_dtype=None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    Workaround for a known scoring.py edge case: at least one cruise test sample
    contains inf in its ground-truth y (test_geom_camber_cruise/000020.pt has
    761 inf values). ``accumulate_batch`` already tries to skip samples with
    non-finite y, but ``err = (pred - y).abs()`` is computed *before* the skip
    mask is applied, and ``inf * 0`` evaluates to ``nan`` in IEEE float — which
    then poisons the entire MAE sum and produces ``test_avg/mae_surf_p = nan``.
    The fix here is to filter bad samples out of the batch *before* calling
    ``accumulate_batch`` so the inf never enters the err computation. The
    semantics match what ``scoring.py`` claims to do (per-sample skip on
    non-finite y).
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
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model({"x": x_norm}, mask=mask)["preds"]
                pred = pred.float()
            else:
                pred = model({"x": x_norm}, mask=mask)["preds"]

            # ---- normalized-space loss accounting (volume + surface) ----
            sq_err = (pred - y_norm) ** 2
            # NaN/Inf-safe: only count valid (mask=True) and finite y nodes.
            y_finite_node = torch.isfinite(y).all(dim=-1)  # [B, N]
            valid = mask & y_finite_node
            vol_mask = valid & ~is_surface
            surf_mask = valid & is_surface
            # Replace any non-finite predictions with 0 in the masked-out positions
            # so they cannot poison the sum. Since vol_mask/surf_mask are False
            # there, this multiplication doesn't affect the metric — but it
            # avoids inf*0 = nan.
            sq_err_safe = torch.where(valid.unsqueeze(-1), sq_err, torch.zeros_like(sq_err))
            vol_loss_sum += (
                (sq_err_safe * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err_safe * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            # ---- denormalized-space MAE accumulation (organizer-shared) ----
            pred_orig = pred * stats["y_std"] + stats["y_mean"]

            # Filter samples with any non-finite y BEFORE handing to accumulate_batch.
            # This preserves the per-sample skip semantics of scoring.py while
            # preventing inf from polluting the err computation inside.
            B = y.shape[0]
            y_finite_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
            if y_finite_sample.any():
                good = y_finite_sample.nonzero(as_tuple=False).flatten()
                pred_g = pred_orig[good]
                y_g = y[good]
                is_g = is_surface[good]
                mk_g = mask[good]
                ds, dv = accumulate_batch(pred_g, y_g, is_g, mk_g, mae_surf, mae_vol)
                n_surf += ds
                n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {
        "vol_loss": vol_loss,
        "surf_loss": surf_loss,
        "loss": vol_loss + surf_weight * surf_loss,
    }
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def _sanitize_artifact_token(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in s)
    return out.strip("-_.") or "run"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_model_artifact(
    run, model_path: Path, model_dir: Path, cfg: "Config",
    best_metrics: dict, best_avg_surf_p: float,
    test_metrics: dict | None, test_avg: dict | None,
    n_params: int, model_config: dict,
) -> None:
    if cfg.wandb_name:
        base = _sanitize_artifact_token(cfg.wandb_name)
    elif cfg.agent:
        base = _sanitize_artifact_token(cfg.agent)
    else:
        base = "tandemfoil"
    artifact_name = f"model-{base}-{run.id}"

    metadata: dict = {
        "run_id": run.id,
        "run_name": run.name,
        "agent": cfg.agent,
        "wandb_name": cfg.wandb_name,
        "wandb_group": cfg.wandb_group,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "model_config": model_config,
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "surf_p_weight": cfg.surf_p_weight,
        "epochs_configured": cfg.epochs,
        "amp": cfg.amp,
        "warmup_frac": cfg.warmup_frac,
        "grad_clip": cfg.grad_clip,
        "use_ada_temp": cfg.use_ada_temp,
        "use_rep_slice": cfg.use_rep_slice,
        "fourier_freq": cfg.fourier_freq,
        "seed": cfg.seed,
    }

    description = (
        f"Transolver checkpoint — best val_avg/mae_surf_p = {best_avg_surf_p:.4f} "
        f"at epoch {best_metrics['epoch']}"
    )

    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        metadata["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                metadata[f"test/{split_name}/mae_surf_p"] = m["mae_surf_p"]
        description += f" | test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}"

    artifact = wandb.Artifact(
        name=artifact_name, type="model",
        description=description, metadata=metadata,
    )
    artifact.add_file(str(model_path), name="checkpoint.pt")
    config_yaml = model_dir / "config.yaml"
    if config_yaml.exists():
        artifact.add_file(str(config_yaml), name="config.yaml")

    aliases = ["best", f"epoch-{best_metrics['epoch']}"]
    run.log_artifact(artifact, aliases=aliases)
    print(f"\nLogged model artifact '{artifact_name}' (aliases: {', '.join(aliases)})")


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
    # Optimization
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    surf_p_weight: float = 1.0  # extra multiplier on pressure channel inside surf_loss
    epochs: int = 50
    warmup_frac: float = 0.0  # linear warmup as a fraction of total steps
    grad_clip: float = 0.0  # 0 disables
    optimizer: str = "adamw"  # adamw | adam
    schedule: str = "cosine"  # cosine | const
    amp: str = "none"  # none | bf16 | fp16
    seed: int = 0

    # Model
    n_hidden: int = 128
    n_layers: int = 5
    n_head: int = 4
    slice_num: int = 64
    mlp_ratio: int = 2
    dropout: float = 0.0
    use_ada_temp: bool = False
    use_rep_slice: bool = False
    fourier_freq: int = 0
    fourier_scale: float = 5.0

    # Train-time subsampling: random K nodes per sample for forward+loss only.
    # Eval always runs on the full mesh (the metric is full-mesh).
    # 0 disables. surf_oversample: fraction of K reserved for surface nodes
    # (rest is uniform over volume). Enabled implicitly when subsample > 0.
    subsample: int = 0
    surf_oversample: float = 0.5  # fraction of subsample budget for surface nodes

    # Train-time augmentation. Stochastic y-axis reflection that negates the
    # z coordinate (x[:, 1]), the foil-1 AoA (x[:, 14]), the foil-2 AoA
    # (x[:, 18]), the y-velocity target (y[:, 1]) and the stagger (x[:, 23]).
    # Pressure (y[:, 2]) is invariant. Applied with prob aug_yflip per sample.
    aug_yflip: float = 0.0  # probability of applying flip per training sample

    # I/O & meta
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum (Chen et al. 2023, arXiv 2302.06675).

    Decoupled-WD sign-momentum optimizer; reported to give comparable or
    better results than AdamW at lower learning rates, with smaller memory
    footprint (one moment buffer instead of two). Used by AB-UPT
    (arXiv 2502.09692) and the 3D transonic wing benchmark
    (arXiv 2511.21474) for Transolver-class CFD surrogates.
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # update direction = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.data.add_(update, alpha=-lr)
                # Update the running momentum with beta2 (decoupled from update).
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def yflip_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    p_flip: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastic y-axis (z-coordinate) reflection per sample in a batch.

    Physics: reflecting z → -z is a valid symmetry of the airfoil flow if we
    also negate (a) the foil AoAs (radians), (b) the y-velocity target Uy,
    and (c) any geometric quantities tied to chord-direction sign. NACA shape
    parameters (camber/position/thickness) are scalar magnitudes — invariant.
    Pressure is a scalar — invariant. Re is invariant. Gap is a magnitude —
    invariant. Stagger sign depends on convention; we negate it conservatively.

    Affects only valid (mask=True) nodes; padding stays untouched.

    Modifies in place; returns the same tensors for chained ops.
    """
    if p_flip <= 0:
        return x, y
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < p_flip
    if not flip.any():
        return x, y
    f_idx = flip.nonzero(as_tuple=False).flatten()
    # Avoid mask handling complications: only flip valid mesh entries.
    # Fields to negate (indices in x):
    #   1: z coordinate
    #   14: foil 1 AoA (radians)
    #   18: foil 2 AoA (radians)
    #   23: stagger (sign-dependent geometric offset)
    # Fields in y:
    #   1: Uy (y-velocity)
    for b in f_idx.tolist():
        m = mask[b]
        x[b, m, 1] = -x[b, m, 1]
        x[b, m, 14] = -x[b, m, 14]
        x[b, m, 18] = -x[b, m, 18]
        x[b, m, 23] = -x[b, m, 23]
        y[b, m, 1] = -y[b, m, 1]
    return x, y


def subsample_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    is_surface: torch.Tensor,
    mask: torch.Tensor,
    n_sub: int,
    surf_oversample: float = 0.5,
):
    """Per-item random subsample of valid nodes, biased toward surface.

    Returns (x_s, y_s, surf_s, mask_s) with N = n_sub. mask_s is True for the
    selected positions (always all True unless a sample has fewer valid nodes
    than n_sub, in which case the tail is padding). Sampling without
    replacement; if a sample has fewer surface nodes than the requested quota
    we just take all surface and fill the rest from volume.
    """
    B, N_max, X_dim = x.shape
    Y_dim = y.shape[-1]
    device = x.device

    n_surf_target = int(n_sub * surf_oversample)
    n_vol_target = n_sub - n_surf_target

    out_x = x.new_zeros(B, n_sub, X_dim)
    out_y = y.new_zeros(B, n_sub, Y_dim)
    out_surf = is_surface.new_zeros(B, n_sub)
    out_mask = mask.new_zeros(B, n_sub)

    for b in range(B):
        m = mask[b]
        s = is_surface[b]
        valid = m.nonzero(as_tuple=False).flatten()
        # Pre-compute surface and volume index pools restricted to valid nodes.
        valid_surf = (m & s).nonzero(as_tuple=False).flatten()
        valid_vol = (m & ~s).nonzero(as_tuple=False).flatten()
        n_surf_avail = valid_surf.numel()
        n_vol_avail = valid_vol.numel()

        # Surface side
        if n_surf_avail >= n_surf_target:
            perm = torch.randperm(n_surf_avail, device=device)[:n_surf_target]
            chosen_surf = valid_surf[perm]
        else:
            # Take all surfaces; fill missing budget with volume below.
            chosen_surf = valid_surf
            n_vol_target_b = n_sub - chosen_surf.numel()
        # Volume side
        n_vol_target_b = n_sub - chosen_surf.numel()
        if n_vol_avail >= n_vol_target_b:
            perm = torch.randperm(n_vol_avail, device=device)[:n_vol_target_b]
            chosen_vol = valid_vol[perm]
        else:
            chosen_vol = valid_vol

        chosen = torch.cat([chosen_surf, chosen_vol], dim=0)
        n_chosen = chosen.numel()
        # Final guard: if we don't have n_sub valid nodes, repeat-last to pad.
        if n_chosen < n_sub:
            # Pad with valid index 0 (any valid) but mark mask=False for those.
            pad_idx = (valid[:1] if valid.numel() else torch.tensor([0], device=device))
            pad = pad_idx.repeat(n_sub - n_chosen)
            chosen = torch.cat([chosen, pad], dim=0)
            out_mask[b, :n_chosen] = True
        else:
            out_mask[b] = True

        out_x[b] = x[b, chosen]
        out_y[b] = y[b, chosen]
        out_surf[b] = is_surface[b, chosen]

    return out_x, out_y, out_surf, out_mask


def main():
    cfg = sp.parse(Config)
    _set_seed(cfg.seed)

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
        n_hidden=cfg.n_hidden,
        n_layers=cfg.n_layers,
        n_head=cfg.n_head,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        use_ada_temp=cfg.use_ada_temp,
        use_rep_slice=cfg.use_rep_slice,
        fourier_freq=cfg.fourier_freq,
        fourier_scale=cfg.fourier_scale,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )

    model = Transolver(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(total_steps * cfg.warmup_frac)

    if cfg.schedule == "cosine":
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        scheduler_step_per_batch = True
    elif cfg.schedule == "const":
        scheduler = None
        scheduler_step_per_batch = False
    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule}")

    amp_dtype = None
    if cfg.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.amp == "fp16":
        amp_dtype = torch.float16
    grad_scaler = torch.amp.GradScaler() if amp_dtype is torch.float16 else None

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        group=cfg.wandb_group,
        name=cfg.wandb_name,
        tags=[cfg.agent] if cfg.agent else [],
        config={
            **asdict(cfg),
            "model_config": model_config,
            "n_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": {k: len(v) for k, v in val_splits.items()},
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    for _name in VAL_SPLIT_NAMES:
        wandb.define_metric(f"{_name}/*", step_metric="global_step")
    wandb.define_metric("lr", step_metric="global_step")

    model_dir = Path(f"models/model-{run.id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "checkpoint.pt"
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(model_config, f)

    best_avg_surf_p = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
            print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        for x, y, is_surface, mask in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False, disable=True,
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Apply y-flip BEFORE subsampling so subsampling sees augmented data.
            if cfg.aug_yflip > 0:
                x, y = yflip_batch(x, y, mask, cfg.aug_yflip)

            if cfg.subsample > 0:
                x, y, is_surface, mask = subsample_batch(
                    x, y, is_surface, mask, cfg.subsample, cfg.surf_oversample
                )

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model({"x": x_norm}, mask=mask)["preds"]
                pred = pred.float()
            else:
                pred = model({"x": x_norm}, mask=mask)["preds"]

            loss, vol_loss, surf_loss = compute_loss(
                pred, y_norm, mask, is_surface, cfg.surf_weight,
                surf_p_weight=cfg.surf_p_weight if cfg.surf_p_weight != 1.0 else None,
            )

            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

            global_step += 1
            wandb.log({
                "train/loss": loss.item(),
                "train/vol_loss": vol_loss.item(),
                "train/surf_loss": surf_loss.item(),
                "global_step": global_step,
            })

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        epoch_vol /= max(n_batches, 1)
        epoch_surf /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        split_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_dtype=amp_dtype)
            for name, loader in val_loaders.items()
        }
        val_avg = aggregate_splits(split_metrics)
        avg_surf_p = val_avg["avg/mae_surf_p"]
        val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
        dt = time.time() - t0

        log_metrics = {
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "val/loss": val_loss_mean,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": dt,
            "global_step": global_step,
            "epoch": epoch + 1,
        }
        for split_name, m in split_metrics.items():
            for k, v in m.items():
                log_metrics[f"{split_name}/{k}"] = v
        for k, v in val_avg.items():
            log_metrics[f"val_{k}"] = v
        wandb.log(log_metrics)

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
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
            f"val_avg_surf_p={avg_surf_p:.4f}{tag}  lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(name, split_metrics[name])

    total_time = (time.time() - train_start) / 60.0
    print(f"\nTraining done in {total_time:.1f} min")

    if best_metrics:
        print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
        wandb.summary.update({
            "best_epoch": best_metrics["epoch"],
            "best_val_avg/mae_surf_p": best_avg_surf_p,
            "total_train_minutes": total_time,
        })

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
                name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_dtype=amp_dtype)
                for name, loader in test_loaders.items()
            }
            test_avg = aggregate_splits(test_metrics)
            print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
            for name in TEST_SPLIT_NAMES:
                print_split_metrics(name, test_metrics[name])

            test_log: dict[str, float] = {}
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    test_log[f"test/{split_name}/{k}"] = v
            for k, v in test_avg.items():
                test_log[f"test_{k}"] = v
            wandb.log(test_log)
            wandb.summary.update(test_log)

        save_model_artifact(
            run=run,
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
        print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping artifact upload.")

    wandb.finish()


if __name__ == "__main__":
    main()
