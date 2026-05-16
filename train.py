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
  python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]
"""

from __future__ import annotations

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
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.optim.swa_utils import AveragedModel, update_bn
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
# Fourier position features
# ---------------------------------------------------------------------------

FOURIER_BANDS = 8


class FourierEncoder(nn.Module):
    """Learnable sinusoidal Fourier encoding of 2D position.

    Initialized to the octave-doubling baseline `[1, 2, 4, ..., 2^(n-1)]` so the
    untrained model matches PR #3200 exactly. Frequencies are stored as a single
    `nn.Parameter` and updated by the same optimizer as the rest of the model.
    """

    def __init__(self, n_bands: int = FOURIER_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.freqs = nn.Parameter(
            torch.tensor([2.0 ** i for i in range(n_bands)], dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """pos: [..., 2] normalized (x, z). Returns [..., 4 * n_bands]."""
        freqs = self.freqs.to(pos.dtype) * math.pi  # [n_bands]
        phases = pos.unsqueeze(-1) * freqs  # [..., 2, n_bands]
        sin_f = torch.sin(phases)
        cos_f = torch.cos(phases)
        out = torch.stack([sin_f, cos_f], dim=-1)  # [..., 2, n_bands, 2]
        return out.reshape(*pos.shape[:-1], 4 * self.n_bands)


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


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on a scalar signal.

    Initialised so gamma≈0 and beta≈0 — at the start of training each FiLM
    layer is identity and the model matches the unconditioned baseline.
    """

    def __init__(self, hidden_dim: int, cond_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, hidden_dim]; cond: [B, cond_dim]
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# Surface-refinement input feature indices into normalized x (24-dim).
# coords (0,1) + log_Re (13) + AoA1 (14) + NACA1 (15-17) + AoA2 (18) +
# NACA2 (19-21) + gap (22) = 12 features. Stagger (23) intentionally omitted
# per the PR spec; n_feat must match SurfaceRefinementMLP construction.
SURF_FEAT_IDXS = list(range(0, 2)) + [13] + list(range(14, 23))


class SurfaceRefinementMLP(nn.Module):
    """Residual MLP refining surface-node predictions post-Transolver.

    Zero-init the output projection so the refinement starts as a no-op
    (identity start) — the baseline prediction is unchanged at init and the
    MLP can only learn corrections that improve the surface metric.
    """

    def __init__(self, n_pred: int = 3, n_feat: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pred + n_feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_pred),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pred_surf: torch.Tensor, x_surf_feat: torch.Tensor) -> torch.Tensor:
        # pred_surf: (S, n_pred), x_surf_feat: (S, n_feat) → (S, n_pred)
        return self.net(torch.cat([pred_surf, x_surf_feat], dim=-1))


def apply_surface_refinement(
    pred: torch.Tensor,
    x_norm: torch.Tensor,
    is_surface: torch.Tensor,
    mask: torch.Tensor,
    surf_refine_mlp: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add a residual refinement to surface-node predictions.

    Returns (pred_refined, delta) where delta is the (S, n_pred) residual
    applied to active surface nodes (None if no active surface nodes).
    Uses a full-shape sparse delta + non-in-place add so the prediction is
    differentiable through both the Transolver and the refinement MLP.
    """
    active_surf = mask & is_surface
    if not active_surf.any():
        return pred, None
    surf_pred = pred[active_surf]
    surf_feat = x_norm[active_surf][:, SURF_FEAT_IDXS]
    delta = surf_refine_mlp(surf_pred, surf_feat)
    full_delta = torch.zeros_like(pred)
    full_delta[active_surf] = delta
    return pred + full_delta, delta


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
                 use_film=False, cond_dim=1):
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
        self.use_film = use_film
        if use_film:
            self.film1 = FiLMLayer(hidden_dim, cond_dim=cond_dim)
            self.film2 = FiLMLayer(hidden_dim, cond_dim=cond_dim)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, re_cond=None):
        h = self.attn(self.ln_1(fx))
        if self.use_film and re_cond is not None:
            h = self.film1(h, re_cond)
        fx = h + fx
        h = self.mlp(self.ln_2(fx))
        if self.use_film and re_cond is not None:
            h = self.film2(h, re_cond)
        fx = h + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False, use_film=False,
                 re_feature_idx=13,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.use_film = use_film
        self.re_feature_idx = re_feature_idx
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
                use_film=use_film,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        # self.apply re-initialises every Linear (incl. FiLM's last linear)
        # with trunc_normal_, which would erase the zero init from
        # FiLMLayer.__init__. Re-zero those weights here so each FiLM starts
        # as near-identity (gamma≈0, beta≈0) per the spec.
        if use_film:
            for block in self.blocks:
                for film in (block.film1, block.film2):
                    nn.init.zeros_(film.net[-1].weight)
                    nn.init.zeros_(film.net[-1].bias)

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
        re_cond = None
        if self.use_film:
            # log(Re) is feature index 13, same value for every real node in a
            # sample. pad_collate pads with zeros at the end, so row 0 is
            # always a real node and reading it directly avoids the padding
            # contamination that mean(dim=1) would suffer (padding rows have
            # normalised value -mean/std ≈ -19 for log(Re), which would
            # dominate the mean for samples with lots of padding).
            re_cond = x[:, 0, self.re_feature_idx:self.re_feature_idx + 1]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, re_cond=re_cond)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, fourier_encoder, loader, stats, surf_weight,
                   smooth_l1_beta, device, surf_refine_mlp=None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    If ``surf_refine_mlp`` is provided, a residual correction is applied to
    surface-node predictions before loss and MAE computation (matching the
    training-time refinement).
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

            # Robust to per-sample non-finite ground truth: zero the mask for
            # any sample whose y has NaN/Inf and sanitize y itself. Otherwise
            # ``inf * 0 = NaN`` (IEEE 754) leaks through the masked sums and
            # produces NaN aggregates even when the offending sample is
            # supposed to be skipped (see test_geom_camber_cruise[20]).
            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            mask = mask & y_finite.view(B, 1)
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            ff = fourier_encoder(x_norm[..., :2])
            x_aug = torch.cat([x_norm, ff], dim=-1)
            pred = model({"x": x_aug})["preds"]

            if surf_refine_mlp is not None:
                pred, _ = apply_surface_refinement(
                    pred, x_norm, is_surface, mask, surf_refine_mlp
                )

            sq_err = F.smooth_l1_loss(pred, y_norm, beta=smooth_l1_beta, reduction="none")
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


def _sanitize_artifact_token(s: str) -> str:
    """wandb artifact names allow alnum, '-', '_', '.' — replace everything else."""
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
    run,
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
    """Log the best checkpoint as a wandb model artifact.

    Name: ``model-<agent-or-wandb_name>-<run.id>`` (run.id guarantees uniqueness).
    Aliases: ``best`` + ``epoch-N`` so the best checkpoint is addressable
    both by role and by the epoch it came from.
    Payload: ``checkpoint.pt`` + ``config.yaml`` (if present).
    Metadata: run context, selected val metric, optional test metrics, git
    commit, model config, and hyperparams — enough to trace and reload.
    """
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
        "epochs_configured": cfg.epochs,
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
        name=artifact_name,
        type="model",
        description=description,
        metadata=metadata,
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
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    fourier_bands: int = FOURIER_BANDS
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    smooth_l1_beta: float = 0.05  # quadratic-to-linear transition in normalized y space
    # SWA: start averaging at this 0-indexed epoch.
    # Default 7 corresponds to ~50% of typical wall-clock-limited training
    # (~14 epochs in the 30-min cap); cfg.epochs=50 would never trigger SWA
    # if we keyed off 50%, so we use the actual-training-aware default.
    swa_start_epoch: int = 7


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
    fun_dim=X_DIM - 2 + 4 * cfg.fourier_bands,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    use_film=True,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

# Surface-refinement residual MLP: takes (pred_surf, surf_feats) → 3-channel
# correction added to surface-node predictions. Zero-init output → identity
# start, matches baseline at init. Refined params join the optimizer below.
surf_refine_mlp = SurfaceRefinementMLP(
    n_pred=3, n_feat=len(SURF_FEAT_IDXS), hidden=64,
).to(device)
n_refine_params = sum(p.numel() for p in surf_refine_mlp.parameters())
print(
    f"Surface refinement MLP: {n_refine_params} params "
    f"(n_feat={len(SURF_FEAT_IDXS)}, hidden=64, zero-init output)"
)

# SWA: wrap a deep copy of the model so per-batch weight averages start
# accumulating once cfg.swa_start_epoch is reached. Default simple averaging.
swa_model = AveragedModel(model)
# Mirror SWA for the refinement MLP so the SWA-evaluated network has
# matched-epoch (model, refine) weights — required for consistent eval.
swa_refine_mlp = AveragedModel(surf_refine_mlp)
swa_n_updates = 0
print(
    f"SWA: averaging starts at 1-indexed epoch {cfg.swa_start_epoch + 1} "
    f"(0-indexed epoch >= {cfg.swa_start_epoch})"
)

fourier_encoder = FourierEncoder(cfg.fourier_bands).to(device)
n_fourier_params = sum(p.numel() for p in fourier_encoder.parameters())
print(f"Fourier encoder: {n_fourier_params} learnable freq parameters "
      f"(init: {[round(f, 3) for f in fourier_encoder.freqs.tolist()]})")

optimizer = torch.optim.AdamW(
    list(model.parameters())
    + list(fourier_encoder.parameters())
    + list(surf_refine_mlp.parameters()),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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
        "n_refine_params": n_refine_params,
        "surf_refine_n_feat": len(SURF_FEAT_IDXS),
        "surf_refine_idxs": SURF_FEAT_IDXS,
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

    for batch_idx, (x, y, is_surface, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        ff = fourier_encoder(x_norm[..., :2])
        x_aug = torch.cat([x_norm, ff], dim=-1)

        if epoch == 0 and batch_idx == 0 and model_config.get("use_film", False):
            with torch.no_grad():
                # First-row read (what the model now uses) — clean per-sample
                # log(Re) signal, immune to padding contamination.
                re_cond_dbg = x_aug[:, 0, 13]
                # Old buggy mean-over-nodes value, logged for the record so
                # the deviation from the original spec is visible.
                re_mean_dbg = x_aug[:, :, 13].mean(dim=1)
                print(
                    f"  [FiLM debug] epoch={epoch+1} re_cond (first-row) = "
                    f"{re_cond_dbg.cpu().tolist()}  re_mean_(buggy) = "
                    f"{re_mean_dbg.cpu().tolist()}"
                )
                wandb.log({
                    "debug/re_cond_min": re_cond_dbg.min().item(),
                    "debug/re_cond_max": re_cond_dbg.max().item(),
                    "debug/re_cond_std": re_cond_dbg.std().item() if re_cond_dbg.numel() > 1 else 0.0,
                    "debug/re_cond_mean": re_cond_dbg.mean().item(),
                    "debug/re_mean_buggy_min": re_mean_dbg.min().item(),
                    "debug/re_mean_buggy_max": re_mean_dbg.max().item(),
                    "global_step": global_step,
                })

        pred_raw = model({"x": x_aug})["preds"]
        pred, delta = apply_surface_refinement(
            pred_raw, x_norm, is_surface, mask, surf_refine_mlp,
        )
        sq_err = F.smooth_l1_loss(pred, y_norm, beta=cfg.smooth_l1_beta, reduction="none")

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        # Diagnostics: did the refinement MLP actually move predictions, and
        # did that movement reduce the surface-loss vs. the raw Transolver
        # output? Computed on the active surface nodes (delta is None only if
        # no surface nodes are active in this batch — extremely rare).
        if delta is not None:
            with torch.no_grad():
                surf_idx = surf_mask  # (B, N) bool
                y_surf = y_norm[surf_idx]  # (S, 3)
                pred_raw_surf = pred_raw[surf_idx]  # (S, 3)
                pred_ref_surf = pred[surf_idx]  # (S, 3)
                surf_loss_pre = F.smooth_l1_loss(
                    pred_raw_surf, y_surf, beta=cfg.smooth_l1_beta, reduction="mean",
                )
                surf_loss_post = F.smooth_l1_loss(
                    pred_ref_surf, y_surf, beta=cfg.smooth_l1_beta, reduction="mean",
                )
                delta_norm = delta.norm(dim=-1).mean()
                refine_log = {
                    "train/surf_delta_norm": delta_norm.item(),
                    "train/surf_loss_pre_refine": surf_loss_pre.item(),
                    "train/surf_loss_post_refine": surf_loss_post.item(),
                    "train/surf_refine_loss_fraction": (
                        surf_loss_post / surf_loss_pre.clamp(min=1e-8)
                    ).item(),
                }
        else:
            refine_log = {}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_refine_mlp.update_parameters(surf_refine_mlp)
            swa_n_updates += 1
        global_step += 1
        wandb.log({"train/loss": loss.item(), **refine_log, "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    surf_refine_mlp.eval()
    split_metrics = {
        name: evaluate_split(model, fourier_encoder, loader, stats,
                             cfg.surf_weight, cfg.smooth_l1_beta, device,
                             surf_refine_mlp=surf_refine_mlp)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    freqs_now = fourier_encoder.freqs.detach().cpu().tolist()
    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.
    for i, f in enumerate(freqs_now):
        log_metrics[f"fourier/freq_{i}"] = f
    log_metrics["fourier/freq_min"] = min(freqs_now)
    log_metrics["fourier/freq_max"] = max(freqs_now)
    log_metrics["fourier/freq_mean"] = sum(freqs_now) / len(freqs_now)
    wandb.log(log_metrics)

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        # Bundle main-model + refinement-MLP state so test-time eval can
        # reconstruct the full prediction pipeline from a single file.
        torch.save({
            "model": model.state_dict(),
            "surf_refine_mlp": surf_refine_mlp.state_dict(),
        }, model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- SWA finalization + validation ---
swa_active = swa_n_updates > 0
swa_val_avg = None
if swa_active:
    n_avg = int(swa_model.n_averaged.item())
    print(
        f"\nSWA: accumulated {swa_n_updates} updates "
        f"(AveragedModel.n_averaged={n_avg}) starting at 1-indexed "
        f"epoch {cfg.swa_start_epoch + 1}"
    )
    # No-op for LayerNorm-only models (no BatchNorm to update), but kept
    # per the SWA recipe — early-returns inside torch when no BN exists.
    update_bn(train_loader, swa_model, device=device)
    swa_model.eval()
    swa_refine_mlp.eval()

    print("Evaluating SWA model on validation splits...")
    swa_val_metrics = {
        name: evaluate_split(swa_model, fourier_encoder, loader, stats,
                             cfg.surf_weight, cfg.smooth_l1_beta, device,
                             surf_refine_mlp=swa_refine_mlp)
        for name, loader in val_loaders.items()
    }
    swa_val_avg = aggregate_splits(swa_val_metrics)
    print(f"  VAL (SWA) avg_surf_p={swa_val_avg['avg/mae_surf_p']:.4f}")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(f"{name} (SWA)", swa_val_metrics[name])

    swa_val_log: dict[str, float] = {"swa_n_updates": swa_n_updates}
    for split_name, m in swa_val_metrics.items():
        for k, v in m.items():
            swa_val_log[f"val/{split_name}/{k}_swa"] = v
    for k, v in swa_val_avg.items():
        swa_val_log[f"val_{k}_swa"] = v
    wandb.log(swa_val_log)
    wandb.summary.update(swa_val_log)

    swa_model_path = model_dir / "swa_checkpoint.pt"
    torch.save({
        "model": swa_model.module.state_dict(),
        "surf_refine_mlp": swa_refine_mlp.module.state_dict(),
    }, swa_model_path)
    print(f"Saved SWA checkpoint to {swa_model_path}")

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    final_freqs = fourier_encoder.freqs.detach().cpu().tolist()
    init_freqs = [2.0 ** i for i in range(cfg.fourier_bands)]
    freq_deltas = [(f - init) / init for f, init in zip(final_freqs, init_freqs)]
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
        "fourier/final_freqs": final_freqs,
        "fourier/init_freqs": init_freqs,
        "fourier/relative_drift": freq_deltas,
        "fourier/max_abs_relative_drift": max(abs(d) for d in freq_deltas),
    })
    print("\nLearned Fourier frequencies:")
    for i, (init, final, delta) in enumerate(zip(init_freqs, final_freqs, freq_deltas)):
        print(f"  freq_{i}: init={init:.4f}  final={final:.4f}  Δrel={delta*100:+.2f}%")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    surf_refine_mlp.load_state_dict(ckpt["surf_refine_mlp"])
    model.eval()
    surf_refine_mlp.eval()

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
            name: evaluate_split(model, fourier_encoder, loader, stats,
                                 cfg.surf_weight, cfg.smooth_l1_beta, device,
                                 surf_refine_mlp=surf_refine_mlp)
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

        if swa_active:
            print("\nEvaluating SWA model on held-out test splits...")
            swa_test_metrics = {
                name: evaluate_split(swa_model, fourier_encoder, loader, stats,
                                     cfg.surf_weight, cfg.smooth_l1_beta, device,
                                     surf_refine_mlp=swa_refine_mlp)
                for name, loader in test_loaders.items()
            }
            swa_test_avg = aggregate_splits(swa_test_metrics)
            print(f"\n  TEST (SWA) avg_surf_p={swa_test_avg['avg/mae_surf_p']:.4f}")
            for name in TEST_SPLIT_NAMES:
                print_split_metrics(f"{name} (SWA)", swa_test_metrics[name])

            swa_test_log: dict[str, float] = {}
            for split_name, m in swa_test_metrics.items():
                for k, v in m.items():
                    swa_test_log[f"test/{split_name}/{k}_swa"] = v
            for k, v in swa_test_avg.items():
                swa_test_log[f"test_{k}_swa"] = v
            wandb.log(swa_test_log)
            wandb.summary.update(swa_test_log)

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
