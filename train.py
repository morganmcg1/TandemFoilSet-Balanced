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
from torch.optim.swa_utils import AveragedModel, SWALR
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
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
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
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, film=None):
        # film: optional (gamma, beta) tensors each [B, 1, H] for this block
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if film is not None:
            gamma, beta = film
            fx = (1 + gamma) * fx + beta
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class FourierCoordFeatures(nn.Module):
    """Random Fourier Features (Tancik et al. NeurIPS 2020) on per-node coordinates.

    Projects (x, z) -> [sin(2*pi*B*x), cos(2*pi*B*x)] where ``B`` is a fixed
    random matrix with entries ~ N(0, sigma^2). The matrix is registered as a
    buffer (frozen), since learning B causes spectral collapse (Tancik 2020).
    """

    def __init__(self, coord_dim: int = 2, num_features: int = 16, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(coord_dim, num_features) * sigma
        self.register_buffer("B", B)
        self.coord_dim = coord_dim
        self.num_features = num_features
        self.sigma = sigma

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [..., coord_dim] (already z-score normalized upstream)
        x_proj = 2.0 * math.pi * (coords @ self.B)  # [..., num_features]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [..., 2*num_features]


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 fourier_features: bool = False,
                 fourier_num_features: int = 16,
                 fourier_sigma: float = 1.0):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        # Random Fourier Features on coords (Tancik 2020). Coords are the first
        # ``space_dim`` channels of x and are already z-score normalized.
        self.fourier_features = fourier_features
        if fourier_features:
            self.fourier_coords = FourierCoordFeatures(
                coord_dim=space_dim,
                num_features=fourier_num_features,
                sigma=fourier_sigma,
            )
            extra_dim = 2 * fourier_num_features
        else:
            self.fourier_coords = None
            extra_dim = 0

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3 + extra_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim + extra_dim, n_hidden * 2, n_hidden,
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
        # Diagnostics filled on first forward pass.
        self._coord_diag: dict[str, float] | None = None
        self._rff_diag: dict[str, float] | None = None
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
        if self.fourier_coords is not None:
            coords = x[..., : self.space_dim]
            rff = self.fourier_coords(coords)  # [B, N, 2*num_features]
            # Capture diagnostics on the very first forward pass only.
            if self._coord_diag is None:
                with torch.no_grad():
                    self._coord_diag = {
                        "min": coords.min().item(),
                        "max": coords.max().item(),
                        "mean": coords.mean().item(),
                        "std": coords.std().item(),
                    }
                    self._rff_diag = {
                        "mean": rff.mean().item(),
                        "std": rff.std().item(),
                        "min": rff.min().item(),
                        "max": rff.max().item(),
                    }
            x = torch.cat([x, rff], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        # Optional FiLM params [B, L, 2, H] — sliced per block as (gamma, beta) each [B, 1, H].
        film_params = data.get("film", None)
        for i, block in enumerate(self.blocks):
            if film_params is not None:
                gamma = film_params[:, i, 0, :].unsqueeze(1)
                beta = film_params[:, i, 1, :].unsqueeze(1)
                fx = block(fx, film=(gamma, beta))
            else:
                fx = block(fx)
        return {"preds": fx}


class FiLMConditioner(nn.Module):
    """Predicts per-layer (gamma, beta) from per-sample global flow conditions.

    Globals are dims 13..23 of x (log(Re), AoA front/rear, NACA front/rear M/P/T,
    gap, stagger; 11 features total). They are constant per-sample over real
    nodes, so we extract them as a masked mean over the node axis to ignore
    padded positions (which sit at 0 in the normalized input).

    Zero-init on the final linear (both weight and bias) so the FiLM transform
    starts as identity ((1 + 0) * h + 0 = h) and does not disturb the
    Transolver init.
    """

    def __init__(self, n_layers, n_hidden, cond_dim=11, mid_dim=64):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 2 * n_layers * n_hidden),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, mask):
        # x: [B, N, 24]  (already normalized by upstream pipeline)
        # mask: [B, N]   (True for real nodes)
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1)
        cond = (x[:, :, 13:24] * m).sum(dim=1) / denom    # [B, cond_dim]
        out = self.net(cond)                              # [B, 2*L*H]
        return out.view(-1, self.n_layers, 2, self.n_hidden)  # [B, L, 2, H]


class FiLMTransolver(nn.Module):
    """Transolver with FiLM global-conditioning injected into every block.

    Holds an unmodified Transolver and a FiLMConditioner. On forward, computes
    one (gamma, beta) pair per Transolver block from the per-sample globals and
    passes them through ``data["film"]`` so the existing Transolver.forward can
    dispatch them inside its block loop.
    """

    def __init__(self, n_layers, n_hidden, cond_dim=11, film_mid_dim=64,
                 **transolver_kwargs):
        super().__init__()
        self.transolver = Transolver(
            n_layers=n_layers, n_hidden=n_hidden, **transolver_kwargs
        )
        self.film = FiLMConditioner(
            n_layers=n_layers, n_hidden=n_hidden,
            cond_dim=cond_dim, mid_dim=film_mid_dim,
        )
        self._last_film: torch.Tensor | None = None  # detached, for diagnostics

    def forward(self, data, **kwargs):
        x = data["x"]
        mask = data["mask"]
        film_params = self.film(x, mask)
        self._last_film = film_params.detach()
        # Forward through Transolver with FiLM injected via data dict.
        data_with_film = {**data, "film": film_params}
        return self.transolver(data_with_film, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

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
            pred = model({"x": x_norm, "mask": mask})["preds"]

            sq_err = F.smooth_l1_loss(pred, y_norm, beta=cfg.huber_beta, reduction='none')
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
# Optimizers
# ---------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023, 'Symbolic Discovery of Optimization Algorithms').

    Sign-of-interpolated-EMA-gradient update with decoupled weight decay. Recommended
    hyperparams are ~3-10× smaller lr and ~3-10× larger weight_decay than AdamW.

    Stores per-step ``last_update_sq_norm`` / ``last_exp_avg_sq_norm`` GPU scalar
    tensors so the training loop can log update-magnitude and EMA-state diagnostics
    (sum of squares across all params with gradients).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_update_sq_norm: torch.Tensor | None = None
        self.last_exp_avg_sq_norm: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        update_sq_total: torch.Tensor | None = None
        exp_avg_sq_total: torch.Tensor | None = None
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # Sign of (beta1 * m + (1-beta1) * g) — bounded magnitude update.
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)
                # EMA update with separate beta2.
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                upd_sq = update.pow(2).sum()
                ema_sq = exp_avg.pow(2).sum()
                update_sq_total = upd_sq if update_sq_total is None else update_sq_total + upd_sq
                exp_avg_sq_total = ema_sq if exp_avg_sq_total is None else exp_avg_sq_total + ema_sq
        self.last_update_sq_norm = update_sq_total.detach() if update_sq_total is not None else None
        self.last_exp_avg_sq_norm = exp_avg_sq_total.detach() if exp_avg_sq_total is not None else None
        return loss


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
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    seed: int = 0
    film_mid_dim: int = 64
    max_norm: float = 0.0  # gradient-norm clipping threshold (0 = disabled, 1.0 = standard)
    use_kendall_uncertainty: bool = False  # learn per-channel log_sigma weighting (Kendall 2018) — overrides surf_weight
    fourier_features: bool = False  # Random Fourier Features on coords (Tancik 2020).
    fourier_num_features: int = 16  # Number of random frequency vectors B columns.
    fourier_sigma: float = 1.0  # Std of random B matrix (controls freq bandwidth).
    huber_beta: float = 1.0  # Smooth-L1 β; lower = more L1-like, higher = more MSE-like
    optimizer: str = "adamw"  # "adamw" (baseline) | "lion" (Chen et al. 2023, sign-of-EMA-grad)
    hybrid_kendall_lr: float = 1e-3  # AdamW lr for log_sigmas when optimizer=lion + use_kendall_uncertainty (Lion's sign-update collapses log_σ channels; AdamW preserves gradient-magnitude per-channel differentiation)
    swa_lr: float = 0.0  # SWALR target lr (the SWA-window floor lr). 0 = use default `cfg.lr * 0.2`. Decouples SWA-floor from cosine-final-lr.


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

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
)

model = FiLMTransolver(
    n_layers=model_config["n_layers"],
    n_hidden=model_config["n_hidden"],
    cond_dim=11,
    film_mid_dim=cfg.film_mid_dim,
    space_dim=model_config["space_dim"],
    fun_dim=model_config["fun_dim"],
    out_dim=model_config["out_dim"],
    n_head=model_config["n_head"],
    slice_num=model_config["slice_num"],
    mlp_ratio=model_config["mlp_ratio"],
    output_fields=model_config["output_fields"],
    output_dims=model_config["output_dims"],
    fourier_features=cfg.fourier_features,
    fourier_num_features=cfg.fourier_num_features,
    fourier_sigma=cfg.fourier_sigma,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_params_film = sum(p.numel() for p in model.film.parameters())
print(f"Model: FiLMTransolver ({n_params/1e6:.2f}M params, FiLM head {n_params_film/1e3:.1f}K)")
if cfg.fourier_features:
    print(
        f"Fourier features: ON (num_features={cfg.fourier_num_features}, sigma={cfg.fourier_sigma}, "
        f"input expanded by {2*cfg.fourier_num_features} channels)"
    )

# Kendall uncertainty heads — per-channel learnable log_sigma scalars (Kendall, Gal, Cipolla 2018).
# Channel order matches loss decomposition: [surf_p, surf_ux, surf_uy, vol_p, vol_ux, vol_uy].
KENDALL_CHANNELS = ("surf_p", "surf_ux", "surf_uy", "vol_p", "vol_ux", "vol_uy")
KENDALL_LOG_SIGMA_CLAMP = 3.0  # clamp |log_sigma| ≤ 3, so sigma ∈ [exp(-3), exp(3)] ≈ [0.05, 20]
def _build_optimizer(name: str, param_groups, lr: float, default_wd: float):
    """Builds an AdamW or Lion optimizer from a param_groups iterable.

    ``param_groups`` may be ``model.parameters()`` (single group) or a list of dicts
    when Kendall log_sigmas need a separate weight_decay=0 group.
    """
    name = name.lower()
    if name == "lion":
        if isinstance(param_groups, list):
            return Lion(param_groups, lr=lr, betas=(0.9, 0.99))
        return Lion(param_groups, lr=lr, weight_decay=default_wd, betas=(0.9, 0.99))
    if name == "adamw":
        if isinstance(param_groups, list):
            return torch.optim.AdamW(param_groups, lr=lr)
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=default_wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


optimizer_kendall: torch.optim.Optimizer | None = None
hybrid_kendall = False
if cfg.use_kendall_uncertainty:
    log_sigmas = nn.Parameter(torch.zeros(len(KENDALL_CHANNELS), device=device))
    # Hybrid path: Lion's sign-update strips gradient magnitude and collapses all log_σ channels
    # to an identical value (verified in PR #2063). When using Lion + Kendall, route log_sigmas
    # through a separate AdamW optimizer so gradient-magnitude per-channel differentiation is
    # preserved. Model params still use Lion (which delivered the 28.5% win).
    if cfg.optimizer.lower() == "lion":
        hybrid_kendall = True
        # Pass `model.parameters()` (a generator) — _build_optimizer's isinstance(list) check
        # otherwise drops the weight_decay kwarg when it sees a list-of-tensors.
        optimizer = _build_optimizer(
            cfg.optimizer,
            model.parameters(),
            lr=cfg.lr,
            default_wd=cfg.weight_decay,
        )
        optimizer_kendall = torch.optim.AdamW(
            [log_sigmas], lr=cfg.hybrid_kendall_lr, weight_decay=0.0
        )
        print(
            f"Hybrid optimizer: Lion(model, lr={cfg.lr:.2e}, wd={cfg.weight_decay:.2e}) + "
            f"AdamW(log_sigmas, lr={cfg.hybrid_kendall_lr:.2e}, wd=0)"
        )
    else:
        # Separate param group with weight_decay=0 — weight decay on log_sigma acts as an unwanted prior toward sigma=1.
        optimizer = _build_optimizer(
            cfg.optimizer,
            [
                {"params": list(model.parameters()), "weight_decay": cfg.weight_decay},
                {"params": [log_sigmas], "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            default_wd=cfg.weight_decay,
        )
    print(f"Kendall uncertainty: ON (6 log_sigmas, clamp ±{KENDALL_LOG_SIGMA_CLAMP}, no weight decay on log_sigmas)")
else:
    log_sigmas = None
    optimizer = _build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr, default_wd=cfg.weight_decay)
print(f"Optimizer: {cfg.optimizer.upper()} (lr={cfg.lr:.2e}, wd={cfg.weight_decay:.2e})")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# SWA (PR #1554): average weights over the final 25% of training to find a
# flatter optimum. Skip update_bn — Transolver uses LayerNorm only.
swa_start_frac = 0.75
swa_start_epoch = int(swa_start_frac * MAX_EPOCHS)  # 0-indexed loop var
swa_model = AveragedModel(model)
swa_lr = cfg.swa_lr if cfg.swa_lr > 0 else cfg.lr * 0.2
swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=2, anneal_strategy="cos")
print(
    f"SWA: start_epoch={swa_start_epoch} (0-indexed), "
    f"swa_lr={swa_lr:.2e}, anneal_epochs=2"
)

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
        "swa_start_frac": swa_start_frac,
        "swa_start_epoch": swa_start_epoch,
        "swa_lr": swa_lr,
        "swa_anneal_epochs": 2,
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
fourier_diag_logged = False

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0
    epoch_grad_norm_sum = 0.0
    epoch_grad_norm_max = 0.0
    epoch_clip_fraction_sum = 0.0
    epoch_lion_update_norm_sum = 0.0
    epoch_lion_update_norm_n = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm, "mask": mask})["preds"]
        sq_err = F.smooth_l1_loss(pred, y_norm, beta=cfg.huber_beta, reduction='none')

        # Per-sample log(Re) from raw (un-normalized) feature dim 13 — constant per real node within a sample.
        m_b = mask.unsqueeze(-1).float()                                   # [B, N, 1]
        denom = m_b.sum(dim=1).clamp(min=1)                                 # [B, 1]
        log_re = (x[:, :, 13:14] * m_b).sum(dim=1) / denom                  # [B, 1]
        # Shift so all values are >= 1 (prevents divide-by-near-zero and keeps weights positive).
        log_re_shifted = log_re - log_re.min().detach() + 1.0               # [B, 1]
        re_weight = 1.0 / log_re_shifted                                    # [B, 1]
        # Normalize so per-batch sample weights mean to 1.0.
        re_weight = re_weight * (re_weight.shape[0] / re_weight.sum().clamp(min=1e-8))
        re_weight_expanded = re_weight.unsqueeze(-1)                        # [B, 1, 1]

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        # Apply per-sample re_weight to the per-element error; surf_weight (or Kendall sigmas) stays on top.
        weighted_err = sq_err * re_weight_expanded

        # Per-channel decomposition: mean-reduced over masked nodes.
        # Y channel order: 0=Ux, 1=Uy, 2=p. Kendall channel order: surf_p, surf_ux, surf_uy, vol_p, vol_ux, vol_uy.
        surf_count = surf_mask.sum().clamp(min=1)
        vol_count = vol_mask.sum().clamp(min=1)
        per_channel_surf = (weighted_err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1)) / surf_count  # [3]
        per_channel_vol = (weighted_err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1)) / vol_count    # [3]
        surf_loss = per_channel_surf.sum()  # sum over 3 channels (matches old surf_loss aggregate)
        vol_loss = per_channel_vol.sum()

        if cfg.use_kendall_uncertainty:
            # 6 per-channel scalar losses in Kendall order.
            kendall_losses = torch.stack([
                per_channel_surf[2],  # surf_p
                per_channel_surf[0],  # surf_ux
                per_channel_surf[1],  # surf_uy
                per_channel_vol[2],   # vol_p
                per_channel_vol[0],   # vol_ux
                per_channel_vol[1],   # vol_uy
            ])
            log_sigmas_clamped = torch.clamp(log_sigmas, -KENDALL_LOG_SIGMA_CLAMP, KENDALL_LOG_SIGMA_CLAMP)
            # Kendall, Gal, Cipolla 2018 eq. 9: L = sum_c [ 0.5 * exp(-2*log_sigma_c) * loss_c + log_sigma_c ]
            precision = torch.exp(-2.0 * log_sigmas_clamped)
            loss = (0.5 * precision * kendall_losses + log_sigmas_clamped).sum()
        else:
            loss = vol_loss + cfg.surf_weight * surf_loss

        # Diagnostic: also compute the unweighted loss (does not flow through .backward()).
        with torch.no_grad():
            vol_loss_unw = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss_unw = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss_unweighted = vol_loss_unw + cfg.surf_weight * surf_loss_unw

        optimizer.zero_grad()
        if optimizer_kendall is not None:
            optimizer_kendall.zero_grad()
        loss.backward()
        grad_norm_val = None
        clip_fired = 0.0
        if cfg.max_norm > 0:
            # Clip model params only; log_sigmas (a tiny ParameterTensor, not in model.parameters())
            # follow their own optimizer's update rule unclipped.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
            grad_norm_val = grad_norm.item()
            clip_fired = 1.0 if grad_norm_val > cfg.max_norm else 0.0
            epoch_grad_norm_sum += grad_norm_val
            epoch_grad_norm_max = max(epoch_grad_norm_max, grad_norm_val)
            epoch_clip_fraction_sum += clip_fired
        optimizer.step()
        if optimizer_kendall is not None:
            optimizer_kendall.step()
        global_step += 1
        log_payload = {
            "train/loss": loss.item(),
            "train/loss_unweighted": loss_unweighted.item(),
            "train/re_weight_min": re_weight.min().item(),
            "train/re_weight_max": re_weight.max().item(),
            "train/re_weight_mean": re_weight.mean().item(),
            "train/log_re_per_batch_mean": log_re.mean().item(),
            "train/log_re_per_batch_std": log_re.std().item() if log_re.numel() > 1 else 0.0,
            "global_step": global_step,
        }
        if grad_norm_val is not None:
            log_payload["train/grad_norm"] = grad_norm_val
            log_payload["train/clip_fraction"] = clip_fired
        if cfg.optimizer == "lion" and isinstance(optimizer, Lion):
            if optimizer.last_update_sq_norm is not None:
                _upd = optimizer.last_update_sq_norm.sqrt().item()
                log_payload["train/optimizer_update_norm"] = _upd
                epoch_lion_update_norm_sum += _upd
                epoch_lion_update_norm_n += 1
            if optimizer.last_exp_avg_sq_norm is not None:
                log_payload["train/exp_avg_norm"] = optimizer.last_exp_avg_sq_norm.sqrt().item()
        if cfg.use_kendall_uncertainty:
            with torch.no_grad():
                ls = log_sigmas.detach()
                ls_c = torch.clamp(ls, -KENDALL_LOG_SIGMA_CLAMP, KENDALL_LOG_SIGMA_CLAMP)
                eff_w = 0.5 * torch.exp(-2.0 * ls_c)
                for ci, name in enumerate(KENDALL_CHANNELS):
                    log_payload[f"train/log_sigma_{name}"] = ls[ci].item()
                    log_payload[f"train/effective_weight_{name}"] = eff_w[ci].item()
                    log_payload[f"train/per_channel_loss_{name}"] = kendall_losses[ci].item()
        if (
            cfg.fourier_features
            and not fourier_diag_logged
            and model.transolver._coord_diag is not None
        ):
            summary_update = {}
            for k, v in model.transolver._coord_diag.items():
                summary_update[f"fourier/coord_{k}"] = v
            for k, v in model.transolver._rff_diag.items():
                summary_update[f"fourier/rff_{k}"] = v
            wandb.summary.update(summary_update)
            log_payload.update(summary_update)
            print(
                "Fourier diag — coord "
                + ", ".join(f"{k}={v:.4f}" for k, v in model.transolver._coord_diag.items())
                + " | RFF "
                + ", ".join(f"{k}={v:.4f}" for k, v in model.transolver._rff_diag.items())
            )
            fourier_diag_logged = True
        wandb.log(log_payload)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model)
        swa_scheduler.step()
        current_lr = swa_scheduler.get_last_lr()[0]
        swa_active = True
    else:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        swa_active = False
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
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "lr": current_lr,
        "swa_active": int(swa_active),
        "epoch_time_s": dt,
        "epoch": epoch + 1,
        "global_step": global_step,
    }
    if cfg.max_norm > 0 and n_batches > 0:
        log_metrics["train/grad_norm_mean"] = epoch_grad_norm_sum / n_batches
        log_metrics["train/grad_norm_max"] = epoch_grad_norm_max
        log_metrics["train/clip_fraction_mean"] = epoch_clip_fraction_sum / n_batches
    if epoch_lion_update_norm_n > 0:
        log_metrics["train/lion_update_norm_epoch_mean"] = (
            epoch_lion_update_norm_sum / epoch_lion_update_norm_n
        )
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.

    # FiLM diagnostics: per-layer mean |gamma| and |beta| from last forward.
    if isinstance(model, FiLMTransolver) and model._last_film is not None:
        film_t = model._last_film  # [B, L, 2, H]
        gamma_per_layer = film_t[:, :, 0, :].abs().mean(dim=(0, 2))  # [L]
        beta_per_layer = film_t[:, :, 1, :].abs().mean(dim=(0, 2))   # [L]
        log_metrics["film/gamma_abs_mean"] = gamma_per_layer.mean().item()
        log_metrics["film/beta_abs_mean"] = beta_per_layer.mean().item()
        log_metrics["film/gamma_l2"] = film_t[:, :, 0, :].norm().item()
        log_metrics["film/beta_l2"] = film_t[:, :, 1, :].norm().item()
        for li in range(gamma_per_layer.numel()):
            log_metrics[f"film/gamma_abs_L{li}"] = gamma_per_layer[li].item()
            log_metrics[f"film/beta_abs_L{li}"] = beta_per_layer[li].item()

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
    swa_tag = " [SWA]" if swa_active else ""
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]{swa_tag}  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"lr={current_lr:.2e}  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    if cfg.use_kendall_uncertainty:
        with torch.no_grad():
            ls = log_sigmas.detach().cpu().numpy()
            ls_clamped = torch.clamp(log_sigmas.detach(), -KENDALL_LOG_SIGMA_CLAMP, KENDALL_LOG_SIGMA_CLAMP).cpu().numpy()
            eff_w_arr = (0.5 * torch.exp(-2.0 * torch.tensor(ls_clamped))).numpy()
        ls_str = "  ".join(f"{n}={s:+.3f}(w={w:.3f})" for n, s, w in zip(KENDALL_CHANNELS, ls, eff_w_arr))
        print(f"    kendall: {ls_str}")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

if cfg.use_kendall_uncertainty:
    with torch.no_grad():
        ls_final = log_sigmas.detach().cpu().numpy()
        ls_clamped = torch.clamp(log_sigmas.detach(), -KENDALL_LOG_SIGMA_CLAMP, KENDALL_LOG_SIGMA_CLAMP).cpu().numpy()
        eff_w_final = (0.5 * torch.exp(-2.0 * torch.tensor(ls_clamped))).numpy()
    print("\nFinal Kendall log_sigmas:")
    for n, s, w in zip(KENDALL_CHANNELS, ls_final, eff_w_final):
        print(f"  {n:<10s} log_sigma={s:+.4f}  effective_weight={w:.4f}")
    kendall_summary = {}
    for n, s, w in zip(KENDALL_CHANNELS, ls_final, eff_w_final):
        kendall_summary[f"final/log_sigma_{n}"] = float(s)
        kendall_summary[f"final/effective_weight_{n}"] = float(w)
    wandb.summary.update(kendall_summary)

# Persist the SWA-averaged weights regardless of best_metrics so they survive
# the run even if no base epoch ever improved val_avg/mae_surf_p.
swa_model_path = model_dir / "checkpoint_swa.pt"
torch.save(swa_model.module.state_dict(), swa_model_path)
print(f"Saved SWA checkpoint to {swa_model_path}")

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest base val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
        "swa_start_epoch": swa_start_epoch,
        "swa_lr": swa_lr,
    })

    # --- Base-best evaluation (for comparison) ---
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    base_test_metrics = None
    base_test_avg = None
    test_loaders = None
    if not cfg.skip_test:
        print("\nEvaluating BASE-BEST checkpoint on held-out test splits...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        base_test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        base_test_avg = aggregate_splits(base_test_metrics)
        print(f"\n  BASE  test_avg/mae_surf_p={base_test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, base_test_metrics[name])

    # --- SWA evaluation (primary numbers) ---
    # Per PR spec: load SWA weights into `model`, then run evaluate_split for
    # val + test. These are the "reported best" numbers.
    model.load_state_dict(swa_model.module.state_dict())
    model.eval()

    print("\nEvaluating SWA model on val splits...")
    swa_val_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
        for name, loader in val_loaders.items()
    }
    swa_val_avg = aggregate_splits(swa_val_metrics)
    swa_val_avg_surf_p = swa_val_avg["avg/mae_surf_p"]
    print(f"  SWA   val_avg/mae_surf_p={swa_val_avg_surf_p:.4f}")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, swa_val_metrics[name])

    swa_test_metrics = None
    swa_test_avg = None
    if not cfg.skip_test and test_loaders is not None:
        print("\nEvaluating SWA model on test splits...")
        swa_test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        swa_test_avg = aggregate_splits(swa_test_metrics)
        print(f"\n  SWA   test_avg/mae_surf_p={swa_test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, swa_test_metrics[name])

    # --- W&B logging: SWA = primary `test_*`/`val_*`, base = `base_test_*` alias ---
    final_log: dict[str, float] = {}

    for split_name, m in swa_val_metrics.items():
        for k, v in m.items():
            final_log[f"swa_val/{split_name}/{k}"] = v
    for k, v in swa_val_avg.items():
        final_log[f"swa_val_{k}"] = v

    if swa_test_metrics is not None:
        for split_name, m in swa_test_metrics.items():
            for k, v in m.items():
                final_log[f"test/{split_name}/{k}"] = v
                final_log[f"swa_test/{split_name}/{k}"] = v
        for k, v in swa_test_avg.items():
            final_log[f"test_{k}"] = v
            final_log[f"swa_test_{k}"] = v

    if base_test_metrics is not None:
        for split_name, m in base_test_metrics.items():
            for k, v in m.items():
                final_log[f"base_test/{split_name}/{k}"] = v
        for k, v in base_test_avg.items():
            final_log[f"base_test_{k}"] = v

    wandb.log(final_log)
    wandb.summary.update(final_log)

    # --- Artifact: SWA model is the reported best ---
    swa_base = _sanitize_artifact_token(cfg.wandb_name or cfg.agent or "tandemfoil")
    swa_artifact_name = f"model-{swa_base}-{run.id}-swa"
    swa_metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "agent": cfg.agent,
        "wandb_name": cfg.wandb_name,
        "wandb_group": cfg.wandb_group,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "model_config": model_config,
        "swa_start_epoch": swa_start_epoch,
        "swa_start_frac": swa_start_frac,
        "swa_lr": swa_lr,
        "swa_anneal_epochs": 2,
        "swa_val_avg/mae_surf_p": swa_val_avg_surf_p,
        "swa_test_avg/mae_surf_p": (swa_test_avg["avg/mae_surf_p"] if swa_test_avg else None),
        "base_best_epoch": best_metrics["epoch"],
        "base_best_val_avg/mae_surf_p": best_avg_surf_p,
        "base_test_avg/mae_surf_p": (base_test_avg["avg/mae_surf_p"] if base_test_avg else None),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "epochs_configured": cfg.epochs,
    }
    description = (
        f"Transolver SWA checkpoint — swa val_avg/mae_surf_p = {swa_val_avg_surf_p:.4f} "
        f"(base-best epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f})"
    )
    if swa_test_avg is not None:
        description += f" | swa test_avg/mae_surf_p = {swa_test_avg['avg/mae_surf_p']:.4f}"
    swa_artifact = wandb.Artifact(
        name=swa_artifact_name,
        type="model",
        description=description,
        metadata=swa_metadata,
    )
    swa_artifact.add_file(str(swa_model_path), name="checkpoint.pt")
    config_yaml = model_dir / "config.yaml"
    if config_yaml.exists():
        swa_artifact.add_file(str(config_yaml), name="config.yaml")
    run.log_artifact(swa_artifact, aliases=["best", "swa-final"])
    print(f"\nLogged SWA model artifact '{swa_artifact_name}'")
else:
    print("\nNo base checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping eval + artifact upload.")

wandb.finish()
