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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a scalar.

    h <- (1 + delta_gamma) * h + beta, with the final MLP layer zero-initialized
    so the transform starts at identity. adaLN-Zero style — see Peebles & Xie
    (2022) and Zhu et al. (ICLR 2025).
    """

    def __init__(self, n_hidden: int, hidden_mlp: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_mlp),
            nn.SiLU(),
            nn.Linear(hidden_mlp, 2 * n_hidden),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c):
        """c: ``[B]`` or ``[B, 1]`` standardized scalar.

        Returns (gamma, beta), each ``[B, 1, n_hidden]`` so they broadcast
        across the node dimension.
        """
        if c.dim() == 1:
            c = c.unsqueeze(-1)
        delta_gamma, beta = self.mlp(c).chunk(2, dim=-1)
        gamma = 1.0 + delta_gamma
        return gamma.unsqueeze(1), beta.unsqueeze(1)


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

    def forward(self, fx, film=None, film_ffn=False):
        fx = self.attn(self.ln_1(fx)) + fx

        ffn_out = self.mlp(self.ln_2(fx))
        if film is not None and film_ffn:
            gamma, beta = film
            ffn_out = gamma * ffn_out + beta
        fx = ffn_out + fx

        if self.last_layer:
            h = self.ln_3(fx)
            h = self.mlp2[0](h)
            h = self.mlp2[1](h)
            if film is not None:
                gamma, beta = film
                h = gamma * h + beta
            return self.mlp2[2](h)
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_film: bool = False, film_mode: str = "output_only",
                 log_re_mean: float = 14.58, log_re_std: float = 0.76):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.use_film = use_film
        self.film_mode = film_mode
        self.log_re_mean = log_re_mean
        self.log_re_std = log_re_std

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
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        if self.use_film:
            self.film = FiLM(n_hidden)

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

        film_params = None
        if self.use_film and "log_re_raw" in data:
            log_re = data["log_re_raw"]
            log_re_n = (log_re - self.log_re_mean) / self.log_re_std
            film_params = self.film(log_re_n)

        n_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            is_last = (i == n_blocks - 1)
            if film_params is not None and self.film_mode == "all_blocks":
                fx = block(fx, film=film_params, film_ffn=True)
            elif film_params is not None and self.film_mode == "output_only" and is_last:
                fx = block(fx, film=film_params, film_ffn=False)
            else:
                fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Fourier positional features over (x, z) coords
# ---------------------------------------------------------------------------

def _apply_fourier(x_norm, model):
    """Append sin/cos Fourier features of normalized (x, z) to ``x_norm``.

    The projection matrix ``B`` is stored as a buffer on the model so the
    same encoding is applied at train and eval time (it travels with
    ``state_dict``). Handles ``AveragedModel`` by falling through to
    ``model.module`` when the buffer is not on the wrapper.
    """
    B = getattr(model, "fourier_B", None)
    if B is None and hasattr(model, "module"):
        B = getattr(model.module, "fourier_B", None)
    if B is None:
        return x_norm
    coord = x_norm[..., :2]                       # [..., 2]
    proj = coord @ B                              # [..., n_fourier]
    ff = torch.cat([torch.sin(2 * math.pi * proj),
                    torch.cos(2 * math.pi * proj)], dim=-1)
    return torch.cat([x_norm, ff], dim=-1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   split_name: str = "") -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    Two-pronged defensive guard (advisor update on PR #3296):
      1. Non-finite model predictions (overflow) → ``nan_to_num`` to 0 so the
         denormalized prediction stays finite. Bad-node MAE becomes ``|y_true|``.
      2. Non-finite *ground truth* values (one known bad sample in
         ``test_geom_camber_cruise``: ``.test_geom_camber_cruise_gt/000020.pt``
         has 761 Inf p-channel values) → drop entire bad samples from the
         mask and replace their y values with 0 so ``(pred - y).abs() * mask``
         does not produce ``inf * 0 = NaN`` that poisons the accumulator.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0
    nf_pred_total = 0
    nf_pred_orig_total = 0
    nf_y_samples_total = 0
    nf_y_nodes_total = 0

    with torch.no_grad():
        for batch_idx, (x, y, is_surface, mask) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            x_norm = _apply_fourier(x_norm, model)
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            log_re_raw = x[:, 0, 13]  # raw log(Re) per sample, [B]
            pred = model({"x": x_norm, "log_re_raw": log_re_raw})["preds"]

            # Diagnostic: model produced non-finite values in normalized space?
            nf_pred = (~torch.isfinite(pred)) & mask.unsqueeze(-1)
            if nf_pred.any():
                n_bad = int(nf_pred.sum().item())
                nf_pred_total += n_bad
                finite_p = pred[torch.isfinite(pred)]
                max_abs_finite = (
                    finite_p.abs().max().item() if finite_p.numel() else float("nan")
                )
                print(
                    f"[DIAGNOSTIC] {split_name} batch {batch_idx}: "
                    f"{n_bad} non-finite normalized pred nodes (model output overflow), "
                    f"pred.shape={tuple(pred.shape)}, max_abs_finite={max_abs_finite:.3e}"
                )
                for i in range(pred.shape[0]):
                    sample_nf = nf_pred[i]
                    if sample_nf.any():
                        ch_counts = sample_nf.sum(dim=0).tolist()  # [Ux, Uy, p]
                        n_bad_surf = int(
                            (sample_nf & is_surface[i].unsqueeze(-1)).sum().item()
                        )
                        print(
                            f"  sample {i}: {int(sample_nf.sum().item())} bad nodes "
                            f"({n_bad_surf} surface), per-channel(Ux,Uy,p)={ch_counts}"
                        )

            # Guard against inf/nan predictions corrupting normalized loss.
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            # Y-side guard: detect samples with non-finite ground-truth values
            # and substitute zeros + drop them from the effective mask so the
            # masked accumulator never sees inf*0 = NaN.
            y_finite_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if (~y_finite_sample).any():
                bad_samples = int((~y_finite_sample).sum().item())
                nf_y_samples_total += bad_samples
                nf_y_mask = ~torch.isfinite(y)
                n_bad_y_nodes = int(nf_y_mask.sum().item())
                nf_y_nodes_total += n_bad_y_nodes
                # Per-sample, per-channel breakdown
                for i in range(y.shape[0]):
                    if not y_finite_sample[i].item():
                        nf_i = nf_y_mask[i]
                        ch_counts = nf_i.sum(dim=0).tolist()  # [Ux, Uy, p]
                        print(
                            f"[DIAGNOSTIC] {split_name} batch {batch_idx} sample {i}: "
                            f"non-finite GROUND TRUTH ({int(nf_i.sum().item())} nodes), "
                            f"per-channel(Ux,Uy,p)={ch_counts} — sample dropped from MAE"
                        )
            y_norm_safe = torch.where(torch.isfinite(y_norm), y_norm, torch.zeros_like(y_norm))
            y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            mask_safe = mask & y_finite_sample[:, None].expand_as(mask)

            sq_err = _residual_err(pred, y_norm_safe, cfg.loss_type, cfg.loss_beta)
            vol_mask = mask_safe & ~is_surface
            surf_mask = mask_safe & is_surface
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

            # Diagnostic: catches denorm overflow (pred finite but pred_orig inf)
            nf_orig = (~torch.isfinite(pred_orig)) & mask.unsqueeze(-1)
            if nf_orig.any():
                n_bad_o = int(nf_orig.sum().item())
                nf_pred_orig_total += n_bad_o
                only_denorm = nf_orig & ~nf_pred
                if only_denorm.any():
                    print(
                        f"[DIAGNOSTIC] {split_name} batch {batch_idx}: "
                        f"{int(only_denorm.sum().item())} extra non-finite nodes "
                        f"introduced by denormalization (pred finite, pred_orig inf)"
                    )

            # Defensive: zero out non-finite preds so the float64 MAE accumulator
            # stays finite. Bad-node MAE becomes |y_true - 0| = |y_true|.
            pred_orig = torch.nan_to_num(pred_orig, nan=0.0, posinf=0.0, neginf=0.0)

            ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask_safe, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    if nf_pred_total or nf_pred_orig_total or nf_y_samples_total:
        print(
            f"[DIAGNOSTIC] {split_name}: totals — "
            f"non-finite pred (norm)={nf_pred_total}, "
            f"non-finite pred (denorm)={nf_pred_orig_total}, "
            f"non-finite y samples={nf_y_samples_total} ({nf_y_nodes_total} nodes)"
        )

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


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023, arXiv 2302.06675).

    Sign-based update with decoupled weight decay; tracks one momentum buffer
    (no second moment), so ~half the memory of AdamW. Paper recipe is 3-10×
    smaller lr and 3-10× larger weight_decay vs AdamW.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - lr * wd)
                update = m.mul(beta1).add(p.grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss


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
    n_fourier: int = 0       # 0 disables; otherwise number of random Fourier features over (x, z)
    fourier_sigma: float = 10.0
    loss_type: str = "mse"   # "mse" or "smooth_l1"
    loss_beta: float = 0.1   # SmoothL1 beta (normalized-space)
    cosine_t_max: int | None = None  # Cosine T_max in epochs; defaults to MAX_EPOCHS
    optimizer_name: str = "adamw"  # "adamw" or "lion"
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    use_film: bool = False   # enable FiLM conditioning on log(Re)
    film_mode: str = "output_only"  # "output_only" or "all_blocks"
    ema_decay: float = 0.0   # 0 disables EMA; e.g. 0.999 enables EMA-of-weights
    grad_clip: float = 0.0   # 0 disables; e.g. 1.0 clips global grad norm
    llrd_gamma: float = 1.0  # 1.0 disables LLRD; <1 applies per-block LR decay from output toward input


def _residual_err(pred, target, loss_type, beta):
    """Per-element residual: squared for MSE, smooth_l1 (Huber) otherwise."""
    if loss_type == "mse":
        return (pred - target) ** 2
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    raise ValueError(loss_type)


def make_llrd_param_groups(model, base_lr, gamma, weight_decay):
    """Build per-block LR groups for Transolver (layer-wise LR decay).

    Tier layout (γ<1 → output trains fastest, input slowest):
        input_embed (preprocess + placeholder)         → base_lr * γ^(N+1)
        block_i   (i=0..N-1, last block excludes head) → base_lr * γ^(N-i)
        output_head (last block ln_3 + mlp2)           → base_lr * γ^0 = base_lr
        film (if present)                              → base_lr
    """
    blocks = list(model.blocks)
    num_blocks = len(blocks)
    last_block = blocks[-1]

    # Output head is the last block's ln_3 + mlp2.
    output_head_param_ids = set()
    output_head_params = []
    for mod in (last_block.ln_3, last_block.mlp2):
        for p in mod.parameters():
            output_head_param_ids.add(id(p))
            output_head_params.append(p)

    # Input embed: preprocess MLP + placeholder bias.
    input_params = list(model.preprocess.parameters()) + [model.placeholder]

    groups = []
    groups.append({
        "params": input_params,
        "lr": base_lr * (gamma ** (num_blocks + 1)),
        "weight_decay": weight_decay,
        "name": "input_embed",
    })
    for i, blk in enumerate(blocks):
        if i == num_blocks - 1:
            blk_params = [p for p in blk.parameters() if id(p) not in output_head_param_ids]
        else:
            blk_params = list(blk.parameters())
        groups.append({
            "params": blk_params,
            "lr": base_lr * (gamma ** (num_blocks - i)),
            "weight_decay": weight_decay,
            "name": f"block_{i}",
        })
    groups.append({
        "params": output_head_params,
        "lr": base_lr,
        "weight_decay": weight_decay,
        "name": "output_head",
    })
    if hasattr(model, "film"):
        groups.append({
            "params": list(model.film.parameters()),
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "film",
        })

    # Sanity: param coverage equals model.parameters() count.
    grouped_ids = set()
    for g in groups:
        for p in g["params"]:
            grouped_ids.add(id(p))
    model_ids = set(id(p) for p in model.parameters() if p.requires_grad)
    missing = model_ids - grouped_ids
    extra = grouped_ids - model_ids
    if missing or extra:
        raise RuntimeError(
            f"[LLRD] param coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    return groups


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

fun_dim = X_DIM - 2 + (2 * cfg.n_fourier if cfg.n_fourier > 0 else 0)
model_config = dict(
    space_dim=2,
    fun_dim=fun_dim,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    use_film=cfg.use_film,
    film_mode=cfg.film_mode,
    log_re_mean=float(stats["x_mean"][13].item()),
    log_re_std=float(stats["x_std"][13].item()),
)

model = Transolver(**model_config).to(device)
if cfg.n_fourier > 0:
    fourier_gen = torch.Generator()
    fourier_gen.manual_seed(0)
    fourier_B = (torch.randn(2, cfg.n_fourier, generator=fourier_gen)
                 * cfg.fourier_sigma).to(device)
    model.register_buffer("fourier_B", fourier_B, persistent=True)
    print(f"Fourier features: n_fourier={cfg.n_fourier}, sigma={cfg.fourier_sigma}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

ema_model = None
if cfg.ema_decay > 0:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay)
    ).to(device)
    print(f"EMA enabled (decay={cfg.ema_decay})")

if cfg.llrd_gamma < 1.0:
    param_groups = make_llrd_param_groups(model, cfg.lr, cfg.llrd_gamma, cfg.weight_decay)
    print(f"[LLRD] gamma={cfg.llrd_gamma}, num_groups={len(param_groups)}")
    for g in param_groups:
        n_params_g = sum(p.numel() for p in g["params"])
        print(f"  group {g['name']:<14s} lr={g['lr']:.3e} ({g['lr']/cfg.lr:.3f}*base) "
              f"params={n_params_g:,}")
    opt_params = param_groups
else:
    opt_params = model.parameters()

if cfg.optimizer_name == "lion":
    optimizer = Lion(opt_params, lr=cfg.lr,
                     betas=(cfg.lion_beta1, cfg.lion_beta2),
                     weight_decay=cfg.weight_decay)
    print(f"Optimizer: Lion(lr={cfg.lr}, betas=({cfg.lion_beta1}, {cfg.lion_beta2}), wd={cfg.weight_decay})")
elif cfg.optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"Optimizer: AdamW(lr={cfg.lr}, wd={cfg.weight_decay})")
else:
    raise ValueError(f"Unknown optimizer_name: {cfg.optimizer_name!r}")
cosine_t_max = cfg.cosine_t_max if cfg.cosine_t_max is not None else MAX_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)
print(f"Scheduler: CosineAnnealingLR(T_max={cosine_t_max})  [epochs cap = {MAX_EPOCHS}]")

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

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        x_norm = _apply_fourier(x_norm, model)
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        log_re_raw = x[:, 0, 13]  # raw log(Re) per sample, [B]
        pred = model({"x": x_norm, "log_re_raw": log_re_raw})["preds"]
        # Two-pronged guard (advisor update on PR #3296):
        #   1. Replace inf/nan predictions with 0 (prevents inf gradients).
        #   2. Drop samples whose GT y is non-finite (one known bad sample
        #      lives in test_geom_camber_cruise; this is purely defensive in
        #      the training loop in case any train sample is similarly bad).
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        y_finite_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
        y_norm_safe = torch.where(torch.isfinite(y_norm), y_norm, torch.zeros_like(y_norm))
        mask_safe = mask & y_finite_sample[:, None].expand_as(mask)
        sq_err = _residual_err(pred, y_norm_safe, cfg.loss_type, cfg.loss_beta)

        vol_mask = mask_safe & ~is_surface
        surf_mask = mask_safe & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        # grad clip (and norm reporting): inf max_norm computes norm without clipping
        clip_at = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_at).item()
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm,
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    # With EMA enabled, evaluate the EMA shadow (the actual predictor we'll
    # checkpoint). The raw model is just an intermediate state.
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    model.eval()
    split_metrics = {
        name: evaluate_split(eval_model, loader, stats, cfg.surf_weight, device, split_name=name)
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
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    # Per-group LRs (LLRD verification + general visibility)
    for i, g in enumerate(optimizer.param_groups):
        gname = g.get("name", f"g{i}")
        log_metrics[f"lr_group/{gname}"] = g["lr"]
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.

    # FiLM γ/β diagnostics across the val_re_rand split's Re range
    if cfg.use_film:
        diag_model = ema_model.module if ema_model is not None else model
        with torch.no_grad():
            re_lo, re_hi = 11.5, 15.4
            log_re_probe = torch.linspace(re_lo, re_hi, 32, device=device)
            log_re_n = (log_re_probe - diag_model.log_re_mean) / diag_model.log_re_std
            gamma, beta = diag_model.film(log_re_n)  # [32, 1, n_hidden]
            log_metrics["film/gamma_max_abs_delta"] = (gamma - 1.0).abs().max().item()
            log_metrics["film/gamma_std"] = gamma.std().item()
            log_metrics["film/beta_abs_max"] = beta.abs().max().item()
            log_metrics["film/beta_std"] = beta.std().item()
    wandb.log(log_metrics)

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        # Save EMA weights when EMA is on — that's the actual predictor.
        sd_to_save = (ema_model.module if ema_model is not None else model).state_dict()
        torch.save(sd_to_save, model_path)
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

# --- Test evaluation + artifact upload ---
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, split_name=name)
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
