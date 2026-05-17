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

import contextlib
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
                 two_shot_film=False):
        super().__init__()
        self.last_layer = last_layer
        self.two_shot_film = two_shot_film
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

    def forward(self, fx, scale=None, shift=None):
        h = self.ln_1(fx)
        if scale is not None:
            h = h * scale.unsqueeze(1) + shift.unsqueeze(1)
        fx = self.attn(h) + fx
        h2 = self.ln_2(fx)
        if self.two_shot_film and scale is not None:
            h2 = h2 * scale.unsqueeze(1) + shift.unsqueeze(1)
        fx = self.mlp(h2) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class FiLMConditioner(nn.Module):
    """Maps per-sample condition scalars to per-layer (scale, shift) for hidden_dim.

    Zero-init the final linear so scale=1, shift=0 at start (identity init).
    """

    def __init__(self, cond_dim: int, hidden_dim: int, n_layers: int, mlp_hidden: int = 128):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, 2 * hidden_dim * n_layers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond):
        out = self.net(cond)
        out = out.reshape(out.size(0), self.n_layers, 2, self.hidden_dim)
        scale = 1.0 + out[:, :, 0]
        shift = out[:, :, 1]
        return scale, shift


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 film_cond: bool = False, film_cond_dim: int = 11,
                 film_cond_slice: tuple[int, int] = (13, 24),
                 film_mlp_hidden: int = 128,
                 two_shot_film: bool = False):
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
        self.n_layers = n_layers
        self.two_shot_film = two_shot_film
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                two_shot_film=two_shot_film,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        self.film_cond = film_cond
        self.film_cond_slice = film_cond_slice
        if film_cond:
            self.film = FiLMConditioner(
                cond_dim=film_cond_dim, hidden_dim=n_hidden,
                n_layers=n_layers, mlp_hidden=film_mlp_hidden,
            )

        self.apply(self._init_weights)
        # Re-zero FiLM output head after generic init (the apply above clobbers it).
        if film_cond:
            nn.init.zeros_(self.film.net[-1].weight)
            nn.init.zeros_(self.film.net[-1].bias)

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
        if self.film_cond:
            s0, s1 = self.film_cond_slice
            cond = x[:, 0, s0:s1]
            scales, shifts = self.film(cond)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for i, block in enumerate(self.blocks):
            if self.film_cond:
                fx = block(fx, scale=scales[:, i], shift=shifts[:, i])
            else:
                fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# EMA — exponential moving average of model weights
# ---------------------------------------------------------------------------


class EMA:
    """Karras-style EMA of model weights with a warmup decay ramp.

    ``decay_eff = min(decay, (1 + step) / (10 + step))`` avoids the early-training
    pathology where the EMA shadow is dominated by random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.step = 0

    def effective_decay(self) -> float:
        return min(self.decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.effective_decay()
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """Swap EMA weights into ``model.parameters().data``; return a restore callback."""
        saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

        @torch.no_grad()
        def restore() -> None:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(saved[n])

        return restore


# ---------------------------------------------------------------------------
# Lion optimizer — sign-based update with EMA momentum (Chen et al. 2023)
# ---------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    """Lion optimizer (arXiv:2302.06675), Algorithm 2.

        update = sign(β1 · m + (1 − β1) · g)
        θ     ← (1 − lr·λ) · θ − lr · update        # additive form expands to this
        m     ← β2 · m + (1 − β2) · g               # uses the pre-step grad

    Half the optimizer state of AdamW (no second moment) and produces unit-L∞-norm
    updates, so lr should typically be 3–10× smaller and weight_decay 3–10× larger
    than the AdamW recipe for the same task.
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
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
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # Interpolated momentum, sign-projected. Clone so exp_avg is intact
                # for its own EMA update below.
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.data.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class LionScheduleFree(torch.optim.Optimizer):
    """Lion + Schedule-Free composition.

    Sign-projected direction (Lion) applied to the SF z-iterate; the Polyak
    average x is swapped into p via .eval() for validation/checkpointing.
    Mirrors AdamWScheduleFree's in-place state layout: p holds y during train
    mode, x is computed on-the-fly via lerp during eval. Maintains the same
    2x param state as plain Lion (z + exp_avg).

    Args:
        params:           model parameters
        lr:               base learning rate
        betas:            (beta1, beta2) for Lion momentum (separate from SF momentum)
        sf_momentum:      SF interpolation weight controlling y = sf_m*x + (1-sf_m)*z
        weight_decay:     decoupled WD applied multiplicatively to y and z
        warmup_steps:     linear LR warmup, mirroring AdamWScheduleFree
        r:                polynomial weighting power for Polyak average
        weight_lr_power:  lr power for Polyak weights
    """

    def __init__(self, params, lr: float = 1.5e-4, betas: tuple[float, float] = (0.9, 0.99),
                 sf_momentum: float = 0.9, weight_decay: float = 0.0,
                 warmup_steps: int = 0, r: float = 0.0, weight_lr_power: float = 2.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 < sf_momentum < 1.0:
            raise ValueError(f"Invalid sf_momentum: {sf_momentum} (must be in (0,1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(
            lr=lr, betas=betas, sf_momentum=sf_momentum,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            r=r, k=0, train_mode=False, weight_sum=0.0,
            lr_max=-1.0, scheduled_lr=0.0, weight_lr_power=weight_lr_power,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                sf_m = group["sf_momentum"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # p = y → x via lerp:  x = (1/sf_m)*y + (1 - 1/sf_m)*z
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / sf_m)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                sf_m = group["sf_momentum"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # p = x → y via lerp:  y = sf_m*x + (1 - sf_m)*z
                        p.lerp_(end=state["z"], weight=1.0 - sf_m)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "LionScheduleFree.step() called outside train mode. "
                "Call .train() before training and .eval() before validation."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            sf_m = group["sf_momentum"]
            decay = group["weight_decay"]
            k = group["k"]
            r_pow = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if (warmup_steps > 0 and k < warmup_steps) else 1.0
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r_pow) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                y = p  # train mode: p holds y
                grad = p.grad
                z = state["z"]
                exp_avg = state["exp_avg"]

                # Lion sign-projected direction evaluated at y (grad was computed at y)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()

                # Schedule-Free y/z updates (z step uses unsigned z_k below)
                y.lerp_(end=z, weight=ckp1)
                y.add_(update, alpha=lr * (sf_m * (1.0 - ckp1) - 1.0))
                z.sub_(update, alpha=lr)

                # Decoupled weight decay applied multiplicatively to both iterates
                if decay != 0.0:
                    decay_factor = 1.0 - lr * decay
                    y.mul_(decay_factor)
                    z.mul_(decay_factor)

                # Lion momentum EMA (uses raw grad, post-step)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

            group["k"] = k + 1
        return loss


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16}


def make_amp_ctx(amp_dtype: str):
    """Build the autocast context for ``amp_dtype`` (``"fp32"`` or ``"bf16"``)."""
    dt = _AMP_DTYPES[amp_dtype]
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dt)


def evaluate_split(model, loader, stats, surf_weight, device, amp_dtype: str = "fp32") -> dict[str, float]:
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
            with make_amp_ctx(amp_dtype):
                pred = model({"x": x_norm})["preds"]
                sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')
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
        "amp_dtype": cfg.amp_dtype,
        "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay if cfg.use_ema else None,
        "grad_clip_norm": cfg.grad_clip_norm,
        "use_schedule_free": cfg.use_schedule_free,
        "sf_warmup_steps": cfg.sf_warmup_steps if cfg.use_schedule_free else None,
        "sf_momentum": cfg.sf_momentum if cfg.use_schedule_free else None,
        "optimizer": cfg.optimizer,
        "optimizer_class": type(optimizer).__name__,
        "lion_lr": cfg.lion_lr if cfg.optimizer == "lion" else None,
        "lion_weight_decay": cfg.lion_weight_decay if cfg.optimizer == "lion" else None,
        "lion_betas": cfg.lion_betas if cfg.optimizer == "lion" else None,
        "seed": cfg.seed,
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
    cosine_t_max: int | None = None  # if None, use MAX_EPOCHS (existing behavior)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    amp_dtype: str = "fp32"  # one of: "fp32", "bf16"
    use_ema: bool = False  # track an EMA shadow copy of weights for val/test/checkpoint
    ema_decay: float = 0.999  # max decay; warmup ramp protects early training
    film_cond: bool = False  # enable per-block FiLM conditioning on x[:,0,13:24]
    film_mlp_hidden: int = 128
    two_shot_film: bool = False  # apply FiLM modulation at both attn and mlp sites per block
    grad_clip_norm: float | None = None  # if set, clip gradient L2 norm before optimizer.step()
    use_schedule_free: bool = False  # AdamWScheduleFree — drop the cosine scheduler
    sf_warmup_steps: int = 500  # warmup steps for Schedule-Free (README recommends warmup)
    sf_momentum: float = 0.9  # SF interpolation weight (used by LionScheduleFree only; AdamW SF reuses beta1)
    optimizer: str = "adamw"  # one of: "adamw", "lion"
    lion_lr: float = 1.5e-4  # Lion lr (typically 3-10x smaller than AdamW; 3x of cfg.lr=5e-4)
    lion_weight_decay: float = 3e-4  # Lion wd (typically 3-10x larger than AdamW; 3x of cfg.weight_decay=1e-4)
    lion_betas: str = "0.9,0.99"  # Lion (beta1, beta2) as comma-separated floats
    seed: int = 0  # RNG seed for model init + data sampler; enables paired-arm reproducibility


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else "") + f"  seed={cfg.seed}")

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
    film_cond=cfg.film_cond,
    film_cond_dim=11,
    film_cond_slice=(13, 24),
    film_mlp_hidden=cfg.film_mlp_hidden,
    two_shot_film=cfg.two_shot_film,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
if ema is not None:
    print(f"EMA: enabled (decay={cfg.ema_decay}, warmup ramp (1+step)/(10+step))")

if cfg.optimizer == "lion" and cfg.use_schedule_free:
    lion_betas = tuple(float(b.strip()) for b in cfg.lion_betas.split(","))
    if len(lion_betas) != 2:
        raise ValueError(f"--lion_betas must be two comma-separated floats, got {cfg.lion_betas!r}")
    optimizer = LionScheduleFree(
        model.parameters(),
        lr=cfg.lion_lr,
        betas=lion_betas,
        sf_momentum=cfg.sf_momentum,
        weight_decay=cfg.lion_weight_decay,
        warmup_steps=cfg.sf_warmup_steps,
    )
    scheduler = None
    t_max_eff = None
    print(
        f"Optimizer: LionScheduleFree(lr={cfg.lion_lr}, betas={lion_betas}, "
        f"sf_momentum={cfg.sf_momentum}, weight_decay={cfg.lion_weight_decay}, "
        f"warmup_steps={cfg.sf_warmup_steps}); no scheduler"
    )
elif cfg.optimizer == "lion":
    lion_betas = tuple(float(b.strip()) for b in cfg.lion_betas.split(","))
    if len(lion_betas) != 2:
        raise ValueError(f"--lion_betas must be two comma-separated floats, got {cfg.lion_betas!r}")
    optimizer = Lion(model.parameters(), lr=cfg.lion_lr,
                     betas=lion_betas, weight_decay=cfg.lion_weight_decay)
    print(f"Optimizer: Lion (lr={cfg.lion_lr}, weight_decay={cfg.lion_weight_decay}, betas={lion_betas})")
    t_max_eff = cfg.cosine_t_max if cfg.cosine_t_max is not None else MAX_EPOCHS
    if cfg.cosine_t_max is not None:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_eff),
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1e-4, total_iters=1_000_000),
            ],
            milestones=[t_max_eff],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    print(f"Scheduler: cosine T_max={t_max_eff}, max_epochs={MAX_EPOCHS}")
elif cfg.use_schedule_free:
    from schedulefree import AdamWScheduleFree
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.sf_warmup_steps,
    )
    scheduler = None
    t_max_eff = None
    print(
        f"Optimizer: AdamWScheduleFree(lr={cfg.lr}, weight_decay={cfg.weight_decay}, "
        f"warmup_steps={cfg.sf_warmup_steps}); no scheduler"
    )
elif cfg.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"Optimizer: AdamW (lr={cfg.lr}, weight_decay={cfg.weight_decay})")
    t_max_eff = cfg.cosine_t_max if cfg.cosine_t_max is not None else MAX_EPOCHS
    if cfg.cosine_t_max is not None:
        # Decay fully across t_max_eff epochs, then hold near zero for any overshoot.
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_eff),
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1e-4, total_iters=1_000_000),
            ],
            milestones=[t_max_eff],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    print(f"Scheduler: cosine T_max={t_max_eff}, max_epochs={MAX_EPOCHS}")
else:
    raise ValueError(f"Unknown --optimizer: {cfg.optimizer!r} (expected 'adamw' or 'lion')")

print(f"Optimizer class: {type(optimizer).__name__}")

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
    if cfg.use_schedule_free:
        optimizer.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0
    grad_norms_epoch: list[float] = []
    clip_active_epoch = 0

    clip_threshold = cfg.grad_clip_norm if cfg.grad_clip_norm is not None else float("inf")

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with make_amp_ctx(cfg.amp_dtype):
            pred = model({"x": x_norm})["preds"]
            sq_err = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
        grad_norm_val = grad_norm_before.item()
        grad_norms_epoch.append(grad_norm_val)
        if cfg.grad_clip_norm is not None and grad_norm_val > cfg.grad_clip_norm:
            clip_active_epoch += 1
        optimizer.step()
        if ema is not None:
            ema.update(model)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    epoch_lr = optimizer.param_groups[0]["lr"]
    if scheduler is not None:
        scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate (with EMA-applied weights if enabled; save best while applied) ---
    model.eval()
    if cfg.use_schedule_free:
        optimizer.eval()  # swap params to Polyak-averaged iterate for eval
    restore = ema.apply_to(model) if ema is not None else None
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.amp_dtype)
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

    if restore is not None:
        restore()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    epoch_record = {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": epoch_lr,
        "cosine_t_max": t_max_eff,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "use_schedule_free": cfg.use_schedule_free,
    }
    if cfg.use_schedule_free:
        epoch_record["sf_warmup_steps"] = cfg.sf_warmup_steps
    if ema is not None:
        epoch_record["ema_step"] = ema.step
        epoch_record["ema_effective_decay"] = ema.effective_decay()
    if grad_norms_epoch:
        gn = torch.tensor(grad_norms_epoch, dtype=torch.float64)
        gn_finite = gn[torch.isfinite(gn)]
        if gn_finite.numel() > 0:
            epoch_record["grad_norm/p50"] = torch.quantile(gn_finite, 0.5).item()
            epoch_record["grad_norm/p90"] = torch.quantile(gn_finite, 0.9).item()
            epoch_record["grad_norm/p99"] = torch.quantile(gn_finite, 0.99).item()
            epoch_record["grad_norm/max"] = gn_finite.max().item()
            epoch_record["grad_norm/mean"] = gn_finite.mean().item()
        epoch_record["grad_norm/n_nonfinite"] = int((gn.numel() - gn_finite.numel()))
        epoch_record["grad_norm/n_steps"] = int(gn.numel())
        if cfg.grad_clip_norm is not None:
            epoch_record["grad_clip_norm"] = cfg.grad_clip_norm
            epoch_record["grad_norm/clip_rate"] = clip_active_epoch / max(gn.numel(), 1)
    append_metrics_jsonl(metrics_jsonl_path, epoch_record)
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.amp_dtype)
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
