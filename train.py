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
# Fourier positional encoding (Tancik et al., 2020, arXiv:2006.10739)
# ---------------------------------------------------------------------------

POS_COLS = slice(0, 2)  # (x, z) spatial coords — dims 0-1 per program.md


def fourier_pos_encode(coords: torch.Tensor, num_freq: int) -> torch.Tensor:
    """NeRF-style log-spaced sinusoidal features for 2D coordinates.

    Returns ``(N, 4*num_freq)``: sin and cos at each of ``num_freq`` log-spaced
    frequencies (``2^k * pi`` for k in 0..num_freq-1) for each of the 2 coords.
    """
    freqs = 2.0 ** torch.arange(num_freq, dtype=coords.dtype, device=coords.device)
    angles = coords.unsqueeze(-1) * freqs[None, None, :] * math.pi   # (N, 2, num_freq)
    enc = torch.cat([angles.sin(), angles.cos()], dim=-1)            # (N, 2, 2*num_freq)
    return enc.reshape(coords.shape[0], -1)                          # (N, 4*num_freq)


def encode_inputs(x_norm: torch.Tensor, num_freq: int) -> torch.Tensor:
    """Replace (x, z) cols of normalized features with Fourier encoding."""
    pos = x_norm[..., POS_COLS]
    flat = pos.reshape(-1, 2)
    enc = fourier_pos_encode(flat, num_freq).reshape(*x_norm.shape[:-1], 4 * num_freq)
    non_pos = torch.cat(
        [x_norm[..., : POS_COLS.start], x_norm[..., POS_COLS.stop :]], dim=-1
    )
    return torch.cat([enc, non_pos], dim=-1)


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
    """SwiGLU FFN (Shazeer 2020, arXiv:2002.05202).

    Replaces a vanilla 2-layer GeLU FFN with a gated variant:
        out = w3( silu(w1(x)) * w2(x) )
    Inner dim is set to `round_to_mult(hidden_dim * 2/3, 8)` so 3 matrices of
    size (in_dim, inner) approximately match the param count of a vanilla
    2-matrix FFN of width `hidden_dim`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        inner = max(8, int((hidden_dim * 2 / 3 + 4) // 8) * 8)
        self.inner = inner
        self.w1 = nn.Linear(in_dim, inner, bias=False)
        self.w2 = nn.Linear(in_dim, inner, bias=False)
        self.w3 = nn.Linear(inner, out_dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 use_qk_norm=False):
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
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(dim_head)
            self.k_norm = nn.LayerNorm(dim_head)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Set _log_diag=True before a forward to capture per-vector L2-norm
        # diagnostics into self._diag (mean over [B, heads, slice_num]). The
        # flag is consumed (reset) inside the forward to avoid stale reads.
        self._log_diag: bool = False
        self._diag: dict[str, float] = {}

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

        log_diag = self._log_diag
        if log_diag:
            self._log_diag = False
            self._diag = {
                "q_norm_pre": q.detach().float().norm(dim=-1).mean().item(),
                "k_norm_pre": k.detach().float().norm(dim=-1).mean().item(),
            }

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if log_diag:
            self._diag["q_norm_post"] = q.detach().float().norm(dim=-1).mean().item()
            self._diag["k_norm_post"] = k.detach().float().norm(dim=-1).mean().item()

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
                 use_qk_norm=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, use_qk_norm=use_qk_norm,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLUFFN(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
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
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_qk_norm=False):
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
                use_qk_norm=use_qk_norm,
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

def _pointwise_loss(pred, y_norm, loss_type: str):
    if loss_type == "mse":
        return (pred - y_norm) ** 2
    if loss_type == "l1":
        return (pred - y_norm).abs()
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, y_norm, reduction="none", beta=1.0)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def _tta_predict(model, x_norm, mask, num_freq, amp_ctx, tta_k: int, tta_noise_std: float,
                 generator: torch.Generator | None):
    """Forward pass with optional K-sample TTA over Gaussian coord-noise perturbations.

    Noise is applied to the raw (x, z) coords (dims 0-1) BEFORE Fourier encoding,
    so each pass sees an independently perturbed geometry. Per-pass noise is IID
    Gaussian with std ``tta_noise_std``, zeroed at padding via ``mask``.

    Returns ``(pred_mean, pred_std)`` in fp32 normalized output space. ``pred_std``
    is zero when ``tta_k <= 1`` (single deterministic pass).
    """
    if tta_k <= 1:
        x_enc = encode_inputs(x_norm, num_freq)
        with amp_ctx:
            pred = model({"x": x_enc})["preds"]
        pred = pred.float()
        return pred, torch.zeros_like(pred)

    pad_mask = mask.unsqueeze(-1).to(x_norm.dtype)  # (B, N, 1); zero at padding
    sum_pred = None
    sum_sq = None
    for _ in range(tta_k):
        noise = torch.randn(
            x_norm.shape[:-1] + (2,),
            dtype=x_norm.dtype, device=x_norm.device, generator=generator,
        ) * tta_noise_std
        x_perturbed = x_norm.clone()
        x_perturbed[..., :2] = x_norm[..., :2] + noise * pad_mask
        x_enc = encode_inputs(x_perturbed, num_freq)
        with amp_ctx:
            pred_k = model({"x": x_enc})["preds"]
        pred_k = pred_k.float()
        if sum_pred is None:
            sum_pred = pred_k
            sum_sq = pred_k * pred_k
        else:
            sum_pred = sum_pred + pred_k
            sum_sq = sum_sq + pred_k * pred_k
    pred_mean = sum_pred / tta_k
    pred_var = (sum_sq / tta_k - pred_mean * pred_mean).clamp(min=0)
    pred_std = pred_var.sqrt()
    return pred_mean, pred_std


def evaluate_split(model, loader, stats, surf_weight, device, num_freq, loss_type: str = "l1",
                   use_bf16: bool = False, tta_k: int = 0, tta_noise_std: float = 0.0,
                   tta_generator: torch.Generator | None = None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    When ``tta_k > 1``, each batch is forwarded ``tta_k`` times with independent
    Gaussian coord-noise (std ``tta_noise_std``) added to dims 0-1 of x_norm
    pre-Fourier; metrics use the averaged prediction. ``mean_tta_std_norm`` is
    the per-element K-pass std (in normalized output space) averaged over valid
    nodes and channels — a diagnostic for how much TTA is actually engaging.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0
    tta_std_sum = 0.0
    tta_std_count = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type="cuda", enabled=False)
    )

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred, pred_std = _tta_predict(
                model, x_norm, mask, num_freq, amp_ctx,
                tta_k=tta_k, tta_noise_std=tta_noise_std, generator=tta_generator,
            )
            if tta_k > 1:
                m3 = mask.unsqueeze(-1).to(pred_std.dtype)
                denom = m3.sum() * pred_std.shape[-1]
                tta_std_sum += (pred_std * m3).sum().item()
                tta_std_count += int(denom.item())

            # NaN guard: some samples (e.g. test_geom_camber_cruise idx 20) have
            # non-finite ground-truth. IEEE 754 propagates NaN through NaN * 0,
            # so masking alone leaks NaN into the per-split loss/MAE sums.
            finite_y = torch.isfinite(y_norm)
            err = _pointwise_loss(pred, torch.where(finite_y, y_norm, torch.zeros_like(y_norm)), loss_type)
            err = torch.where(finite_y, err, torch.zeros_like(err))
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (err * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            # Subset to fully-finite samples so the scoring accumulator's
            # per-sample skip works (it would otherwise still hit NaN*0=NaN
            # when err contains NaN at non-finite y positions).
            B = y.shape[0]
            sample_keep = torch.isfinite(y.view(B, -1)).all(dim=-1)
            if sample_keep.any():
                idx = sample_keep.nonzero(as_tuple=True)[0]
                ds, dv = accumulate_batch(
                    pred_orig[idx], y[idx], is_surface[idx], mask[idx], mae_surf, mae_vol,
                )
                n_surf += ds
                n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    if tta_k > 1 and tta_std_count > 0:
        out["mean_tta_std_norm"] = tta_std_sum / tta_std_count
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
# Lion optimizer (Chen et al. 2023, arxiv:2302.06675)
# ---------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    """Lion optimizer: sign-of-momentum update with decoupled weight decay.

    update_t = sign(beta1 * m_{t-1} + (1-beta1) * g_t)
    theta_t  = (1 - lr*wd) * theta_{t-1} - lr * update_t        # decoupled wd
    m_t      = beta2 * m_{t-1} + (1-beta2) * g_t

    Uses one momentum buffer (vs Adam's two). Sign update gives identical
    per-step magnitude `lr` for every parameter; tune lr ~ AdamW_lr/3..10 and
    wd ~ AdamW_wd*3..10 per the paper.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
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
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # update = sign(beta1*m + (1-beta1)*g). Use non-inplace mul so
                # exp_avg is not mutated before its own beta2 update below.
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)  # decoupled weight decay
                p.data.add_(update, alpha=-lr)  # sign update
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)  # momentum
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
    loss_type: str = "l1"  # mse | l1 | huber — l1 won 12.9% over huber, locking it in
    num_freq: int = 4  # Fourier positional-encoding frequencies (Tancik 2020); 4 won vs 8
    coord_noise_std: float = 0.01  # Gaussian noise std on normalized (x,z) coords during training
    mlp_ratio: int = 2  # FFN expansion ratio; SwiGLU inner = round_to_mult(hidden*mlp_ratio*2/3, 8)
    use_bf16: bool = False  # bf16 autocast (activations only; params/optimizer stay fp32)
    n_hidden: int = 160  # Transolver hidden dim (embedding/attention/FFN base width)
    use_lion: bool = False  # use Lion optimizer (sign-of-momentum) instead of AdamW
    lion_lr: float = 1e-4  # Lion lr; canonical: AdamW_lr / 3..10
    lion_wd: float = 1e-3  # Lion wd; canonical: AdamW_wd * 3..10
    use_qk_norm: bool = False  # ViT-22B-style LayerNorm(head_dim) on Q and K before SDPA
    tta_k: int = 0  # K-sample test-time augmentation at end-of-run eval; 0 = disabled
    tta_noise_std: float = 0.005  # coord-noise std (dims 0-1) for TTA perturbations
    tta_seed: int = 1234  # seed for TTA noise generator (reproducible per-pass draws)


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

ENCODED_X_DIM = 4 * cfg.num_freq + (X_DIM - 2)  # 4*num_freq Fourier feats + 22 non-coord feats
model_config = dict(
    space_dim=2,  # unchanged; only used as input-dim split for preprocess MLP
    fun_dim=ENCODED_X_DIM - 2,  # so fun_dim + space_dim == ENCODED_X_DIM
    out_dim=3,
    n_hidden=cfg.n_hidden,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=cfg.mlp_ratio,
    use_qk_norm=cfg.use_qk_norm,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
ffn_inner = next(m.inner for m in model.modules() if isinstance(m, SwiGLUFFN))
print(f"Model: Transolver ({n_params/1e6:.2f}M params, FFN=SwiGLU inner={ffn_inner})")

if cfg.use_lion:
    optimizer = Lion(model.parameters(), lr=cfg.lion_lr,
                     betas=(0.9, 0.99), weight_decay=cfg.lion_wd)
    print(f"Optimizer: Lion (lr={cfg.lion_lr}, wd={cfg.lion_wd}, betas=(0.9, 0.99))")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

def train_amp_ctx():
    """bf16 autocast for the forward pass when --use_bf16 (no GradScaler needed)."""
    if cfg.use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)

warmup_epochs = 2
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * (epoch + 1) / warmup_epochs  # 0.1 -> 1.0 over warmup
    progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))  # cosine to 0 after warmup
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

GRAD_CLIP_MAX_NORM = 1.0

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
        "ffn_type": "swiglu",
        "ffn_inner": ffn_inner,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
        "warmup_epochs": warmup_epochs,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "scheduler": "linear_warmup_then_cosine",
        "optimizer": "lion" if cfg.use_lion else "adamw",
        "lion_betas": (0.9, 0.99) if cfg.use_lion else None,
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

    for batch_idx, (x, y, is_surface, mask) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        if cfg.coord_noise_std > 0:
            pad_mask = mask.unsqueeze(-1).to(x_norm.dtype)  # (B, N, 1); zero at padding
            noise = torch.randn_like(x_norm[..., :2]) * cfg.coord_noise_std * pad_mask
            x_norm = x_norm.clone()
            x_norm[..., :2] = x_norm[..., :2] + noise
        x_enc = encode_inputs(x_norm, cfg.num_freq)
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        # Capture Q/K norm diagnostics on the first batch of each epoch.
        # Cheap: one extra .item() sync per attention layer, only on this batch.
        log_diag_this_batch = (batch_idx == 0)
        if log_diag_this_batch:
            for blk in model.blocks:
                blk.attn._log_diag = True

        with train_amp_ctx():
            pred = model({"x": x_enc})["preds"]
        # Cast model output back to fp32 so loss/metric arithmetic stays in fp32
        pred = pred.float()
        err = _pointwise_loss(pred, y_norm, cfg.loss_type)

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.num_freq, cfg.loss_type, cfg.use_bf16)
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
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.

    # Per-layer Q/K norm diagnostics captured on this epoch's first batch.
    for layer_idx, blk in enumerate(model.blocks):
        for diag_key, diag_val in blk.attn._diag.items():
            log_metrics[f"qk_diag/layer_{layer_idx}/{diag_key}"] = diag_val

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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.num_freq, cfg.loss_type, cfg.use_bf16)
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

        # --- TTA evaluation: K-pass coord-noise averaging on the best checkpoint ---
        # Same weights as the deterministic K=1 eval above; only the eval-time
        # procedure changes. Logged under *_tta/* / *_tta_avg/* so it does not
        # collide with the baseline test_/test_avg/ metrics used for model
        # selection and prior comparisons.
        if cfg.tta_k > 1:
            print(f"\nTTA evaluation: K={cfg.tta_k} passes, "
                  f"noise_std={cfg.tta_noise_std}, seed={cfg.tta_seed}...")
            tta_gen = torch.Generator(device=device).manual_seed(cfg.tta_seed)
            val_tta_metrics = {
                name: evaluate_split(
                    model, loader, stats, cfg.surf_weight, device, cfg.num_freq,
                    cfg.loss_type, cfg.use_bf16,
                    tta_k=cfg.tta_k, tta_noise_std=cfg.tta_noise_std, tta_generator=tta_gen,
                )
                for name, loader in val_loaders.items()
            }
            val_tta_avg = aggregate_splits(val_tta_metrics)
            print(f"\n  VAL  (TTA K={cfg.tta_k}) avg_surf_p={val_tta_avg['avg/mae_surf_p']:.4f}  "
                  f"(K=1 LIVE best {best_avg_surf_p:.4f}, "
                  f"Δ={val_tta_avg['avg/mae_surf_p'] - best_avg_surf_p:+.4f})")
            for name in VAL_SPLIT_NAMES:
                print_split_metrics(name, val_tta_metrics[name])

            test_tta_metrics = {
                name: evaluate_split(
                    model, loader, stats, cfg.surf_weight, device, cfg.num_freq,
                    cfg.loss_type, cfg.use_bf16,
                    tta_k=cfg.tta_k, tta_noise_std=cfg.tta_noise_std, tta_generator=tta_gen,
                )
                for name, loader in test_loaders.items()
            }
            test_tta_avg = aggregate_splits(test_tta_metrics)
            test_k1 = test_avg["avg/mae_surf_p"]
            test_k = test_tta_avg["avg/mae_surf_p"]
            print(f"\n  TEST (TTA K={cfg.tta_k}) avg_surf_p={test_k:.4f}  "
                  f"(K=1 baseline {test_k1:.4f}, Δ={test_k - test_k1:+.4f})")
            for name in TEST_SPLIT_NAMES:
                print_split_metrics(name, test_tta_metrics[name])

            tta_log: dict[str, float] = {}
            for split_name, m in val_tta_metrics.items():
                for k, v in m.items():
                    tta_log[f"val_tta/{split_name}/{k}"] = v
            for k, v in val_tta_avg.items():
                tta_log[f"val_tta_{k}"] = v
            for split_name, m in test_tta_metrics.items():
                for k, v in m.items():
                    tta_log[f"test_tta/{split_name}/{k}"] = v
            for k, v in test_tta_avg.items():
                tta_log[f"test_tta_{k}"] = v
            tta_log["tta/k"] = cfg.tta_k
            tta_log["tta/noise_std"] = cfg.tta_noise_std
            tta_log["tta/delta_test_avg_mae_surf_p"] = test_k - test_k1
            wandb.log(tta_log)
            wandb.summary.update(tta_log)

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
