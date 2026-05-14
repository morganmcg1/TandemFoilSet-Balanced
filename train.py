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


class Lion(torch.optim.Optimizer):
    """Lion: Symbolic Discovery of Optimization Algorithms (Chen et al., 2023)."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Decoupled weight decay (AdamW-style)
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # Lion update: sign(beta1 * m + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.data.add_(update, alpha=-lr)
                # Momentum update for next step
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


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

    def forward(self, x, mask=None):
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
        if mask is not None:
            # Zero out padded positions so they contribute nothing to slice_token / slice_norm.
            # We mask AFTER softmax: masking before with -inf produces all-(-inf) rows for
            # padded nodes (softmax is over slice_num, not N), which softmaxes to NaN.
            slice_weights = slice_weights * mask[:, None, :, None].to(slice_weights.dtype)
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
                 film_re=False, film_re_hidden: int = 128):
        super().__init__()
        self.last_layer = last_layer
        self.film_re = film_re
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
        if self.film_re:
            # γ-only FiLM-Re: scale-only modulation from log(Re).
            # Identity init (last-linear weight=0, bias=1) sets γ≡1 at epoch 0
            # so the model matches baseline exactly until γ drifts.
            # γ MLP hidden width = film_re_hidden (PR #2948 capacity scan).
            self.film_gamma = nn.Sequential(
                nn.Linear(1, film_re_hidden),
                nn.GELU(),
                nn.Linear(film_re_hidden, hidden_dim),
            )

    def forward(self, fx, mask=None, log_re=None):
        fx = self.attn(self.ln_1(fx), mask=mask) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.film_re and log_re is not None:
            # Post-residual γ modulation, after both attn and mlp residuals.
            # γ shape [B, 1, hidden_dim] broadcasts over node axis.
            gamma = self.film_gamma(log_re.unsqueeze(-1)).unsqueeze(1)
            fx = gamma * fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 init_std: float = 0.02,
                 film_re=False, film_re_hidden: int = 128,
                 log_re_x_index: int = 13):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.init_std = init_std
        self.film_re = film_re
        self.film_re_hidden = film_re_hidden
        self.log_re_x_index = log_re_x_index

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
                film_re=film_re, film_re_hidden=film_re_hidden,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.n_linear_init = 0
        self.apply(self._init_weights)
        print(f"Transolver init: {self.n_linear_init} Linear modules re-init'd with trunc_normal_ std={self.init_std}")
        if self.film_re:
            # Identity-init the FiLM γ output linear AFTER apply() so it isn't
            # overwritten by trunc_normal_/constant_(bias, 0) in _init_weights.
            # γ ≡ 1 at epoch 0 → model exactly matches baseline initially.
            for block in self.blocks:
                nn.init.zeros_(block.film_gamma[-1].weight)
                nn.init.ones_(block.film_gamma[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            self.n_linear_init += 1
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        mask = data.get("mask")  # [B, N] bool or None
        log_re = None
        if self.film_re:
            # x is normalized; log_re is the standardized log(Re) at every node.
            # All nodes in a sample share the same value, so node 0 is safe.
            # The film_gamma MLP can absorb the affine normalization.
            log_re = x[:, 0, self.log_re_x_index]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, mask=mask, log_re=log_re)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

# Y-flip TTA constants (PR #3007): 8-direction dsdf channels permute under y-flip
# as [0,7,6,5,4,3,2,1] (verified via brute-force search over 694 mirror pairs in
# cruise tandem samples; mean L1 diff = 0.18 vs 1.96 for identity). Indices 0 and
# 4 are the two horizontal directions (invariant); the remaining six pair up.
_DSDF_YFLIP_PERM = torch.tensor([0, 7, 6, 5, 4, 3, 2, 1], dtype=torch.long)


def yflip_x(x: torch.Tensor) -> torch.Tensor:
    """Apply y-flip transforms to raw (un-normalized) x [..., 24].

    Negates z-direction features and permutes dsdf channels per the verified
    mirror permutation. Used for y-flip Test-Time Augmentation (PR #3007).

    Negated:
      - x[..., 1]   : node position z
      - x[..., 3]   : saf[1] (signed arc-length z-component)
      - x[..., 14]  : AoA foil 1 (radians)
      - x[..., 18]  : AoA foil 2 (radians)
      - x[..., 22]  : gap (signed z-offset of foil 2 vs foil 1)

    Permuted:
      - x[..., 4:12] : dsdf 8-direction descriptor (perm [0,7,6,5,4,3,2,1])

    Invariant:
      x[..., 0] (x position), x[..., 2] (saf x), x[..., 12] (is_surface),
      x[..., 13] (log Re), x[..., 15:18] (NACA1 M,P,T), x[..., 19:22] (NACA2),
      x[..., 23] (stagger).
    """
    out = x.clone()
    out[..., 1] = -out[..., 1]
    out[..., 3] = -out[..., 3]
    perm = _DSDF_YFLIP_PERM.to(out.device)
    out[..., 4:12] = out.index_select(-1, perm + 4)
    out[..., 14] = -out[..., 14]
    out[..., 18] = -out[..., 18]
    out[..., 22] = -out[..., 22]
    return out


def reflect_pred_yflip(pred: torch.Tensor) -> torch.Tensor:
    """Reflect predicted (Ux, Uy, p) from y-flipped frame back to physical frame.

    Under y-flip: Ux invariant (channel 0), Uy negates (channel 1), p invariant
    (channel 2). Assumes pred is in either normalized or physical space — the
    sign-flip is valid in both (Uy mean is near-zero physically; negation in
    normalized space approximates the physical reflection well enough for the
    averaging step in val/test eval). This function is purely a sign flip on
    channel 1.
    """
    out = pred.clone()
    out[..., 1] = -out[..., 1]
    return out


def evaluate_split(
    model, loader, stats, surf_weight, device, *, use_tta: bool = False,
    sym_z_threshold: float = -0.5,
) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    When ``use_tta=True`` (PR #3007), each batch is forwarded twice — once on
    the original input and once on the y-flipped input — and predictions are
    averaged in physical space (Uy negated on the flipped pred). Two TTA
    variants are logged in addition to the vanilla metrics:

    - ``*_tta``: **uniform** TTA — averaged on every sample regardless of
      mesh symmetry. Matches the PR-spec hypothesis directly.
    - ``*_tta_sym``: **gated** TTA — averaged only on samples whose mesh
      spans z < ``sym_z_threshold`` (freestream/cruise tandem). For
      asymmetric (ground-bound) samples, fall back to the vanilla
      prediction. Tests the smart-subset variant that the PR decision tree
      explicitly contemplates ("try TTA on a subset of splits"). Z-symmetry
      is inferred per-sample from the mesh z range, not from a split label,
      so it works correctly on the mixed-domain re_rand split as well.

    The vanilla (non-TTA) keys remain unchanged so checkpoint selection and
    the existing metric pipeline are not perturbed.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    mae_surf_tta = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol_tta = torch.zeros(3, dtype=torch.float64, device=device)
    mae_surf_tta_sym = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol_tta_sym = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0
    n_surf_tta = n_vol_tta = 0
    n_surf_tta_sym = n_vol_tta_sym = 0
    n_sym = n_total = 0  # diagnostic: how many samples got the TTA average in sym mode

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Exclude samples with non-finite ground truth (e.g. test cruise
            # sample 20 has inf in pressure). scoring.py tries to skip such
            # samples but `0 * inf = NaN` propagates through the mask multiply
            # and breaks mae_surf_p; mirror the same defensive cleanup here for
            # the loss accumulators.
            y_finite_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not y_finite_sample.all():
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                mask = mask & y_finite_sample[:, None]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]
                pred = model({"x": x_norm, "mask": mask})["preds"]
            pred = pred.float()  # back to fp32 for downstream metric accumulation

            huber_err = F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction="none")
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (huber_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (huber_err * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

            if use_tta:
                # Y-flip TTA: forward on the y-mirrored batch, reflect Uy in
                # physical space, average with the original prediction.
                x_flip = yflip_x(x)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_flip_norm = (x_flip - stats["x_mean"]) / stats["x_std"]
                    pred_flip = model({"x": x_flip_norm, "mask": mask})["preds"]
                pred_flip = pred_flip.float()
                pred_flip_phys = pred_flip * stats["y_std"] + stats["y_mean"]
                # Reflect (Ux invariant, Uy negates, p invariant) in physical space.
                pred_flip_phys = reflect_pred_yflip(pred_flip_phys)
                pred_tta_phys = (pred_orig + pred_flip_phys) / 2.0
                ds_t, dv_t = accumulate_batch(
                    pred_tta_phys, y, is_surface, mask, mae_surf_tta, mae_vol_tta,
                )
                n_surf_tta += ds_t
                n_vol_tta += dv_t

                # Sym-only TTA: per-sample, gate on mesh z_min < threshold.
                # Ground-bound meshes (raceCar) have z_min ~ 0 and the y-flip
                # puts the foil below the ground — OOD for the trained model.
                # Freestream meshes (cruise tandem) have z spanning both signs
                # and are valid y-flip targets.
                z_masked = torch.where(
                    mask, x[..., 1],
                    torch.full_like(x[..., 1], float("inf")),
                )
                z_min_per_sample = z_masked.amin(dim=-1)  # [B]
                is_sym = (z_min_per_sample < sym_z_threshold).to(pred_orig.dtype)
                # [B, 1, 1] gating: 1 → use TTA-averaged, 0 → use vanilla
                gate = is_sym[:, None, None]
                pred_tta_sym_phys = gate * pred_tta_phys + (1.0 - gate) * pred_orig
                ds_s, dv_s = accumulate_batch(
                    pred_tta_sym_phys, y, is_surface, mask,
                    mae_surf_tta_sym, mae_vol_tta_sym,
                )
                n_surf_tta_sym += ds_s
                n_vol_tta_sym += dv_s
                n_sym += int(is_sym.sum().item())
                n_total += int(mask.any(dim=-1).sum().item())

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    if use_tta:
        tta_metrics = finalize_split(mae_surf_tta, mae_vol_tta, n_surf_tta, n_vol_tta)
        for k, v in tta_metrics.items():
            out[f"{k}_tta"] = v
        tta_sym_metrics = finalize_split(
            mae_surf_tta_sym, mae_vol_tta_sym, n_surf_tta_sym, n_vol_tta_sym,
        )
        for k, v in tta_sym_metrics.items():
            out[f"{k}_tta_sym"] = v
        out["tta_n_sym"] = float(n_sym)
        out["tta_n_total"] = float(n_total)
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
    weight_decay: float = 2e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    init_std: float = 0.07  # trunc_normal_ std for Linear weight init (σ=0.07 merged PR #2882)
    film_re_hidden: int = 128  # γ MLP hidden width for FiLM-Re (default 128 = current; PR #2948 capacity scan)
    tta_yflip: bool = False  # y-flip Test-Time Augmentation at val/test (PR #3007)


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
    init_std=cfg.init_std,
    film_re=True,  # γ-only FiLM-Re conditioning (PR #2865)
    film_re_hidden=cfg.film_re_hidden,  # γ MLP hidden width (PR #2948 capacity scan)
)

model = Transolver(**model_config).to(device)
model = torch.compile(model, dynamic=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")
print(f"[init] FiLM-Re γ MLP hidden dim = {cfg.film_re_hidden} (default = 128)")

optimizer = Lion(
    model.parameters(),
    lr=cfg.lr * 0.15,       # Bisect upward: 5e-4 * 0.15 = 7.5e-5 (50% above 5e-5)
    weight_decay=cfg.weight_decay * 10.0,  # Lion-recommended: ×10 of AdamW wd
    betas=(0.9, 0.99),       # Lion-paper default
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
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
wandb.define_metric("val_avg/*", step_metric="global_step")
wandb.define_metric("val_avg_tta/*", step_metric="global_step")
wandb.define_metric("val_avg_tta_sym/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

with torch.no_grad():
    param_l2_init = torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())).item()
print(f"Param L2 norm at init: {param_l2_init:.4f}")
wandb.summary["param_l2_init"] = param_l2_init

# Y-symmetry diagnostic over the training set (PR #3007). Verifies the
# y-flip TTA assumptions: mesh z-range, saf/dsdf semantics, AoA/gap ranges.
# Single-foil samples are ground-bound (z >= 0); cruise tandem is symmetric.
# Reported once so the advisor can review against the per-split TTA deltas.
if cfg.tta_yflip:
    print("\n[TTA y-flip] Sampling training-set y-symmetry diagnostics ...")
    with torch.no_grad():
        diag_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        n_seen = 0
        z_min_pos, z_max_pos = float("inf"), float("-inf")
        aoa1_min, aoa1_max = float("inf"), float("-inf")
        aoa2_min, aoa2_max = float("inf"), float("-inf")
        gap_min, gap_max = float("inf"), float("-inf")
        n_ground_bound = 0
        n_symmetric = 0
        for diag_x, _, _, diag_mask in diag_loader:
            for b in range(diag_x.shape[0]):
                m = diag_mask[b]
                z_b = diag_x[b, m, 1]
                aoa1_b = diag_x[b, 0, 14].item()
                aoa2_b = diag_x[b, 0, 18].item()
                gap_b = diag_x[b, 0, 22].item()
                zmin = z_b.min().item()
                zmax = z_b.max().item()
                z_min_pos = min(z_min_pos, zmin)
                z_max_pos = max(z_max_pos, zmax)
                aoa1_min = min(aoa1_min, aoa1_b); aoa1_max = max(aoa1_max, aoa1_b)
                aoa2_min = min(aoa2_min, aoa2_b); aoa2_max = max(aoa2_max, aoa2_b)
                gap_min = min(gap_min, gap_b); gap_max = max(gap_max, gap_b)
                if zmin >= -0.1:
                    n_ground_bound += 1
                elif zmax > 0 and zmin < 0:
                    n_symmetric += 1
                n_seen += 1
            if n_seen >= 200:
                break
        print(f"  Training samples scanned: {n_seen}")
        print(f"  Mesh z range (raw): [{z_min_pos:.3f}, {z_max_pos:.3f}]")
        print(f"  AoA1 range (rad):   [{aoa1_min:.4f}, {aoa1_max:.4f}]  "
              f"(deg [{aoa1_min*180/3.14159265:.2f}°, {aoa1_max*180/3.14159265:.2f}°])")
        print(f"  AoA2 range (rad):   [{aoa2_min:.4f}, {aoa2_max:.4f}]  "
              f"(deg [{aoa2_min*180/3.14159265:.2f}°, {aoa2_max*180/3.14159265:.2f}°])")
        print(f"  gap range:          [{gap_min:.4f}, {gap_max:.4f}]")
        print(f"  Ground-bound (z_min >= -0.1):  {n_ground_bound}/{n_seen}")
        print(f"  Y-symmetric (z spans both signs): {n_symmetric}/{n_seen}")
        wandb.summary.update({
            "tta_diag/z_min": z_min_pos,
            "tta_diag/z_max": z_max_pos,
            "tta_diag/aoa1_min_rad": aoa1_min,
            "tta_diag/aoa1_max_rad": aoa1_max,
            "tta_diag/aoa2_min_rad": aoa2_min,
            "tta_diag/aoa2_max_rad": aoa2_max,
            "tta_diag/gap_min": gap_min,
            "tta_diag/gap_max": gap_max,
            "tta_diag/n_ground_bound": n_ground_bound,
            "tta_diag/n_symmetric": n_symmetric,
            "tta_diag/n_scanned": n_seen,
        })

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

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm, "mask": mask})["preds"]
            # Huber β=0.5 for Ux/Uy (channels 0,1); pinball τ=0.55 for pressure (channel 2).
            # Pinball: ρ_τ(r) = max(τ*r, (τ-1)*r) with r = y - ŷ.
            # τ=0.55 biases the model toward over-predicting pressure when residuals are
            # small (CFD pressure surrogates known to under-predict suction peaks).
            huber_err = F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction="none")
            residual_p = y_norm[..., 2] - pred[..., 2]
            pinball_p = torch.where(residual_p >= 0, 0.55 * residual_p, -0.45 * residual_p)
            err = torch.stack([huber_err[..., 0], huber_err[..., 1], pinball_p], dim=-1)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

            # Diagnostic: signed residual mean for pressure on surface and volume
            # nodes. If pinball τ=0.55 is shifting the bias as hypothesized, surf
            # signed residual should drift toward small negative (over-prediction).
            # Critical check given Lion's sign() update can wash out pinball's
            # magnitude asymmetry — bias shift confirms the mechanism is alive.
            with torch.no_grad():
                vol_mask_f = vol_mask.float()
                surf_mask_f = surf_mask.float()
                p_signed_vol = (
                    (residual_p * vol_mask_f).sum() / vol_mask_f.sum().clamp(min=1)
                )
                p_signed_surf = (
                    (residual_p * surf_mask_f).sum() / surf_mask_f.sum().clamp(min=1)
                )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/p_signed_residual_vol": p_signed_vol.item(),
            "train/p_signed_residual_surf": p_signed_surf.item(),
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
        name: evaluate_split(
            model, loader, stats, cfg.surf_weight, device,
            use_tta=cfg.tta_yflip,
        )
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    # When TTA is on, also aggregate the *_tta and *_tta_sym channels.
    val_avg_tta: dict[str, float] = {}
    val_avg_tta_sym: dict[str, float] = {}
    if cfg.tta_yflip:
        chans = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in ("Ux", "Uy", "p")]
        for k in chans:
            v_tta = [m[f"{k}_tta"] for m in split_metrics.values() if f"{k}_tta" in m]
            if v_tta:
                val_avg_tta[f"avg_tta/{k}"] = sum(v_tta) / len(v_tta)
            v_sym = [m[f"{k}_tta_sym"] for m in split_metrics.values() if f"{k}_tta_sym" in m]
            if v_sym:
                val_avg_tta_sym[f"avg_tta_sym/{k}"] = sum(v_sym) / len(v_sym)
    dt = time.time() - t0

    with torch.no_grad():
        param_l2_epoch = torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())).item()

    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/param_l2": param_l2_epoch,
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
    for k, v in val_avg_tta.items():
        log_metrics[f"val_{k}"] = v  # val_avg_tta/mae_surf_p etc.
    for k, v in val_avg_tta_sym.items():
        log_metrics[f"val_{k}"] = v  # val_avg_tta_sym/mae_surf_p etc.
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
    tta_str = ""
    if cfg.tta_yflip:
        tta_str = (
            f"  tta={val_avg_tta.get('avg_tta/mae_surf_p', float('nan')):.4f}"
            f"  tta_sym={val_avg_tta_sym.get('avg_tta_sym/mae_surf_p', float('nan')):.4f}"
        )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}{tta_str}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

with torch.no_grad():
    param_l2_final = torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())).item()
print(f"Param L2 norm at final epoch: {param_l2_final:.4f}")
wandb.summary["param_l2_final_epoch"] = param_l2_final

# --- γ-only FiLM-Re diagnostics: per-block γ bias and weight L2 ---
# Goal: compare to prior PR #2816 (γ+β FiLM-Re). Does γ drift more or less
# without β present? Log final-epoch bias/weight stats from each block's
# film_gamma[-1] (the last linear that directly produces γ).
if model_config.get("film_re"):
    # Resolve underlying module past torch.compile's _orig_mod wrapper.
    base_model = getattr(model, "_orig_mod", model)
    gamma_diagnostics: dict[str, float] = {}
    print("\nγ-only FiLM-Re per-block diagnostics (final epoch):")
    print(f"  {'block':<6s} {'γ_bias_mean':>14s} {'γ_bias_std':>12s} "
          f"{'γ_w_l2':>10s} {'γ_w_absmax':>12s}")
    for i, block in enumerate(base_model.blocks):
        last_lin = block.film_gamma[-1]
        gb = last_lin.bias.detach().float()
        gw = last_lin.weight.detach().float()
        b_mean, b_std = gb.mean().item(), gb.std().item()
        w_l2 = gw.norm().item()
        w_absmax = gw.abs().max().item()
        gamma_diagnostics[f"film/block{i}/gamma_bias_mean"] = b_mean
        gamma_diagnostics[f"film/block{i}/gamma_bias_std"] = b_std
        gamma_diagnostics[f"film/block{i}/gamma_w_l2"] = w_l2
        gamma_diagnostics[f"film/block{i}/gamma_w_absmax"] = w_absmax
        print(f"  {i:<6d} {b_mean:>14.6f} {b_std:>12.6f} "
              f"{w_l2:>10.4f} {w_absmax:>12.4f}")
    wandb.summary.update(gamma_diagnostics)

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
    with torch.no_grad():
        param_l2_best = torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())).item()
    print(f"Param L2 norm at best checkpoint: {param_l2_best:.4f}")
    wandb.summary["param_l2_best_ckpt"] = param_l2_best

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
            name: evaluate_split(
                model, loader, stats, cfg.surf_weight, device,
                use_tta=cfg.tta_yflip,
            )
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        # Also aggregate TTA channels (uniform + sym-only) if enabled.
        test_avg_tta: dict[str, float] = {}
        test_avg_tta_sym: dict[str, float] = {}
        if cfg.tta_yflip:
            chans = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in ("Ux", "Uy", "p")]
            for k in chans:
                v_tta = [m[f"{k}_tta"] for m in test_metrics.values() if f"{k}_tta" in m]
                if v_tta:
                    test_avg_tta[f"avg_tta/{k}"] = sum(v_tta) / len(v_tta)
                v_sym = [m[f"{k}_tta_sym"] for m in test_metrics.values() if f"{k}_tta_sym" in m]
                if v_sym:
                    test_avg_tta_sym[f"avg_tta_sym/{k}"] = sum(v_sym) / len(v_sym)
        tta_test_str = ""
        if cfg.tta_yflip:
            tta_test_str = (
                f"  tta={test_avg_tta.get('avg_tta/mae_surf_p', float('nan')):.4f}"
                f"  tta_sym={test_avg_tta_sym.get('avg_tta_sym/mae_surf_p', float('nan')):.4f}"
            )
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}{tta_test_str}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])

        test_log: dict[str, float] = {}
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                test_log[f"test/{split_name}/{k}"] = v
        for k, v in test_avg.items():
            test_log[f"test_{k}"] = v
        for k, v in test_avg_tta.items():
            test_log[f"test_{k}"] = v  # test_avg_tta/mae_surf_p etc.
        for k, v in test_avg_tta_sym.items():
            test_log[f"test_{k}"] = v  # test_avg_tta_sym/mae_surf_p etc.
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
