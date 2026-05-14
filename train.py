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


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation channel gate (Hu et al. 2018 CVPR) — masked global
    average pool over tokens, bottleneck MLP, sigmoid gate, broadcast multiply.

    Mask is threaded through forward to exclude padded mesh nodes from the
    global pool (TandemFoilSet meshes range 74K-242K nodes; padding fraction up
    to ~70% within a batch makes unmasked pool biased on small-mesh samples).
    Zero-init fc2 so gate = sigmoid(0) = 0.5 uniformly at step 0.
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        d_hidden = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, dim, bias=True)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x, mask=None):
        # x: (B, N, D); mask: (B, N) bool, True for real nodes
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(x.dtype)
            s = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            s = x.mean(dim=1)
        s = self.fc2(F.gelu(self.fc1(s)))
        gate = torch.sigmoid(s)
        return x * gate.unsqueeze(1)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 layerscale_init=1e-4):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.gamma_attn = nn.Parameter(layerscale_init * torch.ones(hidden_dim))
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        self.gamma_mlp = nn.Parameter(layerscale_init * torch.ones(hidden_dim))
        self.se = SqueezeExcitation(hidden_dim, reduction=8)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, mask=None):
        fx = self.gamma_attn * self.attn(self.ln_1(fx)) + fx
        fx = self.gamma_mlp * self.mlp(self.ln_2(fx)) + fx
        fx = self.se(fx, mask=mask)
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
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        # Feature-stream FiLM gate — zero-init both weight AND bias for identity at step 0.
        # Applied AFTER _init_weights so it overrides the trunc_normal_ init.
        # Channels [13, 14, 18] = [log_Re, AoA0_rad, AoA1_rad] (per #2531 instrumentation).
        self.film = nn.Linear(3, n_hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

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
        mask = data.get("mask")  # [B, N] bool, True for real nodes; threaded to SE pool
        # Per-sample flow scalars (constant across N) — channels [13, 14, 18]
        flow_scalars = x[:, 0, [13, 14, 18]]                       # [B, 3]
        film_scale = self.film(flow_scalars).unsqueeze(1)          # [B, 1, n_hidden]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        # Feature-stream FiLM: zero-init -> identity at step 0
        fx = fx * (1 + film_scale)
        for block in self.blocks:
            fx = block(fx, mask=mask)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, amp_ctx_factory) -> dict[str, float]:
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
            with amp_ctx_factory():
                pred = model({"x": x_norm, "mask": mask})["preds"]
            pred = pred.float()

            sq_err = F.l1_loss(pred, y_norm, reduction='none')
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

            # --- scoring-bug workaround: filter non-finite ground-truth samples
            # *before* accumulate_batch, so 0 * Inf = NaN cannot poison the sums.
            y_finite_mask = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if y_finite_mask.any():
                ds, dv = accumulate_batch(
                    pred_orig[y_finite_mask],
                    y[y_finite_mask],
                    is_surface[y_finite_mask],
                    mask[y_finite_mask],
                    mae_surf,
                    mae_vol,
                )
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


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

SINGLE_DOMAIN_BOOST = 2.0

if not cfg.debug:
    with open(Path(cfg.splits_dir) / "meta.json") as f:
        _meta_for_sampler = json.load(f)
    _domain_groups = _meta_for_sampler["domain_groups"]
    print(f"[sampler-reweight] domain keys: {list(_domain_groups.keys())}")
    print(f"[sampler-reweight] sizes: " + str({k: len(v) for k, v in _domain_groups.items()}))

    single_domain_key = None
    for k in _domain_groups.keys():
        kl = k.lower()
        if "single" in kl and "tandem" not in kl:
            single_domain_key = k
            break
    assert single_domain_key is not None, (
        f"Could not find 'RaceCar single' domain among {list(_domain_groups.keys())}"
    )
    print(f"[sampler-reweight] boosting domain '{single_domain_key}' by {SINGLE_DOMAIN_BOOST}x")

    boosted = sample_weights.clone()
    for idx in _domain_groups[single_domain_key]:
        boosted[idx] = boosted[idx] * SINGLE_DOMAIN_BOOST

    print(f"[sampler-reweight] pre-boost domain mass: "
          + ", ".join(f"{k}={sample_weights[v].sum().item():.4f}" for k, v in _domain_groups.items()))
    print(f"[sampler-reweight] post-boost domain mass: "
          + ", ".join(f"{k}={boosted[v].sum().item():.4f}" for k, v in _domain_groups.items()))

    sample_weights = boosted

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
    n_hidden=96,
    n_layers=4,
    n_head=2,
    slice_num=24,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
print(f"slice_num: {model_config['slice_num']}")
print(f"slice_num: 24 (down from 32, -25% slicing ops/block) — budget-freeing PhysicsAttention granularity probe; 3rd orthogonal budget-bound axis after n_layers (#2268) and n_hidden (#2290)")
print(f"n_head: {model_config['n_head']} (dim_head={model_config['n_hidden'] // model_config['n_head']})")
print(f"Depth: n_layers=4 (TransolverBlock x 4) — depth-down probe, budget-bound vs capacity-saturated diagnostic")
print(f"Width: n_hidden=96 (hidden_dim=96, down from 128) — budget-freeing width-down probe; ~40-45% per-epoch wall-clock savings")

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")
print(f"LayerScale: per-channel learnable gain init=1e-4 on both attn and mlp residual branches in all {model_config['n_layers']} TransolverBlocks")
print(
    f"Feature-stream FiLM gate: zero-init Linear(3, {model_config['n_hidden']}) applied before block 0; "
    f"channels [13, 14, 18] = [log_Re, AoA0_rad, AoA1_rad] (per #2531); "
    f"residual-stream conditioning vs output-side (closed #2531+#2588); "
    f"+~{3 * model_config['n_hidden'] + model_config['n_hidden']} params; "
    f"baseline to beat: val_avg/mae_surf_p < 33.4935"
)
print(f"Actual total params: {n_params}")

# SE diagnostic: count modules and added params
_se_modules = [m for m in model.modules() if isinstance(m, SqueezeExcitation)]
_n_se = len(_se_modules)
_n_se_params = sum(p.numel() for m in _se_modules for p in m.parameters())
print(
    f"Squeeze-Excitation (Hu et al. 2018): added {_n_se} SE modules, +{_n_se_params} params (reduction=8); "
    f"masked global avg pool over tokens -> fc({model_config['n_hidden']}->{model_config['n_hidden']//8}) -> GELU -> "
    f"fc({model_config['n_hidden']//8}->{model_config['n_hidden']}) -> sigmoid -> broadcast multiply; "
    f"applied at END of each TransolverBlock; zero-init fc2 -> gate=0.5 uniform at step 0; "
    f"baseline to beat: val_avg/mae_surf_p < 33.3722"
)

# torch.compile with dynamic=True because pad_collate yields batches with
# variable N_max (longest mesh in batch varies). Without dynamic, compile
# would retrace on every new shape.
compile_active = False
compile_error: str | None = None
try:
    model = torch.compile(model, dynamic=True)
    compile_active = True
    print("torch.compile: enabled (dynamic=True)")
except Exception as e:
    compile_error = repr(e)
    print(f"torch.compile: skipped ({compile_error})")


def amp_ctx_factory():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


print(f"AMP: {'bfloat16' if torch.cuda.is_available() else 'disabled (no CUDA)'}")


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023): sign-based momentum updates.

    Update: theta <- theta - lr * (sign(beta1 * m + (1-beta1) * g) + wd * theta)
    Momentum: m <- beta2 * m + (1-beta2) * g
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss


class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper (Zhang et al. 2019, NeurIPS).

    Maintains slow weights theta_slow; every k regular base-optimizer steps:
        theta_slow <- theta_slow + alpha * (theta_fast - theta_slow)
        theta_fast <- theta_slow

    Inherits from ``torch.optim.Optimizer`` so PyTorch's LRScheduler isinstance
    check passes. Does not call ``super().__init__`` — we delegate
    ``param_groups`` / ``defaults`` / ``state`` to the wrapped base optimizer so
    scheduler updates flow through naturally.

    Args:
        base_optimizer: any torch.optim.Optimizer instance (e.g. Lion).
        k: number of fast steps between slow-weight syncs (default 5).
        alpha: interpolation factor for slow weights (default 0.5).
    """
    def __init__(self, base_optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = []
        for group in base_optimizer.param_groups:
            self.slow_weights.append([p.detach().clone() for p in group['params']])
        # Delegate optimizer-protocol attributes to the base optimizer.
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        # Per-epoch peak-drift diagnostic, reset externally each epoch.
        self.peak_drift = 0.0
        self.peak_drift_ratio = 0.0
        self.last_drift = 0.0
        self.last_drift_ratio = 0.0

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        if self.step_count % self.k == 0:
            l2_drift_sq = 0.0
            l2_fast_sq = 0.0
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    slow = self.slow_weights[group_idx][p_idx]
                    l2_drift_sq += (p.data - slow).pow(2).sum().item()
                    l2_fast_sq += p.data.pow(2).sum().item()
            drift = l2_drift_sq ** 0.5
            fast_l2 = l2_fast_sq ** 0.5
            ratio = drift / max(fast_l2, 1e-12)
            self.last_drift = drift
            self.last_drift_ratio = ratio
            if drift > self.peak_drift:
                self.peak_drift = drift
                self.peak_drift_ratio = ratio
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    slow = self.slow_weights[group_idx][p_idx]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss

    def reset_drift_diagnostics(self):
        self.peak_drift = 0.0
        self.peak_drift_ratio = 0.0

    def state_dict(self):
        return {
            "base": self.base_optimizer.state_dict(),
            "slow_weights": self.slow_weights,
            "step_count": self.step_count,
            "k": self.k,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base"])
        self.slow_weights = state_dict["slow_weights"]
        self.step_count = state_dict["step_count"]
        self.k = state_dict["k"]
        self.alpha = state_dict["alpha"]


_base_optimizer = Lion(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.99),
)
optimizer = Lookahead(_base_optimizer, k=5, alpha=0.5)
# Capture parameter names in optimizer-iteration order so we can save / load
# slow weights as a named state dict for the post-training fast-vs-slow A/B
# diagnostic. model.parameters() (used to construct Lion) and
# model.named_parameters() iterate in the same order.
_param_names_in_order = [name for name, _ in model.named_parameters()]
print(f"Optimizer: Lookahead(k=5, alpha=0.5) wrapping Lion (Chen et al. 2023) | lr={cfg.lr}, wd={cfg.weight_decay}, betas=(0.9, 0.99)")
print(f"Lookahead (Zhang et al. 2019): every k=5 Lion steps, slow weights theta_slow += alpha*(theta_fast - theta_slow); fast weights reset to slow. Targets Lion narrow-basin + in_dist overfitting bottleneck via smooth slow-weight commitment. Baseline to beat: val_avg/mae_surf_p < 33.0195")
warmup_epochs = 3
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(MAX_EPOCHS - warmup_epochs, 1)
        ),
    ],
    milestones=[warmup_epochs],
)
print(f"Scheduler: LinearLR(0.1->1.0 over {warmup_epochs} epochs) -> CosineAnnealingLR(T_max={max(MAX_EPOCHS - warmup_epochs, 1)})")
print(f"LR check ep0: {optimizer.param_groups[0]['lr']:.6f} (expect {0.1 * cfg.lr:.6f})")

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

    optimizer.reset_drift_diagnostics()
    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with amp_ctx_factory():
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm, "mask": mask})["preds"]
            sq_err = F.l1_loss(pred, y_norm, reduction='none')

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    lr_now = optimizer.param_groups[0]['lr']
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx_factory)
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
        # Save Lookahead slow weights at this best-val moment as a named
        # state dict so we can run a fast-vs-slow A/B at end of training.
        _slow_flat = [s for g in optimizer.slow_weights for s in g]
        _slow_state = {name: s.detach().clone()
                       for name, s in zip(_param_names_in_order, _slow_flat)}
        torch.save(_slow_state, model_dir / "checkpoint_slow.pt")
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    _total_syncs = optimizer.step_count // optimizer.k
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": lr_now,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "compile_active": compile_active,
        "lookahead/syncs_total": _total_syncs,
        "lookahead/peak_drift": optimizer.peak_drift,
        "lookahead/peak_drift_ratio": optimizer.peak_drift_ratio,
        "lookahead/last_drift": optimizer.last_drift,
        "lookahead/last_drift_ratio": optimizer.last_drift_ratio,
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

# --- Lion momentum diagnostic: sanity-check optimizer state at terminal ---
_lion_total = 0
_lion_nz = 0
for _group in optimizer.param_groups:
    for _p in _group['params']:
        if _p in optimizer.state:
            _m = optimizer.state[_p].get('momentum')
            if _m is not None:
                _lion_total += _m.numel()
                _lion_nz += (_m.abs() > 1e-8).sum().item()
_lion_nz_frac = _lion_nz / max(_lion_total, 1)
print(f"Lion momentum non-zero fraction at terminal: {_lion_nz_frac:.4f} ({_lion_nz}/{_lion_total} elements)")
append_metrics_jsonl(metrics_jsonl_path, {
    "event": "lion_momentum_diagnostic",
    "nonzero_fraction": _lion_nz_frac,
    "nonzero_count": _lion_nz,
    "total_count": _lion_total,
})

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # LayerScale diagnostic: report final per-block gamma means (best-checkpoint weights)
    _inner = getattr(model, "_orig_mod", model)
    _gamma_log = {"event": "layerscale_gammas", "epoch": int(best_metrics["epoch"]), "blocks": []}
    print("\nLayerScale final gammas (best-checkpoint weights):")
    for _i, _blk in enumerate(_inner.blocks):
        _ga_mean = _blk.gamma_attn.detach().float().mean().item()
        _ga_abs = _blk.gamma_attn.detach().float().abs().mean().item()
        _gm_mean = _blk.gamma_mlp.detach().float().mean().item()
        _gm_abs = _blk.gamma_mlp.detach().float().abs().mean().item()
        print(f"  block[{_i}]: gamma_attn mean={_ga_mean:.6f} abs_mean={_ga_abs:.6f} | "
              f"gamma_mlp mean={_gm_mean:.6f} abs_mean={_gm_abs:.6f}")
        _gamma_log["blocks"].append({
            "block_idx": _i,
            "gamma_attn_mean": _ga_mean,
            "gamma_attn_abs_mean": _ga_abs,
            "gamma_mlp_mean": _gm_mean,
            "gamma_mlp_abs_mean": _gm_abs,
        })
    append_metrics_jsonl(metrics_jsonl_path, _gamma_log)

    # Feature-stream FiLM diagnostic: weight/bias norms and modulation magnitude
    _film = _inner.film
    _film_w_norm = _film.weight.detach().float().norm().item()
    _film_b_norm = _film.bias.detach().float().norm().item()
    print("\nFeature-stream FiLM final diagnostics (best-checkpoint weights):")
    print(f"  film.weight.norm: {_film_w_norm:.4f}")
    print(f"  film.bias.norm:   {_film_b_norm:.4f}")

    # Modulation magnitude on a val_re_rand batch (highest-OOD split for Re).
    _film_scale_mag = None
    _val_re_rand_loader = val_loaders.get("val_re_rand")
    if _val_re_rand_loader is not None:
        with torch.no_grad():
            _x, _y, _is_surface, _mask = next(iter(_val_re_rand_loader))
            _x = _x.to(device, non_blocking=True)
            _x_norm = (_x - stats["x_mean"]) / stats["x_std"]
            _flow_scalars = _x_norm[:, 0, [13, 14, 18]]
            _film_scale = _film(_flow_scalars.float()).unsqueeze(1)
            _film_scale_mag = (1 + _film_scale).abs().mean().item()
        print(f"  film_scale |1+gamma| mean on val_re_rand batch: {_film_scale_mag:.4f}")
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "film_diagnostic",
        "epoch": int(best_metrics["epoch"]),
        "film_weight_norm": _film_w_norm,
        "film_bias_norm": _film_b_norm,
        "film_scale_abs_mean_val_re_rand": _film_scale_mag,
    })

    # Squeeze-Excitation diagnostic: gate stats per block on a sample val batch.
    # Hook each SE module to capture its gate; one forward pass populates them.
    _se_mods = [m for m in _inner.modules() if isinstance(m, SqueezeExcitation)]
    _se_captured: list[tuple[int, torch.Tensor]] = []

    def _make_se_hook(idx: int):
        def _hook(module, args, kwargs, output):
            with torch.no_grad():
                x = args[0] if args else kwargs.get("x")
                m = kwargs.get("mask")
                if m is not None:
                    mf = m.unsqueeze(-1).to(x.dtype)
                    s = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
                else:
                    s = x.mean(dim=1)
                s = module.fc2(F.gelu(module.fc1(s)))
                gate = torch.sigmoid(s).detach().float().cpu()
                _se_captured.append((idx, gate))
        return _hook

    _se_handles = [m.register_forward_hook(_make_se_hook(i), with_kwargs=True)
                   for i, m in enumerate(_se_mods)]

    _se_loader = val_loaders.get("val_single_in_dist") or next(iter(val_loaders.values()))
    with torch.no_grad():
        _x_s, _y_s, _is_s, _m_s = next(iter(_se_loader))
        _x_s = _x_s.to(device, non_blocking=True)
        _m_s = _m_s.to(device, non_blocking=True)
        _x_norm_s = (_x_s - stats["x_mean"]) / stats["x_std"]
        # Route through uncompiled module so Python-level hooks fire
        # (torch.compile bypasses hooks registered after compile).
        with amp_ctx_factory():
            _ = _inner({"x": _x_norm_s, "mask": _m_s})
    for _h in _se_handles:
        _h.remove()

    print("\nSE gate stats per block (sample val batch, best-checkpoint weights):")
    _se_log = {"event": "se_diagnostic", "epoch": int(best_metrics["epoch"]), "blocks": []}
    for _idx, _gate in _se_captured:
        _gm = _gate.mean().item()
        _gs = _gate.std().item()
        _gmin = _gate.min().item()
        _gmax = _gate.max().item()
        print(f"  SE block[{_idx}]: gate mean={_gm:.4f}  std={_gs:.4f}  min={_gmin:.4f}  max={_gmax:.4f}")
        _se_log["blocks"].append({
            "block_idx": _idx,
            "gate_mean": _gm,
            "gate_std": _gs,
            "gate_min": _gmin,
            "gate_max": _gmax,
        })
    append_metrics_jsonl(metrics_jsonl_path, _se_log)

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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx_factory)
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

    # Lookahead diagnostic: re-evaluate val splits using SLOW weights to compare
    # vs fast (= the original training-protocol checkpoint we just evaluated).
    _slow_ckpt_path = model_dir / "checkpoint_slow.pt"
    if _slow_ckpt_path.exists():
        print("\nLookahead diagnostic: re-evaluating val splits with SLOW (Lookahead) weights...")
        _slow_state = torch.load(_slow_ckpt_path, map_location=device, weights_only=True)
        _fast_backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in _slow_state:
                    p.data.copy_(_slow_state[name])
        model.eval()
        slow_split_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx_factory)
            for name, loader in val_loaders.items()
        }
        slow_val_avg = aggregate_splits(slow_split_metrics)["avg/mae_surf_p"]
        print(f"\nFast (best-checkpoint) val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
        print(f"Slow (Lookahead) val_avg/mae_surf_p       = {slow_val_avg:.4f}")
        print(f"Delta (fast - slow): {best_avg_surf_p - slow_val_avg:+.4f}  "
              f"({'slow wins' if slow_val_avg < best_avg_surf_p else 'fast wins'})")
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(name, slow_split_metrics[name])
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "lookahead_slow_diagnostic",
            "fast_val_avg": best_avg_surf_p,
            "slow_val_avg": slow_val_avg,
            "delta_fast_minus_slow": best_avg_surf_p - slow_val_avg,
            "slow_per_split": slow_split_metrics,
        })
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.data.copy_(_fast_backup[name])

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
