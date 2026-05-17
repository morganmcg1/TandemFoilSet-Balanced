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
import random
import statistics
import subprocess
import time
from collections import deque
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
from torch.amp import autocast
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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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


class SwiGLUMlp(nn.Module):
    """SwiGLU FFN: gated linear unit with SiLU/Swish activation.

    Shazeer 2020, "GLU Variants Improve Transformers". Standard FFN in modern
    large transformers (LLaMA, PaLM, Mistral). Uses ``2/3 * n_hidden`` for the
    two parallel projections to keep param count near parity with a vanilla
    ``Linear -> GELU -> Linear`` FFN of the same nominal hidden width.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        swiglu_h = int(round(n_hidden * 2 / 3))
        self.fc_main = nn.Linear(n_input, swiglu_h)
        self.fc_gate = nn.Linear(n_input, swiglu_h)
        self.fc_out = nn.Linear(swiglu_h, n_output)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.fc_out(self.fc_main(x) * self.act(self.fc_gate(x)))


class GeGLUMlp(nn.Module):
    """GeGLU FFN: gated linear unit with GELU activation in the gate.

    Shazeer 2020, "GLU Variants Improve Transformers". Identical structure to
    SwiGLUMlp; only the gate nonlinearity differs (GELU vs SiLU). Same 2/3
    hidden-dim factor keeps param count at parity with SwiGLU. Ablation lever
    for isolating gating mechanism from SiLU-specific behavior.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        geglu_h = int(round(n_hidden * 2 / 3))
        self.fc_main = nn.Linear(n_input, geglu_h)
        self.fc_gate = nn.Linear(n_input, geglu_h)
        self.fc_out = nn.Linear(geglu_h, n_output)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc_out(self.fc_main(x) * self.act(self.fc_gate(x)))


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
                 use_swiglu=False, use_geglu=False):
        super().__init__()
        if use_swiglu and use_geglu:
            raise ValueError("use_swiglu and use_geglu are mutually exclusive")
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if use_swiglu:
            self.mlp = SwiGLUMlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
        elif use_geglu:
            self.mlp = GeGLUMlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
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


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms").

    Reference: https://arxiv.org/abs/2302.06675

    Sign-based update with momentum interpolation; every parameter update has
    magnitude ``lr`` regardless of gradient scale, which helps at small batch
    sizes where AdamW's ``sqrt(v_t)`` denominator is itself noisy. Memory
    footprint is lower than AdamW (no second-moment buffer).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
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
                grad = p.grad
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # .mul()/.add() are OUT-OF-PLACE on purpose; m must remain
                # untouched until the m.mul_().add_() line below.
                update = m.mul(beta1).add(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class Lookahead(torch.optim.Optimizer):
    """Wraps a base optimizer with k-step lookahead (Zhang et al. 2019).

    Reference: "Lookahead Optimizer: k steps forward, 1 step back"
    https://arxiv.org/abs/1907.08610

    Maintains a slow-weight copy and interleaves k fast (base optimizer) steps
    with a slow-weight sync ``θ_slow ← θ_slow + α·(θ_fast - θ_slow)``; the fast
    weights then restart from the new slow weights. The scheduler attaches to
    the base optimizer; param_groups/defaults/state are passthroughs so existing
    code that inspects the wrapper sees the base optimizer's view.

    Side effect: ``self.last_sync_diff_norm`` is set to the L2 norm of
    ``θ_fast - θ_slow`` just before each sync (and ``None`` between syncs) for
    optional logging of how far the fast trajectory wanders from the slow one.

    Optionally, ``alpha`` can be scheduled across training. Pass
    ``alpha_schedule='cosine'`` along with ``alpha_start``, ``alpha_end``, and
    ``total_steps`` to ramp the slow-weight pull strength from ``alpha_start``
    at step 0 to ``alpha_end`` at step ``total_steps`` via a half-cosine. After
    ``total_steps`` the value is held at ``alpha_end``.
    """

    def __init__(self, base_optimizer, k=5, alpha=0.5,
                 alpha_schedule: str | None = None,
                 alpha_start: float | None = None,
                 alpha_end: float | None = None,
                 total_steps: int | None = None):
        self.base = base_optimizer
        self.k = k
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = total_steps
        if alpha_schedule is not None:
            if alpha_schedule != "cosine":
                raise ValueError(f"Unknown alpha_schedule: {alpha_schedule}")
            if alpha_start is None or alpha_end is None or total_steps is None:
                raise ValueError(
                    "alpha_schedule requires alpha_start, alpha_end, total_steps"
                )
        self.current_alpha: float = self._alpha_at_step(0)
        self.step_count = 0
        self.last_sync_diff_norm: float | None = None
        self.slow_state = {}
        for group in base_optimizer.param_groups:
            for p in group["params"]:
                self.slow_state[p] = p.data.clone().detach()
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state

    def _alpha_at_step(self, step: int) -> float:
        if self.alpha_schedule is None:
            return self.alpha
        progress = min(1.0, step / max(1, self.total_steps))
        return (
            self.alpha_start
            + (self.alpha_end - self.alpha_start)
            * 0.5
            * (1.0 - math.cos(math.pi * progress))
        )

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base.step(closure)
        self.step_count += 1
        self.last_sync_diff_norm = None
        self.current_alpha = self._alpha_at_step(self.step_count)
        if self.step_count % self.k == 0:
            sq_sum = None
            for group in self.base.param_groups:
                for p in group["params"]:
                    slow = self.slow_state[p]
                    diff = p.data - slow
                    s = diff.pow(2).sum()
                    sq_sum = s if sq_sum is None else sq_sum + s
                    slow.add_(diff, alpha=self.current_alpha)
                    p.data.copy_(slow)
            if sq_sum is not None:
                self.last_sync_diff_norm = float(sq_sum.sqrt().item())
        return loss


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_swiglu=False, use_geglu=False):
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
                use_swiglu=use_swiglu, use_geglu=use_geglu,
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

            # Defensive: replace non-finite GT values with 0, exclude affected nodes.
            # Prevents inf*0=NaN when scoring.py's accumulate_batch masks them out.
            _y_fin = torch.isfinite(y).all(dim=-1)  # [B, N]
            if not _y_fin.all():
                y = torch.where(_y_fin.unsqueeze(-1), y, torch.zeros_like(y))
                mask = mask & _y_fin

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]

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
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    seed: int = 42
    deterministic: bool = False
    use_swiglu: bool = False
    use_geglu: bool = False
    use_lion: bool = False
    lion_b2: float = 0.99  # Lion beta2; default matches historical baseline
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    lookahead_alpha_schedule: str | None = None  # None or "cosine"
    lookahead_alpha_start: float = 0.5
    lookahead_alpha_end: float = 0.7


cfg = sp.parse(Config)
set_all_seeds(cfg.seed)
if cfg.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  seed={cfg.seed}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

_loader_gen = torch.Generator()
_loader_gen.manual_seed(cfg.seed)
loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2,
                     worker_init_fn=seed_worker, generator=_loader_gen)

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
    use_swiglu=cfg.use_swiglu,
    use_geglu=cfg.use_geglu,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_ffn_params = sum(
    p.numel() for n, p in model.named_parameters()
    if "blocks." in n and ".mlp." in n and ".mlp2." not in n
)
print(f"Model: Transolver ({n_params/1e6:.2f}M params, FFN={n_ffn_params})")

if cfg.use_lion:
    # Lion lr is ~3-10x smaller than AdamW's because sign-based updates have
    # constant magnitude (lr/3 here, matching the PR #4123 recipe).
    base_optimizer = Lion(
        model.parameters(), lr=cfg.lr / 3.0, weight_decay=cfg.weight_decay,
        betas=(0.9, cfg.lion_b2),
    )
else:
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
if cfg.lookahead_k > 0:
    # alpha schedule horizon matches the LR cosine T_max=17 epochs so the slow-weight
    # pull strength reaches alpha_end at the same point the LR reaches 0.
    _alpha_total_steps = (
        len(train_loader) * 17 if cfg.lookahead_alpha_schedule is not None else None
    )
    optimizer = Lookahead(
        base_optimizer,
        k=cfg.lookahead_k,
        alpha=cfg.lookahead_alpha,
        alpha_schedule=cfg.lookahead_alpha_schedule,
        alpha_start=cfg.lookahead_alpha_start,
        alpha_end=cfg.lookahead_alpha_end,
        total_steps=_alpha_total_steps,
    )
else:
    optimizer = base_optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=17)

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
        "n_ffn_params": n_ffn_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)
run.summary["model/param_count"] = n_params
run.summary["model/ffn_param_count"] = n_ffn_params

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

# Wiring sanity: confirm base optimizer betas/lr reached every param group.
_betas_per_group = [tuple(g["betas"]) for g in base_optimizer.param_groups]
_lr_per_group = [g["lr"] for g in base_optimizer.param_groups]
_wd_per_group = [g["weight_decay"] for g in base_optimizer.param_groups]
_base_name = "lion" if cfg.use_lion else "adamw"
_lookahead_on = cfg.lookahead_k > 0
_optim_name = f"lookahead-{_base_name}" if _lookahead_on else _base_name
print(f"Base optimizer: {_base_name}  betas={_betas_per_group}  lr={_lr_per_group}  wd={_wd_per_group}")
_lookahead_sched_desc = ""
if _lookahead_on and cfg.lookahead_alpha_schedule is not None:
    _lookahead_sched_desc = (
        f", alpha_schedule={cfg.lookahead_alpha_schedule}"
        f"({cfg.lookahead_alpha_start}->{cfg.lookahead_alpha_end}"
        f", total_steps={optimizer.total_steps})"
    )
print(f"Lookahead wrap: {'ON' if _lookahead_on else 'OFF'}"
      + (f" (k={cfg.lookahead_k}, alpha={cfg.lookahead_alpha}{_lookahead_sched_desc})" if _lookahead_on else ""))
run.summary["optim/betas"] = list(_betas_per_group[0])
run.summary["optim/n_param_groups"] = len(_betas_per_group)
run.summary["optim/lr_initial"] = _lr_per_group[0]
run.summary["optim/weight_decay"] = _wd_per_group[0]
run.summary["optimizer"] = _optim_name
run.summary["base_optimizer"] = _base_name
run.summary["use_lion"] = cfg.use_lion
run.summary["lion_b2"] = cfg.lion_b2
run.summary["lookahead_k"] = cfg.lookahead_k
run.summary["lookahead_alpha"] = cfg.lookahead_alpha
run.summary["lookahead_on"] = _lookahead_on
run.summary["lookahead_alpha_schedule"] = cfg.lookahead_alpha_schedule or "static"
run.summary["lookahead_alpha_start"] = cfg.lookahead_alpha_start
run.summary["lookahead_alpha_end"] = cfg.lookahead_alpha_end
if _lookahead_on and cfg.lookahead_alpha_schedule is not None:
    run.summary["lookahead_alpha_total_steps"] = optimizer.total_steps

# Rolling buffer for per-step train losses to compute volatility (window=100).
_train_loss_window: deque[float] = deque(maxlen=100)

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    train_loop_t0 = time.time()
    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model({"x": x_norm})["preds"]
            err = F.huber_loss(pred, y_norm, delta=0.1, reduction="none")  # [B, N, 3]

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_mask_3d = vol_mask.unsqueeze(-1)
            surf_mask_3d = surf_mask.unsqueeze(-1)
            vol_loss = (err * vol_mask_3d).sum() / vol_mask_3d.sum().clamp(min=1)
            surf_loss = (err * surf_mask_3d).sum() / surf_mask_3d.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
        optimizer.step()
        global_step += 1
        loss_val = loss.item()
        _train_loss_window.append(loss_val)
        _log_dict = {
            "train/loss": loss_val,
            "train/grad_norm": grad_norm.item(),
            "global_step": global_step,
        }
        if _lookahead_on and optimizer.last_sync_diff_norm is not None:
            _log_dict["train/lookahead_sync_diff_norm"] = optimizer.last_sync_diff_norm
        if _lookahead_on:
            _log_dict["train/lookahead_alpha"] = optimizer.current_alpha
        wandb.log(_log_dict)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
    # Train-loss volatility over last 100 steps in this epoch.
    loss_std_last100 = (
        statistics.pstdev(_train_loss_window) if len(_train_loss_window) >= 2 else 0.0
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_loop_dt = time.time() - train_loop_t0

    scheduler.step()
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

    step_time_ms = train_loop_dt * 1000.0 / max(n_batches, 1)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/loss_std_last100": loss_std_last100,
        "val/loss": val_loss_mean,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "step_time_ms": step_time_ms,
        "gpu_mem_gb_peak": peak_gb,
        "global_step": global_step,
    }
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.
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


    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s, step={step_time_ms:.1f}ms) [{peak_gb:.1f}GB]  "
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
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
