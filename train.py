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


class FourierPositionalEncoding(nn.Module):
    """Sinusoidal position encoding for the first ``space_dim`` channels of x.

    gamma(x) = [sin(2^k pi x), cos(2^k pi x)] for k=0..L-1, applied to each
    spatial dimension independently. Output dim = 2 * space_dim * num_freqs.
    Non-spatial channels (idx >= space_dim) pass through unchanged.
    """

    def __init__(self, space_dim: int, num_freqs: int):
        super().__init__()
        self.space_dim = space_dim
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()  # [1, 2, 4, ..., 2^(L-1)]
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        pos = x[..., : self.space_dim]
        feats = x[..., self.space_dim :]
        arg = pos.unsqueeze(-1) * (math.pi * self.freqs)
        sin = torch.sin(arg).flatten(-2)
        cos = torch.cos(arg).flatten(-2)
        return torch.cat([sin, cos, feats], dim=-1)


class GLUMLP(nn.Module):
    """Gated-Linear-Unit MLP (Shazeer 2020). Drop-in for the in-block MLP.

    SwiGLU/GeGLU: w_down( act(w_gate(x)) * w_up(x) ). Same hidden width as the
    vanilla MLP — accepts ~33% more params in the MLP block for simpler ablation.
    """
    def __init__(self, n_input, n_hidden, n_output, glu_act="silu"):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "gelu": nn.GELU}[glu_act]
        self.w_gate = nn.Linear(n_input, n_hidden)
        self.w_up = nn.Linear(n_input, n_hidden)
        self.w_down = nn.Linear(n_hidden, n_output)
        self.act = act_fn()

    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


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
                 mlp_type="vanilla"):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if mlp_type == "vanilla":
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
        elif mlp_type == "swiglu":
            self.mlp = GLUMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                              glu_act="silu")
        elif mlp_type == "geglu":
            self.mlp = GLUMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                              glu_act="gelu")
        else:
            raise ValueError(f"unknown mlp_type={mlp_type}")
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
                 pos_enc_mode: str = "raw",
                 pos_enc_num_freqs: int = 8,
                 mlp_type: str = "vanilla"):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.pos_enc_mode = pos_enc_mode

        if self.unified_pos:
            self.pos_enc = None
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            if pos_enc_mode == "raw":
                self.pos_enc = None
                preprocess_in = fun_dim + space_dim
            elif pos_enc_mode in ("fourier_basic", "fourier_rich"):
                num_freqs = (pos_enc_num_freqs if pos_enc_mode == "fourier_basic"
                             else pos_enc_num_freqs + 4)
                self.pos_enc = FourierPositionalEncoding(space_dim, num_freqs)
                preprocess_in = fun_dim + 2 * space_dim * num_freqs
            else:
                raise ValueError(f"Unknown pos_enc_mode={pos_enc_mode!r}")
            self.preprocess = MLP(preprocess_in, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                mlp_type=mlp_type,
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
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, loss_fn="mse", eps=1e-3) -> dict[str, float]:
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
            with amp_ctx():
                pred = model({"x": x_norm})["preds"]

            # Drop samples whose ground truth contains any non-finite value
            # before computing loss/MAE. scoring.accumulate_batch tries to skip
            # such samples but NaN * 0.0 == NaN poisons the float64 accumulator;
            # filtering the batch first matches the intended "skip" semantics.
            B = y.shape[0]
            y_finite_per_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not y_finite_per_sample.all():
                if not y_finite_per_sample.any():
                    continue
                pred = pred[y_finite_per_sample]
                y = y[y_finite_per_sample]
                y_norm = y_norm[y_finite_per_sample]
                is_surface = is_surface[y_finite_per_sample]
                mask = mask[y_finite_per_sample]

            with amp_ctx():
                err = _per_node_loss(pred, y_norm, loss_fn, eps)
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

            pred_orig = pred.float() * stats["y_std"] + stats["y_mean"]
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
# Parameter EMA (Polyak averaging for evaluation)
# ---------------------------------------------------------------------------


class ParameterEMA:
    """Shadow-parameter EMA, swap-in for evaluation.

    With ``warmup=True`` the effective decay ramps up as
    ``min(target_decay, (1+t)/(10+t))`` (Karras et al.), so the shadow is not
    biased toward the initialization during early training. This matters most
    for large target decays (e.g. 0.9999) over moderate-length runs.
    """

    def __init__(self, model, decay: float, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.backup: dict = {}

    def effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        t = self.step_count
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        d = self.effective_decay()
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            p.copy_(self.backup[n])
        self.backup = {}


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
    warmup_epochs: int = 3
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    loss_fn: str = "charbonnier"   # "mse" or "charbonnier"
    charbonnier_eps: float = 1e-3  # ε for Charbonnier sqrt(r² + ε²)
    grad_clip_max_norm: float = 0.5  # 0.0 = no clipping, >0 = clip global L2 norm
    pos_enc_mode: str = "raw"        # "raw" | "fourier_basic" | "fourier_rich"
    pos_enc_num_freqs: int = 8        # frequency bands when mode != raw
    amp_dtype: str = "fp32"   # "fp32" | "bf16"
    mlp_type: str = "vanilla"  # "vanilla" | "swiglu" | "geglu"
    use_compile: bool = False  # wrap model with torch.compile if True
    ema_decay: float = 0.0  # 0 disables; >0 enables shadow-parameter EMA for eval
    ema_warmup: bool = True  # Karras-style decay warmup: min(decay, (1+t)/(10+t))


def _per_node_loss(pred, y, fn, eps):
    """Element-wise loss in normalized space, same shape as (pred - y)."""
    diff = pred - y
    if fn == "mse":
        return diff ** 2
    if fn == "charbonnier":
        return torch.sqrt(diff * diff + eps * eps)
    raise ValueError(fn)


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

_AMP_DTYPE_MAP = {"fp32": None, "bf16": torch.bfloat16}
if cfg.amp_dtype not in _AMP_DTYPE_MAP:
    raise ValueError(f"--amp_dtype must be one of {list(_AMP_DTYPE_MAP)}; got {cfg.amp_dtype!r}")
AMP_DTYPE = _AMP_DTYPE_MAP[cfg.amp_dtype]


def amp_ctx():
    if AMP_DTYPE is None:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE)


print(f"AMP: amp_dtype={cfg.amp_dtype}" + (" (autocast disabled)" if AMP_DTYPE is None else ""))

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
    pos_enc_mode=cfg.pos_enc_mode,
    pos_enc_num_freqs=cfg.pos_enc_num_freqs,
    mlp_type=cfg.mlp_type,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

if cfg.use_compile:
    model = torch.compile(model, mode="default")
    print("Wrapped model with torch.compile(mode='default')")

ema = ParameterEMA(model, cfg.ema_decay, warmup=cfg.ema_warmup) if cfg.ema_decay > 0 else None
if ema is not None:
    print(f"EMA: enabled (target decay={cfg.ema_decay}, warmup={cfg.ema_warmup})")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
if cfg.warmup_epochs > 0:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, MAX_EPOCHS - cfg.warmup_epochs), eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs],
    )
else:
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
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")
wandb.define_metric("ema/*", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_avg_surf_p = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()
_logged_x_range = False

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
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        if not _logged_x_range:
            valid = mask
            pos = x_norm[..., :2][valid]
            xn_min = pos.min(dim=0).values.tolist()
            xn_max = pos.max(dim=0).values.tolist()
            xn_mean = pos.mean(dim=0).tolist()
            xn_std = pos.std(dim=0).tolist()
            print(
                f"x_norm[:, :2] first-batch valid-node stats: "
                f"min={xn_min} max={xn_max} mean={xn_mean} std={xn_std}"
            )
            wandb.summary["x_norm_pos_min"] = xn_min
            wandb.summary["x_norm_pos_max"] = xn_max
            wandb.summary["x_norm_pos_mean"] = xn_mean
            wandb.summary["x_norm_pos_std"] = xn_std
            _logged_x_range = True
        with amp_ctx():
            pred = model({"x": x_norm})["preds"]
            err = _per_node_loss(pred, y_norm, cfg.loss_fn, cfg.charbonnier_eps)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip_max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.grad_clip_max_norm
            )
            grad_norm_key = "train/grad_norm_pre_clip"
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            grad_norm_key = "train/grad_norm_unclipped"
        optimizer.step()
        if ema is not None:
            ema.update(model)
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            grad_norm_key: total_norm.item(),
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    # Swap in EMA shadow weights for evaluation and checkpoint saving.
    if ema is not None:
        ema.apply_to(model)

    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.loss_fn, cfg.charbonnier_eps)
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
    if ema is not None:
        log_metrics["ema/effective_decay"] = ema.effective_decay()
        log_metrics["ema/step_count"] = ema.step_count
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
        # Save with EMA still applied — checkpoint holds EMA weights.
        torch.save(model.state_dict(), model_path)
        tag = " *"

    # Restore live (non-EMA) weights to continue training.
    if ema is not None:
        ema.restore(model)

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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, cfg.loss_fn, cfg.charbonnier_eps)
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
