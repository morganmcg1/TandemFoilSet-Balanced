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

import copy
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


class SwiGLUMLP(nn.Module):
    """Gated linear unit MLP block (Shazeer 2020).

    forward: linear_out(SiLU(linear_gate(x)) * linear_value(x))
    """

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.linear_gate = nn.Linear(n_input, n_hidden)
        self.linear_value = nn.Linear(n_input, n_hidden)
        self.linear_out = nn.Linear(n_hidden, n_output)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.linear_out(self.silu(self.linear_gate(x)) * self.linear_value(x))


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
                 use_swiglu=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUMLP(hidden_dim, mlp_hidden, hidden_dim)
        else:
            self.mlp = MLP(hidden_dim, mlp_hidden, hidden_dim,
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


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_swiglu: bool = False):
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
                use_swiglu=use_swiglu,
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
# Output transforms
# ---------------------------------------------------------------------------

def apply_asinh_p(y_norm, scale):
    """Compress the pressure channel of a normalized target tensor with asinh."""
    if scale <= 0:
        return y_norm
    y = y_norm.clone()
    y[..., 2] = torch.asinh(y_norm[..., 2] * scale) / scale
    return y


def invert_asinh_p(pred, scale):
    """Inverse of apply_asinh_p; clamps before sinh to avoid overflow."""
    if scale <= 0:
        return pred
    out = pred.clone()
    out[..., 2] = torch.sinh(pred[..., 2].clamp(-10, 10) * scale) / scale
    return out


def apply_asinh_vel(y_norm, scale):
    """Compress the velocity channels (Ux, Uy) of a normalized target tensor with asinh."""
    if scale <= 0:
        return y_norm
    y = y_norm.clone()
    y[..., 0] = torch.asinh(y_norm[..., 0] * scale) / scale
    y[..., 1] = torch.asinh(y_norm[..., 1] * scale) / scale
    return y


def invert_asinh_vel(pred, scale):
    """Inverse of apply_asinh_vel; clamps before sinh to avoid overflow."""
    if scale <= 0:
        return pred
    out = pred.clone()
    out[..., 0] = torch.sinh(pred[..., 0].clamp(-10, 10) * scale) / scale
    out[..., 1] = torch.sinh(pred[..., 1].clamp(-10, 10) * scale) / scale
    return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   asinh_p_scale: float = 0.0,
                   asinh_vel_scale: float = 0.0) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    When ``asinh_p_scale > 0`` or ``asinh_vel_scale > 0`` the model predicts in
    asinh-compressed normalized space on the corresponding channels; we mirror
    the training loss target with the apply_asinh_* helpers and invert the
    prediction with the invert_asinh_* helpers BEFORE denormalizing for MAE.
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
            y_target = apply_asinh_p(y_norm, asinh_p_scale)
            y_target = apply_asinh_vel(y_target, asinh_vel_scale)
            pred = model({"x": x_norm})["preds"]

            sq_err = (pred - y_target) ** 2
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

            pred_norm = invert_asinh_p(pred, asinh_p_scale)
            pred_norm = invert_asinh_vel(pred_norm, asinh_vel_scale)
            pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
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
    grad_clip: float = 0.0  # max grad norm; 0 disables
    huber_delta: float = 0.0  # Huber transition in normalized space; 0 = MSE
    ema_decay: float = 0.999  # EMA decay rate; smaller = faster shadow tracking
    asinh_p_scale: float = 0.0  # 0 disables; >0 enables asinh on pressure channel
    asinh_vel_scale: float = 0.0  # 0 disables; >0 enables asinh on Ux/Uy channels
    use_swiglu: bool = False  # swap GELU MLP for SwiGLU gated MLP inside TransolverBlocks
    mlp_ratio: float = 2.0  # hidden expansion ratio for the MLP/SwiGLU block; float allows param-match (e.g. 1.333)
    n_head: int = 4  # number of attention heads; n_hidden must be divisible by n_head
    sgdr_t0: int = 0  # CosineAnnealingWarmRestarts cycle length; 0 disables (use plain cosine)
    slice_num: int = 64  # physics-attention slice count (node partitioning granularity)
    adamw_beta2: float = 0.999  # AdamW second-moment EMA decay; default 0.999
    surf_channel_weights: str = "1.0,1.0,1.0"  # per-channel surface loss weights in (Ux, Uy, p) order; mean-preserving normalization applied so (1.0,1.0,1.0) is a no-op


cfg = sp.parse(Config)

# Parse and validate surf_channel_weights — comma-separated 3 floats in (Ux, Uy, p) order
surf_channel_weights = tuple(float(x) for x in cfg.surf_channel_weights.split(","))
assert len(surf_channel_weights) == 3, (
    f"--surf_channel_weights must be 3 comma-separated floats (Ux, Uy, p); got {surf_channel_weights}"
)
assert all(w > 0 for w in surf_channel_weights), (
    f"--surf_channel_weights must be positive; got {surf_channel_weights}"
)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

# Mean-preserving per-channel surface weights: w / w.mean() so uniform (1,1,1) is a no-op
_surf_ch_w_raw = torch.tensor(surf_channel_weights, dtype=torch.float32, device=device)
surf_ch_w = _surf_ch_w_raw / _surf_ch_w_raw.mean()
print(
    f"Surface channel weights (Ux, Uy, p) raw={surf_channel_weights} "
    f"normalized={tuple(surf_ch_w.tolist())}"
)

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
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=cfg.mlp_ratio,
    use_swiglu=cfg.use_swiglu,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)  "
      f"[use_swiglu={cfg.use_swiglu}, mlp_ratio={cfg.mlp_ratio}, n_head={cfg.n_head}, slice_num={cfg.slice_num}]")

ema_model = copy.deepcopy(model)
for p in ema_model.parameters():
    p.requires_grad_(False)
ema_decay = cfg.ema_decay

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, cfg.adamw_beta2),
)
if cfg.sgdr_t0 > 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.sgdr_t0, T_mult=1, eta_min=1e-6
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
        "ema_decay": ema_decay,
        "surf_channel_weights_parsed": list(surf_channel_weights),
        "surf_channel_weights_normalized": surf_ch_w.cpu().tolist(),
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
    epoch_surf_ch = torch.zeros(3, dtype=torch.float64, device=device)
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        y_target = apply_asinh_p(y_norm, cfg.asinh_p_scale)
        y_target = apply_asinh_vel(y_target, cfg.asinh_vel_scale)
        pred = model({"x": x_norm})["preds"]
        if cfg.debug and global_step == 0 and cfg.asinh_p_scale > 0:
            with torch.no_grad():
                yp = y_norm[..., 2][mask]
                yt = y_target[..., 2][mask]
                pp = pred[..., 2][mask]
                pp_inv = invert_asinh_p(pred, cfg.asinh_p_scale)[..., 2][mask]
                pp_phys = (invert_asinh_p(pred, cfg.asinh_p_scale) * stats["y_std"] + stats["y_mean"])[..., 2][mask]
                print(
                    f"[ASINH-DEBUG scale={cfg.asinh_p_scale}] "
                    f"y_norm[p] range=[{yp.min().item():.3f},{yp.max().item():.3f}] "
                    f"y_target[p] range=[{yt.min().item():.3f},{yt.max().item():.3f}] "
                    f"pred[p] range=[{pp.min().item():.3f},{pp.max().item():.3f}] "
                    f"pred_inv[p] range=[{pp_inv.min().item():.3f},{pp_inv.max().item():.3f}] "
                    f"pred_phys[p] range=[{pp_phys.min().item():.3f},{pp_phys.max().item():.3f}]"
                )
        if cfg.huber_delta > 0:
            elem_loss = F.huber_loss(pred, y_target, delta=cfg.huber_delta, reduction="none")
        else:
            elem_loss = (pred - y_target) ** 2

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (elem_loss * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        # Per-channel surface loss (Ux, Uy, p). Each surf_ch_loss[c] is sum_surface(err[c]) / n_surface_nodes
        # so sum_c(surf_ch_loss[c]) reproduces the prior scalar surf_loss exactly.
        n_surf_safe = surf_mask.sum().clamp(min=1)
        surf_ch_loss = (elem_loss * surf_mask.unsqueeze(-1)).sum(dim=(0, 1)) / n_surf_safe  # [3]
        # Mean-preserving channel weights (sum=3 same as uniform) → no loss-scale change vs baseline
        surf_loss = (surf_ch_w * surf_ch_loss).sum()
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm_preclip: float | None = None
        if cfg.grad_clip > 0:
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()
        optimizer.step()
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        global_step += 1
        step_log = {"train/loss": loss.item(), "global_step": global_step}
        if grad_norm_preclip is not None:
            step_log["train/grad_norm_preclip"] = grad_norm_preclip
        wandb.log(step_log)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        epoch_surf_ch += surf_ch_loss.detach().double()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_surf_ch /= max(n_batches, 1)
    epoch_surf_ch_list = epoch_surf_ch.cpu().tolist()  # [Ux, Uy, p]

    # --- Validate ---
    model.eval()
    ema_model.eval()
    split_metrics = {
        name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device,
                             asinh_p_scale=cfg.asinh_p_scale,
                             asinh_vel_scale=cfg.asinh_vel_scale)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    with torch.no_grad():
        ema_lag_sq = 0.0
        model_norm_sq = 0.0
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_lag_sq += (ema_p.data - p.data).pow(2).sum().item()
            model_norm_sq += p.data.pow(2).sum().item()
        ema_lag = ema_lag_sq ** 0.5
        model_norm = model_norm_sq ** 0.5
        ema_lag_rel = ema_lag / max(model_norm, 1e-12)

    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/surf_loss_ux": epoch_surf_ch_list[0],
        "train/surf_loss_uy": epoch_surf_ch_list[1],
        "train/surf_loss_p": epoch_surf_ch_list[2],
        "train/ema_lag": ema_lag,
        "train/ema_lag_rel": ema_lag_rel,
        "train/model_param_norm": model_norm,
        "val/loss": val_loss_mean,
        "lr": scheduler.get_last_lr()[0],
        "train/lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
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
        torch.save(ema_model.state_dict(), model_path)
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                                 asinh_p_scale=cfg.asinh_p_scale,
                                 asinh_vel_scale=cfg.asinh_vel_scale)
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
