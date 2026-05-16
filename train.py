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

# Horizontal-flip TTA: dsdf channels 4..11 reorder as a mirror about the x-axis.
# Assumes the 8 dsdf channels are 8-directional distances around each node;
# mirroring about x-axis swaps (NE↔SE, N↔S, NW↔SW), leaves (E, W) fixed.
# Channel local indices (0..7) permute as [0, 7, 6, 5, 4, 3, 2, 1] →
# global indices [4..11] permute as [4, 11, 10, 9, 8, 7, 6, 5].
DSDF_FLIP_PERM = [4, 11, 10, 9, 8, 7, 6, 5]


def flip_x_horizontal(x: torch.Tensor, _dsdf_idx_cache: dict = {}) -> torch.Tensor:
    """Mirror raw input features about the x-axis (z → -z, plus dependents).

    Operates in raw (un-normalized) feature space. Returns a new tensor.

    Channel layout (data/prepare_splits.py preprocess):
      0-1   pos (x, z)              — z negated
      2-3   saf (rel.x, rel.z)      — rel.z negated
      4-11  dsdf (8 directions)     — permuted (mirror about x-axis)
      12    is_surface              — unchanged
      13    log(Re)                 — unchanged
      14    AoA0 (rad)              — negated
      15    NACA0 camber (M/9)      — negated (mirror foil shape)
      16-17 NACA0 position/thick.   — unchanged
      18    AoA1                    — negated
      19    NACA1 camber            — negated
      20-21 NACA1 position/thick.   — unchanged
      22    gap (z-separation)      — negated
      23    stagger (x-separation)  — unchanged
    """
    idx = _dsdf_idx_cache.get(x.device)
    if idx is None:
        idx = torch.tensor(DSDF_FLIP_PERM, device=x.device, dtype=torch.long)
        _dsdf_idx_cache[x.device] = idx
    xf = x.clone()
    xf[..., 1] = -x[..., 1]
    xf[..., 3] = -x[..., 3]
    xf[..., 4:12] = x.index_select(-1, idx)
    xf[..., 14] = -x[..., 14]
    xf[..., 15] = -x[..., 15]
    xf[..., 18] = -x[..., 18]
    xf[..., 19] = -x[..., 19]
    xf[..., 22] = -x[..., 22]
    return xf


def flip_pred_y(pred: torch.Tensor) -> torch.Tensor:
    """Un-flip a model prediction made on a horizontally-flipped input.

    Outputs are (Ux, Uy, p) in normalized space; only Uy (channel 1) changes sign.
    """
    out = pred.clone()
    out[..., 1] = -pred[..., 1]
    return out


def _make_eval_state(device):
    return {
        "vol_loss_sum": 0.0,
        "surf_loss_sum": 0.0,
        "mae_surf": torch.zeros(3, dtype=torch.float64, device=device),
        "mae_vol": torch.zeros(3, dtype=torch.float64, device=device),
        "n_surf": 0,
        "n_vol": 0,
    }


def _accumulate_pred(state, pred_norm, y_norm, y, is_surface, mask, stats):
    sq_err = (pred_norm - y_norm) ** 2
    vol_mask = mask & ~is_surface
    surf_mask = mask & is_surface
    state["vol_loss_sum"] += (
        (sq_err * vol_mask.unsqueeze(-1)).sum()
        / vol_mask.sum().clamp(min=1)
    ).item()
    state["surf_loss_sum"] += (
        (sq_err * surf_mask.unsqueeze(-1)).sum()
        / surf_mask.sum().clamp(min=1)
    ).item()
    pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
    ds, dv = accumulate_batch(pred_orig, y, is_surface, mask,
                              state["mae_surf"], state["mae_vol"])
    state["n_surf"] += ds
    state["n_vol"] += dv


def _finalize_eval(state, n_batches, surf_weight):
    vol_loss = state["vol_loss_sum"] / max(n_batches, 1)
    surf_loss = state["surf_loss_sum"] / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(state["mae_surf"], state["mae_vol"],
                              state["n_surf"], state["n_vol"]))
    return out


def evaluate_split(model, loader, stats, surf_weight, device) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    Backward-compatible single-pass evaluator (raw predictions only).
    """
    state = _make_eval_state(device)
    n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            _y_fin = torch.isfinite(y).all(dim=-1)
            if not _y_fin.all():
                y = torch.where(_y_fin.unsqueeze(-1), y, torch.zeros_like(y))
                mask = mask & _y_fin

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]

            _accumulate_pred(state, pred, y_norm, y, is_surface, mask, stats)
            n_batches += 1

    return _finalize_eval(state, n_batches, surf_weight)


def evaluate_split_with_tta(model, loader, stats, surf_weight, device):
    """Compute raw and horizontal-flip TTA metrics in one pass.

    For each batch, runs two forward passes:
      1. raw input → raw prediction
      2. flipped input → flipped prediction (un-flipped via Uy negation)
    Then the TTA prediction is the per-node mean of (raw, un-flipped).
    Both raw and TTA predictions feed independent accumulators using the same
    organizer-aligned scoring helpers.

    Returns (metrics_raw, metrics_tta) — each a dict from evaluate_split format.
    """
    raw_state = _make_eval_state(device)
    tta_state = _make_eval_state(device)
    n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            _y_fin = torch.isfinite(y).all(dim=-1)
            if not _y_fin.all():
                y = torch.where(_y_fin.unsqueeze(-1), y, torch.zeros_like(y))
                mask = mask & _y_fin

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred_raw = model({"x": x_norm})["preds"]

            x_flipped = flip_x_horizontal(x)
            x_flipped_norm = (x_flipped - stats["x_mean"]) / stats["x_std"]
            pred_flipped = flip_pred_y(model({"x": x_flipped_norm})["preds"])

            pred_tta = 0.5 * (pred_raw + pred_flipped)

            _accumulate_pred(raw_state, pred_raw, y_norm, y, is_surface, mask, stats)
            _accumulate_pred(tta_state, pred_tta, y_norm, y, is_surface, mask, stats)
            n_batches += 1

    return (
        _finalize_eval(raw_state, n_batches, surf_weight),
        _finalize_eval(tta_state, n_batches, surf_weight),
    )


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
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

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
    wandb.define_metric(f"{_name}_tta/*", step_metric="global_step")
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
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_loop_dt = time.time() - train_loop_t0

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate (raw + TTA in one pass) ---
    model.eval()
    split_metrics_raw: dict[str, dict[str, float]] = {}
    split_metrics_tta: dict[str, dict[str, float]] = {}
    for name, loader in val_loaders.items():
        m_raw, m_tta = evaluate_split_with_tta(model, loader, stats, cfg.surf_weight, device)
        split_metrics_raw[name] = m_raw
        split_metrics_tta[name] = m_tta

    val_avg_raw = aggregate_splits(split_metrics_raw)
    val_avg_tta = aggregate_splits(split_metrics_tta)
    avg_surf_p = val_avg_raw["avg/mae_surf_p"]
    avg_surf_p_tta = val_avg_tta["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics_raw.values()) / len(split_metrics_raw)
    dt = time.time() - t0

    step_time_ms = train_loop_dt * 1000.0 / max(n_batches, 1)
    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "step_time_ms": step_time_ms,
        "global_step": global_step,
    }
    for split_name, m in split_metrics_raw.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for split_name, m in split_metrics_tta.items():
        for k, v in m.items():
            log_metrics[f"{split_name}_tta/{k}"] = v
    for k, v in val_avg_raw.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.
    for k, v in val_avg_tta.items():
        log_metrics[f"val_{k}_tta"] = v  # val_avg/mae_surf_p_tta etc.
    wandb.log(log_metrics)

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "val_avg/mae_surf_p_tta": avg_surf_p_tta,
            "per_split": split_metrics_raw,
            "per_split_tta": split_metrics_tta,
        }
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s, step={step_time_ms:.1f}ms) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f} (TTA={avg_surf_p_tta:.4f}){tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics_raw[name])
        print_split_metrics(f"  └ TTA {name}", split_metrics_tta[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f} "
          f"(TTA at same epoch = {best_metrics['val_avg/mae_surf_p_tta']:.4f})")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "best_val_avg/mae_surf_p_tta": best_metrics["val_avg/mae_surf_p_tta"],
        "total_train_minutes": total_time,
    })

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits (raw + TTA)...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {}
        test_metrics_tta = {}
        for name, loader in test_loaders.items():
            m_raw, m_tta = evaluate_split_with_tta(model, loader, stats, cfg.surf_weight, device)
            test_metrics[name] = m_raw
            test_metrics_tta[name] = m_tta
        test_avg = aggregate_splits(test_metrics)
        test_avg_tta = aggregate_splits(test_metrics_tta)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}  "
              f"avg_surf_p_tta={test_avg_tta['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
            print_split_metrics(f"  └ TTA {name}", test_metrics_tta[name])

        test_log: dict[str, float] = {}
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                test_log[f"test/{split_name}/{k}"] = v
        for split_name, m in test_metrics_tta.items():
            for k, v in m.items():
                test_log[f"test/{split_name}_tta/{k}"] = v
        for k, v in test_avg.items():
            test_log[f"test_{k}"] = v
        for k, v in test_avg_tta.items():
            test_log[f"test_{k}_tta"] = v
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
