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

import copy
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

def evaluate_split(model, loader, stats, surf_weight, device, beta=1.0) -> dict[str, float]:
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

            # Drop samples whose ground truth contains non-finite values; the
            # scoring helper's mask-after-subtract path otherwise propagates NaN
            # through accumulators (NaN * 0 = NaN). One such sample exists in
            # test_geom_camber_cruise (index 20, pressure channel).
            y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not y_finite.any():
                continue
            if not y_finite.all():
                x = x[y_finite]
                y = y[y_finite]
                is_surface = is_surface[y_finite]
                mask = mask[y_finite]

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model({"x": x_norm})["preds"]

                elem_err = F.smooth_l1_loss(pred, y_norm, reduction='none', beta=beta)
                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vol_loss_sum += (
                    (elem_err * vol_mask.unsqueeze(-1)).sum()
                    / vol_mask.sum().clamp(min=1)
                ).item()
                surf_loss_sum += (
                    (elem_err * surf_mask.unsqueeze(-1)).sum()
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
    extras: dict | None = None,
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
        "smooth_l1_beta": cfg.smooth_l1_beta,
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

    if extras:
        summary.update(extras)

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
    surf_weight: float = 25.0
    epochs: int = 50
    smooth_l1_beta: float = 1.0  # transition point between quadratic/linear regions of SmoothL1
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    # SWA tail averaging: uniform average of EMA weights across the last K epochs
    # (Izmailov et al. 2018). Set >0 to enable; 0 disables and the pipeline runs EMA-only.
    swa_tail_epochs: int = 0


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

ema_decay = 0.999
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
print(f"EMA shadow model created with decay={ema_decay}")

# SWA accumulator: uniform average of EMA weights across the last
# cfg.swa_tail_epochs epochs. Stored as a list of state_dicts; we average on
# demand at validation time inside the tail window.
swa_states: list[dict] = []
if cfg.swa_tail_epochs > 0:
    print(f"SWA tail averaging enabled over last {cfg.swa_tail_epochs} epoch(s) of EMA weights")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model({"x": x_norm})["preds"]
            elem_err = F.smooth_l1_loss(pred, y_norm, reduction='none', beta=cfg.smooth_l1_beta)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (elem_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (elem_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
            for eb, b in zip(ema_model.buffers(), model.buffers()):
                eb.copy_(b)

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # SWA bookkeeping: in the last K epochs, snapshot the current EMA state.
    epochs_remaining = MAX_EPOCHS - (epoch + 1)
    in_swa_tail = cfg.swa_tail_epochs > 0 and epochs_remaining < cfg.swa_tail_epochs
    if in_swa_tail:
        swa_states.append({k: v.detach().clone() for k, v in ema_model.state_dict().items()})

    # --- Validate ---
    # Always evaluate the live EMA model so val_avg_ema is logged every epoch.
    # In the tail window, additionally evaluate the SWA-averaged model (uniform
    # average of all EMA snapshots so far) by swapping its weights into
    # ema_model for the duration of the eval, then restoring.
    model.eval()
    ema_model.eval()
    split_metrics_ema = {
        name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device, beta=cfg.smooth_l1_beta)
        for name, loader in val_loaders.items()
    }
    avg_surf_p_ema = aggregate_splits(split_metrics_ema)["avg/mae_surf_p"]

    split_metrics_swa = None
    avg_surf_p_swa = None
    if in_swa_tail and len(swa_states) > 0:
        averaged = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in swa_states[0].items()}
        for state in swa_states:
            for k in averaged:
                averaged[k] += state[k].to(torch.float32) / len(swa_states)
        averaged = {k: v.to(swa_states[0][k].dtype) for k, v in averaged.items()}
        _ema_backup = {k: v.detach().clone() for k, v in ema_model.state_dict().items()}
        ema_model.load_state_dict(averaged)
        split_metrics_swa = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device, beta=cfg.smooth_l1_beta)
            for name, loader in val_loaders.items()
        }
        avg_surf_p_swa = aggregate_splits(split_metrics_swa)["avg/mae_surf_p"]
        ema_model.load_state_dict(_ema_backup)

    # Primary metric used for checkpoint selection: SWA-eval inside the tail
    # window (the experimentally-relevant deployment artifact), EMA-eval
    # outside.
    if avg_surf_p_swa is not None:
        avg_surf_p = avg_surf_p_swa
        split_metrics = split_metrics_swa
        primary_source = "swa"
    else:
        avg_surf_p = avg_surf_p_ema
        split_metrics = split_metrics_ema
        primary_source = "ema"

    dt = time.time() - t0

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
            "primary_source": primary_source,
        }
        torch.save(ema_model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    epoch_record = {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_avg/mae_surf_p_ema": avg_surf_p_ema,
        "val_avg/mae_surf_p_swa": avg_surf_p_swa,
        "val_splits": split_metrics,
        "val_splits_ema": split_metrics_ema,
        "val_splits_swa": split_metrics_swa,
        "is_best": tag == " *",
        "primary_source": primary_source,
        "in_swa_tail": in_swa_tail,
        "n_swa_states": len(swa_states),
    }
    append_metrics_jsonl(metrics_jsonl_path, epoch_record)
    swa_str = f" swa={avg_surf_p_swa:.4f}" if avg_surf_p_swa is not None else ""
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[ema={avg_surf_p_ema:.4f}{swa_str}]  primary={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    ema_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ema_model.eval()

    # --- Build FINAL SWA-averaged model (if active and at least one snapshot) ---
    # The deployment artifact is the uniform average across all K EMA snapshots
    # from the tail window. Test eval is run on this final SWA model per the PR
    # spec; it is independent of the best-checkpoint logic (which uses primary
    # val metric and may save an earlier-tail SWA snapshot).
    final_swa_val_metrics: dict | None = None
    final_swa_val_avg: float | None = None
    swa_path: Path | None = None
    if cfg.swa_tail_epochs > 0 and len(swa_states) > 0:
        averaged_final = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in swa_states[0].items()}
        for state in swa_states:
            for k in averaged_final:
                averaged_final[k] += state[k].to(torch.float32) / len(swa_states)
        averaged_final = {k: v.to(swa_states[0][k].dtype) for k, v in averaged_final.items()}
        swa_path = model_dir / "swa_final.pt"
        torch.save(averaged_final, swa_path)
        print(f"\nFinal SWA model saved (uniform avg over {len(swa_states)} tail EMA snapshots): {swa_path}")

        # Eval the FINAL SWA model on val splits (already computed in last
        # tail epoch but re-run here for a clean separate summary).
        _ema_backup = {k: v.detach().clone() for k, v in ema_model.state_dict().items()}
        ema_model.load_state_dict(averaged_final)
        final_swa_val_metrics = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device, beta=cfg.smooth_l1_beta)
            for name, loader in val_loaders.items()
        }
        final_swa_val_avg = aggregate_splits(final_swa_val_metrics)["avg/mae_surf_p"]
        ema_model.load_state_dict(_ema_backup)
        print(f"  FINAL SWA val_avg/mae_surf_p = {final_swa_val_avg:.4f}")
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(f"final_swa/{name}", final_swa_val_metrics[name])

    # Decide which model to test on:
    # - If SWA is active and we have snapshots, test on the FINAL SWA-averaged
    #   model (per PR spec).
    # - Otherwise, test on the EMA-best checkpoint (standard pipeline).
    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        if final_swa_val_metrics is not None:
            test_model_label = "final_swa"
            print(f"\nEvaluating FINAL SWA model on held-out test splits...")
            _ema_backup = {k: v.detach().clone() for k, v in ema_model.state_dict().items()}
            ema_model.load_state_dict(averaged_final)
        else:
            test_model_label = "ema_best"
            print(f"\nEvaluating EMA-best on held-out test splits...")

        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device, beta=cfg.smooth_l1_beta)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        if final_swa_val_metrics is not None:
            ema_model.load_state_dict(_ema_backup)

        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "test_model": test_model_label,
            "test_avg": test_avg,
            "test_splits": test_metrics,
        })

    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "final",
        "swa_tail_epochs": cfg.swa_tail_epochs,
        "n_swa_states": len(swa_states),
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "best_primary_source": best_metrics.get("primary_source"),
        "final_swa_val_avg/mae_surf_p": final_swa_val_avg,
        "final_swa_val_splits": final_swa_val_metrics,
        "swa_checkpoint": str(swa_path) if swa_path is not None else None,
    })

    extras = {
        "swa_tail_epochs": cfg.swa_tail_epochs,
        "n_swa_states": len(swa_states),
        "best_primary_source": best_metrics.get("primary_source"),
        "final_swa_val_avg/mae_surf_p": final_swa_val_avg,
        "swa_checkpoint": str(swa_path) if swa_path is not None else None,
    }
    if final_swa_val_metrics is not None:
        for split_name, m in final_swa_val_metrics.items():
            for k, v in m.items():
                extras[f"final_swa_val/{split_name}/{k}"] = v

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
        extras=extras,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")
