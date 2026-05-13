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


class SurfaceCorrection(nn.Module):
    """Additive correction applied only at surface nodes (zero-initialized).

    Why: BIVW re-weights samples but does not give the model extra surface-
    specific capacity. This head adds a small MLP whose output starts at
    exactly zero (so training begins identically to the BIVW baseline) and
    can specialize to refine surface predictions over time. Volume nodes are
    routed through `torch.where`, which keeps any NaN in volume positions
    from leaking into surface positions (and vice versa).
    """

    def __init__(self, in_dim, out_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_pred, x_norm, is_surface):
        feats = torch.cat([base_pred, x_norm], dim=-1)
        delta = self.net(feats)
        return base_pred + torch.where(
            is_surface.unsqueeze(-1), delta, torch.zeros_like(delta)
        )


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

def evaluate_split(model, surf_head, loader, stats, surf_weight, device) -> dict[str, float]:
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
            pred = model({"x": x_norm})["preds"]
            pred = surf_head(pred, x_norm, is_surface)

            # Guard: data/scoring.py:accumulate_batch has inf*0=NaN when GT has
            # non-finite values. Pre-filter here: nan_to_num both tensors so no
            # inf reaches the product, and tighten the mask to drop bad samples.
            _B = y.shape[0]
            _y_ok = torch.isfinite(y.reshape(_B, -1)).all(dim=-1)  # [B]
            _y_norm_safe = torch.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)
            _pred_safe = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            sq_err = (_pred_safe - _y_norm_safe) ** 2
            _safe_mask = mask & _y_ok[:, None]
            vol_mask = _safe_mask & ~is_surface
            surf_mask = _safe_mask & is_surface
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
            ds, dv = accumulate_batch(
                torch.nan_to_num(pred_orig, nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
                is_surface,
                mask & _y_ok[:, None],
                mae_surf, mae_vol,
            )
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
    huber_delta: float = 1.0  # Huber threshold (normalised space). 0 ⇒ fallback to MSE.
    surf_head_lr: float = 0.0  # If 0.0, uses cfg.lr (encoder LR) for surf_head too
    use_torch_compile: bool = False    # JIT compile the model via torch.compile
    compile_mode: str = "default"      # "default" | "reduce-overhead" | "max-autotune"
    cosine_restart_T_0: int = 0    # First cycle length; 0 = disabled (use single-cycle cosine)
    cosine_restart_T_mult: int = 1  # Cycle length multiplier on each restart; 1 = constant length
    cosine_restart_eta_min: float = 0.0  # Floor LR at cycle-ends for CosineAnnealingWarmRestarts
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    save_cycle2_snapshots: bool = False  # save weights at e15/e17/e19/e21 for within-cycle-2 SWA eval (PR #2522)


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
surf_head = SurfaceCorrection(in_dim=3 + X_DIM, out_dim=3, hidden=64).to(device)
all_params = list(model.parameters()) + list(surf_head.parameters())
n_params = sum(p.numel() for p in all_params)
print(f"Model: Transolver + SurfaceCorrection ({n_params/1e6:.3f}M params)")

if cfg.use_torch_compile:
    # Variable mesh sizes (74K-242K nodes) → use dynamic=True so we get a
    # single symbolic-shape graph instead of one recompile per unique N_max.
    torch.set_float32_matmul_precision("high")  # TF32 matmul
    torch.backends.cudnn.benchmark = True
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    print(f"[torch.compile] mode={cfg.compile_mode} dynamic=True")
    model = torch.compile(model, mode=cfg.compile_mode, dynamic=True)

_head_lr = cfg.surf_head_lr if cfg.surf_head_lr > 0.0 else cfg.lr
optimizer = torch.optim.AdamW(
    [
        {"params": list(model.parameters()), "lr": cfg.lr},
        {"params": list(surf_head.parameters()), "lr": _head_lr},
    ],
    weight_decay=cfg.weight_decay,
)
if cfg.cosine_restart_T_0 > 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.cosine_restart_T_0,
        T_mult=cfg.cosine_restart_T_mult,
        eta_min=cfg.cosine_restart_eta_min,
    )
    print(f"[lr] CosineAnnealingWarmRestarts T_0={cfg.cosine_restart_T_0} T_mult={cfg.cosine_restart_T_mult} eta_min={cfg.cosine_restart_eta_min}")
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    print(f"[lr] CosineAnnealingLR T_max={MAX_EPOCHS}")

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
wandb.define_metric("lr_surf_head", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_avg_surf_p = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()
compile_warmup_s = 0.0
compile_warmup_logged = False

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    surf_head.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        if cfg.use_torch_compile and not compile_warmup_logged:
            _warmup_t0 = time.time()
        pred = model({"x": x_norm})["preds"]
        pred = surf_head(pred, x_norm, is_surface)

        # Per-node Huber (SmoothL1) for surface to align with MAE metric;
        # keep MSE for volume. delta=0 falls back to MSE for surface too.
        delta = cfg.huber_delta
        abs_err = (pred - y_norm).abs()
        if delta > 0:
            surf_node_loss = torch.where(
                abs_err < delta,
                0.5 * abs_err.pow(2) / delta,
                abs_err - 0.5 * delta,
            )
        else:
            surf_node_loss = abs_err.pow(2)
        sq_err = abs_err.pow(2)  # MSE for volume

        # ── Per-sample inverse-variance weighting (BIVW) ─────────────────────
        # y_norm has shape [B, N, 3]; mask has shape [B, N]. Compute each
        # sample's variance over valid nodes only (pooled across the 3 target
        # channels), then weight the loss contribution by 1/var so low-Re
        # (low normalized variance) samples are not under-trained.
        with torch.no_grad():
            B = y_norm.shape[0]
            y_var = torch.empty(B, device=y_norm.device, dtype=y_norm.dtype)
            for b in range(B):
                valid = y_norm[b][mask[b]]
                if valid.numel() == 0:
                    y_var[b] = 1.0
                else:
                    y_var[b] = valid.var().clamp(min=1e-4)
            sample_w = 1.0 / y_var
            sample_w = sample_w / sample_w.mean()
        sw = sample_w[:, None, None]

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * sw * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (surf_node_loss * sw * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if cfg.use_torch_compile and not compile_warmup_logged:
            torch.cuda.synchronize()
            compile_warmup_s = time.time() - _warmup_t0
            wandb.log({"compile/first_batch_s": compile_warmup_s, "global_step": global_step})
            print(f"[torch.compile] first batch (compile+CUDA-graph warmup): {compile_warmup_s:.1f}s")
            compile_warmup_logged = True
        # Fraction of surface errors in the L1 (linear) regime — informs delta choice.
        with torch.no_grad():
            if delta > 0 and surf_mask.any():
                surf_abs = abs_err[surf_mask.unsqueeze(-1).expand_as(abs_err)]
                surf_l1_frac = (surf_abs >= delta).float().mean().item()
            else:
                surf_l1_frac = 0.0
        wandb.log({
            "train/loss": loss.item(),
            "train/sample_w_max": sample_w.max().item(),
            "train/sample_w_min": sample_w.min().item(),
            "train/sample_w_std": sample_w.std().item() if sample_w.numel() > 1 else 0.0,
            "train/y_var_max": y_var.max().item(),
            "train/y_var_min": y_var.min().item(),
            "train/surf_l1_frac": surf_l1_frac,
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # Cycle-2 snapshot saving for within-cycle-2 SWA (PR #2522).
    # Under T_0=7, T_mult=2: cycle 2 spans e8–e21 with cycle-end at e21.
    # We save weights at e15, e17, e19, e21 — all within cycle 2's descent.
    if cfg.save_cycle2_snapshots and (epoch + 1) in (15, 17, 19, 21):
        snap_path = model_dir / f"cycle2_snap_e{epoch+1}.pt"
        torch.save(
            {"model": model.state_dict(), "surf_head": surf_head.state_dict()},
            snap_path,
        )
        print(f"[cycle2-snap] saved {snap_path.name}")

    # --- Validate ---
    model.eval()
    surf_head.eval()
    split_metrics = {
        name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
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
        "lr_surf_head": scheduler.get_last_lr()[-1],
        "train/lr_encoder": optimizer.param_groups[0]["lr"],
        "train/lr_surf_head": optimizer.param_groups[1]["lr"],
        "epoch": epoch + 1,
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
        torch.save(
            {"model": model.state_dict(), "surf_head": surf_head.state_dict()},
            model_path,
        )
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

if cfg.use_torch_compile:
    try:
        import torch._dynamo as _dynamo
        compile_times = _dynamo.utils.compile_times(repr="csv")
        print(f"[torch.compile] compile_times (csv):\n{compile_times}")
        # Frame count is one proxy for recompilation count.
        frame_count = _dynamo.utils.counters.get("frames", {}).get("ok", 0)
        wandb.summary["compile/frames_ok"] = frame_count
        wandb.summary["compile/first_batch_s"] = compile_warmup_s
        print(f"[torch.compile] frames compiled OK: {frame_count}")
    except Exception as e:
        print(f"[torch.compile] could not collect compile stats: {e}")

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
    })

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    surf_head.load_state_dict(ckpt["surf_head"])
    model.eval()
    surf_head.eval()

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
            name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
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

# Within-cycle-2 SWA A/B evaluation (PR #2522).
# A/B: (a) e21 standalone model vs (b) averaged weights of e15+e17+e19+e21.
# Model uses LayerNorm only (no BatchNorm running stats), so no BN
# recalibration is required after loading the averaged state_dict.
if cfg.save_cycle2_snapshots:
    print("\n=== Within-cycle-2 SWA A/B comparison (PR #2522) ===")
    target_epochs = [15, 17, 19, 21]
    snap_files = {e: model_dir / f"cycle2_snap_e{e}.pt" for e in target_epochs}
    available = [(e, p) for e, p in snap_files.items() if p.exists()]
    available_epochs = [e for e, _ in available]
    print(f"  Available snapshots: e{available_epochs}")

    if not available:
        print("  WARNING: no cycle-2 snapshots saved (likely timed out before e15). Skipping.")
    else:
        # 1) Per-snapshot val — verifies the within-basin hypothesis,
        #    and accumulates for SWA averaging in one pass over disk.
        swa_model_state: dict | None = None
        swa_head_state: dict | None = None
        per_snap_val_full: dict[int, tuple[dict, dict]] = {}

        for _e, _p in available:
            _ckpt = torch.load(_p, map_location=device, weights_only=True)
            model.load_state_dict(_ckpt["model"])
            surf_head.load_state_dict(_ckpt["surf_head"])
            model.eval()
            surf_head.eval()
            _split_m = {
                name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
                for name, loader in val_loaders.items()
            }
            _agg = aggregate_splits(_split_m)
            per_snap_val_full[_e] = (_split_m, _agg)
            print(f"  e{_e}: val_avg/mae_surf_p = {_agg['avg/mae_surf_p']:.4f}")

            if swa_model_state is None:
                swa_model_state = {k: v.float().clone() for k, v in _ckpt["model"].items()}
                swa_head_state = {k: v.float().clone() for k, v in _ckpt["surf_head"].items()}
            else:
                for k in swa_model_state:
                    swa_model_state[k] += _ckpt["model"][k].float()
                for k in swa_head_state:
                    swa_head_state[k] += _ckpt["surf_head"][k].float()

            for k, v in _agg.items():
                wandb.summary[f"swa/snap_e{_e}/{k}"] = v
            for split_name, m in _split_m.items():
                for k, v in m.items():
                    wandb.summary[f"swa/snap_e{_e}/{split_name}/{k}"] = v

        n_snaps = len(available)
        for k in swa_model_state:
            swa_model_state[k] /= n_snaps
        for k in swa_head_state:
            swa_head_state[k] /= n_snaps
        print(f"\n  SWA: averaged {n_snaps} snapshots: e{available_epochs}")

        _snap_vals = [per_snap_val_full[_e][1]["avg/mae_surf_p"] for _e in available_epochs]
        _snap_spread = max(_snap_vals) - min(_snap_vals)
        print(f"  Snapshot val_avg spread (max-min): {_snap_spread:.4f}")
        wandb.summary["swa/snap_val_spread"] = _snap_spread
        wandb.summary["swa/n_snapshots"] = n_snaps
        wandb.summary["swa/snap_epochs"] = ",".join(str(e) for e in available_epochs)

        # Always rebuild test loaders (cheap; avoids skip_test path scoping).
        print("\n  Loading test splits for A/B comparison...")
        _test_datasets_swa = load_test_data(cfg.splits_dir, debug=cfg.debug)
        _test_loaders_swa = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in _test_datasets_swa.items()
        }

        # 2) e21 standalone — explicit A/B baseline (val + test).
        e21_val_full = per_snap_val_full.get(21)
        e21_test_metrics = None
        e21_test_avg = None
        if 21 in available_epochs:
            print("\n  Loading e21 snapshot for explicit A/B baseline test eval...")
            _ckpt = torch.load(snap_files[21], map_location=device, weights_only=True)
            model.load_state_dict(_ckpt["model"])
            surf_head.load_state_dict(_ckpt["surf_head"])
            model.eval()
            surf_head.eval()
            e21_test_metrics = {
                name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
                for name, loader in _test_loaders_swa.items()
            }
            e21_test_avg = aggregate_splits(e21_test_metrics)
            print(f"  e21 test_avg/mae_surf_p = {e21_test_avg['avg/mae_surf_p']:.4f}")
            for name in TEST_SPLIT_NAMES:
                print_split_metrics(name, e21_test_metrics[name])
            for k, v in e21_test_avg.items():
                wandb.summary[f"swa/e21_test_{k}"] = v
            for split_name, m in e21_test_metrics.items():
                for k, v in m.items():
                    wandb.summary[f"swa/e21_test/{split_name}/{k}"] = v
        else:
            print("\n  WARN: e21 snapshot missing — no clean A/B baseline available.")

        # 3) SWA averaged model — val + test.
        print("\n  Loading SWA-averaged state and evaluating on val + test...")
        model.load_state_dict(swa_model_state)
        surf_head.load_state_dict(swa_head_state)
        model.eval()
        surf_head.eval()

        swa_val_split = {
            name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
            for name, loader in val_loaders.items()
        }
        swa_val_avg = aggregate_splits(swa_val_split)
        print(f"  SWA val_avg/mae_surf_p = {swa_val_avg['avg/mae_surf_p']:.4f}")
        for name in VAL_SPLIT_NAMES:
            print_split_metrics(name, swa_val_split[name])

        swa_test_metrics = {
            name: evaluate_split(model, surf_head, loader, stats, cfg.surf_weight, device)
            for name, loader in _test_loaders_swa.items()
        }
        swa_test_avg = aggregate_splits(swa_test_metrics)
        print(f"  SWA test_avg/mae_surf_p = {swa_test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, swa_test_metrics[name])

        for k, v in swa_val_avg.items():
            wandb.summary[f"swa/swa_val_{k}"] = v
        for k, v in swa_test_avg.items():
            wandb.summary[f"swa/swa_test_{k}"] = v
        for split_name, m in swa_val_split.items():
            for k, v in m.items():
                wandb.summary[f"swa/swa_val/{split_name}/{k}"] = v
        for split_name, m in swa_test_metrics.items():
            for k, v in m.items():
                wandb.summary[f"swa/swa_test/{split_name}/{k}"] = v

        # 4) Headline A/B
        print("\n  === Headline A/B (e21 vs SWA) ===")
        if e21_val_full is not None:
            _e21_val = e21_val_full[1]["avg/mae_surf_p"]
            _swa_val = swa_val_avg["avg/mae_surf_p"]
            _dv = _swa_val - _e21_val
            print(f"  val:  e21={_e21_val:.4f}   SWA={_swa_val:.4f}   Δ={_dv:+.4f} (SWA-e21; lower-is-better)")
            wandb.summary["swa/headline_e21_val"] = _e21_val
            wandb.summary["swa/headline_swa_val"] = _swa_val
            wandb.summary["swa/headline_delta_val"] = _dv
        if e21_test_avg is not None:
            _e21_test = e21_test_avg["avg/mae_surf_p"]
            _swa_test = swa_test_avg["avg/mae_surf_p"]
            _dt = _swa_test - _e21_test
            print(f"  test: e21={_e21_test:.4f}   SWA={_swa_test:.4f}   Δ={_dt:+.4f} (SWA-e21; lower-is-better)")
            wandb.summary["swa/headline_e21_test"] = _e21_test
            wandb.summary["swa/headline_swa_test"] = _swa_test
            wandb.summary["swa/headline_delta_test"] = _dt

wandb.finish()
