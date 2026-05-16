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
import random
import subprocess
import time
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
    # PR #3644: cosine10 + constant-LR tail + SWA-over-tail.
    # Phase 1 (epochs 0..t_cosine-1):     cosine annealing from `lr` → ~0
    # Phase 2 (epochs t_cosine..t_cosine+n_tail-1): constant `lr_swa` (the JUMP is intentional)
    t_cosine: int = 10
    lr_swa: float = 1e-4
    n_tail: int = 8


cfg = sp.parse(Config)
set_all_seeds(cfg.seed)
if cfg.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
# PR #3644 piecewise schedule: cosine phase (T_COSINE) + constant-LR tail (N_TAIL).
# In debug mode, run a tiny schedule (2 cosine + 2 tail) just to exercise the wiring.
if cfg.debug:
    T_COSINE = 2
    N_TAIL = 2
    MAX_EPOCHS = T_COSINE + N_TAIL
else:
    T_COSINE = cfg.t_cosine
    N_TAIL = cfg.n_tail
    # cfg.epochs is treated as an upper bound; the actual budget is T_COSINE + N_TAIL.
    MAX_EPOCHS = min(cfg.epochs, T_COSINE + N_TAIL)
SWA_TAIL_START = T_COSINE  # 0-indexed: first constant-LR epoch
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

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
# PR #3644: cosine for the first T_COSINE epochs (lr: cfg.lr → ~0). The constant-LR tail
# (cfg.lr_swa) is applied manually after the cosine phase by overriding param_groups['lr'].
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_COSINE, eta_min=0.0)

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

# PR #3644 three-way comparison: track best-by-val separately for the cosine phase
# (epochs 0..T_COSINE-1) and the constant-LR tail (epochs T_COSINE..MAX_EPOCHS-1).
pre_swa_best_avg = float("inf")
pre_swa_best_metrics: dict = {}
pre_swa_path = model_dir / "pre_swa_best.pt"

tail_best_avg = float("inf")
tail_best_metrics: dict = {}
tail_best_path = model_dir / "tail_best.pt"

# SWA: collect a deep-cloned state_dict at the end of each tail epoch.
# At end of training, average all N_TAIL snapshots (uniform weights) → SWA tail.
swa_snapshots: list[dict] = []
swa_path = model_dir / "swa_tail.pt"

# Per-epoch trajectory of val_avg/mae_surf_p during the constant-LR tail —
# diagnostic for the "bounce vs slow drift" question in the report.
tail_trajectory: list[dict] = []

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
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_loop_dt = time.time() - train_loop_t0

    # Capture the LR that was actually used by this epoch's training steps,
    # BEFORE we advance the schedule for the next epoch.
    epoch_train_lr = optimizer.param_groups[0]["lr"]

    # PR #3644 piecewise schedule:
    #   epochs 0..T_COSINE-2 : advance cosine (next epoch uses next cosine value)
    #   epoch == T_COSINE-1  : jump — set next-epoch LR to cfg.lr_swa (constant tail begins)
    #   epochs >= T_COSINE   : no-op (LR stays at cfg.lr_swa)
    if epoch < T_COSINE - 1:
        scheduler.step()
    elif epoch == T_COSINE - 1:
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr_swa

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
    in_tail = epoch >= SWA_TAIL_START
    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "lr": epoch_train_lr,
        "train/lr": epoch_train_lr,
        "schedule/phase": 1 if in_tail else 0,
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

    # PR #3644 three-way tracking: best-by-val within each phase, plus SWA snapshot
    # collection during the constant-LR tail. The "phase-best" checkpoints are used at
    # end-of-training to evaluate each arm on val+test independently.
    if not in_tail:
        if avg_surf_p < pre_swa_best_avg:
            pre_swa_best_avg = avg_surf_p
            pre_swa_best_metrics = {
                "epoch": epoch + 1,
                "val_avg/mae_surf_p": avg_surf_p,
                "per_split": split_metrics,
            }
            torch.save(model.state_dict(), pre_swa_path)
            tag += " [pre*]"
    else:
        if avg_surf_p < tail_best_avg:
            tail_best_avg = avg_surf_p
            tail_best_metrics = {
                "epoch": epoch + 1,
                "val_avg/mae_surf_p": avg_surf_p,
                "per_split": split_metrics,
            }
            torch.save(model.state_dict(), tail_best_path)
            tag += " [tail*]"
        # Append a deep-cloned snapshot of the post-step model for SWA averaging.
        # bf16 autocast leaves weights in fp32, so .clone() preserves precision.
        swa_snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        tail_trajectory.append({
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": {sp: split_metrics[sp]["mae_surf_p"] for sp in VAL_SPLIT_NAMES},
        })

    phase_tag = "TAIL" if in_tail else "COS"
    print(
        f"Epoch {epoch+1:3d} [{phase_tag}] lr={epoch_train_lr:.2e} "
        f"({dt:.0f}s, step={step_time_ms:.1f}ms) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- PR #3644: three-way evaluation (pre_swa / tail_best / swa_tail) ---
# Hoist test loaders so they're reused across all three arm evaluations.
test_loaders = None
if not cfg.skip_test:
    test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
    test_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }


def _average_state_dicts(state_dicts: list[dict]) -> dict:
    """Uniform mean of N state_dicts. Cast each tensor to float for precision,
    then back to the original dtype so the averaged checkpoint is dtype-compatible
    with the live model (Transolver uses fp32 weights, so this is a no-op cast)."""
    n = len(state_dicts)
    avg = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return avg


def _eval_arm(state: dict, label: str) -> tuple[dict, dict, dict | None, dict | None]:
    """Load `state` into `model`, eval on val + test, return (v_metrics, v_avg, t_metrics, t_avg)."""
    model.load_state_dict(state)
    model.eval()
    v_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
        for name, loader in val_loaders.items()
    }
    v_avg = aggregate_splits(v_metrics)
    t_metrics = t_avg = None
    if test_loaders is not None:
        t_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        t_avg = aggregate_splits(t_metrics)
    t_str = f"  test_avg/mae_surf_p={t_avg['avg/mae_surf_p']:.4f}" if t_avg is not None else ""
    print(f"  [{label}] val_avg/mae_surf_p={v_avg['avg/mae_surf_p']:.4f}{t_str}")
    return v_metrics, v_avg, t_metrics, t_avg


print(
    f"\nThree-way SWA evaluation: "
    f"pre_swa_phase_best={pre_swa_best_avg:.4f} (ep {pre_swa_best_metrics.get('epoch', '-')})  "
    f"tail_phase_best={tail_best_avg:.4f} (ep {tail_best_metrics.get('epoch', '-')})  "
    f"snapshots={len(swa_snapshots)}"
)

arm_results: dict = {}  # flat dict for wandb.log / wandb.summary
arm_payloads: dict = {}  # nested for artifact metadata


def _record_arm(arm: str, v_metrics: dict, v_avg: dict, t_metrics, t_avg) -> None:
    arm_results[f"{arm}/val_avg/mae_surf_p"] = v_avg["avg/mae_surf_p"]
    for split in VAL_SPLIT_NAMES:
        arm_results[f"{arm}/{split}/mae_surf_p"] = v_metrics[split]["mae_surf_p"]
    if t_avg is not None:
        arm_results[f"{arm}/test_avg/mae_surf_p"] = t_avg["avg/mae_surf_p"]
        for split in TEST_SPLIT_NAMES:
            arm_results[f"{arm}/{split}/mae_surf_p"] = t_metrics[split]["mae_surf_p"]
    arm_payloads[arm] = {
        "v_metrics": v_metrics,
        "v_avg": v_avg,
        "t_metrics": t_metrics,
        "t_avg": t_avg,
    }


# Arm 1: best-by-val from cosine phase
if pre_swa_best_metrics:
    state = torch.load(pre_swa_path, map_location=device, weights_only=True)
    v_m, v_a, t_m, t_a = _eval_arm(state, "pre_swa_best")
    arm_results["pre_swa/epoch"] = pre_swa_best_metrics["epoch"]
    _record_arm("pre_swa", v_m, v_a, t_m, t_a)

# Arm 2: best-by-val from constant-LR tail
if tail_best_metrics:
    state = torch.load(tail_best_path, map_location=device, weights_only=True)
    v_m, v_a, t_m, t_a = _eval_arm(state, "tail_best")
    arm_results["tail_best/epoch"] = tail_best_metrics["epoch"]
    _record_arm("tail_best", v_m, v_a, t_m, t_a)

# Arm 3: uniform average of all snapshots collected during the constant-LR tail.
# Transolver uses LayerNorm only (no BatchNorm running stats), so no BN recalibration needed.
swa_state = None
if len(swa_snapshots) >= 2:
    swa_state = _average_state_dicts(swa_snapshots)
    torch.save(swa_state, swa_path)
    v_m, v_a, t_m, t_a = _eval_arm(swa_state, f"swa_tail (n={len(swa_snapshots)})")
    arm_results["swa_tail/n_snapshots"] = len(swa_snapshots)
    _record_arm("swa_tail", v_m, v_a, t_m, t_a)
elif len(swa_snapshots) == 1:
    print(f"\n[warn] only {len(swa_snapshots)} tail snapshot collected — SWA arm skipped")
else:
    print("\n[warn] no tail snapshots collected — SWA arm skipped")

# Per-snapshot val_avg trajectory during the constant-LR tail.
# Logged to W&B summary as a list so the bounce-vs-drift question is answerable post-hoc.
if tail_trajectory:
    arm_results["tail_trajectory/epochs"] = [t["epoch"] for t in tail_trajectory]
    arm_results["tail_trajectory/val_avg_mae_surf_p"] = [
        t["val_avg/mae_surf_p"] for t in tail_trajectory
    ]
    for split in VAL_SPLIT_NAMES:
        arm_results[f"tail_trajectory/{split}/mae_surf_p"] = [
            t["per_split"][split] for t in tail_trajectory
        ]

wandb.log(arm_results)
wandb.summary.update(arm_results)

# Pick the best-by-val arm as the canonical artifact (parity with baseline).
arm_val_scores = {
    arm: arm_results.get(f"{arm}/val_avg/mae_surf_p", float("inf"))
    for arm in ("pre_swa", "tail_best", "swa_tail")
}
winning_arm = min(arm_val_scores, key=arm_val_scores.get)
print(
    f"\nArm comparison (val_avg/mae_surf_p):"
    f"  pre_swa={arm_val_scores['pre_swa']:.4f}"
    f"  tail_best={arm_val_scores['tail_best']:.4f}"
    f"  swa_tail={arm_val_scores['swa_tail']:.4f}"
    f"  → winner: {winning_arm}"
)
wandb.summary.update({"winning_arm": winning_arm})

# --- Artifact upload: best-by-val checkpoint (parity with baseline trainer) ---
if best_metrics:
    print(f"\nBest val (any phase): epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
    })

    # If the SWA-averaged arm beat both best-by-val checkpoints, upload its weights as
    # the artifact so downstream consumers get the winning model.
    artifact_source_path = model_path
    if swa_state is not None and arm_val_scores["swa_tail"] < best_avg_surf_p:
        torch.save(swa_state, model_path)  # overwrite with SWA-averaged weights
        artifact_source_path = model_path
        print(f"  → swa_tail wins overall; uploading averaged weights as artifact")

    model.load_state_dict(torch.load(artifact_source_path, map_location=device, weights_only=True))
    model.eval()

    # Re-run test on the artifact-source state for parity logging (top-level test_*)
    test_metrics = None
    test_avg = None
    if test_loaders is not None:
        test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST (artifact)  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
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
