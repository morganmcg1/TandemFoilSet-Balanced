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
from concurrent.futures import ThreadPoolExecutor
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
from scipy.spatial import cKDTree
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
# Divergence-free auxiliary loss: KNN cache + gradient estimator
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """Wraps a base dataset, adding the sample index and precomputed KNN tensor."""

    def __init__(self, base_ds: Dataset, knn_cache: list[torch.Tensor]):
        self.base_ds = base_ds
        self.knn_cache = knn_cache  # list of [N_i, K] CPU long tensors

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, is_surface = self.base_ds[idx]
        return x, y, is_surface, self.knn_cache[idx]


def indexed_pad_collate(batch):
    """Pad variable-length mesh samples (including KNN indices) into a batch.

    Returns (x, y, is_surface, mask, knn_idx). knn_idx is [B, N_max, K] long.
    Padded KNN rows point to themselves so dx = positions[i] - positions[i] = 0
    and the least-squares solve degenerates cleanly to G=0 (no singular matrix).
    """
    xs, ys, surfs, knns = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    K = knns[0].shape[1]
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    # Default KNN: each node points to itself K times -> dx=0 for padded rows.
    self_idx = torch.arange(max_n, dtype=torch.long).unsqueeze(-1).expand(-1, K)
    knn_pad = self_idx.unsqueeze(0).expand(B, -1, -1).contiguous()
    for i, (x, y, sf, knn) in enumerate(zip(xs, ys, surfs, knns)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
        knn_pad[i, :n] = knn
    return x_pad, y_pad, surf_pad, mask, knn_pad


def _build_knn_for_sample(args: tuple[int, str, int]) -> tuple[int, torch.Tensor]:
    """Build [N, K] long tensor of KNN indices for one sample.

    Loads only the positions from disk (dims 0-1 of x), builds cKDTree, queries
    k+1 (to drop self), returns indices excluding self.
    """
    idx, path, k = args
    s = torch.load(path, weights_only=True)
    pos = s["x"][:, :2].numpy().astype(np.float32)
    tree = cKDTree(pos, leafsize=32)
    # k+1 because the closest point is the query itself
    _, indices = tree.query(pos, k=k + 1, workers=1)
    # indices: [N, k+1]. Drop self-column (typically idx 0, but use mask to be safe).
    self_arr = np.arange(pos.shape[0])[:, None]
    is_self = indices == self_arr  # [N, k+1]
    # For each row, drop the self entry: keep the k columns that are not self.
    # If self isn't found in the k+1 nearest (impossible w/ leafsize default), fall back to dropping col 0.
    keep_mask = ~is_self
    keep_counts = keep_mask.sum(axis=1)
    if (keep_counts == k).all():
        # Standard case: exactly one self entry per row
        knn = indices[keep_mask].reshape(-1, k)
    else:
        # Defensive fallback (duplicate-position rows): drop col 0
        knn = indices[:, 1:]
    return idx, torch.from_numpy(knn.astype(np.int64))


def build_knn_cache(base_ds, k: int, max_workers: int = 8) -> list[torch.Tensor]:
    """Build a list of [N_i, K] long tensors, one per sample in base_ds."""
    files = [str(p) for p in base_ds.files]
    cache: list[torch.Tensor | None] = [None] * len(files)
    args_iter = [(i, p, k) for i, p in enumerate(files)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, knn in tqdm(pool.map(_build_knn_for_sample, args_iter),
                              total=len(files), desc=f"KNN k={k}"):
            cache[idx] = knn
    return cache  # type: ignore[return-value]


def compute_velocity_divergence(
    positions: torch.Tensor,   # [B, N, 2] physical coordinates
    velocities: torch.Tensor,  # [B, N, 2] physical-scale Ux, Uy
    knn_indices: torch.Tensor, # [B, N, K] long
) -> torch.Tensor:             # returns [B, N] divergence
    """KNN least-squares estimator of div(u) on irregular meshes.

    For each node, fits ∇u ∈ R^{2x2} from neighbor displacements via the normal
    equations (dx^T dx) G = dx^T du. Forced to fp32 (autocast disabled) because
    bf16 einsum loses precision on squared-distance accumulations and produces
    NaN on near-degenerate neighborhoods. Divergence = trace(G).
    """
    B, N, K = knn_indices.shape

    # Disable autocast so einsum and the 2x2 inversion run in fp32, not bf16.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        positions_f32 = positions.float()
        velocities_f32 = velocities.float()

        # Gather neighbor positions and velocities. Reshape KNN to [B, N*K, 1]
        # then broadcast to [B, N*K, 2] for a single gather along node dim.
        idx_flat = knn_indices.reshape(B, N * K, 1)
        pos_neighbors = positions_f32.gather(1, idx_flat.expand(-1, -1, 2)).reshape(B, N, K, 2)
        vel_neighbors = velocities_f32.gather(1, idx_flat.expand(-1, -1, 2)).reshape(B, N, K, 2)

        dx = pos_neighbors - positions_f32.unsqueeze(2)   # [B, N, K, 2]
        du = vel_neighbors - velocities_f32.unsqueeze(2)  # [B, N, K, 2]

        # Normal equations: (dx^T dx) G = dx^T du
        AtA = torch.einsum('bnki,bnkj->bnij', dx, dx)  # [B, N, 2, 2]
        Atb = torch.einsum('bnki,bnkj->bnij', dx, du)  # [B, N, 2, 2]

        # Closed-form 2x2 inversion with Tikhonov regularization that scales
        # with the matrix trace. Relative reg ε·trace dominates only when AtA
        # is near-singular; floor at 1e-8 keeps fully-degenerate AtA invertible.
        a = AtA[..., 0, 0]
        b = AtA[..., 0, 1]
        d = AtA[..., 1, 1]
        eps = (1e-4 * (a + d)).clamp(min=1e-8)
        a_reg = a + eps
        d_reg = d + eps
        det = a_reg * d_reg - b * b
        inv00 = d_reg / det
        inv01 = -b / det
        inv11 = a_reg / det
        # We only need trace(G) = ∂Ux/∂x + ∂Uy/∂y
        div = (
            inv00 * Atb[..., 0, 0]
            + inv01 * Atb[..., 1, 0]
            + inv01 * Atb[..., 0, 1]  # symmetric: inv10 == inv01
            + inv11 * Atb[..., 1, 1]
        )
    return div


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

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

    def forward(self, fx, mask=None):
        fx = self.attn(self.ln_1(fx), mask=mask) + fx
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
        mask = data.get("mask")  # [B, N] bool or None
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, mask=mask)
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
    # Divergence-free auxiliary loss: λ_div * mean(|∇·u|_phys) on interior volume nodes.
    # KNN least-squares gradient with K neighbors; velocities denormalized via y_std[:2]
    # before divergence so that ∇·u=0 is the correct physical constraint.
    lambda_div: float = 0.1
    knn_k: int = 8


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

# Precompute KNN indices for the divergence-free auxiliary loss. Geometry is
# fixed per sample, so we cache [N_i, K] long tensors once. Parallel cKDTree
# builds across worker threads to amortize startup cost.
knn_t0 = time.time()
train_knn_cache = build_knn_cache(train_ds, k=cfg.knn_k, max_workers=8)
print(f"KNN cache built for {len(train_knn_cache)} train samples in {time.time()-knn_t0:.1f}s")
indexed_train_ds = IndexedDataset(train_ds, train_knn_cache)

train_loader_kwargs = dict(collate_fn=indexed_pad_collate, num_workers=4,
                           pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(indexed_train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **train_loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(indexed_train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **train_loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **val_loader_kwargs)
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
model = torch.compile(model, dynamic=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

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

    for x, y, is_surface, mask, knn_idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        knn_idx = knn_idx.to(device, non_blocking=True)

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
            primary_loss = vol_loss + cfg.surf_weight * surf_loss

            # Divergence-free auxiliary loss on interior volume nodes.
            # PR-specified: use NORMALIZED velocities so the λ_div=0.1 calibration
            # (assumes ~1e-3 typical |∇·u|) holds. Multiplying pred by y_std[:2]
            # — as the previous revision did — pushed |∇·u| up to ~30, making the
            # weighted aux loss ~30,000x stronger than intended. Since y_std is a
            # constant scalar, ∇·u_norm = 0 ⇔ ∇·u_phys = 0; only the magnitude
            # differs, so calibration matters for λ.
            positions_phys = x[..., :2]  # input dims 0-1 are physical (x, z)
            vel_for_div = pred[..., :2]
            div = compute_velocity_divergence(positions_phys, vel_for_div, knn_idx)
            vol_mask_f = vol_mask.float()
            div_abs = div.abs() * vol_mask_f
            div_count = vol_mask_f.sum().clamp(min=1)
            div_loss_unweighted = div_abs.sum() / div_count
            div_loss = cfg.lambda_div * div_loss_unweighted
            loss = primary_loss + div_loss

            # Diagnostic: signed residual mean for pressure on surface and volume
            # nodes. If pinball τ=0.55 is shifting the bias as hypothesized, surf
            # signed residual should drift toward small negative (over-prediction).
            # Critical check given Lion's sign() update can wash out pinball's
            # magnitude asymmetry — bias shift confirms the mechanism is alive.
            with torch.no_grad():
                surf_mask_f = surf_mask.float()
                p_signed_vol = (
                    (residual_p * vol_mask_f).sum() / vol_mask_f.sum().clamp(min=1)
                )
                p_signed_surf = (
                    (residual_p * surf_mask_f).sum() / surf_mask_f.sum().clamp(min=1)
                )
                div_mean_abs = (div_abs.sum() / div_count).detach()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/primary_loss": primary_loss.item(),
            "train/div_mean_abs": div_mean_abs.item(),
            "train/div_loss_unweighted": div_loss_unweighted.item(),
            "train/div_loss_weighted": div_loss.item(),
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
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
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

with torch.no_grad():
    param_l2_final = torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())).item()
print(f"Param L2 norm at final epoch: {param_l2_final:.4f}")
wandb.summary["param_l2_final_epoch"] = param_l2_final

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
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **val_loader_kwargs)
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
