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
# H4: surface-distance feature + surface-only normalization helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_distance_to_surface(x: torch.Tensor, is_surface: torch.Tensor,
                                device: torch.device, chunk: int = 8192) -> torch.Tensor:
    """Per-node Euclidean distance to nearest surface node, normalized by bbox diagonal.

    Operates on a single (unpadded) sample. Returns a [N, 1] tensor on CPU.
    """
    pos = x[..., :2].to(device)
    surf_mask = is_surface.to(device)
    sp = pos[surf_mask]
    N = pos.shape[0]
    if sp.numel() == 0:
        d = torch.full((N,), 1e3, dtype=pos.dtype, device=device)
    else:
        d = torch.empty(N, dtype=pos.dtype, device=device)
        for i in range(0, N, chunk):
            d[i : i + chunk] = torch.cdist(pos[i : i + chunk], sp).min(dim=-1).values
    bbox = pos.amax(dim=0) - pos.amin(dim=0)
    diag = bbox.norm().clamp(min=1e-6)
    return (d / diag).cpu().unsqueeze(-1)


def precompute_distances(ds, device: torch.device, label: str = "split") -> list[torch.Tensor]:
    """Precompute the surface-distance feature for every sample in a dataset."""
    distances = []
    for i in tqdm(range(len(ds)), desc=f"distance:{label}", leave=False):
        item = ds[i]
        x, _, is_surface = item[0], item[1], item[2]
        distances.append(_sample_distance_to_surface(x, is_surface, device))
    return distances


class DistanceAugmentedDataset(Dataset):
    """Wrap a SplitDataset/TestDataset and append precomputed distance as the 25th feature."""

    def __init__(self, base_ds, distances: list[torch.Tensor]):
        assert len(base_ds) == len(distances)
        self.base_ds = base_ds
        self.distances = distances

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, is_surface = self.base_ds[idx]
        d = self.distances[idx]
        x_aug = torch.cat([x, d], dim=-1)
        return x_aug, y, is_surface


def compute_surface_volume_stats(ds, max_samples: int = 200) -> dict[str, torch.Tensor]:
    """Compute per-channel mean/std for surface and volume target nodes.

    Iterates over up to ``max_samples`` of ``ds`` (works on the augmented or raw dataset
    since y is unchanged) and returns ``{y_mean_surf, y_std_surf, y_mean_vol, y_std_vol}``.
    """
    sums_s = torch.zeros(3, dtype=torch.float64)
    sumsq_s = torch.zeros(3, dtype=torch.float64)
    n_s = 0
    sums_v = torch.zeros(3, dtype=torch.float64)
    sumsq_v = torch.zeros(3, dtype=torch.float64)
    n_v = 0
    n = min(len(ds), max_samples)
    for i in range(n):
        item = ds[i]
        _, y, is_surface = item[0], item[1], item[2]
        y = y.double()
        y_s = y[is_surface]
        y_v = y[~is_surface]
        if y_s.numel() > 0:
            sums_s += y_s.sum(0)
            sumsq_s += (y_s ** 2).sum(0)
            n_s += y_s.shape[0]
        if y_v.numel() > 0:
            sums_v += y_v.sum(0)
            sumsq_v += (y_v ** 2).sum(0)
            n_v += y_v.shape[0]
    y_mean_s = sums_s / max(n_s, 1)
    y_std_s = (sumsq_s / max(n_s, 1) - y_mean_s ** 2).clamp(min=1e-6).sqrt()
    y_mean_v = sums_v / max(n_v, 1)
    y_std_v = (sumsq_v / max(n_v, 1) - y_mean_v ** 2).clamp(min=1e-6).sqrt()
    return {
        "y_mean_surf": y_mean_s.float(),
        "y_std_surf": y_std_s.float(),
        "y_mean_vol": y_mean_v.float(),
        "y_std_vol": y_std_v.float(),
    }


def per_node_y_stats(is_surface: torch.Tensor, surf_stats: dict[str, torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [B, N, 3] per-node y_mean/y_std selecting between surface and volume stats."""
    m = is_surface.unsqueeze(-1).to(surf_stats["y_mean_surf"].dtype)  # [B, N, 1]
    one_minus_m = 1.0 - m
    y_mean = (m * surf_stats["y_mean_surf"].view(1, 1, -1)
              + one_minus_m * surf_stats["y_mean_vol"].view(1, 1, -1))
    y_std = (m * surf_stats["y_std_surf"].view(1, 1, -1)
             + one_minus_m * surf_stats["y_std_vol"].view(1, 1, -1))
    return y_mean, y_std


def sanitize_nonfinite_samples(y: torch.Tensor, mask: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out y and mask for samples whose ground truth has any non-finite value.

    Why: ``data/scoring.py`` computes ``err = |pred - y|`` and then masks out samples
    whose y is non-finite. But IEEE 754 has ``inf * 0 = nan`` — so a single ``-inf``
    in y (e.g. test_geom_camber_cruise sample 20 has 761 ``-inf`` in pressure from
    fp16 overflow in the source data) poisons the entire metric with NaN even though
    the per-sample skip mask is ``False`` for that sample. The same problem hits the
    normalized-space squared-error loss computation in evaluate_split / training.

    Workaround (since ``data/scoring.py`` is read-only): for any sample with a
    non-finite y, replace its y with zeros and zero out its mask. The sample then
    contributes 0 to every accumulator and is correctly excluded from node counts.
    """
    B = y.shape[0]
    y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
    if y_finite.all():
        return y, mask
    keep = y_finite.view(B, *([1] * (y.ndim - 1)))
    y_clean = torch.where(keep, y, torch.zeros_like(y))
    mask_clean = mask & y_finite.unsqueeze(-1)
    return y_clean, mask_clean


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
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 use_surface_norm=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_surface_norm = use_surface_norm and last_layer
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
            if self.use_surface_norm:
                self.mlp2_surf = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                    nn.Linear(hidden_dim, out_dim),
                )
                self.mlp2_vol = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                    nn.Linear(hidden_dim, out_dim),
                )
            else:
                self.mlp2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                    nn.Linear(hidden_dim, out_dim),
                )

    def forward(self, fx, is_surface=None):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            fx = self.ln_3(fx)
            if self.use_surface_norm:
                out_surf = self.mlp2_surf(fx)
                out_vol = self.mlp2_vol(fx)
                m = is_surface.unsqueeze(-1).to(out_surf.dtype)
                return m * out_surf + (1.0 - m) * out_vol
            return self.mlp2(fx)
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_surface_norm: bool = False):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.use_surface_norm = use_surface_norm

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
                use_surface_norm=use_surface_norm,
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
        is_surface = data.get("is_surface", None)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for i, block in enumerate(self.blocks):
            if self.use_surface_norm and i == len(self.blocks) - 1:
                fx = block(fx, is_surface=is_surface)
            else:
                fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   use_distance_feature: bool = False,
                   surf_stats: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    When ``surf_stats`` is provided, surface and volume nodes are normalized with
    their own per-channel mean/std (``y_mean_surf`` etc.) and the model is
    expected to emit the matching gated prediction (``Transolver`` with
    ``use_surface_norm=True``). Otherwise a single global ``stats`` is used.
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

            # Drop samples with non-finite y so Inf in y doesn't poison sums via inf*0=nan.
            y, mask = sanitize_nonfinite_samples(y, mask)

            if use_distance_feature:
                x_norm = torch.cat(
                    [(x[..., :X_DIM] - stats["x_mean"]) / stats["x_std"], x[..., X_DIM:]],
                    dim=-1,
                )
            else:
                x_norm = (x - stats["x_mean"]) / stats["x_std"]

            if surf_stats is not None:
                y_mean_b, y_std_b = per_node_y_stats(is_surface, surf_stats)
                y_norm = (y - y_mean_b) / y_std_b
            else:
                y_mean_b = stats["y_mean"]
                y_std_b = stats["y_std"]
                y_norm = (y - y_mean_b) / y_std_b

            pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
            # Defense: occasionally a CUDA kernel produces a transient inf/nan
            # under memory pressure. Zero those out so a single bad value does
            # not poison the entire split via inf*0 = nan.
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

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

            pred_orig = pred * y_std_b + y_mean_b
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
    # H4 — surface-distance feature + surface-only normalization.
    use_distance_feature: bool = False  # Component 1: append distance-to-surface as 25th feat
    use_surface_norm: bool = False      # Component 2: per-head surface vs volume y-normalization
    surface_stats_max_samples: int = 200  # samples for compute_surface_volume_stats


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

if cfg.use_distance_feature:
    print("Precomputing distance-to-surface feature for train and val splits...")
    t0 = time.time()
    train_distances = precompute_distances(train_ds, device, label="train")
    train_ds = DistanceAugmentedDataset(train_ds, train_distances)
    val_distances = {}
    for name, ds in val_splits.items():
        val_distances[name] = precompute_distances(ds, device, label=name)
        val_splits[name] = DistanceAugmentedDataset(ds, val_distances[name])
    print(f"  Distance precompute done in {time.time() - t0:.1f}s")

surf_stats = None
if cfg.use_surface_norm:
    print("Computing surface vs volume target normalization stats...")
    surf_stats = compute_surface_volume_stats(train_ds, max_samples=cfg.surface_stats_max_samples)
    surf_stats = {k: v.to(device) for k, v in surf_stats.items()}
    print(f"  y_mean_surf={surf_stats['y_mean_surf'].tolist()} "
          f"y_std_surf={surf_stats['y_std_surf'].tolist()}")
    print(f"  y_mean_vol ={surf_stats['y_mean_vol'].tolist()} "
          f"y_std_vol ={surf_stats['y_std_vol'].tolist()}")

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

fun_dim_extra = 1 if cfg.use_distance_feature else 0
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2 + fun_dim_extra,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    use_surface_norm=cfg.use_surface_norm,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        "h4_surface_stats": (
            {k: v.tolist() for k, v in surf_stats.items()} if surf_stats is not None else None
        ),
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

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Defensive: train data is currently clean, but the same Inf-poisoning hazard
        # would silently destroy gradients if a bad sample ever sneaks in.
        y, mask = sanitize_nonfinite_samples(y, mask)

        if cfg.use_distance_feature:
            x_norm = torch.cat(
                [(x[..., :X_DIM] - stats["x_mean"]) / stats["x_std"], x[..., X_DIM:]],
                dim=-1,
            )
        else:
            x_norm = (x - stats["x_mean"]) / stats["x_std"]

        if surf_stats is not None:
            y_mean_b, y_std_b = per_node_y_stats(is_surface, surf_stats)
            y_norm = (y - y_mean_b) / y_std_b
        else:
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

        pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
        sq_err = (pred - y_norm) ** 2

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

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
            use_distance_feature=cfg.use_distance_feature,
            surf_stats=surf_stats,
        )
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
        if cfg.use_distance_feature:
            print("Precomputing distance-to-surface feature for test splits...")
            for name, ds in test_datasets.items():
                test_distances = precompute_distances(ds, device, label=name)
                test_datasets[name] = DistanceAugmentedDataset(ds, test_distances)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(
                model, loader, stats, cfg.surf_weight, device,
                use_distance_feature=cfg.use_distance_feature,
                surf_stats=surf_stats,
            )
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
