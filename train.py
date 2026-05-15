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

# Domain identifiers used by per-domain target normalization.
DOMAIN_NAMES = ["racecar_single", "racecar_tandem", "cruise"]
DOMAIN_TO_ID = {name: i for i, name in enumerate(DOMAIN_NAMES)}
N_DOMAINS = len(DOMAIN_NAMES)

# ---------------------------------------------------------------------------
# Per-domain target normalization helpers
# ---------------------------------------------------------------------------

# Local data manifest (read-only). Used to map each split's file index to a
# physical domain (racecar_single / racecar_tandem / cruise) for per-domain
# target normalization. Avoids any reliance on a flaky x-feature heuristic.
_MANIFEST_PATH = Path(__file__).parent / "data" / "split_manifest.json"


def _global_idx_to_domain(global_idx: int, file_sizes: list[int], pickle_files: list[str]) -> int:
    """Map a global pickle index to a domain id.

    `pickle_files[i]` is the raw source pickle whose name encodes the domain
    (raceCar single vs raceCar tandem vs cruise). The pickle index spans
    cumulative offsets of `file_sizes`.
    """
    offset = 0
    for i, sz in enumerate(file_sizes):
        if offset <= global_idx < offset + sz:
            name = pickle_files[i]
            if "single" in name:
                return DOMAIN_TO_ID["racecar_single"]
            if "raceCar" in name:
                return DOMAIN_TO_ID["racecar_tandem"]
            if "cruise" in name:
                return DOMAIN_TO_ID["cruise"]
            raise ValueError(f"Unrecognized pickle name: {name}")
        offset += sz
    raise IndexError(f"global_idx {global_idx} out of range")


def build_split_domain_labels(split_name: str) -> list[int]:
    """Return a list of domain ids, one per sample in the split's .pt directory.

    File order in the split directory matches the order of indices in
    `split_manifest.json::splits[split_name]` because `prepare_splits.py`
    writes files named `seq_idx:06d.pt` for each entry. Test splits are
    shuffled by `np.random.default_rng(123)` in prepare_splits, so the same
    shuffle is replayed here for test splits.
    """
    import numpy as np

    with open(_MANIFEST_PATH) as f:
        manifest = json.load(f)
    pickle_files = manifest["pickle_files"]
    file_sizes = manifest["file_sizes"]
    g_indices = list(manifest["splits"][split_name])
    if split_name.startswith("test_"):
        np.random.default_rng(123).shuffle(g_indices)
    return [_global_idx_to_domain(g, file_sizes, pickle_files) for g in g_indices]


class DomainTaggedDataset(torch.utils.data.Dataset):
    """Thin wrapper that yields ``(x, y, is_surface, domain_id)`` per sample."""

    def __init__(self, base_ds, domain_labels: list[int]):
        assert len(base_ds) == len(domain_labels), (
            f"len(base_ds)={len(base_ds)} != len(domain_labels)={len(domain_labels)}"
        )
        self.base_ds = base_ds
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, sf = self.base_ds[idx]
        return x, y, sf, self.domain_labels[idx]


def pad_collate_with_domain(batch):
    """Pad variable-length samples + collect per-sample domain id."""
    xs, ys, surfs, doms = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    domain_t = torch.tensor(doms, dtype=torch.long)
    return x_pad, y_pad, surf_pad, mask, domain_t


def compute_per_domain_stats(
    train_base_ds,
    train_domain_labels: list[int],
    per_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (domain_means, domain_stds) each shaped [N_DOMAINS, 3].

    Stats are computed on **un-normalized** training y, over real (non-padded)
    nodes, separately for each domain. With ``per_channel=True`` mean/std are
    per channel; otherwise mean is per channel but std is a single scalar per
    domain (mean of per-channel stds), broadcast across the 3 channels.
    """
    sum_y = torch.zeros(N_DOMAINS, 3, dtype=torch.float64)
    n_nodes = torch.zeros(N_DOMAINS, dtype=torch.float64)
    for idx in range(len(train_base_ds)):
        _, y, _ = train_base_ds[idx]
        d = train_domain_labels[idx]
        sum_y[d] += y.double().sum(0)
        n_nodes[d] += y.shape[0]
    mean_y = sum_y / n_nodes.unsqueeze(-1)  # [N_DOMAINS, 3]

    sq_y = torch.zeros(N_DOMAINS, 3, dtype=torch.float64)
    for idx in range(len(train_base_ds)):
        _, y, _ = train_base_ds[idx]
        d = train_domain_labels[idx]
        sq_y[d] += ((y.double() - mean_y[d]) ** 2).sum(0)
    std_y = (sq_y / (n_nodes.unsqueeze(-1) - 1)).sqrt().clamp(min=1e-6)  # [N_DOMAINS, 3]

    if not per_channel:
        # Arm A: scalar std per domain (mean of per-channel stds), broadcast.
        scalar_std = std_y.mean(dim=-1, keepdim=True).expand_as(std_y)
        std_y = scalar_std.clone()

    return mean_y.float(), std_y.float()


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


class FiLMConditioner(nn.Module):
    """Predicts per-block FiLM scale and shift from global flow condition."""

    def __init__(self, cond_dim: int, n_hidden: int, n_blocks: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2 * n_hidden * n_blocks),
        )
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden

    def forward(self, cond: torch.Tensor):
        # cond: [B, cond_dim]
        out = self.net(cond)  # [B, 2 * n_hidden * n_blocks]
        out = out.view(cond.shape[0], self.n_blocks, 2, self.n_hidden)
        scale = out[:, :, 0, :]  # [B, n_blocks, n_hidden]
        shift = out[:, :, 1, :]  # [B, n_blocks, n_hidden]
        return scale, shift


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

    def forward(self, fx, film_scale=None, film_shift=None):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if film_scale is not None and film_shift is not None:
            fx = fx * (1.0 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


COND_DIMS = slice(13, 24)  # dims 13-23: log Re, AoAs, NACAs, gap, stagger (11 dims, constant per sample)


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
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        cond_dim = COND_DIMS.stop - COND_DIMS.start
        self.film = FiLMConditioner(cond_dim=cond_dim, n_hidden=n_hidden, n_blocks=n_layers)
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
        # Dims 13-23 are constant per sample. pad_collate puts padding at the END,
        # so node 0 is always a real node. Read the condition there to avoid
        # mean-with-zero bias from padded positions.
        cond = x[:, 0, COND_DIMS]  # [B, cond_dim]
        film_scale, film_shift = self.film(cond)  # each [B, n_blocks, n_hidden]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for i, block in enumerate(self.blocks):
            fx = block(fx, film_scale=film_scale[:, i, :], film_shift=film_shift[:, i, :])
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Per-sample scale (for scale-invariant loss)
# ---------------------------------------------------------------------------

def per_sample_field_scale(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample scalar field scale: mean over channels of per-channel std.

    Computed only over real (non-padded) nodes. ``y`` is the globally-normalized
    target so the per-sample std ratio across samples is preserved.

    Args:
        y: [B, N, 3] — normalized target.
        mask: [B, N] — True for real nodes.
    Returns:
        scale: [B] — per-sample scalar scale, clamped to avoid division blowups.
    """
    mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
    n = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
    mean = (y * mask_f).sum(dim=1, keepdim=True) / n  # [B, 1, 3]
    var = ((y - mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / n  # [B, 1, 3]
    std = var.sqrt()  # [B, 1, 3]
    scale = std.mean(dim=-1).squeeze(-1).squeeze(-1)  # [B]
    return scale.clamp(min=1e-4)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update of ``ema_model`` parameters and buffers."""
    for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p_model.detach(), alpha=1.0 - decay)
    for b_ema, b_model in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b_model)


def evaluate_split(model, loader, stats, surf_weight, device,
                   domain_stats: dict | None = None) -> dict[str, float]:
    """Evaluate a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    If ``domain_stats`` is given, the loader is expected to yield batches that
    include a per-sample domain id (5-tuples). Target normalization and
    prediction un-normalization both use the sample's domain stats.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if domain_stats is not None:
                x, y, is_surface, mask, domain_ids = batch
                domain_ids = domain_ids.to(device, non_blocking=True)
            else:
                x, y, is_surface, mask = batch
                domain_ids = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Drop samples whose y is non-finite (e.g. test_geom_camber_cruise idx=20
            # has 761 NaN in the p channel). accumulate_batch tries to skip them via
            # its y_finite check, but ``err = |pred - NaN| = NaN`` then propagates
            # through ``NaN * 0 = NaN`` in the sum. Mask them out here so neither
            # loss nor MAE sees the NaN.
            y_finite_per_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)  # [B]
            sample_keep = y_finite_per_sample.view(-1, 1)
            mask = mask & sample_keep
            y_clean = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            if domain_stats is not None:
                y_mean_per = domain_stats["mean"][domain_ids].unsqueeze(1)
                y_std_per = domain_stats["std"][domain_ids].unsqueeze(1)
                y_norm = (y_clean - y_mean_per) / y_std_per
            else:
                y_norm = (y_clean - stats["y_mean"]) / stats["y_std"]
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

            if domain_stats is not None:
                pred_orig = pred * y_std_per + y_mean_per
            else:
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y_clean, is_surface, mask, mae_surf, mae_vol)
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
        "surf_p_l1_weight": cfg.surf_p_l1_weight,
        "epochs_configured": cfg.epochs,
        "ema_decay": EMA_DECAY,
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
# EMA decay for weight averaging used by validation/checkpointing/test eval.
# Why: with batch_size=4 and ~375 steps/epoch, 0.999 gives an averaging window
# of ~1000 steps ≈ 2.7 epochs — appropriate for the wall-clock-capped 14-epoch runs.
EMA_DECAY = 0.999


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    # Auxiliary L1 loss on surface-node pressure (normalized space). Aligns the
    # training objective with the L1 (MAE) eval metric. 0 disables the term.
    surf_p_l1_weight: float = 0.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    # Per-domain target normalization. When enabled, y is normalized per sample
    # using stats from its physical domain (racecar_single / racecar_tandem /
    # cruise) rather than global stats. ``per_channel_norm`` controls whether
    # the std is per channel (True) or a scalar per domain (False).
    per_domain_norm: bool = False
    per_channel_norm: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

# Per-domain target normalization setup.
domain_stats: dict[str, torch.Tensor] | None = None
if cfg.per_domain_norm:
    train_domain_labels = build_split_domain_labels("train")
    val_domain_labels = {name: build_split_domain_labels(name) for name in VAL_SPLIT_NAMES}
    if cfg.debug:
        # load_data() truncates train_ds.files / val ds.files in debug mode.
        train_domain_labels = train_domain_labels[: len(train_ds)]
        for name, ds in val_splits.items():
            val_domain_labels[name] = val_domain_labels[name][: len(ds)]

    print("Computing per-domain target stats on training data...")
    domain_means, domain_stds = compute_per_domain_stats(
        train_ds, train_domain_labels, per_channel=cfg.per_channel_norm
    )
    domain_stats = {
        "mean": domain_means.to(device),  # [N_DOMAINS, 3]
        "std": domain_stds.to(device),    # [N_DOMAINS, 3]
    }
    for d_id, name in enumerate(DOMAIN_NAMES):
        m = domain_means[d_id].tolist()
        s = domain_stds[d_id].tolist()
        n_train = sum(1 for d in train_domain_labels if d == d_id)
        print(f"  {name:<18s} n_train={n_train:4d}  "
              f"mean=[{m[0]:7.2f},{m[1]:7.2f},{m[2]:8.2f}]  "
              f"std=[{s[0]:7.2f},{s[1]:7.2f},{s[2]:8.2f}]")

    train_ds = DomainTaggedDataset(train_ds, train_domain_labels)
    val_splits = {
        name: DomainTaggedDataset(ds, val_domain_labels[name])
        for name, ds in val_splits.items()
    }
    collate_fn = pad_collate_with_domain
else:
    collate_fn = pad_collate

loader_kwargs = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True,
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

ema_model = copy.deepcopy(model)
ema_model.requires_grad_(False)
ema_model.eval()
print(f"EMA model initialized (decay={EMA_DECAY})")

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
        "ema_decay": EMA_DECAY,
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
    epoch_surf_p_l1 = 0.0
    epoch_scale_mean = epoch_scale_std = 0.0
    n_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        if cfg.per_domain_norm:
            x, y, is_surface, mask, domain_ids = batch
            domain_ids = domain_ids.to(device, non_blocking=True)
        else:
            x, y, is_surface, mask = batch
            domain_ids = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        if cfg.per_domain_norm:
            # Per-sample target normalization using the sample's domain stats.
            y_mean_per = domain_stats["mean"][domain_ids].unsqueeze(1)  # [B, 1, 3]
            y_std_per = domain_stats["std"][domain_ids].unsqueeze(1)    # [B, 1, 3]
            y_norm = (y - y_mean_per) / y_std_per
        else:
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        sq_err = (pred - y_norm) ** 2  # [B, N, 3]

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_mask_f = vol_mask.float().unsqueeze(-1)  # [B, N, 1]
        surf_mask_f = surf_mask.float().unsqueeze(-1)  # [B, N, 1]

        n_vol = vol_mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
        n_surf = surf_mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
        vol_loss_per_sample = (sq_err * vol_mask_f).sum(dim=1) / n_vol  # [B, 3]
        surf_loss_per_sample = (sq_err * surf_mask_f).sum(dim=1) / n_surf  # [B, 3]
        vol_loss_per_sample = vol_loss_per_sample.mean(dim=-1)  # [B]
        surf_loss_per_sample = surf_loss_per_sample.mean(dim=-1)  # [B]

        combined_per_sample = vol_loss_per_sample + cfg.surf_weight * surf_loss_per_sample  # [B]
        sample_scale = per_sample_field_scale(y_norm, mask)  # [B]
        loss = (combined_per_sample / sample_scale.detach()).mean()

        # Auxiliary L1 loss on surface-node pressure (normalized space). Computed
        # as a pooled mean over all surface nodes in the batch — matches the
        # eval-side aggregation more directly than the per-sample-then-average
        # scheme used by the MSE term.
        surf_mask_l1 = (mask & is_surface).float()  # [B, N]
        p_err_abs = (pred[..., 2] - y_norm[..., 2]).abs()  # [B, N]
        surf_p_l1 = (p_err_abs * surf_mask_l1).sum() / surf_mask_l1.sum().clamp(min=1)
        if cfg.surf_p_l1_weight != 0.0:
            loss = loss + cfg.surf_p_l1_weight * surf_p_l1

        # Aggregate logging stats: batch-mean of per-sample vol/surf losses
        # (for parity with the previous epoch_vol/epoch_surf scalars).
        vol_loss_log = vol_loss_per_sample.mean()
        surf_loss_log = surf_loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model, EMA_DECAY)

        epoch_vol += vol_loss_log.item()
        epoch_surf += surf_loss_log.item()
        epoch_surf_p_l1 += surf_p_l1.item()
        epoch_scale_mean += sample_scale.mean().item()
        epoch_scale_std += sample_scale.std(unbiased=False).item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_surf_p_l1 /= max(n_batches, 1)
    epoch_scale_mean /= max(n_batches, 1)
    epoch_scale_std /= max(n_batches, 1)

    # --- Validate (EMA weights) ---
    ema_model.eval()
    split_metrics = {
        name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device,
                             domain_stats=domain_stats)
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
        torch.save(ema_model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/surf_p_l1": epoch_surf_p_l1,
        "train/sample_scale_mean": epoch_scale_mean,
        "train/sample_scale_std": epoch_scale_std,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f} "
        f"surf_p_l1={epoch_surf_p_l1:.4f} "
        f"scale={epoch_scale_mean:.3f}±{epoch_scale_std:.3f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
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

    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        if cfg.per_domain_norm:
            test_domain_labels = {name: build_split_domain_labels(name) for name in TEST_SPLIT_NAMES}
            if cfg.debug:
                for name, ds in test_datasets.items():
                    test_domain_labels[name] = test_domain_labels[name][: len(ds)]
            test_datasets = {
                name: DomainTaggedDataset(ds, test_domain_labels[name])
                for name, ds in test_datasets.items()
            }
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device,
                                 domain_stats=domain_stats)
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
