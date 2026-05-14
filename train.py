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

import contextlib
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


class DropPath(nn.Module):
    """Stochastic Depth / DropPath (Huang et al. 2016 "Deep Networks with Stochastic Depth";
    Larsson et al. 2016 "FractalNet"). Per-sample drop of an entire residual branch
    output at training time; identity at eval. Scale-by-1/keep_prob preserves expectation.

    Built-in counters track empirical drop fraction since last reset, so we can
    sanity-check that the realized rate matches the nominal p across an epoch
    (rate diagnostics are required by PR #3065).
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        assert 0.0 <= p < 1.0, f"drop_path p must be in [0, 1), got {p}"
        self.p = float(p)
        self.register_buffer("_n_branch_total", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_n_branch_dropped", torch.zeros(1, dtype=torch.long), persistent=False)

    def reset_counters(self):
        self._n_branch_total.zero_()
        self._n_branch_dropped.zero_()

    @torch.no_grad()
    def empirical_drop_fraction(self) -> float:
        total = int(self._n_branch_total.item())
        dropped = int(self._n_branch_dropped.item())
        return dropped / total if total > 0 else 0.0

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # Per-sample mask: [B, 1, 1, ...] broadcastable with x ([B, N, D])
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        # Tensor-only counter updates (no .item() to avoid torch.compile graph breaks).
        # Read with .item() only outside the compiled forward (end-of-epoch diagnostic).
        with torch.no_grad():
            self._n_branch_total.add_(mask.numel())
            self._n_branch_dropped.add_((mask == 0).sum().long())
        return x * mask / keep_prob


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP (Shazeer 2020) — out = W_out (silu(W_gate x) * W_value x).

    Param-matched vs. MLP(d, d*mlp_ratio, d): hidden = round(d*mlp_ratio*2/3)
    snapped to a multiple of 8 for tensor-core alignment. 3 matrices of
    d*hidden_swiglu vs. 2 matrices of d*hidden_full keeps Linear param count
    near-identical (within ~0.2% at d=96, mlp_ratio=2).
    """
    def __init__(self, n_input, n_hidden_full, n_output, act_fn=F.silu):
        super().__init__()
        hidden_swiglu = max(8, int(round((n_hidden_full * 2 / 3) / 8) * 8))
        self.hidden_swiglu = hidden_swiglu
        self.linear_gate = nn.Linear(n_input, hidden_swiglu)
        self.linear_value = nn.Linear(n_input, hidden_swiglu)
        self.linear_out = nn.Linear(hidden_swiglu, n_output)
        self.act_fn = act_fn

    def forward(self, x):
        gate = self.act_fn(self.linear_gate(x))
        value = self.linear_value(x)
        return self.linear_out(gate * value)


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


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation channel gate with content-aware attention pool.

    Replaces SE's standard mean-pool with a learned single-head attention pool
    (Lin et al. 2017 "Structured Self-Attentive Sentence Embedding"; Lee et al.
    2019 "Set Transformer"). attn_pool: Linear(C, 1) -> softmax over tokens
    (masked) -> weighted mean -> bottleneck MLP -> sigmoid gate -> broadcast.

    At step 0, attn_pool uses trunc_normal_(std=0.02) init (applied by Transolver
    _init_weights); logit std ~ 0.02 * sqrt(C) * std(x) yields softmax nearly
    uniform over T~100K tokens, i.e. approximately mean pool. Optimizer can then
    sharpen the pool toward informative tokens (surface, wake, boundary).

    Zero-init fc2 so gate = sigmoid(0) = 0.5 uniformly at step 0.
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        d_hidden = max(1, dim // reduction)
        self.attn_pool = nn.Linear(dim, 1, bias=True)
        self.fc1 = nn.Linear(dim, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, dim, bias=True)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x, mask=None):
        # x: (B, N, D); mask: (B, N) bool, True for real nodes
        attn_logits = self.attn_pool(x)  # (B, N, 1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, N, 1)
        s = (x * attn_weights).sum(dim=1)  # (B, D)
        s = self.fc2(F.gelu(self.fc1(s)))
        gate = torch.sigmoid(s)
        return x * gate.unsqueeze(1)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 layerscale_init=1e-4, use_se=True, drop_path=0.0):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        # POST-NORM topology + γ=1.0 constant (Vaswani 2017 recipe). LayerScale
        # removed: γ is a plain Python float, not nn.Parameter — no learnable
        # gain on residual branches. Saves 2×hidden_dim params per block.
        self.gamma_attn = 1.0
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, act_fn=F.silu)
        self.gamma_mlp = 1.0
        # Stochastic Depth / DropPath on each residual branch (#3065).
        # Per-sample drop of attn / mlp branch output at training time; identity at eval.
        self.drop_path_attn = DropPath(p=drop_path)
        self.drop_path_mlp = DropPath(p=drop_path)
        self.se = SqueezeExcitation(hidden_dim, reduction=4) if use_se else None
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, mask=None):
        # POST-NORM: LN applied AFTER residual addition (LN(x + f(x)) vs pre-norm x + f(LN(x)))
        # DropPath wraps each residual branch's contribution (#3065): drop the
        # entire branch with prob p, otherwise scale by 1/(1-p) to preserve mean.
        fx = self.ln_1(self.drop_path_attn(self.gamma_attn * self.attn(fx)) + fx)
        fx = self.ln_2(self.drop_path_mlp(self.gamma_mlp * self.mlp(fx)) + fx)
        if self.se is not None:
            fx = self.se(fx, mask=mask)
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 drop_path: float = 0.0):
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
                use_se=(i == n_layers - 1),
                drop_path=drop_path,
            )
            for i in range(n_layers)
        ])
        # Cross-block residual α: learnable scalar gain applied to each non-final
        # block's output BEFORE the next block consumes it. n_layers-1 scalars
        # (no α after the last block, whose output is already the head prediction).
        # Init 1.0 -> identity at step 0. Scales AFTER the block's post-norm LN
        # so non-1.0 α directly changes feature magnitudes flowing into the next
        # block (LN cannot absorb the scale).
        self.cross_block_alpha = nn.Parameter(torch.ones(n_layers - 1))
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        # Feature-stream FiLM gate — zero-init both weight AND bias for identity at step 0.
        # Applied AFTER _init_weights so it overrides the trunc_normal_ init.
        # Channels [13, 14, 18] = [log_Re, AoA0_rad, AoA1_rad] (per #2531 instrumentation).
        self.film = nn.Linear(3, n_hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

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
        mask = data.get("mask")  # [B, N] bool, True for real nodes; threaded to SE pool
        # Per-sample flow scalars (constant across N) — channels [13, 14, 18]
        flow_scalars = x[:, 0, [13, 14, 18]]                       # [B, 3]
        film_scale = self.film(flow_scalars).unsqueeze(1)          # [B, 1, n_hidden]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        # Feature-stream FiLM: zero-init -> identity at step 0
        fx = fx * (1 + film_scale)
        n_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            fx = block(fx, mask=mask)
            # Apply cross-block α to non-final block outputs; the last block is
            # the head (returns out_dim predictions) and must not be scaled.
            if i < n_blocks - 1:
                fx = self.cross_block_alpha[i] * fx
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, amp_ctx_factory) -> dict[str, float]:
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

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            with amp_ctx_factory():
                pred = model({"x": x_norm, "mask": mask})["preds"]
            pred = pred.float()

            sq_err = F.l1_loss(pred, y_norm, reduction='none')
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

            # --- scoring-bug workaround: filter non-finite ground-truth samples
            # *before* accumulate_batch, so 0 * Inf = NaN cannot poison the sums.
            y_finite_mask = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if y_finite_mask.any():
                ds, dv = accumulate_batch(
                    pred_orig[y_finite_mask],
                    y[y_finite_mask],
                    is_surface[y_finite_mask],
                    mask[y_finite_mask],
                    mae_surf,
                    mae_vol,
                )
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
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    drop_path: float = 0.05  # stochastic-depth p per residual branch (#3065)


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

SINGLE_DOMAIN_BOOST = 2.0

if not cfg.debug:
    with open(Path(cfg.splits_dir) / "meta.json") as f:
        _meta_for_sampler = json.load(f)
    _domain_groups = _meta_for_sampler["domain_groups"]
    print(f"[sampler-reweight] domain keys: {list(_domain_groups.keys())}")
    print(f"[sampler-reweight] sizes: " + str({k: len(v) for k, v in _domain_groups.items()}))

    single_domain_key = None
    for k in _domain_groups.keys():
        kl = k.lower()
        if "single" in kl and "tandem" not in kl:
            single_domain_key = k
            break
    assert single_domain_key is not None, (
        f"Could not find 'RaceCar single' domain among {list(_domain_groups.keys())}"
    )
    print(f"[sampler-reweight] boosting domain '{single_domain_key}' by {SINGLE_DOMAIN_BOOST}x")

    boosted = sample_weights.clone()
    for idx in _domain_groups[single_domain_key]:
        boosted[idx] = boosted[idx] * SINGLE_DOMAIN_BOOST

    print(f"[sampler-reweight] pre-boost domain mass: "
          + ", ".join(f"{k}={sample_weights[v].sum().item():.4f}" for k, v in _domain_groups.items()))
    print(f"[sampler-reweight] post-boost domain mass: "
          + ", ".join(f"{k}={boosted[v].sum().item():.4f}" for k, v in _domain_groups.items()))

    sample_weights = boosted

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
    n_hidden=96,
    n_layers=4,
    n_head=2,
    slice_num=24,
    mlp_ratio=3,
    drop_path=cfg.drop_path,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
print(f"slice_num: {model_config['slice_num']}")
print(f"slice_num: 24 (down from 32, -25% slicing ops/block) — budget-freeing PhysicsAttention granularity probe; 3rd orthogonal budget-bound axis after n_layers (#2268) and n_hidden (#2290)")
print(f"n_head: {model_config['n_head']} (dim_head={model_config['n_hidden'] // model_config['n_head']})")
print(f"Depth: n_layers=4 (TransolverBlock x 4) — depth-down probe, budget-bound vs capacity-saturated diagnostic")
print(f"Width: n_hidden=96 (hidden_dim=96, down from 128) — budget-freeing width-down probe; ~40-45% per-epoch wall-clock savings")

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")
print(
    f"Block topology: POST-NORM (LN AFTER residual sum) at all {model_config['n_layers']}*2={model_config['n_layers']*2} per-block LN sites; "
    f"final ln_3 retained in last block for output-head cleanup; "
    f"forward: ln(gamma * f(x) + x) vs baseline pre-norm x + gamma * f(ln(x)); "
    f"vs #2951 post-norm + LayerScale γ=1e-4: val 35.0624 (+14.73% LOSS), cruise +25.03% (worst). "
    f"This PR isolates topology from LayerScale init."
)
print(
    f"LayerScale: REMOVED — γ_attn = γ_mlp = 1.0 constant (Python float, NOT nn.Parameter); "
    f"-{2 * model_config['n_layers'] * model_config['n_hidden']} params removed (was 8×{model_config['n_hidden']}=768 LayerScale γ params); "
    f"restores Vaswani 2017 original Transformer recipe: standard residual without learnable per-channel gain on residual branch."
)
print(
    f"Feature-stream FiLM gate: zero-init Linear(3, {model_config['n_hidden']}) applied before block 0; "
    f"channels [13, 14, 18] = [log_Re, AoA0_rad, AoA1_rad] (per #2531); "
    f"residual-stream conditioning vs output-side (closed #2531+#2588); "
    f"+~{3 * model_config['n_hidden'] + model_config['n_hidden']} params; "
    f"baseline to beat: val_avg/mae_surf_p < 33.4935"
)
print(f"Actual total params: {n_params}")

# DropPath / Stochastic Depth diagnostic (#3065): p per residual branch
_drop_path_mods = [m for m in model.modules() if isinstance(m, DropPath)]
_drop_path_p = _drop_path_mods[0].p if _drop_path_mods else 0.0
print(
    f"DropPath (Huang et al. 2016): p={_drop_path_p} on every residual branch "
    f"(attention + MLP per block); n_layers={model_config['n_layers']} × 2 branches = "
    f"{len(_drop_path_mods)} DropPath modules total; +0 params; "
    f"per-sample mask, scale-by-1/(1-p) preserves expectation; "
    f"identity at eval (deterministic forward); "
    f"NEW BASELINE to beat: val_avg/mae_surf_p < 29.5318 (PR #3006); test 25.4795"
)

# Cross-block residual α diagnostic: between-block adaptivity at a fresh axis
# (in-block γ closed at all granularities by #2940/#2964/#2977/#2988).
# α scales each non-final block's output AFTER its post-norm LN — non-1.0
# values directly modulate feature magnitudes into the next block. Last
# block's head output is unscaled (no following block).
_inner_for_alpha = getattr(model, "_orig_mod", model)
_alpha_init = _inner_for_alpha.cross_block_alpha.detach().cpu().tolist()
print(
    f"Cross-block residual α (#3006): {_inner_for_alpha.cross_block_alpha.shape[0]} learnable scalars "
    f"(+{_inner_for_alpha.cross_block_alpha.numel()} params) applied between adjacent TransolverBlocks; "
    f"init={[f'{a:.4f}' for a in _alpha_init]} (all 1.0 — identity at step 0); "
    f"in-block γ_attn / γ_mlp UNCHANGED (constant 1.0 from #2964); "
    f"baseline to beat: val_avg/mae_surf_p < 30.0382 (#2964 NEW baseline)"
)

# SE diagnostic: count modules and added params
_se_modules = [m for m in model.modules() if isinstance(m, SqueezeExcitation)]
_n_se = len(_se_modules)
_n_se_params = sum(p.numel() for m in _se_modules for p in m.parameters())
print(
    f"Squeeze-Excitation with ATTENTION POOL (block 3 only, deepest): added {_n_se} SE modules, +{_n_se_params} params (reduction=4); "
    f"attention pool: Linear({model_config['n_hidden']},1) -> softmax over tokens (masked); "
    f"fc1({model_config['n_hidden']}->{model_config['n_hidden']//4}) -> GELU -> "
    f"fc2({model_config['n_hidden']//4}->{model_config['n_hidden']}) -> sigmoid -> broadcast multiply; "
    f"applied at END of TransolverBlock {model_config['n_layers']-1} only (blocks 0..{model_config['n_layers']-2} carry no SE); "
    f"zero-init fc2 -> gate=0.5 uniform at step 0; "
    f"attn_pool trunc_normal_ init (std=0.02) -> softmax over T~100K ~ uniform at step 0 (~mean pool baseline); "
    f"baseline to beat: val_avg/mae_surf_p < 31.3216 (SE r=4 mean pool #2765)"
)

# SwiGLU diagnostic: hidden width and per-block MLP param count vs standard MLP
_swiglu_modules = [m for m in model.modules() if isinstance(m, SwiGLUMLP)]
_swiglu_hidden = _swiglu_modules[0].hidden_swiglu if _swiglu_modules else 0
_swiglu_params = sum(p.numel() for m in _swiglu_modules for p in m.parameters())
_std_mlp_hidden = model_config['n_hidden'] * model_config['mlp_ratio']
_std_mlp_params_per_block = (
    model_config['n_hidden'] * _std_mlp_hidden + _std_mlp_hidden  # linear_pre
    + _std_mlp_hidden * model_config['n_hidden'] + model_config['n_hidden']  # linear_post
)
_std_mlp_params_total = _std_mlp_params_per_block * len(_swiglu_modules)
print(
    f"SwiGLU MLP (Shazeer 2020): replaced GELU-MLP in {len(_swiglu_modules)} TransolverBlocks; "
    f"hidden_swiglu={_swiglu_hidden} (param-matched: round(d*mlp_ratio*2/3)/8 from full hidden {_std_mlp_hidden}); "
    f"act=SiLU; total SwiGLU params={_swiglu_params} vs standard-MLP params={_std_mlp_params_total} "
    f"(delta={_swiglu_params - _std_mlp_params_total:+d}, {(_swiglu_params - _std_mlp_params_total) * 100.0 / max(_std_mlp_params_total, 1):+.2f}%); "
    f"baseline to beat: val_avg/mae_surf_p < 33.0195"
)

# torch.compile with dynamic=True because pad_collate yields batches with
# variable N_max (longest mesh in batch varies). Without dynamic, compile
# would retrace on every new shape.
compile_active = False
compile_error: str | None = None
try:
    model = torch.compile(model, dynamic=True)
    compile_active = True
    print("torch.compile: enabled (dynamic=True)")
except Exception as e:
    compile_error = repr(e)
    print(f"torch.compile: skipped ({compile_error})")


def amp_ctx_factory():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


print(f"AMP: {'bfloat16' if torch.cuda.is_available() else 'disabled (no CUDA)'}")


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023): sign-based momentum updates.

    Update: theta <- theta - lr * (sign(beta1 * m + (1-beta1) * g) + wd * theta)
    Momentum: m <- beta2 * m + (1-beta2) * g
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss


optimizer = Lion(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.99),
)
print(f"Optimizer: Lion (Chen et al. 2023) | lr={cfg.lr}, wd={cfg.weight_decay}, betas=(0.9, 0.99) | sign-based momentum update | replaces AdamW")
print(f"Lion LR sweep: lr={cfg.lr} (1.5x the #2524 baseline lr=1e-4); wd=3e-4, betas=(0.9, 0.99); new baseline to beat: val_avg/mae_surf_p < 36.3994")
warmup_epochs = 3
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(MAX_EPOCHS - warmup_epochs, 1)
        ),
    ],
    milestones=[warmup_epochs],
)
print(f"Scheduler: LinearLR(0.1->1.0 over {warmup_epochs} epochs) -> CosineAnnealingLR(T_max={max(MAX_EPOCHS - warmup_epochs, 1)})")
print(f"LR check ep0: {optimizer.param_groups[0]['lr']:.6f} (expect {0.1 * cfg.lr:.6f})")

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

def capture_residual_magnitudes(model_obj, loader, stats_, amp_ctx, device_, epoch_tag):
    """Probe pre/post-LN residual magnitudes per block on one val batch.

    Logs |x| (RMS) before and after each LN site to distinguish post-norm
    (capped by LN at every block) from pre-norm (residual stream grows with
    depth). Run at ep1 + best-checkpoint for the topology-vs-LayerScale
    attribution analysis required by the PR.
    """
    inner = getattr(model_obj, "_orig_mod", model_obj)
    captured: list[dict] = []

    def make_hook(block_idx, ln_name):
        def hook(_module, args, output):
            with torch.no_grad():
                x_in = args[0]
                pre_rms = x_in.detach().float().pow(2).mean().sqrt().item()
                post_rms = output.detach().float().pow(2).mean().sqrt().item()
                pre_max = x_in.detach().float().abs().max().item()
                post_max = output.detach().float().abs().max().item()
                captured.append({
                    "block_idx": block_idx,
                    "ln_name": ln_name,
                    "pre_ln_rms": pre_rms,
                    "post_ln_rms": post_rms,
                    "pre_ln_abs_max": pre_max,
                    "post_ln_abs_max": post_max,
                })
        return hook

    handles = []
    for bi, blk in enumerate(inner.blocks):
        handles.append(blk.ln_1.register_forward_hook(make_hook(bi, "ln_1")))
        handles.append(blk.ln_2.register_forward_hook(make_hook(bi, "ln_2")))
        if getattr(blk, "last_layer", False) and hasattr(blk, "ln_3"):
            handles.append(blk.ln_3.register_forward_hook(make_hook(bi, "ln_3")))

    model_obj.eval()
    with torch.no_grad():
        x_b, _, _, mask_b = next(iter(loader))
        x_b = x_b.to(device_, non_blocking=True)
        mask_b = mask_b.to(device_, non_blocking=True)
        x_norm = (x_b - stats_["x_mean"]) / stats_["x_std"]
        # Use inner (uncompiled) module so Python forward hooks fire reliably.
        with amp_ctx():
            _ = inner({"x": x_norm, "mask": mask_b})

    for h in handles:
        h.remove()

    print(f"\nResidual magnitude probe @ {epoch_tag} (post-norm topology):")
    log_payload = {
        "event": "residual_magnitude",
        "epoch_tag": epoch_tag,
        "ln_sites": captured,
    }
    for site in captured:
        print(
            f"  block[{site['block_idx']}].{site['ln_name']}: "
            f"pre_LN rms={site['pre_ln_rms']:.4f} max={site['pre_ln_abs_max']:.3f} | "
            f"post_LN rms={site['post_ln_rms']:.4f} max={site['post_ln_abs_max']:.3f}"
        )
    append_metrics_jsonl(metrics_jsonl_path, log_payload)
    return captured


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

    # Reset DropPath counters at the start of each epoch so the empirical
    # drop fraction below reflects only this epoch's forward passes (#3065).
    _inner_dp = getattr(model, "_orig_mod", model)
    for _m in _inner_dp.modules():
        if isinstance(_m, DropPath):
            _m.reset_counters()

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with amp_ctx_factory():
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm, "mask": mask})["preds"]
            sq_err = F.l1_loss(pred, y_norm, reduction='none')

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    lr_now = optimizer.param_groups[0]['lr']
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx_factory)
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
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": lr_now,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "compile_active": compile_active,
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

    # Residual magnitude probe at ep1 — captures the post-norm "capped" pattern
    # early in training (vs pre-norm baseline that grows residuals with depth).
    if (epoch + 1) == 1:
        _probe_loader = val_loaders.get("val_single_in_dist") or next(iter(val_loaders.values()))
        capture_residual_magnitudes(
            model, _probe_loader, stats, amp_ctx_factory, device, epoch_tag="ep1",
        )

    # DropPath empirical drop-fraction diagnostic (#3065). Counters were reset
    # at the start of this epoch, so the fraction is computed across this
    # epoch's training forward passes only. Log every epoch to JSONL; print
    # at ep1 / ep30 / ep(MAX) as called out in the PR.
    _dp_per_branch: list[dict] = []
    for _bi, _blk in enumerate(_inner_dp.blocks):
        if hasattr(_blk, "drop_path_attn"):
            _dp_per_branch.append({
                "block_idx": _bi,
                "branch": "attn",
                "nominal_p": _blk.drop_path_attn.p,
                "empirical_drop_frac": _blk.drop_path_attn.empirical_drop_fraction(),
                "n_branch_total": int(_blk.drop_path_attn._n_branch_total.item()),
            })
        if hasattr(_blk, "drop_path_mlp"):
            _dp_per_branch.append({
                "block_idx": _bi,
                "branch": "mlp",
                "nominal_p": _blk.drop_path_mlp.p,
                "empirical_drop_frac": _blk.drop_path_mlp.empirical_drop_fraction(),
                "n_branch_total": int(_blk.drop_path_mlp._n_branch_total.item()),
            })
    _dp_mean = (
        sum(d["empirical_drop_frac"] for d in _dp_per_branch) / max(len(_dp_per_branch), 1)
    )
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "drop_path_diagnostic",
        "epoch": epoch + 1,
        "branches": _dp_per_branch,
        "empirical_drop_frac_mean": _dp_mean,
    })
    if (epoch + 1) in (1, 30, MAX_EPOCHS):
        print(
            f"  DropPath empirical drop fraction @ ep{epoch+1} "
            f"(nominal p={_dp_per_branch[0]['nominal_p'] if _dp_per_branch else 0.0}):"
        )
        for _d in _dp_per_branch:
            print(
                f"    block[{_d['block_idx']}].{_d['branch']:<4s}: "
                f"emp_drop={_d['empirical_drop_frac']:.4f}"
            )
        print(f"    mean across {len(_dp_per_branch)} branches = {_dp_mean:.4f}")

    # Cross-block α convergence probe: log each scalar at ep1/10/30/60 to track
    # drift away from 1.0 init and per-position pattern (PR #3006 hypothesis test).
    if (epoch + 1) in (1, 10, 30, 60):
        _inner_alpha = getattr(model, "_orig_mod", model)
        _alphas = _inner_alpha.cross_block_alpha.detach().cpu().float().tolist()
        print(
            f"  cross-block α @ ep{epoch+1}: " + ", ".join(
                f"α_{ai}={v:.4f}" for ai, v in enumerate(_alphas)
            )
        )
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "cross_block_alpha",
            "epoch": epoch + 1,
            "alpha_values": _alphas,
            "alpha_mean": sum(_alphas) / max(len(_alphas), 1),
            "alpha_abs_mean_deviation": sum(abs(v - 1.0) for v in _alphas) / max(len(_alphas), 1),
        })

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Lion momentum diagnostic: sanity-check optimizer state at terminal ---
_lion_total = 0
_lion_nz = 0
for _group in optimizer.param_groups:
    for _p in _group['params']:
        if _p in optimizer.state:
            _m = optimizer.state[_p].get('momentum')
            if _m is not None:
                _lion_total += _m.numel()
                _lion_nz += (_m.abs() > 1e-8).sum().item()
_lion_nz_frac = _lion_nz / max(_lion_total, 1)
print(f"Lion momentum non-zero fraction at terminal: {_lion_nz_frac:.4f} ({_lion_nz}/{_lion_total} elements)")
append_metrics_jsonl(metrics_jsonl_path, {
    "event": "lion_momentum_diagnostic",
    "nonzero_fraction": _lion_nz_frac,
    "nonzero_count": _lion_nz,
    "total_count": _lion_total,
})

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Residual magnitude probe @ best-checkpoint (compare to ep1 capture).
    _probe_loader_end = val_loaders.get("val_single_in_dist") or next(iter(val_loaders.values()))
    capture_residual_magnitudes(
        model, _probe_loader_end, stats, amp_ctx_factory, device,
        epoch_tag=f"best_ep{best_metrics['epoch']}",
    )

    # LayerScale diagnostic: γ_attn = γ_mlp = 1.0 CONSTANT (no learnable Parameter).
    # Report the constants for confirmation; per-block dynamics are inapplicable.
    _inner = getattr(model, "_orig_mod", model)
    _gamma_log = {"event": "layerscale_gammas", "epoch": int(best_metrics["epoch"]),
                  "layerscale_mode": "constant_1.0_no_parameter", "blocks": []}
    print("\nLayerScale final gammas (best-checkpoint weights, CONSTANT 1.0 — no learnable γ):")
    for _i, _blk in enumerate(_inner.blocks):
        _ga = float(_blk.gamma_attn) if not torch.is_tensor(_blk.gamma_attn) else float(_blk.gamma_attn.detach().float().mean().item())
        _gm = float(_blk.gamma_mlp) if not torch.is_tensor(_blk.gamma_mlp) else float(_blk.gamma_mlp.detach().float().mean().item())
        print(f"  block[{_i}]: gamma_attn={_ga:.6f} | gamma_mlp={_gm:.6f} (constant; no Parameter)")
        _gamma_log["blocks"].append({
            "block_idx": _i,
            "gamma_attn_mean": _ga,
            "gamma_attn_abs_mean": abs(_ga),
            "gamma_mlp_mean": _gm,
            "gamma_mlp_abs_mean": abs(_gm),
        })
    append_metrics_jsonl(metrics_jsonl_path, _gamma_log)

    # Cross-block α at best checkpoint — final values after loading best weights.
    _alpha_best = _inner.cross_block_alpha.detach().cpu().float().tolist()
    _alpha_best_log = {
        "event": "cross_block_alpha_best",
        "epoch": int(best_metrics["epoch"]),
        "alpha_values": _alpha_best,
        "alpha_mean": sum(_alpha_best) / max(len(_alpha_best), 1),
        "alpha_abs_mean_deviation": sum(abs(v - 1.0) for v in _alpha_best) / max(len(_alpha_best), 1),
    }
    print(
        "\nCross-block α at best checkpoint: "
        + ", ".join(f"α_{ai}={v:.4f}" for ai, v in enumerate(_alpha_best))
        + f" (mean={_alpha_best_log['alpha_mean']:.4f}, |Δ|/n={_alpha_best_log['alpha_abs_mean_deviation']:.4f})"
    )
    append_metrics_jsonl(metrics_jsonl_path, _alpha_best_log)

    # Feature-stream FiLM diagnostic: weight/bias norms and modulation magnitude
    _film = _inner.film
    _film_w_norm = _film.weight.detach().float().norm().item()
    _film_b_norm = _film.bias.detach().float().norm().item()
    print("\nFeature-stream FiLM final diagnostics (best-checkpoint weights):")
    print(f"  film.weight.norm: {_film_w_norm:.4f}")
    print(f"  film.bias.norm:   {_film_b_norm:.4f}")

    # Modulation magnitude on a val_re_rand batch (highest-OOD split for Re).
    _film_scale_mag = None
    _val_re_rand_loader = val_loaders.get("val_re_rand")
    if _val_re_rand_loader is not None:
        with torch.no_grad():
            _x, _y, _is_surface, _mask = next(iter(_val_re_rand_loader))
            _x = _x.to(device, non_blocking=True)
            _x_norm = (_x - stats["x_mean"]) / stats["x_std"]
            _flow_scalars = _x_norm[:, 0, [13, 14, 18]]
            _film_scale = _film(_flow_scalars.float()).unsqueeze(1)
            _film_scale_mag = (1 + _film_scale).abs().mean().item()
        print(f"  film_scale |1+gamma| mean on val_re_rand batch: {_film_scale_mag:.4f}")
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "film_diagnostic",
        "epoch": int(best_metrics["epoch"]),
        "film_weight_norm": _film_w_norm,
        "film_bias_norm": _film_b_norm,
        "film_scale_abs_mean_val_re_rand": _film_scale_mag,
    })

    # Squeeze-Excitation diagnostic: gate + attention-pool stats per block per
    # split (in_dist vs OOD). Hook mirrors the SE forward — attention pool
    # (softmax over masked tokens) then bottleneck MLP — so the captured gate
    # and attn weights match the actual module behaviour.
    _se_mods_with_idx = [(i, blk.se) for i, blk in enumerate(_inner.blocks) if blk.se is not None]
    _se_per_split: dict[str, list[tuple[int, torch.Tensor]]] = {}
    _attn_pool_per_split: dict[str, list[dict]] = {}

    def _make_se_hook(block_idx: int, split_name: str, is_surface_t: torch.Tensor):
        def _hook(module, args, kwargs, output):
            with torch.no_grad():
                x = args[0] if args else kwargs.get("x")
                m = kwargs.get("mask")
                attn_logits = module.attn_pool(x)  # (B, N, 1)
                if m is not None:
                    attn_logits = attn_logits.masked_fill(~m.unsqueeze(-1), -1e9)
                attn_w = torch.softmax(attn_logits, dim=1)  # (B, N, 1)
                s = (x * attn_w).sum(dim=1)  # (B, D)
                s = module.fc2(F.gelu(module.fc1(s)))
                gate = torch.sigmoid(s).detach().float().cpu()
                _se_per_split.setdefault(split_name, []).append((block_idx, gate))

                # Attention pool diagnostics — per-sample over real tokens only
                w = attn_w.squeeze(-1).detach().float()  # (B, N)
                B = w.shape[0]
                max_ws, ents, surf_fracs, n_actives, n_reals = [], [], [], [], []
                for b in range(B):
                    m_b = m[b] if m is not None else torch.ones(w.shape[1], dtype=torch.bool, device=w.device)
                    w_b = w[b][m_b]
                    if w_b.numel() == 0:
                        continue
                    max_ws.append(w_b.max().item())
                    ents.append(-(w_b * (w_b + 1e-12).log()).sum().item())
                    is_surf_b = is_surface_t[b][m_b]
                    surf_fracs.append(w_b[is_surf_b].sum().item() if is_surf_b.any() else 0.0)
                    n_actives.append((w_b > 0.1).sum().item())
                    n_reals.append(int(m_b.sum().item()))
                _attn_pool_per_split.setdefault(split_name, []).append({
                    "block_idx": block_idx,
                    "max_w_mean": float(sum(max_ws) / max(len(max_ws), 1)),
                    "entropy_mean": float(sum(ents) / max(len(ents), 1)),
                    "surface_frac_mean": float(sum(surf_fracs) / max(len(surf_fracs), 1)),
                    "n_active_tokens_mean": float(sum(n_actives) / max(len(n_actives), 1)),
                    "n_real_tokens_mean": float(sum(n_reals) / max(len(n_reals), 1)),
                })
        return _hook

    _se_diag_splits = ["val_single_in_dist", "val_geom_camber_rc",
                       "val_geom_camber_cruise", "val_re_rand"]
    for _split_name in _se_diag_splits:
        _ldr = val_loaders.get(_split_name)
        if _ldr is None:
            continue
        with torch.no_grad():
            _x_s, _y_s, _is_s, _m_s = next(iter(_ldr))
            _x_s = _x_s.to(device, non_blocking=True)
            _is_s_dev = _is_s.to(device, non_blocking=True)
            _m_s = _m_s.to(device, non_blocking=True)
            _se_handles = [
                mod.register_forward_hook(_make_se_hook(bi, _split_name, _is_s_dev), with_kwargs=True)
                for bi, mod in _se_mods_with_idx
            ]
            _x_norm_s = (_x_s - stats["x_mean"]) / stats["x_std"]
            # Route through uncompiled module so Python-level hooks fire
            # (torch.compile bypasses hooks registered after compile).
            with amp_ctx_factory():
                _ = _inner({"x": _x_norm_s, "mask": _m_s})
        for _h in _se_handles:
            _h.remove()

    print("\nSE gate stats per split (best-checkpoint weights, block-3-only):")
    _se_log = {"event": "se_diagnostic", "epoch": int(best_metrics["epoch"]), "splits": {}}
    for _split_name, _captures in _se_per_split.items():
        print(f"  Split: {_split_name}")
        _split_log = []
        for _idx, _gate in _captures:
            _gm = _gate.mean().item()
            _gs = _gate.std().item()
            _gmin = _gate.min().item()
            _gmax = _gate.max().item()
            print(f"    SE block[{_idx}]: gate mean={_gm:.4f}  std={_gs:.4f}  min={_gmin:.4f}  max={_gmax:.4f}")
            _split_log.append({
                "block_idx": _idx,
                "gate_mean": _gm,
                "gate_std": _gs,
                "gate_min": _gmin,
                "gate_max": _gmax,
            })
        _se_log["splits"][_split_name] = _split_log
    append_metrics_jsonl(metrics_jsonl_path, _se_log)

    # SE attention-pool diagnostic: how concentrated is the pool? How much
    # weight on surface tokens vs uniform reference (~surface_frac=0.05-0.3
    # depending on mesh)? Lower entropy / higher max_w = sharper pool.
    print("\nSE attention-pool stats per split (best-checkpoint weights):")
    for _split_name, _ap_list in _attn_pool_per_split.items():
        for _ap in _ap_list:
            print(
                f"  Split: {_split_name}  SE block[{_ap['block_idx']}]: "
                f"max_w_mean={_ap['max_w_mean']:.4f}  "
                f"entropy_mean={_ap['entropy_mean']:.4f}  "
                f"surface_frac_mean={_ap['surface_frac_mean']:.4f}  "
                f"n_active(w>0.1)_mean={_ap['n_active_tokens_mean']:.1f}  "
                f"n_real_mean={_ap['n_real_tokens_mean']:.0f}"
            )
            append_metrics_jsonl(metrics_jsonl_path, {
                "event": "se_attn_pool_diagnostic",
                "epoch": int(best_metrics["epoch"]),
                "split": _split_name,
                "block_idx": _ap["block_idx"],
                "max_w_mean": _ap["max_w_mean"],
                "entropy_mean": _ap["entropy_mean"],
                "surface_frac": _ap["surface_frac_mean"],
                "n_active_tokens": _ap["n_active_tokens_mean"],
                "n_real_tokens": _ap["n_real_tokens_mean"],
            })

    # SwiGLU gate diagnostic — capture per-block gate (silu(W_gate x)) and value
    # (W_value x) statistics on a sample val batch, best-checkpoint weights.
    # gate_zero_frac near 0 ≈ gate passing everything through (acts like a richer
    # MLP); gate_zero_frac > 0.1 ≈ conditional channel suppression engaged.
    _swiglu_blocks = [(i, blk.mlp) for i, blk in enumerate(_inner.blocks)
                      if isinstance(blk.mlp, SwiGLUMLP)]
    _swiglu_captured: list[tuple[int, dict[str, float]]] = []

    def _make_swiglu_hook(idx: int):
        def _hook(module, args, _output):
            x = args[0]
            with torch.no_grad():
                gate_pre = module.linear_gate(x)
                gate_post = module.act_fn(gate_pre).detach().float()
                value = module.linear_value(x).detach().float()
                flat_g = gate_post.flatten()
                flat_v = value.flatten()
                # Correlation in a fixed-size random subsample to avoid GPU OOM
                # on big meshes (242K nodes × 128 hidden = 31M elements).
                n_sub = min(flat_g.numel(), 200_000)
                if n_sub < flat_g.numel():
                    idx_sub = torch.randperm(flat_g.numel(), device=flat_g.device)[:n_sub]
                    fg = flat_g[idx_sub]
                    fv = flat_v[idx_sub]
                else:
                    fg, fv = flat_g, flat_v
                corr = float(
                    torch.corrcoef(torch.stack([fg, fv]))[0, 1].cpu().item()
                ) if fg.numel() > 1 else 0.0
                _swiglu_captured.append((idx, {
                    "gate_mean": float(gate_post.mean().cpu()),
                    "gate_std": float(gate_post.std().cpu()),
                    "gate_abs_mean": float(gate_post.abs().mean().cpu()),
                    "gate_zero_frac": float((gate_post.abs() < 0.01).float().mean().cpu()),
                    "value_abs_mean": float(value.abs().mean().cpu()),
                    "gate_value_corr": corr,
                }))
        return _hook

    _swiglu_handles = [m.register_forward_hook(_make_swiglu_hook(i))
                       for i, m in _swiglu_blocks]
    _swiglu_loader = val_loaders.get("val_single_in_dist") or next(iter(val_loaders.values()))
    with torch.no_grad():
        _x_s, _y_s, _is_s, _m_s = next(iter(_swiglu_loader))
        _x_s = _x_s.to(device, non_blocking=True)
        _m_s = _m_s.to(device, non_blocking=True)
        _x_norm_s = (_x_s - stats["x_mean"]) / stats["x_std"]
        with amp_ctx_factory():
            _ = _inner({"x": _x_norm_s, "mask": _m_s})
    for _h in _swiglu_handles:
        _h.remove()

    print("\nSwiGLU gate stats per block (sample val batch, best-checkpoint weights):")
    _swiglu_log = {"event": "swiglu_diagnostic", "epoch": int(best_metrics["epoch"]), "blocks": []}
    for _idx, _stats in _swiglu_captured:
        print(
            f"  SwiGLU block[{_idx}]: gate_mean={_stats['gate_mean']:.4f}  "
            f"std={_stats['gate_std']:.4f}  abs_mean={_stats['gate_abs_mean']:.4f}  "
            f"zero_frac={_stats['gate_zero_frac']:.4f}  "
            f"value_abs_mean={_stats['value_abs_mean']:.4f}  "
            f"corr(gate,value)={_stats['gate_value_corr']:.4f}"
        )
        _swiglu_log["blocks"].append({"block_idx": _idx, **_stats})
    append_metrics_jsonl(metrics_jsonl_path, _swiglu_log)

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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx_factory)
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
