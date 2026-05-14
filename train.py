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
from copy import deepcopy
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


class FourierFeatures(nn.Module):
    """Replace the first 2 dims of the input (x, z node positions) with
    Fourier feature expansion at K exponentially-spaced frequencies.

    Input shape:  [..., D]  where D = X_DIM.
    Output shape: [..., 4*K + (D-2)]  when K > 0; identity when K == 0.

    Frequencies are logspaced from 1.0 to ``max_freq``. Operates on the
    NORMALIZED input (x_norm = (x - x_mean) / x_std), so positions are
    already O(1) magnitude. No learnable parameters — freqs is a buffer.
    """

    def __init__(self, k: int, max_freq: float = 10.0):
        super().__init__()
        self.k = int(k)
        if self.k > 0:
            freqs = torch.logspace(0.0, math.log10(max_freq), self.k)
            self.register_buffer("freqs", freqs)

    def forward(self, x):
        if self.k == 0:
            return x
        pos = x[..., :2]
        rest = x[..., 2:]
        scaled = pos.unsqueeze(-1) * self.freqs.view(*([1] * pos.ndim), -1) * 2.0 * math.pi
        # scaled: [..., 2, K]
        sin_feats = scaled.sin().reshape(*pos.shape[:-1], 2 * self.k)
        cos_feats = scaled.cos().reshape(*pos.shape[:-1], 2 * self.k)
        return torch.cat([sin_feats, cos_feats, rest], dim=-1)


class GeoContextEncoder(nn.Module):
    """Encode per-sample scalars + masked-mean surface summary into K context tokens.

    Output is (B, n_tokens, d_model) used as additional K/V in PhysicsAttention.
    Final ``token_proj`` is small-init in Transolver._init_weights so the
    cross-attention starts with minimal effect.
    """

    def __init__(self, sample_feat_dim, node_feat_dim, hidden, n_tokens, d_model):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.sample_mlp = nn.Sequential(
            nn.Linear(sample_feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.surface_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.token_proj = nn.Linear(d_model * 2, n_tokens * d_model)

    def forward(self, sample_feats, surface_summary):
        s = self.sample_mlp(sample_feats)
        u = self.surface_mlp(surface_summary)
        ctx = torch.cat([s, u], dim=-1)
        return self.token_proj(ctx).view(-1, self.n_tokens, self.d_model)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 use_geo_cross_attn=False, geo_d_model=None):
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

        self.use_geo_cross_attn = use_geo_cross_attn
        if use_geo_cross_attn:
            geo_d_model = geo_d_model or dim
            self.geo_k_proj = nn.Linear(geo_d_model, inner_dim, bias=False)
            self.geo_v_proj = nn.Linear(geo_d_model, inner_dim, bias=False)

    def forward(self, x, geo_tokens=None):
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

        if self.use_geo_cross_attn and geo_tokens is not None:
            B_g, M, _ = geo_tokens.shape
            k_cross = (
                self.geo_k_proj(geo_tokens)
                .reshape(B_g, M, self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
            )
            v_cross = (
                self.geo_v_proj(geo_tokens)
                .reshape(B_g, M, self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
            )
            k = torch.cat([k, k_cross], dim=2)
            v = torch.cat([v, v_cross], dim=2)

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
                 use_geo_cross_attn=False, geo_d_model=None):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
            use_geo_cross_attn=use_geo_cross_attn, geo_d_model=geo_d_model,
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

    def forward(self, fx, geo_tokens=None):
        fx = self.attn(self.ln_1(fx), geo_tokens=geo_tokens) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    # Last 11 dims of (post-Fourier) x are per-sample BC/geometry scalars:
    # log(Re), AoA1, NACA1[3], AoA2, NACA2[3], gap, stagger. FourierFeatures
    # only rewrites dims 0-1, so the trailing 11 dims align with original
    # x[..., 13:24] regardless of fourier_k.
    GEO_SAMPLE_FEAT_DIM = 11

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 use_geo_cross_attn=False, geo_cross_tokens=4,
                 geo_encoder_hidden=64):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.use_geo_cross_attn = use_geo_cross_attn

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if use_geo_cross_attn:
            self.geo_encoder = GeoContextEncoder(
                sample_feat_dim=self.GEO_SAMPLE_FEAT_DIM,
                node_feat_dim=fun_dim + space_dim,
                hidden=geo_encoder_hidden,
                n_tokens=geo_cross_tokens,
                d_model=n_hidden,
            )
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                use_geo_cross_attn=use_geo_cross_attn, geo_d_model=n_hidden,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        if use_geo_cross_attn:
            # Small-init the gateways into the cross-attention path so the
            # network starts close to the baseline and learns to use the
            # geo tokens gradually instead of perturbing slice attention
            # at step 0.
            nn.init.normal_(self.geo_encoder.token_proj.weight, std=0.01)
            nn.init.zeros_(self.geo_encoder.token_proj.bias)
            for block in self.blocks:
                nn.init.normal_(block.attn.geo_k_proj.weight, std=0.01)
                nn.init.normal_(block.attn.geo_v_proj.weight, std=0.01)

        self._geo_token_shape_printed = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_geo_tokens(self, x, is_surface, mask):
        # x: (B, N, D_aug). Per-sample features are constant across nodes;
        # masked mean gives the per-sample value while ignoring padding.
        mask_f = mask.to(x.dtype).unsqueeze(-1)
        denom = mask_f.sum(1).clamp(min=1.0)

        sample_feats_per_node = x[..., -self.GEO_SAMPLE_FEAT_DIM:]
        sample_feats = (sample_feats_per_node * mask_f).sum(1) / denom

        surf_mask_f = (mask & is_surface).to(x.dtype).unsqueeze(-1)
        surf_denom = surf_mask_f.sum(1).clamp(min=1.0)
        surface_summary = (x * surf_mask_f).sum(1) / surf_denom

        geo_tokens = self.geo_encoder(sample_feats, surface_summary)
        if not self._geo_token_shape_printed:
            print(
                f"  [geo cross-attn] sample_feats={tuple(sample_feats.shape)} "
                f"surface_summary={tuple(surface_summary.shape)} "
                f"geo_tokens={tuple(geo_tokens.shape)}"
            )
            self._geo_token_shape_printed = True
        return geo_tokens

    def forward(self, data, **kwargs):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        geo_tokens = None
        if self.use_geo_cross_attn:
            geo_tokens = self._build_geo_tokens(x, data["is_surface"], data["mask"])
        for block in self.blocks:
            fx = block(fx, geo_tokens=geo_tokens)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   loss_fn: str = "mse", smooth_l1_beta: float = 0.1,
                   amp: bool = False) -> dict[str, float]:
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

            # Drop samples whose y contains any non-finite value; matches
            # scoring.py::accumulate_batch's per-sample skip and prevents
            # Inf*0=NaN propagation through loss/err arithmetic. The
            # test_geom_camber_cruise/000020.pt sample has 761 Inf in y[..., p].
            y_finite_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not bool(y_finite_sample.all()):
                if not bool(y_finite_sample.any()):
                    continue
                x = x[y_finite_sample]
                y = y[y_finite_sample]
                is_surface = is_surface[y_finite_sample]
                mask = mask[y_finite_sample]

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            model_inputs = {"x": x_norm, "is_surface": is_surface, "mask": mask}
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(model_inputs)["preds"]
                pred = pred.float()
            else:
                pred = model(model_inputs)["preds"]

            if loss_fn == "smooth_l1":
                per_elem = F.smooth_l1_loss(
                    pred, y_norm, reduction="none", beta=smooth_l1_beta
                )
            elif loss_fn == "l1":
                per_elem = torch.abs(pred - y_norm)
            else:
                per_elem = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (per_elem * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (per_elem * surf_mask.unsqueeze(-1)).sum()
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
    grad_clip: float = 0.0
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    loss_fn: str = "mse"  # "mse", "smooth_l1", or "l1"
    smooth_l1_beta: float = 0.1
    ema_decay: float = 0.0  # 0.0 disables EMA; >0 enables EMA of weights for val/test/ckpt
    amp: bool = False  # bfloat16 mixed precision autocast for fwd + loss
    warmup_epochs: int = 0  # linear LR warmup epochs before cosine decay (0 = disabled)
    fourier_k: int = 0  # 0 = off (baseline); K > 0 enables Fourier expansion of (x, z) into 4K features
    fourier_max_freq: float = 10.0  # max frequency in the logspaced band; positions are O(1) post-norm
    slice_num: int = 64
    n_layers: int = 5  # Transolver block depth
    optimizer: str = "adamw"  # "adamw" (default — bit-identical to prior) or "lion"
    n_head: int = 4  # Transolver attention head count (dim_head = n_hidden // n_head)
    use_geo_cross_attn: bool = False  # add cross-attention to global geometry/flow context (GeoTransolver H5)
    geo_cross_tokens: int = 4  # number of global context tokens (K_cross)
    geo_encoder_hidden: int = 64  # hidden dim of geometry encoder MLP


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
    n_layers=cfg.n_layers,
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    use_geo_cross_attn=cfg.use_geo_cross_attn,
    geo_cross_tokens=cfg.geo_cross_tokens,
    geo_encoder_hidden=cfg.geo_encoder_hidden,
)

if cfg.fourier_k > 0:
    model_config["space_dim"] = 4 * cfg.fourier_k
    model_config["fun_dim"] = X_DIM - 2


class FourierModel(nn.Module):
    """Wraps Transolver with a FourierFeatures expansion of the first 2 input dims."""

    def __init__(self, fourier: FourierFeatures, base: Transolver):
        super().__init__()
        self.fourier = fourier
        self.base = base

    def forward(self, data, **kwargs):
        x_aug = self.fourier(data["x"])
        new_data = {k: v for k, v in data.items() if k != "x"}
        new_data["x"] = x_aug
        return self.base(new_data, **kwargs)


fourier = FourierFeatures(cfg.fourier_k, cfg.fourier_max_freq).to(device)
base_model = Transolver(**model_config).to(device)
model = FourierModel(fourier, base_model)
n_params = sum(p.numel() for p in model.parameters())
print(
    f"Model: Transolver+FourierFeatures(K={cfg.fourier_k}, max_freq={cfg.fourier_max_freq}) "
    f"({n_params/1e6:.2f}M params)"
)

ema_model = deepcopy(model) if cfg.ema_decay > 0 else None
if ema_model is not None:
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    print(f"EMA enabled (decay={cfg.ema_decay})")

if cfg.optimizer == "lion":
    from timm.optim import Lion
    optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"Optimizer: Lion (lr={cfg.lr}, weight_decay={cfg.weight_decay})")
elif cfg.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r} (expected 'adamw' or 'lion')")

if cfg.warmup_epochs > 0:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
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

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
model_no_ema_path = model_dir / "checkpoint_no_ema.pt"
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

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if cfg.amp and device.type == "cuda"
            else torch.amp.autocast(device_type="cuda", enabled=False)
        )
        with amp_ctx:
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm, "is_surface": is_surface, "mask": mask})["preds"]
            if cfg.loss_fn == "smooth_l1":
                per_elem = F.smooth_l1_loss(
                    pred, y_norm, reduction="none", beta=cfg.smooth_l1_beta
                )
            elif cfg.loss_fn == "l1":
                per_elem = torch.abs(pred - y_norm)
            else:
                per_elem = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (per_elem * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (per_elem * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        clip_value = cfg.grad_clip if cfg.grad_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for ep, p in zip(ema_model.parameters(), model.parameters()):
                    ep.data.mul_(cfg.ema_decay).add_(p.data, alpha=1.0 - cfg.ema_decay)
                for eb, b in zip(ema_model.buffers(), model.buffers()):
                    eb.data.copy_(b.data)
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
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
    eval_model = ema_model if ema_model is not None else model
    split_metrics = {
        name: evaluate_split(eval_model, loader, stats, cfg.surf_weight, device,
                             loss_fn=cfg.loss_fn, smooth_l1_beta=cfg.smooth_l1_beta,
                             amp=cfg.amp)
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
        torch.save(eval_model.state_dict(), model_path)
        if ema_model is not None:
            torch.save(model.state_dict(), model_no_ema_path)
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

    test_eval_model = ema_model if ema_model is not None else model
    test_eval_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_eval_model.eval()

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
            name: evaluate_split(test_eval_model, loader, stats, cfg.surf_weight, device,
                                 loss_fn=cfg.loss_fn, smooth_l1_beta=cfg.smooth_l1_beta,
                                 amp=cfg.amp)
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

        # Dual test eval: when EMA is enabled, also evaluate the non-EMA
        # snapshot saved at the same best-EMA-val epoch so we can isolate
        # variance-reduction-on-eval from any underlying-weights effect.
        if ema_model is not None and model_no_ema_path.exists():
            print("\nEvaluating non-EMA model at the same best-val epoch on test splits...")
            model.load_state_dict(torch.load(model_no_ema_path, map_location=device, weights_only=True))
            model.eval()
            test_metrics_no_ema = {
                name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                                     loss_fn=cfg.loss_fn, smooth_l1_beta=cfg.smooth_l1_beta,
                                     amp=cfg.amp)
                for name, loader in test_loaders.items()
            }
            test_avg_no_ema = aggregate_splits(test_metrics_no_ema)
            print(f"  TEST(no-EMA)  avg_surf_p={test_avg_no_ema['avg/mae_surf_p']:.4f}")
            for name in TEST_SPLIT_NAMES:
                print_split_metrics(f"{name}_no_ema", test_metrics_no_ema[name])

            test_log_no_ema: dict[str, float] = {}
            for split_name, m in test_metrics_no_ema.items():
                for k, v in m.items():
                    test_log_no_ema[f"test_no_ema/{split_name}/{k}"] = v
            for k, v in test_avg_no_ema.items():
                test_log_no_ema[f"test_no_ema_{k}"] = v
            wandb.log(test_log_no_ema)
            wandb.summary.update(test_log_no_ema)

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
