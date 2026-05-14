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

import json
import math
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
from torch.utils.checkpoint import checkpoint
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
    """SwiGLU feed-forward block (Shazeer 2020).

    Replaces a standard 2-matrix MLP `W2 · GELU(W1 · x)` with the gated
    variant `W_down · (SiLU(W_gate · x) ⊙ W_up · x)`.

    `hidden_dim` is the original MLP inner dim; we use the full
    `hidden_dim` here (rounded up to a multiple of 8) to give the
    gate/up/down projections their natural per-token routing capacity
    (was 2/3 × hidden_dim for param-matching with the GELU 2-matrix MLP).
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        inner_dim = 288  # H46: bisect capacity (was hidden_dim=256; 320 was compute-bound)
        inner_dim = ((inner_dim + 7) // 8) * 8  # stays 288 (multiple of 8)
        self.inner_dim = inner_dim
        self.w_gate = nn.Linear(in_dim, inner_dim, bias=False)
        self.w_up = nn.Linear(in_dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, in_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.relu(self.w_gate(x)) * self.w_up(x))   # H39: ReGLU gate (ReLU = max(0,x))


class FourierCoordEnc(nn.Module):
    """Replace the 2 normalized coord dims with 2*2*n_freqs Fourier features.

    Input  shape: [B, N, in_dim]  where in_dim = X_DIM (raw 24-dim feature vector,
                                  already normalized by stats["x_mean"], stats["x_std"]).
    Output shape: [B, N, in_dim + (4*n_freqs - 2)]  -- 2 coord dims replaced by 4*n_freqs
                                                       Fourier features.

    Freqs are learnable (dyadic init), trained with their own optimizer group
    (10x lr, no weight decay) and clamped to [0.1, 100] after each step.
    """

    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.freqs = nn.Parameter(freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :2]
        angles = coords.unsqueeze(-1) * self.freqs[None, None, None, :] * torch.pi
        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)
        fourier = torch.cat([sin_feats, cos_feats], dim=-1).reshape(
            *x.shape[:-1], 4 * self.n_freqs
        )
        return torch.cat([fourier, x[..., 2:]], dim=-1)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    # H67: when True, the forward pass also computes the slice-to-slice
    # softmax weights explicitly and writes the mean / per-head min entropy
    # to ``self.last_attn_entropy_*`` for diagnostic logging. Default False
    # so the production path stays on F.scaled_dot_product_attention.
    record_entropy: bool = False

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        # H73: attention temperature annealing — start sharper (sqrt(3)) and
        # anneal toward sqrt(2) over training. Buffer so the value persists
        # via state_dict (checkpoint load/save) without becoming a learnable
        # parameter. Updated in the training loop, once per epoch.
        self.register_buffer(
            "attn_sharpening_factor", torch.tensor(math.sqrt(3.0))
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.last_attn_entropy_mean: float | None = None
        self.last_attn_entropy_per_head_min: float | None = None

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
        # H73: dynamic sharper temperature — buffer-stored sharpening factor
        # updated per-epoch from the training loop (linear anneal sqrt(3) ->
        # sqrt(2) across the first N_ANNEAL_EPOCHS epochs, clamped to sqrt(2)
        # afterward). Default scale is 1/sqrt(d_head); factor=sqrt(2) matches
        # #2519, factor=sqrt(3) is the sharper early-epoch start.
        sharper_scale = self.attn_sharpening_factor.item() / math.sqrt(self.dim_head)
        out_slice = F.scaled_dot_product_attention(
            q, k, v,
            scale=sharper_scale,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        if PhysicsAttention.record_entropy:
            with torch.no_grad():
                logits = torch.matmul(q, k.transpose(-2, -1)) * sharper_scale
                weights = F.softmax(logits, dim=-1)
                # entropy per (batch, head, query-token): -sum_k p_k log p_k
                ent = -(weights * (weights + 1e-12).log()).sum(dim=-1)  # [B, H, T]
                self.last_attn_entropy_mean = float(ent.mean().item())
                self.last_attn_entropy_per_head_min = float(
                    ent.mean(dim=(0, 2)).min().item()
                )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 stoch_depth_prob: float = 0.0,
                 layer_scale_init: float = 0.1):
        super().__init__()
        self.last_layer = last_layer
        self.stoch_depth_prob = stoch_depth_prob
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim, hidden_dim * mlp_ratio)
        self.layer_scale_attn = nn.Parameter(torch.ones(hidden_dim) * layer_scale_init)
        self.layer_scale_mlp = nn.Parameter(torch.ones(hidden_dim) * layer_scale_init)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        if self.training and self.stoch_depth_prob > 0.0:
            if torch.rand(1, device=fx.device).item() < self.stoch_depth_prob:
                if self.last_layer:
                    return self.mlp2(self.ln_3(fx))
                return fx
        fx = self.layer_scale_attn * self.attn(self.ln_1(fx)) + fx
        fx = self.layer_scale_mlp * self.mlp(self.ln_2(fx)) + fx
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
                stoch_depth_prob=0.1 * (i / max(n_layers - 1, 1)),
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
# H90: GeoMPNN — geometry-aware message-passing GNN
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_knn_graph(
    pos: torch.Tensor, mask: torch.Tensor, k: int, chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KNN graph per sample, chunked along the query axis.

    Naive ``torch.cdist(pos, pos)`` is ``[B, N, N]`` which is ~938 GB at
    ``B=4, N=242K``. We chunk over queries so peak memory is
    ``B * chunk_size * N * 4 bytes``.

    Returns ``(knn_dist [B, N, k], knn_idx [B, N, k])``. Padded source
    nodes (``~mask``) are masked to ``+inf`` distance so they never appear
    in any query's neighbour list (unless a query has fewer than ``k``
    real neighbours, in which case the extra slots have ``+inf`` and are
    replaced with 0 by the caller).
    """
    B, N, _ = pos.shape
    knn_dist = torch.empty(B, N, k, device=pos.device, dtype=pos.dtype)
    knn_idx = torch.empty(B, N, k, device=pos.device, dtype=torch.long)
    invalid_j = ~mask  # [B, N]
    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        chunk_q = pos[:, cs:ce]                # [B, Nc, 2]
        d = torch.cdist(chunk_q, pos)          # [B, Nc, N]
        d = d.masked_fill(invalid_j.unsqueeze(1), float("inf"))
        kd, ki = torch.topk(d, k=k + 1, dim=-1, largest=False)
        # drop self (index 0 in sorted order — query is at distance 0 to itself)
        knn_dist[:, cs:ce] = kd[..., 1:]
        knn_idx[:, cs:ce] = ki[..., 1:]
    return knn_dist, knn_idx


class GeoMPNNLayer(nn.Module):
    """Single GeoMPNN message-passing layer with residual + LayerScale.

    Uses a **factorized message** form to avoid materializing the wide
    ``[B, N, k, 3H]`` concat tensor used by the canonical
    ``msg = MLP(cat(h_i, h_j, e_ij))`` formulation:

        msg = W_out · GELU(W_i · h_i + W_j · h_j + W_e · e_ij)

    Mathematically the same first linear, but ``W_i · h_i`` is computed
    on ``[B, N, H]`` (no ``k`` dim) and broadcast over neighbours.
    """

    def __init__(self, hidden: int, ls_init: float = 0.1):
        super().__init__()
        self.ln_h = nn.LayerNorm(hidden)
        self.msg_i = nn.Linear(hidden, hidden)
        self.msg_j = nn.Linear(hidden, hidden)
        self.msg_e = nn.Linear(hidden, hidden)
        self.msg_out = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.layer_scale = nn.Parameter(torch.ones(hidden) * ls_init)

    def forward(self, h: torch.Tensor, edge_h: torch.Tensor,
                knn_idx: torch.Tensor) -> torch.Tensor:
        B, N, H = h.shape
        K = knn_idx.shape[2]
        h_norm = self.ln_h(h)
        # Gather neighbour features.
        # Critical: do NOT use ``h_norm.unsqueeze(1).expand(-1, N, -1, -1)`` —
        # that creates a [B, N, N, H] *virtual* view, but ``gather``'s backward
        # allocates ``grad_input`` of the source shape, which would be
        # ~134 TB at N=242K. Instead, flatten the (query, k) axis and gather
        # along dim=1 of the original [B, N, H] tensor.
        idx_flat = knn_idx.reshape(B, N * K, 1).expand(B, N * K, H)  # view; stride=0 on H
        h_j_flat = torch.gather(h_norm, dim=1, index=idx_flat)       # [B, N*K, H]
        h_j = h_j_flat.view(B, N, K, H)                              # [B, N, K, H]
        h_i_p = self.msg_i(h_norm).unsqueeze(2)                      # [B, N, 1, H]
        h_j_p = self.msg_j(h_j)                                      # [B, N, K, H]
        e_p = self.msg_e(edge_h)                                     # [B, N, K, H]
        msgs = self.msg_out(F.gelu(h_i_p + h_j_p + e_p))             # [B, N, K, H]
        agg = msgs.mean(dim=2)                                       # [B, N, H]
        update_input = torch.cat([h_norm, agg], dim=-1)              # [B, N, 2H]
        delta = self.update(update_input)                            # [B, N, H]
        return h + self.layer_scale * delta


class GeoMPNN(nn.Module):
    """Geometry-aware message-passing GNN for irregular meshes (H90).

    Replaces Transolver's slice-token attention with explicit local
    message passing along a KNN graph in node-position space. Edges
    carry geometry features ``(rel_pos_xy, distance)``, encoded once
    and reused across all layers.

    Input contract: ``data = {"x": [B, N, in_dim], "pos": [B, N, 2],
    "mask": [B, N]}``; output ``{"preds": [B, N, out_dim]}`` with
    padded rows zeroed.
    """

    def __init__(
        self,
        in_dim: int = 24,
        hidden: int = 288,
        out_dim: int = 3,
        n_layers: int = 5,
        k_neighbors: int = 16,
        ls_init: float = 0.1,
        knn_chunk_size: int = 1024,
        use_checkpoint: bool = True,
        output_fields: list[str] | None = None,
        output_dims: list[int] | None = None,
    ):
        super().__init__()
        self.k = k_neighbors
        self.n_layers = n_layers
        self.hidden = hidden
        self.knn_chunk_size = knn_chunk_size
        self.use_checkpoint = use_checkpoint
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(3, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
        )
        self.layers = nn.ModuleList([
            GeoMPNNLayer(hidden, ls_init=ls_init) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]          # [B, N, in_dim]
        pos = data["pos"]      # [B, N, 2]
        mask = data["mask"]    # [B, N], bool
        B, N, _ = x.shape

        h = self.node_enc(x)   # [B, N, H]

        # KNN graph in fp32 (autocast disabled): cdist + topk need numerical
        # precision; small tensors so bf16 buys nothing here.
        with torch.amp.autocast("cuda", enabled=False):
            knn_dist, knn_idx = compute_knn_graph(
                pos.float(), mask, k=self.k, chunk_size=self.knn_chunk_size,
            )
            # Replace +inf with 0 for queries whose neighbours fell short — those
            # rows are zeroed at the output anyway since the query is padded.
            knn_dist = torch.where(
                torch.isfinite(knn_dist), knn_dist, torch.zeros_like(knn_dist),
            )
            with torch.no_grad():
                idx_exp_pos = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # [B, N, k, 2]
                knn_pos = torch.gather(
                    pos.float().unsqueeze(1).expand(-1, N, -1, -1),        # [B, N, N, 2] view
                    dim=2, index=idx_exp_pos,
                )                                                          # [B, N, k, 2]
                rel_pos = knn_pos - pos.float().unsqueeze(2)               # [B, N, k, 2]
                edge_feat = torch.cat([rel_pos, knn_dist.unsqueeze(-1)], dim=-1)  # [B, N, k, 3]
        edge_h = self.edge_enc(edge_feat)                              # [B, N, k, H]

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = checkpoint(layer, h, edge_h, knn_idx, use_reentrant=False)
            else:
                h = layer(h, edge_h, knn_idx)

        out = self.decoder(h)                                          # [B, N, out_dim]
        out = out * mask.unsqueeze(-1)                                 # zero padded rows
        return {"preds": out}


# ---------------------------------------------------------------------------
# H90: stratified node subsample for GeoMPNN training
# ---------------------------------------------------------------------------


def stratified_subsample_batch(
    x: torch.Tensor, y: torch.Tensor,
    is_surface: torch.Tensor, mask: torch.Tensor,
    n_keep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample stratified subsample to ``n_keep`` nodes.

    Keeps *all* valid surface nodes (small set, primary-metric target) plus a
    uniform-random subset of valid volume nodes filling the rest of the budget.
    Output is padded to ``n_keep`` along dim 1 with ``mask=False`` for short
    samples, so the existing scoring/loss masking continues to work unchanged.
    """
    B, N, X = x.shape
    Y = y.shape[-1]
    device = x.device
    out_x = torch.zeros(B, n_keep, X, dtype=x.dtype, device=device)
    out_y = torch.zeros(B, n_keep, Y, dtype=y.dtype, device=device)
    out_surf = torch.zeros(B, n_keep, dtype=torch.bool, device=device)
    out_mask = torch.zeros(B, n_keep, dtype=torch.bool, device=device)
    for b in range(B):
        valid = mask[b]
        surf = is_surface[b] & valid
        vol = valid & ~surf
        surf_idx = surf.nonzero(as_tuple=True)[0]
        vol_idx = vol.nonzero(as_tuple=True)[0]
        n_surf = int(surf_idx.numel())
        n_vol = int(vol_idx.numel())
        n_keep_surf = min(n_surf, n_keep)
        n_keep_vol = min(n_vol, n_keep - n_keep_surf)
        if n_vol > n_keep_vol:
            perm = torch.randperm(n_vol, device=device)[:n_keep_vol]
            vol_idx = vol_idx[perm]
        chosen = torch.cat([surf_idx[:n_keep_surf], vol_idx], dim=0)
        n_chosen = chosen.numel()
        out_x[b, :n_chosen] = x[b, chosen]
        out_y[b, :n_chosen] = y[b, chosen]
        out_surf[b, :n_chosen] = is_surface[b, chosen]
        out_mask[b, :n_chosen] = mask[b, chosen]
    return out_x, out_y, out_surf, out_mask


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   use_bf16_autocast: bool = False) -> dict[str, float]:
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
            # H90: GeoMPNN takes raw normalized x + pos + mask;
            # Transolver path goes through FourierCoordEnc as before.
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast,
            ):
                if fourier_enc is None:
                    pos = x_norm[..., :2].contiguous()
                    pred = model({"x": x_norm, "pos": pos, "mask": mask})["preds"]
                else:
                    x_norm_fe = fourier_enc(x_norm)
                    pred = model({"x": x_norm_fe})["preds"]
            if use_bf16_autocast:
                pred = pred.float()

            abs_err = (pred - y_norm).abs()
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (abs_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            # H18: per-channel surf-loss weighting (mirrors training loop).
            surf_ch_weights = abs_err.new_tensor([0.5, 0.5, 2.0])
            surf_loss_sum += (
                ((abs_err * surf_ch_weights) * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            B = y.shape[0]
            finite_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if finite_sample.any():
                idx = finite_sample.nonzero(as_tuple=True)[0]
                ds, dv = accumulate_batch(
                    pred_orig[idx], y[idx], is_surface[idx], mask[idx],
                    mae_surf, mae_vol,
                )
            else:
                ds, dv = 0, 0
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


def measure_attention_entropies(
    model, val_loader, stats, fourier_enc, device, n_batches: int = 4,
) -> tuple[list[float], list[float]]:
    """Measure mean and per-head-min slice-to-slice attention entropy per block.

    Toggles ``PhysicsAttention.record_entropy`` on, runs ``n_batches`` of
    ``val_loader`` in eval mode (no_grad), accumulates the per-block
    ``last_attn_entropy_*`` scalars, then disables recording. Cheap enough
    to invoke once per probe epoch.
    """
    was_training = model.training
    model.eval()
    PhysicsAttention.record_entropy = True
    n_blocks = len(model.blocks)
    sum_mean = [0.0] * n_blocks
    sum_min = [0.0] * n_blocks
    n = 0
    with torch.no_grad():
        for batch_idx, (x, _y, _is_surface, _mask) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            x = x.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            x_norm = fourier_enc(x_norm)
            _ = model({"x": x_norm})
            for i, block in enumerate(model.blocks):
                sum_mean[i] += block.attn.last_attn_entropy_mean or 0.0
                sum_min[i] += block.attn.last_attn_entropy_per_head_min or 0.0
            n += 1
    PhysicsAttention.record_entropy = False
    if was_training:
        model.train()
    denom = max(n, 1)
    return ([s / denom for s in sum_mean], [s / denom for s in sum_min])


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
    # H90: model-class switch — "transolver" (default) keeps the established
    # baseline path; "geompnn" swaps in the geometry-aware message-passing GNN.
    model_class: str = "transolver"
    geompnn_hidden: int = 288
    geompnn_layers: int = 5
    geompnn_k: int = 16
    geompnn_chunk: int = 1024  # KNN query chunk size (memory tradeoff)
    geompnn_ls_init: float = 0.1
    geompnn_checkpoint: bool = True  # gradient checkpointing per MP layer (cuts memory ~5x at ~30% time cost)
    # H90: training-time stratified node subsample. KNN graph is O(N^2) per
    # batch; at N=242K it makes full-mesh training infeasible inside the 30 min
    # cap. We keep all surface nodes (~1-3% of N, primary metric) plus a random
    # half-budget of volume nodes, up to ``geompnn_train_subsample`` total per
    # sample. 0 disables subsampling. Eval/test never subsample.
    geompnn_train_subsample: int = 32768
    cosine_t_max_epochs: int = 14  # cosine T_max in epochs (post-warmup)
    # bf16 autocast (Blackwell-friendly) — used for GeoMPNN to keep the wide
    # edge tensors in half precision; Transolver path is unaffected when False.
    use_bf16_autocast: bool = False


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

N_FREQS = 6
USE_GEOMPNN = cfg.model_class == "geompnn"

# H90: for GeoMPNN we keep raw normalized x (24-dim, positions intact at
# dims 0-1 ready for the KNN graph). FourierCoordEnc is Transolver-only.
fourier_enc: FourierCoordEnc | None
if USE_GEOMPNN:
    fourier_enc = None
    print(f"[H90] model_class=geompnn — skipping FourierCoordEnc (raw normalized x).")
else:
    fourier_enc = FourierCoordEnc(n_freqs=N_FREQS).to(device)

if USE_GEOMPNN:
    model_config = dict(
        model_class="geompnn",
        in_dim=X_DIM,
        hidden=cfg.geompnn_hidden,
        out_dim=3,
        n_layers=cfg.geompnn_layers,
        k_neighbors=cfg.geompnn_k,
        ls_init=cfg.geompnn_ls_init,
        knn_chunk_size=cfg.geompnn_chunk,
        use_checkpoint=cfg.geompnn_checkpoint,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model_kwargs = {k: v for k, v in model_config.items() if k != "model_class"}
    model = GeoMPNN(**model_kwargs).to(device)
else:
    model_config = dict(
        model_class="transolver",
        space_dim=2,
        fun_dim=4 * N_FREQS + (X_DIM - 2) - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=4,
        slice_num=64,
        mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model_kwargs = {k: v for k, v in model_config.items() if k != "model_class"}
    model = Transolver(**model_kwargs).to(device)

n_model_params = sum(p.numel() for p in model.parameters())
n_fourier_params = sum(p.numel() for p in fourier_enc.parameters()) if fourier_enc is not None else 0
n_params = n_model_params + n_fourier_params
if USE_GEOMPNN:
    print(
        f"Model: GeoMPNN hidden={cfg.geompnn_hidden} layers={cfg.geompnn_layers} "
        f"k={cfg.geompnn_k} ({n_model_params/1e6:.2f}M params)"
    )
    print(f"n_params (total trainable): {n_params}")
    for i, layer in enumerate(model.layers):
        print(
            f"[H90] layer {i}: layer_scale init avg={layer.layer_scale.mean().item():.4f} "
            f"(target {cfg.geompnn_ls_init})"
        )
else:
    print(f"Model: Transolver ({n_model_params/1e6:.2f}M params) + FourierCoordEnc ({n_fourier_params} freqs)")
    print(f"n_params (total trainable): {n_params}")
    swiglu_inner_dim = model.blocks[0].mlp.inner_dim
    print(f"SwiGLU inner_dim: {swiglu_inner_dim}, total_params: {n_params}")
    # H39: ReGLU gate sanity check (ReLU = max(0,x))
    _h39_test_x = torch.tensor([-1.0, 0.0, 1.0])
    print(f"[H39] ReGLU gate at x=-1: {F.relu(_h39_test_x[0]).item():.4f} (expected 0.0000)")
    print(f"[H39] ReGLU gate at x= 0: {F.relu(_h39_test_x[1]).item():.4f} (expected 0.0000)")
    print(f"[H39] ReGLU gate at x=+1: {F.relu(_h39_test_x[2]).item():.4f} (expected 1.0000)")
    print(f"[H39] SwiGLU inner_dim: {model.blocks[0].mlp.inner_dim}, n_params: {n_params}")
    print(f"[H46] SwiGLU inner_dim: {model.blocks[0].mlp.inner_dim}")
    print(f"[H46] n_params: {n_params}")
    for i, b in enumerate(model.blocks):
        print(
            f"block {i}: layer_scale_attn init avg={b.layer_scale_attn.mean().item():.4f}, "
            f"layer_scale_mlp init avg={b.layer_scale_mlp.mean().item():.4f}"
        )

# H73: Transolver-only attention-temperature annealing. GeoMPNN has no
# attention so this entire block is skipped.
if USE_GEOMPNN:
    N_ANNEAL_EPOCHS = 0
else:
    N_ANNEAL_EPOCHS = 12
    _h73_dim_head = model_config["n_hidden"] // model_config["n_head"]
    _h73_init_factor = math.sqrt(3.0)
    _h73_final_factor = math.sqrt(2.0)
    _h73_init_scale = _h73_init_factor / math.sqrt(_h73_dim_head)
    _h73_final_scale = _h73_final_factor / math.sqrt(_h73_dim_head)
    _h73_default_scale = 1.0 / math.sqrt(_h73_dim_head)
    print(
        f"[H73] attn-temp anneal: linear sqrt(3)={_h73_init_factor:.4f} -> sqrt(2)={_h73_final_factor:.4f} "
        f"over {N_ANNEAL_EPOCHS} epochs (then clamped at sqrt(2)); "
        f"scale {_h73_init_scale:.4f} -> {_h73_final_scale:.4f} "
        f"(default scale would be {_h73_default_scale:.4f}, "
        f"dim_head={_h73_dim_head}, slice_num={model_config['slice_num']}, "
        f"max possible entropy=log({model_config['slice_num']})={math.log(model_config['slice_num']):.4f})"
    )
    for i, b in enumerate(model.blocks):
        print(
            f"[H73] block {i}: attn_sharpening_factor init = "
            f"{float(b.attn.attn_sharpening_factor.item()):.4f}"
        )

# H40/H54: LayerScale params (and learned Fourier freqs) go in a 10x-lr, no-WD
# param group. For GeoMPNN, layer-scale params live at ``model.layers[i].layer_scale``
# (one per layer); there are no Fourier freqs.
if USE_GEOMPNN:
    freq_params = []
    layer_scale_params = [p for n, p in model.named_parameters() if "layer_scale" in n]
    other_params = [p for n, p in model.named_parameters() if "layer_scale" not in n]
    expected_layer_scale = cfg.geompnn_layers * cfg.geompnn_hidden
else:
    freq_params = [p for n, p in fourier_enc.named_parameters() if n == "freqs"]
    layer_scale_params = [p for n, p in model.named_parameters() if "layer_scale" in n]
    other_params = [p for n, p in model.named_parameters() if "layer_scale" not in n]
    assert len(freq_params) == 1 and freq_params[0].numel() == N_FREQS, (
        f"Expected 1 freq tensor with {N_FREQS} elems, "
        f"got {len(freq_params)} with {[p.numel() for p in freq_params]}"
    )
    expected_layer_scale = 2 * len(model.blocks) * model_config["n_hidden"]

all_model_p = {id(p): p for p in model.parameters()}
all_assigned_p = {id(p): p for group in [layer_scale_params, other_params] for p in group}
assert set(all_model_p.keys()) == set(all_assigned_p.keys()), (
    f"Mismatch: {len(all_model_p)} model params, {len(all_assigned_p)} assigned"
)
n_layer_scale = sum(p.numel() for p in layer_scale_params)
assert n_layer_scale == expected_layer_scale, (
    f"Expected {expected_layer_scale} LayerScale params, got {n_layer_scale}"
)
_fourier_total = sum(p.numel() for p in fourier_enc.parameters()) if fourier_enc is not None else 0
assert sum(p.numel() for p in freq_params) + sum(p.numel() for p in layer_scale_params) + sum(p.numel() for p in other_params) == (
    sum(p.numel() for p in model.parameters()) + _fourier_total
)
opt_groups = [
    {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
    {"params": layer_scale_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0},
]
if freq_params:
    opt_groups.insert(1, {"params": freq_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0})
optimizer = torch.optim.AdamW(opt_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
if USE_GEOMPNN:
    print(
        f"[H90] Optimizer param groups: "
        f"other lr={optimizer.param_groups[0]['lr']:.2e} wd={optimizer.param_groups[0]['weight_decay']:.0e} "
        f"({sum(p.numel() for p in other_params)} params), "
        f"layerscale lr={optimizer.param_groups[1]['lr']:.2e} wd={optimizer.param_groups[1]['weight_decay']:.0e} "
        f"({n_layer_scale} params, expected {expected_layer_scale})"
    )
else:
    print(
        f"[H54] Optimizer param groups: "
        f"other lr={optimizer.param_groups[0]['lr']:.2e} wd={optimizer.param_groups[0]['weight_decay']:.0e} "
        f"({sum(p.numel() for p in other_params)} params), "
        f"freqs lr={optimizer.param_groups[1]['lr']:.2e} wd={optimizer.param_groups[1]['weight_decay']:.0e} "
        f"({sum(p.numel() for p in freq_params)} params), "
        f"layerscale lr={optimizer.param_groups[2]['lr']:.2e} wd={optimizer.param_groups[2]['weight_decay']:.0e} "
        f"({n_layer_scale} params, expected {expected_layer_scale})"
    )
    print(f"[H40] Initial freqs: {fourier_enc.freqs.detach().cpu().tolist()}")
# H19: linear warm-up over the first epoch (batches), then cosine annealing for the remaining epochs.
batches_per_epoch = len(train_loader)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=batches_per_epoch
)
cosine_t_max = cfg.cosine_t_max_epochs * batches_per_epoch
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cosine_t_max
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[batches_per_epoch],
)
print(f"LR schedule: linear warmup over {batches_per_epoch} batches (1 epoch), then cosine T_max={cosine_t_max} batches ({cfg.cosine_t_max_epochs} epochs)")

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

    # H73: Transolver-only attention-temperature anneal. Skipped for GeoMPNN.
    if USE_GEOMPNN:
        current_attn_factor = float("nan")
        anneal_frac = float("nan")
    else:
        anneal_frac = min(1.0, epoch / max(1, N_ANNEAL_EPOCHS - 1))
        current_attn_factor = math.sqrt(3.0) + anneal_frac * (math.sqrt(2.0) - math.sqrt(3.0))
        for block in model.blocks:
            block.attn.attn_sharpening_factor.fill_(current_attn_factor)
        print(
            f"[H73] Epoch {epoch+1}: attn_sharpening_factor = {current_attn_factor:.4f} "
            f"(frac={anneal_frac:.3f}, target sqrt(2)={math.sqrt(2.0):.4f})"
        )

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # H90: stratified subsample for GeoMPNN training only. KNN graph is
        # O(N^2) and full meshes (up to 242K nodes) don't fit the 30 min cap.
        # Eval and test runs always use full mesh — never subsampled.
        if USE_GEOMPNN and cfg.geompnn_train_subsample > 0 and x.shape[1] > cfg.geompnn_train_subsample:
            x, y, is_surface, mask = stratified_subsample_batch(
                x, y, is_surface, mask, cfg.geompnn_train_subsample,
            )

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16_autocast,
        ):
            if fourier_enc is None:
                pos = x_norm[..., :2].contiguous()
                pred = model({"x": x_norm, "pos": pos, "mask": mask})["preds"]
            else:
                x_norm_fe = fourier_enc(x_norm)
                pred = model({"x": x_norm_fe})["preds"]
        if cfg.use_bf16_autocast:
            pred = pred.float()
        abs_err = (pred - y_norm).abs()

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        # H18: per-channel surf-loss weighting. Mass-preserving (sum = 3.0).
        # Upweights pressure (channel 2 = p) which defines the primary metric val_avg/mae_surf_p.
        surf_ch_weights = abs_err.new_tensor([0.5, 0.5, 2.0])
        surf_loss = ((abs_err * surf_ch_weights) * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        # Clip model (and fourier_enc when present) params together.
        clip_params = list(model.parameters())
        if fourier_enc is not None:
            clip_params += list(fourier_enc.parameters())
        total_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=25.0)
        optimizer.step()
        if fourier_enc is not None:
            # H40: keep freqs bounded after the update.
            with torch.no_grad():
                fourier_enc.freqs.clamp_(0.1, 100.0)
        scheduler.step()

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                             use_bf16_autocast=cfg.use_bf16_autocast)
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
    current_lr = optimizer.param_groups[0]["lr"]
    # Param group layout depends on model class:
    #   Transolver: [other, freqs, layer_scale]
    #   GeoMPNN:    [other, layer_scale]
    if USE_GEOMPNN:
        freqs_lr = float("nan")
        layer_scale_lr = optimizer.param_groups[1]["lr"]
        freqs_now: list[float] = []
    else:
        freqs_lr = optimizer.param_groups[1]["lr"]
        layer_scale_lr = optimizer.param_groups[2]["lr"]
        freqs_now = fourier_enc.freqs.detach().cpu().tolist()
    epoch_record = {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": current_lr,
        "train/current_lr": current_lr,
        "train/freqs_lr": freqs_lr,
        "train/layer_scale_lr": layer_scale_lr,
        "train/freqs": freqs_now,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/last_grad_norm": float(total_norm),
        "train/model_class": "geompnn" if USE_GEOMPNN else "transolver",
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    }
    if not USE_GEOMPNN:
        epoch_record["train/attn_sharpening_factor"] = current_attn_factor
        epoch_record["train/attn_anneal_frac"] = anneal_frac
        epoch_record["train/attn_anneal_n_epochs"] = N_ANNEAL_EPOCHS
    append_metrics_jsonl(metrics_jsonl_path, epoch_record)
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    if freqs_now:
        print(
            f"    freqs (lr={freqs_lr:.2e}): "
            f"[{', '.join(f'{v:.4f}' for v in freqs_now)}]"
        )
    print(f"    layer_scale (lr={layer_scale_lr:.2e})")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

    # H67: Transolver attention-entropy probe — GeoMPNN has no attention.
    if (not USE_GEOMPNN) and (epoch + 1) in (1, 6, 12):
        probe_loader = val_loaders["val_single_in_dist"]
        ent_means, ent_min_per_head = measure_attention_entropies(
            model, probe_loader, stats, fourier_enc, device, n_batches=4,
        )
        max_entropy = math.log(model_config["slice_num"])
        ent_ratios = [e / max_entropy for e in ent_means]
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "attention_entropy",
            "epoch": epoch + 1,
            "max_entropy": max_entropy,
            "per_block_entropy_mean": ent_means,
            "per_block_entropy_min_per_head": ent_min_per_head,
            "per_block_entropy_ratio": ent_ratios,
        })
        print(
            f"    [H67] attn entropy (max=log(64)={max_entropy:.4f}): "
            + " ".join(
                f"b{i}={ent_means[i]:.4f}({ent_ratios[i]*100:.1f}%, min/head={ent_min_per_head[i]:.4f})"
                for i in range(len(ent_means))
            )
        )

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Log final per-block LayerScale stats (end-of-training state) ---
final_layer_scale_stats: dict[str, float] = {}
if USE_GEOMPNN:
    for i, layer in enumerate(model.layers):
        final_layer_scale_stats[f"final/layer_scale_l{i}_mean"] = float(layer.layer_scale.mean().item())
        final_layer_scale_stats[f"final/layer_scale_l{i}_std"] = float(layer.layer_scale.std().item())
    freqs_summary: dict = {}
else:
    for i, b in enumerate(model.blocks):
        final_layer_scale_stats[f"final/layer_scale_attn_l{i}_mean"] = float(b.layer_scale_attn.mean().item())
        final_layer_scale_stats[f"final/layer_scale_attn_l{i}_std"] = float(b.layer_scale_attn.std().item())
        final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_mean"] = float(b.layer_scale_mlp.mean().item())
        final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_std"] = float(b.layer_scale_mlp.std().item())
    final_freqs = fourier_enc.freqs.detach().cpu().tolist()
    init_freqs = [2.0**k for k in range(N_FREQS)]
    freq_drift = [f - i for f, i in zip(final_freqs, init_freqs)]
    freq_rel_drift = [(f - i) / i for f, i in zip(final_freqs, init_freqs)]
    freqs_summary = {
        "final/freqs": final_freqs,
        "final/freqs_init": init_freqs,
        "final/freqs_abs_drift": freq_drift,
        "final/freqs_rel_drift": freq_rel_drift,
    }
append_metrics_jsonl(
    metrics_jsonl_path,
    {"event": "final", **final_layer_scale_stats, **freqs_summary},
)
print("Final LayerScale stats (end-of-training):")
if USE_GEOMPNN:
    for i in range(len(model.layers)):
        print(
            f"  layer {i}: mean={final_layer_scale_stats[f'final/layer_scale_l{i}_mean']:.4f} "
            f"std={final_layer_scale_stats[f'final/layer_scale_l{i}_std']:.4f}"
        )
else:
    for i in range(len(model.blocks)):
        print(
            f"  block {i}: attn mean={final_layer_scale_stats[f'final/layer_scale_attn_l{i}_mean']:.4f} "
            f"std={final_layer_scale_stats[f'final/layer_scale_attn_l{i}_std']:.4f}  "
            f"mlp mean={final_layer_scale_stats[f'final/layer_scale_mlp_l{i}_mean']:.4f} "
            f"std={final_layer_scale_stats[f'final/layer_scale_mlp_l{i}_std']:.4f}"
        )
    print("[H40] Final learned freqs vs dyadic init:")
    for k in range(N_FREQS):
        print(
            f"  freq[{k}]: init={init_freqs[k]:.4f} -> final={final_freqs[k]:.4f} "
            f"(drift {freq_drift[k]:+.4f}, {freq_rel_drift[k]*100:+.2f}%)"
        )

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

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
                                 use_bf16_autocast=cfg.use_bf16_autocast)
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
