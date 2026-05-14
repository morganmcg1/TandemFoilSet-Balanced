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
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    SplitDataset,
    TestDataset,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)

# ---------------------------------------------------------------------------
# H78: Laplacian eigenvector positional encoding from KNN graph.
# ---------------------------------------------------------------------------

H78_K_NEIGHBORS = 8
H78_N_EIG = 8
H78_CACHE_DIR = Path("data/lappe_cache")


def compute_laplacian_pe_np(coords_np: np.ndarray, k: int = H78_K_NEIGHBORS,
                            n_eig: int = H78_N_EIG) -> np.ndarray:
    """Compute Laplacian eigenvector PE on a KNN graph over node coords.

    L_sym = I - D^(-1/2) A D^(-1/2), then bottom n_eig+1 eigenpairs via
    shift-invert ARPACK (``eigsh(..., which='LM', sigma=0.0)``), trivial
    eigenvector dropped. Output is per-sample standardised to zero-mean /
    unit-std per PE dim so its scale is invariant to N.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    from scipy.spatial import cKDTree

    N = coords_np.shape[0]
    tree = cKDTree(coords_np)
    _, nbr = tree.query(coords_np, k=k + 1, workers=1)
    nbr = nbr[:, 1:]  # drop self
    rows = np.repeat(np.arange(N, dtype=np.int64), k)
    cols = nbr.ravel().astype(np.int64)
    A = sp.csr_matrix((np.ones(N * k, dtype=np.float64), (rows, cols)), shape=(N, N))
    A = A.maximum(A.T)

    n_comp, labels = connected_components(A, directed=False)
    if n_comp > 1:
        # Chain-connect one node per component so the Laplacian is irreducible.
        comp_first = np.array([np.where(labels == c)[0][0] for c in range(n_comp)],
                              dtype=np.int64)
        extra_rows = comp_first[1:]
        extra_cols = np.full(n_comp - 1, comp_first[0], dtype=np.int64)
        extra_data = np.ones(n_comp - 1, dtype=np.float64)
        A_extra = sp.csr_matrix((extra_data, (extra_rows, extra_cols)), shape=(N, N))
        A = A + A_extra + A_extra.T
        A.data = np.minimum(A.data, 1.0)

    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    D = sp.diags(d_inv)
    L = sp.eye(N) - D @ A @ D
    L = 0.5 * (L + L.T)

    try:
        evals, evecs = eigsh(L.tocsc(), k=n_eig + 1, which="LM", sigma=0.0,
                             tol=1e-6, maxiter=3000)
        order = np.argsort(evals)
        evecs = evecs[:, order]
        pe = evecs[:, 1:n_eig + 1]
    except Exception as exc:  # pragma: no cover — eigsh fallback
        print(f"[H78] eigsh failed at N={N} ({exc!s}); using zero PE")
        pe = np.zeros((N, n_eig), dtype=np.float64)

    # Per-sample standardisation: scale invariant to N.
    pe = pe - pe.mean(axis=0, keepdims=True)
    pe = pe / (pe.std(axis=0, keepdims=True) + 1e-8)
    return pe.astype(np.float32)


def _precompute_worker(args):
    src_path, cache_path, k, n_eig = args
    if cache_path.exists():
        return cache_path, 0.0, True
    s = torch.load(src_path, weights_only=True)
    coords = s["x"][:, :2].numpy().astype(np.float64)
    t0 = time.time()
    pe_np = compute_laplacian_pe_np(coords, k=k, n_eig=n_eig)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(pe_np), cache_path)
    return cache_path, time.time() - t0, False


def precompute_lappe(splits_dir: Path, k: int, n_eig: int, n_workers: int,
                     val_names: list[str], test_names: list[str],
                     debug: bool = False, debug_per_split: int = 4) -> None:
    """Parallel precompute of LapPE for every train/val/test sample.

    Cache files live at ``H78_CACHE_DIR/<split>/<sample>.pt`` keyed by the
    underlying sample filename (matches the sorted-glob order used by
    ``SplitDataset`` and ``TestDataset``).
    """
    tasks: list[tuple[Path, Path, int, int]] = []
    train_files = sorted((splits_dir / "train").glob("*.pt"))
    if debug:
        train_files = train_files[:debug_per_split + 2]
    for f in train_files:
        tasks.append((f, H78_CACHE_DIR / "train" / f.name, k, n_eig))
    for name in val_names + test_names:
        files = sorted((splits_dir / name).glob("*.pt"))
        if debug:
            files = files[:debug_per_split]
        for f in files:
            tasks.append((f, H78_CACHE_DIR / name / f.name, k, n_eig))

    pending = [t for t in tasks if not t[1].exists()]
    print(
        f"[H78] LapPE precompute: {len(tasks)} samples (k={k}, n_eig={n_eig}), "
        f"{len(tasks) - len(pending)} cached, {len(pending)} pending, "
        f"n_workers={n_workers}"
    )
    if not pending:
        return

    t0 = time.time()
    # Fork ctx: children inherit the loaded module without re-running its
    # top-level setup (which would itself call precompute_lappe -> fork bomb).
    # Safe here because no CUDA context exists yet at this point.
    with mp.get_context("fork").Pool(n_workers) as pool:
        for i, (cache_path, elapsed, _) in enumerate(
            pool.imap_unordered(_precompute_worker, pending, chunksize=1)
        ):
            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                eta = (len(pending) - (i + 1)) / max(rate, 1e-6)
                print(
                    f"[H78] precompute {i+1}/{len(pending)} "
                    f"({rate:.2f}/s, eta {eta:.0f}s)"
                )
    print(f"[H78] LapPE precompute done in {time.time() - t0:.0f}s")


class LapPEDataset(Dataset):
    """Wrap a SplitDataset/TestDataset and append cached LapPE columns to x.

    Returns ``(x_aug, y, is_surface)`` where ``x_aug = cat([x, pe], -1)`` is
    ``[N, X_DIM + n_eig]``. We rely on ``data.pad_collate`` which infers the
    feature dim from ``xs[0].shape[1]`` — padding regions therefore get zero
    PE entries, matched by the existing mask.
    """

    def __init__(self, base_ds, cache_dir: Path, n_eig: int, k: int,
                 preload: bool = True):
        self.base_ds = base_ds
        self.cache_dir = Path(cache_dir)
        self.n_eig = n_eig
        self.k = k
        # Mirror SplitDataset / TestDataset filename ordering.
        if hasattr(base_ds, "files"):
            self.files = list(base_ds.files)
        else:
            self.files = list(base_ds.x_files)

        # Preload all PE tensors into memory in the parent process. With the
        # Linux fork-based DataLoader workers, the tensor data buffers stay
        # shared via copy-on-write — we don't pay 4× the memory or the disk
        # I/O cost on every __getitem__. Each PE tensor is small (N×8 fp32),
        # so total memory is on the order of a few GB across all splits.
        self._pe_cache: list[torch.Tensor | None] = []
        if preload:
            for f in self.files:
                pe_path = self.cache_dir / f.name
                if pe_path.exists():
                    self._pe_cache.append(
                        torch.load(pe_path, weights_only=True).float()
                    )
                else:
                    self._pe_cache.append(None)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, sf = self.base_ds[idx]
        pe = self._pe_cache[idx] if idx < len(self._pe_cache) else None
        if pe is None:
            pe_path = self.cache_dir / self.files[idx].name
            if pe_path.exists():
                pe = torch.load(pe_path, weights_only=True).float()
            else:
                # Fallback: compute lazily (slow). Happens only if precompute
                # missed this sample, e.g. debug mode subsampling vs cache gap.
                coords = x[:, :2].numpy().astype(np.float64)
                pe = torch.from_numpy(
                    compute_laplacian_pe_np(coords, k=self.k, n_eig=self.n_eig)
                )
                pe_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pe, pe_path)
        return torch.cat([x, pe], dim=-1), y, sf

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
        # H67: fixed sharper temperature (factor of sqrt(2) sharper than the
        # default 1/sqrt(d_head) scale of F.scaled_dot_product_attention).
        sharper_scale = 1.0 / math.sqrt(self.dim_head / 2.0)
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
# Evaluation helpers
# ---------------------------------------------------------------------------

def prepare_input(x: torch.Tensor, stats, fourier_enc, sign_flip: bool) -> torch.Tensor:
    """H78 input pipeline. Loader feeds [B, N, X_DIM + n_eig]; split into the
    raw 24-dim feature vector + cached 8-dim LapPE, normalize and Fourier-encode
    the raw block, optionally sign-flip each PE dim per sample (train only),
    then concatenate. Padding stays as zero in PE columns.
    """
    x_raw = x[..., :X_DIM]
    pe = x[..., X_DIM:X_DIM + H78_N_EIG]
    x_norm = (x_raw - stats["x_mean"]) / stats["x_std"]
    x_norm = fourier_enc(x_norm)
    if sign_flip:
        flip = (torch.randint(0, 2, (pe.shape[0], 1, pe.shape[-1]),
                              device=pe.device, dtype=pe.dtype) * 2 - 1)
        pe = pe * flip
    return torch.cat([x_norm, pe], dim=-1)


def evaluate_split(model, loader, stats, surf_weight, device) -> dict[str, float]:
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

            x_in = prepare_input(x, stats, fourier_enc, sign_flip=False)
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_in})["preds"]

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
            x_in = prepare_input(x, stats, fourier_enc, sign_flip=False)
            _ = model({"x": x_in})
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


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

# H78: precompute Laplacian eigenvector PE for every sample (cached on disk).
# 15 CPU cores available; leave 1 for the main process. Precompute happens
# before CUDA init, so workers don't compete with dataloader CPU workers.
n_precompute_workers = max(1, min(14, (os.cpu_count() or 8) - 1))
precompute_lappe(
    splits_dir=Path(cfg.splits_dir),
    k=H78_K_NEIGHBORS,
    n_eig=H78_N_EIG,
    n_workers=n_precompute_workers,
    val_names=VAL_SPLIT_NAMES,
    test_names=TEST_SPLIT_NAMES,
    debug=cfg.debug,
)

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

# H78: wrap datasets so __getitem__ appends [N, n_eig] PE columns to x.
train_ds = LapPEDataset(train_ds, H78_CACHE_DIR / "train", n_eig=H78_N_EIG, k=H78_K_NEIGHBORS)
val_splits = {
    name: LapPEDataset(ds, H78_CACHE_DIR / name, n_eig=H78_N_EIG, k=H78_K_NEIGHBORS)
    for name, ds in val_splits.items()
}

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
fourier_enc = FourierCoordEnc(n_freqs=N_FREQS).to(device)

# H78: +n_eig dims for Laplacian eigenvector PE appended after fourier_enc.
model_config = dict(
    space_dim=2,
    fun_dim=4 * N_FREQS + (X_DIM - 2) - 2 + H78_N_EIG,
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
n_model_params = sum(p.numel() for p in model.parameters())
n_fourier_params = sum(p.numel() for p in fourier_enc.parameters())
n_params = n_model_params + n_fourier_params  # includes 6 learnable freqs
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
print(f"[H46] SwiGLU inner_dim: {model.blocks[0].mlp.inner_dim}")  # should print 288
print(f"[H46] n_params: {n_params}")  # should print ~892,631
# H78: Laplacian PE verification — input dim widened by n_eig, preprocess +n_eig*2*n_hidden params.
_h78_input_dim = model_config["space_dim"] + model_config["fun_dim"]
_h78_preprocess_added = H78_N_EIG * (model_config["n_hidden"] * 2)
print(
    f"[H78] Laplacian PE: KNN k={H78_K_NEIGHBORS}, n_eig={H78_N_EIG}, "
    f"input dim {_h78_input_dim - H78_N_EIG} -> {_h78_input_dim} "
    f"(preprocess MLP gains +{_h78_preprocess_added} params); "
    f"n_params now {n_params}"
)
for i, b in enumerate(model.blocks):
    print(
        f"block {i}: layer_scale_attn init avg={b.layer_scale_attn.mean().item():.4f}, "
        f"layer_scale_mlp init avg={b.layer_scale_mlp.mean().item():.4f}"
    )

# H67: fixed sharper attention temperature (factor sqrt(2) sharper than the
# F.scaled_dot_product_attention default scale 1/sqrt(d_head)).
_h67_dim_head = model_config["n_hidden"] // model_config["n_head"]
_h67_sharper_scale = 1.0 / math.sqrt(_h67_dim_head / 2.0)
_h67_default_scale = 1.0 / math.sqrt(_h67_dim_head)
print(
    f"[H67] attention scale: {_h67_sharper_scale:.4f} "
    f"(default would be {_h67_default_scale:.4f}, factor sqrt(2)={math.sqrt(2):.4f}, "
    f"dim_head={_h67_dim_head}, slice_num={model_config['slice_num']}, "
    f"max possible entropy=log({model_config['slice_num']})={math.log(model_config['slice_num']):.4f})"
)

# H40: learned Fourier freqs in a separate param group with 10x lr and no weight decay.
# H54: LayerScale params (layer_scale_attn + layer_scale_mlp across 5 blocks = 10 tensors x 128
# channels = 1280 params) also moved to a 10x lr, no-WD group. Same insight as H40: WD pulls
# scaling factors toward zero (init=0.025), fighting the model's learned per-channel
# diversification, and default lr=5e-4 under-trains them.
freq_params = [p for n, p in fourier_enc.named_parameters() if n == "freqs"]
layer_scale_params = [p for n, p in model.named_parameters() if "layer_scale" in n]
other_params = [p for n, p in model.named_parameters() if "layer_scale" not in n]
assert len(freq_params) == 1 and freq_params[0].numel() == N_FREQS, (
    f"Expected 1 freq tensor with {N_FREQS} elems, "
    f"got {len(freq_params)} with {[p.numel() for p in freq_params]}"
)
# H54: param-group consistency check — every model param ends up in exactly one of
# (layer_scale_params, other_params); freq_params come from fourier_enc separately.
all_model_p = {id(p): p for p in model.parameters()}
all_assigned_p = {id(p): p for group in [layer_scale_params, other_params] for p in group}
assert set(all_model_p.keys()) == set(all_assigned_p.keys()), (
    f"Mismatch: {len(all_model_p)} model params, {len(all_assigned_p)} assigned"
)
n_layer_scale = sum(p.numel() for p in layer_scale_params)
expected_layer_scale = 2 * len(model.blocks) * model_config["n_hidden"]
assert n_layer_scale == expected_layer_scale, (
    f"Expected {expected_layer_scale} LayerScale params "
    f"({len(model.blocks)} blocks x 2 paths x {model_config['n_hidden']} channels), got {n_layer_scale}"
)
assert sum(p.numel() for p in freq_params) + sum(p.numel() for p in layer_scale_params) + sum(p.numel() for p in other_params) == (
    sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in fourier_enc.parameters())
)
optimizer = torch.optim.AdamW(
    [
        {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": freq_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0},
        {"params": layer_scale_params, "lr": cfg.lr * 10.0, "weight_decay": 0.0},
    ],
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
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
# H19: linear warm-up over the first epoch (batches), then cosine annealing for the remaining 14.
# scheduler.step() is called once per BATCH below, so total_iters and T_max are expressed in batches.
batches_per_epoch = len(train_loader)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8, end_factor=1.0, total_iters=batches_per_epoch
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=14 * batches_per_epoch
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[batches_per_epoch],
)
print(f"LR schedule: linear warmup over {batches_per_epoch} batches (1 epoch), then cosine T_max={14 * batches_per_epoch} batches (14 epochs)")

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

        x_in = prepare_input(x, stats, fourier_enc, sign_flip=True)
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_in})["preds"]
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
        # H40: clip both model and fourier_enc params together (freqs included).
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(fourier_enc.parameters()), max_norm=25.0
        )
        optimizer.step()
        # H40: keep freqs bounded after the update. Safety net for the higher freqs lr.
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
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
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
    freqs_lr = optimizer.param_groups[1]["lr"]
    layer_scale_lr = optimizer.param_groups[2]["lr"]
    freqs_now = fourier_enc.freqs.detach().cpu().tolist()
    append_metrics_jsonl(metrics_jsonl_path, {
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
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    print(
        f"    freqs (lr={freqs_lr:.2e}): "
        f"[{', '.join(f'{v:.4f}' for v in freqs_now)}]"
    )
    print(f"    layer_scale (lr={layer_scale_lr:.2e})")
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

    # H67: probe slice-to-slice attention entropy at epochs 1, 6, 12 for
    # comparison against the post-#2475 baseline (default scale 1/sqrt(32)).
    if (epoch + 1) in (1, 6, 12):
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
for i, b in enumerate(model.blocks):
    final_layer_scale_stats[f"final/layer_scale_attn_l{i}_mean"] = float(b.layer_scale_attn.mean().item())
    final_layer_scale_stats[f"final/layer_scale_attn_l{i}_std"] = float(b.layer_scale_attn.std().item())
    final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_mean"] = float(b.layer_scale_mlp.mean().item())
    final_layer_scale_stats[f"final/layer_scale_mlp_l{i}_std"] = float(b.layer_scale_mlp.std().item())
# H40: final learned freqs values (compare to dyadic init 1, 2, 4, 8, 16, 32).
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

# H78: input_proj weight L1 norms — raw feature columns vs Laplacian PE columns.
# preprocess.linear_pre[0] is the first Linear in the model; its weight is
# [hidden*2, fun_dim+space_dim]. PE occupies the last n_eig columns; raw
# (fourier-encoded coords + remaining features) occupies the rest.
_h78_weight = model.preprocess.linear_pre[0].weight.detach()  # [hidden*2, in_dim]
_h78_in_dim = _h78_weight.shape[1]
_h78_raw_dim = _h78_in_dim - H78_N_EIG
_h78_col_l1 = _h78_weight.abs().sum(dim=0).cpu()  # per-input-column L1 across output neurons
_h78_raw_per_col = _h78_col_l1[:_h78_raw_dim]
_h78_pe_per_col = _h78_col_l1[_h78_raw_dim:]
h78_diag = {
    "h78/in_dim": _h78_in_dim,
    "h78/raw_dim": int(_h78_raw_dim),
    "h78/pe_dim": H78_N_EIG,
    "h78/raw_col_l1_mean": float(_h78_raw_per_col.mean()),
    "h78/raw_col_l1_std": float(_h78_raw_per_col.std()),
    "h78/pe_col_l1_mean": float(_h78_pe_per_col.mean()),
    "h78/pe_col_l1_std": float(_h78_pe_per_col.std()),
    "h78/pe_to_raw_l1_ratio": float(_h78_pe_per_col.mean() / max(_h78_raw_per_col.mean(), 1e-9)),
    "h78/pe_col_l1_per_dim": _h78_pe_per_col.tolist(),
    "h78/raw_col_l1_min_max": [float(_h78_raw_per_col.min()), float(_h78_raw_per_col.max())],
}
append_metrics_jsonl(metrics_jsonl_path, {"event": "h78_weight_diag", **h78_diag})
print(
    f"[H78] input_proj weight L1: "
    f"raw cols (n={int(_h78_raw_dim)}) mean={h78_diag['h78/raw_col_l1_mean']:.4f} "
    f"+/- {h78_diag['h78/raw_col_l1_std']:.4f}, "
    f"PE cols (n={H78_N_EIG}) mean={h78_diag['h78/pe_col_l1_mean']:.4f} "
    f"+/- {h78_diag['h78/pe_col_l1_std']:.4f} "
    f"(PE/raw ratio={h78_diag['h78/pe_to_raw_l1_ratio']:.3f})"
)
print(f"  per-PE-dim L1: {[f'{v:.3f}' for v in h78_diag['h78/pe_col_l1_per_dim']]}")

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
        # H78: wrap test datasets with the same PE-cache lookup.
        test_datasets = {
            name: LapPEDataset(ds, H78_CACHE_DIR / name, n_eig=H78_N_EIG, k=H78_K_NEIGHBORS)
            for name, ds in test_datasets.items()
        }
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
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
