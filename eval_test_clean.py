"""NaN-safe test re-evaluation.

The shared ``data/scoring.py::accumulate_batch`` masks samples whose GT
contains non-finite values, but it computes ``err = (pred - y).abs()`` BEFORE
multiplying by the mask. IEEE-754 ``Inf * 0 = NaN``, so the cruise test split
(which has 761 ``Inf`` GT pressure values in sample 20) contaminates the
accumulator and produces ``NaN`` for ``test_avg/mae_surf_p``.

This script re-evaluates the saved best-val EMA checkpoint with a one-line
NaN-safe replacement (``err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)`` after
the subtraction) and appends a clean ``event=test_clean`` record into
``metrics.jsonl``.

Usage:
  python eval_test_clean.py --model_dir models/<experiment-dir>

The Transolver class is duplicated here intentionally — importing ``train``
runs the trainer at module-import time, so we keep this script standalone.
The duplicated class is byte-for-byte the same construction used by
``train.py`` so the loaded checkpoint matches exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import TEST_SPLIT_NAMES, load_test_data, pad_collate


CHANNELS = ("Ux", "Uy", "p")


# ---------------------------------------------------------------------------
# Transolver model definition (mirrors train.py — kept in sync intentionally).
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
# NaN-safe accumulation.
# ---------------------------------------------------------------------------

def nan_safe_accumulate(
    pred_orig: torch.Tensor,
    y: torch.Tensor,
    is_surface: torch.Tensor,
    mask: torch.Tensor,
    mae_surf: torch.Tensor,
    mae_vol: torch.Tensor,
) -> tuple[int, int]:
    """Match data.scoring.accumulate_batch but zero out non-finite errors.

    Whole-sample skip is kept; the only change is
    ``err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)`` so the sample-mask
    multiplication can't produce ``NaN`` via ``Inf * 0``.
    """
    B = y.shape[0]
    y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
    if not y_finite.any():
        return 0, 0
    sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
    effective = mask & sample_mask
    surf_mask = effective & is_surface
    vol_mask = effective & ~is_surface
    err = (pred_orig.double() - y.double()).abs()
    err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
    mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    return int(surf_mask.sum().item()), int(vol_mask.sum().item())


def finalize_split(mae_surf, mae_vol, n_surf, n_vol) -> dict[str, float]:
    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    out: dict[str, float] = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = s[i].item()
        out[f"mae_vol_{ch}"] = v[i].item()
    return out


def evaluate_clean(model, loader, stats, device) -> dict[str, float]:
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = nan_safe_accumulate(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.yaml") as f:
        run_cfg = yaml.safe_load(f)
    model_cfg = run_cfg["model_config"]
    print(f"Model config: {model_cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_datasets = load_test_data(args.splits_dir)

    stats_path = Path(args.splits_dir) / "stats.json"
    with open(stats_path) as f:
        raw = json.load(f)
    stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items()}

    model = Transolver(**model_cfg).to(device).eval()
    ckpt_path = model_dir / "checkpoint.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")

    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, ds in test_datasets.items()
    }

    per_split = {name: evaluate_clean(model, loader, stats, device)
                 for name, loader in test_loaders.items()}

    finite_p = [m["mae_surf_p"] for m in per_split.values() if math.isfinite(m["mae_surf_p"])]
    all_p = [m["mae_surf_p"] for m in per_split.values()]
    clean_avg_surf_p = sum(finite_p) / len(finite_p) if finite_p else float("nan")
    raw_avg_surf_p = sum(all_p) / len(all_p) if all_p else float("nan")

    out = {
        "event": "test_clean",
        "test_clean_avg/mae_surf_p_finite_mean": clean_avg_surf_p,
        "test_clean_avg/mae_surf_p_raw_mean": raw_avg_surf_p,
        "test_clean_splits": per_split,
        "finite_splits": [n for n, m in per_split.items() if math.isfinite(m["mae_surf_p"])],
    }
    metrics_jsonl = model_dir / "metrics.jsonl"
    with open(metrics_jsonl, "a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")

    print("\nNaN-safe per-split surface pressure MAE:")
    for n in TEST_SPLIT_NAMES:
        m = per_split[n]
        print(f"  {n:<30s}  mae_surf_p={m['mae_surf_p']:.4f}  "
              f"mae_surf_Ux={m['mae_surf_Ux']:.4f}  mae_surf_Uy={m['mae_surf_Uy']:.4f}")
    print(f"\ntest_clean_avg/mae_surf_p (finite mean, n={len(finite_p)}/4): {clean_avg_surf_p:.4f}")
    print(f"test_clean_avg/mae_surf_p (raw mean, n={len(all_p)}/4):    {raw_avg_surf_p:.4f}")


if __name__ == "__main__":
    main()
