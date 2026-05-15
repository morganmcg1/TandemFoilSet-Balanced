"""NaN-safe test re-evaluation for a saved EMA checkpoint.

Loads the EMA state dict, runs each test split, and computes per-channel
surface/volume MAE with NaN-masked positions zeroed out *before* the sum so a
single corrupt entry can't poison the channel-level aggregate.

Differences from ``data.scoring.accumulate_batch``:
* drops a sample only if its *valid* (mask=True) GT entries contain NaN
* applies ``torch.where(surf_mask, err, 0)`` BEFORE summation so NaN at padded /
  masked positions cannot propagate (``NaN * 0 == NaN`` otherwise).

We re-define the model classes here to avoid executing the top-level training
code in ``train.py``. The class bodies are kept byte-identical with ``train.py``
so the loaded state dict matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    load_data,
    load_test_data,
    pad_collate,
)
from data.scoring import CHANNELS, aggregate_splits


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
                 output_fields=None, output_dims=None):
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


def nan_safe_accumulate(pred_orig, y, is_surface, mask, mae_surf, mae_vol):
    """Per-channel accumulation robust to NaN in masked positions."""
    B = y.shape[0]
    mask_exp = mask.unsqueeze(-1)
    y_at_valid = torch.where(mask_exp, y, torch.zeros_like(y))
    y_finite = torch.isfinite(y_at_valid.reshape(B, -1)).all(dim=-1)
    n_dropped = int((~y_finite).sum().item())
    if not y_finite.any():
        return 0, 0, n_dropped

    sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
    effective = mask & sample_mask
    surf_mask = effective & is_surface
    vol_mask = effective & ~is_surface

    err = (pred_orig.double() - y.double()).abs()
    zeros = torch.zeros_like(err)
    err_surf = torch.where(surf_mask.unsqueeze(-1), err, zeros)
    err_vol = torch.where(vol_mask.unsqueeze(-1), err, zeros)
    mae_surf += err_surf.sum(dim=(0, 1))
    mae_vol += err_vol.sum(dim=(0, 1))
    return int(surf_mask.sum().item()), int(vol_mask.sum().item()), n_dropped


def finalize(mae_surf, mae_vol, n_surf, n_vol):
    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    out = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = s[i].item()
        out[f"mae_vol_{ch}"] = v[i].item()
    return out


def eval_split_clean(model, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_dropped = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv, nd = nan_safe_accumulate(
                pred_orig, y, is_surface, mask, mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv
            n_dropped += nd
    return finalize(mae_surf, mae_vol, n_surf, n_vol), n_dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    summary = yaml.safe_load((model_dir / "metrics.yaml").read_text())
    mc = summary["model_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    model = Transolver(
        space_dim=mc["space_dim"],
        fun_dim=mc["fun_dim"],
        out_dim=mc["out_dim"],
        n_hidden=mc["n_hidden"],
        n_layers=mc["n_layers"],
        n_head=mc["n_head"],
        slice_num=mc["slice_num"],
        mlp_ratio=mc["mlp_ratio"],
        output_fields=mc.get("output_fields"),
        output_dims=mc.get("output_dims"),
    ).to(device)
    state_dict = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and ("model_state_dict" in state_dict or "ema_state_dict" in state_dict):
        state_dict = state_dict.get("ema_state_dict") or state_dict.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    test_datasets = load_test_data(args.splits_dir, debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    out_splits = {}
    drops = {}
    for name in TEST_SPLIT_NAMES:
        m, nd = eval_split_clean(model, test_loaders[name], stats, device)
        out_splits[name] = m
        drops[name] = nd
        print(f"  {name:<28s} mae_surf_p={m['mae_surf_p']:.4f}  mae_surf_Ux={m['mae_surf_Ux']:.4f}  mae_surf_Uy={m['mae_surf_Uy']:.4f}  dropped={nd}")

    avg = aggregate_splits(out_splits)
    print(f"\n  TEST (clean)  avg/mae_surf_p={avg.get('avg/mae_surf_p', float('nan')):.4f}")

    out_path = model_dir / "test_metrics_clean.json"
    out_payload = {
        "test_splits_clean": out_splits,
        "test_avg_clean": avg,
        "dropped_samples": drops,
    }
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"Saved clean test metrics to {out_path}")

    metrics_jsonl = model_dir / "metrics.jsonl"
    record = {
        "event": "test_clean",
        "note": "NaN-safe re-eval: torch.where(mask, err, 0) before sum",
        "best_epoch": summary.get("best_epoch"),
        "test_avg": avg,
        "test_splits": out_splits,
        "dropped_samples": drops,
    }
    with open(metrics_jsonl, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Appended test_clean record to {metrics_jsonl}")


if __name__ == "__main__":
    main()
