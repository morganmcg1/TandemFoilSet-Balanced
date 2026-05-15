"""Re-evaluate held-out test splits with a NaN-safe MAE accumulator.

Background: ``data/scoring.py`` (read-only) computes
``err = abs(pred - y)`` and then multiplies by a boolean mask. When the
ground-truth ``y`` has any non-finite values, the resulting ``err`` carries
``NaN``/``Inf`` that survives the multiplication by 0 (NaN * 0 = NaN), corrupting
the channel-level accumulators even though the offending sample was meant to be
skipped via the per-sample finite check.

For ``test_geom_camber_cruise`` exactly one sample (``000020.pt``) has Inf
ground-truth pressure values, which corrupts ``mae_surf_p`` and ``mae_vol_p``
for the whole split, dragging ``test_avg/mae_surf_p`` to NaN.

This helper loads the saved EMA checkpoint and re-computes the test metrics with
``torch.nan_to_num`` applied to ``err`` before the mask multiplication. The
result corresponds to what scoring would produce if the read-only bug were
fixed. Use it strictly for reporting; do not modify ``data/scoring.py`` in this
PR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    X_DIM,
    finalize_split,
    load_test_data,
    pad_collate,
)


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --- Transolver definition mirrored from train.py for standalone eval ---
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
        fx_mid = (self.in_project_fx(x)
                  .reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x)
                 .reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False,
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


@torch.no_grad()
def evaluate_split_nan_safe(model, loader, stats, surf_weight, device):
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0
    n_skipped = 0

    for x, y, is_surface, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]

        # Loss in normalized space, NaN-safe over the masked nodes
        sq_err = (pred - y_norm) ** 2
        sq_err = torch.nan_to_num(sq_err, nan=0.0, posinf=0.0, neginf=0.0)
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

        # NaN-safe accumulation: skip per-sample if GT is non-finite anywhere,
        # AND zero out any residual NaN/Inf entries in err before the masked sum
        # so that NaN*0 cannot corrupt other channels.
        B = y.shape[0]
        y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
        n_skipped += int((~y_finite).sum().item())
        sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
        effective = mask & sample_mask
        surf_mask_eff = effective & is_surface
        vol_mask_eff = effective & ~is_surface

        err = (pred_orig.double() - y.double()).abs()
        err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
        mae_surf += (err * surf_mask_eff.unsqueeze(-1).double()).sum(dim=(0, 1))
        mae_vol += (err * vol_mask_eff.unsqueeze(-1).double()).sum(dim=(0, 1))
        n_surf += int(surf_mask_eff.sum().item())
        n_vol += int(vol_mask_eff.sum().item())

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {
        "vol_loss": vol_loss,
        "surf_loss": surf_loss,
        "loss": vol_loss + surf_weight * surf_loss,
        "n_skipped_samples": n_skipped,
    }
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_config = cfg["model_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {model_dir / 'checkpoint.pt'}")

    model = Transolver(**model_config).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    test_datasets = load_test_data(args.splits_dir)
    test_loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=4,
            pin_memory=True,
        )
        for name, ds in test_datasets.items()
    }

    # Stats from train pipeline (mirrors load_data())
    from data.loader import _load_stats

    stats = _load_stats(Path(args.splits_dir))
    stats = {k: v.to(device) for k, v in stats.items()}

    surf_weight = float(cfg.get("surf_weight", 10.0))

    test_metrics = {}
    for name in TEST_SPLIT_NAMES:
        loader = test_loaders[name]
        m = evaluate_split_nan_safe(model, loader, stats, surf_weight, device)
        test_metrics[name] = m
        print(f"  {name}: n_skipped={m['n_skipped_samples']}, "
              f"surf_p={m['mae_surf_p']:.4f}, "
              f"surf_Ux={m['mae_surf_Ux']:.4f}, surf_Uy={m['mae_surf_Uy']:.4f}, "
              f"vol_p={m['mae_vol_p']:.4f}, vol_Ux={m['mae_vol_Ux']:.4f}, "
              f"vol_Uy={m['mae_vol_Uy']:.4f}")

    # Aggregate
    keys = ["mae_surf_Ux", "mae_surf_Uy", "mae_surf_p",
            "mae_vol_Ux", "mae_vol_Uy", "mae_vol_p"]
    test_avg = {f"avg/{k}": sum(m[k] for m in test_metrics.values()) / len(test_metrics)
                for k in keys}
    print("\nNaN-safe test averages:")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")

    out_path = model_dir / "test_metrics_nan_safe.json"
    with open(out_path, "w") as f:
        json.dump({"test_metrics": test_metrics, "test_avg": test_avg,
                   "git_commit": _git_commit_short()},
                  f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
