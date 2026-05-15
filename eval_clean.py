"""Post-hoc clean test re-evaluation (standalone).

Re-runs the held-out test evaluation for a saved checkpoint, but masks
non-finite ground-truth entries per-channel BEFORE accumulation. This sidesteps
the pre-existing `0 * NaN = NaN` bug in `data/scoring.py` where samples with
non-finite y entries propagate NaN through the masked product.

Affected sample observed: test_geom_camber_cruise/.test_geom_camber_cruise_gt/000020.pt
has 761 NaN values in the p channel of y.

This script is self-contained (no `train.py` import) so importing it does not
trigger the trainer's module-level data loading / arg parsing.

Usage:
    uv run python eval_clean.py models/model-<id>/checkpoint.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import TEST_SPLIT_NAMES, X_DIM, load_test_data, pad_collate

SPLITS_DIR = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
CHANNELS = ("Ux", "Uy", "p")


# --- Minimal inline Transolver (architecture must match train.py exactly) ---

ACTIVATION = {"gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
              "relu": nn.ReLU, "softplus": nn.Softplus, "ELU": nn.ELU, "silu": nn.SiLU}


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
        fx_mid = (self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, out_dim))

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
            TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                            slice_num=slice_num, last_layer=(i == n_layers - 1))
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


# --- Clean evaluation (per-channel non-finite masking) ---


def _load_stats(splits_dir: Path):
    with open(splits_dir / "stats.json") as f:
        raw = json.load(f)
    return {k: torch.tensor(raw[k], dtype=torch.float32) for k in ("x_mean", "x_std", "y_mean", "y_std")}


def clean_accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol,
                           n_surf_per_ch, n_vol_per_ch):
    """Per-channel non-finite-aware MAE accumulation.

    Unlike `data.scoring.accumulate_batch`, this masks non-finite (y or pred)
    per-channel, per-node BEFORE the masked sum so 0*NaN cannot propagate.
    Counts valid surface/volume nodes PER CHANNEL because a sample may have
    NaN in some channels but not others.
    """
    err = (pred_orig.double() - y.double()).abs()
    valid_node = torch.isfinite(err)  # [B, N, 3]
    err = torch.where(valid_node, err, torch.zeros_like(err))

    surf_mask = (mask & is_surface).unsqueeze(-1).double()
    vol_mask = (mask & ~is_surface).unsqueeze(-1).double()
    valid_d = valid_node.double()

    mae_surf += (err * surf_mask).sum(dim=(0, 1))
    mae_vol += (err * vol_mask).sum(dim=(0, 1))
    n_surf_per_ch += (surf_mask * valid_d).sum(dim=(0, 1))
    n_vol_per_ch += (vol_mask * valid_d).sum(dim=(0, 1))


def clean_evaluate_split(model, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf_per_ch = torch.zeros(3, dtype=torch.float64, device=device)
    n_vol_per_ch = torch.zeros(3, dtype=torch.float64, device=device)
    skipped_samples_by_ch = [0, 0, 0]

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            for b in range(y.shape[0]):
                for c in range(3):
                    if not torch.isfinite(y[b, :, c]).all():
                        skipped_samples_by_ch[c] += 1

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            clean_accumulate_batch(pred_orig, y, is_surface, mask,
                                   mae_surf, mae_vol,
                                   n_surf_per_ch, n_vol_per_ch)

    out = {}
    for i, ch in enumerate(CHANNELS):
        ns = n_surf_per_ch[i].clamp(min=1).item()
        nv = n_vol_per_ch[i].clamp(min=1).item()
        out[f"mae_surf_{ch}"] = (mae_surf[i] / ns).item()
        out[f"mae_vol_{ch}"] = (mae_vol[i] / nv).item()
        out[f"n_surf_{ch}"] = int(n_surf_per_ch[i].item())
        out[f"n_vol_{ch}"] = int(n_vol_per_ch[i].item())
    out["nan_y_samples_by_ch"] = skipped_samples_by_ch
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: eval_clean.py <checkpoint.pt> [splits_dir]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    splits_dir = Path(sys.argv[2] if len(sys.argv) > 2 else SPLITS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {k: v.to(device) for k, v in _load_stats(splits_dir).items()}

    model_config = dict(
        space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=128,
        n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    test_datasets = load_test_data(str(splits_dir))
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=2, pin_memory=True)
    test_loaders = {
        name: DataLoader(ds, batch_size=4, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    per_split = {}
    for name in TEST_SPLIT_NAMES:
        print(f"\n--- {name} ---")
        m = clean_evaluate_split(model, test_loaders[name], stats, device)
        per_split[name] = m
        for ch in CHANNELS:
            print(f"  mae_surf_{ch:2s}={m[f'mae_surf_{ch}']:.4f}  "
                  f"(n_surf={m[f'n_surf_{ch}']:>8d})  "
                  f"mae_vol_{ch:2s}={m[f'mae_vol_{ch}']:.4f}  "
                  f"(n_vol={m[f'n_vol_{ch}']:>8d})")
        print(f"  nan_y_samples_by_ch={m['nan_y_samples_by_ch']}")

    print("\n=== Aggregate (clean) ===")
    keys = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in CHANNELS]
    for k in keys:
        vals = [m[k] for m in per_split.values()]
        print(f"  avg/{k}: {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
