"""Safe 4-split test re-evaluation.

Re-evaluates a saved checkpoint on the held-out test splits with per-node
masking of non-finite ground truth. This works around the bug in
``data/scoring.py`` (read-only) where ``Inf * 0 = NaN`` poisons the
per-channel surface accumulator on ``test_geom_camber_cruise`` (sample 20
has Inf in 761 volume-node pressure values).

Per-node skip (rather than per-sample skip) excludes only the 761 Inf
positions from both numerator and denominator, giving correct surface
AND volume MAE on all 4 splits.

Usage:
    python safe_test_eval.py --model_dir models/model-foo/
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

from data import TEST_SPLIT_NAMES, X_DIM, load_data, load_test_data, pad_collate

ACTIVATION = {"gelu": nn.GELU}


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


class SwiGLUFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim)
        self.w_value = nn.Linear(in_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


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
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, use_swiglu=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads,
                                     dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, dropout=dropout)
        else:
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
                 pos_freq_bands: int = 0,
                 pos_freq_surface_only: bool = False,
                 use_swiglu=False,
                 output_fields=None, output_dims=None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.pos_freq_bands = pos_freq_bands
        self.pos_freq_surface_only = pos_freq_surface_only
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        fourier_dim = 2 * space_dim * pos_freq_bands if pos_freq_bands > 0 else 0
        self.preprocess = MLP(fun_dim + space_dim + fourier_dim, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)
        if pos_freq_bands > 0:
            freqs = 2.0 ** torch.arange(pos_freq_bands, dtype=torch.float32)
            self.register_buffer("_fourier_freqs", freqs)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                            slice_num=slice_num, last_layer=(i == n_layers - 1),
                            use_swiglu=use_swiglu)
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

    def fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        proj = coords.unsqueeze(-1) * (2.0 * math.pi * self._fourier_freqs)
        return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(start_dim=-2)

    def forward(self, data, **kwargs):
        x = data["x"]
        if self.pos_freq_bands > 0:
            coords = x[..., :self.space_dim]
            fun = x[..., self.space_dim:]
            fourier = self.fourier_encode(coords)
            if self.pos_freq_surface_only:
                is_surface = data["is_surface"].to(fourier.dtype).unsqueeze(-1)
                fourier = fourier * is_surface
            x = torch.cat([coords, fourier, fun], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


@torch.no_grad()
def safe_evaluate_split(model, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf_per_ch = torch.zeros(3, dtype=torch.float64, device=device)
    n_vol_per_ch = torch.zeros(3, dtype=torch.float64, device=device)

    for x, y, is_surface, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
        pred_orig = pred * stats["y_std"] + stats["y_mean"]

        y_finite = torch.isfinite(y)  # [B, N, 3]
        eff_mask = mask.unsqueeze(-1) & y_finite
        surf_eff = eff_mask & is_surface.unsqueeze(-1)
        vol_eff = eff_mask & (~is_surface.unsqueeze(-1))

        err = pred_orig.double() - y.double()
        err = torch.where(torch.isfinite(err), err.abs(), torch.zeros_like(err))

        mae_surf += (err * surf_eff.double()).sum(dim=(0, 1))
        mae_vol += (err * vol_eff.double()).sum(dim=(0, 1))
        n_surf_per_ch += surf_eff.double().sum(dim=(0, 1))
        n_vol_per_ch += vol_eff.double().sum(dim=(0, 1))

    s = mae_surf / n_surf_per_ch.clamp(min=1)
    v = mae_vol / n_vol_per_ch.clamp(min=1)
    chs = ("Ux", "Uy", "p")
    return {
        **{f"mae_surf_{c}": s[i].item() for i, c in enumerate(chs)},
        **{f"mae_vol_{c}": v[i].item() for i, c in enumerate(chs)},
        "n_surf_p": int(n_surf_per_ch[2].item()),
        "n_vol_p": int(n_vol_per_ch[2].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg_path = args.model_dir / "config.yaml"
    ckpt_path = args.model_dir / "checkpoint.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = cfg["model_config"]
    model = Transolver(
        space_dim=mc["space_dim"], fun_dim=mc["fun_dim"], out_dim=mc["out_dim"],
        n_hidden=mc["n_hidden"], n_layers=mc["n_layers"], n_head=mc["n_head"],
        slice_num=mc["slice_num"], mlp_ratio=mc["mlp_ratio"],
        pos_freq_bands=mc.get("pos_freq_bands", 0),
        pos_freq_surface_only=mc.get("pos_freq_surface_only", False),
        use_swiglu=mc.get("use_swiglu", False),
        output_fields=mc["output_fields"], output_dims=mc["output_dims"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    _, _, stats, _ = load_data(args.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(args.splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    per_split = {}
    for name in TEST_SPLIT_NAMES:
        m = safe_evaluate_split(model, test_loaders[name], stats, device)
        per_split[name] = m
        print(f"  {name:30s}  surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
              f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]  "
              f"(n_surf_p={m['n_surf_p']}, n_vol_p={m['n_vol_p']})")

    keys = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in ("Ux", "Uy", "p")]
    test_avg = {}
    for k in keys:
        vals = [m[k] for m in per_split.values()]
        test_avg[f"avg/{k}"] = sum(vals) / len(vals)

    print(f"\n  TEST (safe 4-split)  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    print(f"                        avg_surf_Ux={test_avg['avg/mae_surf_Ux']:.4f}  avg_surf_Uy={test_avg['avg/mae_surf_Uy']:.4f}")
    print(f"                        avg_vol_p={test_avg['avg/mae_vol_p']:.4f}  avg_vol_Ux={test_avg['avg/mae_vol_Ux']:.4f}  avg_vol_Uy={test_avg['avg/mae_vol_Uy']:.4f}")

    out_path = args.model_dir / "test_safe_eval.json"
    with open(out_path, "w") as f:
        json.dump({"test_avg": test_avg, "test_splits": per_split,
                   "method": "per-node skip of non-finite y (excludes inf positions from both num and denom)"},
                  f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
