"""Re-evaluate the best checkpoint on test splits with NaN-safe masking.

One-off helper. ``data/scoring.py`` skips non-finite samples via a mask, but
``mask * NaN = NaN`` in PyTorch, so a single hidden-ground-truth NaN
contaminates the per-channel sum. We zero out NaN err contributions inside
the masked region only.

Model classes are duplicated here to avoid triggering ``train.py``'s
top-level training loop on import.
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
from data.scoring import CHANNELS

CKPT = Path(sys.argv[1])
SPLITS_DIR = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Model classes (mirror of train.py) -----------
ACTIVATION = {"gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
              "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU(0.1),
              "softplus": nn.Softplus, "ELU": nn.ELU, "silu": nn.SiLU}


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_freqs=16, sigma=4.0):
        super().__init__()
        B = torch.randn(in_dim, num_freqs) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return self.B.shape[1] * 2

    def forward(self, coords):
        proj = 2 * 3.141592653589793 * coords @ self.B
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


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
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads,
                                     dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num)
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
                 fourier_num_freqs=16, fourier_sigma=4.0,
                 output_fields=None, output_dims=None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        self.fourier = FourierFeatures(in_dim=space_dim, num_freqs=fourier_num_freqs, sigma=fourier_sigma)
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim + self.fourier.out_dim, n_hidden * 2, n_hidden,
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
        pos = x[..., :self.space_dim]
        fourier = self.fourier(pos)
        x_aug = torch.cat([x, fourier], dim=-1)
        fx = self.preprocess(x_aug) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ----------- Setup -----------
with open(SPLITS_DIR + "/stats.json") as f:
    stats_raw = json.load(f)
stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in stats_raw.items()}

model_config = dict(
    space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
)
model = Transolver(**model_config).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

test_datasets = load_test_data(SPLITS_DIR)


def eval_split_clean(loader):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    skipped = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            skipped += int((~y_finite).sum().item())
            sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
            effective = mask & sample_mask
            surf_mask = effective & is_surface
            vol_mask = effective & ~is_surface
            err = (pred_orig.double() - y.double()).abs()
            # NaN-safe: zero out positions where mask=0 first (prevents 0*NaN propagation)
            err_surf = torch.where(surf_mask.unsqueeze(-1), err, torch.zeros_like(err))
            err_vol = torch.where(vol_mask.unsqueeze(-1), err, torch.zeros_like(err))
            mae_surf += err_surf.sum(dim=(0, 1))
            mae_vol += err_vol.sum(dim=(0, 1))
            n_surf += int(surf_mask.sum().item())
            n_vol += int(vol_mask.sum().item())
    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    out = {"_skipped": skipped, "_n_surf": n_surf, "_n_vol": n_vol}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = s[i].item()
        out[f"mae_vol_{ch}"] = v[i].item()
    return out


loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)
results = {}
for name in TEST_SPLIT_NAMES:
    loader = DataLoader(test_datasets[name], batch_size=4, shuffle=False, **loader_kwargs)
    results[name] = eval_split_clean(loader)
    r = results[name]
    print(f"{name:32s}  skipped={r['_skipped']:3d}  "
          f"surf[p={r['mae_surf_p']:.4f}  Ux={r['mae_surf_Ux']:.4f}  Uy={r['mae_surf_Uy']:.4f}]  "
          f"vol[p={r['mae_vol_p']:.4f}  Ux={r['mae_vol_Ux']:.4f}  Uy={r['mae_vol_Uy']:.4f}]")

surf_p_vals = [r["mae_surf_p"] for r in results.values()]
surf_ux_vals = [r["mae_surf_Ux"] for r in results.values()]
surf_uy_vals = [r["mae_surf_Uy"] for r in results.values()]
test_avg = {
    "test_avg/mae_surf_p": sum(surf_p_vals) / len(surf_p_vals),
    "test_avg/mae_surf_Ux": sum(surf_ux_vals) / len(surf_ux_vals),
    "test_avg/mae_surf_Uy": sum(surf_uy_vals) / len(surf_uy_vals),
}
print(f"\n{'test_avg/mae_surf_p':32s} = {test_avg['test_avg/mae_surf_p']:.4f}")
print(f"{'test_avg/mae_surf_Ux':32s} = {test_avg['test_avg/mae_surf_Ux']:.4f}")
print(f"{'test_avg/mae_surf_Uy':32s} = {test_avg['test_avg/mae_surf_Uy']:.4f}")

out_path = CKPT.parent / "test_metrics_clean.json"
with open(out_path, "w") as f:
    json.dump({"splits": results, **test_avg}, f, indent=2)
print(f"\nSaved clean test metrics to {out_path}")
