"""Re-evaluate a saved checkpoint with the current evaluate_split (NaN fix).

Usage:
    python retest_checkpoint.py <model_dir>

Emits a `test_corrected` event line into metrics.jsonl alongside the original
metrics, and prints the corrected test_avg block. The original metrics.yaml
already contains the per-channel surface MAE rows that were not NaN, but
test_avg/mae_surf_p was NaN due to NaN propagation in accumulate_batch
(NaN * 0 = NaN defeats the per-sample skip).

The Transolver class and evaluate_split are duplicated here because train.py
runs simple_parsing at import time. The class definition is byte-identical to
the one in train.py at the commit time of this script — keep them in sync if
that class changes.
"""

from __future__ import annotations

import json
import sys
import time
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
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)

ACTIVATION = {
    "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1), "softplus": nn.Softplus, "ELU": nn.ELU,
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
            self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3).contiguous()
        )
        x_mid = (
            self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3).contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))
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
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if isinstance(slice_num, (list, tuple)):
            assert len(slice_num) == n_layers
            slice_nums = list(slice_num)
        else:
            slice_nums = [slice_num] * n_layers
        self.slice_nums = slice_nums
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_nums[i], last_layer=(i == n_layers - 1),
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


def evaluate_split(model, loader, stats, surf_weight, device):
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
            y_finite_per_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            mask = mask & y_finite_per_sample.view(-1, 1)
            y_clean = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y_clean - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]
            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum()
                             / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum()
                              / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y_clean, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main(model_dir: str) -> None:
    model_path = Path(model_dir)
    cfg_path = model_path / "config.yaml"
    ckpt_path = model_path / "checkpoint.pt"
    metrics_jsonl = model_path / "metrics.jsonl"
    assert cfg_path.exists() and ckpt_path.exists(), f"Missing files under {model_path}"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"slice_num: {mc['slice_num']}")

    model = Transolver(
        space_dim=mc["space_dim"], fun_dim=mc["fun_dim"], out_dim=mc["out_dim"],
        n_hidden=mc["n_hidden"], n_layers=mc["n_layers"], n_head=mc["n_head"],
        slice_num=mc["slice_num"], mlp_ratio=mc["mlp_ratio"],
        output_fields=mc.get("output_fields"), output_dims=mc.get("output_dims"),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    splits_dir = cfg.get("splits_dir", "/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    surf_weight = cfg.get("surf_weight", 10.0)
    batch_size = cfg.get("batch_size", 4)

    _, _, stats, _ = load_data(splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(splits_dir, debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    t0 = time.time()
    test_metrics = {
        name: evaluate_split(model, loader, stats, surf_weight, device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    dt = time.time() - t0

    print(f"\nCorrected test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    print(f"Full test_avg: {test_avg}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(
            f"  {name:<26s} surf_p={m['mae_surf_p']:.4f} "
            f"Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}"
        )

    record = {
        "event": "test_corrected",
        "note": "Re-evaluated with NaN-handling fix in evaluate_split (drop non-finite y samples + nan_to_num).",
        "seconds": dt,
        "test_avg": test_avg,
        "test_splits": test_metrics,
    }
    with open(metrics_jsonl, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"\nAppended test_corrected event to {metrics_jsonl}")


if __name__ == "__main__":
    main(sys.argv[1])
