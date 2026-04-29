"""Run end-of-training test evaluation on a saved Transolver checkpoint.

Self-contained: imports only data/* + the Transolver class definition copied
from train.py. The original train.py runs ``simple_parsing.parse(Config)`` at
import time so we cannot import directly without losing argparse.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_test_data,
    pad_collate,
)


# ---------------------------------------------------------------------------
# Transolver — copied verbatim from train.py
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
            .permute(0, 2, 1, 3).contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3).contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
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
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
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
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


def evaluate_split(model, loader, stats, surf_weight, device):
    """Same metric semantics as ``train.evaluate_split``/``data.scoring`` —
    skips per-sample ground truths that contain any non-finite value (which
    is what ``program.md`` documents). The only difference is a defensive
    pre-zero of non-finite cells *for samples that are dropped anyway*
    before they are multiplied by the boolean mask, because IEEE NaN/Inf
    propagates through ``mask * value`` even when ``mask`` is False
    (``inf * 0 = NaN``). Without this, a single ``inf`` in the held-out
    test data contaminates the float64 accumulator and the per-channel
    MAE comes out as NaN.
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

            # Drop any sample whose ground truth has non-finite values
            # entirely from the loss accumulator AND from the batch passed to
            # accumulate_batch. ``accumulate_batch`` itself will additionally
            # do the same per-sample skip (so we match exactly the train-time
            # semantics from ``data.scoring``).
            y_finite_per_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)

            # Replace any non-finite y/pred entries with zero so the boolean
            # mask multiplications below don't propagate NaN/Inf into the
            # accumulators. The replacement only matters for cells that are
            # already going to be masked out.
            y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y_safe - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]
            pred_safe = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

            sq_err = (pred_safe - y_norm) ** 2
            sample_keep = y_finite_per_sample.view(-1, 1).expand(-1, mask.shape[-1])
            vol_mask = mask & ~is_surface & sample_keep
            surf_mask = mask & is_surface & sample_keep
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1

            pred_orig = pred_safe * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask & sample_keep,
                                      mae_surf, mae_vol)
            n_surf += ds; n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss, "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def print_split_metrics(split_name, m):
    print(
        f"    {split_name:<26s} "
        f"loss={m['loss']:.4f}  "
        f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
        f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config_yaml', required=True)
    ap.add_argument('--splits_dir', default='/mnt/new-pvc/datasets/tandemfoil/splits_v2')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--surf_weight', type=float, default=10.0)
    ap.add_argument('--wandb_name', default=None)
    ap.add_argument('--wandb_group', default=None)
    ap.add_argument('--agent', default='ml-intern-r1')
    ap.add_argument('--no_wandb', action='store_true')
    ap.add_argument('--use_amp', type=lambda s: s.lower() == 'true', default=False,
                    help='If true, run inference under bf16 autocast (matches AMP training).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config_yaml) as f:
        model_config = yaml.safe_load(f)
    print(f'Model config: {model_config}')

    with open(Path(args.splits_dir) / 'stats.json') as f:
        raw = json.load(f)
    stats = {k: torch.tensor(raw[k], dtype=torch.float32, device=device)
             for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}

    model = Transolver(**model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded checkpoint: {n_params/1e6:.2f}M params')

    run = None
    if not args.no_wandb:
        run = wandb.init(
            entity=os.environ.get('WANDB_ENTITY'),
            project=os.environ.get('WANDB_PROJECT'),
            group=args.wandb_group,
            name=args.wandb_name or f'test-eval-{Path(args.checkpoint).parent.name}',
            tags=[args.agent, 'test_eval'],
            config={'checkpoint': args.checkpoint, 'model_config': model_config,
                    'n_params': n_params, 'batch_size': args.batch_size,
                    'surf_weight': args.surf_weight},
            mode=os.environ.get('WANDB_MODE', 'online'),
        )

    test_datasets = load_test_data(args.splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    if args.use_amp:
        # Wrap forward in bf16 autocast to match training-time precision.
        from contextlib import contextmanager
        @contextmanager
        def amp_ctx():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                yield
        # monkey-patch model.forward to apply autocast
        orig_fwd = model.forward
        def amp_fwd(*a, **k):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = orig_fwd(*a, **k)
                # ensure preds are fp32 for scoring (matches train.py)
                out = {k: v.float() for k, v in out.items()}
            return out
        model.forward = amp_fwd

    test_metrics = {
        name: evaluate_split(model, loader, stats, args.surf_weight, device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        print_split_metrics(name, test_metrics[name])

    if run is not None:
        log = {}
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                log[f'test/{split_name}/{k}'] = v
        for k, v in test_avg.items():
            log[f'test_{k}'] = v
        run.log(log)
        run.summary.update(log)
        run.finish()

    return test_avg, test_metrics


if __name__ == '__main__':
    main()
