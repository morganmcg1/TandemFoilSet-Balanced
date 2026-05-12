"""Post-hoc diagnostic: measure divergence between slice_weights_v and slice_weights_k.

Loads the asymmetric-qk checkpoint and runs a few mini-batches forward, capturing
both slice-weight tensors via forward hooks on each PhysicsAttention block.
Reports per-block mean absolute difference + value-side / key-side entropy as a
sanity check on whether the two slice projections actually learned different
partitions, or whether they collapsed to the same distribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add target dir to path so we can import train.py's classes.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import X_DIM, load_data, pad_collate  # noqa: E402


def main() -> None:
    ckpt_dir = ROOT / "models" / "model-charliepai2g24h4-tanjiro-asymmetric-qk-20260512-201340"
    ckpt_path = ckpt_dir / "checkpoint.pt"
    splits_dir = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import Transolver lazily (importing train.py runs the trainer entrypoint).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_train_module", ROOT / "train.py")
    # Don't actually execute train.py; just grab the class definitions by parsing.
    # Easier: copy the class instantiation by reading config.yaml.
    import yaml

    with open(ckpt_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_config = cfg["model_config"]

    # Inline import: avoid running train.py module body.
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    from timm.layers import trunc_normal_

    ACT = {"gelu": nn.GELU}

    class MLP(nn.Module):
        def __init__(self, n_in, n_h, n_out, n_layers=1, act="gelu", res=True):
            super().__init__()
            self.n_layers, self.res = n_layers, res
            self.linear_pre = nn.Sequential(nn.Linear(n_in, n_h), ACT[act]())
            self.linear_post = nn.Linear(n_h, n_out)
            self.linears = nn.ModuleList(
                [nn.Sequential(nn.Linear(n_h, n_h), ACT[act]()) for _ in range(n_layers)]
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
            self.dim_head, self.heads = dim_head, heads
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
            self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_fx = nn.Linear(dim, inner_dim)
            self.in_project_slice = nn.Linear(dim_head, slice_num)
            self.in_project_slice_k = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_slice.weight)
            torch.nn.init.orthogonal_(self.in_project_slice_k.weight)
            self.to_q = nn.Linear(dim_head, dim_head, bias=False)
            self.to_k = nn.Linear(dim_head, dim_head, bias=False)
            self.to_v = nn.Linear(dim_head, dim_head, bias=False)
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            # Diagnostic buffers (filled during forward)
            self.last_sw_v = None
            self.last_sw_k = None

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
            slice_weights_v = self.softmax(self.in_project_slice(x_mid) / self.temperature)
            slice_weights_k = self.softmax(self.in_project_slice_k(x_mid) / self.temperature)
            self.last_sw_v = slice_weights_v.detach()
            self.last_sw_k = slice_weights_k.detach()
            slice_norm_v = slice_weights_v.sum(2)
            st_v = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_v)
            st_v = st_v / ((slice_norm_v + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
            slice_norm_k = slice_weights_k.sum(2)
            st_k = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_k)
            st_k = st_k / ((slice_norm_k + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
            q, k, v = self.to_q(st_v), self.to_k(st_k), self.to_v(st_v)
            out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights_v)
            return self.to_out(rearrange(out_x, "b h n d -> b n (h d)"))

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
                     output_fields=None, output_dims=None):
            super().__init__()
            self.unified_pos = unified_pos
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                   n_layers=0, res=False, act=act)
            self.n_hidden = n_hidden
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

        def forward(self, data, **_):
            x = data["x"]
            fx = self.preprocess(x) + self.placeholder[None, None, :]
            for block in self.blocks:
                fx = block(fx)
            return {"preds": fx}

    model = Transolver(**model_config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    _, val_splits, stats, _ = load_data(splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader = DataLoader(
        val_splits["val_single_in_dist"], batch_size=2, shuffle=False,
        collate_fn=pad_collate, num_workers=0,
    )

    n_blocks = len(model.blocks)
    abs_diff = [[] for _ in range(n_blocks)]
    entropy_v = [[] for _ in range(n_blocks)]
    entropy_k = [[] for _ in range(n_blocks)]
    cos_sim = [[] for _ in range(n_blocks)]

    n_batches_used = 0
    with torch.no_grad():
        for x, _, _, _ in loader:
            x = x.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            _ = model({"x": x_norm})
            for i, blk in enumerate(model.blocks):
                v = blk.attn.last_sw_v.float()
                k = blk.attn.last_sw_k.float()
                # mean absolute difference of softmax probabilities
                abs_diff[i].append((v - k).abs().mean().item())
                # entropy across slice tokens (last dim G)
                ev = -(v * (v + 1e-12).log()).sum(-1).mean().item()
                ek = -(k * (k + 1e-12).log()).sum(-1).mean().item()
                entropy_v[i].append(ev)
                entropy_k[i].append(ek)
                # cosine similarity per node (B,H,N,G)
                vf = v.flatten(0, -2)
                kf = k.flatten(0, -2)
                cs = torch.nn.functional.cosine_similarity(vf, kf, dim=-1).mean().item()
                cos_sim[i].append(cs)
            n_batches_used += 1
            if n_batches_used >= 4:
                break

    summary = {
        "n_batches": n_batches_used,
        "per_block": [
            {
                "block": i,
                "mean_abs_diff_softmax": sum(abs_diff[i]) / len(abs_diff[i]),
                "entropy_v_nats": sum(entropy_v[i]) / len(entropy_v[i]),
                "entropy_k_nats": sum(entropy_k[i]) / len(entropy_k[i]),
                "cos_sim_per_node": sum(cos_sim[i]) / len(cos_sim[i]),
            }
            for i in range(n_blocks)
        ],
    }
    out_path = ckpt_dir / "slice_divergence.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
