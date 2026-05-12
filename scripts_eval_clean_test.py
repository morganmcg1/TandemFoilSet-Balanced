"""Re-evaluate the saved best checkpoint with a NaN-safe scorer.

Loads the model class definitions from train.py without triggering its
module-level training loop (which would launch a fresh 30-min run).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import TEST_SPLIT_NAMES, load_test_data, pad_collate
from data.scoring import CHANNELS


def load_model_classes():
    """Execute only class definitions from train.py (lines 1..214 incl. classes)."""
    src = Path("train.py").read_text()
    # Cut at the line before module-level execution starts. Class block ends at
    # the eval-helpers header comment near line 215.
    cut_marker = "# ---------------------------------------------------------------------------\n# Evaluation helpers"
    cut_idx = src.index(cut_marker)
    code = src[:cut_idx]
    ns: dict = {"__name__": "_train_classes"}
    exec(compile(code, "train.py[classes]", "exec"), ns)
    return ns["Transolver"]


def main():
    ckpt_dir = Path("models/model-charliepai2g24h4-askeladd-surf-weight-30-20260512-175807")
    state_dict = torch.load(ckpt_dir / "checkpoint.pt", map_location="cuda", weights_only=True)

    Transolver = load_model_classes()
    model = Transolver(
        space_dim=2, fun_dim=22, out_dim=3, n_hidden=128, n_layers=5,
        n_head=4, slice_num=64, mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    ).cuda().eval()
    model.load_state_dict(state_dict)

    splits_dir = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    with open(splits_dir / "stats.json") as f:
        raw_stats = json.load(f)
    stats = {k: torch.tensor(raw_stats[k], dtype=torch.float32).cuda()
             for k in ("x_mean", "x_std", "y_mean", "y_std")}

    test_datasets = load_test_data(splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=2, pin_memory=True)

    def safe_eval(loader):
        mae_surf = torch.zeros(3, dtype=torch.float64, device="cuda")
        mae_vol = torch.zeros(3, dtype=torch.float64, device="cuda")
        n_surf = n_vol = 0
        n_samples_skipped = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                is_surface = is_surface.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True)
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                pred = model({"x": x_norm})["preds"]
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                B = y.shape[0]
                y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
                p_finite = torch.isfinite(pred_orig.reshape(B, -1)).all(dim=-1)
                keep = y_finite & p_finite
                n_samples_skipped += (~keep).sum().item()
                if not keep.any():
                    continue
                sample_mask = keep.unsqueeze(-1).expand(-1, mask.shape[-1])
                effective = mask & sample_mask
                sm = effective & is_surface
                vm = effective & ~is_surface
                err = (pred_orig.double() - y.double()).abs()
                err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
                mae_surf += (err * sm.unsqueeze(-1).double()).sum(dim=(0, 1))
                mae_vol += (err * vm.unsqueeze(-1).double()).sum(dim=(0, 1))
                n_surf += int(sm.sum().item()); n_vol += int(vm.sum().item())
        out = {}
        s = mae_surf / max(n_surf, 1); v = mae_vol / max(n_vol, 1)
        for i, ch in enumerate(CHANNELS):
            out[f"mae_surf_{ch}"] = s[i].item()
            out[f"mae_vol_{ch}"] = v[i].item()
        out["n_skipped"] = n_samples_skipped
        out["n_surf_nodes"] = n_surf; out["n_vol_nodes"] = n_vol
        return out

    result: dict = {}
    for name in TEST_SPLIT_NAMES:
        loader = DataLoader(test_datasets[name], batch_size=4, shuffle=False, **loader_kwargs)
        m = safe_eval(loader)
        result[name] = m
        print(f"{name}: surf_p={m['mae_surf_p']:.4f}  vol_p={m['mae_vol_p']:.4f}  "
              f"n_skipped={m['n_skipped']}")

    avg_surf_p = sum(result[n]["mae_surf_p"] for n in TEST_SPLIT_NAMES) / 4
    avg_vol_p = sum(result[n]["mae_vol_p"] for n in TEST_SPLIT_NAMES) / 4
    avg_surf_Ux = sum(result[n]["mae_surf_Ux"] for n in TEST_SPLIT_NAMES) / 4
    avg_surf_Uy = sum(result[n]["mae_surf_Uy"] for n in TEST_SPLIT_NAMES) / 4
    avg_vol_Ux = sum(result[n]["mae_vol_Ux"] for n in TEST_SPLIT_NAMES) / 4
    avg_vol_Uy = sum(result[n]["mae_vol_Uy"] for n in TEST_SPLIT_NAMES) / 4

    summary = {
        "test_splits_clean": result,
        "test_avg_clean/mae_surf_p": avg_surf_p,
        "test_avg_clean/mae_vol_p": avg_vol_p,
        "test_avg_clean/mae_surf_Ux": avg_surf_Ux,
        "test_avg_clean/mae_surf_Uy": avg_surf_Uy,
        "test_avg_clean/mae_vol_Ux": avg_vol_Ux,
        "test_avg_clean/mae_vol_Uy": avg_vol_Uy,
    }
    print("\nClean test_avg/mae_surf_p =", avg_surf_p)
    print("Clean test_avg/mae_vol_p  =", avg_vol_p)
    (ckpt_dir / "test_clean.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
