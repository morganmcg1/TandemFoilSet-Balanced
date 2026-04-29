"""Evaluate an ensemble of saved checkpoints on val + test.

Strategy: average predictions in normalized space across multiple models. All
models must have the same Transolver config. Same NaN-safe filter as
``eval_test.py`` for the test_geom_camber_cruise sample.

Usage:
    python scripts/eval_ensemble.py models/model-A models/model-B [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import (TEST_SPLIT_NAMES, accumulate_batch, aggregate_splits,
                  finalize_split, load_data, load_test_data, pad_collate)
# Reuse Transolver from eval_test.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_test import Transolver


def load_model(model_dir: Path, device) -> tuple[torch.nn.Module, dict]:
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    model = Transolver(**cfg).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def evaluate_ensemble(models, loader, stats, surf_weight, device, label=""):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    n_skipped = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not y_finite.all():
                n_skipped += int((~y_finite).sum().item())
                if not y_finite.any():
                    continue
                idx = y_finite.nonzero().squeeze(-1)
                x = x[idx]; y = y[idx]; is_surface = is_surface[idx]; mask = mask[idx]

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            preds = []
            for m in models:
                p = m({"x": x_norm})["preds"]
                preds.append(p)
            pred = torch.stack(preds, dim=0).mean(dim=0)
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds; n_vol += dv

    if n_skipped:
        print(f"  [{label}] skipped {n_skipped} sample(s) with non-finite GT")
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dirs", nargs="+")
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--surf_weight", type=float, default=10.0)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_splits, stats, _ = load_data(args.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    models = []
    cfgs = []
    for d in args.model_dirs:
        m, cfg = load_model(Path(d), device)
        models.append(m)
        cfgs.append(cfg)
    print(f"Loaded {len(models)} models for ensemble")

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=2, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)

    # VAL
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    val_metrics = {n: evaluate_ensemble(models, l, stats, args.surf_weight, device, n)
                   for n, l in val_loaders.items()}
    val_avg = aggregate_splits(val_metrics)
    print(f"\n--- ENS VAL --- avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for name, m in val_metrics.items():
        print(f"  {name:<26s}  surf[p={m['mae_surf_p']:.4f}]  vol[p={m['mae_vol_p']:.4f}]")

    # TEST
    test_datasets = load_test_data(args.splits_dir)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }
    test_metrics = {n: evaluate_ensemble(models, l, stats, args.surf_weight, device, n)
                    for n, l in test_loaders.items()}
    test_avg = aggregate_splits(test_metrics)
    print(f"\n--- ENS TEST --- avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(f"  {name:<26s}  surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]"
              f"  vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]")

    if args.out_json:
        out = {
            "model_dirs": list(args.model_dirs),
            "configs": cfgs,
            "val_avg": val_avg,
            "val_per_split": val_metrics,
            "test_avg": test_avg,
            "test_per_split": test_metrics,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
