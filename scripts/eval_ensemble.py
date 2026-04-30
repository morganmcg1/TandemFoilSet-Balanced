"""Ensemble evaluation: average predictions from multiple checkpoints.

For each batch, run the forward pass for every model and average the
*denormalized* predictions before computing MAE. This is the standard
test-time ensemble for regression models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)
from train import Transolver


def load_model(model_dir: Path, device: torch.device) -> Transolver:
    cfg = yaml.safe_load((model_dir / "config.yaml").open())
    model = Transolver(**cfg).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_ensemble(models, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0

    for x, y, is_surface, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        # Average the denormalized predictions across models.
        pred_orig_avg = None
        for m in models:
            pred = m({"x": x_norm}, mask=mask)["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            if pred_orig_avg is None:
                pred_orig_avg = pred_orig.clone()
            else:
                pred_orig_avg += pred_orig
        pred_orig_avg /= len(models)

        # Filter samples with non-finite y (avoids inf*0 = nan)
        B = y.shape[0]
        y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
        if y_finite.any():
            good = y_finite.nonzero(as_tuple=False).flatten()
            ds, dv = accumulate_batch(
                pred_orig_avg[good], y[good], is_surface[good], mask[good], mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv

    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_ids", nargs="+", required=True, help="wandb run ids of checkpoints to ensemble")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = Path(__file__).resolve().parents[1] / "models"
    models = []
    for rid in args.run_ids:
        model_dir = base / f"model-{rid}"
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        models.append(load_model(model_dir, device))
        print(f"Loaded {rid}")
    print(f"Ensemble of {len(models)} models")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    val_loaders = {n: DataLoader(d, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                   for n, d in val_splits.items()}
    val_metrics = {n: evaluate_ensemble(models, l, stats, device) for n, l in val_loaders.items()}
    val_avg = aggregate_splits(val_metrics)
    print(f"VAL ensemble avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for n, m in val_metrics.items():
        print(f"  {n:<26s} surf_p={m['mae_surf_p']:.4f}")

    print()
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {n: DataLoader(d, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                    for n, d in test_datasets.items()}
    test_metrics = {n: evaluate_ensemble(models, l, stats, device) for n, l in test_loaders.items()}
    test_avg = aggregate_splits(test_metrics)
    print(f"TEST ensemble avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for n, m in test_metrics.items():
        print(f"  {n:<26s} surf_p={m['mae_surf_p']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "run_ids": args.run_ids,
            "val_avg": val_avg,
            "test_avg": test_avg,
            "val_per_split": val_metrics,
            "test_per_split": test_metrics,
        }, default=str, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
