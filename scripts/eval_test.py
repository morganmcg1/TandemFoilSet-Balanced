"""Load a saved checkpoint and re-evaluate val + test with the fixed eval code.

Usage:
  python scripts/eval_test.py --run_id <wandb-run-id> [--config config.yaml] [--gpu 0]

This is needed because runs that finished before the inf-handling fix in
train.py produced ``test_avg/mae_surf_p = nan`` whenever the cruise test split
contained sample 000020.pt (which has 761 inf values in y). The fix is in
``evaluate_split`` in train.py; this script re-runs that evaluation against
the saved best checkpoint so we get a real number out.
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
    X_DIM,
    aggregate_splits,
    load_data,
    load_test_data,
    pad_collate,
)
from train import Transolver, evaluate_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True, help="wandb run id (model-{run_id} dir name)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--surf_weight", type=float, default=10.0)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(__file__).resolve().parents[1] / f"models/model-{args.run_id}"
    cfg_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "checkpoint.pt"

    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    with cfg_path.open() as f:
        model_cfg = yaml.safe_load(f)
    print(f"Model config: {model_cfg}")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                   for name, ds in val_splits.items()}

    model = Transolver(**model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint with {n_params/1e6:.2f}M params")

    # Val
    val_metrics = {name: evaluate_split(model, l, stats, args.surf_weight, device)
                   for name, l in val_loaders.items()}
    val_avg = aggregate_splits(val_metrics)
    print(f"VAL avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for n, m in val_metrics.items():
        print(f"  {n:<26s} surf_p={m['mae_surf_p']:.4f} surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    # Test
    print()
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                    for name, ds in test_datasets.items()}
    test_metrics = {name: evaluate_split(model, l, stats, args.surf_weight, device)
                    for name, l in test_loaders.items()}
    test_avg = aggregate_splits(test_metrics)
    print(f"TEST avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for n, m in test_metrics.items():
        print(f"  {n:<26s} surf_p={m['mae_surf_p']:.4f} surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    if args.out:
        out = {
            "run_id": args.run_id,
            "model_config": model_cfg,
            "val_avg": val_avg,
            "test_avg": test_avg,
            "val_per_split": val_metrics,
            "test_per_split": test_metrics,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, default=str, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
