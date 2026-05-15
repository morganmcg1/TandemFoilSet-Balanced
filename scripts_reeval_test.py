"""Post-hoc NaN-safe test eval for an existing checkpoint.

Loads the best checkpoint from a `models/<dir>` produced by `train.py` and
re-runs the test splits with `evaluate_split_nan_safe`, writing a follow-up
record into the same `metrics.jsonl` and producing a small JSON summary.

Usage:
  python scripts_reeval_test.py --model_dir models/model-bf16-amp-...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import TEST_SPLIT_NAMES, load_test_data, pad_collate
from train import (
    Transolver,
    aggregate_splits,
    aggregate_splits_finite,
    append_metrics_jsonl,
    evaluate_split_nan_safe,
    load_data,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg_path = model_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = cfg["model_config"]
    model = Transolver(**mc).to(device)
    ckpt = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    _, _, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(args.splits_dir, debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }
    test_metrics = {
        name: evaluate_split_nan_safe(model, loader, stats, cfg["surf_weight"], device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    test_avg_finite = aggregate_splits_finite(test_metrics)

    print("NaN-safe test re-eval results:")
    print(f"  test_avg/mae_surf_p (all 4 splits):    {test_avg.get('avg/mae_surf_p', float('nan')):.4f}")
    print(f"  test_avg/mae_surf_p (finite-split):    {test_avg_finite.get('avg/mae_surf_p', float('nan')):.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        p = m.get("mae_surf_p", math.nan)
        n_surf = m.get("n_surf_nodes", 0)
        print(f"    {name}: mae_surf_p={p:.4f} (n_surf_nodes={n_surf})")

    append_metrics_jsonl(model_dir / "metrics.jsonl", {
        "event": "test_nan_safe_reeval",
        "test_avg": test_avg,
        "test_avg_finite": test_avg_finite,
        "test_splits": test_metrics,
    })
    print(f"\nAppended NaN-safe test re-eval to {model_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
