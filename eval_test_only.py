"""Post-hoc test evaluation of a saved Transolver checkpoint.

Loads the checkpoint produced by ``train.py`` and re-runs the test-split
evaluation only — used to recover ``test_avg/mae_surf_p`` after a training
run that hit the ``Inf in y`` scoring bug (test_geom_camber_cruise/000020).

Usage:
    python eval_test_only.py --model_dir models/<experiment>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    aggregate_splits,
    load_test_data,
    pad_collate,
)
from train import Transolver, evaluate_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--surf_weight", type=float, default=10.0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "checkpoint.pt"
    cfg_path = model_dir / "config.yaml"
    with open(cfg_path) as f:
        saved_cfg = yaml.safe_load(f)
    model_config = saved_cfg["model_config"]

    stats_path = Path(args.splits_dir) / "stats.json"
    with open(stats_path) as f:
        raw_stats = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {k: torch.tensor(raw_stats[k], dtype=torch.float32, device=device)
             for k in ("x_mean", "x_std", "y_mean", "y_std")}

    model = Transolver(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_datasets = load_test_data(args.splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    test_metrics = {
        name: evaluate_split(model, loader, stats, args.surf_weight, device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)

    print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(
            f"    {name:<28s} "
            f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
            f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
        )

    metrics_jsonl_path = model_dir / "metrics.jsonl"
    with open(metrics_jsonl_path, "a") as f:
        f.write(json.dumps({
            "event": "test_posthoc",
            "test_avg": test_avg,
            "test_splits": test_metrics,
        }, sort_keys=True) + "\n")
    print(f"\nAppended post-hoc test event to {metrics_jsonl_path}")

    summary_path = model_dir / "metrics.yaml"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = yaml.safe_load(f) or {}
        summary["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                summary[f"test/{split_name}/{k}"] = v
        with open(summary_path, "w") as f:
            yaml.safe_dump(summary, f, sort_keys=True)
        print(f"Updated test_* entries in {summary_path}")


if __name__ == "__main__":
    main()
