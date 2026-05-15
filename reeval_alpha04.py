"""Re-evaluate the alpha=0.4 checkpoint with the patched evaluate_split.

The original alpha=0.4 run finished before the NaN-safe pre-filter was added
to evaluate_split, so test_avg/mae_surf_p came out as NaN. This script reloads
the saved best checkpoint and runs the patched test-time eval, then appends a
"test_corrected" event to that model's metrics.jsonl and updates metrics.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Reuse model + eval helpers from train.py — running this from target/ uses the
# same X_DIM etc.
from train import (
    Transolver,
    evaluate_split,
    aggregate_splits,
)
from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    load_data,
    load_test_data,
    pad_collate,
)


MODEL_DIR = Path("models/model-charliepai2i24h5-alphonse-camber-mixup-alpha04-20260515-133139")
SPLITS_DIR = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
BATCH_SIZE = 4
SURF_WEIGHT = 10.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load stats (and discard train data — we only need val stats / test data)
    _, _, stats, _ = load_data(SPLITS_DIR, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    model_config = dict(
        space_dim=2,
        fun_dim=X_DIM - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=4,
        slice_num=64,
        mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)

    ckpt_path = MODEL_DIR / "checkpoint.pt"
    print(f"Loading checkpoint {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    # Re-run validation with patched code as well — best_val metrics in the
    # original metrics.yaml were computed before the patch and may include
    # poisoned splits.
    print("\nRe-evaluating on validation splits (patched)...")
    _, val_splits, _, _ = load_data(SPLITS_DIR, debug=False)
    val_loaders = {
        name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    val_metrics = {
        name: evaluate_split(model, loader, stats, SURF_WEIGHT, device)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(val_metrics)
    print(f"  VAL  avg_surf_p={val_avg['avg/mae_surf_p']:.4f}")
    for name in VAL_SPLIT_NAMES:
        m = val_metrics[name]
        print(
            f"  {name:26s} loss={m['loss']:.4f} surf[p={m['mae_surf_p']:.4f}"
            f" Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]"
            f" vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f}"
            f" Uy={m['mae_vol_Uy']:.4f}]"
        )

    print("\nEvaluating on test splits (patched)...")
    test_datasets = load_test_data(SPLITS_DIR, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }
    test_metrics = {
        name: evaluate_split(model, loader, stats, SURF_WEIGHT, device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    print(f"  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(
            f"  {name:26s} loss={m['loss']:.4f} surf[p={m['mae_surf_p']:.4f}"
            f" Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]"
            f" vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f}"
            f" Uy={m['mae_vol_Uy']:.4f}]"
        )

    # Append corrected event to metrics.jsonl
    jsonl_path = MODEL_DIR / "metrics.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({
            "event": "test_corrected",
            "note": "Re-eval with NaN-safe evaluate_split (filter non-finite-y samples).",
            "best_epoch": 13,
            "val_avg_corrected": val_avg,
            "val_splits_corrected": val_metrics,
            "test_avg": test_avg,
            "test_splits": test_metrics,
        }, sort_keys=True) + "\n")
    print(f"\nAppended test_corrected event to {jsonl_path}")

    # Patch the yaml summary as well, replacing NaN test fields and adding
    # corrected val/test averages without losing the original record.
    yaml_path = MODEL_DIR / "metrics.yaml"
    with open(yaml_path, "r") as f:
        summary = yaml.safe_load(f)

    summary["test_avg/mae_surf_p_corrected"] = test_avg["avg/mae_surf_p"]
    summary["best_val_avg/mae_surf_p_corrected"] = val_avg["avg/mae_surf_p"]
    for split_name, m in test_metrics.items():
        for k, v in m.items():
            summary[f"test_corrected/{split_name}/{k}"] = v
    for split_name, m in val_metrics.items():
        for k, v in m.items():
            summary[f"best_val_corrected/{split_name}/{k}"] = v

    with open(yaml_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"Updated {yaml_path} with *_corrected fields.")


if __name__ == "__main__":
    main()
