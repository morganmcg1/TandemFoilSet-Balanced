"""Re-evaluate saved Arm A / Arm B checkpoints with the NaN-propagation fix.

Loads each checkpoint, runs the (patched) ``evaluate_split`` over both val and
test splits, and appends an ``event: "test_corrected"`` line into the
arm's ``metrics.jsonl``. Also updates ``metrics.yaml`` with corrected test
fields. The patched evaluator lives in ``train.py``; this script imports the
function directly so the fix and the re-eval cannot drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import importlib.util


def _load_train_module():
    """Import train.py as a module without triggering its top-level training run.

    train.py does ``sp.parse(Config)`` at import; we side-step that by reading
    the source up to the ``cfg = sp.parse(Config)`` line and executing only
    the prefix (class definitions + helpers).
    """
    src = Path("train.py").read_text()
    cutoff = src.index("cfg = sp.parse(Config)")
    prefix = src[:cutoff]
    import types
    mod = types.ModuleType("train_partial")
    mod.__dict__["__name__"] = "train_partial"
    sys.modules["train_partial"] = mod
    exec(compile(prefix, "train.py", "exec"), mod.__dict__)
    return mod


def reeval_checkpoint(model_dir: Path, train_mod, splits_dir: str) -> dict:
    cfg_path = model_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_config = cfg["model_config"]
    n_freqs = model_config["surface_pe_n_freqs"]

    from data import (
        TEST_SPLIT_NAMES,
        VAL_SPLIT_NAMES,
        load_data,
        load_test_data,
        pad_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_splits, stats, sample_weights = load_data(splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_datasets = load_test_data(splits_dir, debug=False)

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    model = train_mod.Transolver(**model_config).to(device)
    ckpt_path = model_dir / "checkpoint.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"\n=== Re-eval {model_dir.name} (n_freqs={n_freqs}) ===")

    val_metrics = {
        name: train_mod.evaluate_split(model, loader, stats, cfg["surf_weight"], device)
        for name, loader in val_loaders.items()
    }
    from data import aggregate_splits
    val_avg = aggregate_splits(val_metrics)
    print(f"  VAL avg_surf_p={val_avg['avg/mae_surf_p']:.4f}")
    for name in VAL_SPLIT_NAMES:
        m = val_metrics[name]
        print(f"    {name:<26s} surf_p={m['mae_surf_p']:.4f}  vol_p={m['mae_vol_p']:.4f}")

    test_metrics = {
        name: train_mod.evaluate_split(model, loader, stats, cfg["surf_weight"], device)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    print(f"  TEST avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(f"    {name:<26s} surf_p={m['mae_surf_p']:.4f}  vol_p={m['mae_vol_p']:.4f}")

    # Append the corrected event into metrics.jsonl
    jsonl_path = model_dir / "metrics.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({
            "event": "test_corrected",
            "note": "Re-evaluated with NaN-propagation fix in evaluate_split (NaN samples skipped via y_finite_per_sample mask + nan_to_num).",
            "n_freqs": n_freqs,
            "val_avg": val_avg,
            "val_splits": val_metrics,
            "test_avg": test_avg,
            "test_splits": test_metrics,
        }, sort_keys=True) + "\n")

    # Update metrics.yaml with the corrected fields
    yaml_path = model_dir / "metrics.yaml"
    with open(yaml_path) as f:
        summary = yaml.safe_load(f)
    summary["test_avg_corrected/mae_surf_p"] = test_avg["avg/mae_surf_p"]
    for split_name, m in test_metrics.items():
        for k, v in m.items():
            summary[f"test_corrected/{split_name}/{k}"] = v
    summary["val_avg_corrected/mae_surf_p"] = val_avg["avg/mae_surf_p"]
    for split_name, m in val_metrics.items():
        for k, v in m.items():
            summary[f"val_corrected/{split_name}/{k}"] = v
    with open(yaml_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)

    return {
        "model_dir": str(model_dir),
        "n_freqs": n_freqs,
        "val_avg/mae_surf_p": val_avg["avg/mae_surf_p"],
        "test_avg/mae_surf_p": test_avg["avg/mae_surf_p"],
        "test_per_split": {name: test_metrics[name] for name in TEST_SPLIT_NAMES},
        "val_per_split": {name: val_metrics[name] for name in VAL_SPLIT_NAMES},
    }


def main():
    train_mod = _load_train_module()
    splits_dir = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"

    arm_a = Path("models/model-charliepai2i24h5-askeladd-surface-pe-nfreqs16-20260515-133406")
    arm_b = Path("models/model-charliepai2i24h5-askeladd-surface-pe-nfreqs8-20260515-142835")

    results = {}
    for key, model_dir in [("arm_a_n16", arm_a), ("arm_b_n8", arm_b)]:
        results[key] = reeval_checkpoint(model_dir, train_mod, splits_dir)

    out_path = Path("logs/reeval_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=float)
    print(f"\nWrote summary: {out_path}")
    print(json.dumps({k: {"val_avg": v["val_avg/mae_surf_p"], "test_avg": v["test_avg/mae_surf_p"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
