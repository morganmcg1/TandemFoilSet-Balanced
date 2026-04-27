"""Re-evaluate saved best checkpoints on the test splits with the NaN fix.

Loads each checkpoint at models/model-<run_id>/checkpoint.pt and runs the fixed
``evaluate_split`` over the test loaders. Prints per-split + averaged metrics.
Optionally updates the corresponding W&B run summary.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Re-exec only the class/function/import top-level statements from train.py so
# we get the Transolver model + the FIXED evaluate_split without triggering
# the training script body (sp.parse, data loading, training loop, etc.).
_train_src = (Path(__file__).parent / "train.py").read_text()
_tree = ast.parse(_train_src)
# Keep imports + class/function defs + a small allowlist of module-level
# constants used by the model classes. Skip everything else (CLI parsing,
# data loading, model construction, training loop).
_KEEP_NAMES = {"ACTIVATION"}


def _keep(node):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in _KEEP_NAMES:
        return True
    return False


_safe_body = [n for n in _tree.body if _keep(n)]
import types as _types
_module = _types.ModuleType("_train_lib")
_module.__file__ = str(Path(__file__).parent / "train.py")
sys.modules["_train_lib"] = _module
_module_ns = _module.__dict__
exec(compile(ast.Module(body=_safe_body, type_ignores=[]),
             str(Path(__file__).parent / "train.py"), "exec"),
     _module_ns)

Transolver = _module_ns["Transolver"]
evaluate_split = _module_ns["evaluate_split"]

from data import (
    TEST_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_data,
    load_test_data,
    pad_collate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--surf_weight", type=float, default=10.0)
    p.add_argument("--runs", nargs="+", required=True,
                   help="W&B run IDs (each maps to models/model-<id>/checkpoint.pt)")
    p.add_argument("--update_wandb", action="store_true",
                   help="Update W&B run summary with corrected test metrics")
    p.add_argument("--wandb_entity", default="wandb-applied-ai-team")
    p.add_argument("--wandb_project", default="senpai-charlie-wilson-willow-r2")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stats from the dataset (load_data handles normalization stats).
    print("Loading data stats...")
    _, _, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    print("Loading test datasets...")
    test_datasets = load_test_data(args.splits_dir, debug=False)

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

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

    all_results = {}

    for run_id in args.runs:
        ckpt_path = Path(f"models/model-{run_id}/checkpoint.pt")
        print(f"\n=== Run {run_id} ===")
        print(f"Loading checkpoint: {ckpt_path}")
        if not ckpt_path.exists():
            print(f"  MISSING: {ckpt_path}")
            continue

        model = Transolver(**model_config).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        test_metrics = {
            name: evaluate_split(model, loader, stats, args.surf_weight, device)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        test_avg_surf_p = test_avg["avg/mae_surf_p"]

        for name in TEST_SPLIT_NAMES:
            m = test_metrics[name]
            print(f"  {name}: surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} "
                  f"Uy={m['mae_surf_Uy']:.4f}]  vol[p={m['mae_vol_p']:.4f} "
                  f"Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]")
        print(f"  test_avg/mae_surf_p = {test_avg_surf_p:.4f}")

        all_results[run_id] = {
            "test_avg/mae_surf_p": test_avg_surf_p,
            "per_split": test_metrics,
            "test_avg_all": test_avg,
        }

        if args.update_wandb:
            import wandb
            api = wandb.Api()
            run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{run_id}")
            print(f"  Updating W&B summary for {run.name}...")
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    run.summary[f"test/{split_name}/{k}"] = v
            for k, v in test_avg.items():
                run.summary[f"test_{k}"] = v
            run.summary["test_avg/mae_surf_p_fixed"] = test_avg_surf_p
            run.summary.update()
            print(f"  Updated.")

    print("\n=== Summary ===")
    for run_id, r in all_results.items():
        print(f"  {run_id}: test_avg/mae_surf_p = {r['test_avg/mae_surf_p']:.4f}")

    if all_results:
        vals = [r["test_avg/mae_surf_p"] for r in all_results.values()]
        if len(vals) > 1:
            t = torch.tensor(vals, dtype=torch.float64)
            print(f"\n  mean = {t.mean().item():.4f}")
            print(f"  std  = {t.std(unbiased=True).item():.4f}")


if __name__ == "__main__":
    main()
