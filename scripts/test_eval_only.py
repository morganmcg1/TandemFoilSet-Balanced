"""Run end-of-run test evaluation on a saved checkpoint.

Recovers the test_avg/test_3split metrics when the original training run's
test eval step crashed (e.g. OOM from a co-tenant on the GPU).

Usage:
  python scripts/test_eval_only.py --run_id 3qudhi04 --batch_size 1 --update_wandb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import wandb
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_test_data,
    pad_collate,
)
from data.loader import load_data


def _load_train_module_defs() -> dict:
    """Execute only the importable top of train.py to get the model classes.

    train.py has module-level script code starting at ``cfg = sp.parse(Config)``;
    we slice everything before that and exec it so Transolver/evaluate_split
    become available without launching a training run.
    """
    src = (ROOT / "train.py").read_text()
    marker = "cfg = sp.parse(Config)"
    if marker not in src:
        raise RuntimeError("train.py layout changed; update marker in test_eval_only.py")
    header = src.split(marker, 1)[0]
    import types
    mod = types.ModuleType("train_classes_only")
    mod.__file__ = str(ROOT / "train.py")
    sys.modules["train_classes_only"] = mod
    ns = mod.__dict__
    ns["__name__"] = "train_classes_only"
    exec(compile(header, str(ROOT / "train.py"), "exec"), ns)
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Use 1 to minimize peak memory during test eval (default).")
    p.add_argument("--update_wandb", action="store_true",
                   help="Update the W&B run summary with the recovered test metrics")
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    args = p.parse_args()

    api = wandb.Api()
    run_path = f"wandb-applied-ai-team/senpai-v1/{args.run_id}"
    run = api.run(run_path)
    cfg = dict(run.config)
    print(f"Run: {run.name}  (state={run.state})")
    print(f"  slice_num={cfg.get('slice_num')} n_head={cfg.get('n_head')} "
          f"mlp_ratio={cfg.get('mlp_ratio')} use_swiglu={cfg.get('use_swiglu')}")
    print(f"  asinh_p_scale={cfg.get('asinh_p_scale')} asinh_vel_scale={cfg.get('asinh_vel_scale')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = ROOT / "models" / f"model-{args.run_id}" / "checkpoint.pt"
    print(f"Checkpoint: {ckpt_path}")
    assert ckpt_path.exists(), f"Missing checkpoint at {ckpt_path}"

    defs = _load_train_module_defs()
    Transolver = defs["Transolver"]
    evaluate_split = defs["evaluate_split"]

    model_config = dict(
        space_dim=2,
        fun_dim=X_DIM - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=cfg["n_head"],
        slice_num=cfg["slice_num"],
        mlp_ratio=cfg["mlp_ratio"],
        use_swiglu=cfg["use_swiglu"],
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    _, _, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(args.splits_dir, debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    test_metrics: dict[str, dict[str, float]] = {}
    for name in TEST_SPLIT_NAMES:
        loader = test_loaders[name]
        t0 = time.time()
        with torch.no_grad():
            m = evaluate_split(
                model, loader, stats, cfg.get("surf_weight", 10.0), device,
                asinh_p_scale=cfg.get("asinh_p_scale", 0.0),
                asinh_vel_scale=cfg.get("asinh_vel_scale", 0.0),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        test_metrics[name] = m
        print(
            f"  {name:30s}  {dt:.1f}s  peak={peak_gb:.1f}GB  "
            f"surf[p={m['mae_surf_p']:.4f}, Ux={m['mae_surf_Ux']:.4f}, "
            f"Uy={m['mae_surf_Uy']:.4f}]  vol_p={m['mae_vol_p']:.4f}"
        )

    test_avg = aggregate_splits(test_metrics)
    print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")

    three_splits = ["test_single_in_dist", "test_geom_camber_rc", "test_re_rand"]
    test_3split_surf_p = sum(test_metrics[s]["mae_surf_p"] for s in three_splits) / 3.0
    print(f"  TEST  test_3split/mae_surf_p={test_3split_surf_p:.4f}  (mean of 3 valid splits)")
    print(f"\nPer-split test mae_surf_p:")
    for name in TEST_SPLIT_NAMES:
        print(f"  {name:30s}  {test_metrics[name]['mae_surf_p']:.4f}")

    if args.update_wandb:
        print(f"\nUpdating W&B summary for {args.run_id}...")
        test_log: dict[str, float] = {}
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                test_log[f"test/{split_name}/{k}"] = v
        for k, v in test_avg.items():
            test_log[f"test_{k}"] = v
        test_log["test_3split/mae_surf_p"] = test_3split_surf_p
        test_log["test_3split/mae_surf_Ux"] = sum(test_metrics[s]["mae_surf_Ux"] for s in three_splits) / 3.0
        test_log["test_3split/mae_surf_Uy"] = sum(test_metrics[s]["mae_surf_Uy"] for s in three_splits) / 3.0
        for k, v in test_log.items():
            run.summary[k] = v
        run.summary.update()
        print(f"  Updated {len(test_log)} keys")


if __name__ == "__main__":
    main()
