"""Post-hoc nansafe test evaluation.

Re-evaluates a saved checkpoint on the test splits using a per-element/channel
nansafe accumulator. This avoids the in-tree ``data/scoring.py`` bug where one
``-inf`` ground-truth value contaminates the entire split metric via
``0 * inf = NaN``.

Usage:
    python research/nansafe_eval.py --run_id <wandb-run-id> [--upload]

The checkpoint is loaded from ``models/model-<run_id>/checkpoint.pt`` (the path
``train.py`` saves to). Model config is reconstructed from the W&B run config.

Computes the nansafe metric: a value ``y[i, n, c]`` is included only if
``isfinite(y[i, n, c])`` is True; surface vs volume membership uses the
existing ``is_surface`` flag. The bad cruise sample contributes its surface
nodes (all finite) and its non-pressure interior nodes; only its -inf interior
pressure positions are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

# Add parent so we can import data + Transolver from train.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import TEST_SPLIT_NAMES, X_DIM, load_test_data, pad_collate  # noqa: E402

# train.py runs ``simple_parsing.parse(Config)`` at import time, so spoof argv
# while importing the model class.
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from train import Transolver  # noqa: E402
finally:
    sys.argv = _saved_argv


CHANNELS = ("Ux", "Uy", "p")


def nansafe_evaluate(model, loader, stats, device) -> dict[str, float]:
    """Evaluate one split with per-element/channel nansafe accumulation.

    Returns per-channel surf/vol MAE plus per-channel counts so multiple runs
    can be averaged consistently.
    """
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = torch.zeros(3, dtype=torch.float64, device=device)
    n_vol = torch.zeros(3, dtype=torch.float64, device=device)

    skipped_elements = 0
    total_elements = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]

            err = (pred_orig.double() - y.double()).abs()
            y_finite = torch.isfinite(y)  # [B, N, 3] per element
            err = torch.where(y_finite, err, torch.zeros_like(err))

            surf3 = (mask & is_surface).unsqueeze(-1) & y_finite  # [B, N, 3]
            vol3 = (mask & ~is_surface).unsqueeze(-1) & y_finite

            mae_surf += (err * surf3.double()).sum(dim=(0, 1))
            mae_vol += (err * vol3.double()).sum(dim=(0, 1))
            n_surf += surf3.sum(dim=(0, 1)).double()
            n_vol += vol3.sum(dim=(0, 1)).double()

            valid_node_mask = mask.unsqueeze(-1).expand_as(y_finite)
            total_elements += int(valid_node_mask.sum().item())
            skipped_elements += int((valid_node_mask & ~y_finite).sum().item())

    out: dict[str, float] = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}_nansafe"] = (mae_surf[i] / n_surf[i].clamp(min=1)).item()
        out[f"mae_vol_{ch}_nansafe"] = (mae_vol[i] / n_vol[i].clamp(min=1)).item()
        out[f"n_surf_{ch}_nansafe"] = int(n_surf[i].item())
        out[f"n_vol_{ch}_nansafe"] = int(n_vol[i].item())
    out["skipped_elements"] = skipped_elements
    out["total_elements"] = total_elements
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True, help="W&B run ID")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--upload", action="store_true", help="Write nansafe metrics back into the W&B run summary")
    args = p.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    cfg = dict(run.config)
    model_cfg = cfg["model_config"]
    print(f"Run: {run.name} ({args.run_id})")
    print(f"Best val_avg/mae_surf_p: {run.summary.get('best_val_avg/mae_surf_p')}")

    ckpt = Path("models") / f"model-{args.run_id}" / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transolver(**model_cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Stats: derive from the splits_v2 stats.json.
    import json
    with open(Path(args.splits_dir) / "stats.json") as f:
        raw_stats = json.load(f)
    stats = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in raw_stats.items()}

    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, ds in test_datasets.items()
    }

    per_split: dict[str, dict[str, float]] = {}
    for name, loader in test_loaders.items():
        print(f"\nEvaluating {name} (nansafe)...")
        m = nansafe_evaluate(model, loader, stats, device)
        per_split[name] = m
        print(f"  mae_surf_p_nansafe = {m['mae_surf_p_nansafe']:.4f}  "
              f"(skipped {m['skipped_elements']}/{m['total_elements']} elements)")

    # Aggregate equal-weight across splits.
    avg: dict[str, float] = {}
    for ch in CHANNELS:
        for loc in ("surf", "vol"):
            key = f"mae_{loc}_{ch}_nansafe"
            vals = [per_split[s][key] for s in per_split]
            avg[f"avg/{key}"] = sum(vals) / len(vals)

    print("\n=== Per-split nansafe surf_p ===")
    for name in TEST_SPLIT_NAMES:
        print(f"  {name:<32s}  {per_split[name]['mae_surf_p_nansafe']:.4f}")
    print(f"  {'test_avg_nansafe/mae_surf_p':<32s}  {avg['avg/mae_surf_p_nansafe']:.4f}")

    if args.upload:
        live = wandb.init(
            entity=args.entity,
            project=args.project,
            id=args.run_id,
            resume="must",
        )
        log: dict[str, float] = {}
        for split_name, m in per_split.items():
            for k, v in m.items():
                log[f"test/{split_name}/{k}"] = v
        for k, v in avg.items():
            log[f"test_{k}"] = v
        wandb.summary.update(log)
        live.finish()
        print(f"\nUploaded {len(log)} nansafe entries to W&B summary for {args.run_id}")


if __name__ == "__main__":
    main()
