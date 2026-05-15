"""Recompute test metrics with NaN-safe accumulation.

The current ``data/scoring.py`` uses ``err * mask`` where ``err`` can contain
NaN at masked-out positions (when GT has NaN), and NaN*0 = NaN in PyTorch,
so the entire MAE sum becomes NaN. To get a correct test_avg without
touching scoring.py, this script filters out the sample(s) with non-finite
GT before running the standard evaluator. Skipping the whole sample matches
what scoring.py *intends* to do for non-finite GT samples.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import (
    SPLITS_DIR,
    TEST_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_test_data,
    pad_collate,
)
from data.loader import _load_stats


def _load_train_module() -> object:
    """Import train.py with sentinel guard so its top-level training loop
    doesn't fire when we just want the model class.

    We rely on ``__name__`` to gate; since train.py executes at import,
    we instead splice out the model classes via importlib and stop at the
    line that begins script execution.
    """
    # The Transolver class is defined entirely above the training code.
    src_path = Path(__file__).parent / "train.py"
    text = src_path.read_text()
    cutoff = text.index("# ---------------------------------------------------------------------------\n# Evaluation helpers")
    header = text[:cutoff]
    # Create a synthetic module
    spec = importlib.util.spec_from_loader("train_models_only", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(header, mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--slice_num", type=int, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, default=str(SPLITS_DIR))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--cond_dim", type=int, default=0,
                    help="FiLM conditioning dim (0=disabled). H13 uses 0.")
    ap.add_argument("--surface_head_hidden_dim", type=int, default=64,
                    help="Hidden dim of the post-trunk SurfaceHead MLP. "
                         "Set 0 to disable (H13 default is 64).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_mod = _load_train_module()
    Transolver = train_mod.Transolver
    SurfaceHead = train_mod.SurfaceHead
    apply_surface_head = train_mod.apply_surface_head

    stats = _load_stats(Path(args.splits_dir))
    stats = {k: v.to(device) for k, v in stats.items()}

    model_config = dict(
        space_dim=2,
        fun_dim=X_DIM - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=4,
        slice_num=args.slice_num,
        mlp_ratio=2,
        cond_dim=args.cond_dim,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    surface_head = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if args.surface_head_hidden_dim > 0 and "surface_head" in ckpt:
            surface_head = SurfaceHead(
                in_dim=3, hidden_dim=args.surface_head_hidden_dim, out_dim=3
            ).to(device)
            surface_head.load_state_dict(ckpt["surface_head"])
            surface_head.eval()
            print(
                f"Loaded SurfaceHead (hidden_dim={args.surface_head_hidden_dim}) "
                f"from ckpt at epoch {ckpt.get('epoch', '?')}"
            )
    else:
        # Legacy: bare state_dict (no surface head)
        model.load_state_dict(ckpt)
    model.eval()

    test_datasets = load_test_data(Path(args.splits_dir), debug=False)

    # Filter out samples whose GT has non-finite values. The scoring code is
    # supposed to skip these but its mask-multiply path propagates NaN.
    filtered_counts: dict[str, int] = {}
    for name, ds in test_datasets.items():
        keep_x, keep_gt = [], []
        n_skipped = 0
        for xf, gf in zip(ds.x_files, ds.gt_files):
            gt = torch.load(gf, weights_only=True)
            if torch.isfinite(gt["y"]).all():
                keep_x.append(xf)
                keep_gt.append(gf)
            else:
                n_skipped += 1
        ds.x_files = keep_x
        ds.gt_files = keep_gt
        filtered_counts[name] = n_skipped
    print(f"Skipped GT-nonfinite test samples: {filtered_counts}")

    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=4, pin_memory=True)
        for name, ds in test_datasets.items()
    }

    # Use the standard scoring path: it is correct when all samples are clean.
    from data.scoring import accumulate_batch, finalize_split

    def evaluate_clean(loader):
        vol_loss_sum = surf_loss_sum = 0.0
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = n_batches = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]
                pred = model({"x": x_norm})["preds"]
                if surface_head is not None:
                    pred = apply_surface_head(pred, surface_head, is_surface, mask)
                sq_err = (pred - y_norm) ** 2
                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vol_loss_sum += (
                    (sq_err * vol_mask.unsqueeze(-1)).sum()
                    / vol_mask.sum().clamp(min=1)
                ).item()
                surf_loss_sum += (
                    (sq_err * surf_mask.unsqueeze(-1)).sum()
                    / surf_mask.sum().clamp(min=1)
                ).item()
                n_batches += 1
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                ds_n, dv_n = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
                n_surf += ds_n
                n_vol += dv_n
        vol_loss = vol_loss_sum / max(n_batches, 1)
        surf_loss = surf_loss_sum / max(n_batches, 1)
        out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
               "loss": vol_loss + args.surf_weight * surf_loss}
        out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
        return out

    test_metrics = {name: evaluate_clean(loader) for name, loader in test_loaders.items()}
    test_avg = aggregate_splits(test_metrics)
    print(f"\n  TEST (NaN-skip)  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(
            f"    {name:<26s} "
            f"loss={m['loss']:.4f}  "
            f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
            f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
        )

    record = {
        "event": "test_recompute_skip_nan_gt",
        "checkpoint": args.checkpoint,
        "slice_num": args.slice_num,
        "cond_dim": args.cond_dim,
        "surface_head_hidden_dim": (
            args.surface_head_hidden_dim if surface_head is not None else 0
        ),
        "filtered_counts": filtered_counts,
        "test_avg": test_avg,
        "test_splits": test_metrics,
    }
    with open(args.out_jsonl, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"\nSaved corrected test metrics to {args.out_jsonl}")


if __name__ == "__main__":
    main()
