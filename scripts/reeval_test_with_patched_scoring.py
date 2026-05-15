"""Re-evaluate the saved checkpoint on test splits with a patched accumulator
that handles non-finite GT correctly (avoiding Inf * 0 = NaN poisoning).

This is a one-shot recovery script for the existing surf_weight=25 run whose
test_geom_camber_cruise MAE was reported as NaN due to one corrupt sample
(.test_geom_camber_cruise_gt/000020.pt has 761 Inf values in pressure).

It does NOT modify data/scoring.py — that file is read-only during normal
experiment PRs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    finalize_split,
    load_test_data,
    pad_collate,
)

# Import Transolver without triggering train.py's module-level training run.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_train_src", REPO_ROOT / "train.py")
_src = _spec.loader.get_source("_train_src")
_marker = "# ---------------------------------------------------------------------------\n# Evaluation helpers"
_model_src = _src.split(_marker)[0]
_mod = type(sys)("_train_models")
exec(compile(_model_src, str(REPO_ROOT / "train.py"), "exec"), _mod.__dict__)
Transolver = _mod.Transolver

CHECKPOINT_PATH = REPO_ROOT / "models" / "model-hkka77kg" / "checkpoint.pt"
SPLITS_DIR = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
SURF_WEIGHT = 25.0
BATCH_SIZE = 4


def patched_accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol):
    """Same semantics as data.scoring.accumulate_batch but Inf-safe.

    The original multiplies err by a 0/1 mask after computing
    (pred - y).abs(), which yields NaN when y has Inf (Inf * 0 = NaN). We
    zero out the bad sample rows BEFORE multiplying so the running sum stays
    finite. Per-sample skipping semantics are preserved.
    """
    B = y.shape[0]
    y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
    if not y_finite.any():
        return 0, 0

    sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
    effective = mask & sample_mask
    surf_mask = effective & is_surface
    vol_mask = effective & ~is_surface

    y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
    err = (pred_orig.double() - y_safe.double()).abs()
    mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    return int(surf_mask.sum().item()), int(vol_mask.sum().item())


def evaluate_split(model, loader, stats, surf_weight, device):
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

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface

            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
            vol_mask_safe = vol_mask & sample_mask
            surf_mask_safe = surf_mask & sample_mask

            sq_err_safe = torch.where(torch.isfinite(sq_err), sq_err,
                                      torch.zeros_like(sq_err))
            vol_loss_sum += (
                (sq_err_safe * vol_mask_safe.unsqueeze(-1)).sum()
                / vol_mask_safe.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err_safe * surf_mask_safe.unsqueeze(-1)).sum()
                / surf_mask_safe.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = patched_accumulate_batch(
                pred_orig, y, is_surface, mask, mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {
        "vol_loss": vol_loss,
        "surf_loss": surf_loss,
        "loss": vol_loss + surf_weight * surf_loss,
    }
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from data.loader import _load_stats
    stats = _load_stats(Path(SPLITS_DIR))
    stats = {k: v.to(device) for k, v in stats.items()}

    model_config = dict(
        space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=128, n_layers=5,
        n_head=4, slice_num=64, mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    test_datasets = load_test_data(SPLITS_DIR)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    print("Re-evaluating test splits with Inf-safe accumulator...")
    test_metrics = {}
    for name in TEST_SPLIT_NAMES:
        m = evaluate_split(model, test_loaders[name], stats, SURF_WEIGHT, device)
        test_metrics[name] = m
        print(
            f"  {name:<26s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
            f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
        )

    test_avg = aggregate_splits(test_metrics)
    print("\n--- test_avg (4-split mean) ---")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
