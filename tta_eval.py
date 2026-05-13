"""Test-time augmentation eval: average predictions over N AoA-jittered copies.

Lives alongside ``train.py`` and reuses its model class + augment function so
the perturbations applied at test time match the training-time augmentation
distribution exactly. The canonical (no-jitter) pass is one of the N — i.e.
N=1 reduces to the same eval as ``train.py``'s test step.

The cruise split has 761 +Inf in y[:, 2] (sample 20); we zero-fill non-finite
ground truth before the subtraction so the cruise split is reported alongside
the other three (matching ``models/.../safe_re_eval.py`` semantics).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)


def _load_train_module():
    """Import train.py without running its module-level training code.

    Stubs ``simple_parsing.parse`` to raise ``SystemExit`` at the
    ``cfg = sp.parse(Config)`` line; by that point all class and function
    definitions (Transolver, augment_geometry, ...) have executed.
    """
    import simple_parsing as sp_mod
    original_parse = sp_mod.parse
    sp_mod.parse = lambda *a, **k: (_ for _ in ()).throw(SystemExit)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_for_tta", ROOT / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_for_tta"] = mod
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sp_mod.parse = original_parse
    return mod


def tta_forward(
    model,
    x: torch.Tensor,
    stats: dict,
    augment_geometry,
    n_passes: int,
    aoa_jitter_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model n_passes times (canonical + jittered), average normalized preds.

    Returns (mean_pred_normalized, std_pred_normalized). Both ``[B, N, 3]``.
    """
    preds = []
    # Canonical pass (no jitter) — one of the N
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    preds.append(model({"x": x_norm})["preds"])
    # N-1 jittered passes — AoA-only jitter
    jitter_cfg = SimpleNamespace(aoa_jitter_rad=aoa_jitter_rad, naca_jitter=0.0)
    for _ in range(n_passes - 1):
        x_jit = augment_geometry(x, jitter_cfg)
        x_jit_norm = (x_jit - stats["x_mean"]) / stats["x_std"]
        preds.append(model({"x": x_jit_norm})["preds"])
    stacked = torch.stack(preds, dim=0)  # [N, B, N_nodes, 3]
    return stacked.mean(dim=0), stacked.std(dim=0)


def evaluate_tta_split(
    model,
    loader,
    stats,
    augment_geometry,
    device,
    n_passes: int,
    aoa_jitter_rad: float,
):
    """Returns (per-split metrics dict, audit dict including pred-std stats)."""
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    n_samples_seen = 0
    n_samples_with_nonfinite_y = 0
    pred_std_surf_p_sum = 0.0
    pred_std_surf_p_count = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # TTA forward (in normalized space)
            mean_pred_norm, std_pred_norm = tta_forward(
                model, x, stats, augment_geometry,
                n_passes=n_passes, aoa_jitter_rad=aoa_jitter_rad,
            )
            # Denormalize
            pred_orig = mean_pred_norm * stats["y_std"] + stats["y_mean"]
            # Pred std in physical units (per channel scale = y_std)
            pred_std_phys = std_pred_norm * stats["y_std"]

            # Safe zero-fill non-finite y before subtraction
            B = y.shape[0]
            y_finite_per_sample = torch.isfinite(
                y.reshape(B, -1)
            ).all(dim=-1)
            n_samples_seen += B
            n_samples_with_nonfinite_y += int((~y_finite_per_sample).sum().item())
            y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

            ds, dv = accumulate_batch(
                pred_orig, y_safe, is_surface, mask, mae_surf, mae_vol,
            )
            n_surf += ds
            n_vol += dv

            # Average prediction-std on surface nodes for the pressure channel,
            # over finite-y samples only (matching the metric semantics).
            sample_mask = y_finite_per_sample.unsqueeze(-1).expand(-1, mask.shape[-1])
            surf_eff = mask & is_surface & sample_mask
            n_surf_batch = int(surf_eff.sum().item())
            if n_surf_batch > 0:
                surf_p_std = (pred_std_phys[..., 2] * surf_eff.double()).sum().item()
                pred_std_surf_p_sum += surf_p_std
                pred_std_surf_p_count += n_surf_batch

    metrics = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
    audit = {
        "n_samples_total": n_samples_seen,
        "n_samples_with_nonfinite_y": n_samples_with_nonfinite_y,
        "n_surf_accumulated": n_surf,
        "n_vol_accumulated": n_vol,
        "mean_pred_std_surf_p": (
            pred_std_surf_p_sum / max(pred_std_surf_p_count, 1)
        ),
    }
    return metrics, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Path to models/<experiment-stamp>/ directory")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--n_passes", type=int, default=5)
    parser.add_argument("--aoa_jitter_rad", type=float, default=0.00873)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <model_dir>/tta_eval.json)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for TTA random perturbations (reproducibility)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    ckpt_path = model_dir / args.checkpoint
    cfg_path = model_dir / "config.yaml"
    assert ckpt_path.exists(), f"missing checkpoint: {ckpt_path}"
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())

    out_path = (
        Path(args.output).resolve() if args.output
        else model_dir / f"tta_eval_N{args.n_passes}_jit{args.aoa_jitter_rad:g}.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_mod = _load_train_module()
    Transolver = train_mod.Transolver
    augment_geometry = train_mod.augment_geometry

    # stats from training data
    _, _, stats, _ = load_data(cfg["splits_dir"])
    stats = {k: v.to(device) for k, v in stats.items()}

    # build and load model
    model_cfg = cfg["model_config"]
    model = Transolver(**model_cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    # test loaders
    test_splits = load_test_data(cfg["splits_dir"])
    test_loaders = {
        name: DataLoader(
            ds, batch_size=cfg["batch_size"],
            shuffle=False, collate_fn=pad_collate, num_workers=2,
        )
        for name, ds in test_splits.items()
    }

    per_split: dict[str, dict[str, float]] = {}
    audit_per_split: dict[str, dict] = {}
    for name in TEST_SPLIT_NAMES:
        metrics, audit = evaluate_tta_split(
            model, test_loaders[name], stats, augment_geometry,
            device, n_passes=args.n_passes, aoa_jitter_rad=args.aoa_jitter_rad,
        )
        per_split[name] = metrics
        audit_per_split[name] = audit

    test_avg = aggregate_splits(per_split)
    record = {
        "event": "tta_test_eval",
        "model_dir": str(model_dir),
        "checkpoint": args.checkpoint,
        "n_passes": args.n_passes,
        "aoa_jitter_rad": args.aoa_jitter_rad,
        "seed": args.seed,
        "test_avg": test_avg,
        "test_splits": per_split,
        "audit": audit_per_split,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")
    print(f"\ntest_avg (TTA N={args.n_passes}, jitter={args.aoa_jitter_rad}):")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
