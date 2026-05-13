"""Safe re-eval of test metrics for this checkpoint.

The cruise test split has one ground-truth sample with +Inf in y[:, 2] (p
channel). `data/scoring.py` is read-only, and its `err * surf_mask` step
hits `Inf * 0 = NaN`, which propagates through `.sum()` and poisons the
per-channel accumulator. The fix documented in BASELINE.md is to zero-fill
non-finite y values BEFORE the subtraction, then proceed normally — this
preserves the intended "skip non-finite samples" semantics and recovers a
clean number for all 4 test splits.

This script does not modify `data/scoring.py` (read-only); it reuses the
exact accumulator/finalize helpers from `data/scoring.py` with one
pre-processing step: clone y and set y[non-finite] = 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Make the target package importable when run from anywhere
ROOT = Path(__file__).resolve().parents[2]
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


def _load_transolver_class():
    """Import the Transolver class from train.py without running module-level
    training code. We stub `simple_parsing.parse` to raise SystemExit at the
    `cfg = sp.parse(Config)` line in train.py — by that point the class
    definitions (MLP, TransolverBlock, Transolver) have already executed."""
    import importlib.util
    import simple_parsing as sp_mod
    original_parse = sp_mod.parse
    sp_mod.parse = lambda *a, **k: (_ for _ in ()).throw(SystemExit)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_for_eval", ROOT / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_for_eval"] = mod  # dataclass needs this
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sp_mod.parse = original_parse
        sys.modules.pop("_train_for_eval", None)
    return mod.Transolver


Transolver = _load_transolver_class()


def main() -> None:
    here = Path(__file__).resolve().parent
    ckpt_path = here / "checkpoint.pt"
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    splits_dir = cfg["splits_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, stats, _ = load_data(splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_splits = load_test_data(splits_dir)

    model_cfg = cfg["model_config"]
    model = Transolver(**model_cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    per_split: dict[str, dict[str, float]] = {}
    safe_eval_log: list[dict] = []

    with torch.no_grad():
        for name in TEST_SPLIT_NAMES:
            loader = DataLoader(
                test_splits[name], batch_size=cfg["batch_size"],
                shuffle=False, collate_fn=pad_collate, num_workers=2,
            )
            mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
            mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
            n_surf = n_vol = 0
            n_samples_seen = 0
            n_samples_with_nonfinite_y = 0

            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # Safe pre-processing: zero-fill non-finite y BEFORE any
                # subtraction so Inf cannot leak into `err = (pred - y).abs()`.
                # accumulate_batch already drops samples whose y is non-finite
                # anywhere via its `y_finite` mask, so this zero-fill changes
                # nothing about which samples count — it only prevents the
                # `Inf * 0 = NaN` accumulator poison.
                B = y.shape[0]
                y_finite_per_sample = torch.isfinite(
                    y.reshape(B, -1)
                ).all(dim=-1)
                n_samples_seen += B
                n_samples_with_nonfinite_y += int((~y_finite_per_sample).sum().item())
                y_safe = torch.where(
                    torch.isfinite(y), y, torch.zeros_like(y),
                )

                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                pred = model({"x": x_norm})["preds"]
                pred_orig = pred * stats["y_std"] + stats["y_mean"]

                ds, dv = accumulate_batch(
                    pred_orig, y_safe, is_surface, mask, mae_surf, mae_vol,
                )
                n_surf += ds
                n_vol += dv

            per_split[name] = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
            safe_eval_log.append({
                "split": name,
                "n_samples_total": n_samples_seen,
                "n_samples_with_nonfinite_y": n_samples_with_nonfinite_y,
                "n_surf_accumulated": n_surf,
                "n_vol_accumulated": n_vol,
                "metrics": per_split[name],
            })

    test_avg = aggregate_splits(per_split)
    record = {
        "event": "test_safe_reeval",
        "best_epoch": cfg.get("epochs", None),
        "test_avg": test_avg,
        "test_splits": per_split,
        "safe_eval_audit": safe_eval_log,
    }

    out_path = here / "test_safe_eval.jsonl"
    with open(out_path, "w") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    log_path = here / "test_safe_eval.log"
    with open(log_path, "w") as f:
        f.write("Safe re-eval of test metrics (zero-fill non-finite y before subtraction)\n")
        f.write("=" * 72 + "\n\n")
        for entry in safe_eval_log:
            f.write(
                f"{entry['split']}: {entry['n_samples_total']} samples, "
                f"{entry['n_samples_with_nonfinite_y']} skipped (non-finite y)\n"
            )
            m = entry["metrics"]
            f.write(
                f"  mae_surf  p={m['mae_surf_p']:.4f}  "
                f"Ux={m['mae_surf_Ux']:.4f}  Uy={m['mae_surf_Uy']:.4f}\n"
            )
            f.write(
                f"  mae_vol   p={m['mae_vol_p']:.4f}  "
                f"Ux={m['mae_vol_Ux']:.4f}  Uy={m['mae_vol_Uy']:.4f}\n\n"
            )
        f.write("Aggregated (equal-weight mean across 4 splits):\n")
        for k, v in test_avg.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    print("\ntest_avg (safe re-eval):")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
