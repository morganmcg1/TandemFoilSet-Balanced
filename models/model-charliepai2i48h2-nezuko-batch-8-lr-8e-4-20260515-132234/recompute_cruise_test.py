"""One-off recomputation of test metrics with NaN-safe accumulation.

Pre-existing bug in data/scoring.py: when GT y has a NaN at any node, the
intent is to skip that sample. But the implementation does
    err = (pred - y).abs()                 # NaN propagates here
    mae += (err * mask).sum(...)           # NaN * 0 = NaN in IEEE float
so a single NaN sample contaminates the whole split.

In test_geom_camber_cruise/000020.pt, the pressure channel of y contains
NaN values, so the split's mae_*_p comes back NaN, and the overall
test_avg/mae_surf_p is NaN.

This script reloads the best checkpoint and accumulates MAE for ALL four
test splits, using torch.nan_to_num on the error tensor to neutralise the
contaminated sample before masking — i.e., the metric the scoring code is
*supposed* to produce.

Uses batch_size=8 to match the original test-eval attention behavior
(padding interactions on attention).
"""
from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load Transolver class without triggering the full training-script execution.
# We do this by parsing train.py source and exec'ing only the lines up to the
# point where simple-parsing kicks in.
TRAIN_PY = ROOT / "train.py"
src = TRAIN_PY.read_text()
cut_marker = "cfg = sp.parse(Config)"
src_classes_only = src[: src.find(cut_marker)]
exec_ns: dict = {}
exec(compile(src_classes_only, str(TRAIN_PY), "exec"), exec_ns)
Transolver = exec_ns["Transolver"]

from data import load_test_data, pad_collate  # noqa: E402
from data.scoring import CHANNELS, finalize_split  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent
CKPT = MODEL_DIR / "checkpoint.pt"
BATCH_SIZE = 8  # match the original train-time test eval (cfg.batch_size)

with open(MODEL_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transolver(**cfg["model_config"]).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

splits_dir = Path(cfg["splits_dir"])
with open(splits_dir / "stats.json") as f:
    raw = json.load(f)
stats = {
    k: torch.tensor(raw[k], dtype=torch.float32).to(device)
    for k in ("x_mean", "x_std", "y_mean", "y_std")
}

test_splits = load_test_data(splits_dir)
out_all: dict[str, dict[str, float]] = {}
for name, ds in test_splits.items():
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=0
    )
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    skipped = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device); y = y.to(device); is_surface = is_surface.to(device); mask = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]

            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not y_finite.any():
                skipped += int(B)
                continue
            sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
            effective = mask & sample_mask
            surf_mask = effective & is_surface
            vol_mask = effective & ~is_surface

            err = (pred_orig.double() - y.double()).abs()
            # Bug fix: zero out NaN before masked-sum so NaN*0 doesn't poison the
            # accumulator. surf_mask is already False on samples we want to skip.
            err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
            mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
            mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
            n_surf += int(surf_mask.sum().item())
            n_vol += int(vol_mask.sum().item())
            skipped += int((~y_finite).sum().item())

    out = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
    out_all[name] = {**out, "n_skipped_nonfinite_gt": float(skipped)}
    print(f"{name}: skipped={skipped} samples")
    for k, v in out.items():
        print(f"    {k} = {v:.6f}")

print()
print("--- Corrected test_avg ---")
agg: dict[str, float] = {}
keys = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in CHANNELS]
for k in keys:
    vals = [m[k] for m in out_all.values() if k in m]
    agg[f"avg/{k}"] = sum(vals) / len(vals)
    print(f"  avg/{k} = {agg[f'avg/{k}']:.6f}")

out_path = MODEL_DIR / "test_metrics_corrected.json"
with open(out_path, "w") as f:
    json.dump({"test_splits": out_all, "test_avg": agg, "batch_size": BATCH_SIZE}, f, indent=2)
print(f"\nWrote {out_path}")
