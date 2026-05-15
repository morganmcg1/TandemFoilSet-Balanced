"""Re-evaluate a saved checkpoint on test splits with a corrected metric
computation that handles NaN-in-y samples robustly.

Usage: ``python reeval.py <model-dir-or-id>``  (default: model-hncmk6wk)

The scoring.py logic intends to skip samples with non-finite y, but
``NaN * 0 = NaN`` propagates through the mask multiplication, contaminating
the per-channel MAE sums. Here we sanitize y before computing err, while
still excluding those samples via sample_mask. This matches the documented
intent of scoring.py without modifying that read-only module.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data import TEST_SPLIT_NAMES, load_test_data, pad_collate
from data.scoring import CHANNELS

from model_def import Transolver


def evaluate_split_clean(model, loader, stats, device):
    """Compute MAE per split, robustly handling NaN-in-y samples."""
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    n_bad_samples = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]

            B = y.shape[0]
            y_finite_per_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
            n_bad_samples += int((~y_finite_per_sample).sum().item())
            # Zero out non-finite y values so multiplication by 0 mask works correctly.
            y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            sample_keep = y_finite_per_sample.unsqueeze(-1).expand(-1, mask.shape[-1])
            effective = mask & sample_keep
            surf_mask = effective & is_surface
            vol_mask = effective & ~is_surface

            err = (pred_orig.double() - y_safe.double()).abs()
            mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
            mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
            n_surf += int(surf_mask.sum().item())
            n_vol += int(vol_mask.sum().item())

    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    out = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = float(s[i].item())
        out[f"mae_vol_{ch}"] = float(v[i].item())
    out["n_surf_nodes"] = n_surf
    out["n_vol_nodes"] = n_vol
    out["n_bad_samples_excluded"] = n_bad_samples
    return out


def main():
    import sys as _sys
    device = torch.device("cuda")
    ckpt_id = _sys.argv[1] if len(_sys.argv) > 1 else "model-hncmk6wk"
    ckpt_dir = Path(f"models/{ckpt_id}") if not ckpt_id.startswith("models/") else Path(ckpt_id)
    with open(ckpt_dir / "config.yaml") as f:
        model_config = yaml.safe_load(f)
    print(f"Loading checkpoint {ckpt_dir}: {model_config}")

    model = Transolver(**model_config).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=True))
    model.eval()

    splits_dir = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    with open(splits_dir / "stats.json") as f:
        stats_raw = json.load(f)
    stats = {k: torch.tensor(v, device=device) for k, v in stats_raw.items()}

    test_datasets = load_test_data(str(splits_dir), debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=2, pin_memory=True)

    results = {}
    for name in TEST_SPLIT_NAMES:
        loader = DataLoader(test_datasets[name], batch_size=4, shuffle=False, **loader_kwargs)
        m = evaluate_split_clean(model, loader, stats, device)
        results[name] = m
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")

    avg = {}
    for ch in CHANNELS:
        for loc in ("surf", "vol"):
            key = f"mae_{loc}_{ch}"
            avg[key] = sum(results[s][key] for s in TEST_SPLIT_NAMES) / len(TEST_SPLIT_NAMES)
    print("\n=== TEST AVG ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    out_path = f"logs/reeval_{ckpt_dir.name}.json"
    with open(out_path, "w") as f:
        json.dump({"per_split": results, "test_avg": avg}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
