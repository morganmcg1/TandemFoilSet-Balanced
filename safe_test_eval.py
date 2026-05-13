"""Safe 4-split test re-evaluation.

Re-evaluates a saved checkpoint on the held-out test splits with per-node
masking of non-finite ground truth. This works around the bug in
``data/scoring.py`` (read-only) where ``Inf * 0 = NaN`` poisons the
per-channel surface accumulator on ``test_geom_camber_cruise`` (sample 20
has Inf in 761 volume-node pressure values).

Per-node skip (rather than per-sample skip) excludes only the 761 Inf
positions from both numerator and denominator, giving correct surface
AND volume MAE on all 4 splits.

The Transolver class is loaded from ``train.py`` via importlib so that any
model-config field (e.g. ``pos_freq_bands``, ``pos_freq_surface_only``)
introduced in ``train.py`` is automatically supported here without
duplicating the model code.

Usage:
    python safe_test_eval.py --model_dir models/model-foo/
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

from data import TEST_SPLIT_NAMES, load_data, load_test_data, pad_collate


def _load_transolver_class() -> type:
    """Import the Transolver class from train.py without running module-level
    training code. We stub ``simple_parsing.parse`` to raise SystemExit at the
    ``cfg = sp.parse(Config)`` line in train.py — by that point the class
    definitions (MLP, TransolverBlock, Transolver) have already executed."""
    here = Path(__file__).resolve().parent
    import simple_parsing as sp_mod
    original_parse = sp_mod.parse
    sp_mod.parse = lambda *a, **k: (_ for _ in ()).throw(SystemExit)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_for_eval", here / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_for_eval"] = mod
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sp_mod.parse = original_parse
        sys.modules.pop("_train_for_eval", None)
    return mod.Transolver


Transolver = _load_transolver_class()


@torch.no_grad()
def safe_evaluate_split(model, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf_per_ch = torch.zeros(3, dtype=torch.float64, device=device)
    n_vol_per_ch = torch.zeros(3, dtype=torch.float64, device=device)

    for x, y, is_surface, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
        pred_orig = pred * stats["y_std"] + stats["y_mean"]

        y_finite = torch.isfinite(y)  # [B, N, 3]
        eff_mask = mask.unsqueeze(-1) & y_finite
        surf_eff = eff_mask & is_surface.unsqueeze(-1)
        vol_eff = eff_mask & (~is_surface.unsqueeze(-1))

        err = pred_orig.double() - y.double()
        err = torch.where(torch.isfinite(err), err.abs(), torch.zeros_like(err))

        mae_surf += (err * surf_eff.double()).sum(dim=(0, 1))
        mae_vol += (err * vol_eff.double()).sum(dim=(0, 1))
        n_surf_per_ch += surf_eff.double().sum(dim=(0, 1))
        n_vol_per_ch += vol_eff.double().sum(dim=(0, 1))

    s = mae_surf / n_surf_per_ch.clamp(min=1)
    v = mae_vol / n_vol_per_ch.clamp(min=1)
    chs = ("Ux", "Uy", "p")
    return {
        **{f"mae_surf_{c}": s[i].item() for i, c in enumerate(chs)},
        **{f"mae_vol_{c}": v[i].item() for i, c in enumerate(chs)},
        "n_surf_p": int(n_surf_per_ch[2].item()),
        "n_vol_p": int(n_vol_per_ch[2].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg_path = args.model_dir / "config.yaml"
    ckpt_path = args.model_dir / "checkpoint.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transolver(**cfg["model_config"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    _, _, stats, _ = load_data(args.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(args.splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    per_split = {}
    for name in TEST_SPLIT_NAMES:
        m = safe_evaluate_split(model, test_loaders[name], stats, device)
        per_split[name] = m
        print(f"  {name:30s}  surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
              f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]  "
              f"(n_surf_p={m['n_surf_p']}, n_vol_p={m['n_vol_p']})")

    keys = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in ("Ux", "Uy", "p")]
    test_avg = {}
    for k in keys:
        vals = [m[k] for m in per_split.values()]
        test_avg[f"avg/{k}"] = sum(vals) / len(vals)

    print(f"\n  TEST (safe 4-split)  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    print(f"                        avg_surf_Ux={test_avg['avg/mae_surf_Ux']:.4f}  avg_surf_Uy={test_avg['avg/mae_surf_Uy']:.4f}")
    print(f"                        avg_vol_p={test_avg['avg/mae_vol_p']:.4f}  avg_vol_Ux={test_avg['avg/mae_vol_Ux']:.4f}  avg_vol_Uy={test_avg['avg/mae_vol_Uy']:.4f}")

    out_path = args.model_dir / "test_safe_eval.json"
    with open(out_path, "w") as f:
        json.dump({"test_avg": test_avg, "test_splits": per_split,
                   "method": "per-node skip of non-finite y (excludes inf positions from both num and denom)"},
                  f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
