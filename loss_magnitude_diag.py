"""Loss-magnitude characterization at convergence.

Reproduces the diagnostic from PR #601 results comment for a δ=0.05 run:
- Loads the best checkpoint
- Computes per-element |residual| in normalized space across all 4 val splits
- Reports lin fraction at thresholds 0.05 (current δ) and 0.10 (prior δ)
- Per-channel breakdown and aggregate

Run from target/:
    python loss_magnitude_diag.py --model_dir models/<exp>
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

from data import VAL_SPLIT_NAMES, X_DIM, load_data, pad_collate

THRESHOLDS = [0.05, 0.10]
CHANNELS = ["Ux", "Uy", "p"]


def _import_model_classes():
    """Load Transolver class from train.py without running its top-level training code."""
    train_path = Path(__file__).parent / "train.py"
    src = train_path.read_text()
    # Trim everything from the first top-level "cfg = sp.parse(Config)" onward.
    idx = src.find("\ncfg = sp.parse(Config)")
    if idx == -1:
        raise RuntimeError("Could not find cfg = sp.parse(Config) in train.py")
    safe_src = src[:idx]
    mod = type(sys)("train_models")
    mod.__file__ = str(train_path)
    sys.modules["train_models"] = mod
    mod.__dict__["__name__"] = "train_models"
    exec(compile(safe_src, str(train_path), "exec"), mod.__dict__)
    return mod.Transolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg_path = args.model_dir / "config.yaml"
    ckpt_path = args.model_dir / "checkpoint.pt"
    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"Loading checkpoint: {ckpt_path}")

    Transolver = _import_model_classes()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    model_config = dict(
        space_dim=2,
        fun_dim=X_DIM - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=4,
        slice_num=64,
        mlp_ratio=2,
        drop_path_max=cfg.get("drop_path_max", 0.1),
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    per_split = {}
    agg_total = 0
    agg_counts = {t: 0 for t in THRESHOLDS}
    agg_chan_total = [0, 0, 0]
    agg_chan_counts = {t: [0, 0, 0] for t in THRESHOLDS}

    for split_name in VAL_SPLIT_NAMES:
        loader = val_loaders[split_name]
        thr_counts = {t: 0 for t in THRESHOLDS}
        chan_counts = {t: [0, 0, 0] for t in THRESHOLDS}
        chan_total = [0, 0, 0]
        total = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]
                pred = model({"x": x_norm})["preds"]
                err = (pred - y_norm).abs()  # [B, N, 3]
                m = mask.unsqueeze(-1)  # [B, N, 1]
                # Aggregate (surf+vol combined)
                m_full = m.expand_as(err)
                total += int(m_full.sum().item())
                for t in THRESHOLDS:
                    thr_counts[t] += int(((err >= t) & m_full).sum().item())
                # Per-channel
                for c in range(3):
                    chan_total[c] += int(mask.sum().item())
                    for t in THRESHOLDS:
                        chan_counts[t][c] += int(((err[..., c] >= t) & mask).sum().item())

        per_split[split_name] = {
            "n_total": total,
            "thr_counts": {str(t): v for t, v in thr_counts.items()},
            "chan_counts": {str(t): v for t, v in chan_counts.items()},
            "chan_total": chan_total,
        }
        agg_total += total
        for t in THRESHOLDS:
            agg_counts[t] += thr_counts[t]
            for c in range(3):
                agg_chan_counts[t][c] += chan_counts[t][c]
        for c in range(3):
            agg_chan_total[c] += chan_total[c]

        print(f"\n{split_name} (N={total/1e6:.2f} M points)")
        for t in THRESHOLDS:
            lin = thr_counts[t] / max(total, 1)
            print(f"  thr={t}: lin={lin*100:.1f}%  quad={(1-lin)*100:.1f}%")
        for c, ch in enumerate(CHANNELS):
            line = f"  per-channel {ch}:"
            for t in THRESHOLDS:
                f = chan_counts[t][c] / max(chan_total[c], 1)
                line += f"  thr={t}: lin={f*100:.1f}% quad={(1-f)*100:.1f}%"
            print(line)

    print(f"\n=== AGGREGATE (N={agg_total/1e6:.2f} M points) ===")
    for t in THRESHOLDS:
        lin = agg_counts[t] / max(agg_total, 1)
        print(f"  thr={t}: lin={lin*100:.2f}%  quad={(1-lin)*100:.2f}%")
    for c, ch in enumerate(CHANNELS):
        line = f"  per-channel {ch}:"
        for t in THRESHOLDS:
            f = agg_chan_counts[t][c] / max(agg_chan_total[c], 1)
            line += f"  thr={t}: lin={f*100:.2f}% quad={(1-f)*100:.2f}%"
        print(line)

    out = {
        "per_split": per_split,
        "aggregate": {
            "n_total": agg_total,
            "thr_counts": {str(t): v for t, v in agg_counts.items()},
            "chan_counts": {str(t): v for t, v in agg_chan_counts.items()},
            "chan_total": agg_chan_total,
        },
        "thresholds": THRESHOLDS,
    }
    out_path = args.model_dir / "loss_magnitude.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
