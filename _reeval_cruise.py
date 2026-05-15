"""Re-evaluate test_geom_camber_cruise excluding sample 20 (which has inf in y[:, 2]).

The scoring helper in `data/scoring.py` filters non-finite y per-sample, but then
multiplies `(pred - y).abs()` by a zeroed mask — and inf*0 = NaN in PyTorch.
That makes `test_geom_camber_cruise/mae_surf_p` and `mae_vol_p` NaN whenever any
sample in the split has inf in y[..., 2].

This script loads each arm's best checkpoint and recomputes the surf/vol p MAE
on test_geom_camber_cruise with sample 20 manually excluded, mirroring the
intent of scoring.py's `y_finite` filter without the inf*0=NaN pitfall.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data import load_test_data, pad_collate
from data.scoring import CHANNELS
from train import Transolver, X_DIM


def reeval(ckpt_path: Path, splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_splits = load_test_data(splits_dir)
    cruise = test_splits["test_geom_camber_cruise"]

    # Same model_config as train.py
    model = Transolver(
        space_dim=2,
        fun_dim=X_DIM - 2,
        out_dim=3,
        n_hidden=128,
        n_layers=5,
        n_head=4,
        slice_num=64,
        mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Load stats (same as data.load_data)
    import json
    with open(Path(splits_dir) / "stats.json") as f:
        s = json.load(f)
    stats = {
        "x_mean": torch.tensor(s["x_mean"], dtype=torch.float32, device=device),
        "x_std":  torch.tensor(s["x_std"],  dtype=torch.float32, device=device),
        "y_mean": torch.tensor(s["y_mean"], dtype=torch.float32, device=device),
        "y_std":  torch.tensor(s["y_std"],  dtype=torch.float32, device=device),
    }

    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = 0
    n_vol = 0

    excluded = []

    loader = DataLoader(cruise, batch_size=1, shuffle=False, collate_fn=pad_collate, num_workers=0)
    with torch.no_grad():
        for idx, (x, y, is_surface, mask) in enumerate(loader):
            if not torch.isfinite(y).all().item():
                excluded.append(idx)
                continue
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = model({"x": x_norm})["preds"]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]

            err = (pred_orig.double() - y.double()).abs()
            surf_m = (mask & is_surface).unsqueeze(-1).double()
            vol_m = (mask & ~is_surface).unsqueeze(-1).double()
            mae_surf += (err * surf_m).sum(dim=(0, 1))
            mae_vol += (err * vol_m).sum(dim=(0, 1))
            n_surf += int((mask & is_surface).sum().item())
            n_vol += int((mask & ~is_surface).sum().item())

    s_div = mae_surf / max(n_surf, 1)
    v_div = mae_vol / max(n_vol, 1)
    out = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = s_div[i].item()
        out[f"mae_vol_{ch}"] = v_div[i].item()
    print(f"\n{ckpt_path}")
    print(f"  excluded samples (non-finite y): {excluded}")
    print(f"  evaluated samples: {len(cruise) - len(excluded)}")
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")
    return out


if __name__ == "__main__":
    arm1 = Path("models/model-h3-gradclip-warmup-20260515-135926/checkpoint.pt")
    arm2 = Path("models/model-h3-gradclip0.5-warmup-20260515-143246/checkpoint.pt")
    r1 = reeval(arm1)
    r2 = reeval(arm2)

    # Test split metrics from metrics.yaml of each arm (3 splits that already passed).
    other_arm1 = {
        "test_geom_camber_rc": 118.02985627330379,
        "test_re_rand": 100.26568464156199,
        "test_single_in_dist": 135.6643475671379,
    }
    other_arm2 = {
        "test_geom_camber_rc": 123.25113797061165,
        "test_re_rand": 112.21467181451344,
        "test_single_in_dist": 140.96257823119316,
    }
    avg1 = (r1["mae_surf_p"] + sum(other_arm1.values())) / 4
    avg2 = (r2["mae_surf_p"] + sum(other_arm2.values())) / 4
    print(f"\nCorrected test_avg/mae_surf_p (arm1 grad_clip=1.0): {avg1:.4f}")
    print(f"Corrected test_avg/mae_surf_p (arm2 grad_clip=0.5): {avg2:.4f}")
