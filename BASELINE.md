# Baseline — TandemFoilSet (willow-pai2i-48h-r5)

## Current best — PR #3123 (2026-05-15)

**val_avg/mae_surf_p = 130.46** (W&B run: `24yldhv7`)  
**test_avg/mae_surf_p = NaN** ⚠️ — test_geom_camber_cruise split produces NaN for all runs (baseline-side bug, not introduced by this PR — tracked in follow-up NaN-fix PR)

| Split | val mae_surf_p |
|-------|---------------|
| val_single_in_dist | 159.57 |
| val_geom_camber_rc | 150.12 |
| val_geom_camber_cruise | **89.02** |
| val_re_rand | 123.13 |

Added: Random Fourier positional features over (x,z) coordinates, n_fourier=16, sigma=10.0

**Reproduce:**
```bash
cd target/
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group fourier-pe-thorfinn \
  --n_fourier 16 --fourier_sigma 10.0 \
  --wandb_name thorfinn-arm-C-fourier16
```

---

## Starting point (unmerged baseline reference)

**val_avg/mae_surf_p = 135.23** (Arm A from PR #3123, W&B: `jyqygcbx` — no Fourier features)

| Split | val mae_surf_p |
|-------|---------------|
| val_single_in_dist | 156.98 |
| val_geom_camber_rc | 144.01 |
| val_geom_camber_cruise | 119.48 |
| val_re_rand | 120.44 |

The baseline architecture is defined in `target/train.py`:

| Param | Value |
|-------|-------|
| model | Transolver |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| params | ~1.5M |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| loss | MSE (normalized space) |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| epochs cap | 50 |
| timeout cap | 30 min |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)  
**Test metric:** `test_avg/mae_surf_p` (lower is better)

Per-split metrics for each run are logged to W&B under `wandb-applied-ai-team/senpai-v1`.

---

_Updated when a PR is merged: add PR#, W&B run ID, val_avg/mae_surf_p, test_avg/mae_surf_p, and per-split breakdown._
