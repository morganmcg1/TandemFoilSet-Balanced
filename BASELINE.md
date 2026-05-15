# Baseline — TandemFoilSet (willow-pai2i-48h-r5)

## Current best — PR #3444 (Cosine T_max=14)

**val_avg/mae_surf_p = 93.1996** (W&B run: `1hx2rm1n`, PR #3444 cosine T_max=14 on Huber + Fourier σ=10)
**test_avg/mae_surf_p = 83.5377** (same run `1hx2rm1n`, clean 4-split thanks to merged #3296)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 114.80 | 105.93 |
| geom_camber_rc | 104.16 | 90.03 |
| geom_camber_cruise | **68.17** | **57.65** ¹ |
| re_rand | 85.66 | 80.55 |

¹ 199/200 samples evaluated — `splits_v2/.test_geom_camber_cruise_gt/000020.pt` has 761 inf y-values, dropped from MAE accumulator. Pred-side `nan_to_num` guards against any model overflow.

**Comparison vs prior best (PR #3098 + #3296):**

| Split | Prior test mae (xvn4gllg) | New test mae (1hx2rm1n) | Δ |
|-------|---------------------------:|--------------------------:|---:|
| single_in_dist | 109.30 | 105.93 | −3.1% |
| geom_camber_rc | 103.19 | 90.03 | **−12.8%** |
| geom_camber_cruise | 60.61 | 57.65 | −4.9% |
| re_rand | 86.90 | 80.55 | −7.3% |
| **avg** | **90.00** | **83.54** | **−7.2%** |

Every test split improves; the biggest win is on `geom_camber_rc` (−12.8%), the previously hardest split.

**PR #3444 (cosine T_max=14):** 1-LOC scheduler period change. The 30-min wall-clock binds at ~epoch 14, but the cosine schedule was tuned for T_max=50 → LR never decayed below 82% of peak. Setting T_max=14 lets the cosine schedule complete inside the wall-clock budget, giving the final 2-4 epochs proper fine-tuning at low LR.

**Reproduce (PR #3444):**
```bash
cd target/
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group round2-cosine-tmax-thorfinn \
  --cosine_t_max 14 \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 16 --fourier_sigma 10.0 \
  --wandb_name thorfinn-arm-B-tmax-14
```

---

## PR #3098 + #3296 (prior best) — Huber + NaN guard

**val_avg/mae_surf_p = 96.0548** (W&B run: `md6so639`, PR #3098 SmoothL1 β=0.05)
**test_avg/mae_surf_p = 90.0004** (W&B run: `xvn4gllg`, PR #3296 two-pronged NaN guard on Huber)

| Split | val mae_surf_p (md6so639) | test mae_surf_p (xvn4gllg) |
|-------|---------------------------|----------------------------|
| single_in_dist | 109.64 | 109.30 |
| geom_camber_rc | 112.30 | 103.19 |
| geom_camber_cruise | **73.22** | **60.61** |
| re_rand | 89.06 | 86.90 |

**PR #3098 (val):** SmoothL1 (Huber) loss with β=0.05 replacing MSE. Effect: val_avg 130.46 → 96.05 (-26.4% vs PR #3123). All 4 val splits improved.

**PR #3296 (test):** Two-pronged NaN guard (pred-side `nan_to_num` + y-side sample mask) in both `evaluate_split` and the training loop. First valid test_avg of the launch.

---

## PR #3123 (2026-05-15) — earlier reference

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
