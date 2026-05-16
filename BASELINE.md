# Baseline — TandemFoilSet (willow-pai2i-48h-r5)

## Current best — PR #3672 (Fourier ablation: n_fourier=0 under FiLM+Lion+EMA)

**val_avg/mae_surf_p = 70.3432** (W&B run: `297qot5r`, PR #3672 n_fourier=0 + FiLM-output log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14)
**test_avg/mae_surf_p = 61.6253** (same run `297qot5r`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 79.64 | 69.97 |
| geom_camber_rc | 82.43 | 73.96 |
| geom_camber_cruise | **51.50** | **42.22** |
| re_rand | 67.80 | 60.35 |

**Comparison vs prior best (PR #3405 FiLM+Lion+EMA, val 71.6544 / test 62.1091):**

| Split | Prior val (ksltdq7a) | New val (297qot5r) | Δval | Prior test | New test | Δtest |
|-------|---------------------:|--------------------:|-----:|-----------:|---------:|------:|
| single_in_dist | 81.17 | 79.64 | −1.53 | 71.30 | 69.97 | −1.33 |
| geom_camber_rc | 84.45 | 82.43 | −2.02 | 73.87 | 73.96 | +0.09 |
| geom_camber_cruise | 51.99 | 51.50 | −0.49 | 42.84 | 42.22 | −0.62 |
| re_rand | 69.01 | 67.80 | −1.21 | 60.43 | 60.35 | −0.08 |
| **avg** | **71.65** | **70.34** | **−1.31** | **62.11** | **61.63** | **−0.48** |

All 4 val splits improve; 3/4 test splits improve (camber_rc +0.09 test, within noise).

**PR #3672 (Fourier ablation — n_fourier=0):** Under FiLM+Lion+EMA, dropping Fourier positional features entirely (n_fourier=0) slightly outperforms both σ=3 (val 71.28) and σ=10 baseline (val 71.65). FiLM conditioning on log(Re) already encodes the flow-regime information that Fourier PE was providing, making Fourier redundant and slightly harmful. Dropping Fourier simplifies the architecture (removes ~1.1K RFF params, one hyperparameter, one coordinate transform per forward pass).

**Reproduce (PR #3672):**
```bash
cd target/
python train.py --agent willowpai2i48h5-alphonse --epochs 50 \
  --wandb_group round5-film-fourier-alphonse \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name alphonse-r5-film-nofourier
```

---

## Prior best — PR #3405 (FiLM conditioning + Lion + EMA)

**val_avg/mae_surf_p = 71.6544** (W&B run: `ksltdq7a`, PR #3405 FiLM-output on log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) on Huber + Fourier σ=10 + T_max=14)
**test_avg/mae_surf_p = 62.1091** (same run `ksltdq7a`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 81.17 | 71.30 |
| geom_camber_rc | 84.45 | 73.87 |
| geom_camber_cruise | **51.99** | **42.84** ¹ |
| re_rand | 69.01 | 60.43 |

¹ 199/200 samples evaluated — `splits_v2/.test_geom_camber_cruise_gt/000020.pt` dropped via y-side mask. Previously null due to flat key naming; now correctly resolved to 42.84 by scan of nested W&B keys.

**Comparison vs prior best (PR #3537 Lion, val 77.58 / test 68.88):**

| Split | Prior test mae (yvkf9glr) | New test mae (ksltdq7a) | Δ |
|-------|---------------------------:|--------------------------:|---:|
| single_in_dist | 81.69 | 71.30 | **−12.7%** |
| geom_camber_rc | 77.94 | 73.87 | **−5.2%** |
| geom_camber_cruise | 48.83 | 42.84 | **−12.3%** |
| re_rand | 67.04 | 60.43 | **−9.9%** |
| **avg** | **68.88** | **62.11** | **−9.8%** |

All 4 splits improve; largest gains on `single_in_dist` and `geom_camber_cruise`.

**PR #3405 (FiLM conditioning):** FiLM (Feature-wise Linear Modulation) conditions the surface-pressure model on log(Re) via gamma/beta affine transforms at the network output. log(Re) encodes the Reynolds-number regime of each flow sample. On the OOD `re_rand` split, FiLM adds the most value (val 69.01 vs Lion-only 72.93), and the `geom_camber_cruise` split sees the largest absolute improvement on test (48.83 → 42.84). Combined with Lion optimizer + EMA(0.997) + Fourier σ=10 + Huber β=0.05 + cosine T_max=14.

**Reproduce (PR #3405):**
```bash
cd target/
python train.py --agent willowpai2i48h5-nezuko --epochs 50 \
  --wandb_group round4-film-ema-lion-nezuko \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 16 --fourier_sigma 10.0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name nezuko-r4-film-ema997-lion
```

---

## PR #3537 (prior best) — Lion optimizer

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 90.85 | 81.69 |
| geom_camber_rc | 87.72 | 77.94 |
| geom_camber_cruise | **58.81** | **48.83** ¹ |
| re_rand | 72.93 | 67.04 |

¹ 199/200 samples evaluated — `splits_v2/.test_geom_camber_cruise_gt/000020.pt` has 761 inf y-values, dropped from MAE accumulator.

**Comparison vs prior best (PR #3444, T_max=14):**

| Split | Prior test mae (1hx2rm1n) | New test mae (yvkf9glr) | Δ |
|-------|---------------------------:|--------------------------:|---:|
| single_in_dist | 105.93 | 81.69 | **−22.9%** |
| geom_camber_rc | 90.03 | 77.94 | **−13.4%** |
| geom_camber_cruise | 57.65 | 48.83 | **−15.3%** |
| re_rand | 80.55 | 67.04 | **−16.8%** |
| **avg** | **83.54** | **68.88** | **−17.5%** |

Every test split improves substantially; the biggest gain is on `single_in_dist` (−22.9%). This is the largest single-mechanism improvement in the launch.

**PR #3537 (Lion optimizer):** Sign-based update rule (Chen et al. 2023, arXiv 2302.06675) replacing AdamW. Decoupled weight decay → sign update → momentum decay. β₁=0.9, β₂=0.99. Note: paper recommends batch ≥ 64 but Lion works strongly even at batch_size=4 in our regime, likely because the irregular-mesh CFD loss landscape is well-suited to sign updates.

**Reproduce (PR #3537):**
```bash
cd target/
python train.py --agent willowpai2i48h5-askeladd --epochs 50 \
  --wandb_group round3-lion-askeladd \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 16 --fourier_sigma 10.0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --wandb_name askeladd-r3-arm-A-lion-lr5e5-wd1e3
```

---

## PR #3444 (prior best) — Cosine T_max=14

**val_avg/mae_surf_p = 93.1996** (W&B run: `1hx2rm1n`)
**test_avg/mae_surf_p = 83.5377** (same run)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 114.80 | 105.93 |
| geom_camber_rc | 104.16 | 90.03 |
| geom_camber_cruise | 68.17 | 57.65 |
| re_rand | 85.66 | 80.55 |

**PR #3444:** 1-LOC scheduler period change. The 30-min wall-clock binds at ~epoch 14, but the cosine schedule was tuned for T_max=50 → LR never decayed below 82% of peak. Setting T_max=14 lets the cosine schedule complete inside the wall-clock budget.

---

## PR #3098 + #3296 (earlier baseline) — Huber + NaN guard

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
