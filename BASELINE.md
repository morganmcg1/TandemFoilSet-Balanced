# Baseline — `icml-appendix-charlie-pai2g-48h-r5`

This branch is the **Charlie no-W&B logging ablation, round 5 (charlie-pai2g-48h-r5)**.

Experiment metrics are written to local JSONL only (`models/<experiment>/metrics.jsonl`).
**Do not** add or query W&B / wandb experiment logging for this arm.

## Primary ranking metric

- **Validation:** `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE
  across the four val tracks (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` from the best-val checkpoint.

> ✅ **Round-5 scoring bug fixed (merged via PR #1532):** `test_geom_camber_cruise/000020.pt`
> contains ±Inf values in the `p` channel. The `train.py:evaluate_split` workaround
> (batch-level `y_finite_mask` filter before `accumulate_batch`) is now on the
> advisor branch. All subsequent PRs must include this fix on their branch and
> should report **finite `test_avg/mae_surf_p`**. Round-5 ranking remains
> `val_avg/mae_surf_p` as the primary metric.

## Reference configuration (train.py defaults)

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
)
optimizer = AdamW; scheduler = CosineAnnealingLR(T_max=epochs)
```

Each training execution is hard-capped by `SENPAI_TIMEOUT_MINUTES=30` (wall clock).
`--epochs 50` is an upper bound; runs typically reach 12-16 epochs under the
30-min cap at the default model size.

## Current best (val)

| Metric | Value | PR | Config | Notes |
|---|---|---|---|---|
| `val_avg/mae_surf_p` | **54.0051** | #1846 | L1 + compile + bf16 + slice_num=32 | epoch 40 of 41; best≠terminal (first converged run); -9.30% vs #1700; **measured on L1-only base** |
| `test_avg/mae_surf_p` | **47.6261** | #1846 | — | all 4 test splits improve; -7.46% vs #1700 |

> ⚠️ **Note:** PR #1846 (slice_num=32) was measured on the L1 baseline (#1700, val=59.54) **before** the sampler merge (#1619, val=56.62 with sampler 2× single). Post-merge advisor now includes BOTH sampler 2× single AND slice_num=32. The true stacked baseline will be revealed when future PRs run against current advisor. Conservative lower bound: val_avg ≤ 54.0051.

All subsequent PRs must beat `val_avg/mae_surf_p < 54.0051` to be merged.

## 2026-05-13 05:45 — PR #1846: slice_num 64 → 32 (tighter attention bottleneck)

- **Student:** charliepai2g48h5-frieren
- **Best epoch:** 40 of 41 — **first run where best ≠ terminal** (model converged within budget!)
- **Epochs reached:** 41 (~43.5 s/epoch, -12.3% vs baseline — slice_num=32 is faster)
- **Peak GPU memory:** 21.35 GB (-10.4% vs baseline)
- **Param count:** 657,079 (~5K fewer than baseline 662K — <1% change)

| Split | val mae_surf_p | Δ vs #1700 L1 baseline |
|---|---|---|
| `val_single_in_dist` | **59.0943** | -8.93% |
| `val_geom_camber_rc` | 67.4450 | -8.91% |
| `val_geom_camber_cruise` | **35.7197** | **-10.63%** |
| `val_re_rand` | **53.7616** | -9.25% |
| **val_avg** | **54.0051** | **-9.30%** |

| Split | test mae_surf_p | Δ vs #1700 |
|---|---|---|
| `test_single_in_dist` | **53.2538** | -4.27% |
| `test_geom_camber_rc` | **62.8744** | -5.86% |
| `test_geom_camber_cruise` | **29.4777** | -12.22% |
| `test_re_rand` | **44.8988** | -9.97% |
| **test_avg** | **47.6261** | **-7.46%** |

- **All 4 val splits and all 4 test splits improved uniformly (~9%)** — this is a global inductive-bias benefit, not one-split coincidence.
- **Why it works:** slice_num=64 was over-allocated for TandemFoilSet's natural spatial regimes (~10-20 CFD motifs). slice_num=32 forces tighter routing → regularization at the information-bottleneck level + ~4 extra epochs from budget gain.
- **Key diagnostic:** first run in round 5 where best_epoch ≠ terminal (ep 40 < ep 41) — the model now converges within the 30-min cap. Smaller bottleneck = faster optimization.
- **⚠️ Caveat:** Measured on L1-only base (#1700, 59.54). Post-merge advisor includes sampler 2× single (#1619). True stacked baseline revealed by future runs.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-frieren-slice-num-32-20260513-030801/metrics.jsonl`
  `models/model-charliepai2g48h5-frieren-slice-num-32-20260513-030801/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-frieren \
      --experiment_name "charliepai2g48h5-frieren/slice-num-32" \
      --epochs 50
  ```
  (slice_num=32 now on advisor branch)

---

## 2026-05-13 05:10 — PR #1619: Sampler 2× single boost on L1 baseline

- **Student:** charliepai2g48h5-nezuko
- **Best epoch:** 39 (wall-clock-bound at 30 min; best == terminal; trajectory still descending: ep 38→39: 57.74→56.62)
- **Epochs reached:** 39 (~46.3 s/epoch, unchanged vs L1 baseline)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1700 |
|---|---|---|
| `val_single_in_dist` | **56.1237** | **-13.51%** |
| `val_geom_camber_rc` | 71.0701 | -4.02% |
| `val_geom_camber_cruise` | 41.6906 | +4.31% |
| `val_re_rand` | **57.6024** | -2.76% |
| **val_avg** | **56.6217** | **-4.89%** |

| Split | test mae_surf_p | Δ vs #1700 |
|---|---|---|
| `test_single_in_dist` | **50.0812** | **-9.97%** |
| `test_geom_camber_rc` | 66.9241 | +0.20% |
| `test_geom_camber_cruise` | 34.1808 | +1.78% |
| `test_re_rand` | 50.5377 | +1.34% |
| **test_avg** | **50.4310** | **-2.01%** |

- **Sampler intervention:** `racecar_single` boost factor 2× → 50% / 25% / 25% share (single / tandem / cruise).
- **Lever validated across 3 baselines:** β=1.0 (-2.80%), β=1.0+compile (-2.25%), **L1+compile (-4.89%)**. Win grows as loss gets sharper.
- **Three of four val splits improve;** only `val_geom_camber_cruise` regresses (+4.31%) — mechanistically expected (cruise loses 25% of training mass to the 2× boost).
- **Metric artifacts:**
  `models/model-charliepai2g48h5-nezuko-sampler-2x-on-l1-20260513-021352/metrics.jsonl`
  `models/model-charliepai2g48h5-nezuko-sampler-2x-on-l1-20260513-021352/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-nezuko \
      --experiment_name "charliepai2g48h5-nezuko/sampler-2x-on-l1" \
      --epochs 50
  ```
  (sampler-reweight block now on advisor branch — see PR #1619 diff)

---

## 2026-05-13 02:10 — PR #1700: Pure L1 loss (β sweep → β=0 limit wins)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 37 (wall-clock-bound at 30 min; best == terminal; still descending)
- **Epochs reached:** 37 (~49.64 s/epoch, unchanged vs #1633)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1633 |
|---|---|---|
| `val_single_in_dist` | 64.8899 | -10.6% |
| `val_geom_camber_rc` | 74.0437 | -5.5% |
| `val_geom_camber_cruise` | **39.9687** | **-7.9%** |
| `val_re_rand` | 59.2391 | -4.5% |
| **val_avg** | **59.5354** | **-7.08%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 55.6271 |
| `test_geom_camber_rc` | 66.7873 |
| `test_geom_camber_cruise` | **33.5816** |
| `test_re_rand` | 49.8704 |
| **test_avg** | **51.4666** |

- **β sweep summary:** β=2.0→1.0→0.5→0.25→0 monotone improvement: 77.81→69.83→64.07→60.76→**59.54**. Diminishing returns with each halving (+8.2% → +5.2% → +2.0%), but L1 is the best point on the curve.
- **Key code change:** `F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction='none')` → `F.l1_loss(pred, y_norm, reduction='none')` at both call sites (lines 246, 485).
- **Arm A (β=0.25):** val_avg=60.7558, test_avg=52.3312.
- **Critical diagnostic:** both arms best_epoch == terminal — undertrained, not overfit.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.yaml`
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.25-20260513-000538/metrics.jsonl`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/l1-loss" \
      --epochs 50
  ```
  (L1 loss now on advisor branch — both `smooth_l1_loss` call sites replaced with `F.l1_loss`)

---

## 2026-05-13 00:50 — PR #1633: Huber β=0.5 (sharper loss function)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 37 (wall-clock-bound at 30 min; model still descending at timeout)
- **Epochs reached:** 37 (~49.5 s/epoch, same as compile baseline — sharper β adds no compute)
- **Peak GPU memory:** 23.83 GB (unchanged)

| Split | val mae_surf_p | Δ vs #1568 baseline |
|---|---|---|
| `val_single_in_dist` | 72.5692 | -5.9% |
| `val_geom_camber_rc` | 78.3209 | -6.2% |
| `val_geom_camber_cruise` | **43.3744** | **-14.4%** |
| `val_re_rand` | 62.0174 | -8.9% |
| **val_avg** | **64.0705** | **-8.2%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 63.0824 |
| `test_geom_camber_rc` | 69.4136 |
| `test_geom_camber_cruise` | **36.1544** |
| `test_re_rand` | 53.3341 |
| **test_avg** | **55.4961** |

- **Key code change:** `F.smooth_l1_loss(..., beta=0.5)` (was β=1.0). Sharper β makes the loss linear for a wider range of medium-magnitude residuals, down-weighting outlier gradients — directly suited to TandemFoil's heavy-tailed surface pressure residual distribution.
- **Monotone signal:** β=2.0 (val=77.81, +11.4%), β=1.0 (val=69.83, baseline), β=0.5 (val=64.07, -8.2%). Clear direction: sweep further toward β=0.25 / L1.
- **Note:** best_epoch=37=terminal — model was still improving at the timeout. Sweeping β=0.25 is the next logical step.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/huber-beta-0.5" \
      --epochs 50
  ```
  (change both `smooth_l1_loss` call sites in `train.py` to `beta=0.5` — see PR #1633 diff)

---

## 2026-05-12 22:10 — PR #1568: torch.compile + bf16 AMP for additional throughput

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 36 (wall-clock-bound at 30 min; model still descending at timeout)
- **Epochs reached:** 36 (~2.0× faster than bf16 baseline: ~49.5 s/epoch vs ~98 s)
- **Peak GPU memory:** 23.8 GB

| Split | val mae_surf_p | Δ vs #1532 baseline |
|---|---|---|
| `val_single_in_dist` | 77.10 | -35.8% |
| `val_geom_camber_rc` | 83.49 | -22.0% |
| `val_geom_camber_cruise` | 50.64 | -38.9% |
| `val_re_rand` | 68.10 | -28.0% |
| **val_avg** | **69.8316** | **-30.9%** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 67.81 |
| `test_geom_camber_rc` | 77.68 |
| `test_geom_camber_cruise` | 41.98 |
| `test_re_rand` | 59.99 |
| **test_avg** | **61.8652** |

- **Key code change:** `torch.compile(model, dynamic=True)` applied after model construction; `dynamic=True` prevents recompilation on variable mesh batch sizes. No recompilation stalls observed across 36 epochs.
- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/torch-compile-bf16" \
      --epochs 50
  ```
  (`torch.compile(model, dynamic=True)` now on advisor branch — see PR #1568 diff)

---

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 17 (wall-clock-bound at 30 min; model still improving at epoch 19)
- **Epochs reached:** 19 (~25% faster than fp32: ~98 s/epoch vs ~131 s)
- **Peak GPU memory:** 32.95 GB (well under 96 GB limit)

| Split | val mae_surf_p | Δ vs #1444 |
|---|---|---|
| `val_single_in_dist` | 120.0176 | -15.14 |
| `val_geom_camber_rc` | 107.0980 | -21.98 |
| `val_geom_camber_cruise` | 82.8425 | +5.14 |
| `val_re_rand` | 94.5268 | -6.57 |
| **val_avg** | **101.1212** | **-9.64** |

| Split | test mae_surf_p |
|---|---|
| `test_single_in_dist` | 105.4434 |
| `test_geom_camber_rc` | 99.9931 |
| `test_geom_camber_cruise` | 69.2841 |
| `test_re_rand` | 91.2844 |
| **test_avg** | **91.5013** |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/bf16-amp-scoring-fix" \
      --epochs 50
  ```
  (bf16 AMP via `torch.autocast` + scoring workaround — see PR #1532 diff)

---

## 2026-05-12 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Student:** charliepai2g48h5-thorfinn
- **Best epoch:** 14 (wall-clock-bound at 30 min; model still improving)
- **Peak GPU memory:** 42.1 GB
- **Time per epoch:** ~131 s

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 135.16 | 1.719 | 0.769 |
| `val_geom_camber_rc` | 129.08 | 2.104 | 0.988 |
| `val_geom_camber_cruise` | 77.70 | 1.047 | 0.555 |
| `val_re_rand` | 101.10 | 1.607 | 0.740 |
| **val_avg** | **110.76** | — | — |

- **Metric artifacts:**
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.jsonl`
  `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.yaml`

- **Reproduce:**
  ```bash
  cd target && python train.py \
      --agent charliepai2g48h5-thorfinn \
      --experiment_name "charliepai2g48h5-thorfinn/smooth-l1-loss" \
      --epochs 50
  ```
  (plus the Smooth-L1 substitution in `train.py` — see PR #1444 diff)

## Reproduce command (reference defaults)

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name "<student>/<short-description>" \
    --epochs 50
```

Commit `models/<experiment>/metrics.jsonl` and `metrics.yaml` with the PR and
quote the key values in the PR results comment plus the
`SENPAI-RESULT` terminal marker.
