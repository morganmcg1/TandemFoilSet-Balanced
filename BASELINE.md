# Baseline — icml-appendix-charlie-pai2g-24h-r5

Charlie no-W&B logging arm, round 5. Each training execution is capped at
`SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap). Local JSONL metrics only,
no W&B.

## Current best

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p** | **114.40** | PR #1519 (merged 2026-05-12) |
| **test_avg/mae_surf_p** | **107.57** | PR #1564 (merged 2026-05-12) — GT-NaN fix |

### Per-split val (epoch 13)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist | 140.78 | 1.701 | 0.801 |
| val_geom_camber_rc | 123.10 | 2.406 | 1.012 |
| val_geom_camber_cruise | 89.71 | 1.177 | 0.590 |
| val_re_rand | 104.02 | 1.737 | 0.805 |
| **val_avg** | **114.40** | 1.756 | 0.802 |

### Per-split test (epoch 13, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist | 122.65 | 1.663 | 0.769 |
| test_geom_camber_rc | 111.09 | 2.332 | 0.942 |
| test_geom_camber_cruise | 92.41 | 1.179 | 0.612 |
| test_re_rand | 104.14 | 1.595 | 0.775 |
| **test_avg** | **107.57** | 1.692 | 0.775 |

## 2026-05-12 21:05 — PR #1564: GT-NaN fix in evaluate_split (MERGED)

- **val_avg/mae_surf_p: 114.40** (unchanged — bit-identical to #1519; fix is a no-op on clean GT)
- **test_avg/mae_surf_p: 107.57** (was NaN; now the first valid paper-facing test number)
- **Metric artifacts:** `models/model-gt_nan_fix_baseline-20260512-201204/metrics.jsonl`
- **What changed:** In `evaluate_split`, filter non-finite GT samples before calling `accumulate_batch`:
  `gt_finite_mask = torch.isfinite(y).all(dim=-1)` then AND into `mask` and `is_surface`. Fixes
  IEEE `NaN * 0 = NaN` leakage in `data/scoring.py` (which is read-only). Val results are bit-identical
  to #1519; only `test_geom_camber_cruise/mae_surf_p` changes (NaN → 92.41).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name gt_nan_fix_baseline --agent <student>
  ```

## 2026-05-12 20:10 — PR #1519: Warmup + cosine to 13-epoch budget (MERGED)

- **val_avg/mae_surf_p: 114.40** (↓ 8.6% from informal 125.20 baseline)
- **test_avg/mae_surf_p: NaN** (cruise GT issue; 3-split clean = 112.63)
- **Metric artifacts:** `models/model-warmup3_cosine13-20260512-190738/metrics.jsonl`
- **What changed:** Added 3-epoch linear warmup before cosine; matched `--epochs 13` to
  actual wall-clock budget so cosine LR reaches near-zero (was T_max=50, only 14 epochs ran,
  LR never decayed meaningfully). Added seed=42 and `nan_to_num` prediction guard.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name warmup3_cosine13 --agent <student>
  ```
  (Plus seed pin and nan_to_num guard from merged `train.py`.)

## Baseline configuration (from `target/train.py`)

| Lever | Value |
|---|---|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Activation | GELU |
| Loss | MSE in normalized space: `vol_loss + 10 * surf_loss` |
| Surface weight | 10.0 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| LR schedule | CosineAnnealingLR, T_max=epochs |
| Batch size | 4 (variable mesh sizes, pad_collate to N_max) |
| Sampler | WeightedRandomSampler (balanced domain mix) |
| Max epochs | 50 (configurable via `--epochs`) |
| Timeout | 30 min wall-clock (hard cap) |
| Precision | FP32 |

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four
validation splits:

- `val_single_in_dist` — single-foil random holdout (sanity check)
- `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar)
- `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise)
- `val_re_rand` — stratified Re holdout across tandem domains

Best checkpoint = lowest `val_avg/mae_surf_p`. Test eval at end of training
uses that checkpoint and reports `test_avg/mae_surf_p` for the paper.

## How to claim a win

A PR is a winner if its terminal `SENPAI-RESULT` marker reports a strictly
lower `val_avg/mae_surf_p` than this file's current best (or, before the
first merge, beats the out-of-the-box baseline of the same epoch/timeout
budget by a clearly statistically meaningful margin and reports the
matching test number).

## Update procedure

When a new winner merges, update this file with:

- The merged PR number
- New `val_avg/mae_surf_p`
- Matching `test_avg/mae_surf_p`
- One-line note on what changed
