# Baseline — icml-appendix-charlie-pai2g-24h-r5

Charlie no-W&B logging arm, round 5. Each training execution is capped at
`SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap). Local JSONL metrics only,
no W&B.

## Current best

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p** | **105.46** | PR #1483 (merged 2026-05-12) — grad clip + warmup+cosine |
| test_avg/mae_surf_p | TBD* | PR #1483 reported test NaN due to source-branch lacking GT-NaN fix; merged code now has both grad_clip AND GT-NaN fix — re-measure on next run |

*PR #1483's source branch did not include the GT-NaN fix from PR #1564, so its test_avg was NaN. The merged code combines `clip_grad_norm_(max=1.0)` (from #1483) with warmup+cosine schedule (from #1519) and the GT-NaN evaluate_split filter (from #1564). The next reproduce run will produce a finite test_avg.

### Per-split val (PR #1483, epoch 13, **note: ran on pre-warmup baseline**)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist | 112.93 | 1.445 | 0.699 |
| val_geom_camber_rc | 122.87 | 2.467 | 0.957 |
| val_geom_camber_cruise | 83.98 | 1.001 | 0.556 |
| val_re_rand | 102.08 | 1.763 | 0.745 |
| **val_avg** | **105.46** | 1.669 | 0.739 |

### Per-split test (PR #1483, partial — cruise NaN under source branch's old eval code)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist | 107.65 | 1.521 | 0.675 |
| test_geom_camber_rc | 106.42 | 2.324 | 0.881 |
| test_geom_camber_cruise | NaN | 0.980 | 0.514 |
| test_re_rand | 97.10 | 1.495 | 0.703 |
| **test_avg (3-split clean)** | **103.72** | — | — |

## 2026-05-12 21:55 — PR #1483: Gradient clipping max_norm=1.0 (MERGED)

- **val_avg/mae_surf_p: 105.46** (↓ 7.8% from 114.40 — biggest single-step gain this round)
- **test_avg/mae_surf_p:** Reported NaN by source branch (lacked GT-NaN fix). Merged code now has both fixes — re-measure on next run.
- **Metric artifacts:** `models/model-grad_clip_1-20260512-210428/metrics.jsonl`
- **What changed:** Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()` — 1 line in train.py.
- **Why it worked (per tanjiro's analysis):** Pre-clip grad norms are 45–112 throughout training, all well above max_norm=1.0. Clipping fires on EVERY step → effectively renormalises every gradient to unit norm. This is much stronger than "tame occasional outliers" — it's closer to "Adam on g/‖g‖". The largest gains are on splits with highest target magnitudes (val_single_in_dist −34.4%, val_geom_camber_rc −29.2%) — consistent with a Re-rebalancing interpretation: extreme-Re samples no longer dominate gradient steps.
- **Note on baseline measurement:** PR #1483 measured 105.46 against the OLD baseline (no warmup+cosine, no GT-NaN fix), reporting a 26.7% within-PR improvement. The merged code now stacks: warmup+cosine + GT-NaN fix + grad_clip. Composed val_avg may differ slightly from 105.46; next experiments to verify.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name baseline_check --agent <student>
  ```

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
