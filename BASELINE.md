# Baseline — icml-appendix-charlie-pai2g-24h-r5

Charlie no-W&B logging arm, round 5. Each training execution is capped at
`SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap). Local JSONL metrics only,
no W&B.

## Current best

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p** | **73.15** | PR #1641 (merged 2026-05-13) — Lion optimizer (lr=3e-4) on BF16+grad_clip stack |
| **test_avg/mae_surf_p** | **66.76** | PR #1641 — all 4 splits finite |
| Peak VRAM | 42.11 GB | PR #1641 — Lion run (FP32; merged stack now has BF16, expect ~33 GB) |
| s/epoch | 134.3 s | PR #1641 — Lion run pre-BF16; with BF16 expect ~100 s |

### Per-split val (PR #1641, epoch 13)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist | 80.78 | 0.906 | 0.498 |
| val_geom_camber_rc | 90.86 | 1.614 | 0.736 |
| val_geom_camber_cruise | 51.56 | 0.567 | 0.400 |
| val_re_rand | 69.39 | 1.078 | 0.554 |
| **val_avg** | **73.15** | 1.041 | 0.547 |

### Per-split test (PR #1641, epoch 13 best checkpoint)

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 69.02 |
| test_geom_camber_rc | 77.38 |
| test_geom_camber_cruise | 59.49 |
| test_re_rand | 61.14 |
| **test_avg** | **66.76** |

## 2026-05-13 01:20 — PR #1641: Lion optimizer (lr=3e-4) (MERGED)

- **val_avg/mae_surf_p: 73.15** (↓ 22.4% from 94.22 — largest single-PR gain this round)
- **test_avg/mae_surf_p: 66.76** (↓ 23.4% from 87.10)
- **Peak VRAM: 42.11 GB** (FP32 run; merged stack now has BF16 so future runs ~33 GB)
- **Metric artifacts:** `models/model-charliepai2g24h5-frieren-lion_lr3e4-20260512-225827/metrics.jsonl` (winner); `models/model-charliepai2g24h5-frieren-lion_lr1_5e4-20260512-235646/metrics.jsonl` (arm 1)
- **What changed:** Replaced AdamW with Lion optimizer from `lion-pytorch`. Lion lr=3e-4 (= AdamW lr/3.3), lion_wd=6e-5 (= wd/1.67). Added `lion_lr` and `lion_weight_decay` fields to Config. All other config identical: grad_clip(max_norm=1.0), warmup3+cosine13, MSE loss, batch=4, surf_weight=10.0, seed=42.
- **Why it worked:** Lion's per-parameter sign-based update is a stronger version of the global gradient renormalization our baseline already applies via grad_clip. Where grad_clip renormalizes the full gradient vector to L2-norm ≤ 1, Lion takes each parameter's gradient and applies `±lr` regardless of its magnitude — per-parameter sign quantization. This eliminates gradient magnitude variance entirely across the parameter space, producing uniform step sizes across attention slices, MLP layers, and projection matrices. The Transolver's heterogeneous parameter structure (different gradient scales in PhysicsAttention vs MLP projections) benefits strongly from this uniformity. Two arms: lr=1.5e-4 (val 75.17) and lr=3e-4 (val 73.15 — winner).
- **Reproduce:**
  ```bash
  pip install lion-pytorch && cd target/ && python train.py --epochs 13 --lion_lr 3e-4 --lion_weight_decay 6e-5 --experiment_name lion_lr3e4 --agent <student>
  ```

## 2026-05-13 01:05 — PR #1565: BF16 autocast (MERGED)

- **val_avg/mae_surf_p: 94.22** (↓ 1.3% from 95.44)
- **test_avg/mae_surf_p: 87.10** (↓ 0.8% from 87.83)
- **Peak VRAM: 32.94 GB** (↓ 22% from 42.11 GB — unlocks future wider-model experiments)
- **s/epoch: 100.87** (↓ 23% from 131.44 — same 30-min budget now fits more iterations)
- **Metric artifacts:** `models/model-charliepai2g24h5-fern-bf16_only_lr1e3-20260513-001209/metrics.jsonl`
- **What changed:** Added `torch.cuda.amp.autocast(dtype=torch.bfloat16)` wrapping the forward pass inside `train_epoch`. No batch-size change (kept batch=4). No schedule change. Single-line addition.
- **Why it worked:** BF16 reduces memory bandwidth for activations, speeding up compute. The slight metric improvement (−1.3% val, −0.8% test) likely comes from a mild implicit regularisation effect from reduced precision. The VRAM win (+9 GB headroom) is the more important outcome — it enables revisiting n_hidden=192 or n_layers=7 with the current stack.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name bf16_only_lr1e3 --agent <student>
  ```
  (train.py now has BF16 autocast in train_epoch by default)

## 2026-05-12 23:05 — PR #1638: LR=1e-3 with grad_clip (MERGED)

- **val_avg/mae_surf_p: 95.44** (↓ 9.5% from 105.46 — biggest gain this round)
- **test_avg/mae_surf_p: 87.83** (all 4 splits finite, tested from best-val checkpoint epoch 13)
- **Metric artifacts:** `models/model-charliepai2g24h5-tanjiro-lr1e3_gradclip-20260512-221259/metrics.jsonl`
- **What changed:** `lr: 5e-4 → 1e-3` in `train.py` Config dataclass (1-line change, commit `a1b596d`). All other config identical to #1483 baseline.
- **Why it worked:** Grad-clip at max_norm=1.0 fires every step (pre-clip norms 45–112 >> 1.0), effectively renormalising every gradient vector to unit norm. This bounded-step regime can safely absorb a 2× LR increase: each step is geometrically identical but with larger step size. The biggest gains are on OOD splits (val_geom_camber_rc −16.9, val_re_rand −12.6) consistent with improved cross-domain generalisation from the renorm regime.
- **Baseline configuration delta:** `lr: 5e-4 → 1e-3` (AdamW).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name lr1e3_gradclip --agent <student>
  ```
  (train.py now has `lr: float = 1e-3` as default)

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
| Optimizer | **Lion, lr=3e-4, weight_decay=6e-5** (changed from AdamW lr=1e-3) |
| LR schedule | LambdaLR, 3-epoch warmup + cosine to T_max=epochs |
| Grad clip | max_norm=1.0 |
| Batch size | 4 (variable mesh sizes, pad_collate to N_max) |
| Sampler | WeightedRandomSampler (balanced domain mix) |
| Max epochs | 13 (within 30-min wall-clock) |
| Timeout | 30 min wall-clock (hard cap) |
| Precision | BF16 autocast in forward pass |

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
