# Baseline — `icml-appendix-charlie-pai2g-24h-r4`

Fresh start for round 4 of the Charlie / pai2g 24h logging ablation. No
prior experiments on this branch — the implicit baseline is the unmodified
`train.py` config inherited from `icml-appendix-charlie`. The first merged
winner sets the first numeric reference value.

## Reference config (default `train.py` at HEAD)

- **Model**: Transolver
  - `n_hidden = 128`
  - `n_layers = 5`
  - `n_head = 4`
  - `slice_num = 64`
  - `mlp_ratio = 2`
  - `space_dim = 2`, `fun_dim = X_DIM - 2 = 22`
  - `out_dim = 3` (`Ux`, `Uy`, `p`)
  - `unified_pos = False`
- **Optimizer**: AdamW (`lr = 5e-4`, `weight_decay = 1e-4`)
- **LR schedule**: CosineAnnealingLR with `T_max = 15` (aligned to actual training horizon under 30 min cap) _(updated 2026-05-12 by PR #1611)_
- **Loss**: **L1 (MAE) in normalized target space**, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight = 10.0` _(updated 2026-05-12 by PR #1397)_
- **Stochastic depth**: per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]` (linear schedule, last layer is the output head and never dropped) _(added 2026-05-12 by PR #1552)_
- **`evaluate_split` NaN-safe pre-filter**: skip samples with non-finite `y` before `accumulate_batch` to keep the 4-split test mean finite despite the `test_geom_camber_cruise/000020.pt` data bug _(added 2026-05-12 by PR #1552)_
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)` immediately before `optimizer.step()`; the pre-clip total_norm is also logged to metrics.jsonl as `train/last_grad_norm` _(added 2026-05-12 by PR #1637)_
- **Fourier coord positional encoding**: `FourierCoordEnc(n_freqs=6)` applied after `(x - x_mean)/x_std` normalization; replaces the 2 raw `(x, z)` coord dims with 24 Fourier features (`sin/cos` at frequencies `2^k · π`, `k=0..5`). `fun_dim = 4 * 6 + 22 - 2 = 44`. _(updated 2026-05-13 by PR #1772, was L=4 in #1548)_
- **Batch size**: `4`
- **Epochs**: configured `50`, capped by `SENPAI_TIMEOUT_MINUTES = 30`
- **Sampler**: `WeightedRandomSampler` with equal-domain weights from `meta.json`

## Metrics contract

- Primary ranking metric: `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits.
- Paper-facing comparison metric: `test_avg/mae_surf_p` — same aggregation on the four test splits at the best val checkpoint.
- All metrics computed in physical (denormalized) units in `data/scoring.py`.

## Current best result

### 2026-05-13 02:50 — PR #1772 (`charliepai2g24h4-edward/fourier-coords-L6`)

Fourier positional encoding bumped from L=4 → L=6 (24 Fourier features
replacing the 16 L=4 features). Single-knob bracket-up of the merged
#1548 Fourier mechanism — the L=4 → L=6 trajectory is still on the
upward slope of Tancik's curve, with the predicted plateau at L=8-10.
Every val split and every test split improves; magnitude is at the
upper end of the pre-registered prediction band (-0.5% to -2.5%) on val
and middle of the band on test.

- **`val_avg/mae_surf_p`** = **82.311** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **73.330**
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 93.299 (-3.89% vs L=4)
  - `val_geom_camber_rc` = 92.965 (-2.14% vs L=4)
  - `val_geom_camber_cruise` = 63.131 (-0.91% vs L=4)
  - `val_re_rand` = 79.848 (-4.10% vs L=4)
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 83.323 (-2.91% vs L=4)
  - `test_geom_camber_rc` = 81.867 (-1.39% vs L=4)
  - `test_geom_camber_cruise` = 54.094 (-1.43% vs L=4)
  - `test_re_rand` = 74.038 (-1.17% vs L=4)
- **Δ vs PR #1548 baseline (84.762 / 74.659)**: **-2.89%** on val_avg, **-1.78%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → #1772 → val_avg has improved from 100.957 to 82.311 = **-18.5% over 6 merges**.
- **Param count**: 667,991 (+2,048 over #1548; +0.31%; from wider preprocess MLP first-layer input).
- **Surprise finding**: `val_re_rand` improved -4.10% (pre-registered as "likely stays flat" since its OOD axis is Reynolds, not spatial frequency). Plausible mechanism: at L=4 the network was over-spending capacity on low-freq geometry encoding; with L=6 it can encode geometry in higher Fourier bands and free up MLP capacity for Reynolds-dependent features. The consistent -1.2% to -1.4% gain on test_re_rand corroborates this is not pure noise.
- **Metric artifacts**: `models/model-charliepai2g24h4-edward-fourier-coords-L6-20260513-011437/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-edward --experiment_name charliepai2g24h4-edward/fourier-coords-L6`

### 2026-05-13 01:15 — PR #1548 (`charliepai2g24h4-edward/fourier-coords-L4-rebased`)

Fourier positional encoding (`L=4`, 16 Fourier features replacing the 2 raw
`(x, z)` coord dims, applied after normalization). Stacks cleanly with the
merged compound (stoch-depth + cosine T_max=15 + grad-clip 25) — every val
split improves, and test improves more than val. The split pattern matches
the spectral-bias hypothesis: largest gains where high-frequency spatial
structure dominates (`val_single_in_dist` -11.35%, `val_geom_camber_cruise`
-7.94%), minimal movement on `val_re_rand` (-0.30%) whose OOD axis is
Reynolds (flow-condition) not spatial coords.

- **`val_avg/mae_surf_p`** = **84.762** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **74.659**
- **Per-split val** `mae_surf_p` at the best val checkpoint:
  - `val_single_in_dist` = 97.074
  - `val_geom_camber_rc` = 94.997
  - `val_geom_camber_cruise` = 63.711
  - `val_re_rand` = 83.266
- **Per-split test** `mae_surf_p` at the best val checkpoint:
  - `test_single_in_dist` = 85.819
  - `test_geom_camber_rc` = 83.023
  - `test_geom_camber_cruise` = 54.879
  - `test_re_rand` = 74.916
- **Δ vs PR #1637 baseline (90.294 / 81.243)**: **-6.13%** on val_avg, **-8.10%** on 4-split test.
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → val_avg has improved from 100.957 to 84.762 = **-16.0% over 5 merges**.
- **Param count**: 665,943 (+5.4K, +0.82% over previous baseline; from wider preprocess MLP input).
- **Metric artifacts**: `models/model-charliepai2g24h4-edward-fourier-coords-L4-rebased-20260512-235326/metrics.jsonl` and `metrics.yaml`
- **Reproduce**: `cd target/ && python train.py --agent charliepai2g24h4-edward --experiment_name charliepai2g24h4-edward/fourier-coords-L4-rebased`

### 2026-05-12 22:55 — PR #1637 (`charliepai2g24h4-askeladd/grad-clip-25`)

Permissive gradient clipping at `max_norm=25.0` immediately before
`optimizer.step()` — a single-line addition. Diagnostic-informed
follow-up to closed PR #1529 (`max_norm=1.0`, +5.4% regression): with the
threshold raised from 1.0 to 25.0, clipping fires on the outlier-spike
steps (the largest grad norm observed in training is 110.04 at epoch 8)
without touching typical 30-70-range gradients. The mechanism is
compatible with stoch-depth (block-drop spikes are suppressed) and
cosine T_max=15 (the late-epoch cooldown phase relies on stable
gradients to fine-tune).

- **`val_avg/mae_surf_p`** = **90.294** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **81.243**

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) |
|-------|-----------:|
| single_in_dist     | 109.497 |
| geom_camber_rc     |  98.952 |
| geom_camber_cruise |  69.208 |
| re_rand            |  83.520 |
| **avg**            | **90.294** |

vs PR #1611 baseline:
- val_avg: 94.217 → 90.294 (**-4.16% improvement**)
- All four val splits improved uniformly (-3.14% to -5.61%) — no
  split-specific direction, exactly as the hypothesis predicted ("stable
  descent helps everywhere").
- test_avg: 84.859 → 81.243 (-4.26% improvement)

Diagnostic from the per-epoch `train/last_grad_norm` trace: 14/15 epochs
had end-of-epoch grad_norm > 25 (the clip threshold), confirming
clipping is active throughout training. The largest spike (110.04 at
epoch 8) was suppressed; typical training-step norms stayed in the
30-70 range. Val MAE descended monotonically epoch 9 → 15, with the
biggest single-epoch drop (-13.7%) coinciding with the only epoch where
the end-of-epoch norm fell below 25 (22.40 at epoch 12).

- **Metric artifacts**:
  `models/model-charliepai2g24h4-askeladd-grad-clip-25-20260512-221014/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.11 GB, **wall time**: 30 min (cap).

### 2026-05-12 21:16 — PR #1611 (`charliepai2g24h4-askeladd/cosine-tmax-15`)

CosineAnnealingLR schedule horizon aligned to the actual training duration:
`T_max=15` (matching the empirical epoch count at the 30-min cap), replacing
the previous `T_max=MAX_EPOCHS=50`. Under the old schedule, LR was at ~80% of
peak (≈4.0e-4) when training terminated — the full cosine cooldown phase
never ran. With `T_max=15`, LR completes its full cosine arc to ~0 over the
actual training horizon (verified by jsonl LR trace: 4.945e-4 → 5.463e-6 → 0).
Zero added compute, zero added parameters, single-line change.

- **`val_avg/mae_surf_p`** = **94.217** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **84.859**

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) | mae_surf_p (test) |
|-------|-----------:|-----------:|
| single_in_dist     | 114.200 |  ? |
| geom_camber_rc     | 102.157 |  ? |
| geom_camber_cruise |  73.321 |  ? |
| re_rand            |  87.188 |  ? |
| **avg**            | **94.217** |  **84.859** |

vs PR #1552 baseline:
- val_avg: 98.353 → 94.217 (**-4.21% improvement** — largest single-arm gain of wave 2)
- All four val splits neutral-to-positive (camber_rc -8.04% largest gain).
- test_avg: 87.995 → 84.859 (-3.57% improvement)

Val MAE descended monotonically every epoch — the model was still improving
at the wall-clock cap, suggesting further headroom with more time. The new
LR cooldown phase let the model find a better minimum within the same
30-min budget.

- **Metric artifacts**:
  `models/model-charliepai2g24h4-askeladd-cosine-tmax-15-20260512-211600/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.1 GB, **wall time**: 30 min (cap).

### 2026-05-12 20:52 — PR #1552 (`charliepai2g24h4-frieren/stoch-depth-0.1`)

Stochastic depth (Huang et al., ECCV 2016) added to the 5-layer Transolver
with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
At eval/test time it is a no-op (all blocks always used). Also bundles the
NaN-safe pre-filter in `evaluate_split` that produces the first finite
4-split `test_avg/mae_surf_p` on this branch.

- **`val_avg/mae_surf_p`** = **98.353** (best @ epoch 15, last epoch before 30 min timeout)
- **`test_avg/mae_surf_p` (4-split, NaN-safe)** = **87.995** — first finite 4-split test
  reference on this branch; new paper-facing baseline.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p (val) | mae_surf_p (test) |
|-------|-----------:|-----------:|
| single_in_dist     | 119.159 | 104.953 |
| geom_camber_rc     | 111.093 | 101.883 |
| geom_camber_cruise |  73.323 |  62.243 |
| re_rand            |  89.837 |  82.901 |
| **avg**            | **98.353** |  **87.995** |

vs. L1 baseline (PR #1397):
- val_avg: 100.957 → 98.353 (**-2.58% improvement**)
- val_single_in_dist: -6.45% / val_geom_camber_cruise: -5.21% (largest gains)
- val_geom_camber_rc: +0.24% (flat) / val_re_rand: +1.77% (small regression)

The OOD-specific framing was only half-supported — the biggest gain landed
on `val_single_in_dist` (in-distribution), not the camber OOD splits as
predicted. Stoch-depth's implicit ensemble flattened split-specific overfit
modes regardless of OOD axis. Best epoch landed at the wall-clock cap
(epoch 15), so more training time would likely extend the gain.

Caveat: `loss`/`surf_loss` aggregates for `test_geom_camber_cruise` still
show NaN/Inf in `metrics.yaml` because the normalized-space loss path
runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary
ranking metric is clean.

- **Metric artifacts**:
  `models/model-charliepai2g24h4-frieren-stoch-depth-0.1-20260512-201730/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M (unchanged), **peak GPU memory**: 42.1 GB, **wall time**: 30 min (cap).

### 2026-05-12 19:05 — PR #1397 (`charliepai2g24h4-alphonse/l1-loss`)

L1 (MAE) loss replaces MSE in normalized-space training. First numeric
baseline on this branch.

- **`val_avg/mae_surf_p`** = **100.9574** (best @ epoch 13/14 before 30 min timeout)
- **`test_avg/mae_surf_p` (3-split mean, excludes `test_geom_camber_cruise`)** = **100.8314**
- **`test_avg/mae_surf_p` (all 4 splits, raw)** = NaN — pre-existing data
  bug: `test_geom_camber_cruise/000020.pt` has 761 nodes with `inf` in
  pressure y. Affects every arm in this round; `data/scoring.py` is
  marked read-only. See PR #1397 comment for full trace and proposed
  fixes. Until resolved we record the 3-split test mean.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 127.371 |
| val_geom_camber_rc     | 110.832 |
| val_geom_camber_cruise |  77.353 |
| val_re_rand            |  88.273 |
| **val_avg/mae_surf_p** | **100.957** |
| test_single_in_dist    | 116.622 |
| test_geom_camber_rc    |  97.209 |
| test_geom_camber_cruise| NaN (data bug, surf_Ux/Uy still ok) |
| test_re_rand           |  88.663 |
| **test_avg/mae_surf_p (3-split)** | **100.831** |

- **Metric artifacts**:
  `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M, **peak GPU memory**: 42.1 GB, **wall time**: 30.7 min.

## Reproduce baseline

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <student>/baseline
```
