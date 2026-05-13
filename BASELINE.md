<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Best Baseline — `icml-appendix-willow-pai2g-24h-r3` (willow-pai2g-24h-r3)

Primary metric (lower is better): `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits).
Paper-facing metric: `test_avg/mae_surf_p` (4 test splits; the cruise-NaN-y bug was fully fixed in code by PR #1615 — `train.py::evaluate_split` now applies the per-sample `torch.isfinite(y).all(dim=-1)` filter before forward pass, matching the `data/scoring.py::accumulate_batch` per-sample skip semantics.).

## 2026-05-13 14:05 — PR #1986: Fourier positional features for (x,z) node coordinates — K=12 (MERGED)

- **`val_avg/mae_surf_p` (primary):** **73.1596** (W&B run `osxp8woj`)
- **`test_avg/mae_surf_p` (4-split, finite):** **63.8880**
- **Per-split val surface-p MAE (`osxp8woj`, K=12+warmup5, best-val epoch 19):**
  - val_single_in_dist: 80.28  (vs 88.79 baseline → −9.6%) ✅
  - val_geom_camber_rc: 81.88  (vs 89.23 → −8.2%) ✅
  - val_geom_camber_cruise: 57.96  (vs 54.47 → +6.4%) ❌ regression
  - val_re_rand: 72.52  (vs 71.33 → +1.7%) ~noise
- **Per-split test surface-p MAE (`osxp8woj`, 4-split clean):**
  - test_single_in_dist: 71.08  (vs 80.20 → −11.4%) ✅
  - test_geom_camber_rc: 70.73  (vs 79.83 → −11.4%) ✅
  - test_geom_camber_cruise: 47.87  (vs 45.41 → +5.4%) ❌ regression
  - test_re_rand: 65.87  (vs 64.70 → +1.8%) ~noise
- **Mechanism:** Fourier position encoding (sin/cos features for x,z) adds inductive bias for sharp pressure peak localization. K=12 resolves λ down to ~0.1 in normalized coord space — big gains on sharp-peak splits (single_in_dist, camber_rc), slight regression on smooth-field splits (cruise). Zero latency overhead; 3,072 extra params.
- **W&B run:** `osxp8woj`
- **W&B group:** `willow-r3-fourier-features`
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5 --fourier_k 12
  ```

## 2026-05-13 10:50 — PR #1438: 5-epoch linear LR warmup before cosine decay (MERGED)

- **`val_avg/mae_surf_p` (primary):** **75.9562** (W&B run `d1lqln08`; warmup-3ep sibling `fzxx54lu` at 76.72)
- **`test_avg/mae_surf_p` (4-split, finite):** **67.5326**
- **Per-split val surface-p MAE (`d1lqln08`, warmup-5ep, best-val epoch 19):**
  - val_single_in_dist: 88.79  (vs 90.76 baseline → −2.2%)
  - val_geom_camber_rc: 89.23  (vs 90.73 → −1.7%)
  - val_geom_camber_cruise: 54.47  (vs 54.88 → −0.7%)
  - val_re_rand: 71.33  (vs 73.12 → −2.4%)
- **Per-split test surface-p MAE (`d1lqln08`, 4-split clean):**
  - test_single_in_dist: 80.20  (vs 79.88 → +0.3% slight regress)
  - test_geom_camber_rc: 79.83  (vs 81.08 → −1.5%)
  - test_geom_camber_cruise: 45.41  (vs 45.88 → −1.0%)
  - test_re_rand: 64.70  (vs 65.99 → −2.0%)
- **W&B run:** `d1lqln08` (warmup-5ep); sibling `fzxx54lu` (warmup-3ep, val=76.72 / test=67.24)
- **W&B group:** `willow-r3-warmup-5ep`
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp --warmup_epochs 5
  ```

**Mechanism summary**: 5-epoch linear LR warmup ramps from 0 to peak LR over epochs 1-5, then hands off to the existing CosineAnnealingLR(T_max=50). Improvement is small but consistent: 4/4 val splits improve, 3/4 test splits improve, test_avg −1.00%. Both warmup arms beat baseline; warmup-5ep slightly better on val, warmup-3ep slightly better on test_avg (67.24 vs 67.53). The directional signal across 8 sub-metrics is clean despite in-band magnitude. The improvement is plausibly from stabilized early-training dynamics — warmup prevents overshooting the initial learning-rate peak and landing in a worse basin.

**Key open question**: warmup effect may be larger if combined with a longer schedule. The cosine LR was still at ~37% of peak at epoch 19 (same as AMP baseline) — warmup doesn't change epoch budget. The principal next gain axis is improving the per-epoch learning rate coverage, not reducing early-epoch instability.

## 2026-05-13 06:51 — PR #1440: bfloat16 AMP (torch.autocast) on top of EMA+SmoothL1+grad-clip (MERGED)

- **`val_avg/mae_surf_p` (primary):** **77.3716** (W&B run `30wvu5r0`; AMP-only supplementary arm `rn1gkw8h` at 86.03)
- **`test_avg/mae_surf_p` (4-split, finite):** **68.2053**
- **Per-split val surface-p MAE (`30wvu5r0`, AMP+EMA, best-val epoch 19):**
  - val_single_in_dist: 90.76  (vs 108.52 EMA-only → −16.4%)
  - val_geom_camber_rc: 90.73  (vs 104.81 → −13.4%)
  - val_geom_camber_cruise: 54.88  (vs 68.50 → −19.9%)
  - val_re_rand: 73.12  (vs 84.78 → −13.7%)
- **Per-split test surface-p MAE (`30wvu5r0`, 4-split clean):**
  - test_single_in_dist: 79.88
  - test_geom_camber_rc: 81.08
  - test_geom_camber_cruise: 45.88
  - test_re_rand: 65.99
- **W&B run:** `30wvu5r0` (AMP+EMA merge candidate); supplementary AMP-only arm `rn1gkw8h`
- **W&B group:** `willow-r3-amp-bf16` in `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3`
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp
  ```

**Mechanism summary**: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass and loss computation. BF16 uses the same exponent range as FP32 (no overflow risk), cuts per-epoch wall-clock ~25% (97.8s vs 131s), frees ~22% VRAM (32.9 GB vs 42 GB), and delivers +35% more gradient steps inside the 30-min cap (19 epochs vs ~14). AMP and EMA are orthogonal: AMP touches the per-step precision pipeline (forward + loss in bf16, master weights fp32 inside AdamW), EMA averages the parameter trajectory in fp32 buffers outside the autocast context. No numerical interaction. Val curve was strictly monotonic through all 19 completed epochs — the 30-min cap fires while training is still actively improving (LR at 37% of peak with T_max=50).

**EMA decomposition at same step (AMP+EMA arm, best-val epoch 19):**

| Branch | test_avg/mae_surf_p |
|--------|-------------------:|
| EMA weights (saved ckpt, primary) | **68.21** |
| Raw weights at same step (`test_no_ema/*`) | 71.39 |
| AMP-only arm best-val (`rn1gkw8h`) | 74.28 |

EMA on top of AMP adds ≈ −4.5% (variance-reduction-at-eval) + ≈ −3.7% (better-epoch-selection).

**Key open question**: val curve still strictly descending at epoch 19 with T_max=50. With AMP, the cosine LR at epoch 19 is still ~37% of peak — the optimizer is cut off mid-schedule. The next natural hypothesis is `--epochs 20` (matching T_max to the AMP epoch budget) so cosine LR completes a full annealing cycle within 30 min.

## 2026-05-13 04:52 — PR #1437: EMA of model weights (decay=0.999) for val/test/checkpoint (MERGED)

- **`val_avg/mae_surf_p` (primary):** **91.6553** (best of two EMA reproductions; sibling at 93.70; baselines at 101.06/104.03/105.18, mean ≈ 103.4)
- **`test_avg/mae_surf_p` (4-split, finite):** **81.2845** (best EMA, sibling 83.46; baselines 89.41 / 94.79)
- **Per-split val surface-p MAE (EMA `emqh79b0`, best-val epoch 14):**
  - val_single_in_dist: 108.52  (vs 129.82 baseline → −16.4%)
  - val_geom_camber_rc: 104.81  (vs 110.53 → −5.2%)
  - val_geom_camber_cruise: 68.50  (vs 80.24 → −14.6%)
  - val_re_rand: 84.78  (vs 95.52 → −11.2%)
- **Per-split test surface-p MAE (EMA `emqh79b0`, 4-split clean):**
  - test_single_in_dist: 98.53
  - test_geom_camber_rc: 89.74
  - test_geom_camber_cruise: 57.58
  - test_re_rand: 79.30
- **W&B run:** `emqh79b0` (best EMA arm); sibling reproduction `zzv8ke31` at val=93.70 / test=83.46
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999
  ```

**Mechanism summary**: EMA(0.999) is a Polyak average over the AdamW trajectory with effective window ≈ 1000 steps. Decomposition via dual eval at the same training step (EMA-weights vs raw-weights):

| Branch                              | test_avg/mae_surf_p |
|-------------------------------------|--------------------:|
| EMA weights (saved ckpt)            | **81.28**           |
| Raw weights at same step (no EMA eval) | 85.46            |
| Baseline (no EMA in training, `7xv82fez`) | 89.41         |

Two roughly equal effects compound: (a) variance-reduction-on-eval at the same step (85.46 → 81.28, ≈ −5%); (b) better epoch selection because the EMA val curve is monotonically descending so best-val lands on a later, better epoch (89.41 → 85.46, ≈ −4%). Per-step EMA update overhead is negligible (~1 ms; epoch wall-clock identical to baseline at 130–133 s). Decay=0.9995 (effective window ≈ 2000 steps) was tested in the original PR pass and is too slow at the 30-min budget.

**Orthogonality**: EMA stacks cleanly on top of SmoothL1(β=0.1) + grad_clip(1.0) without interaction — SmoothL1 stabilizes individual gradient steps, grad-clip caps per-batch gradient spikes, EMA averages the parameter trajectory itself. Distinct mechanisms on distinct objects, all compose. EMA also added a non-EMA dual-test-eval logged under `test_no_ema/*` — useful for future EMA-extension PRs.

## 2026-05-13 01:35 — PR #1615: Add pure L1 loss + cruise-NaN code fix (MERGED)

- **`val_avg/mae_surf_p` (primary):** **104.03** (pure-L1 variant, run `mc22t7l2`)
- **`test_avg/mae_surf_p` (4-split, finite):** **95.09**
- **Per-split val surface-p MAE (pure-L1, best-val checkpoint):**
  - val_single_in_dist: 129.82
  - val_geom_camber_rc: 110.53
  - val_geom_camber_cruise: 80.24
  - val_re_rand: 95.52
- **Per-split test surface-p MAE (pure-L1, post-fix 4-split):**
  - test_single_in_dist: 119.46
  - test_geom_camber_rc: 101.44
  - test_geom_camber_cruise: 68.55
  - test_re_rand: 90.93
- **W&B run:** `mc22t7l2` (pure-L1 variant)
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn l1
  ```

Same-run sibling SmoothL1(β=0.1) baseline arm (`x0ud9i0a`): val=102.17, test=92.04 (4-split) — confirming pure-L1 ≈ SmoothL1 within the ±7 single-seed noise band.

**Mechanism summary**: SmoothL1's win in #1441 was the linear-region gradient cap on outlier residuals — *not* the quadratic-near-zero smoothness. The pure-L1 variant drops the quadratic entirely and lands statistically indistinguishable from tuned-β SmoothL1 (within the ±7 single-seed noise band; three independent SmoothL1 reproductions span 102.17 → 103.57 → 125.94). Parameter-free L1 is the simpler, equivalent option. The per-split delta shows SmoothL1 only edges out pure-L1 on `val_geom_camber_cruise` / `test_geom_camber_cruise` (the low-|p| split where residuals are small enough to enter the quadratic region) — confirming the textbook Huber picture.

**Bug-fix component**: BASELINE.md previously claimed the cruise-NaN filter was in `train.py::evaluate_split` via PR #1433, but only the docs landed — the code change was missing on advisor branch. This PR adds the actual per-sample `torch.isfinite()` filter (lines 240-250 of train.py), so future PRs will natively report finite 4-split `test_avg/mae_surf_p`.

## 2026-05-12 21:05 — PR #1441: Replace MSE with SmoothL1 (Huber, beta=0.1) loss

- **`val_avg/mae_surf_p` (primary):** **104.6982** (epoch 13)
- **`test_avg/mae_surf_p` (3 splits, ex-cruise):** **101.0793**
- **`test_avg/mae_surf_Ux`** (4-split): 1.6194
- **`test_avg/mae_surf_Uy`** (4-split): 0.5998
- **Per-split surface-p MAE (val, best-val epoch):**
  - val_single_in_dist: 120.6339
  - val_geom_camber_rc: 117.4537
  - val_geom_camber_cruise: 82.3642
  - val_re_rand: 98.3412
- **W&B run:** `d53f0jn4` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3/runs/d53f0jn4)
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --smooth_l1_beta 0.1
  ```

Mechanism summary: SmoothL1 caps the per-element gradient magnitude at `beta=0.1` in normalized space, so high-Re/high-`|p|` outlier samples no longer dominate the quadratic MSE gradient. Training curve becomes monotonically descending instead of oscillating ±20–40 MAE points per epoch under MSE. Predicted delta was 2–5%; observed delta was −20.6% on the primary metric.

## 2026-05-12 21:06 — PR #1433: Add gradient norm clipping (max_norm=1.0)

Merged on top of #1441. PR #1433 was measured under MSE (PR's own baseline arm = 131.96, clip=1.0 variant = **114.18**, −13.5%) so it does not dethrone the 104.70 high-water mark from #1441, but it ships:

- `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` (orthogonal hygiene lever)
- **Inline cruise-test NaN fix** in `train.py::evaluate_split` — drops non-finite-`y` samples before the forward pass and `accumulate_batch`. Once a 4-split test pass succeeds end-to-end, future PRs will report the true `test_avg/mae_surf_p` rather than 3-split-ex-cruise.

- **W&B run:** `qof1cbki` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3/runs/qof1cbki) — recorded for reference; not the branch's empirical best.

Current advisor-branch code = SmoothL1(β=0.1) + grad_clip(1.0) + inline cruise-NaN fix. The combined-config run has not been measured yet (PR #1441 was SmoothL1-only, PR #1433 was grad-clip-only). Best-verified empirical metric remains **104.70**; expect future PRs that re-establish a baseline arm under the current advisor code to land near or below that point.

## 2026-05-13 00:02 — Supplemental: Current advisor-branch combined-config measurements (NOT a new merged PR)

The following measurements come from baseline arms of in-flight Round 2 PRs and characterize the current state of the advisor branch (SmoothL1 + grad_clip + cruise-NaN fix). These are not new code changes — they measure the code already merged via #1441 + #1433.

| Source | Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Note |
|---|---|---|---|---|
| #1616 (askeladd uniform-baseline, closed) | `eztvtkxc` | **90.91** | **86.87 (4-split, all finite!)** | Cleanest measurement; first paper-facing 4-split test pass |
| #1615 (tanjiro smooth-l1-v2 baseline, WIP) | `x0ud9i0a` | 102.17 | (TBD) | Independent run |
| #1437 (fern baseline-newbase, WIP) | `r7ysmbfi` | 104.84 | (TBD) | Independent run |

Range: 90.91 to 104.84 (15% spread; ±7 noise around mean ~99).

**Per-split val surface-p MAE (askeladd `eztvtkxc`, best-val epoch 14):**
- val_single_in_dist: 103.90
- val_geom_camber_rc: 105.34
- val_geom_camber_cruise: 68.99
- val_re_rand: 85.40

**Per-split test surface-p MAE (askeladd `eztvtkxc`, 4-split clean):**
- test_single_in_dist: 94.28
- test_geom_camber_rc: 95.81
- test_geom_camber_cruise: **76.61** (finite — cruise-NaN-y filter from #1433 worked)
- test_re_rand: 80.78

Reproduce (current advisor-branch HEAD):
```bash
cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0
```

Implication for Round 2 hypothesis ranking: any claim of <5% improvement is in the single-seed noise band. Practical merge bar = around the lower end of the variance band (~91); verified high-water-mark for *merged* code stays at 104.70 until a winning hypothesis PR with a terminal `SENPAI-RESULT` marker lands.
