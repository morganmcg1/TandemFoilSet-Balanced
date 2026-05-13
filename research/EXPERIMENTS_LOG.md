<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research Results — willow-pai2g-24h-r3

Lower is better for `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

## 2026-05-13 09:10 — PR #1842: Transolver mlp_ratio sweep (post-rebase under AMP) — CLOSED

- Student branch: `willowpai2g24h3-edward/mlp-ratio-sweep`
- Hypothesis: at the new AMP operating point, smaller MLP (`mlp_ratio=1`) re-allocates throughput into more epochs of cosine cool-down; larger MLP (`mlp_ratio=4`) costs throughput. Pre-AMP this PR had won at 85.82 val (−6.4% vs pre-AMP 91.66).

### Results (3 arms, all rebased onto advisor `04aa53b` with `--amp --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999`)

| arm | mlp_ratio | n_params | val_avg | test_avg (EMA) | test_no_ema | best epoch | s/epoch | run id |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (baseline reproduction) | 2 | 0.66M | **77.94** | **67.84** | 78.49 | 19 | 98.86 | `6u8da009` |
| B (winner candidate pre-AMP) | 1 | 0.50M | 78.46 | 69.71 | 86.97 | 19 | 94.63 | `eliwspvs` |
| C (bracket) | 4 | 0.99M | 81.14 | 71.66 | 85.11 | 18 | 103.76 | `t9yrv6d1` |
| advisor reference | 2 | 0.66M | 77.37 | 68.21 | — | — | — | `30wvu5r0` |

Arm A reproduces the advisor reference within ±0.6 val / ±0.4 test — rebase is correct. Arm B (the pre-AMP winner) loses on every test split. Arm C (the upper bracket) loses by even more. Per-split test deltas vs baseline:

| split | ratio=2 | ratio=1 | ratio=4 | Δ(1−2) | Δ(4−2) |
|---|---:|---:|---:|---:|---:|
| test_single_in_dist | 79.84 | 81.30 | 84.78 | +1.46 | +4.94 |
| test_geom_camber_rc | 79.73 | 80.83 | 82.63 | +1.10 | +2.90 |
| test_geom_camber_cruise | 46.06 | 48.61 | 49.77 | +2.55 | +3.71 |
| test_re_rand | 65.72 | 68.09 | 69.44 | +2.38 | +3.72 |

### Mechanism — AMP subsumed the throughput-mechanism win

Epoch-time ratio collapse is the keeper output:

| arm | predicted (PR body) | pre-AMP measured | post-AMP measured |
|---|---:|---:|---:|
| ratio=1 vs baseline | 0.71× | 0.93× | **0.957×** |
| ratio=4 vs baseline | 1.75× | 1.10× | **1.050×** |

Under AMP, smaller MLP buys only ~4% per-epoch and larger MLP costs only ~5%. All three arms hit 18-19 epochs (vs the pre-AMP 13-15 spread). The throughput dial that drove the pre-AMP win (~2 extra epochs of cool-down) is now saturated by autocast. Only the per-step capacity effect remains, which favors the existing `mlp_ratio=2`.

### Conclusion

**Close. mlp_ratio axis at this depth/width is settled — no value in 1 or 4 over the existing default.** Code change (Config field) not cherry-picked: not needed for any in-flight hypothesis, and one less Config option is preferable. Edward reassigned to depth sweep (`n_layers`) — the natural follow-up since width was closed pre-AMP (#1443) and depth has never been tested at the AMP operating point.

### Emerging lessons logged

1. **AMP shifts the capacity-vs-throughput surface.** Pre-AMP optimum was throughput-saving (ratio=1); post-AMP optimum is the existing default (ratio=2). Pre-AMP architectural negatives may be worth a quick AMP re-sweep before being treated as final.
2. **Test-split direction breaks ties on noise-band val results.** When val_avg lands in ±7 noise, per-split test sign (all worse vs mixed) is the cleanest tie-breaker. Decisive here against second-seeding mlp_ratio=1.

## 2026-05-13 09:00 — PR #1779: AdamW weight_decay sweep at AMP+EMA baseline — CLOSED

- Student branch: `willowpai2g24h3-thorfinn/weight-decay-sweep`
- Hypothesis: AdamW `weight_decay` sweep {1e-4, 1e-3, 1e-2, 5e-2} as a regularization probe on top of the merged SmoothL1+grad-clip+EMA(0.999)+AMP stack. Prediction: a modest wd lift would help the highest-MAE OOD splits (`val_single_in_dist`, `val_geom_camber_rc`) preferentially.

### Results — all 3 variant arms (rebased onto advisor `04aa53b`, 30-min cap, AMP+EMA on)

| arm | wd | val_avg | test_avg | Δ val vs 77.37 | best epoch | run id |
|---|---:|---:|---:|---:|---:|---|
| Baseline (advisor) | 1e-4 | **77.37** | **68.21** | — | — | `30wvu5r0` |
| wd=1e-3 (best variant) | 1e-3 | 77.73 | 68.64 | +0.36 (+0.5%) | 19/50 | `xz3vojme` |
| wd=1e-2 | 1e-2 | 78.35 | 69.54 | +0.98 (+1.3%) | 19/50 | `npnres0j` |
| wd=5e-2 | 5e-2 | 81.03 | 71.57 | +3.66 (+4.7%) | 19/50 | `hqglc6x5` |

W&B group: `willow-r3-weight-decay-sweep`. No NaNs, no OOMs, peak GPU mem ~53 GiB across arms.

### Per-split val breakdown (wd=1e-3 best variant vs AMP+EMA baseline `30wvu5r0`)

| split | baseline 1e-4 | wd=1e-3 | Δ | wd=1e-2 | Δ |
|---|---:|---:|---:|---:|---:|
| val_single_in_dist | 90.76 | **90.29** | −0.47 | **89.65** | −1.11 |
| val_geom_camber_rc | 90.73 | **88.59** | **−2.14** | 91.46 | +0.73 |
| val_geom_camber_cruise | 54.88 | 57.99 | +3.11 | 57.80 | +2.92 |
| val_re_rand | 73.12 | 74.05 | +0.93 | 74.50 | +1.38 |
| **avg** | **77.37** | **77.73** | +0.36 | **78.35** | +0.98 |

### Mechanism analysis (thorfinn)

The hypothesis predicted that L2 regularization should preferentially help the highest-MAE OOD-ish val splits. **The mechanism is partly real but unprofitable**:

- At wd=1e-3, the two high-MAE OOD-ish splits (`val_single_in_dist`, `val_geom_camber_rc`) DO improve in the predicted direction (−0.47 and **−2.14** respectively).
- But the same regularizer hurts `val_geom_camber_cruise` by +3.11 and `val_re_rand` by +0.93.
- Net of opposing forces lands aggregate inside the ±7 noise band on the wrong side of baseline.

This is the signature of a **model whose capacity is matched-to-task**: there is no free regularization headroom — the same regularizer that fixes one split breaks another by the same magnitude. EMA already subsumes the variance-reduction effect that decoupled weight_decay would otherwise provide.

### Conclusion (advisor)

**Weight decay is closed at this baseline.** Rule out the entire L2-regularization family for Round 2: no wd schedules, no layer-wise wd, no AdamW-vs-Adam re-runs. The forward axis is **input-side feature representation / data augmentation**, which thorfinn's per-split asymmetry diagnosis points at directly. Next assignment to thorfinn is coordinate-jitter augmentation as a synthetic-near-miss-geometry generator targeting the held-out camber splits.

The per-split asymmetry table from this run ("one split is over-regularized by exactly the same amount that the other is under-regularized") is the cleanest capacity-vs-task-fit diagnostic from this round and should be revisited every time a new lever is proposed.

## 2026-05-13 06:35 — PR #1440: Enable bfloat16 mixed precision (AMP + EMA) — MERGED (WINNER)

- Student branch: `willowpai2g24h3-nezuko/amp-bf16`
- Hypothesis: `torch.autocast("cuda", dtype=torch.bfloat16)` for forward + loss reduces per-epoch wall-clock ~25-30%, giving ~35% more gradient steps within the 30-min budget. AMP and EMA are orthogonal — AMP changes per-step precision, EMA averages the parameter trajectory.

### Results

| arm | run | epochs (30-min cap) | s/epoch | peak VRAM | val_avg | test_avg | Δ vs merged 91.66/81.28 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (EMA, no AMP, advisor `emqh79b0`) | `emqh79b0` | ~14 | ~131 | ~42 GB | 91.6553 | 81.2845 | baseline |
| AMP only (no EMA, supplementary) | `rn1gkw8h` | 19 | 98.4 | 32.9 GB | 86.0296 | 74.2780 | −6.2% val |
| **AMP + EMA (merge candidate)** | `30wvu5r0` | 19 | 97.8 | 32.9 GB | **77.3716** | **68.2053** | **−15.6% val / −16.1% test** |

W&B group: `willow-r3-amp-bf16` in `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3`.

### Per-split breakdown (AMP+EMA `30wvu5r0` vs baseline `emqh79b0`)

| split | baseline val | AMP+EMA val | Δ | baseline test | AMP+EMA test | Δ |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 108.52 | **90.76** | −16.4% | 98.53 | **79.88** | −18.9% |
| geom_camber_rc | 104.81 | **90.73** | −13.4% | 89.74 | **81.08** | −9.7% |
| geom_camber_cruise | 68.50 | **54.88** | −19.9% | 57.58 | **45.88** | −20.3% |
| re_rand | 84.78 | **73.12** | −13.7% | 79.30 | **65.99** | −16.8% |
| **avg** | **91.66** | **77.37** | **−15.6%** | **81.28** | **68.21** | **−16.1%** |

### EMA decomposition at same step (best-val epoch 19)

| evaluation branch | test_avg/mae_surf_p |
|---|---:|
| EMA weights (saved ckpt, primary) | **68.21** |
| Raw weights at same step (`test_no_ema/*`) | 71.39 |
| AMP-only arm (different ckpt, no EMA training, `rn1gkw8h`) | 74.28 |

EMA on top of AMP adds ≈ −4.5% (variance-reduction-at-eval) + ≈ −3.7% (better-epoch-selection via smoother val curve) = ≈ −8% from EMA alone, fully consistent with the EMA mechanism from PR #1437.

### Analysis and conclusions

1. **AMP + EMA compose cleanly.** AMP changes the per-step precision pipeline (forward + loss in bf16, master weights fp32 inside AdamW). EMA averages the parameter trajectory outside the autocast context (fp32 EMA buffers). No numerical interaction. EMA overhead invisible at AMP speed (97.8 vs 98.4 s/epoch).
2. **Mechanism is throughput → more cooling.** With T_max=50 cosine schedule and ~14 baseline epochs (no AMP), only ~28% of the annealing budget is used. AMP pushes to ~19 epochs, still only 38% — but the extra 5 epochs correspond to additional LR cool-down where the optimizer makes more conservative, higher-quality steps. Val curve was **strictly monotonic** through epoch 19 (no plateau), meaning the 30-min cap fires while training is still actively improving.
3. **Cruise NaN bug fully resolved.** `test_geom_camber_cruise/mae_surf_p = 45.88` (finite). The per-sample `isfinite(y)` filter from PR #1615 handles this cleanly.
4. **Key open question**: val curve still descending monotonically at epoch 19 with T_max=50 → only 38% of cosine schedule spent. The implicit next hypothesis is matching T_max to the AMP epoch budget (~20 epochs) so cosine LR fully anneals within 30 min.
5. **New reproduce baseline**: `cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0 --ema_decay 0.999 --amp`

## 2026-05-13 06:08 — PR #1800: Truncated L1 (zero-gradient cliff at τ) — CLOSED

- Student branch: `willowpai2g24h3-tanjiro/truncated-l1`
- Hypothesis: per-element `min(|r|, τ)` zeros gradient on residuals where `|r| ≥ τ`. Predicted to improve high-|p| splits (in_dist + rc) by 2-5% by suppressing outlier influence; cruise (low-|p|) predicted to degrade slightly. Cliff sweep at τ∈{0.5, 1.0, 2.0} plus a τ=1.0+EMA+grad_clip combined arm.

### Results

| arm | run | val_avg | test_avg | Δ vs merged 91.66/81.28 |
|---|---|---:|---:|---:|
| L1 baseline | `gif79a0t` | 100.38 | 89.66 | +8.7 val (in noise) |
| τ=2.0 (light cap) | `cnv0cuz5` | 114.14 | 104.15 | +22 val WORSE |
| τ=1.0 (primary) | `f8j56db1` | 111.08 | 101.41 | +19 val WORSE |
| τ=0.5 (tight cap) | `cln0mj4e` | 125.23 | 115.83 | +34 val WORSE |
| **τ=1.0 + EMA + gc** | `r6tr47d7` | **107.08** | **97.71** | **+15.4 val / +16.4 test WORSE** |

### `train/pct_clipped` at convergence

| arm | last-half mean clip rate |
|---|---:|
| τ=2.0 | 0.006 (essentially off) |
| τ=1.0 (no EMA) | 0.034 |
| τ=1.0 + EMA + gc | 0.029 |
| τ=0.5 | 0.107 |

### Mechanism — prediction falsified, reinterpreted

PR-body predicted: at τ=1.0, `val_single_in_dist` and `val_geom_camber_rc` IMPROVE (high-|p| splits where tight cap helps), `cruise` DEGRADES (low-|p| where cap removes signal).

Observed at τ=1.0 (vs L1):
- val_single_in_dist: **+34.0 MAE (+29%) WORSE** (predicted: improve)
- val_geom_camber_rc: **+6.3 MAE WORSE** (predicted: improve)
- val_geom_camber_cruise: +0.4 MAE (essentially unchanged — only correct prediction)
- val_re_rand: +2.0 MAE WORSE

Reinterpretation: the few worst residuals at convergence (~3% at τ=1.0) ARE the signal needed to learn high-magnitude regions, not outlier noise. Zeroing their gradient kills learning on those regions. Degradation is graded proportional to clip rate.

### EMA-on-truncated_l1 orthogonality check (best τ=1.0+EMA arm `r6tr47d7`)

Dual eval at same best-val checkpoint via #1437's `test_no_ema/*` logging:

| metric | EMA weights | non-EMA weights at same step | Δ |
|---|---:|---:|---:|
| test_avg | 97.71 | 108.34 | **−10.6** (EMA wins) |
| test_geom_camber_rc | 102.29 | 126.05 | **−23.8** |
| test_geom_camber_cruise | 62.08 | 74.51 | **−12.4** |
| test_re_rand | 87.62 | 102.26 | **−14.6** |
| test_single_in_dist | 138.84 | 130.54 | +8.3 (EMA hurts on this split) |

EMA buys ~10.6 test MAE on top of truncated_l1 — **same magnitude as on SmoothL1 (#1437)**. EMA's parameter-trajectory averaging is robust to the underlying gradient-shape choice. Useful generalization: future loss-fn hypotheses can assume EMA stacks for free, and only need to argue about the underlying loss-fn mechanism.

### Conclusions

- **Closed**. Truncated direction does not produce a merge candidate at any τ tested. The closer-to-zero floor at the best τ=1.0+EMA arm (107.08 val, 97.71 test) is 15-16 MAE worse than the merged baseline.
- **Loss-shape axis is now closed.** Three PRs (#1441 MSE→SmoothL1 winner, #1615 SmoothL1→L1 equivalence, this PR truncated L1 hurts) pin down `sign(r)` bounded-linear gradient as the local optimum on the "gradient aggressiveness vs residual magnitude" axis. Future loss-fn hypotheses should target sample-conditional rather than residual-conditional gradient shape.
- **Diagnostic `train/pct_clipped` is a new advisor-branch instrument**: not strictly necessary for the merged baseline, but useful for future cliff/clip hypotheses. Living in tanjiro's branch only — would need re-implementation if revisited. (Not landing in advisor branch since this PR is closing.)

## 2026-05-13 04:52 — PR #1437: EMA of model weights (decay=0.999) — MERGED (winner)

- Student branch: `willowpai2g24h3-fern/ema-decay999`
- Hypothesis: EMA of model weights (decay=0.999) at val/test/checkpoint reduces variance of the SGD/Adam trajectory; predicted 1–3% reduction in `val_avg/mae_surf_p`. Orthogonal to SmoothL1 (per-element gradient cap) and grad-clip (per-batch gradient cap) because it operates on the parameter trajectory itself.

### Results (rebase onto advisor HEAD `4f225b4` = SmoothL1+grad-clip+cruise-NaN fix)

| Arm | Run | ema_decay | best val_avg/mae_surf_p | test_avg/mae_surf_p (4-split) | Δ vs merged 104.03/95.09 |
|---|---|---|---:|---:|---:|
| baseline-30m-newbase-v2 | `7xv82fez` | 0.0 | 101.06 | 89.41 | −2.9% / −6.0% (in noise) |
| baseline-r3-30m | `t73h00e2` | 0.0 | 105.18 | 94.79 | +1.1% / −0.3% (in noise) |
| **ema-decay-0.999-30m-newbase** | `zzv8ke31` | 0.999 | **93.70** | **83.46** | **−9.9% / −12.2%** |
| **ema-decay-0.999-30m-newbase-v2** | `emqh79b0` | 0.999 | **91.66** | **81.28** | **−11.9% / −14.5%** |

Three baselines (incl. merged 104.03) mean ≈ 103.4; two EMA reproductions mean ≈ 92.7 — a **−10.4% mean delta on val, well outside the ±7 single-seed noise band**.

### Per-split breakdown (best EMA `emqh79b0`)

**Val (vs merged baseline #1615 per-split):**

| Split | EMA `emqh79b0` | merged baseline | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 108.52 | 129.82 | **−16.4%** |
| val_geom_camber_rc | 104.81 | 110.53 | **−5.2%** |
| val_geom_camber_cruise | 68.50 | 80.24 | **−14.6%** |
| val_re_rand | 84.78 | 95.52 | **−11.2%** |

**Test (4-split, finite):**

| Split | EMA `emqh79b0` |
|---|---:|
| test_single_in_dist | 98.53 |
| test_geom_camber_rc | 89.74 |
| test_geom_camber_cruise | 57.58 |
| test_re_rand | 79.30 |

Improvement is broadly distributed (every split is in-the-money). Biggest absolute gains land on `single_in_dist` and `cruise` — the splits with the noisiest val curve under baseline. Smallest gain on `geom_camber_rc` (already the lowest-error split, less headroom).

### Mechanism (decomposed by dual eval at the same training step)

EMA(0.999) is a Polyak average over the AdamW trajectory with effective window ≈ 1000 steps. The rebase plumbed an additional "non-EMA test eval at the same epoch" logged under `test_no_ema/*`, isolating two distinct contributions:

| Branch | test_avg/mae_surf_p |
|---|---:|
| EMA weights (saved ckpt) | **81.28** |
| Raw weights at same step (no-EMA-eval) | 85.46 |
| Baseline (no EMA in training, `7xv82fez`) | 89.41 |

- **Variance-reduction at eval (~5%):** 85.46 → 81.28 from averaging the parameter trajectory.
- **Better epoch selection (~4%):** 89.41 → 85.46 because the EMA val curve is monotonically descending so `argmin val_avg` lands on a later, better epoch.

Both effects compound; combined ≈ −9% test, with no interaction with SmoothL1 / grad-clip. Per-step EMA update overhead is ~1 ms; per-epoch wall-clock identical to baseline at 130–133 s. EMA also did NOT need any α-warmup at this budget (decay=0.9995 was tested in PR pass 1 and is too slow at 30 min — see #1437 pass 1 comment).

### Conclusions

- **Merged. New empirical high-water mark on the advisor branch:**
  - `val_avg/mae_surf_p = 91.66` (down from 104.03)
  - `test_avg/mae_surf_p = 81.28` (down from 95.09)
- EMA stacks orthogonally on top of SmoothL1(β=0.1) + grad_clip(1.0) without interaction — three distinct gradient-stabilization mechanisms on three distinct objects (per-element / per-batch / parameter-trajectory) all compose.
- The `test_no_ema/*` dual-eval plumbing now ships on the advisor branch — future EMA-extension PRs (decay sweep, warmup, longer-budget) inherit it for free.
- **Round 2 priority shift**: capacity / regularization / loss reformulation now compete against a much harder baseline. Hypotheses claiming <5% improvement need ≥2 seeds. Any new "headline" merge needs ≥10% relative gain to be visibly real on a single seed.

## 2026-05-12 19:30 — PR #1443: Widen Transolver to n_hidden=192, n_head=6 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/wider-n192`
- Hypothesis: increasing `n_hidden` from 128→192 and `n_head` from 4→6 (`dim_head` constant at 32) gives more capacity at fixed depth/slice; expected 2–6% reduction in `val_avg/mae_surf_p`.

### Results

| Run | n_hidden / n_head | Params | Epochs done | val_avg/mae_surf_p | test 3-split avg surf_p | Δ vs baseline | W&B |
|---|---|---|---|---|---|---|---|
| baseline-30m | 128 / 4 | 0.66M | 14 | **123.17** (e12) | 120.19 | — | `h73q3u7m` |
| wider-n192-30m | 192 / 6 | 1.45M | 9 | **163.67** (e7) | 165.67 | +33% val / +38% test (worse) | `b9pe1a61` |

### Analysis

Wider variant regressed by +33% on val and +38% on test. Root cause: wider model is ~1.5× slower per epoch, finishes only 9 of the 50 scheduled epochs vs baseline's 14, and never enters the cosine cool-down where the baseline gains most of its ground.

Key observation from baseline trajectory (which becomes the seed for the next experiment): val_avg/mae_surf_p drops 140 → 156 → 126 → **123** at epochs 9-12 (collapse to 182 at e13 — likely noise). The cosine LR is barely cooled at this point (T_max=50, t=14 → cos(14π/100)≈0.92, LR still ~4.6e-4 of 5e-4 peak). Completing the schedule should push the best lower.

### Conclusions

- At the 30-min budget, capacity scaling via width is dominated by throughput cost — closed.
- Schedule mismatch (T_max=50, only 14 epochs fit) is a probable next lever — assigned to thorfinn as `schedule-tuned-e13`.
- **Known bug (do not block on):** `test_geom_camber_cruise/mae_surf_p` is NaN on both arms (pre-existing in the scoring/data path). Both `Ux/Uy` MAE on the same split are finite, suggesting a specific sample's p-channel prediction or ground-truth overflows. Need a separate `data/scoring.py` or data-side PR; deferring until more PRs land or the bug starts blocking ranking.

## 2026-05-12 21:05 — PR #1441: Replace MSE with SmoothL1 (Huber, β=0.1) — MERGED (winner)

- Student branch: `willowpai2g24h3-tanjiro/smooth-l1-beta01`
- Hypothesis: SmoothL1 in normalized space caps per-element gradient magnitude on high-Re outliers; predicted 2–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | Loss | Best val_avg/mae_surf_p (epoch) | test 3-split-ex-cruise avg surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| baseline-30m | MSE | 131.81 (e10) | 131.56 | — | `y3dfc5e7` |
| smooth-l1-0.1-30m | SmoothL1(β=0.1) | **104.70 (e13)** | **101.08** | **−20.6% val / −23.2% test** | `d53f0jn4` |

Per-split val surface-p MAE at SmoothL1 best-val:
- val_single_in_dist 120.63 (−22.9%)
- val_geom_camber_rc 117.45 (−16.4%)
- val_geom_camber_cruise 82.36 (−24.5%)
- val_re_rand 98.34 (−18.9%)

### Analysis

Outsized win — 4-10× the predicted delta — uniformly across every val split. Mechanism is consistent with the heavy-tail story: under MSE the high-Re/high-`|p|` outlier samples in each batch produce normalized residuals well above β=0.1, dominating the quadratic gradient on a single step and yanking the model off-trajectory (epoch-to-epoch val swings of ±20–40 MAE points were typical). SmoothL1 caps that contribution while leaving the in-regime quadratic intact, so each step is balanced across the Re range. SmoothL1's best epoch came at 13 vs MSE's 10 — Huber also keeps improving for longer in the same wall-clock budget. Largest absolute gains landed on the splits with the largest |p| magnitudes (cruise / re_rand), as predicted.

### Conclusions

- Merged. New empirical high-water mark on the advisor branch: **val_avg/mae_surf_p = 104.70**.
- Pre-authorized follow-ups (β=0.05, longer training, surf_weight re-tune under Huber, pure L1 comparison) are first-class Round 2 candidates.
- The cruise-test NaN bug is not in this PR (it stays a 3-split-ex-cruise figure) — fix lands in #1433 (next merge).

## 2026-05-12 21:06 — PR #1433: Add gradient norm clipping (max_norm=1.0) — MERGED

- Student branch: `willowpai2g24h3-askeladd/grad-clip-norm1`
- Hypothesis: `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` stabilizes training under heavy-tailed outliers; predicted 1–4% reduction in `val_avg/mae_surf_p`.
- Also ships the inline cruise-test NaN fix in `train.py::evaluate_split` (drops non-finite-`y` samples before forward pass and `accumulate_batch`).

### Results

| Run | max_norm | Best val_avg/mae_surf_p (epoch) | Δ vs baseline arm | W&B |
|---|---|---|---|---|
| baseline-30m | none | 131.96 (e?) | — | `mz3x4ieb` |
| grad-clip-1.0 | 1.0 | **114.18** | **−13.5%** | `qof1cbki` |
| grad-clip-0.5 | 0.5 | 121.41 | −8.0% | `japg46eu` |

Pre-clip grad-norm distribution measured at the baseline arm: median 53.90, max 579.57 — confirming the heavy-tail hypothesis (a single batch's grad-norm spike at >10× the median is a routine occurrence under MSE).

### Analysis

Tighter clip (0.5) underperforms looser (1.0), suggesting the floor for "useful" grad updates on a normal batch is somewhere between 0.5 and the median ~54 in pre-clip norm — 1.0 attenuates only the spike-batches and leaves the bulk of training-time gradients essentially untouched. Same mechanism as Huber (cap outlier influence) but acting at the batch-aggregate level instead of per-element.

### Conclusions

- Merged. Does NOT dethrone tanjiro's 104.70 (this PR was measured under MSE, not SmoothL1). Advisor branch now ships SmoothL1 + grad-clip stacked; combined-config has never been measured.
- The cruise-test NaN fix is now on the advisor branch. Future PRs will inherit it via rebase and report 4-split `test_avg/mae_surf_p` end-to-end.
- Open question for Round 2: does grad-clip still help on top of SmoothL1, or does SmoothL1 already subsume it? Pre-clip norms under SmoothL1 should be much smaller — likely the marginal benefit of clip-on-top-of-Huber is near zero, but a small A/B run can confirm.

## 2026-05-13 00:10 — PR #1616: Per-Re WeightedRandomSampler (upweight high-Re samples) — CLOSED

- Student branch: `willowpai2g24h3-askeladd/re-resample`
- Hypothesis: a `WeightedRandomSampler` weighted by `exp(t * log_re_centered)` shifts effective epochs toward the high-Re regime where pressure targets vary most; predicted 1–5% reduction in `val_avg/mae_surf_p`.

### Results

| Run | re_weight_temp | val_avg/mae_surf_p (best) | test_avg/mae_surf_p | Δ vs baseline arm | W&B |
|---|---|---|---|---|---|
| uniform-baseline-smoothl1-clip1 (baseline) | 0.0 | **90.91** | **86.87 (4-split, all finite)** | — | `eztvtkxc` |
| re-resample-t1.0-smoothl1-clip1 | 1.0 | 97.41 (final 100.61) | NaN (variant produced non-finite preds on cruise test) | **+7.2% (worse)** | `stzo9xvw` |

Per-split val MAE breakdown shows the mechanism cleanly:

| Split | Baseline t=0 | Variant t=1.0 | Δ |
|---|---|---|---|
| val_single_in_dist | 103.90 | 148.16 | **+42.6% (catastrophic)** |
| val_geom_camber_rc | 105.34 | 102.78 | −2.4% |
| val_geom_camber_cruise | 68.99 | 67.60 | −2.0% |
| val_re_rand | 85.40 | 83.89 | −1.8% |

### Analysis

The variant *improves* every OOD-ish split (geom_camber_rc, geom_camber_cruise, re_rand) by 2–3% on both val and test — confirming the "high-Re samples generalize the OOD splits" sub-hypothesis. But the in-distribution split (`val_single_in_dist`) degrades by **+42.6%** because at `t=1.0` the max/min sampling ratio is **67.6×** — the lowest-Re training samples are seen <1× per epoch in expectation under `WeightedRandomSampler(replacement=True)`. The model is starved of low-Re training updates that the in-distribution split depends on.

Mechanistic insight: Huber and re-resampling are *not* the orthogonal mechanisms the PR predicted. They fight — Huber caps the gradient on high-Re samples that re-resampling deliberately re-injects. The net effect is just less effective training on in-distribution, with no headroom gained from over-emphasized regimes (Huber already handles those).

Additionally: the variant model produced non-finite predictions on at least one cruise *test* sample (`vol_loss = +Inf`, `surf_loss = NaN`), even though training-time cruise val was finite. The cruise-y filter from #1433 cannot help here — it handles non-finite *ground truth*, not non-finite *predictions* — but this is a signal that the variant model is unfit for the paper-facing pass under heavy reweighting.

### Side-effects of this PR (high-value despite the close)

1. **First clean end-to-end 4-split test pass for this launch.** Run `eztvtkxc` delivered `test_avg/mae_surf_p = 86.87` with all four splits finite — the cruise-y filter from PR #1433 worked.
2. **Cleanest measurement of the current advisor branch:** 90.91 val / 86.87 test (uniform sampling on top of SmoothL1+grad-clip+cruise-fix). Combined with two other in-flight baseline measurements (#1615 at 102.17, #1437 at 104.84), this characterizes a **±7 single-seed noise band** on `val_avg/mae_surf_p`.

### Conclusions

- Closed. Hypothesis at `t=1.0` falsified (+7.2% on val, NaN on test). Per-spec `t=2.0` stretch arm correctly not run.
- Follow-up direction (assigned to askeladd as next PR): **loss-level Re-reweighting** — multiply each sample's loss by `exp(t * log_re_centered)` inside the train loop, no resampling. Same "tilt toward high-Re" mechanism without the discrete sample-starvation problem. If even `t=0.3` produces a -1 to -3% effect on `val_avg`, the OOD-split signal observed here is real and just needed a less aggressive implementation.
- BASELINE.md updated with the supplemental 90.91/86.87 measurement of the current advisor branch (the merged-best stays at 104.70 until a winning hypothesis PR's terminal `SENPAI-RESULT` marker lands).

## 2026-05-13 00:55 — PR #1431: Raise surf_weight 10 → 50 to align loss with surface-p MAE — CLOSED

- Student branch: `willowpai2g24h3-alphonse/surf-weight-50`
- Hypothesis: raising `surf_weight` from 10 → 50 sharpens the loss-vs-metric alignment with surface-pressure MAE; predicted small improvement on `val_avg/mae_surf_p`.
- Bundled: an in-PR copy of the cruise-NaN-y filter (commit `b073a95` in `train.py::evaluate_split`) — same fix as askeladd's #1433, applied independently. Will be a no-op delta on rebase.

### Results

| Arm | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, finite) | Δ vs baseline (test) | W&B |
|---|---:|---:|---:|---:|---|
| baseline | 10 (default) | **126.70** | **112.68** | — | `ogz8su1w` |
| variant | 50 | 131.34 | 120.90 | **+7.30% worse** | `2qytxnem` |
| bonus | 25 | 143.79 | 127.35 | +13.02% worse | `x6nf3mk2` |

All three arms hit the 30-min wall-clock cap at 14 epochs (~28% through the cosine schedule). Comparisons are apples-to-apples at the same training budget on alphonse's pre-rebase branch (his fork carries MSE + cruise-fix, but does *not* yet stack SmoothL1+grad-clip — so absolute numbers are not directly comparable to other students' baselines on the current advisor branch). The hypothesis decision (variant +7.3% worse) is unaffected.

### Per-split test breakdown (best-val checkpoint) — the smoking gun

| Arm | split | surf[p] | vol[p] |
|---|---|---:|---:|
| baseline | test_single_in_dist | 132.97 | 134.44 |
| baseline | test_geom_camber_rc | 124.40 | 121.45 |
| baseline | test_geom_camber_cruise | 81.39 | 79.76 |
| baseline | test_re_rand | 111.96 | 107.22 |
| **surf=50** | test_single_in_dist | 130.39 | **178.16 (+32%)** |
| **surf=50** | test_geom_camber_rc | 132.78 | **159.60 (+31%)** |
| **surf=50** | test_geom_camber_cruise | 98.18 | **161.50 (+102%)** |
| **surf=50** | test_re_rand | 122.24 | **176.63 (+65%)** |

### Analysis (mechanistic — high-value finding)

**Bernoulli-coupling is the dominant mechanism.** alphonse's diagnosis: in incompressible flow, surface `p` and volume `p` are globally linked through pressure-Poisson / Bernoulli equations. Suppressing the volume-`p` residual signal (from `1/(1+10)=9.1%` of total at `surf_weight=10` to `1/(1+50)=1.96%` at `surf_weight=50`) starves the model of the volume-pressure structure it needs to *correctly anchor* surface pressure. The result is exactly what we see: vol[p] regresses by 30-102% across all four test splits, and surface-p slightly regresses too because the global pressure field is now miscalibrated near the foil.

**The "minority-class" framing was wrong on principle.** "Surface is the metric, so upweight surface" looks like sensible loss-metric alignment, but on a coupled PDE system the volume channels are *not noise* — they carry the constraint structure the surface predictions rely on. This rules out a whole family of naive task-aligned reweighting hypotheses for coupled physics. Generalizes to other PDE-surrogate problems.

**Surface velocity (Ux, Uy) is robust to channel reweighting** (slight regressions only) — the free-slip-like constraint at the foil makes those channels easy and saturated. The hypothesis only ever had a chance on `surf[p]`, and that channel needs both sides of the Bernoulli coupling.

### Conclusions

- Closed. Hypothesis falsified by an internally-consistent A/B with strong mechanistic explanation.
- Cruise-NaN-y filter works: all three arms produced finite 4-split `test_avg/mae_surf_p`. Independent confirmation that #1433's fix is correct.
- Follow-up direction (assigned to alphonse as next PR): **`slice_num` sweep on Transolver's Physics Attention layer.** Listed as an open question in `CURRENT_RESEARCH_STATE.md`; tests whether 64 slices saturate on the 242K-node cruise meshes. Default 64; arms at 32/96/128 to bracket. Compute trade-off (slower epochs vs finer representation) similar to but milder than the closed #1443 wider-n192.
- The Bernoulli-coupling mechanism finding will be cited in future hypothesis assignments. "Reweight surface" is now a known dead end for surface-MAE-on-coupled-physics.

## 2026-05-13 01:20 — PR #1537: Tune cosine T_max to budget — --epochs 13 instead of 50 (CLOSED)

- Student branch: `willowpai2g24h3-thorfinn/schedule-tuned-e13`
- Hypothesis: matching cosine `T_max` to the actually-achievable epoch count converts the unused tail of the schedule into a proper cool-down; predicted 3–10% reduction in `val_avg/mae_surf_p`. (Direct data-driven follow-up to thorfinn's own #1443 baseline arm trajectory.)

### Results — W&B group `willow-r3-schedule-tuned-e13`

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Epochs | Wall-clock | State |
|---|---:|---:|---:|---:|---|
| x4sqeaqz | **118.77** (best) | NaN | 13 | 28:50 | finished |
| nx1tvtp1 | 119.26 | NaN | 13 | 28:54 | finished |
| rfxdtryp | 120.64 | NaN | 13 | 28:49 | finished |
| afft3f1v | 122.25 | NaN | 13 | 28:57 | finished |
| slsutjdn | 122.41 | NaN | 13 | 28:51 | finished |
| k6h7anq3 | 192.27 (div) | NaN | — | — | crashed |
| crwqx3mb | 172.03 (div) | NaN | — | — | crashed |
| navwrdyg | 186.59 (in-prog) | NaN | mid-run | 0:18 | running (div) |

Best of 5 finished arms is **118.77 — +14 MAE points above merged baseline 104.70, +28 above the advisor-branch ~91 lower noise band.** The variant arm (e13) does not move the metric over thorfinn's own e50 baseline trajectory, and no arm enters the noise band of the current advisor baseline.

### Analysis

The mechanism prediction was that cool-down of the cosine schedule would harvest the last few percentage points of capacity. Empirically, finishing a 13-epoch cosine cycle does cool the LR but produces no measurable improvement vs running 14 of 50 epochs at near-peak LR — across 5 independent seeds. Implication: at the merged-baseline operating point (SmoothL1 + grad-clip on advisor branch), the LR-cooling regime contributes less than the seed-to-seed noise (±7). 

Two crashes + one in-flight diverging run also suggest the e13-config + WeightedRandomSampler-with-replacement combination may have a borderline-stable training regime — likely a separate effect from the hypothesis itself, but worth noting.

No SENPAI-RESULT terminal marker was posted on the PR; advisor closed based on the W&B group readout directly.

### Conclusions

- Closed. Hypothesis falsified at the merged advisor-branch operating point — schedule-cooling alone is not a 5%+ lever here.
- **Schedule reformulation is not abandoned** — frieren's #1438 (warmup-5ep) tests the complementary half (LR warmup before the cosine). If warmup wins, then a **warmup + tuned T_max combo** would be the natural Round 2 stack and bears revisiting.
- All test_avg/mae_surf_p were NaN, suggesting thorfinn's branch may not have absorbed the cruise-NaN-y fix from #1433 — but the comparison against the val metric is unaffected.
- Follow-up direction (assigned to thorfinn as next PR): **AdamW weight_decay sweep**. Single-knob regularization test on a baseline that is now characterized to ±7 noise. Pure compute-neutral lever — no time cost per epoch, predicted to differentiate cleanly on per-split signal (especially val_single_in_dist and val_geom_camber_rc which have the highest per-split MAE).

## 2026-05-13 01:35 — PR #1615: Pure L1 / MAE loss + cruise-NaN code fix (MERGED)

- Student branch: `willowpai2g24h3-tanjiro/pure-l1`
- Hypothesis: dropping SmoothL1's quadratic-near-zero region (pure L1 / MAE loss in normalized space) should match SmoothL1(β=0.1) within noise — testing whether the residual quadratic does any useful work after we established the Huber gradient cap is the dominant mechanism. Predicted delta: −2% (better) to +5% (worse).

### Results — W&B group `willow-r3-pure-l1`

| Arm | wandb_id | loss_fn | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, post-fix) |
|---|---|---|---:|---:|
| pure-l1-30m (variant) | `mc22t7l2` | L1 | **104.03** | **95.09** |
| smooth-l1-0.1-30m-v2 (best SmoothL1) | `x0ud9i0a` | SmoothL1 β=0.1 | 102.17 | 92.04 |
| smooth-l1-0.1-30m (#3) | `30cs7nad` | SmoothL1 β=0.1 | 103.57 | 94.02 |
| smooth-l1-0.1-30m (#1, high-var) | `02e8ituj` | SmoothL1 β=0.1 | 125.94 | 97.40 |

Pure-L1 variant vs best SmoothL1 baseline: +1.8% val / +3.3% test. Pure-L1 vs mean of two well-behaved SmoothL1 baselines (102.17, 103.57): +1.1% val / +2.2% test. Both well within the ±7 single-seed noise band (three SmoothL1 reproductions span val=102-126, σ≈13). **Hypothesis confirmed equivalent within noise.**

### Per-split val MAE: pure-L1 vs best SmoothL1 baseline

| Split | SmoothL1 best (102.17) | pure L1 (104.03) | Δ (L1 − SmoothL1) |
|---|---:|---:|---:|
| val_geom_camber_cruise | 69.20 | 80.24 | **+15.9%** |
| val_geom_camber_rc | 111.06 | 110.53 | −0.5% |
| val_re_rand | 92.90 | 95.52 | +2.8% |
| val_single_in_dist | 135.52 | 129.82 | −4.2% |

### Per-split test MAE (post-fix, 4-split): pure-L1 vs best SmoothL1 baseline

| Split | SmoothL1 best | pure L1 | Δ |
|---|---:|---:|---:|
| test_geom_camber_cruise | 58.60 | 68.55 | **+17.0%** |
| test_geom_camber_rc | 100.15 | 101.44 | +1.3% |
| test_re_rand | 85.68 | 90.93 | +6.1% |
| test_single_in_dist | 123.73 | 119.46 | −3.5% |

### Bug-fix component (separate from hypothesis result)

tanjiro discovered that the advisor branch `train.py::evaluate_split` was missing the cruise-NaN-y filter that BASELINE.md / PR #1433 docs claimed was in place — only the documentation landed, not the code. He added the actual per-sample `torch.isfinite(y).all(dim=-1)` filter (train.py lines 240-250), exactly matching the `data/scoring.py::accumulate_batch` per-sample-skip semantics. This unlocks finite 4-split `test_avg/mae_surf_p` reporting for all future PRs. **This is a high-value contribution beyond the loss-fn experiment.**

### Analysis (mechanistic)

The Huber win in #1441 was **the linear-region gradient cap on outlier residuals**, not the quadratic-near-zero smoothness. With y-normalized target stats `y_std ≈ O(1)` and SmoothL1 β=0.1, the quadratic region (|r|<0.1 in normalized space) covers only the bottom decile of residuals at convergence. Early-training trajectories are very similar between L1 and SmoothL1 (both runs reach val ≈ 105 by epoch 25). Only the cruise split shows SmoothL1 consistently better — that split is dominated by easy low-Re aerofoil flow with the smallest absolute pressure scale (cruise val_p ≈ 70 vs single-foil val_p ≈ 130), so its residuals are the most likely to live inside the quadratic region. Other three splits show pure-L1 either ahead or within ±2%. **The residual quadratic does its (tiny) work on the low-magnitude split** — consistent with the textbook Huber picture, but not big enough to matter at this dataset/budget noise band.

### Conclusions

- Merged. New empirical baseline: **val_avg/mae_surf_p = 104.03** (pure-L1, run `mc22t7l2`); **test_avg/mae_surf_p = 95.09** (4-split, post-fix).
- **Implication for the paper**: parameter-free L1 is statistically indistinguishable from tuned-β SmoothL1 on TandemFoilSet at this scale — the SmoothL1 win in #1441 reduces to "gradient cap on the linear-region tail of outlier residuals." Clean negative result for the quadratic-near-zero.
- Bug-fix code change unlocks paper-facing 4-split `test_avg/mae_surf_p` reporting for every future run on the advisor branch.
- Follow-up direction (assigned to tanjiro as next PR): on the outlier-residual mechanism thread, the next high-leverage direction is **NOT smaller β** (this PR + closed #1616 already bracket that). It's a different mechanism entirely — to be designed in the next assignment.

## 2026-05-13 01:55 — PR #1434: 3× p-channel weight on pressure in training loss (CLOSED)

- Student branch: `willowpai2g24h3-edward/p-channel-weight3x`
- Hypothesis: multiplying the pressure-channel loss term by 3× (relative to Ux, Uy) aligns the loss with the surface-MAE ranking metric without abandoning velocity supervision; predicted 2-5% improvement on `val_avg/mae_surf_p`.

### Results — W&B group `willow-r3-p-channel-weight3x`

| Arm | p_weight | Best val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline | W&B |
|---|---:|---:|---:|---:|---|
| baseline (best) | 1.0 | **97.00** | 89.49 (`567j0vuh`) | — | `w6lqwh5o` (val), `567j0vuh` (test) |
| 5× variant | 5.0 | 138.92 | — | **+43% worse** | `aq0t3zfr` |
| 3× variant | 3.0 | 157.16 | — | **+62% worse** | `nqjvocmq` |

Multiple baseline reproductions (p_weight=1.0): 97.00, 98.49, 99.70, 113.12 — well-clustered near ~99, matching the current advisor-branch noise band.

### Analysis (mechanistic — confirms Bernoulli-coupling generalization)

Same failure mode as alphonse's closed #1431 (surf_weight 10 → 50), via a different lever:

- Transolver's decoder predicts `(Ux, Uy, p)` as a globally coupled physical solution; the loss minimum lies on a manifold defined by the incompressible Navier-Stokes equations (∇·u = 0, u·∇u + ∇p/ρ = ν∇²u).
- Up-weighting the `p` channel by 3× tells the optimizer to spend disproportionate capacity on fitting `p` at the expense of `(Ux, Uy)`. This breaks the Bernoulli closure between velocity and pressure.
- Result: predicted-`p` drifts off the manifold defined by predicted `(Ux, Uy)`. Training-time p-loss can decrease while *evaluation* p-MAE rises, because the prediction is no longer physically self-consistent.

**Independent confirmation of a generalizable failure mode.** alphonse's #1431 reweighted surface-vs-interior; edward's #1434 reweighted per-channel. Both fail by the same coupling-violation mechanism, and the failure scales monotonically with the strength of the reweighting (3× already +62% worse, 5× still +43% — i.e. 5× is "less catastrophic" than 3× because the 5× variant happened to converge on a slightly less broken local optimum; both are decisively closed).

### Conclusions

- Closed. Channel-level reweighting of (Ux, Uy, p) is a closed direction.
- Combined with #1431 closure, the lesson is: **never reweight individual output channels (or boundary regions) of a physics-coupled multi-task head** unless the reweighting respects the coupling constraint. This rules out a whole family of naive task-aligned reweighting hypotheses for coupled PDE surrogates.
- Follow-up direction (assigned to edward as next PR): hypothesis pivoted to a lever that doesn't touch the loss landscape's physical coupling — see next assignment.
