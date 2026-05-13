<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# SENPAI Research State — TandemFoilSet

- **Date**: 2026-05-13 (updated 01:40)
- **Launch**: `willow-pai2g-24h-r3` (isolated 24h appendix experiment)
- **Advisor branch**: `icml-appendix-willow-pai2g-24h-r3`
- **W&B project**: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3`
- **Target metric** (lower is better): `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 splits)
- **Paper-facing test metric**: `test_avg/mae_surf_p`
- **Hard caps**: 30 min wall-clock per training run, 50 epochs, 1 GPU (96GB) per student
- **Verified best (merged)**: `val_avg/mae_surf_p = 104.03` (PR #1615, run `mc22t7l2`, pure-L1). Same-PR sibling SmoothL1(β=0.1) baseline arm landed at **102.17 val / 92.04 test** (`x0ud9i0a`) — within noise of the pure-L1 variant.
- **Paper-facing test metric**: `test_avg/mae_surf_p = 95.09` (4-split, post-fix, PR #1615 variant) — first verified 4-split test pass on a merged PR.
- **Cruise-test NaN bug — FULLY FIXED in code (PR #1615, MERGED)**: `test_geom_camber_cruise/000020.pt` has 761 Inf values in the p channel of `y`. The advisor branch was previously missing the per-sample finite-y filter in `train.py::evaluate_split` (BASELINE.md / PR #1433 docs claimed it was in code, but only the docs landed). PR #1615 adds `y_finite = torch.isfinite(y).all(dim=-1)` filter before forward pass at train.py:240-250, exactly matching `data/scoring.py::accumulate_batch` per-sample-skip semantics. All future PRs will natively report finite 4-split `test_avg/mae_surf_p` without student-side workarounds.
- **Single-seed noise band ≈ ±7 around mean ≈ 99** (confirmed by three independent SmoothL1 reproductions of the same code at 102.17 / 103.57 / 125.94 in PR #1615 alone, plus advisor-branch baselines 90.91, 102.17, 104.84). Implication: hypotheses claiming <5% improvement need multi-seed confirmation. "Headline" Round 2 merges need ≥10% relative gain to be visibly real on a single seed.

## Most recent direction from human researcher team
None yet — no GitHub issues open at start of launch.

## Current research focus

This is a fresh research track on a clean baseline. Round 1 covers eight orthogonal "compound levers" that each address a different known limitation of the baseline training recipe, all testable inside a single 30-min run.

The baseline Transolver recipe has several obvious soft spots:

1. **Loss alignment with the metric.** Ranking is surface-pressure MAE only, but the loss uses equal channel weights and surf_weight=10 (modest).
2. **Heavy-tailed targets.** y std varies by ~10x within a single split (high-Re samples dominate MSE gradients).
3. **No gradient hygiene.** No clipping, no warmup, no EMA — all standard transformer-training practices are absent.
4. **Slow throughput.** No mixed precision, so 50 epochs at batch=4 is the budget ceiling.
5. **Small capacity.** n_hidden=128 may be capacity-limited across order-of-magnitude Re range.

## Round 1 hypothesis assignments

| Slug | Status | Mechanism | Risk |
|---|---|---|---|
| ~~`surf-weight-50`~~ (alphonse, #1431) | **CLOSED** | Raise surf_weight 10→50 — variant +7.3% worse on test (Bernoulli coupling breaks); vol[p] degraded 30-102% across all test splits | — |
| ~~`grad-clip-norm1`~~ (askeladd, #1433) | **MERGED** | clip_grad_norm_ at 1.0; ALSO ships cruise-NaN fix. 114.18 (−13.5%) under MSE | Low |
| `p-channel-weight3x` (edward, #1434) | WIP-rebased-running | 3× weight on pressure channel in loss; new arm `wdmlxbyx` showing 106.6 mid-run; 3 prior crashes (incl. one NaN) post-rebase | Low–Med |
| `ema-decay999` (fern, #1437) | WIP-rebased-running | EMA model weights; rebase done, baseline arm landed at 104.84 (in-band w/ 104.70); variant arm pending | Low |
| `warmup-5ep` (frieren, #1438) | WIP-rebase-needed | 5-epoch linear LR warmup; advisor told frieren to rebase onto current advisor branch | Low |
| `amp-bf16` (nezuko, #1440) | WIP-likely-dead | bfloat16 autocast; multiple finished arms all 131–184, plus 1 crash. Likely close after terminal results post | Medium |
| ~~`smooth-l1-beta01`~~ (tanjiro, #1441) | **MERGED (WINNER)** | SmoothL1(β=0.1). **104.70 (−20.6%)** — new best | Medium |
| ~~`wider-n192`~~ (thorfinn, #1443) | **CLOSED** | n_hidden 128→192, n_head 4→6 — variant +33% (compute-bound under 30-min cap) | — |
| ~~`schedule-tuned-e13`~~ (thorfinn, #1537) | **CLOSED** | --epochs 13; 5 finished arms in 118-122 (best 118.77, +14 above 104.70); 2 crashes + 1 diverging mid-run. Schedule-cooling <noise | — |
| ~~`pure-l1`~~ (tanjiro, #1615) | **MERGED** | Drop SmoothL1 quadratic-near-zero; variant 104.03 val / 95.09 test (4-split, post-fix) ≈ SmoothL1 baseline 102.17/92.04 within ±7 noise. **Also ships the missing cruise-NaN-y filter code in evaluate_split** (only docs had landed before — a high-value bug-fix beyond the loss-fn experiment) | Low |
| ~~`re-resample`~~ (askeladd, #1616) | **CLOSED** | log(Re)-weighted WeightedRandomSampler; uniform baseline arm `eztvtkxc` hit **90.91/86.87** (new W&B low); variant 100.61 (+7.2% worse, starves low-Re) | — |

## Round 2 hypothesis assignments

| Slug | Status | Mechanism | Risk |
|---|---|---|---|
| `slice-num-sweep` (alphonse, #1747) | WIP | Transolver Physics Attention `slice_num` sweep at 32/64/96/128 — capacity test on 242K-node cruise meshes | Low–Med |
| `re-loss-weight` (askeladd, #1721) | WIP | Loss-level Re-reweighting (continuous control replaces discrete sampler starvation from closed #1616) | Low |
| `weight-decay-sweep` (thorfinn, NEW) | WIP | AdamW weight_decay sweep {0, 1e-3, 1e-2, 5e-2} — regularization probe at characterized ±7 noise floor | Low |

Full Round 1 hypothesis design lives in `research/RESEARCH_IDEAS_2026-05-12_initial.md`. Results log lives in `research/EXPERIMENTS_LOG.md`.

## Potential next directions (Round 2+)

After Round 1 results land, the next wave will likely come from:

- **Combos of Round 1 winners.** Stack the strongest of {`surf-weight`, `grad-clip`, `p-weight`, `ema`} which are mostly independent levers.
- **Loss reformulation deeper.** Per-Re loss reweighting (downweight low-Re where MAE is naturally tiny; upweight high-Re). Relative MAE in normalized space.
- **Architecture beyond width.** Deeper Transolver (n_layers 5→8) and slice_num sweep (64→128) — but only if combined with schedule-tuning, since #1443 confirmed width-at-fixed-budget loses under 30-min cap.
- **Geometric features.** Fourier positional features for (x, z), surface-distance features, or per-foil indicator channels.
- **Augmentation.** Horizontal mirroring (with AoA sign flip and rear-foil swap for tandem) — exact CFD symmetry that doubles effective dataset.
- **Optimizer.** AdamW → Lion, Adan; larger weight_decay; layer-wise LR decay.
- **Cruise-test NaN bug investigation.** A separate scoring/data PR may be needed once we have a winner ready to merge — clean 4-split test_avg/mae_surf_p is required for paper-facing numbers.

The most promising single direction beyond Round 1 is likely a **best-checkpoint combo** of metric-aligned losses (`surf-weight` + `p-channel`) with training hygiene (`grad-clip` + `warmup` + `ema`) — stacking should compound to a multi-percent reduction in `val_avg/mae_surf_p`. Capacity scaling stays off the table until schedule-tuning is resolved.

## Active emerging lessons

- **Schedule-budget mismatch is a structural baseline weakness, but schedule-cooling alone does not move the metric.** thorfinn's `schedule-tuned-e13` (#1537, closed) ran 5 finished arms of `--epochs 13`; best 118.77, all clustered 118-122 — **+14 above merged 104.70 baseline and +28 above the advisor-branch ~91 noise floor.** Implication: the cooling regime contributes less than the seed-to-seed ±7 noise on this baseline. The schedule-reformulation lever is not dead, but it needs to be paired with warmup or peak-LR retuning (frieren's `warmup-5ep` #1438 tests the warmup half). The natural Round 2 stack remains **warmup + tuned T_max** — but only after the merged advisor branch grows the warmup component.
- **Width is compute-bound at this budget.** Any future architecture-scaling proposal must either reduce other compute (e.g. slice_num, mlp_ratio) or budget for fewer-but-fully-cooled epochs.
- **SmoothL1 is a 20% headline lever, not a 2-5% tweak.** Predicted delta was 2–5%; observed was 20.6%. Mechanism: capping per-element gradient magnitude on high-`|p|` outlier samples in normalized space. Gain is uniform across val splits, with the largest absolute gain on `val_geom_camber_cruise` (highest-magnitude split). This reframes Round 2 priorities: any technique that further attenuates per-sample outlier influence (β=0.05, pure L1, per-Re reweighting, gradient-norm-aware sample reweighting) jumps in priority above standard hygiene levers.
- **Grad-clip is a milder version of the same mechanism.** Under MSE, clip(1.0) recovered −13.5%, clip(0.5) only −8.0%. The looser clip helps because median pre-clip grad-norms are ~54 — only the spike-batches above 1.0 actually get attenuated. Under SmoothL1 the median norm should be much smaller, so the marginal benefit of grad-clip on top of SmoothL1 is an open question (probably small).
- **Run-to-run variance on the combined-config baseline is large (≈ ±7 on `val_avg/mae_surf_p`, σ ≈ 13 over the widest spread).** Tanjiro's three independent SmoothL1(β=0.1) reproductions of the *same code* landed at 102.17, 103.57, 125.94 (σ ≈ 13). Plus advisor-branch baselines spanned 90.91, 102.17, 104.84. Implication: any hypothesis claiming <5% improvement needs ≥2 seeds to overcome noise. Any "headline" merge in Round 2 likely needs to be a ≥10% relative gain over baseline to be visibly real on a single run. Sources of variance to investigate: WeightedRandomSampler-with-replacement vs `shuffle=True` (askeladd's loader uses the former by default — could be a meaningful regime difference), default seed, num_workers nondeterminism. **Note**: the 125.94 outlier from `02e8ituj` finished training fine but its best-val checkpoint clearly fell into a worse basin — suggests early-training stability matters (potential warmup-lr signal from frieren #1438).
- **Pure L1 ≈ SmoothL1(β=0.1)** (tanjiro #1615 MERGED). Variant `pure-l1-30m` at 104.03 vs his own SmoothL1 baseline at 102.17 → ~+1.8%, just outside his pre-defined ±0.5% equivalence threshold but well within the ±7 noise band. The per-split breakdown shows SmoothL1 only edges out pure-L1 on `val_geom_camber_cruise` / `test_geom_camber_cruise` (the low-|p| split where residuals are most likely to enter the quadratic region |r|<β=0.1) — a clean textbook Huber confirmation. **Conclusion**: SmoothL1's win in #1441 was almost entirely the linear-region gradient cap on outliers, NOT the quadratic-near-zero smoothness. The β-sweep direction is now closed (pure-L1 brackets from β→0; askeladd #1616 confirmed sample-reweighting hurts; closing this whole sub-family).
- **Re-weighted sampling by log(Re) does not help on top of SmoothL1** (askeladd #1616 unconfirmed terminal). `re-resample-t1.0` = 100.61 vs uniform-baseline = 90.91 → re-weighting hurts. Mechanistic interpretation: SmoothL1 already neutralizes the per-element outlier-gradient problem, so artificially upweighting the same outlier samples *for more frequent updates* is now redundant and biases the model toward high-Re geometry at the cost of low-Re generalization. The "heavy-tail" story was already addressed loss-side; addressing it again sample-side overshoots.

## Open questions to keep in view

- Are the val/test split distributions drifting? Worth comparing val and test surface-p MAE for each split after Round 1 to confirm val is a faithful predictor of test.
- Which split dominates the `val_avg`? If `val_re_rand` or `val_geom_camber_cruise` is much larger than the others, methods that help only the easy splits will be misleading.
- Does Transolver's `slice_num=64` saturate on 242K-node cruise meshes? Could be a capacity bottleneck specifically for cruise.
