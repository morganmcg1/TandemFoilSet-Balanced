# SENPAI Research State

- **Last updated:** 2026-05-15 17:55 (tanjiro #3294 awaiting rebase; thorfinn #3223 closed; thorfinn assigned n_hidden=96)
- **Most recent research direction from human researcher team:** none (no open issues).
- **Current best (in-flight, pending merge):** `val_avg/mae_surf_p` = **100.811** (PR #3294 warmup+cosine 14ep, lr=7e-4) — rebasing
- **Official baseline:** `val_avg/mae_surf_p` = **109.681** (PR #3276 grad-clip + AdamW selective decay)
- **Current focus:** merge tanjiro's #3294 (new baseline ~100.81), then retest frieren/fern's orthogonal axes on new baseline.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — `val_avg/mae_surf_p` **109.68** (current best)

Key config: SmoothL1 (Huber, β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (LN/bias/1D no-decay) + NaN sample guard in evaluate_split.

## Active PRs

| PR | Student | Hypothesis | Status | Result |
|----|---------|-----------|--------|--------|
| #3294 | tanjiro | Warmup+cosine 14ep, lr=7e-4 | WIP (rebase) | **100.811 val (-8.08%)** → pending merge |
| #3304 | frieren | surf_weight=20 | REVIEW HOLD | 103.668 val (-5.49%) → retest after #3294 merges |
| #3314 | fern | weight_decay=3e-4 | REVIEW HOLD | 105.640 val (-3.69%) → retest after #3294 merges |
| #3295 | edward | slice_num=128 | WIP | — |
| #3301 | alphonse | width-192, epochs=10 | WIP | — |
| #3344 | nezuko | Random Fourier Features (Tancik 2020 RFF) | WIP | — |
| #3362 | askeladd | n_head 4→8 (single-axis) | WIP (new) | — |
| #3377 | thorfinn | n_hidden=96 (width sweep counterpart to #3301) | WIP (new) | replaces #3223 (closed, BF16+batch=8 regressed +34%) |

## Confirmed design insights (from completed rounds)

### Budget-matching (critical from round-1)
Under a 30-min cap with 50-epoch cosine, LR never anneals. Matching epochs to completable epochs so cosine cools fully is the key fix. Tanjiro's #3294 confirms: warmup+cosine over 14 epochs gives the best epoch = epoch 14 with monotonically improving late-stage checkpoints.

### Regularization levers both help independently (round-2 partial)
- grad-clip + selective decay: +5.94% (PR #3276, merged baseline)
- surf_weight=20: +5.49% vs old baseline (frieren #3304, on hold)
- weight_decay=3e-4: +3.69% vs old baseline (fern #3314, on hold)
- warmup+cosine 14ep: +8.08% vs old baseline (tanjiro #3294, pending merge)

All improvements concentrate on `single_in_dist` (hardest split) with minor cruise regression.

### Depth-8 needs more budget
Depth-8 at ~205 s/epoch is squeezed into 9 epochs — still descending. Hold until BF16/batch8 reduces per-epoch cost.

## Open questions (in-flight)
- Does width-192 actually help when schedule fits? (alphonse #3301)
- Does width-96 help via faster training + less overfitting? (thorfinn #3377)
- Does slice_num=128 beat 64 single-axis? (edward #3295)
- Does corrected RFF improve geometry-split generalization? (nezuko #3344)
- Does n_head=8 help at current slice_num=64? (askeladd #3362)
- Does surf_weight=20 still help ON TOP of tanjiro's warmup+cosine? (frieren #3304 on hold)
- Does weight_decay=3e-4 still optimal on warmup+cosine? (fern #3314 on hold)

## Closed/regressed (round-2/3)
- #3302 askeladd depth-8: +1.53% regression (budget-bound; revisit if BF16/batch8 lands)
- #3223 thorfinn BF16+batch=8: +34% regression (padding overhead dominates; bug-fix work merged via #3276)

## Next priorities (round-3+)

Immediate next steps after tanjiro #3294 merges (~100.81 baseline):
1. Release frieren/fern holds → rebase+retest on new baseline
2. Process alphonse/edward/thorfinn results as they land
3. If thorfinn's BF16+batch8 gives speedup → depth-8 becomes viable again
4. Tanjiro's own follow-ups: per-step cosine LR, eta_min>0, lr sweep around 7e-4

Longer-term research directions:
- H2: Per-sample output scale normalization (y-std variability is 40× across dataset) — likely high impact
- H1: FiLM conditioning on global Re/geometry params
- H3: Separate surface/volume decoder heads
- H13: Log-scale pressure loss in train only
- Diagnose why geom_rc lags — harder geometry-OOD split, capacity or domain coverage

## Plateau watch
Not yet — tanjiro's result (-8.08%) is the largest single improvement so far. Next threshold: can we reach val_avg/mae_surf_p < 95 by compounding schedule + surf_weight + weight_decay?
