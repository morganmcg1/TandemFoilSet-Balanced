# SENPAI Research State

- **Last updated:** 2026-05-15 21:36 (round-4 retests in flight; rate-limit storm cleared; all 8 students 100% GPU)
- **Most recent research direction from human researcher team:** none (no open issues).
- **Current best (merged):** `val_avg/mae_surf_p` = **97.757** (PR #3399 slice_num=96 on warmup+cosine baseline)
- **Current focus:** Verify whether frieren/fern/nezuko/askeladd axes (surf_weight, weight_decay, RFF, n_head=8) compound on top of warmup+cosine+slice_num=96 baseline.
- **Operational state:** All 8 student pods at 100% GPU after 38-min GH rate-limit storm cleared at ~21:21Z. Stale-base PR branches (#3304/#3314/#3344/#3362/#3377/#3301) have not been pushed with rebase commits but pod VRAM signatures suggest most students applied rebase locally before launching training. Verify final config from committed metrics.yaml on each student's results commit.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD, and #3399 pending merge)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — `val_avg/mae_surf_p` 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — `val_avg/mae_surf_p` 100.811
4. **PR #3399** (slice_num=96, pending merge) — `val_avg/mae_surf_p` **97.757** (-3.03%, NEW BEST)

Key config (after #3399 merge): SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=1e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12) + lr=7e-4 + epochs=14 + **slice_num=96**.

## Active PRs

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #3304 | frieren | surf_weight=20 (RETEST) | WIP (stale base) | Training on pre-#3294 config; needs rebase onto new HEAD after #3399 merge |
| #3314 | fern | weight_decay=3e-4 (RETEST) | WIP (stale base) | Same — training on old config stack |
| #3344 | nezuko | RFF 32-freq (RETEST) | WIP (stale base) | Same — needs rebase onto new HEAD |
| #3301 | alphonse | width-192 (rebase pending) | WIP (rebase) | Sent back: rebase onto HEAD + epochs=10 for budget |
| #3362 | askeladd | n_head 4→8 | WIP (stale base) | No code commit yet; recovering from rate-limit storm |
| #3377 | thorfinn | n_hidden=96 (rebase pending) | WIP (rebase) | Sent back: rebase + retest on new HEAD (val 102.082 on old base) |
| #3397 | tanjiro | eta_min=1e-5 in cosine | WIP (correct base) | Correct base; training in progress |
| #3399 | edward | slice_num=96 | **MERGED** | val 97.757 (-3.03% vs 100.811) → new baseline |
| #3453 | edward | T_max=10 calibration (slice96 budget) | WIP (new) | Completes cosine fully within ~12-ep cap |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → -8.08% improvement (PR #3294, merged)
- Without budget-matching, 50-epoch cosine never anneals → flat high-LR training throughout

### Slot-count is the second lever
- slice_num=64 → 96 (PR #3399): -3.03% improvement on top of warmup+cosine baseline
- slice_num=128: +20% regression (budget-mismatch, too slow per-epoch — closed PR #3295)
- slice_num=96 the sweet spot: richer attention, still fits ~12 epochs in 30-min cap

### Regularization axes both helped vs OLDER baseline (need clean retest on new warmup+cosine+slice96 stack)
- surf_weight=20: val 103.668 vs old 109.68 baseline (-5.49%, frieren #3304)
- weight_decay=3e-4: val 105.640 vs old 109.68 baseline (-3.69%, fern #3314)
- RFF 32-freq: val 103.891 vs old 109.68 baseline (-5.28%, nezuko #3344)
- All three concentrate improvement on single_in_dist (hardest split).

### Width sweep — stale results, retest pending
- n_hidden=96 (thorfinn #3377, stale base): val 102.082 vs old baseline, +1.26% vs new → needs rebase
- n_hidden=192 (alphonse #3301, stale base): val 99.611 vs new but wrong schedule → needs rebase
- Direction likely UP (alphonse's 99.611 under old schedule suggests width matters)

### Depth and head-count untested on new stack
- depth-8: +1.5% regression (budget-bound, closed #3302)
- n_head=8: no result yet (askeladd #3362, recovering from rate-limit)

## Open questions
- Does surf_weight=20 still help ON TOP of warmup+cosine+slice_num=96? (frieren #3304, needs rebase)
- Does weight_decay=3e-4 still optimal with new full stack? (fern #3314, needs rebase)
- Does corrected RFF compound with schedule + slot change? (nezuko #3344, needs rebase)
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3397 in-flight)
- Does n_head=4→8 help? (askeladd #3362 in-flight)
- Width 192 vs 128 on new full stack? (alphonse #3301 rebasing)
- Width 96 vs 128 on new full stack? (thorfinn #3377 rebasing)

## Critical: stale base issue (round 3 cascading failure)
Most round-3 students were assigned before tanjiro's #3294 merge. Their branches diverge from pre-#3294 HEAD, so they are running the OLD config (lr=5e-4, T_max=50, no warmup). Results from stale-base runs are not directly comparable to new baseline. After #3399 merges, ALL remaining students need to rebase onto the NEW HEAD (warmup+cosine + slice_num=96) before retesting.

## Plateau watch
Not yet — best is 97.757 after 4 rounds; total -12% from original 116.61 Huber baseline.
Next threshold: can we reach val < 95 by stacking schedule + surf_weight + weight_decay + RFF?
Key opportunity: the 3 known regularization improvements (surf_weight, weight_decay, RFF) each gave 3-5% vs OLD baseline — if they compound on the NEW stack, there's meaningful headroom.

## Closed/regressed
- #3302 askeladd depth-8: +1.53% (budget-bound)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch, too slow)
- #3377 thorfinn n_hidden=96 (stale base): sent back for rebase, +1.26% vs new baseline
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends
