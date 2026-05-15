# SENPAI Research State

- **Last updated:** 2026-05-15 19:30 (round-3 fanout: 8 students all bootstrapping training as of ~19:24 after GH rate-limit cleared)
- **Most recent research direction from human researcher team:** none (no open issues).
- **Current best:** `val_avg/mae_surf_p` = **100.811** (PR #3294 warmup+cosine 14ep, lr=7e-4)
- **Current focus:** Verify whether frieren/fern/nezuko axes (surf_weight, weight_decay, RFF) compound on top of the new schedule baseline.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — `val_avg/mae_surf_p` 109.68
3. **PR #3294** (warmup+cosine 14ep, lr=7e-4) — `val_avg/mae_surf_p` **100.811** (current best, -8.08%)

Key config: SmoothL1 (Huber β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (wd=1e-4) + NaN guard + SequentialLR (LinearLR 2ep warmup + CosineAnnealingLR T_max=12) + lr=7e-4 + epochs=14.

## Active PRs

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #3304 | frieren | surf_weight=20 single-axis (RETEST) | WIP (rebase) | Beat old baseline 109.68; retesting on new 100.81 |
| #3314 | fern | weight_decay=3e-4 (RETEST) | WIP (rebase) | Beat old baseline 109.68; retesting on new 100.81 |
| #3344 | nezuko | RFF Tancik 2020 (RETEST) | WIP (rebase) | Beat old baseline 109.68; retesting on new 100.81 |
| #3301 | alphonse | width-192, epochs=10 | WIP (just bootstrapped) | Pod idle until 19:22 due to GH rate-limit; now training |
| #3362 | askeladd | n_head 4→8 single-axis | WIP (new) | — |
| #3377 | thorfinn | n_hidden=96 (width sweep) | WIP (new) | — |
| #3397 | tanjiro | eta_min=1e-5 in cosine | WIP (new) | Follow-up to merged #3294 |
| #3399 | edward | slice_num=96 (mid-point) | WIP (new) | Follow-up to closed #3295 |

## Confirmed design insights

### Schedule is the dominant lever
- Budget-matched warmup+cosine (14ep, lr=7e-4) → -8.08% improvement (PR #3294, merged)
- Without budget-matching, 50-epoch cosine never anneals → flat high-LR training throughout

### Regularization axes both helped vs old baseline (need retest on new)
- surf_weight=20: val 103.668 vs old 109.68 baseline (-5.49%, frieren #3304)
- weight_decay=3e-4: val 105.640 vs old 109.68 baseline (-3.69%, fern #3314)
- RFF 32-freq: val 103.891 vs old 109.68 baseline (-5.28%, nezuko #3344)
- All three improvements concentrate on single_in_dist (hardest split). Mild cruise regression.

### Depth and slot-width fail under the wall-clock budget
- slice_num=128: +20% regression (budget-mismatch, too slow per-epoch)
- depth-8: +1.5% regression (same issue)
- Both need per-epoch cost reduction to be viable

### Width sweep in progress
- n_hidden=128 (baseline)
- n_hidden=192 (alphonse #3301, actively training)
- n_hidden=96 (thorfinn #3377, just assigned)

## Open questions
- Does surf_weight=20 still help ON TOP of warmup+cosine? (frieren #3304 retesting)
- Does weight_decay=3e-4 still optimal with lr=7e-4 + cosine? (fern #3314 retesting)
- Does corrected RFF compound with schedule? (nezuko #3344 retesting)
- Does eta_min=1e-5 extract more from the cosine tail? (tanjiro #3397 new)
- Does slice_num=96 balance slot-count vs per-epoch cost? (edward #3399 new)
- Does n_head=8 help? (askeladd #3362)
- Does n_hidden=96/192 expand or contract the right capacity? (thorfinn #3377, alphonse #3301)

## Plateau watch
Not yet — best is 100.81 after 3 rounds; ~8% total improvement per round. 
Theoretical floor: compression ratio of pressure field, not well-defined without ablations.
Next threshold: can we reach val < 95 by stacking schedule + surf_weight + weight_decay + RFF?

## Closed/regressed
- #3302 askeladd depth-8: +1.53% (budget-bound)
- #3223 thorfinn BF16+batch=8: +34% (padding overhead)
- #3295 edward slice_num=128: +20% (budget-mismatch, too slow)
- #3205, #3179, #3183, #3214, #3216, #3220 — round-1 dead ends
