# SENPAI Research State

- **Date**: 2026-05-15 15:35
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 1 experiments completing; Round 2 assignments active
- **Most recent human research directive**: None received

## Current Best

**PR #3166 (FiLM) — val_avg/mae_surf_p = 114.6268** (merged 2026-05-15)
⚠ Note: this includes FiLM conditioning (cond_dim=11) — not a pure baseline.
⚠ Clean baseline from PR #3168 (slice_num=96, no FiLM): 149.27 → confirms FiLM is contributing ~35 points (-23% improvement).

## Key Confirmed Insights

1. **FiLM conditioning is very effective**: clean unmodified Transolver (PR #3168, slice_num=96) gives val_avg=149.27 vs FiLM baseline 114.63 — FiLM provides ~35 point improvement. This is now confirmed (not just assumed).
2. **T_max=50 mismatch**: All Round 1 experiments run CosineAnnealingLR(T_max=50) but timeout at ~14 epochs. LR never anneals. Future experiments should use T_max=14 or T_max=actual_achievable_epochs.
3. **Wider model stresses budget**: n_hidden=256 only fits 7 epochs in the 30-min wall; half the epoch budget means invalid comparison.
4. **Scoring NaN bug confirmed**: `test_geom_camber_cruise/000020.pt` has non-finite GT; `nan*0=nan` propagates through masked accumulator. All test_avg values are NaN. Workaround: tanjiro's `recompute_test.py` skips NaN-GT samples.

## Active WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3154 | alphonse | H5: Matched-budget wider model (A: std, B: 256) | Sent back, revising |
| #3156 | askeladd | H1: p-channel surf loss upweight 3x, 5x | Active WIP |
| #3158 | edward | H2: EMA paired comparison (A: no-EMA baseline, B: EMA-only) | Sent back, revising |
| #3160 | fern | H4: Huber loss delta=1.0, 0.5 | Active WIP |
| #3163 | frieren | H3: Grad clip + warmup | Active WIP |
| #3170 | thorfinn | H11: Deeper model (5→7, 5→8 layers) | Active WIP |
| #3284 | nezuko | H12: Clean baseline + T_max=15 ablation (no FiLM) | Active WIP |
| #3297 | tanjiro | H13: Surface dual-head (dedicated surface MLP) | New assignment |

## Key Open Questions

1. **Does T_max fix matter (PR #3284 Arm B)?** If T_max=15 beats Arm A (~149), the schedule fix is a mandatory default for all future experiments.
2. **How much of Round 1 was hurt by T_max mismatch?** PR #3154 and #3158 sent back for paired comparison.
3. **Will the surface dual-head work without FiLM?** If PR #3297 beats 114.63 with no FiLM, the surface head alone is a huge win.
4. **Do Huber / loss-upweight / grad-clip actually help?** PRs #3156, #3160, #3163 still in flight.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all models. File is read-only. Use `recompute_test.py` (from tanjiro PR #3168 branch) or report 3-split average as workaround.

## Potential Next Research Directions

- **FiLM + T_max fix compound**: Apply both together as the new default config once T_max impact is measured.
- **FiLM + surface dual-head**: If both help independently, compound them.
- **Per-sample adaptive loss**: Normalize loss contribution per sample by its y_std — high-Re samples dominate because of large absolute errors.
- **Graph-based positional encoding**: Current coordinates (x,z,sdf,dsdf) are Euclidean. Geodesic distances along the foil surface could help surface node specialization.
- **Stochastic weight averaging (SWA)**: Alternative to EMA that averages along the cosine schedule valley.
- **Mesh-size-aware slice budget (PR #3168 insight)**: cruise large meshes benefit from slice_num=128, raceCar small meshes benefit from 96. Dynamic slice allocation by domain.
- **WSD (Warmup-Stable-Decay) schedule**: Replace cosine with WSD to get stable plateau followed by sharp decay — better suited to the short wall budget.
- **Lower slices**: PR #3168 suggests optimum may be at slice_num=32 or 48, not 64. Explore downward.
