# SENPAI Research State

- **Date**: 2026-05-15 15:55
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 1 completing; Round 2 (compounds) ramping up
- **Most recent human research directive**: None received

## Current Best

**PR #3160 (Huber δ=0.5, no FiLM) — val_avg/mae_surf_p = 112.8406** (merged 2026-05-15)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| Huber δ=0.5 (no FiLM) | **112.84** | Current best |
| FiLM (MSE) | 114.63 | Previous best |
| Huber δ=1.0 (no FiLM) | 115.99 | Other arm of PR #3160 |
| Clean no-mod baseline (slice_num=96) | 149.27 | PR #3168 — establishes raw reference |

Huber δ=0.5 beats FiLM by 1.79 pts. Two confirmed independent improvements (FiLM ~+35 vs raw, Huber δ=0.5 ~+1.8 vs FiLM) — compound being tested in PR #3311 (fern).

## Key Confirmed Insights

1. **FiLM conditioning is very effective**: clean unmodified Transolver (PR #3168, slice_num=96) gives val_avg=149.27 vs FiLM baseline 114.63 — FiLM provides ~35 point improvement (~23% reduction).
2. **Huber δ=0.5 beats FiLM alone**: PR #3160 confirms Huber δ=0.5 (no FiLM) reaches 112.84 — better than FiLM (114.63). Tighter Huber threshold linearizes more of the right tail, damping extreme-Re gradient dominance. Mild win (-1.6%) but real.
3. **Huber δ trend monotone**: δ=1.0 → 115.99, δ=0.5 → 112.84. Sweet spot may be tighter (δ=0.25 being tested in PR #3311).
4. **Huber wins on 3/4 splits, loses on val_geom_camber_rc**: Huber's right-tail damping hurts the hardest extreme-Re split where genuine large errors should drive learning. Possibly per-split or per-channel Huber would help.
5. **T_max=50 mismatch**: All Round 1 experiments run CosineAnnealingLR(T_max=50) but timeout at ~14 epochs. LR never anneals. PR #3284 will test T_max=15 fix.
6. **Wider model stresses budget**: n_hidden=256 only fits 7 epochs in the 30-min wall; PR #3154 sent back for matched-budget paired comparison.
7. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 has non-finite GT; `nan*0=nan` propagates. All test_avg values are NaN. Workaround: report 3-split test_avg excluding cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3154 | alphonse | H5: Matched-budget wider model (A: std, B: 256) | Sent back, revising |
| #3156 | askeladd | H1: p-channel surf loss upweight 3x, 5x | Active WIP — GPU 99%, slow run |
| #3158 | edward | H2: EMA paired comparison | Sent back, revising |
| #3163 | frieren | H3: Grad clip + warmup | ⚠ GPU idle 3h+, status pinged |
| #3170 | thorfinn | H11: Deeper model (5→7, 5→8 layers) | ⚠ GPU idle 3h+, status pinged |
| #3284 | nezuko | H12: Clean baseline + T_max=15 ablation (no FiLM) | Active WIP |
| #3297 | tanjiro | H13: Surface dual-head (dedicated surface MLP) | Active WIP |
| #3311 | fern | H14: FiLM + Huber compound (δ=0.5, 0.25) | Just assigned |

## Key Open Questions

1. **Does FiLM + Huber compound (PR #3311)?** Hypothesis says yes (mechanistically orthogonal). If val_avg < 110, the compound stacks. If ≈ 112-114, FiLM and Huber overlap. If > 114, they interfere.
2. **Does δ=0.25 continue the Huber trend (PR #3311 Arm B)?** Monotone improvement δ=1.0→0.5 suggests yes; if val_avg drops further, the sweet spot is even tighter.
3. **Does T_max fix matter (PR #3284 Arm B)?** If T_max=15 beats Arm A, schedule fix becomes new default.
4. **Will the surface dual-head work (PR #3297)?** Direct architectural target for the primary metric.
5. **Are frieren/thorfinn pods stuck?** GPU 0% for 3h. Pinged for status — if no progress, may need to reassign their PRs.

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
