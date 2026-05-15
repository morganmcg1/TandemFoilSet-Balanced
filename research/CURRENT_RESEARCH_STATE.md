# SENPAI Research State

- **Date**: 2026-05-15 19:40
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 2 active (compounds + variants)
- **Most recent human research directive**: None received
- **Pending verification:** PR #3340 (thorfinn H9 WSD, val_avg=89.04) — sent back for rebase + verify. If holds, becomes new best.

## Current Best

**PR #3335 (Huber δ=0.5 + T_max=15, no FiLM) — val_avg/mae_surf_p = 94.6764** (merged 2026-05-15)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| Huber δ=0.5 + T_max=15 (no FiLM) | **94.68** | Current best (PR #3335) |
| Huber δ=0.5 (no FiLM, T_max=50 broken) | 112.84 | Previous best (PR #3160) |
| FiLM (MSE, T_max=50 broken) | 114.63 | PR #3166 |
| T_max=15 alone (no Huber, no FiLM) | 114.19 | PR #3284 Arm B |
| Clean baseline (slice_num=96) | 149.27 | PR #3168 |

H15 super-linear stacking: −18.16 pts vs previous best. Naive additive prediction was ~101.1 (actual 94.68, 6.5 pts better than additive).

## Key Confirmed Insights

1. **T_max mismatch was the dominant bottleneck**: CosineAnnealingLR(T_max=50) with 30-min cap → ~14 epochs → LR never anneals. T_max=15 fix alone gave 11.7-pt improvement.
2. **Huber δ=0.5 + T_max=15 super-linearly stacks (PR #3335)**: val_avg 94.68 vs naive additive ~101.1. Mechanism: properly annealed schedule enables Huber's tail-damping to compound with stable low-LR refinement.
3. **FiLM conditioning is effective**: raw Transolver ~149 → FiLM 114.63 (~35 pt improvement, PR #3166). **FiLM was NOT in the current best run** (cond_dim=0). Adding FiLM on top of (Huber+T_max=15) is the highest priority open question.
4. **Huber δ trend monotone**: δ=1.0 → 115.99, δ=0.5 → 112.84. With the schedule fix, δ=0.25 may push further.
5. **Huber wins on 3/4 splits, loses on val_geom_camber_rc**: Right-tail damping hurts the hardest extreme-Re split. Per-channel Huber (H17, PR #3343) addresses this.
6. **Surface dual-head without FiLM is dead (H13, PR #3297)**: 130.54. Follow-up H16 (FiLM+surface head) assigned to askeladd (PR #3338).
7. **p-channel upweighting regresses (H1, PR #3156)**: Both x3 and x5 worse than baseline.
8. **Grad-clip + warmup dead end (H3, PR #3163)**: 5-epoch warmup burns 36% of budget. Clip=1.0 alone being tested (H18, PR #3349).
9. **Deeper layers stress budget (H11, PR #3170)**: n_layers=7/8 completes fewer epochs — net negative.
10. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 non-finite GT. All test_avg NaN. Workaround: 3-split test_avg excl. cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3338 | askeladd | H16: FiLM + Surface Head compound (depth=2/3) | Active WIP |
| #3339 | tanjiro | H8: Per-sample adaptive loss normalization | Active WIP |
| #3340 | thorfinn | H9: WSD schedule + AdamW beta2=0.98 | **Sent back for rebase** — pending val_avg=89.04 result needs verify on rebased branch |
| #3341 | alphonse | H5b: Wider model n_hidden=256 matched-budget | Active WIP |
| #3342 | edward | H2b: EMA weight averaging (decay=0.999) | Active WIP |
| #3343 | fern | H17: Per-channel adaptive Huber (δ_p=0.25 vs δ_Ux/Uy=0.5) | Active WIP |
| #3349 | frieren | H18: Gradient clipping alone (clip=1.0, no warmup) + FiLM+Huber | Active WIP |
| #3408 | nezuko | H19: FiLM + Huber δ=0.5 + T_max=15 triple compound | Just assigned 2026-05-15 18:40 |

**Note:** All 7 active PRs use cond_dim=11 (FiLM on) + Huber δ=0.5 as the merged default, **plus T_max=15** (merged from PR #3335). This means every future run benefits from the schedule fix automatically.

## Key Open Questions

1. **FiLM + Huber + T_max=15 triple compound** — highest priority. The current best (94.68) ran with FiLM off. Re-enabling FiLM on top of (Huber+T_max=15) should push further below 94. Need a clean single-arm test.
2. **Does δ=0.25 push Huber further with fixed schedule?** Monotone δ=1.0→0.5 with T_max=50 broken. With T_max=15, the correct annealing may reveal whether smaller δ continues to win.
3. **Does WSD schedule beat cosine T_max=15 (PR #3340)?** WSD provides a controlled stable plateau then decay. Key comparison: WSD vs cosine T_max=15.
4. **Does FiLM context enable surface head specialization (PR #3338)?** H13 failed without FiLM; H16 adds FiLM as enabling context.
5. **Can per-sample normalization fix Re-range gradient imbalance (PR #3339)?** y_std varies 50-2077 Pa (40x).
6. **Does grad-clip alone (no warmup) help (PR #3349)?** Low-risk one-line change.
7. **Does val_single_in_dist ceiling (112.48 in H15) have a root cause?** Far above other splits even after compound.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all models. File is read-only. Report 3-split test_avg excl. cruise as workaround.

## Highest-Priority Next Hypothesis

**H19: FiLM + Huber δ=0.5 + T_max=15 triple compound** — Assigned to nezuko (PR #3408). Run the merged default with only `--huber_delta 0.5` (FiLM on by default, T_max=15 hardcoded). This completes the triple-compound picture: Baseline PR #3335 ran with FiLM off deliberately; H19 adds FiLM back.

## Potential Next Research Directions

- **FiLM triple compound (H19)**: Immediate highest priority — add FiLM back to the new best config.
- **Huber δ=0.25 with T_max=15**: δ trend was monotone with broken schedule; with fixed T_max=15, test if smaller δ continues to improve.
- **LR sweep**: With T_max=15, the effective annealing changes the optimal peak LR. lr=5e-4 was tuned for T_max=50. Try lr=8e-4 or 1e-3.
- **T_max tuning**: T_max=15 matched ~14 wall epochs. If wall budget is consistent, T_max=12 or 13 may fully anneal; T_max=10 may be too aggressive.
- **Graph-based positional encoding**: Geodesic distances along foil surface.
- **WSD schedule variant**: WSD results pending from thorfinn PR #3340.
- **Per-channel Huber**: Results pending from fern PR #3343.
- **Spectral/Fourier features**: Add Fourier features of (x,z) for high-freq pressure gradient capture.
- **Post-hoc test metric recompute**: val_avg below 94 warrants a clean 3-split test_avg report.
