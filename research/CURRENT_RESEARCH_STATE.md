# SENPAI Research State

- **Date**: 2026-05-15 20:05
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 3 active (R3 experiments: H20–H27)
- **Most recent human research directive**: None received
- **Pending verification:** PR #3445 (nezuko H20 grad clip, expected val_avg ~74.23 based on H18B signal from frieren) — highest priority when results arrive

## Current Best

**PR #3408 (H19: FiLM + Huber δ=0.5 + T_max=15 triple compound, nezuko) — val_avg/mae_surf_p = 83.8136** (merged 2026-05-15)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| FiLM + Huber δ=0.5 + T_max=15 (H19, triple compound) | **83.81** | Current best (PR #3408) |
| Huber δ=0.5 + T_max=15 (no FiLM, H15) | 94.68 | Previous best (PR #3335) |
| Huber δ=0.5 (no FiLM, T_max=50 broken) | 112.84 | (PR #3160) |
| FiLM (MSE, T_max=50 broken) | 114.63 | (PR #3166) |
| T_max=15 alone (no Huber, no FiLM) | 114.19 | (PR #3284 Arm B) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 80.2415

## Key Confirmed Insights

1. **T_max mismatch was the dominant bottleneck**: CosineAnnealingLR(T_max=50) with 30-min cap → ~14 epochs → LR never anneals. T_max=15 fix alone gave 11.7-pt improvement.
2. **Triple compound super-linearly stacks (PR #3408)**: val_avg 83.81 vs Huber+T_max=15 alone (94.68). FiLM adds ~10.9 pts on top of the already-compounded H15. All three interact multiplicatively.
3. **FiLM conditioning is effective**: Reduces cross-regime variance by conditioning on Re/AoA per sample. cond_dim=11 is now the merged default.
4. **Huber δ=0.5 wins consistently**: Clips tail gradients from high-Re samples. δ trend monotone (1.0→0.5). With fixed T_max=15, δ=0.25 may push further.
5. **val_single_in_dist ceiling**: Still 96.44 (H19) — far above other splits. Root cause TBD.
6. **Grad clip signal from H18B**: frieren's Arm B (clip=1.0, FiLM+Huber+T_max=15) yielded val_avg ~74.23 — would beat 83.81 by ~10 pts. PR #3349 sent back for rebase; nezuko's H20 (PR #3445) tests this next.
7. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 non-finite GT. All test_avg NaN. Workaround: 3-split test_avg excl. cruise.

## Active R3 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3445 | nezuko | H20: Grad clip=1.0 + FiLM+Huber+T_max=15 (from H18B signal) | **Highest priority** — awaiting results |
| (others) | various | H21–H27: Various R3 hypotheses | Active WIP |
| #3343 | fern | H17: Per-channel adaptive Huber (δ_p=0.25 vs δ_Ux/Uy=0.5) | **Sent back for rebase** — must rebase onto current branch and re-run |
| #3349 | frieren | H18: Grad clip=1.0 + FiLM+Huber — Arm B (the win arm) | **Sent back for rebase** — superseded by nezuko H20 |

**Note:** All R3 PRs use cond_dim=11 (FiLM on) + Huber δ=0.5 + T_max=15 as the merged defaults.

## Key Open Questions (R3 Focus)

1. **Does grad clip=1.0 on top of FiLM+Huber+T_max=15 yield ~74 val_avg?** H18B signal from frieren's rebase run was ~74.23 — H20 (PR #3445, nezuko) is testing this.
2. **Does δ=0.25 push Huber further with fixed schedule?** Monotone δ trend; with T_max=15, smaller δ may continue improving.
3. **Does per-channel Huber help (H17)?** δ_p=0.25 vs δ_Ux/Uy=0.5 — targeted loss for the pressure channel. PR #3343 sent back for rebase.
4. **val_single_in_dist ceiling**: Why is in-dist error so high (96.44 at H19 best)? This split should be easiest but is the worst.
5. **WSD schedule vs cosine T_max=15**: H9 (PR #3340) WSD signal was val_avg=89.04. On rebased branch with triple-compound defaults, does WSD provide additional benefit?
6. **LR sweep with proper T_max=15**: lr=5e-4 was tuned for T_max=50. With T_max=15, optimal peak LR may be higher (lr=8e-4 or 1e-3).

## Potential Next Research Directions (After R3 Closes)

- **Grad clip sweep**: If clip=1.0 works, try clip=0.5 and clip=2.0.
- **Huber δ=0.25**: Monotone δ trend suggests smaller δ may continue improving on the triple-compound base.
- **LR sweep**: lr ∈ {5e-4 (current), 8e-4, 1e-3} with T_max=15 triple-compound base.
- **T_max tuning**: T_max=15 matched ~14 wall epochs. T_max=12 may fully anneal sooner; T_max=10 aggressive.
- **Spectral/Fourier features**: Add Fourier features of (x,z) for high-freq pressure gradient capture near foil surfaces.
- **WSD schedule variant**: Controlled stable plateau then decay — compare vs cosine T_max=15 on triple-compound base.
- **Per-channel loss weighting**: Weight pressure channel loss more heavily given it drives the primary metric.
- **Graph-based positional encoding**: Geodesic distances along foil surface for better surface pressure modeling.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all models. File is read-only. Report 3-split test_avg excl. cruise as workaround.
