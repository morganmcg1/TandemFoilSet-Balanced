# SENPAI Research State

- **Date**: 2026-05-15 23:50
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 3 active (R3 experiments: H20–H28)
- **Most recent human research directive**: None received

## Current Best

**PR #3450 (H25: Per-channel Huber δ_vel=1.0, δ_p=0.25 on H19 stack, askeladd) — val_avg/mae_surf_p = 75.7713** (merged 2026-05-15)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **Per-channel Huber (δ_vel=1.0, δ_p=0.25) on H19 (H25 Arm B)** | **75.77** | **Current best (PR #3450)** |
| Per-channel Huber (δ_vel=0.5, δ_p=0.25) on H19 (H25 Arm A) | 78.23 | (PR #3450) |
| Uniform Huber δ=0.1 on H19 (H22 winning arm) | 78.83 | (PR #3447, merged) |
| Uniform Huber δ=0.25 on H19 (H22 second arm) | 79.15 | (PR #3447, merged) |
| FiLM + Huber δ=0.5 + T_max=15 (H19) | 83.81 | Previous best (PR #3408) |
| Huber δ=0.5 + T_max=15 (no FiLM, H15) | 94.68 | (PR #3335) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 73.0704 (H25 Arm B)

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck**: T_max=15 fix alone gave 11.7-pt improvement (114→102 etc.). Cosine T_max=epochs=50 never anneals at 30-min cap.
2. **Per-channel Huber decouples pressure clipping (H25)**: δ_p=0.25 (aggressive clip on heavy-tailed pressure) + δ_vel=1.0 (near-MSE on Gaussian velocities) yields 9.6% improvement over uniform-Huber H19. Pressure outliers were dominating optimization.
3. **Triple compound (H19) was the prior win**: FiLM + Huber δ=0.5 + T_max=15. FiLM adds 10.9 pts on top of compounded H15. H25 then improved on this by decoupling Huber per channel.
4. **Huber δ trend is monotone down to 0.1 for uniform**: H22 confirmed — δ=0.1 (78.83) better than δ=0.25 (79.15) when applied uniformly. But per-channel asymmetric δ wins outright.
5. **FiLM conditioning is effective**: cond_dim=11 merged default. Reduces cross-regime variance via Re/AoA conditioning.
6. **EMA at d=0.999 fails at 4,875-step budget (H24, closed)**: shadow weights lag a still-rapidly-improving live model. Pivoted to SWA (H28).
7. **val_single_in_dist remains the ceiling split**: 86.55 at H25 — still 12+ pts above the val_geom_camber_cruise easy split. Root cause TBD.
8. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 non-finite GT. Workaround: 3-split test_avg excl. cruise.

## Active R3 WIP / Review-Ready

| PR | Student | Hypothesis | Status | val_avg (vs H25 75.77) |
|----|---------|------------|--------|------------------------|
| #3452 | frieren | H27: LR=7e-4 with T_max=15 base | **Review-ready** | ~79.79 — does not beat H25 |
| #3448 | tanjiro | H23: surf_weight=5 (vs default 10) | **Review-ready** | ~81.91 — does not beat H25 |
| #3446 | thorfinn | H21: WSD schedule on H19 stack | **Review-ready** | ~96.71 — dead end candidate |
| #3445 | nezuko | H20: Grad clip=1.0 + FiLM+Huber+T_max=15 | WIP | — |
| #3447 | (merged) | H22: uniform Huber δ sweep | Merged | 78.83 (won arm) |
| #3450 | (merged) | H25: per-channel Huber | **Merged — NEW BEST** | 75.77 |
| #3451 | ? | (stale label) | Investigate | — |
| #3491 | edward | H28: SWA on H19 stack | WIP | — |
| #3343 | fern | H17: Per-channel adaptive Huber (older variant) | **Sent back for rebase** — likely superseded by H25 | — |
| #3349 | frieren | H18: Grad clip+FiLM+Huber | **Sent back for rebase** — superseded by nezuko H20 | — |
| #3340 | thorfinn | H9: WSD schedule | **Sent back for rebase + verify** | 89.04 prelim |

**Note:** All R3 PRs use cond_dim=11 (FiLM on) + Huber δ=0.5 + T_max=15 as the merged defaults. H25's per-channel Huber is now the new base for follow-up experiments.

## Key Open Questions (Post-H25)

1. **Push per-channel Huber further**: δ_p=0.1 or even smaller for pressure on top of δ_vel=1.0? Monotone δ_p trend suggests room.
2. **Decouple per-channel for velocities asymmetrically**: δ_Ux=1.0/δ_Uy=0.5/δ_p=0.25, or even per-channel separately tuned.
3. **Grad clip on H25 base**: H18B signal (~74) from frieren on H19 — does grad clip=1.0 + per-channel Huber yield ~70?
4. **SWA on H25 base**: H28 (edward) tests SWA on H19 base; rebase onto H25 would be the right next step.
5. **LR sweep on H25 base**: H27 (frieren) tried lr=7e-4 on H19 base. Does lr ∈ {7e-4, 1e-3} beat lr=5e-4 on H25?
6. **surf_weight on H25 base**: H23 (tanjiro) tried surf_weight=5 on H19. Now per-channel Huber down-weights pressure gradient already — does surf_weight↓ help or hurt?
7. **val_single_in_dist ceiling (86.55 at H25)**: Why is in-dist the worst split? Suggests model has high variance on samples it should ace.
8. **WSD vs cosine T_max=15 on triple+per-channel base (H9 pending rebase)**.

## Potential Next Research Directions (After R3 Reviews Close)

- **Per-channel Huber refinement**: δ_p ∈ {0.1, 0.05}, δ_vel asymmetric, learned δ.
- **Combine grad clip + per-channel Huber + lr sweep**: triple-knob compound on H25 base.
- **Pressure-only auxiliary head**: separate decoder branch for surface pressure with sharper loss.
- **Architecture (deferred until simpler levers exhausted)**: spectral/Fourier features, graph-based positional encoding for surface, n_hidden/depth sweep with proper schedule.
- **Per-split weighted loss**: val_single_in_dist drives error — upweight in-distribution samples?
- **SWA tuned for short budget**: average final K=3 epochs only (skip noisy mid-training).

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all models. File is read-only. Report 3-split test_avg excl. cruise as workaround.
