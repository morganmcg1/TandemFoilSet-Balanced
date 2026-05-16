# SENPAI Research State

- **Date**: 2026-05-16 00:45
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 3 active (R3 experiments: H20–H33)
- **Most recent human research directive**: None received

## Current Best

**PR #3445 (H20: Grad clip=1.0 on H19 triple compound, nezuko) — val_avg/mae_surf_p = 75.4955** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **Grad clip=1.0 + H19 (H20 Arm A)** | **75.4955** | **Current best (PR #3445)** |
| Per-channel Huber δ_vel=1.0/δ_p=0.25 (H25 Arm B) | 75.7713 | (PR #3450) |
| Per-channel Huber δ_vel=0.5/δ_p=0.25 (H25 Arm A) | 78.2286 | (PR #3450) |
| Uniform Huber δ=0.1 on H19 (H22 Arm B) | 78.8321 | (PR #3447) |
| Clip=0.5 on H19 (H20 Arm B) | 77.0687 | (PR #3445) |
| FiLM + Huber δ=0.5 + T_max=15 (H19) | 83.8136 | (PR #3408) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 73.1556 (H20 Arm A)

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck**: CosineAnnealingLR(T_max=50) with 30-min cap → ~14 epochs → LR never anneals. T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins over uniform (H25)**: δ_p=0.25 for pressure (heavy-tailed), δ_vel=1.0 for velocities (near-MSE) — decoupling channels targets the right statistical structure per field.
3. **Grad clip=1.0 is independently effective (H20)**: Clipping was active every step (pre-clip norm 5–17). Caps per-step update magnitude, preventing any single high-Re sample from dominating a training step.
4. **H20 and H25 use the SAME δ_p and δ_vel defaults**: H20's effective config is δ_vel=0.5/δ_p=0.25 (from merged defaults) + clip=1.0; H25 differs only in δ_vel=1.0 (no clip). The two wins are orthogonal: pressure clipping vs gradient clipping.
5. **Compound H25+H20 (H29) is the top priority**: Neither has the other's improvement — combining should yield ~74 or better.
6. **Averaging (EMA H24, SWA H28) fails at 14-epoch budget**: Model is in steep descent throughout; no converged basin to average. Both closed as dead ends.
7. **WSD fails on this budget (H21 confirmed H9)**: Decay phase never fires within 14 epochs. CosineAnnealingLR T_max=15 is the right schedule.
8. **cond_dim=3 beats cond_dim=11 (H26)**: Geometry tail dims zero out for single-foil samples (~50% of training), creating noise. cond_dim=3 (Re, AoA1, NACA1_camber) is the cleaner FiLM conditioning.
9. **surf_weight=5 beats 10 (H23)**: Lowering surface weight improves both surface AND volume metrics simultaneously — 10 is too high.
10. **lr=7e-4 beats lr=5e-4 on H19 (H27)**: Monotone LR trend confirmed; higher peak LR is better under T_max=15 cosine.
11. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 non-finite GT. Workaround: 3-split test_avg excl. cruise.

## Active R3/R4 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3549 | nezuko | **H29: Per-channel δ_vel=1.0/δ_p=0.25 + clip=1.0 (compound H25+H20)** | WIP — **HIGHEST PRIORITY** |
| #3551 | askeladd | H30: Clip sweep (2.0, 1.5) on H20 base | WIP |
| #3553 | fern | H31: δ_p push (0.1, 0.05) with clip=1.0 | WIP |
| #3557 | thorfinn | H32: LR sweep (1e-3, 8e-4) on H20 base | WIP |
| #3561 | edward | H33: n_hidden=192/256 on H20 base | WIP |
| #3452 | frieren | H27b: LR sweep (7e-4, 1e-3) + clip=1.0 rebase | WIP (rebase in progress) |
| #3448 | tanjiro | H23b: surf_weight=5/2 + clip=1.0 rebase | WIP (rebase in progress) |
| #3451 | alphonse | H26b: cond_dim=3/2 + clip=1.0 rebase | WIP (rebase in progress) |

**Note:** All new R4 PRs use cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25 (merged defaults), CosineAnnealingLR T_max=15 as the base. clip_grad_norm=1.0 is explicitly set in all new assignments.

## Key Open Questions (Post-H20)

1. **Does per-channel δ_vel=1.0 compound with clip=1.0?** H29 (nezuko) — critical, expected ~74 or better.
2. **Optimal clip threshold?** H30 (askeladd) — does clip=2.0 continue the monotone trend, or is 1.0 the optimum?
3. **How far can δ_p go?** H31 (fern) — δ_p=0.1 and δ_p=0.05; if it's nearly L1 at 0.05, does the loss destabilize?
4. **Higher LR with clip stability?** H32 (thorfinn) + H27b (frieren) — clip=1.0 may allow lr=1e-3 without instability; H27b and H32 cover complementary ranges.
5. **Architecture headroom on tuned base?** H33 (edward) — n_hidden=192/256 on H20 base. H5 (R1) was inconclusive on untuned stack.
6. **cond_dim=3 with clip?** H26b (alphonse) — if cond_dim=3 compounds with clip, the combined FiLM+conditioning improvement is an additive win.
7. **surf_weight=5/2 with clip?** H23b (tanjiro) — if lighter surface weighting helps on H20 base, may reveal gradient balance insight.
8. **val_single_in_dist ceiling (~85.7 at H20)**: Still the worst split. Hypothesis: in-dist samples have high variance (single foil, diverse Re/AoA), model under-regularized for these.

## Potential Next Research Directions (After Active PRs Close)

- **Compound triple: δ_vel=1.0 + δ_p=0.1 + clip=1.0**: If H29 and H31 both win, compound all three.
- **Per-sample adaptive loss weighting**: Reweight training samples by recent loss history to focus on hard samples (curriculum).
- **Pressure-only auxiliary head**: Separate decoder branch with sharper Huber δ_p for surface pressure.
- **Non-contiguous FiLM conditioning**: Try indices (Re, AoA1, AoA2) vs contiguous slice — alphonse's H26 suggested non-contiguous could test different structural hypothesis.
- **Per-foil FiLM heads**: Single vs tandem mode switch to handle distribution mismatch in single-foil samples.
- **Graph-based positional encoding**: Geodesic distances along foil surface for better surface-local representation.
- **SWA with late start (start_epoch=11+) if budget allows**.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT. File is read-only. Report 3-split test_avg excl. cruise as workaround.
- train.py: `huber_delta` Config field is present but NOT used in loss computation — loss always uses `huber_delta_vel` and `huber_delta_p`. Students passing `--huber_delta` have no effect on the loss.
