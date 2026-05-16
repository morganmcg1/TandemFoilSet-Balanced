# SENPAI Research State

- **Date**: 2026-05-16 02:50
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 3/4 active (R3 base: H20–H34; R4 new assignments: H35–H37)
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
4. **H20 and H25 use the SAME δ_p and δ_vel defaults**: H20's effective config is δ_vel=0.5/δ_p=0.25 (from merged defaults) + clip=1.0; H25 differs only in δ_vel=1.0 (no clip).
5. **Gradient-magnitude-shaping interventions don't compound with clip=1.0 (H29, H23b, H30 pattern)**: Three independent confirmations:
   - H29: per-channel δ_vel=1.0 + clip=1.0 → global rescale defeats per-channel benefit (76.80)
   - H23b: surf_weight=5 + clip=1.0 → clip already controls step magnitude; sw reduction redundant (75.91)
   - H30: clip=2.0/1.5 → grad norms stay above thresholds; still "active active clipping" (75.68/76.55)
   **Principle:** Future high-value experiments should target **directions** (attention structure, conditioning, representation budget), not magnitude (loss weights, clip thresholds, surf_weight).
6. **Averaging (EMA H24, SWA H28) fails at 14-epoch budget**: Model is in steep descent throughout; no converged basin to average. Both closed as dead ends.
7. **WSD fails on this budget (H21 confirmed H9)**: Decay phase never fires within 14 epochs. CosineAnnealingLR T_max=15 is the right schedule.
8. **cond_dim=3 beats cond_dim=11 (H26)**: Geometry tail dims zero out for single-foil samples (~50% of training), creating noise. cond_dim=3 (Re, AoA1, NACA1_camber) is the cleaner FiLM conditioning.
9. **surf_weight=5 beats 10 on H19 base (H23) but NOT on H20 base (H23b)**: On unclipped H19, sw=5 was implicitly correcting gradient magnitude. On H20 with clip=1.0, sw=10 is optimal — clip already controls step magnitude.
10. **lr=7e-4 beats lr=5e-4 on H19 (H27)**: Monotone LR trend confirmed; higher peak LR is better under T_max=15 cosine.
11. **Architecture headroom: n_hidden capacity is NOT the bottleneck (H33)**: n_hidden=192/256 both regress (+15%/+22.6%). The model can't use extra capacity in 14 epochs. Architecture changes must use the same parameter budget or focus on structure, not size.
12. **Scoring NaN bug confirmed**: `test_geom_camber_cruise` sample 20 non-finite GT. Workaround: 3-split test_avg excl. cruise.

## Active R3/R4 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3587 | nezuko | **H34: Element-wise clip (clip_grad_value) + per-channel Huber** | WIP — **highest priority (tests H29 mechanism)** |
| #3557 | thorfinn | H32: LR sweep (1e-3, 8e-4) on H20 base | WIP |
| #3553 | fern | H31: δ_p push (0.1, 0.05) with clip=1.0 | WIP |
| #3452 | frieren | H27b: LR sweep (7e-4, 1e-3) + clip=1.0 rebase | WIP (rebase in progress) |
| #3451 | alphonse | H26b: cond_dim=3/2 + clip=1.0 rebase | WIP (rebase in progress) |
| #3623 | edward | H35: slice_num sweep (96, 128) on H20 base | NEW — architectural representation budget |
| #3626 | askeladd | H36: AdamW beta2 sweep (0.95, 0.999) on H20 base | NEW — optimizer momentum tuning |
| #3629 | tanjiro | H37: n_head sweep (8, 2) on H20 base | NEW — attention head granularity |

**Note:** All new R4 PRs use cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25 (merged defaults), CosineAnnealingLR T_max=15, clip_grad_norm=1.0 as the base.

## Key Open Questions (Post-H20)

1. ~~Does per-channel δ_vel=1.0 compound with clip=1.0?~~ **ANSWERED: NO** — H29 closed. Global clip rescale defeats per-channel benefit.
2. **Does element-wise clip preserve per-channel benefit?** H34 (nezuko) — directly tests the H29 mechanism. Highest priority.
3. ~~Optimal global-norm clip threshold?~~ **H30 CLOSED** — null result. Budget asymmetry confounded; clip=1.0 remains optimal.
4. **How far can δ_p go?** H31 (fern) — δ_p=0.1 and δ_p=0.05.
5. **Higher LR with clip stability?** H32 (thorfinn) + H27b (frieren) — covers complementary LR ranges.
6. **Physical representation budget (slice_num)?** H35 (edward) — slice_num=96/128 on tuned H20 stack. H10 on broken stack was uninterpretable.
7. **Optimizer second-moment decay?** H36 (askeladd) — beta2=0.95 may be better calibrated for 14-epoch runs than 0.999.
8. **Attention head granularity?** H37 (tanjiro) — n_head=8 (more specialization) vs n_head=2 (richer per-head).
9. **cond_dim=3 with clip?** H26b (alphonse) — if cond_dim=3 compounds with clip, combined FiLM+conditioning improvement is additive.
10. **val_single_in_dist ceiling (~85.7 at H20)**: Still the worst split. Single-foil diversity suggests model under-regularized for in-distribution samples.

## Guiding Principle (Post-H23b/H29/H30)

**Clip absorbs magnitude-shaping interventions.** Anything that reshapes the gradient magnitude profile (surf_weight, per-channel Huber deltas as amplifiers, looser clip threshold) stops compounding once `clip_grad_norm=1.0` is in the stack. Future high-value experiments should target:
- **Structural/directional changes**: attention head count (H37), physical representation budget (H35), conditioning architecture (H26b)
- **Optimizer dynamics** beyond step magnitude: momentum decay (H36), LR schedule shape (H32, H27b)
- **Loss semantics**: pushing δ_p further toward L1 (H31), element-wise clip that preserves per-channel ratio (H34)

## Potential Next Research Directions (After Active PRs Close)

- **Element-wise clip compound**: If H34 confirms mechanism, compound δ_vel=1.0 + clip_grad_value=1.0 as the new default.
- **LR warmup on H20**: H3 failed on broken stack; short linear warmup (1-2 ep) before cosine may help with high clip threshold.
- **FiLM head decoupling**: separate LR group for FiLM scale/shift parameters (they have different gradient structure from backbone).
- **Non-contiguous FiLM conditioning**: indices (Re, AoA1, AoA2) rather than contiguous slice — different structural hypothesis.
- **Per-foil FiLM heads**: single vs tandem mode routing to address single-foil distribution mismatch.
- **Graph-based positional encoding**: geodesic distances along foil surface for better surface-local representation.
- **AdamW eps sweep**: eps=1e-6 vs 1e-8 to moderate per-param LR scaling (less extreme in early training).
- **SWA with late start (start_epoch=11+) if budget allows**.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT. File is read-only. Report 3-split test_avg excl. cruise as workaround.
- `train.py`: `huber_delta` Config field present but NOT used in loss computation — loss always uses `huber_delta_vel` and `huber_delta_p`. Students passing `--huber_delta` have no effect on the loss (but the flag is preserved for config.yaml traceability).
