# SENPAI Research State

- **Updated:** 2026-05-15 23:05
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (multiple students have independently confirmed the root cause: inf×0=NaN in the surf_mask-zeroed pressure sum during evaluation).

## Current baseline

**`val_avg/mae_surf_p = 90.04`** — L1 surface loss + CosineAnnealingWarmRestarts(T_0=5, T_mult=2) + grad_clip max_norm=1.0, PR #3434, **merged** (2026-05-15 21:30)

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 108.95 |
| val_geom_camber_rc | 97.70 |
| val_geom_camber_cruise | 70.40 |
| val_re_rand | 83.11 |
| **val_avg** | **90.04** |

test 3-split (excl. cruise) = **87.78** | W&B run: `tcci4fzk`

## Winners pending confirmation

- **#3307 askeladd OneCycleLR right-sized → 97.44 (OLD baseline, single arm)**: Now branching from L1 baseline. Askeladd sent back to rebase (keep OneCycleLR, keep L1 surf, drop warm-restarts) and run 2 replication arms. Target: 3-arm mean < 90.04 to merge as compound winner.

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3434 edward | L1 surface loss (vol MSE + surf L1) | **90.04** | −8.94% ✓ **MERGED** |
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | **98.88** | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 3 portfolio (active experiments)

| Student | PR | Change | Rationale |
|---|---|---|---|
| nezuko | #3512 | Higher nominal LR: 5e-4 → 1e-3 (3-arm) | Grad clip normalizes step magnitude; doubling lr doubles effective LR from 1.1e-5 to 2.2e-5 while preserving warm-restart structure |
| askeladd | #3307 | OneCycleLR right-sized + rebase to L1 baseline | Winner pending — needs rebase + 2 replication arms against 90.04 |
| tanjiro | #3360 | grad clip max_norm=0.5 on warm-restarts+L1 baseline | Rebase + 3-arm retest on new baseline (confirmed on old 117.16 baseline) |
| fern | #3462 | surf_weight 10 → 5 (multi-arm) | Test if sweet spot is below 10; labels now fixed |
| frieren | #3464 | slice_num 64 → 32 (multi-arm) | Counter-probe to closed #3146; labels now fixed |
| alphonse | #3487 | weight_decay 1e-4 → 0 (multi-arm) | Test if wd is pure penalty in underfit regime |
| edward | #3488 | Full L1 both vol+surf loss | Extend #3434 win: test if vol L1 also helps |
| thorfinn | #3489 | Huber surf loss (delta={1.0, 0.5}) | Between L1 and L2: does smooth-near-zero help fine-tuning? |

## Round 3 closed (before completion of main portfolio)

| PR | Change | Best val_avg | Δ | Decision |
|---|---|---|---|---|
| #3431 nezuko | EMA weights (decay=0.999) | 102.63 (best of 3) | +14.0% | closed — EMA incompatible with warm-restarts; ema metric worse than raw model |
| #3436 alphonse | T_0=3 warm-restarts | 116.32 (1 arm) | +17.6% | closed — too many restart penalties for 14-epoch budget |
| #3416 thorfinn | p×3 per-channel weighting | 118.74 (best of 3) | +32.0% | closed — same mechanism as surf_weight=25 failure |

## Key research findings

1. **Loss formulation is the dominant active lever** — L1 surf loss gave 8.94% improvement (#3434 merged). Mechanism: L1 = median estimator, MAE-optimal; L2 = mean estimator, chases heavy-tailed OOD outliers. With grad clip normalizing step sizes, convergence dynamics are similar; the minimum reached is better with L1.
2. **Schedule is the second major lever** — warm-restarts gave 15.6% on top of grad-clip baseline; T_0=5 is near-optimal (T_0=3 regresses, more restarts = more recovery penalty per budget). OneCycle right-sized pending. EMA (decay=0.999) is incompatible with warm-restarts — the averaging window spans multiple restart cycles and the EMA metric was worse than the raw model.
3. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; median pre-clip norm ~45 → effective LR ~1.1e-5. Tighter (0.5) improved on old baseline (-4.33 pp); testing on new baseline.
4. **Model is NOT capacity-limited** — width, depth, and slice count increases all fail in this budget.
5. **Loss weighting and channel-weighting hurt** — surf_weight=25 and p×3 both regress. Volume term provides inductive structure that helps surface generalization. Don't destabilize the loss balance.
6. **Restart frequency: T_0=5 is near-optimal** — T_0=3 costs 17.6% regression (restart penalties dominate 14-epoch budget). T_0=7 not tested but theoretical argument: one restart at epoch 7 gives less exploration than two restarts at 5, 10.
7. **Run variance is lever-dependent** — base 15-pp spread, warm-restarts reduced to ~3 pp, surf_weight=25 blew to 66 pp. L1 single-arm at 8.84 pp improvement is noise-floor robust (even assuming ±3 pp spread, still >5 pp gain).

## Potential next research directions (round 4+)

1. **Compound stack** — if round-3 experiments win, compound: L1 + OneCycle, L1 + clip=0.5, L1 + wd=0. Stack orthogonal wins.
2. **FiLM conditioning of global features** — Re, AoA, NACA codes encoded once via small MLP → (γ, β) → modulate Transolver block activations. Adds multiplicative interactions that additive concat can't represent. ~few hundred extra params.
3. **Higher nominal LR** — given effective LR = lr / grad_norm ≈ 5e-4 / 45 = 1.1e-5, try lr=1e-3 to get eff_LR = 2.2e-5 without changing clip. Simple 1-line change.
4. **Coordinate features** — random Fourier features `[sin(2^k π x), cos(2^k π x)]` for k∈{0..5} to improve high-frequency spatial representation near airfoil surface. Could help val_single_in_dist (currently highest split at 108.95).
5. **SWA (Stochastic Weight Averaging)** — collect weights from final K epochs, average uniformly. Different from EMA (nezuko). Wider flat minimum → better generalization.
6. **Smaller batch_size=2** — 2× more gradient updates per epoch at same wall-clock; batch variance increases but grad clip handles it; warm-restarts per-batch scheduler timing preserves.
7. **Loss re-balancing after full L1** — if edward's vol+surf L1 changes the loss scale, surf_weight may need re-tuning (surf_weight=7 or 15 on full-L1 baseline).
8. **Per-domain adaptive slice_num** — mesh sizes vary 3× across physical domains; fixed slice_num gives very different token/node ratios (frieren's follow-up #5).
9. **Log-space pressure loss** — surface pressure has wide dynamic range; log-space distributes gradient weight per decade rather than per magnitude unit. May help OOD camber splits.
