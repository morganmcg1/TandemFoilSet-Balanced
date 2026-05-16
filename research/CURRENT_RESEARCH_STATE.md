# SENPAI Research State

- **Updated:** 2026-05-16 05:00
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received in current heartbeat cycle. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (multiple students have independently confirmed: inf×0=NaN in the surf_mask-zeroed pressure sum during evaluation).

## Current baseline

**`val_avg/mae_surf_p = 81.66`** (3-arm mean) — L1 surface loss + OneCycleLR right-sized to actual budget + grad_clip max_norm=1.0, PR #3307, **merged** (2026-05-16 01:35)

| Split | val mae_surf_p (best arm `iomzoqit`) | 3-arm mean |
|---|---|---|
| val_single_in_dist | 92.04 | 93.33 |
| val_geom_camber_rc | 92.30 | 92.67 |
| val_geom_camber_cruise | 60.31 | 61.87 |
| val_re_rand | 76.60 | 78.79 |
| **val_avg** | **80.31** | **81.66** |

test 3-split (excl. cruise) = **77.97** (best arm) / **79.28** (mean) | W&B best run: `iomzoqit`

Head config: `OneCycleLR(max_lr=1e-3, total_steps=len(train_loader)*14, pct_start=0.1, div_factor=25, final_div_factor=1e4)` + per-batch step + `if global_step < scheduler.total_steps` guard + L1 surf loss + grad_clip max_norm=1.0

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3307 askeladd | OneCycleLR right-sized + L1 surf (compound) | **81.66** (mean) / 80.31 (best) | −9.30% ✓ **MERGED** |
| #3434 edward | L1 surface loss (vol MSE + surf L1) | **90.04** | −8.94% ✓ **MERGED** |
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | **98.88** | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 4 active portfolio (8 students, all GPUs assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3613 | askeladd | OneCycleLR `pct_start=0.05` (faster warmup, longer anneal) | Running — 2 arm1 attempts both regress (84.59, 85.94); arm running |
| #3614 | thorfinn | OneCycleLR **max_lr=1.5e-3** (midpoint retest after 2e-3 regressed) | Sent back; 3 new arms incoming |
| #3720 | nezuko | **Lion optimizer (sign-based momentum)** — 3-arm max_lr sweep | New (replaces #3615 SWA close) |
| #3616 | fern | `batch_size=2` (2× gradient updates/epoch) | Running — arm1 attempt 1 hit **77.98** (potential round winner); follow-on arms running |
| #3617 | edward | log-space L1 surface pressure loss | Running — arm1 attempt 1 hit **78.13** (potential round winner); high variance attempt 2 = 82.78 |
| #3619 | alphonse | weight_decay=0 retest on OneCycle baseline | Running |
| #3696 | frieren | OneCycleLR **max_lr=5e-4** (low end of LR sweep) | New (after #3622 close) |
| #3699 | tanjiro | Lookahead(AdamW, k=5, α=0.5) optimizer wrapper | New (after #3360 close) |

## Round 4 closed/sent-back so far

| PR | Student | Change | Decision |
|---|---|---|---|
| #3615 | nezuko | SWA on final 3-4 OneCycle epochs | closed — SWA mean 85.09 (+4.2%); second weight-averaging failure |
| #3622 | frieren | final_div_factor ∈ {1e3, 1e5} | closed — locks 1e4 (both sides regress ~0.2 pp) |
| #3614 | thorfinn | max_lr=2e-3 | **sent back** for max_lr=1.5e-3 retest (val regresses but test improves) |
| #3360 | tanjiro | grad_clip=0.5 retest on OneCycle | closed — schedule subsumes clip-strength (+1.94% regression mean) |
| #3464 | frieren | slice_num=32 (corrected close) | closed — beats old baseline but regresses on OneCycle (+15.7%) |

## Round 3 closed

| PR | Student | Change | Decision |
|---|---|---|---|
| #3512 | nezuko | lr=1e-3 warm-restarts | closed — obsolete config (warm-restarts superseded by OneCycle) |
| #3489 | thorfinn | Huber surf (delta={1.0, 0.5}) | closed — pure L1 wins; grad clip neutralizes Huber advantage |
| #3488 | edward | Full L1 vol+surf | closed — vol MSE provides stronger far-field structure |
| #3487 | alphonse | weight_decay=0 (on warm-restarts baseline) | closed THERE; **reassigned for OneCycle retest** (#3619) |
| #3462 | fern | surf_weight 10 → 5 | closed — 10 is right balance |
| #3431 | nezuko | EMA weights (decay=0.999) | closed — incompatible with warm-restarts cycles |
| #3436 | alphonse | T_0=3 warm-restarts | closed — too many restart penalties |
| #3416 | thorfinn | p×3 per-channel weighting | closed — same as surf_weight=25 failure |

## Key research findings

1. **Loss formulation is the dominant active lever** — L1 surf loss gave 8.94% improvement (#3434 merged). Mechanism: L1 = median estimator, MAE-optimal; L2 = mean estimator, chases heavy-tailed OOD outliers. With grad clip normalizing step sizes, convergence dynamics are similar; the minimum reached is better with L1. Huber (delta=1.0, 0.5) is WORSE than pure L1 because grad clip already handles gradient-magnitude normalization.
2. **OneCycleLR right-sized to budget is the dominant schedule** — OneCycle + L1 compounded for −9.30% (81.66 vs 90.04). Right-sizing `total_steps` to the actual wall-clock budget is critical. OneCycle's aggressive anneal in the final epochs is responsible for the gain. Warm-restarts gave 15.6% on top of grad-clip baseline but is now superseded by OneCycleLR.
3. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; median pre-clip norm ~45 → effective LR ~1.1e-5. Tighter (0.5) gave +1.33% on old warm-restarts baseline; retesting on OneCycle baseline (#3360).
4. **Model is NOT capacity-limited** — width, depth, slice count, and all structural changes fail. slice_num=64 is the floor.
5. **Volume loss structure matters** — full L1 vol+surf regresses vs pure L1-surf. MSE vol loss provides stronger far-field consistency gradient. Locked in: vol MSE + surf L1.
6. **Loss weighting: surf_weight=10 is optimal** — surf_weight=5 regresses by 10.8%; surf_weight=25 regresses even more.
7. **Weight averaging is incompatible with our setup, regardless of schedule** — EMA failed on warm-restarts (#3431), SWA failed on OneCycle (#3615). OneCycle's monotonic anneal settles into a single sharp basin; averaging blurs the cruise-OOD specialization in the final epochs rather than widening it. Don't retry averaging-based methods on this baseline.

## Potential next research directions (round 5+ backlog)

### If a round-4 hypothesis wins and locks in
- **Follow-ups to winners** — extend dose-response (e.g., if `max_lr=2e-3` wins → try 3e-3 and 1.5e-3; if `pct_start=0.05` wins → try 0.025; if SWA wins → try wider averaging window).

### If round 4 plateaus
- **FiLM conditioning** — Re, AoA, NACA codes → (γ, β) → modulate Transolver blocks. Multiplicative interactions additive concat can't represent.
- **Coordinate features** — random Fourier features for spatial encoding near airfoil surface.
- **Adam → AdamW or LION** — different optimizer family entirely.
- **Augmentation** — spatial transforms on point clouds (small jitter, scale).
- **Knowledge distillation from a long-trained teacher** — but the budget is the constraint.
- **Architecture pivot** — try a new model entirely (e.g., GINO, FNO, or PointTransformer-style head). Mode change per the Plateau Protocol.

### Frontier diagnostics (could run anytime)
- **OOD error analysis** — read worst predictions in `val_geom_camber_rc` and `val_re_rand`; what's the residual pattern? Pressure peak misalignment vs amplitude error?
- **W&B GH #3292 fix** — `test_geom_camber_cruise` NaN root cause; would unlock paper-facing 4-split test metric. Could assign as utility PR once a student frees up.
