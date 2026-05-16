# SENPAI Research State

- **Updated:** 2026-05-16 01:45
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (multiple students have independently confirmed the root cause: inf×0=NaN in the surf_mask-zeroed pressure sum during evaluation).

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

## Round 4 active portfolio (waiting for rate limit reset ~02:20 UTC to create PRs)

| Student | PR | Change | Status |
|---|---|---|---|
| tanjiro | #3360 | grad_clip max_norm=0.5 on **OneCycleLR** baseline (sent back for retest) | Rebasing + re-running 3 arms vs 81.66 baseline |
| alphonse | #3487 | weight_decay=0 (arm 3 still running; needs retest on OneCycle) | Waiting for arm 3, then send-back for retest |
| nezuko | #3512 | lr=1e-3 warm-restarts (obsolete config) | Close pending — fails even old baseline |
| edward | #3488 | Full L1 vol+surf | Close pending — 92-93 fails both baselines |
| frieren | #3464 | slice_num=32 | Close pending — 125-163 decisive dead end |
| askeladd | — | IDLE — pending new assignment (round 4) | Assign: SWA or max_lr=2e-3 probe |
| thorfinn | — | IDLE after #3489 close | Assign: round 4 experiment |
| fern | — | IDLE after #3462 close | Assign: round 4 experiment |

## Round 3 closed

| PR | Change | Best val_avg | Δ | Decision |
|---|---|---|---|---|
| #3489 thorfinn | Huber surf loss (delta={1.0, 0.5}) | 93.45 (d=1.0) | +3.79% | closed — pure L1 wins; grad clip neutralizes Huber's advantage |
| #3462 fern | surf_weight 10 → 5 | 99.60 (best) | +10.8% | closed — regresses vs 90.04 baseline; surf_weight=10 is better balance |
| #3431 nezuko | EMA weights (decay=0.999) | 102.63 (best of 3) | +14.0% | closed — EMA incompatible with warm-restarts; ema metric worse than raw model |
| #3436 alphonse | T_0=3 warm-restarts | 116.32 (1 arm) | +17.6% | closed — too many restart penalties for 14-epoch budget |
| #3416 thorfinn | p×3 per-channel weighting | 118.74 (best of 3) | +32.0% | closed — same mechanism as surf_weight=25 failure |

## Key research findings

1. **Loss formulation is the dominant active lever** — L1 surf loss gave 8.94% improvement (#3434 merged). Mechanism: L1 = median estimator, MAE-optimal; L2 = mean estimator, chases heavy-tailed OOD outliers. With grad clip normalizing step sizes, convergence dynamics are similar; the minimum reached is better with L1. Huber (delta=1.0, 0.5) is WORSE than pure L1 because grad clip already handles gradient-magnitude normalization.
2. **OneCycleLR right-sized to budget is the dominant schedule** — OneCycle + L1 compounded for −9.30% (81.66 vs 90.04). Right-sizing `total_steps` to the actual wall-clock budget is critical. OneCycle's aggressive anneal in the final epochs is responsible for the gain. Warm-restarts gave 15.6% on top of grad-clip baseline but is now superseded by OneCycleLR.
3. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; median pre-clip norm ~45 → effective LR ~1.1e-5. Tighter (0.5) gave +1.33% on old warm-restarts baseline; retesting on OneCycle baseline.
4. **Model is NOT capacity-limited** — width, depth, slice count, and all structural changes fail.
5. **Volume loss structure matters** — full L1 vol+surf (edward) regresses vs pure L1-surf (93 vs 90.04 old baseline). MSE vol loss provides stronger far-field consistency gradient. Do not switch vol loss to L1.
6. **Loss weighting: surf_weight=10 is optimal** — surf_weight=5 regresses by 10.8%; surf_weight=25 regresses even more. The balance between vol (structure) and surf (target metric) must be preserved.
7. **EMA is incompatible with warm-restarts / oscillatory schedules** — the averaging window spans restart cycles, mixing exploring and settled weights. OneCycle may be more EMA-friendly (monotonic anneal).

## Potential next research directions (round 4)

### On-deck / high priority
1. **Tanjiro #3360 retest** — does clip=0.5 stack with OneCycleLR? (retesting on new head)
2. **Alphonse #3487 retest** — does wd=0 stack with OneCycleLR? (wait for arm 3, then send back)
3. **max_lr=2e-3 for OneCycleLR** — askeladd suggested: now that shape is right, can we push peak LR higher? 2× current max_lr. Simple 1-line change.
4. **pct_start=0.05** — faster warmup means more time in anneal phase; test whether the peak timing matters.
5. **batch_size=2** — 2× gradient updates per epoch; OneCycleLR `total_steps` scales automatically with train_loader length.

### Medium priority
6. **SWA** — collect weights from final K epochs, average. OneCycle brings model near flat basin at end; SWA should widen it. Different from EMA (fixed window, not decay).
7. **FiLM conditioning** — Re, AoA, NACA codes → (γ, β) → modulate Transolver blocks. ~few hundred extra params. Adds multiplicative interactions additive concat can't represent.
8. **Coordinate features** — random Fourier features for spatial encoding near airfoil surface. Could help val_single_in_dist (still at 92-93 mean).
9. **Log-space pressure loss** — log-space MAE for surface pressure. Wide dynamic range → log-scale distributes gradient per decade. Could help OOD camber splits.
