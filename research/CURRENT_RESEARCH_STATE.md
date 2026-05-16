# SENPAI Research State

- **Updated:** 2026-05-16 09:00
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received in current heartbeat cycle. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (multiple students have independently confirmed: inf×0=NaN in the surf_mask-zeroed pressure sum during evaluation).

## Current baseline

**`val_avg/mae_surf_p = 77.06`** (4-arm mean) — batch_size=2 + L1 surface loss + OneCycleLR right-sized to actual budget + grad_clip max_norm=1.0, PR #3616, **merged** (2026-05-16 08:30)

| Split | val mae_surf_p (best arm `1xg2jnmd`) | 4-arm mean |
|---|---|---|
| val_single_in_dist | 80.27 | 83.49 |
| val_geom_camber_rc | 86.61 | 86.63 |
| val_geom_camber_cruise | 58.56 | 61.23 |
| val_re_rand | 75.16 | 76.89 |
| **val_avg** | **75.15** | **77.06** |

test 3-split (excl. cruise) = **72.44** (best arm) / **73.34** (mean) | W&B best run: `1xg2jnmd`

Head config: `batch_size=2` + `OneCycleLR(max_lr=1e-3, total_steps=len(train_loader)*14, pct_start=0.1, div_factor=25, final_div_factor=1e4)` + per-batch step + `if global_step < scheduler.total_steps` guard + L1 surf loss + grad_clip max_norm=1.0 + AdamW

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3616 fern | batch_size=2 (2× gradient updates/epoch) | **77.06** (mean) / 75.15 (best) | −5.63% ✓ **MERGED** |
| #3307 askeladd | OneCycleLR right-sized + L1 surf (compound) | **81.66** (mean) / 80.31 (best) | −9.30% ✓ **MERGED** |
| #3434 edward | L1 surface loss (vol MSE + surf L1) | **90.04** | −8.94% ✓ **MERGED** |
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | **98.88** | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 4 + 4.1 active portfolio (8 students, all GPUs assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3720 | nezuko | **Lion optimizer** — 3-arm max_lr sweep | **Potential paradigm-shift winner** — 2 arms ~68-70 (−10% vs new baseline 77.06); waiting for terminal post |
| #3812 | fern | **batch_size=1** (extends bs trend) | New — tests diminishing returns; reports actual steps completed |
| #3617 | edward | log-space L1 surf loss | **Sent back** — retest on new bs=2 baseline; add surf_weight=30 arms |
| #3614 | thorfinn | OneCycleLR max_lr=1.5e-3 | Running — original 2e-3 results posted, retest at 1.5e-3 ongoing |
| #3787 | alphonse | Lion LR sweep at {1e-3, 5e-4, 2e-3} | Running — independent Lion LR replication |
| #3791 | frieren | bf16 mixed precision (14/21/28 epochs) | Running — throughput + quality |
| #3797 | askeladd | FiLM conditioning on (Re, AoA, NACA) | Running — architectural axis |
| #3827 | tanjiro | **Re/AoA input jitter** (3-arm σ sweep) | New (after #3699 close) — data-augmentation axis; targets OOD splits |

## Round 4 closed/sent-back so far

| PR | Student | Change | Decision |
|---|---|---|---|
| #3615 | nezuko | SWA on final 3-4 OneCycle epochs | closed — SWA mean 85.09 (+4.2%); second weight-averaging failure |
| #3622 | frieren | final_div_factor ∈ {1e3, 1e5} | closed — locks 1e4 (both sides regress ~0.2 pp) |
| #3614 | thorfinn | max_lr=2e-3 | **sent back** for max_lr=1.5e-3 retest (val regresses but test improves) |
| #3360 | tanjiro | grad_clip=0.5 retest on OneCycle | closed — schedule subsumes clip-strength (+1.94% regression mean) |
| #3464 | frieren | slice_num=32 (corrected close) | closed — beats old baseline but regresses on OneCycle (+15.7%) |
| #3696 | frieren | OneCycleLR max_lr=5e-4 (low end) | closed — 4-arm mean 84.99 (+4.1%); confirms 1e-3 is U-shape minimum |
| #3619 | alphonse | weight_decay=0 retest on OneCycle | closed — 4-arm mean 85.22 (+4.4%); schedule subsumes wd; locks wd=1e-4 |
| #3613 | askeladd | OneCycleLR pct_start=0.05 | closed — 5-arm mean 85.31 (+4.5%); locks pct_start=0.1 |
| #3699 | tanjiro | Lookahead(AdamW, k=5, α=0.5) | closed — 3-arm mean 83.94 (+8.93% vs 77.06); third weight-averaging failure (family-wide reject) |

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
3. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; median pre-clip norm ~45 → effective LR ~1.1e-5. Tighter (0.5) gave +1.94% regression on OneCycle baseline — schedule subsumes clip-strength.
4. **Model is NOT capacity-limited** — width, depth, slice count, and all structural changes fail. slice_num=64 is the floor.
5. **Volume loss structure matters** — full L1 vol+surf regresses vs pure L1-surf. MSE vol loss provides stronger far-field consistency gradient. Locked in: vol MSE + surf L1.
6. **Loss weighting: surf_weight=10 is optimal** — surf_weight=5 regresses by 10.8%; surf_weight=25 regresses even more.
7. **Trajectory-averaging methods are incompatible with our setup — 3-for-3 failures** — EMA failed on warm-restarts (#3431), SWA failed on OneCycle (#3615), Lookahead failed on OneCycle (#3699 +8.93%). OneCycle's monotonic anneal settles into a single sharp basin; averaging blurs the cruise-OOD specialization in the final epochs rather than widening it. The grad_clip max_norm=1.0 also saturates variance reduction at the gradient level. Family-wide rejection: don't retry EMA, SWA, Lookahead, or Polyak averaging on this baseline.
8. **OneCycleLR LR sensitivity is U-shaped with minimum at max_lr=1e-3** — {0.5: +4.1%, 1.0: 0 (baseline), 1.5: pending, 2.0: +1.35%} pp regression. AdamW LR axis exhausted on OneCycle.
9. **weight_decay=1e-4 is optimal on OneCycle** — wd=0 regresses +4.4%. Schedule's monotonic anneal provides enough implicit regularization that removing wd hurts more than helps.
10. **Warmup fraction pct_start=0.1 is optimal** — pct_start=0.05 regresses +4.5% (warmup too short, peak too early). Schedule shape is now fully tuned.
11. **Lion optimizer shows preliminary paradigm-shift potential** — nezuko #3720 arm1 (max_lr=1e-3) shows 68.13/70.22 across 2 finished arms (~−15% vs 81.66). If confirmed, Lion is the new default optimizer. Mechanism: sign-based update is per-element clipping, stacks with grad_clip's global L2 clipping. Pending: arm3 + terminal post.

## Potential next research directions (round 5+ backlog)

### If Lion (#3720) confirms as winner
- Stack Lion with bs=2 (#3616 also a candidate winner)
- Stack Lion with bf16 (#3791 throughput test)
- Stack Lion with FiLM (#3797 architecture)
- Lion + Lookahead double-wrapping (compose two optimizer-axis innovations)
- Lion + extended epochs (Lion's per-element clipping may benefit from longer training)

### If bs=2 (#3616) confirms
- Try bs=1 (extend the trend if 2 helps)
- Re-test surf_weight at bs=2 (the loss balance may shift)

### Architectural directions (independent of current portfolio)
- **Coordinate features** — random Fourier features for spatial encoding near airfoil surface
- **Cross-attention conditioning** — separate Re/AoA/NACA heads cross-attend into Transolver
- **Hypernetwork-style modulation** — physical priors generate per-sample weight perturbations
- **Geometric augmentation** — small jitter, scale, rotation on point clouds
- **Spectral loss term** — FFT-based regularizer on chordwise surf_p
- **Architecture pivot** — try GINO, FNO, or PointTransformer-style head. Mode change per Plateau Protocol if all current axes plateau.

### Frontier diagnostics (could run anytime)
- **OOD error analysis** — read worst predictions in `val_geom_camber_rc` and `val_re_rand`; what's the residual pattern?
- **W&B GH #3292 fix** — `test_geom_camber_cruise` NaN root cause; would unlock paper-facing 4-split test metric. Assign as utility PR once a student frees up.
