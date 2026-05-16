# SENPAI Research State

- **Updated:** 2026-05-16 11:05
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received in current heartbeat cycle. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (multiple students have independently confirmed: inf×0=NaN in the surf_mask-zeroed pressure sum during evaluation).

## Current baseline

**`val_avg/mae_surf_p = 63.79`** (3-arm mean) / **63.47** (best arm `jj0kve7c`) — **FiLM conditioning** on (Re, AoA, NACA, gap, stagger) physical priors + Lion optimizer (lr=1.5e-4, betas=(0.9, 0.99), wd=1e-4) + OneCycleLR(max_lr=3e-4) + batch_size=2 + L1 surface loss + grad_clip max_norm=1.0, PR #3797, **merged** (2026-05-16 10:55)

**Note:** FiLM result was at bs=4+AdamW (pre-stack). Confirmation run of FiLM+Lion+bs=2 (PR #3910) in progress — expected to improve further.

| Split | val mae_surf_p (best arm `jj0kve7c`) | 3-arm mean |
|---|---|---|
| val_single_in_dist | 69.21 | 70.49 |
| val_geom_camber_rc | 81.74 | 80.84 |
| val_geom_camber_cruise | 40.42 | 41.33 |
| val_re_rand | 62.49 | 62.50 |
| **val_avg** | **63.47** | **63.79** |

test 3-split (excl. cruise) = **61.23** (best arm) / **61.50** (3-arm mean)

Head config: `Lion(lr=1.5e-4, betas=(0.9, 0.99), weight_decay=1e-4)` + `batch_size=2` + `OneCycleLR(max_lr=3e-4, total_steps=len(train_loader)*14, pct_start=0.1, div_factor=25, final_div_factor=1e4)` + per-batch step + L1 surf loss + grad_clip max_norm=1.0 + **FiLM conditioning** (cond_dim=11, hidden_dim=64, zero-init, always-on)

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3797 askeladd | **FiLM conditioning (Re, AoA, NACA) physical priors** | **63.79** (mean) / **63.47** (best) | **−4.35%** ✓ **MERGED** |
| #3720 nezuko | Lion optimizer + OneCycleLR(max_lr=3e-4) | 66.69 (best) | −13.46% ✓ **MERGED** |
| #3616 fern | batch_size=2 (2× gradient updates/epoch) | 77.06 (mean) / 75.15 (best) | −5.63% ✓ **MERGED** |
| #3307 askeladd | OneCycleLR right-sized + L1 surf (compound) | 81.66 (mean) / 80.31 (best) | −9.30% ✓ **MERGED** |
| #3434 edward | L1 surface loss (vol MSE + surf L1) | 90.04 | −8.94% ✓ **MERGED** |
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | 98.88 | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 5 active portfolio (8 students, all GPUs assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3910 | askeladd | **FiLM + Lion + bs=2 stacked confirmation** (3-arm) | **New (round 5, 11:00)** — establishes true compounded baseline |
| #3911 | thorfinn | **surf_weight sweep on FiLM+Lion+bs=2** {5, 10, 20} | **New (round 5, 11:00)** — loss-balance sensitivity after FiLM merge |
| #3860 | nezuko | Lion betas (β1, β2) 4-arm sweep | Running — fine-tunes Lion's momentum hyperparameters |
| #3812 | fern | batch_size=1 (extends bs trend) | Running (stale — nudged 10:40) |
| #3617 | edward | log-space L1 + Lion + bs=2 (5-arm, sw∈{10, 30}) | Sent back 2026-05-16 10:40 — stacks log-space L1 with FiLM+Lion+bs=2 |
| #3787 | alphonse | Lion LR sweep at {1e-3 repl, 5e-4, 2e-3} | Nudged 10:40 — awaiting status |
| #3791 | frieren | bf16 mixed precision (14/21/28 epochs) | Nudged 10:40 — awaiting status |
| #3827 | tanjiro | Re/AoA input jitter (3-arm σ sweep) | Running — data-augmentation axis; targets OOD splits |

## Recently closed/sent-back

| PR | Student | Change | Decision |
|---|---|---|---|
| #3614 | thorfinn | bs=2 + AdamW max_lr=1.5e-3 | **closed** — LR axis exhausted (all bumps above 1e-3 regress at bs=4 and bs=2); AdamW superseded by Lion |
| #3615 | nezuko | SWA on final 3-4 OneCycle epochs | closed — SWA mean 85.09 (+4.2%); weight-averaging failure |
| #3622 | frieren | final_div_factor ∈ {1e3, 1e5} | closed — locks 1e4 |
| #3699 | tanjiro | Lookahead(AdamW) | closed — +8.93% regression; third trajectory-averaging failure |

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

1. **FiLM conditioning is now the leading architectural lever** — #3797 merged, −4.35% on top of Lion+bs=2 baseline (63.79 vs 66.69). Mechanism: direct per-block multiplicative modulation of (Re, AoA, NACA) physical priors gives per-channel, per-regime feature gating. Zero-init ensures no representational cost at step 0. **FiLM is now always-on in the merged head.** The cruise split improved −7.41 pp (48.74→41.33) — FiLM's conditioning is especially powerful for OOD geometric shifts.
2. **Lion optimizer is the dominant optimizer** — #3720 merged, −13.46% (66.69 vs 77.06). Sign-based update + grad_clip global L2 = double-bounded trajectory. Lion's optimal LR is 3e-4 (3.3× smaller than AdamW's 1e-3), confirming the Lion paper recommendation.
3. **Loss formulation is the dominant active lever** — L1 surf loss gave 8.94% improvement (#3434). Mechanism: L1 = median estimator, MAE-optimal; L2 chases OOD outliers. Huber WORSE than pure L1 because grad clip already handles gradient-magnitude normalization.
4. **OneCycleLR right-sized to budget is the dominant schedule** — OneCycle + L1 compounded −9.30% (81.66 vs 90.04). Right-sizing `total_steps` critical.
5. **Grad clip is gradient normalization** — max_norm=1.0 fires 100% of steps; median pre-clip norm ~45. Tighter (0.5) gave +1.94% regression.
6. **Model capacity is NOT the bottleneck (historically)** — width, depth, slice changes all failed. FiLM's win is conditioning-based, not capacity-based. Post-FiLM, this finding may need revisiting.
7. **Volume loss structure matters** — vol MSE + surf L1 locked in. Full L1 vol+surf regresses.
8. **surf_weight=10 is optimal on non-FiLM baseline** — under investigation whether FiLM changes this (#3911 thorfinn). Prior: sw=5 regresses 10.8%, sw=25 worse.
9. **Trajectory-averaging methods are incompatible — 3-for-3 failures** — EMA (#3431), SWA (#3615), Lookahead (#3699 +8.93%). Family-wide rejection: don't retry.
10. **AdamW LR axis is exhausted** — max_lr=1e-3 is the ceiling for AdamW+this setup. All bumps at {1.5, 2.0}e-3 regress at both bs=4 and bs=2. Lion's optimal is 3e-4.
11. **batch_size=2 compounded with all subsequent wins** — −5.63% (#3616). bs=1 under test (#3812 fern).

## Potential next research directions (round 5+ backlog)

### FiLM follow-ups (highest priority — active frontier)
- **FiLM + Lion + bs=2 confirmation** (askeladd #3910, running) — critical to establish true stacked baseline
- **surf_weight sensitivity on FiLM+Lion+bs=2** (thorfinn #3911, running) — loss-balance may shift with FiLM's internal conditioning
- **Separate FiLM heads per prior axis** — split Re, AoA, geometry into separate conditioning MLPs; better OOD attribution
- **FiLM hidden_dim sweep** (64→32 for fewer params/faster epochs, or 64→128 for more expressivity)
- **FiLM only on early/late layers** — ablation to understand where conditioning is most useful
- **FiLM + log-space L1** (edward #3617 testing this)
- **FiLM + input jitter** (tanjiro #3827 testing jitter independently; stack once result known)
- **FiLM + extended epochs** (val curve still descending at timeout; need faster epochs via bf16 or smaller eval frequency)

### Optimizer / schedule
- **Lion betas tuning** (nezuko #3860, running)
- **Lion + bf16** (frieren #3791, running) — throughput for more epochs
- **Lion LR revalidation at FiLM baseline** (alphonse #3787 running, but on pre-FiLM baseline — results inform LR axis awareness)

### Data augmentation
- **Re/AoA input jitter** (tanjiro #3827, running)

### Architectural (post-FiLM, lower priority)
- **Cross-attention conditioning** — replace FiLM scalars with cross-attention over conditioning tokens (more expressive, more compute)
- **Spectral loss term** — FFT-based regularizer on chordwise surf_p
- **Hypernetwork-style modulation** — physical priors generate per-sample weight perturbations (harder to implement, high ceiling)

### Frontier diagnostics
- **GH #3292 fix** — `test_geom_camber_cruise` NaN; would unlock paper-facing 4-split test metric
- **OOD error analysis** — worst predictions in `val_geom_camber_rc` and `val_re_rand`; what pattern remains after FiLM?
