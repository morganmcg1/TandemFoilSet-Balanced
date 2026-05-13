# SENPAI Research State

- 2026-05-13 18:45 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=51.5839 (PR #2456 post-LN + wd=0 + slice_num=24)**. Cumulative gain from PR #1391: 121.28 → 51.58 = **−57.5%**.
- No directives from human researcher team yet.

## Current baseline (PR #2456 merged — post-LN swap)

**test_avg/mae_surf_p = 51.5839** | val = 59.1952 (best epoch 18, still descending at cutoff)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + β2=0.99 + **wd=0** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0 + **post-LN**. W&B run: ovv9h3s7.

Per-split: in_dist=51.59, rc=61.37, cruise=39.33, re_rand=54.04.

**Reproduce:** `cd target/ && python train.py --batch_size 4 --accumulation_steps 2 --grad_clip_max_norm 5.0 --weight_decay 0.0`

⚠️ ALL future run commands must include `--weight_decay 0.0` (not yet the default in train.py).
⚠️ Post-LN is now merged into train.py default via PR #2456.

## Win history (this round)
| PR | Config change | test_avg | Δ | Cumulative |
|---|---|---|---|---|
| #2282 | slice_num=24 | 61.8457 | −1.52% | −49.0% |
| #2343 | wd=0 | 60.7447 | −1.78% | −49.9% |
| **#2456** | **post-LN swap** | **51.5839** | **−15.08%** | **−57.5%** |

## Major milestones

### Lion optimizer fully tuned (pre-LN stack — may need re-calibration)
| Lever | Status | Optimal | Source |
|---|---|---|---|
| `lr` | **NEEDS RECHECK** (post-LN) | 1.5e-4 (pre-LN) | Finding #20 / askeladd #2494 |
| `β1` | CLOSED (pre-LN) | 0.9 | #2237 / Finding #32 |
| `β2` | CLOSED (pre-LN) | 0.99 | #2382 / Finding #39 |
| `weight_decay` | CLOSED | 0.0 | #2343 / Finding #38 |

### Post-LN as new foundation (Finding #45)
PR #2456 post-LN swap: **−15.08%** (60.74 → 51.58). All 4 splits improved uniformly (13.5–17.3%). Key mechanism: placement-after-residual keeps residual stream stationary — representation-level effect, not IID/OOD redistribution. Sharp contrast with RMSNorm (#2425): **computation type is second-order; placement is first-order**. best_epoch=18 with loss still descending — model has more headroom.

### IID/OOD redistribution meta-pattern (Finding #41) — REVISED
Originally: capacity/resolution increases improve IID, harm OOD. Post-LN's uniform IID+OOD improvement BREAKS this pattern. Revised: the pattern holds for capacity/resolution axes; normalization position is an orthogonal axis that can improve both simultaneously.

## Round-3 status (updated 2026-05-13 18:45)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| thorfinn | #2433 | Per-iter warmup (correct SequentialLR impl) | **wip** |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **wip** |
| nezuko | #2466 | SwiGLU MLP (param-matched) | **wip** |
| edward | #2473 | Slot routing temperature: fixed T=1.0/0.5 vs learnable | **wip** |
| fern | #2474 | Coord-noise augmentation: Gaussian jitter on mesh coords (train only) | **wip** |
| alphonse | #2485 | Lion gradient noise: LR-scaled Langevin perturbation | **wip** |
| askeladd | #2494 | Post-LN LR re-calibration: lr=3e-4 vs 1.5e-4 (Finding #20 stale) | **wip** (NEW) |
| tanjiro | #2499 | RMSNorm under post-LN: stack computation on placement | **wip** (NEW) |
| tanjiro | #2456 | Pre-LN → Post-LN swap | **MERGED** ✓ | test=51.5839 (−15.08%). **NEW BEST.** |
| askeladd | #2458 | Lookahead-wrapped Lion (k=5/10) | **CLOSED** ✗ | +9.76% regression. Lion binary updates incompatible with Polyak avg. Finding #45b. |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **CLOSED** ✗ | +6.24% regression. LR floor ≥ 1e-5 harmful. Finding #44. |
| edward | #2327 | GELU → SiLU activation swap | **CLOSED** ✗ | +16% regression all 3 trials. Finding #42. |
| fern | #2117 | EMA decay=0.95/0.99 | **CLOSED** ✗ | Diag-ratio collapse; EMA gain budget-dependent. Finding #43. |
| nezuko | #2393 | Fourier L=12/4 | **CLOSED** ✗ | Regime-neutral; IID/OOD redistribution. Finding #41. |
| tanjiro | #2425 | LayerNorm → RMSNorm | **CLOSED** ✗ | Regime-neutral aggregate. Finding #40. |
| askeladd | #2382 | Lion β2=0.999/0.95 | **CLOSED** ✗ | Symmetric +13%. Finding #39. |
| tanjiro | #2344 | attention dropout | **CLOSED** ✗ | Capacity-limited. |
| frieren | #2294 | surf_weight=15/20 | **CLOSED** ✗ | rc +3.42. Finding #37. |


## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED (pre-LN)**: lr=1.5e-4 correct for pre-LN. NEEDS RECHECK under post-LN (#2494).
21. **Clip mechanism fully characterized**: clip=5.0 in bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24.
24. **EMA gain is budget-dependent**: SUPERSEDED by Finding #43.
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED**: T_max=18 optimal.
27. **Capacity-budget structural bound**: >+5% per-epoch overhead = net loser.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule.
29. **Per-epoch warmup = step function (CLOSED BUG)**: Warmup must be per-iteration.
30. **Attention dropout CLOSED**: Slot routing needs determinism.
31. **Inter-seed variance noise floor ~1.4% rel**: Multi-seed needed for sub-noise differences.
32. **Lion β1 lever CLOSED**: β1=0.9 optimal.
33. **Locality-prior OOD tradeoff**: slice_num lever trades OOD, not in-dist accuracy.
34. **Capacity-limited regime** (#2344): 1.47M+18 epochs = capacity-limited, not overfitting.
35. **Locality regularization incompatible with stochastic attention** (#2344).
36. **surf_weight has split-asymmetric effects** (#2294): sw↑ hurts rc. Optimum below 10.
37. **surf_weight lever CLOSED** (#2294): sw=15/20 all worse on rc. sw=5/7 testing.
38. **Lion wd=0 optimal** (#2343): NEW BASELINE.
39. **Lion β2=0.99 sharp sweet spot** (#2382): Lion fully tuned milestone (pre-LN).
40. **Normalization-type lever closed** (#2425): LayerNorm optimal; RMSNorm neutral aggregate.
41. **IID/OOD redistribution meta-pattern** (#2393): Capacity/resolution axes → IID gains, OOD regressions. REVISED: post-LN breaks this pattern — normalization position is an orthogonal axis.
42. **SiLU vs GELU convergence-rate asymmetry** (#2327): GELU optimal.
43. **EMA gain is timeout-budget-dependent** (#2117): EMA shadow degenerates at eta_min=0 + full schedule.
44. **LR floor ≥ 1e-5 in cosine tail is harmful** (#2326): eta_min=0 required for convergence. Revival paths closed.
45. **Post-LN swap is decisive** (#2456): −15.08% across ALL splits uniformly (13.5–17.3%). Residual-stream stationarity is the load-bearing lever. Computation type (LayerNorm vs RMSNorm) is second-order. Post-LN best_epoch=18 still descending — minimum has headroom. **New baseline: 51.5839**.
45b. **Lookahead incompatible with Lion's binary updates** (#2458): Polyak averaging of binary direction choices is destructive at k=5. Meta-optimizer lever closed.

## Active experiments (7 students — tanjiro IDLE)

### Tier 1: LR re-calibration for post-LN (URGENT — Finding #20 stale)
| PR | Student | Expected gain |
|---|---|---|
| #2494 | askeladd | lr=3e-4 under post-LN: −2% to −8% (LR ceiling shifted) |

### Tier 2: Architecture / normalization
| PR | Student | Expected gain |
|---|---|---|
| #2433 | thorfinn | Per-iteration warmup: −0.5% to −2% |
| #2466 | nezuko | SwiGLU MLP (param-matched): −0.3% to −2% |

### Tier 3: OOD-targeted levers (per Finding #41)
| PR | Student | Expected gain |
|---|---|---|
| #2426 | frieren | surf_weight DOWN (sw=5/7): rc improvement expected |
| #2474 | fern | Coord-noise augmentation (σ=0.005): OOD-targeted |
| #2485 | alphonse | Lion gradient noise (σ=0.01): Langevin flat-minimum |

### Tier 4: Architecture probe
| PR | Student | Expected gain |
|---|---|---|
| #2473 | edward | Slot routing temperature ablation: fixed T=1.0 |

## Key open questions
1. **Does lr=3e-4 help under post-LN?** (askeladd #2494) — Finding #20 was on pre-LN; post-LN changes gradient stability boundary. HIGHEST PRIORITY.
2. **What should tanjiro try next?** — RMSNorm under post-LN? More epoch budget? SwiGLU + post-LN stack? NEEDS ASSIGNMENT.
3. **Does SwiGLU's gating compound with post-LN?** (nezuko #2466)
4. **Does per-iteration warmup help post-LN converge faster?** (thorfinn #2433) — warmup was for pre-LN; now critical with best_epoch=18 still descending.
5. **Does coord-noise target OOD without hurting IID?** (fern #2474) — first data-side intervention.
6. **Does slot routing temperature matter?** (edward #2473) — tests whether learnable T is load-bearing.
7. **Does Lion gradient noise help find flat minima?** (alphonse #2485) — Langevin regularizer.

## IMPORTANT NOTES
- **All new/updated run commands must include `--weight_decay 0.0`**
- **Post-LN is now the default** in train.py (merged via #2456)
- **New baseline: 51.5839** — ALL comparisons use this
- **tanjiro is idle** — must be assigned next experiment immediately
- **Many lever closures from pre-LN era may be stale** — Finding #20 (lr), Finding #32 (β1), Finding #39 (β2) all need recheck under post-LN if significant gains continue

## Lever status summary
| Lever | Status | Best value |
|---|---|---|
| Learning rate | **RECHECKING** (#2494, post-LN) | 1.5e-4 (pre-LN; may shift) |
| Optimizer β1 | CLOSED (pre-LN) | β1=0.9 |
| Optimizer β2 | CLOSED (pre-LN) | β2=0.99 |
| Grad clip | CLOSED | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | CLOSED | slice_num=24 |
| mlp_ratio | CLOSED | mlp_ratio=2 |
| T_max cosine | CLOSED | T_max=18 (full) |
| weight_decay | CLOSED — wd=0 optimal | wd=0 |
| Attention dropout | CLOSED | 0.0 |
| Normalization type | CLOSED | LayerNorm |
| Fourier L | CLOSED | L=8 |
| Activation function | CLOSED | GELU |
| Normalization position | **CLOSED — post-LN wins** | post-LN |
| EMA | CLOSED (budget-dependent) | — |
| eta_min | CLOSED | eta_min=0 |
| Meta-optimizer (Lookahead) | CLOSED — incompatible with Lion | — |
| Warmup | **ACTIVE** (#2433) | — |
| surf_weight | **ACTIVE** (#2426, sw↓) | — |
| MLP architecture | **ACTIVE** (#2466, SwiGLU) | — |
| Slot routing temperature | **ACTIVE** (#2473) | — |
| Coord-noise augmentation | **ACTIVE** (#2474) | — |
| Gradient noise (Lion) | **ACTIVE** (#2485) | — |
