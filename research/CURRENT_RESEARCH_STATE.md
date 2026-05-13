# SENPAI Research State

- 2026-05-13 19:45 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=49.3466 (PR #2508 T_max=20 + post-LN + wd=0)**. Cumulative gain from PR #1391: 121.28 → 49.35 = **−59.3%**.
- No directives from human researcher team yet.

## Current baseline (PR #2508 merged — T_max=20 under post-LN)

**test_avg/mae_surf_p = 49.3466** | val = 56.5563 (best epoch 18, still descending at cutoff Δ=−2.64)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + β2=0.99 + **wd=0** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0 + **post-LN** + **t_max=20**. W&B run: i2pxi78b.

Per-split: in_dist=50.894, rc=61.814, cruise=35.172, re_rand=49.507.

**Reproduce:** `cd target/ && python train.py --batch_size 4 --accumulation_steps 2 --grad_clip_max_norm 5.0 --weight_decay 0.0 --t_max 20`

⚠️ ALL future run commands must include `--weight_decay 0.0` and `--t_max 20`.
⚠️ Post-LN is now merged into train.py default via PR #2456.
⚠️ `--t_max 20` is required (default is still 18 in Config dataclass).

## Win history (this round)
| PR | Config change | test_avg | Δ | Cumulative |
|---|---|---|---|---|
| #2282 | slice_num=24 | 61.8457 | −1.52% | −49.0% |
| #2343 | wd=0 | 60.7447 | −1.78% | −49.9% |
| #2456 | post-LN swap | 51.5839 | −15.08% | −57.5% |
| **#2508** | **T_max=20** | **49.3466** | **−4.34%** | **−59.3%** |

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

## Round-3 status (updated 2026-05-13 19:45)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| thorfinn | #2527 | T_max=22: probe Finding #44 boundary on stretched cosine | **wip** (NEW) |
| edward | #2528 | n_layers=6: depth increase enabled by post-LN stability | **wip** (NEW) |
| alphonse | #2530 | Lion β1 re-calibration under post-LN (Finding #32 stale) | **wip** (NEW) |
| fern | #2533 | Lion β2 re-calibration under post-LN (Finding #39 stale) | **wip** (NEW) |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **wip** |
| nezuko | #2466 | SwiGLU MLP (param-matched) | **wip** (stale, status check sent) |
| askeladd | #2494 | Post-LN LR re-calibration: lr=3e-4 vs 1.5e-4 (Finding #20 stale) | **wip** |
| tanjiro | #2499 | RMSNorm under post-LN: stack computation on placement | **wip** |
| thorfinn | #2508 | T_max=20 cosine extension under post-LN | **MERGED** ✓ | test=49.3466 (−4.34%). **NEW BEST.** |
| tanjiro | #2456 | Pre-LN → Post-LN swap | **MERGED** ✓ | test=51.5839 (−15.08%). |
| alphonse | #2485 | Lion gradient noise: LR-scaled Langevin perturbation | **CLOSED** ✗ | +3.46% regression. Noise compounds dithering. Finding #49. |
| edward | #2473 | Slot routing temperature: fixed T=1.0/0.5 vs learnable | **CLOSED** ✗ | +4.87% regression uniform. T learnability is load-bearing. Finding #48. |
| fern | #2474 | Coord-noise augmentation: Gaussian jitter on mesh coords | **CLOSED** ✗ | Within noise floor. rc-specific high-freq sensitivity. Finding #50. |
| askeladd | #2458 | Lookahead-wrapped Lion (k=5/10) | **CLOSED** ✗ | +9.76% regression. Finding #45b. |
| thorfinn | #2433 | Per-iter warmup (correct SequentialLR impl) | **CLOSED** ✗ | +7.78–9.19% regression. Finding #46. |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **CLOSED** ✗ | +6.24% regression. Finding #44. |
| edward | #2327 | GELU → SiLU activation swap | **CLOSED** ✗ | +16% regression. Finding #42. |
| fern | #2117 | EMA decay=0.95/0.99 | **CLOSED** ✗ | Diag-ratio collapse. Finding #43. |
| nezuko | #2393 | Fourier L=12/4 | **CLOSED** ✗ | Regime-neutral. Finding #41. |
| tanjiro | #2425 | LayerNorm → RMSNorm | **CLOSED** ✗ | Regime-neutral. Finding #40. |
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
46. **Warmup incompatible with Lion's sign-update, regardless of normalization position** (#2433): Lion's per-coordinate ±lr bound makes warmup's "shrink step magnitude" irrelevant — lr is the only per-step magnitude control. clip_fire stayed ~100% in epoch 1 WITH warmup (the instability warmup targets does not exist under Lion). Finding applies to all LN positions — tanjiro's proposed "post-LN + warmup stack" is dead. The fix in #2303 corrected a discontinuous step-function BUG, not a missing warmup feature.
47. **T_max=18 was pre-LN optimum; post-LN requires T_max=20** (#2508): Post-LN reaches a deeper minimum that needs more cosine schedule budget. Crossover at epoch 16, accelerating gain through ep-18 (Δ=−2.64). Load-bearing factor is extended tail LR (ep-18: 3.67e-6 vs 0). Higher mid-training LR with T_max=20 actually slightly hurt (epochs 9-12). Finding #26 (T_max=18 closed) was pre-LN-specific. New optimum: T_max=20. Constraint: T_max≤20 per Finding #44 (ep-18 LR stays below 1e-5 threshold).
48. **Slot routing temperature T is load-bearing** (#2473): Removing per-head learnable temperature scalars (20 params) regresses all 4 splits uniformly (+4.87%). Not dead weight at 1.47M. Routing sharpness per head is a learned, necessary signal.
49. **Gradient noise contraindicated for Lion on TandemFoilSet** (#2485): LR-scaled Langevin noise (sigma=0.01) regresses +3.46%. Lion's sign-update is already a noisy direction estimator; compounding with stochastic perturbation adds dithering, not flat-minimum search. Two consecutive negatives from the noise family (Finding #44 + #49). OOD splits hurt most — noise makes this loss surface more opaque, not more generalizable.
50. **Coord-noise has narrow per-split-heterogeneous sigma-curve** (#2474): sigma=0.005 improves cruise −6.58% and re_rand −2.31% but regresses rc +2.96%; sigma=0.01 reverses everything. Aggregate within noise floor. rc behaves opposite to other OOD splits under coord perturbation — non-monotonic camber sensitivity to Fourier smearing.

## Active experiments (8 students — all occupied)

### Tier 1: Schedule extension and pre-LN calibration checks (URGENT)
| PR | Student | Expected gain |
|---|---|---|
| #2527 | thorfinn | T_max=22: probe Finding #44 boundary (ep-18 LR 1.21e-5 on decaying cosine vs 1.5e-5 fixed floor) |
| #2494 | askeladd | lr=3e-4 under post-LN: −2% to −8% (Finding #20 was pre-LN) |

### Tier 2: Architecture depth + optimizer re-calibration
| PR | Student | Expected gain |
|---|---|---|
| #2528 | edward | n_layers=6: depth increase enabled by post-LN gradient stability |
| #2530 | alphonse | β1=0.95/0.85: post-LN cleaner gradients may shift β1 optimum (Finding #32 stale) |
| #2533 | fern | β2=0.999/0.95: post-LN gradient variance profile may shift β2 optimum (Finding #39 stale) |

### Tier 3: OOD-targeted levers
| PR | Student | Expected gain |
|---|---|---|
| #2426 | frieren | surf_weight DOWN (sw=5/7): rc improvement expected |
| #2466 | nezuko | SwiGLU MLP (param-matched) — stale, status check sent |

### Tier 4: Normalization probe
| PR | Student | Expected gain |
|---|---|---|
| #2499 | tanjiro | RMSNorm under post-LN: does computation type interact with placement? |

## Key open questions
1. **Does T_max=22 help further?** (thorfinn #2527) — val delta at ep-18 was −2.64 (accelerating). Finding #44 was about a fixed floor, not a stretched cosine. T_max=22 gives ep-18 LR=1.21e-5 on a decaying curve. HIGHEST PRIORITY (immediately follows the T_max=20 win).
2. **Does lr=3e-4 help under post-LN?** (askeladd #2494) — Finding #20 was on pre-LN; post-LN changes gradient stability boundary.
3. **Do β1 or β2 optimal values shift under post-LN?** (alphonse #2530, fern #2533) — Findings #32 and #39 were both pre-LN. Post-LN's cleaner gradient signal may broaden or shift both optima.
4. **Does n_layers=6 help under post-LN?** (edward #2528) — Post-LN removes the pre-LN gradient variance accumulation that made depth 5 the ceiling.
5. **Does SwiGLU's gating compound with post-LN?** (nezuko #2466) — stale, awaiting student response.
6. **Does RMSNorm stack with post-LN?** (tanjiro #2499) — computation type was second-order on pre-LN.
7. **Can lr=3e-4 + T_max=20 stack?** (future) — if both askeladd #2494 and the current T_max=20 win independently, this combination is the next priority. Do not merge prematurely.

## IMPORTANT NOTES
- **All new/updated run commands must include `--weight_decay 0.0` and `--t_max 20`**
- **Post-LN is now the default** in train.py (merged via #2456)
- **t_max=20 now merged** into train.py but the Config default is still 18 — use `--t_max 20` explicitly
- **New baseline: 49.3466** — ALL comparisons use this
- **All 8 students are occupied** — no idle GPUs
- **Pre-LN lever closures being systematically re-checked** under post-LN: Finding #20 (lr), Finding #32 (β1), Finding #39 (β2) in flight; Finding #26 (T_max) closed and revised to T_max=20

## Lever status summary
| Lever | Status | Best value |
|---|---|---|
| Learning rate | **RECHECKING** (#2494, post-LN) | 1.5e-4 (pre-LN; may shift) |
| Optimizer β1 | **RECHECKING** (#2530, post-LN) | β1=0.9 (pre-LN; may shift) |
| Optimizer β2 | **RECHECKING** (#2533, post-LN) | β2=0.99 (pre-LN; may shift) |
| Grad clip | CLOSED | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | CLOSED | slice_num=24 |
| mlp_ratio | CLOSED | mlp_ratio=2 |
| T_max cosine | **PROBING** T_max=22 (#2527) | T_max=20 (post-LN; Finding #47) |
| weight_decay | CLOSED — wd=0 optimal | wd=0 |
| Attention dropout | CLOSED | 0.0 |
| Normalization type | CLOSED | LayerNorm |
| Fourier L | CLOSED | L=8 |
| Activation function | CLOSED | GELU |
| Normalization position | **CLOSED — post-LN wins** | post-LN |
| EMA | CLOSED (budget-dependent) | — |
| eta_min | CLOSED | eta_min=0 |
| Meta-optimizer (Lookahead) | CLOSED — incompatible with Lion | — |
| Warmup | CLOSED — incompatible with Lion sign-update | — |
| surf_weight | **ACTIVE** (#2426, sw↓) | — |
| MLP architecture | **ACTIVE** (#2466, SwiGLU) | — |
| Slot routing temperature | **ACTIVE** (#2473) | — |
| Coord-noise augmentation | **ACTIVE** (#2474) | — |
| Gradient noise (Lion) | **ACTIVE** (#2485) | — |
