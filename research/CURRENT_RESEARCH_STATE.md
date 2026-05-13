# SENPAI Research State

- 2026-05-13 20:15 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=47.9076 (PR #2494 Lion lr=2e-4 + post-LN + T_max=18 + wd=0)**. Cumulative gain from PR #1391: 121.28 → 47.91 = **−60.5%**.
- No directives from human researcher team yet.

## Current baseline (PR #2494 merged — lr=2e-4 under post-LN)

**test_avg/mae_surf_p = 47.9076** | val = 55.9044 (best epoch 18/18)
Config: bf16 + bs=4 + accum=2 + Lion **lr=2e-4** + β1=0.9 + β2=0.99 + **wd=0** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0 + **post-LN** + **t_max=18**. W&B run: 1vr2l3if.

Per-split: in_dist=47.82, rc=60.53, cruise=34.40, re_rand=48.88.

**Reproduce:** `cd target/ && python train.py --batch_size 4 --accumulation_steps 2 --grad_clip_max_norm 5.0 --weight_decay 0.0 --lr 2e-4`

⚠️ ALL future run commands must include `--weight_decay 0.0` and `--lr 2e-4`.
⚠️ Post-LN is now merged into train.py default via PR #2456.
⚠️ Default `t_max=18` in Config dataclass; the T_max=20 win (#2508) was superseded by lr=2e-4 — use **`--lr 2e-4`** alone for the current best, and the **lr=2e-4 + T_max=20 stack** is being tested by askeladd #2568.

## Win history (this round)

| PR | Config change | test_avg | Δ | Cumulative |
|---|---|---|---|---|
| #2282 | slice_num=24 | 61.8457 | −1.52% | −49.0% |
| #2343 | wd=0 | 60.7447 | −1.78% | −49.9% |
| #2456 | post-LN swap | 51.5839 | −15.08% | −57.5% |
| #2508 | T_max=20 | 49.3466 | −4.34% | −59.3% |
| **#2494** | **lr=2e-4** | **47.9076** | **−2.92%** | **−60.5%** |

## Major milestones

### Lion optimizer fully tuned (pre-LN closures revised under post-LN)

| Lever | Status | Optimal | Source |
|---|---|---|---|
| `lr` | **CLOSED** (post-LN) | 2e-4 | #2494 / Finding #54 |
| `β1` | TESTING (post-LN) | 0.9 (pre-LN) | #2530 alphonse (wip) |
| `β2` | **CLOSED** (post-LN — same as pre-LN) | 0.99 | #2533 closed, Finding #51 |
| `weight_decay` | CLOSED | 0.0 | #2343 / Finding #38 |

### Post-LN as new foundation (Finding #45)
PR #2456 post-LN swap: **−15.08%** (60.74 → 51.58). All 4 splits improved uniformly (13.5–17.3%). Key mechanism: placement-after-residual keeps residual stream stationary — representation-level effect, not IID/OOD redistribution. Sharp contrast with RMSNorm (#2425): **computation type is second-order; placement is first-order**.

### Optimizer scale + schedule shape — both moved under post-LN
- **T_max** moved 18 → 20 (Finding #47, PR #2508)
- **lr** moved 1.5e-4 → 2e-4 (Finding #54, PR #2494)
- **β2** stayed at 0.99 (Finding #51 confirmed under post-LN — β2=0.999 +35.1% catastrophic)

The two moves are individually validated; the stack (lr=2e-4 + T_max=20) is now being tested by askeladd #2568. Predicted additional gain: −1 to −3%.

### IID/OOD redistribution meta-pattern (Finding #41) — REVISED
Originally: capacity/resolution increases improve IID, harm OOD. Post-LN's uniform IID+OOD improvement BREAKS this pattern. Revised: the pattern holds for capacity/resolution axes; normalization position is an orthogonal axis that can improve both simultaneously. **Under lr=2e-4, the IID split now dominates the gain** (in_dist −6%) — the OOD/IID redistribution may be reasserting itself in the lr-scale axis even though it was broken in the LN-position axis.

### Per-split lever-flip under post-LN (Finding #53, frieren #2426)
The pre-LN sign of the surf_weight–rc relationship **inverted** under post-LN. Pre-LN: sw=5 → rc −0.74 (DOWN helps rc). Post-LN: sw=5 → rc +1.49 (DOWN hurts rc). Cruise mirror-inverted. Mechanism: post-LN's compressed residual stream makes surface/volume token representations more similar, breaking the surf_weight gradient signal that pre-LN had. **Implication:** other per-split findings from pre-LN should be re-validated case-by-case under post-LN.

## Round-3 status (updated 2026-05-13 20:15)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| askeladd | #2568 | lr=2e-4 + T_max=20 STACK (new winner compound) | **wip** (NEW) |
| frieren | #2572 | lr=2.25e-4 LR midpoint refinement | **wip** (NEW) |
| fern | #2573 | mlp_ratio=3 capacity width without depth | **wip** (NEW) |
| edward | #2577 | slice_num=32 physics-aware re-cal under post-LN | **wip** (NEW) |
| thorfinn | #2527 | T_max=22: probe Finding #44 boundary on stretched cosine | **wip** |
| alphonse | #2530 | Lion β1 re-calibration under post-LN (Finding #32 stale) | **wip** |
| nezuko | #2466 | SwiGLU MLP (param-matched) | **wip** (status check #2 sent — `d0nsbeam` run finished at test=51.59 on old config, awaiting confirmation arm with lr=2e-4) |
| tanjiro | #2499 | RMSNorm under post-LN | **wip** (status check sent — last run crashed) |
| askeladd | #2494 | Post-LN LR re-calibration: lr=2e-4 wins | **MERGED** ✓ test=47.9076 (−2.92%). **NEW BEST.** |
| thorfinn | #2508 | T_max=20 cosine extension under post-LN | **MERGED** ✓ test=49.3466 (−4.34%) |
| tanjiro | #2456 | Pre-LN → Post-LN swap | **MERGED** ✓ test=51.5839 (−15.08%). |
| fern | #2533 | Lion β2 re-calibration under post-LN | **CLOSED** ✗ β2=0.999 catastrophic +35.1%. Finding #51. |
| edward | #2528 | n_layers=6: depth increase under post-LN | **CLOSED** ✗ +8.78% timeout-bound. Finding #52. |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **CLOSED** ✗ Flat +0.60%, per-split flip. Finding #53. |
| alphonse | #2485 | Lion gradient noise: LR-scaled Langevin perturbation | **CLOSED** ✗ +3.46% regression. Finding #49. |
| edward | #2473 | Slot routing temperature: fixed T | **CLOSED** ✗ +4.87% regression. Finding #48. |
| fern | #2474 | Coord-noise augmentation | **CLOSED** ✗ Within noise floor. Finding #50. |
| askeladd | #2458 | Lookahead-wrapped Lion | **CLOSED** ✗ +9.76% regression. Finding #45b. |
| thorfinn | #2433 | Per-iter warmup | **CLOSED** ✗ +7.78–9.19% regression. Finding #46. |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **CLOSED** ✗ +6.24%. Finding #44. |
| edward | #2327 | GELU → SiLU activation swap | **CLOSED** ✗ +16% regression. Finding #42. |
| fern | #2117 | EMA decay=0.95/0.99 | **CLOSED** ✗ Finding #43. |
| nezuko | #2393 | Fourier L=12/4 | **CLOSED** ✗ Finding #41. |
| tanjiro | #2425 | LayerNorm → RMSNorm (standalone) | **CLOSED** ✗ Finding #40. |
| askeladd | #2382 | Lion β2=0.999/0.95 (pre-LN) | **CLOSED** ✗ Finding #39. |

## Key research findings (cumulative — latest at bottom)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling (pre-LN closed)**: lr=1.5e-4 correct for pre-LN. **SUPERSEDED by Finding #54.**
21. **Clip mechanism fully characterized**: clip=5.0 in bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24 (under pre-LN — re-cal testing as of #2577).
24. **EMA gain is budget-dependent**: SUPERSEDED by Finding #43.
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED (pre-LN)**: T_max=18 optimal pre-LN. **SUPERSEDED by Finding #47.**
27. **Capacity-budget structural bound**: >+5% per-epoch overhead = net loser. Reinforced by Finding #52.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule.
29. **Per-epoch warmup = step function (CLOSED BUG)**.
30. **Attention dropout CLOSED**: Slot routing needs determinism.
31. **Inter-seed variance noise floor ~1.4% rel**.
32. **Lion β1 lever CLOSED (pre-LN)**: β1=0.9 optimal. Testing under post-LN (#2530).
33. **Locality-prior OOD tradeoff**: slice_num lever trades OOD, not in-dist accuracy (pre-LN).
34. **Capacity-limited regime** (#2344): 1.47M+18 epochs = capacity-limited.
35. **Locality regularization incompatible with stochastic attention** (#2344).
36. **surf_weight has split-asymmetric effects** (#2294): pre-LN.
37. **surf_weight lever CLOSED (pre-LN)** (#2294, #2426 follow-up): sw=15/20 worse on rc pre-LN; sw=5 flat under post-LN with per-split flip (Finding #53). DOWN closed.
38. **Lion wd=0 optimal** (#2343): NEW BASELINE.
39. **Lion β2=0.99 sharp sweet spot (pre-LN)** (#2382). **REINFORCED by Finding #51 under post-LN.**
40. **Normalization-type lever closed** (#2425): LayerNorm optimal; RMSNorm neutral aggregate.
41. **IID/OOD redistribution meta-pattern** (#2393, REVISED): post-LN breaks the capacity-tradeoff; lr-scale axis may reassert it.
42. **SiLU vs GELU convergence-rate asymmetry** (#2327): GELU optimal (pre-LN).
43. **EMA gain is timeout-budget-dependent** (#2117).
44. **LR floor ≥ 1e-5 in cosine tail is harmful** (#2326): applies to FIXED floor; T_max=22 (#2527) probes whether decaying cosine through 1e-5 is harmful too.
45. **Post-LN swap is decisive** (#2456): −15.08% across ALL splits uniformly. Residual-stream stationarity is the load-bearing lever.
45b. **Lookahead incompatible with Lion's binary updates** (#2458).
46. **Warmup incompatible with Lion's sign-update** (#2433): clip_fire stays ~100% in epoch 1 WITH warmup; instability doesn't exist under Lion.
47. **T_max=18 was pre-LN-specific; post-LN requires T_max=20** (#2508): Extended tail LR is load-bearing.
48. **Slot routing temperature T is load-bearing** (#2473): 20 params not dead weight.
49. **Gradient noise contraindicated for Lion** (#2485): Lion's sign-update already a noisy direction estimator.
50. **Coord-noise has narrow per-split-heterogeneous sigma-curve** (#2474): rc anti-correlated with cruise/re_rand on sigma.
51. **β2=0.99 confirmed optimal under post-LN** (#2533 closed): β2=0.999 → +35.1% catastrophic test regression. Post-LN gradient statistics make over-smoothing **worse** not better — Lion's sign-of-momentum needs gradient variance to make adaptive per-coordinate decisions. β2 valley appears NARROWER under post-LN, not broader (asymmetric sensitivity).
52. **n_layers=6 timeout-bound under 30-min cap** (#2528 closed): Post-LN does enable stable depth-6 training (sanity gates pass, gn_mean monotonically descends), but +23% per-epoch cost cuts schedule by 3 epochs → +8.78% regression. Mechanism viable; budget constraint blocks merge under fixed cap. Reinforces Finding #27.
53. **Per-split lever-flip under post-LN** (#2426 closed): surf_weight–rc gradient SIGN flipped pre-LN → post-LN. sw=5 helps rc pre-LN, hurts rc post-LN. Cruise mirror-inverted. Mechanism: post-LN's compressed residual stream makes surface/volume tokens more similar. **Implication for round design**: per-split findings from pre-LN are not transferable — must re-validate case-by-case under each new optimizer config.
54. **Lion lr=2e-4 optimal under post-LN** (#2494): pre-LN's lr=1.5e-4 (Finding #20) was a pre-LN-specific calibration. Under post-LN's bounded residual stream, lr=2e-4 wins (1.33× pre-LN, not 2×). Both lr=2e-4 and lr=3e-4 beat baseline — lr=3e-4 wins on rc OOD by small margin, lr=2e-4 wins on 3 of 4 splits and dominates on test_avg. e1 gn_mean for lr=3e-4 was 95.9 (predicted 120–160 was too pessimistic). **Implication**: the post-LN landscape has substantial unused headroom.

## Active experiments (8 students — all occupied)

### Tier 1: Winner compounds + LR neighborhood (HIGHEST EV)
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2568 | askeladd | **lr=2e-4 + T_max=20 STACK** | −1% to −3% (compound of two wins) |
| #2572 | frieren | lr=2.25e-4 LR midpoint refinement | −0% to −2% (LR optimum mapping) |
| #2527 | thorfinn | T_max=22 (older — pre-#2494 LR config) | probe Finding #44 boundary |

### Tier 2: Capacity expansion (orthogonal to LR axis)
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2573 | fern | mlp_ratio=3 (width without depth) | −1% to −3% (capacity unused under post-LN+lr=2e-4) |
| #2577 | edward | slice_num=32 (physics-aware re-cal) | OOD-focused, especially rc=60.53 |

### Tier 3: Sub-architectural + optimizer probes
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2530 | alphonse | Lion β1 re-cal under post-LN | direction unknown (Finding #32 stale) |
| #2466 | nezuko | SwiGLU MLP (status check pending — old run at test=51.59, needs lr=2e-4 retry) | awaiting student |
| #2499 | tanjiro | RMSNorm under post-LN (status check pending — last run crashed) | awaiting student |

## Key open questions

1. **Does the lr=2e-4 + T_max=20 stack compound?** (askeladd #2568) — HIGHEST PRIORITY. If yes, test=46.5–47.2 expected.
2. **Is lr=2.25e-4 the true LR optimum?** (frieren #2572) — LR-curve mapping. The midpoint test will tell us whether the optimum sits between 2e-4 and 3e-4 or exactly at 2e-4.
3. **Does T_max=22 cross the Finding #44 boundary?** (thorfinn #2527) — was assigned under the old lr=1.5e-4 baseline. Even if T_max=22 wins on the OLD baseline, it needs to be re-tested with the new lr=2e-4 next round.
4. **Does Lion β1 shift under post-LN?** (alphonse #2530) — Finding #32 was pre-LN.
5. **Can mlp_ratio=3 add capacity without breaking budget?** (fern #2573) — capacity-via-width vs the depth-failure mode of n_layers=6.
6. **Does slice_num=32 unlock more physics-aware OOD performance?** (edward #2577) — slice_num=24 was a pre-LN closure; post-LN's stationary residual stream may support finer groupings.
7. **Does SwiGLU's gating compound with lr=2e-4?** (nezuko #2466) — old run at test=51.59 was under-config.
8. **Does RMSNorm stack with post-LN placement + lr=2e-4?** (tanjiro #2499) — crashed last run.

## Outstanding follow-ups (not yet assigned)

- Stretch: lr=2e-4 + T_max=20 + epochs=20 (only after #2568 lands and if budget allows — currently timeout-risky at 20×102s ≈ 34 min).
- Stretch: mlp_ratio=4 (if mlp_ratio=3 wins clearly).
- Stretch: slice_num=48 (if slice_num=32 wins and per-epoch time stays under 110s).
- Stretch: grad_clip=3.0 tightening (askeladd noted clip is still 50% firing at e18).
- Stretch: surf_weight UP (sw=12, 15) under post-LN+lr=2e-4 — frieren's Finding #53 found per-split flip; mirror experiment.
- Stretch: epochs=20 under lr=2e-4 (best_epoch=18/18 is still cutoff). Compute-risky.

## IMPORTANT NOTES

- **All new/updated run commands must include `--weight_decay 0.0` and `--lr 2e-4`**
- **Post-LN is now the default** in train.py (merged via #2456)
- **lr=2e-4 is the new optimal** but Config default is still lr=1.5e-4 — use `--lr 2e-4` explicitly
- **t_max default is 18** in Config; the lr=2e-4 win at T_max=18 is current best — `--lr 2e-4` alone reproduces it. T_max=20 stack is being tested.
- **New baseline: 47.9076** — ALL comparisons use this
- **All 8 students are occupied** — no idle GPUs
- **Pre-LN findings are not transferable wholesale** (Finding #53: per-split flips can occur under post-LN). Re-validate case-by-case.

## Lever status summary

| Lever | Status | Best value |
|---|---|---|
| Learning rate | **CLOSED** (post-LN) | lr=2e-4 (Finding #54) |
| Optimizer β1 | TESTING (#2530) | β1=0.9 (pre-LN; may shift) |
| Optimizer β2 | CLOSED — same as pre-LN | β2=0.99 (Finding #51) |
| Grad clip | CLOSED (provisional — may revisit under lr=2e-4) | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED — n=6 budget-bound (Finding #52) | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | **TESTING** (#2577) under post-LN+lr=2e-4 | slice_num=24 (pre-LN closure) |
| mlp_ratio | **TESTING** (#2573) under post-LN+lr=2e-4 | mlp_ratio=2 (pre-LN closure) |
| T_max cosine | **TESTING** T_max=22 (#2527) and T_max=20 stack (#2568) | T_max=18 (current best is lr=2e-4 + T_max=18) |
| weight_decay | CLOSED | wd=0 |
| Attention dropout | CLOSED | 0.0 |
| Normalization type | CLOSED | LayerNorm |
| Fourier L | CLOSED | L=8 |
| Activation function | CLOSED (pre-LN) | GELU |
| Normalization position | CLOSED — post-LN wins | post-LN |
| EMA | CLOSED (budget-dependent) | — |
| eta_min | CLOSED | eta_min=0 |
| Meta-optimizer (Lookahead) | CLOSED — incompatible with Lion | — |
| Warmup | CLOSED — incompatible with Lion sign-update | — |
| surf_weight | CLOSED (DOWN, Finding #53 per-split flip) | surf_weight=10 |
| MLP architecture | TESTING (#2466 SwiGLU; awaiting lr=2e-4 retry) | — |
| Gradient noise (Lion) | CLOSED (Finding #49) | — |
