# SENPAI Research State

- 2026-05-13 11:00 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=68.0957 (PR #2090 grad_clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 68.10 = −43.8%.
- No directives from human researcher team yet.
- **Assignment routing bug fixed**: PRs #2165 and #2166 were created on branches `tanjiro/...` and `thorfinn/...` (missing `willowpai2g48h1-` prefix), so student pods never polled them. Closed and reassigned as #2208 (tanjiro) and #2209 (thorfinn) on correctly-prefixed branches.

## Current baseline (PR #2090 merged — grad_clip_max_norm=5.0)

**test_avg/mae_surf_p = 68.0957** | val = 75.8431 (best epoch 14)
Config: bf16 + batch_size=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 + **grad_clip_max_norm=5.0**. W&B run: 0w7kkvb8.

Per-split: in_dist=68.29, rc=82.24, cruise=50.71, re_rand=71.14.

**Mechanism**: clip=5.0 is a bulk Lion direction rescaler (fire_rate 84–100%, mean grad norm 19–109 vs threshold 5.0). Smooths the direction signal before Lion's momentum buffer update, reducing sign-vote variance under grad-accum=2. Opposite to AdamW clip behavior: Lion's sign update discards magnitude, so clipping g is "free" while the direction smoothing is pure upside.

## Previous baselines
- PR #1980 (gradient accumulation accum=2): test=80.62 | val=90.82
- PR #1395 (Lion optimizer): test=83.77 | Lion lr=1.5e-4, no accumulation
- PR #1387 (Fourier+wider): test=93.29 | AdamW lr=7e-4, space_dim=34, n_hidden=192
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-3 status (updated 2026-05-13 11:00)

| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| tanjiro | #2208 | grad-clip-sweep (max_norm=2.0/10.0/50.0) | **wip** (REASSIGNED) | Bracket optimal clip threshold; mechanism validation |
| thorfinn | #2209 | cosine-realign-epochs15 | **wip** (REASSIGNED) | T_max=15 to match actual 30-min budget |
| edward | #2141 | layerscale-1e-4 | **wip** | CaiT-style per-channel residual scaling, γ_init=1e-4 |
| alphonse | #2115 | mesh-node-dropout=0.1 | **CLOSED** ✗ | +84.7% catastrophic. Mechanism: PointNet-style dropout poisons dense physics attention. → **assigned #2191 n_layers=6+clip** |
| fern | #2117 | ema-decay-095 | **wip** | 14-step half-life EMA, follow-up to closed #2050 |
| frieren | #2118 | fourier-per-axis-L (Lx=8, Ly=4) | **CLOSED** ✗ | +4.71%. Hypothesis falsified (in_dist hit worst → info loss, not aliasing). Fourier ceiling L=8 confirmed. → **assigned #2190 accum=4+clip** |
| nezuko | #2121 | slice_num-48 + clip=5.0 retest | **wip** (SENT BACK) | Retest on new baseline=68.10; previous result 79.60 vs old 80.62 |
| askeladd | #2088 | lion-lr-2.1e-4-sqrt2 | **wip** | Lion lr sqrt(2) scaling for eff_bs=8 |
| tanjiro | #2090 | grad-norm-clip-5-on-lion-stack | **MERGED** ✓ | **test=68.0957** (−15.5%); new best. Bulk Lion rescaler, not tail-only. |
| nezuko | #2121 | slice-num-48 (vs old baseline) | **SENT BACK** | test=79.60 vs old 80.62 (−1.27% vs old baseline). Superseded by new 68.10 baseline. |
| thorfinn | #2030 | drop-path-stochastic-depth | **CLOSED** ✗ | +6.0% (85.47 vs 80.62). Underfitting: depth=5 too shallow, 18 epochs too few for ensemble. |
| edward | #2010 | swiglu-activation (GELU→SiLU) | **CLOSED** ✗ | +14.3% (92.17 vs 80.62, 2-seed). Activation-swap lever CLOSED. |
| alphonse | #2047 | n-hidden-224-rescaled-cosine | **CLOSED** ✗ | +11.8%. Width saturated at 30-min cap. |
| fern | #1969 | decoupled-weight-decay | **CLOSED** ✗ | +1.2% null. |
| frieren | #2050 | ema-weights-decay-0999 | **CLOSED** ✗ | +29.9%. Horizon mismatch. |
| nezuko | #1967 | slice-num-96 | **CLOSED** ✗ | +11.3%. Capacity-up cost dominates. |

## Key research findings (cumulative)

1. **Throughput at 30-min budget**: bf16+batch-8 → 17 epochs → round-1 win.
2. **Schedule alignment**: T_max=actual epochs → −7.67%. T_max=14 over-corrects.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%. High-freq basis helps near-foil.
5. **Lion optimizer biggest lever (before clip)**: sign-momentum → −15.97% (83.77 from 99.69).
6. **Lion opens memory budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4.
7. **Depth dead at all tested widths**: horizon-vs-depth tradeoff. CLOSED permanently.
8. **Gradient accumulation (accum=2) wins**: −3.77%, free. Tighter micro-batch padding reduces sign-vote noise.
9. **Width lever CLOSED**: empirical O(n_hidden^2.43) per-epoch cost; n_hidden>192 infeasible at 30-min.
10. **EMA decay=0.999 CLOSED**: Rapid descent regime ≠ stationary trajectory. Short-horizon (0.95) being tested.
11. **Slice-num=96 CLOSED**: Capacity-up cost. Reversed direction (48) being retested.
12. **Weight-decay lever CLOSED across both axes**: magnitude+structure both null.
13. **Fourier lever permanently CLOSED at L=8 uniform**: L=16 aliases, Lx=8/Ly=4 also closed (+4.71% — y-axis carries info, not aliasing). Three points of evidence all confirm L=8 as the optimum.
14. **grad-accum=4 CLOSED**: Step starvation at eff_bs=16.
15. **DropPath CLOSED**: Underfitting regime, depth=5 too shallow, 18 epochs too few.
16. **Activation-swap CLOSED (SiLU, +14.3%)**: GELU near-optimal at depth=5 width=192.
17. **grad_clip=5.0 MASSIVE WIN**: −15.5% (68.10 from 80.62). Bulk Lion direction rescaler. New best. Mechanism: clip before momentum buffer smooths sign-vote direction variance, free since Lion discards magnitude anyway.
18. **Fourier lever permanently CLOSED at L=8 uniform**: 3 points of evidence — L=16 +4.6%, Ly=4 +4.7%, L=8 winning. y-axis high-freq carries info (transition regions, wake shed, leading/trailing edge pressure).
19. **Mesh-node-dropout CLOSED at p=0.1**: PointNet-style dropout fundamentally incompatible with Transolver's dense physics attention. Three failure modes: slice-token poisoning, eval-time distribution shift, dense-coverage requirement for unseen geometry.

## Active hypotheses — new priority order

Given new baseline test=68.10 and the clip=5.0 breakthrough, priority order shifts:

### Tier 1: Direct follow-ups to clip breakthrough
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2165 | tanjiro | grad_clip sweep (2.0, 10.0, 50.0) | −1% to −3% if 2.0 is better |
| #2166 | thorfinn | cosine T_max realign epochs=15 | −0.5% to −2% free win |
| #2121 | nezuko | slice_num=48 + clip=5.0 retest | −0.5% to −1.5% if stacks |

### Tier 2: Independent levers (may beat 68.10 on their own)
| PR | Student | Hypothesis | Likely gain range |
|---|---|---|---|
| #2088 | askeladd | Lion lr=2.1e-4 (sqrt(2)) | Likely direction-swap given new clip smoothing |
| #2117 | fern | EMA decay=0.95 | −0.3% to −1.5% |
| #2141 | edward | LayerScale γ=1e-4 | −0.3% to −1.5% |

### Tier 1 NEW (assigned 2026-05-13 10:52)
| PR | Student | Hypothesis | Expected gain | Discriminating? |
|---|---|---|---|---|
| #2190 | frieren | accumulation_steps=4 + clip=5.0 (mechanism-changed retest of closed #2009) | −1% to −3% if clip changes step-starvation | YES — validates/falsifies clip-as-direction-rescaler |
| #2191 | alphonse | n_layers=6 + clip=5.0 (mechanism-changed retest of closed #1862) | −2% to −5% if depth ceiling was clip-conditional | YES — tests whether depth=5 is genuine architectural ceiling |

## Key open questions (updated)
1. **Is 2.0 better or worse than 5.0 for max_norm?** (#2165) — characterizes whether bulk rescaling benefits from being even tighter, or whether 5.0 was the right trade-off.
2. **Does fully-annealed cosine (T_max=15) help with clip?** (#2166) — low-LR refinement + well-conditioned gradient directions at the same time.
3. **Does slice_num=48 stack with clip=5.0?** (#2121 retest) — capacity-down + direction-smoothing could compound.
4. **Do the independent levers (EMA, Fourier, LayerScale, mesh-dropout) still help vs the new harder 68.10 baseline?** All were designed vs 80.62; the mechanism improvements are mostly orthogonal, so they should still add value.
5. **Is there a deeper OOD gap between cruise (50.71) and rc (82.24) that can be closed?** The 31.5-point gap across test splits points to a physics specialization problem.

## Plateau watch
NOT in plateau — #2090 (−15.5%) is a massive breakthrough. New strategies are flowing from this result. Cosine realignment, clip sweep, and clip stacking are all direct extensions of this breakthrough with clear mechanisms. Continue mining the clip direction before escalating.

## Next milestones
- Tanjiro #2165: grad_clip sweep to bracket optimal threshold
- Thorfinn #2166: cosine realignment to 15 epochs
- Nezuko #2121: slice_num=48 retest with clip=5.0
- All 5 in-flight independents (#2088, #2115, #2117, #2118, #2141): first results vs new 68.10 baseline
