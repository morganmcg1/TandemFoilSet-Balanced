# SENPAI Research State

- 2026-05-13 12:00 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=65.3734 (PR #2121 slice_num=48 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 65.37 = −46.1%.
- No directives from human researcher team yet.

## Current baseline (PR #2121 merged — slice_num=48 + grad_clip=5.0)

**test_avg/mae_surf_p = 65.3734** | val = 71.9613 (best epoch 15)
Config: bf16 + batch_size=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=48**, mlp_ratio=2 + **grad_clip_max_norm=5.0**. W&B run: vyjph01c.

Per-split: in_dist=67.70, rc=74.63, cruise=51.29, re_rand=67.87.

## Round-3 status (updated 2026-05-13 12:00)

| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #2236 | n_head=8 + clip + slice=48: attention diversification | **wip** (NEW) | Zero per-epoch cost capacity add; rc split key diagnostic |
| askeladd | #2237 | Lion β1 sweep (0.95 primary, 0.85 secondary) | **wip** (NEW) | Untested momentum recalibration under clip direction smoothing |
| nezuko | #2226 | slice_num=32 + clip=5.0 — find slot floor | **wip** | Continue capacity scan; cruise split key diagnostic |
| tanjiro | #2208 | grad-clip-sweep (max_norm=2.0/10.0/50.0) | **wip** | Bracket optimal clip threshold |
| thorfinn | #2209 | cosine-realign-epochs15 | **wip** | T_max=15 to match actual 30-min budget |
| edward | #2141 | layerscale-1e-4 | **wip** | CaiT-style per-channel residual scaling, γ_init=1e-4 |
| fern | #2117 | ema-decay-095 + clip=5.0 | **wip** (restarted) | Run ckmhwg39 in flight — fixed missing clip issue |
| frieren | #2190 | accumulation_steps=4 + clip=5.0 | **wip** | Mechanism-changed retest of step starvation |
| alphonse | #2191 | n_layers=6 + clip=5.0 | **CLOSED** ✗ | +9.24% vs new baseline. Depth lever CLOSED under 30-min budget (not clip-conditional — clip preserved mechanism). +21% per-epoch tax truncates cosine refinement. |
| askeladd | #2088 | lion-lr-2.1e-4-sqrt2 | **CLOSED** ✗ | All 3 LR arms regressed (test=85-90). Lion LR scaling CLOSED — sqrt(2) rule doesn't apply with clip bulk rescaling. |

## Key research findings (cumulative)

1. **Throughput at 30-min budget**: bf16+batch-8 → 17 epochs → round-1 win.
2. **Schedule alignment**: T_max=actual epochs → −7.67%.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%.
5. **Lion optimizer biggest lever (before clip)**: sign-momentum → −15.97% (83.77 from 99.69).
6. **Lion opens memory budget**: AdamW ~94 GB → Lion ~43 GB.
7. **Depth lever CLOSED under 30-min wall clock**: +21% per-epoch tax truncates cosine refinement. Clip mechanism preserved at depth=6 — the ceiling is budget, not stability.
8. **Gradient accumulation (accum=2) wins**: −3.77%.
9. **Width lever CLOSED**: n_hidden>192 infeasible at 30-min.
10. **EMA decay=0.999 CLOSED**: Rapid descent regime mismatch. Short-horizon 0.95 + clip being retested.
11. **Slice-num monotonic trend**: 96 regresses, 48 wins (−4% on clip baseline), 32 being tested. Mechanism: regularization via leaner slot partitioning, not capacity.
12. **Weight-decay lever CLOSED**.
13. **Fourier lever CLOSED at L=8**: 3 evidence points.
14. **grad-accum=4 CLOSED (pre-clip)**: Step starvation. Being retested with clip in #2190.
15. **DropPath CLOSED**: Underfitting.
16. **Activation-swap CLOSED (SiLU)**: +14.3%.
17. **grad_clip=5.0 MASSIVE WIN**: −15.5% (68.10 from 80.62). Bulk Lion direction rescaler.
18. **Mesh-node-dropout CLOSED**: Dense physics attention incompatible with PointNet-style dropout.
19. **slice_num=48 + clip COMPOUND WIN**: −3.99% super-additive. New best 65.3734. rc improved −9.25%.
20. **Lion LR scaling CLOSED**: sqrt(2) rule fails with clip. lr=1.5e-4 correctly calibrated for clip stack.
21. **Clip mechanism preserved at depth=6**: Closure of depth is budget-conditional (30-min hard limit), not stability-conditional.

## Active hypotheses — priority order

### Tier 1: Direct stack follow-ups (high expected value)
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2226 | nezuko | slice_num=32 + clip=5.0 | −0.5% to −2%; cruise split key diagnostic |
| #2208 | tanjiro | grad_clip sweep (2.0, 10.0, 50.0) | −0.5% to −2%; bracket optimal clip threshold |
| #2209 | thorfinn | cosine T_max=15 | −0.5% to −1.5%; low-LR refinement on new stack |

### Tier 2: Mechanism-changed retests
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2190 | frieren | accum=4 + clip=5.0 | −1% to −3%; clip-resolves-step-starvation test |

### Tier 3: New independent levers
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2236 | alphonse | n_head=8 + clip + slice=48 | −0.5% to −2%; attention diversification, zero time cost |
| #2237 | askeladd | Lion β1=0.95 (primary), 0.85 (secondary) | −0.5% to −2%; momentum recalibration under clip |
| #2141 | edward | LayerScale γ=1e-4 | −0.3% to −1.5% |
| #2117 | fern | EMA decay=0.95 + clip=5.0 | −0.3% to −1.5% |

## Key open questions
1. **Is slice_num=32 better?** (#2226) — cruise diagnostic: holds flat = floor below 32; regresses = 48 is the floor.
2. **Is clip threshold 2.0 tighter better?** (#2208) — characterizes bulk-rescaling sensitivity.
3. **Does T_max=15 help on new stack?** (#2209) — schedule alignment compounds with clip+slice.
4. **Does clip resolve accum=4 step starvation?** (#2190) — discriminating mechanism test.
5. **Does n_head=8 improve OOD generalization?** (#2236) — rc is key; zero per-epoch cost capacity.
6. **What's optimal Lion β1 with clip?** (#2237) — fresh untested optimizer lever.
7. **Why is rc (74.63) still 23 points above cruise (51.29)?** Both should benefit from capacity+regularization but rc is the harder OOD split.

## Plateau watch
NOT in plateau. Consecutive wins (#2090 −15.5%, #2121 −3.99%). Six promising experiments in flight plus two new assignments. Continue mining.

## Next milestones
- nezuko #2226: slot floor scan at 32
- tanjiro #2208: clip threshold characterization
- alphonse #2236: n_head=8 first result
- askeladd #2237: β1=0.95 primary arm
