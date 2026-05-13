# SENPAI Research State

- 2026-05-13 12:15 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=65.3734 (PR #2121 slice_num=48 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 65.37 = −46.1%.
- No directives from human researcher team yet.

## Current baseline (PR #2121 merged — slice_num=48 + grad_clip=5.0)

**test_avg/mae_surf_p = 65.3734** | val = 71.9613 (best epoch 15)
Config: bf16 + batch_size=4 + accumulation_steps=2 (eff_bs=8) + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=48**, mlp_ratio=2 + **grad_clip_max_norm=5.0**. W&B run: vyjph01c.

Per-split: in_dist=67.70, rc=74.63, cruise=51.29, re_rand=67.87.

## Round-3 status (updated 2026-05-13 12:15)

| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| edward | #2258 | mlp_ratio=4 + clip + slice=48: wider FFN | **wip** (NEW) | Capacity-add complementing n_head=8; rc OOD key diagnostic |
| alphonse | #2236 | n_head=8 + clip + slice=48: attention diversification | **wip** | Zero per-epoch cost capacity |
| askeladd | #2237 | Lion β1 sweep (0.95/0.85) + clip | **wip** | First β1 test under bulk clip rescaling |
| nezuko | #2226 | slice_num=32 + clip=5.0 | **wip** | Capacity scan; cruise diagnostic |
| tanjiro | #2208 | grad-clip-sweep (2.0/10.0/50.0) | **wip** | Bracket optimal clip threshold |
| thorfinn | #2209 | cosine T_max=15 | **wip** | Schedule alignment on new stack |
| fern | #2117 | EMA decay=0.95 + clip=5.0 | **wip** | Run ckmhwg39 in flight |
| frieren | #2190 | accumulation_steps=4 + clip=5.0 | **wip** | Step starvation mechanism retest |
| edward | #2141 | LayerScale γ=1e-4/1e-3 | **CLOSED** ✗ | Both γ regress >29%. Mechanism mismatch: depth=5 doesn't need LayerScale's init suppression; Lion+clip already stabilizes gradients. |
| alphonse | #2191 | n_layers=6 + clip=5.0 | **CLOSED** ✗ | +9.24% vs baseline. Depth closed under 30-min budget (+21% per-epoch tax). Clip mechanism intact. |
| askeladd | #2088 | lion-lr=2.1e-4 sqrt(2) sweep | **CLOSED** ✗ | All arms test=85-90. LR scaling closed. |

## Key research findings (cumulative)

1. **Throughput at 30-min budget**: bf16+batch-8 → 17 epochs.
2. **Schedule alignment**: T_max=actual epochs → −7.67%.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%.
5. **Lion optimizer**: sign-momentum → −15.97%.
6. **Lion opens memory budget**: ~43 GB vs AdamW 94 GB.
7. **Depth lever CLOSED under 30-min budget**: ceiling is wall-clock, not stability. +21% per-epoch tax eats cosine refinement. Clip intact at depth=6.
8. **Gradient accumulation (accum=2) wins**: −3.77%.
9. **Width lever CLOSED**: n_hidden>192 infeasible.
10. **EMA 0.999 CLOSED**: Horizon mismatch. Short-horizon 0.95+clip being retested.
11. **Slice-num monotonic**: 96 regresses, 48 wins (−4%). Scan continues at 32.
12. **Weight-decay CLOSED**.
13. **Fourier CLOSED at L=8**: 3 evidence points.
14. **grad-accum=4 CLOSED pre-clip**: Being retested with clip.
15. **DropPath CLOSED**.
16. **Activation-swap CLOSED (SiLU +14.3%)**.
17. **grad_clip=5.0 MASSIVE WIN**: −15.5%. Bulk Lion direction rescaler.
18. **Mesh-node-dropout CLOSED**: Dense physics attention incompatible.
19. **slice_num=48 + clip COMPOUND WIN**: −3.99% super-additive. New best 65.3734.
20. **Lion LR scaling CLOSED**: sqrt(2) rule fails with clip. lr=1.5e-4 correctly calibrated.
21. **Clip mechanism preserved at depth=6**: Budget closes depth, not stability.
22. **LayerScale CLOSED at depth=5**: γ=1e-4 and γ=1e-3 both regress >29%. Mechanism mismatch — init suppression counterproductive when Lion+clip already stabilizes directions.

## Active experiments (8 students)

### Tier 1: Direct stack follow-ups
| PR | Student | Expected gain |
|---|---|---|
| #2226 | nezuko | slice_num=32 + clip: −0.5% to −2% |
| #2208 | tanjiro | clip sweep 2.0/10.0/50.0: −0.5% to −2% |
| #2209 | thorfinn | T_max=15: −0.5% to −1.5% |

### Tier 2: Mechanism retests
| PR | Student | Expected gain |
|---|---|---|
| #2190 | frieren | accum=4 + clip: −1% to −3% |

### Tier 3: New capacity levers (complementary pair)
| PR | Student | Expected gain |
|---|---|---|
| #2236 | alphonse | n_head=8: attention diversification, zero time cost |
| #2258 | edward | mlp_ratio=4: FFN width, ~+15% time cost |

### Tier 4: Optimizer tuning
| PR | Student | Expected gain |
|---|---|---|
| #2237 | askeladd | Lion β1=0.95: momentum recalibration under clip |
| #2117 | fern | EMA 0.95 + clip: short-horizon weight averaging |

## Key open questions
1. Does slice_num=32 further reduce test? (cruise diagnostic)
2. Is clip threshold 2.0 better than 5.0?
3. Does T_max=15 help on new stack?
4. Does clip fix accum=4 step starvation?
5. Do n_head=8 and mlp_ratio=4 add capacity without budget penalty?
6. What's optimal Lion β1 with clip? (β1=0.9 inherited from Lion paper)
7. Why is the rc-cruise gap (74.63 vs 51.29) still 23 points?

## Plateau watch
NOT in plateau. Consecutive wins. 8 active experiments covering clip-stack follow-ups, mechanism retests, capacity levers, and optimizer tuning. Continue.
