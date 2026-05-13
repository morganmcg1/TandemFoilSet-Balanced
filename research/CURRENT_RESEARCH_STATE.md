# SENPAI Research State

- 2026-05-13 13:15 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=62.8014 (PR #2226 slice_num=32 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 62.80 = −48.2%.
- No directives from human researcher team yet.

## Current baseline (PR #2226 merged — slice_num=32 + grad_clip=5.0)

**test_avg/mae_surf_p = 62.8014** | val = 71.7560 (best epoch 17)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=32**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10. W&B run: 9u8p8npt.

Per-split: in_dist=64.70, rc=71.97, cruise=48.79, re_rand=65.75.

## Round-3 status (updated 2026-05-13 13:15)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2294 | surf_weight sweep (15/20) | **wip** (NEW) |
| nezuko | #2282 | slice_num=24 + clip | **wip** |
| fern | #2117 | EMA 0.99 on slice=32 (retest) | **wip** (SENT BACK) |
| alphonse | #2236 | n_head=8 | **wip** |
| askeladd | #2237 | Lion β1 sweep | **wip** |
| tanjiro | #2208 | clip threshold sweep | **wip** |
| thorfinn | #2209 | T_max=15 cosine | **wip** |
| edward | #2258 | mlp_ratio=4 | **wip** |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | All 3 arms +22-26%. accum=4 step starvation is structural, not clip-conditional. Discriminating negative. |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism preserved at depth=6**: budget, not stability, closes depth lever.
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num monotonic scan**: floor below 32 (cruise improves at every reduction). Scan at 24.
24. **EMA 0.99 wins on old stack**: −5.29%; slice=32 confirmation pending (fern #2117).
25. **accum=4 PERMANENTLY CLOSED**: clip does not fix step starvation. Gain at accum=2 was direction-smoothing at micro-batch level, NOT effective-batch-size benefit. accum=2 is the optimal accumulation.

## Active experiments (8 students)

### Tier 1: Slot scan + EMA confirmation (highest expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2282 | nezuko | slice=24: cruise diagnostic for floor; −0.5% to −2% |
| #2117 | fern | EMA 0.99 on slice=32: −1% to −4% if stacks (exp ~59-60) |

### Tier 2: Loss weighting (new direct lever)
| PR | Student | Expected gain |
|---|---|---|
| #2294 | frieren | surf_weight=15: −0.3% to −2% on rc/re_rand |

### Tier 3: Schedule / threshold
| PR | Student | Expected gain |
|---|---|---|
| #2209 | thorfinn | T_max=15: −0.5% to −1.5% |
| #2208 | tanjiro | clip threshold sweep: −0.5% to −2% |

### Tier 4: Capacity levers
| PR | Student | Expected gain |
|---|---|---|
| #2236 | alphonse | n_head=8: −0.5% to −2% |
| #2258 | edward | mlp_ratio=4: −0.5% to −2% |

### Tier 5: Optimizer tuning
| PR | Student | Expected gain |
|---|---|---|
| #2237 | askeladd | Lion β1=0.95: −0.5% to −2% |

## Key open questions
1. **Slot floor location?** Cruise −4.87% at slice=32, still improving. Will it improve at 24?
2. **Does EMA 0.99 stack with slice=32?** The key compound experiment. Expected ~59-60.
3. **Does surf_weight=15 help?** Fresh lever; gradient budget tilted to primary metric.
4. **Optimal clip threshold?** Is 2.0 better than 5.0?
5. **Does n_head=8 or mlp_ratio=4 add capacity?**
6. **What's optimal Lion β1 with clip?**

## Plateau watch
NOT in plateau. Three consecutive wins. accum=4 closure strengthens understanding of clip mechanism. Continue mining.
