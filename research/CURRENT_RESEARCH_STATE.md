# SENPAI Research State

- 2026-05-13 14:10 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=62.8014 (PR #2226 slice_num=32 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 62.80 = −48.2%.
- No directives from human researcher team yet.

## Current baseline (PR #2226 merged — slice_num=32 + grad_clip=5.0)

**test_avg/mae_surf_p = 62.8014** | val = 71.7560 (best epoch 17)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=32**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + **act=gelu** + **eta_min=0**. W&B run: 9u8p8npt.

Per-split: in_dist=64.70, rc=71.97, cruise=48.79, re_rand=65.75.

## Round-3 status (updated 2026-05-13 14:10)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2294 | surf_weight sweep (15/20) | **wip** |
| nezuko | #2282 | slice_num=24 + clip | **wip** |
| fern | #2117 | EMA 0.99 on slice=32 (retest) | **wip** (SENT BACK) |
| alphonse | #2326 | cosine eta_min=1.5e-5 floor | **wip** (NEW) |
| askeladd | #2237 | Lion β1 sweep | **wip** |
| tanjiro | #2208 | clip threshold sweep | **wip** (status request) |
| thorfinn | #2303 | 1-epoch LR warmup (linear) | **wip** |
| edward | #2327 | SiLU activation swap | **wip** (NEW) |
| alphonse | #2236 | n_head=8 | **CLOSED** ✗ | test=69.53 (+10.7%). +18% per-epoch tax → schedule truncation. d_head=24 too narrow for slot-routing physics. |
| edward | #2258 | mlp_ratio=4 | **CLOSED** ✗ | test=69.48 (+10.6%, two-seed). +9% per-epoch tax → schedule truncation. |
| thorfinn | #2209 | T_max=15 cosine | **CLOSED** ✗ | test=69.86 (+11.2%). |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | accum=4 step starvation structural. |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism preserved at depth=6**: budget, not stability, closes depth lever.
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num monotonic scan**: floor below 32 (cruise improves at every reduction). Scan at 24.
24. **EMA 0.99 wins on old stack**: −5.29%; slice=32 confirmation pending (fern #2117).
25. **accum=4 PERMANENTLY CLOSED**: clip does not fix step starvation. accum=2 optimal.
26. **T_max sweep CLOSED (#2209)**: shortening cosine T_max lowers LR at every epoch, not just deepens the tail. Refinement tail extensions need orthogonal mechanisms (warmup, eta_min floor), not T_max shortening.
27. **Capacity-budget structural bound CONFIRMED** (joint #2191 + #2258 + #2236): three independent capacity-adding interventions all failed because of schedule truncation under 30-min cap. Pattern: any per-epoch overhead >+5% has been a net loser. depth=6 (+21% tax), n_head=8 (+18% tax, cuBLAS kernel path issue at d_head=24), mlp_ratio=4 (+9% tax) all closed. **Remaining experimental directions must be per-epoch-cost-neutral or negative.**

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

### Tier 3: Schedule / threshold (per-epoch-cost neutral levers)
| PR | Student | Expected gain |
|---|---|---|
| #2303 | thorfinn | 1-epoch LR warmup (Lion init smoothing): −0.3% to −2% |
| #2326 | alphonse | cosine eta_min=1.5e-5 floor (refinement tail): −0.3% to −2% (NEW) |
| #2208 | tanjiro | clip threshold sweep: −0.5% to −2% (status request sent) |

### Tier 4: Free architectural changes (per-epoch-cost neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation swap (GELU → SiLU): −0.3% to −2% (NEW) |

### Tier 5: Optimizer tuning
| PR | Student | Expected gain |
|---|---|---|
| #2237 | askeladd | Lion β1=0.95: −0.5% to −2% |

## Key open questions
1. **Slot floor location?** Cruise −4.87% at slice=32, still improving. Will it improve at 24?
2. **Does EMA 0.99 stack with slice=32?** Key compound experiment. Expected ~59-60.
3. **Does surf_weight=15 help?** Fresh lever; gradient budget tilted to primary metric.
4. **Optimal clip threshold?** Is 2.0 better than 5.0? (Awaiting tanjiro completion + status post.)
5. **What's optimal Lion β1 with clip?**
6. **Does LR warmup help Lion?** Smoother initialization at peak LR start.
7. **Does eta_min refinement-tail floor help?** (NEW) Refinement-tail lever directly orthogonal to #2209 closure.
8. **Does SiLU outperform GELU for Lion+clip?** (NEW) Free activation lever; Lion paper used SiLU.

## Plateau watch
NOT in plateau. Three consecutive wins (slice=48, slice=32). Five discriminating negatives (#2117 → #2236 capacity-budget bound now formalized). The capacity-budget pattern is the strongest insight of round 3 — it constrains all future architecture work to per-epoch-cost-neutral directions. Remaining direction-space: schedule shape (eta_min, warmup), activation choice, loss weighting, slot floor, EMA, β1 tuning, clip threshold. Continue mining.
