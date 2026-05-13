# SENPAI Research State

- 2026-05-13 14:25 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=61.8457 (PR #2282 slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 61.85 = **−49.0%**.
- No directives from human researcher team yet.

## Current baseline (PR #2282 merged — slice_num=24 + grad_clip=5.0)

**test_avg/mae_surf_p = 61.8457** | val = 70.7422 (best epoch 18, schedule fully completed)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0. W&B run: evcflzgo.

Per-split: in_dist=64.56, rc=72.29, cruise=46.72, re_rand=63.82.

## Slot scan history (monotonic regularization, all wins)
| slice_num | test_avg/mae_surf_p | cruise | Δ test | Δ cruise |
|---|---|---|---|---|
| 96 (orig) | ~67+ | ~54+ | baseline | baseline |
| 48 (PR #2121) | 65.37 | 51.29 | −2.4% | −5.2% |
| 32 (PR #2226) | 62.80 | 48.79 | −3.9% | −4.9% |
| 24 (PR #2282) | 61.85 | 46.72 | **−1.5%** | **−4.2%** |
| **16 (PR #2333)** | **TBD** | **TBD** | **in progress** | **in progress** |

## Round-3 status (updated 2026-05-13 14:25)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2294 | surf_weight sweep (15/20) | **wip** |
| nezuko | #2333 | slice_num=16 + clip | **wip** (NEW) |
| fern | #2117 | EMA 0.99 on slice=32 (retest) | **wip** (SENT BACK — needs rebase to slice=24) |
| alphonse | #2326 | cosine eta_min=1.5e-5 floor | **wip** |
| askeladd | #2237 | Lion β1 sweep | **wip** |
| tanjiro | #2208 | clip threshold sweep | **wip** (status request sent) |
| thorfinn | #2303 | 1-epoch LR warmup (linear) | **wip** |
| edward | #2327 | SiLU activation swap | **wip** |
| nezuko | #2282 | slice_num=24 + clip | **MERGED** ✓ | test=61.85 (−1.52%). New best. Schedule fully completed 18/18. |
| alphonse | #2236 | n_head=8 | **CLOSED** ✗ | +10.7% (+18% per-epoch tax, cuBLAS slow path at d_head=24) |
| edward | #2258 | mlp_ratio=4 | **CLOSED** ✗ | +10.6%, two-seed (+9% per-epoch tax) |
| thorfinn | #2209 | T_max=15 cosine | **CLOSED** ✗ | +11.2% — T_max shortening lowers high-LR phase |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | accum=4 step starvation structural |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism preserved at depth=6**: budget, not stability, closes depth lever.
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num monotonic scan ONGOING**: 96→48→32→24 all improve. Floor below 24. Scan at 16.
24. **EMA 0.99 wins on old stack**: −5.29%; slice=24 confirmation pending (fern #2117 — needs rebase).
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED (#2209)**: T_max shortening lowers high-LR exploration at every epoch. Refinement-tail fixes must be orthogonal (warmup, eta_min floor).
27. **Capacity-budget structural bound CONFIRMED** (joint #2191+#2258+#2236): depth=6 (+21%), n_head=8 (+18%, cuBLAS slow path at d_head=24), mlp_ratio=4 (+9%) all closed. Any >+5% per-epoch overhead is a net loser under 30-min cap.
28. **Schedule completion bonus (slice=24)**: First full 18/18 epochs in slot scan. Smaller slice matrix → faster per-epoch → more refinement steps in budget. Dual win mechanism confirmed.

## Active experiments (8 students)

### Tier 1: Slot scan (highest expected value — compound wins every step)
| PR | Student | Expected gain |
|---|---|---|
| #2333 | nezuko | slice=16: −0.5% to −2% if floor still below 16 |

### Tier 2: EMA + loss weighting (high expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2117 | fern | EMA 0.99 on slice=24 (NEEDS REBASE): −1% to −4% compound expected |
| #2294 | frieren | surf_weight=15: −0.3% to −2% on rc/re_rand |

### Tier 3: Schedule shape (per-epoch-cost-neutral levers)
| PR | Student | Expected gain |
|---|---|---|
| #2303 | thorfinn | 1-epoch LR warmup: −0.3% to −2% |
| #2326 | alphonse | cosine eta_min=1.5e-5 floor: −0.3% to −2% |
| #2208 | tanjiro | clip threshold sweep: −0.5% to −2% |

### Tier 4: Free architectural + optimizer tuning
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation swap: −0.3% to −2% |
| #2237 | askeladd | Lion β1=0.95: −0.5% to −2% |

## Key open questions
1. **Slot floor location?** Cruise −4.24% at slice=24, still improving. Will it improve at 16?
2. **Does EMA 0.99 stack with slice=24?** The key compound experiment. Fern needs to rebase.
3. **Does surf_weight=15 help?** Fresh lever; gradient budget tilted to primary metric.
4. **Optimal clip threshold?** Awaiting tanjiro status post.
5. **What's optimal Lion β1 with clip?**
6. **Does LR warmup help Lion?** (thorfinn #2303)
7. **Does eta_min=1.5e-5 refinement tail floor help?** (alphonse #2326)
8. **Does SiLU outperform GELU for Lion+clip?** (edward #2327)

## Plateau watch
NOT in plateau. FOUR consecutive wins (slice=48, slice=32, slice=24 now merged). The slot scan is the dominant positive driver. Capacity-budget bound (finding #27) has consolidated understanding — all future architectural experiments must be per-epoch-cost-neutral. Continue mining slot floor + free-lever stack.

## IMPORTANT — fern rebase note
Fern (#2117) was assigned EMA 0.99 on slice=32. Two merges have since happened (#2226 slice=32, #2282 slice=24). The baseline is now slice=24. Fern should rebase onto current advisor branch and re-run on slice=24 stack to get the compound result. This is the highest-value pending confirmation experiment.
