# SENPAI Research State

- 2026-05-13 15:45 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=61.8457 (PR #2282 slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 61.85 = **−49.0%**.
- No directives from human researcher team yet.

## Current baseline (PR #2282 merged — slice_num=24 + grad_clip=5.0)

**test_avg/mae_surf_p = 61.8457** | val = 70.7422 (best epoch 18, schedule fully completed)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + β2=0.99 + wd=1e-4 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0. W&B run: evcflzgo.

Per-split: in_dist=64.56, rc=72.29, cruise=46.72, re_rand=63.82.

## Slot scan history (complete — floor found at slice_num=24)
| slice_num | test_avg/mae_surf_p | cruise | Δ test | Δ cruise |
|---|---|---|---|---|
| 96 (orig) | ~67+ | ~54+ | baseline | baseline |
| 48 (PR #2121) | 65.37 | 51.29 | −2.4% | −5.2% |
| 32 (PR #2226) | 62.80 | 48.79 | −3.9% | −4.9% |
| 24 (PR #2282) | **61.85** | **46.72** | **−1.5%** | **−4.2%** ← FLOOR |
| 16 (PR #2333) | 63.01 | 48.14 | +1.9% (regression) | +3.0% (regression) |

**Slot scan lever CLOSED. slice_num=24 is optimal.**

## Round-3 status (updated 2026-05-13 15:45)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2294 | surf_weight sweep (15/20) | **wip** (STALE — status check sent) |
| nezuko | #2393 | Fourier L=12/4 sweep | **wip** (NEW) |
| fern | #2117 | EMA 0.99 on slice=24 (retest) | **wip** (SENT BACK — needs rebase) |
| alphonse | #2326 | cosine eta_min=1.5e-5 floor | **wip** |
| askeladd | #2382 | Lion β2=0.999/0.95 sweep | **wip** (NEW) |
| tanjiro | #2344 | attention dropout=0.1 | **wip** |
| thorfinn | #2343 | weight decay ablation (wd=0) | **wip** |
| edward | #2327 | SiLU activation swap | **wip** |
| nezuko | #2333 | slice_num=16 + clip | **CLOSED** ✗ | cruise +3.04% regression. Slot floor found at slice_num=24. |
| askeladd | #2237 | Lion β1 sweep (0.95/0.85) | **CLOSED** ✗ | β1=0.9 default optimal within ~1.4% inter-seed variance. |
| nezuko | #2282 | slice_num=24 + clip | **MERGED** ✓ | test=61.85 (−1.52%). New best. |
| thorfinn | #2303 | 1-epoch LR warmup | **CLOSED** ✗ | +7.3%. Per-epoch LinearLR = step function. |
| tanjiro | #2208 | clip threshold sweep (2/5/10/50) | **CLOSED** ✗ | Clip mechanism fully characterized. clip=5.0 optimal. |
| alphonse | #2236 | n_head=8 | **CLOSED** ✗ | +10.7% (+18% per-epoch tax) |
| edward | #2258 | mlp_ratio=4 | **CLOSED** ✗ | +10.6% (+9% per-epoch tax) |
| thorfinn | #2209 | T_max=15 cosine | **CLOSED** ✗ | +11.2% |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | accum=4 step starvation structural |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism fully characterized** (joint #2090+#2208): bulk Lion direction rescaling in [2,5] optimal. clip=5.0 robust-safe.
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num monotonic scan COMPLETE**: 96→48→32→24 all improve; 24→16 regressed. **Floor confirmed at slice_num=24.**
24. **EMA 0.99 wins on old stack**: −5.29%; slice=24 confirmation pending (fern #2117 — needs rebase).
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED (#2209)**: T_max shortening lowers high-LR exploration at every epoch.
27. **Capacity-budget structural bound CONFIRMED** (joint #2191+#2258+#2236): any >+5% per-epoch overhead is net loser under 30-min cap.
28. **Schedule completion bonus (slice=24)**: Smaller slice matrix → faster per-epoch → full 18/18 schedule for first time.
29. **Per-epoch warmup = step function (CLOSED)**: Warmup must be per-iteration. Epoch-1 grad-norm spike 114.9 confirmed the bug.
30. **Weight decay untested**: Testing wd=0 vs wd=1e-4 (thorfinn #2343).
31. **Attention dropout untested**: Testing dropout=0.1 as OOD regularizer (tanjiro #2344).
32. **Inter-seed variance noise floor ~1.4% rel** (askeladd #2237): Single-seed differences <1.4% require multi-seed confirmation.
33. **Lion β1 lever CLOSED** (#2237): β1=0.9 optimal within noise floor. β1=0.85 hurts (clip fire-rate 92.6%, grad-norm 22.8). β2 is the remaining untested Lion parameter (askeladd #2382).
34. **Locality-prior OOD tradeoff confirmed** (#2333 slot scan closure): slice_num lever trades OOD generalization, not in-dist accuracy. in_dist was least sensitive across entire 96→16 scan. cruise/re_rand move first and largest — use as primary evaluation split for future capacity-tuning experiments.

## Active experiments (8 students)

### Tier 1: EMA + loss weighting (high expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2117 | fern | EMA 0.99 on slice=24 (NEEDS REBASE): −1% to −4% compound |
| #2294 | frieren | surf_weight=15/20: −0.3% to −2% (STALE — status check sent) |

### Tier 2: Schedule + regularization levers (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2326 | alphonse | cosine eta_min=1.5e-5 floor: −0.3% to −2% |
| #2343 | thorfinn | weight decay wd=0 ablation: −0.3% to −1.5% |
| #2344 | tanjiro | attention dropout=0.1: −0.3% to −2% (rc/re_rand focus) |

### Tier 3: Architecture + optimizer (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation swap: −0.3% to −2% |
| #2382 | askeladd | Lion β2=0.999/0.95 sweep: −0.3% to −2% |
| #2393 | nezuko | Fourier L=12/4 sweep: −0.3% to −2% |

## Key open questions
1. **Does EMA 0.99 stack with slice=24?** Fern needs rebase. Expected ~58-60 if stacks. Highest-EV pending experiment.
2. **Does surf_weight=15 help?** Loss weighting lever. Frieren stale — awaiting report.
3. **Does weight decay impede Lion+slice regularization?** (thorfinn #2343)
4. **Does attention dropout help rc/re_rand OOD?** (tanjiro #2344)
5. **Does eta_min refinement tail help?** (alphonse #2326)
6. **Does SiLU outperform GELU?** (edward #2327)
7. **What's optimal Lion β2?** β2=0.999 vs 0.99 vs 0.95 on slice=24 stack. (askeladd #2382)
8. **Does Fourier L=12 improve positional resolution for slot routing?** (nezuko #2393)

## Plateau watch
NOT in plateau. Four consecutive wins (slot scan). Two levers CLOSED this cycle (slot scan at slice=16, Lion β1). Eight students active. EMA compound (fern #2117) is the highest-expected-value pending result. Continue mining.

## IMPORTANT — fern rebase note
Fern (#2117) must rebase onto current advisor branch before re-running. Baseline is slice=24 (PR #2282). The EMA compound (expected ~58-60 if stacks) is the highest-value pending experiment in the fleet.

## Lever status summary
| Lever | Status | Best value |
|---|---|---|
| Learning rate | CLOSED | lr=1.5e-4 |
| Optimizer | Active: Lion | β1=0.9 closed, β2 in-flight |
| Grad clip | CLOSED | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED (capacity) | n_layers=5 |
| n_head | CLOSED (capacity) | n_head=4 |
| slice_num | **CLOSED — floor at 24** | slice_num=24 |
| mlp_ratio | CLOSED (capacity) | mlp_ratio=2 |
| T_max cosine | CLOSED | T_max=18 (full schedule) |
| Warmup | CLOSED (needs per-iter impl) | — |
| accum=4 | CLOSED | accum=2 |
| EMA | PENDING (fern #2117) | 0.99 on slice=24 |
| surf_weight | PENDING (frieren #2294) | — |
| eta_min | PENDING (alphonse #2326) | — |
| weight_decay | PENDING (thorfinn #2343) | — |
| dropout | PENDING (tanjiro #2344) | — |
| activation | PENDING (edward #2327) | — |
| Fourier L | PENDING (nezuko #2393) | — |
