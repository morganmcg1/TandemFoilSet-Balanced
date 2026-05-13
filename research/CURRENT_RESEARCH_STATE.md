# SENPAI Research State

- 2026-05-13 15:30 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=61.8457 (PR #2282 slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 61.85 = **−49.0%**.
- No directives from human researcher team yet.

## Current baseline (PR #2282 merged — slice_num=24 + grad_clip=5.0)

**test_avg/mae_surf_p = 61.8457** | val = 70.7422 (best epoch 18, schedule fully completed)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + β2=0.99 + wd=1e-4 + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0. W&B run: evcflzgo.

Per-split: in_dist=64.56, rc=72.29, cruise=46.72, re_rand=63.82.

## Slot scan history (monotonic regularization, all wins)
| slice_num | test_avg/mae_surf_p | cruise | Δ test | Δ cruise |
|---|---|---|---|---|
| 96 (orig) | ~67+ | ~54+ | baseline | baseline |
| 48 (PR #2121) | 65.37 | 51.29 | −2.4% | −5.2% |
| 32 (PR #2226) | 62.80 | 48.79 | −3.9% | −4.9% |
| 24 (PR #2282) | 61.85 | 46.72 | **−1.5%** | **−4.2%** |
| **16 (PR #2333)** | **TBD** | **TBD** | **in progress** | **in progress** |

## Round-3 status (updated 2026-05-13 15:30)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2294 | surf_weight sweep (15/20) | **wip** |
| nezuko | #2333 | slice_num=16 + clip | **wip** |
| fern | #2117 | EMA 0.99 on slice=24 (retest) | **wip** (SENT BACK — needs rebase) |
| alphonse | #2326 | cosine eta_min=1.5e-5 floor | **wip** |
| askeladd | #2382 | Lion β2=0.999/0.95 sweep | **wip** (NEW) |
| tanjiro | #2344 | attention dropout=0.1 | **wip** |
| thorfinn | #2343 | weight decay ablation (wd=0) | **wip** |
| edward | #2327 | SiLU activation swap | **wip** |
| nezuko | #2282 | slice_num=24 + clip | **MERGED** ✓ | test=61.85 (−1.52%). New best. |
| askeladd | #2237 | Lion β1 sweep (0.95/0.85) | **CLOSED** ✗ | β1=0.9 default well-calibrated. Best arm (β1=0.95 retry) test=64.97 (+5.06% vs 61.85). All arms on slice=48 stack. Inter-seed variance ~1.4% rel established as noise floor. |
| thorfinn | #2303 | 1-epoch LR warmup | **CLOSED** ✗ | +7.3%. Per-epoch LinearLR = step function; epoch-1 grad-norm spike 114.9. Warmup lever needs per-iteration implementation, not per-epoch. |
| tanjiro | #2208 | clip threshold sweep (2/5/10/50) | **CLOSED** ✗ | Best arm clip=2.0 test=67.60 (+9.3% vs 61.85). All arms on old stack; no arm beats current best. Clip mechanism fully characterized: bulk bulk rescaling [2,5] optimal, clip=5.0 is safe-robust. |
| alphonse | #2236 | n_head=8 | **CLOSED** ✗ | +10.7% (+18% per-epoch tax, cuBLAS slow path d_head=24) |
| edward | #2258 | mlp_ratio=4 | **CLOSED** ✗ | +10.6%, two-seed (+9% per-epoch tax) |
| thorfinn | #2209 | T_max=15 cosine | **CLOSED** ✗ | +11.2% |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | accum=4 step starvation structural |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism fully characterized** (joint #2090+#2208): bulk Lion direction rescaling in [2,5] optimal. clip=5.0 is the robust choice (clip=2.0 improvement not statistically significant, high variance). clip=50.0 = essentially no-op (only tail clipping). **Fire-rate data confirmed: mechanism is bulk rescaling, not tail trimming.**
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num monotonic scan ONGOING**: 96→48→32→24 all improve. Floor below 24. Scan at 16.
24. **EMA 0.99 wins on old stack**: −5.29%; slice=24 confirmation pending (fern #2117 — needs rebase).
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED (#2209)**: T_max shortening lowers high-LR exploration at every epoch.
27. **Capacity-budget structural bound CONFIRMED** (joint #2191+#2258+#2236): any >+5% per-epoch overhead is a net loser under 30-min cap. depth=6 (+21%), n_head=8 (+18%, d_head=24 cuBLAS slow path), mlp_ratio=4 (+9%). **Remaining experiments must be per-epoch-cost-neutral.**
28. **Schedule completion bonus (slice=24)**: Smaller slice matrix → faster per-epoch → full 18/18 schedule completed for first time. Dual win: regularization + budget.
29. **Per-epoch warmup = step function (CLOSED)**: LinearLR(total_iters=1, step_per_epoch) is NOT a smooth ramp — it's a cold epoch + hard LR step-jump. Warmup must be implemented per-iteration (not per-epoch) to be genuine. Epoch-1 grad-norm spike 114.9 confirmed the bug.
30. **Weight decay untested** (NEW): Lion wd=1e-4 has never been tuned on this stack. At slice_num=24 locality regularization, L2 shrinkage may be redundant. Testing wd=0 vs 1e-4.
31. **Attention dropout untested** (NEW): PhysicsAttention has dropout=0.0 by default. Testing dropout=0.1 as OOD regularizer for rc/re_rand splits.
32. **Inter-seed variance noise floor ~1.4% rel** (askeladd #2237 two-seed at β1=0.95): Two seeds differed by 64.97→65.90 = 1.43% rel test_avg/mae_surf_p. Single-seed differences <1.4% are indistinguishable from seed noise. Multi-seed required for sub-noise-floor gain attribution.
33. **Lion β1 lever CLOSED** (#2237): β1=0.9 optimal within measured inter-seed variance at slice_num=48 + clip=5.0. Range [0.85, 0.95] fully characterized. β1=0.85 hurts (clip fire-rate 92.6%, grad-norm 22.8 vs 11.1) due to momentum-memory instability when clip already smooths magnitude. β2 (persistent EMA state) is the remaining untested Lion parameter.

## Active experiments (8 students)

### Tier 1: Slot scan (highest expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2333 | nezuko | slice=16: −0.5% to −2% if floor still below 16 |

### Tier 2: EMA + loss weighting (high expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2117 | fern | EMA 0.99 on slice=24 (NEEDS REBASE): −1% to −4% compound |
| #2294 | frieren | surf_weight=15: −0.3% to −2% |

### Tier 3: Schedule + regularization levers (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2326 | alphonse | cosine eta_min=1.5e-5 floor: −0.3% to −2% |
| #2343 | thorfinn | weight decay wd=0 ablation: −0.3% to −1.5% |
| #2344 | tanjiro | attention dropout=0.1: −0.3% to −2% (rc/re_rand focus) |

### Tier 4: Optimizer + architectural (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation swap: −0.3% to −2% |
| #2382 | askeladd | Lion β2=0.999/0.95 sweep: −0.3% to −2% (direction-memory horizon) |

## Key open questions
1. **Slot floor?** Cruise −4.24% at slice=24, still improving. Will it improve at 16?
2. **Does EMA 0.99 stack with slice=24?** Fern needs rebase. Expected ~58-60 if stacks.
3. **Does surf_weight=15 help?** Loss weighting lever.
4. **Does weight decay impede Lion+slice regularization?** (thorfinn #2343)
5. **Does attention dropout help rc/re_rand OOD?** (tanjiro #2344)
6. **Does eta_min refinement tail help?** (alphonse #2326)
7. **Does SiLU outperform GELU?** (edward #2327)
8. **What's optimal Lion β2?** β2 controls the persistent EMA state's memory horizon. β2=0.999 (longer) is the natural test on the current clip-stabilized stack. (askeladd #2382)

## Plateau watch
NOT in plateau. FOUR consecutive wins (slice scan). Capacity-budget bound fully formalized. Clip mechanism fully characterized. Warmup lever closed (per-epoch bug). Lion β1 lever closed (β1=0.9 optimal, noise floor established at ~1.4% rel). All 8 students have active per-epoch-cost-neutral experiments. Continue mining.

## IMPORTANT — fern rebase note
Fern (#2117) must rebase onto current advisor branch before re-running. The baseline is now slice=24. The EMA compound (expected ~58-60) is the highest-value pending experiment.
