# SENPAI Research State

- 2026-05-13 16:15 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=61.8457 (PR #2282 slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 61.85 = **−49.0%**.
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
| 16 (PR #2333) | 63.01 | 48.14 | +1.9% ✗ | +3.0% ✗ |

**Slot scan lever CLOSED. slice_num=24 is optimal.**

## Round-3 status (updated 2026-05-13 16:15)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **wip** (NEW) |
| nezuko | #2393 | Fourier L=12/4 sweep | **wip** |
| fern | #2117 | EMA 0.99 on slice=24 (retest) | **wip** (SENT BACK — needs rebase) |
| alphonse | #2326 | cosine eta_min=1.5e-5 floor | **wip** |
| askeladd | #2382 | Lion β2=0.999/0.95 sweep | **wip** |
| tanjiro | #2425 | LayerNorm → RMSNorm swap | **wip** (NEW) |
| thorfinn | #2343 | weight decay ablation (wd=0) | **wip** |
| edward | #2327 | SiLU activation swap | **wip** |
| tanjiro | #2344 | attention dropout=0.1 | **CLOSED** ✗ | cruise +2.86%. Capacity-limited not overfitting-limited. Locality regularization incompatible with stochastic attention (finding #36). |
| frieren | #2294 | surf_weight=15/20 | **CLOSED** ✗ | rc +3.42 dominated. Split-asymmetric effect: sw↑ helps in_dist+cruise, hurts rc. Sweep redirected down (finding #37). |
| nezuko | #2333 | slice_num=16 + clip | **CLOSED** ✗ | cruise +3.04% → slot floor at slice_num=24. |
| askeladd | #2237 | Lion β1 sweep (0.95/0.85) | **CLOSED** ✗ | β1=0.9 optimal within ~1.4% noise floor. |
| nezuko | #2282 | slice_num=24 + clip | **MERGED** ✓ | test=61.85 (−1.52%). New best. |
| thorfinn | #2303 | 1-epoch LR warmup | **CLOSED** ✗ | +7.3%. Per-epoch warmup = step function bug. |
| tanjiro | #2208 | clip threshold sweep | **CLOSED** ✗ | Clip mechanism fully characterized. clip=5.0 optimal. |
| alphonse | #2236 | n_head=8 | **CLOSED** ✗ | +10.7% (+18% per-epoch tax) |
| edward | #2258 | mlp_ratio=4 | **CLOSED** ✗ | +10.6% (+9% per-epoch tax) |
| thorfinn | #2209 | T_max=15 cosine | **CLOSED** ✗ | +11.2% |
| frieren | #2190 | accum=4 + clip | **CLOSED** ✗ | accum=4 step starvation |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correctly calibrated.
21. **Clip mechanism fully characterized** (joint #2090+#2208): clip=5.0 in optimal bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: γ init suppression counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24. 96→48→32→24 all improve; 24→16 regressed.
24. **EMA 0.99 wins on old stack**: −5.29%; slice=24 confirmation pending (fern #2117 — needs rebase).
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED**: T_max=18 (full schedule) optimal.
27. **Capacity-budget structural bound CONFIRMED**: any >+5% per-epoch overhead is net loser under 30-min cap.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule for first time.
29. **Per-epoch warmup = step function (CLOSED)**: needs per-iteration implementation.
30. **Weight decay untested**: Testing wd=0 vs wd=1e-4 (thorfinn #2343).
31. **Attention dropout CLOSED** (#2344): Dropout interferes with locality regularization. Slot routing wants determinism. Finding #36.
32. **Inter-seed variance noise floor ~1.4% rel** (#2237): Single-seed differences <1.4% require multi-seed confirmation.
33. **Lion β1 lever CLOSED** (#2237): β1=0.9 optimal. β2 in-flight (askeladd #2382).
34. **Locality-prior OOD tradeoff** (#2333): slice_num lever trades OOD generalization, not in-dist accuracy.
35. **Capacity-limited regime confirmed** (#2344): 1.47M params + 18 epochs = capacity-limited, NOT overfitting-limited. Best epoch unchanged at 18 with dropout. Standard regularizers don't transfer.
36. **Locality regularization incompatible with stochastic attention** (#2344): Slot routing at slice=24 needs deterministic attention-weight paths. Dropout at the attention-weight level disrupts the geometric basis of slot allocation.
37. **surf_weight has split-asymmetric effects** (#2294): sw↑ → better in_dist+cruise, worse rc. Mechanism: more surface weight → more surface-specialized → less volumetric context → worse geometry-camber OOD. Optimum is BELOW sw=10. Testing sw=5/7 on slice=24 (#2426).

## Active experiments (8 students)

### Tier 1: EMA (highest expected value — pending for weeks)
| PR | Student | Expected gain |
|---|---|---|
| #2117 | fern | EMA 0.99 on slice=24 (NEEDS REBASE): −1% to −4% compound |

### Tier 2: Loss weighting (high expected value from asymmetry finding)
| PR | Student | Expected gain |
|---|---|---|
| #2426 | frieren | surf_weight DOWN (sw=5/7): rc improvement expected; net −0.5% to −2% |

### Tier 3: Schedule + optimizer levers (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2326 | alphonse | cosine eta_min=1.5e-5 floor: −0.3% to −2% |
| #2343 | thorfinn | weight decay wd=0 ablation: −0.3% to −1.5% |
| #2382 | askeladd | Lion β2=0.999/0.95 sweep: −0.3% to −2% |

### Tier 4: Architecture (per-epoch-cost-neutral)
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation swap: −0.3% to −2% |
| #2393 | nezuko | Fourier L=12/4 sweep: −0.3% to −2% |
| #2425 | tanjiro | LayerNorm → RMSNorm swap: −0.3% to −2% |

## Key open questions
1. **Does EMA 0.99 stack with slice=24?** Fern needs rebase. Expected ~58-60 if stacks. Highest-EV pending.
2. **Does surf_weight=5/7 improve rc generalization?** (frieren #2426) — asymmetry finding points strongly downward.
3. **Does weight decay impede Lion+slice regularization?** (thorfinn #2343)
4. **Does eta_min refinement tail help?** (alphonse #2326)
5. **Does SiLU outperform GELU?** (edward #2327)
6. **What's optimal Lion β2?** (askeladd #2382)
7. **Does Fourier L=12 improve positional resolution for slot routing?** (nezuko #2393)
8. **Does RMSNorm improve gradient flow over LayerNorm?** (tanjiro #2425)

## Plateau watch
NOT in plateau. Four consecutive wins (slot scan). Multiple levers closed this cycle (dropout, surf_weight↑, slot scan, β1). New findings: capacity-limited regime (#35), locality-regularization constraint (#36), surf_weight asymmetry (#37). Eight students active. EMA compound (fern #2117) is highest-EV. Continue mining.

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
| T_max cosine | CLOSED | T_max=18 (full) |
| Warmup | CLOSED (needs per-iter impl) | — |
| Attention dropout | **CLOSED — capacity-limited** | 0.0 (no dropout) |
| surf_weight | PARTIAL — sw↑ closed; testing sw↓ | #2426 (sw=5/7) |
| EMA | PENDING (fern #2117) | 0.99 on slice=24 |
| eta_min | PENDING (alphonse #2326) | — |
| weight_decay | PENDING (thorfinn #2343) | — |
| activation | PENDING (edward #2327) | — |
| Fourier L | PENDING (nezuko #2393) | — |
| norm type | PENDING (tanjiro #2425) | — |
