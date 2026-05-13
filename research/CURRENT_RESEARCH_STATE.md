# SENPAI Research State

- 2026-05-13 16:30 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=60.7447 (PR #2343 wd=0 + slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 60.74 = **−49.9%**.
- No directives from human researcher team yet.

## Current baseline (PR #2343 merged — wd=0)

**test_avg/mae_surf_p = 60.7447** | val = 69.3303 (best epoch 18, schedule fully completed)
Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + β1=0.9 + β2=0.99 + **wd=0** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0. W&B run: rxid6958.

Per-split: in_dist=62.37, rc=70.92, cruise=46.91, re_rand=62.78.

**Reproduce:** `cd target/ && python train.py --batch_size 4 --accumulation_steps 2 --grad_clip_max_norm 5.0 --weight_decay 0.0`

⚠️ ALL future run commands must include `--weight_decay 0.0` (not yet the default in train.py).

## Win history (this round)
| PR | Config change | test_avg | Δ | Cumulative |
|---|---|---|---|---|
| #2282 | slice_num=24 | 61.8457 | −1.52% | −49.0% |
| **#2343** | **wd=0** | **60.7447** | **−1.78%** | **−49.9%** |

## Slot scan history (complete — floor at slice_num=24)
| slice_num | test_avg/mae_surf_p | cruise | Δ test | Δ cruise |
|---|---|---|---|---|
| 24 (PR #2282) | 61.85 | 46.72 | — | — |
| 16 (PR #2333) | 63.01 | 48.14 | +1.9% ✗ | +3.0% ✗ |

**Slot scan lever CLOSED.**

## Round-3 status (updated 2026-05-13 16:30)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| thorfinn | #2433 | Per-iter warmup (correct SequentialLR impl) | **wip** (NEW) |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **wip** |
| nezuko | #2393 | Fourier L=12/4 sweep | **wip** |
| fern | #2117 | EMA 0.99 on slice=24 | **wip** (SENT BACK — needs rebase) |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **wip** (STALE — status check sent, updated run cmd) |
| askeladd | #2382 | Lion β2=0.999/0.95 sweep | **wip** |
| tanjiro | #2425 | LayerNorm → RMSNorm swap | **wip** |
| edward | #2327 | SiLU activation swap | **wip** (STALE — status check sent, updated run cmd) |
| thorfinn | #2343 | wd=0 ablation | **MERGED** ✓ | test=60.7447 (−1.78%). New best. |
| tanjiro | #2344 | attention dropout=0.1 | **CLOSED** ✗ | Capacity-limited not overfitting. Locality incompatible with stochastic attention. |
| frieren | #2294 | surf_weight=15/20 | **CLOSED** ✗ | rc +3.42 dominated. Split-asymmetric: sw↑ hurts rc. Optimum is below 10. |
| nezuko | #2333 | slice_num=16 | **CLOSED** ✗ | Slot floor at slice_num=24. |
| askeladd | #2237 | Lion β1 sweep | **CLOSED** ✗ | β1=0.9 optimal within noise floor. |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correct.
21. **Clip mechanism fully characterized**: clip=5.0 in bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24.
24. **EMA 0.99 wins on old stack**: slice=24 confirmation pending (fern #2117).
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED**: T_max=18 optimal.
27. **Capacity-budget structural bound**: >+5% per-epoch overhead = net loser.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule.
29. **Per-epoch warmup = step function (CLOSED BUG)**: Warmup must be per-iteration.
30. **Attention dropout CLOSED**: Slot routing needs determinism (finding #36).
31. **Inter-seed variance noise floor ~1.4% rel**: Multi-seed needed for sub-noise differences.
32. **Lion β1 lever CLOSED**: β1=0.9 optimal. β2 in-flight.
33. **Locality-prior OOD tradeoff**: slice_num lever trades OOD, not in-dist accuracy.
34. **Capacity-limited regime** (#2344): 1.47M+18 epochs = capacity-limited, not overfitting.
35. **Locality regularization incompatible with stochastic attention** (#2344): Dropout disrupts slot routing.
36. **surf_weight has split-asymmetric effects** (#2294): sw↑ hurts rc via reduced volumetric context. Optimum below 10.
37. **Lion wd=0 optimal** (#2343): L2 weight decay redundant at slice=24+clip=5. Late-epoch grad-norm LOWER without wd — confirms L2 was competing with locality prior. NEW BASELINE.

## Active experiments (8 students)

### Tier 1: EMA (highest expected value)
| PR | Student | Expected gain |
|---|---|---|
| #2117 | fern | EMA 0.99 on slice=24+wd=0 (NEEDS REBASE): −1% to −4% |

### Tier 2: High-EV levers
| PR | Student | Expected gain |
|---|---|---|
| #2433 | thorfinn | Per-iteration warmup (correct): −0.5% to −2% |
| #2426 | frieren | surf_weight DOWN (sw=5/7): rc improvement expected |

### Tier 3: Schedule + optimizer
| PR | Student | Expected gain |
|---|---|---|
| #2326 | alphonse | cosine eta_min=1.5e-5 (STALE, pinged) |
| #2382 | askeladd | Lion β2=0.999/0.95 |

### Tier 4: Architecture
| PR | Student | Expected gain |
|---|---|---|
| #2327 | edward | SiLU activation (STALE, pinged) |
| #2393 | nezuko | Fourier L=12/4 |
| #2425 | tanjiro | LayerNorm → RMSNorm |

## Key open questions
1. **Does EMA 0.99 stack with wd=0+slice=24?** Fern needs rebase. Expected ~57-59 if stacks.
2. **Does per-iteration warmup help Lion momentum bootstrap?** (thorfinn #2433)
3. **Does surf_weight=5/7 improve rc generalization?** (frieren #2426)
4. **Does Lion β2 affect direction-memory quality?** (askeladd #2382)
5. **Does Fourier L=12 improve OOD via better positional resolution?** (nezuko #2393)
6. **Does RMSNorm improve gradient flow?** (tanjiro #2425)
7. **Does SiLU outperform GELU with Lion?** (edward #2327, stale)
8. **Does eta_min refinement floor help?** (alphonse #2326, stale)

## IMPORTANT NOTES
- **All new/updated run commands must include `--weight_decay 0.0`** — wd=0 is the new best config but NOT yet the default in train.py
- **Fern #2117** needs rebase onto current advisor branch — EMA compound is highest-EV pending
- **Edward #2327 and alphonse #2326** are stale (2.5h+, no activity) — status checks sent with updated baseline and `--weight_decay 0.0` flag
- Baseline has shifted twice this session: 61.8457 → 60.7447

## Lever status summary
| Lever | Status | Best value |
|---|---|---|
| Learning rate | CLOSED | lr=1.5e-4 |
| Optimizer | Active | β1=0.9 closed, β2 in-flight |
| Grad clip | CLOSED | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | **CLOSED — floor at 24** | slice_num=24 |
| mlp_ratio | CLOSED | mlp_ratio=2 |
| T_max cosine | CLOSED | T_max=18 (full) |
| Warmup | **RETESTING** (#2433, per-iter) | — |
| weight_decay | **CLOSED — wd=0 optimal** | wd=0 |
| Attention dropout | CLOSED | 0.0 (no dropout) |
| surf_weight | PARTIAL — testing sw↓ | #2426 (sw=5/7) |
| EMA | PENDING (fern #2117) | 0.99 on slice=24+wd=0 |
| eta_min | STALE (alphonse #2326) | — |
| activation | STALE (edward #2327) | — |
| Fourier L | PENDING (nezuko #2393) | — |
| norm type | PENDING (tanjiro #2425) | — |
