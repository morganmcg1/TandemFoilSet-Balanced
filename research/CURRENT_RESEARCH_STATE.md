# SENPAI Research State

- 2026-05-13 18:00 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=60.7447 (PR #2343 wd=0 + slice_num=24 + clip=5.0)**. Cumulative gain from PR #1391: 121.28 → 60.74 = **−49.9%**.
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

## Major milestones

### Lion optimizer fully tuned
| Lever | Status | Optimal | Source |
|---|---|---|---|
| `lr` | CLOSED | 1.5e-4 | Finding #20 |
| `β1` | CLOSED | 0.9 | #2237 / Finding #32 |
| `β2` | CLOSED | 0.99 (sharp sweet spot) | #2382 / Finding #39 |
| `weight_decay` | CLOSED | 0.0 | #2343 / Finding #38 |

### IID/OOD redistribution meta-pattern (Finding #41)
Three independent axes now confirm: **capacity/resolution increases improve IID, harm OOD uniformly.**

| Lever | in_dist Δ | OOD Δ | Source |
|---|---|---|---|
| surf_weight↑ (sw=15) | −1.49% | rc +3.42% | Finding #37 (#2294) |
| RMSNorm swap | −5.61% | OOD +0.78–2.10% | Finding #40 (#2425) |
| Fourier L=12 | −2.21% | OOD +0.87–1.28% | Finding #41 (#2393) |

Implication: test_avg improvements must come from OOD-specific levers or levers that improve both IID and OOD simultaneously (like wd=0 did). Any "fit harder" axis is dead for test_avg at current scale.

## Round-3 status (updated 2026-05-13 18:00)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| thorfinn | #2433 | Per-iter warmup (correct SequentialLR impl) | **wip** |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **wip** |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **wip** (STALE) |
| askeladd | #2458 | Lookahead-wrapped Lion (k=5/10) | **wip** |
| tanjiro | #2456 | Pre-LN → Post-LN swap | **wip** |
| nezuko | #2466 | SwiGLU MLP (param-matched) | **wip** |
| edward | #2473 | Slot routing temperature: fixed T=1.0/0.5 vs learnable | **wip** (NEW) |
| fern | #2474 | Coord-noise augmentation: Gaussian jitter on mesh coords (train only) | **wip** (NEW) |
| thorfinn | #2343 | wd=0 ablation | **MERGED** ✓ | test=60.7447 (−1.78%). New best. |
| edward | #2327 | GELU → SiLU activation swap | **CLOSED** ✗ | +16% regression all 3 trials. Finding #42. |
| fern | #2117 | EMA decay=0.95/0.99 | **CLOSED** ✗ | Diag-ratio collapse; EMA gain budget-dependent. Finding #43. |
| nezuko | #2393 | Fourier L=12/4 | **CLOSED** ✗ | Regime-neutral; IID/OOD redistribution. Finding #41. |
| tanjiro | #2425 | LayerNorm → RMSNorm | **CLOSED** ✗ | Regime-neutral aggregate. Finding #40. |
| askeladd | #2382 | Lion β2=0.999/0.95 | **CLOSED** ✗ | Symmetric +13%. Finding #39. |
| tanjiro | #2344 | attention dropout | **CLOSED** ✗ | Capacity-limited. |
| frieren | #2294 | surf_weight=15/20 | **CLOSED** ✗ | rc +3.42. Finding #37. |
| nezuko | #2333 | slice_num=16 | **CLOSED** ✗ | Floor at slice_num=24. |
| askeladd | #2237 | Lion β1 sweep | **CLOSED** ✗ | β1=0.9 optimal. |

## Key research findings (cumulative)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling CLOSED**: lr=1.5e-4 correct.
21. **Clip mechanism fully characterized**: clip=5.0 in bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24.
24. **EMA 0.99 wins on old stack**: slice=24 confirmation pending → SUPERSEDED by Finding #43.
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED**: T_max=18 optimal.
27. **Capacity-budget structural bound**: >+5% per-epoch overhead = net loser.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule.
29. **Per-epoch warmup = step function (CLOSED BUG)**: Warmup must be per-iteration.
30. **Attention dropout CLOSED**: Slot routing needs determinism.
31. **Inter-seed variance noise floor ~1.4% rel**: Multi-seed needed for sub-noise differences.
32. **Lion β1 lever CLOSED**: β1=0.9 optimal.
33. **Locality-prior OOD tradeoff**: slice_num lever trades OOD, not in-dist accuracy.
34. **Capacity-limited regime** (#2344): 1.47M+18 epochs = capacity-limited, not overfitting.
35. **Locality regularization incompatible with stochastic attention** (#2344).
36. **surf_weight has split-asymmetric effects** (#2294): sw↑ hurts rc. Optimum below 10.
37. **surf_weight lever CLOSED** (#2294): sw=15/20 all worse on rc. sw=5/7 testing.
38. **Lion wd=0 optimal** (#2343): NEW BASELINE.
39. **Lion β2=0.99 sharp sweet spot** (#2382): Lion fully tuned milestone.
40. **Normalization-type lever closed** (#2425): LayerNorm optimal; RMSNorm neutral aggregate.
41. **IID/OOD redistribution meta-pattern** (#2393): Three independent axes confirm — capacity/resolution increases improve IID, harm OOD uniformly. Fourier L lever closed. Future improvements must target OOD-specific mechanisms.
42. **SiLU vs GELU convergence-rate asymmetry** (#2327): SiLU +16% regression across all 3 trials. Mechanistic explanation: Lion sign-momentum + fixed budget favors activations matching GELU's smooth gradient profile. Slow-converging architectural choices lose at fixed epoch budget regardless of theoretical asymptotics. GELU is optimal.
43. **EMA gain is timeout-budget-dependent** (#2117): EMA 0.95 and 0.99 both tie baseline (diag-ratio collapses from 0.95% to 0.08% by epoch 18). EMA needs late-training parameter noise to average; full-schedule completion + eta_min=0 eliminates that noise. EMA may revive if alphonse #2326 (eta_min>0) lands a gain.

## Active experiments (8 students)

### Tier 1: Architecture / meta-optimizer
| PR | Student | Expected gain |
|---|---|---|
| #2433 | thorfinn | Per-iteration warmup: −0.5% to −2% |
| #2458 | askeladd | Lookahead-wrapped Lion (k=5/10): −0.5% to −3% |
| #2456 | tanjiro | Pre-LN → Post-LN swap: −0.5% to −2.5% |
| #2466 | nezuko | SwiGLU MLP (param-matched): −0.3% to −2% |

### Tier 2: OOD-targeted levers (priority per Finding #41)
| PR | Student | Expected gain |
|---|---|---|
| #2426 | frieren | surf_weight DOWN (sw=5/7): rc improvement expected |
| #2474 | fern | Coord-noise augmentation (σ=0.005): OOD-targeted data augmentation |
| #2473 | edward | Slot routing temperature ablation: T=1.0 vs learnable |

### Tier 3: Schedule
| PR | Student | Expected gain |
|---|---|---|
| #2326 | alphonse | cosine eta_min=1.5e-5 (STALE) |

## Key open questions
1. **Does Post-LN improve OOD on this 5-layer stack?** (tanjiro #2456) — does it follow Finding #41?
2. **Does Lookahead's Polyak averaging reduce Lion sign-flip variance?** (askeladd #2458)
3. **Does SwiGLU's gating break Finding #41 (improve OOD too) or follow it (IID only)?** (nezuko #2466)
4. **Does per-iteration warmup help Lion momentum bootstrap?** (thorfinn #2433)
5. **Does surf_weight=5/7 improve rc generalization?** (frieren #2426)
6. **Does coord-noise jitter on mesh coords target OOD splits without hurting IID?** (fern #2474) — first data-side intervention; key test of Finding #41 scope.
7. **Is slot routing temperature learnable T=0.5 actually doing work, or is fixed T=1.0 equivalent?** (edward #2473) — tests whether slot routing is genuinely a learned lever or just a tunable constant.
8. **Does eta_min refinement floor help?** (alphonse #2326, stale)

## IMPORTANT NOTES
- **All new/updated run commands must include `--weight_decay 0.0`**
- **Alphonse #2326** is stale (4h+) — status check pending
- **IID/OOD redistribution meta-pattern** now governs hypothesis selection: prioritize OOD-specific levers
- **EMA revival condition**: EMA may become viable again IF alphonse #2326 (eta_min=1.5e-5) shows that non-zero eta_min prevents full convergence — that would reintroduce late-epoch noise for EMA to average.

## Lever status summary
| Lever | Status | Best value |
|---|---|---|
| Learning rate | CLOSED | lr=1.5e-4 |
| Optimizer β1 | CLOSED | β1=0.9 |
| Optimizer β2 | CLOSED — sharp sweet spot | β2=0.99 |
| Grad clip | CLOSED | clip=5.0 |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | CLOSED — floor at 24 | slice_num=24 |
| mlp_ratio | CLOSED | mlp_ratio=2 |
| T_max cosine | CLOSED | T_max=18 (full) |
| weight_decay | CLOSED — wd=0 optimal | wd=0 |
| Attention dropout | CLOSED | 0.0 (no dropout) |
| Normalization type | CLOSED — LayerNorm optimal | LayerNorm |
| Fourier L | CLOSED — L=8 optimal | L=8 |
| Activation function | CLOSED — GELU optimal | GELU |
| EMA | CLOSED (budget-dependent) — may revive at eta_min>0 | — |
| Normalization position | **ACTIVE** (#2456, post-LN) | — |
| Meta-optimizer | **ACTIVE** (#2458, Lookahead) | — |
| MLP architecture | **ACTIVE** (#2466, SwiGLU) | — |
| Warmup | **RETESTING** (#2433, per-iter) | — |
| surf_weight | PARTIAL — testing sw↓ | #2426 (sw=5/7) |
| Slot routing temperature | **ACTIVE** (#2473, fixed T ablation) | — |
| Coord-noise augmentation | **ACTIVE** (#2474, σ=0.005) | — |
| eta_min | STALE (alphonse #2326) | — |
