# SENPAI Research State

- **Date:** 2026-05-17 06:05
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4322 (askeladd, MERGED THIS TURN):** BF16 + LayerScale γ=0.01 + n_freqs=10 + batch_size=2 + Huber δ=0.10 + T_max=20 + clip=0.25 + slice_num=32 + **weight_decay=0.001**
- **val_avg/mae_surf_p: 55.799** | **test_avg/mae_surf_p: 48.846**
- Per-split test surf_p: single=51.318, rc=62.165, cruise=33.069, re_rand=48.835
- best_epoch=22/22 (timeout-bound)
- **Cumulative improvement: -56.7% val from round-5 start (~128.69)**

**Note arm-2 (wd=0.005)**: val=56.080, test=**48.496** — best test overall; wins EVERY test split but val criterion not met.

**Also strong:** PR #4146 (val=57.11): bs=2+n=8+lr=7e-4 — test=49.24. Different lineage.
**Test winner (unmerged)**: wd=0.005 arm, test=48.496 (all 4 test splits improved vs baseline).

## Two competing lineages

| Lineage | Stack | val | test | Strength |
|---|---|---|---|---|
| A | bs=2+n=8+lr=7e-4 (#4146) | 57.11 | 49.24 | n=8 aliasing reduction + larger steps |
| **B (current best)** | bs=2+n=10+δ=0.10+slice=32 (#4221) | **56.124** | 49.696 | Tight Huber + smaller routing slots = more epochs |

**4-way merger (both lineages)**: bs=2+n=8+lr=7e-4+δ=0.10 → **tanjiro #4220 arm-1 is testing this.**

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4322 (askeladd, merged)** | **BF16 + LS + n10 + bs=2 + δ=0.10 + slice=32 + wd=0.001** | **55.799** | **48.846** | **-0.58%** |
| #4221 (thorfinn, merged) | BF16 + LS + n10 + bs=2 + δ=0.10 + slice=32 | 56.124 | 49.696 | -1.40% |
| #4103 (tanjiro, merged) | BF16 + LS + n10 + bs=2 + δ=0.10 | 56.92 | 49.32 | -0.33% |
| #4146 (alphonse, merged) | BF16 + LS + n8 + bs=2 + lr=7e-4 | 57.11 | 49.24 | -1.99% |
| #4083 (alphonse, merged) | BF16 + LS + n8 + batch_size=2 | 58.27 | 51.12 | -3.96% |
| #4026 (alphonse, merged) | BF16 + LS + n10 + batch_size=2 | 60.67 | 53.11 | -5.32% |
| #4006 (fern, merged) | BF16 + LS + n_freqs=8 (clip=0.25) | 64.08 | 55.05 | -2.47% |
| #4009 (nezuko, merged) | BF16 + LS + n10 + clip=1.0 | 65.70 | 57.80 | -2.22% |
| #3527 (tanjiro, merged) | BF16 + LayerScale + n_freqs=10 | 67.19 | 58.05 | -5.6% |

## Active WIP (8 students)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| edward | #4367 | n_head sweep {2, 8} — per-head capacity vs diversity | wave-15 NEW (just assigned) |
| fern | #4396 | n_freqs sweep {8, 12} on new best stack | wave-15 NEW (just assigned) |
| tanjiro | #4349 | 5-way compound (slice=32 + n=8 + lr=7e-4 + δ=0.10) | wave-15 active |
| thorfinn | #4407 | T_max bracket {16, 18} on new best stack (wd=0.001+slice=32) | wave-15 NEW (just assigned) |
| frieren | #4352 | surf_weight upper sweep {12, 15} on new best | wave-15 active |
| nezuko | #4368 | clip bracket {0.18, 0.20} — fill optimum between 0.15 and 0.25 | wave-15 active |
| alphonse | #4330 | slice=32 + lr=7e-4 compound on new best (4-way merge w/lineage A's lr win) | wave-15 active |
| askeladd | #4406 | weight_decay bracket {0.002, 0.003} — fill val optimum on new best stack | wave-15 NEW (just assigned) |

## Current research themes

### Wave-14/15: Lineage merger + capacity expansion

**Settled knowledge from PR #4199 (edward, closed):**
- At the **n=8+lr=7e-4 lineage, δ optimum = 0.30** (NOT lower). δ=0.15 essentially flat (+0.42%), δ=0.20 regresses (+3.95%). Opposite of n=10 lineage. Mechanism: residual saturation + clip-saturation absorb any benefit from tighter Huber knee.
- **Per-split signature**: δ=0.15 helps cruise (-2.75%) and re_rand (-0.81%) but hurts single/rc — partial split benefit but no avg gain.
- **δ is now settled for ALL lineage A** (n=8, with or without lr=7e-4): PR #4179 askeladd confirms δ=0.30 optimal at n=8 without lr. Combined with edward #4199 (n=8+lr=7e-4), δ mapping complete for lineage A. Do NOT sweep δ below 0.30 on any n=8 stack.

**Active highest-priority experiments:**

1. **tanjiro #4220** (highest priority): 4-way merge (bs=2+n=8+lr=7e-4+δ=0.10) + δ=0.05 on n=10. Multiple arms running, val unknown — potential breakthrough.
2. **edward #4279** (new): n_hidden capacity expansion {160, 192} on new best stack. Memory budget is 80 GB — capacity may be the binding constraint.
3. **alphonse #4198**: LR upper search {9e-4, 1.2e-3} on bs=2+n=8 lineage.
4. **askeladd #4179**: 3-way δ={0.15, 0.20} on n=8 without lr — completes the δ×n=8 interaction matrix (complement to edward #4199).
5. **frieren #4222**: lr=7e-4+clip=1.0 on n=10+δ=0.10 (5-way compound test).

### Critical findings accumulated this round

- **weight_decay=0.001 NEW BEST** (askeladd #4322 merged): val=55.799 (-0.58%), test=48.846 (-1.71%). Default wd=0.0001 was over-regularizing. arm-2 wd=0.005 has BEST-EVER test=48.496 (wins all 4 test splits) but val criterion not met. Classic regularization tradeoff: RC wants lower wd, cruise wants higher wd. Now filling bracket {0.002, 0.003} in askeladd #4406.
- **slice_num lever SETTLED** (thorfinn #4298 closed): {32 wins val, 48 wins test, 40 wins nothing}. slice=40 is worst of both worlds. T_max=24 at slice=48 has zero effect. DO NOT sweep slice_num further.
- **fourier_base=2.0 PERMANENTLY SETTLED** (fern #4331 closed): 2nd independent confirmation across stacks. PR #4060 (n=8 stack) and this PR (n=10+slice=32) both find 2.0 optimal. Strong per-split signal: fb=1.5 helps cruise+re_rand, hurts single (smooth basis can't represent sharp leading-edge features). Per-domain positional encoding now a viable future direction (code change required).
- **n_head=4 confirmed best under wall-clock budget** (edward #4289 closed): n_hidden=160/192 wall-clock-bound, not capacity-limited. Per-epoch val curves ~identical across widths. Narrower wins because more epochs in 30 min. Pivot: test n_head {2, 8} (edward #4367) — near-free lever.
- **clip=0.15 ties current best on val** (nezuko #4293 closed): val=56.127 vs 56.124 (+0.003 = noise). Test improves -0.35%. Monotonicity confirmed: 56.13 < 56.92 < 58.05. Bracket {0.18, 0.20} → nezuko #4368. Per-split: clip tightening helps single+cruise, hurts rc — split-selective lever.
- **δ=0.30 confirmed optimal for lineage A** (n=8+lr=7e-4). Settling this knob — closed edward #4199.
- **n=8 × clip=1.0 SUBSTITUTES**: Do NOT combine. clip=1.0 on n=10, n=8 on clip=0.25.
- **bs=1 ceiling found**: bs=2 is step-count optimum for 30-min budget.
- **slice_num non-monotone**: slice>64 fails hard (routing softmax flattening). slice<64 improves via epoch-budget mechanism (slice=32: +4 epochs in 30 min → val=56.124 NEW BEST). Val/test winner split: slice=32 wins val, slice=48 wins test (best test=48.578 seen). Sweet spot likely between 32-48. Testing slice=40 + slice=48+T_max=24 in thorfinn #4298.
- **Monotonic Huber**: δ=0.10 profitable on n=10 stack; δ floor not yet found (δ=0.05 in tanjiro #4220).
- **EMA alive at bs=2** (PR #4130 closed): EMA gap +4.18 at τ=0.998 on δ=0.30 stack. Mechanism confirmed (noise-averaging at 13,500 steps).
- **EMA × δ=0.10 ANTI-ADDITIVE** (PR #4288 closed): EMA gap shrinks to +2.23 at δ=0.10 (Huber already does noise reduction) AND the ~12-14% per-epoch overhead costs 2 epochs in 30-min budget. Net: arm-1 val=60.42 (+7.6% vs new best). EMA is DEAD on current best stack at this budget.
- **LR ceiling at 7e-4 for bs=2+n=8 lineage** (PR #4198 closed): lr=9e-4 val=59.02 (+3.34%), lr=1.2e-3 val=57.95 (+1.46%). Non-unimodal curve — γ-collapse partially offsets larger LR. clip-saturation robust to LR within 18-epoch window.
- **Per-split signature emerging**: cruise + re_rand respond differently than single + rc to optimization changes. Three independent observations now (alphonse #4198, askeladd #4179, thorfinn #4221). Lower-magnitude / less-clip-saturated splits gain when single/rc don't. Strong candidate for per-split loss / per-split δ in future.
- **lr=7e-4 does NOT transfer to lineage B** (frieren #4222 closed): val 57.53 (+2.5% vs new best). At δ=0.10+n=10, lr=7e-4 produces noisier L1-regime gradients; late-cosine can't recover at 18-epoch budget. May still work on slice=32 stack (+4 epoch budget) — alphonse #4330 testing.
- **clip × δ reversal THIRD confirmation** (frieren #4222): clip_frac collapses 0.992→0.644 at clip=1.0+δ=0.10+lr=7e-4. 36% of late-training steps escape clip. STOP testing clip≥1.0 with δ=0.10 anywhere.
- **δ=0.10 is the Huber floor on n=10** (tanjiro #4220 arm-2 closed): δ=0.05 regresses +3.7% vs new best. Monotonic Huber tightening 0.3→0.15→0.10 does NOT continue.
- **Stack-lineage merger is sub-additive** (tanjiro #4220 arm-1 closed): 4-way compound (n=8+lr=7e-4+δ=0.10) val=56.39 — beats old baseline 56.92 but NOT new best 56.124. Adding slice=32 = 5-way (tanjiro #4349 testing).
- **clip × δ interaction REVERSES at tight knee** (PR #4223 closed): clip=1.0 + δ=0.10 → clip_frac drops to 0.716 at ep17 (vs 1.0 on δ=0.30 stack). Tight Huber knee → smaller late-epoch gradients → clip rarely engages. clip=1.0 regresses +1.66% val on this stack. **Implies tighter clip {0.15, 0.10} may now help** (testing in nezuko #4293).
- **surf_weight=5 regresses on n=10+δ=0.10** (PR #4223): val 57.594 +1.18%. surf_weight=10 already well-calibrated; rebalancing trades surf↑ vs vol↓ unfavorably.
- **Memory headroom**: 18.43 GB peak at bs=2 vs 96 GB. n_hidden expansion is viable.

## Key insights accumulated

- **Current best stack**: BF16 + LS + n10 + bs=2 + δ=0.10 + slice_num=32 + **wd=0.001** (val=55.799/test=48.846)
- **Alternative strong stack**: BF16 + LS + n8 + bs=2 + lr=7e-4 (val=57.11/test=49.24)
- **δ=0.30 is optimal for lineage A** (n=8+lr=7e-4); **δ=0.10 for lineage B** (n=10)
- **n=8 and clip=1.0 are substitutive**. Do NOT combine.
- **bs lever exhausted** at bs=1. bs=2 is the step-count optimum.
- **Memory headroom: 18.43 GB at bs=2** — n_hidden=192 (~41 GB) and n_hidden=256 (~74 GB) viable.
- **T_max=20 confirmed at slice=48** (T_max=24 had zero effect). **T_max on new best stack untested** — thorfinn #4407 testing {16, 18}.
- **lr=7e-4 is the LR ceiling for lineage A** (alphonse #4198 closed). LR window: 7e-4 < 1.2e-3 (regress +1.46%) < 9e-4 (regress +3.34%). Non-unimodal curve from γ-collapse. lr=7e-4 on the NEW best stack untested — testing now (alphonse #4330).
- **EMA alive at bs=2** (PR #4130): noise-averaging mechanism confirmed (+4.18 EMA gap, beats no-EMA bs=2+n=10 baseline by 1.3-1.6 val). But doesn't beat current best alone. EMA × δ=0.10 compound running (fern #4288).
- **Capacity expansion**: n_hidden={160, 192} first screen (edward #4279), then wider if it wins.

## Potential next research directions

- **clip optimum found ≈ 0.15-0.20**: nezuko #4368 bracket {0.18, 0.20} will confirm exact val optimum. If found, natural compound: clip_best + lr=7e-4 (if alphonse #4330 wins).
- **n_head {2, 8}**: edward #4367 testing — near-free lever (no VRAM cost). If n_head=2 helps, head_dim=64 is the bottleneck; push n_head=1 next. If n_head=8 helps, pattern diversity matters.
- **n_freqs {8, 12}**: fern #4396 testing — fourier_base settled, now n_freqs revisit on new stack.
- **T_max bracket {16, 18}**: thorfinn #4407 testing on new best stack (wd=0.001+slice=32).
- **wd bracket {0.002, 0.003}**: askeladd #4406 testing — find val optimum between 0.001 (val winner) and 0.005 (test winner).
- **Per-domain positional encoding** (mixed basis: smooth for cruise, sharper for single): from fern #4331 per-split signal. Would require code change to mix multiple fourier_base values in feature dim or per-split adapter heads.
- **n_hidden expansion revisited**: arm-1 attention scales linearly (not quadratic). Budget ~80 GB unused. With 45-min timeout OR with T_max extension, n_hidden=160 could outperform 128. Deferred until we know if wider T_max (from thorfinn #4298) helps.
- **δ=0.05 settled**: tanjiro #4220 closed — regresses. δ=0.10 is the floor on n=10.
- **LR warmup**: 5-epoch warmup on new best — untested lever.
- **weight_decay {0.001, 0.005}**: askeladd #4322 testing on new best stack.
- **Per-domain loss weighting**: cruise responds to clip/δ distinctly. A domain-specific surf_weight could exploit per-split structure.
- **slice refinement**: thorfinn #4298 testing {40, 48+T_max=24} — will settle slice sweet spot.
- **5-way compound (slice=32+n=8+lr=7e-4+δ=0.10)**: tanjiro #4349 — the boldest merger. If this wins, it integrates both lineages.
