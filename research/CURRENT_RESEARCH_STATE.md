# SENPAI Research State

- **Date:** 2026-05-17 01:40
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4103 (tanjiro, merged):** BF16 + LayerScale γ=0.01 + n_freqs=**10** + **batch_size=2** + **Huber δ=0.10** + T_max=20 + clip=0.25 (no EMA)
- **val_avg/mae_surf_p: 56.92** | **test_avg/mae_surf_p: 49.32**
- Per-split test surf_p: single=54.68, rc=61.34, cruise=32.89, re_rand=48.35
- best_epoch=18/18 (timeout-bound, still descending)
- **Cumulative improvement: -55.8% val from round-5 start (~128.69)**

**Also strong:** PR #4146 (val=57.11): bs=2+n=8+lr=7e-4 — test=49.24, all 4 splits improve. Different stack lineage, coexisting.

## Two competing lineages

| Lineage | Stack | val | test | Strength |
|---|---|---|---|---|
| A | bs=2+n=8+lr=7e-4 (#4146) | 57.11 | 49.24 | n=8 aliasing reduction + larger steps |
| **B (current best)** | bs=2+n=10+δ=0.10 (#4103) | **56.92** | 49.32 | Tight Huber (L1-like) on late residuals |

**4-way merger (both lineages)**: bs=2+n=8+lr=7e-4+δ=0.10 → **tanjiro #4220 arm-1 is testing this.**

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4103 (tanjiro, merged)** | **BF16 + LS + n10 + bs=2 + δ=0.10** | **56.92** | **49.32** | **-0.33%** |
| #4146 (alphonse, merged) | BF16 + LS + n8 + bs=2 + lr=7e-4 | 57.11 | 49.24 | -1.99% |
| #4083 (alphonse, merged) | BF16 + LS + n8 + batch_size=2 | 58.27 | 51.12 | -3.96% |
| #4026 (alphonse, merged) | BF16 + LS + n10 + batch_size=2 | 60.67 | 53.11 | -5.32% |
| #4006 (fern, merged) | BF16 + LS + n_freqs=8 (clip=0.25) | 64.08 | 55.05 | -2.47% |
| #4009 (nezuko, merged) | BF16 + LS + n10 + clip=1.0 | 65.70 | 57.80 | -2.22% |
| #3527 (tanjiro, merged) | BF16 + LayerScale + n_freqs=10 | 67.19 | 58.05 | -5.6% |

## Active WIP (8 students)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| edward | #4279 | n_hidden capacity {160, 192} on new best (bs=2+n=10+δ=0.10) | wave-15 NEW (just assigned) |
| tanjiro | #4220 | 4-way merge (n=8+lr=7e-4+δ=0.10) + δ=0.05 | wave-14 WIP (long-running, multiple arms) |
| thorfinn | #4221 | slice_num lower bracket {32, 48} on new best | wave-14 WIP |
| frieren | #4222 | lr=7e-4+clip=1.0 on bs=2+n=10+δ=0.10 (5-way compound) | wave-14 WIP |
| nezuko | #4223 | clip=1.0 + surf_weight=5 on bs=2+n=10+δ=0.10 | wave-14 WIP |
| alphonse | #4198 | LR upper search {9e-4, 1.2e-3} on bs=2+n=8 | wave-14 WIP |
| askeladd | #4179 | bs=2+n=8 + Huber δ={0.15, 0.20} — 3-way compound | wave-14 WIP |
| fern | #4130 | EMA re-test at bs=2 (τ={0.998, 0.995}) | wave-14 WIP (delayed by rate limit) |

## Current research themes

### Wave-14/15: Lineage merger + capacity expansion

**Settled knowledge from PR #4199 (edward, closed):**
- At the **n=8+lr=7e-4 lineage, δ optimum = 0.30** (NOT lower). δ=0.15 essentially flat (+0.42%), δ=0.20 regresses (+3.95%). Opposite of n=10 lineage. Mechanism: residual saturation + clip-saturation absorb any benefit from tighter Huber knee.
- **Per-split signature**: δ=0.15 helps cruise (-2.75%) and re_rand (-0.81%) but hurts single/rc — partial split benefit but no avg gain.
- **δ is now settled for lineage A**: do not sweep δ below 0.30 on n=8+lr=7e-4 stack without a new mechanism.

**Active highest-priority experiments:**

1. **tanjiro #4220** (highest priority): 4-way merge (bs=2+n=8+lr=7e-4+δ=0.10) + δ=0.05 on n=10. Multiple arms running, val unknown — potential breakthrough.
2. **edward #4279** (new): n_hidden capacity expansion {160, 192} on new best stack. Memory budget is 80 GB — capacity may be the binding constraint.
3. **alphonse #4198**: LR upper search {9e-4, 1.2e-3} on bs=2+n=8 lineage.
4. **askeladd #4179**: 3-way δ={0.15, 0.20} on n=8 without lr — completes the δ×n=8 interaction matrix (complement to edward #4199).
5. **frieren #4222**: lr=7e-4+clip=1.0 on n=10+δ=0.10 (5-way compound test).

### Critical findings accumulated this round

- **δ=0.30 confirmed optimal for lineage A** (n=8+lr=7e-4). Settling this knob — closed edward #4199.
- **n=8 × clip=1.0 SUBSTITUTES**: Do NOT combine. clip=1.0 on n=10, n=8 on clip=0.25.
- **bs=1 ceiling found**: bs=2 is step-count optimum for 30-min budget.
- **slice_num>64 fails hard**: routing softmax flattening. Testing {32, 48} with thorfinn.
- **Monotonic Huber**: δ=0.10 profitable on n=10 stack; δ floor not yet found (δ=0.05 in tanjiro #4220).
- **EMA dead on BF16**: 3+ tests. Fern re-testing at bs=2.
- **Memory headroom**: 18.43 GB peak at bs=2 vs 96 GB. n_hidden expansion is viable.

## Key insights accumulated

- **Current best stack**: BF16 + LS + n10 + bs=2 + δ=0.10 (val=56.92/test=49.32)
- **Alternative strong stack**: BF16 + LS + n8 + bs=2 + lr=7e-4 (val=57.11/test=49.24)
- **δ=0.30 is optimal for lineage A** (n=8+lr=7e-4); **δ=0.10 for lineage B** (n=10)
- **n=8 and clip=1.0 are substitutive**. Do NOT combine.
- **bs lever exhausted** at bs=1. bs=2 is the step-count optimum.
- **Memory headroom: 18.43 GB at bs=2** — n_hidden=192 (~41 GB) and n_hidden=256 (~74 GB) viable.
- **T_max=20 confirmed optimal** for both lineages.
- **lr=7e-4 confirmed optimal** for lineage A. LR ceiling still unknown (testing {9e-4, 1.2e-3}).
- **EMA dead on BF16 stack**: 3+ tests confirm. Final check at bs=2 running (fern #4130).
- **Capacity expansion**: n_hidden={160, 192} first screen (edward #4279), then wider if it wins.

## Potential next research directions

- **n_hidden > 192**: if edward #4279 confirms capacity helps, push to 224/256.
- **δ=0.05 or L1**: tanjiro #4220 arm-2 testing. If monotonic, push lower.
- **LR warmup**: 5-epoch warmup on new best — untested.
- **weight_decay sweep**: current 0.0001 default, never explored. {0.001, 0.01} possible.
- **Per-domain loss weighting**: cruise responds to δ=0.15 distinctly (−2.75%). A domain-specific δ or surf_weight could exploit this.
- **Lower slice_num {32, 48}**: thorfinn #4221 testing.
- **5-way compound (n=10+δ=0.10+lr=7e-4+clip=1.0)**: frieren #4222 testing.
