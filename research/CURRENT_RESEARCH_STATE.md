# SENPAI Research State

- **Date:** 2026-05-17 00:00
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

**4-way merger (both lineages)**: bs=2+n=8+lr=7e-4+δ=0.10 → **tanjiro #4220 arm-1 is testing this.** If additive, predicted val ~54-55.

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
| tanjiro | #4220 | 4-way merge (n=8+lr=7e-4+δ=0.10) + δ=0.05 | wave-14 NEW (just assigned) |
| thorfinn | #4221 | slice_num lower bracket {32, 48} on new best | wave-14 NEW (just assigned) |
| frieren | #4222 | lr=7e-4+clip=1.0 on bs=2+n=10+δ=0.10 (5-way compound) | wave-14 NEW (just assigned) |
| nezuko | #4223 | clip=1.0 + surf_weight=5 on bs=2+n=10+δ=0.10 | wave-14 NEW (just assigned) |
| alphonse | #4198 | LR upper search {9e-4, 1.2e-3} on bs=2+n=8 | wave-14 WIP |
| edward | #4199 | 4-way Huber δ={0.15, 0.20} on bs=2+n=8+lr=7e-4 | wave-14 WIP |
| askeladd | #4179 | bs=2+n=8 + Huber δ={0.15, 0.20} — 3-way compound | wave-14 WIP |
| fern | #4130 | EMA re-test at bs=2 (τ={0.998, 0.995}) | wave-14 WIP |

## Current research themes

### Wave-14: Merging the two lineages + pushing the δ floor

Key insight this round: TWO competing stacks at val~57. The highest-value experiments now test whether these stacks can be merged:

1. **tanjiro #4220** (highest priority): 4-way merge (bs=2+n=8+lr=7e-4+δ=0.10) vs δ=0.05 on n=10. If the 4-way compound works, val should drop to ~54-55.
2. **frieren #4222**: 4-way (bs=2+n=10+δ=0.10+lr=7e-4) + 5-way (+ clip=1.0). Tests whether the n=10+δ=0.10 stack absorbs lr=7e-4 and clip=1.0.
3. **edward #4199**: 4-way δ={0.15, 0.20} on the n=8+lr=7e-4 stack. Note: δ=0.10 (from #4103) now has a new claim — edward should ideally see if δ=0.15 beats the new baseline 56.92.
4. **alphonse #4198**: LR ceiling on n=8 stack — does 9e-4 or 1.2e-3 beat 7e-4?
5. **askeladd #4179**: 3-way δ on n=8 (without lr) — completes the δ×n=8 interaction matrix.

### Critical findings from this round

- **Monotonic Huber tightening**: δ=0.3 → 0.15 → 0.10 → each step profitable at bs=2. L1 floor still not found; δ=0.05 (tanjiro arm-2) and δ=0.10 on n=8 lineage (#4199) are next.
- **n=8 × clip=1.0 SUBSTITUTES** (nezuko #4095): These two levers attack the same DOF (gradient noise). Do NOT combine. Use clip=1.0 on n=10 or n=8 on clip=0.25.
- **bs=1 ceiling found** (frieren #4125): bs=1 is neutral vs bs=2 on val at the current best. The lever is exhausted. Memory headroom (9.25 GB) opens capacity experiments.
- **slice_num>64 fails hard** (#4131): Step-count loss + routing softmax flattening. Direction reversal to slice_num=32/48 (thorfinn #4221).

## Key insights accumulated

- **Monotonic Huber: δ=0.10 is profitable** (confirmed BS2+n10). δ=0.05 still untested.
- **n=8 and clip=1.0 are substitutive** (#4095). Do NOT combine.
- **Current best stack**: BF16 + LS + n10 + bs=2 + δ=0.10 (val=56.92/test=49.32)
- **Alternative strong stack**: BF16 + LS + n8 + bs=2 + lr=7e-4 (val=57.11/test=49.24)
- **bs lever exhausted** at bs=1. bs=2 is the step-count optimum for 30-min budget.
- **Memory headroom: 9.25 GB at bs=1** (vs 96 GB available) — capacity expansion possible
- **Memory headroom: 18.43 GB at bs=2** — still plenty for wider models
- **T_max=20 confirmed optimal** — T_max=18, 22 both regress.
- **lr=7e-4 confirmed** on n=8+bs=2 stack. Effect sub-additive (-1.99% vs -8.75% at bs=8). LR ceiling above 7e-4 unknown.
- **clip_frac=0.988 at ep18** for bs=2 — approaching but not reaching saturation escape.
- **EMA dead on BF16 stack**: 3+ tests confirm. Being re-tested at bs=2 with 13,500 steps (fern #4130).

## Potential next research directions

- **5-way compound**: bs=2+n=10+δ=0.10+lr=7e-4+clip=1.0 — theoretically justified (n=10 × clip=1.0 compounds, δ × clip orthogonal). Frieren #4222 arm-2 testing this.
- **δ=0.05 or L1**: monotonic trend through δ=0.10, next is 0.05 (tanjiro #4220 arm-2) or pure L1 (Huber δ→0).
- **LR above 7e-4**: alphonse #4198 testing {9e-4, 1.2e-3}.
- **LR warmup**: 5-epoch warmup on the new best stack — untested.
- **Capacity expansion at bs=1**: n_hidden=192, n_layers=6 (requires code change). 80 GB headroom available.
- **Lower slice_num {32, 48}**: thorfinn #4221 testing — coarser routing may reduce slot redundancy in smooth flow regions.
