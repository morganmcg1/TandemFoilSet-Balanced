# SENPAI Research State

- **Date:** 2026-05-16 23:10
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4146 (alphonse, merged):** BF16 + LayerScale γ=0.01 + n_freqs=**8** + **batch_size=2** + **lr=7e-4** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **val_avg/mae_surf_p: 57.11** | **test_avg/mae_surf_p: 49.24**
- Per-split test surf_p: single=53.80, rc=61.64, cruise=32.83, re_rand=48.69 — **all 4 splits improve**
- best_epoch=18/18 (timeout-bound, still descending — headroom remains)
- Peak memory **18.43 GB**; throughput 102.4 s/epoch; clip_frac=0.988 at ep18
- **Cumulative improvement: -55.7% val from round-5 start (~128.69)**
- Key insight: lr=7e-4 compounds **sub-additively** on bs=2+n=8 (predicted -8.75%, got -1.99% val). clip-saturation robust vs lr knob in this range.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4146 (alphonse, merged)** | **BF16 + LS + n8 + bs=2 + lr=7e-4** | **57.11** | **49.24** | **-1.99%** |
| **#4083 (alphonse, merged)** | **BF16 + LS + n8 + batch_size=2** | **58.27** | **51.12** | **-3.96%** |
| #4026 (alphonse, merged) | BF16 + LS + n10 + batch_size=2 | 60.67 | 53.11 | -5.32% |
| #4006 (fern, merged) | BF16 + LS + n_freqs=8 (clip=0.25) | 64.08 | 55.05 | -2.47% |
| #4009 (nezuko, merged) | BF16 + LS + n10 + clip=1.0 | 65.70 | 57.80 | -2.22% |
| #3527 (tanjiro, merged) | BF16 + LayerScale + n_freqs=10 | 67.19 | 58.05 | -5.6% |
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |

## Active WIP (8 students)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | #4198 | LR upper search {9e-4, 1.2e-3} on bs=2+n=8 | wave-14 NEW (just assigned) |
| edward | #4199 | 4-way Huber δ={0.15, 0.20} on bs=2+n=8+lr=7e-4 | wave-14 NEW (just assigned) |
| askeladd | #4179 | bs=2+n=8 + Huber δ={0.15, 0.20} — 3-way compound | wave-14 WIP (updated baseline 57.11) |
| fern | #4130 | EMA re-test at bs=2 (τ={0.998, 0.995}) | wave-14 WIP |
| thorfinn | #4131 | slice_num sweep {128, 192} at bs=2 | wave-14 WIP |
| frieren | #4125 | bs=1 sweep — extreme steps (n=10 and n=8 arms) | wave-14 WIP (local metrics done, not yet pushed) |
| tanjiro | #4103 | bs=2 + Huber δ={0.15, 0.10} on n=10 | wave-14 WIP (stale baseline) |
| nezuko | #4095 | bs=2 + clip=1.0 compound | wave-14 WIP (stale baseline) |

**Note on stale baselines:** Tanjiro (#4103) and nezuko (#4095) were assigned before the current best shifted to 57.11. Results will be informative for mechanism, but may not beat the new baseline. Stale = still valid science, not a blocker for completion.

## Current research themes

### Wave-14: Compounding on the new best stack (57.11 baseline)

The current best stack is: **BF16 + LS γ=0.01 + n_freqs=8 + bs=2 + lr=7e-4 + Huber-0.3 + T_max=20 + clip=0.25**

Active compound tests targeting 57.11:

1. **alphonse #4198 (new)**: LR upper search {9e-4, 1.2e-3} — find the LR ceiling. If 7e-4 is not the peak, pushing higher may unlock more. At lr=1.2e-3, may enter "clip escape" regime (clip_frac drops meaningfully).
2. **edward #4199 (new)**: 4-way Huber δ={0.15, 0.20} on full new stack (bs=2+n=8+lr=7e-4). δ=0.15 gave -3.19% on bs=8/n=10 — if orthogonal, predicts val ≈ 55.3 for the 4-way compound.
3. **askeladd #4179**: 3-way Huber δ={0.15, 0.20} on bs=2+n=8 (without lr=7e-4) — complements edward's 4-way test; reveals δ×lr interaction structure.
4. **fern #4130**: EMA re-test at bs=2 (τ={0.998, 0.995}) — EMA was dead at bs=8 (window=50% of training); at bs=2 with 13,500 steps it may revive.
5. **thorfinn #4131**: slice_num={128, 192} at bs=2 — first-ever Transolver architectural sweep, enabled by 18.43 GB memory headroom.
6. **frieren #4125**: bs=1 sweep — extreme steps lever (~27,000 updates), n=10 and n=8 arms. Local metrics committed on pod, not yet pushed to GitHub.
7. **tanjiro #4103**: bs=2 + δ={0.15, 0.10} on n=10 (stale). Informative for δ mechanism at bs=2.
8. **nezuko #4095**: bs=2 + clip=1.0 compound (stale). Tests whether clip=1.0 helps at bs=2.

### Key open questions

- **LR ceiling**: 7e-4 is better than 5e-4 (sub-additively). Does 9e-4 or 1.2e-3 push further? (#4198)
- **δ=0.15 on full stack**: 4-way compound predicted val 55.3 if additive. (#4199)
- **bs=1**: does extreme step-count (27k updates) outperform bs=2 (13.5k)? (#4125)
- **slice_num**: can wider attention slot granularity improve OOD generalization? (#4131)
- **EMA at bs=2**: 13,500 steps — enough for AveragedModel to work? (#4130)
- **Sub-50 val target**: current best 49.24 on test. val below 55 would be a major milestone.

### Val floor analysis

Current val=57.11, test=49.24. Route to sub-55 val:
- δ=0.15 alone: -3.19% → 55.3 predicted
- LR above 7e-4: unknown
- All three combined (δ + higher lr + stacked compounds): sub-53 possible

## Key insights accumulated

- **Current best stack**: BF16 + LS + n_freqs=8 + bs=2 + lr=7e-4 (PR #4146, val=57.11)
- **bs=2 + n=8 + lr=7e-4 are all sub-additive with each other**: each gives smaller gains when stacked on the other. Both bs and lr lengthen optimization distance under clip-saturation — partially overlapping.
- **Memory headroom: 18.43 GB peak** — huge headroom for width/depth experiments.
- **best_epoch=18/18 at bs=2**: always timeout-bound — more epochs would help.
- **Clip-saturation NO LONGER absolute at bs=2+lr=7e-4**: clip_frac=0.988 at ep18 (vs 0.987 at baseline, 0.984 at #4115). Natural late-epoch gradient escape.
- **T_max=20 confirmed optimal**: both T_max=18 and T_max=22 regress. Schedule shape matters.
- **lr=7e-4 on bs=2+n=8**: sub-additive compound (predicted -8.75%, got -1.99% val).
- **n_freqs=8 confirmed**: local minimum on test; n=12 hurts OOD geometry (rc split); going below n=8 is a dead end.
- **clip=1.0 wins on n=10 stack** (#4009). Not yet tested on bs=2+n=8.
- **EMA dead on BF16 stack**: 3 separate tests confirm. Smoothing window covers only 41% of training at bs=8; being re-tested at bs=2 (#4130).
- **BF16 regime shift**: extended convergence horizon (17-18 epochs vs 12) changes optimal hyperparameters.
- **LayerScale γ=0.01 fully confirmed** across multiple bracket experiments.
- **Huber δ=0.15 wins** at bs=8/n=10 (-3.19% val). Not yet confirmed on new bs=2+n=8 stack.

## Potential next research directions

- **Width expansion (n_hidden=256)**: 18.43 GB headroom makes this safe. First architecture change in many waves.
- **mlp_ratio sweep**: currently hardcoded 2; mlp_ratio=4 doubles FFN capacity.
- **bs=2 + clip=1.0 + n=8 triple compound**: clip×n8×bs=2 — three independent wins not yet combined.
- **Longer training budget**: best_epoch=18/18 always timeout-bound. If budget extended to 45 min, another 4-5 epochs of descent.
- **Residual shape below δ=0.10**: tanjiro #4033 suggested δ=0.10 might be even better (gradient discontinuity risk).
- **LR warmup**: current setup starts at lr=7e-4 immediately. Warmup (5 epochs to peak LR) may stabilize early training.
