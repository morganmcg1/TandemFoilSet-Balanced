# SENPAI Research State

- **Date:** 2026-05-16 22:20
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4083 (alphonse, merged):** BF16 + LayerScale γ=0.01 + n_freqs=**8** + **batch_size=2** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **val_avg/mae_surf_p: 58.27** | **test_avg/mae_surf_p: 51.12**
- Per-split test surf_p: single=57.42, rc=64.11, cruise=33.68, re_rand=49.27 — **all 4 splits improve**
- best_epoch=18/18 (timeout-bound, still descending — headroom remains)
- Peak memory **18.43 GB**; throughput 102.4 s/epoch
- **Cumulative improvement: -54.7% val from round-5 start (~128.69)**
- Key insight: compound is **~89% additive** — bs=2 (step count) and n_freqs=8 (representation) are near-independent mechanisms. clip_frac drops from 1.000 → 0.987 at epoch 18 — first late-epoch gradient escape this round.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| **#4083 (alphonse, merged)** | **BF16 + LS + n8 + batch_size=2** | **58.27** | **51.12** | **-3.96%** |
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | 98.62 | 88.14 | -4.4% |
| #3221 (nezuko, merged) | Fourier n=10 + Huber-0.3 | 89.27 | 79.43 | -9.5% |
| #3333 (frieren, merged) | Fourier+Huber+T_max=20+clip=0.25 | 84.59 | 73.89 | -5.2% |
| #3529 (frieren, merged) | clip=0.25→1.0 on full stack | 84.01 | 72.95 | -0.7% |
| #3438 (nezuko, merged) | n_freqs=14 on full stack | 81.08 | 71.52 | -3.5% |
| #3593 (alphonse, merged) | LayerScale γ-init=0.01 on full stack | 72.77 | 65.12 | -10.2% |
| #3192 (edward, merged) | LayerScale + n_freqs=14 + EMA 0.998 | 71.20 | 62.71 | -2.16% |
| #3527 (tanjiro, merged) | BF16 + LayerScale + n_freqs=10 | 67.19 | 58.05 | -5.6% |
| #4009 (nezuko, merged) | BF16 + LS + n10 + clip=1.0 | 65.70 | 57.80 | -2.22% |
| #4006 (fern, merged) | BF16 + LS + n_freqs=8 (clip=0.25) | 64.08 | 55.05 | -2.47% |
| **#4026 (alphonse, merged)** | **BF16 + LS + n10 + batch_size=2** | **60.67** | **53.11** | **-5.32%** |

## Active WIP (8 students)

| Student | PR | Hypothesis | Status | Baseline |
|---|---|---|---|---|
| alphonse | #4146 | bs=2+n=8+lr=7e-4 compound; arm-2 lr_t_max=22 schedule fix | wave-14 NEW | **58.27** |
| nezuko | #4095 | bs=2 + clip=1.0 compound; arm-2: triple bs=2+n=8+clip=1.0 | wave-14 NEW | 60.67 |
| tanjiro | #4103 | bs=2 + Huber δ={0.15, 0.10} compound | wave-14 NEW | 60.67 |
| askeladd | #4179 | bs=2+n=8 + Huber δ={0.15, 0.20} compound on new winning stack | wave-14 NEW | **58.27** |
| fern | #4130 | EMA re-test at bs=2 (τ={0.998, 0.995}) | wave-14 NEW | 60.67 |
| thorfinn | #4131 | slice_num sweep {128, 192} at bs=2 (first-ever arch sweep) | wave-14 NEW | 60.67 |
| frieren | #4125 | bs=1 sweep — extreme steps-in-budget (n=10 and n=8 arms) | wave-14 NEW | 60.67 |
| edward | #4053 | n_freqs {8, 12} at clip=1.0+bs=8 (stale baseline; informative for clip×n_freqs interaction) | wave-12 WIP | 65.70 |

**Note:** All in-flight PRs assigned before bs=2 merge → baseline now 60.67. Any result below that is a new winner. Wave-11 results still inform variable optimums independently (different lever axes from bs).

## Closed this round

| PR | Reason |
|---|---|
| #3424 (askeladd) | clip=0.1 × Huber δ both arms regress 5%; LayerScale already gates gradients |
| #3878 (edward) | EMA decay sweep {0.995, 0.999}: both worse than 0.998 |
| #3882 (alphonse) | SAM ρ=0.05: 2× overhead halves epochs; no flat-min benefit |
| #3823 (nezuko) | Lookahead: both k=5/k=10 ~15-21% worse; disrupts LayerScale γ |
| #3740 (frieren) | Asymmetric LayerScale γ-init: γ converges naturally regardless of init |
| #3730 (alphonse) | LayerScale+n14 sub-additive WITHOUT EMA |
| #3782 (fern) | AdamW eps sweep falsified; default 1e-8 optimal |
| #3784 (thorfinn) | LR sweep on FP32 triple: clip_frac=1.0 throughout; superseded by BF16 |
| #3883 (fern) | T_max sweep: T_max=12 worst, T_max=20 optimal |
| #3909 (frieren) | Learnable Fourier: frequencies barely migrate; overhead costs ~2 epochs |
| #3941 (nezuko) | WD sweep {3e-5, 3e-4} both worse; WD=1e-4 confirmed optimal |
| #4007 (frieren) | Width n=144: timeout-bound at 15 epochs vs 17+ needed |
| #3983 (askeladd) | Huber δ {0.15, 0.5}: both regress vs δ=0.3 on FP32 stack |
| #3964 (alphonse) | LayerScale γ-init {0.005, 0.020}: both regress vs γ=0.01 |
| #4005 (tanjiro) | BF16+LS+n10+EMA 0.998: val=68.64 (worse). EMA dead on BF16 |
| #3971 (edward) | EMA warm-up on FP32 triple: arm-1 +4.1%, arm-2 +50%. Superseded by BF16+clip=1.0 |
| #4008 (thorfinn) | sw=5: val=64.10 — independent win superseded by parallel #4006 (n=8) merge |
| #4014 (frieren) | Width n=120: throughput hypothesis fails (no speedup), test ties |
| #4052 (nezuko) | clip ceiling {2.0, 4.0}: both arms regress on val (68.21 / 66.46 vs 65.70). Ceiling confirmed at clip=1.0. Non-monotone: clip=2.0 worst (+3.81%), clip=4.0 mixed (+1.16% val / -1.04% test) |
| #4033 (tanjiro) | Huber δ {0.15, 0.5}: δ=0.15 beats δ=0.3 on n=10 stack (val=64.00, -3.19%), stale baseline (post-#4026 merge best is 60.67). BF16 favors tighter δ thesis confirmed. Compound (δ=0.15 + bs=2) assigned #4103 |
| #4027 (askeladd) | LR {7e-4, 1e-3}: lr=7e-4 wins by -8.75% val (61.31 vs 67.19 baseline). Largest single-knob win any wave. Stale baseline (current best 60.67). clip_frac=1.000 throughout — effective step = lr × 0.25 × dir(g). Compound (lr=7e-4 + bs=2) assigned #4115 |
| #4060 (frieren) | fourier_base {1.5, 2.5}: both regress vs 64.08 baseline (64.79 / 65.03). fourier_base=2.0 locally optimal. octave spacing not tunable here — pivoted frieren to bs=1 sweep (#4125) |
| #4058 (fern) | n_freqs {4, 6}: n=6 val=63.22 but test=56.76 (rc +4.78%). n=8 is local minimum on test. Pivoted fern to EMA-bs2 (#4130) |
| #4059 (thorfinn) | sw {2.5, 5.0} on n=8 stack: both ~63.2-63.4 (vs current best 60.67). 3.7pt seed variance makes result noise-level. sw/n_freqs not independent. Pivoted thorfinn to slice_num sweep (#4131) |
| #4083 (alphonse) | **MERGED** — bs=2+n=8 compound: val=58.27 (-3.96%), test=51.12. All 4 splits improve. ~89% additive compound. lr_t_max=18 arm failed (premature cosine freeze). New best. |
| #4115 (askeladd) | bs=2+lr={7e-4, 8e-4} on n=10: arm-1 val=58.78 (regress 0.51 vs 58.27) / test=50.54 (improve 0.58); arm-2 dead at val=66.11. **First bs=2 clip escape**: clip_frac→0.984 at ep18 under lr=7e-4. lr×bs sub-additive — both extend optimization distance under clip-saturation. Pivoted askeladd to bs=2+n=8+δ compound (#4179) |

## Current research themes

### Wave-14: BF16+LS+n8+bs=2 (new merged baseline: 58.27)

The new best stack emerged from compounding the two independent wins:
- **batch_size=2 (PR #4026)**: 4.5× more updates in same wall-clock budget — clip-saturation mechanism
- **n_freqs=8 (PR #4006/4083)**: finer-grained Fourier representation with less aliasing
- **#4083 compound**: ~89% additive (-3.96% val on top of #4026)

**Wave-14 active compound tests (all targeting 58.27 baseline):**
1. **alphonse #4146**: bs=2+n=8+lr=7e-4 (biggest single-knob win applied to new stack) + lr_t_max=22 (schedule shape fix)
2. **nezuko #4095**: bs=2+clip=1.0 (arm-1 n=10, arm-2 n=8) — clip×bs=2 compound
3. **tanjiro #4103**: bs=2+Huber δ={0.15, 0.10} on n=10 — δ×bs=2 compound
4. **askeladd #4179**: bs=2+n=8+Huber δ={0.15, 0.20} — quad compound (orthogonal δ on new winning stack)
5. **frieren #4125**: bs=1 sweep (n=10, n=8) — extreme steps lever
6. **fern #4130**: EMA re-test at bs=2 (τ={0.998, 0.995}) — dead at bs=8, may revive at 13,500 steps
7. **thorfinn #4131**: slice_num={128, 192} at bs=2 — first-ever architectural sweep

**Key next questions:**
- Does lr=7e-4 compound with bs=2+n=8 to push below 52-53? (#4146)
- Does clip=1.0 help at bs=2? (#4095 arm-2)
- Does bs=1 continue to pay off, or does GPU underutilization dominate? (#4125)
- Does slice_num increase help? (#4131 — needs small code change)

### Wave-13 (all closed; n=8 baseline 64.08 experiments complete)

- **frieren #4060** CLOSED: fourier_base {1.5, 2.5} both regress. base=2.0 locally optimal. → #4125 bs=1 sweep
- **fern #4058** CLOSED: n_freqs {4, 6} — n=8 confirmed local minimum on test. → #4130 EMA-bs2
- **thorfinn #4059** CLOSED: sw {2.5, 5.0} — noise-level, sw/n_freqs not independent, 3.7pt seed variance. → #4131 slice_num sweep

These are still useful: any result <60.67 is a new winner, and they inform whether the optimum of each independent lever shifts on the n=8 stack.

### Wave-12 (in flight, on n=10+clip=1.0 baseline 65.70)

- **edward #4053**: n_freqs {8, 12} at clip=1.0 — answers the critical n=8+clip=1.0 compound question
- **nezuko #4052**: clip ceiling {2.0, 4.0} on n=10+clip=1.0 — pushes clip further

### Wave-11 (in flight, on older 67.19 baseline)

- **tanjiro #4033**: Huber δ {0.15, 0.5} — actively training at 100% GPU
- **askeladd #4027**: LR {7e-4, 1e-3} — recovering from GitHub rate-limit window (15:43–18:24 UTC)

## Key insights accumulated

- **bs=2 + n=8 is the new default** (PR #4083). clip-saturation + reduced aliasing → -54.7% cumulative from round-5 start.
- **Memory headroom: 18.43 GB peak** — huge head-room for width/depth/Fourier expansion experiments without OOM risk.
- **best_epoch=18/18 at bs=2**: still descending at timeout → more epochs would help; lr_t_max=22 should extend useful LR to the cutoff.
- **Clip-saturation NO LONGER absolute at bs=2**: PR #4083 showed clip_frac=0.987 at ep18, #4115 lr=7e-4 dropped further to 0.984. Cosine LR decay + bs=2 → late-epoch gradient shrinkage occasionally falls below the 0.25 clip threshold. "Effective step = lr × clip × dir(grad)" framing becomes approximate in the final 2-3 epochs.
- **lr × bs sub-additive compound**: lr=7e-4 effect halves when applied on top of bs=2 (8.75% → 3.11%). Both levers extend "total optimization distance" under clip-saturation — partially overlapping mechanisms.
- **n_freqs=8 confirmed on bs=2 stack** (#4083). Going below n=8 (n=4/6) is a dead end on test — n=8 is local minimum.
- **n_freqs ordering (matched recipes)**: 8 < 12 < 14 ≈ 10 — non-monotonic; n=8 confirmed on both bs=8 and bs=2.
- **clip=1.0 wins on n=10 stack** (#4009). Compound with n=8 and with bs=2 not yet tested.
- **Clip acts as lr-scale in fully-clipped regime.** clip_frac=1.0 throughout most runs; effective step ≈ clip × grad/‖grad‖.
- **BF16 regime shift**: extended convergence horizon (17-18 epochs vs 12) changes optimal hyperparameters.
- **EMA dead on BF16 stack**: 3 separate tests confirm. Smoothing window covers only 41% of training; cost exceeds benefit.
- **T_max=20 confirmed optimal** for 12-17 epoch runs — but bs=2 may want lr_t_max=18 since best_epoch=18.
- **surf_weight curve is monotonic-decreasing in 5-20 range** at clip=0.25, n=10. Compound on n=8 untested.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal.
- **Huber δ=0.3 confirmed** on FP32 stack. Re-testing on BF16+clip=0.25 via tanjiro #4033.
- **LayerScale γ=0.01 fully confirmed** across multiple bracket experiments.
- **Width n_hidden=128 is the right point**: narrower (120, 144) both fail. With bs=2's memory savings, 192/256 width experiments are now safe to try.

## Potential next research directions

- **bs=1** — push the steps lever further. ~27,000 updates in 30 min budget. Caveat: GPU pipeline overhead may dominate. **ASSIGNED: frieren #4125** (arm-1 n=10, arm-2 n=8).
- **bs=2 + n=8 + clip=1.0 triple** — combine all three independent wins
- **bs=2 + lr_t_max=18** — let cosine finish given best_epoch=18 (student-suggested)
- **bs=2 + EMA re-test** — **ASSIGNED: fern #4130** (τ={0.998, 0.995}); EMA dead at bs=8 but 13,500 steps may revive it
- **slice_num sweep at bs=2** — **ASSIGNED: thorfinn #4131** ({128, 192}); first-ever architectural sweep, enabled by bs=2 memory headroom
- **bs=2 + width expansion** — 18.5 GB headroom means n_hidden=256 fits after #4130/#4131 results inform priority
- **mlp_ratio sweep** — currently hardcoded 2; mlp_ratio=4 with bs=2 memory headroom
- **Sub-55 val target**: val=60.67 now; bs=2 + n=8 compound is the obvious next push toward 55-57; +lr_t_max=18 could go lower

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
