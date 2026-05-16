# SENPAI Research State

- **Date:** 2026-05-16 17:10
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4009 (nezuko, merged):** BF16 + LayerScale γ=0.01 + n_freqs=10 + **clip=1.0** (no EMA)
- **val_avg/mae_surf_p: 65.70** | **test_avg/mae_surf_p: 57.80**
- Per-split test surf_p: single=65.24, rc=71.43, cruise=38.31, re_rand=56.21
- **Cumulative improvement: -49.0% val from round-5 start (~128.69)**
- Key insight: clip=0.25 was double-regularizing with LayerScale's gating — clip_frac=1.000 throughout (every step was clipped). clip=1.0 frees ~5% of late steps; the improvement is mostly "larger effective step" (clip acts as lr-scale in clip-bound regime). LayerScale γ stable throughout.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
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
| **#4009 (nezuko, merged)** | **BF16 + LS + n10 + clip=1.0** | **65.70** | **57.80** | **-2.22%** |

## Active WIP (8 students, full deck)

| Student | PR | Hypothesis | Status | Baseline |
|---|---|---|---|---|
| edward | #4053 | n_freqs sweep {8, 12} on BF16+LS+n10+**clip=1.0** | wave-12 NEW | new 65.70 |
| nezuko | #4052 | Clip ceiling {2.0, 4.0} on BF16+LS+n10+clip=1.0 | wave-12 NEW | new 65.70 |
| tanjiro | #4033 | Huber δ sweep {0.15, 0.5} on BF16+LS+n10+clip=0.25 | wave-11 WIP | old 67.19 |
| fern | #4006 | n_freqs sweep {8, 12} on BF16+LS+clip=0.25 | wave-11 WIP | old 67.19 |
| thorfinn | #4008 | surf_weight sweep {5.0, 20.0} on BF16+LS+n10+clip=0.25 | wave-11 WIP | old 67.19 |
| alphonse | #4026 | Batch size sweep {2, 8} on BF16+LS+n10+clip=0.25 | wave-11 WIP | old 67.19 |
| askeladd | #4027 | LR sweep {7e-4, 1e-3} on BF16+LS+n10+clip=0.25 | wave-11 WIP | old 67.19 |
| frieren | #4014 | Width scaling narrower: n_hidden=120 on BF16+LS+n10+clip=0.25 | wave-11 WIP | old 67.19 |

**Note:** wave-11 PRs (#4006, #4008, #4014, #4026, #4027, #4033) were assigned against the old clip=0.25 baseline (67.19). Any result that beats 65.70 is still a new winner. Results that beat 67.19 but not 65.70 confirm the variable's effect direction but won't merge until superseded or re-tested on clip=1.0.

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
| #4005 (tanjiro) | BF16+LS+n10+EMA 0.998: val=68.64 (worse). EMA dead on BF16 — window covers only 41% of training |
| #3971 (edward) | EMA warm-up ramp on FP32 triple: arm-1 val=74.11 (+4.1%), arm-2 val=106.71 (disaster). Superseded by BF16+clip=1.0 stack |

## Current research themes

### Wave-12: BF16+LS+n10+clip=1.0 (new default stack)

Clip=1.0 opened the new baseline. The clip-as-lr-scale mechanism means:
- All clip sweep experiments on old stack now complete: clip={0.1, 0.25, 0.5, 1.0} fully mapped
- clip=1.0 is the new default for all future BF16 work
- **Two critical next questions:**
  1. **Does clip ceiling continue? (nezuko #4052)** clip={2.0, 4.0} — if clip_frac keeps dropping, we gain more honest gradients; if it destabilizes, we've found the LayerScale-protected ceiling
  2. **Does n_freqs optimum shift at clip=1.0? (edward #4053)** n={8,12} — larger effective step may change the aliasing tradeoff

### Wave-11: Old clip=0.25 sweeps (completing)

Six PRs still running on old 67.19 baseline. Results valuable as each variable is being isolated:
- **tanjiro #4033**: Huber δ bracket — does BF16's 17-epoch horizon change optimal δ from 0.3?
- **fern #4006**: n_freqs bracket (parallel to edward but at clip=0.25) — can be compared
- **thorfinn #4008**: surf_weight sweep — never tested in programme; high priority to understand
- **alphonse #4026**: batch_size sweep — first ever; BF16 freed memory makes bs=8 viable
- **askeladd #4027**: LR sweep — re-test LR at extended 17-epoch BF16 horizon
- **frieren #4014**: Width n=120 bracket

## Key insights accumulated

- **clip=1.0 is the new default** on BF16+LS+n10 stack. clip was double-regularizing with LayerScale's gating.
- **Clip acts as lr-scale in the fully-clipped regime.** clip × grad/‖grad‖ = effective step direction + magnitude. Prior clip=0.25 was 4× under-stepping vs clip=1.0.
- **BF16 regime shift**: n_freqs=10 > n_freqs=14 at 17 epochs. Everything from FP32 triple needs re-evaluation on BF16+LS+n10+clip=1.0.
- **EMA on BF16 stack is definitively dead.** All three EMA tests regress: n14+EMA (68.50), n10+EMA (68.64), FP32 EMA warm-up (74.11). EMA's smoothing window covers only 41% of training at 30-min budget.
- **T_max=20 confirmed optimal** for 12-17 epoch runs.
- **surf_weight=10.0 never swept**: Major hyperparameter unexplored — high priority.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal. Inner-optimizer exhausted.
- **Huber δ=0.3 confirmed** on FP32 stack. Re-testing on BF16+clip=1.0 via #4033.
- **LayerScale γ=0.01 fully confirmed** (three bracket experiments all regress from this value).

## Potential next research directions

- **n_freqs=6** — if n=8 wins on edward/fern sweeps, continue the aliasing trend lower
- **clip=2.0/4.0 ceiling** — nezuko #4052 will decide whether clip can keep going
- **surf_weight on clip=1.0 stack** — re-test once thorfinn #4008 result comes in
- **Batch size on clip=1.0 stack** — once alphonse #4026 result comes in
- **Width n=160** — once frieren #4014 n=120 bracket result is clear
- **n_layers depth scaling** — BF16 memory headroom; requires train.py modification
- **Sub-60 val target**: val=65.70 now; clip ceiling + n_freqs optimum could push to 62-63

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
