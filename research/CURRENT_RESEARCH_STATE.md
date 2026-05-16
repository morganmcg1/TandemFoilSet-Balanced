# SENPAI Research State

- **Date:** 2026-05-16 04:36
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3438 (nezuko, merged):** Fourier n_freqs=14 + Huber-0.3 + T_max=20 + clip=0.25
- **val_avg/mae_surf_p: 81.08** | **test_avg/mae_surf_p: 71.52**
- Per-split test surf_p: single=81.31, rc=84.95, cruise=49.89, re_rand=69.92
- **Note: clip=1.0 (from PR #3529) and n_freqs=14 (from PR #3438) were merged in parallel; the combined stack (n=14+clip=1.0) hasn't been tested yet — this is PR #3650 (nezuko), highest priority**

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | 98.62 | 88.14 | -4.4% |
| #3221 (nezuko, merged) | Fourier n=10 + Huber-0.3 | 89.27 | 79.43 | -9.5% |
| #3333 (frieren, merged) | Fourier+Huber+T_max=20+clip=0.25 | 84.59 | 73.89 | -5.2% |
| #3529 (frieren, merged) | clip=0.25→1.0 on full stack | 84.01 | 72.95 | -0.7% |
| **#3438 (nezuko, merged)** | **n_freqs=14 on full stack** | **81.08** | **71.52** | **-3.5%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| nezuko | #3650 | n_freqs=14+clip=1.0 AND n_freqs=18+clip=1.0 | NEW (wave-6) — highest priority |
| frieren | #3648 | LR linear warmup {1,2 ep} + clip=1.0 on full stack | NEW (wave-6) |
| alphonse | #3593 | LayerScale γ-init∈{0.01,0.1} on full stack | WIP (training) |
| tanjiro | #3527 | Mixed precision BF16 + n_hidden=128/160 | WIP (training) |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta | WIP (training) |
| thorfinn | #3682 | Peak LR sweep lr∈{7e-4, 1e-3} on n_freqs=14+clip=1.0 | NEW (wave-7) |
| edward | #3192 | EMA decay=0.999 + T_max=20 on full stack | Needs rebase |
| fern | #3439 | Gaussian RFF σ∈{3,7} on full stack + T_max=20 | Sent back (rebase+σ refinement) |

## Closed this round

| PR | Reason |
|---|---|
| #3225 (tanjiro) | Multiscale (32+128) too slow: only 8ep in budget |
| #3178 (alphonse) | Per-sample scale + Huber anti-synergistic |
| #3334 (tanjiro) | n_hidden=192: 5× per-epoch slowdown, only 9ep |
| #3355 (alphonse) | Physics features: MLP synthesizes implicitly |
| #3199 (fern) | Dualhead redundant with Huber loss |
| #3420 (alphonse) | Log-space loss: anti-aligned with MAE |
| #3419 (tanjiro) | n_hidden=160: 4.6× per-epoch slowdown, only 10-11ep |
| #3509 (alphonse) | DropPath: underfit regime, 5-block net, convergence-speed penalty |
| #3227 (thorfinn) | Surf-anneal: 13h stale, no rebase after 2 requests, advisor branch moved 4+ merges ahead |

## Current research themes

1. **Compose merged wins (nezuko #3650) — HIGHEST PRIORITY**: clip=1.0 and n_freqs=14 were merged from parallel branches (both vs clip=0.25 base). Their combination is untested. arm-1 = n_freqs=14+clip=1.0 → expected ~80.5 val; arm-2 = n_freqs=18+clip=1.0 → spectrum scaling continues.

2. **LR warmup (frieren #3648)**: Cold-start penalty visible at every epoch-1 (val ~210). clip=1.0's diagnostic showed clip_frac breaks below 1.0 at epoch 10, not earlier — warmup attacks the root cause (high gnorm_max=48 at epoch 1 due to random init).

3. **BF16 speed (tanjiro #3527)**: 2× speedup would unlock n_hidden=160 (killed by 4.6× slowdown), LayerScale longer runs, and more epochs for all experiments.

4. **EMA checkpoint averaging (edward #3192)**: Confirmed large effect (-14.3% val contribution from EMA alone). Retesting with T_max=20. If EMA composes with n_freqs=14+clip=1.0, could land below val=75.

5. **RFF Gaussian σ refinement (fern #3439)**: σ=5 competitive, σ=1→5 monotonically improved. Testing σ∈{3,7} on full stack with T_max=20.

6. **LayerScale (alphonse #3593)**: Zero convergence penalty. Per-channel attenuation of residuals. Expected OOD improvement.

7. **Tighter clip + Huber delta (askeladd #3424)**: Exploring clip=0.1.

8. **Peak LR sweep (thorfinn #3682)**: default lr=5e-4 set pre-Fourier, pre-clip. With clip=1.0 no longer saturating (clip_frac=0.984 at ep14), effective step size is now LR-bound for the first time. Testing lr=7e-4 and lr=1e-3 on the full stack (n_freqs=14 + clip=1.0).

## Key insights accumulated

- **Fourier scaling not saturated**: n_freqs: 6→10 (+9.5%), 10→14 (+4.2%). n=12 is mixed; n=14 improves all splits. n=18 warranted.
- **clip_frac=1.000 → 0.984**: clip=1.0 is FIRST threshold where clip stops saturating within budget (epoch 10 onwards). clip=0.25 and 0.5 are both fully saturated throughout.
- **EMA confirmed effective**: -14.3% val gain from EMA alone at epoch 13 (decay=0.999). Fills averaging window in ~3 epochs.
- **Parallel merges create untested combination**: clip=1.0 × n_freqs=14 is highest priority to test.

## Potential next research directions

- **n_freqs=20** — if n=18 still saturates (timeout-bound), scale further
- **EMA + n_freqs=14 + clip=1.0** — triple compound after edward rerun
- **LR warmup + everything** — may replace need for tight clip
- **Higher peak LR (7e-4, 1e-3)** — IN FLIGHT (thorfinn #3682)
- **Lookahead optimizer** — wrap AdamW
- **SAM** — flat minima optimization

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
