# SENPAI Research State

- **Date:** 2026-05-15 23:40
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3333 (frieren, merged):** Fourier n=10 + Huber-0.3 + LR T_max=20 + clip=0.25
- **val_avg/mae_surf_p: 84.59** | **test_avg/mae_surf_p: 73.89**
- Per-split test surf_p: single=86.87, rc=86.21, cruise=51.47, re_rand=71.01
- Cumulative improvement: -34% val from round-5 start (~128.69)
- Key insight: All four orthogonal improvements compose cleanly. Monotone val improvement throughout all 14 epochs — model still learning at timeout. clip_frac=1.0 every step indicates clip fires on all batches, not just heavy-tail spikes.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | 98.62 | 88.14 | -4.4% |
| #3221 (nezuko, merged) | Fourier n=10 + Huber-0.3 | 89.27 | 79.43 | -9.5% |
| **#3333 (frieren, merged)** | **Fourier+Huber+T_max=20+clip=0.25** | **84.59** | **73.89** | **-5.2%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| frieren | #3529 | Grad-clip sweep max_norm∈{0.5,1.0} on full stack | NEW (wave-5) |
| tanjiro | #3527 | Mixed precision BF16, n_hidden=128 + n_hidden=160 | NEW (wave-5) |
| alphonse | #3509 | Stochastic depth DropPath drop_path∈{0.05,0.10} | NEW (wave-5) |
| nezuko | #3438 | n_freqs=14 on full stack (deconfounded rerun) | Sent back (rebase) |
| fern | #3439 | Gaussian random Fourier features σ∈{1.0,5.0} | WIP (training now) |
| askeladd | #3424 | Tighter clip max_norm=0.1 × Huber delta sweep | WIP (training now) |
| thorfinn | #3227 | Surf-anneal + full stack | Needs rebase (stale) |
| edward | #3192 | EMA checkpoint averaging on full stack | Stale (rate-limit lockout) |

## Closed this round

| PR | Reason |
|---|---|
| #3225 (tanjiro) | Multiscale (32+128) too slow: only 8ep in budget |
| #3178 (alphonse) | Per-sample scale + Huber anti-synergistic |
| #3334 (tanjiro) | n_hidden=192: 5× per-epoch slowdown, only 9ep |
| #3355 (alphonse) | Physics features: MLP synthesizes implicitly |
| #3199 (fern) | Dualhead redundant with Huber loss |
| #3420 (alphonse) | Log-space loss: gradient inversely proportional to |p|, anti-aligned with MAE |
| #3419 (tanjiro) | n_hidden=160: 4.6× per-epoch slowdown, only 10-11ep |

## Current research themes

1. **Clip threshold optimization** — clip_frac=1.0 every step suggests 0.25 may be over-tight with current full stack. Frieren (#3529) testing 0.5/1.0 to find the right operating point.

2. **Speed infrastructure** — Tanjiro (#3527) testing BF16 mixed precision. If 2× speedup materializes, we get 20+ epochs in budget AND n_hidden=160 becomes viable again.

3. **Regularization for OOD** — Alphonse (#3509) testing stochastic depth. Both OOD splits (geom_camber_rc, re_rand) still at 71-86 MAE; regularization targeting these is the priority.

4. **Fourier frequency scaling** — n=6→10 showed massive gains; n=12/14+clip was confounded. Nezuko (#3438) deconfounding with n=14 on full stack.

5. **Unexplored orthogonal levers** — Fern (#3439) testing Gaussian random Fourier; askeladd (#3424) testing tighter clip/lower Huber delta; thorfinn (#3227) surf-anneal; edward (#3192) EMA averaging.

## Potential next research directions

- **LR warmup (1-2 epochs)** — epoch-1 val (~210) >> epoch-2 val (~167) in all runs; warmup could reduce this cold-start penalty and unlock higher effective LR
- **lr_t_max sweep {18, 22, 25}** — T_max=20 beats T_max=14; optimum not yet found  
- **Fourier on other geometric inputs** — arc-length (saf), shape descriptors could benefit
- **Higher peak LR** — now that clip+Fourier reshape gradient landscape, lr=7e-4 or 1e-3 untested
- **AMP + n_hidden scaling** — if BF16 works, revisit n_hidden=160 with the additional epochs
- **Relative position attention** — rotary embeddings for pairwise mesh-node distances

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
