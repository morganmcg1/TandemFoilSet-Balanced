# SENPAI Research State

- **Date:** 2026-05-16 01:33
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3333 (frieren, merged):** Fourier n=10 + Huber-0.3 + LR T_max=20 + clip=0.25
- **val_avg/mae_surf_p: 84.59** | **test_avg/mae_surf_p: 73.89**
- Per-split test surf_p: single=86.87, rc=86.21, cruise=51.47, re_rand=71.01
- Cumulative improvement: -34% val from round-5 start (~128.69)

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
| frieren | #3529 | Grad-clip sweep max_norm∈{0.5,1.0} on full stack | WIP (training) |
| tanjiro | #3527 | Mixed precision BF16, n_hidden=128 + n_hidden=160 | WIP (training) |
| nezuko | #3438 | n_freqs=14 on full stack (deconfounded rerun) | WIP (training) |
| askeladd | #3424 | Tighter clip max_norm=0.1 × Huber delta sweep | WIP (training) |
| thorfinn | #3227 | Surf-anneal + full stack | Needs rebase (stale, pod restarted) |
| edward | #3192 | EMA decay=0.999 + T_max=20 on full stack (rerun) | Sent back (add T_max=20) |
| fern | #3439 | Gaussian RFF σ∈{3,7} on full stack (refined sweep) | Sent back (σ refinement + T_max=20) |
| alphonse | #3593 | LayerScale γ-init∈{0.01,0.1} on full stack | NEW (wave-6) |

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
| #3509 (alphonse) | DropPath: convergence-speed penalty too high at 13-epoch budget; underfit regime |

## Current research themes

1. **EMA checkpoint averaging (edward #3192)** — Confirmed: EMA decay=0.999 delivers -14.3% val reduction at epoch 13. Without T_max=20, tied current baseline (84.61 vs 84.59). Retesting with T_max=20 — this is highest-priority pending result, expected to clearly beat 84.59.

2. **Clip threshold exploration (frieren #3529)** — clip_frac=1.0 every step means clip fires on every batch. Testing max_norm∈{0.5,1.0} to find right operating point.

3. **Speed infrastructure (tanjiro #3527)** — BF16 mixed precision. If 2× speedup confirmed, unlocks n_hidden=160 (previously infeasible at 4.6× epoch slowdown).

4. **Fourier frequency scaling (nezuko #3438)** — n=14 deconfounded on full stack (prior run was confounded with clip addition).

5. **Gaussian RFF refinement (fern #3439)** — σ=5 competitive on smoother splits; σ=1→5 monotonically improved. Retesting σ∈{3,7} on full stack. Optimum likely near σ=5-7.

6. **LayerScale (alphonse #3593)** — Per-channel residual gain (γ-init=0.01/0.1) from CaiT. Unlike DropPath, no convergence penalty — just attenuates residuals initially. Zero overhead. Expected OOD improvement through implicit per-channel regularization.

7. **Tighter clip + Huber delta (askeladd #3424)** — Clip=0.1 interaction with Huber delta sweep.

8. **Surf-weight curriculum (thorfinn #3227)** — Anneal 1→20 on full stack. Awaiting rebase.

## Key insights accumulated

- **clip_frac=1.0 throughout**: Clip at 0.25 fires on every batch. This is gradient-direction-only training — clip may be over-tight.
- **EMA confirmed effective**: decay=0.999 gives -14.3% val at epoch 13. Fills averaging window in ~3 epochs. Decay=0.9995 too slow for 14-epoch budget.
- **DropPath failure mode**: 13 epochs + 5 blocks + underfit → adding noise slows convergence; no overfitting to regularize.
- **RFF vs log-spaced**: Log-spaced 10-octave coverage beats targeted Gaussian distribution overall. RFF wins on smoother flow splits (cruise, re_rand) but loses on geom_camber_rc.

## Potential next research directions

- **LR warmup (1-2 epochs)** — epoch-1 val (~210) >> epoch-2 (~167) across all runs
- **lr_t_max sweep {18, 22, 25}** — T_max=20 beats T_max=14; optimum not found
- **Higher peak LR** — lr=7e-4 or 1e-3 untested with current Fourier+clip stack
- **AMP + n_hidden scaling** — if BF16 confirmed, revisit n_hidden=160
- **Lookahead optimizer** — wrap AdamW k=5/α=0.5
- **SAM (Sharpness-Aware Minimization)** — flat minima; may compound with EMA

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
