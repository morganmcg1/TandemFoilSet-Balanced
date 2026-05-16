# SENPAI Research State

- **Date:** 2026-05-16 06:25
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3593 (alphonse, merged):** Fourier n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 + **LayerScale γ-init=0.01**
- **val_avg/mae_surf_p: 72.77** | **test_avg/mae_surf_p: 65.12**
- Per-split test surf_p: single=78.83, rc=75.82, cruise=43.86, re_rand=61.97
- **Cumulative improvement: -43.5% val from round-5 start (~128.69)**
- **Note: LayerScale + n_freqs=14 combination is untested and is the HIGHEST PRIORITY (PR #3730, alphonse)**

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
| **#3593 (alphonse, merged)** | **LayerScale γ-init=0.01 on full stack** | **72.77** | **65.12** | **-10.2%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | #3730 | LayerScale γ=0.01 + n_freqs=14 compound | NEW (wave-7) — HIGHEST PRIORITY |
| nezuko | #3732 | n_freqs={18,20} + clip=0.25 (tight clip for Fourier scaling) | NEW (wave-7) |
| frieren | #3648 | LR linear warmup {1,2 ep} + clip=1.0 on full stack | WIP (stale) |
| tanjiro | #3527 | Mixed precision BF16 training | WIP (stale) |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta | WIP (stale) |
| thorfinn | #3682 | Peak LR sweep lr∈{7e-4, 1e-3} on n_freqs=14+clip=1.0 | WIP |
| edward | #3192 | EMA decay=0.999 + T_max=20 on full stack | WIP |
| fern | #3708 | AdamW β2 sweep {0.99, 0.95} on n_freqs=14+clip=1.0 | WIP |

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
| #3439 (fern) | RFF σ∈{1,3,5,7} all worse than log-spaced; σ=3 best at val=85.25 vs baseline 81.08 (-5%) |
| #3650 (nezuko) | Compound fail: n=14+clip=1.0 val=81.20 (worse than 81.08); n=18+clip=1.0 val=84.71. clip=0.25 is regularizer at n_freqs≥14 |

## Current research themes

1. **LayerScale + n_freqs=14 compound (alphonse #3730) — HIGHEST PRIORITY**: LayerScale γ=0.01 won by -10.2% (val=72.77) on n_freqs=10. n_freqs=14 won by -4.2% (val=81.08). Never combined. Expected val ~65-70 if composition holds. NOTE: Use clip=0.25 — PR #3650 confirmed clip=1.0 doesn't help at n_freqs≥14.

2. **n_freqs=18+clip=0.25 (nezuko #3732)**: PR #3650 tested n=18+clip=1.0 (bad). The correct test n=18+clip=0.25 is untested. If Fourier scaling continues beyond n=14, it'll compound further with LayerScale.

3. **LR warmup (frieren #3648)**: Cold-start penalty visible at every epoch-1 (val ~210). clip=1.0's diagnostic showed clip_frac breaks below 1.0 at epoch 10, not earlier — warmup attacks the root cause (high gnorm_max=48 at epoch 1).

4. **EMA checkpoint averaging (edward #3192)**: Confirmed large effect (-14.3% val at epoch 13). Retesting with T_max=20. If EMA composes with LayerScale+n_freqs=14, could land below val=60.

5. **AdamW β2 sweep (fern #3708)**: Heavy-tailed gradients (Huber + clip needed) contaminate β2=0.999 second moment. Testing β2∈{0.99, 0.95}.

6. **Peak LR sweep (thorfinn #3682)**: Testing lr=7e-4 and lr=1e-3 on n_freqs=14+clip=1.0 stack.

7. **BF16 speed (tanjiro #3527)**: 2× speedup would unlock more epochs for all experiments.

8. **Tighter clip + Huber delta (askeladd #3424)**: Exploring clip=0.1.

## Key insights accumulated

- **Fourier scaling not saturated at n=14**: n_freqs: 6→10 (+9.5%), 10→14 (+4.2%). n=12 mixed; n=14 all splits improve. n=18+clip=0.25 untested.
- **clip=0.25 is regularization at n_freqs≥14**: PR #3650 confirmed — clip=1.0 removes implicit regularization at high n_freqs (cruise blows up +5-16%). Always use clip=0.25 when n_freqs≥12.
- **clip=1.0 win at n=10 was likely split-noise**: margin was -0.69% (PR #3529); same order as variation in PR #3650 arm-1 (+0.15%). Not a real structural win.
- **LayerScale γ=0.01 is the biggest win**: -10.2% val from per-channel selective residual gating. Mechanism: most γ-attn channels near zero, a few 8×; γ-mlp grows 3×. OOD splits improve most (rc -10.7%, cruise -12.1%, re_rand -11.4%).
- **EMA confirmed effective**: -14.3% val gain from EMA alone at epoch 13 (decay=0.999).
- **γ=0.1 high variance**: run-to-run spread of 7 points; γ=0.01 cleaner dynamics.

## Potential next research directions

- **LayerScale + n_freqs=14 + EMA** — triple compound if all compose; could land sub-60 val
- **LayerScale γ-init ∈ {0.003, 0.005}** — even smaller init may increase channel selectivity (γ-attn stays near zero anyway)
- **Asymmetric LayerScale** — separate init for attn vs mlp branches (γ-mlp needs bigger; γ-attn barely moves)
- **n_freqs=20** — if n=18 still saturates with clip=0.25
- **LR warmup + everything** — may improve cold-start convergence
- **Lookahead optimizer** — wrap AdamW
- **SAM** — flat minima optimization

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
