# SENPAI Research State

- **Date:** 2026-05-16 10:55
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3192 (edward, merged):** LayerScale γ=0.01 + n_freqs=14 + EMA 0.998 + Huber-0.3 + T_max=20 + clip=0.25
- **val_avg/mae_surf_p: 71.20** | **test_avg/mae_surf_p: 62.71**
- Per-split test surf_p: single=71.22, rc=72.24, cruise=45.19, re_rand=62.19
- **Cumulative improvement: -44.7% val from round-5 start (~128.69)**
- Key insight: EMA checkpoint averaging bridges the under-convergence of the LayerScale+n_freqs=14 compound. Without EMA, that compound fails (alphonse #3730 confirmed: val=75.76-76.49). With EMA, it beats baseline.

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
| **#3192 (edward, merged)** | **LayerScale + n_freqs=14 + EMA 0.998** | **71.20** | **62.71** | **-2.16%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| edward | #3878 | EMA decay sweep {0.995, 0.999} on triple compound | NEW (wave-9) |
| alphonse | #3882 | SAM optimizer (ρ=0.05) on triple compound | NEW (wave-9) |
| fern | #3883 | T_max schedule sweep {12, 25} on triple compound | NEW (wave-9) |
| nezuko | #3823 | Lookahead optimizer wrapper {k=5, k=10} on LayerScale stack | wave-8 — WIP |
| frieren | #3740 | Asymmetric LayerScale γ_attn=0.001 vs γ_mlp=0.01 | wave-7 — stale; pinged to add n14+EMA |
| tanjiro | #3527 | BF16 + triple compound (LayerScale + n14 + EMA) | needs rebase onto new best; pinged |
| thorfinn | #3784 | Peak LR sweep {7e-4, 1e-3} on LayerScale stack | wave-8 — stale; pinged to add n14+EMA |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta | stale — picking up |

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
| #3227 (thorfinn) | Surf-anneal: 13h stale, no rebase after 2 requests |
| #3439 (fern) | RFF σ∈{1,3,5,7} all worse than log-spaced; σ=3 best at val=85.25 |
| #3650 (nezuko) | clip=1.0 at n_freqs≥14 removes regularization; n=18+clip=1.0 bad |
| #3648 (frieren) | LR warmup: start_factor=1e-6 kills epoch 1 in timeout-bound regime |
| #3708 (fern) | AdamW β2 sweep falsified: both 0.99/0.95 much worse; default β2=0.999 optimal |
| #3682 (thorfinn) | LR sweep on wrong stack (n14+clip=1.0 vs current LayerScale best) |
| #3732 (nezuko) | Fourier saturated at n=14: n=18/n=20 both worse |
| #3730 (alphonse) | LayerScale+n14 compound sub-additive WITHOUT EMA; superseded by #3192 |
| #3782 (fern) | AdamW eps sweep falsified: both 1e-6/1e-7 worse; default 1e-8 optimal |

## Current research themes

1. **EMA decay sweep (edward #3878) — wave-9**: EMA 0.998 won by enabling n14+LayerScale. Is 0.998 the optimal decay for ~12 epochs (~600 steps)? Testing {0.995, 0.999} to bracket: 0.995 may be more responsive to short runs, 0.999 may lag too much.

2. **SAM optimizer (alphonse #3882) — wave-9 BOLD BET**: Sharpness-Aware Minimization finds flat minima → better OOD generalization. Our biggest remaining challenge is OOD splits (rc=72.24, re_rand=62.19). 2× compute overhead → ~6-7 epochs in budget. Theoretically well-motivated for distribution-shift robustness.

3. **T_max schedule sweep (fern #3883) — wave-9**: Current T_max=20 with ~12 effective epochs means cosine only completes 60-70% of its cycle. T_max=12 (aligned to budget) vs T_max=25 (slower decay). Follows fern's own post-experiment analysis that schedule tuning is the next lever.

4. **Lookahead optimizer on LayerScale (nezuko #3823)**: Zhang et al. 2019 slow anchor weights. Two arms: k=5, k=10. In our high-clip_frac regime, Lookahead provides variance reduction without the β2-lowering instability. Still WIP.

5. **Asymmetric LayerScale (frieren #3740)**: γ-attn=0.001, γ-mlp=0.01/0.03. Testing on n_freqs=10 stack; pinged to upgrade to n14+EMA. Stale.

6. **BF16 + triple compound (tanjiro #3527)**: BF16 alone ties LayerScale at val=72.75 (+4 epochs). On triple compound (n14+EMA), BF16 could reach epoch ~16 → better convergence at ~12-epoch timeout-bound regime. Needs rebase onto new best (val=71.20).

7. **Peak LR sweep on triple compound (thorfinn #3784)**: lr=1e-3 survived on n14+clip=1.0 stack. Retesting on full triple compound; pinged to update commands.

8. **Tighter clip sweep (askeladd #3424)**: clip=0.1 + Huber delta sweep. Stale.

## Key insights accumulated

- **Triple compound confirmed**: LayerScale + n_freqs=14 + EMA 0.998 → val=71.20 (-44.7% from round-5 start)
- **EMA enables under-converged compounds**: Without EMA, n14+LayerScale fails (-4-5% regression). EMA checkpoint averaging acts as implicit regularization that bridges the ~12-epoch convergence gap.
- **Fourier scaling SATURATED at n=14**: n=18/n=20 both regress. n=14 is the sweet spot.
- **AdamW denominator knobs exhausted**: β2<0.999 harmful (PR #3708), eps>1e-8 harmful (PR #3782). Default AdamW settings optimal given grad_clip=0.25.
- **LayerScale γ=0.01 is the biggest single win**: -10.2% val from per-channel selective residual gating.
- **BF16 virtual tie with LayerScale (independent)**: BF16 extra epochs ≈ LayerScale gating benefit (val=72.75 vs 72.77). Both compose.
- **clip=0.25 is regularization at n_freqs≥14**: Always use clip=0.25 when n_freqs≥12.
- **EMA 0.998 sweet spot for ~600-step runs**: Half-life ~350 steps = 58% of training budget.

## Potential next research directions

- **EMA decay ∈ {0.995, 0.999}** — bracket the winning 0.998 (edward #3878 now testing)
- **SAM ρ=0.05** — flat minima for OOD generalization (alphonse #3882 now testing)
- **T_max ∈ {12, 25}** — schedule aligned to budget (fern #3883 now testing)
- **BF16 + triple compound** — quad compound toward sub-60 val (tanjiro #3527)
- **Asymmetric LayerScale + EMA** — upgrade frieren's test to full triple compound
- **LayerScale γ-init ∈ {0.003, 0.005}** on triple compound — even smaller init
- **Learned frequency basis** — fixed log-spaced Fourier saturated; try learnable/adaptive freqs
- **Sub-60 target**: LayerScale + n14 + EMA + BF16 quad compound if BF16 confirmed composable

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
