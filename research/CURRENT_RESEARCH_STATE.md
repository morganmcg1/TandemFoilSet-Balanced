# SENPAI Research State

- **Date:** 2026-05-16 08:35
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3593 (alphonse, merged):** Fourier n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 + **LayerScale γ-init=0.01**
- **val_avg/mae_surf_p: 72.77** | **test_avg/mae_surf_p: 65.12**
- Per-split test surf_p: single=78.83, rc=75.82, cruise=43.86, re_rand=61.97
- **Cumulative improvement: -43.5% val from round-5 start (~128.69)**
- **HIGHEST PRIORITY: LayerScale + n_freqs=14 compound (alphonse #3730)**
- **UPCOMING HIGH VALUE: BF16 + LayerScale composition (tanjiro #3527 rebasing)**

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
| alphonse | #3730 | LayerScale γ=0.01 + n_freqs=14 compound | Wave-7 — HIGHEST PRIORITY |
| nezuko | #3823 | Lookahead optimizer wrapper {k=5, k=10} on LayerScale stack | NEW (wave-8) |
| frieren | #3740 | Asymmetric LayerScale γ_attn=0.001 vs γ_mlp=0.01/0.03 | Wave-7 |
| edward | #3192 | EMA 0.998 on LayerScale stack | Sent back (rebase in progress) |
| askeladd | #3424 | Tighter clip sweep max_norm=0.1 × Huber delta | WIP (stale — picking up after API rate-limit recovery) |
| tanjiro | #3527 | BF16 + LayerScale composition | Sent back (rebase onto LayerScale — arm-1 BF16 tied val=72.75; compose BF16+LayerScale expected ~65-70) |
| fern | #3782 | AdamW eps sweep {1e-6, 1e-7} on LayerScale stack | NEW (wave-8) |
| thorfinn | #3784 | Peak LR sweep {7e-4, 1e-3} on LayerScale stack | NEW (wave-8) |

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
| #3682 (thorfinn) | LR sweep on wrong stack (n14+clip=1.0 vs current LayerScale best); arm-2 (lr=1e-3) promising but retesting on LayerScale |
| #3732 (nezuko) | Fourier saturated at n=14: n=18 val=83.06, n=20 val=81.24 — both much worse than LayerScale. Stop scaling n_freqs. |

## Current research themes

1. **LayerScale + n_freqs=14 compound (alphonse #3730) — HIGHEST PRIORITY**: LayerScale γ=0.01 won by -10.2% (val=72.77) on n_freqs=10. n_freqs=14 won by -4.2% (val=81.08). Never combined. Expected val ~65-70 if composition holds. Use clip=0.25.

2. **BF16 + LayerScale composition (tanjiro #3527 rebasing)**: BF16 alone ties LayerScale at val=72.75 (+4 epochs/1.30× speed). Composition BF16+LayerScale expected val 65-70. Needs rebase.

3. **Lookahead optimizer on LayerScale (nezuko #3823)**: Zhang et al. 2019 slow anchor weights with k-step pull-back. In heavy-tail/high clip_frac regime, Lookahead provides variance reduction without the β2-lowering instability (confirmed harmful by PR #3708). Two arms: k=5 (default) and k=10 (longer lookback). Near-zero compute overhead.

4. **Asymmetric LayerScale (frieren #3740)**: γ-attn stays near 0.01, γ-mlp grows 3×. Separate inits (γ_attn=0.001, γ_mlp=0.01/0.03) target natural trajectory. Expected 1-3% beyond symmetric γ=0.01.

5. **EMA 0.998 on LayerScale stack (edward #3192)**: EMA 0.998 earlier beat old n14 baseline by -1.16%. On LayerScale stack expected val ~67. Triple compound (LayerScale + n14 + EMA) could reach sub-60.

6. **Peak LR sweep on LayerScale (thorfinn #3784)**: lr=1e-3 showed healthy gradient stats in PR #3682 but on wrong stack. Retesting on current best: LayerScale gating may tolerate higher base LR as γ damps attn branch updates.

7. **AdamW eps sweep on LayerScale (fern #3782)**: eps ∈ {1e-6, 1e-7} on LayerScale stack. Near-zero γ-attn channels have small v_t → tiny eps amplifies their updates. Raising eps damps → more uniform per-param effective LR.

8. **Tighter clip sweep (askeladd #3424)**: clip=0.1 + Huber delta sweep. Currently stale after API rate-limit recovery.

## Key insights accumulated

- **Fourier scaling SATURATED at n=14**: n_freqs: 6→10 (+9.5%), 10→14 (+4.2%), 14→18 (+2.4% regression), 14→20 (+0.2% regression, essentially tied). Non-monotone pattern above n=14. Stop scaling n_freqs. n=14 is sweet spot in this budget.
- **clip=0.25 is regularization at n_freqs≥14**: PR #3650 confirmed — clip=1.0 removes implicit regularization at high n_freqs. Always use clip=0.25 when n_freqs≥12.
- **LayerScale γ=0.01 is the biggest single win**: -10.2% val from per-channel selective residual gating. OOD splits improve most.
- **BF16 virtual tie with LayerScale (independent paths to same performance)**: BF16 extra epochs ≈ LayerScale gating benefit. Both likely compose.
- **AdamW β2<0.999 is harmful**: clip already detoxifies heavy-tail gradients before v_t; lower β2 introduces denominator instability. Default optimal.
- **lr=1e-3 not divergent**: clip_frac drops to 0.97s, grad_norm_mean trends lower; potentially beneficial on current stack.
- **EMA 0.998 effective**: -1.16% vs prior stack; composition with LayerScale untested.
- **γ=0.1 high variance**: run-to-run spread of 7 points; γ=0.01 cleaner dynamics.

## Potential next research directions

- **LayerScale + n_freqs=14 + EMA** — triple compound; could land sub-60 val
- **LayerScale + BF16 + n_freqs=14** — triple compound if BF16 confirmed to compose
- **LayerScale γ-init ∈ {0.003, 0.005}** — even smaller init may increase channel selectivity
- **Learned frequency basis** — Fourier scaling with fixed log-spaced freqs saturated; try learnable/adaptive freq components
- **Lookahead optimizer** — wrap AdamW (no hyperparameter changes to the inner optimizer needed)
- **SAM** — flat minima; theoretically motivated for OOD generalization

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
