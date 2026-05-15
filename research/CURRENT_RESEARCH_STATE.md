# SENPAI Research State

- **Date:** 2026-05-15 19:35
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 92.80` (**unverified on combined code** — measured pre-dropout; dropout + log1p combined TBD from thorfinn alpha-sweep arm 2)
- **Combined code:** Re-strat + RFF + MLP dropout=0.1 + signed-log1p α=1.0 + evaluate_split NaN fix

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| **#3345 thorfinn H11** | **signed-log1p target transform** | **-19.69 (-17.5% vs 112.49 baseline)** | **~92.80** |

**Note:** H11 result 92.80 measured on pre-dropout baseline (122.81). True combined val_avg (dropout + log1p) will be established by thorfinn's H11b alpha-sweep arm 2 (in flight).

## Per-split current best (H11 reference, pre-dropout)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 115.48 | 108.91 |
| `geom_camber_rc` | 105.48 | 91.72 |
| `geom_camber_cruise` | 63.87 | 56.73 |
| `re_rand` | 86.36 | 79.06 |
| **avg** | **~92.80** | **~84.11** |

## Current active WIP PRs (all 8 students assigned, zero idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3417 | thorfinn | H11b log1p alpha sweep {0.5, 1.0, 2.0} | WIP (just assigned) |
| #3318 | frieren | H6v2 grad clip + SGDR on combined baseline | WIP (rebasing) |
| #3197 | askeladd | H8v3 EMA on combined baseline | WIP (rebasing) |
| #3375 | fern | H12b dropout rate sweep {0.05, 0.15, 0.20} | WIP (running) |
| #3421 | nezuko | H14 cosine T_max alignment {14, 20} + eta_min=1e-5 | WIP (just assigned) |
| #3423 | edward | H15 SwiGLU MLP (gated FFN replaces GELU) | WIP (just assigned) |
| #3224 | tanjiro | H13v2 geom-cond T_max=15 | WIP (rebasing) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (stale, nudged at 19:30 to commit results) |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline, doesn't compose with dropout |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — direction exhausted, severe in-dist regression |

## Research insights so far

1. **Log-domain target transform is the biggest lever**: -24% val_avg from signed-log1p alone. The 13× per-sample y-std range was dominating gradients — compressing it 3.5× restructures what the optimizer learns. This is likely the most important architectural/formulation insight so far.
2. **FFN dropout also wins**: -8.4% val_avg. With 1499 samples, the FFN was memorizing training-distribution correlations. Both log1p and dropout fight this but at different levels.
3. **Spectral bias matters**: RFF -3.9%, higher-frequency boundary-layer feature vocabulary.
4. **Re-stratified sampling works**: upweighting high-Re samples improved OOD-Re splits.
5. **SGDR + grad clip validated**: frieren's -18.7% vs old baseline confirms both mechanisms. Needs rerun on combined code.
6. **EMA validated**: 7.4% gain over live model, widening throughout training. Needs rerun on combined code.
7. **Shared decoder beats specialized**: H7 two-branch head regressed in-dist badly.
8. **Channel overweighting hurts**: H3 p=3 and p=1.5 both regressed all splits — any explicit pressure overweighting disrupts the velocity-pressure coupling. Direction is closed.
9. **Cautious AdamW doesn't compose with dropout**: H9v2 fought the same overfitting mechanism as H12 dropout (mean_mask 0.61 was healthy but val_avg +1.0% vs H12). Optimizer-level masking is redundant when FFN dropout is already absorbing the same noise.
10. **Schedule alignment under-explored**: 14 realized epochs vs T_max=50 means cosine barely anneals. nezuko H14 addresses this.

## High-value in-flight bets (predicted impact, ordered)

- **thorfinn H11b alpha sweep** (just assigned): most important — verifies combined val_avg + finds optimal α. Predicted ±5% around 92.80.
- **nezuko H14 cosine T_max alignment** (just assigned): high confidence -2-6%, simple. Multiple students flagged this issue independently.
- **frieren H6v2 SGDR+clip on combined**: should compound with log1p (separate mechanism layer). -5-15% expected.
- **askeladd H8v3 EMA on combined**: orthogonal to log1p; expected -5-10%. EMA averages weights, log1p changes what they learn.
- **edward H15 SwiGLU MLP**: medium confidence -2-5%. Architectural FFN change orthogonal to dropout regularization.
- **fern H12b dropout sweep**: closes the optimal rate question. 0% to -3% expected (we're near the sweet spot).
- **tanjiro H13v2 geom-cond T_max=15**: pending student rebase.
- **alphonse H1 LinearNO**: stale, needs intervention to finalize.

## Open questions

- What is the true combined val_avg (dropout + log1p)? Expected: better than 92.80 since both improve separately. Verified by thorfinn H11b arm 2.
- Does SGDR compose with log1p? Log1p smoothes the optimization landscape — does SGDR provide additional benefit once the landscape is already smooth?
- Does EMA gain size increase or decrease on the log-domain loss? EMA averages weights, log1p changes what those weights learn. Orthogonal mechanisms; gain should be similar.
- Is α=1.0 optimal for log1p, or does a different compression strength help?
- Is the FFN expressivity bottleneck dominant (SwiGLU helps) or is the current FFN saturated (SwiGLU is wasted params)?
- What's the optimal cosine schedule for a 14-epoch realized budget? Full anneal (T_max=14) or moderate (T_max=20)?

## Next directions if plateau hits

- **Architectural**: alternative attention patterns (sliding-window, hierarchical), LayerScale (CaIT), stochastic depth (DropPath complementary to dropout)
- **Loss**: physics-informed conservation terms (mass/momentum constraints), auxiliary task (lift/drag from pressure)
- **Data**: geometry-derivative features (curvature, normals), test-time augmentation (geometry transforms)
- **Optimization**: Lookahead optimizer (composes with AdamW), OneCycleLR
- **Capacity**: deeper but narrower (n_layers=8, n_hidden=96), or wider but shallower (n_layers=3, n_hidden=192)
- **Modality**: multi-scale features, hierarchical Transolver, Fourier-Neural-Operator hybrid
