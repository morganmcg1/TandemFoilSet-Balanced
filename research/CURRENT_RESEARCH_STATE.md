# SENPAI Research State

- **Date:** 2026-05-15 21:35
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 85.16` (PR #3224 tanjiro H13 geom-cond GALE, on full combined stack)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| #3345 thorfinn H11 | signed-log1p target transform | -19.69 (-17.5% vs 112.49) | ~92.80 (pre-dropout, unverified combined) |
| **#3224 tanjiro H13** | **GALE-style geom-cond per block + T_max=15** | **-7.64 (-8.2% vs 92.80)** | **85.16** |

## Per-split current best (H13 reference, full combined stack)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 106.160 | 99.658 |
| `geom_camber_rc` | 92.098 | 84.121 |
| `geom_camber_cruise` | 61.360 | 52.932 |
| `re_rand` | 81.005 | 73.739 |
| **avg** | **85.16** | **77.61** |

## Current active WIP PRs (all 8 students assigned, zero idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3417 | thorfinn | H11b log1p alpha sweep {0.5, 1.0, 2.0} | WIP (running) |
| #3318 | frieren | H6v2 grad clip + SGDR on combined baseline | WIP (running, active at 21:24) |
| #3197 | askeladd | H8v3 EMA on combined baseline | WIP (running, pod restarted 21:25) |
| #3375 | fern | H12b dropout rate sweep {0.05, 0.15, 0.20} | WIP (running) |
| #3421 | nezuko | H14 cosine T_max alignment {14, 20} + eta_min=1e-5 | WIP (just assigned — note: baseline now has T_max=15 baked in) |
| #3423 | edward | H15 SwiGLU MLP (gated FFN replaces GELU) | WIP (just assigned) |
| **#3461** | **tanjiro** | **H16 FiLM geom-cond (γ + β, extends H13 GALE)** | **WIP (just assigned)** |
| #3184 | alphonse | H1 LinearNO ablation | WIP (running, pod restarted 21:22) |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline, doesn't compose with dropout |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — direction exhausted, severe in-dist regression |

## Research insights so far

1. **GALE-style geometry conditioning works**: -8.2% val_avg from additive geom-cond with learned per-block gates. camber_rc split benefited most (-12.7%), confirming OOD geometry interpolation regime. T_max=15 alignment critical — made the cosine schedule actually anneal within the 30-min cap.
2. **Log-domain target transform is the biggest lever**: -17.5% val_avg from signed-log1p. The 13× per-sample y-std range was dominating gradients.
3. **FFN dropout also wins**: -8.4% val_avg. With 1499 samples, the FFN was memorizing.
4. **Spectral bias matters**: RFF -3.9%, higher-frequency boundary-layer feature vocabulary.
5. **Re-stratified sampling works**: upweighting high-Re samples improved OOD-Re splits.
6. **SGDR + grad clip validated on RFF baseline**: -18.7% vs old baseline (99.85). Needs rerun on full combined code — frieren running now.
7. **EMA validated on Re-strat+RFF baseline**: +7.4% gain over live weights. Needs rerun on full combined code — askeladd running now.
8. **Shared decoder beats specialized**: H7 two-branch head regressed in-dist badly.
9. **Channel overweighting hurts**: H3 p=3 and p=1.5 both regressed all splits.
10. **Cautious AdamW doesn't compose with dropout**: H9v2 redundant when FFN dropout absorbs the same overfitting noise.
11. **Schedule alignment matters**: T_max=15 > T_max=50 when realized epochs ≈ 14. The cosine schedule needs to align to actual training budget, not nominal.

## High-value in-flight bets (predicted impact, ordered)

- **frieren H6v2 SGDR+clip on combined**: SGDR validated at -18.7% vs old baseline. On new 85.16 baseline with log1p smoothing landscape, SGDR may compound less, but direction is sound. -2-10% expected.
- **askeladd H8v3 EMA on combined**: EMA is orthogonal to geom-cond and log1p. -3-8% expected.
- **thorfinn H11b alpha sweep**: Verifies true combined baseline + optimal α. Current α=1.0 is a reasonable default but α=2.0 (more compression) could be better for the tail of the y distribution.
- **tanjiro H16 FiLM geom-cond**: Multiplicative path on top of additive. Gates showed monotone growth — more geometry capacity likely helps. -2-6%.
- **nezuko H14 cosine T_max**: With T_max=15 now baked in from H13, this tests T_max=14 vs 15 vs 20. Most important thing: eta_min=1e-5 floor (current baseline has no eta_min). -1-3%.
- **edward H15 SwiGLU MLP**: Architectural FFN change. -2-5%. Medium confidence — small dataset may not benefit from extra FFN capacity.
- **fern H12b dropout sweep**: Closes optimal rate question. 0-3% expected.
- **alphonse H1 LinearNO**: Stale, low confidence. Informative as an ablation.

## Open questions

- Does SGDR compose with log1p + geom-cond? Log1p smoothes the landscape; geom-cond adds conditioning; does SGDR still provide additional benefit?
- Does EMA gain size hold on the full combined stack? Dropout and EMA both fight overfitting; do they conflict on 1499 samples?
- Is α=1.0 optimal for log1p, or does α=2.0 (more compression) help on the tail of the y distribution?
- Is FiLM's multiplicative path better than additive-only for 11-dim geometry context?
- Does the multiplicative path change which split benefits most (H13 saw camber_rc +12.7%)?

## Next directions if plateau hits

- **Architectural**: alternative attention patterns (sliding-window, hierarchical), LayerScale (CaIT), stochastic depth (DropPath complementary to dropout)
- **Loss**: physics-informed conservation terms (mass/momentum constraints), auxiliary task (lift/drag from pressure)
- **Data**: geometry-derivative features (curvature, normals), test-time augmentation (geometry transforms)
- **Optimization**: Lookahead optimizer (composes with AdamW), OneCycleLR
- **Capacity**: deeper but narrower (n_layers=8, n_hidden=96), or wider but shallower (n_layers=3, n_hidden=192)
- **Conditioning**: per-block geom_proj (shared vs independent), LoRA-style low-rank delta, FiLM on slice tokens (not hidden state)
