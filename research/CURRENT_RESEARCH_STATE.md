# SENPAI Research State

- **Date:** 2026-05-15 22:40
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 80.21` (PR #3423 edward H15 SwiGLU, 2 runs; seed variance ~10%)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| #3345 thorfinn H11 | signed-log1p target transform | -19.69 (-17.5%) | ~92.80 |
| #3224 tanjiro H13 | GALE-style geom-cond per block + T_max=15 | -7.64 (-8.2%) | 85.16 |
| **#3423 edward H15** | **SwiGLU gated FFN (replaces GELU)** | **-4.95 (-5.8%)** | **80.21** |

## Per-split current best (H15 reference, run 2)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 104.46 | — |
| `geom_camber_rc` | 88.50 | — |
| `geom_camber_cruise` | 53.88 | — |
| `re_rand` | 74.00 | — |
| **avg** | **80.21** | **73.20** |

Note: per-split test metrics not separately reported by edward. test_avg=73.20.

## Current active WIP PRs (8 students, edward idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3417 | thorfinn | H11b log1p alpha sweep {0.5, 1.0, 2.0} | WIP (training done, awaiting commit post rate-limit) |
| #3318 | frieren | H6v2 grad clip + SGDR on combined baseline | WIP (training done, awaiting commit post rate-limit) |
| #3197 | askeladd | H8v3 EMA on combined baseline | WIP (training done, awaiting commit post rate-limit) |
| **#3421** | **nezuko** | **H14 cosine T_max {14, 20} + eta_min=1e-5** | **status:review (both arms done, SENPAI-RESULT posted 22:33)** |
| #3461 | tanjiro | H16 FiLM geom-cond (γ + β, extends H13 GALE) | WIP (training started, post-rate-limit) |
| #3467 | fern | H17 attention dropout sweep {0.05, 0.10} | WIP (training started, post-rate-limit) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (training done, awaiting commit post rate-limit) |
| _(edward)_ | edward | _idle — assigning H18 LayerScale_ | _to assign_ |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep {0.05, 0.15, 0.20} | Closed — U-shape minimum at 0.10 |

## Research insights so far

1. **SwiGLU FFN is a structural win**: -5.8% from replacing GELU with gated multiplicative FFN. OOD gains 1.5–1.7× larger than in-dist — gate modulation reduces co-adaptation like dropout but structurally. Best epoch at 10 (faster convergence).
2. **GALE-style geometry conditioning works**: -8.2% from additive geom-cond with zero-init gates. Camber_rc split benefited most (-12.7%). T_max=15 alignment critical.
3. **Log-domain target transform is the biggest lever**: -17.5% from signed-log1p. Compresses 13× y-std range.
4. **FFN dropout confirmed optimal at 0.10**: U-shape minimum, all alternatives regress. Narrow basin.
5. **Spectral bias matters**: RFF -3.9%.
6. **Re-stratified sampling works**: upweighting high-Re samples improved OOD-Re splits.
7. **SGDR + grad clip validated**: -18.7% vs old baseline, needs rerun on new stack.
8. **EMA validated**: +7.4% gain over live, needs rerun on new stack.
9. **Schedule alignment matters**: T_max=15 > T_max=50 for 14 realized epochs.
10. **Seed variance is real with SwiGLU**: H15 showed 10% val_avg spread between 2 seeds. Gating amplifies lucky init states. Keep this in mind for future architectural changes.

## High-value in-flight bets

- **nezuko H14 cosine T_max**: Both arms completed (22:33 UTC SENPAI-RESULT). Need to read results and decide. With T_max=15 already baked, this tests {14, 20} vs 15.
- **frieren H6v2 SGDR+clip**: On full combined stack now. SGDR validated before at -18.7% — should compound.
- **askeladd H8v3 EMA**: On full combined stack. EMA orthogonal to geom-cond and SwiGLU.
- **thorfinn H11b alpha sweep**: Verifies combined baseline. α=2.0 could help on tail of y distribution.
- **tanjiro H16 FiLM**: Multiplicative geom-cond. Gate magnitudes from H13 suggest model wants more geometry capacity.
- **fern H17 attention dropout**: Unregularized attention (dropout=0.0) on 1499 samples. 0.05 or 0.10 could help.
- **edward H18 LayerScale (to assign)**: Per-channel residual scaling (init=1e-6). Complements SwiGLU's gated FFN with gradual depth activation.
- **alphonse H1 LinearNO**: Diagnostic ablation — removes inter-slice QKV. Expected to regress but quantifies attention contribution.

## Open questions

- Does SGDR compose with log1p + geom-cond + SwiGLU? (frieren testing)
- Does EMA gain survive on the full stack? (askeladd testing)
- Is α=1.0 optimal for log1p? (thorfinn testing)
- Does FiLM beat additive geom-cond? (tanjiro testing)
- Does attention dropout help in addition to FFN dropout? (fern testing)
- Does LayerScale help with SwiGLU's high-capacity FFN? (edward to test)
- What is the true cosine T_max optimum now that T_max=15 is baked? (nezuko tested, pending review)

## Next directions if plateau hits

- **Architectural**: LayerScale (CaIT) [in queue for edward], DropPath (stochastic depth), n_layers=7, hierarchical attention
- **Loss**: physics-informed conservation terms, auxiliary lift/drag prediction
- **Data**: geometry-derivative features (curvature, normals), TTA
- **Optimization**: Lookahead, OneCycleLR, LR warmup before cosine
- **Conditioning**: per-block geom_proj (independent), FiLM on slice tokens instead of hidden state
