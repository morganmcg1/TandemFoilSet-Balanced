# SENPAI Research State

- **Date:** 2026-05-15 23:05
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

## Current active WIP PRs (8 students, all assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3417 | thorfinn | H11b log1p alpha sweep {0.5, 1.0, 2.0} | WIP (training done, awaiting commit post rate-limit) |
| #3421 | nezuko | H14v2 cosine T_max=14 + eta_min=1e-5 single-arm retest | WIP (sent back for v2 retest on current code) |
| #3197 | askeladd | H8v3 EMA on combined baseline | WIP (training done, awaiting commit post rate-limit) |
| #3461 | tanjiro | H16 FiLM geom-cond (γ + β, extends H13 GALE) | WIP (training started, post-rate-limit) |
| #3467 | fern | H17 attention dropout sweep {0.05, 0.10} | WIP (training started, post-rate-limit) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (training done, awaiting commit post rate-limit) |
| **#3514** | **edward** | **H18 LayerScale residual scaling (CaIT init=1e-6)** | **WIP (just assigned)** |
| **#3517** | **frieren** | **H19 DropPath stochastic depth {0.10, 0.20}** | **WIP (just assigned)** |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep {0.05, 0.15, 0.20} | Closed — U-shape minimum at 0.10 |
| **#3318** | **frieren** | **H6v2 SGDR+grad-clip** | **Closed — +7.5% regression; SGDR can't fire 2nd restart in 14-epoch budget** |

## Research insights so far

1. **SwiGLU FFN is a structural win**: -5.8% from replacing GELU with gated multiplicative FFN. OOD gains 1.5–1.7× larger than in-dist — gate modulation reduces co-adaptation like dropout but structurally. Best epoch at 10 (faster convergence).
2. **GALE-style geometry conditioning works**: -8.2% from additive geom-cond with zero-init gates. Camber_rc split benefited most (-12.7%). T_max=15 alignment critical.
3. **Log-domain target transform is the biggest lever**: -17.5% from signed-log1p. Compresses 13× y-std range.
4. **FFN dropout confirmed optimal at 0.10**: U-shape minimum, all alternatives regress. Narrow basin.
5. **Spectral bias matters**: RFF -3.9%.
6. **Re-stratified sampling works**: upweighting high-Re samples improved OOD-Re splits.
7. **SGDR does NOT compose with the current stack**: Only one LR restart fires in 14 realized epochs; second restart is structurally unreachable. Log1p + dropout absorb grad-clip's stabilization benefit. The SGDR direction is closed.
8. **EMA validated earlier**: +7.4% gain over live (pre-GALE stack) — retest on current stack pending (askeladd).
9. **Schedule alignment matters**: T_max=15 > T_max=50 for 14 realized epochs. T_max=14 + eta_min=1e-5 may give marginal further gain (nezuko retesting).
10. **Seed variance is real with SwiGLU**: H15 showed 10% val_avg spread between 2 seeds. Gating amplifies lucky init states.

## High-value in-flight bets

- **edward H18 LayerScale**: Per-channel residual scaling (init=1e-6). Synergistic with SwiGLU — gradually activates depth capacity. Diagnostic: deeper blocks should develop larger gamma norms.
- **frieren H19 DropPath**: Stochastic depth {0.10, 0.20}. Block-level regularization complementary to FFN dropout. Hypothesis: SwiGLU made each block more expressive; DropPath prevents block co-adaptation.
- **fern H17 attn-dropout**: Unregularized attention on 1499 samples. 0.05 or 0.10 could help.
- **tanjiro H16 FiLM**: Multiplicative geom-cond extends H13 GALE. Gate magnitudes from H13 suggest model wants more geometry capacity.
- **askeladd H8v3 EMA**: On full combined stack. EMA orthogonal to geom-cond and SwiGLU.
- **thorfinn H11b alpha sweep**: Verifies combined baseline. α=2.0 could help on tail of y distribution.
- **nezuko H14v2**: Single-arm T_max=14 + eta_min=1e-5 on current code. Marginal gain expected; close if within seed variance.
- **alphonse H1 LinearNO**: Diagnostic ablation — removes inter-slice QKV. Expected to regress but quantifies attention contribution.

## Open questions

- Does LayerScale's gradual depth activation compose with SwiGLU gating? (edward H18 testing)
- Does DropPath's block-level regularization help on 1499 samples? (frieren H19 testing)
- Does EMA gain survive on the full stack? (askeladd H8v3 testing)
- Is α=1.0 optimal for log1p? (thorfinn H11b testing)
- Does FiLM beat additive geom-cond? (tanjiro H16 testing)
- Does attention dropout help in addition to FFN dropout? (fern H17 testing)
- Does T_max=14 + eta_min=1e-5 give marginal gain over T_max=15? (nezuko H14v2 testing)

## Next directions if plateau hits

- **Architectural**: n_layers=7 (deeper), hierarchical attention, slice_num sweep (32 vs 64 vs 128)
- **Loss**: physics-informed conservation terms, auxiliary lift/drag prediction head
- **Optimizer**: OneCycleLR superconvergence (Smith 2017), LR warmup before cosine
- **Data**: geometry-derivative features (curvature, normals), TTA, data augmentation (geometry perturbations)
- **Conditioning**: per-block independent geom_proj (vs shared), FiLM on slice tokens instead of hidden state
- **Learning rate sweep**: {1e-4, 5e-4 (current), 1e-3} — not yet ablated
- **Preprocess MLP**: deeper (3 layers) or wider (256 hidden) to expand feature extraction capacity
