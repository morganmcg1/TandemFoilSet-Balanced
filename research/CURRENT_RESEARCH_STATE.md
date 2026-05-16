# SENPAI Research State

- **Date:** 2026-05-16 00:45
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 79.52, test_avg=68.95` (PR #3514 edward H18 LayerScale; **#3197 askeladd EMA 74.18 is winner pending rebase**)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| #3345 thorfinn H11 | signed-log1p target transform | -19.69 (-17.5%) | ~92.80 |
| #3224 tanjiro H13 | GALE-style geom-cond per block + T_max=15 | -7.64 (-8.2%) | 85.16 |
| #3423 edward H15 | SwiGLU gated FFN (replaces GELU) | -4.95 (-5.8%) | 80.21 |
| **#3514 edward H18** | **LayerScale residual scaling (CaIT init=1e-6)** | **-0.69 (-0.86%)** | **79.52** |
| *#3197 askeladd H8v3* | *EMA weights (decay=0.999) — PENDING MERGE* | *-5.34 (-6.7%)* | *74.18* |

## Per-split current best (H18 baseline)

| Split | val (H18) | val (EMA, pending) | test (H18) |
|---|---|---|---|
| `single_in_dist` | 104.62 | 98.18 | — |
| `geom_camber_rc` | 93.29 | 81.38 | — |
| `geom_camber_cruise` | 50.00 | 49.79 | — |
| `re_rand` | 70.14 | 67.37 | — |
| **avg** | **79.52** | **74.18** | **68.95** |

## Current active WIP PRs (8 students, all assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#3197** | **askeladd** | **H8v3 EMA (val_avg=74.18, WINNER 6.7% gain)** | **Sent back for rebase onto new H18 baseline; merge after fix** |
| #3421 | nezuko | H14v2 cosine T_max=14 + eta_min=1e-5 single-arm retest | WIP |
| #3467 | fern | H17 attention dropout sweep {0.05, 0.10} | WIP (picked up 00:21 UTC) |
| #3517 | frieren | H19 DropPath stochastic depth {0.10, 0.20} | WIP |
| #3538 | thorfinn | H22 LR warmup (2-epoch linear) + cosine eta_min=1e-5 | WIP |
| #3539 | alphonse | H23 slice_num sweep {32, 64, 128} | WIP |
| #3540 | tanjiro | H24 OneCycleLR super-convergence | WIP |
| **#3559** | **edward** | **H25 n_layers=6 deeper Transolver with LayerScale** | **WIP (just assigned)** |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep {0.05, 0.15, 0.20} | Closed — U-shape minimum at 0.10 |
| #3318 | frieren | H6v2 SGDR+grad-clip | Closed — SGDR can't fire 2nd restart in 14-epoch budget |
| #3417 | thorfinn | H11b log1p alpha sweep | Closed — α=1.0 confirmed optimal |
| #3184 | alphonse | H1 LinearNO ablation | Closed — +16% regression, attention essential |
| #3461 | tanjiro | H16 FiLM multiplicative geom-cond | Closed — camber_rc structural regression |

## Research insights so far

1. **EMA is a major win**: -6.7% val, -8.9% test on full combined stack (pending merge). Orthogonal to architecture.
2. **LayerScale confirms**: -0.86% val, -5.80% test. FFN gamma monotone with depth (textbook). Attention U-shaped. Cam_rc regression pattern (same as FiLM) — watch this.
3. **SwiGLU FFN**: -5.8% val. OOD gains 1.5-1.7× in-dist. Gate modulation structural.
4. **GALE geom-cond**: -8.2% val. Additive shift — NOT multiplicative (FiLM closed).
5. **Log-domain transform**: -17.5%, α=1.0 optimal (confirmed).
6. **FFN dropout=0.1**: Optimal, U-shape confirmed. Closed.
7. **Cam_rc regression pattern**: Both FiLM and LayerScale regress on geom_camber_rc (+3.18 and +5.42 respectively). This split requires OOD geometry interpolation — multiplicative/scale-based mechanisms seem to hurt it. Only additive mechanisms (GALE) helped it. Flag for future experiments.

## Open questions

- Does EMA compose with LayerScale? (Askeladd rebase will answer this post-merge)
- Does n_layers=6 help now that LayerScale enables identity init for new blocks? (edward H25)
- Does DropPath block-level regularization help on 1499 samples? (frieren H19)
- Does LR warmup reduce early noise without hurting peak? (thorfinn H22)
- What's optimal slice_num? (alphonse H23)
- Does OneCycleLR super-convergence beat cosine for 14-epoch budget? (tanjiro H24)
- Does attention dropout help? (fern H17)
- Does T_max=14 + eta_min=1e-5 beat T_max=15? (nezuko H14v2)
- **Cam_rc recovery**: How do we recover the geom_camber_rc split? It was gained by GALE (-12.7%) but partially lost by LayerScale (+5.4%). EMA pending shows rc=81.38 (partly recovered via weight averaging).

## Next directions

- **Architectural**: Hierarchical attention, per-block independent geom_proj
- **Loss**: Auxiliary lift/drag prediction head, physics-informed conservation terms
- **Optimizer**: Lookahead, Lion, weight decay sweep
- **Data**: Geometry-derivative features (curvature, normals), TTA
- **EMA refinements**: decay {0.9999, 0.999, 0.99}, EMA + BN/LN stats
- **Cam_rc focus**: The hardest-to-improve OOD split. Pure additive mechanisms help; scale-based mechanisms hurt. Needs dedicated investigation.
