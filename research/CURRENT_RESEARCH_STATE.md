# SENPAI Research State

- **Date:** 2026-05-16 02:20
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 67.64, test_avg=62.12` (PR #3540 tanjiro H24 OneCycleLR, epoch 12/15 truncated)
- **Key pending wins:** askeladd EMA v4 (retesting on OneCycleLR baseline), frieren DropPath rebase (was 73.55 on SwiGLU — if it composes with OneCycleLR could go much lower)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| #3345 thorfinn H11 | signed-log1p target transform | -19.69 (-17.5%) | ~92.80 |
| #3224 tanjiro H13 | GALE-style geom-cond per block + T_max=15 | -7.64 (-8.2%) | 85.16 |
| #3423 edward H15 | SwiGLU gated FFN (replaces GELU) | -4.95 (-5.8%) | 80.21 |
| #3514 edward H18 | LayerScale residual scaling (CaIT init=1e-6) | -0.69 (-0.86%) | 79.52 |
| **#3540 tanjiro H24** | **OneCycleLR super-convergence** | **-11.88 (-14.9%)** | **67.64** |

## Per-split current best (H24 OneCycleLR baseline)

| Split | val | test |
|---|---|---|
| `single_in_dist` | 80.32 | — |
| `geom_camber_rc` | 81.81 | — |
| `geom_camber_cruise` | 44.46 | — |
| `re_rand` | 63.96 | — |
| **avg** | **67.64** | **62.12** |

## Current active WIP PRs (8 students, all assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#3197** | **askeladd** | **H8v3 EMA decay=0.999 — v3 val=74.18 (stale), sent back for v4 retest with OneCycleLR** | **WIP (retest)** |
| **#3517** | **frieren** | **H19 DropPath=0.20 — val=73.55 on SwiGLU, sent back for rebase onto OneCycleLR+LayerScale** | **WIP (rebase)** |
| #3583 | fern | H26 AdamW weight_decay sweep {0.001, 0.01, 0.05} | WIP |
| **#3625** | **tanjiro** | **H27 OneCycleLR max_lr sweep {1e-3, 2e-3}** | **WIP (just assigned)** |
| **#3627** | **thorfinn** | **H28 Deeper preprocess MLP (86→256→128)** | **WIP (just assigned)** |
| **#3628** | **nezuko** | **H29 Per-block independent geom_proj (cam_rc recovery)** | **WIP (just assigned)** |
| #3539 | alphonse | H23 slice_num sweep {32, 64, 128} | WIP |
| #3559 | edward | H25 n_layers=6 deeper Transolver with LayerScale | WIP |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep | Closed — U-shape, 0.10 confirmed optimal |
| #3318 | frieren | H6v2 SGDR+grad-clip | Closed — can't fire 2nd restart in 14 epochs |
| #3417 | thorfinn | H11b log1p alpha sweep | Closed — α=1.0 confirmed optimal |
| #3184 | alphonse | H1 LinearNO ablation | Closed — +16% regression, attention essential |
| #3461 | tanjiro | H16 FiLM multiplicative geom-cond | Closed — camber_rc structural regression |
| #3467 | fern | H17 attention dropout {0.05, 0.10} | Closed — both arms regress; softmax routing low-entropy |
| #3538 | thorfinn | H22 LR warmup + cosine | Closed — redundant, subsumed by OneCycleLR (#3540) |
| #3421 | nezuko | H14 cosine T_max sweep {14, 20} | Closed — obsolete after OneCycleLR merged |

## Research insights so far

1. **OneCycleLR is the biggest win** (-14.9% val, -9.9% test in single PR). Super-convergence effect clear: rapid cosine fall forces LR to 9.4e-5 by epoch 12 vs cosine at 1.7e-4. Schedule didn't complete (epoch 12/15) so result is a lower bound.
2. **LR schedule is the dominant lever**: Both H22 warmup (-14.0%) and H24 OneCycleLR (-14.9%) produced massive gains. OneCycleLR subsumed warmup. max_lr tuning is the natural next step.
3. **Regularization compounds orthogonally**: FFN dropout + DropPath (pending) + EMA (pending) + weight decay (pending) all target different axes. Stack expected.
4. **Multiplicative/scale mechanisms hurt cam_rc**: FiLM, LayerScale — both regressed cam_rc. Additive mechanisms (GALE) helped. Per-block geom_proj is additive → may help.
5. **Attention-weight noise doesn't help**: H17 attn dropout failed. Residual-level (DropPath H19) worked on val_single (-23.1%) and val_rc (-7.6%).

## Open questions

- Does EMA compose with OneCycleLR? (askeladd H8v3 v4 retest — prediction: stronger gain, EMA most powerful at low-LR tail)
- Does DropPath compose with LayerScale + OneCycleLR? (frieren H19 rebase)
- Optimal max_lr for OneCycleLR? (tanjiro H27 {1e-3, 2e-3})
- Does deeper preprocess MLP help with heterogeneous input? (thorfinn H28)
- Per-block geom_proj for cam_rc recovery? (nezuko H29)
- Optimal weight_decay? (fern H26)
- Optimal slice_num? (alphonse H23)
- Does n_layers=6 help with LayerScale+OneCycleLR? (edward H25)

## Next directions (after current wave resolves)

- **DropPath extend**: sweep 0.30-0.40 if 0.20 confirmed (frieren suggested)
- **EMA decay sweep**: {0.9999, 0.999, 0.99} once EMA v4 confirms
- **Auxiliary physics loss**: lift/drag prediction head
- **Geometry features**: curvature/normals
- **Batch size + linear scaling rule** with OneCycleLR
- **SAM (Sharpness-Aware Minimization)**: orthogonal to schedule, no new packages needed (inline impl)
