# SENPAI Research State

- **Date:** 2026-05-16 01:45
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 79.52, test_avg=68.95` (PR #3514 edward H18 LayerScale)
- **Top pending winner:** `#3517 frieren H19 DropPath=0.20: 73.55 val / 67.05 test` — **SENT BACK FOR REBASE + RETEST on H18 baseline**
- **Second pending winner:** `#3197 askeladd H8v3 EMA=0.999: 74.18 val` — **rebased, WIP retest in progress**

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

## Per-split current best (H18 LayerScale baseline)

| Split | val (H18) | val (DropPath 0.20, pending rebase) | test (H18) |
|---|---|---|---|
| `single_in_dist` | 104.62 | 91.50 | — |
| `geom_camber_rc` | 93.29 | 84.83 | — |
| `geom_camber_cruise` | 50.00 | 49.93 | — |
| `re_rand` | 70.14 | 67.92 | — |
| **avg** | **79.52** | **73.55** | **68.95** |

## Current active WIP PRs (8 students, all assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#3197** | **askeladd** | **H8v3 EMA decay=0.999 — val=74.18 on H15 baseline, now rebased onto H18** | **WIP retest** |
| **#3517** | **frieren** | **H19 DropPath=0.20 — val=73.55 on H15 baseline, SENT BACK for H18 rebase + retest** | **WIP (rebase pending)** |
| #3421 | nezuko | H14v2 cosine T_max=14 + eta_min=1e-5 single-arm retest | WIP |
| **#3583** | **fern** | **H26 AdamW weight_decay sweep {0.001, 0.01, 0.05}** | **WIP (just assigned)** |
| #3538 | thorfinn | H22 LR warmup (2-epoch linear) + cosine eta_min=1e-5 | WIP |
| #3539 | alphonse | H23 slice_num sweep {32, 64, 128} | WIP |
| #3540 | tanjiro | H24 OneCycleLR super-convergence | WIP |
| #3559 | edward | H25 n_layers=6 deeper Transolver with LayerScale | WIP |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep | Closed — U-shape, 0.10 confirmed optimal |
| #3318 | frieren | H6v2 SGDR+grad-clip | Closed — can't fire 2nd restart in 14 epochs |
| #3417 | thorfinn | H11b log1p alpha sweep | Closed — α=1.0 confirmed optimal |
| #3184 | alphonse | H1 LinearNO ablation | Closed — +16% regression, attention essential |
| #3461 | tanjiro | H16 FiLM multiplicative geom-cond | Closed — camber_rc structural regression |
| #3467 | fern | H17 attention dropout {0.05, 0.10} | Closed — both arms regress; softmax routing low-entropy |

## Research insights so far

1. **Regularization is the dominant lever at 1499 samples**: Every major win this round is some form of regularization — FFN dropout (-8.4%), DropPath (-8.3%), EMA (-6.7%). Architecture wins (SwiGLU, LayerScale) are smaller but compound.
2. **DropPath=0.20 is the best raw result** (73.55 val, 67.05 test, pending H18 rebase confirmation). At 5 layers, 0.20 beats 0.10 because the regularizer needs more force at this depth.
3. **EMA** (74.18 val) and **DropPath** (73.55 val) are orthogonal: stochastic depth operates at forward-pass training time; EMA operates on weight averaging at inference. Should compose.
4. **Multiplicative/scale mechanisms hurt cam_rc**: FiLM (+structural regression) and LayerScale (+5.4% cam_rc, partly offset by test gains) — both harmed the hardest OOD split. Only additive (GALE) helped it.
5. **Attention-weight noise doesn't help**: H17 attn dropout tried; failed. Residual-level stochastic depth (H19 DropPath) works. The mechanism matters: routing softmax is already low-entropy.
6. **Weight decay 1e-4 is suspect**: Never tuned for this dataset. AdamW typically uses 0.01–0.1. Next probe: H26 sweep.

## Open questions

- Does DropPath compose with LayerScale? (frieren H19 rebase will answer — both are standard CaIT/DeiT companions targeting different axes)
- Does EMA compose with DropPath+LayerScale? (askeladd H8v3 rebase/retest in progress)
- Optimal weight_decay? (fern H26, 3-arm sweep)
- Does LR warmup reduce early noise without hurting peak? (thorfinn H22)
- What's optimal slice_num? (alphonse H23)
- Does OneCycleLR super-convergence beat cosine for 14-epoch budget? (tanjiro H24)
- Does T_max=14 + eta_min=1e-5 beat T_max=15? (nezuko H14v2)
- Does n_layers=6 help with LayerScale identity-init of new blocks? (edward H25)

## Next directions (after current wave resolves)

- **Cam_rc recovery**: Still the hardest split. Pure additive mechanisms help; scale-based mechanisms hurt. Needs dedicated investigation — possibly per-split loss weighting.
- **DropPath refinement**: Sweep 0.30–0.40 if 0.20 is confirmed optimal (frieren suggested it, trajectory still descending at timeout)
- **Auxiliary loss**: Lift/drag prediction head — physics-informed auxiliary signal
- **Optimizer**: Lion optimizer, Lookahead wrapper
- **Data**: Geometry-derivative features (curvature, normals), TTA
- **Architecture**: Per-block independent geom_proj, hierarchical attention
- **EMA refinements**: decay {0.9999, 0.999, 0.99} sweep once H8v3 merges
