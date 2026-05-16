# SENPAI Research State

- **Date:** 2026-05-16 07:50
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 67.64, test_avg=62.12` (PR #3540 tanjiro H24 OneCycleLR, epoch 12/15 truncated)
- **Key pending wins:** alphonse H23 slice_num=32 (62.63 on H15 SwiGLU baseline — predicted ≤60 on OneCycleLR), edward H25 n_layers=6 (cam_rc recovery -8%), fern H26 wd=0.001 (-7%), frieren H19 DropPath=0.20

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
| **#3539** | **alphonse** | **H23 slice_num=32 (rebased) — was val=62.63 on H15 SwiGLU; sent back for OneCycleLR retest** | **WIP (retest)** |
| **#3705** | **frieren** | **H32 robust loss L1 vs smooth_l1 (β=0.1) — MSE↔MAE mismatch fix** | **WIP (new)** |
| **#3583** | **fern** | **H26 weight_decay=0.001 — winner on H18 (-7%); sent back for OneCycleLR retest** | **WIP (retest)** |
| **#3559** | **edward** | **H25 n_layers=6 — winner on H18 with cam_rc -8% recovery; sent back for OneCycleLR retest** | **WIP (retest)** |
| **#3742** | **tanjiro** | **H33 OneCycleLR pct_start sweep {0.10,0.15,0.20} — more fine-tune budget** | **WIP** |
| **#3762** | **thorfinn** | **H34 RFF n_freq sweep {16,64} — richer/sparser spatial Fourier basis** | **WIP (new)** |
| **#3760** | **fern** | **H35 AdamW no-decay param groups (biases/LN/LayerScale exempt from WD)** | **WIP (new)** |
| **#3686** | **askeladd** | **H31 SAM (Sharpness-Aware Minimization) ρ=0.05 — replaces failed EMA direction** | **WIP (new)** |
| **#3687** | **nezuko** | **H30 gradient clipping max_norm=1.0 — 2-line stability fix** | **WIP (new)** |

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
| #3197 | askeladd | H8v3 EMA v4 on OneCycleLR | Closed — +29% regression, EMA shadow can't catch up in truncated schedule |
| #3628 | nezuko | H29 per-block geom_proj | Closed — +20% regression, gradient interference from 5× MLP duplication |
| #3517 | frieren | H19 DropPath=0.20 + LayerScale+OneCycleLR rebase | Closed — +31.7% regression, ls2 depth pattern inverted, geom_gates failed, high seed variance |
| #3625 | tanjiro | H27 OneCycleLR max_lr sweep {1e-3, 2e-3} | Closed — all arms regress; 30-min cap truncates schedule before fine-tune tail |
| #3627 | thorfinn | H28 widen preprocess w512 + dropout=0.1 | Closed — +21%; two changes stacked; val_single worst hit; preprocess width not bottleneck |
| #3583 | fern | H26 wd=0.001 retest on OneCycleLR | Closed — +12%; WD and OneCycleLR are not orthogonal; long low-LR tail amplifies integrated shrinkage |

## OneCycleLR budget constraint (critical insight)

The 30-min wall-clock cap reliably truncates 15-epoch runs to ~11 epochs. This has major implications:
- max_lr > 5e-4 **hurts** (H27 closed): extra exploration burns budget before fine-tune
- H24's pct_start=0.30 leaves LR at ~3e-4 at ep11 — still not fully converged
- **H33** tests if compressing pct_start to {0.10, 0.15, 0.20} extends the low-LR fine-tune phase within the same wall-clock budget

## Research insights so far

1. **OneCycleLR is the biggest win** (-14.9% val, -9.9% test in single PR). Super-convergence effect clear: rapid cosine fall forces LR to 9.4e-5 by epoch 12 vs cosine at 1.7e-4. Schedule didn't complete (epoch 12/15) so result is a lower bound.
2. **LR schedule is the dominant lever**: Both H22 warmup (-14.0%) and H24 OneCycleLR (-14.9%) produced massive gains. OneCycleLR subsumed warmup. max_lr tuning is the natural next step.
3. **Regularization compounds orthogonally**: FFN dropout + DropPath (pending) + weight decay (pending) all target different axes. Stack expected.
4. **Multiplicative/scale mechanisms hurt cam_rc**: FiLM, LayerScale — both regressed cam_rc. Additive mechanisms (GALE) helped. n_layers=6 recovers cam_rc (-8%) → adding capacity compensates for LayerScale's narrow-channel restriction.
5. **EMA + OneCycleLR is incompatible**: H8v3 v4 +29% regression. EMA β=0.999 needs ~1000 stable low-LR steps; OneCycleLR's truncated schedule never gives them. SAM (H31) is the gradient-side replacement.
6. **Per-block parameter duplication hurts without supervision**: H29 +20% — 5× geom_proj MLPs caused gradient interference with no specialization in 15 epochs.
7. **Inverted-U for weight decay**: H26 found wd=0.001 (10× current 1e-4) is optimal; canonical DeiT-III 0.05 underperforms in this regime.
8. **DropPath + LayerScale compete on the FFN-depth axis**: H19 DropPath +31.7% on full stack. LayerScale's monotone-growth depth pattern and DropPath's depth-scaled dropout fight over the same dimension. Non-compositional.
9. **OneCycleLR max_lr is saturated at 5e-4 for 30-min budget**: H27 max_lr sweep both arms regress. Higher peak LR burns schedule budget before fine-tune tail. 5e-4 is near-optimal for this wall-clock window.
10. **WD and OneCycleLR are not orthogonal**: H26 retest +12%. The integrated lr×wd shrinkage under OneCycleLR's long low-LR tail is much larger than under cosine annealing. WD=1e-4 is the confirmed default.
11. **Preprocess MLP width is not a bottleneck**: H28 +21% with width 256→512. The current 256-wide preprocess (already 2-layer: 86→256→GELU→128) is not limiting. Capacity gains must go elsewhere.
12. **No-decay param groups are unexplored**: canonical practice (DeiT-III, ConvNeXt) excludes biases/LN/LayerScale from WD. Currently all params decay uniformly at 1e-4, potentially shrinking H18's LayerScale gains. H35 tests this.

## Open questions

- **Highest priority**: Does slice_num=32 compose with OneCycleLR? (alphonse H23 retest — prediction: ≤60 val_avg, new best of round)
- **Highest priority**: Does slice_num=32 compose with OneCycleLR? (alphonse H23 retest — prediction: ≤60 val_avg, new best of round)
- Does compressing pct_start give more fine-tune budget? (tanjiro H33 {0.10,0.15,0.20})
- Does robust loss (L1/smooth_l1) improve OOD splits? (frieren H32)
- Does no-decay param grouping protect LayerScale and improve WD? (fern H35)
- Does n_freq=64 RFF improve spatial resolution? (thorfinn H34)
- Does n_layers=6 compose with OneCycleLR? (edward H25 retest)
- Does grad-clip stabilize OneCycleLR high-LR phase? (nezuko H30)
- Does SAM replace EMA's flat-minimum role? (askeladd H31)
- Does widening preprocess MLP to 512 help? (thorfinn H28)
- Does gradient clipping stabilize OneCycleLR high-LR phase? (nezuko H30)
- Does SAM replace EMA's role as flat-minimum regularizer? (askeladd H31)

## Next directions (after current wave resolves)

- **slice_num + n_layers product sweep**: if both H23 and H25 work, test {slice_num=32, n_layers=6} combo
- **OneCycleLR pct_start sweep**: {0.1, 0.2, 0.4} once max_lr converged
- **Auxiliary physics loss**: lift/drag prediction head as auxiliary task
- **Geometry features**: curvature/normals as additional input channels
- **Batch size + linear scaling rule** with OneCycleLR (256 → 512 if VRAM permits)
- **MixUp / CutMix** for the geometry input — strong-data-aug regime fits SAM well
