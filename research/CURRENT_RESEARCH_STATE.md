# SENPAI Research State

- **Date:** 2026-05-16 12:24
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 67.64, test_avg=62.12` (PR #3540 tanjiro H24 OneCycleLR, epoch 12/15 truncated)
- **Key pending wins:** thorfinn H41 domain-type indicator (predicted -3 to -8, highest researcher priority, attacks val_single=80.32), nezuko H42 aux Re/AoA head (attacks val_re=63.96), alphonse H43 flow-condition MixUp (first data-aug), frieren H45 pressure head split (channel-imbalance architectural fix), edward H40 SWA tail-averaging

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
| **#3929** | **alphonse** | **H43 Flow-condition MixUp (dims 13-23, α={0.4, 0.2}) — first data-augmentation experiment** | **WIP (new, replaces H23 retest)** |
| **#3928** | **frieren** | **H45 Pressure-specific output head (vel_head + p_head SwiGLU split) — channel-imbalance architectural fix** | **WIP (new, replaces H36)** |
| **#3824** | **fern** | **H38 input Gaussian noise injection sweep {0.01, 0.03, 0.10} — Bishop 1995 Tikhonov-equiv** | **WIP** |
| **#3852** | **edward** | **H40 SWA tail-averaging K={3,5} — pseudo-ensemble from converged tail epochs** | **WIP** |
| **#3930** | **tanjiro** | **H44 Foil-relative coords (log1p centroid distance) — boundary-layer physics prior** | **WIP (new, replaces H33)** |
| **#3948** | **thorfinn** | **H47 Focal regression loss (gamma sweep {1.0, 2.0}) — per-sample hard-example weighting** | **WIP (new, replaces H41)** |
| **#3931** | **askeladd** | **H46 Gradient accumulation 2x (effective batch 8) — variance-reducing optimizer steps** | **WIP (new, replaces H39)** |
| **#3900** | **nezuko** | **H42 aux Re/AoA prediction head (aux_weight={0.1, 0.05}) — val_re targeting** | **WIP** |

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
| #3705 | frieren | H32 robust loss L1/smooth_l1 β=0.1 | Closed — L1 +7.7%, smooth_l1 +11.9%. signed_log1p already neutralizes heavy tail; MSE confirmed optimal for this dataset |
| #3559 | edward | H25 n_layers=6 OneCycleLR retest | Closed — +26.5% (2 seeds, spread 4.1). Depth + OneCycleLR non-orthogonal: 199s/epoch model gets only 10 epochs, never reaches fine-tune tail (LR=2.3e-4 at termination vs 9.4e-5 for baseline) |
| #3760 | fern | H35 AdamW no-decay param groups | Closed — +13.3%. Mechanism empirically absent: ls2 norms flat at ~0.42 under no_decay vs baseline's monotone 0.43→0.51. Integrated WD shrinkage on LS is ~1e-4 over 4125 steps — numerically negligible at wd=1e-4 |
| #3686 | askeladd | H31 SAM ρ=0.05 OneCycleLR | Closed — 84-95% worse baseline. 5.3 min/epoch (2.1×) → only 5-6 epochs fit in 30-min. Budget collision, not SAM failure. Best contingency val=124.68 (ep 6/10 truncated). AMP (H39) assigned as structural fix. |
| #3792 | edward | H37 OneCycleLR epochs=12 | Closed — +14.3% (val=77.30). Lost ~20% gradient steps. Per-epoch drift 130s→166s (+27%) meant realized 10.8 epochs, schedule still truncated. Schedule truncation cannot be fixed by epoch-count alone under variable wall-time. |
| #3762 | thorfinn | H34 RFF n_freq sweep {16,64} | Closed — n_freq=16: +13.1% (val=76.52), n_freq=64: +21.2% (val=81.97). Both regress uniformly. n_freq=32 is empirically optimal. Asymmetry (doubling hurts more than halving) — larger input dilutes per-input-dim weights at fixed n_hidden=256 bottleneck. |
| #3687 | nezuko | H30 gradient clipping max_norm=1.0 | Closed — mean 4-seed val=71.93 (+6.3%). Natural gradient norm 4-30 with surf_weight=10; max_norm=1.0 fires 100% of batches (not spike-filtering but constant damping). OneCycleLR + log1p is already stable — no instability to clip. |
| #3539 | alphonse | H23 slice_num=32 rebased retest (4 seeds) | Closed — mean val=72.22 (+6.8%), best 71.16. slice_num=32 and OneCycleLR are anti-synergistic — both attack the same overfitting axis. Original SwiGLU win doesn't compose with OneCycleLR. test_avg=62.80 within noise (+1.1%). |
| #3742 | tanjiro | H33 OneCycleLR pct_start sweep {0.10,0.15,0.20} | Closed — best val=77.77 (+15%). All arms regress; best_epoch stays at 10-11 across all. Compression shifts LR to lower values sooner without moving the optimum. val_single hit hardest (+17 to +21) — high-LR exploration matters for in-dist generalization. Schedule axis is now exhausted. |
| #3788 | frieren | H36 channel-weighted surf_loss surf_p_weight sweep {2,3,5} | Closed — best val=76.73 (+13.4%, w=2). All arms regress monotonically. Ux/Uy supervision on surface nodes is auxiliary supervision regularizing the shared backbone — cutting velocity weight corrupts the multi-task representation. val_single (most p-dominated) hit hardest. Rules out output-loss reweighting axis; pivots to architectural head-split (H45 frieren). |
| #3850 | askeladd | H39 BF16 AMP mixed precision | Closed — val=69.69 (+3.0% on primary), test=60.92 (-1.9% improved). Schedule completion mechanism worked (LR=2.1e-9, best_epoch=15). val_single regression (+10.3%) dominates val story (same structural single-foil bottleneck thorfinn H41 targets). Speedup only 5-25% — attention dominates, not matmul. AMP stays available for future depth/capacity experiments. |
| #3867 | thorfinn | H41 domain-type indicator (is_tandem embedding) | Closed — val=78.13 (+15.5%), val_single=96.39 (+20.0% on target split, opposite direction). Embedding norms grew from 0 to ~0.10 (mechanism active) but model didn't extract useful signal. dims 18-23 are EXACTLY zero for single-foil — model trivially learns the AND-of-zero via single hidden unit. val_single is NOT a routing/disambiguation problem — must be rarity, variance, or shape distribution. |

## OneCycleLR budget constraint (critical insight)

The 30-min wall-clock cap reliably truncates 15-epoch runs to ~11-13 epochs depending on per-epoch cost. This has major implications:
- max_lr > 5e-4 **hurts** (H27 closed): extra exploration burns budget before fine-tune
- H24's pct_start=0.30 gives best_epoch=12 at LR=9.4e-5 — schedule end (LR=0) never reached
- **H33** tests if compressing pct_start to {0.10, 0.15, 0.20} extends the low-LR fine-tune phase within the same wall-clock budget
- **H37** tests if compressing total epochs (epochs=12) so schedule actually completes helps
- **n_layers=6 is structurally incompatible** with 30-min budget: 199s/epoch → only 10 epochs, never reaches fine-tune tail

## Research insights so far

1. **OneCycleLR is the biggest win** (-14.9% val, -9.9% test in single PR). Super-convergence effect clear: rapid cosine fall forces LR to 9.4e-5 by epoch 12 vs cosine at 1.7e-4. Schedule didn't complete (epoch 12/15) so result is a lower bound.
2. **LR schedule is the dominant lever**: Both H22 warmup (-14.0%) and H24 OneCycleLR (-14.9%) produced massive gains. OneCycleLR subsumed warmup. max_lr tuning is the natural next step.
3. **Regularization compounds orthogonally**: FFN dropout + DropPath (pending) + weight decay (pending) all target different axes. Stack expected.
4. **Multiplicative/scale mechanisms hurt cam_rc**: FiLM, LayerScale — both regressed cam_rc. Additive mechanisms (GALE) helped. n_layers=6 recovers cam_rc (-8%) → adding capacity compensates for LayerScale's narrow-channel restriction.
5. **EMA + OneCycleLR is incompatible**: H8v3 v4 +29% regression. EMA β=0.999 needs ~1000 stable low-LR steps; OneCycleLR's truncated schedule never gives them. SAM (H31) is the gradient-side replacement.
6. **Per-block parameter duplication hurts without supervision**: H29 +20% — 5× geom_proj MLPs caused gradient interference with no specialization in 15 epochs.
7. **Inverted-U for weight decay**: H26 found wd=0.001 (10× current 1e-4) is optimal on cosine; but WD and OneCycleLR are non-orthogonal. WD=1e-4 is confirmed default on OneCycleLR.
8. **DropPath + LayerScale compete on the FFN-depth axis**: H19 DropPath +31.7% on full stack. LayerScale's monotone-growth depth pattern and DropPath's depth-scaled dropout fight over the same dimension. Non-compositional.
9. **OneCycleLR max_lr is saturated at 5e-4 for 30-min budget**: H27 max_lr sweep both arms regress. Higher peak LR burns schedule budget before fine-tune tail. 5e-4 is near-optimal for this wall-clock window.
10. **WD and OneCycleLR are not orthogonal**: H26 retest +12%. The integrated lr×wd shrinkage under OneCycleLR's long low-LR tail is much larger than under cosine annealing. WD=1e-4 is the confirmed default.
11. **Preprocess MLP width is not a bottleneck**: H28 +21% with width 256→512. The current 256-wide preprocess (already 2-layer: 86→256→GELU→128) is not limiting. Capacity gains must go elsewhere.
12. **No-decay param groups are unexplored**: canonical practice (DeiT-III, ConvNeXt) excludes biases/LN/LayerScale from WD. Currently all params decay uniformly at 1e-4, potentially shrinking H18's LayerScale gains. H35 tests this.
13. **MSE in signed_log1p space is optimal for this dataset**: H32 closed. signed_log1p already neutralizes the heavy Re tail (residuals are O(0.1-1), not blowing up). L1's robustness buys nothing. Smooth_L1 β=0.1 was 5× MSE curvature in the wrong regime.
14. **Depth (n_layers=6) + OneCycleLR is structurally non-orthogonal under 30-min cap**: H25 retest closed +26.5% (2 seeds). 199s/epoch model never reaches fine-tune tail. The architecture is sound (H18 baseline showed -2.7%); the budget interaction is the barrier.
15. **Primary metric is p-channel only but loss is channel-uniform**: val_avg/mae_surf_p is pure pressure. `surf_loss` averages Ux, Uy, p equally → p gets 1/3 of gradient signal. H36 tests upweighting p directly.
16. **Schedule truncation is a first-class bug**: H24 baseline reaches LR=9.4e-5 at best_epoch=12 but schedule ends at LR=0 (epoch 15, never reached). H33 (pct_start) and H37 (epochs=12) both attack this truncation from different angles.
17. **No-decay carve-out needs higher WD to matter**: H35 closed +13.3%. ls2 norm comparison showed flat ~0.42 under no_decay vs baseline's monotone 0.43→0.51 — deep-block LS actually grew LESS, refuting the "uniform WD silently shrinks LayerScale" premise. Integrated shrinkage at wd=1e-4 is ~1e-4 relative — numerically negligible. The canonical DeiT-III recipe needs wd=0.05 to shine, which H26 closure rules out.
18. **Input-side regularization is unexplored**: We've tested FFN dropout (H12 merged), attention dropout (H17 closed), weight decay (H26 closed), no-decay groups (H35 closed). Input-Jacobian regularization (Bishop 1995 equivalence) is a third axis — H38 tests this.

## Open questions

- Does flow-condition MixUp expand effective dataset diversity and improve OOD splits? (alphonse H43 — prediction: -2% to -6%, concentrated on val_rc and val_re)
- Does splitting p output head from velocity head fix channel imbalance architecturally? (frieren H45 — prediction: -2% to -5%; orthogonal to closed H36 loss approach)
- Does foil-relative log-distance feature add useful boundary-layer prior? (tanjiro H44 — prediction: -1% to -4%; first input-feature engineering since H5 RFF)
- Does gradient accumulation (effective batch 8) reduce gradient variance and improve stability? (askeladd H46 — prediction: -1% to -4%)
- Does SWA tail averaging K={3,5} give pseudo-ensemble benefit? (edward H40 — prediction: -0.5% to -3%)
- Does input Gaussian noise regularize the input Jacobian? (fern H38 — prediction: -0.5% to -2%)
- Does focal regression loss (γ-weighted hard-sample mining) close val_single and val_rc gaps? (thorfinn H47 — prediction: -1% to -4%; new direction after H41 routing-thesis refutation)
- Does aux Re/AoA head force Re-aligned encoder representations and close val_re gap? (nezuko H42 — prediction: -2% to -6%, concentrated on val_re)

## Research insights so far (27 total)

1. **OneCycleLR is the biggest win** (-14.9% val, -9.9% test in single PR). Super-convergence effect clear: rapid cosine fall forces LR to 9.4e-5 by epoch 12 vs cosine at 1.7e-4. Schedule didn't complete (epoch 12/15) so result is a lower bound.
2. **LR schedule is the dominant lever**: Both H22 warmup (-14.0%) and H24 OneCycleLR (-14.9%) produced massive gains. OneCycleLR subsumed warmup. max_lr tuning is the natural next step.
3. **Regularization compounds orthogonally**: FFN dropout + DropPath (pending) + weight decay (pending) all target different axes. Stack expected.
4. **Multiplicative/scale mechanisms hurt cam_rc**: FiLM, LayerScale — both regressed cam_rc. Additive mechanisms (GALE) helped. n_layers=6 recovers cam_rc (-8%) → adding capacity compensates for LayerScale's narrow-channel restriction.
5. **EMA + OneCycleLR is incompatible**: H8v3 v4 +29% regression. EMA β=0.999 needs ~1000 stable low-LR steps; OneCycleLR's truncated schedule never gives them. SAM (H31) is the gradient-side replacement.
6. **Per-block parameter duplication hurts without supervision**: H29 +20% — 5× geom_proj MLPs caused gradient interference with no specialization in 15 epochs.
7. **Inverted-U for weight decay**: H26 found wd=0.001 (10× current 1e-4) is optimal on cosine; but WD and OneCycleLR are non-orthogonal. WD=1e-4 is confirmed default on OneCycleLR.
8. **DropPath + LayerScale compete on the FFN-depth axis**: H19 DropPath +31.7% on full stack. LayerScale's monotone-growth depth pattern and DropPath's depth-scaled dropout fight over the same dimension. Non-compositional.
9. **OneCycleLR max_lr is saturated at 5e-4 for 30-min budget**: H27 max_lr sweep both arms regress. Higher peak LR burns schedule budget before fine-tune tail. 5e-4 is near-optimal for this wall-clock window.
10. **WD and OneCycleLR are not orthogonal**: H26 retest +12%. The integrated lr×wd shrinkage under OneCycleLR's long low-LR tail is much larger than under cosine annealing. WD=1e-4 is the confirmed default.
11. **Preprocess MLP width is not a bottleneck**: H28 +21% with width 256→512. The current 256-wide preprocess (already 2-layer: 86→256→GELU→128) is not limiting. Capacity gains must go elsewhere.
12. **No-decay param groups are unexplored**: canonical practice (DeiT-III, ConvNeXt) excludes biases/LN/LayerScale from WD. Currently all params decay uniformly at 1e-4, potentially shrinking H18's LayerScale gains. H35 tests this.
13. **MSE in signed_log1p space is optimal for this dataset**: H32 closed. signed_log1p already neutralizes the heavy Re tail (residuals are O(0.1-1), not blowing up). L1's robustness buys nothing. Smooth_L1 β=0.1 was 5× MSE curvature in the wrong regime.
14. **Depth (n_layers=6) + OneCycleLR is structurally non-orthogonal under 30-min cap**: H25 retest closed +26.5% (2 seeds). 199s/epoch model never reaches fine-tune tail. The architecture is sound (H18 baseline showed -2.7%); the budget interaction is the barrier.
15. **Primary metric is p-channel only but loss is channel-uniform**: val_avg/mae_surf_p is pure pressure. `surf_loss` averages Ux, Uy, p equally → p gets 1/3 of gradient signal. H36 tests upweighting p directly.
16. **Schedule truncation is a first-class bug**: H24 baseline reaches LR=9.4e-5 at best_epoch=12 but schedule ends at LR=0 (epoch 15, never reached). H33 (pct_start) attacks this from the warmup-compression angle. H37 failed because epoch-count compression lost too many gradient steps.
17. **No-decay carve-out needs higher WD to matter**: H35 closed +13.3%. ls2 norm comparison showed flat ~0.42 under no_decay vs baseline's monotone 0.43→0.51 — deep-block LS actually grew LESS, refuting the "uniform WD silently shrinks LayerScale" premise. Integrated shrinkage at wd=1e-4 is ~1e-4 relative — numerically negligible.
18. **Input-side regularization is unexplored**: We've tested FFN dropout (H12 merged), attention dropout (H17 closed), weight decay (H26 closed), no-decay groups (H35 closed). Input-Jacobian regularization (Bishop 1995 equivalence) is a third axis — H38 tests this.
19. **SAM structurally incompatible with 30-min cap**: H31 closed. 2.1× per-step slowdown → 5-6 SAM epochs max. OneCycleLR schedules need 12+ epochs for fine-tune tail. Per-epoch wall-time variance (130s → 166s across node SKUs) is a confound for all schedule-fitting experiments — future proposals must use step-count targets, not epoch-count.
20. **n_freq=32 is empirically optimal for RFF on this dataset**: H34 closed. n_freq=16: +13.1%, n_freq=64: +21.2%. CFD pressure fields are spatially smoother than 3D NeRF textures — 32-mode basis with σ=1.0 captures all relevant frequencies. Asymmetry (doubling worse than halving) → fixed preprocess bottleneck (n_hidden=256) dilutes per-input-dim weights as rff_dim grows.
21. **val_single=80.32 anomaly is NOT a routing problem (refuted by H41)**: Dims 18-23 are EXACTLY zero for single-foil — the AND pattern is tight, not fuzzy. The preprocess MLP can learn this with a single hidden unit (one specific weight config suffices). Explicit is_tandem embedding regressed val_single +20% on the target split. The bottleneck must be something else: rarity (single-foil ~25% of balanced batches), variance (no foil2 to provide multi-modal pressure regime), shape distribution (single-foil surface node count), or output-scale calibration. H47 thorfinn tests rarity hypothesis via focal regression loss.
22. **Gradient clipping incompatible with surf_weight=10 + signed_log1p loss**: H30 closed +6.3%. Natural gradient norm is mean 4-30, max 150 under the current loss. max_norm=1.0 fires 100% of batches — transforms AdamW into sign-of-gradient updates, destroying moment estimates. OneCycleLR + log1p produces stable monotonic descent (H24 curve confirms); there is no instability to fix. Any grad-clip intervention needs max_norm ≥ 30 to be spike-only.
23. **OneCycleLR schedule axis is exhausted**: H27 max_lr, H33 pct_start, H37 epochs=12 all close. pct_start=0.30 + epochs=15 + max_lr=5e-4 are jointly near-optimal under the 30-min cap. Schedule-side work should pivot to fundamentally different cadences (warmup-free, snapshot ensembles, SWA tail-averaging via H40) rather than parameter sweeps. val_single regression in every compressed-warmup arm confirms: high-LR exploration is *essential* for in-dist generalization, not just OOD.
24. **slice_num=32 + OneCycleLR is anti-synergistic**: H23 retest closed +6.8% (4 tight seeds, σ=1.1). Both regularize the same overfitting axis — fast schedule already provides the regularization that slice_num bottleneck was providing. When stacked, OneCycleLR dominates and slice_num=32 just removes capacity. Critical compositionality lesson: when adding an axis, check whether it regularizes the same dimension as an already-merged change.
25. **Output-loss reweighting destroys multi-task representations**: H36 closed +13.4% (best arm). Ux/Uy surface supervision is NOT wasted gradient — it's auxiliary supervision regularizing the shared backbone. Cutting velocity loss weight corrupts the multi-task representation that p depends on. Rules out output-loss axis; channel imbalance must be attacked architecturally (H45 frieren pressure head split) not via gradient magnitudes.
26. **BF16 AMP works mechanistically but speedup is too small for our compute regime**: H39 closed +3.0% on val, -1.9% improved on test. OneCycleLR completed cleanly (LR=2e-9 by epoch 15, monotone descent). val_single regression dominates val_avg (+10.3%). Speedup was 5-25%, not 30-40% — Transolver's attention dominates compute (already fast in FP32), not matmul-heavy. AMP remains a tool for hypotheses needing more wall-clock budget (deeper models, slower per-step variants).
27. **val_single bottleneck is mechanistic, not routing-based** (H41 closure): The "researcher highest-priority" hypothesis refuted. Single-foil's exactly-zero dims 18-23 are trivially discriminated by the preprocess MLP. Explicit token-type embedding actively hurt val_single by +20%. Single-foil's high error must come from one of: (a) sampling rarity in batches → less gradient signal per natural-distribution sample; (b) higher target variance without foil2 interaction; (c) different surface-node density and mask structure; (d) output-scale calibration that's tuned to tandem distribution. H47 thorfinn focal loss tests (a) via hard-example up-weighting.

## Next directions (after current wave resolves)

- **slice_num=32 + n_layers=6 product sweep**: if H23 wins, revisit depth with cheaper attention (H25 showed depth is good architecturally, just budget-limited; slice_num=32 cuts ~50% per-epoch cost AND if AMP (H39) also wins, depth becomes viable again)
- **pct_start=0.20 + AMP combo**: if H33 and/or H39 win, combine them — rearrange schedule AND give it more epochs
- **per-channel loss with per-split weighting**: if H36 wins, consider also weighting re_rand vs single vs rc differently
- **Auxiliary physics loss**: lift/drag prediction head as auxiliary task
- **Geometry features**: curvature/normals as additional input channels
- **Deeper sweep below slice_num=32**: alphonse suggested {8, 16, 24} — monotonic trend suggests optimum still not found
