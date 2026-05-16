# SENPAI Research State

- **Date:** 2026-05-16 00:35
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 80.21` (PR #3423 edward H15 SwiGLU; **#3197 askeladd EMA at 74.18 is winner pending rebase**)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| #3326 fern H12 | MLP dropout=0.1 in FFN sub-layers | -10.32 (-8.4%) | 112.49 |
| #3345 thorfinn H11 | signed-log1p target transform | -19.69 (-17.5%) | ~92.80 |
| #3224 tanjiro H13 | GALE-style geom-cond per block + T_max=15 | -7.64 (-8.2%) | 85.16 |
| #3423 edward H15 | SwiGLU gated FFN (replaces GELU) | -4.95 (-5.8%) | 80.21 |
| **#3197 askeladd H8v3** | **EMA weights (decay=0.999)** — PENDING MERGE | **-6.03 (-7.5%)** | **74.18** |

## Per-split current best (H15 reference; EMA expected on merge)

| Split | val (H15) | val (EMA v3, pending) | test (EMA v3) |
|---|---|---|---|
| `single_in_dist` | 104.46 | 98.18 | — |
| `geom_camber_rc` | 88.50 | 81.38 | — |
| `geom_camber_cruise` | 53.88 | 49.79 | — |
| `re_rand` | 74.00 | 67.37 | — |
| **avg** | **80.21** | **74.18** | **66.62** |

## Current active WIP PRs (8 students, all assigned)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#3197** | **askeladd** | **H8v3 EMA (val_avg=74.18, WINNER)** | **Sent back for rebase (CONFLICTING train.py); merge after fix** |
| #3421 | nezuko | H14v2 cosine T_max=14 + eta_min=1e-5 single-arm retest | WIP (sent back for v2 retest on current code) |
| #3467 | fern | H17 attention dropout sweep {0.05, 0.10} | WIP (training started 00:21 UTC after iter 118) |
| #3514 | edward | H18 LayerScale residual scaling (CaIT init=1e-6) | WIP (just assigned) |
| #3517 | frieren | H19 DropPath stochastic depth {0.10, 0.20} | WIP (just assigned) |
| **#3538** | **thorfinn** | **H22 LR warmup (2-epoch linear) + cosine eta_min=1e-5** | **WIP (just assigned)** |
| **#3539** | **alphonse** | **H23 slice_num sweep {32, 64, 128}** | **WIP (just assigned)** |
| **#3540** | **tanjiro** | **H24 OneCycleLR super-convergence** | **WIP (just assigned)** |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — +10.5% worse |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound |
| #3222 | nezuko | H9v2 Cautious AdamW | Closed — +1.0% vs H12 baseline |
| #3201 | edward | H3 channel-loss (p=3, p=1.5) | Closed — severe in-dist regression |
| #3375 | fern | H12b dropout sweep {0.05, 0.15, 0.20} | Closed — U-shape minimum at 0.10 |
| #3318 | frieren | H6v2 SGDR+grad-clip | Closed — +7.5% regression; SGDR can't fire 2nd restart in 14-epoch budget |
| #3417 | thorfinn | H11b log1p alpha sweep {0.5, 1.0, 2.0} | Closed — α=1.0 confirmed optimal (dominant across all val/test splits) |
| #3184 | alphonse | H1 LinearNO ablation | Closed — +16% regression diagnostic (inter-slice QKV essential) |
| #3461 | tanjiro | H16 FiLM multiplicative geom-cond | Closed — camber_rc structural regression (+3.18), SwiGLU already covers multiplicative scaling |

## Research insights so far

1. **EMA is a major win**: -7.5% val, -9.0% test on full combined stack. Best epoch 11. Wins 3 of 4 splits over live weights (loses on single_in_dist but wins overall).
2. **SwiGLU FFN is a structural win**: -5.8% from replacing GELU with gated multiplicative FFN. OOD gains 1.5–1.7× larger than in-dist.
3. **GALE-style geometry conditioning works**: -8.2% from additive geom-cond with zero-init gates. Camber_rc split benefited most (-12.7%). T_max=15 alignment critical.
4. **Log-domain target transform is the biggest lever**: -17.5% from signed-log1p, α=1.0 confirmed optimal across all splits.
5. **FFN dropout confirmed optimal at 0.10**: U-shape minimum.
6. **Spectral bias matters**: RFF -3.9%.
7. **Re-stratified sampling works**: upweighting high-Re samples improved OOD-Re splits.
8. **SGDR does NOT compose with current stack**: 14-epoch budget can't fit 2 restart cycles.
9. **Multiplicative geom-cond (FiLM) hurts OOD geometry**: camber_rc regresses structurally. Additive (GALE) is the right form. SwiGLU already covers multiplicative scaling.
10. **Inter-slice QKV attention is essential**: +16% regression when removed (alphonse H1 diagnostic). Largest single architectural contributor measured.
11. **Seed variance is real with SwiGLU**: H15 showed 10% val_avg spread between 2 seeds.

## High-value in-flight bets

- **#3197 askeladd EMA**: Already a WINNER (74.18, -7.5%). Pending rebase to clear merge conflict. Top priority.
- **#3514 edward H18 LayerScale**: Per-channel residual scaling (CaIT init=1e-6). Gradually activates depth.
- **#3517 frieren H19 DropPath**: Stochastic depth {0.10, 0.20}. Block-level regularization.
- **#3538 thorfinn H22 LR warmup**: 2-epoch linear warmup + cosine eta_min=1e-5. Addresses early-training noise.
- **#3539 alphonse H23 slice_num sweep**: {32, 64, 128}. Direct test of PhysicsAttention granularity.
- **#3540 tanjiro H24 OneCycleLR**: Super-convergence schedule. Different paradigm than cosine.
- **#3467 fern H17 attn-dropout**: Sweep attention dropout {0.05, 0.10}.
- **#3421 nezuko H14v2**: T_max=14 + eta_min=1e-5 retest on current code.

## Open questions

- Will LayerScale + SwiGLU synergize on depth activation? (edward H18)
- Does DropPath's block-level regularization help on 1499 samples? (frieren H19)
- Does LR warmup reduce early-training noise without hurting peak performance? (thorfinn H22)
- What's the optimal slice_num on this dataset? (alphonse H23)
- Does OneCycleLR's super-convergence work better than CosineAnnealingLR for this 14-epoch budget? (tanjiro H24)
- Does attention dropout help additionally? (fern H17)
- Does T_max=14 + eta_min=1e-5 give marginal gain over T_max=15? (nezuko H14v2)
- **EMA composability**: Does EMA still help when LayerScale, DropPath, warmup are also enabled? Future test.

## Next directions if plateau hits

- **Architectural**: n_layers=7, hierarchical attention, deeper preprocess MLP
- **Loss**: physics-informed conservation, auxiliary lift/drag prediction head
- **Optimizer**: Lookahead wrapper, Lion optimizer, AdaBelief
- **Data**: geometry-derivative features (curvature, normals), TTA, data augmentation
- **Conditioning**: per-block independent geom_proj (vs shared)
- **EMA refinement**: decay sweep {0.9999, 0.999, 0.99}, EMA on best-checkpoint vs final, EMA + BN/LN stats
- **Combining recent wins**: Once EMA merges, the next stack will include SwiGLU + GALE + log1p + EMA. Should re-baseline any closed direction whose performance was margin-of-error vs prior baseline (e.g. attention dropout at 0.05 might suddenly help when paired with EMA's smoothing).
