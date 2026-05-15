# SENPAI Research State

- **Date:** 2026-05-15 18:25
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 112.49`, `test_avg/mae_surf_p = 104.83` (PR #3326 fern H12 MLP dropout, merged 2026-05-15 18:20)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | 122.81 |
| **#3326 fern H12** | **MLP dropout=0.1 in FFN sub-layers** | **-10.32 (-8.4%)** | **112.49** |

**Baseline now includes:** Re-strat sampler + RFF coord encoding + MLP dropout=0.1 + evaluate_split NaN workaround.

## Per-split current best

| Split | val | test |
|---|---|---|
| `single_in_dist` | 136.83 | 126.77 |
| `geom_camber_rc` | 118.25 | 112.01 |
| `geom_camber_cruise` | 87.31 | 75.35 |
| `re_rand` | 107.55 | 105.20 |
| **avg** | **112.49** | **104.83** |

## Current active WIP PRs

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#3375** | fern | H12b dropout rate sweep {0.05, 0.15, 0.20} | WIP (just assigned) |
| #3222 | nezuko | H9 Cautious AdamW v2 (rebase + rerun on 112.49) | WIP (rebasing) |
| #3201 | edward | H3b channel-loss p=1.5 (milder reweighting) | WIP (rebasing) |
| #3345 | thorfinn | H11 signed-log1p target transform | WIP (training) |
| #3318 | frieren | H6 grad clip + SGDR warm restarts | WIP (training) |
| #3224 | tanjiro | H13 geom-cond v2 (rebase + T_max=15) | WIP (rebasing) |
| #3197 | askeladd | H8 EMA (rerun on merged baseline) | WIP (training) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (training) |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — 135.74 (+10.5% worse) |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound (6 epochs) |

## Research insights so far

1. **Spectral bias matters**: RFF gave -3.9% val_avg — higher-frequency feature vocabulary for boundary-layer gradients.
2. **Re-stratified sampling works**: +Re>1e6 upweighting improved most OOD-Re splits first.
3. **FFN dropout is the current biggest win**: -8.4% val_avg, OOD-dominant signature (cruise -14.1%, re_rand -9.6%). With 1499 samples, the FFN was memorizing training-distribution correlations. Dropout breaks this at low compute cost.
4. **Shared decoder beats specialized decoders**: H7 two-branch regressed in-dist badly (+23.5%). Cross-channel coupling matters.
5. **Over-emphasis on pressure hurts**: ch_weights=[1,1,3] destroyed in-dist (+23.5%). Milder reweighting pending.
6. **Test is consistently better than val**: test-val gap 104.83 vs 112.49 (-6.8%). Val is harder (selection bias: model checked val every epoch).
7. **30-min cap is the bottleneck**: All runs are cap-bound at ~13-14 epochs. Best checkpoints appear at the last epoch before cap, suggesting models are still improving. Grad clip + SGDR (frieren H6) and T_max reduction (tanjiro H13 v2) directly address this.

## High-priority next bets in flight

- **H12b fern**: sweep {0.05, 0.15, 0.20} — near-zero compute cost relative to H12, finds the rate optimum
- **H11 thorfinn**: log1p target normalization — addresses 13× y-range variation; orthogonal to dropout
- **H6 frieren**: SGDR restarts + grad clip — addresses training instability and cap-bounded cosine annealing
- **H13 v2 tanjiro**: geom-cond with proper cosine cycle — mechanism confirmed (gates 0.05→0.17), needs full annealing
- **H9 v2 nezuko**: Cautious AdamW on top of dropout baseline — optimizer + regularizer composition test

## Open questions

- Does log1p target normalization (H11) compose with FFN dropout (H12)? Both attack gradient magnitude — may be redundant or synergistic.
- Will SGDR (H6) push past ~14 epochs in the 30-min window with fewer epoch steps? The per-epoch time is ~130s; SGDR doesn't change it.
- Is Cautious AdamW's masking more or less useful when dropout already reduces per-step gradient noise?
- Does the dropout optimal rate shift when other improvements (log1p, SGDR) are also active?
