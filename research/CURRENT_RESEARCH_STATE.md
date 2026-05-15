# SENPAI Research State

- **Date:** 2026-05-15 16:30
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 122.81`, `test_avg/mae_surf_p = 111.16` (PR #3217 frieren H5 RFF, merged 2026-05-15 16:25)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | **122.81** |

**Baseline now includes:** Re-strat sampler + RFF coord encoding + evaluate_split NaN workaround.

## Per-split current best

| Split | val | test |
|---|---|---|
| `single_in_dist` | 144.70 | 123.91 |
| `geom_camber_rc` | 125.95 | 114.82 |
| `geom_camber_cruise` | 101.61 | 88.14 |
| `re_rand` | 119.00 | 117.78 |
| **avg** | **122.81** | **111.16** |

## Current active WIP PRs (round 2)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head (surface vs volume) | WIP (just assigned) |
| #3222 | nezuko | H9 Cautious AdamW | WIP (training, GPU 99%) |
| #3224 | tanjiro | H13 geom-cond v2 (re-run + T_max=15) | WIP (re-running on baseline) |
| #3217 | frieren | **MERGED** — now assigned H6 | (see below) |
| #3210 | fern | H2 scale v2 (grad clip + n_hidden=192) | WIP (revising) |
| #3201 | edward | H3 channel-weighted loss | WIP (training, GPU 99%) |
| #3197 | askeladd | H8 EMA (re-run over merged baseline) | WIP (re-running) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (setup) |

## Upcoming new work

- **frieren → H6**: grad clip (max_norm=1.0) + CosineAnnealingWarmRestarts (T_0=10, T_mult=2). Targets the noisy training curve seen in the RFF run (216→189→198→... before cosine annealed); SGDR gives multiple LR restarts vs. one monotone decay. (PR being created this session.)

## Research insights so far

1. **Spectral bias matters**: RFF gave -3.9% val_avg, acting on boundary-layer gradients that raw (x,z) coordinates can't represent efficiently.
2. **High-Re upweighting works**: Re-strat sampler gave -5.03% val_avg in its first pass; val_re_rand (119.00) and val_geom_camber_cruise (101.61) are the two strongest splits.
3. **val_single_in_dist is the remaining bottleneck**: at 144.70 it's 20% above the best other split. Two-branch head (thorfinn H7) is the most targeted mechanism for this.
4. **Test metrics now reliable**: frieren's NaN fix means test_avg is a real number. The test-val gap is significant (111.16 vs 122.81, -9.5%) — test consistently lower (better) than val, suggesting val is harder or that the test-optimal checkpoint differs from val-optimal.

## Potential next research directions (round 3+)

After round-2 WIP resolves:
- **Stacking round** — compose any round-2 winners (EMA + two-branch head + cautious adamw, etc.)
- **H4 asymmetric Q/K**: orthogonal attention modification
- **H11 log1p target normalization**: y values span orders of magnitude; compressing y space may help optimization
- **H12 MLP dropout 0.1**: lightweight regularization
- **Architecture search**: try slice_num=128 or n_layers=6 with the same 1M param budget repartitioned
- **val_single_in_dist targeted**: data analysis to understand why in-distribution single foil is harder than OOD camber splits

## Open questions

- Will EMA + RFF + Re-strat compose to sub-120 val_avg?
- Does Cautious AdamW's masking benefit increase or decrease after the optimizer has stable RFF gradients?
- What explains the 20% harder val_single_in_dist vs. other splits? Is it the wake-interaction complexity, sample count imbalance, or Re distribution within that split?
