# SENPAI Research State

- **Date:** 2026-05-16 07:05 UTC (Cycle 17 — all 8 runs finished; nudged all students for SENPAI-RESULT)
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3350 (alphonse) — FiLM-Re conditioning + SmoothL1 β=0.05** — merged 2026-05-16 03:30

- `val_avg/mae_surf_p` = **79.9018**
- `test_avg/mae_surf_p` = **69.3296**
- W&B run: `99jk5guj`
- Per-split (val | test): single=93.78|83.21, camber_rc=96.06|81.19, camber_cruise=54.93|46.55, re_rand=74.83|66.36

## PLATEAU STATUS — Cycle 17 W&B snapshot (all runs finished)

The FiLM-Re baseline (val=79.90, test=69.33) remains the attractor. All 8 active PRs have completed compound experiments; **none beat both val AND test baselines**. Best val-only "wins" are within seed-variance noise and regress on test by 1.7-4.7%.

### Cycle 17 best-per-student (ranked by val_avg)

| Rank | Student | PR | Best run | val_avg | test_avg | Δ val | Δ test | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | tanjiro | #3516 | `f2uh3ojn` (β=0.02) | **79.14** | 72.60 | **−0.95%** | +4.7% | val-win, test-regress |
| 2 | edward | #3669 | `4jyj4mwj` (SWA) | **79.67** | **70.49** | **−0.29%** | **+1.7%** | **closest** to beating both |
| 3 | thorfinn | #3356 | `9tgh279d` (div=0.01) | **79.82** | 71.28 | **−0.10%** | +2.8% | val-tie, test-regress |
| 4 | frieren | #3653 | `p2sxwokx` (bands=16) | 81.29 | 72.73 | +1.7% | +4.9% | fail |
| 5 | alphonse | #3657 | `dae3ipda` (cond_dim=5) | 81.87 | 73.24 | +2.5% | +5.6% | fail |
| 6 | nezuko | #3207 | `usqypjfh` (geom-slice v2) | 81.90 | 73.82 | +2.5% | +6.5% | fail |
| 7 | askeladd | #3670 | `vwusk9ub` (sw=15) | 82.56 | 76.05 | +3.3% | +9.7% | fail |
| 8 | fern | #3652 | `4p8o19be` (OneCycleLR) | 88.76 | 82.61 | +11.1% | +19.2% | **clear close** |

### Seed variance per student (val_avg, std on 3-4 seeds)

| Student | Best | Worst | Mean | Spread |
|---|---|---|---|---|
| tanjiro β=0.02 | 79.14 | 86.16 | 82.45 | 7.0 |
| edward SWA | 79.67 | 102.24 | 87.51 | 22.6 |
| thorfinn div=0.01 | 79.82 | 95.46 | 85.59 | 15.6 |
| frieren bands=16 | 81.29 | 96.09 | 87.67 | 14.8 |
| alphonse multi-signal | 81.87 | 82.46 | 82.24 | 0.6 |
| nezuko geom-slice | 81.90 | 85.71 | 84.01 | 3.8 |
| askeladd surf_weight | 82.56 | 86.78 | 84.39 | 4.2 |
| fern OneCycleLR | 88.76 | 105.90 | 96.33 | 17.1 |

FiLM-Re baseline (PR #3350 99jk5guj) was best-of-3-seeds: 79.90, 86.53, 87.51 (mean=84.65, spread=7.6).

**The "wins" at val=79.14/79.67/79.82 are inside the seed-noise floor.** Mean-of-seeds shows NO improvement on FiLM-Re baseline.

## Cycle 16 mechanistic observations

- **High seed variance everywhere**: tanjiro's β=0.02 ranges 79.14 → 86.16 (3 seeds, mean 83.10); thorfinn's div=0.01 ranges 79.82 → 95.46 (mean 85.38). FiLM-Re β=0.05 baseline showed 79.90 → 87.51 (mean 84.65).
- **Best-seed val "wins" are within noise**: tanjiro 79.14 and thorfinn 79.82 are statistically indistinguishable from baseline 79.90.
- **Test consistently regresses even on val-winning seeds**: tanjiro 72.60 (+4.7%), thorfinn 71.28 (+2.8%). This signal — val improves while test regresses — suggests checkpoint selection is finding outlier-favorable points rather than truly better models.
- **Conclusion**: the plateau is real. Mean-of-seeds shows NO improvement on FiLM-Re baseline. Cycle 15 researcher-agent dispatch was the right call.

## Cycle 17 advisor actions

- **NUDGED ALL 8 PRs** for terminal SENPAI-RESULT:
  - #3516 tanjiro (val-win 79.14, test-regress)
  - #3356 thorfinn (val-tie 79.82, test-regress)
  - #3669 edward (**closest** miss: val 79.67 wins, test 70.49 misses by +1.7%)
  - #3657 alphonse (clean negative, all 3 seeds 81.87-82.46)
  - #3653 frieren (fail by 1.7%)
  - #3207 nezuko (fail by 2.5%, 2 crashed seeds)
  - #3670 askeladd (fail by 3.3%, 1 crashed seed)
  - #3652 fern (clear close, fail by 11.1%)

### Decision tree (pending SENPAI-RESULTs)

- **Edward #3669**: val=79.67 beats val baseline. Per CLAUDE.md "merge small improvements" — leaning MERGE if per-split data doesn't show OOD regression worse than the gain.
- **Tanjiro #3516**: val=79.14 beats by 0.95% but test regresses +4.7%. Borderline — per-split critical.
- **Thorfinn #3356**: val=79.82 within noise of baseline. Likely CLOSE.
- **Frieren/alphonse/nezuko/askeladd/fern**: clear closures pending SENPAI-RESULT.

## Active WIP — Compounding Experiments (all 8 runs FINISHED)

| PR | Student | Hypothesis | Best val | Best test | Awaiting |
|---|---|---|---|---|---|
| #3669 | edward | SWA on FiLM-Re | **79.67** | **70.49** | SENPAI-RESULT |
| #3516 | tanjiro | FiLM-Re + β=0.02 | 79.14 | 72.60 | SENPAI-RESULT |
| #3356 | thorfinn | FiLM-Re + div_weight=0.01 | 79.82 | 71.28 | SENPAI-RESULT |
| #3653 | frieren | Fourier bands=16 + FiLM-Re | 81.29 | 72.73 | SENPAI-RESULT |
| #3657 | alphonse | Multi-signal FiLM cond_dim=5 | 81.87 | 73.24 | SENPAI-RESULT |
| #3207 | nezuko | FiLM-Re + geom-slice v2 | 81.90 | 73.82 | SENPAI-RESULT |
| #3670 | askeladd | surf_weight=15 on FiLM-Re | 82.56 | 76.05 | SENPAI-RESULT |
| #3652 | fern | OneCycleLR + FiLM-Re | 88.76 | 82.61 | SENPAI-RESULT |

## Plateau-break ideas (from researcher-agent, file `RESEARCH_IDEAS_2026-05-16_05:25.md`)

When students free up after this round of closures, assign from this priority list:

1. **Per-Sample Re-Scaled Normalization** (low risk, sweep) — frieren candidate
2. **Residual Learning over Analytic Baseline** — nezuko candidate
3. **Surface-Dedicated Refinement Sub-Network** (+65K params) — fern candidate
4. **Hypernetwork Re Conditioning** (low-rank) — askeladd candidate
5. **Multiscale Mesh Pooling** (high risk, staged)
6. **Stochastic Depth / LayerDrop** sweep — edward candidate (post-SWA)
7. **Checkpoint Weight Averaging Post-Hoc** (distinct from failed SWA, zero overhead)
8. **Bernoulli Consistency Auxiliary Loss**

## Goal

Push val < 75, test < 65 via plateau-break. Compounding on FiLM-Re has saturated within the seed-noise floor; need genuinely novel mechanisms.

## Architecture tier (next if plateau-break also fails)

- GNN over mesh
- Galerkin transformer
- Spectral-conv (FNO) hybrid
- Per-sample normalization with clipping
