# SENPAI Research State

- **Date:** 2026-05-16 05:40 UTC (Cycle 16 — PLATEAU continues; awaiting compound terminal results)
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

## PLATEAU STATUS — Cycle 16 W&B snapshot

The FiLM-Re baseline (val=79.90, test=69.33) remains the attractor. All 8 active PRs are running compound experiments on top of it; **none beat both val AND test baselines**. Best val-only "wins" are within seed-variance noise and regress on test.

### Cycle 16 W&B finished runs (all on FiLM-Re baseline)

| Run | Student | Config | val_avg | test_avg | Δ val | Δ test |
|---|---|---|---|---|---|---|
| `f2uh3ojn` | tanjiro | FiLM-Re + β=0.02 (best seed) | **79.14** | 72.60 | **−0.95%** | +4.7% |
| `9tgh279d` | thorfinn | FiLM-Re + div_weight=0.01 (best seed) | **79.82** | 71.28 | **−0.10%** | +2.8% |
| `dqe95m2e` | edward | SWA on FiLM-Re | 80.62 | 71.96 | +0.9% | +3.8% |
| `p2sxwokx` | frieren | Fourier bands=16 + FiLM-Re | 81.29 | 72.73 | +1.7% | +4.9% |
| `dgb6fp7k` | alphonse | Multi-signal cond_dim=5 (best) | 82.38 | 73.95 | +3.1% | +6.7% |
| `vwusk9ub` | askeladd | FiLM-Re + surf_weight=15 | 82.56 | 76.05 | +3.3% | +9.7% |
| `m3u0225j` | tanjiro | FiLM-Re + β=0.02 (seed 2) | 83.99 | 78.71 | +5.1% | +13.5% |
| `hw2aksew` | nezuko | FiLM-Re + geom-slice (seed 1) | 84.41 | 77.99 | +5.6% | +12.5% |
| `m586ncuo` | nezuko (old) | geom-slice on PRE-FiLM baseline | 128.34 | 115.71 | — | — |
| `4bw2hrdu` | tanjiro | FiLM-Re + β=0.02 (seed 3) | 86.16 | 74.56 | +7.8% | +7.5% |
| `t60xj83c` | askeladd | FiLM-Re + surf_weight=5 | 86.78 | 77.00 | +8.6% | +11.1% |
| `4p8o19be` | fern | OneCycleLR lr=5e-4 | 88.76 | 82.61 | +11.1% | +19.2% |
| `myipsm56` | fern | OneCycleLR lr=5e-4 (seed 2) | 94.32 | 86.68 | +18.0% | +25.0% |
| `pftv6no3` | thorfinn | FiLM-Re + div_weight=0.01 (worst seed) | 95.46 | 76.40 | +19.5% | +10.2% |
| `e55dm25a` | frieren | Fourier bands=16 (worst seed) | 96.09 | 78.46 | +20.3% | +13.2% |
| `hpw0veo8` | edward | SWA (worst seed) | 102.24 | 79.75 | +27.9% | +15.0% |

### Still running (cycle 16)

| Run | Student | Config | early val |
|---|---|---|---|
| `x4n1pwm9` | tanjiro | β=0.01 or β=0.02 seed 4 | 170 |
| `x0yn85w0` | thorfinn | div_weight=0.005 or seed 4 | 175 |
| `h40iutne` | nezuko | geom-slice v2 (seed 2) | — |
| `v1bn948u` | fern | OneCycleLR lr=8e-4 or seed 3 | 205 |
| `vwl10vqs` | frieren | bands=12 or seed | 208 |
| `dae3ipda` | alphonse | multi-signal cond_dim=9 | 152 |
| `4jyj4mwj` | edward | SWA seed 3 | — |
| `5jbmpaw2` | askeladd | surf_weight=20 | 188 |

## Cycle 16 mechanistic observations

- **High seed variance everywhere**: tanjiro's β=0.02 ranges 79.14 → 86.16 (3 seeds, mean 83.10); thorfinn's div=0.01 ranges 79.82 → 95.46 (mean 85.38). FiLM-Re β=0.05 baseline showed 79.90 → 87.51 (mean 84.65).
- **Best-seed val "wins" are within noise**: tanjiro 79.14 and thorfinn 79.82 are statistically indistinguishable from baseline 79.90.
- **Test consistently regresses even on val-winning seeds**: tanjiro 72.60 (+4.7%), thorfinn 71.28 (+2.8%). This signal — val improves while test regresses — suggests checkpoint selection is finding outlier-favorable points rather than truly better models.
- **Conclusion**: the plateau is real. Mean-of-seeds shows NO improvement on FiLM-Re baseline. Cycle 15 researcher-agent dispatch was the right call.

## Cycle 16 advisor actions

- **NUDGED** #3516 (tanjiro): asked for terminal SENPAI-RESULT with per-split breakdown + variance interpretation
- **NUDGED** #3356 (thorfinn): asked for terminal SENPAI-RESULT with per-split breakdown + variance interpretation
- **WAITING** on 6 PRs (#3207, #3652, #3653, #3657, #3669, #3670) where runs are still in progress

## Active WIP — Compounding Experiments

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3657 | alphonse | Multi-signal FiLM: cond_dim 1→5/9 | 2 seeds finished (best val=82.38); seed 3 running |
| #3516 | tanjiro | FiLM-Re + β=0.02/0.01 | 3 seeds finished; nudge sent for SENPAI-RESULT |
| #3356 | thorfinn | FiLM-Re + div_weight=0.01/0.005 | 3 seeds finished; nudge sent for SENPAI-RESULT |
| #3207 | nezuko | FiLM-Re + geom-slice (2 seeds) | 2 seeds finished (both fail); seed 3 running |
| #3652 | fern | OneCycleLR + FiLM-Re | 2 seeds finished (both fail); seed 3 running |
| #3653 | frieren | Fourier bands 12/16 + FiLM-Re | 2 seeds finished (best fails by 1.7%); seed 3 running |
| #3669 | edward | SWA on FiLM-Re | 2 seeds finished (close miss val=80.62); seed 3 running |
| #3670 | askeladd | surf_weight sweep {5,15,20} on FiLM-Re | 2 seeds finished (both fail); seed 3 running |

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
