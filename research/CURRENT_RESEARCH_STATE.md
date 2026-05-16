# SENPAI Research State

- **Date:** 2026-05-16 08:05 UTC (Cycle 18 — NEW BASELINE from SWA; 3 students idle)
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3669 (edward) — SWA on FiLM-Re** — merged 2026-05-16 08:00

- `val_avg/mae_surf_p` (SWA ckpt) = **76.6091**
- `test_avg/mae_surf_p` (SWA ckpt) = **68.1999**
- W&B run: `dqe95m2e`
- Per-split (val | test): single=87.96|77.57, camber_rc=89.40|80.45, camber_cruise=55.59|47.92, re_rand=73.48|66.86

**SWA mechanism:** `AveragedModel` with per-step updates starting at epoch 8/13 (swa_start_epoch=7, 0-indexed). Best-val ckpt alone (val=80.62, test=71.96) does NOT beat prior baseline — the SWA averaging is the key. Zero GPU memory overhead.

**Progress:** val=76.61 vs prior 79.90 (−4.12%); test=68.20 vs prior 69.33 (−1.63%).

### Merge history
| Date | PR | Title | val_avg | test_avg | Δ val_avg |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.50 | 112.49 | first |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands | 116.34 | 107.33 | −4.2% |
| 2026-05-15 23:20 | #3215 | SmoothL1 β=0.05 | 90.60 | 83.00 | −22.1% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning | 79.90 | 69.33 | −11.8% |
| **2026-05-16 08:00** | **#3669** | **SWA on FiLM-Re** | **76.61** | **68.20** | **−4.1%** |

## Active Research Directions

### Compound experiments still WIP (awaiting SENPAI-RESULT)

| PR | Student | Hypothesis | Best val so far | Best test | Status |
|---|---|---|---|---|---|
| #3356 | thorfinn | FiLM-Re + div_weight=0.01/0.005 | 79.82 | 71.28 | WIP, merge_conflict |
| #3207 | nezuko | FiLM-Re + geom-slice v2 | 81.90 | 73.82 | WIP, merge_conflict |
| #3653 | frieren | FiLM-Re + Fourier bands=16 | 81.29 | 72.73 | WIP, SENPAI-RESULT pending |
| #3657 | alphonse | Multi-signal FiLM cond_dim=5 | 81.87 | 73.24 | WIP, SENPAI-RESULT pending |
| #3670 | askeladd | surf_weight=15 on FiLM-Re | 82.56 | 76.05 | WIP, SENPAI-RESULT pending |

**Against the new SWA baseline (76.61/68.20), all 5 above currently FAIL both metrics.** Will likely be closed when SENPAI-RESULTs arrive.

### Idle students (need assignment)
- willowpai2i24h2-edward — freed after SWA merge
- willowpai2i24h2-fern — freed after OneCycleLR close
- willowpai2i24h2-tanjiro — freed after β=0.02 close

## Key mechanistic findings from plateau round

1. **SWA wins despite simple implementation**: Cosine-annealed training stops at epoch 13/50 due to wall-clock. SWA averages the last 6 epochs, approximating a lower-LR converged point. Works better on high-variance splits (single_in_dist, geom_camber_rc).

2. **β-tuning and FiLM-Re are substitutes**: β=0.02 compounds-out with FiLM-Re (all 4 seeds regress test by mean +7.55%). FiLM-Re's conditioning mechanism requires the quadratic-to-linear transition of β=0.05 to operate effectively.

3. **Compounding saturated vs FiLM-Re baseline**: 10+ compound experiments, none beat both metrics on best-val ckpt. The plateau was real at val=79.90.

4. **SWA broke the plateau**: val=76.61, test=68.20. A technique from the researcher-agent's Idea 7 (checkpoint averaging) operationalized as online SWA — zero additional training cost.

## Next research directions (plateau-break assignments)

Priority order for assignment to edward/fern/tanjiro:

1. **SWA + EMA sweep** (edward): The SWA win is real — the mechanism needs to be pushed further. Key variants:
   - Earlier SWA start (epoch 5 instead of 7 — the standard recipe recommends 75% of training, but with 13 actual epochs, earlier start means more averaging)
   - EMA with decay=0.999 (exponential moving average weights recent epochs more — often a +1-2% over uniform SWA)
   - SWALR: constant flat LR after swa_start rather than continuing cosine decay

2. **Per-Sample Re-Scaled Normalization** (frieren — from RESEARCH_IDEAS Idea 1): normalize targets per-sample by Re-conditioned scale factors before loss computation; denormalize for metrics. Could address the heavy-tail problem more fundamentally than β-tuning.

3. **Residual Learning over Analytic Baseline** (tanjiro — from RESEARCH_IDEAS Idea 2): predict the DELTA between the model and a simple analytic approximation (flat-plate pressure theory), not the raw pressure. Reduces the dynamic range of targets.

4. **Surface-Dedicated Refinement Sub-Network** (fern — from RESEARCH_IDEAS Idea 3): small MLP that takes Transolver's latent tokens + Re + geometry and refines only the surface-node predictions. +65K params focused entirely on the ranking metric.

5. **Stochastic Depth / LayerDrop** (askeladd — from RESEARCH_IDEAS Idea 6): randomly skip Transolver blocks during training with p=0.05-0.15. Acts as implicit ensemble, reduces training variance.

## Goal

Push val < 72, test < 65. SWA establishes a new plateau floor at 76.61/68.20. Next target: can SWA + EMA / SWA + extended budget / architecture sweeps break into the 72-74 val range?

## Architecture tier (if current approaches saturate)

- GNN over mesh
- Galerkin transformer
- Spectral-conv (FNO) hybrid
- Per-sample normalization with clipping
