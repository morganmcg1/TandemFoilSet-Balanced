# SENPAI Research State

- **Date:** 2026-05-16 08:15 UTC (Cycle 19 — plateau-break round launched; 0 idle students)
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

**SWA mechanism:** `AveragedModel` with per-step updates starting at epoch 8/13 (swa_start_epoch=7, 0-indexed). Best-val ckpt alone (val=80.62, test=71.96) does NOT beat prior baseline — the SWA averaging is the key. Zero GPU memory overhead. Now default in `train.py`.

### Merge history
| Date | PR | Title | val_avg | test_avg | Δ val_avg |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.50 | 112.49 | first |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands | 116.34 | 107.33 | −4.2% |
| 2026-05-15 23:20 | #3215 | SmoothL1 β=0.05 | 90.60 | 83.00 | −22.1% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning | 79.90 | 69.33 | −11.8% |
| **2026-05-16 08:00** | **#3669** | **SWA on FiLM-Re** | **76.61** | **68.20** | **−4.1%** |

## Cycle 19 actions (just executed)

**Closed** (no longer compounding against new SWA baseline):
- PR #3356 (thorfinn) — divergence-free aux loss; student replied "Option A — closing" with mechanistic reasoning that FiLM-Re's per-block Re-conditioning already imposes a physics-aware prior the div-free penalty competes with.
- PR #3653 (frieren) — Fourier bands=16; student's own SENPAI-RESULT recommended closure (3-seed mean misses both baselines by 6-11%; sample-complexity-limited at 13 epochs, not capacity-limited).
- PR #3207 (nezuko) — FiLM-Re + geom-slice v2 compound; v2 primary (usqypjfh) val=81.90 / test=73.82 vs SWA 76.61/68.20. Per-split signature shows OOD splits regressing while in-dist improves — exactly the wrong direction for an inductive-bias improvement.

**Assigned** (3 new plateau-break experiments now running):

| Student | PR | Idea | Why this student |
|---|---|---|---|
| thorfinn | #3813 | Per-Sample Re-Scaled Normalization (Idea 1, ranked #1 by researcher-agent) | Strong physics-loss/scaling instincts; this is the loss-side analog of FiLM-Re |
| frieren | #3816 | Stochastic Depth / LayerDrop sweep {0.05, 0.10, 0.15} (Idea 6) | Has demonstrated clean multi-seed sweep methodology |
| nezuko | #3820 | Residual learning over linear baseline (Idea 2, DeltaPhi-style) | Meticulous data-side work; found the scoring.py NaN bug |

## Active Research Directions — Cycle 19 in-flight (8 WIP)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3657 | alphonse | Multi-signal FiLM cond_dim=5/9 (geometry-augmented FiLM) | WIP, SENPAI-RESULT pending |
| #3670 | askeladd | surf_weight sweep {5,15,20} on FiLM-Re | WIP, SENPAI-RESULT pending |
| #3799 | edward | EMA vs uniform SWA (decay sweep) | WIP |
| #3803 | tanjiro | SWA start epoch sweep {4,6,8,10} | WIP |
| #3806 | fern | Surface-Dedicated Refinement MLP (Idea 3) | WIP |
| **#3813** | **thorfinn** | **Per-Sample Re-Scaled Norm (Idea 1)** | **NEW — just assigned** |
| **#3816** | **frieren** | **Stochastic Depth / LayerDrop sweep (Idea 6)** | **NEW — just assigned** |
| **#3820** | **nezuko** | **Residual learning over linear baseline (Idea 2)** | **NEW — just assigned** |

## Plateau-break strategy (this cycle)

The researcher-agent's cycle-15 diagnosis identifies three orthogonal axes that haven't been pulled:

1. **Loss-side** — directly condition the OBJECTIVE on Re (per-sample normalization), not just the model weights. (#3813 thorfinn)
2. **Regularization-side** — stochastic depth implicitly ensembles sub-networks and reduces per-step compute. Orthogonal to FiLM-Re and SWA. (#3816 frieren)
3. **Target-side** — residualize the target with a linear baseline, reducing magnitude and freeing capacity for the nonlinear physics. (#3820 nezuko)

In parallel, edward (#3799), tanjiro (#3803), and fern (#3806) are exploring SWA mechanism variants and surface-specialized capacity. Total coverage: 8 active experiments spanning 5 orthogonal mechanisms.

## Goal

Push val < 72, test < 65. SWA broke the 79.90 → 76.61 plateau. The next break needs to come from one of the 8 active mechanisms above — most likely loss-side or target-side (Idea 1 or 2 in the researcher-agent's ranking).

## Architecture tier (if plateau-break round stalls)

- Hypernetwork Re conditioning (Idea 4) — only after low-risk ideas have results
- Multiscale mesh pooling / domain-aware coarsening (Idea 5) — staged probe first
- Bernoulli consistency aux loss (Idea 8) — if ideas 1-4 fail
- GNN over mesh / Galerkin transformer / FNO hybrid — architecture-tier reset
