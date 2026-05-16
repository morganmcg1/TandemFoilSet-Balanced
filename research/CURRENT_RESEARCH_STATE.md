# SENPAI Research State

- **Date:** 2026-05-16 08:40 UTC (Cycle 20 — close 2 PRs, full plateau-break round (8/8) now in flight)
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

### Merge history
| Date | PR | Title | val_avg | test_avg | Δ val_avg |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.50 | 112.49 | first |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands | 116.34 | 107.33 | −4.2% |
| 2026-05-15 23:20 | #3215 | SmoothL1 β=0.05 | 90.60 | 83.00 | −22.1% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning | 79.90 | 69.33 | −11.8% |
| **2026-05-16 08:00** | **#3669** | **SWA on FiLM-Re** | **76.61** | **68.20** | **−4.1%** |

## Cycle 20 actions (just executed)

**Closed** (2 PRs, both with student-recommended closure and thorough mechanistic analyses):

- **PR #3670 (askeladd, surf_weight sweep)** — best arm sw=15 val=82.56/test=76.05 vs SWA 76.61/68.20 (+7.8% val, +11.5% test). Curve mapped at {5,10,15,20} — sw=10 wins. Load-bearing insight from student: volume loss is a *structural prior* for surface generalisation, not just a regulariser (camber holdouts most volume-sensitive).
- **PR #3657 (alphonse, multi-signal FiLM cond_dim=5)** — 3-seed clean negative (mean val=82.24, test=73.30; +7.4%/+7.5%). Regression driven entirely by single_in_dist (+15% val). Student diagnosis: FiLM bottleneck is narrow (`Linear(cond_dim, 32) → GELU → Linear(32, 2·hidden)`), wider input through fixed bottleneck doesn't help; geometry info redundant with per-node features.

**Assigned** (2 new plateau-break experiments now running):

| Student | PR | Idea | Why this student |
|---|---|---|---|
| alphonse | #3828 | Low-rank hypernetwork Re conditioning on PhysicsAttention to_v (Idea 4) | Alphonse built FiLM-Re; diagnosed bottleneck issue cleanly; hypernetwork is the natural "widen the bottleneck" follow-up they proposed |
| askeladd | #3831 | Bernoulli consistency aux loss (Idea 8) | Askeladd's surf_weight closing insight (volume-as-structural-prior, channel coupling matters) motivates the Bernoulli p+0.5|U|² approach directly |

## Active Research Directions — Cycle 20 in-flight (8 WIP, 0 idle)

All 8 students now running plateau-break experiments spanning 6 orthogonal mechanisms:

| PR | Student | Idea / Mechanism | Status |
|---|---|---|---|
| **#3799** | edward | EMA vs uniform SWA (decay sweep) — SWA mechanism variant | WIP, started cycle 18 |
| **#3803** | tanjiro | SWA start epoch sweep {4,6,8,10} — SWA mechanism variant | WIP, started cycle 18 |
| **#3806** | fern | Surface-Dedicated Refinement MLP (Idea 3) | WIP, started cycle 18 |
| **#3813** | thorfinn | Per-Sample Re-Scaled Normalization (Idea 1, rank #1) | WIP, started cycle 19 |
| **#3816** | frieren | Stochastic Depth / LayerDrop sweep (Idea 6) | WIP, started cycle 19 |
| **#3820** | nezuko | Residual learning over linear baseline (Idea 2) | WIP, started cycle 19 |
| **#3828** | alphonse | Low-rank hypernetwork Re conditioning (Idea 4) | **NEW — just assigned** |
| **#3831** | askeladd | Bernoulli consistency aux loss (Idea 8) | **NEW — just assigned** |

## Plateau-break strategy (full coverage now)

Six orthogonal mechanisms covering the researcher-agent's cycle-15 plateau diagnosis:

1. **SWA tuning** (edward EMA, tanjiro start-epoch): refine the SWA win itself.
2. **Surface specialisation** (fern): add capacity dedicated to the metric-relevant nodes.
3. **Loss-side normalisation** (thorfinn): directly condition the OBJECTIVE on Re.
4. **Regularisation** (frieren): stochastic depth as orthogonal ensemble.
5. **Target-side** (nezuko): residualize target with linear baseline.
6. **Conditioning expressivity** (alphonse): weight-generating hypernetwork generalises FiLM beyond diagonal modulation.
7. **Physics coupling** (askeladd): Bernoulli aux loss explicitly couples (p, Ux, Uy).

This is the broadest mechanism-coverage round so far. If 2-3 of these compound, val can plausibly reach the 72-74 range targeted.

## Goal

Push val < 72, test < 65. SWA broke the 79.90 → 76.61 plateau. Cycle 20 has 8 orthogonal mechanisms running simultaneously — at least one is likely to show signal in the next 30-90 minutes as runs complete.

## Architecture tier (if cycle-20 round stalls)

- Multiscale mesh pooling / domain-aware coarsening (Idea 5) — staged probe first
- GNN over mesh / Galerkin transformer / FNO hybrid — architecture-tier reset
- Boundary-first tokenization (SpiderSolver-style) — surface-specialised architecture
- Re-trigger researcher-agent for fresh ideas if all 8 cycle-20 mechanisms underperform

## What has been definitively ruled out (do not retry on this baseline)

- β-tuning of SmoothL1 (β=0.02, 0.01 — both regress in compound)
- OneCycleLR / cosine warmup at 30-min budget
- Depth/width scaling (mlp_ratio=4, larger hidden)
- More Fourier bands (16) — marginal signal, hurts at 13 epochs
- Divergence-free physics aux loss (fragile, competes with FiLM-Re)
- surf_weight off baseline (full curve mapped at {5, 10, 15, 20})
- Geom-slice (standalone and in compound with FiLM-Re)
- Multi-signal FiLM (cond_dim=5 — narrow bottleneck, redundant signal)
- Per-channel surface loss weights (edward #3198 in pre-FiLM era)
- Domain one-hot embedding (edward #3523)
