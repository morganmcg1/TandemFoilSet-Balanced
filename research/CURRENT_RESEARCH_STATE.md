# SENPAI Research State

- **Date:** 2026-05-16 09:45 UTC (Cycle 21 — close 1 PR (#3820), assign Idea 5 multiscale probe; all 8 students WIP)
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

## Cycle 21 actions (just executed)

**Closed (1 PR):**
- **PR #3820 (nezuko, residual learning over linear baseline)** — SWA val=80.40/test=71.70 (+3.8%/+3.5% worse than baseline). The student produced a load-bearing diagnostic: per-sample DC R² of 0.53 looks promising but corresponds to only 3.7% of node-level p variance. The mechanism collapsed because surface pressure variance is dominated by *intra-sample* structure (geometry, boundary layer) — not between-sample DC level. Per-sample residualisation is closed; per-node panel-method variant remains viable for future cycles.

**Assigned (1 new PR):**

| Student | PR | Idea | Why |
|---|---|---|---|
| nezuko | #3856 | Multiscale background subsampling probe (Idea 5, staged) | Nezuko's diagnostic rigor + geometry awareness is the right fit for this probe; saf feature gives natural zone partition; uses 5-15% near-foil + K background tokens |

## Active Research Directions — Cycle 21 in-flight (8 WIP, 0 idle)

All 8 students running plateau-break experiments spanning 7 orthogonal mechanisms:

| PR | Student | Idea / Mechanism | Started |
|---|---|---|---|
| #3799 | edward | EMA vs uniform SWA (decay sweep) — SWA mechanism variant | cycle 18 |
| #3803 | tanjiro | SWA start epoch sweep {4,6,8,10} — SWA mechanism variant | cycle 18 |
| #3806 | fern | Surface-Dedicated Refinement MLP (Idea 3) | cycle 18 |
| #3813 | thorfinn | Per-Sample Re-Scaled Normalization (Idea 1, rank #1) | cycle 19 |
| #3816 | frieren | Stochastic Depth / LayerDrop sweep (Idea 6) | cycle 19 |
| #3828 | alphonse | Low-rank hypernetwork Re conditioning (Idea 4) | cycle 20 |
| #3831 | askeladd | Bernoulli consistency aux loss (Idea 8) | cycle 20 |
| **#3856** | **nezuko** | **Multiscale BG subsampling probe (Idea 5, staged)** | **cycle 21 — NEW** |

## Plateau-break strategy (complete coverage of cycle-15 ideas)

With nezuko's multiscale probe now in flight, **all 8 ideas from the researcher-agent's cycle-15 plateau-break analysis are either running or merged**:

| Idea | Status | Student/PR |
|---|---|---|
| 1. Per-sample Re-scaled loss norm | Running | thorfinn #3813 |
| 2. Residual learning over analytic | **Closed** (per-sample DC tested); panel-method variant deferred | nezuko #3820 (closed) |
| 3. Surface-dedicated refinement | Running | fern #3806 |
| 4. Hypernetwork Re conditioning | Running | alphonse #3828 |
| 5. Multiscale mesh pooling | **Running** (staged probe) | nezuko #3856 |
| 6. Stochastic depth / LayerDrop | Running | frieren #3816 |
| 7. Checkpoint weight averaging | **Merged** as SWA baseline | edward #3669 |
| 8. Bernoulli consistency loss | Running | askeladd #3831 |

Plus 2 SWA-mechanism-tuning experiments (edward EMA, tanjiro start-epoch).

## Goal

Push val < 72, test < 65. Cycle 21 closes the residual-baseline thread and operationalises the last remaining plateau-break idea (multiscale, staged probe). Results from the 8 in-flight experiments should arrive over the next 30-90 min as each 30-minute training run completes. With 8 orthogonal mechanisms simultaneously, at least one is likely to show signal.

## Next steps once cycle-21 results arrive

- If 2+ mechanisms beat baseline, consider compounding the winners.
- If 0-1 mechanisms beat baseline, dispatch researcher-agent for a fresh round of ideas (deepening into multiscale architecture if probe shows promise, or new directions like Galerkin transformer, GNN, FNO hybrid).

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
- **Per-sample DC residual baseline** (NEW — 3.7% node-level variance reduction too small; competes with FiLM-Re)
