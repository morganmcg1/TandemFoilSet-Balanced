# SENPAI Research State

- **Date:** 2026-05-16 11:35 UTC (Cycle 23 — merge #3806 (fern, surface refinement); close #3816 (frieren, layerdrop); assign #3917 (fern, EMA compound) + #3920 (frieren, EMA decay sweep))
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3806 (fern) — Surface-Dedicated Refinement MLP** — merged 2026-05-16 11:28

- `val_avg/mae_surf_p_swa` (SWA ckpt) = **76.2033**
- `test_avg/mae_surf_p_swa` (SWA ckpt) = **67.1099**
- W&B run: `pnmb6bd5`
- Per-split (val | test): single=89.13|77.64, camber_rc=88.15|77.27, camber_cruise=54.46|47.31, re_rand=73.07|66.21

### Merge history
| Date | PR | Title | val_swa | test_swa | Δ val |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.50 | 112.49 | first |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands | 116.34 | 107.33 | −4.2% |
| 2026-05-15 23:20 | #3215 | SmoothL1 β=0.05 | 90.60 | 83.00 | −22.1% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning | 79.90 | 69.33 | −11.8% |
| 2026-05-16 08:00 | #3669 | SWA on FiLM-Re | 76.61 | 68.20 | −4.1% |
| **2026-05-16 11:28** | **#3806** | **Surface-Dedicated Refinement MLP** | **76.20** | **67.11** | **−0.5%** |

## Cycle 23 actions (just executed)

**Merged:** PR #3806 (fern, surface refinement MLP) — new baseline val=76.2033, test=67.1099. Gain concentrated on test_geom_camber_rc (−3.18, −4%).

**Closed:** PR #3816 (frieren, LayerDrop) — both p=0.05 arms worse than new baseline (val +0.88/+0.93, test +1.68/+0.92). Root cause: 5-layer Transolver has no redundant depth for LayerDrop to regularize away. Smoke test at p=0.10 diverged immediately.

**Assigned (2 new PRs):**

| Student | PR | Hypothesis |
|---|---|---|
| fern | #3917 | EMA decay=0.99 on surface-refinement baseline (compound of two best wins) |
| frieren | #3920 | EMA decay sweep {0.999, 0.95, 0.90} on surface-refinement baseline (complementary to fern's 0.99) |

Together, #3917 (fern) + #3920 (frieren) map the full EMA decay response curve on the new compound baseline, covering {0.90, 0.95, 0.99, 0.999}. If 0.99 works on the new baseline as it did on the old (val −6.04), we expect val ~70 test ~61 as a new floor.

## Active Research Directions — Cycle 23 in-flight

| PR | Student | Idea / Mechanism | Started |
|---|---|---|---|
| #3813 | thorfinn | Per-Sample Re-Scaled Normalization (Idea 1) | cycle 19 |
| #3828 | alphonse | Low-rank hypernetwork Re conditioning (Idea 4) | cycle 20 |
| #3831 | askeladd | Bernoulli consistency aux loss (Idea 8) | cycle 20 |
| #3856 | nezuko | Multiscale BG subsampling probe (Idea 5, staged) | cycle 21 |
| **#3917** | **fern** | **EMA decay=0.99 compound on surface-refinement baseline** | **cycle 23 NEW** |
| **#3920** | **frieren** | **EMA decay sweep {0.999, 0.95, 0.90} on surface-refinement baseline** | **cycle 23 NEW** |

Awaiting student submission (code already trained, just needs push + SENPAI-RESULT):

| PR | Student | Status |
|---|---|---|
| #3799 | edward | Major winner (val=70.57, test=61.98) pending push + SENPAI-RESULT |
| #3803 | tanjiro | Sweep incomplete; awaiting pivot to swa_start ∈ {8,9,10} |

## Plateau-break strategy (updated)

| Idea | Status |
|---|---|
| 1. Per-sample Re-scaled loss norm | Running (thorfinn #3813) |
| 2. Residual learning over analytic | Closed (#3820); per-node panel-method deferred |
| 3. Surface-dedicated refinement | **Merged** (#3806) → val=76.20, test=67.11 |
| 4. Hypernetwork Re conditioning | Running (alphonse #3828) |
| 5. Multiscale mesh pooling | Running (staged probe, nezuko #3856) |
| 6. Stochastic depth / LayerDrop | **Closed** (#3816; 5-layer arch has no depth headroom) |
| 7. EMA-weighted averaging (ema_decay=0.99) | MAJOR WIN (#3799) pending submission; compounding (#3917/#3920) |
| 8. Bernoulli consistency loss | Running (askeladd #3831) |

## Goal

Push val < 72, test < 65. **Edward's xuugyx5t (val=70.57, test=61.98) already crosses both thresholds** — once merged it becomes the new baseline. Fern (#3917) and frieren (#3920) testing whether EMA gains compound with surface-refinement further (expected val ~70, test ~61 if both mechanisms stack).

## What has been definitively ruled out (do not retry on this baseline)

- β-tuning of SmoothL1 (β=0.02, 0.01 — both regress in compound)
- OneCycleLR / cosine warmup at 30-min budget
- Depth/width scaling (mlp_ratio=4, larger hidden)
- More Fourier bands (16) — marginal signal, hurts at 13 epochs
- Divergence-free physics aux loss (fragile, competes with FiLM-Re)
- surf_weight off baseline (full curve mapped at {5, 10, 15, 20})
- Geom-slice (standalone and in compound with FiLM-Re)
- Multi-signal FiLM (cond_dim=5 — narrow bottleneck, redundant signal)
- Per-channel surface loss weights (#3198)
- Domain one-hot embedding (#3523)
- Per-sample DC residual baseline (#3820; 3.7% node-level p variance)
- **swa_start=4** (tanjiro #3803; too early — drift dominates averaging)
- **LayerDrop / stochastic whole-layer skip on 5-layer Transolver** (NEW #3816; no depth headroom)
