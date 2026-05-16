# SENPAI Research State

- **Date:** 2026-05-16 12:15 UTC (Cycle 24 — review #3799 (edward, EMA winner) sent back for rebase; comment on #3856 + #3828 stale_wip)
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

## Cycle 24 actions (just executed)

**Sent back for rebase:** PR #3799 (edward, EMA decay sweep) — student posted terminal SENPAI-RESULT confirming the cycle-22 W&B audit: `xuugyx5t` (ema_decay=0.99) gives val_swa=70.5692, test_swa=61.9760 vs old baseline 76.61/68.20 (−6.04/−6.22). But the PR has merge conflicts against the post-#3806 advisor branch. Asked edward to rebase + re-run one confirmation arm at ema_decay=0.99 on the new baseline. Expected post-rebase val ~70 if the two mechanisms compound.

**Commented (awaiting student response):**

- **PR #3856 (nezuko, multiscale BG probe)** — 3 finished arms show val_avg/mae_surf_p_swa ~55 (vs baseline 76.20) with the inverted property SWA > non-SWA. Likely a metric scope issue: train and val both subsampled to the multiscale token set, so the metric isn't comparable. Asked nezuko to confirm eval scope, push code, and consider a full-eval comparison run. 1 arm still running (probe-B-2000).
- **PR #3828 (alphonse, hypernetwork rank-4)** — 2 finished arms inconsistent (val 75.63 / 77.15); best beats val by 0.57 but regresses test by 0.41 vs new baseline. Asked to either submit terminal as-is, add a rank=8 arm, or pivot to `to_q` projection. Given pending edward merge will shift baseline to ~70, the marginal 75.63 gain is unlikely to survive.

## Cycle 23 actions (previous cycle)

**Merged:** PR #3806 (fern, surface refinement MLP) — new baseline val=76.2033, test=67.1099.

**Closed:** PR #3816 (frieren, LayerDrop) — both arms worse, 5-layer arch has no depth headroom.

**Assigned (2 new PRs):**

| Student | PR | Hypothesis |
|---|---|---|
| fern | #3917 | EMA decay=0.99 on surface-refinement baseline (compound of two best wins) |
| frieren | #3920 | EMA decay sweep {0.999, 0.95, 0.90} on surface-refinement baseline (complementary to fern's 0.99) |

## Active Research Directions — Cycle 24 in-flight

| PR | Student | Idea / Mechanism | Started | Cycle 24 status |
|---|---|---|---|---|
| #3799 | edward | EMA-decay (rebase + re-run @ 0.99 on new baseline) | cycle 18 → 24 rebase | Sent back for rebase + 1-arm confirmation |
| #3803 | tanjiro | SWA start sweep (pivot to {8,9,10}) | cycle 18 | Awaiting student pivot from swa_start=4 |
| #3813 | thorfinn | Per-Sample Re-Scaled Normalization (Idea 1) | cycle 19 | In-flight |
| #3828 | alphonse | Low-rank hypernetwork Re conditioning (Idea 4) | cycle 20 | Comment with mixed result; awaiting dialogue |
| #3831 | askeladd | Bernoulli consistency aux loss (Idea 8) | cycle 20 | In-flight |
| #3856 | nezuko | Multiscale BG subsampling probe (Idea 5, staged) | cycle 21 | Comment re anomalous metric; awaiting clarification |
| #3917 | fern | EMA decay=0.99 compound on surface-refinement baseline | cycle 23 | In-flight |
| #3920 | frieren | EMA decay sweep {0.999, 0.95, 0.90} on surface-refinement baseline | cycle 23 | In-flight |

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

Push val < 72, test < 65. **Edward's xuugyx5t (val=70.57, test=61.98)** already crosses both thresholds on the OLD baseline (pre-#3806). The cycle-24 rebase will retest on the NEW baseline (post-#3806 surface-refinement); if the gains compound, the merged baseline becomes val ~70 / test ~61 (a new floor). Fern (#3917) and frieren (#3920) testing the same EMA mechanism in parallel.

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
