# SENPAI Research State

- **Date:** 2026-05-16 10:30 UTC (Cycle 22 — W&B-verified 3 stale_wip PRs; 1 major winner + 1 marginal winner pending student submission)
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3669 (edward) — SWA on FiLM-Re** — merged 2026-05-16 08:00

- `val_avg/mae_surf_p_swa` (SWA ckpt) = **76.6091**
- `test_avg/mae_surf_p_swa` (SWA ckpt) = **68.1999**
- W&B run: `dqe95m2e`
- Per-split (val | test): single=87.96|77.57, camber_rc=89.40|80.45, camber_cruise=55.59|47.92, re_rand=73.48|66.86

### Merge history
| Date | PR | Title | val_swa | test_swa | Δ val |
|---|---|---|---|---|---|
| 2026-05-15 17:22 | #3200 | Fourier position encoding (8 bands) | 121.50 | 112.49 | first |
| 2026-05-15 19:28 | #3352 | Learnable Fourier frequency bands | 116.34 | 107.33 | −4.2% |
| 2026-05-15 23:20 | #3215 | SmoothL1 β=0.05 | 90.60 | 83.00 | −22.1% |
| 2026-05-16 03:30 | #3350 | FiLM-Re conditioning | 79.90 | 69.33 | −11.8% |
| **2026-05-16 08:00** | **#3669** | **SWA on FiLM-Re** | **76.61** | **68.20** | **−4.1%** |

## Cycle 22 actions (just executed)

**W&B audit of 3 stale_wip PRs:** GitHub API rate-limit at ~09:39 UTC interrupted polling for edward / fern / tanjiro. All three finished training but never pushed code or posted SENPAI-RESULT. I independently fetched their W&B metrics and commented on each PR with the correct apples-to-apples results vs the SWA baseline.

| PR | Student | Best W&B run | val_swa | test_swa | Status |
|---|---|---|---|---|---|
| #3799 | edward | `xuugyx5t` (ema_decay=0.99) | **70.57** | **61.98** | **MAJOR WINNER** awaiting student submission |
| #3806 | fern | `pnmb6bd5` (surf-refine seed 2) | 76.20 | 67.11 | Marginal winner awaiting student submission |
| #3803 | tanjiro | `wr1yyf4l` (swa_start=4) | 83.23 | 75.04 | swa_start=4 worse; sweep incomplete (no {6,8,10} arms) — asked to pivot |

Edward's ema_decay=0.99 is the **largest single-PR improvement** on this baseline (−6.04 val, −6.22 test). Mechanism: ema_decay=0.99 weights the late cosine-tail epochs much more than uniform SWA (effective decay ≈ 0.9996 over 2250 updates), placing averaged weights closer to the converged low-LR basin.

## Active Research Directions — Cycle 22 in-flight

5 students with WIP experiments still running (cycle 19-21 assignments):

| PR | Student | Idea / Mechanism | Started |
|---|---|---|---|
| #3813 | thorfinn | Per-Sample Re-Scaled Normalization (Idea 1) | cycle 19 |
| #3816 | frieren | Stochastic Depth / LayerDrop sweep (Idea 6) | cycle 19 |
| #3828 | alphonse | Low-rank hypernetwork Re conditioning (Idea 4) | cycle 20 |
| #3831 | askeladd | Bernoulli consistency aux loss (Idea 8) | cycle 20 |
| #3856 | nezuko | Multiscale BG subsampling probe (Idea 5, staged) | cycle 21 |

3 PRs awaiting student close-out / submission (this cycle):

| PR | Student | Status |
|---|---|---|
| #3799 | edward | Winner pending push + SENPAI-RESULT (xuugyx5t, ema_decay=0.99) |
| #3806 | fern | Marginal winner pending push + SENPAI-RESULT (pnmb6bd5) |
| #3803 | tanjiro | Sweep incomplete; asked to pivot to swa_start ∈ {8,9,10} or post terminal |

## Plateau-break strategy (complete coverage of cycle-15 ideas)

| Idea | Status |
|---|---|
| 1. Per-sample Re-scaled loss norm | Running (thorfinn #3813) |
| 2. Residual learning over analytic | Closed (#3820 nezuko); per-node panel-method variant deferred |
| 3. Surface-dedicated refinement | **Marginal win** awaiting submit (fern #3806) |
| 4. Hypernetwork Re conditioning | Running (alphonse #3828) |
| 5. Multiscale mesh pooling | Running (staged probe, nezuko #3856) |
| 6. Stochastic depth / LayerDrop | Running (frieren #3816) |
| 7. Checkpoint weight averaging — variant: EMA decay | **MAJOR WIN** awaiting submit (edward #3799) |
| 8. Bernoulli consistency loss | Running (askeladd #3831) |

Plus: SWA-start sweep (tanjiro #3803) — swa_start=4 underperformed; pivot pending.

## Goal

Push val < 72, test < 65. **Edward's xuugyx5t (val=70.57, test=61.98) crosses both thresholds.** Once merged, this becomes the new baseline and every running experiment will be re-evaluated against it. The remaining 5 in-flight mechanisms (per-sample norm, stochastic depth, hypernetwork, Bernoulli, multiscale) are orthogonal to EMA decay and could compound on top.

## Next steps once cycle-22 results land

1. **Highest priority:** merge edward #3799 (ema_decay=0.99) as soon as student pushes code + posts terminal SENPAI-RESULT + marks ready. Update baseline to val=70.57 / test=61.98.
2. **Second priority:** merge fern #3806 (pnmb6bd5) once student submits — marginal but consistent across two seeds. Re-evaluate against the new edward-merged baseline.
3. **Tanjiro #3803:** decide based on student's response — either run swa_start ∈ {8,9,10} arms or close.
4. **In-flight cycle-19/20/21 PRs (5):** review as they come in. With the baseline shifting downward by ~6 points after edward merges, the bar for "beats baseline" tightens — re-evaluate those experiments vs the new baseline.
5. **If edward merges and no other mechanism stacks:** consider compounding combinations (e.g., ema_decay=0.99 + surface-refinement, ema_decay=0.99 + hypernetwork) in a follow-up cycle.

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
- Per-sample DC residual baseline (#3820, 3.7% node-level variance reduction too small)
- **swa_start=4** (NEW from tanjiro #3803; too early — drift dominates averaging)
