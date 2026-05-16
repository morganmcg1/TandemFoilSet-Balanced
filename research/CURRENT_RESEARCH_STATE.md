<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 01:45 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 4 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 4 running. New SOTA: alphonse's Lion+bf16+clip+floor stack (PR #3427, merged 01:30 UTC).**

Round-4 SOTA: **alphonse's lion-bf16-clip-floor (PR #3427)**, val_avg/mae_surf_p=**69.86**, test_nansafe=**65.88** — a −25.75% improvement over the Lion-only baseline (94.08) and −35.0% over the Huber baseline (107.46).

The three stacked levers in the merged baseline:
1. **Lion optimizer** (PR #3387): sign-based update rule, orthogonal to Huber's loss-level capping
2. **bf16 autocast** (PR #3427): 19 epochs in 30 min (vs 14 fp32) — 5 extra optimizer steps
3. **grad-clip(max_norm=1.0)** (PR #3427): engaged at 99.7% of steps — per-step Lion momentum normalizer
4. **eta_min=1e-5** (PR #3427): standby floor, not yet engaging at T_max=50 + 19 epochs

Key signals:
- Val curve **still descending at epoch 19** (best=final epoch) — material headroom remains
- VRAM: **33 GB / 96 GB** — 63 GB unused headroom for bigger model
- LR at ep19: **7.16e-5** — only 28% into cosine decay; eta_min floor not engaged

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| alphonse | #3590 | `lion-clip-sweep` | Clip values {0.25, 0.5, 2.0} vs current clip=1.0 | **WIP (just assigned)** |
| nezuko | #3592 | `deeper-model` | n_layers=7 and n_hidden=160 on new stack (63 GB headroom) | **WIP (just assigned)** |
| tanjiro | #3596 | `lion-tmax-newbase` | T_max=19 to engage eta_min=1e-5 within bf16 budget | **WIP (just assigned)** |
| fern | #3598 | `p-weight-surf-loss` | Per-channel p weighting in surf_loss (2× and 4×) | **WIP (just assigned)** |
| frieren | #3604 | `lion-warmup-newbase` | Warmup on new stacked baseline — does clip make warmup redundant? | **WIP (just assigned)** |
| edward | #3518 | `lion-tmax14` | T_max=14 on old Lion baseline (arm 3 running) | WIP — awaiting terminal |
| thorfinn | #3541 | `lion-lr-wd-sweep` | Lion lr/wd sweep on old Lion baseline (arm 2 running) | WIP — awaiting terminal |
| askeladd | #3385 | `warmup-cosine-stacked` | warmup2+clip50 on new Lion stack (rebase needed) | WIP — awaiting rebase |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **69.8562**
- `test_avg_nansafe/mae_surf_p` = **65.8812**
- W&B run: `f6lnbssy` (alphonse, group `bf16-stable`, PR #3427)
- Stack: Lion + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5
- VRAM: 33 GB / 96 GB. Best epoch = final epoch 19. Val still descending at timeout.

## Key research signals — round 4

### Mechanism insight: clip=1.0 as Lion momentum normalizer

99.7% of steps are clipped at clip=1.0 on Lion, median pre-clip grad norm 16.6. This means clip is not a "spike ceiling" but a **per-step normalizer for Lion's momentum input**. This is the dominant active lever in the merged stack — more so than bf16 throughput:

- **Lion+bf16 alone** (fern #3481): val=89.53 (before closure)
- **Lion+bf16+clip+floor** (alphonse #3427): val=69.86

The clip lever accounts for ~20 additional points beyond bf16 alone.

### Eliminated approaches (round 4)

| Approach | Best val | Decision |
|---|---:|---|
| Huber δ sweep (tanjiro, 3 arms) | 108.37 (δ=1.0) | Closed — optimizer dominates δ tuning |
| Surf weight sw=5/20 (nezuko) | 111.08 (sw=5) | Closed — sw=10 is near-optimal |
| NACA Fourier features (thorfinn) | 115.45 | Closed — features need raw not standardized inputs |
| LR warmup on old Lion (frieren) | 100.80 (1-epoch) | Closed — warmup doesn't help bare Lion |
| Lion+bf16 without clip (fern) | 89.53 | Closed — superseded by merged clip stack |
| Surface-only Huber (frieren) | 103.20 | Closed — hurts more than it helps |

### In-flight experiments prioritized

| Priority | PR | Student | Hypothesis | Expected val |
|---|---|---|---|---|
| HIGH | #3590 | alphonse | Clip sweep (0.25, 0.5, 2.0) — find optimal clip | 63–70 |
| HIGH | #3592 | nezuko | Deeper model (L=7, n_hidden=160) — use 63 GB headroom | 60–67 |
| HIGH | #3596 | tanjiro | T_max=19 fix — engage eta_min floor | 64–68 |
| MED | #3598 | fern | Per-channel p weighting (2×, 4×) | 64–69 |
| MED | #3604 | frieren | Warmup on new stack — clip vs warmup redundancy test | 67–70 |
| MED | #3518 | edward | T_max=14 on old Lion (awaiting arm 3) | 90–95 diagnostic |
| MED | #3541 | thorfinn | Lion lr/wd sweep on old Lion (awaiting arm 2) | 95–100 diagnostic |
| LOW | #3385 | askeladd | warmup2+clip50 on new stack (rebase needed) | 67–72 |

### Round-5 directions (speculative, contingent on in-flight results)

1. **Full stack**: optimal clip + T_max=19 + larger model (if all three beat baseline)
2. **Bigger model**: if nezuko's deeper model helps, L=9 or n_hidden=192 next
3. **Clip+LR interaction**: the optimal clip threshold likely depends on Lion's LR — if clip sweep shows a clear winner, the clip/lr ratio becomes a tunable lever
4. **Longer schedule**: T_max and bf16 both hit the timeout constraint; if we can get >30 min (two chained runs, or a 60-min timeout), the descending val curve suggests >10 more points available

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]`. NaN poisons surface MAE. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`.

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` at `slice_num=128`. Future slice_num arms must pair with stability guard.

### 3. senpai-pr-guard.py code-fence bug
Guard picks up template SENPAI-RESULT markers inside code fences as invalid JSON, blocking mark_ready_for_review. Both tanjiro and nezuko independently identified and suggested a fix. Advisor-side fix pending (add `in_fence` tracking to `result_markers()`).

## Operational notes

- All new assignments use fixed seed (torch.manual_seed(42)) — mandated after tanjiro's 108→141 spread finding
- All PRs use `--wandb_group <slug>` for W&B grouping
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run
- All PRs log `test_avg_nansafe/mae_surf_p` (cruise NaN workaround)
</content>
</invoke>