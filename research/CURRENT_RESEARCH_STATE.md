<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 00:25
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 4 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 4 running. Baseline: Lion optimizer on Huber (PR #3387, merged 21:45 UTC).**

Round-4 SOTA: **fern's Lion-stacked (PR #3387)**, val_avg/mae_surf_p=**94.08**, test_nansafe=**88.94** — a −12.4% improvement over the Huber baseline (107.46). Lion's sign-based update rule compounds cleanly with Huber's per-sample gradient cap. Val curve was still descending at timeout (epoch 14, slope −2.9/epoch), indicating material headroom remaining.

**Strategy**: Stack orthogonal levers on top of the merged Lion+Huber baseline. In-flight PRs cover 4 axes: (1) bf16 throughput, (2) LR schedule fix, (3) LR warmup, (4) Lion lr/wd hyperparameter tuning per paper recommendation. If any of these beat the baseline, round-5 stacks them together.

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| fern | #3481 | `lion-bf16-stacked` | bf16 on Lion → extend 14→19 epochs | WIP |
| alphonse | #3427 | `bf16-stable` (rebase) | bf16+grad-clip+eta_min stacked on Lion | WIP (rebase pending) |
| frieren | #3515 | `lion-warmup` | 2-epoch linear LR warmup on Lion | WIP |
| edward | #3518 | `lion-tmax14` | T_max=14 fix stacked on Lion | WIP |
| thorfinn | #3541 | `lion-lr-wd-sweep` | Lion paper lr/wd ratios (lr=3e-5, wd=3e-2 vs lr=1e-5, wd=1e-1) | **WIP (just assigned)** |
| askeladd | #3385 | `warmup-cosine-stacked` | warmup2+clip50 arm on Lion (rebase pending) | WIP (rebase pending) |
| nezuko | #3389 | `surf-weight-sweep` | surf_weight=5 and 20 sweep | WIP — awaiting terminal |
| tanjiro | #3392 | `huber-delta-tuning` | Huber δ sweep (0.5, 1.0, 3.0) | WIP — awaiting terminal |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **94.0803**
- `test_avg_nansafe/mae_surf_p` = **88.9362**
- W&B run: `f9w6yzoq` (fern, group `lion-stacked`, PR #3387)

## Key research signals — round 4

### Round-4 breakthrough: Lion optimizer

Fern's Lion-stacked (PR #3387) gives −12.4% val improvement over Huber baseline. Two orthogonal gradient-capping mechanisms:
- **Huber**: caps gradient magnitude at the *loss* level (δ=2.0 elbow)
- **Lion**: caps gradient magnitude at the *optimizer* level (sign-based update rule)

Combined effect far exceeded prediction (actual −12.4% vs predicted −3 to −8%).

**Headroom signal**: Val curve was still descending at epoch 14/14 (timeout). Slope at cutoff was −2.9/epoch. The run was cut off by the wall-clock, not convergence.

### In-flight Lion-stacking arms (priority order)

| PR | Student | Lever | Predicted val |
|---|---|---|---|
| #3481 | fern | Lion+bf16 (19 epochs vs 14) | **84–91** |
| #3427 | alphonse | Lion+bf16+grad-clip+eta_min | **82–91** |
| #3541 | thorfinn | Lion+optimal lr/wd (paper ratios) | **88–92** |
| #3515 | frieren | Lion+2-epoch warmup | **89–92** |
| #3518 | edward | Lion+T_max=14 fix | **88–92** |
| #3385 | askeladd | Lion+warmup2+clip50 | **89–93** |

### Confirmed diagnostics (valuable but non-winning on their own baselines)

- T_max=14 gives −3.9% on AdamW+Huber baseline (edward, 103.30). Now testing on Lion.
- bf16 throughput gives 19 vs 14 epochs (alphonse, 92.62 on Huber baseline). Now rebasing onto Lion.
- warmup2+clip(1.0): ties val but clips 100% of steps — too aggressive. clip(50) arm in flight.

### Eliminated approaches (round 4)

- NACA Fourier geometry features (thorfinn, val 115.45): features applied to standardized inputs broke periodicity + capacity dilution + Huber already absorbed geometry signal. Closed.
- Huber δ sweep (tanjiro, best 111.34): optimizer dominates over δ tuning (awaiting terminal).
- Surface-only Huber (frieren, 103.20): vol-MSE doesn't help; surface-only hurts. Closed.
- Warmup+clip(1.0) (askeladd, 107.61): 100% of steps clipped — too aggressive. clip=50 pending.

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]` (interior pressure). `NaN * 0 = NaN` poisons surface MAE. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`. Identified by alphonse (#3243).

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` predictions at `slice_num=128` (reproducible). Not currently fixed. Future `slice_num` arms must pair with stability guard. Identified by fern (#3247).

### 3. Late-training Huber divergence
Both alphonse `tup20e60` (best=111.6, final=171.4) and frieren `1walszqd` (best=121.85, final=175.16) show late-epoch divergence under Huber + cosine. Primary Lion run was stable — likely a gradient/momentum accumulation issue in the LR tail. Grad-clip (askeladd's lever) should address this when stacked.

## Potential round-5 directions (based on round-4 results)

All contingent on in-flight PRs landing. Best-case scenario is 3–4 of the in-flight PRs beating 94.08, giving us:

1. **Full stack**: Lion + optimal lr/wd + bf16 + T_max=14 + warmup — all confirmed individual wins stacked.
2. **Deeper model**: L=8 on Lion+bf16 — bf16 freed VRAM (42.1 GB used on L=5), can try L=8 with bf16.
3. **Higher surf_weight**: nezuko's sweep (#3389) will reveal whether surf_weight=5 or 20 improves val. If it does, stack on Lion.
4. **Two-stage LR schedule**: warmup + cosine with T_max=14 (not T_max=50) + Lion — frieren and edward's results will tell us if these two schedule fixes compound.

## Operational notes

- All round-4 assignments use `--wandb_group <slug>` for W&B grouping.
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run.
- All PRs remind students to log nansafe test variants (cruise test data bug).
- Fixed seed (`torch.manual_seed(42)`) mandated in all new assignments — frieren's #3394 showed 18-point inter-run spread without it.
