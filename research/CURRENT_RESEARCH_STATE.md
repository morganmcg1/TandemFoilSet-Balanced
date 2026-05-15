<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15 23:10
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 4 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 4 running. New baseline: Lion optimizer on Huber (PR #3387 merged 21:45 UTC).**

Round-4 SOTA: **fern's Lion-stacked (PR #3387)**, val_avg/mae_surf_p=**94.08**, test_nansafe=**88.94** — a −12.4% improvement over the Huber baseline (107.46). Lion's sign-based update rule compounds cleanly with Huber's per-sample gradient cap: two orthogonal mechanisms both limiting optimizer sensitivity to large gradients. Val curve was still descending at timeout (epoch 14), indicating material headroom remaining.

Round-3 winner now superseded: ~~frieren's Huber loss (107.46, PR #3248)~~ — now the foundation layer that Lion stacks on.

**Strongest stack in flight:**
- **alphonse `bf16-stable`** (#3427): **Terminal confirmed val=92.6166, test=87.6987** — beats Lion baseline by −1.46. BUT branch needs rebase onto Lion. Sent back to rebase + verify Lion+bf16+grad-clip+eta_min stack. Expect rebased rerun val ~84–92.
- **frieren `lion-warmup`** (#3515): 2-epoch warmup on Lion baseline. Targeting val 89–92.
- **edward `lion-tmax14`** (#3518): T_max=14 fix on Lion baseline. Targeting val 88–92.
- **fern `lion-bf16-stacked`** (#3481): bf16 on Lion baseline — extends 14→19 epochs. Targeting val 84–91.

**Current baseline (BASELINE.md):**
- `val_avg/mae_surf_p` = **94.0803** (NEW)
- `test_avg_nansafe/mae_surf_p` = **88.9362** (NEW, via eval_nansafe.py)
- W&B run: `f9w6yzoq` (fern, group `lion-stacked`, PR #3387, merged 21:45 UTC)

## Round 4 PRs — current status (23:10 UTC)

| PR | Student | Hypothesis | Slug | Status |
|---|---|---|---|---|
| #3387 | fern | Lion optimizer on Huber baseline | `lion-stacked` | **MERGED 21:45** — new baseline 94.08 |
| #3394 | frieren | Surface-only Huber + delta tuning | `huber-surface-only` | **CLOSED** val=103.20, no rebase — reassigned |
| #3403 | edward | Cosine T_max fix | `lr-tmax-fix` | **CLOSED** val=103.30, T_max=14 diagnostic confirmed — reassigned |
| #3427 | alphonse | bf16 + grad-clip + eta_min on Huber | `bf16-stable` | **WIP (rebase pending)** — val=92.62 confirmed, sent back to rebase onto Lion |
| #3481 | fern | Lion + bf16 (timeout extension) | `lion-bf16-stacked` | WIP — just assigned |
| #3515 | frieren | Lion + 2-epoch warmup | `lion-warmup` | WIP — just assigned |
| #3518 | edward | Lion + T_max=14 | `lion-tmax14` | WIP — just assigned |
| #3385 | askeladd | Warmup+cosine+clip (Huber) | `warmup-cosine-stacked` | WIP (needs_rebase) — nudged to rebase+run warmup2-clip50 on Lion |
| #3389 | nezuko | surf_weight sweep | `surf-weight-sweep` | WIP — awaiting terminal |
| #3391 | thorfinn | NACA Fourier on Huber | `naca-fourier-stacked` | WIP — 6 runs done (best 115.45, regression), nudged for terminal |
| #3392 | tanjiro | Huber delta sweep | `huber-delta-tuning` | WIP — awaiting terminal |

## Key research signals — round 4 update

### Round-4 breakthrough: Lion optimizer

Fern's Lion-stacked (PR #3387, merged 21:45 UTC) gives −12.4% val improvement over Huber baseline. The two levers compound cleanly:
- **Huber**: caps gradient magnitude at the *loss* level (δ=2.0 elbow)
- **Lion**: caps gradient magnitude at the *optimizer* level (sign-based update rule)

These are truly orthogonal — Huber limits what gets backpropped; Lion limits how the optimizer acts on it. Combined effect far exceeded prediction (actual −12.4% vs predicted −3 to −8%).

**Headroom signal**: Lion val curve was still descending at epoch 14/14 (timeout). The slope at cutoff was −2.9/epoch. With more epochs (bf16 throughput would give ~19 epochs) or a better LR schedule (edward's T_max fix), the floor hasn't been reached.

### Round-4 confirmed wins and elimination

**New SOTA stack (confirmed terminal):**
- **Lion+Huber (fern, 94.08)**: merged baseline. −12.4% over Huber.
- **bf16+grad-clip+eta_min+Huber (alphonse, 92.62)**: confirmed terminal, beats Lion. Needs rebase to validate Lion+bf16+clip+floor stack.

**Pending Lion-stacking arms (highest priority):**
- **fern #3481 `lion-bf16-stacked`**: Lion+bf16, predict 84–91.
- **alphonse #3427 rebased**: Lion+bf16+grad-clip+eta_min, predict 82–91.
- **frieren #3515 `lion-warmup`**: Lion+warmup, predict 89–92.
- **edward #3518 `lion-tmax14`**: Lion+T_max=14, predict 88–92.

**Confirmed diagnostics (valuable but non-winning):**
- T_max=14 gives −3.9% on AdamW+Huber baseline (edward, 103.30). Now testing on Lion.
- bf16 throughput gives 19 vs 14 epochs (alphonse, 92.62). Now rebasing onto Lion.

**Eliminated on Lion baseline:**
- Huber δ sweep (tanjiro, 111.34 best): optimizer dominates over δ tuning.
- Surface-only Huber (frieren, 103.20 best): vol-MSE doesn't help, surface-only hurts.
- NACA Fourier (thorfinn, 115.45 best): geometry encoding not competitive.
- Warmup+clip(1.0) (askeladd, 107.61 on old Huber base): clip=1.0 too aggressive.
- Warmup+clip(50) still pending (askeladd, needs rebase onto Lion).

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]` (interior pressure). `NaN * 0 = NaN` in IEEE-754 poisons surface MAE. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`. Identified by alphonse (#3243).

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` predictions at `slice_num=128` (reproducible). Not currently fixed. Future `slice_num` arms must pair with stability guard. Identified by fern (#3247).

### 3. Late-training Huber divergence
Both alphonse `tup20e60` (best=111.6, final=171.4) and frieren `1walszqd` (best=121.85, final=175.16) show late-epoch divergence under Huber + cosine. The primary run `mp8s8okf` was stable — likely a gradient/momentum accumulation issue in the LR tail. Grad-clip (askeladd's lever) should address this when stacked.

## Active PRs summary (21:50 UTC)

| PR | Student | Status | Next action |
|---|---|---|---|
| #3387 | fern | **MERGED** 21:45 UTC | New baseline 94.08. Reassigned to `lion-bf16-stacked` (PR #3481) |
| #3481 | fern | WIP (assigned 21:55 UTC) | lion-bf16-stacked: add bf16 to extend val descent past timeout |
| #3385 | askeladd | WIP — `warmup2-clip50` in flight | await terminal |
| #3389 | nezuko | WIP — awaiting terminal | 2 arms (sw5, sw20) in flight |
| #3391 | thorfinn | WIP — awaiting terminal | best 130.99, regression vs Lion baseline |
| #3392 | tanjiro | WIP — awaiting terminal | best 111.34 (δ=1.0), only arm run so far |
| #3394 | frieren | WIP — awaiting terminal | best 103.20 (δ=1.0 surf-only), vs Lion 94.08 |
| #3403 | edward | WIP — awaiting terminal (nudged 21:50 UTC) | val 103.30 (old baseline basis), T_max=12 arm pending |
| #3427 | alphonse | WIP — awaiting terminal (nudged 21:50 UTC) | val 92.62 potential SOTA vs new 94.08 baseline |

## Operational notes

- All round-4 assignments use `--wandb_group <slug>` for W&B grouping.
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run.
- All PRs remind students to log nansafe test variants (cruise test data bug).
- Next researcher-agent run will generate round-5 ideas (launched async this invocation).
