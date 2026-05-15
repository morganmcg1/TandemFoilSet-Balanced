<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-15 21:50
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 4 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 4 running. New baseline: Lion optimizer on Huber (PR #3387 merged 21:45 UTC).**

Round-4 SOTA: **fern's Lion-stacked (PR #3387)**, val_avg/mae_surf_p=**94.08**, test_nansafe=**88.94** — a −12.4% improvement over the Huber baseline (107.46). Lion's sign-based update rule compounds cleanly with Huber's per-sample gradient cap: two orthogonal mechanisms both limiting optimizer sensitivity to large gradients. Val curve was still descending at timeout (epoch 14), indicating material headroom remaining.

Round-3 winner now superseded: ~~frieren's Huber loss (107.46, PR #3248)~~ — now the foundation layer that Lion stacks on.

**Two potentially stronger results in flight:**
- **alphonse `bf16-stable`** (#3427): W&B shows `to8x5txt` val=92.62, test=87.70 — would be **new SOTA (−1.46 below Lion baseline)** if terminal confirms. Nudged for terminal.
- **edward `lr-tmax-fix`** (#3403): W&B shows val=103.30 (old Huber baseline basis), test=98.64 — didn't beat the new Lion baseline, but the T_max diagnostic has high round-5 strategic value. Nudged for terminal.

**Current baseline (BASELINE.md):**
- `val_avg/mae_surf_p` = **94.0803** (NEW)
- `test_avg_nansafe/mae_surf_p` = **88.9362** (NEW, via eval_nansafe.py)
- W&B run: `f9w6yzoq` (fern, group `lion-stacked`, PR #3387, merged 21:45 UTC)

## Round 4 PRs (7 WIP, 1 merged)

| PR | Student | Hypothesis | Slug | Status |
|---|---|---|---|---|
| #3387 | fern | Lion optimizer on Huber baseline | `lion-stacked` | **MERGED 21:45 UTC** → new baseline 94.08 |
| #3385 | askeladd | Warmup+cosine+grad-clip on Huber baseline | `warmup-cosine-stacked` | WIP — `warmup2-clip50` arm in flight |
| #3389 | nezuko | surf_weight sweep (5, 20) | `surf-weight-sweep` | WIP — awaiting terminal |
| #3391 | thorfinn | NACA Fourier features on Huber baseline | `naca-fourier-stacked` | WIP — awaiting terminal |
| #3392 | tanjiro | Huber delta sweep (0.5, 1.0, 3.0) | `huber-delta-tuning` | WIP — awaiting terminal |
| #3394 | frieren | Surface-only Huber + delta tuning | `huber-surface-only` | WIP — awaiting terminal |
| #3403 | edward | Cosine T_max fix (round-5 idea 4) | `lr-tmax-fix` | WIP — awaiting terminal (val 103.30 observed) |
| #3427 | alphonse | bf16 + grad-clip + LR floor on Lion baseline | `bf16-stable` | WIP — awaiting terminal (val 92.62 potential SOTA!) |

## Key research signals — round 4 update

### Round-4 breakthrough: Lion optimizer

Fern's Lion-stacked (PR #3387, merged 21:45 UTC) gives −12.4% val improvement over Huber baseline. The two levers compound cleanly:
- **Huber**: caps gradient magnitude at the *loss* level (δ=2.0 elbow)
- **Lion**: caps gradient magnitude at the *optimizer* level (sign-based update rule)

These are truly orthogonal — Huber limits what gets backpropped; Lion limits how the optimizer acts on it. Combined effect far exceeded prediction (actual −12.4% vs predicted −3 to −8%).

**Headroom signal**: Lion val curve was still descending at epoch 14/14 (timeout). The slope at cutoff was −2.9/epoch. With more epochs (bf16 throughput would give ~19 epochs) or a better LR schedule (edward's T_max fix), the floor hasn't been reached.

### Next convergence targets

Based on what's in flight:
- **alphonse `bf16-stable` (val 92.62 mid-flight)**: bf16 throughput + grad-clip + eta_min on Huber base. If this holds vs Lion baseline, it may stack with Lion to push further.
- **edward `lr-tmax-fix` (val 103.30)**: Isolated T_max diagnostic. Doesn't beat Lion, but validates that the LR schedule was misconfigured — sets up Lion+T_max-fix stacking.
- **askeladd `warmup2-clip50` (in flight)**: The revised clip ceiling (max_norm=50 vs 1.0) may give a clear val win now that clipping is sparse (only true spikes clipped vs every step).

### Round-4 levers confirmed irrelevant vs new Lion baseline

- **NACA Fourier (thorfinn, 130.99 best)**: geometry features don't provide competitive advantage at this experiment scale.
- **Huber δ sweep (tanjiro, 111.34 best)**: Delta tuning doesn't matter much when the optimizer is the binding constraint.
- **Surface-only Huber (frieren, 103.20 best)**: Positive result on old baseline but doesn't beat new Lion 94.08.
- **surf_weight sweep (nezuko)**: Awaiting terminal.

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
