<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 07:30 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 6 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 6 starting. Baseline still at 65.7375 (tanjiro #3596). Strongest in-flight candidate: edward's EMA d=0.999 (sent back for rebase to T_max=21 stack).**

Key finding from round 5 review: **EMA of weights (d=0.999) on old stack → val=64.5125, test=60.2569** — a −7.6%/−8.5% improvement over the old baseline. The run compared against T_max=50 stack (PR #3427); edward is rebasing to the current T_max=21 stack. If the EMA improvement replicates (expected: likely), the new SOTA will be ~63–64 val.

Critical structural insight from alphonse: **clip=1.0 engages on 100% of steps** → all Lion gradient updates are unit-direction vectors going into momentum. This means clip is not a ceiling but a *per-step direction normalizer*. The β₁ in Lion controls how many past unit-vectors are averaged. Two new hypotheses follow from this: lion-beta-sweep tests β₁ and wd-sweep tests whether the optimizer magnitude lever (wd) is well-calibrated.

The full current stack (6 levers stacked across 5 rounds):
1. **Lion optimizer** (PR #3387): sign-based update
2. **bf16 autocast** (PR #3427): 19 epochs in 30 min
3. **grad-clip(max_norm=1.0)** (PR #3427): 100% engagement — direction normalizer not spike ceiling
4. **eta_min=1e-5** (PR #3427): floor
5. **T_max=21** (PR #3596): cosine fully traverses productive low-LR zone
6. **EMA d=0.999** (edward, pending rebase): strong improvement expected on top of T_max=21

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| alphonse | #3590 | `lion-clip-sweep` | Clip=off arm (add Arm 4 after Lion+always-clip invariance finding) | WIP — clip=2.0 running, clip=off directed |
| edward | #3640 | `ema-weights` | EMA d=0.999 — sent back for rebase to T_max=21 stack | DRAFT (sent back for rebase) |
| frieren | #3675 | `lion-lr-sweep` | LR {2e-4, 3e-4} — lr2e4 val=65.30 (beats SOTA), lr3e4=67.32 | WIP — awaiting terminal |
| tanjiro | #3713 | `eta-min-sweep` | eta_min {2e-5, 3e-5} — arm1 val=67.16 (worse), arm2 running | WIP — awaiting terminal |
| nezuko | #3745 | `h160-tmax-calibrated` | H=160 with T_max=16 (calibrated to epoch budget) | NEW |
| fern | #3747 | `vol-loss-p-weight` | vol_loss p-weight {2.0, 1.5} | NEW |
| askeladd | #3749 | `lion-beta-sweep` | Lion β₁ {0.8, 0.95} on T_max=21 stack | NEW |
| thorfinn | #3751 | `wd-sweep` | Weight decay {1e-3, 5e-2} on T_max=21 stack | NEW |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **65.7375**
- `test_avg_nansafe/mae_surf_p` = **61.7003**
- W&B run: `tew7xthq` (tanjiro, group `lion-tmax-newbase`, PR #3596)
- Stack: Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21**
- VRAM: 33 GB / 96 GB. Best epoch = **18** (ep19 mildly regresses).

## Key research signals — round 5 review

### Round-5 final results (this boot)

| PR | Student | Hypothesis | Result | Decision |
|---|---|---|---|---|
| **#3640** | **edward** | **EMA d=0.999** | **val=64.5125 (−1.22 vs SOTA on old stack)** | **Sent back for rebase to T_max=21 stack** |
| #3675 | frieren | LR sweep lr=2e-4 | val=65.2991 (−0.44 vs SOTA); lr=3e-4=67.32 | Awaiting terminal |
| #3598 | fern | surf_loss p_weight {2×, 4×} | val=77.18/79.94 monotonic regression | CLOSED |
| #3592 | nezuko | Deeper model (L7=78.31, H160=70.49) | T_max budget mismatch | CLOSED, H160 reassigned |
| #3641 | askeladd | Batch size bs=8 | val=86.34 (+20.60) — fewer epochs dominate | CLOSED |
| #3541 | thorfinn | Lion lr/wd sweep (paper range) | val=98.95 — sweep collapsed to arm1 | CLOSED |

### Critical finding: alphonse's Lion+clip invariance (#3590)

clip=1.0 engages on 100% of steps with pre-clip grad-norm median=15.44. Since clip
rescales each gradient to unit direction, any positive clip threshold produces **identical
training trajectories** under Lion (sign invariance). The clip threshold is a no-op
within the always-clipping regime. Directed alphonse to run clip=off arm to probe whether
removing clip entirely changes the trajectory.

### In-flight experiments (training or pending)

| Priority | PR | Student | Hypothesis | Status |
|---|---|---|---|---|
| HIGH | #3640 | edward | EMA d=0.999 on T_max=21 stack | Awaiting rebase + rerun |
| HIGH | #3675 | frieren | LR sweep lr=2e-4 (−0.44 vs SOTA) | Awaiting terminal |
| MED | #3713 | tanjiro | eta_min sweep (arm1 worse, arm2 running) | Awaiting terminal |
| MED | #3590 | alphonse | clip=off arm (probe Lion without clipping) | Awaiting clip=2 + clip=off |
| MED | #3745 | nezuko | H=160 + T_max=16 calibrated | NEW — training |
| MED | #3747 | fern | vol_loss p-weight {1.5, 2.0} | NEW — training |
| LOW | #3749 | askeladd | Lion β₁ {0.8, 0.95} | NEW — training |
| LOW | #3751 | thorfinn | WD sweep {1e-3, 5e-2} | NEW — training |

## Round-6 directions (active)

1. **EMA + T_max=21 stack** (edward rebasing): EMA d=0.999 on old stack gave −7.6%/−8.5%. On T_max=21, the low-LR refinement region (ep 16-18) is where EMA earns its keep most. Expected to be NEW SOTA.
2. **H=160 + calibrated T_max** (nezuko): Prior H=160 was T_max-misaligned; retest with T_max=16.
3. **LR=2e-4 + T_max interaction** (frieren): If lr=2e-4 holds, optimal T_max will shift (higher LR decays faster, T_max ~25-28 may be needed).
4. **Lion beta sweep** (askeladd): β₁ controls direction averaging; with 100% clip, this is purely temporal averaging of unit-vectors.
5. **Weight decay sweep** (thorfinn): wd=1e-2 never re-tuned on SOTA stack. OOD splits are diagnostic.
6. **EMA finer sweep** (round 7): {0.995, 0.997, 0.9995} around d=0.999 after edward's rebase confirms.

## Eliminated approaches

| Approach | Best val | Decision |
|---|---:|---|
| surf_loss p-weight 2×/4× (fern #3598) | 77.18 | Closed — monotonic regression all channels |
| Deeper model L=7 (nezuko #3592 arm1) | 78.31 | Dead end |
| Batch size bs=8 (askeladd #3641) | 86.34 | Closed — fewer epochs dominate |
| Lion lr=3e-5/wd=3e-2 paper range (thorfinn #3541) | 98.95 | Closed — obsolete |
| LR warmup on new Lion+clip stack (frieren #3604) | 76.12 | Closed |
| Surf weight sw=5/20 (nezuko) | 111.08 | Closed |
| NACA Fourier features (thorfinn) | 115.45 | Closed |
| LR warmup on old Lion (frieren) | 100.80 | Closed |
| Lion+bf16 without clip (fern) | 89.53 | Closed |

## Known infra bugs (unchanged)

### 1. `data/scoring.py` NaN propagation
`test_geom_camber_cruise/000020.pt` has 761 `-inf` entries in `y[:, 2]`. Workaround: all PRs log `test_avg_nansafe/mae_surf_p`.

### 2. PhysicsAttention inf at `slice_num=128`
Model produces `±inf` at `slice_num=128`. Future slice_num arms must pair with stability guard.

### 3. senpai-pr-guard.py code-fence bug
Guard picks up template SENPAI-RESULT markers inside code fences as invalid JSON. Advisor-side fix pending.

## Operational notes

- All new assignments use fixed seed (torch.manual_seed(42))
- All PRs use `--wandb_group <slug>` for W&B grouping
- Hard limits: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50` per run
- All PRs log `test_avg_nansafe/mae_surf_p` (cruise NaN workaround)
