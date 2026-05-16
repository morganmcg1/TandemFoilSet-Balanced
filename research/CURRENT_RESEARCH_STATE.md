<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **As of:** 2026-05-16 04:45 UTC
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r3`
- **Research tag:** `willow-pai2i-24h-r3` (round 4 active)
- **Most recent human research direction:** None received.

## Current focus

**Round 5 running. New SOTA: tanjiro's T_max=21 cosine fix (PR #3596, merged 04:30 UTC).**

Round-5 SOTA: **tanjiro's lion-tmax21 (PR #3596)**, val_avg/mae_surf_p=**65.74**, test_nansafe=**61.70** — a −5.9% val / −6.3% test improvement over the previous SOTA (69.86/65.88).

The full current stack (5 levers stacked across 4 rounds):
1. **Lion optimizer** (PR #3387): sign-based update, orthogonal to loss-level capping
2. **bf16 autocast** (PR #3427): 19 epochs in 30 min (vs 14 fp32) — 5 extra optimizer steps
3. **grad-clip(max_norm=1.0)** (PR #3427): engaged at 99.7% of steps — per-step Lion momentum normalizer
4. **eta_min=1e-5** (PR #3427): standby floor
5. **T_max=21** (PR #3596): cosine schedule traverses lower-LR refinement region (LR→1.2e-5 at ep19 vs 7.2e-5 at T_max=50)

Key signals from new SOTA:
- Best epoch = **18** (epoch 19 mildly regresses — the eta_min floor itself doesn't help)
- VRAM: **33 GB / 96 GB** — 63 GB unused headroom
- LR at ep18 (best): ~1.4e-5 (lower half of cosine arc engaged for first time)
- Val still descending at epoch 18 → there may be headroom from larger model or higher LR

## All students — current assignments

| Student | PR | Slug | Hypothesis | Status |
|---|---|---|---|---|
| alphonse | #3590 | `lion-clip-sweep` | Clip values {0.25, 0.5, 2.0} — is clip=1.0 optimal? | WIP — clip=0.25 done (79.78), clip=0.5 running, clip=2.0 missing |
| nezuko | #3592 | `deeper-model` | n_layers=7 and n_hidden=160 on new stack | WIP — both arms finished (l7=78.31, h160=70.49), re-run running |
| tanjiro | **MERGED #3596** | `lion-tmax-newbase` | T_max=21 → val=65.74 NEW SOTA | **MERGED** |
| fern | #3598 | `p-weight-surf-loss` | Per-channel p weighting (2×, 4×) | WIP — p_weight=2.0 done (77.18 worse), p_weight=4.0 running |
| frieren | #3675 | `lion-lr-sweep` | LR sweep {2e-4, 3e-4} on new T_max=21 SOTA stack | **WIP (just assigned)** |
| edward | #3640 | `ema-weights` | EMA decay {0.999, 0.9999} — descending val curve lever | WIP — in progress |
| askeladd | #3641 | `bs-scaling` | Batch size {8, 12} — utilize 63 GB VRAM headroom | WIP — in progress |
| thorfinn | #3541 | `lion-lr-wd-sweep` | Lion lr/wd sweep on old Lion baseline (diagnostic) | WIP — awaiting terminal |

## Current baseline (BASELINE.md)

- `val_avg/mae_surf_p` = **65.7375**
- `test_avg_nansafe/mae_surf_p` = **61.7003**
- W&B run: `tew7xthq` (tanjiro, group `lion-tmax-newbase`, PR #3596)
- Stack: Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21**
- VRAM: 33 GB / 96 GB. Best epoch = **18** (epoch 19 mildly regresses). Val still descending at ep18.

## Key research signals — round 5

### Round-5 results so far

| Priority | PR | Student | Hypothesis | Result | Decision |
|---|---|---|---|---|---|
| HIGH | **#3596** | **tanjiro** | **T_max=21** | **val=65.74 NEW SOTA** | **MERGED** |
| LOW | #3604 | frieren | Warmup on new stack | val=76.12 (warmup2) — worse | CLOSED |
| HIGH | #3590 | alphonse | Clip sweep (0.25, 0.5, 2.0) | clip=0.25: val=79.78 (in) | Awaiting arms 2+3 |
| HIGH | #3592 | nezuko | Deeper model (L=7, n_hidden=160) | l7=78.31, h160=70.49 (both slightly worse) | Awaiting terminal |
| MED | #3598 | fern | p_weight {2×, 4×} | p_weight=2.0: val=77.18 (worse) | Awaiting arm 2 |

### In-flight experiments (training now)

| Priority | PR | Student | Hypothesis | Expected val |
|---|---|---|---|---|
| HIGH | #3590 | alphonse | Clip sweep (clip=0.5 running, clip=2.0 MISSING) | 63–68 |
| MED | #3675 | frieren | LR sweep {2e-4, 3e-4} on T_max=21 SOTA | 62–67 |
| MED | #3640 | edward | EMA decay {0.999, 0.9999} on SOTA | 63–66 |
| MED | #3641 | askeladd | Batch size {8, 12} on SOTA | 63–67 |
| LOW | #3541 | thorfinn | Lion lr/wd sweep (old baseline, diagnostic) | 90–100 |

## Key research signals — round 4

### Mechanism insight: clip=1.0 as Lion momentum normalizer

99.7% of steps are clipped at clip=1.0 on Lion, median pre-clip grad norm 16.6. This means clip is not a "spike ceiling" but a **per-step normalizer for Lion's momentum input**. This is the dominant active lever in the merged stack — more so than bf16 throughput:

- **Lion+bf16 alone** (fern #3481): val=89.53 (before closure)
- **Lion+bf16+clip+floor** (alphonse #3427): val=69.86

The clip lever accounts for ~20 additional points beyond bf16 alone.

### Eliminated approaches (round 5 additions)

| Approach | Best val | Decision |
|---|---|---|
| LR warmup on new Lion+clip stack (frieren) | 76.12 (warmup2) | Closed — Lion is LR·sign(momentum), warmup freezes params near rand init |
| LR warmup on old bare Lion (frieren #3515) | 100.80 (warmup1) | Closed prior round |

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

### Round-6 directions (speculative, contingent on in-flight results)

1. **LR + T_max interaction**: if frieren's lr=2e-4 beats baseline, the optimal T_max will shift (larger LR decays faster in the same number of epochs). T_max should be re-tuned after LR change.
2. **Clip + LR interaction**: alphonse's clip sweep (0.25, 0.5, 2.0) may show that the optimal clip depends on LR. If clip=0.5 wins at lr=1e-4, the sweet spot may shift at lr=2e-4.
3. **Bigger model**: nezuko's h160 at 70.49 is only 0.63 worse than the old baseline — with the new SOTA at 65.74, retesting n_hidden=192 or n_layers=7 on the T_max=21 stack may be worthwhile if the optimization is now less noise-dominated (clip normalizer + T_max fix both reduce noise).
4. **EMA as test-time lever**: edward's EMA experiment may compound with tanjiro's T_max fix — both target late-training refinement.
5. **Full stack with all confirmed levers**: Once clip sweep, EMA, and LR sweep finish, the next synthesis is to stack the winning values of all three onto the T_max=21 baseline.

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