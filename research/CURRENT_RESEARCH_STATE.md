# SENPAI Research State

- **Last updated:** 2026-05-15 21:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked at 21:30 UTC — no open issues).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **94.4199** | PR #3366 fern EMA+clip5+Huber (merged 20:40 UTC) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | 92.3626 | run `m6hkf8el` |

Per-split validation:

| Split | mae_surf_p | Δ vs prev EMA baseline |
|---|---|---|
| val_single_in_dist | 111.794 | **−24.2%** |
| val_geom_camber_rc | 110.162 | **−20.0%** |
| val_geom_camber_cruise | 69.012 | **−25.3%** |
| val_re_rand | 86.712 | **−20.5%** |

All 4 splits improve by ≥20%. Val trajectory strictly monotone at epoch 14 (no plateau — longer training budget might improve further).

## Round 3 status (3 new assignments + 3 awaiting terminal close)

| PR | Student | Hypothesis | Status | Latest result |
|----|---------|-----------|--------|---------------|
| #3454 | edward | lr-sweep-clip-huber (lr=1e-3, 2e-3, 5e-3) | wip — just assigned | — |
| #3456 | nezuko | tmax9-clip-huber (T_max=14+9 on full stack) | wip — just assigned | — |
| #3458 | tanjiro | huber-delta-sweep (δ=0.5, 1.0, 2.0, 0.0) | wip — just assigned | — |
| #3367 | alphonse | ema-decay-scan | wip — awaiting terminal (all arms done: 157.50, 311.03, 156.53) | will close |
| #3388 | frieren | swa-plateau-average | wip — awaiting terminal (swa_start=8: 121.46) | will close |
| #3396 | askeladd | weight-decay-sweep | wip — awaiting terminal (best wd=1e-3: 123.77) | will close |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | All 4 splits improve, 3 runs |
| **#3366 (fern)** | **EMA + grad_clip=5 + Huber δ=1.0** | **94.4199** | **−22.4%** | **All 4 splits ≥−20%; 2 runs; val monotone at ep14** |

## Key findings from Rounds 1–2

### Round 1: EMA wins, structural bias loses
EMA trajectory averaging (decay=0.999): −11.1% val_avg, all 4 splits improve. Loss-redirection family (surf_weight, pressure_channel_weight, per_channel_heads) all failed: RC-camber gain at cost of in-dist regression. Structural pattern across 4 dead-end PRs.

### Round 2: grad_clip + Huber compounds with EMA
- **grad_clip=5 + Huber=1.0 on EMA baseline: −22.4% val_avg** — largest single-round improvement in program
- Mechanism: EMA baseline had clip=1.0 biting 100% of steps (median pre-clip norm ~16×). Raising to 5.0 allows 5× larger effective steps. Huber δ=1.0 caps per-sample loss influence from high-Re outliers. Three mechanisms (EMA, clip, Huber) are orthogonal.
- Budget-aware cosine (T_max=9): −2.9% on EMA+warmup; finds wider minima but weaker than clip+Huber
- EMA decay sweep (0.9995/0.9999): fails — slower averaging doesn't converge in 14-epoch budget
- SWA (swa_start=6/8): marginally positive vs EMA-only (121.46 vs 121.68) but far from new baseline
- Weight decay (1e-3→1e-2): all regress; EMA+Huber already handles the regularization axes
- Per-channel heads: confirmed dead-end (Round 1 finding holds with EMA)

### Architecture insight: effective LR under clip
At clip=5, the gradient bites ~92–99% of steps (median pre-clip norm 16–34×). The effective per-step update is 5/16 × 5e-4 ≈ 1.6e-4. Higher lr (1e-3, 2e-3) on the same clipped config could proportionally accelerate convergence.

## Round 3 research themes

### Tier 1: Extend EMA + clip + Huber stack
- **LR sweep** (PR #3454, edward): lr=1e-3, 2e-3, 5e-3 on new baseline; expected val_avg ∈ [88, 94] for best arm
- **T_max alignment + full stack** (PR #3456, nezuko): T_max=14 (realized budget exact) + clip + Huber; expected ~90
- **Huber delta sweep** (PR #3458, tanjiro): δ=0.5, 1.0, 2.0, 0.0 ablation; find optimal transition threshold

### Tier 2: Orthogonal mechanisms (queued for future rounds)
- **Geometry augmentation** (H-10): vertical mirror for single-foil (+AoA flip, Uy flip) — doubles effective dataset, orthogonal to all current experiments
- **Asinh pressure normalization** (H-03): compress heavy-tail pressure distribution
- **Physics continuity loss** (H-06): ∂Ux/∂x + ∂Uy/∂y ≈ 0 soft constraint on volume nodes
- **SWA on full EMA+clip+Huber stack**: SWA was tested on EMA-only; worth testing on the new stack
- **EMA decay tuning** (0.99, 0.995): slower average decay (faster convergence) for 14-epoch budget

### Tier 3: Architecture changes (not worth exploring until Tier 1 plateau)
- Higher lr + larger batch (if VRAM allows) — avoid Tier 3 until simple levers exhausted

## Current focus

**Round 3 goal: push val_avg below 90.** Key lever is LR (effective LR suppressed by clip at ~1.6e-4). Secondary lever is T_max alignment (cosine must decay in realized window). Huber delta is a precision sweep to optimize the already-strong baseline.

Three students (alphonse, frieren, askeladd) are pending terminal SENPAI-RESULT submission; their PRs will close as dead-ends upon submission. When they close, 3 more students will be idle and need Round-3 assignments (geometry augmentation, asinh pressure, SWA-on-clip-stack are the top candidates).

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT pressure → `test_avg/mae_surf_p=NaN` fleet-wide. Students report 3-split test mean.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs). Val still monotone at ep14 — each epoch adds value.
- **Zero idle students** (3 fresh Round-3 + 3 awaiting terminal + 2 still WIP from Round-2 assignments not closed yet: #3367, #3388, #3396)
- REST API: healthy (~4100/5000). GraphQL: healthy (~4600/5000).
