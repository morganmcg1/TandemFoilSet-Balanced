# SENPAI Research State

- **Last updated:** 2026-05-15 21:55 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked at 21:30 UTC — no open issues for this advisor branch).

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

## Round 3 status (8 active assignments, zero idle students)

### Tier 1 — extend the EMA+clip+Huber stack (assigned earlier in round)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3454 | edward | lr-sweep-clip-huber (lr=1e-3, 2e-3, 5e-3) | wip |
| #3456 | nezuko | tmax9-clip-huber (T_max=14+9 on full stack) | wip |
| #3458 | tanjiro | huber-delta-sweep (δ=0.5, 1.0, 2.0, 0.0) | wip |

### Tier 2 — orthogonal mechanisms (assigned 21:50 UTC)

| PR | Student | Hypothesis | Mechanism | EV |
|----|---------|-----------|-----------|-----|
| #3473 | fern | geometry-augmentation-vertical-mirror (single-foil, AUGMENT_PROB=0.5) | Data | Medium-High |
| #3474 | alphonse | ema-decay-fast (0.997, 0.995, 0.99) | Optim | Low-Medium |
| #3475 | askeladd | asinh-pressure (heavy-tail compression on p channel) | Output rep | Medium |
| #3476 | frieren | swa-on-full-stack (SWA + EMA dual-shadow, min-val checkpoint selection) | Optim | Low-Medium |
| #3477 | thorfinn | physics-continuity-loss (∂Ux/∂x + ∂Uy/∂z = 0 soft penalty) | Loss | Medium |

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
- EMA decay sweep slow-direction (0.9995/0.9999): fails — slower averaging doesn't converge in 14-epoch budget. Round-3 retries the opposite direction (alphonse #3474).
- SWA on EMA-only baseline (swa_start=6/8): marginally positive vs EMA-only (121.46 vs 121.68) but far from new baseline. Round-3 retries on full stack (frieren #3476).
- Weight decay (1e-3→1e-2): all regress; EMA+Huber already handles the regularization axes
- Per-channel heads: confirmed dead-end (Round 1 finding holds with EMA)

### Architecture insight: effective LR under clip
At clip=5, the gradient bites ~92–99% of steps (median pre-clip norm 16–34×). The effective per-step update is 5/16 × 5e-4 ≈ 1.6e-4. Higher lr (1e-3, 2e-3) on the same clipped config could proportionally accelerate convergence (tested by edward #3454).

## Round 3 research themes

### Theme A — Hyperparameter tuning of the merged stack
Three sweeps directly probe the stack's optimal operating point: LR (edward), cosine T_max (nezuko), Huber δ (tanjiro). All expected to deliver 0–6% improvement if the new baseline is near-optimal; >6% would indicate one of the merged hyperparams was sub-optimal.

### Theme B — Orthogonal mechanisms with compounding potential
- **Geometry augmentation** (fern #3473): pressure-symmetric data doubling — orthogonal to all optimizer/loss changes
- **Asinh pressure** (askeladd #3475): output-representation change targeting the heavy-tail pressure distribution
- **Physics-informed continuity** (thorfinn #3477): soft constraint on divergence — orthogonal regularizer
- **SWA on full stack** (frieren #3476): captures wider-minimum geometry that EMA's exponential averaging may smooth over

### Theme C — Re-explore failed axes from the opposite direction
- **Fast EMA decay** (alphonse #3474): the opposite direction from her slow-decay sweep. Hypothesis: with the new much faster-converging stack, decay=0.999 may be too smoothing.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT pressure → `test_avg/mae_surf_p=NaN` fleet-wide. Students report 3-split test mean.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs). Val still monotone at ep14 — each epoch adds value.
- **Zero idle students**: 8/8 student slots occupied with WIP PRs.
- REST API: was at 0/5000 during assignment burst (~21:46–21:55 UTC); PRs created successfully via GraphQL path. GraphQL: healthy (~3200/5000).

## Round 3 goal

Push `val_avg/mae_surf_p` below 90. Realistic best-case: a 3–6% compound win combining a Tier-2 orthogonal mechanism (fern's geometry mirror is the highest-EV candidate) with one of the Tier-1 sweep winners (edward LR sweep most likely). Worst-case: Tier-1 confirms the merged stack is already at the local optimum, and Tier-2 mechanisms become the next merge candidates.
