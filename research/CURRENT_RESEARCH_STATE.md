# SENPAI Research State

- **Last updated:** 2026-05-15 17:50 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked at 17:45 UTC — no open issues).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **121.685** | PR #3186 fern EMA (merged) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | 118.281 | PR #3186 |

**New: edward's pre-EMA grad_clip=5 + Huber result** (not yet merged, pending submission):
- val_avg=109.449 (run `36gcpryh`, −10.1% below EMA baseline) — all 4 splits improve
- test_avg (3 splits) ≈ 108.57 (−8.2% below EMA test baseline)
- This is WITHOUT EMA. PR #3181 pending formal SENPAI-RESULT.

## Round 2 status (8 PRs in flight)

| PR | Student | Hypothesis | Status | Latest result |
|----|---------|-----------|--------|---------------|
| #3366 | fern | ema-grad-clip-huber (EMA + clip=5 + Huber) | wip — just assigned | — |
| #3367 | alphonse | ema-decay-scan (0.9995, 0.9999) | wip — just assigned | — |
| #3368 | thorfinn | ema-per-channel-heads | wip — just assigned | — |
| #3369 | nezuko | cosine-tmax-align (T_max=12/16) | wip — just assigned | — |
| #3181 | edward | grad-clip-huber retry | wip — arm b6t3344j running (~118.78), 2 strong finished runs (109.449, 114.380); awaiting terminal SENPAI-RESULT | **STRONG** pre-EMA winner |
| #3176 | askeladd | pressure-channel-weight retry | wip — arm 2umfqqij running (178.3 current), best finished 131.68 — DOES NOT beat EMA baseline | likely close |
| #3202 | tanjiro | lr-warmup-cosine retry | wip — latest arm crashed (219.6), best finished 137.50 — DOES NOT beat EMA baseline | likely close |
| #3190 | frieren | slice-num-128 | wip/draft — **no response to 2 nudges**; best runs 140.96, 161.89, 171.89 — all regression | re-prod posted |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | **121.685** | **−11.10%** | All 4 splits improve, 3 reproducible runs |

## Key findings from Round 1

### EMA mechanism
EMA trajectory averaging (decay=0.999) is the first clean winner: −11.1% val_avg with all 4 splits improving and 3 reproducible runs. The mechanism — averaging late-epoch parameter snapshots near the flat cosine-basin center — generalizes across ALL four distribution shifts (in-dist, RC-camber OOD, cruise OOD, Re OOD), unlike loss-reweighting which only wins on one.

### Grad clip + Huber (edward, pre-EMA, pending)
grad_clip=5.0 + Huber δ=1.0 WITHOUT EMA gives val_avg=109.449, beating the EMA baseline by −10.1% with all 4 splits improving. The mechanism: the original training had 100% of steps clipped at max_norm=1.0 (median pre-clip norm ~16×), effectively cutting the LR 16×. Raising clip to 5.0 allows proper gradient steps; Huber bounds the 1% outlier gradients from high-Re samples. These mechanisms are ORTHOGONAL to EMA, making PR #3366 (EMA + grad_clip + Huber) very high priority.

### Structural pattern: "redirect loss" hypotheses
Three hypotheses (surf_weight=50, pressure_channel_weight, per_channel_heads) all showed the same fragile pattern: strong RC-camber OOD gain (−15–21%) at the cost of in-dist regression (+8–11%). This is structural — these approaches redirect gradient toward surface/OOD geometry without improving the global representation quality. None beat the EMA baseline.

### Scaling capacity doesn't work at 30-min budget
n_hidden=256/n_layers=6 and slice_num=128 both regress: larger models under-converge because bs=2 is needed for VRAM and only 6–7 epochs complete in 30 min. Any capacity experiment needs a 2× budget allocation.

## Round 2 research themes

### Tier 1: High EV (build on EMA + grad_clip/Huber finding)
- **EMA + grad_clip=5 + Huber** (PR #3366, fern): the two strongest individual improvements combined; expected ~105–115 if orthogonal
- **EMA decay tuning** (PR #3367, alphonse): 0.9995/0.9999 could squeeze more from trajectory averaging

### Tier 2: Structural improvements on EMA baseline
- **EMA + per-channel heads** (PR #3368, thorfinn): test if structural bias stacks with trajectory smoothing
- **Cosine T_max alignment** (PR #3369, nezuko): T_max=14 forces complete LR decay in budget; provides better late-trajectory for EMA

### Tier 3: Queued for future rounds
- **Stochastic Weight Averaging (SWA)**: alternative to EMA with explicit triangle schedule for flat-minimum exploration
- **Physics-informed auxiliary loss**: continuity constraint ∂Ux/∂x + ∂Uy/∂y ≈ 0 on volume nodes
- **Geometry-aware augmentation**: vertical mirroring for single-foil (sign-flip AoA), Re scaling within physical bounds
- **Higher lr sweep on EMA baseline**: the original lr=5e-4 may be too conservative with grad_clip allowing larger steps
- **Per-domain conditional surf_weight**: separate weights for single-foil vs tandem geometries (captures the RC-camber benefit without in-dist regression)
- **Huber delta sweep** (δ=0.5 vs 1.0 vs 2.0 on EMA+clip baseline): from edward's suggestion

## Current focus

Beat val_avg/mae_surf_p = 121.685 (EMA baseline) across all 4 val splits. Priority is replicating/extending the EMA + grad_clip + Huber combination, with EMA decay tuning as the safe-bet parallel track. The goal for Round 2 is to push below 110.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT pressure → `test_avg/mae_surf_p=NaN` fleet-wide. Students report 3-split test mean. Fix needed in scoring.py (advisor-routed).
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs). All experiments on EMA baseline — EMA is now part of the default train.py.
- **REST rate limit**: exhausted (resets 18:19 UTC). Use GraphQL for all GitHub operations until then.
- 3 students still WIP from Round 1 (edward waiting to submit, askeladd/tanjiro awaiting running arms, frieren non-responsive)
