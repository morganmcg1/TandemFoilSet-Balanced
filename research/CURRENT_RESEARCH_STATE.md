# SENPAI Research State

- **Last updated:** 2026-05-15 18:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked at 17:45 UTC — no open issues).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **121.685** | PR #3186 fern EMA (merged) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | 118.281 | PR #3186 |

**Pending merges that would update this baseline:**
- **edward `b6t3344j` (NEW STRONGEST):** `best_val_avg = 106.7216` (−12.3% below EMA baseline), pre-EMA codebase, awaiting terminal SENPAI-RESULT.
- **tanjiro `3kervu49`:** val_avg = 119.7996 (−1.55% below EMA baseline), pre-EMA codebase, awaiting terminal SENPAI-RESULT. Mergeable but likely superseded by edward.

## Round 2 status (5 PRs in flight, 2 pending terminal, 1 student to assign)

| PR | Student | Hypothesis | Status | Latest result |
|----|---------|-----------|--------|---------------|
| #3366 | fern | ema-grad-clip-huber (EMA + clip=5 + Huber) | wip | not yet running |
| #3367 | alphonse | ema-decay-scan (0.9995, 0.9999) | wip | not yet running |
| #3368 | thorfinn | ema-per-channel-heads | wip | not yet running |
| #3369 | nezuko | cosine-tmax-align (T_max=9/12/16) | wip | not yet running |
| #3388 | frieren | swa-plateau-average (SWA alongside EMA) | wip — just assigned | — |
| #3181 | edward | grad-clip-huber retry | wip — awaiting terminal | **NEW STRONGEST: b6t3344j best_val_avg=106.72** |
| #3202 | tanjiro | lr-warmup-cosine retry | wip — awaiting terminal | 3kervu49: val_avg=119.80 (BEATS baseline by −1.55%) |
| TBD | askeladd | (to assign — H-02 weight-decay-sweep candidate) | — | #3176 closed |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | **121.685** | **−11.10%** | All 4 splits improve, 3 reproducible runs |

## Key findings from Round 1

### EMA mechanism
EMA trajectory averaging (decay=0.999) is the first clean winner: −11.1% val_avg with all 4 splits improving and 3 reproducible runs. The mechanism — averaging late-epoch parameter snapshots near the flat cosine-basin center — generalizes across ALL four distribution shifts.

### Grad clip + Huber (edward, pre-EMA, b6t3344j = 106.72, pending submission)
grad_clip=5.0 + Huber δ=1.0 WITHOUT EMA: 3-run reproducibility (106.72, 109.45, 114.38). Mechanism: original training had 100% of steps clipped at max_norm=1.0 (median pre-clip norm ~16×), effectively cutting LR 16×. Raising clip to 5.0 allows proper gradient steps; Huber bounds 1% outlier gradients from high-Re samples. ORTHOGONAL to EMA → PR #3366 (EMA + grad_clip + Huber) is highest-priority Round-2 PR.

### Budget-aware cosine schedule (tanjiro 3kervu49, pre-EMA)
warmup_epochs=3 + cosine_t_max=9 (matched to realized 14-epoch budget) + warmup_start_factor=0.1 gives val_avg=119.80 — −1.55% on the pre-EMA codebase. T_max=9 ensures cosine fully decays in the wall-clock window; T_max=50 (default for MAX_EPOCHS=50) leaves LR near peak when training stops. Validates nezuko's PR #3369 hypothesis. May or may not stack with EMA — that's what #3369 tests.

### Structural pattern: "redirect loss" hypotheses (closed family)
Four hypotheses (surf_weight=50, pressure_channel_weight at 3×/5× and at 1.5×/2×, per_channel_heads) all showed the same fragile pattern: strong RC-camber OOD gain at the cost of in-dist regression. This is structural — these approaches redirect gradient toward surface/OOD geometry without improving the global representation quality. None beat the EMA baseline.

### Scaling capacity doesn't work at 30-min budget
n_hidden=256/n_layers=6 and slice_num=128 both regress: larger models under-converge because bs=2 is needed for VRAM and only 6–7 epochs complete in 30 min. Any capacity experiment needs a 2× budget allocation.

## Round 2 research themes

### Tier 1: High EV (extend EMA + grad_clip/Huber finding)
- **EMA + grad_clip=5 + Huber** (PR #3366, fern): the two strongest individual improvements combined; expected ~105–115 if orthogonal
- **EMA decay tuning** (PR #3367, alphonse): 0.9995/0.9999 could squeeze more from trajectory averaging
- **SWA alongside EMA** (PR #3388, frieren): uniform-snapshot averager, orthogonal mechanism to EMA's exponentially-weighted average

### Tier 2: Structural improvements on EMA baseline
- **EMA + per-channel heads** (PR #3368, thorfinn): test if structural bias stacks with trajectory smoothing
- **Cosine T_max alignment** (PR #3369, nezuko): tanjiro's 119.80 result validates this works *without* EMA — does it stack with EMA?

### Tier 3: Pending assignment / queued for future rounds
- **Weight decay sweep** (H-02 — to assign to askeladd next): 1e-3 / 5e-3 / 1e-2 vs current 1e-4
- **Asinh pressure output normalization** (H-03): compress heavy-tail pressure distribution
- **Dropout regularization** (H-04)
- **Physics-informed continuity loss** (H-06): ∂Ux/∂x + ∂Uy/∂y ≈ 0 on volume nodes
- **Geometry-aware augmentation** (H-10): vertical mirroring for single-foil (sign-flip AoA + Uy)
- **Higher lr sweep on EMA baseline**: original lr=5e-4 may be too conservative with grad_clip allowing larger steps
- **Per-domain conditional surf_weight**: separate weights for single-foil vs tandem (captures RC-camber benefit without in-dist regression)
- **Huber delta sweep** (δ=0.5 vs 1.0 vs 2.0 on EMA+clip baseline): from edward's suggestion
- **LR one-cycle** (H-07, High EV but overlaps with nezuko's #3369)
- **Longer budget rerun** (H-09 — NOT ASSIGNABLE: overrides hard timeout cap)

## Current focus

**Two parallel tracks:**
1. **Merge edward's b6t3344j** as soon as he submits terminal SENPAI-RESULT. This sets a much lower baseline (~106) and reframes all in-flight Round-2 work against it.
2. **Round-2 stack experiments** (PRs #3366–#3369, #3388) test whether the orthogonal mechanisms (EMA + grad_clip/Huber, EMA decay tuning, EMA + per-channel heads, T_max alignment, SWA) compound.

Goal for Round 2 is to push below val_avg=110, with the long-term target sub-100.

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT pressure → `test_avg/mae_surf_p=NaN` fleet-wide. Students report 3-split test mean. Fix needed in scoring.py (advisor-routed).
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs). 
- **REST rate limit**: Recovered (~3000/5000 remaining; resets at 18:19 UTC have passed). GraphQL preferred for label and comment mutations.
- 1 student to assign (askeladd, freed by #3176 close). Zero idle students otherwise.
