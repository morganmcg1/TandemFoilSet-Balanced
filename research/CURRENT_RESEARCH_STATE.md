# SENPAI Research State

- **Last updated:** 2026-05-15 16:30 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None on this launch.

## Branch baseline established

First reproducible measurement on this branch came from PR #3176's baseline-w1 reference arm (`07efagec`).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | **136.8873** (best @ epoch 14) |
| `test_avg/mae_surf_p` (3 valid splits; cruise GT corrupted) | 137.6945 |

See `BASELINE.md` for the per-split table.

## Round 1 status (8 PRs in flight)

| PR | Student | Hypothesis | Status | Best val_avg seen | Δ vs 136.89 |
|----|---------|-----------|--------|---------------------|------|
| #3173 | alphonse | surf-weight-scan (25, 50) | wip — **W&B shows results but no commit/submit; nudged** | 130.29 (w=50, run `mdkp6avx`) | **−4.82%** but +11.4% on val_single_in_dist |
| #3176 | askeladd | pressure-channel-weight | wip — retry running for w=1.5/2.0 (GPU 99% as of 16:28 UTC) | 134.63 (w=3, prev arm) | −1.65% (single-split-carried) |
| #3181 | edward | grad-clip-huber retry (5.0/10.0) | wip — assignment picked up at 16:21 UTC, training pending | 110.55 prev (w=1.0, under-converged) | — |
| #3186 | fern | ema-weights (decay=0.999) | wip — **STRONG W&B winner not yet pushed; nudged** | **121.69** (run `2i7tmbir`) | **−11.10%** — ALL 4 SPLITS IMPROVE |
| #3190 | frieren | slice-num-128 | wip — **W&B shows regressions (140.96, 161.89, 171.89); nudged to close** | 140.96 best | +2.98% |
| #3196 | nezuko | hidden-256-depth6 | wip — **W&B shows 3 failures + 2 regressions; nudged to close** | 152.48 best finished | +11.4% |
| #3202 | tanjiro | lr-warmup-cosine retry (2ep/3ep) | wip — assignment picked up at 16:19 UTC, training pending | 149.84 prev (5ep, under-converged) | — |
| #3211 | thorfinn | per-channel-output-heads | wip — **W&B shows marginal result; nudged** | 133.70 (run `x3h1o3id`) | −2.33% (single-split-carried) |

**Operational gap surfaced at 16:25 UTC:** Five students (alphonse, fern, frieren, nezuko, thorfinn) completed multiple training runs each with results logged to W&B but **never committed their `train.py` code change or posted a `SENPAI-RESULT` marker**. Their branch HEADs still equal the original assignment commit from 12:52 UTC; the working-tree `M train.py` modification is uncommitted. Without code in the branch, neither merge nor review is possible — there is literally nothing to merge. Advisor posted explicit nudge comments on all 5 PRs at 16:28 UTC instructing them to commit + push + post SENPAI-RESULT + invoke `senpai:submit-experiment-results`.

**Operational status:**
- **2 students actively training** (askeladd #3176 retry GPU 99% @ 16:28, tanjiro #3202 picked up @ 16:19) — results expected ~16:51–16:58 UTC.
- **5 students hold-by-nudge** (alphonse, fern, frieren, nezuko, thorfinn) — awaiting commit + push + submit before review can proceed.
- **1 student early-stage** (edward retry picked up at 16:21 UTC, training pending).
- **GitHub rate limits:** REST core 1670/5000 (resets 17:19), GraphQL 4300/5000 (resets 16:53). Both healthy.

**Operational alert — `data/scoring.py` NaN bug.** PR #3181 + PR #3176 both confirm: `.test_geom_camber_cruise_gt/000020.pt` contains `inf` AND 761 sample-level NaN values in the pressure channel of GT. `inf * 0 = NaN` in `err * sample_mask` propagates. NaNs out `test_avg/mae_surf_p` AND `vol_loss` fleet-wide. Needs an advisor-routed fix; students report `test_avg` from 3 clean splits.

## Most likely Round-1 winner: EMA weights (fern, #3186)

If fern submits, the EMA winner (`val_avg = 121.69`, `decay=0.999`) is the cleanest result yet:

| Split | EMA (`2i7tmbir`) | baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 147.55 | 151.85 | **−2.83%** |
| val_geom_camber_rc | 137.68 | 173.91 | **−20.83%** |
| val_geom_camber_cruise | 92.42 | 101.41 | **−8.86%** |
| val_re_rand | 109.09 | 120.38 | **−9.38%** |
| **val_avg** | **121.69** | 136.89 | **−11.10%** |

All four splits improve, and three independent runs (121.69, 122.64, 123.13) cluster within ±0.7 — high reproducibility. This is the "common-recipe change that survives across all four tracks" the branch charter calls for, and is a candidate for merge as soon as fern submits.

## Current research focus

Beat the Transolver baseline of `val_avg/mae_surf_p = 136.8873`. The four val splits stress:

1. `val_single_in_dist` — sanity check, raceCar single-foil random holdout
2. `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar tandem)
3. `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise tandem)
4. `val_re_rand` — stratified Re holdout across tandem domains

Prefer common-recipe changes that survive across all four tracks over hacks that only help one.

## Round 1 themes

Initial round explores the cheapest, highest-EV levers before touching architecture deeply. Each student tests one orthogonal axis:

- **Loss reformulation** — surface weighting (`surf-weight-scan`), per-channel weighting (`pressure-channel-weight`), robust loss + gradient clipping (`grad-clip-huber`)
- **Optimization** — EMA weights (`ema-weights`), linear warmup (`lr-warmup-cosine`)
- **Architecture** — more physics tokens (`slice-num-128`), wider+deeper (`hidden-256-depth6`), decoupled output heads (`per-channel-output-heads`)

Full hypothesis details: [`research/RESEARCH_IDEAS_2026-05-15_init.md`](RESEARCH_IDEAS_2026-05-15_init.md).

## Early signals from Round 1 (incl. W&B surfacing)

- **EMA weights (`decay=0.999`) is the clearest winner so far** — three independent runs all hit −10% to −11% on `val_avg` with all 4 splits improving. Awaiting formal submission.
- **Pressure-channel weighting helps RC-camber OOD specifically** but hurts everything else. Same per-split pattern shows up across `pressure-channel-weight` (askeladd), `surf-weight-scan` (alphonse), and `per-channel-output-heads` (thorfinn) — they all *redirect* loss weight toward the surface or pressure channel and they all get the RC-camber gain at the cost of in-dist regression. This is structural, not coincidence — RC-camber is the OOD split that is geometry-similar to in-dist but unseen, so it benefits most from extra pressure signal, while the in-dist split was already well-fit and now over-fits.
- **5-epoch warmup is too expensive under wall-clock cap** — under-converges because cosine tail never activates. Even small warmups need `T_max` aligned to realized epoch count (~12), not nominal `MAX_EPOCHS=50`.
- **`max_norm=1.0` is way too aggressive** for this gradient distribution (median pre-clip grad norm 16.15, p99 75.69).
- **`slice_num=128` and `n_hidden=256, n_layers=6` are clear regressions** at this training budget — extra capacity under-converges and adds variance. Reject for now.

## Potential next research directions

If round-1 yields one or more clean winners after the in-flight retries + nudge-driven submissions, the natural follow-ups are:

- **Combinations of orthogonal winners** — EMA × pressure-channel-weighting × shorter warmup, etc.
- **Physics-informed auxiliary losses** — continuity (∂Ux/∂x + ∂Uy/∂y ≈ 0) on volume nodes; tangential-pressure smoothness on surface nodes
- **Geometry-aware augmentation** — vertical mirroring for single-foil (sign-flip AoA), Re scaling within plausible bounds, foil-pair stagger jitter
- **Foil-1 vs foil-2 disambiguation** — add a learned token or feature that distinguishes which foil a node is near
- **Output transformations** — predict residuals over a cheap analytic baseline (potential flow / thin-airfoil), asinh-transformed pressure for high-Re samples
- **Multi-scale attention** — cross-attention between mesh nodes and a small set of geometry tokens summarizing each foil
- **Per-OOD-split conditional losses** — domain-aware surf-loss weighting (different weight per-sample based on tandem-vs-single, camber band, or Re band) to capture the OOD gain without the in-dist regression
- **Longer-epoch budgets for over-capacity models** — the failed `hidden-256-depth6` and `slice_num=128` may be salvageable with `MAX_EPOCHS=100` and a fitness/wall-clock-pegged schedule.

## Operational notes

- 8 students at launch — every GPU must remain assigned. Stale-WIP-without-commit is the new failure mode to detect.
- Per-run budget: 30 min wall clock and 50 epochs hard cap. Wall clock typically binds first (~14 epochs land). Design schedules around realized epoch budget, not nominal cap.
- Students edit `train.py` only. `data/` files are read-only.
- **W&B is now part of the advisor's review surface.** If a student goes silent on a PR but has W&B runs in the expected wandb_group, surface those runs and prod the student to formally submit before assuming nothing happened.
