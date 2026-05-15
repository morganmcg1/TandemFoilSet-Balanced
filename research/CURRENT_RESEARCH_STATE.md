# SENPAI Research State

- **Last updated:** 2026-05-15 15:43 UTC
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

| PR | Student | Hypothesis | Status | val_avg/mae_surf_p | Δ vs 136.89 |
|----|---------|-----------|--------|---------------------|------|
| #3173 | alphonse | surf-weight-scan (25, 50) | wip, training | — | — |
| #3176 | askeladd | pressure-channel-weight | **sent back** — sweep 1.5/2.0 | 134.63 (w=3) / 165.22 (w=5) | −1.65% / +20.69% |
| #3181 | edward | grad-clip-huber (max_norm=1.0) | sent back — sweep 5.0/10.0 — assignment poll blocked by REST rate limit until 16:19 UTC | 110.55 @ ep11 (under-converged @ w=1.0) | — |
| #3186 | fern | ema-weights | wip, training | — | — |
| #3190 | frieren | slice-num-128 | wip — assignment poll blocked by REST rate limit until 16:19 UTC | — | — |
| #3196 | nezuko | hidden-256-depth6 | wip, training | — | — |
| #3202 | tanjiro | lr-warmup-cosine (5ep) | **sent back** — sweep 2ep/3ep with T_max realigned | 149.84 (5ep, under-converged) | +9.46% |
| #3211 | thorfinn | per-channel-output-heads | wip — assignment poll blocked by REST rate limit until 16:19 UTC | — | — |

**Operational status:**
- **3 students actively training** (alphonse, fern, nezuko) — GPU 99% util as of 15:31–15:33 UTC. Results expected ~15:51–15:57 UTC.
- **3 students blocked on REST rate limit** (edward retry, frieren, thorfinn) — pods are healthy but their assignment-poll script hits HTTP 403 from a shared 5000/hr core quota that drains across all pods + advisor. Core reset is 16:19:22 UTC. With 5-min poll cadence they will pick up assignments within ~5 min after reset; training results land ~30 min after that (~16:55 UTC).
- **2 students re-assigned** (askeladd, tanjiro) — their PRs were sent back at 15:41/15:42 UTC and they will see the new instructions on their next poll. No new PR needed since they iterate within the same `wandb_group`.
- **GitHub rate limits:** REST core 0/5000 (resets 16:19), GraphQL 3360/5000 (resets 15:53). Use GraphQL exclusively until REST reset.

**Operational alert — `data/scoring.py` NaN bug.** PR #3181 + PR #3176 both confirm: `.test_geom_camber_cruise_gt/000020.pt` contains `inf` AND 761 sample-level NaN values in the pressure channel of GT. The scoring code multiplies `err = (pred − y).abs()` (which becomes `inf` or `NaN`) by `sample_mask` AFTER computing it, so `inf * 0 = NaN` propagates into the accumulator. NaNs out `test_avg/mae_surf_p` AND `vol_loss` for every run on this branch. Fix: zero out non-finite-y samples in `err` before the mask multiply. Needs an advisor-routed fix; for now students report `test_avg/mae_surf_p` from the 3 clean splits.

## Current research focus

Beat the Transolver baseline of `val_avg/mae_surf_p = 136.8873` (equal-weight mean surface pressure MAE across four val splits) — the same metric is computed on the four test splits as `test_avg/mae_surf_p` for paper-facing numbers. The four tracks stress different generalization axes:

1. `val_single_in_dist` — sanity check, raceCar single-foil random holdout
2. `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar tandem)
3. `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise tandem)
4. `val_re_rand` — stratified Re holdout across tandem domains

Prefer common-recipe changes that survive across all four tracks over hacks that only help one. When splits disagree, the disagreement is information. Per the askeladd PR #3176 review, a 1.65% headline win that comes from one big OOD-split win and 3 regressions is *not* a merge — it is a signal to sweep finer.

## Round 1 themes

Initial round explores the **cheapest, highest-EV levers** before touching architecture in deeper ways. Each student tests one orthogonal axis:

- **Loss reformulation** — surface weighting (`surf-weight-scan`), per-channel weighting (`pressure-channel-weight`), robust loss + gradient clipping (`grad-clip-huber`)
- **Optimization** — EMA weights (`ema-weights`), linear warmup (`lr-warmup-cosine`)
- **Architecture** — more physics tokens (`slice-num-128`), wider+deeper (`hidden-256-depth6`), decoupled output heads (`per-channel-output-heads`)

Full hypothesis details: [`research/RESEARCH_IDEAS_2026-05-15_init.md`](RESEARCH_IDEAS_2026-05-15_init.md).

## Early signals from Round 1 reviews

- **Pressure-channel weighting helps RC-camber OOD specifically** (−19% at w=3) but hurts everything else. This is a per-split lever, not a global one. If a gentler weight (1.5–2.0) can preserve the OOD gain without trashing in-dist, that becomes a candidate for combination with architecture/optimization winners.
- **5-epoch warmup is too expensive under the wall-clock cap** — under-converges because cosine tail never activates. Even small warmups need T_max aligned to realized epoch count (~12), not nominal MAX_EPOCHS (50).
- **`max_norm=1.0` is way too aggressive** for this gradient distribution (median pre-clip grad norm 16.15, p99 75.69). 100% of steps clipped at 1.0 cut effective LR ~16× and prevented convergence even though val 110.55 still beat the new baseline by coincidence.

## Potential next research directions

If round-1 yields one or more clean winners after the in-flight retries, the natural follow-ups are:

- **Combinations of orthogonal winners** (loss × architecture × optimization stack)
- **Physics-informed auxiliary losses** — continuity (∂Ux/∂x + ∂Uy/∂y ≈ 0) on volume nodes; tangential-pressure smoothness on surface nodes
- **Geometry-aware augmentation** — vertical mirroring for single-foil (sign-flip AoA), Re scaling within plausible bounds, foil-pair stagger jitter
- **Foil-1 vs foil-2 disambiguation** — add a learned token or feature that distinguishes which foil a node is near
- **Output transformations** — predict residuals over a cheap analytic baseline (potential flow / thin-airfoil), asinh-transformed pressure for high-Re samples
- **Multi-scale attention** — cross-attention between mesh nodes and a small set of geometry tokens summarizing each foil
- **Per-OOD-split conditional losses** — given that `p_surf_weight=3` helps RC-camber dramatically but hurts in-dist, a *domain-aware* surf-loss weighting (different weight per-sample based on tandem-vs-single, camber band, or Re band) could capture the OOD gain without the in-dist regression. This is now a top candidate motivated directly by PR #3176's per-split structure.

## Operational notes

- 8 students at launch — every GPU must remain assigned to a hypothesis this round.
- Per-run budget: 30 min wall clock and 50 epochs hard cap. Wall clock typically binds first (~14 epochs land). Design schedules around the realized epoch budget, not the nominal cap.
- Students edit `train.py` only. `data/` files are read-only.
- GraphQL is mandatory for any label / comment / mutation operations from the advisor until REST resets at 16:19 UTC.
