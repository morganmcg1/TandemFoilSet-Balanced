# SENPAI Research State

- 2026-04-27 23:30 — round 1 in progress on `icml-appendix-willow-pai2d-r2`
- Baseline: unmodified `train.py`. **No concrete metric anchored yet** —
  no finished baseline-config run on W&B; first PR to beat baseline (or
  first dedicated baseline run) populates `BASELINE.md`.
- Primary metric: `val_avg/mae_surf_p`. Paper-facing: `test_avg/mae_surf_p`
  (currently blocked by a scoring NaN — see below).
- W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2`.
- Wall-clock cap per run: `SENPAI_TIMEOUT_MINUTES=30`. Critically, this
  cap turns out to allow only **9–14 epochs** at the round-1 model
  configurations, not the planned 50 — every finished round-1 run is
  hit by the timeout and is undertrained.

## Round 1 status

| PR  | Student   | Axis                       | Status       | Best val_avg/mae_surf_p |
|-----|-----------|----------------------------|--------------|-------------------------|
| 311 | alphonse  | width 128 → 192            | sent back    | 134.13 (epoch 10/50)    |
| 325 | askeladd  | depth 5 → 8                | wip          | 150.06 (W&B; not reviewed)|
| 326 | edward    | mlp_ratio 2 → 4            | review (next)| 137.83 (W&B; not reviewed)|
| 328 | fern      | slice_num 64 → 128         | wip          | 133.55 (W&B; not reviewed)|
| 330 | frieren   | MSE → Huber β=1            | wip          | (multiple crashes; debugging)|
| 332 | nezuko    | surf_weight 10 → 25 (sweep)| wip          | 137.42 surf-15 (sweep ongoing)|
| 335 | tanjiro   | warmup + cos, peak 1e-3    | sent back    | 154.57 (epoch 13/14)    |
| 337 | thorfinn  | BS 4→8, lr 7e-4            | wip          | 139.39 (W&B; not reviewed)|

PRs surfaced for advisor review this cycle: **#311, #335** (both sent
back with specific iteration instructions — see `EXPERIMENTS_LOG.md`).

### Send-back instructions (this cycle)

- **#311 (alphonse)** → `willow-r2-alphonse-width-160` (compute-equal
  middle ground). Optional same-PR AMP-only baseline at width-128 to
  disentangle precision from architecture.
- **#335 (tanjiro)** → parametrize `--cosine_t_max` CLI flag, run sweep
  `(lr 7e-4, T_max 18)` and `(lr 1e-3, T_max 15)` on shared
  `--wandb_group "willow-r2-tanjiro-sched-v2"`.

## Open issues blocking the research programme

1. **`test_geom_camber_cruise/mae_surf_p` is NaN on every round-1 run.**
   Root cause: `data/scoring.py::accumulate_batch` skips samples whose
   GT is non-finite but does not guard against non-finite **predictions**.
   A single divergent prediction on the cruise test set poisons the
   split metric. Blocks `test_avg/mae_surf_p` (paper-facing) for every
   run until fixed. Out of scope for individual round-1 PRs (data/
   files are read-only for normal experiments). **Plan: dedicated
   bug-fix PR after round-1 settles**, with the minimal change being
   to mirror the GT-finiteness skip on `pred_orig` inside
   `accumulate_batch`. Worth flagging to the human research team if the
   cruise test set itself contains pathological samples.

2. **30-min wall-clock cap binds at 9–14 epochs**, not the 50 I
   originally assumed. Implications: (a) capacity-scaling axes
   (width, depth, mlp_ratio) are confounded with undertraining;
   (b) schedule axes (cosine T_max=50) don't engage their decay
   phase; (c) we may need to re-think round-1 conclusions in terms
   of "best at the 30-min budget" rather than "best at convergence."
   The compute-equal width-160 variant (alphonse iteration) is the
   first explicit test of this framing.

3. **No finished baseline run.** Every finished run on this project
   is a hypothesis variant. We don't have a clean "unmodified
   `train.py`" number to anchor against. Once round-1 sweep variants
   land (and ideally an idle student kicks off a 50-epoch baseline
   run with `SENPAI_TIMEOUT_MINUTES` honored as an ordinary cap),
   we'll have the missing reference point.

## Round 2 candidate stacks (post round-1 settle)

- **Compound winner.** Pick the strongest arch axis × strongest
  optimizer/loss axis. Current cohort suggests slice-128 (fern,
  133.55) is the most promising arch axis; surf-weight (nezuko)
  and mlp-ratio (edward) are tied for second.
- **AMP / mixed precision.** Likely a free unlock — at 30-min cap
  every architecture is undertrained, so any throughput gain
  translates directly to better convergence. Alphonse's iteration
  may give us the first AMP signal.
- **Schedule that fits the budget.** OneCycleLR over `total_steps`
  (not epochs) is robust to wall-clock cuts. May supersede the
  `T_max=epochs` cosine pattern entirely if tanjiro's iteration
  shows the budget-matched schedule pays.
- **Target-space reformulation.** asinh / per-sample normalization
  on the pressure channel — pairs especially well with Huber if
  frieren's huber-loss eventually finishes (currently crashing).
- **Per-channel loss weighting.** Up-weight the `p` channel (the
  metric only cares about pressure). Pairs with surf_weight winner.
- **Curriculum / data axes.** Sort batches by per-sample y std,
  warmup the model on low-magnitude samples first.
- **Test-scoring NaN fix** (above) — must land before any
  `test_avg`-driven decisions.

## Constraints respected

- `SENPAI_MAX_EPOCHS=999` and `SENPAI_TIMEOUT_MINUTES=30` not
  overridden.
- Loaders + scoring are read-only — bug-fix PR for `accumulate_batch`
  is the documented exception, deferred to round-1 settle.
- One hypothesis per PR. Send-backs ask for variants of the same
  axis, not bundled changes.
