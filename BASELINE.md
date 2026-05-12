# Charlie pai2g 48h r1 — Baseline

Branch: `icml-appendix-charlie-pai2g-48h-r1`
Research tag: `charlie-pai2g-48h-r1`

## Status (2026-05-12)

**No measured baseline yet.** This advisor branch was created fresh from
`icml-appendix-charlie`. Round-1 student experiments (PRs #1355, #1381, #1385,
#1389, #1393, #1399, #1405, #1410) include a wall-clock-tuned vanilla run on
several PR arms — the first terminal `SENPAI-RESULT` from any of them
establishes the de-facto baseline measurement. We will update this file with
the actual `val_avg/mae_surf_p` and `test_avg/mae_surf_p` once the first PR
completes.

## Reference configuration (unchanged `train.py`)

- **Optimizer:** `AdamW(lr=5e-4, weight_decay=1e-4)`
- **Scheduler:** `CosineAnnealingLR(T_max=epochs)`
- **Loss:** MSE in normalized space, `vol_loss + 10.0 * surf_loss`
- **Model:** Transolver
  - `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
  - `space_dim=2, fun_dim=22 (= X_DIM - 2), out_dim=3`
  - ~1.1M params
- **Training:** `batch_size=4`, fp32
  - `WeightedRandomSampler` for equal-weight domain sampling across the three
    training-domain groups (raceCar single, raceCar tandem, cruise tandem).
- **Wall-clock cap:** `SENPAI_TIMEOUT_MINUTES=30` per training execution.
- **Eval splits:** `val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand` (100 samples each).
- **Test splits:** matching test versions, 200 samples each. Evaluated once at
  the end of training using the best-val checkpoint.

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four
val splits, in the original target space (physical units). Lower is better.

## Paper-facing metric

`test_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four
test splits, evaluated from the best-val checkpoint at the end of training.

## Notes for reviewers

- The default `--epochs 50` is wasted under the 30-min cap: at ~2 min/epoch
  only ~10-15 epochs actually run, so cosine annealing only enters its first
  ~30% of the decay curve. All round-1 PRs explicitly tune `--epochs` to fit
  the cap; this is itself an implicit common-recipe improvement.
- Compute headroom: 96 GB VRAM is heavily underused at batch=4, fp32 — room
  for wider models, larger batches, AMP, etc.
- Local JSONL metrics only on this branch (`models/<exp>/metrics.jsonl`);
  no W&B.
