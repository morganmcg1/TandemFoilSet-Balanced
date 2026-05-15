# Charlie Round 3 — Baseline (`icml-appendix-charlie-pai2h-48h-r3`)

## Current best

No experiment has been merged into `icml-appendix-charlie-pai2h-48h-r3` yet — this round is starting from `target/train.py` defaults.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | TBD (no merged experiment) |
| `test_avg/mae_surf_p` | TBD |
| Best PR | — |
| Date | 2026-05-15 (round start) |

## Reference configuration (`train.py` defaults)

- **Model:** Transolver (physics-aware slice attention)
  - `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `dropout=0.0`, `space_dim=2`, `fun_dim=22`, `out_dim=3`
  - Approx **0.5 M** parameters
- **Optimizer:** AdamW `lr=5e-4`, `weight_decay=1e-4`, cosine annealing `T_max=epochs`
- **Loss:** MSE in normalized target space, `loss = vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- **Batching:** `batch_size=4`, `WeightedRandomSampler` for domain-balanced training (raceCar single / raceCar tandem / cruise tandem each weighted ~1/3)
- **Schedule:** Per-training-run hard caps — `SENPAI_TIMEOUT_MINUTES=30.0`, `SENPAI_MAX_EPOCHS=50`. Best checkpoint selected by lowest `val_avg/mae_surf_p`, then evaluated on the four test splits.

## Primary metric (lower is better)

`val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE across the four validation tracks (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Aggregation is global over all surface nodes in the split (so the larger cruise meshes contribute proportionally more nodes than the smaller raceCar-single meshes).

Paper-facing decision metric is `test_avg/mae_surf_p` evaluated from the best-val checkpoint.

## Notes

- This is the Charlie *local-metrics* arm: experiment metrics are committed as JSONL under `models/<experiment>/metrics.jsonl` and a `metrics.yaml` summary. No remote experiment tracking is enabled.
- The baseline above is the starting point of round 3. The first merged improvement on this branch will replace these defaults as the new baseline reference.
