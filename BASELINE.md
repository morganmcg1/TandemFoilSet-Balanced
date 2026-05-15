# Baseline — icml-appendix-charlie-pai2i-48h-r5

## Current Best

### 2026-05-15 16:28 — PR #3213: Huber loss (delta=0.3) — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **103.18** (best_epoch=13/50)
- **test_avg/mae_surf_p**: **92.02** (NaN-safe re-eval; baseline eval had NaN from data bug)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 111.93 |
  | geom_camber_rc | 102.85 |
  | geom_camber_cruise | 62.84 |
  | re_rand | 90.45 |
- **Metric artifacts**: `models/model-huber-0.3-20260515-140457/metrics.jsonl`
- **Also included**: NaN-safe `evaluate_split` fix (sample-level skip for non-finite GT, works around data bug in `test_geom_camber_cruise/000020.pt`)
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --experiment_name huber-0.3 \
      --huber_delta 0.3 \
      --agent charliepai2i48h5-frieren
  ```

## Reference Configuration (baseline `train.py`)
- Model: Transolver
  - n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  - space_dim=2, fun_dim=22, out_dim=3
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Cosine annealing LR schedule, T_max=epochs
- batch_size=4
- surf_weight=10.0 (loss = vol_loss + 10 * surf_loss)
- Default epochs=50 cap, SENPAI_TIMEOUT_MINUTES wall-clock cap
- Balanced WeightedRandomSampler across domains (single/RC-tandem/cruise-tandem)
- Loss in normalized target space; metrics in denormalized physical units

## Splits (lower is better — surface MAE on pressure)
| Split | Test source | Notes |
|---|---|---|
| val_single_in_dist | random holdout from single-foil | sanity |
| val_geom_camber_rc | raceCar M=6-8 front foil | geometry extrapolation |
| val_geom_camber_cruise | cruise M=2-4 front foil | geometry extrapolation |
| val_re_rand | stratified Re across all tandem domains | Re generalization |

Primary metric: equal-weight average across all 4 splits.

## Notes
- Round 5, charlie arm, 48h budget, local JSONL metrics only.
- 8 students, 1 GPU (96GB) each.
- First batch of hypotheses includes a clean baseline run to anchor the metric.
