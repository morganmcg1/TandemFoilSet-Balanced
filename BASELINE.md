# Baseline — `icml-appendix-charlie-pai2g-24h-r4`

Fresh start for round 4 of the Charlie / pai2g 24h logging ablation. No
prior experiments on this branch — the implicit baseline is the unmodified
`train.py` config inherited from `icml-appendix-charlie`. The first merged
winner sets the first numeric reference value.

## Reference config (default `train.py` at HEAD)

- **Model**: Transolver
  - `n_hidden = 128`
  - `n_layers = 5`
  - `n_head = 4`
  - `slice_num = 64`
  - `mlp_ratio = 2`
  - `space_dim = 2`, `fun_dim = X_DIM - 2 = 22`
  - `out_dim = 3` (`Ux`, `Uy`, `p`)
  - `unified_pos = False`
- **Optimizer**: AdamW (`lr = 5e-4`, `weight_decay = 1e-4`)
- **LR schedule**: CosineAnnealingLR with `T_max = MAX_EPOCHS`
- **Loss**: **L1 (MAE) in normalized target space**, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight = 10.0` _(updated 2026-05-12 by PR #1397)_
- **Batch size**: `4`
- **Epochs**: configured `50`, capped by `SENPAI_TIMEOUT_MINUTES = 30`
- **Sampler**: `WeightedRandomSampler` with equal-domain weights from `meta.json`

## Metrics contract

- Primary ranking metric: `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits.
- Paper-facing comparison metric: `test_avg/mae_surf_p` — same aggregation on the four test splits at the best val checkpoint.
- All metrics computed in physical (denormalized) units in `data/scoring.py`.

## Current best result

### 2026-05-12 19:05 — PR #1397 (`charliepai2g24h4-alphonse/l1-loss`)

L1 (MAE) loss replaces MSE in normalized-space training. First numeric
baseline on this branch.

- **`val_avg/mae_surf_p`** = **100.9574** (best @ epoch 13/14 before 30 min timeout)
- **`test_avg/mae_surf_p` (3-split mean, excludes `test_geom_camber_cruise`)** = **100.8314**
- **`test_avg/mae_surf_p` (all 4 splits, raw)** = NaN — pre-existing data
  bug: `test_geom_camber_cruise/000020.pt` has 761 nodes with `inf` in
  pressure y. Affects every arm in this round; `data/scoring.py` is
  marked read-only. See PR #1397 comment for full trace and proposed
  fixes. Until resolved we record the 3-split test mean.

Per-split surface pressure MAE at the best val checkpoint:

| Split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 127.371 |
| val_geom_camber_rc     | 110.832 |
| val_geom_camber_cruise |  77.353 |
| val_re_rand            |  88.273 |
| **val_avg/mae_surf_p** | **100.957** |
| test_single_in_dist    | 116.622 |
| test_geom_camber_rc    |  97.209 |
| test_geom_camber_cruise| NaN (data bug, surf_Ux/Uy still ok) |
| test_re_rand           |  88.663 |
| **test_avg/mae_surf_p (3-split)** | **100.831** |

- **Metric artifacts**:
  `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`
  and `metrics.yaml`.
- **n_params**: 0.66M, **peak GPU memory**: 42.1 GB, **wall time**: 30.7 min.

## Reproduce baseline

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <student>/baseline
```
