# Baseline Metrics

## Current Baseline

**Status**: No experiments completed yet (round 1 in progress).

The baseline code is the unmodified Transolver in `train.py`:
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1M params)
- `AdamW(lr=5e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=epochs)`
- `batch_size=4`, `surf_weight=10.0`
- MSE loss in normalized space

**Primary metric to beat**: `val_avg/mae_surf_p` — will be established once round 1 results arrive.

---

## Update Log

| Date | PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|------|----|--------------------|---------------------|-------|
| — | — | — | — | Awaiting round 1 |
