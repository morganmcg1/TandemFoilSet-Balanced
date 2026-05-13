# epochs=30 on n_layers=3+slice_num=32: squeeze remaining budget headroom

## Hypothesis

The current best (PR #2107: n_layers=3 + slice_num=32, val=39.143) terminated at best_epoch=27/27 STILL DESCENDING. Per-epoch ~57s, 27 epochs = 25.6 min — 4.4 min headroom remains. T_max auto-aligns to epochs in train.py. At 57s/epoch: 30 epochs × 57s = 28.5 min (within budget).

**Prediction:** val < 39.143.

## Instructions

Change ONLY `--epochs` from 27 to 30. Use EXACT same config as PR #2107.

Do NOT pass `--n_head` (default n_head=4 is the current best).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name epochs-30-nlayers3-slicenum32 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 32
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=39.143 / test=33.571)
2. Per-split `mae_vol_p`
3. Per-epoch wall-clock (epoch 1 and last)
4. Per-epoch val_avg at epochs 27–30
5. Best epoch, total wall-clock, epochs completed
6. Parameter count (~515K), peak memory

## Baseline (PR #2107: n_layers=3 + slice_num=32 + epochs=27)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.405 | 35.977 |
| geom_camber_rc | 51.895 | 47.136 |
| geom_camber_cruise | 22.756 | 19.101 |
| re_rand | 41.517 | 32.070 |
| **avg** | **39.143** | **33.571** |

**Target to beat:** `val_avg/mae_surf_p < 39.143`

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
