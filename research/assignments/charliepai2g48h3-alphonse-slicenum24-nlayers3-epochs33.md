# slice_num=24 on n_layers=3: continue partition sweep at new best depth

## Hypothesis

The partition sweep has consistently freed per-epoch budget: slice_num=64→48→32, each step fitting more epochs in the 30-min cap. The current best is n_layers=3+slice_num=32+epochs=27 (val=39.143, best_epoch=27 STILL DESCENDING).

**slice_num=24** is the natural next point (25% fewer partitions). At n_layers=3+slice_num=32 per-epoch is ~57s. With slice_num=24:
- Per-epoch estimate: ~50s
- Budget: ~33 epochs in 30 min cap
- T_max auto-aligns to epochs in train.py

**Prediction:** val < 39.143 if epoch-budget mechanism remains binding.

## Instructions

Change `--slice_num` from 32 to 24 and `--epochs` from 27 to 33. Same config as PR #2107 otherwise.

Do NOT pass `--n_head` (default n_head=4).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name slicenum24-nlayers3-epochs33 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs baseline (val=39.143 / test=33.571)
2. Per-split `mae_vol_p`
3. **Epoch 1 wall-clock** — critical for budgeting slice_num=24 at n_layers=3
4. Last epoch wall-clock, total wall-clock, epochs completed
5. Best epoch — still final? Per-epoch trajectory last 5 epochs
6. Parameter count, peak memory

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
