# weight_decay=5e-5 on n_layers=3+slice_num=16+epochs=36: WD axis at new optimum

## Hypothesis

Weight decay was tested only at slice_num=24 (old baseline):
- wd=0 (PR #2274): LOST +2.2% at slice_num=24
- wd=1e-4 (current baseline): val=35.548 at slice_num=16

Two reasons wd=5e-5 might win at slice_num=16:
1. **Capacity-regularization balance shifts with partition**: At slice_num=16 the model is underfit (best_epoch=final at every run). Reducing regularization slightly may help the model fit more aggressively within the budget while avoiding the WD=0 collapse seen at slice_num=24.
2. **Partition-dependent optimum**: Like LR and (potentially) surf_weight, weight_decay optimum may shift with slice_num. wd=0 was decisively bad at slice_num=24, but wd=5e-5 (halfway between 0 and 1e-4) may be a sweet spot at the new partition.

## Instructions

Single flag change from baseline: `--weight_decay 5e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name wd5e-5-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 5e-5 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (val=35.548 / test=30.345)
2. Per-split mae_vol_p
3. Best epoch (expected final; does lower WD push best_epoch earlier?)
4. Train loss trajectory epochs 1-5, peak memory, wall-clock per epoch

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |

**Reproduce baseline:**
```bash
cd target/ && python train.py \
  --epochs 36 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --n_layers 3 --slice_num 16
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
