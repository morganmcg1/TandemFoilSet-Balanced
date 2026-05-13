# surf_weight=8 on n_layers=3+slice_num=16+epochs=36: surf_weight axis at new optimum

## Hypothesis

surf_weight was last tested at slice_num=24 (old baseline). Like LR, the optimal surf_weight may be partition-dependent. With slice_num=16 now the proven optimum (PR #2348), the surf/vol loss balance may shift.

Prior surf_weight evidence (at OLDER stacks):
- sw=2 (PR #2248): LOST significantly at slice_num=24
- sw=10 (current baseline): val=35.548 at slice_num=16
- sw=12, sw=16: tested at slice_num=24, none won

**Why surf_weight=8 might help at slice_num=16:**
- slice_num=16 has finer geometric resolution than slice_num=24 — vol gradients may now carry more useful signal relative to surf gradients.
- Reducing surf_weight from 10→8 lets the optimizer respond more to vol_p errors, potentially regularizing the surface predictions implicitly.
- Coarsely-partitioned models (slice_num=24) likely needed strong surf weighting to compensate for blurry geometric features. Finer partitions (slice_num=16) may benefit from a more balanced surf/vol loss.

## Instructions

Single flag change from baseline: `--surf_weight 8`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name surfweight8-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 8 \
  --n_layers 3 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (val=35.548 / test=30.345)
2. Per-split mae_vol_p — especially check if vol metrics improve significantly (mechanism canary)
3. Best epoch (expected near final), total wall-clock, peak memory
4. Train loss trajectory epochs 1-5 (compare to lr=1e-4 sw=10 baseline)

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
