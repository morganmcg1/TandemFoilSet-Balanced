# slice_num=20 on n_layers=2+epochs=44: Partition retest at new depth (counter in-dist regression)

## Hypothesis

slice_num=20 was tested at n_layers=3+epochs=36 (PR #2475: val=36.835, slightly worse than 35.548). The new best stack (PR #2468) is n_layers=2+slice_num=16+epochs=46, val=35.256.

The depth-down + epoch-up mechanism produced gains primarily on OOD splits (camber_rc, camber_cruise, re_rand) but **regressed single_in_dist** (+1.21). slice_num=20 may be the right partition for the n_layers=2 model to recover the in-distribution patterns.

Reasoning:
1. **Partition optimum is depth-dependent**: At n_layers=3, slice_num=16 won over 20. At n_layers=2, the lower-capacity model may benefit from MORE partitions (finer granularity = more attention heads to learn separately) to compensate for fewer transformer layers. The optimum may shift to slice_num=20.
2. **Counter single_in_dist regression**: A finer slice partition might let the model represent the dense in-distribution geometries better, recovering some of the in-dist accuracy lost at n_layers=2+slice_num=16.
3. **Budget accounting**: slice_num=20 at n_layers=2 should run ~38s/epoch (vs 35s at slice_num=16). At 44 epochs × 38s = 27.9 min, safely inside 30-min cap.

## Instructions

Two flag changes from PR #2468 winner: `--slice_num 20` AND `--epochs 44` (budget reduction to stay within timecap).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name slicenum20-nlayers2-epochs44 \
  --epochs 44 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 20
```

**Wall-clock gate:** If epoch 1 wall-clock > 40s, reduce epochs to 42 (42×40=28.0 min, safe).

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — KEY METRIC: did slice_num=20 recover the in-dist regression?
4. Best epoch, total wall-clock, epoch-1 wall-clock vs 38s expectation
5. Peak memory

## Baseline (PR #2468)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.476 | 33.035 |
| geom_camber_rc | 48.297 | 44.333 |
| geom_camber_cruise | 18.326 | 15.496 |
| re_rand | 37.923 | 28.116 |
| **avg** | **35.256** | **30.245** |

**Reproduce baseline:**
```bash
cd target/ && python train.py \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --n_layers 2 --slice_num 16
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
