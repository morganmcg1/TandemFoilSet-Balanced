# n_layers=2+slice_num=16+epochs=50: extend depth-down winner

## Hypothesis

Direct extension of PR #2468 (val=35.256, new baseline). best_epoch=46/46 was still descending at epoch 46 with slope ~−0.1/epoch. The run finished 3.1 min under the 30-min cap (26.9 min total). Pushing to epochs=50 uses that margin.

Estimated gain from the current local slope (~−0.1/epoch):
- epochs 47–50 = ~4 epochs × −0.1/epoch = ~0.4 val improvement
- Projection: ~34.85 (optimistic, if slope holds)

At 35s/epoch × 50 epochs = 29.2 min — safely inside 30-min cap.

**Only change from winning config: `--epochs 50`.**

## Instructions

Single flag change from PR #2468 winner: `--epochs 50`. ALSO update T_max to match: cosine annealing uses T_max=epochs automatically.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name nlayers2-slicenum16-epochs50 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

Expected: 35s/epoch × 50 epochs = 29.2 min (0.8 min margin under 30-min cap).

**Wall-clock gate:** If epoch 1 wall-clock > 36s, reduce to 48 epochs (36×48=28.8 min, safe). If > 37s, reduce to 46 (same as previous winner, will tie rather than improve).

## Reporting

1. **Epoch 1 wall-clock** (calibrate against 35s expectation)
2. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
3. Epochs 45–50 val trajectory (is the slope still ~−0.1/epoch?)
4. Best epoch, total wall-clock
5. **single_in_dist val mae_surf_p** — did the in-dist regression worsen, stay, or improve vs epoch 46?

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
