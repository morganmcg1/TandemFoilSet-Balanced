# surf_weight=8 on n_layers=2+slice_num=16+epochs=46: Loss-weight axis at new stack

## Hypothesis

surf_weight axis was tested only at older stacks (n_layers=3 / 4 / 5). The new winning stack is n_layers=2+slice_num=16+epochs=46 (val=35.256). surf_weight=8 is genuinely untested at this depth.

**Key motivation from PR #2525 (closed lr=1.5e-4 loss):** The lr=1.5e-4 result revealed the n_layers=2 baseline has a clear in-dist-vs-OOD tradeoff — single_in_dist improved with higher LR but ALL 3 OOD splits got worse. This means OOD generalization dominates the avg metric loss.

Lowering surf_weight from 10 → 8 may help OOD by:
1. **Rebalancing loss focus**: Less aggressive weighting of surface points reduces pressure on overfitting to in-distribution surface patterns. The model attends more to volume points which carry richer geometric structure that may generalize better.
2. **Indirect OOD regularization**: Reducing surf_weight has a regularizing effect (lower magnitude of surface-loss gradient signal). At the smaller n_layers=2 capacity, mild loss-magnitude regularization may help generalization without hurting fit.
3. **Untested at new depth**: At n_layers=3 stack, sw axis was characterized; sw=5, 3, 2 lost (+2.6%, +3.9%, +3.1%) but those were at older stacks. The optimum sw may shift modestly with depth.

## Instructions

Single flag change from PR #2468 winner: `--surf_weight 8`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name surfweight8-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 8 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p — KEY: did lower surf_weight improve mae_vol_p?
3. **single_in_dist val mae_surf_p** — does sw=8 close the in-dist regression at n_layers=2?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does lower surf_weight help OOD generalization (per PR #2525 hypothesis)?
5. Best epoch, total wall-clock, peak memory

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
