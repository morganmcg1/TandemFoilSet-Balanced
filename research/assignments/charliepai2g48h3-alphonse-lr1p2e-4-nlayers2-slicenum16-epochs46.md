# lr=1.2e-4 on n_layers=2+slice_num=16+epochs=46: LR fine probe at new depth stack

## Hypothesis

The new best stack (PR #2468) uses lr=1e-4 at n_layers=2. Alongside askeladd's lr=1.5e-4 (big-step) and thorfinn's lr=5e-5 (lower bound), this run tests a SMALL upward step (1.2× vs 1.5× and 0.5×). 

A finer LR sweep helps locate the optimum if it's near (but not exactly at) lr=1e-4. Reasoning:
1. **Optimum could be at modestly higher LR**: At n_layers=2 with fewer params (361K vs 515K), the model may want slightly more aggressive updates. lr=1.2e-4 lets us see if a small bump helps without overshooting.
2. **In-dist regression context**: single_in_dist worsened (+1.21) at n_layers=2+lr=1e-4. A modest LR bump may help in-distribution learning without destabilizing OOD generalization.
3. **Three-way LR sweep coverage**: Combined with askeladd (1.5e-4) and thorfinn (5e-5), this gives 4-point coverage (5e-5, 1e-4, 1.2e-4, 1.5e-4) — enough granularity to triangulate the optimum at the new depth stack.

## Instructions

Single flag change from PR #2468 winner: `--lr 1.2e-4`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name lr1p2e-4-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1.2e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — small LR bump helps in-dist regression?
4. Train loss epochs 1-5: faster vs lr=1e-4?
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
