# mlp_ratio=2 on n_layers=2+slice_num=16+epochs=46: Narrower FFN at new stack

## Hypothesis

mlp_ratio axis at n_layers=3 was closed: mlp_ratio=2 (+2.3% loss, PR #2350), mlp_ratio=4 (winner), mlp_ratio=6 (+5.4% loss, PR #2278). At the shallower n_layers=2 stack, the FFN axis is **fully untested**. tanjiro #2571 is testing mlp_ratio=3 (intermediate); this PR tests mlp_ratio=2 (bottom of axis).

**Why narrower FFN might help at n_layers=2:**
1. **Parameter budget reallocation**: At mlp_ratio=2 vs 4, FFN params drop by ~50%. The model has 361K total params at mlp_ratio=4; mlp_ratio=2 brings it to ~270K. This frees memory and compute, but mostly tests whether FFN capacity is actually load-bearing at this depth.
2. **OOD bottleneck targeting**: Per PR #2525, OOD splits dominate the loss at n_layers=2. Narrower FFN acts as a strong regularizer — may help OOD generalization by reducing memorization of in-distribution patterns.
3. **Variance context (PR #2523)**: Run-to-run variance ~±1.0 val units. mlp_ratio=4→2 is a 50% reduction — large signal expected.
4. **Combined with tanjiro #2571 (mlp_ratio=3)**: Provides 2-point lower-bound characterization of FFN axis at new stack. If both lose, FFN axis at n_layers=2 is closed-down. If one wins, we have new direction.

## Instructions

Single flag change from PR #2468 winner: `--mlp_ratio 2`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name mlpratio2-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16 \
  --mlp_ratio 2
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does narrow FFN help in-dist or hurt it?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does narrower FFN regularize OOD well?
5. Best epoch, total wall-clock, peak memory
6. Param count vs baseline (expect ~50% FFN param reduction; total ~270K vs 361K)

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
