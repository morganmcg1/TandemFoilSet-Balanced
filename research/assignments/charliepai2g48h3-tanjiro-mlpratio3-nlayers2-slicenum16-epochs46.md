# mlp_ratio=3 on n_layers=2+slice_num=16+epochs=46: FFN width at new stack

## Hypothesis

mlp_ratio axis was characterized at n_layers=3: mlp_ratio=2 (+2.3% loss, PR #2350), mlp_ratio=4 (winner, current default), mlp_ratio=6 (+5.4% loss, PR #2278). mlp_ratio=3 — the intermediate value between 2 and 4 — has never been tested at any depth.

**Key motivation:** At n_layers=2 (shallow), the model has fewer transformer blocks to express the function, so each block's FFN capacity matters more. The optimum mlp_ratio may shift with depth:
1. **Compensation hypothesis**: Shallower depth → larger ratio could help. But mlp_ratio=6 lost at n_layers=3, so blowing up FFN width is bad.
2. **Optimum may shift DOWN**: At shallower depth with fewer parameters total (361K), a smaller mlp_ratio could reduce parameter waste and allow other capacity dimensions to do more work. mlp_ratio=3 (vs 4) reduces FFN params by 25%, freeing budget.
3. **Targets OOD bottleneck**: Per PR #2525, OOD is the bottleneck at n_layers=2. Wider FFN (mlp_ratio=6) overfit. Narrower FFN (mlp_ratio=3) may be a useful regularizer for OOD generalization.
4. **Single untested intermediate point**: mlp_ratio=3 has never been tested. It's the only unfilled gap in the {2, 3, 4, 6} sweep.

## Instructions

Single flag change from PR #2468 winner: `--mlp_ratio 3`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name mlpratio3-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16 \
  --mlp_ratio 3
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does narrower FFN help in-dist regression?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does narrower FFN help OOD?
5. Best epoch, total wall-clock, peak memory
6. Param count vs baseline (expect ~25% FFN param reduction)

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
