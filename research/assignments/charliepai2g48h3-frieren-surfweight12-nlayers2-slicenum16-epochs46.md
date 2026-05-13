# surf_weight=12 on n_layers=2+slice_num=16+epochs=46: Higher loss-weight at new stack

## Hypothesis

surf_weight=12 has never been tested at any depth/partition. Existing surf_weight knowledge:
- sw=10 (baseline): optimal at all stacks tested
- sw=8 (fern #2570 in-flight at new stack)
- sw=15: neutral at n_layers=4
- sw=5/3/2: lost at n_layers=3 stack (+2.5% to +3.9%)

**Critical context — seed variance insight from PR #2523:**
Single-seed comparisons of small (~5%) changes are below the noise floor (~1.0 val units). A 20% change in surf_weight should produce signal above noise.

**Why surf_weight=12 might win:**
1. **Bigger gradient signal on surface points**: The metric we optimize for IS surface MAE. Increasing surface loss weight from 10→12 directly weights gradients toward improving surface predictions.
2. **OOD bottleneck targeted**: PR #2525 showed OOD splits dominate the loss at n_layers=2. The OOD splits (geom_camber_rc=48.3, cruise=18.3, re_rand=37.9) are all surface MAE measurements — stronger surface focus may directly help.
3. **sw=15 was neutral at n_layers=4**: sw=15 didn't help at deeper stack. But at the shallower n_layers=2 stack, the model has less capacity to "waste" on vol predictions, so increased surface focus may produce a cleaner signal.
4. **Untested gap**: sw=10 (winning), sw=15 (neutral) — sw=12 is the obvious untested intermediate point.

## Instructions

Single flag change from PR #2468 winner: `--surf_weight 12`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name surfweight12-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 12 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p — KEY: how much does higher surf_weight worsen vol predictions?
3. **single_in_dist val mae_surf_p** — does sw=12 help in-dist?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does stronger surface focus help OOD generalization?
5. Best epoch, total wall-clock, peak memory
6. **Acknowledge seed variance**: This is a 20% surf_weight change, expected to produce signal above noise floor. Compare to baseline using the (val, test) signs together as evidence.

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
