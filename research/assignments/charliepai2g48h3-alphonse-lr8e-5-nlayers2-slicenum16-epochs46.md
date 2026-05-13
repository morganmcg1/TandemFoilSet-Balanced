# lr=8e-5 on n_layers=2+slice_num=16+epochs=46: LR low-side fine probe at new stack

## Hypothesis

The LR axis at n_layers=2 currently has only 2 confirmed data points:
- lr=1e-4 (baseline, val=35.256)
- lr=1.5e-4 (PR #2525, +3.30% LOSS) — the loss is OOD-dominated; in-dist actually IMPROVED

lr=8e-5 fills the gap between baseline and any lower bound:
1. **Lower LR for tighter OOD convergence**: PR #2525 showed lr=1.5e-4 helped in-dist but hurt OOD (per-split: in-dist −6.35%, OOD +3-10%). A LOWER LR may have the opposite trade — slightly worse in-dist but better OOD. Since OOD dominates the avg metric, this could be net positive.
2. **20% LR reduction is borderline signal**: PR #2523 revealed seed variance ~±1.0 val units. 20% LR change is at the noise floor edge — interpret with care, but cleanly above noise if combined with test metric.
3. **Bracket the LR axis**: With askeladd's #2601 testing higher LR (1.5e-4) compound, this provides the lower-side probe. Together they characterize the full LR curve at the new stack.

## Instructions

Single flag change from PR #2468 winner: `--lr 8e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name lr8e-5-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 8e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does lower LR worsen in-dist as expected (mirror of lr=1.5e-4 result)?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does lower LR help OOD?
5. Best epoch, total wall-clock, peak memory
6. **Variance acknowledgment**: 20% LR change is at the noise floor edge. Use both val AND test consistency to assess significance.

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
