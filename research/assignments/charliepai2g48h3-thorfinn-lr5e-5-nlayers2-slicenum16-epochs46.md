# lr=5e-5 on n_layers=2+slice_num=16+epochs=46: LR lower bound at new depth stack

## Hypothesis

The new best stack (PR #2468) is n_layers=2+slice_num=16+epochs=46, val=35.256. LR has been tested only at n_layers=3. This run completes the LR axis at the new depth alongside askeladd's lr=1.5e-4 upper probe.

Evidence and reasoning:
1. **LR optimum shifts with depth**: Smaller models (n_layers=2 has 30% fewer params: 361K vs 515K) sometimes prefer LOWER LR because each step has more leverage and overshoot is more damaging.
2. **In-dist regression at n_layers=2**: single_in_dist got worse (+1.21) at n_layers=2+lr=1e-4. A LOWER LR might let the smaller model converge more carefully on the in-distribution manifold.
3. **Cosine T_max=46 with low LR**: A halved peak LR with cosine annealing still covers significant parameter space — the model has 46 epochs to converge. Final LR will be lower (cosine to 0).
4. **Brackets the LR axis**: With askeladd at lr=1.5e-4 and this at lr=5e-5, we have a 3× span across the LR axis at the new depth, giving clean signal on the optimum location.

## Instructions

Single flag change from PR #2468 winner: `--lr 5e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr5e-5-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — did the in-dist regression (36.476 at lr=1e-4) improve with lower LR?
4. Train loss epochs 1-5: slower descent vs lr=1e-4?
5. Best epoch (does lower LR push best_epoch earlier or final?)
6. Total wall-clock, peak memory

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
