# lr=5e-5 on n_layers=2+slice_num=16+epochs=46: LR lower bound at new stack (retry)

## Hypothesis

This is a retry of the original lr=5e-5 hypothesis from the stale PR #2549 (closed due to rate-limit-stuck pod). The hypothesis remains valid and the experimental setup is unchanged.

The LR axis at n_layers=2 currently has:
- lr=1e-4 (baseline, val=35.256)
- lr=1.5e-4 (PR #2525, +3.30% LOSS) — OOD-dominated loss
- lr=1.5e-4 + wd=3e-4 (askeladd #2601 fresh, in-flight compound)

lr=5e-5 brackets the LR axis on the lower side:
1. **Lower LR → tighter OOD convergence**: Lower LR is generally associated with better generalization. At n_layers=2 (where OOD is the bottleneck), a halved LR may help. 50% LR reduction is well above the seed-variance noise floor (~±1.0 val units per PR #2523).
2. **Smaller model may prefer lower LR**: n_layers=2 has 361K params vs the older 515K at n_layers=3. Smaller models often prefer lower LR (each step has more leverage; overshoot is more damaging).
3. **In-dist vs OOD tradeoff at low LR**: Per PR #2525, higher LR hurt OOD. Mirroring this: lower LR may help OOD at small cost to in-dist. Since OOD dominates avg, this could be net positive.
4. **Final cosine LR**: T_max=46 with peak LR=5e-5 gives final LR of cos(π)*5e-5 = 0, so model still converges to zero by the end. Late-stage training is slow but precise.

## Instructions

Single flag change from PR #2468 winner: `--lr 5e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr5e-5-nlayers2-slicenum16-epochs46-r2 \
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
3. **single_in_dist val mae_surf_p** — does the in-dist regression at n_layers=2 worsen further with lower LR (mirror of #2525 in-dist gain)?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does lower LR help OOD generalization?
5. Best epoch — does lower LR push best_epoch earlier or to final?
6. Train loss epochs 1-5: slower descent vs lr=1e-4?
7. Total wall-clock, peak memory

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
