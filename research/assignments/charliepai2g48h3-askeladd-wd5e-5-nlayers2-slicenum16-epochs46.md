# weight_decay=5e-5 on n_layers=2+slice_num=16+epochs=46: Lower WD bound

## Hypothesis

**Brackets the WD axis on the lower side.** Combined with frieren #2614 (wd=3e-4 alone), this gives a clean 3-point WD axis at the new stack: 5e-5 (askeladd), 1e-4 (baseline), 3e-4 (frieren).

**Why halved WD might help at n_layers=2:**

1. **Smaller model has less risk of overfitting**: n_layers=2 has 361K params (vs 515K at n_layers=3). Smaller models need less regularization to begin with. Lower WD may allow tighter fit without overfitting risk.
2. **In-dist headroom**: PR #2601 showed wd=3e-4 *hurts* in-dist (single_in_dist val 36.48 → 36.88). The mirror hypothesis: lower WD may improve in-dist (single_in_dist baseline=36.48 has slack vs higher-LR-achievable 34.16).
3. **OOD risk**: Lower WD = less flat minima preference → may hurt OOD. But the OOD splits already dominate the baseline — if in-dist gains are large enough, net could be positive.
4. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. WD 1e-4 → 5e-5 is a 50% reduction — expect signal above noise floor.
5. **Clean diagnostic**: If wd=5e-5 wins → less regularization is the right direction → consider wd=1e-5. If both wd=5e-5 (askeladd) and wd=3e-4 (frieren) lose → baseline WD=1e-4 is at the WD optimum.

## Instructions

Single flag change from PR #2468 winner: `--weight_decay 5e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name wd5e-5-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 5e-5 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist** — KEY: does halved WD help in-dist (mirror of #2601 in-dist regression under WD=3e-4)?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — does less regularization hurt OOD?
5. Best epoch — does weaker WD shift best_epoch earlier (faster convergence) or expose overfitting late?
6. Train loss curve epochs 1-5: faster descent vs WD=1e-4?
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
