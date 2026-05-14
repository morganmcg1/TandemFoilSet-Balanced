# n_layers=1 + epochs=60 on slice_num=16: Extreme depth-down + epoch-up

## Hypothesis

**Continues the depth-down + epoch-up trajectory.** The mechanism has been monotonic: n_layers=6→5→4→3→2, each step a big gain by freeing per-epoch budget for cosine annealing. n_layers=1 is the next bold extension.

**Why n_layers=1 might extend the win:**

1. **Per-epoch saving**: n_layers=2 → ~35s/epoch. n_layers=1 estimated ~22-25s/epoch (single transformer block + decoder). Each epoch saves ~30-40% wall-clock → ~60 epochs fit within 30-min cap (vs 46 at n_layers=2).
2. **best_epoch=46 (final) at baseline**: Model is still descending at the budget cap. More epochs is the dominant lever per the trajectory history.
3. **Capacity risk acknowledged**: n_layers=1 may be insufficient for the task. The same was said about n_layers=2 vs 3 — and it worked. The 4 val/test split metrics will diagnose: if OOD splits crater, capacity floor reached. If they're similar to baseline, depth-down mechanism still has room.
4. **Combines with Round 37 #2638 insight**: The student found geom_camber OOD is capacity-limited. n_layers=1 reduces depth-capacity but increases epoch-budget. Tests whether more epochs trumps lost layer capacity.
5. **Variance context**: With epochs jumping from 46→60 (+30%), expect signal well above ±1.0 seed-variance noise floor.

**Wall-clock gate:** If epoch 1 wall-clock > 28s, reduce epochs to 55 (55×28=25.6 min). If > 32s, reduce to 50 (50×32=26.7 min). Goal: stay under 28 min total to leave buffer.

## Instructions

Two flag changes from PR #2468 winner: `--n_layers 1 --epochs 60`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name nlayers1-slicenum16-epochs60 \
  --epochs 60 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 1 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits** — KEY: do geom_camber splits crater (capacity floor) or stay close (still in epoch-budget regime)?
4. **single_in_dist** — does losing 1 layer help (less overfitting) or hurt (less capacity)?
5. Best epoch — is best_epoch=60 (final)? If yes, more epochs may still help.
6. Per-epoch wall-clock
7. Param count vs baseline (expect ~30-40% reduction)
8. Total wall-clock, peak memory

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
