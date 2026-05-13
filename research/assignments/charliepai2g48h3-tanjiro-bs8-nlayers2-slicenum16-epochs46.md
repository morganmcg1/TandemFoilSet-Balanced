# batch_size=8 on n_layers=2+slice_num=16+epochs=46: Larger batch at new stack

## Hypothesis

**batch_size is the most-untested axis at the n_layers=2 stack.** Every variant since Round 32 has held bs=4 fixed. Both bs=8 (tanjiro) and bs=2 (fern #2612) test this axis cleanly.

**Why batch_size=8 might help at n_layers=2:**

1. **In-dist gradient quality**: bs=4→8 halves gradient variance. If in-dist learning at n_layers=2 is currently noise-limited (rather than capacity-limited), more accurate gradients → better optimum. PR #2525 showed in-dist val=34.16 was achievable with higher LR; this is a competing path to in-dist sharpness without the OOD damage.
2. **More samples / fewer steps / same total data**: 46 epochs at bs=8 = 1820 batches (vs 3640 at bs=4). Cosine LR schedule will reach min faster relative to wall-clock — different convergence trajectory.
3. **VRAM headroom**: At 13.5 GB peak for bs=4, bs=8 should still fit comfortably (we have 96 GB VRAM).
4. **Combined with fern bs=2 (#2612)**: Clean 3-point batch_size axis (2, 4, 8) at the new stack.
5. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. bs=4→8 is a substantial gradient-noise reduction — expect signal above noise floor.

**Wall-clock gate:** If epoch 1 wall-clock > 30s, no change (default 46 epochs). bs=8 is expected to be **faster** per epoch than bs=4 (fewer steps); should comfortably fit in <30 min.

## Instructions

Single flag change from PR #2468 winner: `--batch_size 8`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name bs8-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 8 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist** — does smoother gradient help in-dist (mirror of high-LR effect in #2525)?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does less gradient noise hurt OOD?
5. Best epoch — does larger batch shift best_epoch later (slower convergence) or earlier (better per-step quality)?
6. Train loss curve epochs 1-5: smoother descent vs bs=4?
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
