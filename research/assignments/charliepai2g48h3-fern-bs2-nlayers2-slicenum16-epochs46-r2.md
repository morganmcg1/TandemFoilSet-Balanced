# batch_size=2 on n_layers=2+slice_num=16+epochs=46: gradient-noise regularization (RETRY)

## Hypothesis

**This is a retry of PR #2636 (closed as stale_wip — pod stuck in rate-limit polling). The hypothesis is unchanged.**

**batch_size is the most-untested axis at the n_layers=2 stack.** Every variant since Round 32 has held bs=4 fixed. Both bs=2 (this PR) and bs=8 (tanjiro #2693) test this axis cleanly.

**Why batch_size=2 might help at n_layers=2:**

1. **OOD bottleneck context (PR #2525, #2601, #2638)**: OOD splits dominate val_avg at n_layers=2. Lower batch size injects more gradient noise per step, which acts as an implicit regularizer — known to favor flatter minima.
2. **Per Round 37 #2638 split-dependent OOD finding**: re_rand is regularization-friendly. bs=2 should help re_rand (additional regularization effect from noisier gradients).
3. **More gradient updates per epoch**: bs=4→2 doubles the steps/epoch. With cosine LR schedule (T_max=46), this means finer-grained LR descent.
4. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. bs=4→2 is a substantial gradient-noise change — expect signal above noise floor.

**Wall-clock gate:** If epoch 1 wall-clock > 45s, reduce epochs to 35. If > 50s, reduce to 30.

## Instructions

Single flag change from PR #2468 winner: `--batch_size 2`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name bs2-nlayers2-slicenum16-epochs46-r2 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 2 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits** — KEY: per #2638 prediction, re_rand should improve (regularization), geom_camber may not respond (capacity-limited)
4. **single_in_dist** — does in-dist suffer from gradient noise?
5. Best epoch — does smaller batch shift best_epoch?
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
