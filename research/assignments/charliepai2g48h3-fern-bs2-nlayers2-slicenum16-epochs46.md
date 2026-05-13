# batch_size=2 on n_layers=2+slice_num=16+epochs=46: Smaller batch noise at new stack

## Hypothesis

**batch_size is the most-untested axis at the n_layers=2 stack.** Every variant since Round 32 has held bs=4 fixed. Both bs=2 (fern) and bs=8 (tanjiro #2613) test this axis cleanly.

**Why batch_size=2 might help at n_layers=2:**

1. **OOD bottleneck context (PR #2525, #2601)**: OOD splits dominate val_avg at n_layers=2. Lower batch size injects more gradient noise per step, which acts as an implicit regularizer — known to favor flatter minima and improve OOD generalization.
2. **More gradient updates per epoch**: bs=4→2 doubles the steps/epoch. With cosine LR schedule (T_max=46), this means finer-grained LR descent. Effective LR per sample stays the same.
3. **Surface-loss SNR**: surf_weight=10 puts heavy weight on surface losses; smaller batch may improve surface-prediction stability because each batch contains fewer geometrically-correlated samples, reducing intra-batch correlation in surface gradient estimates.
4. **Wall-clock budget**: bs=2 doubles step count but each step is faster. Expect ~50s/epoch, 46×50=38 min — may need to drop to epochs=35 if epoch wall-clock >40s.
5. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. bs=4→2 is a substantial gradient-noise change — expect signal well above noise floor.

**Wall-clock gate:** If epoch 1 wall-clock > 45s, reduce epochs to 35 (35×45=26 min, safe). If > 50s, reduce to 30.

## Instructions

Single flag change from PR #2468 winner: `--batch_size 2`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name bs2-nlayers2-slicenum16-epochs46 \
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
3. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does gradient noise help OOD generalization?
4. **single_in_dist** — does in-dist suffer from increased gradient noise?
5. Best epoch — does smaller batch shift best_epoch earlier (faster convergence) or later (slower with noisier updates)?
6. Train loss curve epochs 1-5: faster initial descent vs bs=4?
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
