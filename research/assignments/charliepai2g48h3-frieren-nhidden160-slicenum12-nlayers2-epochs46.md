# n_hidden=160 + slice_num=12 + epochs=46 on n_layers=2: ISO-EPOCH capacity test

## Hypothesis

**Follow-up to your own PR #2685 closure (your suggestion #1).** The previous experiment confounded `n_hidden=160` (+capacity) with `epochs=40` (vs baseline's 46) due to wall-clock cap. Best_epoch=40 (final, still descending at ~0.13/epoch) proved the model was epoch-starved at the new capacity.

**This experiment isolates the capacity effect by clawing back per-epoch time via `slice_num=12`**, allowing us to match baseline's epoch count.

**Wall-clock budget reasoning:**
- Previous run: ~42.4 s/epoch at n_hidden=160 + slice_num=16
- slice_num=16 → 12 is a 25% reduction in attention compute → ~32 s/epoch (your prediction)
- 46 epochs × 32 s = 24.5 min, comfortably under cap

**Why iso-epoch capacity test matters:**

1. **#2638 split-dependent OOD finding**: geom_camber OOD is capacity-limited (your own diagnostic). The capacity hypothesis has not been refuted — it has been incompletely tested at unequal epoch budget.
2. **Best_epoch=final has held for 13+ baselines**: the model is *always* training-time-limited. Adding capacity without adding training time was the confound.
3. **slice_num=12 at n_layers=3 lost (PR #2351, +1.18% val)**: BUT was at slower per-step time without capacity bump compensating. At n_layers=2 with n_hidden=160 boosting per-block capacity, the slice_num=12 might transfer differently — and the iso-epoch comparison to baseline is cleaner.
4. **Risk acknowledged**: this compounds two changes (n_hidden=160 + slice_num=12). If it wins, follow-up needed to disentangle (slice_num=12 alone at n_layers=2 — askeladd #?? is testing this in parallel).
5. **Variance context**: ~±1.0 val seed-variance noise floor. A clean iso-epoch capacity gain should be well above noise.

**Wall-clock gate:** If epoch 1 wall-clock > 35s, reduce epochs to 42 (42×35=24.5 min). If > 38s, reduce to 39. If > 42s, abort and report — slicing reduction didn't claw back enough time.

## Instructions

Three flag changes from PR #2468 winner: `--n_hidden 160 --slice_num 12 --epochs 46`. (The --n_hidden CLI flag was already added in your previous PR #2685 — verify the branch picks it up.)

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name nhidden160-slicenum12-nlayers2-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 12 \
  --n_hidden 160
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits, especially geom_camber_rc and geom_camber_cruise** — KEY: at iso-epoch budget, does capacity bump rescue the camber OOD bottleneck?
4. **single_in_dist** — test split improved -2.49% in your prior run; does it hold or improve at iso-epoch?
5. **Best epoch** — is best_epoch=46 (final)? If yes, capacity + iso-epochs is still epoch-starved (need more reduction).
6. Per-epoch wall-clock at n_hidden=160 + slice_num=12
7. Param count and peak memory (expect ~560K params, ~16-17 GB)
8. Total wall-clock

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
