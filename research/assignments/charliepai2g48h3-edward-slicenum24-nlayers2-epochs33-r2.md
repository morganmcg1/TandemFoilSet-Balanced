# slice_num=24+epochs=33 on n_layers=2: Wider partition retest at new depth (retry)

## Hypothesis

**This is a retry of the slice_num=24 hypothesis from the stale PR #2609 (closed due to rate-limit-stuck pod). The hypothesis remains valid; the experimental setup is unchanged.**

The partition axis was fully closed at n_layers=3 (slice_num=24 → 35.969, slice_num=16 → 35.548). At n_layers=2, the partition optimum may shift because:

1. **Per-block routing capacity matters more at shallow depth**: With only 2 transformer blocks, each PhysicsAttention's slice routing must do more work per layer. Wider slicing (24) gives more attention pathways and could help OOD by improving feature diversity.

2. **OOD bottleneck context (Round 37 #2638 insight)**: The geom_camber OOD splits are **capacity-limited**, not regularization-limited. Wider partitioning provides more capacity per block — directly testing the capacity hypothesis on the slice axis.

3. **Variance context (PR #2523)**: Run-to-run variance ~±1.0 val units. slice_num=16→24 is a +50% partition change — well above the noise floor.

4. **Wall-clock**: slice_num=24 at n_layers=2 estimated ~50s/epoch (vs 35s at slice_num=16). 33 epochs × 50s = 27.5 min, safely inside 30-min cap.

**Wall-clock gate:** If epoch 1 wall-clock > 52s, reduce epochs to 30 (30×52=26 min, safe). If > 54s, reduce to 28.

## Instructions

Two flag changes from PR #2468 winner: `--slice_num 24 --epochs 33`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name slicenum24-nlayers2-epochs33-r2 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 24
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY per #2638: does wider partition help geom_camber (the capacity-limited splits)?
4. **single_in_dist** — does the in-dist regression close with wider slicing?
5. Best epoch, total wall-clock, peak memory
6. Param count change

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
