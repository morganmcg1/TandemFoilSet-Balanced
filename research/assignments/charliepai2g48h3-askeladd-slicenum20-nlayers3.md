# slice_num=20 on n_layers=3+epochs=34: fine-grain partition sweep at compact stack

## Hypothesis

The partition-sweep mechanism is the dominant lever. The ladder so far:
- slice_num=32 (PR #2107): val=39.143 at epochs=27
- slice_num=24 (PR #2229): val=37.366 at epochs=33 ← current baseline
- slice_num=20 (this run): hypothesized val ~36.9 at epochs=34
- slice_num=16 (alphonse #2348, fixed body, in flight): hypothesized val ~36.5 at epochs=36
- slice_num=12 (tanjiro #2351, floor probe, in flight): unknown

This fills in the partition sweep between current best 24 and the active probe 16. Per-epoch estimate: slice_num=20 → ~52s/epoch → ~34 epochs in 30 min.

**Why this matters:** If slice_num=16 wins, slice_num=20 may also win (fine-grain bracket). If slice_num=16 loses on capacity, slice_num=20 could be the next sweet spot below 24. Either way the result is informative.

## Instructions

Single flag change: `--slice_num 20`. Set epochs=34. If epoch 1 wall-clock > 53s, reduce to 33.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name slicenum20-nlayers3 \
  --epochs 34 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 20
```

## Reporting

1. **Epoch 1 wall-clock** — critical for budget calibration
2. Per-split val/test mae_surf_p vs baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Per-epoch wall-clock last 5, best epoch, total
5. Parameter count (~514K), peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
