# lr=5e-5 on n_layers=3+slice_num=24+epochs=33: bracket LR lower bound at compact stack

## Hypothesis

LR axis bracketing at compact stack (n_layers=3, slice_num=24):
- lr=2e-4 (#2367): val=39.028 (+4.4% worse) — LR ceiling confirmed below 2e-4
- lr=1.5e-4 (#2353, thorfinn in flight): testing upper-middle bracket
- **this run: lr=5e-5** → tests the lower bound

If lr=5e-5 wins: LR trough is below 1e-4, axis has more to explore downward.
If lr=5e-5 loses: lr=1e-4 confirmed at/near trough; LR axis fully closed.

## Instructions

Single flag change: `--lr 5e-5`. Same config as PR #2229 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name lr-5e-5-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (val=37.366 / test=31.371)
2. Per-split mae_vol_p
3. Train loss epochs 1-5 — does lr=5e-5 converge more slowly early on?
4. Best epoch (earlier or later than baseline's epoch 33?)
5. Per-epoch wall-clock (~53.7s expected), parameter count, peak memory

## Baseline (PR #2229)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
