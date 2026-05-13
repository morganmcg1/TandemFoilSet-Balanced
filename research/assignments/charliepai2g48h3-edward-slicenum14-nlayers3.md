# slice_num=14 on n_layers=3+epochs=36: partition neighborhood probe

## Hypothesis

Probes the partition neighborhood around the current local optimum (slice_num=16).

Sweep so far:
- slice_num=24 → val=37.366 (PR #2229)
- slice_num=16 → val=35.548 (PR #2348, CURRENT BASELINE)
- slice_num=12 → val=35.969 (PR #2351, WORSE than 16 — non-monotone)

Per-epoch cost is flat across 12–16 (~50s/epoch). Testing 14 fills the gap and determines whether 16 is the exact local optimum or the floor lies slightly lower.

## Instructions

Single flag change: `--slice_num 14`. Use `--epochs 36`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-edward \
  --experiment_name slicenum14-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 14
```

## Reporting

1. Epoch 1 wall-clock (expected ~50s)
2. Per-split val/test mae_surf_p vs baseline (val=35.548 / test=30.345)
3. Per-split mae_vol_p
4. Best epoch, total wall-clock, peak memory
5. **Capacity canary:** if geom_camber_rc val mae_vol_p rises sharply vs baseline (~53.6), report early

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
