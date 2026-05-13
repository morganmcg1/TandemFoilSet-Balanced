# slice_num=16 on n_layers=3+epochs=36: continue partition sweep

## Hypothesis

PR #2229 confirmed the partition-sweep mechanism at n_layers=3+slice_num=24: 57s→53.7s/epoch allowed 33 epochs in the 30-min cap (vs 27 epochs at slice_num=32). All four splits improved; best_epoch=33/33 STILL DESCENDING.

The mechanism is not saturated. slice_num=24 per-epoch estimate (~53.7s) still has headroom:
- slice_num=16 estimate: ~50s/epoch → ~36 epochs in 30 min (T_max auto-aligns)
- Epoch gain: 33→36 = 3 more descent steps at the tail of the cosine schedule
- Previous descent rate in ep33 tail: ~0.4-0.5 val/epoch → ~1.2-1.5 further improvement possible

If slice_num=16 wins: mechanism still binding, probe slice_num=12 next.
If slice_num=16 loses: slice_num=24 is the optimal; partition axis saturated.

**Risk:** At slice_num=16, PhysicsAttention has fewer tokens per slice — representational capacity of the attention may degrade on the harder splits (geom_camber_rc). If it does, this tells us the capacity floor is slice_num=24.

## Instructions

Single flag change: `--slice_num 16`. Adjust epochs to `--epochs 36` to use the freed budget (verify epoch 1 wall-clock first; if >52s, reduce to `--epochs 34`).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

## Reporting

1. **Epoch 1 wall-clock** — critical for budget calibration
2. Per-split val/test mae_surf_p vs NEW baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Per-epoch wall-clock last 5 epochs, best epoch, total
5. Parameter count (~514K expected), peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
