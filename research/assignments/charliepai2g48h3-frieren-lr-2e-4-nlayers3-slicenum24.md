# lr=2e-4 on n_layers=3+slice_num=24+epochs=33: bracket LR axis upper bound at new baseline

## Hypothesis

The LR axis at the NEW baseline stack (n_layers=3+slice_num=24) is being bracketed:
- baseline lr=1e-4 (PR #2229) → val=37.366
- thorfinn #2353: lr=1.5e-4 (in flight) → tests if higher LR helps
- **this run: lr=2e-4** → tests the upper end of plausible LR

Lion + cosine + L1 loss has historically been LR-sensitive. At the smaller-model+longer-budget compact stack, the LR optimum may have shifted higher because:
- Smaller model (514K) → faster traversal of loss landscape → tolerates larger LR
- Longer cosine (T_max=33) → more aggressive decay toward end → can afford higher peak

If lr=2e-4 wins (vs baseline AND thorfinn's lr=1.5e-4 if available): LR axis still has room.
If lr=2e-4 plateaus or diverges early: the LR ceiling at compact stack is between 1.5e-4 and 2e-4.

**Risk:** Lion + lr=2e-4 may diverge early. If train loss diverges in epoch 1-3, report and abort.

## Instructions

Single flag change: `--lr 2e-4`. Same config as PR #2229 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name lr-2e-4-nlayers3-slicenum24 \
  --epochs 33 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 24
```

## Reporting

1. **Train loss epochs 1-5** — critical for divergence detection
2. Per-split val/test mae_surf_p vs baseline (val=37.366 / test=31.371)
3. Per-split mae_vol_p
4. Best epoch (earlier peak with higher LR?)
5. Per-epoch wall-clock (~53.7s expected), parameter count, peak memory

## Baseline (PR #2229)

| Split | val | test |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | 51.356 | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |
