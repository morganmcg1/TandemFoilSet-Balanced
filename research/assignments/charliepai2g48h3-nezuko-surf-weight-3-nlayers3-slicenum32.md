# surf_weight=3 on n_layers=3+slice_num=32+epochs=27: fill sw curve

## Hypothesis

Nezuko's PR #2214 confirmed vol-gradient mechanism active on compact stack: mae_vol_p improved on every split (−7.9 to −14.9%), test surface (−3.42%) showed strong generalization.

Fill the sw curve at n_layers=3:
- askeladd #2248: sw=2
- nezuko (this): sw=3
- fern #2245: sw=5
- baseline: sw=10

At sw=3, vol gets ~3.3× more gradient mass than baseline. Expect larger vol MAE improvement than at sw=5.

## Instructions

Single flag change: `--surf_weight 3`. Same config as PR #2107 otherwise.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name surf-weight-3-nlayers3-slicenum32 \
  --epochs 27 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 3 \
  --n_layers 3 \
  --slice_num 32
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (38.270/32.470)
2. **Per-split mae_vol_p** (should improve more than at sw=5)
3. Per-epoch wall-clock, best epoch
4. Parameter count (~515K), peak memory

## Baseline (PR #2228)

| Split | val | test |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | 52.042 | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |
