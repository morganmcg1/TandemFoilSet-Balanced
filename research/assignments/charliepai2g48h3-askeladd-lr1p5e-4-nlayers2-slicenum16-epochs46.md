# lr=1.5e-4 on n_layers=2+slice_num=16+epochs=46: LR probe at new depth stack

## Hypothesis

The new best stack (PR #2468) is n_layers=2+slice_num=16+epochs=46, val=35.256. LR has only been tested at n_layers=3; the LR optimum may shift at n_layers=2.

Evidence and reasoning:
1. **LR is partition+depth dependent**: At n_layers=3+slice_num=16, lr=1.5e-4 LOST (+7.3%). But at n_layers=3+slice_num=24, lr=1.5e-4 WON. The LR optimum shifts with architecture.
2. **Smaller model may need higher LR**: n_layers=2 has 30% fewer params (361K vs 515K). With fewer parameters, each update step has more leverage — a higher LR may be appropriate.
3. **More epochs amplify LR sensitivity**: With 46 cosine epochs and T_max=46, a higher LR covers more of the parameter space during the high-LR phase (first ~half of cosine schedule).
4. **In-dist regression at lr=1e-4**: single_in_dist got worse (+1.21) at n_layers=2+lr=1e-4. A higher LR early in training may help the smaller model learn in-distribution patterns more quickly.

## Instructions

Single flag change from PR #2468 winner: `--lr 1.5e-4`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name lr1p5e-4-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1.5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — did the in-dist regression (36.476 at lr=1e-4) improve or worsen?
4. Train loss epochs 1–5: faster descent vs lr=1e-4?
5. Best epoch, total wall-clock, peak memory

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
