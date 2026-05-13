# n_layers=2 on slice_num=32: continue depth sweep (epoch-budget mechanism step 4)

## Hypothesis

Every depth reduction has produced a win: n_layers=6→5→4→3, accelerating not diminishing (−6.98%, −1.07%, −8.58% on last step). Current best: n_layers=3+slice_num=32+epochs=27, val=39.143, best_epoch=27 STILL DESCENDING.

n_layers=2 removes one full Transolver block (~17% fewer FLOPs):
- Per-epoch estimate: ~48s
- Budget: ~33 epochs in 30 min cap (33 × 48 = 26.4 min)
- T_max auto-aligns to epochs

Parameter estimate: ~360K (vs 515K at n_layers=3). Key risk: may be too shallow for geometry diversity in geom_camber_rc.

**Predictions:**
1. If win → test n_layers=1 or n_hidden reduction next
2. If loss → capacity floor at n_layers=3; explore slice_num, surf_weight, n_head

## Instructions

Change ONLY `--n_layers` from 3 to 2 and set `--epochs 33`. Same config as PR #2107 otherwise.

Do NOT pass `--n_head` (default n_head=4).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name nlayers2-slicenum32-epochs33 \
  --epochs 33 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 32
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs baseline (val=39.143 / test=33.571)
2. Per-split `mae_vol_p`
3. **Epoch 1 wall-clock** — key for n_layers=2 speed characterization
4. Last epoch wall-clock, total wall-clock, epochs completed
5. Best epoch — still final? Per-epoch trajectory last 5 epochs
6. Parameter count (expect ~360K), peak memory

## Baseline (PR #2107: n_layers=3 + slice_num=32 + epochs=27)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.405 | 35.977 |
| geom_camber_rc | 51.895 | 47.136 |
| geom_camber_cruise | 22.756 | 19.101 |
| re_rand | 41.517 | 32.070 |
| **avg** | **39.143** | **33.571** |

**Target to beat:** `val_avg/mae_surf_p < 39.143`

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
