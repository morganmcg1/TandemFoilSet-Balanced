# lr=7e-5 on n_layers=2+slice_num=16+epochs=46: LR fine-grain low-side probe (pivot)

## Hypothesis

**Pivot from lr=5e-5 which has failed to land 3 times (#2549, #2611, #2683 all stale_wip from rate-limit-stuck pods).** Collecting NEW data at a fresh LR point is more valuable than further retries of the failing pattern.

LR axis at n_layers=2 stack:
- lr=5e-5 (3x stale, untested)
- lr=7e-5 **(this PR — fresh probe at 30% below baseline)**
- lr=8e-5 (alphonse #?? in flight, 3rd retry)
- lr=1e-4 (baseline, val=35.256)
- lr=1.2e-4 (lost — original alphonse #2543 stale, but lost on closure)
- lr=1.5e-4 (PR #2525, +3.30% LOSS)

**Why lr=7e-5 might help at n_layers=2:**

1. **Fine-grain LR sweep**: lr=7e-5 is between baseline 1e-4 and lr=5e-5. 30% LR reduction is at the edge of the noise floor (PR #2523 variance ~±1.0 val units; would be ~2-3% gain to be clearly above noise).
2. **In-dist vs OOD tradeoff (PR #2525, #2638)**: Higher LR hurt OOD; mirror hypothesis says lower LR may help OOD generalization. lr=7e-5 is intermediate — provides a 3rd data point on the LR axis without being too extreme.
3. **Smaller model may prefer lower LR**: n_layers=2 has 361K params; smaller models often prefer slightly lower LR.
4. **Combined with alphonse lr=8e-5 (#?? in flight, 3rd retry)**: Gives clean 4-point LR axis at the new stack if both land — 7e-5 (this), 8e-5 (alphonse), 1e-4 (baseline), 1.5e-4 (lost).
5. **Final cosine LR**: T_max=46 with peak LR=7e-5 gives final LR ≈ 0 by cosine schedule. Model still converges to zero by end.

## Instructions

Single flag change from PR #2468 winner: `--lr 7e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr7e-5-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 7e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does in-dist regression worsen with 30% lower LR?
4. **OOD splits** — KEY: does lower LR help OOD generalization?
5. Best epoch — does lower LR push best_epoch?
6. Total wall-clock, peak memory

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
