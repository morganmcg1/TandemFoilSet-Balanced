# lr=5e-5 on n_layers=2+slice_num=16+epochs=46: LR lower bound at new stack (3rd retry)

## Hypothesis

**This is the THIRD retry of the lr=5e-5 hypothesis (after stale PR #2549 and #2611 both closed due to rate-limit-stuck pods). The hypothesis remains valid; the experimental setup is unchanged.**

The LR axis at n_layers=2 still needs the 50% reduction lower-bound:
- **lr=5e-5 (this PR)** — 50% reduction
- lr=8e-5 (alphonse in-flight, fine-grain probe)
- lr=1e-4 (baseline, val=35.256)
- lr=1.5e-4 (PR #2525, +3.30% LOSS)

**Why lr=5e-5 might help at n_layers=2:**

1. **Tighter OOD convergence**: Lower LR is generally associated with better generalization. At n_layers=2 (OOD bottleneck), a halved LR may help. 50% LR reduction is well above the seed-variance noise floor (~±1.0 val units per PR #2523).
2. **Smaller model may prefer lower LR**: n_layers=2 has 361K params vs older 515K at n_layers=3. Smaller models often prefer lower LR.
3. **In-dist vs OOD tradeoff at low LR (PR #2525)**: Higher LR hurt OOD. Mirroring this: lower LR may help OOD at small cost to in-dist.
4. **Combined with alphonse lr=8e-5 (R37)**: Gives a clean 3-point LR axis at the new stack: 5e-5 (this), 8e-5 (alphonse), 1e-4 (baseline). If 5e-5 wins → lower LR direction confirmed; consider 3e-5 next.

## Instructions

Single flag change from PR #2468 winner: `--lr 5e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name lr5e-5-nlayers2-slicenum16-epochs46-r3 \
  --epochs 46 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does in-dist regression worsen with lower LR (mirror of #2525)?
4. **OOD splits** — KEY: does lower LR help OOD generalization?
5. Best epoch — does lower LR push best_epoch earlier or to final?
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
