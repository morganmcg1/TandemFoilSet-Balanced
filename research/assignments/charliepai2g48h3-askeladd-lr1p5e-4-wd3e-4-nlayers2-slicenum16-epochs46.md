# lr=1.5e-4 + weight_decay=3e-4 on n_layers=2+slice_num=16+epochs=46: Compound LR+WD probe targeting OOD

## Hypothesis

This is a **compound experiment** testing two axes simultaneously, motivated by the PR #2525 (closed) finding:

**PR #2525 result (lr=1.5e-4 alone):**
- single_in_dist: −6.35% IMPROVED (val), −4.25% IMPROVED (test)
- ALL 3 OOD splits: +3.4% to +10.5% WORSE
- Net: +3.30% LOSS on avg

**The hypothesis:** Higher LR fixes the n_layers=2 in-dist regression but causes overfitting/oscillation that hurts OOD. **Strong weight decay (3× baseline) may regularize the OOD damage while preserving the in-dist benefit.**

Mechanism:
1. **Higher LR drives faster in-dist fitting**: The model converges quickly on dense in-distribution training samples.
2. **Higher WD penalizes large weight magnitudes**: This counteracts the noise/oscillation caused by larger LR steps and acts as an L2 prior on flat solutions (which generalize better to OOD).
3. **Both axes targeted ONE PR — compound test**: If both win, we get a bigger gain than either alone (LR fixes in-dist, WD fixes OOD).
4. **Seed variance context (PR #2523)**: Run-to-run variance is ~1.0 val units. This compound change is a 50% LR increase + 200% WD increase — big enough to produce signal above noise.

## Instructions

Two flag changes from PR #2468 winner: `--lr 1.5e-4 --weight_decay 3e-4`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name lr1p5e-4-wd3e-4-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1.5e-4 \
  --weight_decay 3e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does the in-dist gain from lr=1.5e-4 still happen with WD=3e-4? (was 34.161 with lr=1.5e-4 alone)
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does WD=3e-4 rescue the OOD regression seen in #2525?
5. Best epoch, total wall-clock, peak memory
6. **Comparison to PR #2525 lr=1.5e-4 alone**: For each split, did adding WD=3e-4 help (better than 36.419 avg) or hurt?

## Baseline (PR #2468)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.476 | 33.035 |
| geom_camber_rc | 48.297 | 44.333 |
| geom_camber_cruise | 18.326 | 15.496 |
| re_rand | 37.923 | 28.116 |
| **avg** | **35.256** | **30.245** |

**Reference: lr=1.5e-4 + WD=1e-4 baseline (PR #2525, closed +3.30%):**
| Split | val | test |
|---|---|---|
| single_in_dist | 34.161 | 31.629 |
| geom_camber_rc | 51.816 | 45.840 |
| geom_camber_cruise | 20.251 | 16.886 |
| re_rand | 39.449 | 29.122 |
| **avg** | **36.419** | **30.869** |

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
