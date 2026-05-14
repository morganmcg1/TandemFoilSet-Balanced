# lr=8e-5 on n_layers=2+slice_num=16+epochs=46: LR low-side fine probe (retry)

## Hypothesis

**This is a retry of the original lr=8e-5 hypothesis from the stale PR #2608 (closed due to rate-limit-stuck pod). The hypothesis remains valid; the experimental setup is unchanged.**

The LR axis at n_layers=2 stack still needs a fine-grained probe below baseline LR=1e-4:
- lr=5e-5 (thorfinn retry, in-flight) — 50% reduction
- **lr=8e-5 (this PR)** — 20% reduction, fine probe
- lr=1e-4 (baseline, val=35.256)
- lr=1.5e-4 (PR #2525, +3.30% LOSS) — OOD-dominated loss

**Why lr=8e-5 might help at n_layers=2:**

1. **Fine-grain LR sweep**: 20% LR reduction is at the noise floor edge (PR #2523 variance ~±1.0 val units; baseline 35.256 → 35.96 = 2% would be borderline). A win here would confirm the LR optimum has shifted down at the new stack.
2. **In-dist vs OOD tradeoff (PR #2525)**: Higher LR hurt OOD; lower LR may help OOD generalization while still allowing convergence within 46 epochs.
3. **Smaller model preference**: n_layers=2 has 361K params vs 515K at n_layers=3. Smaller models often prefer slightly lower LR (each step has more leverage).
4. **Final cosine LR**: T_max=46 with peak LR=8e-5 gives final LR ≈ 0, so model still converges to zero by end.

## Instructions

Single flag change from PR #2468 winner: `--lr 8e-5`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-alphonse \
  --experiment_name lr8e-5-nlayers2-slicenum16-epochs46-r2 \
  --epochs 46 \
  --lr 8e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does in-dist regression vs baseline 36.476 worsen?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does lower LR help OOD?
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
