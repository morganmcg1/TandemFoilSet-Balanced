# epochs=24 + T_max=24 on slice_num=32 stack: squeeze remaining epoch budget

## Hypothesis

PR #2108 (thorfinn slice_num=32 + n_layers=4 + T_max=21) gave the current baseline val=42.815/test=36.899, with **best_epoch=21 STILL DESCENDING at the final epoch**. The training wall-clock was 25.94 min — well inside the 30 min cap. The cosine schedule was correctly aligned with T_max=21, but the model hadn't saturated when the schedule expired.

This is a clear signal: at slice_num=32 + n_layers=4, the 21-epoch budget under-uses the wall-clock cap. Per-epoch time is ~74s, so:
- 24 epochs × 74s = 1776s = **29.6 min** (just inside the 30-min cap)
- 25 epochs × 74s = 1850s = 30.8 min (over budget — would be cut)

Setting `--epochs 24 --T_max=24` (automatic via `T_max=cfg.epochs`) gives:
- Same per-epoch speed
- 3 additional epochs of cosine-aligned training
- LR decays more slowly during early epochs (slower descent → more learning at each level)

**Two predictions:**
1. **If epochs=24 wins:** The "epoch-count is binding constraint" mechanism is still operating even at slice_num=32. Would then test epochs=25 (if it fits in 30 min) or pivot to further per-epoch speedups.
2. **If epochs=24 loses or ties:** The slice_num=32 stack has approached its convergence ceiling at 21 epochs; the LR-floor (cosine going to 0) matters more than additional epochs. Close this micro-axis and pivot to architectural changes.

**Mechanism rationale:** Best epoch has been the final epoch across n_layers=6→5→4 and slice_num=64→48→32. This is a 5-experiment streak indicating that the model has consistently been budget-limited. Pushing the budget to its hard limit is the lowest-risk test of how much epoch-axis headroom remains.

## Instructions

Change ONLY `--epochs` from 21 to 24. Use `--slice_num 32` and `--n_layers 4` to match the current baseline. Keep all other defaults.

**Budget guardrail:** If per-epoch time is significantly above 75s at epoch 1, the 30-min cap may cut you before epoch 24. Report however many epochs completed.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name epochs-24-slicenum32-nlayers4 \
  --epochs 24 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.815 / test=36.899)
2. Per-split `mae_vol_p`
3. **Per-epoch val_avg trajectory across all completed epochs** — critical for determining whether the val curve flattens between epochs 21 and 24 or continues to descend
4. Per-epoch wall-clock (especially epochs 1 and last) — confirm timing fits budget
5. Best epoch number, total wall-clock, epochs completed
6. Peak memory

**Critical analysis:** Compare val_avg/mae_surf_p at epochs 21 (baseline reference point) and 24. If val_21 ≈ val_24, the budget extension didn't help and we hit a ceiling. If val_24 < val_21, the epoch axis still has slack.

## Baseline (PR #2108: n_layers=4 + slice_num=32 + T_max=21)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 44.963 | 40.717 |
| geom_camber_rc | 56.766 | 51.074 |
| geom_camber_cruise | 25.476 | 21.158 |
| re_rand | 44.053 | 34.646 |
| **avg** | **42.815** | **36.899** |

**Target to beat:** `val_avg/mae_surf_p < 42.815`

Baseline reproduce:
```bash
cd target/ && python train.py --epochs 21 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
