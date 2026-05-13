# slice_num=24: continue slice-partition sweep on new compound stack

## Hypothesis

The slice_num reduction sweep has produced two consecutive wins:
- **PR #1996 (slice_num=64→48)**: ~18% per-epoch speedup on n_layers=6 → 15 epochs in budget → val −1.33%
- **PR #2108 (slice_num=48→32)**: ~21% per-epoch speedup on n_layers=4 → 21 epochs in budget → val −7.6%

Both best epochs were the **final epoch** (still descending), meaning the epoch-count constraint is still binding even after two rounds of speedup. The mechanism is:
- Fewer slices → smaller `in_project_slice` linear map → less FLOPs/epoch → more cosine-aligned epochs in 30-min budget

**slice_num=32 gave 21 epochs at ~74s each.** Reducing to 24 should give:
- Estimated ~64-68s/epoch → 26-27 epochs in 30-min budget → T_max=26 aligned

**The key question:** At slice_num=24, do PhysicsAttention partitions still capture the meaningful CFD subregions (boundary layer / wake / freestream / near-field) for this 2D dataset, or do we cross the granularity floor where 24 slices is too coarse?

**Two predictions:**
1. **If slice_num=24 wins:** Continued epoch-budget gains dominate granularity loss at this scale. Would test slice_num=16 next.
2. **If slice_num=24 loses:** The granularity floor is between 24 and 32 for this problem. Close this axis at slice_num=32.

**Your PR #2108 observation:** Val curve still descending at epoch 21 — another T_max extension may also work. But isolating slice_num is cleaner.

## Instructions

Change ONLY `--slice_num` from 32 to 24. Set `--epochs 26` and use `--slice_num 24`. Keep all other current baseline defaults (n_layers=4, surf_weight=10, Lion lr=1e-4 WD=1e-4, bf16, batch=4, n_head=4).

Note: `n_layers` defaults to 5 in train.py — you MUST pass `--n_layers 4`.

**Budget guardrail:** If epoch 1 exceeds ~70s, the 30-min cap may cut before epoch 26. Report however many epochs completed. If per-epoch time ends up between 68-72s, also try `--epochs 25` as the T_max.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name slice-num-24-nlayers4 \
  --epochs 26 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 24
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.815 / test=36.899)
2. Per-split `mae_vol_p`
3. Per-epoch wall-clock (epoch 1 vs last) — key to confirm timing prediction
4. Total wall-clock, epochs completed, best epoch
5. Parameter count (slice_num doesn't change param count significantly — expect ~667K)
6. Peak memory

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
