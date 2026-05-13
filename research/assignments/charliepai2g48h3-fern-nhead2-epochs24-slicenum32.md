# n_head=2 + epochs=24 compound: combine both recent wins

## Hypothesis

Two independent improvements have just been merged:
1. **PR #2149 (n_head=2):** head_dim=32→64, −0.25% val at epochs=21, n_head=4 stack
2. **PR #2172 (epochs=24):** 21→24 epochs, −6.21% vs #2108 baseline, but used n_head=4 (default at run time)

These are orthogonal axes (attention head structure vs training budget). They have NOT been tested in combination. The natural compound test is **n_head=2 + epochs=24 + slice_num=32 + n_layers=4**.

**Expected outcome:** If gains compound:
- Your PR #2172 result at n_head=4 + epochs=24: val=40.158
- PR #2149's n_head=2 improvement at epoch=21: −0.25% = ~−0.11 val improvement relative
- Combined: ~39.9–40.0 val (conservative) or better if the mechanisms interact positively

**Why this is important:** PR #2172 showed `best_epoch=24 STILL DESCENDING`, and PR #2149 showed `best_epoch=21 STILL DESCENDING`. Both axes have slack. The compound run will use the current train.py default of n_head=2 (no override needed) and epochs=24.

**Note:** The current train.py now has `n_head=2` as default (merged in PR #2149). Your previous run (PR #2172) was at n_head=4 because it was created before that merge. This run uses the current default — no `--n_head` override needed.

## Instructions

Use `--epochs 24` and `--slice_num 32` and `--n_layers 4`. Do NOT override `--n_head` — the current default is n_head=2 (the winning configuration from PR #2149).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-fern \
  --experiment_name nhead2-epochs24-slicenum32-nlayers4 \
  --epochs 24 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32
```

**Note:** n_head defaults to 2 in current train.py — do NOT pass `--n_head` to get the compound. If you see n_head=4 in your config.yaml after running, something went wrong (the --n_head default changed back).

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=40.158 / test=34.904)
2. Per-split `mae_vol_p`
3. Per-epoch wall-clock — should be ~65s (n_head=2 is slightly faster than n_head=4 per your original observation)
4. **Per-epoch val_avg trajectory at epochs 21, 22, 23, 24** — key for comparing with PR #2172
5. Best epoch, total wall-clock, epochs completed
6. Parameter count — confirm n_head=2 used (should be ~708K vs #2172's 667K)
7. Peak memory

**Critical check:** Report `config.yaml`'s `n_head` value to confirm the compound is being tested. Should show `n_head: 2`.

## Baseline (PR #2172: epochs=24 + n_layers=4 + slice_num=32, n_head=4)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.610 | 38.553 |
| geom_camber_rc | 54.872 | 49.316 |
| geom_camber_cruise | 23.477 | 19.263 |
| re_rand | 41.675 | 32.483 |
| **avg** | **40.158** | **34.904** |

**Target to beat:** `val_avg/mae_surf_p < 40.158`

Baseline reproduce (fern's exact config):
```bash
cd target/ && python train.py --epochs 24 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32 --n_head 4
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
