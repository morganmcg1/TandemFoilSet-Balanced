# surf_weight=5 on current best stack: test vol-gradient mechanism on compact + extended config

## Hypothesis

The surf_weight axis has been bracketed on n_layers=4 + slice_num=48 (old stack):
- sw=15: neutral (−0.44% val vs old baseline, PR #2143)
- sw=10: current default (baseline)
- sw=2: never completed (PR #2109 closed stale)

But these tests used the OLD slice_num=48 + epochs=17/21 stack. The current best config is significantly different:
- slice_num=32 (21% fewer partitions per epoch)
- epochs=24 (3 more training epochs)
- n_head=2 (doubled per-head capacity in progress, pending compound)

The vol-gradient mechanism (sw lowering reallocates L1 gradient from surface to volume nodes) has never been tested on the current compact + extended-budget stack. The question: **does sw=5 help on n_layers=4 + slice_num=32 + epochs=24?**

Historical evidence:
- sw=5 on n_layers=6: WON (−9.0%, PR #1836)
- sw=5 on n_layers=5: WON (−4.0% vs n_layers=5 baseline, PR #2048, but lost vs current)
- sw=5 on n_layers=4 + slice_num=48: untested on this stack
- sw=15 on n_layers=4: neutral → sw=10 is near optimum in the HIGH direction

sw=5 is the most natural remaining test point. If it wins, the vol-gradient and epoch-budget mechanisms are orthogonal and compound. If it loses, the surf_weight axis is saturated for this depth.

**Note on n_head:** The current train.py default is n_head=2 (merged in PR #2149). Do NOT override --n_head — use the current default. The compound of n_head=2 + sw=5 vs n_head=2 + sw=10 is the clean test.

## Instructions

Change ONLY `--surf_weight` from 10 to 5. Use the CURRENT best config: `--epochs 24 --slice_num 32 --n_layers 4`. Do NOT pass `--n_head` — let the current default (n_head=2) apply.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name surf-weight-5-slicenum32-nhead2 \
  --epochs 24 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 5 \
  --n_layers 4 \
  --slice_num 32
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=40.158 / test=34.904)
2. Per-split `mae_vol_p` — diagnostic for vol-gradient mechanism (should improve if mechanism is active)
3. Per-epoch wall-clock (should be ~65s at n_head=2)
4. Best epoch, total wall-clock
5. Parameter count, peak memory

**Mechanism check:** If `mae_vol_p` improves and `mae_surf_p` also improves → vol-gradient mechanism is active and beneficial on this stack. If `mae_vol_p` improves but `mae_surf_p` worsens → the vol→surface pathway is too weak at this depth. If both worsen → sw=10 is optimal.

## Baseline (PR #2172: epochs=24 + n_layers=4 + slice_num=32, n_head=4 at run time)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 40.610 | 38.553 |
| geom_camber_rc | 54.872 | 49.316 |
| geom_camber_cruise | 23.477 | 19.263 |
| re_rand | 41.675 | 32.483 |
| **avg** | **40.158** | **34.904** |

**Target to beat:** `val_avg/mae_surf_p < 40.158`

Baseline reproduce (fern's exact run):
```bash
cd target/ && python train.py --epochs 24 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32 --n_head 4
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
