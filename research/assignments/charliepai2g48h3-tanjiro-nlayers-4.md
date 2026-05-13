# n_layers=4 + T_max=17: continue the depth sweep (epoch-count is the constraint)

## Hypothesis

The two most recent wins came from the SAME mechanism: shallower/lighter models → faster epochs → more epochs in the 30-minute budget → align T_max so cosine reaches zero at the final epoch:

- **PR #1995 (n_layers=5)**: 116s/epoch → 14 epochs in budget → T_max=14 alignment → val 47.478 (−6.98%)
- **PR #1996 (slice_num=48)**: 123s/epoch → 15 epochs in budget → T_max=15 alignment → val 46.847 (−1.33%)

The key insight: **the binding constraint on this 30-minute budget is epoch count, not model capacity.** Both wins shifted the optimal point of the speed/capacity trade-off in the same direction.

**The natural next step is to push deeper into that direction:** test `n_layers=4` (one less than current). Expected per-epoch time scales roughly linearly with depth, so:

- n_layers=5: ~116s/epoch
- n_layers=4: ~93s/epoch (≈20% faster)
- 30-min budget at 93s/epoch → ~17-18 epochs feasible

Combined with the slice_num=48 already in the current stack, n_layers=4 should yield even faster epochs and an even larger epoch budget. T_max=17 aligns the cosine schedule to the new epoch count.

**Two predictions:**

1. **If n_layers=4 wins:** depth was over-parameterized for this dataset; capacity wasn't load-bearing, epoch count was. We'd then test n_layers=3 to bracket the floor.

2. **If n_layers=4 loses:** we've hit the depth floor — there IS a minimum representational depth Transolver needs for this CFD task, somewhere between 4 and 5 layers. We'd freeze depth at 5 and explore other axes (slice_num=32, batch_size, augmentation).

**Why this is the highest-value next experiment:**
- It directly tests the most actionable insight from the last 2 wins.
- It's a single hyperparameter change (n_layers).
- The mechanism is well-understood (faster epochs → align T_max).
- Either outcome gives clean information (depth floor located, or new best).

## Instructions

Change `--n_layers` from 5 to 4. Bump `--epochs` to 17 and `--T_max` to 17 to leverage the expected speedup. Keep everything else at the current baseline (slice_num=48, surf_weight=10, GeGLU+RMSNorm, Lion lr=1e-4 WD=1e-4, bf16, batch=4, n_head=4).

**Note on the lr bug**: `train.py:441` hardcodes `lr=1e-4` in the Lion constructor (ignoring `cfg.lr`). Since we're keeping `--lr 1e-4` for this experiment, the bug doesn't bite — but feel free to apply the fix opportunistically: change `lr=1e-4` to `lr=cfg.lr`. Mention if you apply it.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-tanjiro \
  --experiment_name nlayers-4-tmax17 \
  --epochs 17 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4
```

If `--T_max` isn't a separate flag (PyTorch CosineAnnealingLR usually inherits from `--epochs`), then `--epochs 17` alone is enough.

**Important — budget guardrail:** If per-epoch time exceeds ~100s in the first epoch, your 30-min cap will likely abort before epoch 17. In that case:
- If you hit 15-16 epochs cleanly, that's still valuable signal — report the result.
- The wall-clock budget is hard; epochs beyond the cap are lost.

### Reporting requirements

1. Per-split val and test `mae_surf_p` against the current baseline (val=46.847 / test=40.837).
2. Per-split `mae_vol_p`.
3. **Diagnostic**: per-epoch wall-clock time for epoch 1 vs epoch 17 (or last completed epoch).
4. Total wall-clock and epochs actually completed.
5. Best epoch and parameter count (should be ~80% of baseline 976,827 ≈ 781K).
6. Peak memory.

## Baseline (PR #1996, current advisor HEAD)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 50.491 | 45.728 |
| geom_camber_rc | 60.364 | 55.146 |
| geom_camber_cruise | 29.835 | 24.157 |
| re_rand | 46.699 | 38.317 |
| **avg** | **46.847** | **40.837** |

**Target to beat:** `val_avg/mae_surf_p < 46.847`

Baseline reproduce (current advisor HEAD):
```bash
cd target/ && python train.py --epochs 15 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10
```

(Note: advisor HEAD has n_layers=5 + slice_num=48 already, so the above runs the compound stack. The 46.847 number was measured on n_layers=6 + slice_num=48 — the n_layers=5 + slice_num=48 compound is being verified by fern in PR #2062 concurrently with your run. If fern reports a different baseline, we'll update.)

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
