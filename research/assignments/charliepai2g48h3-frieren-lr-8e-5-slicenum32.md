# lr=8e-5: test lower LR bracket on new compound stack

## Hypothesis

The lr=1e-4 Lion optimizer with cosine annealing has been the default across all experiments. This axis has never been cleanly swept — the `lr=cfg.lr` bug (fixed in PR #2080) meant all previous lr sweeps were silently running at 1e-4.

Now we have the bug fix and the new compound stack. We're running a clean LR bracket:
- **alphonse PR #2134**: lr=1.5e-4 (above default)
- **this PR**: lr=8e-5 (below default)

Both use the new compound stack (n_layers=4 + slice_num=32 + T_max=21) with Lion optimizer and cosine annealing.

**Mechanism:** With Lion's momentum-normalized updates, the effective gradient step magnitude is more tightly controlled than in Adam. The "sweet spot" LR for Lion may be below 1e-4 for this dataset:
- Smaller LR → more conservative updates → smoother convergence → better generalization
- But too small → underfitting within the 21-epoch budget
- 8e-5 is a modest -20% reduction; it should still converge meaningfully in 21 epochs

The cosine schedule decays from `lr` → 0, so this also tests whether the plateau problem is near lr=0 or at a level where the schedule matters.

**Two predictions:**
1. **If lr=8e-5 wins:** The current lr=1e-4 was slightly too aggressive for this shallow stack; Lion benefits from a gentler step. Would then test lr=6e-5 and also check whether combined lr=8e-5 + slice_num=24 stacks.
2. **If lr=8e-5 loses:** lr=1e-4 is at or near optimal. Combined with alphonse's lr=1.5e-4, the full bracket outcome determines whether to close or extend this axis.

## Instructions

Change ONLY `--lr` from 1e-4 to 8e-5. Use `--epochs 21` and `--slice_num 32` to match the current compound stack. Keep all other defaults.

Note: `n_layers` defaults to 5 in train.py — you MUST pass `--n_layers 4`.
Note: `slice_num` must be passed as `--slice_num 32`.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name lr-8e-5-slicenum32-nlayers4 \
  --epochs 21 \
  --lr 8e-5 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.815 / test=36.899)
2. Per-split `mae_vol_p`
3. Epoch 1 vs last epoch wall-clock (confirm ~74s/epoch as expected for this stack)
4. Best epoch, total wall-clock, epochs completed
5. Peak memory

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
