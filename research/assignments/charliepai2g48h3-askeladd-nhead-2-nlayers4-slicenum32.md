# n_head=2 on new compound stack: test per-head capacity on shallow+narrow model

## Hypothesis

The current stack uses `n_head=4` with `n_hidden=128`, giving `head_dim=32`. Halving to `n_head=2` gives `head_dim=64`, doubling per-head representational capacity.

Previous test: `n_head=2` on the OLD stack (n_layers=6 + slice_num=64, val baseline ~51.040) lost by +12.4% — attention diversity mattered more than per-head capacity on a deeper, wider model.

But the model has changed significantly since then:
- **n_layers: 6 → 4** (much shallower — each layer must do more work)
- **slice_num: 64 → 32** (fewer attention partitions — each slice is coarser)
- **n_params: ~976K → 667K** (−31.6%)

On this new stack, `head_dim=32` may be too small for the coarser 32-slice partitions. With fewer, coarser PhysicsAttention slices, each head needs more capacity to model inter-slice relationships. The trade-off has shifted: with only 32 partitions, 4 heads × 32-dim is likely under-parameterized per head, while 2 heads × 64-dim may better utilize the available attention budget.

**Two predictions:**
1. **If n_head=2 wins:** Per-head capacity is the bottleneck on the new compact stack. Would then test n_head=1 to bracket further.
2. **If n_head=2 loses:** Attention diversity matters even at slice_num=32 — the 4 parallel "modes" are load-bearing even at head_dim=32. Close this axis.

**Key diagnostic:** Check `mae_vol_p` per split. n_head=2 changing vol reconstruction implies the attention re-wiring is genuine, not just noise.

## Instructions

Change ONLY `--n_head` from 4 to 2. Use `--epochs 21` and `--slice_num 32` to match the current compound stack. Keep all other defaults.

Note: `n_layers` defaults to 5 in train.py — you MUST pass `--n_layers 4`.
Note: `slice_num` must be passed as `--slice_num 32` (default is now 48 via CLI).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name nhead-2-slicenum32-nlayers4 \
  --epochs 21 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32 \
  --n_head 2
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.815 / test=36.899)
2. Per-split `mae_vol_p` — diagnostic for whether head rewiring changes volume reconstruction
3. Per-epoch wall-clock (epoch 1 vs last) — n_head=2 should not significantly change timing
4. Best epoch, total wall-clock, epochs completed
5. Parameter count (n_head=2 with n_hidden=128 changes attention proj sizes — report actual)
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
