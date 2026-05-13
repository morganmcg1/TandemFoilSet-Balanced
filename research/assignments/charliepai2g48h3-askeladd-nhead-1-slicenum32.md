# n_head=1 on new compact stack: bracket the per-head capacity axis

## Hypothesis

PR #2149 (your previous experiment) established that n_head=2 wins over n_head=4 on the new compact stack (n_layers=4 + slice_num=32) by −0.25% val / −0.31% test. The result is consistent in direction on both val and test, confirming that **per-head capacity beats attention diversity on this compact architecture**.

The PR predicted: "If n_head=2 wins → n_head=1 is the natural bracket." This closes the axis with one more data point:
- n_head=4 (old baseline): attention diversity mattered more at deep/wide stack
- n_head=2 (current baseline): per-head capacity wins at compact stack — new best
- **n_head=1 (this PR):** zero attention diversity, maximum per-head capacity (head_dim=128)

With n_head=1, the PhysicsAttention module computes a single attention head over all 32 slices with head_dim=128. This is the extreme end of "capacity over diversity."

**Two predictions:**
1. **If n_head=1 wins:** Attention diversity is entirely redundant at n_layers=4 + slice_num=32; a single high-capacity head is sufficient. The model is effectively learning one global attention pattern over the 32 CFD subregions. Would then test whether n_head=2 was the source of the win, or whether it was the param increase (+6.3%).
2. **If n_head=1 loses (expected by prior intuition):** n_head=2 is the sweet spot — *some* diversity (2 modes) is needed, but 4 heads is over-diversified for 32 coarse slices. This cleanly brackets the head axis: n_head=2 is optimal on this stack.

**Param note:** Your prior writeup flagged that n_head=2 vs n_head=4 was mildly param-confounded (+6.3% params). For n_head=1, you may get yet another param increase or roughly the same — report the actual param count so we can track this. If n_head=1 wins, a param-matched n_head=4 test (e.g., n_hidden=120 instead of 128) would be the cleaner isolation, but that's a secondary follow-up.

## Instructions

Change ONLY `--n_head` from 2 to 1. Use `--slice_num 32` and `--n_layers 4` to match the current baseline. Keep epochs=21, T_max=21 (auto-aligned). Keep all other defaults.

Note: `--n_head` is now a CLI arg (merged in PR #2149). Pass `--n_head 1`.
Note: `n_layers` defaults to 5 in train.py — you MUST pass `--n_layers 4`.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name nhead-1-slicenum32-nlayers4 \
  --epochs 21 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32 \
  --n_head 1
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=42.709 / test=36.784)
2. Per-split `mae_vol_p`
3. Per-epoch wall-clock (epoch 1 and last) — should be similar to n_head=2 (~65s)
4. Best epoch, total wall-clock, epochs completed
5. **Parameter count** — critical for evaluating the param-vs-diversity confound
6. Peak memory

## Baseline (PR #2149: n_head=2 + n_layers=4 + slice_num=32 + T_max=21)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 45.089 | 41.257 |
| geom_camber_rc | 57.248 | 50.023 |
| geom_camber_cruise | 25.495 | 21.336 |
| re_rand | 43.004 | 34.519 |
| **avg** | **42.709** | **36.784** |

**Target to beat:** `val_avg/mae_surf_p < 42.709`

Baseline reproduce:
```bash
cd target/ && python train.py --epochs 21 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32 --n_head 2
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
