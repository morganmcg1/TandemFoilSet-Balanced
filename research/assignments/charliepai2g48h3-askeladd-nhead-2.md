# n_head=2: trade attention parallelism for per-head capacity

## Hypothesis

The current stack uses `n_head=4` with `n_hidden=128`, giving `head_dim=32`. In modern transformer literature (and the original "Attention Is All You Need"), `head_dim` is typically 64–128. `head_dim=32` is on the small side and may be undersized for the rich slice-wise features that Transolver's PhysicsAttention produces over `slice_num=64` learned partitions.

Halving the head count to `n_head=2` gives `head_dim=64`, doubling per-head representational capacity at the cost of half the parallel attention "modes". For physics problems where each slice represents a meaningful subregion of the flow field (boundary layer, wake, freestream), having fewer but more expressive attention heads may better match the underlying inductive bias — each head can dedicate more dimensions to encoding which slices interact and how.

This is an architectural axis we have NOT tested. Adjacent axes recently swept:
- `n_layers=5` vs 6 (edward #1995, in flight)
- `slice_num=48` vs 64 (fern #1996, in flight)
- `mlp_ratio=2` vs 4 (tanjiro #2007, in flight)
- `n_hidden=160` vs 128 (closed, val/test inversion)

`n_head` is the last unswept architectural lever on the current stack.

**Two complementary predictions:**

1. **If `n_head=2` wins:** head_dim=32 was undersized; the model was wasting parameters on too many small heads. We'd then test `n_head=1` to bracket the floor.
2. **If `n_head=2` loses:** attention diversity was load-bearing, and we should test `n_head=8` (`head_dim=16`) in the opposite direction to see if even MORE parallel modes help.

Either outcome is informative.

## Instructions

Change ONLY `--n_head` from 4 to 2. Keep everything else at the current baseline defaults (PR #1956: T_max=12, surf_weight=5, GeGLU+RMSNorm, Lion lr=1e-4 WD=1e-4, bf16, batch=4).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name nhead-2 \
  --epochs 12 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 5 \
  --n_head 2
```

Make sure `--n_head 2` is plumbed through to the model config (the `Config` dataclass in `train.py` should already accept it; verify before running). The Transolver `MODEL_CONFIG` dict in `train.py` reads `n_head` from `cfg.n_head`.

### Reporting requirements

1. Per-split val and test `mae_surf_p` against the current baseline (val=51.040 / test=44.390).
2. Per-split `mae_vol_p` — useful to see if the volume reconstruction changes with different attention dimensionality.
3. Best epoch and parameter count (should be ~similar to baseline 979,995 since `n_hidden` is unchanged).
4. Peak memory.

## Baseline (PR #1956)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 56.933 | 50.459 |
| geom_camber_rc | 64.886 | 59.341 |
| geom_camber_cruise | 31.056 | 25.501 |
| re_rand | 51.287 | 42.260 |
| **avg** | **51.040** | **44.390** |

**Target to beat:** `val_avg/mae_surf_p < 51.040`

Baseline reproduce:
```bash
cd target/ && python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 5
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
