# Gradient clipping max_norm=1.0: stabilize Lion updates

## Hypothesis

The current stack has NO explicit gradient clipping. Lion's sign-based updates inherently bound parameter step magnitude (each update is `±lr` per dimension), but the underlying gradients can still spike during training — particularly in the early epochs where loss is high (epoch 1 val=200+, decaying to 51 by epoch 12). Large unbounded gradients before the sign operation contribute to the moving averages (m_t with β1=0.9) used to compute the eventual update direction, which can cause direction estimates to be dominated by outlier gradients for several subsequent steps.

Adding `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()` bounds gradient norms while preserving direction. For Lion specifically, this stabilizes the first-moment EMA used to compute the sign update, leading to more representative direction estimates per step. The mechanism is orthogonal to all current wins (T_max=12 schedule alignment, surf_weight=5 budget reallocation, GeGLU+RMSNorm regularization, Lion sign-bounded updates).

**Why max_norm=1.0:** Standard default in transformer training. Conservative enough to bite only on outlier gradient spikes, not regular updates. Common choices in literature span 0.5–10.0; 1.0 is the most-cited default.

**Expected mechanism:** Slightly faster convergence in early epochs (better direction estimates), small improvement at the cosine tail (cleaner EMA contribution). Most likely outcome: small uniform improvement across splits, or wash. Small chance of significant gain if early-epoch gradient spikes have been silently bottlenecking us.

This is a single-line change in train.py — minimal implementation surface, easy to validate.

## Instructions

Add ONE gradient clipping call in the training loop, immediately before `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If using bf16 autocast with a GradScaler, the unscale step happens first — but since we use bf16 autocast directly (no scaler), no special handling is needed.

Keep all other defaults at the current baseline (PR #1956: T_max=12, surf_weight=5, GeGLU+RMSNorm, Lion lr=1e-4 WD=1e-4, bf16, batch=4, n_head=4).

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name grad-clip-1 \
  --epochs 12 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 5
```

If you want to expose this as a CLI flag (recommended for clean experiments and follow-ups), add a `--grad_clip_max_norm` flag defaulting to `0.0` (no clipping) and threshold the clip call on it. But hardcoded `max_norm=1.0` for this single experiment is also fine — note the change in the PR description.

### Reporting requirements

1. Per-split val and test `mae_surf_p` against the current baseline (val=51.040 / test=44.390).
2. Per-split `mae_vol_p`.
3. **Diagnostic:** Log the maximum gradient norm seen per epoch (`total_norm` returned by `clip_grad_norm_`). If `total_norm` is always < 1.0, the clipping never fires and the experiment is effectively a no-op confirmation — flag this in your writeup.
4. Best epoch and peak memory.

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
