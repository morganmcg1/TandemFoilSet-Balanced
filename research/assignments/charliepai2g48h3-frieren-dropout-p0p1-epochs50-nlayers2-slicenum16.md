# Assignment: frieren — Dropout p=0.1 on attention + FFN on epochs=50 stack

**Branch (use exactly):** `charliepai2g48h3-frieren/dropout-p0p1-epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

The seed-variance finding from #2888 (three same-config runs gave val [34.544, 35.414, 35.697], sample std ≈ 0.605) is most parsimoniously explained by **overfitting / sensitivity to initialization-path**. The model converges to a different basin depending on init seed, and these basins differ by 0.6-1.1 val units. Regularization should:
1. Reduce the spread (variance) across seeds by pulling all init paths toward a wider, flatter basin
2. Possibly improve OOD splits (geom_camber_rc dominates val_avg at 48; if it's overfit to specific in-distribution geometric features, dropout would help)

**Dropout p=0.1 on attention + FFN is the canonical transformer regularization.** It's been the standard since Vaswani et al. 2017; absence of dropout in our current config means we're an unusual outlier. At n_layers=2 + slice_num=16 (361K params), the model is small enough that dropout p=0.1 won't underfit, but large enough that it's plausibly overfitting to train geometries.

**This is a true axis-test:** dropout has not been tried at this stack. If it helps, we've found a fresh regularization lever. If neutral or hurts, we've closed the dropout axis at this stack.

## Why this is mechanism-distinct from prior attempts

- **Aux head (#2871):** adds redundant gradient signal in same direction as surf_weight. Refuted.
- **Specialized decoders (#2883/#2904):** changes readout architecture; schedule-sensitive. Refuted.
- **SWA (#2857/#2888):** post-hoc weight averaging; depends on trajectory geometry. Refuted.
- **Dropout (this PR):** stochastic regularization DURING training, applied to attention weights and FFN activations. Affects what the model learns, not just how outputs are averaged. Mechanistically distinct.

Dropout is the standard regularization missing from this stack — and is precisely the mechanism most likely to address the seed-variance pattern (different basins → wider, flatter basin).

## Implementation

Find the Transolver model class in `train.py` (or in the imported model module). Identify:
1. The attention module — likely a standard scaled-dot-product attention, possibly with `nn.MultiheadAttention` or a custom implementation.
2. The FFN / MLP block — usually 2-layer with GELU/GEGLU activation between.

**Add a Config flag:**
```python
dropout: float = 0.0   # Dropout rate for attention weights and FFN (0.0 = disabled, matches current behavior)
```

**Apply dropout in TWO places:**

**1. Attention dropout** — after the softmax, before multiplying by V. This is the standard "attention dropout" (e.g., `MultiheadAttention(dropout=p)`).
```python
attn_weights = F.softmax(scores, dim=-1)
attn_weights = F.dropout(attn_weights, p=cfg.dropout, training=self.training)
out = attn_weights @ V
```

**2. FFN dropout** — after the activation, before the second linear (standard transformer convention).
```python
h = self.fc1(x)
h = self.gelu(h)   # or GeGLU
h = F.dropout(h, p=cfg.dropout, training=self.training)
h = self.fc2(h)
```

If the model uses `nn.MultiheadAttention`, you can pass `dropout=cfg.dropout` directly to its constructor. If GEGLU is used, dropout should be after the gated activation.

**Eval automatically disables dropout** (PyTorch's `.eval()` handles this); no eval changes needed.

**Note:** Do NOT add residual-path / embedding dropout — keep this minimal to one-axis test of attention+FFN dropout only.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name dropout-p0p1-epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --dropout 0.1
```

## Baseline to beat

PR #2872 (n_layers=2 + slice_num=16 + epochs=50, **no dropout**) — current best:

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **34.544** |
| val_single_in_dist | 35.113 |
| val_geom_camber_rc | 48.106 |
| val_geom_camber_cruise | 18.895 |
| val_re_rand | 36.060 |
| **test_avg/mae_surf_p** | **29.916** |

**Important: seed variance context.** Per #2888, same-config runs gave val [34.544, 35.414, 35.697]. The true mean of this config is ~35.2. Dropout should be evaluated against this distribution, not the lucky 34.544 baseline. **A val of 33-34 with dropout would be a clear win**; a val of 35-36 would be ambiguous (within noise); a val ≥36.5 would be a clear loss.

## Per-run constraints

- Hard timeout: 30 min (`SENPAI_TIMEOUT_MINUTES=30`). Dropout overhead is negligible (~0.1% per step). Should be ~29.3 min, same as baseline #2872.
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Include:
1. Full per-split val + test table at best_epoch
2. Per-epoch val_avg trajectory for last 8 epochs (so we can see whether dropout changes the plateau location/shape)
3. The actual `best_epoch` reached
4. Total wall-clock + per-epoch timing

## Decision criteria

- **Clear win (val < 34.0, ≥1.0 below mean of 35.2):** Dropout works. Try sweeping p ∈ {0.05, 0.1, 0.15, 0.2} to find optimal rate.
- **Ambiguous (val ∈ [34.5, 35.8]):** Within seed noise. Re-run 2x at p=0.1 to estimate variance; if mean clearly below 35.2, accept. If at-noise, close axis at p=0.1.
- **Loss (val > 36.0):** Dropout hurts. May indicate underfitting; close axis or try p=0.05.

## Suggested follow-ups (depending on outcome)

- **If wins:** sweep dropout rate; try compound with EMA (askeladd #2907 if EMA wins).
- **If neutral:** consider DropPath / stochastic depth as a sample-level regularization alternative; or input feature noise.
- **If loses:** rules out simple regularization. Pivot to data augmentation (geometric flips for allowed splits) or physics-informed loss.

## EV assessment

**Medium-high.** Dropout is the standard transformer regularization that's missing from this stack — a fundamental gap. At seed variance ~0.6 val-std, even small improvements in basin geometry should manifest. The strongest case for dropout is the geom_camber_rc dominance (val 48 — clear overfitting target); a regularizer that lowers this split would have outsize effect on val_avg. Worst case: dropout neutral or mild loss (axis closed, expected per-epoch budget unchanged). Best case: dropout finds wider basin → val drop 1-3% on the dominant OOD split → val_avg win of similar size.

This is the **last canonical regularization axis** to test before pivoting to data augmentation or physics-informed loss.
