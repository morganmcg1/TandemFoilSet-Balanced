# Round 133 — Zero-init the final output projection (head)

## Hypothesis

**Zero-init the weights of the final output projection layer** (the Linear that produces the per-token surface/volume predictions). Tests whether the default Kaiming-init head is biasing early training toward arbitrary noise patterns that compete with the network's emerging signal.

## Why this might WIN

1. **Classic recipe-gap test never applied in this launch.** Zero-init of the final layer is the canonical "warm start" trick from DiT (Peebles & Xie 2022), ResNet (He et al. zero-init final BN γ), and fixup-init transformers. Each block's residual `x = x + γ * block(x)` already has LayerScale γ=1e-4 zero-ish init, so the network is essentially identity for the first few epochs. But the OUTPUT HEAD is currently default-init (Kaiming-style), meaning at step 0 the model predicts random non-zero values that get penalized by loss → wasted early gradient updates pushing the head toward zero before the body's representations have meaningful content.

2. **Zero-init head + zero-init LayerScale γ = clean inductive bias.** At step 0, the network predicts exactly 0 everywhere. Gradients flow through to the body's parameters, which learn representations, and the head learns the projection magnitude. This avoids the "wasted epoch 0-1" where the head is being randomly subtracted from random body output.

3. **Especially useful for regression tasks.** Classification heads can be left default-init because softmax normalizes the output. Regression heads benefit more from zero-init because the loss directly penalizes magnitude — and a randomly-initialized head produces predictions of arbitrary scale.

4. **Tiny change: 1-2 lines.** Locate the final Linear projection (the last layer that produces per-token outputs) and zero-init its weight (and optionally its bias).

5. **Zero new parameters, zero new flops.** Pure initialization change.

## Why this might LOSS

1. **Body may need a non-zero head signal to learn.** If the head is zero-init, the gradient signal through to the body's parameters is also zero-magnitude initially — the network might fail to start learning at all. Mitigation: this is FALSE because the LOSS gradient w.r.t. the head's input (the body output) is non-zero — the head's weight gradient is non-zero whenever the body's output has any signal. The body can learn even when the head is zero.

2. **Default init may already be near-zero in expectation.** Kaiming-init with `fan_in=96` produces weights of std ≈ √(2/96) ≈ 0.144 — small but not negligible. The bias is typically zero. So the head's expected output at init is small. The improvement from zero-init may be marginal.

3. **The cruise/in_dist trade-off is structural, not head-init-driven.** If the trade-off is determined by feature representations, head init won't break it.

## Falsifiable predictions

- **WIN** (val < 30.5605): Head init was a recipe gap. Try also zero-init the head bias (if not already).
- **WASH** (val ≈ 30.5605 ± 0.3%): Default-init head was already near-zero enough. Close axis.
- **LOSS** (val > 31.0): Some non-zero head init is needed for stable early training. Surprising; would falsify the DiT/Fixup intuition for this scale.

## Implementation

### Step 1: Find the final output projection in `train.py`

The model produces per-token surface predictions. Look in `Transolver` (or `TransolverBlock` for the `last_layer=True` case) for the final Linear layer that maps the block's hidden features to the output channel count (`out_dim`).

Likely candidates:
- The `last_layer=True` branch in `TransolverBlock` may have a separate output projection
- OR the `Transolver` class may have a `self.out_proj = nn.Linear(n_hidden, out_dim)` or similar
- OR the final block's MLP may be the projection

### Step 2: Zero-init the final projection's weight

Add immediately after the final projection's construction:

```python
# Zero-init the output head — canonical regression recipe gap
nn.init.zeros_(<final_projection_layer>.weight)
# bias usually already 0 by default, but be explicit:
if <final_projection_layer>.bias is not None:
    nn.init.zeros_(<final_projection_layer>.bias)
```

Where `<final_projection_layer>` is whichever module produces the final per-token output (e.g., `self.out_proj`, or the last `nn.Linear` in `TransolverBlock` when `last_layer=True`).

### Step 3: Startup diagnostics

```python
# Verify zero-init landed
final_layer = <whichever module>
print(f"Final head weight norm at init: {final_layer.weight.norm().item():.6f}")  # Expected: 0
print(f"Final head bias norm at init: {final_layer.bias.norm().item():.6f}")        # Expected: 0
print(f"Model output at step 0 (dummy input): should be ~0")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")
```

### Step 4: Verify model predictions are ~0 at step 0

Before training starts, run one forward pass on a batch and confirm:
- `pred.abs().mean()` is very small (~1e-5 or less)
- `loss` at step 0 should be close to the loss of predicting all zeros (essentially the target magnitude itself)

This sanity-check confirms the zero-init landed and the entire forward chain is wired correctly.

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 |

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/head-zero-init" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Init-sanity diagnostic:** final-layer weight/bias norms at step 0 (should be 0), model output magnitude at step 0 (should be ~0)
4. **Head learning trajectory:** final-layer weight norm at ep1, 5, 30, 60. Does the head grow steadily from 0?
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train loss at ep1-3 vs expected baseline ep1-3
8. **Plain-language verdict:** zero-init head WIN/WASH/LOSS — was head init a real recipe gap or marginal?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
