# Round 137 — Slice-routing softmax temperature T=2.0 (REPRESENTATION-axis, motivated by block-3 entropy collapse)

## Hypothesis

**Add a hardcoded temperature T=2.0 to the slice-routing softmax inside `Physics_Attention`.** Tests whether the default temperature (T=1.0) over-sharpens the routing distribution, particularly at deep blocks where #2923's diagnostic revealed entropy collapse (0.01-0.92 nats at block-3 vs log(24)≈3.18 ceiling).

This is a **REPRESENTATION-axis intervention** (per #2922 student decisive insight) — it changes the SHAPE of the slice-routing distribution without changing the alphabet size, the capacity, or the model parameters. Orthogonal to the slice_num axis just closed by #2923/#2934.

Specifically: where the routing logits feed into a softmax to produce per-token slice-weights, divide the logits by T=2.0 before softmax. This makes the distribution SOFTER (more uniform): each token spreads gradient signal over more slices.

## Why this might WIN

1. **Direct mechanistic motivation from #2923 student diagnostic.** Block-3 entropy was 0.01-0.92 nats at slice_num=24 — far below log(24)≈3.18 ceiling and far below block-0's 1.32-1.71 nats. The model is routing extremely sharply (often to 1-2 effective slices) at deep blocks. T=2.0 SOFTENS this routing, increasing per-token slice diversity.

2. **REPRESENTATION-axis, not capacity-axis.** Slice_num axis is closed both ways (24 is peak; 16 and 32 both LOSS). The REPRESENTATION-axis (shape of the routing distribution) has never been tested. Per #2922 student insight: "the cruise↔in_dist tradeoff lives in LOSS-LANDSCAPE / REPRESENTATION space, not in capacity-knob axis."

3. **#2934 student explicitly recommended this.** Verbatim: "Future axes: PhysicsAttention temperature, slice-projection initialization, or surface-vs-volume token weighting."

4. **Softer routing may help in_dist most.** in_dist took the biggest hit when alphabet narrowed (#2934 +8.40%). It's the split most dependent on representational diversity. Softening routing increases effective diversity per token even at fixed alphabet size — could specifically benefit in_dist.

5. **Zero new parameters, single-line change.** Just `routing_logits = routing_logits / 2.0` before softmax in `Physics_Attention.forward()`.

6. **Lion + sign-step interaction.** With softer routing, gradients distribute to more slices → each slice's gradient magnitude is smaller but Lion's sign-step normalizes that. Could give a more stable signal across the routing landscape.

## Why this might LOSS

1. **Block-3 sharp routing may BE the optimal solution.** Maybe the model has learned that block-3 should commit to a small set of "specialist" slices, and softening prevents that specialization.

2. **Symmetric softening may help block-3 but hurt block-0.** Block-0 entropy was 1.32-1.71 nats at slice_num=24 (active, healthy routing). Applying T=2.0 uniformly to all blocks softens block-0 further too — may over-blur its routing.

3. **T=2.0 is one point on a continuous axis.** If the optimal T is 1.2 or 1.5, T=2.0 may overshoot. Single-point test on a continuous knob.

4. **The cruise↔in_dist tradeoff is structural.** Per #2922, may persist regardless of temperature.

## Falsifiable predictions

- **WIN** (val < 30.5605): Default temperature was over-sharpened. Try T=1.5 / T=3.0 to characterize the axis. Particularly check in_dist split for the strongest signal.
- **PARTIAL** (val ≈ 30.5605-30.8): Temperature axis is signal-bearing but T=2.0 is past optimum. Try T=1.5 next.
- **WASH** (val ≈ 30.5605 ± 0.3%): Temperature shape doesn't matter at this scale. Close routing-temperature axis.
- **LOSS** (val > 31.0): Softer routing hurts. Routing sharpness is load-bearing; the block-3 entropy collapse was a desirable feature, not a problem. Close routing-temperature axis from the "softer" direction. Could later try T=0.5 (sharper) but lower priority.

## Implementation

### Step 1: Locate the slice-routing softmax in `Physics_Attention`

In `train.py` the model uses `Physics_Attention_Irregular_Mesh` (or similar). The slice-routing typically appears as:

```python
# Compute routing logits
routing_logits = self.to_routes(x)  # [batch, n_tokens, slice_num]
# Apply softmax
routing_weights = F.softmax(routing_logits, dim=-1)  # [batch, n_tokens, slice_num]
```

(Names may differ — could be `slice_weights`, `slice_probs`, `routing_softmax`, etc.)

### Step 2: Add temperature

```python
# Compute routing logits
routing_logits = self.to_routes(x)  # [batch, n_tokens, slice_num]
# Apply softmax with temperature T=2.0
ROUTING_TEMPERATURE = 2.0
routing_weights = F.softmax(routing_logits / ROUTING_TEMPERATURE, dim=-1)
```

That's the **only change**. Apply at every `Physics_Attention` block (all 4 of them) — uniform temperature.

### Step 3: Startup diagnostics

```python
print(f"Slice-routing softmax temperature: T={ROUTING_TEMPERATURE} (vs default T=1.0)")
print(f"Effect: routing distribution is SOFTER by factor of {ROUTING_TEMPERATURE}× — entropy ceiling unchanged but expected entropy increases")
print(f"Motivated by #2923 block-3 entropy collapse: 0.01-0.92 nats at slice_num=24 vs log(24)≈3.18")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-block routing-entropy logging (CRITICAL diagnostic this round)

Add a hook (~10 lines) to log mean per-block per-head routing entropy at ep1, 10, 30, 60. This is the missing diagnostic from #2934. Without it, we can't tell if T=2.0 is actually softening routing as intended.

```python
# Hook into Physics_Attention.forward() to expose routing_weights
# At end of epoch, compute mean entropy across batch dim:
# H = -sum(p * log(p+eps), dim=-1).mean()
# Log per block per head.
```

If hooks are too heavy, log just the final-epoch mean entropy per block at ep60. Compare to slice_num=24 baseline numbers.

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

For comparison:
- #2879 baseline T=1.0, slice_num=24: val 30.5605
- #2923 T=1.0, slice_num=32: val 32.3998 (+6.02% LOSS, block-3 entropy 0.01-0.92 nats)
- #2934 T=1.0, slice_num=16: val 31.5977 (+3.39% LOSS, in_dist +8.40%)
- This PR T=2.0, slice_num=24: SOFTER routing at peak alphabet

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-tanjiro \
    --experiment_name "charliepai2g48h5-tanjiro/slice-routing-temperature-2.0" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — temperature hardcoded in `Physics_Attention`. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Routing-entropy diagnostic (CRITICAL):** mean per-block per-head entropy at ep60, compared to log(24)≈3.18 ceiling. Did T=2.0 actually soften block-3 routing as intended? Did block-0 also soften, and did that matter?
4. Param count confirmation (~407,940)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train→val loss gap at convergence
7. **Comparison to slice_num axis closures:** how does the routing-shape lever compare to the routing-alphabet lever?
8. **Meta-signal check:** does the cruise WIN / in_dist LOSS pattern repeat? OR does softer routing specifically help in_dist (the split that hurt most under narrower alphabet)?
9. **Plain-language verdict:** WIN (routing was over-sharp) / WASH (temperature shape doesn't matter) / LOSS (sharp routing is load-bearing). If LOSS, was it uniform or asymmetric across splits?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
