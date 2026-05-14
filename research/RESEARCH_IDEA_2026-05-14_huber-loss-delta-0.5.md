# Round 136 — Huber loss δ=0.5 (gradient-shape LOSS-axis test)

## Hypothesis

**Replace L1 loss with Huber loss (SmoothL1) at δ=0.5** on both `surf_loss` and `vol_loss`. Tests the gradient-SHAPE axis of the loss function — orthogonal to #2933 alphonse's per-channel-WEIGHT axis. Huber transitions from L2-like (small residuals) to L1-like (large residuals) at the threshold δ.

This is the second loss-function-axis test in this round, following the decisive student of #2922 insight: *"the cruise↔in_dist trade-off lives in the LOSS LANDSCAPE itself. Future attacks should target the LOSS FUNCTION or REPRESENTATION rather than optimizer dynamics."* #2933 attacks per-channel weight; this PR attacks per-residual gradient shape.

## Why this might WIN

1. **L2-like gradient for small residuals** provides smoother gradient signal at convergence. Lion's sign(grad) is doubly coarse with L1 (sign of an already coarse sub-gradient); Huber near 0 gives Lion a SMOOTH gradient to take sign of, which actually carries magnitude information up to ±δ. This can unlock fine-tuning the L1+Lion combo cannot capture.

2. **L1-like robustness for large residuals** preserves the L1 baseline's robustness to outlier tokens (e.g., near-singularity points at trailing-edge corners). Doesn't degrade the regime where L1 already works.

3. **Standard recipe in PDE-surrogate and vision regression literature.** SmoothL1 is well-tested; δ=0.5 is a moderate choice between L2 (δ→∞) and L1 (δ→0).

4. **Direct test of #2922 student LOSS-FUNCTION axis recommendation.** Different from #2933 (per-channel) — this is per-residual-magnitude reweighting via the loss curve shape itself.

5. **Zero new parameters, single-line change.** Pure loss-function edit. No architecture, no scheduler, no optimizer change.

## Why this might LOSS

1. **Eval metric is MAE (L1), train loss would be Huber.** Train/eval mismatch could create a suboptimal training objective relative to the metric. Mitigation: at δ=0.5, the loss is L1 for any residual above 0.5 (in normalized space), which is most of the surf MAE regime (typical normalized residuals are ~0.1-1.0 magnitude).

2. **Lion was calibrated for L1.** Lion's sign(exp_avg + β·grad) update doesn't care about gradient magnitude; it only cares about sign. With L2-like Huber near 0, the sub-gradient direction is still correct, but the magnitudes are wrong — though this still doesn't matter for sign step. Should be neutral or beneficial in practice.

3. **The cruise/in_dist trade-off is structural.** If the LOSS LANDSCAPE has the trade-off baked in (per #2922 insight), changing the loss SHAPE may not move the trade-off — only the LANDSCAPE itself (via REPRESENTATION change). LOSS WIN here would falsify or refine this.

## Falsifiable predictions

- **WIN** (val < 30.5605): L2-like gradient near 0 unlocks fine-tuning. Try δ=1.0 (more L2-leaning) and δ=0.1 (more L1-leaning) to characterize.
- **PARTIAL** (in_dist WIN at cruise cost, or vice-versa): Meta-signal persists. Huber doesn't break the structural trade-off, just nudges the landing point on it.
- **WASH** (val ≈ 30.5605 ± 0.3%): Gradient shape near 0 doesn't matter for Lion. Close gradient-shape axis.
- **LOSS** (val > 31.0): Train/eval mismatch hurt. Close gradient-shape axis.

## Implementation

### Step 1: Locate the L1 loss computation in `train.py`

The current loss block uses `(pred - y).abs().mean()` (or equivalent) for both surf_loss and vol_loss. Find the lines that compute these.

### Step 2: Replace L1 with Huber (SmoothL1) at β=0.5

PyTorch's `F.smooth_l1_loss(input, target, beta=δ, reduction='none')` computes:
- 0.5 * (x - y)² / β when |x - y| < β
- |x - y| - 0.5 * β when |x - y| ≥ β

So `beta=0.5` is the δ=0.5 Huber transition.

```python
import torch.nn.functional as F

# Replace
diff_surf = (pred - y).abs()  # OLD: L1

# With
diff_surf = F.smooth_l1_loss(pred, y, beta=0.5, reduction='none')  # NEW: Huber δ=0.5
```

Apply the same change at both surf_loss and vol_loss sites. Keep all mask normalization and surf_weight scaling identical.

**Important:** Keep `reduction='none'` if the existing code applies a mask to the per-element diff before reducing — the per-element shape must match exactly.

### Step 3: Startup diagnostics

```python
print(f"Loss function: SmoothL1 (Huber) β=0.5 instead of L1 .abs()")
print(f"Huber transition: residuals |r| < 0.5 use 0.5·r²/0.5 = r² (L2-like)")
print(f"                  residuals |r| ≥ 0.5 use |r| - 0.25 (L1-like, shifted)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 4: Per-epoch logging

Track:
- `train/loss` — will be slightly different magnitude than baseline (Huber near 0 is ½ smaller than L1)
- `val_avg/mae_surf_p` — directly comparable, uses L1 metric regardless of train loss
- Fraction of residuals < 0.5 (i.e., in the L2-like regime) at ep1, 30, 60 — confirms which regime dominates

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

Current loss: L1 (`.abs().mean()`) on both surf and vol channels.

After change: SmoothL1/Huber β=0.5 on both surf and vol channels. surf_weight=10 unchanged.

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/huber-loss-delta-0.5" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — Huber β=0.5 hardcoded in loss compute. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Huber-regime diagnostic:** at convergence, what fraction of residuals are in the L2-like regime (|r| < 0.5)? Is the model effectively training under L2 or L1? Report at ep1, 30, 60.
4. Param count confirmation (~407,940)
5. Epochs completed (target: 60), sec/epoch, peak GPU memory
6. Train→val loss gap (note: train loss is now Huber-scale, not directly comparable to baseline train loss)
7. **Meta-signal check:** does this experiment join the cruise WIN / in_dist LOSS pattern, break it uniformly, or break it in the WIN direction?
8. **Plain-language verdict:** WIN / WASH / LOSS — was the L2-like gradient shape near 0 useful for Lion+L1's coarse update, or does train/eval mismatch dominate?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
