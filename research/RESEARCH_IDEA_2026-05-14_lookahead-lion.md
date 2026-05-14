# Round 130 — Lookahead-Lion optimizer wrapper (k=5, α=0.5)

## Hypothesis

Wrap the Lion optimizer with **Lookahead** (Zhang et al. 2019, "Lookahead Optimizer: k steps forward, 1 step back"). Lookahead maintains a separate set of **slow weights** that update toward the **fast weights** (the actual Lion-trained weights) every `k=5` steps via `θ_slow ← θ_slow + α·(θ_fast − θ_slow)` with `α=0.5`. Tests whether explicit slow/fast separation in the optimizer state breaks the **cruise/in_dist over-specialization meta-signal** observed across 5 architecturally-distinct losing experiments.

## Why this might WIN

1. **Lookahead is a known anti-over-specialization tool.** Zhang et al. 2019 explicitly motivate Lookahead as a remedy for "high variance in inner-loop training" — exactly what produces over-specialization in small-data regimes. By anchoring the model to a slow EMA-like average, fast weights cannot drift too far into any single local basin (in_dist over-fit, in our case). Empirically shows 0.3-0.6% improvement on ImageNet across multiple optimizers.

2. **Different mechanism from EMA-eval (#2892, closed).** EMA-eval ran a parallel weight average that was used only at evaluation time and lost in our 30-min budget regime because of overhead. Lookahead is INSIDE the optimizer — every 5 steps, the SLOW weights are updated AND the FAST weights are pulled toward them. The fast weights themselves benefit from the anchor. No 2x eval overhead; ~1.02-1.05x train step overhead.

3. **Directly targets the meta-signal mechanism.** The 5-experiment meta-signal shows ANY architectural attack drifts the model toward in_dist features and away from OOD ones. Lookahead's slow weights act as an explicit drift-resistant anchor, especially in the deep cosine tail where small LR allows fast weights to wander into specialized basins. The slow-fast pull-back is designed for exactly this phenomenon.

4. **Compounds well with Lion.** Lion's sign-step is bold — large effective updates per step. Lookahead's k=5 inner loop gives Lion's bold updates room to explore (k steps forward), then the slow-weight pull-back compresses the exploration toward a more stable basin. This is a natural pairing: bold optimizer + slow-weight anchor.

5. **Cheap implementation.** ~30 lines of code. No new model parameters. Slow weights are a copy of the model weights (407,940 extra GPU buffers, ~1.6 MB at fp32 — negligible vs 16 GB peak memory).

## Why this might LOSS

1. **The 30-min budget compresses Lookahead's benefit.** Lookahead's improvement is largest with long training (≥100 epochs). At our 60-epoch budget reaching ep58 best-val, the slow weights may not have time to crystallize a useful anchor. Mitigation: Lookahead can still help via inner-loop variance reduction even with limited outer steps.

2. **k=5 may be wrong for this scale.** Standard k is 5-10; with batch_size=4 and 3000 samples we have ~375 steps/epoch, so k=5 means ~75 slow-updates per epoch. That's a lot of slow-pulls — could over-anchor. Mitigation: 0.5 alpha is conservative; we move only halfway each pull.

3. **Slow-weight anchoring could SUPPRESS the cruise-WIN signal.** If the cruise improvement in losing experiments came from beneficial drift into a different basin (specialization in one direction), anchoring the model could prevent reaching that basin. The meta-signal might be a fluctuation that Lookahead damps out.

4. **Best-val checkpoint vs slow-weight checkpoint.** Standard Lookahead implementations save the FAST weights for eval but the SLOW weights are also a valid checkpoint. We need to choose. Recommendation: report both, save fast weights for primary metric (matches baseline #2879's setup).

## Falsifiable predictions

- **WIN** (val < 30.5605): Lookahead's anchor mechanism helps. Try larger α=0.8 or k=10 next.
- **PARTIAL** (camber_cruise within 1% of baseline AND in_dist within 1% of baseline AND val_avg ≈ baseline): meta-signal partially broken — model neither over-fits in_dist NOR specializes for cruise. Try increasing α toward 1.0.
- **WASH** (val ≈ 30.5605 ± 0.5%): No effect at this k/α. Try k=10 or α=0.7.
- **LOSS** (val > 31.0): Lookahead's anchor prevents reaching useful basins. Close axis.

## Implementation

### Step 1: Add `Lookahead` class to `train.py`

Place after `class SwiGLUMLP` and before `class PhysicsAttention` (or in a `class Lookahead(torch.optim.Optimizer)` location appropriate to the file structure):

```python
class Lookahead(torch.optim.Optimizer):
    \"\"\"Lookahead wrapper: maintains slow weights pulled toward fast weights every k steps.

    Reference: Zhang et al. 2019, "Lookahead Optimizer: k steps forward, 1 step back"
    \"\"\"
    def __init__(self, base_optimizer, k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        # Reuse base_optimizer's param_groups so the LR scheduler can drive it.
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        # Allocate slow buffers shadowing every parameter.
        self.slow_weights = []
        for group in self.param_groups:
            for p in group["params"]:
                self.slow_weights.append(p.data.clone())

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            i = 0
            for group in self.param_groups:
                for p in group["params"]:
                    # θ_slow ← θ_slow + α (θ_fast − θ_slow)
                    self.slow_weights[i].add_(p.data - self.slow_weights[i], alpha=self.alpha)
                    # θ_fast ← θ_slow (pull fast weights back to the new slow position)
                    p.data.copy_(self.slow_weights[i])
                    i += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base": self.base_optimizer.state_dict(),
            "step_counter": self.step_counter,
            "slow_weights": self.slow_weights,
        }

    def load_state_dict(self, state):
        self.base_optimizer.load_state_dict(state["base"])
        self.step_counter = state["step_counter"]
        self.slow_weights = state["slow_weights"]
```

### Step 2: Wrap Lion in `Lookahead` after optimizer construction

In `train.py`, find where the Lion optimizer is constructed (search for `Lion(` or `optim.Lion`). After construction, wrap it:

```python
# Existing Lion construction:
optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))

# NEW: wrap in Lookahead
optimizer = Lookahead(optimizer, k=5, alpha=0.5)
```

The LambdaLR/CosineAnnealingLR scheduler binds to `optimizer.param_groups`, which Lookahead exposes verbatim — no scheduler change required.

### Step 3: Verify scheduler still works

The scheduler will call `scheduler.step()` per-batch as usual. Since Lookahead exposes the same `param_groups`, the LR updates happen on the SAME param dicts that Lion's inner step reads. Confirm this by printing the LR at the start of each epoch — should match baseline's cosine curve.

### Step 4: Diagnostics

Print at training start:
```python
print(f"Optimizer: Lion wrapped in Lookahead(k=5, alpha=0.5)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
print(f"Slow buffer memory: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB")  # ~1.6 MB
```

Per epoch, optionally log:
- `optim/fast_minus_slow_norm` — L2 distance between fast and slow weights. Should be non-zero between slow-updates, small after each pull-back.

## Baseline (PR #2879, current best)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 (Lookahead adds 0 trainable params; ~1.6 MB of slow buffers in optimizer state) |

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/lookahead-lion-k5-a0.5" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60. No new CLI flag — Lookahead's k=5 and α=0.5 are hardcoded. **No W&B / wandb.** SENPAI_TIMEOUT_MINUTES=30 hard cap.

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Meta-signal table** — add this PR as the 6th row to the 5-experiment over-specialization table. Did Lookahead's slow-weight anchor break the cruise/in_dist coupling?
4. Param count confirmation (407,940 trainable; ~1.6 MB slow buffers in optimizer state)
5. Epochs completed (target: 60), sec/epoch (expect within 5% of baseline), peak GPU memory (expect within 1.6 MB of baseline)
6. Train-loss vs val-loss gap (Lookahead should slightly reduce this gap)
7. **Verdict on the over-specialization meta-signal:** did Lookahead break the coupling? If YES → try k=10, α=0.7. If WASH → try k=10. If LOSS → close optimizer-wrapper axis.

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
