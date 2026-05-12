# Research Ideas — 2026-05-12 21:00
## TandemFoilSet / Transolver — Wave 3 Hypotheses

**Context:**
- Baseline: `val_avg/mae_surf_p = 98.353`, `test_avg = 87.995` (PR #1552, stoch-depth merged)
- Per-split val: `single_in_dist=119.16`, `camber_rc=111.09`, `re_rand=89.84`, `camber_cruise=73.32`
- Hard cap: 30 min / ~13-15 epochs; train.py-only edits; no wandb; no >10% per-step compute
- Already WIP: tied-projection (#1555), Gumbel-Softmax (#1553), FiLM (#1549), Fourier coords (#1548)
- Dead ends (do not repeat): surf_weight tuning, grad-clip=1.0, Kendall uncertainty, Ada-Temp scalar per head, Asymmetric Q/K
- Note: `WeightedRandomSampler` with domain-balanced weights is ALREADY active in baseline

---

## H12 (Rank 1): Local Adaptive Temperature (Per-Node)

**Mechanism:** Replace the shared scalar temperature per head with a per-node offset:
`τᵢ = clamp(self.temperature + self.temp_proj(x_mid_i), min=0.1)` where `temp_proj = nn.Linear(dim_head, 1, bias=False)`.
Each mesh point independently controls how sharply it concentrates on a physics slice, rather than using one global sharpness per head.

**Why it should help:** The exhausted Ada-Temp experiments (PR #1514) used a scalar per head — every node got the same sharpness. Local adaptive temperature is a strictly more expressive variant from Transolver++ (arXiv:2502.02414). In that ablation, local Ada-Temp gave ~46% of the total relative error reduction. The `single_in_dist` split (worst at 119.16) likely contains nodes spanning boundary layers, wake regions, and far-field where optimal softness differs by orders of magnitude.

**Implementation sketch in `PhysicsAttention.__init__`:**
```python
# After self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
self.temp_proj = nn.Linear(self.dim_head, 1, bias=False)
```
**In `PhysicsAttention.forward`:**
```python
# Replace the single line:
#   slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
# with:
temp_offset = self.temp_proj(x_mid)  # [B, H, N, 1]
local_temp = torch.clamp(self.temperature + temp_offset, min=0.1)
slice_weights = self.softmax(self.in_project_slice(x_mid) / local_temp)
```
**Critical pitfall:** `Transolver.__init__` calls `self.apply(self._init_weights)` which re-initializes all `nn.Linear` weights with `trunc_normal_(std=0.02)`. This will overwrite the zero-init of `temp_proj` if done before `self.apply`. The fix: add `temp_proj` to `PhysicsAttention.__init__` normally (no special init there), and it will be initialized by `self.apply` automatically. The `torch.ones * 0.5` for `self.temperature` lives in `nn.Parameter`, not in `_init_weights`, so it is unaffected. No extra re-zeroing needed.

**Compute overhead:** One `nn.Linear(dim_head, 1)` per head per forward pass. For dim_head=32 (128/4), this is 32 multiplies per node per head — roughly 1-2% overhead. Well within the 10% cap.

**Expected direction:** `val_avg/mae_surf_p` decrease (improvement), especially on `single_in_dist` which spans the most diverse physics regimes.

**Risk:** The node-level temperature offset can produce very small `local_temp` values causing numerical overflow before the clamp. The `clamp(min=0.1)` guard is essential. Also verify `temp_proj` weight ends up small after `_init_weights` — `trunc_normal_(std=0.02)` on a 1-output linear is fine (outputs will be ~0.02 * mean(x_mid), which is a small perturbation on the 0.5 base).

---

## H13 (Rank 2): EMA Weight Averaging (decay=0.999)

**Mechanism:** Maintain an exponential moving average (EMA) of the model weights throughout training. Use the EMA copy exclusively for validation and test evaluation; the live model continues to receive gradient updates.

**Why it should help:** The 30-minute cap forces stopping at ~13-15 epochs. The live model weights at epoch 15 may be on a sharp loss landscape valley that generalizes poorly to OOD splits. EMA weights track a smoother, more stable version of the weight trajectory. arXiv:2411.18704 shows EMA is equivalent to implicit ensemble averaging and consistently improves OOD robustness and calibration at negligible cost. With only 13-15 epochs, the EMA at decay=0.999 retains information from all epochs — it does not warm up and reset like longer-run EMA.

**Implementation sketch (train-loop level):**
```python
# After model creation:
ema_model = Transolver(**model_config).to(device)
ema_model.load_state_dict(model.state_dict())
ema_decay = 0.999

# Inside the training loop, after optimizer.step():
with torch.no_grad():
    for p_ema, p_live in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_decay).add_(p_live.data, alpha=1.0 - ema_decay)

# For validation, use ema_model.eval() instead of model.eval()
# For test evaluation, load EMA weights from ema_model (do NOT torch.save/load the live model checkpoint;
# instead save and load ema_model.state_dict())
```
**Key subtlety for eval in evaluate_split:** Pass `ema_model` to evaluate_split during the validation loop. The checkpoint saved to disk for test evaluation should be the EMA state dict, not the live model. This means changing `torch.save(model.state_dict(), model_path)` to `torch.save(ema_model.state_dict(), model_path)` in the best-checkpoint block.

**Compute overhead:** EMA update is an in-place multiply + add on all parameters. At ~3M params, this is ~6M FLOPs per step — negligible relative to the forward/backward pass (billions of FLOPs). Essentially zero overhead.

**Expected direction:** Primary improvement on the OOD splits (`camber_rc`, `re_rand`, potentially `single_in_dist`). May slightly hurt the easiest split (`camber_cruise`) if that split is already well-fit.

**Risk:** With only 13-15 epochs, the EMA at decay=0.999 retains ~1 - 0.999^15 ≈ 1.5% weight on the final model — meaning the EMA is almost entirely dominated by earlier, less-trained checkpoints. Consider using decay=0.99 or 0.995 for short runs. At decay=0.99, the EMA has a timescale of ~100 steps. For ~375 steps/epoch × 15 epochs = ~5600 steps, decay=0.999 gives a timescale of 1000 steps = ~2.7 epochs, which is appropriate. Use 0.999 as specified but include 0.995 as a fallback.

---

## H14 (Rank 3): Cosine Schedule T_max Alignment

**Mechanism:** Change `CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)` to `T_max=MAX_EPOCHS` is correct but `MAX_EPOCHS` is set to 50 from `cfg.epochs=50`. Since the 30-minute cap terminates training at epoch ~13-15, the scheduler is using T_max=50 but only running ~28% of the cosine cycle. The LR at epoch 15 with T_max=50 is: `lr_15 = 0.5 * (1 + cos(π * 15/50)) * 5e-4 ≈ 0.5 * (1 + cos(54.6°)) * 5e-4 ≈ 0.5 * 1.579 * 5e-4 ≈ 3.95e-4`. By contrast, with T_max=15 the LR reaches near-zero at epoch 15. **The hypothesis:** the current schedule is running with the LR at ~79% of its initial value when training terminates, meaning the model never benefits from the final low-LR refinement phase. Setting `T_max=15` (or dynamically from `MAX_TIMEOUT_MIN`-estimated epochs) should allow the full cosine cycle to complete within the budget.

**Implementation — single line change in train.py:**
```python
# Replace:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
# With:
EFFECTIVE_EPOCHS = 15  # empirically ~13-15 epochs in 30 min at batch_size=4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EFFECTIVE_EPOCHS)
```
Alternatively, use the timeout-adaptive estimate: set `T_max=max(int(MAX_TIMEOUT_MIN * 0.5), 10)` as a heuristic.

**Compute overhead:** Zero — purely a scheduler parameter.

**Expected direction:** Improvement across all splits, especially in later epochs when the low-LR refinement phase (currently truncated) would have allowed the optimizer to settle into a flatter, better-generalizing minimum.

**Risk:** If the model is undertrained with T_max=15, the LR might drop too aggressively in early epochs, slowing convergence before the model has properly fit the training data. A diagnostic: look at the per-epoch training loss curve — if it is still declining at epoch 13-15 with T_max=50, then the LR is not the bottleneck. If it has plateaued, the schedule fix matters more. Recommend trying T_max=15 first, and if worse, try T_max=20 as a middle ground.

---

## H15 (Rank 4): Gradient Clipping at Natural Norm (~25)

**Mechanism:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)` before `optimizer.step()`. The dead-end was `max_norm=1.0` which was severely too aggressive (typical pre-clip norms for this model on large-mesh batches are ~10-50). A clip at ~25 removes only catastrophic gradient spikes from large-mesh samples (RaceCar tandem ~127K nodes, Cruise tandem ~210K nodes) while leaving typical gradients unaffected.

**Why it matters here:** Large-mesh samples produce gradients with larger norms simply because more nodes contribute. Without clipping, a single large-mesh batch can produce a gradient spike that disturbs convergence. The `single_in_dist` split (worst, 119.16) uses RaceCar meshes (~85K nodes) while `camber_cruise` (best, 73.32) uses Cruise meshes (~210K nodes) — counterintuitively the larger mesh may be better regularized because more nodes average out per batch.

**Implementation:**
```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)
```

**Compute overhead:** One pass over all parameters for the norm computation. For ~3M params: negligible (~1% overhead).

**Expected direction:** Reduced variance in training loss curve; potential improvement on `single_in_dist` and `camber_rc` (both RaceCar mesh) which currently are the worst-performing splits.

**Risk:** The right `max_norm` is dataset-dependent. If the natural gradient norm is actually below 25, this is a no-op. Recommend logging `grad_norm` for 2-3 epochs first (add `actual_norm = sum(p.grad.data.norm(2)**2 for p in model.parameters() if p.grad is not None)**0.5` before clipping) to confirm the norm range. Key difference from the dead-end: clip=1.0 was 20-50x below the natural norm; clip=25 is designed to be at or just above the natural norm.

---

## H16 (Rank 5): log1p Target Reparameterization for Surface Pressure

**Mechanism:** Before computing the L1 loss, apply `torch.sign(y_norm) * torch.log1p(y_norm.abs())` to normalized pressure predictions and targets. Invert with `torch.expm1` for final evaluation. This compresses the heavy tail of the normalized pressure distribution, making the L1 loss weight under-predicted high-pressure events more equally with over-predicted low-pressure events.

**Why it might help:** Surface pressure `p` on foils has a strongly skewed distribution: high-suction peaks near the leading edge have large normalized values, while trailing-edge pressure is near the freestream. In L1 loss on normalized space, the leading-edge suction peaks dominate the gradient because their residuals are larger in absolute terms. The `single_in_dist` split at 119.16 vs `camber_cruise` at 73.32 suggests that high-suction configurations (single foil, high AoA) are harder to predict — consistent with a loss that over-indexes on large-magnitude errors.

**Implementation:**
```python
# After y_norm is computed, apply compression for loss only:
def log1p_compress(t):
    return torch.sign(t) * torch.log1p(t.abs())

# In the training step, replace:
#   abs_err = (pred - y_norm).abs()
# with (only for pressure channel, dim index 2):
pred_p = log1p_compress(pred[..., 2:3])
y_norm_p = log1p_compress(y_norm[..., 2:3])
pred_uv = pred[..., :2]
y_norm_uv = y_norm[..., :2]
pred_compressed = torch.cat([pred_uv, pred_p], dim=-1)
y_compressed = torch.cat([y_norm_uv, y_norm_p], dim=-1)
abs_err = (pred_compressed - y_compressed).abs()
```
**Note:** The validation `evaluate_split` function uses `pred_orig = pred * stats["y_std"] + stats["y_mean"]` before calling `accumulate_batch` — this bypasses the log1p compression entirely. So the compression only affects the training loss, not the eval metric. This is correct behavior.

**Compute overhead:** Two extra element-wise ops per training step. Essentially zero.

**Expected direction:** Most benefit on `single_in_dist` which has the most variable pressure distribution. May slightly hurt `camber_cruise` if that split's pressure distribution is already Gaussian-like.

**Risk:** If normalized pressure is already near-Gaussian (confirmed by looking at distribution), log1p has no effect. The bigger risk is that compressing predictions in loss space but evaluating in original space creates a mismatch that confuses the optimizer — it will try to minimize log1p-MAE but be graded on original-space MAE. This is an acceptable proxy mismatch only if the log1p-MAE and original MAE are well-correlated, which holds when the distribution is moderately heavy-tailed.

---

## H17 (Rank 6): Pressure-Weighted Bias in the Slice Projection Output Layer

**Mechanism:** Add a learned scalar bias to the final output projection for the pressure channel (index 2) while leaving Ux/Uy unchanged. The final `mlp2` in the last TransolverBlock outputs 3 dimensions; the pressure dimension is the one driving `mae_surf_p`. A separate learnable bias term for pressure, initialized to the training set mean pressure in normalized space (which is 0, by construction), allows the model to shift its pressure predictions globally without interfering with velocity gradient flow.

**Why it might help:** The model uses a shared output head for all 3 output fields. In normalized space the mean is 0 for all channels, so a zero-initialized bias is correct. But if the model's intermediate representation is biased toward velocity (which has much higher normalized variance), the shared output head may be compromised. A pressure-specific final bias (and scale) lets the model independently calibrate the pressure output.

**Implementation:**
```python
# In TransolverBlock.__init__, after mlp2 is created (only when last_layer=True):
if self.last_layer:
    self.p_output_bias = nn.Parameter(torch.zeros(1))
    self.p_output_scale = nn.Parameter(torch.ones(1))

# In TransolverBlock.forward, replace:
#   return self.mlp2(self.ln_3(fx))
# with (last layer):
out = self.mlp2(self.ln_3(fx))
# Apply learned scale+bias to pressure channel only:
out_p = out[..., 2:3] * self.p_output_scale + self.p_output_bias
return torch.cat([out[..., :2], out_p], dim=-1)
```
**Critical pitfall:** `self.apply(self._init_weights)` in `Transolver.__init__` initializes only `nn.Linear` instances. The new `p_output_bias` and `p_output_scale` are plain `nn.Parameter` objects, not `nn.Linear`, so they are NOT re-initialized by `_init_weights`. Their initial values of `zeros(1)` and `ones(1)` respectively are preserved through the `self.apply` call. No special post-apply re-init needed.

**Compute overhead:** Two scalar multiply-adds per node per forward pass. Genuinely negligible.

**Expected direction:** Small but consistent improvement on all `mae_surf_p` splits. The primary benefit is decoupling pressure calibration from the shared output head's gradient signal.

**Risk:** This is a low-capacity change — 2 parameters. The benefit may be indistinguishable from noise at the scale of our runs. If the model is already well-calibrated for pressure mean/scale, this adds nothing. Worth testing only if H12-H15 are all in flight.

---

## Summary Table

| Rank | ID  | Title                         | Compute Overhead | Primary Mechanism                         | Target Split(s)          |
|------|-----|-------------------------------|------------------|-------------------------------------------|--------------------------|
| 1    | H12 | Local Adaptive Temperature    | ~1-2%            | Per-node slice sharpness (Transolver++)   | single_in_dist, all      |
| 2    | H13 | EMA Weight Averaging 0.999    | ~0%              | Implicit ensemble / OOD smoothing        | camber_rc, re_rand       |
| 3    | H14 | Cosine T_max Alignment (=15)  | 0%               | Full LR cycle in 30-min budget            | all                      |
| 4    | H15 | Gradient Clip max_norm=25     | ~1%              | Suppress large-mesh gradient spikes       | single_in_dist, camber_rc|
| 5    | H16 | log1p Pressure Compression    | ~0%              | Compress heavy-tailed pressure loss       | single_in_dist           |
| 6    | H17 | Pressure Output Scale+Bias    | ~0%              | Decouple pressure head calibration        | all mae_surf_p           |

**WIP (do not duplicate):** #1555 tied-projection, #1553 Gumbel-Softmax, #1549 FiLM, #1548 Fourier coords
**Dead ends (do not repeat):** surf_weight, grad-clip=1.0, Kendall, Ada-Temp scalar, Asymmetric Q/K, unified_pos
**Already active in baseline:** WeightedRandomSampler with domain-balanced inverse-frequency weights
