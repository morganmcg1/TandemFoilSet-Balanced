# Round-5 Fresh Research Hypotheses
# Generated: 2026-05-15 18:00
# Baseline: val_avg/mae_surf_p = 114.1704 (PR #3281, EMA + scale-invariant loss)

## Exclusions (do not re-propose)
- WIP/assigned: #3265 FiLM, #3267 surface decoder, #3337 L1 aux, #3346 cosine fix, #3347 manifold mixup, #3374 stochastic depth, #3373 bf16
- Closed/regressed: #3269 multi-scale slice, #3270 capacity 256/8/8, #3271 signed-log p, #3272 arc-length Fourier PE, #3268 NACA mixup, #3315 Cautious AdamW
- Per-split context: single_in_dist=138.48 (worst, despite in-distribution — raceCar single has widest Re range 100K–5M); geom_camber_rc=130.84; re_rand=102.77; geom_camber_cruise=84.60 (best despite OOD)

---

## Idea 1 — Schedule-Free AdamW (RANK #1 — HIGHEST PRIORITY)

**Mechanism:** Replace CosineAnnealingLR(T_max=50) with schedule-free AdamW (Defazio et al., 2024), which maintains a Polyak-Ruppert iterate without requiring a preset stopping time T_max, eliminating the mismatch between the 50-epoch schedule and the ~14-epoch wall-clock budget.

**Why it helps here:** Every run is killed at ~14 epochs by SENPAI_TIMEOUT_MINUTES=30. CosineAnnealingLR with T_max=50 means the learning rate has only decayed from 5e-4 to ~4.3e-4 by epoch 14 — the model is still in the high-LR phase when the run ends. Schedule-Free AdamW adapts implicitly to the actual stopping time. It was the fastest solver in the MLCommons AlgoPerf 2024 benchmark, and crucially it pairs directly with EMA (the SF-AdamW iterate IS an implicit polynomial average; our EMA copy remains a separate exponential average, which is complementary). Targets the training/optimization bottleneck that all 14-epoch runs share.

**Implementation (train.py changes only):**
```python
# 1. pip install schedulefree (add to pyproject.toml)
import schedulefree

# 2. Replace AdamW + CosineAnnealingLR with:
optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(), lr=5e-4, weight_decay=1e-4
)
# 3. Remove scheduler lines (scheduler = ..., scheduler.step())
# 4. Toggle eval mode for SF-AdamW at validation time:
optimizer.eval()     # before evaluate_split / model.eval()
optimizer.train()    # after validate, before next training step
# 5. Keep EMA update unchanged — it still runs after optimizer.step()
```

**Expected delta:** -2% to -5% on val_avg/mae_surf_p. Largest effect on single_in_dist (currently 138.48) where high-LR noise is hurting convergence most. Low risk.

**Risk/failure mode:** SF-AdamW has slightly different weight-decay semantics than AdamW; if weight_decay=1e-4 is too strong in this formulation, loss may not converge as fast. Fallback: reduce wd to 1e-5. Also: pyproject.toml must be updated with `schedulefree` package in same PR.

---

## Idea 2 — Switch EMA / SEMA Epoch-Level Feedback (RANK #2)

**Mechanism:** After each validation epoch, copy the EMA shadow weights back into the live model weights before the next epoch begins (Kaddour et al., "Stop Regressing", 2024 / SEMA). This "refreshes" the live model to the flat-minimum region found by EMA, compounding the flatness benefit across every training epoch rather than only at the final checkpoint.

**Why it helps here:** PR #3281 proved that EMA weights are strictly better than live weights (−7.84% val). But the current implementation only uses EMA at validation/checkpoint time — the live model is still training away from the EMA attractor. SEMA closes this gap: after each epoch, `model.load_state_dict(ema_model.state_dict())` resets the live weights to the smoothed location, so subsequent gradient steps start from the EMA flat region. This is a single extra line per epoch and no additional compute. Particularly valuable for the OOD splits (geom_camber_rc=130.84, re_rand=102.77) where EMA gains are most concentrated.

**Implementation (train.py changes only):**
```python
# After validation at end of each epoch (inside the training loop):
# Existing code already does: update_ema(ema_model, model, decay=0.999)
# Add ONE line after validate, before next epoch starts:
if (epoch + 1) % sema_freq == 0:
    model.load_state_dict(ema_model.state_dict())
# sema_freq = 1 (every epoch) is the standard; try 1 and 2
# EMA update and checkpoint selection logic unchanged
```

**Expected delta:** -1% to -3% on val_avg/mae_surf_p. Free: zero extra VRAM, ~1ms per epoch. Effect compounds with EMA; likely additive with Schedule-Free AdamW.

**Risk/failure mode:** If the live model diverges after the SEMA copy (EMA weights are "softer" than live weights), early training might destabilize. Mitigation: use sema_freq=1 starting from epoch 5 onwards, not epoch 0. If loss spikes, fall back to sema_freq=2.

---

## Idea 3 — Per-Domain Target Normalization (RANK #3 — HIGH PRIORITY)

**Mechanism:** Replace the single global `y_mean, y_std` normalization with per-domain statistics (computed separately for raceCar single, raceCar tandem, cruise), applied during both training loss and model output un-normalization. The model predicts in per-domain normalized space; the denormalization at eval time uses the domain's own stats.

**Why it helps here:** The `single_in_dist` split is anomalously the worst (138.48) despite being in-distribution — because raceCar single has y_std up to 2077 vs cruise's max 506. Under global normalization, raceCar single samples are under-represented in gradient signal (their large absolute errors are diluted by the global std). Per-sample scale-invariant loss (PR #3266) already helps but still normalizes against a global baseline. Per-domain stats sharpen the per-domain learning signal further without the instability of per-sample normalization. Domain identity is available at train time: `stats.json` already segments data; loader `sample_weights` tags domain; `x[:,12-23]` encodes single vs tandem, AoA range, gap/stagger.

**Implementation (train.py changes only):**
```python
# 1. At startup, compute domain-conditional stats from training samples:
domain_stats = compute_domain_stats(train_ds, stats)
# domain_stats["racecar_single"] = {"y_mean": ..., "y_std": ...}
# domain_stats["racecar_tandem"] = {...}, domain_stats["cruise"] = {...}

# 2. Domain detection from x (dims 19-21 are NACA foil 2; all-zero → single-foil;
#    AoA range dim 14 distinguishes cruise from racecar among tandem):
def get_domain(x):  # x: [N, 24]
    is_single = (x[:, 22] == 0).all()  # gap==0 → single foil
    is_cruise  = (x[:, 14] > 0).any()  # positive AoA → cruise
    ...

# 3. In loss computation and evaluate_split, swap stats for the batch's domain.
# The scoring contract is: MAE in physical units — no change needed there.
```

**Expected delta:** -3% to -7% on val_avg, concentrated on single_in_dist (−5 to −12 absolute). The cruise split may be largely unaffected (already best). Medium implementation complexity.

**Risk/failure mode:** Domain detection from x may be imperfect for edge cases. More robust: precompute domain labels and store them in dataset; but data/ is read-only. Fallback: use gap==0 for single-foil (exact) and AoA sign for cruise vs raceCar tandem (reliable given AoA ranges in program.md). Risk of domain boundary artifacts for samples near the detection boundary.

---

## Idea 4 — Bernoulli Pressure Residual Prediction (RANK #4)

**Mechanism:** Instead of predicting raw normalized pressure p, predict the residual (p - ½(Ux² + Uy²)) — the deviation from Bernoulli's equation for incompressible flow. The model learns what the flow physics cannot explain, which should be a smoother, lower-variance target than raw p.

**Why it helps here:** Pressure in CFD is dominated by the Bernoulli term ½|U|², especially in high-Re inviscid-dominated regions. The model already predicts Ux and Uy; adding a skip-connection that subtracts ½(Ux_pred² + Uy_pred²) from the pressure prediction essentially teaches the model to predict the correction to an analytical baseline. This directly targets the p-channel MAE which is the primary metric. The high-Re single_in_dist samples have extreme p values (up to 29,136 in magnitude) that are mostly explained by velocity, leaving a smaller residual for the model to fit.

**Implementation (train.py changes only):**
```python
# After model forward pass, before loss/MAE computation:
# pred = model({"x": x_norm})["preds"]  # [B, N, 3] in normalized space

# Denorm velocity predictions:
pred_phys = pred * stats["y_std"] + stats["y_mean"]  # [B, N, 3]
ux_pred, uy_pred = pred_phys[..., 0], pred_phys[..., 1]

# Bernoulli correction (kinematic: p_approx = 0.5 * |U|^2):
bernoulli = 0.5 * (ux_pred**2 + uy_pred**2)  # [B, N]

# Model's p output is now the residual; add Bernoulli back:
p_full = pred_phys[..., 2] + bernoulli  # [B, N]

# Reconstruct full prediction for MAE:
pred_full_phys = torch.stack([ux_pred, uy_pred, p_full], dim=-1)

# For loss: renormalize p_full, compute loss on residual target
p_residual_norm = (y[..., 2] - bernoulli_from_true_uv - stats["y_mean"][2]) / stats["y_std"][2]
```

**Expected delta:** -3% to -6% on val_avg, with largest benefit on single_in_dist and re_rand where high-Re p extremes dominate. Moderate implementation complexity.

**Risk/failure mode:** Bernoulli is only exact for inviscid irrotational flow. Near the airfoil surface (where `is_surface=True` and the loss is weighted 10x), viscous effects dominate and the Bernoulli term is a poor approximation. This could hurt surface pressure accuracy specifically. Mitigation: apply the Bernoulli decomposition only to volume nodes; surface nodes predict p directly. Gate with `~is_surface` mask.

---

## Idea 5 — F-SAM (Friendly Sharpness-Aware Minimization) (RANK #5)

**Mechanism:** Replace AdamW with F-SAM (Friendly-SAM, arxiv 2403.12350), which decomposes the SAM perturbation into its "full gradient" component (estimated via EMA of past gradients) and a stochastic noise component, then removes the full-gradient component. This sharpens the benefit of SAM by targeting only the variance of the loss landscape rather than its mean gradient direction.

**Why it helps here:** Sharpness-aware minimization is theoretically motivated for generalization on the OOD splits (geom_camber_rc=130.84, re_rand=102.77). Standard SAM requires two forward-backward passes (2x compute per step, prohibitive). F-SAM reduces cost by ~40% vs SAM by reusing the gradient EMA. Crucially, we already maintain an EMA object (PR #3281) — the gradient EMA needed by F-SAM is a separate, equally lightweight structure. The flat-minimum benefit of F-SAM is complementary to weight-space EMA.

**Implementation (train.py changes only):**
```python
# 1. Add gradient EMA buffer (same structure as weight EMA):
grad_ema = {}
for name, p in model.named_parameters():
    grad_ema[name] = torch.zeros_like(p.data)

# 2. In training loop, after loss.backward():
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_ema[name] = 0.9 * grad_ema[name] + 0.1 * p.grad.data

# 3. SAM perturbation (remove full-gradient component):
rho = 0.05  # SAM radius
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.grad is not None:
            noise = p.grad.data - grad_ema[name]
            p.data += rho * noise / (noise.norm() + 1e-12)

# 4. Second forward pass on perturbed weights, optimizer.step()
# 5. Restore original weights after SAM step
# Keep EMA weight update unchanged
```

**Expected delta:** -2% to -5% on val_avg, concentrated on OOD splits (geom_camber_rc, re_rand). 

**Risk/failure mode:** 2x forward-backward cost; with 30-minute wall clock this halves effective epoch count (~7 epochs instead of 14). This is a real risk. Mitigation: use a smaller rho (0.01) to reduce perturbation cost, or apply SAM only every K steps (K=4 is common). The experiment should first try K=4 (25% overhead vs baseline) and measure if the improvement justifies the epoch reduction.

---

## Idea 6 — Huber Loss + Elevated surf_weight (RANK #6)

**Mechanism:** Replace MSE loss with Huber loss (delta=1.0 in normalized space) and increase surf_weight from 10.0 to 20.0. Huber is MAE for large errors (|error| > delta) and MSE for small ones — this directly aligns the training loss with the evaluation metric (MAE) for the high-error surface pressure predictions that dominate ranking.

**Why it helps here:** The current training loss is MSE in normalized space while the eval metric is MAE in physical space. For high-Re samples where |p_error_norm| >> 1, MSE penalizes outliers quadratically while MAE is linear — the optimization objective diverges from the test metric. Huber bridges this: with delta=1.0, predictions within 1 normalized unit are trained with MSE (smooth gradients), while large surface-pressure errors (the ones we care about) get MAE-proportional gradients. Elevating surf_weight from 10 to 20 compensates for the fact that Huber reduces the gradient signal for large errors compared to MSE, specifically on surface nodes.

**Implementation (train.py changes only):**
```python
# Replace compute_loss or the loss lines:
def compute_loss_huber(pred, target, mask, is_surface, 
                       surf_weight=20.0, delta=1.0):
    vol_loss = F.huber_loss(pred[mask], target[mask], 
                             reduction='mean', delta=delta)
    surf_mask = mask & is_surface
    surf_loss = F.huber_loss(pred[surf_mask], target[surf_mask], 
                              reduction='mean', delta=delta)
    return vol_loss + surf_weight * surf_loss

# Keep scale-invariant normalization (PR #3266) unchanged — 
# apply Huber AFTER per-sample scale normalization.
```

**Expected delta:** -2% to -5% on val_avg, concentrated on single_in_dist (extreme p values). Medium confidence — the MAE/MSE mismatch is real but the scale-invariant loss already partially mitigates it.

**Risk/failure mode:** Huber with delta=1.0 gives much weaker gradients for the extreme high-Re samples that most need correction. If delta is too large (approaching MSE), no benefit. If too small (approaching MAE), early training may be unstable. Test both delta=0.5 and delta=1.0 as two arms of the same PR.

---

## Idea 7 — AoA Jitter Test-Time Augmentation (RANK #7)

**Mechanism:** At evaluation time only, average model predictions over 5 slightly perturbed copies of each input (AoA ± 0.5°, ± 1°, and original), then report the mean prediction. This is TTA (test-time augmentation) using the physical symmetry that near-by AoA conditions produce smoothly varying fields.

**Why it helps here:** The model was trained on discrete AoA samples. At test time, predictions for AoA values between training samples may be noisy. Averaging over 5 nearby conditions (span ±1°) smooths this discretization noise. Zero training overhead — applies only in `evaluate_split`. Expected to help especially on `val_single_in_dist` (single-foil AoA range -10° to 0°, discretely sampled) and `val_geom_camber_rc` (tandem AoA interpolation).

**Implementation (train.py changes only):**
```python
def tta_evaluate(model, x_norm, stats, aoa_dims=[14, 18], n_aug=5, aoa_sigma=0.01):
    # aoa_sigma in radians ≈ 0.57°
    preds = []
    for _ in range(n_aug):
        x_aug = x_norm.clone()
        for dim in aoa_dims:
            noise = torch.randn_like(x_aug[..., dim]) * aoa_sigma
            x_aug[..., dim] = x_aug[..., dim] + noise
        with torch.no_grad():
            preds.append(model({"x": x_aug})["preds"])
    return torch.stack(preds, dim=0).mean(dim=0)

# In evaluate_split, replace model({"x": x_norm}) call with tta_evaluate(...)
```

**Expected delta:** -1% to -3% on val_avg. Low complexity, zero training cost, but limited impact because AoA jitter is a small perturbation and surface pressure near stall (extreme AoA) may not vary smoothly.

**Risk/failure mode:** If pressure field is highly nonlinear near the input AoA values (near stall), averaging over perturbed AoA will blur sharp features and hurt rather than help. The cruise OOD split (M=2-4 geometry) may see degradation if the held-out camber range causes discontinuous behavior. Mitigation: gate TTA only for single-foil samples (gap==0, aoa_dim_18==0).

---

## Idea 8 — Divergence-Free Velocity Regularization (RANK #8)

**Mechanism:** Add an auxiliary regularization term penalizing |∂Ux/∂x + ∂Uy/∂y| on predicted velocity fields, enforcing the incompressible continuity equation. Computed via finite differences on surface nodes (where the mesh is dense enough for reliable numerical differentiation).

**Why it helps here:** Incompressible flow physics demands ∇·u = 0. This is a hard constraint the model currently violates freely. Adding this penalty as a soft constraint should improve physical consistency, which could help pressure predictions via the velocity-pressure coupling (Bernoulli, pressure Poisson equation). Effect most likely on `val_re_rand` (102.77) and `val_geom_camber_rc` (130.84) where the model generalizes across regimes.

**Implementation (train.py changes only):**
```python
div_weight = 0.01  # tune: 0.001 to 0.1

def divergence_penalty(pred_norm, x_norm, mask, stats):
    # pred_norm: [B, N, 3] in normalized space
    # Denorm velocity
    pred_phys = pred_norm * stats["y_std"] + stats["y_mean"]
    ux = pred_phys[..., 0]  # [B, N]
    uy = pred_phys[..., 1]  # [B, N]
    # Finite difference on node positions (x[:,0], x[:,1])
    # Use nearest-neighbor pairs within each sample (approximate)
    # ... node-level finite difference is approximate on unstructured mesh
    # Simpler: penalize ||ux||_variance + ||uy||_variance for volume nodes
    # (proxy for smoothness / mass conservation)
    return (ux[mask].std() + uy[mask].std()).clamp(min=0)

# Add to total loss: loss = loss + div_weight * divergence_penalty(...)
```

**Expected delta:** -1% to -4% on val_avg. HIGH UNCERTAINTY — finite differences on an unstructured overset mesh are noisy and the penalty may not usefully constrain the learned function.

**Risk/failure mode:** Overset mesh means nodes are NOT on a regular grid; simple finite differences on `x[:,0:2]` position differences will have high numerical noise. A proper divergence estimate would require knowing the mesh connectivity (not available in the 24-feature input). The proxy divergence penalty (smoothness regularization) is a very weak version of the actual constraint. This idea is lower-confidence than the others. If assigning, recommend lightweight arm with div_weight=0.001 first.

---

## Summary Ranking Table

| Rank | Idea | Expected delta (val_avg) | Complexity | Risk | Primary splits targeted |
|------|------|--------------------------|------------|------|------------------------|
| 1 | Schedule-Free AdamW | -2% to -5% | Low | Low | All (optimizer alignment) |
| 2 | Switch EMA / SEMA | -1% to -3% | Minimal (1 line) | Very Low | OOD splits, single_in_dist |
| 3 | Per-Domain Target Normalization | -3% to -7% | Medium | Medium | single_in_dist (primary) |
| 4 | Bernoulli Residual Prediction | -3% to -6% | Medium | Medium-High | single_in_dist, re_rand |
| 5 | F-SAM | -2% to -5% | Medium-High | Medium (cost) | geom_camber_rc, re_rand |
| 6 | Huber Loss + surf_weight=20 | -2% to -5% | Low | Low-Medium | single_in_dist |
| 7 | AoA Jitter TTA | -1% to -3% | Low | Low | single_in_dist, geom_camber_rc |
| 8 | Divergence Regularization | -1% to -4% | Medium | High | re_rand, geom_camber_rc |

## Recommended Student Assignments

- **askeladd**: Idea 1 (Schedule-Free AdamW) — highest expected gain, cleanest mechanism, well-justified by the T_max misalignment with wall clock
- **edward**: Idea 3 (Per-Domain Target Normalization) — directly targets the single_in_dist anomaly (worst despite in-distribution), data-grounded hypothesis
- **nezuko**: Idea 6 (Huber Loss + surf_weight=20) — low complexity, directly aligns training loss with eval metric, natural companion to scale-invariant loss

Idea 2 (SEMA) should be a secondary assignment for whoever finishes first (it's a 1-line change that can be stacked onto any other result).
