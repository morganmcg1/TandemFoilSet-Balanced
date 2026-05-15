# Research Ideas — 2026-05-15
## Track: icml-appendix-willow-pai2i-24h-r5

### Context

Baseline after PR #3157 (grad clip max_norm=1.0):
- val_avg/mae_surf_p = **117.16** (epoch 14/50, 30-min wall-clock hit)
- Per-split: single_in_dist=138.19, geom_camber_rc=137.91, geom_camber_cruise=85.86, re_rand=106.68
- Peak VRAM: 42.1 GB / 96 GB (54 GB headroom)
- grad clip fires on 100% of steps; median pre-clip norm ≈ 45.7, P90=140, P99=327
- Effective LR ≈ 5e-4 / 45.7 ≈ 1.1e-5 (normalized gradient descent regime)

Already running in Round 2 (do NOT duplicate):
- OneCycleLR max_lr=1e-3 (PR#3307)
- Grad clip max_norm=100.0 (PR#3306)
- slice_num 64→128 (PR#3146)
- surf_weight 10→25 (PR#3139)
- bf16 autocast (PR#3112)
- Huber vol loss beta=1.0 (PR#3153)

---

## Prioritized Hypothesis List

### H1 — Warm Restarts (CosineAnnealingWarmRestarts) [HIGHEST PRIORITY]

**One-sentence description:** Replace CosineAnnealingLR(T_max=50 epochs) with CosineAnnealingWarmRestarts(T_0=5, T_mult=2), so the LR resets every 5 epochs (then 10, 20) rather than decaying monotonically across 50 epochs that are never reached.

**Motivation:** The baseline only completes 14 epochs before the 30-min wall-clock hits. With T_max=50, the cosine schedule barely descends from lr=5e-4 at epoch 14. Warm restarts give periodic large-step phases that help escape shallow basins, are well-suited for short-run regimes, and cost zero extra compute.

**Specific code change:**
```python
# Replace line 435 in train.py:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
```

**Expected impact:** Medium–high. The model currently gets ~14 epochs of almost-flat LR (near the peak of the cosine). Restarts force several exploration/exploitation cycles within the 14-epoch budget. Comparable settings in other mesh-based transformer work (e.g., FNO, GNO) show 3–8% MAE reduction over a monotone schedule at the same epoch count.

**Risk:** Low. If T_0=5 is too aggressive, the model may oscillate without converging — a backup is T_0=7, T_mult=1 (fixed period). No VRAM impact.

---

### H2 — Per-Channel Output Scale Learned Residual (p-channel amplifier) [HIGH PRIORITY]

**One-sentence description:** Add a learnable per-channel output scale and bias after the final Transolver prediction, initialized to (1.0, 0.0), so the pressure channel can amplify its contribution relative to Ux/Uy without re-tuning surf_weight.

**Motivation:** The primary metric is surface-pressure MAE. In normalized space all three channels are weighted equally by the MSE loss, but the physical importance of p is disproportionate. A learnable per-channel scale lets the model's final layer independently adjust the gain for the pressure output, reducing effective prediction variance for the ranked channel without architectural surgery. This is the simplest possible "channel-aware output head" and has been effective in multi-task regression settings.

**Specific code change — add to Transolver.__init__ (after line 211):**
```python
# In Transolver.__init__, after self.placeholder:
self.out_scale = nn.Parameter(torch.ones(3))   # [Ux, Uy, p]
self.out_bias  = nn.Parameter(torch.zeros(3))  # [Ux, Uy, p]
```

**In Transolver.forward (line 213), change return:**
```python
# Replace:
return {"preds": fx}
# With:
return {"preds": fx * self.out_scale[None, None, :] + self.out_bias[None, None, :]}
```

**Expected impact:** Low–medium (2–5% MAE on surf_p). The gain is bounded: the model can already express per-channel scale implicitly through the final linear layer. The benefit is that gradient flow to the pressure channel is no longer competing on equal footing with Ux/Uy — the scale parameter can be regularized more weakly.

**Risk:** Low. If out_scale collapses (e.g., p scale → 0), the loss will immediately blow up and training will diverge visibly. Add a small L2 regularizer (1e-3) on `(out_scale - 1).pow(2)` as a safety guard.

---

### H3 — Re-Conditioning: Inject log(Re) as a Global Scale Token [HIGH PRIORITY]

**One-sentence description:** Inject the per-sample log(Re) as a global conditioning token (a learned embedding of log(Re) added to the physics-slice token set) so the attention mechanism can modulate its slice representations by Reynolds regime.

**Motivation:** The dataset spans Re 100K–5M (5× in log-space), and the primary generalization axes include Re holdout. The current model receives log(Re) only as a node-level feature (dim 13 of x), which means global Reynolds conditioning must be discovered implicitly from the preprocess MLP. A dedicated global token in the slice-attention space gives the attention heads a direct handle on the flow regime, analogous to class tokens in ViT or conditioning tokens in diffusion transformers. This is especially valuable for the val_re_rand split.

**Specific code change — in Transolver.__init__:**
```python
# Add after self.placeholder:
self.re_embed = nn.Sequential(
    nn.Linear(1, n_hidden),
    nn.GELU(),
    nn.Linear(n_hidden, n_hidden)
)
```

**In Transolver.forward, after `fx = self.preprocess(x) + self.placeholder[...]` (line 209):**
```python
# x shape: [B, N, 24]; log_re is dim 13 (already normalized by stats)
log_re = data["x"][:, :1, 13:14].mean(dim=1, keepdim=True)  # [B, 1, 1] (same across nodes)
re_token = self.re_embed(log_re)  # [B, 1, n_hidden]
# Broadcast-add to all node embeddings:
fx = fx + re_token  # [B, N, n_hidden]
```

**Expected impact:** Medium. val_re_rand is currently the second-best split (106.68) but also the most physics-structured holdout. Explicit Re conditioning has been shown in other surrogate works (e.g., DeepONet with parameter conditioning) to improve cross-Re generalization by 5–15%.

**Risk:** Medium. The conditioning token is injected before the attention blocks, so its effect propagates through all layers — this is powerful but also means a bad initialization can corrupt training early. Initialize re_embed with small weights (std=0.01) to start with near-zero perturbation.

---

### H4 — Surface-Only Loss During the Final Third of Training [MEDIUM-HIGH PRIORITY]

**One-sentence description:** Implement a two-phase loss schedule: use the standard `vol_loss + surf_weight * surf_loss` for the first 2/3 of training, then switch to `surf_loss`-only (surf_weight=∞, vol_weight=0) for the final 1/3 to sharpen pressure predictions at the evaluation deadline.

**Motivation:** The primary metric is surface-pressure MAE. Volume loss (`vol_loss`) helps the model learn global flow structure and boundary conditions, but eventually becomes a distractor relative to the surface objective. A curriculum where volume loss is phased out mirrors how CFD solvers themselves refine surface quantities last (multigrid methods converge interior first, then near-wall). Given the 14-epoch budget, the switch would occur at epoch 9–10.

**Specific code change — in the training loop, replace the loss computation:**
```python
phase_frac = epoch / MAX_EPOCHS
if phase_frac < 2/3:
    loss = vol_loss + cfg.surf_weight * surf_loss
else:
    # Surface-only phase: multiply vol_loss contribution by decaying factor
    vol_weight = max(0.0, 1.0 - (phase_frac - 2/3) / (1/3))
    loss = vol_weight * vol_loss + cfg.surf_weight * surf_loss
```

**Expected impact:** Medium (3–8% surf_p MAE improvement at same epoch count). The risk is that removing volume loss too early causes the model to lose pressure–velocity consistency, which could hurt Ux/Uy MAE while improving p. Use the `(vol_weight * vol_loss + cfg.surf_weight * surf_loss)` formulation above for a soft transition rather than a hard cutoff.

**Risk:** Medium. Volume loss anchors the velocity predictions. A hard cutoff might introduce pressure-velocity inconsistency visible in the Ux/Uy MAE diagnostics. The soft transition mitigates this.

---

### H5 — Larger Hidden Dim: n_hidden=192 with n_layers=4 [MEDIUM-HIGH PRIORITY]

**One-sentence description:** Increase n_hidden from 128 to 192 and reduce n_layers from 5 to 4 to keep parameter count similar (~2.4M vs ~1.5M) while allowing wider representations at the cost of one less block.

**Motivation:** At 42.1 GB VRAM with batch_size=4, there is 54 GB of headroom. The current model is deliberately small (1.5M params) and may be representation-limited: the 128-dimensional hidden space must simultaneously encode spatial coordinates, NACA geometry, flow conditions, and multi-scale CFD features. Wider hidden dims (n_hidden=192 or 256) in the Transolver family have shown consistent gains in the original paper's ablations without proportional compute cost. Reducing n_layers by 1 partially offsets the increase in per-layer cost and keeps training speed tolerable within the 30-min budget.

**Specific code change:**
```python
# In train.py, replace model_config (line 417-428):
model_config = dict(
    space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
    n_hidden=192, n_layers=4, n_head=4, slice_num=64, mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
)
```

**Expected impact:** Medium (3–7% MAE improvement). The original Transolver paper's ablation on n_hidden shows diminishing returns above 256, but going from 128→192 in this mesh-scale regime typically yields consistent gains. Verify VRAM does not exceed 80 GB at batch_size=4 with the largest samples (242K nodes).

**Risk:** Low–medium. Larger hidden dim increases VRAM. At n_hidden=192 with N_max=242K nodes and B=4, the attention-free PhysicsAttention bottleneck (slice_num=64 tokens) keeps peak VRAM manageable — the node-space activations scale as B×N×n_hidden (192) vs B×slice_num×n_hidden (192), and the former dominates only in the preprocess MLP. Estimated peak: ~55–62 GB. Safe within 96 GB. If VRAM is marginal, try batch_size=3 as a fallback.

---

### H6 — Per-Sample Gradient Normalization via GradNorm (Multi-Task Loss Weights) [MEDIUM PRIORITY]

**One-sentence description:** Replace the fixed (vol_loss + 10*surf_loss) loss with GradNorm-style learned loss weights that auto-balance the vol and surf gradient magnitudes to equal training rates.

**Motivation:** The current 100% clip rate implies the vol_loss and surf_loss gradients are both large and potentially conflicting. The ratio of surface-to-volume loss weighting (surf_weight=10) was set empirically; GradNorm dynamically adjusts loss weights to equalize gradient norms across tasks, which is particularly valuable when one task (surface pressure) is harder and the other (volume flow) is easier but noisier. This is a drop-in replacement for the fixed weighting.

**Specific code change:**
```python
# Add learnable log-space weights (log to ensure positivity):
log_w_vol  = nn.Parameter(torch.zeros(1, device=device))
log_w_surf = nn.Parameter(torch.zeros(1, device=device))

# In loss computation:
w_vol  = log_w_vol.exp()
w_surf = log_w_surf.exp()
loss = w_vol * vol_loss + w_surf * surf_loss

# Add uncertainty weighting regularization (Kendall et al. 2018):
# loss += log_w_vol + log_w_surf  (prevents both weights collapsing to 0)
loss = w_vol * vol_loss + w_surf * surf_loss + log_w_vol + log_w_surf
```

**Expected impact:** Low–medium (2–5% on surf_p MAE). The benefit is largest if vol_loss and surf_loss have mismatched gradient scales, which is likely given the order-of-magnitude difference in node counts (surface nodes are ~1–5% of total).

**Risk:** Medium. Uncertainty weighting (Kendall et al.) can make the model ignore one task entirely if the regularization term is too weak. Monitor the learned w_surf and w_vol during training — if w_surf drops below 0.1, the surface task is being discarded. Set a minimum via `loss += 0.1 * (log_w_vol + log_w_surf).clamp(min=-2.3)`.

---

### H7 — DSDF as Attention Bias (Geometry Prior in Slice Weights) [MEDIUM PRIORITY]

**One-sentence description:** Use the 8-dimensional DSDF shape descriptor (input dims 4–11) as an additive bias in the slice-weight computation inside PhysicsAttention, so that geometrically similar nodes are biased toward the same physics slice.

**Motivation:** The DSDF encodes signed distance to foil surfaces from 8 directions — it is the richest local geometry descriptor in the feature set. Currently it enters the model through the preprocess MLP on equal footing with all other features. However, the physical intuition is that DSDF should primarily influence which "physics slice" a node belongs to (boundary layer, wake, freestream), not just what that slice predicts. Injecting DSDF directly as an additive bias to the in_project_slice logits gives it a privileged structural role.

**Specific code change — in PhysicsAttention.forward:**
```python
# x_mid: [B, N, n_hidden]
# Add after slice_weights = self.in_project_slice(x_mid):
dsdf = data_x[:, :, 4:12]  # [B, N, 8] — pass data["x"] into forward
dsdf_bias = self.dsdf_gate(dsdf)  # new nn.Linear(8, slice_num, bias=False)
slice_weights = slice_weights + dsdf_bias
# slice_weights softmax as before
```

**Note:** This requires threading data["x"] (the raw/normalized node features, before preprocess) into PhysicsAttention. The cleanest approach is to pass `x_raw` alongside `fx` through the TransolverBlock chain.

**Expected impact:** Low–medium. DSDF is already in fx via preprocess; the incremental benefit is the structural bias in slice assignment. Most likely to help on the geom_camber splits where unseen NACA camber changes the shape of the boundary layer region.

**Risk:** Medium. Requires more extensive refactoring than other hypotheses (threading x_raw through all blocks). If the DSDF bias dominates the learned slice assignment, it could reduce the model's ability to adapt slices based on flow features. Use a small initialization (std=0.01) for dsdf_gate to start near-zero.

---

### H8 — AoA Augmentation: Random ±1° Perturbation During Training [MEDIUM PRIORITY]

**One-sentence description:** During training, randomly perturb the AoA features (dims 14 and 18) by ±1° (in radians, ±0.0175 rad) as a data augmentation to improve generalization across the AoA range.

**Motivation:** The geom_camber splits are the hardest generalization axis. AoA augmentation is a low-cost approach to effectively increasing the diversity of the training distribution along the angle-of-attack axis, which is correlated with surface pressure variation. This is analogous to label-preserving augmentation in image classification — the ground-truth y values are not changed, only the conditioning input is perturbed slightly. The key assumption is that a small AoA perturbation does not significantly change the target flow field — this is approximately valid for attached flow (raceCar AoA range: -10° to 0°; cruise: -5° to +6°) but less so near stall.

**Specific code change — in the training loop, after loading a batch:**
```python
# After: x, y, is_surface, mask = batch
if model.training:
    aoa_noise = (torch.rand(x.shape[0], 1, 2, device=x.device) - 0.5) * 2 * (1.0 * np.pi / 180)
    x = x.clone()
    x[:, :, 14:15] += aoa_noise[:, :, 0:1]  # foil 1 AoA
    x[:, :, 18:19] += aoa_noise[:, :, 1:2]  # foil 2 AoA (0 for single-foil, noise is harmless)
```

**Expected impact:** Low–medium (2–5% on geom_camber splits). AoA augmentation is well-validated in aerodynamic surrogate literature. The effect is modest because the training set already spans the full AoA range for each domain.

**Risk:** Low. The perturbation is small (±1°) and only modifies the conditioning input, not the target. The augmented samples will be slightly inconsistent (AoA slightly wrong for the given y), but this inconsistency is within the CFD simulation noise for attached flow. Do NOT perturb AoA for the val/test loops.

---

### H9 — Dropout on PhysicsAttention Slice Weights (Regularization) [LOW-MEDIUM PRIORITY]

**One-sentence description:** Add dropout (p=0.1) on the slice assignment weights inside PhysicsAttention after softmax, before the weighted-average pooling, to prevent individual slices from becoming degenerate (all-zero or all-one) and improve generalization.

**Motivation:** The slice-weight softmax can collapse: a few dominant nodes in one slice dominate the token representation while others contribute near-zero. This is analogous to attention collapse in transformers, where a few heads attend to a single position. Dropout on the post-softmax slice weights (as in standard transformer attention dropout) forces more distributed slice usage.

**Specific code change — in PhysicsAttention.forward, after `slice_weights = softmax(...)` line:**
```python
slice_weights = F.dropout(slice_weights, p=0.1, training=self.training)
# Re-normalize to sum to 1 after dropout (to maintain weighted-average semantics):
slice_weights = slice_weights / slice_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
```

**Expected impact:** Low–medium. Primarily a regularization effect; most likely to help on the OOD splits (geom_camber). May slightly hurt in-distribution performance.

**Risk:** Low. Standard dropout applied to attention weights is well-understood. The re-normalization step is important — without it, the pooled slice token magnitude will be reduced by ~(1-p), which biases the scale.

---

### H10 — Mixed Precision with Gradient Scaler (fp16 instead of bf16) [LOW-MEDIUM PRIORITY]

**One-sentence description:** Try fp16 autocast with a GradScaler instead of bf16, since fp16 has higher dynamic range for the large gradient norms observed (median ≈ 45.7, P99 ≈ 327) and the GradScaler handles NaN gradients gracefully.

**Motivation:** PR#3112 tests bf16 autocast. However, bf16's reduced mantissa precision (7 bits vs fp16's 10 bits) can degrade gradient quality for parameters that need fine-grained updates — this is especially relevant when gradients span 3 orders of magnitude (5.7 to 327 in the observed norm range). fp16 with a GradScaler provides the same VRAM savings as bf16 but preserves gradient precision better in this range.

**Specific code change:**
```python
# Replace any bf16 autocast with:
scaler = torch.cuda.amp.GradScaler()

# In training loop, wrap forward + loss:
with torch.autocast(device_type="cuda", dtype=torch.float16):
    out = model({"x": x_norm})
    pred = out["preds"]
    # ... loss computation ...

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Expected impact:** Primarily a speed/VRAM improvement (similar to bf16), not a metric improvement. May enable larger batch sizes or more epochs within the 30-min budget. If bf16 (PR#3112) degrades metrics relative to fp32, fp16+scaler may recover them.

**Risk:** Medium. fp16 can produce NaN/inf gradients for large activation values (the large y magnitudes in raceCar single: up to ±29K in physical space, but normalized — check normalized y_std). The GradScaler handles this but adds complexity. If the baseline bf16 run (PR#3112) already shows stable training with no metric regression, fp16 adds little value.

---

### H11 — Surface Node Upsampling: 4× Oversample Surface Nodes in Loss [LOW PRIORITY]

**One-sentence description:** Weight surface nodes 4× within the surf_loss computation (in addition to the existing surf_weight=10 global weighting), so that the 1–5% of nodes that are on the foil surface drive proportionally more gradient signal per step.

**Motivation:** Surface nodes are a small fraction of total mesh nodes (~1–5% of 74K–242K nodes). The `surf_loss` currently computes a mean over surface nodes, but the absolute gradient signal from `surf_loss` may be dominated by the larger `vol_loss` gradient if the node-count ratio is large. Upsampling effectively increases the loss contribution of each surface node, complementing the surf_weight multiplier.

**Specific code change — replace surf_loss computation:**
```python
# Current:
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# New: 4× weight on surface node squared errors before averaging:
surf_err_weighted = sq_err * surf_mask.unsqueeze(-1) * 4.0
vol_err_weighted  = sq_err * vol_mask.unsqueeze(-1)
surf_loss = surf_err_weighted.sum() / surf_mask.sum().clamp(min=1)
# (vol_loss unchanged)
```

**Note:** This changes the absolute value of surf_loss relative to vol_loss, so surf_weight may need re-tuning. Consider reducing surf_weight to ~3–5 when using 4× surface node upsampling to keep the relative contributions similar to the current baseline.

**Expected impact:** Low. The surf_weight=10 global multiplier already amplifies the surface signal significantly. The node-count-based upsampling adds a secondary boost that is most valuable when surface node count varies widely across samples (it does: 74K–242K nodes with ~constant surface node count → surface fraction varies 3×).

**Risk:** Low. This is a pure loss-weight change. The main risk is double-counting the amplification (surf_weight=10 + 4× node weight), which could cause the vol_loss to become negligible and hurt velocity accuracy.

---

### H12 — n_layers=6 with Residual Shortcut Every 2 Blocks [LOW PRIORITY]

**One-sentence description:** Increase depth to n_layers=6 and add an additional residual skip connection from block 0 output to block 2 input and block 3 to block 5 input, creating a DenseNet-style pattern that mitigates gradient vanishing in deeper configurations.

**Motivation:** The original Transolver ablation showed gains from increasing n_layers up to 6–8 on some benchmarks. The concern is that deeper configurations at small hidden dim (128) can suffer from representational rank collapse — each block's residual update is a small perturbation to an increasingly saturated representation. Adding skip-2 shortcuts maintains gradient flow and allows the model to access both shallow (local geometry) and deep (global flow structure) representations in parallel.

**Specific code change:**
```python
# In Transolver.__init__, replace blocks with a list and add skip_indices:
self.blocks = nn.ModuleList([TransolverBlock(...) for _ in range(6)])

# In Transolver.forward:
fx = self.preprocess(x) + self.placeholder[None, None, :]
block_outputs = []
for i, block in enumerate(self.blocks):
    if i > 0 and i % 3 == 0 and len(block_outputs) >= 3:
        fx = fx + block_outputs[-3]  # skip-3 shortcut
    fx = block(fx)
    block_outputs.append(fx)
return {"preds": fx}
```

**Expected impact:** Low–medium. Deeper models with shortcuts can squeeze more out of limited hidden dimension, but the benefit is often small at n_hidden=128 where the bottleneck is width not depth. Most value if n_hidden=192 (H5) is already adopted.

**Risk:** Medium. Adding shortcuts to a Transolver requires care — the PhysicsAttention slice mechanism accumulates slice-level context across blocks, and a skip from block 0 (no slice context) to block 3 (rich slice context) can confuse the representation. Test n_layers=6 without shortcuts first to isolate the depth effect.

---

## Prioritization Summary

| Rank | ID | Hypothesis | Expected Impact | Risk | Priority |
|------|----|-----------|----------------|------|---------|
| 1 | H1 | Warm restarts (T_0=5, T_mult=2) | Med-High | Low | **HIGHEST** |
| 2 | H5 | n_hidden 128→192, n_layers 5→4 | Med | Low-Med | **HIGH** |
| 3 | H3 | log(Re) global conditioning token | Med | Med | **HIGH** |
| 4 | H4 | Surface-only loss final 1/3 training | Med | Med | **HIGH** |
| 5 | H2 | Per-channel output scale (p-amplifier) | Low-Med | Low | **MED-HIGH** |
| 6 | H6 | GradNorm learned loss weights | Low-Med | Med | **MEDIUM** |
| 7 | H8 | AoA augmentation ±1° | Low-Med | Low | **MEDIUM** |
| 8 | H7 | DSDF as attention bias | Low-Med | Med | **MEDIUM** |
| 9 | H9 | Dropout on slice weights p=0.1 | Low-Med | Low | **LOW-MED** |
| 10 | H10 | fp16 + GradScaler (vs bf16) | Speed only | Med | **LOW-MED** |
| 11 | H11 | 4× surface node upsampling | Low | Low | **LOW** |
| 12 | H12 | n_layers=6 with skip-3 shortcuts | Low-Med | Med | **LOW** |

## Key Insight from Round 1

The 100% clip rate at max_norm=1.0 is the most diagnostic result in the baseline. It means the model is operating in a normalized-gradient-descent regime (effective lr ≈ 1e-5). This has two implications for Round 3+ prioritization:

1. **The LR is the real bottleneck.** If Round 2's OneCycleLR (PR#3307) and max_norm=100 (PR#3306) both improve metrics, the mechanism is simply "more useful gradient signal per step." The correct follow-up is to understand the optimal effective step size, not just try more LR variants.

2. **Architecture changes only matter after optimization is fixed.** If the model is effectively underfitting due to suppressed gradient flow, adding capacity (H5, H12) will not help until the step size is corrected. The priority order above assumes Round 2 results will clarify the optimization picture first.
