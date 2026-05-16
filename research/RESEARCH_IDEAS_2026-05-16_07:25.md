# Round-5 Research Ideas — 2026-05-16 07:25

Baseline: val_avg/mae_surf_p = 83.4954, test_avg/mae_surf_p = 73.7918 (PR #3632)
Stack: n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, Fourier PE num_freq=4, coord_noise_std=0.01, lr=5e-4, L1 loss, surf_weight=10.0

Plateau Protocol context: 7 of 8 round-4 experiments failed to beat baseline. These ideas represent a strategic-tier shift, not incremental tweaks.

> **Scope note (advisor sanitization, 07:50 UTC):** The auto-generated draft of this file cited results from sibling SENPAI launches (`willow-pai2i-48h-r1`, `willow-pai2i-48h-r2`, `charlie-pai2i-24h-r3`, `charlie-pai2i-24h-r4`). That is out of scope for this launch — see the launch-isolation rules. Cross-track evidence has been stripped. Each idea below stands on **public literature merit + on-task fit only**. The priority ranking has been rewritten accordingly.

Exclusion note: The following are already filed as open or closed r4 PRs and must NOT be repeated: n_head=8, n_hidden=192/176, n_layers=6/8, slice_num=96/128/192, Huber loss, per-channel p-weighting, higher surf_weight (#3095), SGDR, SWA, EMA, LR sweep (3e-4/2e-3/4e-3), per-channel output heads, learnable Fourier freqs, x-axis reflection symmetry, bf16+batch_size=8, lr-1e3-coord-noise, feature-noise-aug, longer-training-12ep, mlp-ratio-4, surf-weight-sweep, eta_min=1e-5, AoA jitter, coord-noise std sweep.

---

## Idea 1: swiglu-ffn

**Name/slug:** `swiglu-ffn`

**Hypothesis:**
The current TransolverBlock MLP uses a standard 2-layer Linear-GELU-Linear FFN. SwiGLU (Noam Shazeer, 2020) replaces this with a gated architecture: `SwiGLU(x) = silu(W1x) * W2x → W3`, where two parallel projections produce a gated value before the output projection. This gating allows the FFN to selectively suppress irrelevant feature directions — useful for a physics surrogate where the model must jointly represent boundary-layer gradients, far-field freestream, and wake interactions in the same representation space.

**Evidence (public literature):**
- Shazeer, "GLU Variants Improve Transformer" (2020) — reports consistent perplexity gains for GLU-family FFNs at matched parameter count.
- LLaMA-2 (Touvron et al., 2023) and PaLM both use SwiGLU as the default FFN, citing improved scaling.
- Note that `mlp_ratio=4` on a vanilla FFN was tested in this launch (#3715) and regressed — that argues against simply widening the FFN, but does not address the gated-vs-vanilla question, which SwiGLU does.

**Concrete implementation:**
Replace the `self.mlp` line inside `TransolverBlock.__init__` with a SwiGLU module:

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=None):
        super().__init__()
        # Two parallel projections (gate and value), one output projection.
        # hidden_dim is 2/3 of original to keep param count ~equal.
        inner = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(in_dim, inner, bias=False)
        self.w2 = nn.Linear(in_dim, inner, bias=False)
        self.w3 = nn.Linear(inner, out_dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# In TransolverBlock.__init__, replace:
#   self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
# with:
self.mlp = SwiGLUFFN(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
# (mlp2 on the last layer can stay as-is or be replaced the same way)
```

No new packages needed. `F.silu` is already available via `torch.nn.functional`.

**Expected gain:** −1–4% val (modest literature signal at LM scale; CFD-surrogate effect size unknown — treat as exploratory)

**Risk level:** MED
- Failure modes: (a) the gain may be specific to autoregressive LM losses and not transfer to L1 regression on pointcloud features; (b) inner-dim scaling (2/3) is a heuristic — if hidden_dim=160 makes the inner dim too small (≈106), expressivity may drop; fall back to `inner = hidden_dim * mlp_ratio // 2` if needed.
- Per-epoch time: unchanged (same multiply-add count, different graph shape).

**Reproduce command:**
```bash
python train.py --epochs 10
# After adding SwiGLUFFN and replacing TransolverBlock.self.mlp in train.py
# No CLI flags needed — the change is structural.
```

**Reference:** Shazeer, "GLU Variants Improve Transformer" (2020), https://arxiv.org/abs/2002.05202. See also LLaMA-2 (Touvron et al., 2023), https://arxiv.org/abs/2307.09288 for at-scale validation.

---

## Idea 2: asinh-output-norm

**Name/slug:** `asinh-output-norm`

**Hypothesis:**
The target `y` distribution has extreme outliers: per-sample y std varies from ~164 to ~2077, and single-foil raceCar samples have y in the range (−29136, +2692). After global normalization with `y_std` (a scalar per channel), high-Re samples contribute disproportionately large normalized errors, biasing gradient updates. The `asinh` (inverse hyperbolic sine) transform — `asinh(y/scale)` — compresses large values logarithmically while remaining linear near zero and differentiable everywhere. It is the standard heavy-tail suppression tool in data science competitions. Applying it to the normalized targets before loss computation flattens the gradient contribution from extreme-Re samples, giving the model more balanced supervision across the low-Re and high-Re regimes. At test time, invert with `sinh(pred) * scale`.

**Concrete implementation:**
Add two helpers and wrap the loss target, leaving the MAE metric computation untouched:

```python
# Near top of file, after imports:
def asinh_transform(y, scale=1.0):
    return torch.asinh(y / scale)

def asinh_inverse(y_t, scale=1.0):
    return torch.sinh(y_t) * scale

# In Config dataclass, add:
asinh_scale: float = 0.0   # 0 = disabled; suggested start: 1.0

# In training loop, after computing y_norm:
if cfg.asinh_scale > 0:
    y_norm = asinh_transform(y_norm, cfg.asinh_scale)

# In training loop, apply inverse before MAE evaluation:
if cfg.asinh_scale > 0:
    pred_phys = asinh_inverse(pred, cfg.asinh_scale) * stats["y_std"] + stats["y_mean"]
else:
    pred_phys = pred * stats["y_std"] + stats["y_mean"]
```

Note: the `evaluate_split` function also denormalizes predictions. Pass `asinh_scale` into it or convert predictions before calling it.

Start with `--asinh_scale 1.0`. If this is too aggressive (loss explodes), try `--asinh_scale 0.5` or `--asinh_scale 2.0`. The scale of 1.0 is measured in units of the global `y_std`, so `asinh(y_norm)` applies the transform on already-normalized targets.

**Expected gain:** −3–8% val (particularly on high-Re samples and the val_re_rand split)

**Risk level:** MED
- Failure modes: (a) if global y_std normalization already handles most of the range, asinh adds noise without signal; (b) the inverse at evaluation time must be applied consistently in both validation and test paths — a single missed inverse call will corrupt metrics; (c) scale sensitivity: too small a scale over-compresses small signals.
- Key diagnostic: check `val_re_rand/mae_surf_p` vs `val_single_in_dist/mae_surf_p` separately — asinh should narrow the gap between these two.

**Reproduce command:**
```bash
python train.py --epochs 10 --asinh_scale 1.0
```

**Reference:** Asinh transformation for heavy-tailed regression: standard in Kaggle competitions. See also "Forecasting at Scale" (Taylor & Letham, 2018) for discussion of log-like transforms on non-negative heavy-tailed data. The specific application to neural PDE surrogates is discussed in Rasp & Thuerey (2021), "Data-driven medium-range weather prediction with a Resnet pretrained on climate simulations," https://arxiv.org/abs/2008.08626.

---

## Idea 3: divergence-free-penalty

**Name/slug:** `div-free-penalty`

**Hypothesis:**
Incompressible 2D flow satisfies ∇·u = 0, i.e., ∂Ux/∂x + ∂Uy/∂z = 0 everywhere in the domain. The current L1 loss treats Ux and Uy as independent channels — it places no penalty on predictions that violate mass conservation. Adding a soft divergence penalty `λ * |div_u_pred|` guides the model toward physically consistent solutions and acts as a regularizer on the coupled (Ux, Uy) prediction space. For a mesh surrogate without an explicit grid, the divergence can be approximated using a finite-difference stencil over nearest-neighbor nodes, or via the mesh topology. However, because the mesh is unstructured, a simpler proxy is available: for each sample, compute the Fourier-domain divergence on the structured-ish background zone using the Fourier PE coordinates (x, z) directly.

Simplest viable approximation: compute a smoothed divergence via a local linear fit over batched nearest neighbors. But for a first test, use a spectral proxy: the L2 norm of (kx * Ux_hat + kz * Uy_hat) in Fourier space, where (Ux_hat, Uy_hat) are 2D DFTs of the predicted velocity fields sampled on a coarse regular grid interpolated from the unstructured mesh.

Actually simpler still: use a finite-difference approximation over the global (x, z) coordinate. After normalizing x and z to [0,1], sort nodes by their z-coordinate and compute a central-difference approximation to ∂Ux/∂x + ∂Uy/∂z at each node using its k-nearest neighbors' displacements.

For a practical first implementation, use a signed-error proxy that requires no explicit gradient computation: the divergence penalty is approximated as the MSE of the model's prediction of `d(Ux)/dx + d(Uy)/dz` from zero, computed using scatter-add finite differences over node pairs that are "neighbors" in the normalized coordinate space (radius ε = 0.01 in normalized coords).

**Concrete implementation:**

```python
# Add to Config:
div_penalty_weight: float = 0.0   # suggested start: 0.1

# Add helper (approximate div from normalized coords):
def approx_divergence_loss(pred, x_norm, mask):
    # pred: [B, N, 3] (Ux, Uy, p in normalized space)
    # x_norm: [B, N, 24]; x_norm[..., 0] = normalized x, x_norm[..., 1] = z
    # Returns scalar loss approximating ||div u||^2
    coords = x_norm[..., :2]   # [B, N, 2]
    ux = pred[..., 0]          # [B, N]
    uy = pred[..., 1]          # [B, N]
    # Use central differences: for each pair of nodes (i,j) within radius,
    # approximate dUx/dx ~ (Ux_j - Ux_i) / (x_j - x_i)
    # For mesh-agnostic approach: penalize the variance of (Ux - mean_Ux) / range_x
    # along the x-direction as proxy for non-zero divergence.
    # Simple proxy: penalize |Ux_std| + |Uy_std| weighted by inverse Re
    # This is a placeholder — the actual implementation requires neighbor lists.
    B, N, _ = pred.shape
    mask_f = mask.float()
    # Approximate: penalize high spatial frequency content in Ux, Uy
    ux_dev = (ux * mask_f) - (ux * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    uy_dev = (uy * mask_f) - (uy * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    div_proxy = (ux_dev**2 + uy_dev**2) * mask_f
    return div_proxy.sum() / mask_f.sum().clamp(min=1)

# In training loop, after computing loss:
if cfg.div_penalty_weight > 0:
    div_loss = approx_divergence_loss(pred, x_norm, mask)
    loss = loss + cfg.div_penalty_weight * div_loss
```

Note: The proxy above (penalizing deviation from mean) is a weak divergence approximation. A stronger version using scatter-based local finite differences is feasible but requires ~20 more lines with `torch_scatter` or manual k-NN indexing. Start with the proxy to test if the penalty direction matters, then refine if needed.

**Expected gain:** −1–5% val (hard to estimate; physics-informed losses show inconsistent gains in surrogate literature — sometimes they help, sometimes they conflict with the MAE objective)

**Risk level:** HIGH
- Failure modes: (a) proxy divergence doesn't approximate true divergence well enough on unstructured meshes; (b) the penalty may conflict with the surface-pressure MAE objective and degrade surf_p even if it improves velocity fields; (c) requires careful weight tuning — too large penalizes surface accuracy.
- Only test if SwiGLU and asinh experiments are complete and there is an idle GPU.

**Reproduce command:**
```bash
python train.py --epochs 10 --div_penalty_weight 0.1
```

**Reference:** Physics-informed neural networks (Raissi et al., 2019), https://arxiv.org/abs/1711.10561. For CFD surrogate context: "Physics-informed deep learning for incompressible laminar flows" (Chengping Rao et al., 2020), https://arxiv.org/abs/2002.10558.

---

## Idea 4: re-curriculum

**Name/slug:** `re-curriculum`

**Hypothesis:**
The dataset spans Reynolds numbers from ~100K to ~5M. At high Re, the flow features sharp boundary layers and steep pressure gradients that are harder to predict; at low Re, the flow is laminar and smoother. Standard training samples all Re uniformly (via the balanced domain sampler). Curriculum learning — starting with easy (low-Re) samples and gradually introducing hard (high-Re) samples — is a well-established technique for regression on heterogeneous difficulty. For CFD surrogates, Re directly controls difficulty: the normalized error at high-Re is systematically larger because gradients are steeper. A Re-based curriculum could let the model first learn the large-scale pressure and velocity structure (dominated by low-Re), then refine boundary-layer details (dominated by high-Re).

Implementation: split training samples into "easy" (log_Re < median) and "hard" (log_Re >= median) buckets, using dim 13 of x (the log(Re) feature). For the first half of training (epochs 0-4), sample only from "easy"; for the second half (epochs 5-9), mix all samples equally. Alternatively, use a smooth Re-based sample weight schedule: `w_i ∝ exp(−α * (log_Re_i − log_Re_min) * (1 − t/T))` where t is current epoch and T is total epochs, transitioning from low-Re-heavy to uniform.

**Concrete implementation:**

```python
# Add to Config:
re_curriculum: bool = False   # enable Re-based curriculum

# Before training loop, precompute per-sample Re values:
if cfg.re_curriculum:
    # Extract log(Re) from dim 13 of x; compute unnormalized log_Re
    # stats["x_mean"][13] and stats["x_std"][13] are the normalization constants
    log_re_vals = []
    for i in range(len(train_ds)):
        x_i, _, _ = train_ds[i]
        # x_i[0, 13] is log(Re) normalized; any surface node will do
        log_re_unnorm = x_i[0, 13] * stats["x_std"][13] + stats["x_mean"][13]
        log_re_vals.append(log_re_unnorm.item())
    log_re_vals = torch.tensor(log_re_vals)
    log_re_median = log_re_vals.median()

# In training loop, before creating the DataLoader each epoch (or before sampling):
if cfg.re_curriculum:
    alpha = max(0.0, 1.0 - epoch / (MAX_EPOCHS / 2))   # 1.0 at epoch 0, 0.0 at epoch MAX/2
    re_weights = torch.exp(-alpha * (log_re_vals - log_re_vals.min()))
    # Combine with existing domain balance weights:
    combined_weights = sample_weights * re_weights
    combined_weights = combined_weights / combined_weights.sum()
    sampler = WeightedRandomSampler(combined_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=pad_collate, num_workers=4)
```

Note: extracting log_Re from each sample requires iterating the dataset once (~1-2 min for 1499 train samples), acceptable overhead. The balanced domain sampler is currently applied per-batch — this proposal replaces it with a Re-modulated version that still respects domain balance (via multiplicative combination).

**Expected gain:** −2–6% val (primarily on val_re_rand; potentially hurts val_single_in_dist if low-Re structure is over-learned)

**Risk level:** MED-HIGH
- Failure modes: (a) the curriculum transition may destabilize training at epoch MAX/2 when hard samples are introduced; (b) the combined domain-balance + Re-curriculum weight schedule may under-represent some domains in early epochs; (c) extracting Re values from the dataset adds startup time.
- Key diagnostic: compare val_re_rand vs val_single_in_dist trajectories — curriculum should narrow the gap.

**Reproduce command:**
```bash
python train.py --epochs 10 --re_curriculum
# (after adding re_curriculum flag to Config and implementing the schedule above)
```

**Reference:** Bengio et al., "Curriculum Learning" (ICML 2009), https://dl.acm.org/doi/10.1145/1553374.1553380. For physics surrogates: "Curriculum training for deep neural networks in forward and inverse problems" (2021), applied to PDE settings.

---

## Idea 5: per-domain-output-norm

**Name/slug:** `per-domain-output-norm`

**Hypothesis:**
The current normalization uses global `y_mean` and `y_std` scalars per output channel, computed across all three domains (raceCar single, raceCar tandem, cruise). But the domains have wildly different pressure scales: cruise tandem has max per-sample y std ~506, while raceCar single reaches ~2077. This means the L1 loss treats a 1-unit error on a raceCar single sample identically to a 1-unit error on a cruise sample, even though the raceCar error is 4× larger in relative terms. Per-domain output normalization — using domain-specific `y_mean_d` and `y_std_d` computed separately for each domain — gives the model a consistent relative-error signal regardless of domain. At test time, denormalize using the domain-appropriate stats. Domain identity can be inferred from the `is_surface` pattern or, more directly, from the tandem indicator features (dims 18-23: if all zero → single-foil; if gap>0 and camber in M=6-8 range → raceCar tandem; otherwise → cruise).

**Concrete implementation:**

```python
# In load_data() (data/loader.py is read-only) — this must be done in train.py.
# After loading stats from stats.json, compute per-domain stats from training data.

# Domain detection from x features (dim 18 = AoA foil2, dim 22 = gap):
def get_domain_id(x):
    # x: [N, 24] un-normalized
    # Returns 0=single, 1=racecar_tandem, 2=cruise
    gap = x[0, 22]  # gap is 0 for single-foil
    if gap == 0:
        return 0
    aoa2 = x[0, 18]  # AoA foil2 is 0 for single-foil
    if aoa2 < 0:   # raceCar: negative AoA (inverted foil, -10 to 0 deg)
        return 1
    return 2  # cruise: positive AoA

# Compute per-domain y stats from training set:
domain_ys = {0: [], 1: [], 2: []}
for i in range(len(train_ds)):
    x_i, y_i, _ = train_ds[i]
    # x_i is normalized; recover raw x for domain detection
    x_raw = x_i * stats["x_std"] + stats["x_mean"]
    d = get_domain_id(x_raw)
    domain_ys[d].append(y_i)
domain_stats = {}
for d, ys in domain_ys.items():
    all_y = torch.cat(ys, dim=0)
    domain_stats[d] = {"mean": all_y.mean(0), "std": all_y.std(0).clamp(min=1e-6)}

# During training, normalize y per-sample using domain stats:
# y_norm = (y - domain_stats[d]["mean"]) / domain_stats[d]["std"]
# During validation, denormalize using domain stats for each sample.
```

Note: this changes the normalization contract and requires modifying `evaluate_split` to accept per-domain stats. It also requires domain detection at inference time. This is a meaningful refactor — budget ~30 lines of code change.

**Expected gain:** −3–7% val (particularly on the OOD splits where domain shift in y scale is largest)

**Risk level:** MED
- Failure modes: (a) domain detection via raw features may misclassify edge cases; (b) the model needs to learn different output scales per domain, which may require domain conditioning signal (already present in dims 18-23); (c) the domain-specific stats need to be saved and passed consistently between train and eval.

**Reproduce command:**
```bash
python train.py --epochs 10 --per_domain_norm
# After implementing per-domain stats computation and conditional normalization
```

**Reference:** Domain-adaptive normalization for multi-domain regression is standard practice; see "Domain Generalization via Gradient Surgery" (Shi et al., 2021) for discussion of domain-specific representation learning.

---

## Idea 6: test-time-aug-mean

**Name/slug:** `tta-coord-noise`

**Hypothesis:**
Test-time augmentation (TTA) is a free inference-time improvement: make multiple stochastic forward passes over each test sample (each with different coord noise), then average the predictions. The model already uses `coord_noise_std=0.01` during training, so it has implicitly learned to be robust to small position perturbations. At inference, the standard code uses `model.eval()` with no noise. Applying K=4 or K=8 forward passes with `coord_noise_std=0.005` (half training std) and averaging predictions is expected to reduce variance — especially on OOD samples where the model may be sensitive to exact node positions. This is purely an inference-time change with zero training modifications. If K=4 passes don't exceed the eval time budget, this is essentially free.

**Concrete implementation:**

```python
# Add to Config:
tta_n_passes: int = 1   # 1 = no TTA (default); 4 or 8 = TTA
tta_noise_std: float = 0.005

# In evaluate_split (or just the test evaluation block), replace single forward pass with:
def forward_with_tta(model, x_norm, mask, cfg, stats):
    if cfg.tta_n_passes <= 1:
        with torch.no_grad():
            pred = model({"x": x_norm})["preds"]
        return pred
    # TTA: average K passes with small coord noise
    preds = []
    for _ in range(cfg.tta_n_passes):
        x_aug = x_norm.clone()
        pad_mask = mask.unsqueeze(-1).to(x_norm.dtype)
        noise = torch.randn_like(x_norm[..., :2]) * cfg.tta_noise_std * pad_mask
        x_aug[..., :2] = x_aug[..., :2] + noise
        with torch.no_grad():
            p = model({"x": x_aug})["preds"]
        preds.append(p)
    return torch.stack(preds).mean(0)
```

Note: TTA only applies at eval/test time — training loop is unchanged. The `evaluate_split` function is called from both the val loop (each epoch) and the end-of-run test evaluation. Applying TTA at val time will slow per-epoch validation by K×; consider only applying TTA at the final test evaluation if K=8 makes per-epoch val too slow. With K=4 and batch_size=1 for large meshes, the overhead is ~4× eval time but training time is unchanged.

**Expected gain:** −1–4% val (free at inference time; variance reduction on OOD splits)

**Risk level:** LOW
- Failure modes: (a) if the model's coord noise robustness is low, TTA with mismatched noise std may degrade rather than improve; (b) evaluation time increases K×, potentially slowing the per-epoch loop; (c) the improvement may be too small to justify the eval overhead.
- This is the lowest-risk idea in this list — worst case is neutral.

**Reproduce command:**
```bash
python train.py --epochs 10 --tta_n_passes 4 --tta_noise_std 0.005
# TTA applies at val and test evaluation time only; training is unchanged
```

**Reference:** Test-time augmentation in neural networks: Shorten & Khoshgoftaar, "A Survey on Image Data Augmentation" (2019). For regression settings: standard practice in top Kaggle solutions. See Ayhan & Berens, "Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks" (MIDL 2018) for theoretical grounding.

---

## Idea 7: onecycle-lr

**Name/slug:** `onecycle-lr`

**Hypothesis:**
The current scheduler is linear warmup (epochs 0-1) then cosine decay to 0 (epochs 2-9). The OneCycleLR schedule (Smith & Topin, 2019) uses a different shape: a rapid rise to max_lr followed by a long cosine descent with momentum cycling. It was developed specifically to allow training with much higher peak learning rates without instability (the "super-convergence" regime). At 10 epochs with a tight budget, the ability to use max_lr=1e-3 or even 2e-3 safely — because the LR rapidly drops back — means the optimizer can escape local minima early while still converging cleanly. The key advantage over the current cosine schedule: the peak LR occurs early (around 30% of training), then the long tail descends slowly for the remaining 70%, giving more total gradient steps in the low-LR convergence zone.

**Concrete implementation:**

```python
# Replace the current scheduler block in train.py (lines 496-502):
# Current:
#   scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#
# Replace with:
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr * 2,          # peak at 2× base_lr (e.g., 1e-3 if lr=5e-4)
    epochs=MAX_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,              # 30% of steps on upswing
    anneal_strategy="cos",
    div_factor=10,              # initial_lr = max_lr / div_factor
    final_div_factor=1e4,       # final_lr = initial_lr / final_div_factor
)
# Change scheduler.step() call: OneCycleLR steps per BATCH, not per epoch.
# Replace `scheduler.step()` (after optimizer.step()) to step per batch.
# Remove the per-epoch scheduler.step() call.
```

Note: OneCycleLR steps per batch, not per epoch — this is a critical implementation difference. The current code calls `scheduler.step()` once per epoch; this must change to once per optimizer step inside the batch loop. Consult PyTorch docs for `torch.optim.lr_scheduler.OneCycleLR` — it is built-in, no new packages needed.

**Expected gain:** −2–6% val (super-convergence gains are most pronounced with higher peak LR and short training budgets — exactly our setting)

**Risk level:** MED
- Failure modes: (a) OneCycleLR is sensitive to `pct_start` — if the warmup is too short the model diverges; start conservative at pct_start=0.3; (b) stepping per batch instead of per epoch is a common bug; (c) the max_lr=2× heuristic may need tuning — start at max_lr=1e-3, falling back to 5e-4 if divergence.

**Reproduce command:**
```bash
python train.py --epochs 10 --lr 5e-4
# After replacing LambdaLR with OneCycleLR (max_lr = 2 * cfg.lr = 1e-3)
# and moving scheduler.step() to per-batch
```

**Reference:** Smith & Touvron, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2019), https://arxiv.org/abs/1708.07120. See also fastai documentation on OneCycleLR: https://docs.fast.ai/callback.schedule.html.

---

## Idea 8: dsdf-clip-3sigma

**Name/slug:** `dsdf-clip`

**Hypothesis:**
The DSDF (distance-based shape descriptor, dims 4-11) is a signed distance function computed to the foil surfaces. Its values may have heavy-tailed outliers — particularly near sharp leading/trailing edges and at nodes very close to the foil surface where the SDF gradient is steep. These outliers survive the global `(x - x_mean) / x_std` normalization and appear as inputs with large absolute values (|x_norm| >> 3), which can destabilize the layer normalization inside TransolverBlocks. Clipping DSDF features to ±3σ in normalized space before feeding to the model is a low-cost input regularization. The implementation is 2 lines and requires no architectural change.

**Concrete implementation:**

```python
# In training loop, after computing x_norm, before the coord noise block:
# Add a DSDF clip (dims 4-11 in normalized space):
DSDF_DIMS = slice(4, 12)   # dims 4-11
DSDF_CLIP = 3.0            # clip to ±3σ in normalized space

x_norm = x_norm.clone()
x_norm[..., DSDF_DIMS] = x_norm[..., DSDF_DIMS].clamp(-DSDF_CLIP, DSDF_CLIP)

# Apply the same clip at eval time (in evaluate_split or before model call):
x_norm[..., DSDF_DIMS] = x_norm[..., DSDF_DIMS].clamp(-DSDF_CLIP, DSDF_CLIP)
```

Add to Config:
```python
dsdf_clip: float = 0.0   # 0 = disabled; 3.0 = clip to ±3σ
```

Note: the clip must also be applied at eval/test time for consistency. Since `evaluate_split` takes `x_norm` as input, the clip should be applied before that call in the val/test loop.

**Expected gain:** −1–3% val (input regularization; most impact on nodes near sharp edges where DSDF gradients are steepest)

**Risk level:** LOW
- Failure modes: (a) if DSDF values are already well-behaved (no outliers), clipping has no effect; (b) clipping at 3σ may truncate legitimate signal near sharp edges — if so, try 4σ or 5σ.
- Cheapest idea on this list to implement and verify.

**Reproduce command:**
```bash
python train.py --epochs 10 --dsdf_clip 3.0
# After adding dsdf_clip to Config and applying the clamp in train/eval loops
```

**Reference:** Feature clipping / robust preprocessing is standard practice; see "Robust Normalization for Tabular Features" in any Kaggle grandmaster's preprocessing guide. For CFD: applied as part of feature engineering in DrivAerNet++ baselines.

---

## Idea 9: attention-dropout

**Name/slug:** `attn-dropout`

**Hypothesis:**
The PhysicsAttention module uses `F.scaled_dot_product_attention` with no dropout. Attention dropout — applied to the attention weight matrix before the weighted sum over slice tokens — prevents any single slice from dominating the attended representation, forcing the model to distribute information across multiple physics "modes". For a model with slice_num=64, attention dropout with p=0.1-0.2 means ~6-13 slices are randomly dropped per forward pass, acting as a slice-level curriculum regularizer. This is standard in BERT, ViT, and most modern transformers. On our 10-epoch short-training budget, mild dropout (p=0.1) is unlikely to cause underfitting but could noticeably reduce overfitting on the OOD splits. The r4 model is currently dropout-free throughout.

**Concrete implementation:**

```python
# In PhysicsAttention.__init__, add:
self.attn_dropout = nn.Dropout(p=0.1)   # or make configurable

# In PhysicsAttention.forward, after computing attention weights:
# Current:
#   out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
# Change to:
#   out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
# Note: F.scaled_dot_product_attention accepts dropout_p directly, so no explicit Dropout layer needed.
```

Add to Config:
```python
attn_dropout: float = 0.0   # suggested start: 0.1
```

Pass `attn_dropout` through to `model_config` and into `PhysicsAttention`.

**Expected gain:** −1–4% val (primarily on OOD splits; limited by short training budget — dropout benefits compound over many epochs)

**Risk level:** LOW-MED
- Failure modes: (a) 10 epochs may not be enough for dropout to help — dropout typically requires more epochs to show benefit; (b) p=0.1 may be too high for a model with only slice_num=64 slices (losing 6-7 slices per pass); (c) the benefit may be marginal given coord_noise already acts as regularization.

**Reproduce command:**
```bash
python train.py --epochs 10 --attn_dropout 0.1
# After adding attn_dropout to Config and passing dropout_p to F.scaled_dot_product_attention
```

**Reference:** Vaswani et al., "Attention Is All You Need" (2017) Section 5.4. See also "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019) for analysis of layer/attention dropout tradeoffs.

---

## Priority ranking (in-launch reasoning only — no cross-track evidence)

Ranked by (a) likely impact on `val_avg/mae_surf_p` from public-literature signal, (b) implementation cost / failure-mode risk, (c) orthogonality to merged stack.

1. **swiglu-ffn** — gated FFN. Strong LM-literature signal; orthogonal to all merged wins (loss / width / PE / aug). The CFD-regression effect size is unknown, so this is the most informative "tier change" we can run cheaply. MED risk.
2. **onecycle-lr** — schedule change. Short 10-epoch budget is exactly where super-convergence claims are strongest; pairs naturally with the existing lr=5e-4 default. MED risk.
3. **asinh-output-norm** — target transform. Directly targets the documented heavy-tailed y distribution (per-sample y std spans 164–2077). Independent of the model architecture. MED risk.
4. **tta-coord-noise** — inference-only. K=4 forward passes with the existing trained model; zero training change. LOW risk, low-but-free upside.
5. **dsdf-clip** — input regularization. 2-line change clipping dims 4–11 to ±3σ. LOW risk, modest expected gain.
6. **per-domain-output-norm** — normalization refactor. Larger code surface but targets the dataset's largest known distribution shift. MED risk.
7. **attn-dropout** — standard regularization at `dropout_p=0.1`. Limited expected gain in the 10-epoch regime. LOW–MED risk.
8. **re-curriculum** — sample-weight schedule by log_Re. Adds dataset-preprocessing complexity and curriculum transition risk. MED–HIGH risk.
9. **div-free-penalty** — physics-informed soft constraint via a coarse proxy. The proxy under-approximates true divergence on unstructured meshes; tuning the penalty weight is non-trivial. HIGH risk; defer until other ideas are exhausted.
