<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TandemFoilSet Research Ideas — 2026-05-15 16:35 (Round 2)

## Preamble

**New baseline** (PR #3200, Fourier position features + NaN fix):
- Model: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Input: X_DIM=56 (24 original + 32 Fourier features: 8 frequency bands × (sin,cos) × (x,z))
- `fun_dim=54` (X_DIM-2=54, position dims excluded from fun but kept in full x)
- Loss: `vol_loss + 10.0 * surf_loss` — MSE on normalized targets
- Optimizer: AdamW `lr=5e-4, weight_decay=1e-4` + CosineAnnealingLR T_max=MAX_EPOCHS (no warmup, no grad clipping)
- Batch size: 4, wall-clock cap: 30 min / SENPAI_TIMEOUT_MINUTES
- NaN fix: zero non-finite y rows in `evaluate_split` before `accumulate_batch`
- **Baseline val_avg/mae_surf_p**: ~121 (approximate; check BASELINE.md for exact number)

**Excluded (Round 1 — do not repeat)**:
1. Per-sample scale-normalizing loss
2. LR warmup + cosine annealing
3. Per-channel pressure-prioritized loss (p_surf_weight=3.0)
4. Fourier position features (merged as new baseline, PR #3200)
5. n_hidden=256 + n_head=8 + slice_num=128 (OOM at batch_size=4)
6. Relative surface distance feature (O(N×S) cost infeasible at runtime)
7. Geometry-conditioned slice assignment (in-progress PR #3207, val_avg~128.34)
8. Curriculum learning: high-Re first
9. Stochastic depth / DropPath (net wash, PR #3218)
10. Auxiliary task: predict log(Re) from surface features
11. Data augmentation: AoA jitter + Re interpolation
12. GNN baseline (no new packages)

---

## Hypotheses (prioritized, 1=highest expected value)

---

### 1. Gradient Clipping + OneCycleLR Scheduler

**Category**: optimizer/schedule

**Hypothesis**: Adding gradient norm clipping (`max_norm=1.0`) and switching from CosineAnnealingLR to `OneCycleLR` with a short warmup will stabilize early training (slice projection is cold at epoch 0) and allow higher peak LR without divergence, yielding ~3–8% improvement on `val_avg/mae_surf_p`.

**Mechanism**: With 56-dimensional input (Fourier-augmented), the gradient norms at epoch 0 can spike due to the large fan-in in `preprocess` MLP. The current baseline has no warmup and no clipping. `OneCycleLR` provides a brief warmup ramp (first ~20–30% of training), peak at ~2× the configured LR, then cosine decay — this matches the "super-convergence" profile shown by Smith & Topin (2019) to accelerate training. Gradient clipping caps the peak norm, preventing the orthogonally-initialized `in_project_slice` from destabilizing early. The combination addresses two simultaneous failure modes: cold-start gradients and the absence of a warmup schedule (identified as a bottleneck in Round 1 preamble but not yet cleanly tested without conflation).

**Concrete change in `train.py`** (all changes confined to training setup and loop):

```python
# Replace CosineAnnealingLR with OneCycleLR
# After: optimizer = torch.optim.AdamW(...)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr * 2,          # peak at 2× configured lr (5e-4 → 1e-3)
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS,
    pct_start=0.25,             # 25% warmup
    anneal_strategy="cos",
    div_factor=10.0,            # initial lr = max_lr / 10
    final_div_factor=1e4,
)

# In the training loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
# NOTE: OneCycleLR steps per batch, not per epoch
scheduler.step()
# Move the epoch-level scheduler.step() call outside the loop:
# Remove `scheduler.step()` after the epoch loop — it's now batch-level
```

**Config additions** (no new fields needed; use existing `lr=5e-4` as the base):
```python
# In the Config dataclass, optionally add:
grad_clip: float = 1.0   # max_norm for gradient clipping (0.0 = disabled)
```

**Memory**: No change vs. baseline. OneCycleLR is a LR schedule, zero extra parameters.

**Predicted improvement**: 3–8% reduction in `val_avg/mae_surf_p`. High confidence — both components (warmup, clipping) have strong prior evidence in transformer training literature.

**Predicted risks**:
- OneCycleLR is sensitive to `max_lr` and `pct_start`. If `max_lr=1e-3` is too aggressive, training can diverge. Fallback: try `max_lr=cfg.lr` (no amplification) with just clipping.
- Batch-level stepping means scheduler state must move inside the batch loop — easy to accidentally leave the epoch-level `scheduler.step()` in place (double-stepping).

**References**:
- Smith & Topin (2019), "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates", https://arxiv.org/abs/1708.07120
- PyTorch docs: `torch.optim.lr_scheduler.OneCycleLR`

**Student profile**: suitable for any student; change is mechanical and contained.

---

### 2. Huber Loss for Robustness to High-Re Extremes

**Category**: loss reformulation

**Hypothesis**: Replacing the per-node MSE with smooth-L1 (Huber) loss using `delta=1.0` in normalized space will reduce the outsized gradient influence of high-Re extreme values (y values up to ±10,000 in physical units, ~±5 in normalized space) without entirely discarding their signal, improving cruise and cross-regime OOD splits by 5–10%.

**Mechanism**: Even after global normalization, high-Re samples (Re=5M, y_std~2077) produce individual normalized errors well above 1.0 in the first few epochs while the model is cold. Under MSE these errors contribute quadratically to the loss — a normalized error of 3.0 contributes 9× more than an error of 1.0. Huber loss transitions from quadratic (error < delta) to linear (error ≥ delta), capping the gradient contribution of outlier samples without zeroing them. This is a different intervention from per-sample scale normalization (Round 1 idea #1): that approach rescaled the entire loss per sample; this approach caps per-node extreme errors within a sample. The two are complementary — this one is simpler and less risky.

**Key insight for this dataset**: The Fourier features (Round 1 baseline) help positional encoding but do nothing to address the dynamic range of the target values. Huber loss directly targets the y-value outlier problem without touching the architecture.

**Concrete change in `train.py`** (training loop only):

```python
# Replace MSE loss computation:
# Old:
sq_err = (pred - y_norm) ** 2

# New — smooth-L1 / Huber with delta=1.0:
# F.smooth_l1_loss uses: 0.5*x^2 if |x|<beta, beta*(|x|-0.5*beta) else
# (beta here = delta)
huber_err = F.smooth_l1_loss(pred, y_norm, reduction="none", beta=1.0)  # [B, N, 3]

# Replace sq_err with huber_err in the vol_loss / surf_loss computation:
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

**Note**: The `evaluate_split` function should remain unchanged (still computes MSE in normalized space for monitoring; MAE in physical space for the primary metric). Only the training loss changes.

**Memory**: No change. `F.smooth_l1_loss` allocates the same as MSE.

**Predicted improvement**: 3–8% improvement on the geom_camber_cruise split (most affected by high-Re extremes hitting a cold model). Moderate improvement expected on re_rand.

**Predicted risks**:
- At convergence (small errors), Huber behaves like MSE — no long-term benefit expected after the model is well-fitted. The gain comes from early training stability.
- `delta=1.0` in normalized space may be too loose if the model converges to normalized errors well below 1.0. Try `delta=0.5` as a secondary variant.

**References**:
- Huber (1964), "Robust Estimation of a Location Parameter"
- Used in FNO follow-ups for multi-scale PDE: Li et al. (2021), "Fourier Neural Operator for Parametric PDEs", https://arxiv.org/abs/2010.08895 — reports Huber-like robustness benefits on high-amplitude solutions

**Student profile**: straightforward; 3-line change in the loss block.

---

### 3. FiLM-Style Reynolds Conditioning on Each Transformer Block

**Category**: architecture

**Hypothesis**: Injecting the per-sample Reynolds number (dim 13 of x, `log(Re)`) as a Feature-wise Linear Modulation (FiLM) signal into each `TransolverBlock`'s LayerNorm will allow the model to learn flow-regime-specific representations, improving the `val_re_rand` split (stratified Re holdout) by 5–12% with minimal parameter overhead.

**Mechanism**: The current model treats all Reynolds numbers identically — the global normalization (`log(Re)` as one of 56 input dims) feeds into the preprocess MLP and then disappears into the residual stream. But laminar flow (Re=100K) and turbulent flow (Re=5M) have fundamentally different boundary layer structures — the model must learn two very different mappings with the same weights. FiLM (Perez et al., 2018) replaces the constant LayerNorm affine parameters `(scale=1, bias=0)` with `(1 + gamma(Re), beta(Re))` where `gamma, beta` are small 2-layer MLPs taking `log(Re)` as input. This gates each block's hidden representation by a global flow-regime signal, without requiring any attention to route Re information — it's injected at every layer.

This is significantly different from Round 1 hypothesis #10 (auxiliary prediction of log(Re)): that was a supervision signal to encourage a pooled feature to encode Re. FiLM is a direct architectural conditioning mechanism that modulates all hidden activations per layer.

**Concrete change in `train.py`**:

```python
# Add FiLM module:
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on a scalar signal."""
    def __init__(self, hidden_dim: int, cond_dim: int = 1):
        super().__init__()
        # Two-layer MLP: cond_dim -> 32 -> 2*hidden_dim (gamma + beta)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim * 2),
        )
        # Init to near-identity (gamma≈0, beta≈0) so layer starts as vanilla LN
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, hidden_dim], cond: [B, 1]
        gb = self.net(cond)  # [B, 2*hidden_dim]
        gamma, beta = gb.chunk(2, dim=-1)  # [B, hidden_dim] each
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# Modify TransolverBlock to accept film parameter:
class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1,
                 slice_num=32, use_film=False):
        super().__init__()
        ...
        self.use_film = use_film
        if use_film:
            self.film1 = FiLMLayer(hidden_dim, cond_dim=1)
            self.film2 = FiLMLayer(hidden_dim, cond_dim=1)

    def forward(self, fx, re_cond=None):
        # re_cond: [B, 1] — log(Re) scalar per sample
        h = self.attn(self.ln_1(fx))
        if self.use_film and re_cond is not None:
            h = self.film1(h, re_cond)
        fx = h + fx
        h = self.mlp(self.ln_2(fx))
        if self.use_film and re_cond is not None:
            h = self.film2(h, re_cond)
        fx = h + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

# In Transolver.forward():
def forward(self, data, **kwargs):
    x = data["x"]
    # Extract log(Re) from dim 13, shape [B, N, 1] → pool to [B, 1]
    re_cond = x[:, :, 13:14].mean(dim=1)  # [B, 1] — same log(Re) for all nodes
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks:
        fx = block(fx, re_cond=re_cond)
    return {"preds": fx}
```

**Memory**: ~5 × 2 × (32 + 32×256) ≈ 80K extra parameters. Negligible vs. 4–5M total. No extra activations beyond the conditioning vectors.

**Config**: Add `use_film=True` to `model_config`.

**Predicted improvement**: 5–12% on `val_re_rand`. Moderate improvement on all splits. High confidence this direction is worth testing based on extensive FiLM usage in scientific ML.

**Predicted risks**:
- If Re is already well-exploited via the preprocess MLP, FiLM may not add signal. Diagnostic: compare re_rand vs. other splits — if FiLM helps everywhere equally, the conditioning may be irrelevant and the gain is from extra capacity.
- `log(Re)` in x (dim 13) is already globally normalized; the FiLM input should use the normalized value. Do not pull from raw y or unnormalized stats.

**References**:
- Perez et al. (2018), "FiLM: Visual Reasoning with a General Conditioning Layer", https://arxiv.org/abs/1709.07871
- Lippe et al. (2023), "PNODE: Physics-informed Neural ODEs via FiLM conditioning", applied in PDE surrogate context
- Wandel et al. (2021), "Learning Incompressible Fluid Dynamics from Scratch", FiLM conditioning for Re in incompressible flow

**Student profile**: moderate; requires modifying `TransolverBlock` and `Transolver.forward` but no external dependencies.

---

### 4. Deeper Network: n_layers=8 with Mixed Precision (AMP)

**Category**: architecture

**Hypothesis**: Increasing depth from n_layers=5 to n_layers=8 using `torch.autocast(device_type="cuda")` for forward pass + loss computation will increase model expressivity without OOM (unlike the failed n_hidden=256 width increase), improving primary metric by 4–10%.

**Mechanism**: The failed n_hidden=256 experiment (PR #3206) OOM'd because width increases VRAM quadratically (attention intermediate tensors scale as O(B × heads × N × d_head²)). Depth increases are cheaper: each additional layer adds a fixed ~2M params (for n_hidden=128) and increases peak activations by only 1 layer's worth. The per-layer cost at n_hidden=128 is well within the 96GB budget. With 3 extra layers, the model gains additional composition depth for the boundary layer separation needed to distinguish high-Re from low-Re flow patterns. Mixed precision (AMP) using `torch.autocast("cuda")` further reduces the memory footprint of intermediate activations by ~40% (float32 → float16 for most ops), making the 8-layer model safe even at batch_size=4 with 242K-node meshes.

**Concrete change in `train.py`**:

```python
# 1. Add torch.autocast to training loop (forward + loss only, not backward):
for x, y, is_surface, mask in tqdm(train_loader, ...):
    ...
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (y - stats["y_mean"]) / stats["y_std"]

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = model({"x": x_norm})["preds"]
        sq_err = (pred.float() - y_norm) ** 2  # cast back to fp32 before loss
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 2. In evaluate_split: optionally add autocast for faster inference:
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    pred = model({"x": x_norm})["preds"]
pred = pred.float()   # cast back before denormalize+MAE

# 3. model_config change:
model_config = dict(
    ...
    n_layers=8,   # was 5
    ...
)
```

**Memory estimate**: n_layers=8 at n_hidden=128 adds ~6M more params but no non-linear VRAM scaling. AMP reduces forward pass activations by ~40%. Expected peak VRAM: ~35–45 GB at batch_size=4.

**Important**: Do NOT use GradScaler with AMP here — the loss is already in float32 (we cast pred back before loss). If using full AMP pipeline with fp16 loss, add `scaler = torch.cuda.amp.GradScaler()` and `scaler.step(optimizer)` / `scaler.update()`.

**Predicted improvement**: 4–10% on all splits (capacity gain is general). Depth is a stronger lever than width for this architecture given the slice-attention bottleneck.

**Predicted risks**:
- 8 layers × ~1.5 min/epoch = slightly slower. Verify ≥14 epochs within 30 min before full run.
- AMP can cause NaN in softmax if attention logits overflow fp16 range. Mitigation: `torch.autocast` with `dtype=torch.float16` handles this via the `scaled_dot_product_attention` path. Monitor for NaN loss at epoch 1.
- LayerNorm in fp16 can reduce precision for extreme values; cast pred to float32 before loss computation.

**References**:
- Micikevicius et al. (2018), "Mixed Precision Training", https://arxiv.org/abs/1710.03740
- PyTorch AMP tutorial: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

**Student profile**: moderate; AMP wrapping is mechanical but requires careful cast management.

---

### 5. Surface-Node Weighted MAE in Training Loss

**Category**: loss reformulation

**Hypothesis**: The current surface loss is MSE (L2), but the evaluation metric is MAE (L1). Replacing the surface component of the loss with L1 (MAE) while keeping the volume component as L2 will align training geometry with the evaluation signal, reducing the surface pressure MAE by 5–10%.

**Mechanism**: MSE and MAE optimize different statistics: MSE minimizes the squared deviation (penalizes large errors quadratically) while MAE minimizes the mean absolute deviation (linear penalty, robust to outliers). The primary metric is `mae_surf_p` — a pure L1 quantity. Training with L2 creates a systematic mismatch: the model over-attends to the few nodes with large surface errors (MSE's quadratic regime), which may coincide with geometric stagnation points and leading edges. Switching the surface component to L1 aligns gradient direction with the evaluation metric. The volume component stays L2 because volume flow has no direct primary metric and L2 training is typically more stable for initial convergence.

**Concrete change in `train.py`**:

```python
# In the training loop, after computing pred:
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface

# Volume: keep L2 (MSE)
sq_err_vol = (pred - y_norm) ** 2
vol_loss = (sq_err_vol * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)

# Surface: switch to L1 (MAE)
abs_err_surf = (pred - y_norm).abs()
surf_loss = (abs_err_surf * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * surf_loss
```

**Note**: The `evaluate_split` monitoring loss will now be heterogeneous (L2 vol + L1 surf). This is fine — only the MAE metrics matter for checkpoint selection.

**Memory**: No change. L1 is implemented as `.abs()` rather than `.pow(2)`.

**Predicted improvement**: 3–8% on all splits; stronger effect on `geom_camber_rc` and `geom_camber_cruise` where the surface geometry is OOD and MSE's quadratic regime may be systematically misleading.

**Predicted risks**:
- L1 has a subgradient at zero — small errors near convergence may oscillate. Mitigation: use `F.smooth_l1_loss(reduction="none", beta=0.1)` as a compromise (L2 for errors < 0.1, L1 for errors ≥ 0.1).
- The `surf_weight=10.0` was calibrated for L2 surface loss; with L1 surface loss the effective weighting changes because the magnitudes differ. May need to re-tune `surf_weight` (try 15.0 as a variant).

**References**:
- Zhao et al. (2017), "Loss Functions for Image Restoration with Neural Networks" — systematic comparison of L1 vs L2 in field prediction tasks
- Li et al. (2023), "Long-term Weather Forecasting with FourCastNet" — uses L1 loss aligned with evaluation metric

**Student profile**: easy; 5-line change in loss block.

---

### 6. Learnable Fourier Frequency Bands

**Category**: features/data

**Hypothesis**: Replacing the fixed octave-doubling Fourier feature frequencies (1, 2, 4, 8, 16, 32, 64, 128 cycles/unit) with jointly-learned frequency parameters will allow the model to concentrate encoding capacity on the spatial scales that matter most for the foil boundary layer, improving OOD geometry splits by 4–8%.

**Mechanism**: The Round 1 Fourier feature baseline (PR #3200) uses a fixed geometric progression of frequencies. These are motivated by NeRF-style positional encoding but are not adapted to the TandemFoilSet spatial structure. The foil boundary layer has characteristic scales determined by chord length and Re. The background flow field has much larger scales. Fixed octave doubling is agnostic to this structure. Learning the frequencies (as a `nn.Parameter` initialized to the octave-doubling baseline) allows gradient descent to discover which scales are most discriminative for this dataset. This is the "random Fourier features with learned frequencies" idea from Tancik et al. (2020) / Rahimi & Recht (2007), adapted to make frequencies trainable rather than random-fixed.

**Implementation detail**: The Fourier features are appended to x in `train.py` (not in the dataloader), so this is entirely in-scope for `train.py`.

**Concrete change in `train.py`** (add before model instantiation):

```python
# --- Learnable Fourier position encoding ---
N_FOURIER_BANDS = 8   # matches Round 1 baseline
# Initialize to octave-doubling baseline, make trainable
fourier_freqs = nn.Parameter(
    torch.tensor([2.0 ** i for i in range(N_FOURIER_BANDS)],
                 dtype=torch.float32),
    requires_grad=True
)
# Register with optimizer (add separately from model params):
optimizer = torch.optim.AdamW(
    list(model.parameters()) + [fourier_freqs],
    lr=cfg.lr, weight_decay=cfg.weight_decay
)

def apply_fourier_features(x_raw: torch.Tensor) -> torch.Tensor:
    """Append 32 learnable Fourier features to the 24-dim raw input."""
    # x_raw: [B, N, 24]; dims 0-1 are (x, z) position
    pos = x_raw[..., :2]  # [B, N, 2]
    # freqs: [N_FOURIER_BANDS] — broadcast over [B, N, 2, N_FOURIER_BANDS]
    angles = pos.unsqueeze(-1) * fourier_freqs * 2 * torch.pi  # [B, N, 2, 8]
    feats = torch.cat([angles.sin(), angles.cos()], dim=-1)    # [B, N, 2, 16]
    feats = feats.reshape(*x_raw.shape[:-1], 32)               # [B, N, 32]
    return torch.cat([x_raw, feats], dim=-1)                   # [B, N, 56]

# In the training loop:
x_norm = (x - stats["x_mean"]) / stats["x_std"]
x_norm = apply_fourier_features(x_norm)  # [B, N, 56]
pred = model({"x": x_norm})["preds"]

# In evaluate_split, same apply_fourier_features call before model forward.
```

**Note**: `X_DIM` stays 56, `fun_dim=54` stays — only the frequency values change from fixed to learned. The frequency gradient is tiny vs. model gradients; AdamW will handle both.

**Memory**: 8 extra float32 parameters (negligible). No activation overhead.

**Predicted improvement**: 4–8% on OOD geometry splits (geom_camber_rc, geom_camber_cruise) where the fixed frequencies may not optimally encode unseen NACA profiles.

**Predicted risks**:
- Frequencies can drift to near-zero or collapse to the same value (degenerate solution). Add a log-barrier or simply monitor the `fourier_freqs` values via W&B.
- The apply_fourier_features function must be called consistently in both the training loop AND `evaluate_split`. Easy to forget the eval-time call.
- Gradient flow through `fourier_freqs` is through a `sin`/`cos` — well-behaved but the learning signal is indirect (through the model, not direct supervision on frequencies).

**References**:
- Tancik et al. (2020), "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", https://arxiv.org/abs/2006.10739
- Li et al. (2021), "Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding", https://arxiv.org/abs/2106.02795

**Student profile**: moderate; requires wrapping the Fourier computation in a function and ensuring consistent application in both train and eval paths.

---

### 7. Domain-Type One-Hot Embedding as Extra Input Feature

**Category**: features/data

**Hypothesis**: Injecting a 3-dim one-hot domain indicator (raceCar-single, raceCar-tandem, cruise-tandem) as an additional input feature will allow the model to learn domain-specific representations without requiring architectural changes, improving in-dist and OOD metrics by 3–7%.

**Mechanism**: The three training domains (raceCar single, raceCar tandem, cruise tandem) have fundamentally different geometry (ground-effect vs. freestream), flow regimes (inverted vs. upright foil), mesh structure (85K vs. 127K vs. 210K nodes), and target value ranges. The model currently must infer the domain from dims 18–23 (which are 0 for single, nonzero for tandem) and the NACA parameters (which separate raceCar from cruise by AoA sign). This is implicit and requires the model to learn the domain structure from weak signal. An explicit one-hot domain indicator gives the model direct access to this coarse partition, reducing the burden on the attention mechanism to re-discover it.

**Domain detection rule** (fully derivable from existing x features, no new data needed):
- raceCar-single: `x[:, :, 18] == 0` AND `x[:, :, 14] < 0` (no foil2, negative AoA)
- raceCar-tandem: `x[:, :, 18] != 0` AND `x[:, :, 14] < 0`
- cruise-tandem: `x[:, :, 18] != 0` AND `x[:, :, 14] >= 0`

**Concrete change in `train.py`**:

```python
def compute_domain_onehot(x_raw: torch.Tensor) -> torch.Tensor:
    """
    Returns a 3-dim one-hot domain indicator repeated for all nodes.
    x_raw: [B, N, 24] — raw (unnormalized) features
    Returns: [B, N, 3]
    """
    # Use unnormalized features for reliable thresholding
    aoa1 = x_raw[:, 0, 14]           # [B] — AoA foil1, dim 14
    foil2_present = (x_raw[:, 0, 18] != 0).float()  # [B] — AoA foil2 nonzero
    
    # Domain indices: 0=rc_single, 1=rc_tandem, 2=cruise_tandem
    is_cruise = (foil2_present > 0) & (aoa1 >= 0)  # [B]
    is_rc_tandem = (foil2_present > 0) & (aoa1 < 0)
    # rc_single = not tandem (foil2_present == 0)

    domain_idx = torch.zeros(x_raw.shape[0], dtype=torch.long, device=x_raw.device)
    domain_idx[is_rc_tandem] = 1
    domain_idx[is_cruise] = 2

    onehot = F.one_hot(domain_idx, num_classes=3).float()  # [B, 3]
    return onehot.unsqueeze(1).expand(-1, x_raw.shape[1], -1)  # [B, N, 3]

# In the training loop, after normalization:
x_norm = (x - stats["x_mean"]) / stats["x_std"]
domain_oh = compute_domain_onehot(x)  # uses raw x for threshold safety
x_aug = torch.cat([x_norm, domain_oh], dim=-1)  # [B, N, 59]
pred = model({"x": x_aug})["preds"]

# Update model_config:
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2 + 3,  # 54 + 3 = 57
    ...
)
# X_DIM is 56 (post-Fourier baseline); the aug input is 59 dims
# fun_dim = 59 - 2 = 57 (subtract space_dim=2 for position dims)
```

**Memory**: 3 extra input channels. Slightly larger preprocess MLP input (59 vs. 56), negligible increase.

**Predicted improvement**: 3–7% overall; strongest benefit expected on val_geom_camber splits where domain-specific geometric priors matter most.

**Predicted risks**:
- The domain-detection rule uses `x_raw[:, 0, 14]` (node 0) to read AoA. This assumes all nodes in a sample have the same global flow parameters, which is true by construction (global flow conditions are replicated per node). But verify that `x_raw[:, 0, ...]` is not padding. Use `mask` to find a valid node: `valid_node = mask.float().argmax(dim=1)` to safely index.
- Val samples that fall in Part 2 (OOD camber splits) still belong to raceCar-tandem or cruise-tandem domains — the one-hot is still correct for them.
- One-hot is not continuous — if the model learns to key off these bits too hard, it may fail to generalize across domains.

**References**:
- Domain-conditional inputs are standard in multi-task learning literature
- Pathak et al. (2022), "FourCastNet" — uses explicit geometry/condition indicators as extra channels

**Student profile**: easy; pure input augmentation in the training loop.

---

### 8. AdamW + Separate Learning Rate for Fourier Frequency Parameters

**Category**: optimizer/schedule

**Hypothesis**: Using parameter-group-specific learning rates — a lower LR for Transolver model parameters and a higher LR for the Fourier frequency parameters (from hypothesis #6) — will improve training stability by letting the architectural parameters converge slowly while the encoding parameters adapt quickly to the dataset's characteristic spatial scales.

**Note**: This hypothesis builds on hypothesis #6 (Learnable Fourier Features). It should be tried as a combined experiment or as a follow-up to #6 if #6 succeeds. As a standalone experiment (with the new baseline's fixed Fourier features), this reduces to testing multi-rate AdamW groups, which is still valuable.

**For standalone use (fixed Fourier, new baseline)**: Use parameter groups to give `slice_weights` projection (`in_project_slice`) a higher LR than other parameters, since the slice assignment is the most dynamic part of the architecture and benefits from faster convergence.

**Concrete change in `train.py`**:

```python
# Parameter groups with different LRs:
slice_params = [
    p for name, p in model.named_parameters()
    if "in_project_slice" in name or "temperature" in name
]
other_params = [
    p for name, p in model.named_parameters()
    if "in_project_slice" not in name and "temperature" not in name
]

optimizer = torch.optim.AdamW(
    [
        {"params": other_params, "lr": cfg.lr},             # 5e-4
        {"params": slice_params, "lr": cfg.lr * 5.0},       # 2.5e-3
    ],
    weight_decay=cfg.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS
)
```

**Memory**: No change. Only the optimizer state grows by a tiny amount (one extra momentum buffer per parameter group).

**Predicted improvement**: 2–5% across all splits. This is a lower-confidence hypothesis but very cheap to test.

**Predicted risks**:
- The slice projection (`in_project_slice`) has orthogonal initialization — a higher LR may destroy the orthogonality faster. This may actually be beneficial (faster departure from the symmetric initial state) or harmful (instability). Monitor the val loss at epoch 1.
- If orthogonality is destroyed, the physics-aware attention may degrade in the first few epochs before recovering.

**References**:
- Karpathy (2022), "Recipe for Training Neural Networks" — parameter-group-specific LR is a standard trick for fine-tuning transformers
- Transolver (2024), https://arxiv.org/abs/2402.02366 — notes the temperature parameter and slice projection as the most sensitive parts of PhysicsAttention

**Student profile**: easy; optimizer configuration only.

---

### 9. Divergence-Free Velocity Auxiliary Loss (Physics Constraint)

**Category**: bold/high-risk

**Hypothesis**: Adding a weak auxiliary loss term penalizing the divergence of the predicted velocity field `∇·u = ∂Ux/∂x + ∂Uy/∂z ≈ 0` (incompressible flow condition) will regularize the model toward physically consistent velocity predictions, improving both velocity channels and indirectly pressure accuracy via the coupling.

**Mechanism**: The Navier-Stokes incompressibility constraint `∇·u = 0` holds exactly in steady 2D incompressible flow (which this dataset represents). The current model is trained purely on MSE — it can predict velocity fields that are non-divergence-free without any penalty. Adding `λ * ||∇·u_pred||²` to the loss will steer the model toward physically realizable solutions. Even when the divergence loss is not directly tied to the primary metric (surface pressure), pressure and velocity are coupled through the Navier-Stokes pressure Poisson equation — improving velocity field consistency should improve pressure predictions indirectly.

**Implementation**: Finite differences on the mesh. For each sample, the mesh is not structured, but the local node neighborhoods can be approximated via finite differences if we assume locally uniform spacing at each node. Alternatively, we approximate the divergence at each internal node using nearby neighbors from the input feature `dsdf` (distance-based shape descriptor, dims 4–11 of x) — but this approach is expensive and imprecise.

**Simpler viable approach**: Instead of computing actual spatial derivatives, use the normalized prediction and its L2 norm as a proxy, OR apply the constraint only to the regular background zone (Zone 0), where the mesh is approximately structured. Since we do not have mesh adjacency in the loader, use finite differences in the sorted x-position order as an O(N) approximation.

**Concrete change in `train.py`** (training loop only):

```python
# After computing pred: [B, N, 3] in normalized space
# Denormalize velocity to physical space for physics constraint
pred_phys = pred * stats["y_std"] + stats["y_mean"]  # [B, N, 3]
ux_pred = pred_phys[..., 0]  # [B, N]
uy_pred = pred_phys[..., 1]  # [B, N]

# Approximate divergence using adjacent node pairs along sorted x-axis
# Sort by x-coordinate (dim 0 of x); use finite differences on sorted neighbors
# This is an approximation valid for the ~structured Zone 0 background mesh
x_pos = x[..., 0]  # [B, N] — x-position in physical space
# Sort nodes by x-position for each sample in the batch
sort_idx = x_pos.argsort(dim=1)  # [B, N]
ux_sorted = ux_pred.gather(1, sort_idx)  # [B, N]
uy_sorted = uy_pred.gather(1, sort_idx)
x_sorted = x_pos.gather(1, sort_idx)
z_sorted = x[..., 1].gather(1, sort_idx)

# Central differences (interior nodes only)
dux_dx = (ux_sorted[:, 2:] - ux_sorted[:, :-2]) / (x_sorted[:, 2:] - x_sorted[:, :-2] + 1e-6)
duy_dz = (uy_sorted[:, 2:] - uy_sorted[:, :-2]) / (z_sorted[:, 2:] - z_sorted[:, :-2] + 1e-6)
divergence = dux_dx + duy_dz  # [B, N-2]

div_loss = divergence.pow(2).mean()
loss = vol_loss + cfg.surf_weight * surf_loss + cfg.div_weight * div_loss

# Add to Config:
div_weight: float = 0.01  # small regularizer; tune in [0.001, 0.1]
```

**Memory**: Sort + gather adds O(B×N) temporary memory. At B=4, N=242K, this is ~4×242K×4 bytes × 3 tensors ≈ 12 MB. Negligible.

**Important caveat**: The sorted-x finite difference is a rough approximation — the mesh is unstructured, and sorting by x-position gives random z-spacing. This means `duy_dz` computed on x-sorted nodes is not a true z-derivative. A cleaner approach: use only pairs of nodes that share a mesh edge. But edge connectivity is not in the dataloader. A middle ground: random node pairs and check if a divergence-free field has lower pairwise velocity differences along approximate normals. Given the approximation quality, start with `div_weight=0.001` (very weak regularizer).

**Predicted improvement**: 2–8% on velocity channels; unclear benefit on `mae_surf_p` directly. This is a high-variance experiment — it may improve velocity physics consistency with no benefit to the primary metric, or it may act as a useful regularizer.

**Predicted risks**:
- The finite-difference approximation on unstructured meshes is noisy — the div_loss signal may be dominated by approximation noise, providing no useful gradient. Diagnostic: log `div_loss` separately and check it decreases.
- Zone 0 (background) has ~60% of nodes and is quasi-structured; Zone 1/2 (dense foil zones) are highly unstructured. The sorted-x approximation will be worst for Zone 1/2 nodes. Possible mitigation: apply div_loss only to non-surface nodes using the `vol_mask`.

**References**:
- Wandel et al. (2021), "Learning Incompressible Fluid Dynamics from Scratch", https://arxiv.org/abs/2006.05796 — direct ∇·u=0 penalty in loss
- Mohan et al. (2020), "Embedding Hard Physical Constraints in Convolutional Neural Networks for 3D Turbulence", https://arxiv.org/abs/2002.00021 — hard vs. soft physics constraints

**Student profile**: challenging; requires careful implementation of the approximation and monitoring div_loss separately.

---

### 10. Slice Count Scaling: slice_num=96 with Gradient Checkpointing

**Category**: bold/high-risk

**Hypothesis**: Increasing `slice_num` from 64 to 96 (50% more slice tokens) with gradient checkpointing to compensate for the larger O(slice_num²) attention matrix will increase the model's ability to represent complex flow patterns over larger neighborhoods, improving OOD geometry splits by 5–12%.

**Mechanism**: The slice-based attention in PhysicsAttention is O(slice_num²) in the attention computation and O(N × slice_num) in the pooling. The current `slice_num=64` creates 64 physics-aware tokens per sample. For tandem foil configurations with 127K–210K nodes across 2–3 mesh zones, 64 slices may be insufficient — the two foil boundary layers, their interaction zone, and the background flow are qualitatively different regions that may each need multiple slices to represent well. Increasing to 96 adds 50% more slice tokens (9,216 → 9,216 attention elements; 64×4 → 96×4 head × slice matrices) at modest cost. Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) on the TransolverBlocks reduces the peak activation memory at the cost of ~33% extra computation (one extra forward pass per block for the checkpoint).

**Concrete change in `train.py`**:

```python
import torch.utils.checkpoint as cp

# Modify TransolverBlock.forward to support gradient checkpointing:
class TransolverBlock(nn.Module):
    def forward_with_checkpoint(self, fx):
        return cp.checkpoint(self._forward_inner, fx, use_reentrant=False)

    def _forward_inner(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

    def forward(self, fx):
        return self._forward_inner(fx)  # default: no checkpointing

# In Transolver.forward with checkpointing enabled:
def forward(self, data, use_checkpoint=False, **kwargs):
    x = data["x"]
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks:
        if use_checkpoint and self.training:
            fx = cp.checkpoint(block._forward_inner, fx, use_reentrant=False)
        else:
            fx = block(fx)
    return {"preds": fx}

# In model_config:
model_config = dict(
    ...
    slice_num=96,  # was 64
    ...
)

# In the training forward call:
pred = model({"x": x_norm}, use_checkpoint=True)["preds"]
# In eval (evaluate_split): use_checkpoint=False (default)
```

**Memory estimate**: slice_num=96 increases PhysicsAttention tensors by (96/64)² = 2.25× for the attention matrix portion. With gradient checkpointing active during training, peak activation memory is bounded by O(B × N × n_hidden × 1 layer) rather than O(B × N × n_hidden × n_layers). Expected peak VRAM: ~45–55 GB at batch_size=4. Should fit within 96 GB.

**Predicted improvement**: 5–12% on tandem configurations (geom_camber_rc, geom_camber_cruise, re_rand), where the dual-foil interaction zone benefits most from additional slice resolution.

**Predicted risks**:
- Gradient checkpointing adds ~33% training time per epoch. Verify ≥14 epochs still fit within 30 min.
- `use_reentrant=False` is required for compatibility with autocast and `in_project_slice`'s orthogonal initialization. Do not use `use_reentrant=True`.
- If slice_num=96 is still OOM at batch_size=4 even with checkpointing, reduce to slice_num=80 as a fallback.

**References**:
- Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost", https://arxiv.org/abs/1604.06174
- PyTorch gradient checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- Transolver (2024), https://arxiv.org/abs/2402.02366 — slice_num ablation shows monotonic improvement up to slice_num=128

**Student profile**: moderate; checkpointing API is well-documented but requires care with `use_reentrant`.

---

## Summary Table

| # | Category | Hypothesis | Expected gain | Risk | Wall-clock safety |
|---|----------|------------|---------------|------|-------------------|
| 1 | optimizer/schedule | OneCycleLR + gradient clipping | 3–8% | Medium (LR sensitivity) | Safe (no arch change) |
| 2 | loss | Huber loss (beta=1.0) for high-Re robustness | 3–8% | Low | Safe |
| 3 | architecture | FiLM Re conditioning per block | 5–12% | Medium (conditioner integration) | Safe (+80K params) |
| 4 | architecture | n_layers=8 + AMP (mixed precision) | 4–10% | Medium (AMP NaN risk) | Safe if AMP enabled |
| 5 | loss | L1 surface loss (L2 vol, L1 surf) | 3–8% | Low (surf_weight re-tuning) | Safe |
| 6 | features/data | Learnable Fourier frequencies | 4–8% | Medium (frequency collapse) | Safe |
| 7 | features/data | Domain one-hot as extra feature | 3–7% | Low | Safe |
| 8 | optimizer/schedule | Per-group LR: slice projection 5× higher | 2–5% | Low | Safe |
| 9 | bold | Divergence-free velocity aux loss | 2–8% | High (approx quality) | Safe |
| 10 | bold | slice_num=96 + gradient checkpointing | 5–12% | Medium (VRAM, time) | Tight (verify epoch time) |

## Prioritization for student assignment

**Highest priority** (strongest expected signal, lowest complexity):
- #2 Huber loss (3-line change, clear mechanism, no risk)
- #5 L1 surface loss (5-line change, directly targets metric misalignment)
- #7 Domain one-hot (pure input augmentation, zero architectural risk)

**High priority** (strong mechanism, moderate complexity):
- #3 FiLM Re conditioning (addresses re_rand split weakness directly)
- #1 OneCycleLR + grad clipping (two components, both well-motivated)
- #4 n_layers=8 + AMP (capacity increase with concrete memory plan)

**Medium priority** (promising but requires careful implementation):
- #6 Learnable Fourier frequencies (builds on new baseline, creative)
- #10 slice_num=96 + checkpointing (strong Transolver-specific lever)

**Lower priority / exploratory**:
- #8 Per-group LR (small expected gain, useful as add-on to #6)
- #9 Divergence-free aux loss (high implementation complexity, uncertain transfer to primary metric)
