# Research Ideas — 2026-05-13 19:00
# Branch: icml-appendix-charlie-pai2g-24h-r1
# Baseline: PR #2011 val_avg/mae_surf_p=28.8762, test_avg=24.9992

---

## Idea 1 — Surface / Volume Split Head (Hard Inductive Bias)

**Hypothesis:** The model is penalized simultaneously on interior volume nodes (low-stakes) and surface nodes (high-stakes for `mae_surf_p`). A shared output projection forces the model to compromise. Splitting the final linear head into two separate MLPs — one for `is_surface=True` nodes, one for volume nodes — allows the surface head to specialize entirely on the pressure-dominated surface regime, while the volume head focuses on the smooth Navier-Stokes interior. This is a near-zero-parameter change that targets the exact bottleneck: surface pressure prediction.

**Implementation:**
- In `Transolver` decoder, after the transformer layers produce node embeddings `h ∈ [B, N, d]`, replace the single `nn.Linear(d, 3)` output projection with:
  ```python
  self.surf_head = nn.Linear(d, 3)
  self.vol_head  = nn.Linear(d, 3)
  # forward:
  surf_mask = x_dict['is_surface'].bool()  # shape [B, N]
  out = torch.where(surf_mask.unsqueeze(-1), self.surf_head(h), self.vol_head(h))
  ```
- Initialize both heads identically to current head weights (or from scratch with zero-init on bias).
- Keep `ReScaleHead` conditioning on both heads via the same `re_scale` FiLM signal.
- Loss: keep current weighted cross-field loss unchanged — the split head does not require any loss change.
- `surf_weight` is already 10, so surface nodes dominate loss — the head specialization aligns the capacity with what the loss already emphasizes.

**Why it might beat baseline:** The surface pressure field has sharp gradients near leading/trailing edges and a different characteristic length scale than the interior. A shared head must learn a basis that spans both, which dilutes expressiveness. Specialized heads are a known win in multi-task output settings (citation: MT-DNN, multi-head decoders in weather models).

**Risk:** Very low — one extra `nn.Linear(d,3)` (~6K params on d=256). Could slightly increase overfit on 1499 samples but the split is physically motivated. Ablate with volume head weight-decay 10× if needed.

---

## Idea 2 — Dynamic Pressure as Derived Input Feature

**Hypothesis:** The 24 input features include `Ux, Uy` as part of the output targets, NOT the inputs — inputs are purely geometric/parametric (position, shape descriptors, Re, AoA, gap). However, we can compute a proxy for local flow speed from the mesh geometry: the signed arc-length `s` and `dsdf` encode proximity to the wall, and the Reynolds number encodes the freestream speed scale. The missing link is a per-node estimate of dynamic pressure `q_proxy = Re^α × f(dsdf, s)`. A simpler version: add `log(Re)^2` and `Re × AoA` as explicit scalar input features broadcast to all nodes — these polynomial interaction terms encode the nonlinear Re-AoA coupling that governs lift/drag and hence surface pressure.

**Implementation:**
- In the data loader (read-only) workaround: construct extra features in `train.py` before passing to model by patching the input tensor in the collate step:
  ```python
  # x shape: [B, N, 24]
  log_re   = x[..., feat_idx['log_re']].unsqueeze(-1)   # [B,N,1]
  aoa1     = x[..., feat_idx['aoa1']].unsqueeze(-1)
  x_extra  = torch.cat([x, log_re**2, log_re * aoa1], dim=-1)  # [B,N,26]
  ```
- Update `Transolver` `input_dim` from 24 → 26 (two extra channels).
- Normalize new features with running stats over training set (or use the existing per-channel normalizer if it already handles extras).
- Keep everything else fixed.

**Why it might beat baseline:** Polynomial feature interaction terms are a cheap form of feature engineering that neural networks can in principle learn — but with 1499 samples and a deep transformer, the network may not reliably discover the Re²-AoA coupling. Making it explicit reduces the learning burden on early layers and is standard practice in tabular/sparse-data settings where sample count is tight.

**Risk:** Low. If the normalizer contract in `scoring.py` is hard-coded to 24-dim inputs, verify the model-side only change is transparent to the scoring pipeline. Extra features are model-internal; scoring.py operates on outputs, not inputs, so this is safe.

---

## Idea 3 — Stochastic Depth (Drop-Path) Regularization

**Hypothesis:** With 1499 training samples and a ~4M-parameter Transolver, the primary failure mode is overfitting. All attempted regularization axes (weight-decay grid, LayerScale) were architectural. Stochastic Depth (drop-path) is a different regularization modality: it randomly zeros entire residual branches during training, forcing the model to learn redundant representations across layers. It is the single most effective regularizer in ViT-family models on small datasets (DeiT, ConvNeXt papers demonstrate consistent 0.5–2% improvement on ImageNet with N<1M samples; the analogy holds for mesh transformers).

**Implementation:**
- Add `drop_path_rate` parameter to `Transolver` (e.g., 0.1).
- Use `timm`'s `DropPath` or a simple inline implementation:
  ```python
  from timm.layers import DropPath
  # in each TransolverBlock.__init__:
  self.drop_path = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()
  # in forward, after residual:
  x = x + self.drop_path(attn_out)
  x = x + self.drop_path(ffn_out)
  ```
- Use linear stochastic depth schedule: layer 0 → 0.0, layer L-1 → 0.1.
- Apply ONLY during training; eval uses identity (drop_path handles this automatically).
- Keep all other hyperparameters at baseline.

**Why it might beat baseline:** All recent plateau experiments adjust optimizer or loss weights. Stochastic depth is an orthogonal regularization axis that has not been tested. On small datasets it typically outperforms dropout (which breaks attention patterns). The mechanism is well-validated in ViT literature and the physics-attention architecture is structurally identical to a ViT in terms of residual connections.

**Risk:** Low-medium. If the model is already underfitting (unlikely at 28.8 with 1499 samples), drop-path will hurt. Cheap diagnostic: check if train MAE << val MAE in baseline — if yes, overfit is real and drop-path is the right tool.

---

## Idea 4 — Fourier Feature Positional Encoding for Mesh Coordinates

**Hypothesis:** The current input features include raw (x, z) mesh node coordinates. The model must learn to represent spatial frequency content from raw coordinates, which is inefficient for shallow early layers. Random Fourier Features (RFF) / Gaussian Fourier positional encodings convert raw coordinates to a fixed-frequency embedding that encodes multi-scale spatial structure explicitly. This is the mechanism behind NeRF's positional encoding and has been shown in several physics-ML papers (Fourier Neural Operator, implicit neural representations) to dramatically reduce the number of layers needed to represent high-frequency spatial variation — directly relevant to sharp pressure gradients near leading edges.

**Implementation:**
- Pre-compute Fourier features for (x, z) coordinates:
  ```python
  # In train.py, augment node features before model forward pass
  def fourier_encode(coords, num_freq=8, sigma=1.0):
      # coords: [B, N, 2]
      B = torch.randn(2, num_freq, device=coords.device) * sigma  # fixed across training
      proj = coords @ B  # [B, N, num_freq]
      return torch.cat([torch.sin(2*pi*proj), torch.cos(2*pi*proj)], dim=-1)  # [B,N,16]
  ```
- Fix `B` at initialization (not learned) with `sigma` tuned to coordinate scale (~1.0 normalized).
- Append to input features: 24 → 40 dims. Update `input_dim`.
- Try `num_freq` in {4, 8} — prefer 4 (adds 8 dims) for speed.

**Why it might beat baseline:** The Transolver's physics-attention slice tokens aggregate spatially, but the node feature encoder before slicing uses raw coordinates. Fourier encoding gives each node a rich multi-scale spatial identity that distinguishes leading-edge nodes from trailing-edge nodes at multiple length scales. Related work: `MeshGraphNets` with Fourier node features outperform raw-coord variants on aerodynamic meshes.

**Risk:** Medium. The `sigma` / frequency scale requires tuning to match the coordinate normalization. Start with `sigma=1.0` on normalized coords. If the coordinate normalization in `data/loader.py` produces values in [-1, 1], use `sigma=0.5`.

---

## Idea 5 — Mixup / CutMix in Parameter Space (Physics-Consistent Augmentation)

**Hypothesis:** With 1499 training samples, data augmentation is the most direct lever against overfitting. Geometric/coordinate augmentations (jitter, translation) were closed because they perturb physical meaning. However, linear interpolation in the **parameter space** (Re, AoA1, AoA2, NACA1, NACA2, gap, stagger) is physically meaningful: the convex combination of two flow configurations is approximately a valid flow if the parameters vary smoothly (this is the basis of reduced-order modeling). Mixup on the (parameter vector, output field) pairs — weighted average of two samples' inputs AND their target fields — creates synthetic training examples with correct physical labels, massively expanding the effective training set.

**Implementation:**
- In the training loop, after loading a batch `(x1, y1)` and a randomly shuffled batch `(x2, y2)`:
  ```python
  lam = np.random.beta(alpha, alpha)  # alpha=0.4 suggested
  x_mix = lam * x1 + (1-lam) * x2    # [B, N, 24] — mix ALL features including coords
  y_mix = lam * y1 + (1-lam) * y2    # [B, N, 3]
  ```
- CRITICAL: only mix within the same mesh topology (same number of nodes after padding) OR mix at the sample level before pad_collate. Since meshes have variable sizes, the safest implementation is to draw two same-sized sub-batches and mix them before collation.
- Alternative safe implementation: mix only the **scalar global parameters** (Re, AoA1, AoA2, NACA1, NACA2, gap, stagger — the 7 broadcast features) and their corresponding outputs, keeping the geometric node features fixed from one parent sample. This avoids mesh topology conflicts entirely.
- `alpha=0.2` (conservative) → `alpha=0.4` (moderate). Avoid `alpha>0.5` — too far from real data.

**Why it might beat baseline:** On small datasets (N<2000), Mixup consistently reduces generalization gap by 1-3% in classification and regression settings. For CFD surrogates with smooth parameter-to-field mappings, the interpolation assumption is better justified than for image tasks. This is an entirely untested axis on this branch.

**Risk:** Medium-high. The variable mesh size makes naive Mixup tricky. The safest version (mix only scalar features) degrades to a mild input regularizer. The full version (mix all features) requires careful same-topology batching. Recommend starting with scalar-only Mixup.

---

## Idea 6 — Multi-Query / Grouped-Query Attention with Wider Hidden Dim

**Hypothesis:** Current architecture: n_layers=4, n_head=4, d=256. Experiments with n_layers=6 and n_head=8 were closed (presumably increased capacity without improving generalization). The missing direction is **width vs. depth vs. heads** trade-off under a fixed parameter budget: replacing standard multi-head attention with Grouped-Query Attention (GQA) halves the KV projection cost, freeing parameters for a wider feed-forward MLP or wider hidden dim. At the same parameter count, GQA-wide beats MHA-narrow on small-data transformers because the wider FFN improves representational capacity while the reduced KV redundancy improves generalization.

**Implementation:**
- In `PhysicsAttention`, replace 4-head MHA with 2-group GQA (2 query heads per KV head):
  ```python
  # num_heads=4, num_kv_heads=2 (GQA)
  self.q_proj = nn.Linear(d, d)       # 4 heads, d/4 each
  self.k_proj = nn.Linear(d, d//2)    # 2 KV heads, d/4 each
  self.v_proj = nn.Linear(d, d//2)
  # expand KV for grouped attention during forward
  ```
- Simultaneously, increase `d_ffn` from `4*d=1024` → `6*d=1536` (recovered from KV savings).
- Keep `n_layers=4`, `n_head=4` total, `d=256` — same parameter budget, different allocation.
- Or simpler: just increase `mlp_ratio` from 4 → 5 on existing architecture (mlp_ratio=3 was closed, but 3 < 4 baseline; 5 > 4 is unexplored upward).

**Why it might beat baseline:** GQA + wider FFN is the dominant architectural pattern in recent LLM work (Llama 2/3, Mistral, Gemma) precisely because it improves quality per parameter. The physical motivation: the attention mechanism aggregates spatial context (structure-preserving), while the FFN applies nonlinear transformation per node (function approximation). On CFD surrogates, the function approximation bottleneck is likely more limiting than the attention bottleneck.

**Risk:** Medium. Implementation requires modifying `PhysicsAttention` forward pass carefully. The GQA version is more complex; the `mlp_ratio=5` version is a one-line change and should be tried first as a fast diagnostic.

---

## Idea 7 — Bernoulli-Based Surface Pressure Soft Constraint

**Hypothesis:** For inviscid, incompressible, steady flow, Bernoulli's equation gives: `p + ½ρ|U|² = const` along a streamline. On the airfoil surface where `|U|_surface ≈ 0` (no-slip), the stagnation pressure is approximately `p_stag ≈ p_ref + ½ρU_∞²`. This means surface pressure at each surface node should satisfy a soft constraint relative to freestream conditions. Adding an auxiliary physics loss that penalizes violations of this soft constraint on surface nodes — weighted by distance from stagnation point — injects physical inductive bias that directly targets `mae_surf_p`.

**Implementation:**
- In the loss function, add a Bernoulli residual term on surface nodes only:
  ```python
  # pred_u, pred_p: predicted [Ux, Uy, p] at surface nodes (after denorm)
  q_dynamic = 0.5 * (pred_u[...,0]**2 + pred_u[...,1]**2)  # ½|U|²
  p_total   = pred_p[...,2] + q_dynamic                      # p + q
  # Penalize variance of p_total across surface nodes (should be ~const per sample)
  bernoulli_loss = p_total.var(dim=-1).mean()  # var over surface nodes, mean over batch
  loss_total = loss_main + bernoulli_weight * bernoulli_loss
  ```
- Start with `bernoulli_weight=0.01` (very small, so it doesn't destabilize training).
- Apply ONLY to predicted values on surface nodes (`is_surface=True`).
- Use normalized predictions for the loss (match normalization level of main loss).

**Why it might beat baseline:** The physics Laplacian regularizer was closed, but that tested volumetric smoothness. The Bernoulli constraint is a surface-specific physics constraint that directly targets the metric we care about. It's a different physical mechanism — not gradient smoothness but pressure-velocity coupling consistency. Even as a soft constraint it guides the model toward physically realizable surface pressure distributions.

**Risk:** Medium. The Bernoulli approximation holds for inviscid flow; at high Re or high AoA, viscous effects and separation break it. If the constraint is too strong it could introduce bias toward ideal solutions. Keep `bernoulli_weight` small (0.001–0.01) and treat it as a regularizer, not a hard constraint. The Laplacian loss failure may indicate that physics losses on this dataset add noise rather than signal — this Bernoulli version is more targeted so may succeed where volumetric smoothness failed.

---

## Idea 8 — Test-Time Augmentation via Reynolds Number Bracketing

**Hypothesis:** At test time, running the model at `Re ± δ` and averaging predictions exploits the smoothness of the Re → field mapping to reduce prediction variance without any training changes. Since Re is a scalar global feature broadcast to all nodes, it is trivially modifiable at inference. If the model has learned a locally smooth Re-response function (which it must, given the training data spans a range of Re), averaging predictions at `{Re - δ, Re, Re + δ}` reduces the effective prediction variance by ~3× for the stochastic component and can sharpen the surface pressure estimate. This is a pure inference-time improvement — zero training cost.

**Implementation:**
- At evaluation time:
  ```python
  def predict_tta(model, batch, delta_re_frac=0.05):
      re_idx = feat_idx['log_re']
      preds = []
      for delta in [-delta_re_frac, 0.0, +delta_re_frac]:
          x_aug = batch['x'].clone()
          x_aug[..., re_idx] += delta   # shift log(Re) by ±5%
          preds.append(model(x_aug, batch))
      return torch.stack(preds).mean(0)
  ```
- `delta_re_frac` in {0.02, 0.05, 0.10} — try 0.05 first (5% Re shift in log-space).
- Only apply TTA during validation/test scoring, not during training.
- Cost: 3× inference time per sample — acceptable for evaluation.

**Why it might beat baseline:** TTA is consistently one of the highest-ROI techniques in Kaggle competitions and scientific ML: it reduces variance without any training risk. The Re-smoothness assumption is validated by the presence of `val_re_rand` split — the dataset explicitly tests Re generalization, implying smooth Re dependence. This is also the cheapest experiment possible: zero training change, purely a scoring modification.

**Risk:** Very low. If the Re-response function has sharp transitions (unlikely for smooth CFD), TTA will smear predictions. The `delta_re_frac` tuning is the only hyperparameter. Can be validated cheaply by comparing TTA vs non-TTA predictions on the validation set.
