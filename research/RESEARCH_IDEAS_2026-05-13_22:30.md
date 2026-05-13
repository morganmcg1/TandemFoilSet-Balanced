# SENPAI Research Ideas — 2026-05-13 22:30

**Context**: Generated after the 20th compound win (PR #2519, fixed attention τ=√2 × default, −3.68% val / −4.38% test) and a complete washout of the subsequent 6-experiment wave (#2574 attn-temp-√3, #2575 latent-mixup, #2576 per-head-τ, #2517 Q-bias, #2518 β2=0.99, #2511 input-gate, #2515 per-block-LS-lr). The meta-diagnosis: additive-scale and optimizer-config axes are dry post-#2519. The sharper attention absorbed the residual error signal those mechanisms were compensating for. The OOD splits (camber_rc, camber_cruise) have plateaued while single_in_dist keeps improving.

**Closed axes (do NOT re-propose)**: all optimizer-config variations, layerscale init/lr/decoupled variants, per-head τ multiplier, latent/slice-token mixup, y-reflection TTA, Q-projection bias, β2=0.99, input-feature gate, per-block LS lr. Full closed-axes table in CURRENT_RESEARCH_STATE.md.

**Primary target metric**: `val_avg/mae_surf_p` = equal-weight mean surface-pressure MAE across 4 splits: single_in_dist=66.511, camber_rc=68.819, camber_cruise=34.782, re_rand=54.590. OOD improvement (camber_rc, camber_cruise) is highest-priority because most historical wins came from single_in_dist.

---

## Idea 1: Tangent-Plane Feature Augmentation (FOMA-inspired, applied at node-feature level)

**One-line hypothesis**: Augmenting node features with small random displacements in the tangent space of the training manifold, computed per-batch via input Jacobians, will improve OOD robustness by teaching the model to be locally invariant in the directions the data naturally varies.

**Mechanism**: FOMA (First-Order Manifold Augmentation, ICML 2024) showed that sampling from local tangent planes of the training distribution is empirically superior to standard mixup for regression OOD tasks, because it generates perturbations that are on-manifold rather than off-manifold interpolations. Unlike slice-token mixup (#2575, catastrophic +28.65%), this operates entirely at the 24-dim node-feature level before any representation is built — there is no slice-token destruction. Concretely: for each training sample, compute a finite-difference tangent estimate by perturbing the 24-dim input x with ε ~ N(0, σ²I), project this perturbation to remove the component in the loss gradient direction, and add the result to x. The model is then trained on both the original and the augmented version. This forces the learned representation to be locally stable in the directions orthogonal to the loss — exactly the directions that camber variation occupies.

**Minimal implementation sketch**:
```python
# In train.py, inside the training loop, after loading batch (x, y, is_surface, mask)
# Apply tangent-plane augmentation with prob p_aug=0.5 per batch

if self.training and torch.rand(1).item() < p_aug:
    # Estimate tangent direction via finite difference on input features
    sigma = 0.01  # small perturbation in feature space
    noise = sigma * torch.randn_like(x)
    
    # Forward pass with noise to get directional loss gradient estimate
    with torch.no_grad():
        pred_perturbed = model(x + noise, ...)
        # Project noise to remove loss-gradient component
        loss_perturbed = criterion(pred_perturbed, y, is_surface, mask)
    
    # Simple implementation: use the noise directly (tangent approx without projection)
    # Full FOMA: project out gradient component using autograd
    tangent_x = x + noise  # first-order approx
    
    # Train on augmented version (can double-batch or replace original)
    x = torch.cat([x, tangent_x], dim=0)
    y = torch.cat([y, y], dim=0)
    is_surface = torch.cat([is_surface, is_surface], dim=0)
    mask = torch.cat([mask, mask], dim=0)
```

Simpler approximation (no autograd overhead): add Gaussian noise directly to the 24-dim input features during training (this is the tangent approximation without the projection step). This is a 2-line change: add a `feature_noise_std` hyperparameter and apply `x = x + torch.randn_like(x) * feature_noise_std` during training. The projection step can be added as a follow-up.

Start with `feature_noise_std = 0.005` (≈0.5% of typical feature magnitude). Since the input features include log(Re), AoA, NACA codes, gap, stagger — all of which are the axes of OOD variation — even small Gaussian noise on these features may teach the model the local invariance structure.

**Pre-registered prediction**: val_avg improves by 1-3% primarily through camber_rc and camber_cruise splits. If in-distribution (single_in_dist) degrades while OOD improves, that is also a win because it signals the mechanism is working. Failure mode: noise std too large destroys the AoA/NACA signal entirely (watch for catastrophic single_in_dist regression > 10%).

**Citation**: FOMA (First-Order Manifold Augmentation), ICML 2024. https://proceedings.mlr.press/v235/fast24a.html — showed tangent-plane augmentation outperforms mixup on OOD regression benchmarks; key insight is that the tangent space approximation needs to be local (σ small) to avoid off-manifold samples.

---

## Idea 2: Spectral Frequency Contrastive Loss on Slice Tokens

**One-line hypothesis**: Adding an auxiliary loss that enforces frequency-domain consistency between the slice-token representations of in-distribution and OOD-like augmented inputs will teach the model to extract physically invariant features that generalize to unseen camber configurations.

**Mechanism**: Inspired by iMOOE (ICLR 2026, Li et al.), which introduced a frequency-enriched invariant objective to achieve zero-shot OOD generalization for PDE dynamics. The core insight: physical systems that look different in the spatial domain (different camber profiles) often look similar in the frequency domain (same dominant aerodynamic modes). The iMOOE approach enforces two-fold invariance: the operator-level representations and the compositional (multi-scale) representations should be consistent across domain shifts. Adapted here: after the slice-token aggregation, compute the FFT of each slice token's feature vector across the feature dimension, and apply a contrastive loss that pulls together the low-frequency components of slice tokens from geometrically similar but domain-different configurations.

**Minimal implementation sketch**:
```python
# In TransolverBlock or at the model level, after computing slice_token
# slice_token shape: (B, n_heads, slice_num, dim_head) = (B, 4, 64, 32)

# Compute FFT across feature dimension for each slice token
slice_fft = torch.fft.rfft(slice_token, dim=-1, norm='ortho')
slice_freq_mag = slice_fft.abs()  # (B, 4, 64, 17)

# Low-frequency components (first n_low=4 components)
n_low = 4
slice_low_freq = slice_freq_mag[..., :n_low]  # (B, 4, 64, 4)

# Augmented version: add small AoA perturbation to input and re-run
# (or use a different sample from same batch with same NACA code)
# Simple approximation: use pairs within the batch that share NACA code

# Contrastive loss: minimize L2 distance between low-freq components
# of same-NACA pairs, maximize distance between different-NACA pairs
# (cosine similarity version is simpler)

freq_loss = F.mse_loss(
    slice_low_freq[0::2],  # even-indexed samples
    slice_low_freq[1::2]   # odd-indexed samples from same domain
)

total_loss = main_loss + lambda_freq * freq_loss  # lambda_freq = 0.1
```

Simpler version without explicit pairing: add an L2 regularization term that penalizes high-frequency components in the slice-token features, enforcing smooth (low-frequency) representations. This is a single-line auxiliary loss: `aux_loss = slice_freq_mag[..., n_low:].pow(2).mean()`. This biases the representation toward capturing dominant physical modes rather than camber-specific high-frequency artifacts.

**Pre-registered prediction**: camber_rc and camber_cruise improve by 2-5%; single_in_dist approximately stable or slightly worse. The mechanism is OOD-targeted regularization, not capacity improvement, so in-dist should be minimally affected. Failure mode: if `n_low=4` is too aggressive, the model loses the ability to distinguish between flow regimes entirely.

**Citation**: iMOOE (Physics-Guided Invariant Multiphysics Operator for OOD Generalization), ICLR 2026. Described in the ICLR 2026 accepted papers listing — introduces frequency-enriched invariant objectives for zero-shot OOD generalization in PDE settings with operator-level and compositional invariance.

---

## Idea 3: Thin-Airfoil Theory Auxiliary Residual Loss

**One-line hypothesis**: Adding an auxiliary loss that penalizes deviation from thin-airfoil theory pressure predictions (analytic potential-flow solution) on the surface nodes will provide a free physics prior that constrains camber extrapolation.

**Mechanism**: Inspired by the multiphysics training line (ICLR 2026 under review), which showed that joint training with simplified physics forms as an auxiliary task improves OOD generalization for parameter shifts. Thin-airfoil theory (Joukowski/thin-airfoil) predicts the leading-order surface pressure coefficient Cp from camber and AoA via an analytic closed form: `Cp = 2*(AoA - dc/dx)` where `dc/dx` is the local camber slope. This is cheap to compute analytically from the input features (which include NACA codes + AoA) and provides a *camber-aware* signal that generalizes across all camber values, not just trained ones. The model sees `camber=6-8 (raceCar)` during training but thin-airfoil theory gives it a systematic extrapolation anchor.

**Minimal implementation sketch**:
```python
# Thin-airfoil approximation for NACA 4-digit series:
# NACA MPXX: M = max camber / 100, P = camber position / 10
# Cp_tat(x) = 2*(AoA - camber_slope(x)) for thin airfoil, inviscid

def thin_airfoil_cp(x_coord, m_camber, p_position, aoa_rad):
    """
    Thin-airfoil theory Cp estimate.
    x_coord: normalized chord position [0, 1]
    m_camber: max camber fraction (NACA M/100)
    p_position: camber position fraction (NACA P/10)
    Returns: Cp_TAT (scalar or array, valid where inviscid assumption holds)
    """
    # Camber line derivative from NACA definition
    # For x < p_position: dc/dx = 2*m/p^2 * (p - x)
    # For x >= p_position: dc/dx = -2*m/(1-p)^2 * (p - x)
    # (standard NACA 4-digit camber line)
    dc_dx = torch.where(
        x_coord < p_position,
        2 * m_camber / (p_position**2 + 1e-8) * (p_position - x_coord),
        -2 * m_camber / ((1 - p_position)**2 + 1e-8) * (p_position - x_coord)
    )
    return 2.0 * (aoa_rad - dc_dx)

# In loss computation, for surface nodes only:
# x_features contains (x, y, arc_length, ..., log_Re, aoa, naca_m, naca_p, ...)
# Parse from the 24-dim feature vector using known column layout

cp_tat = thin_airfoil_cp(x_coord, m_camber, p_position, aoa_rad)
cp_pred = pred[is_surface, 2]  # channel 2 is pressure p
# Scale cp_tat to match the model's unnormalized pressure units (y-scaler)
aux_physics_loss = F.l1_loss(cp_pred, cp_tat_rescaled)

total_loss = main_loss + lambda_tat * aux_physics_loss  # lambda_tat=0.01 start
```

Key implementation note: the exact column indices for naca_m, naca_p, aoa in the 24-dim feature vector must be verified from `data/` (read program.md for the exact 24-dim feature layout). The thin-airfoil approximation breaks down near the trailing edge and in the boundary layer — use only on nodes with x_coord ∈ [0.1, 0.85] to avoid these regions.

**Pre-registered prediction**: camber_rc and camber_cruise improve 2-4% since the physics anchor is directly tied to the camber variation axis. single_in_dist may be slightly hurt (model is regularized toward an imperfect physics prior). The most important observable: does the camber_rc–camber_cruise split gap narrow (both ~50 now, thin-airfoil should help both roughly equally)?

**Citation**: Multiphysics Training for Neural PDE Surrogates, ICLR 2026 under review. Also related: FEA-Net (ICLR 2020) for physics-constraint auxiliary losses; Raissi et al. PINN (2019) for the general paradigm of physics residuals as auxiliary losses.

---

## Idea 4: Geometry-Aware Heat-Diffusion Positional Encoding

**One-line hypothesis**: Replacing the current FourierCoordEnc (which only sees x,y coordinates independently) with heat-diffusion spectral embeddings that encode mesh geodesic structure will give the model a geometry-aware positional signal that extrapolates better to unseen airfoil shapes.

**Mechanism**: Farazi & Wang (arXiv 2411.00164, Oct 2024) showed that heat diffusion structural embeddings — computed from the eigenvectors of the mesh Laplacian — capture geodesic geometry that Euclidean Fourier features fundamentally cannot represent. The key insight: for CFD meshes, the geodesic distance along the airfoil surface encodes the physical boundary condition in a way that (x,y) Euclidean coordinates do not. Two points at the same (x,y) might be on the upper and lower surface, but they have very different geodesic distances from the leading edge. The current FourierCoordEnc maps (x,y)→24-dim, which conflates these. Heat diffusion PE: compute the diffusion matrix `H_t = exp(-t * L)` where L is the normalized graph Laplacian of the mesh, then use the diagonal of `H_t` as a positional embedding. This gives each node a signature that depends on its local connectivity, not just its (x,y).

**Minimal implementation sketch**:
```python
# Precompute: for each sample at load time (or cached), compute:
# 1. Build sparse graph Laplacian from mesh connectivity
# 2. Compute top-K eigenvectors of L (K=16 eigenvectors)
# 3. Use eigenvector values at each node as positional embedding

# At training time, replace dims 0-1 in x (currently x,y coords that
# FourierCoordEnc processes) with the heat-diffusion eigenvector values

# Simple approximation (no precomputation needed): use the arc-length
# feature (already in dims 2-3) + Manhattan distance from leading edge
# computed from (x, y). This is a cheap proxy for geodesic distance.

# Full implementation:
# In data loading (or as a collation hook in train.py):
def compute_laplacian_pe(pos, edge_index, n_eig=16, t=1.0):
    """
    pos: (N, 2) node positions
    edge_index: (2, E) edge connectivity from mesh
    Returns: (N, n_eig) heat diffusion PE
    """
    # Build Laplacian, compute eigenvectors
    # Use scipy.sparse.linalg.eigsh for efficiency
    L = build_normalized_laplacian(pos, edge_index)
    eigenvalues, eigenvectors = eigsh(L, k=n_eig, which='SM')
    H_t = eigenvectors * np.exp(-t * eigenvalues)[None, :]  # (N, n_eig)
    return torch.tensor(H_t, dtype=torch.float32)

# In train.py model forward: concatenate laplacian_pe to node features
# (alongside or replacing current FourierCoordEnc output)
```

Important constraint from `program.md`: data loaders are read-only. The precomputation must be done inside `train.py` by reading the mesh connectivity from the loaded batch (which includes `x` with coordinate features). The most practical approach: compute a simple spectral proxy from the batch data using the coordinate and surface-flag features that are already present, without requiring mesh edge connectivity.

**Pre-registered prediction**: camber_rc and camber_cruise improve 1-3% since the diffusion PE encodes surface topology that is invariant to camber magnitude. Failure mode: if mesh connectivity is not available in the loaded batch (only node features are loaded), this approach needs a fallback using coordinate-based graph construction, which adds latency.

**Citation**: Geometry-Aware 3D Mesh Transformers via Heat Diffusion Structural Embeddings, Farazi & Wang, arXiv:2411.00164, Oct 2024. https://arxiv.org/abs/2411.00164 — showed heat diffusion PE outperforms Fourier PE and sinusoidal PE on mesh-based 3D shape analysis tasks.

---

## Idea 5: Mixture-of-Experts Slice-to-Output Projection (Regime-Routing MoE)

**One-line hypothesis**: Replacing the single dense `out_project` linear layer (which maps slice tokens back to node features) with a sparse mixture-of-experts layer that routes by physical regime will allow the model to specialize different expert sub-networks for different airfoil geometries, directly addressing the camber OOD gap.

**Mechanism**: MoE-POT (NeurIPS 2025) demonstrated that sparse MoE with 16 routed experts (4 activated) + 2 shared experts reduced zero-shot error by 40% on PDE operator pretraining. The key architectural insight for this setting: the `out_project` layer (dim 128 → 128) is the single point where all physical regimes must be handled by the same weights. If we replace it with 4 experts (2 active per token) routed by a lightweight router, different experts can specialize for raceCar-high-camber vs. cruise-low-camber configurations. The routing signal comes from the slice-token content itself (which encodes the AoA and geometry features from the input). This is a small, targeted MoE — not a full-model replacement.

**Minimal implementation sketch**:
```python
class SliceMoEProjection(nn.Module):
    """
    Sparse MoE replacement for the single out_project linear in PhysicsAttention.
    n_experts=4, top_k=2 (2 experts active per token).
    """
    def __init__(self, in_dim=128, out_dim=128, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Expert projections
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(n_experts)
        ])
        # Router: small linear from slice token to expert logits
        self.router = nn.Linear(in_dim, n_experts, bias=False)
        # Load balancing auxiliary loss weight
        self.aux_loss_weight = 0.01
    
    def forward(self, x):
        # x: (..., in_dim)
        router_logits = self.router(x)  # (..., n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute output as weighted sum of top-k expert outputs
        out = sum(
            top_k_probs[..., i:i+1] * self.experts[top_k_indices[..., i]](x)
            # Note: above indexing simplified; proper impl uses scatter
            for i in range(self.top_k)
        )
        
        # Load balancing loss (prevent expert collapse)
        # Return as tuple (out, aux_loss) and accumulate in training loop
        mean_router_probs = router_probs.mean(dim=(0, 1, 2))  # (n_experts,)
        aux_loss = self.aux_loss_weight * (n_experts * mean_router_probs * mean_router_probs).sum()
        
        return out, aux_loss

# Replace in PhysicsAttention.__init__:
# self.out_project = nn.Linear(dim_head, dim_head, ...)
# → self.out_project = SliceMoEProjection(dim_head, dim_head, n_experts=4, top_k=2)
```

Parameter budget: 4 experts × 128×128 = 65,536 extra params + router 128×4 = 512 params = ~66K increase from ~893K to ~959K. This is within acceptable range (no architectural violation). Add the aux load-balancing loss to avoid expert collapse. Expect expert specialization to emerge: one expert for leading edge, one for trailing edge, one for surface upper, one for lower — or by camber regime.

**Pre-registered prediction**: camber_rc and camber_cruise improve 2-5%; single_in_dist approximately stable. Expert routing entropy should be non-uniform (check that not all tokens go to expert 0). Failure mode: with only 12 epochs, the router may not diversify (similar to the per-channel γ finding in #2488 — too many params to route cleanly). If load balancing shows one expert getting >80% of tokens, increase aux_loss_weight to 0.1.

**Citation**: MoE-POT (Mixture-of-Experts for Physics Operator Training), NeurIPS 2025 AI4Science Workshop. Demonstrated 40% zero-shot error reduction with 16-expert sparse MoE on PDE pre-training tasks.

---

## Idea 6: Geometry-Conditioning via Signed Distance Function Features

**One-line hypothesis**: Augmenting the 24-dim input features with per-node signed distance function (SDF) values — distances to both airfoil surfaces and to the domain boundary — will give the model an explicit geometry-aware signal that extends to unseen camber shapes without requiring training examples.

**Mechanism**: SDF features are a standard geometric primitive that encode the local geometric context in a way that is both differentiable and inherently invariant to how the shape is parameterized (e.g., which NACA code describes it). The key advantage over NACA parameters: the NACA encoding in the current 24-dim features is a 2-number description of shape (M and P), which requires learning that "M=6 and M=8 have similar leading-edge curvature" purely from data. SDF features directly encode the local geometric relationship: "this node is 0.01 chord lengths from the upper surface, 0.05 from the lower surface". This shifts the representation from parametric (NACA codes) to geometric (local topology), enabling the model to generalize to unseen parametric configurations that share local geometric properties with trained ones.

**Minimal implementation sketch**:
```python
# Compute SDF features from the node coordinates already in x
# Node features x: (N, 24) includes x-coord, y-coord, arc-length, surface_flag, ...

# For each batch sample, compute SDF to the two airfoil surfaces
# using the surface nodes (identified by is_surface and surface_flag features)

def compute_sdf_features(x, is_surface, n_sdf_channels=4):
    """
    x: (N, 24) node features; x[:, 0:2] = (x_coord, y_coord)
    is_surface: (N,) bool mask for surface nodes
    Returns: (N, n_sdf_channels) SDF features
    
    Features:
    0: distance to nearest surface node (foil 1 or 2)
    1: distance to foil 1 surface specifically
    2: distance to foil 2 surface specifically
    3: signed distance (negative inside, positive outside)
    """
    coords = x[:, 0:2]  # (N, 2)
    surface_coords = coords[is_surface]  # (S, 2)
    
    # Distance from each node to nearest surface point
    # Using efficient pairwise distance (approximate via batch kNN)
    dists = torch.cdist(coords.unsqueeze(0), surface_coords.unsqueeze(0))[0]
    min_dist = dists.min(dim=-1).values  # (N,)
    
    # Split by surface flag to get foil-specific distances
    foil1_mask = is_surface & (x[is_surface, surface_flag_col] == 0)  # adjust col idx
    foil2_mask = is_surface & (x[is_surface, surface_flag_col] == 1)
    
    dist_foil1 = torch.cdist(coords.unsqueeze(0), coords[foil1_mask].unsqueeze(0))[0].min(-1).values
    dist_foil2 = torch.cdist(coords.unsqueeze(0), coords[foil2_mask].unsqueeze(0))[0].min(-1).values
    
    return torch.stack([min_dist, dist_foil1, dist_foil2, dist_foil1 - dist_foil2], dim=-1)

# In train.py model forward, before the input projection:
sdf_feats = compute_sdf_features(x, is_surface)  # (N, 4)
x_augmented = torch.cat([x, sdf_feats], dim=-1)   # (N, 28)
# Input projection layer n_hidden→128 needs input_dim=28 instead of 24
```

This requires changing `input_proj = nn.Linear(24, n_hidden)` to `nn.Linear(28, n_hidden)` — a minimal architectural change that does not touch any downstream components. The SDF computation is done in the forward pass using the existing coordinate features; no new data loading is required.

Note: `torch.cdist` on variable-size meshes (74K–242K nodes) may be expensive. A practical approximation: precompute SDF using the surface nodes as anchor points and use a k-nearest-neighbor (k=5) distance rather than full pairwise. Under the 30-min wall-clock cap, this needs to be benchmarked.

**Pre-registered prediction**: camber_rc and camber_cruise improve 2-4%; single_in_dist minimally affected. The geometric signal is invariant to NACA parameterization, so the model should handle camber extrapolation better. Failure mode: if the SDF computation adds >5 min per epoch, this is not feasible under the 30-min cap. Precompute and cache SDF as part of the batch.

**Citation**: Implicit Neural Representations / SDF literature (Park et al. DeepSDF, CVPR 2019) for the geometric representation principle. Also: physics-agnostic pretraining (Zhang et al., ICLR 2026) for the insight that geometry-only representations improve operator generalization.

---

## Idea 7: Per-Block Attention Temperature Annealing Schedule

**One-line hypothesis**: Instead of a fixed √2 attention temperature across all blocks and all training epochs, using a curriculum that starts sharper (τ=√3) and anneals toward √2 over training will find a better sharpening attractor than any fixed scalar can achieve.

**Mechanism**: The √2 win (#2519) established that the attention sharpening axis is real. The failure of √3 (#2574) established that the optimum is between √2 and √3, not beyond √3. A curriculum approach that starts at √3 and anneals toward √2 (or searches the interval) gives the model the benefit of high sharpening early (when representations are noisy and need strong filtering) while settling at the √2 attractor that has proven stable. This is inspired by temperature annealing in simulated annealing and by the observation that in many attention-based models, the optimal attention temperature is training-stage-dependent (early training prefers softer, late training prefers sharper — or vice versa).

**Minimal implementation sketch**:
```python
# In PhysicsAttention, replace the fixed scalar with a schedule-updated value
# sharper_scale is not a learnable param — it is a hyperparameter updated by the scheduler

class PhysicsAttention(nn.Module):
    def __init__(self, ..., attn_temp_init=3.0**0.5, attn_temp_final=2.0**0.5):
        ...
        self.register_buffer('attn_temp', torch.tensor(attn_temp_init))
        self.attn_temp_init = attn_temp_init
        self.attn_temp_final = attn_temp_final
    
    def update_temperature(self, frac):
        """Call with frac = current_epoch / max_epochs"""
        # Linear annealing from init to final
        current_temp = self.attn_temp_init + frac * (self.attn_temp_final - self.attn_temp_init)
        self.attn_temp.fill_(current_temp)
    
    def forward(self, x):
        ...
        sharper_scale = self.attn_temp / math.sqrt(self.dim_head)
        out_slice = F.scaled_dot_product_attention(q, k, v, scale=sharper_scale, ...)

# In training loop (train.py):
for epoch in range(n_epochs):
    frac = epoch / n_epochs
    for block in model.blocks:
        block.attn.update_temperature(frac)
    ...
```

Three variants worth trying as arms:
- Arm A: √3 → √2 linear annealing over 12 epochs
- Arm B: √2 → √3 linear annealing (inverse schedule — start sharp, get sharper)
- Arm C: Cosine schedule peak at √3 at epoch 6, return to √2 at epoch 12

**Pre-registered prediction**: at least one arm beats the fixed √2 baseline. The √3→√2 schedule (Arm A) is the primary bet: early high-sharpening forces the model to commit to coarse structure (which is more camber-invariant), then √2 at convergence allows fine-tuning. Failure mode: if the model's optimization trajectory is strongly dependent on early-epoch attention temperature, curriculum scheduling may produce unstable learning dynamics.

**Citation**: Temperature annealing in Boltzmann machines (Hinton, 1986) for the general principle; attention temperature scheduling has been used in knowledge distillation (Hinton et al., 2015) and transformer pre-training (Press et al., 2020). No direct PDE surrogate citation — this is an extrapolation of the #2519 finding.

---

## Summary and Top 6 Assignments

### Ranking by expected OOD impact + implementation confidence

| Rank | Idea | Expected val_avg gain | OOD targeting | Impl. risk | Taste scores (Mech / State-value / Exec) |
|------|------|----------------------|---------------|------------|------------------------------------------|
| 1 | Idea 3: Thin-Airfoil Auxiliary Loss | 2-4% | Direct (camber axis) | Low (analytic formula) | 4 / 4 / 4 |
| 2 | Idea 1: Tangent-Plane Feature Noise | 1-3% | Indirect (all OOD) | Very low (2-line change) | 3 / 4 / 4 |
| 3 | Idea 5: SliceMoE Projection | 2-5% | Direct (regime routing) | Medium (~66K new params) | 3 / 3 / 3 |
| 4 | Idea 7: Attention Temp Annealing | 1-2% | Indirect (attractor) | Very low (schedule only) | 3 / 3 / 4 |
| 5 | Idea 2: Spectral Freq Contrastive Loss | 2-5% | Direct (invariant repr) | Medium (FFT + pairing) | 3 / 3 / 3 |
| 6 | Idea 6: SDF Features | 2-4% | Direct (geometry) | Medium (cdist overhead) | 3 / 3 / 2 |
| 7 | Idea 4: Heat Diffusion PE | 1-3% | Direct (mesh topology) | High (Laplacian eigvec) | 3 / 3 / 2 |

### Top 6 for immediate assignment:

1. **Idea 3 (Thin-Airfoil Aux Loss)** — highest mechanistic grounding, directly targets camber OOD axis, analytic computation (no new compute overhead), strong prior in multiphysics training literature
2. **Idea 1 (Tangent-Plane Feature Noise)** — simplest possible implementation (add Gaussian noise to 24-dim inputs during training), high OOD upside, zero-cost at inference, FOMA-backed
3. **Idea 7 (Attn Temp Annealing)** — zero-risk extension of the #2519 win, 3-arm test covers the full interval, very cheap
4. **Idea 5 (SliceMoE Projection)** — bold architectural bet, strong NeurIPS 2025 evidence from MoE-POT, targeted at exactly the regime-routing bottleneck
5. **Idea 2 (Spectral Freq Contrastive Loss)** — novel to this programme, targets invariant representation learning, iMOOE evidence from ICLR 2026
6. **Idea 6 (SDF Features)** — geometry-aware representation, extends input feature space in a physically motivated direction, risk is compute overhead from cdist

Idea 4 (Heat Diffusion PE) is ranked 7th due to implementation risk (requires mesh connectivity in the batch) and can be assigned once Idea 6 resolves whether geometry-aware features help at all.

---

## Research State Update

**Current best explanation for OOD plateau**: The model's internal representation uses the NACA parametric features (M, P codes) as a proxy for geometry, but has not learned a representation that is locally invariant to camber variation. The sharper attention (√2) improved generalization uniformly because it forces the slice tokens to be more selective (lower entropy routing), reducing the influence of high-frequency artifacts. But this is a structural fix, not a representational fix — the underlying feature space still lacks explicit camber-invariant signal.

**What would falsify this**: if Idea 3 (thin-airfoil auxiliary loss) fails to improve camber_rc/camber_cruise while degrading single_in_dist, it suggests the bottleneck is NOT the physics prior but rather the representational capacity (Idea 5 path). If Idea 1 (input noise) fails to improve OOD, it suggests the bottleneck is NOT local invariance but rather something deeper in the architecture.

**Next discriminating experiment**: Idea 3 (Thin-Airfoil Aux Loss), because it directly injects camber-scale physics knowledge with zero architectural risk. A positive result (even 1% camber_rc improvement) would confirm the physics-prior hypothesis and justify larger investments (full PINN residual, multiphysics training). A null result narrows to representation capacity (Ideas 5, 2).

**Stop condition for this direction**: if all 6 ideas fail to improve camber_rc or camber_cruise by more than 0.5%, the OOD gap is likely a data density problem that cannot be solved by regularization or physics priors — pivot to data augmentation via synthetic CFD samples or geometry interpolation.
