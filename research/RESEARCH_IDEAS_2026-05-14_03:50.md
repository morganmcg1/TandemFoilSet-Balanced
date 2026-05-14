# Wave 20 Research Ideas — 2026-05-14 03:50

**Context**: Wave 19 was a complete washout (7/7 PRs regressed). The attention-schedule axis is fully closed. Geo aux head, Laplacian PE, slice_num variants all failed. The local neighborhood of the current Transolver/L1/Fourier-coords/slice-token config is exhausted. These 8 hypotheses represent a paradigm shift — moving to different abstraction levels rather than incremental tuning.

**Current baseline**: val_avg/mae_surf_p = 55.1595 (PR #2648). Per-split: single_in_dist=60.851 / camber_rc=68.657 / camber_cruise=35.762 / re_rand=55.368.

**Primary OOD gap**: camber_rc = 68.657 is 12.8% worse than single_in_dist despite being geometry interpolation. This is the primary target.

---

## H1 — Relative L1 Loss (Per-Sample Magnitude Normalization)

**Family**: LOSS REFORMULATION

**Mechanism**: Replace the current global-normalize L1 loss with a per-sample relative L1 that normalizes by the ground-truth magnitude within each sample. Currently, high-Re samples with large pressure ranges dominate gradient signal during training. Low-Re (in-distribution) samples contribute proportionally tiny loss gradients. A per-sample relative loss equalizes gradient magnitude across Reynolds number regimes and forces the model to represent relative rather than absolute errors — which is the physically correct objective for a surrogate model that must generalize across Re.

**Implementation outline**:
```python
# In normalized space, after computing pred and y_norm:
# Current loss: l1 = masked_l1(pred, y_norm, mask)
# Proposed: normalize each sample by its ground-truth magnitude
y_phys = y_norm * stats["y_std"] + stats["y_mean"]  # denormalize
pred_phys = pred * stats["y_std"] + stats["y_mean"]

# Per-sample denominator: mean absolute value of GT over valid nodes
denom = (y_phys.abs() * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
        mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
denom = denom.clamp(min=1.0)  # avoid div-by-zero for near-zero samples

rel_loss = ((pred_phys - y_phys).abs() / denom * mask.unsqueeze(-1)).sum() / \
           mask.sum().clamp(min=1)
```
Apply this to surface channel-weighted loss (keep [0.5, 0.5, 2.0] surf-ch-weight) and vol loss. Add a stability floor `denom.clamp(min=1.0)` to protect near-zero samples.

**Why it might break the plateau**:
- The camber_rc failure mode is likely that the model sees large Re-driven gradients from racecar-single and racecar-tandem Part 1/3 samples and under-fits the pressure variation pattern specific to M=6-8 camber (Part 2, held out). Relative L1 removes the Re-scaling bias from gradient weighting.
- The primary metric is MAE in physical units — which is equivalent to absolute L1 after denormalization. But during training, the loss the model minimizes is in normalized space, which does NOT equalize cross-sample gradients. This is a fundamental mismatch between training objective and evaluation objective.
- Strong analogy: in molecular property prediction, normalizing losses by molecule size dramatically improves OOD generalization (Gilmer et al.). The same principle applies here for Re-dependent magnitude variation.

**Risk**: Relative L1 can destabilize training if denominators approach zero. The `clamp(min=1.0)` guard should handle this but needs testing. The per-sample normalization also changes the gradient landscape significantly — may need LR adjustment (try 3e-4 and 5e-4 arms).

**Confidence**: HIGH. This is a mechanistically clean fix to an objective mismatch. The camber_rc OOD gap is consistent with a Re-scale-bias failure mode. Implementation is simple, no new parameters.

**Expected impact**: Potential -5 to -15% on camber_rc and re_rand splits. Small or neutral effect on single_in_dist and camber_cruise (which the model already handles well). Net val_avg improvement if OOD splits respond.

---

## H2 — Sobolev Loss (First-Order Gradient Matching on Surface Pressure)

**Family**: LOSS REFORMULATION

**Mechanism**: Add a Sobolev-type auxiliary loss term that penalizes mismatches in the spatial gradient of surface pressure, in addition to the pointwise L1 loss. Surface pressure gradients encode aerodynamic loading distributions (lift, pitching moment, separation indicators). A model trained only on pointwise L1 can match average pressure accurately while getting the gradient (pressure recovery slope, suction peak sharpness) completely wrong — which matters for both physical accuracy and for generalizing to unseen camber geometries where the pressure gradient pattern changes qualitatively.

**Implementation outline**:
```python
# Surface nodes only; compute approximate spatial gradient via finite differences
# on arc-length coordinate (dim 2-3 of input x: signed arc-length saf)
# Step 1: sort surface nodes by arc-length within each sample
# Step 2: compute finite differences of predicted vs GT pressure on surface

def sobolev_surf_loss(pred, y, x, is_surface, mask, lambda_sob=0.1):
    # pred, y: [B, N, 3]; x: [B, N, 24]; is_surface: [B, N]
    surf_mask = is_surface & mask  # [B, N]
    # Arc-length from x dim 2 (saf_0) — use as approximate 1D coordinate
    s = x[..., 2]  # [B, N] — signed arc-length foil 1 surface parameter
    
    # For each sample: get surface nodes, sort by s, compute dp/ds
    # Use central finite differences on sorted surface nodes
    p_pred = pred[..., 2]  # pressure channel
    p_true = y[..., 2]
    
    # Approximate gradient: difference between adjacent surface nodes sorted by arc-length
    # Sort indices by arc-length within surf_mask
    # ... (implement per-sample sort + diff; standard pattern)
    
    # Sobolev term: L1 on gradient difference
    grad_loss = (dp_pred - dp_true).abs().mean()
    return lambda_sob * grad_loss
```
`lambda_sob` = 0.05 to 0.1. Keep main L1 loss unchanged; add this as auxiliary term. Operates only on surface nodes — no interference with vol loss.

**Why it might break the plateau**:
- Current L1 loss on pressure is global and treats all surface nodes equally. The suction peak and pressure recovery region drive most of the aerodynamic loading. A Sobolev term forces the model to get the shape of the pressure distribution right, not just its mean value.
- For unseen camber geometries (camber_rc), the pressure gradient pattern changes qualitatively with camber. A model trained only on pointwise pressure will struggle with these qualitative shape changes — but a model trained to match pressure gradients will have learned to represent the arc-length-dependent loading pattern.
- Connection to physics: Kutta condition and thin-airfoil theory both operate on pressure gradient at the trailing edge. Matching dp/ds implicitly enforces physical constraints the model has not seen explicitly.

**Risk**: Arc-length coordinate (x dim 2, signed arc-length saf) is available as a raw input feature but may require careful handling for tandem foils where foil 1 and foil 2 surface nodes share the same arc-length range. Need to test whether the foil 1 vs foil 2 surface arc-length coordinates are distinguishable from input features. If not, may need to restrict to foil 1 surface nodes using NACA dim 18-21 to detect tandem vs single.

**Confidence**: MEDIUM-HIGH. The mechanism is sound and the implementation is feasible in 12 epochs. The main uncertainty is whether the arc-length coordinate is clean enough for reliable finite differencing on irregular meshes.

**Expected impact**: Potential -3 to -10% on camber_rc and camber_cruise; may slightly increase single_in_dist as a side effect of stricter surface gradient matching.

---

## H3 — Geometry-Aware Message Passing GNN (Model Class Change)

**Family**: MODEL CLASS CHANGE

**Mechanism**: Replace the Transolver's slice-token attention mechanism with a geometry-aware message passing neural network (GeoMPNN-style) that propagates information along mesh edges. Unlike slice-based pooling, edge-based message passing explicitly encodes local mesh geometry (edge vectors, distances, surface normals) and can learn to propagate pressure information along streamlines and surface boundary layers — the physically correct information pathways for aerodynamic surrogates. GeoMPNN won Best Student Submission at NeurIPS 2024 ML4CFD Competition.

**Implementation outline**:
Keep the same input interface (x: [B, N, 24] → preds: [B, N, 3]) and normalization contract. Replace the internal Transolver architecture with:

```python
class GeoMPNN(nn.Module):
    def __init__(self, in_dim=24, hidden=288, out_dim=3, n_layers=5, k_neighbors=16):
        # Node encoder: x → hidden
        self.node_enc = nn.Linear(in_dim, hidden)
        # Edge encoder: relative position + distance → edge features
        self.edge_enc = nn.Sequential(
            nn.Linear(3, hidden),  # (dx, dz, dist)
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GeoMPLayer(hidden) for _ in range(n_layers)
        ])
        # Output decoder
        self.decoder = nn.Linear(hidden, out_dim)
    
    def forward(self, batch):
        x = batch["x"]  # [B, N, 24]
        # Build KNN graph from node positions x[..., 0:2]
        # Use torch_geometric or manual KNN
        # ...
```
For KNN graph construction, use `torch.cdist` + `torch.topk` on the (x, z) coordinates (dims 0-1) within each sample. k=16 neighbors. This avoids `torch_geometric` dependency if not already installed.

The main risk is memory: for N=242K nodes and k=16 neighbors, the edge tensor is 242K×16 = 3.9M edges per sample, which at batch_size=1 fits comfortably in 96GB VRAM.

**Why it might break the plateau**:
- The Transolver's slice-token mechanism pools global context via learned soft assignments to 64 "slice" tokens. This is a global pooling operation — it cannot represent local mesh topology or surface curvature. For camber OOD generalization, the local curvature and streamline topology change qualitatively with M=6-8 vs M=2-5 camber. A GNN can learn local structure invariances that a global attention mechanism cannot.
- NeurIPS 2024 ML4CFD Competition: GeoMPNN showed strong generalization to unseen geometries, specifically because edge-based message passing encodes local geometry rather than learned global assignments.
- Comparable parameter count: 5-layer GeoMPNN with hidden=288 and k=16 is ~900K-1.2M parameters, within the 2× budget.

**Risk**: GNN training is more expensive per epoch than Transolver due to irregular adjacency structure. May not converge in 12 epochs without careful LR scheduling. KNN graph construction at each forward pass adds significant overhead for N=242K; consider precomputing KNN graphs offline (but data/ is read-only — must do it in train.py's dataloader side).

**Confidence**: MEDIUM. The mechanism is strong and external evidence exists. The main uncertainty is whether 12 epochs is enough for GNN convergence from scratch, and whether KNN graph construction overhead fits within 30-min wallclock.

**Expected impact**: High variance — could be -10 to -20% if GNN converges well, or could be +10% if underfit. Recommend a 6-epoch screening run first.

---

## H4 — SE(2)-Equivariant Transformer for Airfoil Fields (Model Class Change)

**Family**: MODEL CLASS CHANGE

**Mechanism**: Replace the Transolver with an SE(2)-equivariant transformer that is exactly equivariant to 2D rotations and translations of the airfoil geometry. The current Transolver is NOT equivariant — if you rotate the mesh, the model output changes in a non-trivial way because the Fourier coordinate encoding and slice-token assignments are orientation-dependent. Enforcing SE(2)-equivariance means the model learns to decompose the pressure and velocity field into equivariant quantities (force magnitudes) and invariant quantities (flow coefficients), which improves generalization to geometries seen at different orientations (AoA variation) and unseen camber profiles.

**Implementation outline**:
Use the equivariant attention formulation from Liao et al. (2025), "Equivariant Graph Neural Networks for 2D Navier-Stokes" (arxiv 2405.20287):

```python
# Key idea: represent node features as (invariant scalars, equivariant vectors)
# Invariant: [N, d_s] — Re, NACA params, distance features
# Equivariant: [N, d_v, 2] — positions, velocity vectors

# Equivariant message: m_ij = f(||x_i - x_j||, s_i, s_j) * (x_j - x_i)/||x_j - x_i||
# This vector message is automatically equivariant under rotation

# Node update: equivariant aggregation via
# h_i^new = sum_j m_ij  (equivariant part)
# s_i^new = g(sum_j f(...))  (invariant part)
```

Target ~1M params (slightly larger than current 892K). AoA is already in the input features (dims 14, 18) — handle as a rotation conditioning variable. Output Ux, Uy as equivariant 2-vector, p as invariant scalar (matches physical symmetry: velocity transforms under rotation, pressure does not).

**Why it might break the plateau**:
- The camber_rc OOD gap is a geometry generalization problem. SE(2)-equivariant networks have a built-in inductive bias for aerodynamic geometry: they cannot confuse "the airfoil is at AoA=-5°" with "the airfoil has camber M=8°" because they represent these as fundamentally different mathematical quantities. This constraint forces better factorization of geometry vs. flow condition.
- Physical correctness: pressure is a scalar field (SO(2)-invariant), while velocity is a vector field (SO(2)-equivariant). Current Transolver treats all 3 output channels identically. An equivariant architecture with this output decomposition is a strictly more physically correct representation.
- Liao et al. 2025 reported 0.5-1% median relative error on 2D NS pressure/velocity with SE(2)-equivariant GNNs — competitive with specialized CFD surrogates.

**Risk**: SE(2)-equivariant networks require careful handling of the rotation representation. Implementation complexity is high. The signed arc-length feature (dims 2-3) is a surface-intrinsic coordinate that is already pseudo-equivariant; integrating it into an SE(2)-equivariant framework is non-trivial. May need 2-3 student iterations to get right. Recommend starting with a simplified E(2)-equivariant (invariant to rotation but not translation) version first.

**Confidence**: MEDIUM. Strong theoretical motivation and external evidence. High implementation complexity — student must be capable. Suggest assigning to strongest coder.

**Expected impact**: High variance. If implemented correctly, potential -10 to -20% OOD improvement. If implementation has subtle equivariance violations, may perform similarly to Transolver with added complexity.

---

## H5 — Camber-Difficulty Curriculum with Re-Weighted Sampler (CURRICULUM/SAMPLING + OOD-targeted)

**Family**: CURRICULUM / SAMPLING, OOD-targeted

**Mechanism**: Implement a curriculum sampler that starts training on easy samples (low camber, in-distribution Re) and gradually introduces harder samples (high camber, extreme Re) over the 12-epoch budget. The core insight: the current equal-domain sampler treats a raceCar-single sample at Re=100K and a raceCar-tandem M=9 sample at Re=5M as equally weighted — but the model is clearly struggling with the latter (camber_rc val=68.657). A curriculum that starts with low-variance samples allows the model to learn the baseline pressure field topology before being asked to generalize to high-camber, high-Re configurations.

**Implementation outline**:
```python
# Compute per-sample difficulty score from input features
# (available at dataset construction time, no labels needed)
# Difficulty = f(camber, Re, is_tandem)

def compute_difficulty(x_sample):
    # x_sample: [N, 24] (before normalization)
    # NACA foil 1 camber: dim 15 (raw, 0-1 normalized: 0=M=0, 1=M=9)
    # log(Re): dim 13
    camber1 = x_sample[0, 15]  # same for all nodes in sample
    re_val = x_sample[0, 13]   # log(Re), same for all nodes
    is_tandem = (x_sample[0, 22] != 0) or (x_sample[0, 23] != 0)  # gap or stagger nonzero
    
    # Simple scalar difficulty score
    difficulty = camber1 * 2.0 + (re_val - 11.0) / 4.0 + (1.0 if is_tandem else 0.0)
    return float(difficulty)

# Curriculum schedule: sample weight for epoch e
def curriculum_weight(difficulty, epoch, total_epochs=12, warmup_epochs=4):
    if epoch < warmup_epochs:
        # Strongly downweight hard samples in first warmup_epochs
        alpha = epoch / warmup_epochs  # 0→1 linearly
        weight = 1.0 / (1.0 + difficulty * (1.0 - alpha) * 3.0)
    else:
        # After warmup: uniform weights (standard training)
        weight = 1.0
    return weight

# In DataLoader: rebuild sample_weights each epoch
# Pass updated weights to WeightedRandomSampler
```

This requires modifying the training loop to rebuild the sampler at each epoch start, which is done in `train.py` (modifiable). Keep the equal-domain balance from `load_data()` sample_weights as a baseline; multiply by curriculum weights on top.

**Why it might break the plateau**:
- The camber_rc OOD gap is not primarily a data density gap (closed in Wave 18, PR #2391). It is likely a representation capacity gap: the model's slice-token dispatch was trained on samples where the "hard" camber configurations appear at random during training — the model's representations are contaminated by easy-sample gradient signal during early training when representations are being formed.
- Curriculum learning is well-validated in NLP and CV settings for improving generalization to rare or hard classes. In CFD surrogates, the analog is unseen geometry configurations.
- The first 4 epochs (warmup) are where the slice-token dispatch topology forms (as established by the attention-temperature mechanism from PR #2648). Controlling what the model sees during this critical window could qualitatively change which representations get formed.

**Risk**: The difficulty score is heuristic — camber is the main axis but Re and tandem/single also matter. Getting the difficulty ordering wrong could hurt more than help. The 12-epoch budget means the full-uniform phase only gets 8 epochs — may underfit. Recommend testing with warmup_epochs=3 and warmup_epochs=5 as two arms.

**Confidence**: MEDIUM-HIGH. Curriculum learning has strong external evidence in multiple domains. The specific failure mode (camber OOD) is consistent with a curriculum diagnosis. Implementation is simple. Main uncertainty is difficulty score design.

**Expected impact**: Potential -5 to -12% on camber_rc and re_rand. May slightly worsen single_in_dist if curriculum suppresses racecar-single samples too aggressively during warmup. Net val_avg effect depends on OOD response magnitude.

---

## H6 — Camber-Conditional Normalization via Scale-Shift Conditioning (OOD-targeted)

**Family**: OOD-targeted, ARCHITECTURE

**Mechanism**: Apply geometry-conditional feature normalization (similar to AdaIN/FiLM but conditioned only on the NACA camber parameter, not the full geometry vector). The current architecture applies the same learned LayerNorm to all samples regardless of airfoil geometry. A camber-conditional scale-shift allows the model to dynamically adjust its feature normalization based on the front foil camber parameter — which is known to be the primary OOD axis (M=6-8 held out). This is distinct from closed FiLM conditioning (PR #2453, which conditioned on the full geometry + Re vector and helped ID while hurting OOD) — here we condition specifically and only on camber to force geometry-aware normalization without Re entanglement.

**Implementation outline**:
```python
# Instead of full FiLM (which was closed), use camber-only conditioning
# Camber signal: extract mean of dim 15 (NACA foil 1 camber) from input x

class CamberCondLayerNorm(nn.Module):
    def __init__(self, d_model, cond_dim=8):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        # Condition only on camber (1D scalar → cond_dim embedding → scale+shift)
        self.cond_net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, 2 * d_model)  # scale and shift
        )
        # Initialize to identity: zero shift, unit scale
        nn.init.zeros_(self.cond_net[-1].weight)
        nn.init.zeros_(self.cond_net[-1].bias)
        # bias for scale = 1.0
        self.cond_net[-1].bias.data[:d_model] = 1.0
    
    def forward(self, x, camber):
        # x: [B, N, d_model]; camber: [B, 1] — single scalar per sample
        x_norm = self.ln(x)
        gamma_beta = self.cond_net(camber.unsqueeze(-1))  # [B, 1, 2*d_model]
        gamma = gamma_beta[..., :d_model]
        beta = gamma_beta[..., d_model:]
        return gamma * x_norm + beta

# In Transolver attention layers: replace standard LayerNorm with CamberCondLayerNorm
# Pass camber = x[:, 0, 15] (constant across all nodes in a sample)
# Parameter overhead: ~5 * (2 * 288 * 8 + 8 + 576) = ~30K params (negligible)
```

Apply this to the pre-attention LayerNorm in each of the 5 Transolver layers. Keep existing LayerScale γ=0.1 unchanged. Initialize to identity to not disturb baseline.

**Why it might break the plateau**:
- Camber_rc failure is specifically: the held-out camber M=6-8 configurations have a qualitatively different pressure distribution (higher suction peak, larger adverse pressure gradient). The model's internal representations are normalized under a distribution that never saw M=6-8 during training. Camber-conditional normalization allows each layer to dynamically rescale its feature distribution to match the expected statistics for a given camber level — interpolating into M=6-8 from the M=2-5 and M=9 training examples.
- This is distinct from FiLM (PR #2453) because: (a) we condition only on camber, not Re or full geometry; (b) we apply it to LayerNorm gamma/beta (multiplicative normalization) rather than post-activation (which FiLM did); (c) we initialize to identity to ensure the baseline trajectory is not disturbed.
- Conditional normalization is a core technique in domain adaptation (AdaIN for style transfer, SPADE for spatially-adaptive normalization) — both address the same problem of adapting feature statistics to unseen input conditions.

**Risk**: If the camber signal (x dim 15) is already well-represented in the Fourier coordinate encoding and slice-token dispatch, this adds conditioning that is redundant with existing information. The identity initialization makes failure mode benign (worst case: learns identity, equivalent to standard LayerNorm). The main risk is that camber-only conditioning is too weak a signal to produce meaningful normalization shifts.

**Confidence**: MEDIUM. The camber-OOD hypothesis is specific and well-motivated. The main uncertainty is whether LayerNorm conditioning is the right intervention given that FiLM (a similar idea) was closed. However, the conditioned signal scope and application point are both different from the closed FiLM experiment.

**Expected impact**: Potential -3 to -10% on camber_rc. Neutral to small effect on other splits. Low risk due to identity initialization.

---

## H7 — Self-Supervised Pre-Training on Masked Node Prediction (SSL Pretraining)

**Family**: SELF-SUPERVISED PRETRAINING

**Mechanism**: Pre-train the Transolver encoder on a masked node prediction task using all training data (no labels required for pretraining) before fine-tuning on the supervised pressure/velocity prediction task. The pre-training task: randomly mask 30% of mesh nodes' position features (dims 0-1, x/z coordinates) and train the encoder to reconstruct them from context. This forces the encoder to learn a rich representation of local mesh topology and global geometry context — which is exactly the information needed to generalize to unseen camber configurations.

**Implementation outline**:
Phase 1 (pre-training, epochs 1-4 of 12):
```python
class MaskedNodePretrainer(nn.Module):
    def __init__(self, base_encoder, d_model=288, in_dim=24):
        self.encoder = base_encoder
        self.reconstruction_head = nn.Linear(d_model, 2)  # reconstruct (x, z)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))  # learnable mask token
    
    def forward(self, x, mask_ratio=0.30):
        # x: [B, N, 24]
        B, N, D = x.shape
        # Sample random nodes to mask
        rand = torch.rand(B, N)
        node_mask = rand < mask_ratio  # True = masked
        
        # Replace masked node position features with mask token
        x_masked = x.clone()
        x_masked[node_mask] = self.mask_token
        
        # Forward through encoder
        enc_out = self.encoder({"x": x_masked})["preds"]  # [B, N, out_dim]
        # We repurpose the encoder output; need intermediate representations
        # Alternative: add a hook to get slice-token representations
        
        # Reconstruction head
        reconstructed = self.reconstruction_head(enc_out)  # [B, N, 2]
        
        # Loss: L1 on (x, z) reconstruction for masked nodes only
        loss = (reconstructed[node_mask] - x[node_mask, :2]).abs().mean()
        return loss
```
Phase 2 (fine-tuning, epochs 5-12): load pre-trained weights, reset reconstruction head, attach standard pressure/velocity decoder, train with standard supervised L1 loss.

Simpler alternative (if encoder hook is complex): use a separate lightweight 2-layer encoder for pre-training, then use its learned embeddings as additional input features for the main Transolver.

**Why it might break the plateau**:
- The Transolver's slice-token dispatch learns soft assignments from scratch with random initialization, using only 12 epochs of supervised signal. For OOD geometries (camber M=6-8), this means the slice assignments may be poorly calibrated because the training distribution (M=2-5, M=9) doesn't cover M=6-8. Pre-training on masked reconstruction teaches the encoder what the mesh topology looks like for ALL geometries in the training set, including the edges of the geometry distribution.
- Masked node prediction is essentially a self-supervised graph completion task — analogous to BERT for language models. It forces the model to learn geometric context from the mesh structure rather than from pressure/velocity labels. Geometric understanding is exactly what's missing for camber generalization.
- SSL pretraining adds no new labeled data — it's a free lunch from the existing training samples.

**Risk**: The 12-epoch budget is tight for a 2-phase approach (4 pre-train + 8 fine-tune). May need to test with (3+9) or (2+10) splits. The masked reconstruction task uses node positions as targets, which may be too easy (nodes are spatially clustered, neighbors are predictable) — if the task is too easy, pre-training learns trivial features. Recommend checking reconstruction loss curves in phase 1.

**Confidence**: MEDIUM. SSL pretraining has strong evidence in NLP and CV. Application to irregular mesh GNNs is less established but several 2024 papers show promising results. Main uncertainty is whether 12 epochs is sufficient.

**Expected impact**: Potential -5 to -15% on OOD splits. May slightly improve all splits if encoder learns richer geometry representations.

---

## H8 — NSE-Residual Physics-Informed Auxiliary Loss (LOSS REFORMULATION + Physics)

**Family**: LOSS REFORMULATION, PHYSICS-INFORMED

**Mechanism**: Add a Navier-Stokes equation residual auxiliary loss computed on a random subset of volume nodes each forward pass. The NSE residual penalizes predicted (Ux, Uy, p) fields that violate incompressibility (∇·U ≈ 0) and momentum conservation (ρ(U·∇)U = -∇p + μ∇²U). This injects a strong physical prior that is completely missing from the current L1 loss, which treats all 3 output channels independently. For OOD camber generalization, the model can exploit the NSE constraint to infer physically consistent pressure fields from velocity fields (or vice versa) — even for unseen camber configurations.

**Implementation outline**:
```python
# Approximate NSE residual using finite differences on mesh
# For incompressibility: divergence of velocity field
# For momentum: approximated on surface nodes using Cp = (p - p_inf) / (0.5 * rho * U_inf^2)

def approx_incompressibility_residual(pred, x, mask, lambda_incomp=0.01):
    # pred: [B, N, 3] (Ux, Uy, p in normalized space — denormalize first)
    pred_phys = pred * stats["y_std"] + stats["y_mean"]
    Ux = pred_phys[..., 0]  # [B, N]
    Uy = pred_phys[..., 1]  # [B, N]
    
    # Approximate ∂Ux/∂x + ∂Uy/∂z using node positions
    # Build local neighborhood for finite differences (same KNN as H3)
    # For each node i, neighbors j: 
    # ∂Ux/∂x ≈ sum_j (Ux_j - Ux_i) * (x_j - x_i) / ||x_j - x_i||^2
    
    # Simplified: only on a random 10% of volume nodes per forward pass
    # to keep memory and compute manageable
    
    div_U = approx_divergence(Ux, Uy, x, mask)  # [B, N_sampled]
    incomp_loss = div_U.abs().mean()
    return lambda_incomp * incomp_loss

# Bernoulli pressure-velocity consistency (simpler, cheaper):
# On surface nodes: p ≈ p_0 - 0.5 * (Ux^2 + Uy^2) (Bernoulli approximation)
# Penalty: |p_pred - (p_ref - 0.5 * (Ux_pred^2 + Uy_pred^2))| on surface nodes
# This is a soft constraint, not exact — but provides direction

def bernoulli_consistency_loss(pred, x, is_surface, mask, lambda_bern=0.05):
    pred_phys = pred * stats["y_std"] + stats["y_mean"]
    Ux, Uy, p = pred_phys[..., 0], pred_phys[..., 1], pred_phys[..., 2]
    surf_mask = is_surface & mask
    
    # Bernoulli: p + 0.5*rho*(Ux^2 + Uy^2) ≈ const along streamline
    # Use as consistency constraint: variance of p + 0.5*(Ux^2+Uy^2) on surface
    # should be small (for incompressible, inviscid flow approximation)
    q_surf = (p + 0.5 * (Ux**2 + Uy**2)) * surf_mask
    bernoulli_var = (q_surf - q_surf.sum(dim=1, keepdim=True) / surf_mask.sum(dim=1, keepdim=True).clamp(min=1)).pow(2)
    loss = (bernoulli_var * surf_mask).sum() / surf_mask.sum().clamp(min=1)
    return lambda_bern * loss
```

Start with Bernoulli consistency (simpler, no KNN needed) as the primary arm. Add incompressibility if Bernoulli shows promise. `lambda_bern = 0.01` to `0.05`.

**Why it might break the plateau**:
- The current model has learned purely from data with no physical constraint. For unseen camber M=6-8, the data distribution gap means the model has no anchor for what the pressure field "should look like". The Bernoulli constraint provides a physical anchor: the total pressure (static + dynamic) must be approximately conserved along streamlines, regardless of camber.
- For the Re-rand OOD split, the momentum balance changes with Re (viscous terms dominate at low Re). An NSE-residual loss that correctly incorporates Re-dependence would help here.
- This is a different physical signal than the closed thin-airfoil Cp_TAT aux loss (PR #2652), which used a specific 4-digit NACA approximation formula. The Bernoulli/incompressibility constraints are general physical laws, not domain-specific approximations.

**Risk**: The Bernoulli approximation is only valid for inviscid, incompressible flow — but CFD simulations include viscosity (Re-dependent) and may have compressible effects at high speeds. At low Re (100K), viscous effects are dominant and Bernoulli is a poor approximation. Setting `lambda_bern` too high could actively hurt low-Re predictions. Recommend testing with small lambda (0.01, 0.03) and monitoring per-Re performance.

**Confidence**: MEDIUM-LOW. Physics-informed losses for CFD surrogates have mixed results in the literature — they help when the physics is accurately modeled and hurt when the physical approximations are too crude for the actual simulation conditions. The Bernoulli approximation is rougher than the closed thin-airfoil TAT approximation, which itself was too crude (PR #2652). This is a bold bet.

**Expected impact**: High variance. Could improve all splits by -3 to -8% if the Bernoulli consistency constraint adds useful signal. Could hurt low-Re splits if the inviscid approximation is too poor. Monitor re_rand and single_in_dist carefully.

---

## Implementation Priority Ranking

For Wave 20 assignment, priority order based on mechanism strength, implementation risk, and OOD targeting:

1. **H1 — Relative L1 Loss** [HIGHEST PRIORITY — LOW RISK, HIGH MECHANISM CONFIDENCE]
   - Clean fix to a real objective mismatch; no new parameters; easy to implement
   - Should be first assigned and first to complete
   
2. **H5 — Camber-Difficulty Curriculum** [HIGH PRIORITY — SPECIFICALLY TARGETS camber_rc]
   - Directly attacks the primary OOD gap via training-time intervention
   - Implementation in train.py only; no new dependencies
   
3. **H6 — Camber-Conditional LayerNorm** [HIGH PRIORITY — camber_rc targeted]
   - Mechanistically distinct from closed FiLM (different conditioning signal, different application point)
   - Identity initialization makes failure benign
   
4. **H2 — Sobolev Surface Loss** [MEDIUM-HIGH PRIORITY — gradient matching]
   - Targets pressure gradient pattern matching for unseen camber
   - Needs arc-length coordinate handling for tandem foils
   
5. **H3 — GeoMPNN Model Class Change** [MEDIUM PRIORITY — full model class change]
   - Strong external evidence; NeurIPS 2024 ML4CFD winner
   - Higher implementation complexity; assign to capable student
   
6. **H7 — Masked Node SSL Pretraining** [MEDIUM PRIORITY — 2-phase training]
   - Free signal from existing data; addresses representation initialization
   - 12-epoch budget is tight for 2-phase approach
   
7. **H4 — SE(2)-Equivariant Transformer** [MEDIUM PRIORITY — principled model class change]
   - Strong theoretical motivation; highest implementation complexity
   - Assign to most capable student only; expect 2-3 iteration cycles
   
8. **H8 — NSE Bernoulli Consistency Loss** [LOWER PRIORITY — physics approximation risk]
   - Bold physical prior injection; but Bernoulli approximation quality is uncertain
   - Run last or alongside H1 as a risk diversifier

---

## Ruled-Out Ideas (Do Not Repeat)

The following were either closed in prior waves or are mechanistically overlapping with closed experiments:
- FOMA input noise (closed PR #2649 — too weak in 12-epoch budget)
- Thin-airfoil Cp_TAT aux loss (closed PR #2652 — NACA 4-digit approximation too crude)
- Spectral HF penalty on slice tokens (closed PR #2653 — wrong axis)
- SDF geometry features (closed PR #2654 — conflicts with FourierCoordEnc)
- Geo aux head / camber+Re prediction from slice tokens (closed PR #2719)
- Laplacian eigenvector PE from KNN graph (closed PR #2656 thorfinn WIP)
- Slice-token mixup (closed PR #2575 — catastrophic)
- All attention-temperature schedule variants (fully closed Waves 18-19)
- Y-axis reflection (closed PR #2514 — tandem foil NOT y-symmetric)
- FiLM conditioning (closed PR #2453 — helps ID, hurts OOD)
- Equal-weight OOD upsampling (closed PR #2391 — not a density gap)
