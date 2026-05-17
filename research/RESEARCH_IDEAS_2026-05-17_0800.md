# SENPAI Research Ideas — 2026-05-17 08:00
# Cycle 44 / R5 Plateau Protocol — Post-H120 K=1 Frontier

## Research Context

**Current best**: H120 Arm B Fourier PE K=1 → val_avg=35.6651 / test 3-split=33.3976 (PR #4394)

**Active WIP (do NOT duplicate)**:
- H123 (askeladd): Fourier K=0 ablation + K=1 scale=0.5
- H124 (alphonse): EMA τ=0.999/0.9995

**Plateau state**: 8 consecutive post-merge negatives. All obvious hyperparameter levers exhausted.
Canonical anti-overfitting mechanisms (H125–H130) are in-flight. These hypotheses are for cycle 45+
when those return, targeting architecture-level and representation-level changes.

**Primary bottleneck**: val_geom_camber_rc ≈ 47.56 — 12 pts above val_avg=35.67. This split
tests unseen front-foil camber in the RaceCar domain (M=6-8 held out). No lever has closed
this gap disproportionately.

**Signal from Fourier sweep**: K=8→K=4→K=2→K=1 monotone test improvement confirms the model
is overfitting to training-set spatial detail at sub-chord scale. The right direction for
geometry encoding is fewer, coarser features — not more detail.

---

## Ranked Hypotheses

### H-A: Geometry Cross-Attention Context (GALE-style)

**Rank: 1 (highest priority)**

**What it is**: Add a shared geometry-context cross-attention pathway alongside the existing
PhysicsAttention self-attention in each TransolverBlock. Concretely: extract a fixed-size
pool of "geometry tokens" from the surface nodes (using their DSDF and arc-length features),
then let each TransolverBlock cross-attend to this context after the self-attention step.

**Mechanism**: GALE (GeoTransolver, arxiv 2512.20399, NVIDIA/MIT, Dec 2024) replaces Transolver's
pure self-attention with physics-aware self-attention on learned state slices PLUS cross-attention
to a shared multi-scale geometry/BC context built via ball queries. On DrivAerML and Luminary
SHIFT-SUV/Wing it shows improved accuracy and OOD robustness over stock Transolver, specifically
in out-of-distribution geometry splits. The tandem foil val_geom_camber_rc split is a near-exact
analogue: held-out camber values not seen in training. The cross-attention context provides a
stable geometric anchor that is not blended into the spatial encoding (unlike Fourier PE, which
embeds geometry implicitly and can overfit).

**Why it might break the plateau**: The model currently has no explicit pathway to reason about
geometry globally. FiLM conditioning encodes NACA parameters numerically but cannot capture
local camber shape in the DSDF. Cross-attending to surface geometry tokens makes the model's
internal representation of each mesh node explicitly dependent on its position relative to the
foil surface, which is the feature that val_geom_camber_rc tests.

**Implementation steps** (students implement; do not modify program baseline files):
1. Add a `GeomContext` module that takes surface node features (rows where is_surface=1),
   projects them through an MLP to n_hidden, and pools to a fixed K=32 geometry tokens via
   learned slot attention or simple mean-pooling over uniformly spaced arc-length bins.
2. Add an optional `cross_attn` `nn.MultiheadAttention(n_hidden, n_head, batch_first=True)`
   in `TransolverBlock`, applied after PhysicsAttention and FiLM, with `fx` as query and
   geometry tokens as key/value.
3. Control with `--geom_context_tokens K` (default 0 = disabled, K=32 first trial).
4. Share the geometry token pool across all blocks (compute once in `Transolver.forward`,
   pass as optional argument to each block) to keep cost linear in n_layers.
5. Baseline sanity: run K=32 tokens, 1 cross-attn head. If unstable, try K=16 or detach
   the geometry tokens from the gradient of cross-attention.

**Expected gain**: 0.5–2.0 pts val_avg, with disproportionate improvement on val_geom_camber_rc.

**Risk / failure mode**: The surface token pool may collapse to a flat mean that carries no
discriminative camber information. Mitigation: use arc-length-binned pooling (not global mean)
so each slot contains a local surface region.

**External evidence**: GeoTransolver reports 3–8% MAE improvement on OOD geometry splits vs
Transolver baseline. Paper: arxiv 2512.20399. Code not public but architecture is clear.

---

### H-B: LE+TE Dual Coordinate System

**Rank: 2**

**What it is**: Augment each node's spatial representation with coordinates expressed in
two additional frames: (1) leading-edge-centered (x_LE = x - x_LE, z_LE = z - z_LE) and
(2) trailing-edge-centered (x_TE = x - x_TE, z_TE = z - z_TE). These are node coordinates
relative to the two stagnation-point landmarks, not the global mesh frame.

**Mechanism**: Geometry-aware MPNN (arxiv 2412.09399, Texas A&M, Dec 2024) uses a dual
coordinate system in leading-edge and trailing-edge frames for airfoil surface representation.
The key property is scale-invariance within chord length: for any chord-scaled variant of a
foil geometry, LE/TE-relative coordinates are the same function of arc-length position.
This means stagnation and separation regions are described by the same LE/TE coordinates
regardless of absolute chord scaling or global position — which directly addresses the
camber OOD split, because camber affects LE/TE suction peak location but not the coordinate
system itself.

**Why it might break the plateau**: Current input features (x[0:2]) are in global mesh
coordinates, which conflate geometry variation with domain position. DSDF features
(x[4:11]) are already partially geometry-aware but are computed from the closest surface
point, not from stagnation landmarks. LE/TE coordinates provide a canonical, physically
meaningful frame tied to aerodynamic significance.

**Implementation steps**:
1. At data-loading time (or as a transform applied in `Transolver.forward` before preprocess),
   identify LE and TE node indices. LE = node with minimum x-coordinate on upper surface of
   foil 1 (or can be identified as arg-min of arc_length channel x[2]). TE = node with
   maximum x-coordinate.
2. Compute `x_LE = (x[...,0] - x_LE_coords[0])`, `z_LE = (x[...,1] - x_LE_coords[1])` and
   similarly for TE. Append as 4 extra input features.
3. Since foil 1 and foil 2 are present in tandem configurations, compute LE/TE for each
   foil separately (8 features total for tandem; 4 for single foil).
4. These features can be precomputed and stored alongside the existing data, or computed
   on-the-fly using the is_surface mask + x/z coordinates.
5. Control with `--dual_coord` flag. space_dim stays 2; the extra 4-8 features go into
   fun_dim (same path as Fourier PE).
6. First trial: foil-1 LE+TE only (4 features). Second trial: both foils (8 features).

**Expected gain**: 0.3–1.5 pts val_avg with particular sensitivity to camber OOD.

**Risk / failure mode**: LE/TE identification from mesh node coordinates may be unreliable
for highly cambered foils if the "minimum x" heuristic breaks down. Better: use the arc_length
channel minimum as the LE proxy (canonical stagnation parametrization). Should be tested with
a quick sanity visualization before the full training run.

**External evidence**: Texas A&M paper shows the dual-frame encoding gives their MPNN
significantly better generalization on unseen geometries vs single-frame. The mechanism is
tested on NACA family airfoils, directly analogous.

---

### H-C: DSDF Fourier Features

**Rank: 3**

**What it is**: Apply Fourier positional encoding to the DSDF channels (x[4:11]) at K=1
frequency (same as the current coordinate Fourier PE) rather than only to (x, z) spatial
coordinates. The DSDF channels encode signed distance from each of 4-8 foil surface segments.

**Mechanism**: The current Fourier PE (K=1) encodes chord-scale positional information via
sin/cos of (2π·x, 2π·z). DSDF values are implicitly geometric: they encode how far each
mesh point is from each surface segment in physical space. Applying Fourier features to DSDF
channels adds a richer non-linear representation of boundary proximity that is inherently
camber-sensitive — a cambered foil has a different DSDF profile than a symmetric one — but
at the chord-scale resolution that worked best in the K sweep.

**Why it might break the plateau**: The K=1 coordinate Fourier PE captures periodic position
information but cannot distinguish foil-specific geometry (two foils with the same (x,z) but
different camber would have the same coordinate Fourier features but different DSDF features).
DSDF Fourier features would give the model richer distance-to-surface signals without adding
the sub-chord spatial detail that caused K>1 to overfit.

**Implementation steps**:
1. In `Transolver.forward`, after the existing coordinate Fourier PE block, compute
   `dsdf_ff = fourier_features(x[..., 4:11], num_freqs=1)` (shape: B×N×14 for 7 DSDF channels).
2. Concatenate to x before preprocess, updating fun_dim accordingly.
3. Control with `--dsdf_fourier_pe` flag (default off). Number of frequencies controlled by
   `--dsdf_fourier_pe_freqs` (default 1).
4. First trial: K=1 DSDF Fourier combined with K=1 coordinate Fourier (current best).
5. Ablation: K=1 DSDF only (without coordinate Fourier) to test whether DSDF features
   subsume coordinate Fourier.

**Expected gain**: 0.3–1.2 pts val_avg, primarily on geom_camber splits.

**Risk / failure mode**: DSDF channels may already encode sufficient boundary-proximity signal
in raw form; Fourier encoding might only add numerical redundancy. Diagnostic: check if DSDF
variance across training samples is larger than coordinate variance → if so, Fourier encoding
adds more discriminative power for DSDF.

---

### H-D: Cross-Domain Feature Alignment Loss (MMD/CORAL)

**Rank: 4**

**What it is**: Add an auxiliary loss that penalizes distributional discrepancy between
the slice-token distributions of the RaceCar domain and the Cruise domain during training.
Specifically: at each step, accumulate the mean slice-token activations across the two domains
and minimize Maximum Mean Discrepancy (MMD) or second-order statistics alignment (CORAL).

**Mechanism**: Domain Adaptation theory (Gretton et al., Ganin & Lempitsky) shows that
matching feature distributions across source and target domains reduces generalization gap.
The tandem foil dataset has three training domains (RaceCar single, RaceCar tandem, Cruise)
with different mesh topologies and Re ranges. The val_geom_camber_rc split is within RaceCar
but with unseen camber; the val_geom_camber_cruise split is Cruise with unseen camber.
A model that learns domain-invariant slice-token representations would be forced to encode
physics (pressure, velocity) rather than domain-specific layout patterns.

**Implementation steps**:
1. In the training loop, track domain labels from the dataset metadata (the domain is already
   implicit in the split structure; need to add a domain_id field to the dataset or infer
   from sample index).
2. After each forward pass, collect mean slice-token activations from the last TransolverBlock
   (`fx` shape B×N×n_hidden) for each domain in the batch.
3. Compute MMD loss using a Gaussian RBF kernel or CORAL (second-order moment matching):
   `L_mmd = ||mu_racercar - mu_cruise||² + trace(C_racecar - C_cruise)²`
4. Add to total loss with weight `lambda_mmd=0.01` initially.
5. Control with `--mmd_weight` (default 0.0).
6. First trial: last-block only. Second trial: average over all blocks.

**Expected gain**: 0.5–1.5 pts on geom_camber_cruise and geom_camber_rc splits specifically,
potentially small regression on single_in_dist.

**Risk / failure mode**: Domain labels are at the sample level; with batch_size=4, a batch
may not always contain both domains, making MMD estimates noisy. Fix: require each batch to
contain at least one sample from each domain (modify `WeightedRandomSampler` or add a
domain-stratified sampler). Alternatively use CORAL which is more stable with small N.

**External evidence**: CORAL loss (Sun & Saenko 2016) has been applied successfully to mesh
simulation OOD generalization. Domain adversarial training (DANN) is a stronger variant
but adds architecture complexity; CORAL is a single loss term. For physics surrogates,
see PhysDG (2024) and EquivPDE (2023) which apply feature alignment in the latent PDE space.

---

### H-E: Geometry-Space NACA Mixup

**Rank: 5**

**What it is**: During training, randomly interpolate between two samples that differ only
in their NACA parameters, computing the interpolated output as a linear combination of the
two CFD solutions. This is a domain-specific Mixup that operates on the camber-relevant axis
rather than arbitrary sample pairs.

**Mechanism**: Standard Mixup (H129, in-flight) blends arbitrary training pairs (x_i, y_i)
and (x_j, y_j). Geometry-space Mixup restricts blending to pairs where the mesh topology is
compatible (same domain, similar Re and AoA) but NACA parameters differ. The interpolated
target y_mix = λy_i + (1−λ)y_j approximates what the flow field would look like for an
intermediate foil shape, which is approximately linear for small NACA perturbations.
This directly augments training data along the camber axis — exactly the dimension that
val_geom_camber_rc tests.

**Implementation steps**:
1. Pre-compute a compatibility index: pairs (i, j) where same domain, |Re_i - Re_j| < 0.1×Re,
   |AoA_i - AoA_j| < 2 degrees, and same tandem configuration (single vs tandem).
2. At each training step, with probability p_geom_mix=0.3, select a compatible pair and
   blend: x_mix_cond = λ·NACA_i + (1−λ)·NACA_j (replace dims 15-21 in x), y_mix = λy_i + (1−λ)y_j.
3. Mesh geometry (x[0:2], DSDF) cannot be directly blended (different meshes); use only the
   conditioning features (NACA params, Re, AoA) for blending in x, while keeping y_mix as the
   target — this tests whether the model can interpolate on the conditioning manifold.
4. λ ~ Beta(0.5, 0.5) (U-shaped, emphasizes extremes) or λ=0.5 deterministically for stable
   first trial.
5. Control with `--geom_mixup_prob` (default 0.0).

**Expected gain**: 0.3–1.0 pts specifically on camber OOD splits. Relatively cheap to implement.

**Risk / failure mode**: Blending conditioning features while keeping separate mesh geometries
is an approximation that may produce inconsistent (x_cond, y) pairs. The model might learn to
ignore inconsistencies rather than generalize. A cleaner version would require aligned meshes
(re-meshed for each NACA variant) which we don't have. Validate by checking if val_geom_camber_rc
improves disproportionately vs single_in_dist.

---

### H-F: Spectral Normalization on in_project_slice

**Rank: 6**

**What it is**: Apply spectral normalization to the `in_project_slice` linear layer inside
`PhysicsAttention` — the layer that maps each per-head token to the slice_num=96 logits
used to assign tokens to physics slices.

**Mechanism**: `in_project_slice` (shape: dim_head → slice_num) is initialized with orthogonal
weights and trained unconstrained thereafter. If the largest singular value grows, this layer
becomes an amplifier of specific spatial directions, making slice assignments brittle to the
exact spatial frequency content of training geometries. Spectral normalization (Miyato et al.
2018, SN-GAN) constrains σ_max=1 via power iteration, providing a Lipschitz-1 constraint that
regularizes the assignment map without changing architecture capacity.

**Why it might break the plateau**: The K=8→K=1 improvement told us the model overfits to
sub-chord spatial frequencies. The overfitting may be encoded in the in_project_slice singular
vectors — high-singular-value directions may correspond to training-set-specific spatial modes.
Constraining σ_max reduces the effective rank of slice assignments, which is equivalent to
reducing the Fourier PE frequency sensitivity of the slice assignment operator.

**Implementation steps**:
1. Wrap `self.in_project_slice` in `nn.utils.spectral_norm` in `PhysicsAttention.__init__`.
2. `torch.nn.utils.spectral_norm(nn.Linear(dim_head, slice_num))` — applies power iteration
   with `n_power_iterations=1` per step by default. Use `n_power_iterations=2` for better
   approximation quality.
3. Control with `--spectral_norm_slice` flag (default off). No hyperparameter tuning required.
4. Can be combined with K=1 Fourier PE directly — only one weight matrix changes.

**Expected gain**: 0.2–0.8 pts val_avg. Cheap and low-risk.

**Risk / failure mode**: Spectral norm adds a power-iteration step per forward pass through
in_project_slice, adding small computational cost (~1% per block). The n_power_iterations=1
approximation may be too coarse early in training. Monitor train loss closely — if it stalls
in early epochs, increase n_power_iterations or apply spectral norm only after warmup.

**External evidence**: Spectral normalization for generalization in neural operators has been
used in NeuralOperator-style architectures (Kovachki et al., 2023) and is a common stabilization
technique in graph neural networks applied to mesh simulations (e.g., MeshGraphNets ablations).

---

### H-G: Surface Normal + Curvature Input Features

**Rank: 7**

**What it is**: Compute the surface normal vector (dx/ds, dz/ds) and signed curvature
(d²x/ds², d²z/ds²) from the arc_length features already present in x[2:4], and append them
as additional input features (4 extra dims) for surface nodes, zero elsewhere.

**Mechanism**: The arc-length channel (x[2:4]) encodes parametric position along the foil
surface. Its finite-difference derivative gives the unit tangent vector (dx/ds, dz/ds); the
second derivative gives the curvature κ, which directly encodes the camber distribution.
Curvature is the invariant that distinguishes a highly cambered foil from a symmetric one at
the same arc-length position — it is the feature val_geom_camber_rc tests. The model currently
has no access to this geometric invariant in explicit form.

**Implementation steps**:
1. During data loading or in `Transolver.forward`, for surface nodes (is_surface=1), compute
   finite differences of x[0:2] with respect to x[2] (arc_length_foil1 channel):
   `tangent = (x_next - x_prev) / (arc_next - arc_prev)` (central difference where possible).
2. Normalize tangent to unit vector; compute curvature as `||d_tangent/ds||`.
3. Append [tangent_x, tangent_z, curvature, sign_curvature] as 4 new input features.
   For non-surface nodes (is_surface=0), set to zeros.
4. fun_dim increases by 4. Model capacity unchanged, only input dimensionality changes.
5. Control with `--surface_curvature` flag (default off).

**Expected gain**: 0.3–1.0 pts on geom_camber splits. Signal is concentrated in camber-OOD.

**Risk / failure mode**: The arc-length coordinates in x[2:4] may be parameterized differently
for foil 1 vs foil 2 in tandem configurations, requiring care in finite-difference computation.
The curvature signal may be noisy at mesh resolution limits (O(1/Δs) sensitivity); smooth with
a 3-node Gaussian kernel if needed. Validate by checking that curvature values are plausible
before the training run (plot a few samples).

---

### H-H: Stochastic Depth / Drop-Path on TransolverBlocks

**Rank: 8**

**What it is**: Apply stochastic depth (Huang et al. 2016, "Deep Networks with Stochastic Depth")
to the TransolverBlocks during training: each block is randomly dropped with probability p_drop,
linearly scaled from 0 (first block) to p_max (last block), with residual identity bypass when
dropped.

**Mechanism**: With n_layers=4 blocks and stochastic depth, the model sees shorter effective
paths during training (average depth ≈ 2.5 blocks at p_max=0.2), which reduces co-adaptation
across blocks. Unlike dropout (H126, in-flight, applied inside FFN), stochastic depth randomizes
the global depth of each forward pass, providing a qualitatively different regularization
signal. This is standard in modern vision transformers (DeiT, Swin) and was shown by Touvron
et al. (2021) to specifically help with OOD generalization by preventing last-layer overfitting.

**Implementation steps**:
1. In `Transolver.forward`, after computing `fx = block(fx, cond=cond)` for each block,
   apply stochastic depth residual: if training and `torch.rand(1) < p_drop_i`, replace
   `fx = fx_prev + (block_output - fx_prev)` with `fx = fx_prev` (skip the block delta).
2. Use linear schedule: `p_drop_i = p_max * i / (n_layers - 1)` for block i.
3. First trial: p_max=0.1. Second trial: p_max=0.2.
4. Control with `--drop_path_rate` (default 0.0).

**Expected gain**: 0.2–0.7 pts val_avg. Orthogonal to dropout (H126) if that is positive.

**Risk / failure mode**: With only n_layers=4, stochastic depth may be overly aggressive
even at p_max=0.1 (25% chance of dropping last block). Monitor train loss for instability.
The drop rate must be per-step (not per-sample) to avoid gradient issues with residual sharing.

---

### H-I: Learnable Huber Scale (Laplace Output Head)

**Rank: 9**

**What it is**: Replace the fixed Huber δ=0.25 for pressure with a learnable per-channel
scale parameter σ under the Laplace likelihood: `L = |y - ŷ| / σ + log(σ)`. This is
equivalent to adaptive Huber regression where the model learns the optimal scale per output
channel from the data.

**Mechanism**: The current Huber loss uses fixed δ_p=0.25, δ_vel=0.5. These were set by
prior experiments but not re-tuned after the Fourier PE changed the effective output range.
Laplace likelihood with learnable σ interpolates between L1 (large σ) and L2 (small σ)
loss, automatically adapting to the noise distribution of each channel. For pressure near
stagnation points (high MAE in camber-OOD), σ_p may be too small, causing the loss to
upweight hard examples inappropriately; adaptive σ would rebalance.

**Implementation steps**:
1. Add `log_sigma_p`, `log_sigma_Ux`, `log_sigma_Uy` as learnable `nn.Parameter` scalars
   (initialized to log(0.25), log(0.5), log(0.5) respectively).
2. Replace Huber loss with:
   `sigma = log_sigma.exp().clamp(min=1e-4)`
   `L_ch = (|y - ŷ|_ch / sigma + log(sigma)).mean()`
3. Keep surf_weight=10 as the weighting between surface and volume terms.
4. Control with `--learnable_loss_scale` flag (default off).

**Expected gain**: 0.1–0.5 pts. This is a refinement hypothesis, not a major architectural
change. Most useful if pressure MAE on camber OOD is specifically the limiting factor.

**Risk / failure mode**: σ may collapse to a very large value (large σ → small gradient →
model stops learning). Fix: clip log_sigma to [-3, 2] range. Also ensure σ parameters
use a separate, higher learning rate than the rest of the model (add to a separate param
group with lr×10) to allow rapid initial calibration.

---

### H-J: Attention Temperature Annealing

**Rank: 10**

**What it is**: Replace the fixed `temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)`
in PhysicsAttention with a scheduled temperature that starts high (0.9) and is cosine-annealed
to 0.3 over training. This controls how "sharp" the slice-assignment attention becomes over time.

**Mechanism**: The PhysicsAttention temperature parameter scales the dot-product attention
logits for slice token pooling. A high temperature produces soft, diffuse slice assignments
(each token contributes to many slices), which is a strong regularizer against spatial
overfitting. A low temperature produces hard, peaked assignments (each token is committed
to one slice), which recovers the original Transolver behavior. Starting high and annealing
low is analogous to the "temperature annealing" in self-supervised learning (DINO, VICReg),
where progressive commitment regularizes feature learning.

**The OOD connection**: During early training, high temperature prevents slice assignments
from committing to training-set-specific spatial modes. By the time temperature is low and
assignments harden, the model has already learned physics-consistent representations that
generalize to unseen camber values. Without annealing, the model commits to training-set
geometry early and cannot recover.

**Implementation steps**:
1. Change `temperature` from `nn.Parameter` to a buffer (non-trainable).
2. Pass a `temperature_value` argument to `PhysicsAttention.forward` (or set directly on
   the module), controlled by the training loop's epoch counter.
3. Schedule: `T(epoch) = T_max - (T_max - T_min) * (1 - cos(π * epoch / T_max)) / 2`
   with T_min=0.3, T_max=0.9, cosine over training duration.
4. First trial: T_min=0.3, T_max=0.9. Second trial: T_min=0.5, T_max=1.2.
5. Control with `--temp_anneal_max` / `--temp_anneal_min` flags (default: no annealing,
   use existing learned temperature as a fallback).

**Expected gain**: 0.2–0.8 pts. Mechanism is architecture-level; orthogonal to all current
regularization experiments.

**Risk / failure mode**: Making temperature non-trainable removes an implicit adaptive
mechanism. Hybrid: keep it trainable but add the cosine schedule as a lower-bound constraint,
or initialize it to T_max and let it be trainable from there. If temperature collapses to
the floor in early training despite annealing, add a floor of T_min throughout.

---

### H-K (Stretch): Implicit Neural Representation (INR) Decoder

**Rank: 11 (bold tier-shift for plateau break)**

**What it is**: Replace the TransolverBlock output decoder (the `last_layer=True` block that
projects from n_hidden to out_dim=3) with a small INR decoder — an MLP that takes as input
the node's (x, z) coordinate plus the Transolver latent token and outputs (Ux, Uy, p).

**Mechanism**: Aero-Nef (Airbus/ISAE-Supaero, arxiv 2407.19916, 2024) is an INR-based
aerodynamic surrogate that achieves 3× lower test error and markedly better generalization
on unseen geometries vs GNNs, specifically because the coordinate-conditioned MLP decoder
provides a continuous, geometry-respecting reconstruction. The core insight is that the
decoder should be a function of continuous coordinates, not just a linear projection of
latent tokens. For our setting: replace `self.to_out` (the last linear layer of the last
TransolverBlock) with a 2-hidden-layer MLP:
`output(node) = MLP([latent(node), x(node), z(node)], → 3)`.

This is a zero-cost architectural change at inference (same parameter count roughly) but
forces the output to vary continuously with node position given the same latent, which
prevents the model from memorizing discrete mesh-node patterns in training geometries.

**Implementation steps**:
1. In the `last_layer=True` branch of TransolverBlock's `__init__`, replace the current
   `to_out = nn.Linear(n_hidden, out_dim)` with
   `to_out = MLP(n_input=n_hidden+2, n_hidden=n_hidden//2, n_output=out_dim, n_layers=1)`.
2. In the last block's `forward`, concatenate the (x, z) coordinates to `fx` before passing
   to `to_out`: `preds = self.to_out(torch.cat([fx, coords], dim=-1))`.
3. The model needs to receive the spatial coordinates in the last block's forward call. Pass
   them as an optional argument from `Transolver.forward`.
4. Control with `--inr_decoder` flag (default off). This is a structural change that requires
   careful testing with `--debug` first.

**Expected gain**: 0.5–2.5 pts on OOD splits if the mechanism is alive. High variance.

**Risk / failure mode**: The MLP decoder may not improve if the TransolverBlock output already
has sufficient geometry specificity (the current architecture might effectively be doing this
already via FiLM conditioning). A cheap diagnostic: train with `--debug` and check if the
INR decoder variant produces qualitatively different predictions on a camber-OOD sample.

---

## Summary Ranking

| Rank | ID | Mechanism | OOD target | Complexity | Expected gain |
|------|----|-----------|------------|------------|---------------|
| 1 | H-A | Geometry cross-attention context | val_geom_camber_rc | High | 0.5–2.0 pts |
| 2 | H-B | LE+TE dual coordinate system | val_geom_camber_rc | Medium | 0.3–1.5 pts |
| 3 | H-C | DSDF Fourier features | Both camber splits | Low | 0.3–1.2 pts |
| 4 | H-D | MMD/CORAL cross-domain alignment | Both camber splits | Medium | 0.5–1.5 pts |
| 5 | H-E | NACA Mixup (geometry-space) | val_geom_camber_rc | Medium | 0.3–1.0 pts |
| 6 | H-F | Spectral norm on in_project_slice | All splits | Low | 0.2–0.8 pts |
| 7 | H-G | Surface normal + curvature | val_geom_camber_rc | Low | 0.3–1.0 pts |
| 8 | H-H | Stochastic depth / drop-path | All splits | Low | 0.2–0.7 pts |
| 9 | H-I | Learnable Huber scale (Laplace) | Pressure OOD | Low | 0.1–0.5 pts |
| 10 | H-J | Attention temperature annealing | All splits | Low | 0.2–0.8 pts |
| 11 | H-K | INR decoder (Aero-Nef style) | OOD generalization | Medium | 0.5–2.5 pts |

## Experimental Decision Tree

```
H123 (K=0) + H124 (EMA) + H125-H130 results arrive
        │
        ├── Any positive (>2σ=1.67 pts improvement)
        │       └── Merge, assign more students to promising directions
        │           If H125 (wd) or H126 (dropout) wins → try H-F (spectral norm, orthogonal)
        │           If H124 (EMA) wins → try H-H (stochastic depth, orthogonal)
        │           If H129 (mixup) wins → try H-E (NACA Mixup, same mechanism, targeted)
        │
        └── All negative (full plateau confirmed)
                └── TIER SHIFT: assign bold architectural hypotheses
                    Priority order for 8 students:
                    1. H-A (geom cross-attn, bold, directly targets OOD bottleneck)
                    2. H-B (dual coords, cheap diagnostic, direct OOD mechanism)
                    3. H-G (surface curvature, very cheap, OOD-targeted)
                    4. H-C (DSDF Fourier, low-risk incremental, K=1 compatible)
                    5. H-H (stochastic depth, cheap regularizer, independent)
                    6. H-F (spectral norm, near-zero cost, independent)
                    7. H-D (MMD domain alignment, medium complexity)
                    8. H-K (INR decoder, bold/risky, tier shift)
                    
                    Hold H-E (NACA Mixup) and H-J (temp annealing) for second round
                    after above results, unless H-K fails → then try H-J as lower-risk.
```

## Implementation Priority Notes

- H-A and H-B share a common feature: they require identifying structural landmarks (surface
  nodes, LE/TE positions). If H-B is assigned first, the infrastructure for surface node
  identification can be reused by H-A.
- H-F (spectral norm) is the lowest-risk experiment in the list and can be bundled with any
  other hypothesis as a secondary arm at near-zero extra cost.
- H-K (INR decoder) should be validated with a 3-epoch debug run before committing to a full
  training run, as it changes the output pathway and may require LR adjustment.
- H-D (MMD) requires domain label access in the training loop; verify that the dataset
  already provides domain tags (from the splits_v2 directory structure) before assigning.

## Source Literature

- GeoTransolver (GALE): arxiv 2512.20399, Dec 2024 — NVIDIA/MIT, Transolver+ball-query cross-attn
- Geometry-Aware MPNN: arxiv 2412.09399, Dec 2024 — Texas A&M, LE+TE dual frames for airfoils
- Adversarial Distillation for OOD: arxiv 2510.18989, Oct 2024 — PGD attack + KD for neural operators
- Aero-Nef (INR surrogate): arxiv 2407.19916, Jul 2024 — Airbus/ISAE-Supaero, INR 3× lower OOD error
- Stochastic Depth: Huang et al. 2016 NeurIPS, DeiT (Touvron 2021) for OOD regularization
- Spectral Normalization: Miyato et al. 2018 ICLR (SN-GAN); NeuralOperator applicability
- CORAL: Sun & Saenko 2016 — second-order feature alignment for domain adaptation
- MMD / DANN: Gretton et al. 2012 (kernel MMD), Ganin & Lempitsky 2015 (adversarial DA)
- Mixup: Zhang et al. 2018 ICLR; geometry-space variant from Aubin et al. 2020 (manifold mixup)
- Temperature annealing: DINO (Caron et al. 2021), VICReg (Bardes et al. 2022)
