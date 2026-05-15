<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15

Fresh hypotheses for TandemFoilSet CFD surrogate improvement.
Primary metric: `val_avg/mae_surf_p` (lower is better).

Excluded (already drafted as first-round ideas): width scale-up, depth scale-up,
attention capacity increase, Huber loss, surf-weight ramp, Fourier position features,
longer schedule with warmup, mixed precision.

---

## H1 — FiLM Global-Parameter Conditioning

**Statement.** Inject a FiLM (Feature-wise Linear Modulation) layer after the
Transolver preprocessing MLP, conditioning every node embedding on a learned
embedding of the global flow parameters (log Re, per-foil AoA, NACA codes, gap,
stagger).

**Why it might help.** The current model sees global parameters (dims 13–23) mixed
uniformly with local geometry features (dims 0–11) via the same linear projection.
A FiLM layer lets the model learn a global-to-local modulation function: for a given
Re and geometry, scale and shift the intermediate node representations before attention.
This is the mechanism by which FiLMNet outperformed GNOT on surface pressure in the
BlendedNet++ benchmark (arxiv 2512.03280). The geometry-interpolation val splits
(val_geom_camber_rc/cruise) test exactly the scenario where the model must extrapolate
a new front-foil camber value — a per-sample conditional rescaling of the feature space
directly targets this failure mode.

**Predicted delta.** 5–15% reduction in val_avg/mae_surf_p. The BlendedNet++ paper
reports FiLMNet narrowing the gap over standard MLP baselines by ~10–20% on OOD
geometry splits. The benefit depends on how well the model currently disentangles
global from local features — if it conflates them, FiLM has large headroom.

**Specific code change (train.py).**
1. Extract global features: `g = x_norm[:, :, 13:]` — shape `[B, N, 11]`. Because
   all nodes in a sample share the same global parameters, take the first node:
   `g_global = g[:, 0, :]` — shape `[B, 11]`.
2. Add a `FiLMConditioner` module:
   ```python
   class FiLMConditioner(nn.Module):
       def __init__(self, cond_dim, hidden_dim):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(cond_dim, hidden_dim),
               nn.GELU(),
               nn.Linear(hidden_dim, 2 * hidden_dim),
           )
       def forward(self, h, cond):
           # h: [B, N, D], cond: [B, D_cond]
           params = self.net(cond)  # [B, 2D]
           gamma, beta = params.chunk(2, dim=-1)
           return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
   ```
3. In `Transolver.forward`, after `fx = self.preprocess(x) + self.placeholder`:
   ```python
   g_global = data["x"][:, 0, 13:]  # raw not normalized; or pass separately
   fx = self.film(fx, g_global)
   ```
4. Instantiate in `__init__`: `self.film = FiLMConditioner(cond_dim=11, hidden_dim=n_hidden)`.

Key hyperparameters: `cond_dim=11` (dims 13–23), hidden_dim matches model's `n_hidden=128`.
No new packages needed.

**Risk / failure mode.** If global parameters are already captured well by the
current linear projection, FiLM adds ~20K parameters with no gain. The conditioning
uses raw (unnormalized) `x[:, 0, 13:]` — must use the normalized version
`x_norm[:, 0, 13:]` for stable training, otherwise log Re (dim 13 after normalization)
has mismatched scale with the rest of the conditioning.

**Literature anchor.** Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Approach," AAAI 2018. Applied to aerodynamic surrogates: "BlendedNet++:
A Multi-Scale Architecture for Aerodynamic Prediction," arXiv 2512.03280.

---

## H2 — Per-Sample Output Scale Normalization (Re-Adaptive Targets)

**Statement.** Before computing the loss, normalize each sample's targets by its
per-sample standard deviation (computed from the non-padding nodes), then rescale
predictions back for evaluation. This makes the loss magnitude invariant to the
order-of-magnitude variation in flow velocity with Re.

**Why it might help.** The dataset's per-sample y std ranges from ~50 (low-Re cruise)
to ~2077 (high-Re raceCar) — a 40x span. Global normalization with a single y_mean/y_std
from `stats.json` means the loss is dominated by high-Re samples. Low-Re samples
contribute essentially zero gradient signal because their normalized residuals are tiny.
This is a known training pathology in multi-regime surrogate regression; per-sample
scale normalization is analogous to per-utterance normalization in speech synthesis.
Fixing this should improve val_re_rand (stratified Re holdout) and low-Re cruise
samples specifically.

**Predicted delta.** 3–12% on val_avg/mae_surf_p. The effect should concentrate on
val_geom_camber_cruise (low-Re cruises dominate that split) and val_re_rand.

**Specific code change (train.py).** In the training loop, after computing `y_norm`:
```python
# Per-sample y std (over non-padding nodes) for adaptive scaling
with torch.no_grad():
    valid_y = y * mask.unsqueeze(-1)  # zero-out padding
    n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1]
    y_bar = valid_y.sum(dim=1) / n_valid  # [B, 3]
    sq_dev = ((y - y_bar.unsqueeze(1)) ** 2 * mask.unsqueeze(-1))
    per_sample_std = (sq_dev.sum(dim=1) / n_valid).sqrt().clamp(min=1.0)  # [B, 3]
    # Rescale targets: [B, N, 3] / [B, 1, 3]
    y_scaled = y_norm / per_sample_std.unsqueeze(1)
    pred_scaled = pred / per_sample_std.unsqueeze(1)
sq_err = (pred_scaled - y_scaled) ** 2
```
The normalization uses the physical y (before global norm) std for a clean scale
estimate, then divides normalized y by it so the model still predicts in normalized
space. At eval time, this transform is NOT applied — `evaluate_split` uses the
standard MAE without per-sample rescaling.

**Risk / failure mode.** If `per_sample_std` is computed from the full padded batch,
padding zeros will artificially deflate the std estimate. The mask-aware computation
above avoids this. A second risk: if all samples in a mini-batch happen to be
high-Re, the per-batch std estimate is noisy; averaging over the physical y rather
than normalized y gives a more stable signal.

**Literature anchor.** Instance normalization for time-series forecasting (Kim et al.,
"Reversible Instance Normalization for Accurate Time-Series Forecasting," ICLR 2022).
Multi-magnitude PDE loss equalization (arXiv 2308.06672, "Gradient-Enhanced
Physics-Informed Neural Networks with Grouping Regularization").

---

## H3 — Separate Surface and Volume Decoder Heads

**Statement.** Replace the single output MLP in the final `TransolverBlock` with two
separate output heads — one for surface nodes and one for volume nodes — applied
conditionally based on `is_surface`.

**Why it might help.** Surface nodes (boundary layer, foil surface) have fundamentally
different physics from volume nodes (far-field, wake region): surface pressure is
determined by the boundary condition and the no-slip constraint, while volume pressure
is governed by the Bernoulli equation and the inviscid outer flow. The current single
decoder must simultaneously represent both regimes from the same hidden state. Two
separate heads with independent final projections allow each to specialize its
activation range and bias toward the relevant regime. The primary metric
`mae_surf_p` is exclusively a surface quantity, so a head specialized on surface
physics is a direct architectural alignment with the objective.

**Predicted delta.** 4–8% on val_avg/mae_surf_p. Effect should be most visible in
splits where surface/volume balance is challenging (raceCar tandem at high Re, where
surface pressures are extreme).

**Specific code change (train.py).** Modify `TransolverBlock`:
```python
# In __init__, when last_layer:
self.surf_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
    nn.Linear(hidden_dim, out_dim),
)
self.vol_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
    nn.Linear(hidden_dim, out_dim),
)
```
In forward:
```python
if self.last_layer:
    fx_ln = self.ln_3(fx)
    surf_out = self.surf_head(fx_ln)   # [B, N, out_dim]
    vol_out  = self.vol_head(fx_ln)    # [B, N, out_dim]
    return surf_out, vol_out           # caller selects
```
In `Transolver.forward`, pass `is_surface` through to the last block and select:
```python
# is_surface: [B, N] bool
surf_pred, vol_pred = last_block(fx, is_surface)
preds = torch.where(is_surface.unsqueeze(-1), surf_pred, vol_pred)
return {"preds": preds}
```
This requires threading `is_surface` into the model forward; add it to the `data`
dict: `model({"x": x_norm, "is_surface": is_surface})`.

**Risk / failure mode.** During training the model receives `is_surface` and can
simply memorize which head to use by type rather than learning physics. Ensure
gradients flow through both heads every batch (the balanced sampler guarantees
surface nodes exist in every batch). Also: the head selection must be consistent
between training loss and evaluation.

**Literature anchor.** Multi-task learning with task-specific heads in neural
operators (Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D PDEs,"
NeurIPS 2023, arXiv 2309.00583). Boundary vs. interior decomposition in PDE solvers
(Lagaris et al., "Artificial Neural Networks for Solving Ordinary and Partial
Differential Equations," IEEE TNNLS 1998).

---

## H4 — Per-Channel Loss Weighting (Upweight p Relative to Ux, Uy)

**Statement.** Replace the uniform MSE loss across all three output channels with a
weighted per-channel loss that assigns a higher weight to the pressure channel (p)
relative to the velocity channels (Ux, Uy), since `mae_surf_p` is the sole ranking
metric.

**Why it might help.** The current loss treats Ux, Uy, and p with equal weight in
normalized space. However, pressure in normalized space has a systematically
different residual scale from velocity because the global normalization statistics
are computed over all three channels independently (`y_std` is a 3-vector). The
ranking metric cares only about p on surface nodes. By upweighting p in the training
loss, the optimizer is told explicitly what matters. This is analogous to task
weighting in multi-task learning and should produce a Pareto shift along the
velocity-pressure tradeoff.

**Predicted delta.** 3–10% on val_avg/mae_surf_p; some degradation expected on
`mae_surf_Ux` and `mae_surf_Uy`. The net effect depends on whether velocity and
pressure are tightly coupled through the model's internal representation or
effectively independent.

**Specific code change (train.py).** Introduce a `channel_weights` parameter:
```python
# In Config:
p_channel_weight: float = 3.0  # weight for p channel (index 2) relative to Ux, Uy
```
In the training loop, after computing `sq_err`:
```python
ch_w = torch.tensor([1.0, 1.0, cfg.p_channel_weight],
                    dtype=sq_err.dtype, device=sq_err.device)  # [3]
sq_err = sq_err * ch_w[None, None, :]  # broadcast over [B, N, 3]
```
The rest of the loss computation is unchanged. Suggested sweep: `p_channel_weight` in
{2.0, 3.0, 5.0} across three short runs.

**Risk / failure mode.** High p_channel_weight can destabilize training if the
pressure gradient dominates. Start at 3.0. Also: the validation `evaluate_split`
function must NOT apply channel weighting — it computes unweighted MAE for reporting,
which is correct as-is.

**Literature anchor.** GradNorm (Chen et al., "GradNorm: Gradient Normalization for
Adaptive Loss Balancing in Deep Multitask Networks," ICML 2018). Task weighting in
physics-informed networks for multi-output PDEs.

---

## H5 — Magnitude-Equalized Loss via Power Normalization

**Statement.** Apply a power operation to each loss term (vol_loss, surf_loss) to
equalize their magnitudes before combining, following the grouping regularization
approach from arXiv 2308.06672.

**Why it might help.** The current loss is `vol_loss + 10 * surf_loss`. The raw
magnitudes of vol_loss and surf_loss vary across samples and across training stages.
At initialization, surf_loss may be 100x larger than vol_loss (surface nodes have
much more extreme values); by epoch 10 the ratio may invert. The fixed weight 10
cannot track this ratio dynamically. Power normalization raises each term to a power
alpha ∈ (0, 1) so that the combined gradient scale is invariant to the absolute
magnitude of each term. This is a principled fix for the multi-magnitude loss
problem documented in the multi-magnitude PINN paper, which reports 10–30% better
convergence on PDE systems with disparate loss scales.

**Predicted delta.** 5–12% on val_avg/mae_surf_p through better optimization
dynamics, especially early in training.

**Specific code change (train.py).** In the training loop:
```python
# In Config:
loss_power: float = 0.5  # power for magnitude equalization (0.5 = sqrt)

# In training step, replace:
# loss = vol_loss + cfg.surf_weight * surf_loss
# With:
vol_term  = vol_loss.pow(cfg.loss_power)
surf_term = surf_loss.pow(cfg.loss_power)
# Renormalize so the combination has roughly the same scale as before
loss = vol_term + cfg.surf_weight * surf_term
```
Note: `vol_loss` and `surf_loss` are scalars (already reduced), so `.pow()` works
directly. Suggested sweep: `loss_power` in {0.5, 0.7, 0.9}.

**Risk / failure mode.** Power < 0.5 can produce very flat gradients when losses are
small (near convergence). Power = 0.5 (square root) is the theoretically motivated
choice from the paper. The interaction with `surf_weight=10` must be re-tuned; when
using power=0.5, the effective surf weight becomes `10 * (surf_loss / vol_loss)^0.5`,
which is adaptive. Initial `surf_weight` may need to be reduced to ~3–5.

**Literature anchor.** "Gradient-Enhanced Physics-Informed Neural Networks with
Grouping Regularization for Forward and Inverse Problems of PDEs," arXiv 2308.06672.

---

## H6 — Two-Stage Training: Uniform Pre-train then Surface-Focused Fine-tune

**Statement.** Train for the first 60% of epochs with `surf_weight=1.0` (uniform
loss) to build a globally accurate representation, then switch to `surf_weight=20.0`
for the final 40% to specialize the model toward surface pressure accuracy.

**Why it might help.** Starting with high surf_weight from epoch 1 biases the model
toward surface predictions before it has learned a good global flow representation.
The volume flow (Bernoulli, far-field decay) constrains the surface solution via
continuity and momentum — a model that has first learned the volume structure should
produce more physically consistent surface predictions when fine-tuned on the surface
objective. This two-stage approach is analogous to pre-training on a surrogate loss
then fine-tuning on the target objective in NLP/vision transfer learning.

**Predicted delta.** 3–8% on val_avg/mae_surf_p. The benefit should concentrate
in geometry-OOD splits where the model needs a strong global prior to generalize to
unseen foil shapes.

**Specific code change (train.py).** Add a schedule for surf_weight:
```python
# In Config:
surf_weight_stage1: float = 1.0   # uniform stage
surf_weight_stage2: float = 20.0  # surface-focus stage
stage2_start_frac: float = 0.6    # fraction of MAX_EPOCHS for stage 1
```
In the training loop:
```python
stage2_epoch = int(cfg.stage2_start_frac * MAX_EPOCHS)
current_surf_weight = (cfg.surf_weight_stage1
                       if epoch < stage2_epoch
                       else cfg.surf_weight_stage2)
# Replace cfg.surf_weight → current_surf_weight in the loss line
loss = vol_loss + current_surf_weight * surf_loss
```

**Risk / failure mode.** The transition at epoch `stage2_epoch` can cause a loss
spike if `surf_weight_stage2` is too large relative to `stage1`. Use `stage2_start_frac=0.6`
and `stage2=20.0` as the first point; also check whether the best checkpoint is
selected from the fine-tuning phase (it should be, since that's where surface
metrics improve).

**Literature anchor.** Curriculum and staged training in deep learning (Bengio et al.,
"Curriculum Learning," ICML 2009). Two-stage fine-tuning for physics surrogates
(NeuralFoil, arXiv 2503.16323, which uses staged training from coarse to fine
resolution).

---

## H7 — Signed Re-Conditioned Residual Connection

**Statement.** Add a lightweight physics-prior residual: after the final Transolver
output, add a learned linear correction `A * log_Re_normalized + b` applied
per-output-channel, so the model predicts a Reynolds-number-aligned prior plus a
learned residual.

**Why it might help.** The dominant effect of Reynolds number on the pressure field
is well understood: surface Cp (pressure coefficient) scales approximately as
`~Re^(-0.5)` in the laminar regime and `~Re^(-0.2)` in turbulent flow. A linear
function of log(Re) can approximate this power-law scaling. Giving the model an
explicit prior that the output scales with Re means the residual that the attention
mechanism must predict is small and smooth — reducing the effective problem
complexity. This is analogous to the "physics residual" formulation used in
data-driven CFD surrogates (GINO, arXiv 2309.00583).

**Predicted delta.** 4–10% on val_avg/mae_surf_p; the largest gain on val_re_rand
where the Re range is widest. Low risk of hurting other splits.

**Specific code change (train.py).** After `Transolver.forward` computes `fx` (which
is now `preds`), add a global bias:
```python
class Transolver(nn.Module):
    def __init__(self, ...):
        ...
        # Re-conditional prior: maps log(Re) → per-channel linear prior
        # log_re is x_norm[:, :, 13] (dim 13, already normalized)
        self.re_prior = nn.Linear(1, out_dim, bias=True)

    def forward(self, data):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        # x[..., 13] is log(Re) position (already in normalized space)
        log_re = x[:, :, 13:14]  # [B, N, 1]
        re_bias = self.re_prior(log_re)  # [B, N, out_dim]
        return {"preds": fx + re_bias}
```
This adds only 4 parameters (3 weights + 3 biases for 3 output channels from a 1-dim
input). Use `x_norm` (the normalized version) so log_re is zero-centered.

**Risk / failure mode.** The linear prior may over-fit the training Re distribution
if the model learns to rely on it exclusively for Re-scaling, reducing the attention
mechanism's role. Monitor val_re_rand — if it improves but val_geom_camber_* does
not, the prior is being used correctly. If val_re_rand also regresses, the prior is
distorting the gradient.

**Literature anchor.** Physics-prior residual correction in neural operators (GINO,
arXiv 2309.00583, Section 3.2). Re-scaling arguments in aerodynamic surrogate
modelling (NeuralFoil, arXiv 2503.16323).

---

## H8 — AoA Reflection Symmetry Augmentation for Single-Foil Samples

**Statement.** During training, randomly apply a reflection augmentation to
single-foil samples: negate the y-coordinate, negate AoA, negate Uy, and negate Ux
sign (where appropriate for the reflected flow), exploiting the bilateral symmetry of
NACA airfoils at ±AoA.

**Why it might help.** NACA 4-digit profiles are symmetric about the chord line for
zero camber and approximately symmetric about their camber line for finite camber.
For the raceCar single domain (AoA -10° to 0°), each physical simulation at AoA=-α
approximately mirrors a hypothetical AoA=+α case. Augmenting with reflections
doubles the effective dataset size for single-foil samples and teaches the model the
physical symmetry as an inductive bias. This is the primary data augmentation
technique used in NeuralFoil (arXiv 2503.16323) for generalization across AoA.

**Predicted delta.** 3–8% on val_single_in_dist; modest improvement on geometry-OOD
splits if the symmetry prior transfers. No benefit to tandem samples (stagger and gap
break the symmetry), so the augmentation must be applied only to single-foil samples
(dims 18–23 all zero).

**Specific code change (train.py).** In the training loop, after loading the batch,
before normalization:
```python
# Identify single-foil samples: gap == 0 and stagger == 0
is_single_foil = (x[:, 0, 22] == 0) & (x[:, 0, 23] == 0)  # [B]
if is_single_foil.any():
    flip = torch.rand(B, device=x.device) < 0.5  # random 50% flip
    flip = flip & is_single_foil  # only flip single-foil
    # x: negate z-coord (dim 1), negate AoA (dim 14), negate saf (dim 3)
    x[flip, :, 1] = -x[flip, :, 1]    # z coordinate
    x[flip, :, 3] = -x[flip, :, 3]    # signed arc-length z-component
    x[flip, :, 14] = -x[flip, :, 14]  # AoA foil 1
    # y: negate Uy (dim 1), p unchanged (symmetric), Ux unchanged
    y[flip, :, 1] = -y[flip, :, 1]    # Uy sign flips under z-reflection
```
Note: `x` and `y` are in physical (un-normalized) space here; normalization happens
after. The dsdf (dims 4–11) encodes distances and may need careful handling — for a
pure z-flip, distances to surface points are preserved, but their z-components in
the vector encoding should be negated. If dsdf is unsigned, leave it unchanged.
Check dims 4–11 semantics in `data/prepare_splits.py` before implementing.

**Risk / failure mode.** Incorrect reflection of multi-dim features (e.g., dsdf
vector components) will corrupt the augmented samples and harm training. Start by
only augmenting dims {1, 3, 14} for x and {1} for y, and verify on a small debug
run that loss is stable. The `data/` files are read-only; the augmentation must live
in `train.py` in the training loop.

**Literature anchor.** Symmetry augmentation for airfoil ML (NeuralFoil,
arXiv 2503.16323). Equivariant augmentation in PDE surrogates (Gao et al.,
"PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for
Solving Parametric PDEs on Irregular Domain," JCP 2021).

---

## H9 — Relative Position Features (Node Position Relative to Foil Centroid)

**Statement.** Append relative position features to the input — specifically, for
each mesh node, compute its position relative to the centroid of the surface nodes
in the same sample, and include this as additional input dimensions alongside the
existing absolute (x, z) position.

**Why it might help.** The current model sees absolute mesh positions (dims 0–1),
which depend on the specific placement of the foil geometry in the domain. For a
different foil at a different position, the absolute coordinates shift, making
generalization harder. Relative coordinates that are invariant to rigid translation
are a standard inductive bias in graph neural networks and attention-based PDE
solvers. For geometry-interpolation generalization (val_geom_camber splits), the
model must relate node positions to the foil boundary — relative coordinates make
the relevant geometric relationship explicit in the input.

**Predicted delta.** 4–10% on val_geom_camber splits specifically. The effect on
val_re_rand and val_single_in_dist should be neutral to mildly positive.

**Specific code change (train.py).** The data/ loader returns normalized x, but
normalization is applied in the trainer. Before normalization, compute:
```python
# Compute centroid of surface nodes per sample
# is_surface: [B, N] bool, x: [B, N, 24] in physical space
surf_pos = x[:, :, 0:2] * is_surface.unsqueeze(-1).float()  # [B, N, 2]
n_surf = is_surface.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
centroid = surf_pos.sum(dim=1) / n_surf  # [B, 2]
rel_pos = x[:, :, 0:2] - centroid.unsqueeze(1)  # [B, N, 2]
# Append to x: [B, N, 26]
x = torch.cat([x, rel_pos], dim=-1)
```
This adds 2 dimensions, making x [B, N, 26]. Update `model_config`:
```python
fun_dim = X_DIM - 2 + 2  # = 24
```
Also update `stats` normalization: the two new dims need mean/std. Compute these
from the training set or simply normalize with a fixed scale (e.g., divide rel_pos
by 10.0, which is roughly the domain half-width).

**Risk / failure mode.** The centroid of all surface nodes may include both foil 1
and foil 2 surfaces for tandem samples, placing it between the foils — which is not
meaningful for either foil individually. For tandem samples, compute separate
centroids per foil and use the nearest centroid. However, the model cannot distinguish
foil 1 vs. foil 2 nodes from the input (per program.md). A simpler fallback: use the
global domain centroid (0, 0) if the mesh is centered, which makes rel_pos just a
re-centered version of absolute position.

**Literature anchor.** Relative position encoding in graph neural networks (Gilmer
et al., "Neural Message Passing for Quantum Chemistry," ICML 2017). GINO's
point-cloud representation uses relative positions (arXiv 2309.00583). PointNet++
uses relative coordinates within each local neighborhood.

---

## H10 — Hierarchical Two-Level PhysicsAttention (Coarse + Fine Slices)

**Statement.** Replace the single-level PhysicsAttention (64 slices) with a
two-level hierarchy: a coarse level with 32 global slices that captures domain-wide
structure (background flow, wake), plus a fine level with 64 slices computed on the
surface-node subset only, then combine both outputs per node.

**Why it might help.** The current 64-slice attention must simultaneously represent
both far-field flow structure (smooth, large-scale variations) and near-wall boundary
layer effects (sharp gradients, small-scale variations). These two regimes benefit
from different resolutions. A hierarchical approach analogous to Erwin's ball-tree
transformer (arXiv 2502.17019) or PointNet++'s multi-scale grouping allows the model
to allocate separate representational capacity to each regime. The primary metric
cares about surface pressure — giving the surface regime its own attention level
directly addresses the architectural bottleneck.

**Predicted delta.** 6–15% on val_avg/mae_surf_p; the gain comes from improved
surface representation. The risk is increased VRAM usage from the additional
attention computation.

**Specific code change (train.py).** Add a `HierarchicalPhysicsAttention` module:
```python
class HierarchicalPhysicsAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, coarse_slices=32, fine_slices=64):
        super().__init__()
        self.coarse_attn = PhysicsAttention(dim, heads, dim_head, dropout, coarse_slices)
        self.fine_attn   = PhysicsAttention(dim, heads, dim_head, dropout, fine_slices)
        self.combine     = nn.Linear(2 * dim, dim)

    def forward(self, x, is_surface=None):
        coarse_out = self.coarse_attn(x)  # full-mesh attention
        # Fine attention: run on all nodes but weight by is_surface
        fine_out = self.fine_attn(x)
        return self.combine(torch.cat([coarse_out, fine_out], dim=-1))
```
This version runs fine attention on all nodes but uses additional slices for
finer partitioning. An alternative: mask attention to surface-only nodes before
computing fine_out, then broadcast back. The simpler version above avoids masking
complexity and should be tried first.

**VRAM note.** Two attention modules increase VRAM. With slice_num=(32, 64) the
additional overhead is ~2x the current PhysicsAttention memory, which should fit
within 96GB at batch_size=4.

**Risk / failure mode.** The `combine` linear layer may learn to ignore one of the
two branches. Add a gate: `out = coarse_out + sigmoid(gate) * fine_out` where
gate is a learned scalar. Also, if `is_surface` is not threaded into
HierarchicalPhysicsAttention, the fine branch has no information about which nodes
are surface nodes — it must infer this from the features alone (which is possible
since `x[:, :, 12]` is the is_surface flag).

**Literature anchor.** Erwin hierarchical transformer for irregular meshes
(arXiv 2502.17019). PointNet++ multi-scale grouping (Qi et al., NeurIPS 2017).
Hierarchical neural operators (Herde et al., "Poseidon: Efficient Foundation Models
for PDEs," arXiv 2405.19101).

---

## H11 — Domain-Adversarial Training for Geometry Generalization

**Statement.** Add a gradient reversal layer and domain classifier that predicts
which domain (raceCar single, raceCar tandem, cruise) a sample came from, with the
main model trained to fool this classifier. This encourages the shared representation
to be domain-invariant.

**Why it might help.** The three training domains have systematically different
geometry (ground effect vs. freestream), AoA ranges, and Re distributions. The
geometry-OOD val splits test generalization to unseen front-foil camber within
raceCar and cruise separately — meaning the model must generalize within a domain to
a new geometry. A domain-invariant representation reduces the model's ability to
shortcut by memorizing domain-specific patterns. This is the standard DANN (Domain
Adversarial Neural Network) approach applied to physics surrogate generalization.

**Predicted delta.** 3–10% on val_geom_camber splits; potential regression on
val_single_in_dist if the domain classifier over-regularizes. Net effect on
val_avg is uncertain but worth testing.

**Specific code change (train.py).** Add a gradient reversal function and domain
classifier:
```python
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, grad): return -ctx.alpha * grad, None

class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim, n_domains=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(),
            nn.Linear(64, n_domains),
        )
    def forward(self, pooled_feat, alpha):
        rev = GradReverse.apply(pooled_feat, alpha)
        return self.net(rev)
```
In the training loop, pool the penultimate hidden state (mean over non-padding nodes),
pass through DomainClassifier, compute cross-entropy loss against domain labels.
Domain labels: identify single-foil vs. tandem samples from `x[:, 0, 22]` (gap=0 →
single-foil), and cruise vs. raceCar from `x[:, 0, 18]` (AoA foil 2, positive →
cruise). Total loss: `loss + lambda_adv * domain_loss`.

**Risk / failure mode.** The gradient reversal alpha schedule matters — start at
alpha=0 and linearly ramp to 1.0 over the first 30% of training (per DANN paper).
If domain labels are noisy or the domain classifier collapses, the adversarial
signal is useless. Also: requires domain labels that are derivable from the input
features in `train.py` without touching data/. The derivation from `x` dims 18–23
is sufficient.

**Literature anchor.** DANN (Ganin et al., "Domain-Adversarial Training of Neural
Networks," JMLR 2016). Application to fluid simulations: "Deep Learning for Real-
Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning"... (redirect)
— more relevant: domain adaptation for aerodynamic shape optimization (transfer
learning across Re regimes in CFD).

---

## H12 — Curriculum Learning Ordered by Reynolds Number (Easy-to-Hard)

**Statement.** Train with a curriculum that starts on low-Re samples (smaller
velocity magnitudes, simpler flow structure) and gradually introduces high-Re samples
as training progresses, rather than the current balanced random sampling.

**Why it might help.** High-Re flows involve turbulent structures, boundary-layer
separation, and strong pressure gradients that are hard to predict. Starting on
simpler low-Re flows may allow the model to learn the basic geometric and boundary
conditions first, before encountering the harder regime. Curriculum learning has
strong empirical support in deep learning (Bengio et al., ICML 2009) and has been
applied in weather forecasting models (e.g., NowcastNet uses curriculum over forecast
horizon). For CFD surrogates, the curriculum along Re is a natural "difficulty"
ordering because the physics complexity increases monotonically.

**Predicted delta.** 3–8% on val_avg/mae_surf_p, primarily via improved convergence
stability. The effect is most likely to show up in training curves (faster convergence)
rather than final accuracy.

**Specific code change (train.py).** Add Re-stratified curriculum scheduling. Since
data/ is read-only, implement in train.py via a custom `SubsetWeightedSampler`:
```python
# After loading train_ds and sample_weights:
log_re_values = []  # collect log(Re) for each training sample
for i in range(len(train_ds)):
    x_i, _, _ = train_ds[i]
    log_re_values.append(x_i[0, 13].item())  # dim 13 = log(Re), raw
log_re_arr = torch.tensor(log_re_values)

# In the training loop, compute epoch-dependent weights:
def get_curriculum_weights(epoch, max_epoch, base_weights, log_re_arr):
    frac = epoch / max_epoch  # 0 → 1
    # Sigmoid re-weighting: early training down-weights high Re
    re_threshold = log_re_arr.min() + frac * (log_re_arr.max() - log_re_arr.min())
    re_weight = torch.sigmoid((log_re_arr - re_threshold) * (-5.0) + 3.0)
    return base_weights * (re_weight + 0.1)  # 0.1 floor to not starve high-Re
```
Recreate the `WeightedRandomSampler` at the start of each epoch with updated weights.
This requires recreating `train_loader` each epoch, which adds ~1s overhead.

**Risk / failure mode.** Loading all log(Re) values at startup (length ~1500 samples)
is fast. Recreating the DataLoader each epoch adds ~1s/epoch overhead. The curriculum
may be too aggressive — ensure the high-Re samples are never fully excluded (the
`+ 0.1` floor prevents zero weight). Verify that the baseline `sample_weights`
domain balance is preserved within the curriculum weighting.

**Literature anchor.** Curriculum learning (Bengio et al., "Curriculum Learning,"
ICML 2009). Re-stratified sampling for CFD surrogates (see TandemFoilSet data splits
design in data/SPLITS.md, which already uses stratified Re for the val_re_rand split).

---

## H13 — Log-Scale Target Transformation for Pressure

**Statement.** For the pressure channel only, apply a signed log transform
`p_transformed = sign(p) * log(1 + |p| / scale)` in the loss, effectively training
the model to predict log-pressure rather than linear pressure. Inverse-transform
predictions before MAE computation.

**Why it might help.** Surface pressure MAE is dominated by high-Re samples where
|p| can exceed 10,000. In normalized space, even after global normalization, the
pressure residuals from high-Re samples are orders of magnitude larger than those
from low-Re samples. A log transform compresses this range: for high-Re samples, a
relative error of 1% in pressure contributes the same loss as a 1% error in a low-Re
case, rather than contributing 100x more. This is the ML analogue of the standard
numerical CFD practice of solving for log-pressure in highly compressible flows to
improve conditioning. The transform is applied only to the loss computation — the
model still predicts in (log-transformed) normalized space.

**Predicted delta.** 5–15% on val_avg/mae_surf_p; the effect should particularly
improve val_re_rand and val_geom_camber_rc where high-Re samples are most prevalent.

**Specific code change (train.py).** Define the transform and its inverse:
```python
def signed_log(x, scale=1.0):
    return x.sign() * torch.log1p(x.abs() / scale)

def inv_signed_log(x, scale=1.0):
    return x.sign() * (torch.expm1(x.abs()) * scale)
```
In the training loop, after computing `y_norm` and `pred`:
```python
# Apply signed log to pressure channel (index 2) only
y_log = y_norm.clone()
y_log[:, :, 2] = signed_log(y_norm[:, :, 2], scale=1.0)
pred_log = pred.clone()
pred_log[:, :, 2] = signed_log(pred[:, :, 2], scale=1.0)
sq_err = (pred_log - y_log) ** 2
```
At eval time in `evaluate_split`, do NOT apply the transform — the model still
predicts in normalized linear space, and the evaluation uses raw denormalized
predictions. The transform is purely a loss-space reweighting.

**Risk / failure mode.** The `scale` parameter controls the transition from linear
to log behavior. scale=1.0 means the log regime kicks in when |y_norm| > 1, which
is appropriate for normalized targets. If scale is too large, the transform reduces
to linear and has no effect. If too small, gradients become extremely small for large
pressure values. Ablate scale in {0.5, 1.0, 2.0}. Also: the signed log is
applied to the NORMALIZED pressure (after global normalization) — the inflection
point must be checked against the actual distribution of y_norm[:, :, 2].

**Literature anchor.** Log-transformed regression for heavy-tailed targets (common
in financial ML and meteorological prediction). Huber loss as a robust alternative
(robust to large residuals, but does not directly address scale variation). The signed
softplus / log1p transform is used in OpenFold for log-scale distance prediction
in protein structure modelling.

---

## H14 — NACA Geometry Embedding as Explicit Conditioning

**Statement.** Add a dedicated 8-dimensional learned embedding of the NACA parameters
(camber, position, thickness for both foils) and gap/stagger, projected to the
model's hidden dimension and added to every node's embedding as a global geometry
bias.

**Why it might help.** The NACA parameters (dims 15–21) and gap/stagger (dims 22–23)
are currently treated as additional input features averaged with local geometry
features through the same MLP. But these are fundamentally different: NACA parameters
describe the global geometry of the foil, not any local property of a mesh node. They
are the same for every node in a given sample. An explicit embedding that is added to
every node (like a "geometry token" or "condition token" in transformers) gives the
model a clean architectural separation between global geometry conditioning and local
mesh processing, similar to how class tokens work in ViT. For geometry-OOD splits,
this may improve generalization by learning a compact, reusable geometry code.

**Predicted delta.** 3–8% on val_geom_camber splits. The effect on val_re_rand
should be neutral (Re is separately handled by dim 13).

**Specific code change (train.py).** Add a geometry conditioner:
```python
class GeometryEmbedding(nn.Module):
    def __init__(self, geom_dim=9, hidden_dim=128):
        super().__init__()
        # 9 dims: NACA foil1 (3) + NACA foil2 (3) + AoA foil2 (1) + gap (1) + stagger (1)
        self.embed = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, x_norm):
        # x_norm: [B, N, 24]; take per-sample global params from first node
        geom = x_norm[:, 0, 15:24]  # [B, 9]: NACA1, NACA2, AoA2, gap, stagger
        return self.embed(geom).unsqueeze(1)  # [B, 1, H] broadcast over N
```
In `Transolver.forward`:
```python
fx = self.preprocess(x) + self.placeholder[None, None, :] + self.geom_embed(x)
```
The geometry embedding is broadcast over all N nodes. This adds ~33K parameters.

**Risk / failure mode.** If NACA params for single-foil samples are (0, 0, 0) for
foil 2, the embedding must not confuse "zero NACA params" with "identical foil 2";
this is handled correctly since the embedding input includes gap=0 and stagger=0
which identify single-foil samples. AoA foil 1 (dim 14) is also relevant but
already in the local node features; including it in the geometry embedding would
create a duplicate — keep it in the local features only.

**Literature anchor.** Geometry tokens in neural operators for parametric PDEs
(DeepONet with input-space conditioning, Lu et al., Nature Machine Intelligence 2021).
Parameter conditioning in GINO (arXiv 2309.00583), which concatenates global SDF
features with local mesh features. Hypernetworks for geometry-conditioned networks
(Ha et al., "HyperNetworks," ICLR 2017).

---

## H15 — Gradient Clipping Tightened + AdamW Decoupled Decay Only on Weights

**Statement.** Add gradient norm clipping (`max_norm=1.0`) and ensure AdamW
weight decay is applied only to weight matrices (not biases and LayerNorm parameters),
following the GPT-2 training recipe. These are low-risk, high-value training
stabilization changes.

**Why it might help.** The current optimizer applies weight decay uniformly to all
parameters, including LayerNorm scale/bias and attention temperature parameters.
Decaying LayerNorm parameters is known to interfere with normalization dynamics
(it pulls scale toward zero). The current code has no gradient clipping, which means
large batches with extreme pressure gradients can produce gradient spikes that destabilize
training. Both changes improve training stability at zero architectural cost,
following best practice from large transformer training (GPT-2, BERT). The effect
on a 5-layer model is modest but consistent.

**Predicted delta.** 2–5% reduction in val_avg/mae_surf_p through more stable
convergence. The effect is most visible in early training curves (lower variance
across runs) and in seeds that currently diverge.

**Specific code change (train.py).** Replace the optimizer initialization:
```python
# Separate parameter groups: decay weights only, not biases/LayerNorm
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim < 2 or 'bias' in name or 'ln_' in name or 'norm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params,    "weight_decay": cfg.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=cfg.lr)
```
Add gradient clipping in the training loop after `loss.backward()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Risk / failure mode.** Gradient clipping with `max_norm=1.0` may be too aggressive
for this model (slows convergence if gradients are normally ~0.5 and clipping rarely
fires, but fine). Check the average gradient norm in the first few epochs; if it is
consistently < 0.5, use `max_norm=5.0`. The parameter group split must correctly
exclude the `temperature` parameters in PhysicsAttention (which are single-element
tensors, ndim=4 but are attention temperatures and should be no_decay — use name
check `'temperature' in name`).

**Literature anchor.** GPT-2 training recipe (Radford et al. 2019; no weight decay
on biases/norms). AdamW decoupled decay (Loshchilov and Hutter, "Decoupled Weight
Decay Regularization," ICLR 2019). Gradient clipping in transformer training
(Pascanu et al., "On the difficulty of training recurrent neural networks," ICML 2013).

---

## Priority Ordering and Decision Tree

### Tier 1 (highest expected gain, low implementation risk)

1. **H4** — Per-channel loss weighting (p upweighted): 2 lines of code, direct
   alignment with primary metric. If it works, combine with H5.
2. **H15** — Gradient clipping + selective weight decay: 10 lines of code, zero
   architectural change, known best practice.
3. **H2** — Per-sample output scale normalization: targets the 40x Re magnitude
   span that dominates the loss. Direct fix for a diagnosed pathology.

### Tier 2 (medium implementation, strong theoretical motivation)

4. **H1** — FiLM conditioning: directly addresses geometry-OOD failure mode. External
   evidence from BlendedNet++ is strong.
5. **H6** — Two-stage training: analogous to successful fine-tuning in NLP/vision.
   Adds zero new architecture.
6. **H7** — Re-conditioned residual: 4 new parameters, mechanistically targets Re
   generalization.

### Tier 3 (larger implementation, higher upside, higher risk)

7. **H3** — Separate surface/volume decoder heads: directly targets the primary
   metric through architectural alignment.
8. **H10** — Hierarchical two-level PhysicsAttention: largest architectural change,
   highest upside for surface-specific accuracy.
9. **H13** — Log-scale pressure loss: targets the heavy tail in pressure distribution.

### Tier 4 (data/training regime changes)

10. **H8** — AoA reflection symmetry augmentation: needs careful feature reflection
    mapping; start with a debug run.
11. **H12** — Curriculum learning by Re: simple to implement, benefits likely show
    in convergence speed not final accuracy.
12. **H9** — Relative position features: adds input dims, needs normalization update.
13. **H11** — Domain-adversarial training: most complex, clearest theoretical
    motivation for geometry generalization.
14. **H14** — NACA geometry embedding: simple but overlaps with FiLM (H1).

### Experiment tree

```
START
├── H4 + H15 (cheap, parallel)
│   ├── H4 wins (p↓): add H4 to baseline, next try H1 (FiLM) on top
│   ├── H15 wins (stability↑): add H15 to baseline, next try H2 or H6
│   ├── Both win: merge both, try H2 + H1 together
│   └── Neither wins: H4 tells us channel coupling is tight → try H3 (separate heads)
│
├── H2 (Re-magnitude normalization)
│   ├── Wins: signals training is Re-magnitude dominated → also try H7 (Re residual)
│   └── Loses: normalization not the bottleneck → ruled out; move to H1
│
├── H1 (FiLM, after baseline stabilized)
│   ├── Wins on geom-OOD splits: architecture bottleneck confirmed → try H10
│   └── Wins only on in-dist: Re-conditioning is stronger signal → focus on H7/H12
│
└── H10 (hierarchical attention, after H1/H2 integrated)
    ├── Wins: tier-shift confirmed, surface-specialized attention works → extend
    └── Fails: slice mechanism is not the bottleneck → revisit data/augmentation (H8)
```

### Stop conditions

- **Stop H1 (FiLM)** if val_geom_camber splits do not improve by >2% over H1-inclusive
  baseline after 30 epochs. The mechanism is not active.
- **Stop H2 (per-sample scale)** if val_re_rand does not improve. The Re-magnitude
  hypothesis is falsified.
- **Stop H10 (hierarchical)** if VRAM exceeds 80GB or val_avg/mae_surf_p regresses
  vs. baseline at epoch 10 check.
- **Stop curriculum (H12)** if training loss curve is not smoother than baseline
  by epoch 5. Curriculum learning shows up in early training behavior first.
