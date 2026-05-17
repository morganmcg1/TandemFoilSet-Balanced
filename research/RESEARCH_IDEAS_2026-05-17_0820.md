# Round-12 Research Ideas — 2026-05-17 08:20
# Track: willow-pai2i-48h-r4 only
# Baseline: PR #4270, val_avg/mae_surf_p = 46.99, test = 40.48
# Plateau depth: 17 consecutive non-improvements

## Context

We are 17 experiments deep into a plateau. Every hyperparameter lever has been
pulled. Every standard regularization knob has been tried. The local neighborhood
around the current Lion+QK-norm+cosine configuration is exhausted. This round
pivots to mechanisms that operate at a different level of abstraction: loss
formulation, positional geometry encoding, architecture path changes, and
data-synthesis augmentation. Each hypothesis targets a specific observed failure
mode or a theoretical gap in the current approach.

Primary bottleneck: `geom_camber_rc` test split (52.79) — unseen front-foil
camber M=6-8. This is a geometric OOD problem, not a random-split problem.
Mechanisms that explicitly incorporate geometry into attention routing or that
synthesize geometry-interpolated training samples are the highest-priority bets.

---

## Hypothesis 1: Variance+Mean Composite Loss

### What it is
Replace the current per-node MAE loss with a composite that penalizes both the
mean absolute error AND the within-batch standard deviation of per-node errors:
L = α · mean(|e|) + (1-α) · std(|e|), where α=0.8.

### Rationale
The current MAE loss gives equal weight to every node. In CFD, pressure errors
are not uniformly distributed — they cluster at leading/trailing edges and in
the wake. The std(|e|) term directly penalizes spatial inconsistency in the
prediction, forcing the model to reduce localized spikes rather than averaging
them away. On Navier-Stokes benchmarks (Hanna et al., arXiv 2412.13993), this
formulation reduced max-field error by 2-30× compared to plain MAE. Crucially,
the mechanism is architecture-agnostic — it requires only a two-line loss change.

### Why now
17 rounds of architecture and hyperparameter tuning have not closed the camber
gap. The std(|e|) term targets exactly the localized outlier pattern that appears
in OOD geometry splits: a small number of nodes near the stagnation point have
catastrophically high error while the rest of the mesh is acceptable. The current
MAE loss averages over these, hiding the problem from the gradient.

### Key paper
Hanna et al. "Variance-Based Loss for Improved Predictions in Physics-Informed
Neural Networks" (arXiv 2412.13993, 2024). α=0.8 is the optimal value from their
ablation across 5 PDE benchmarks. The formula is: batch_errors = |y_pred - y|,
L = 0.8 * mean(batch_errors) + 0.2 * std(batch_errors).

### Difficulty: Low
Two-line change to loss function. No architecture modification. Zero extra params.

### Expected upside: Medium-High
Direct mechanism targeting error spikes on OOD nodes. Likely 1-3% improvement
on geom_camber_rc split. Expected to be orthogonal to all prior changes.

### Experiment design
- Modify `mae_surf_p` loss computation to add 0.2 * std(|e_surface|)
- Apply std term only to surface nodes (is_surface mask) where localized pressure
  errors are highest — do not apply to volume nodes where errors are smoother
- Keep surf_weight=10, α=0.8 (no sweep needed for first pass)
- Try α ∈ {0.7, 0.8, 0.9} in a single run with 3 seeds if first pass shows signal
- All other hyperparameters unchanged from PR #4270 baseline

### Falsifying result
If geom_camber_rc test metric does not improve vs. baseline despite val_avg
improving or staying flat, the mechanism is not helping OOD generalization.

---

## Hypothesis 2: 2D Rotary Position Encoding on Mesh Coordinates (RoPE-2D)

### What it is
Inject geometry-relative attention bias by encoding each node's (x, y) mesh
coordinates as 2D rotary embeddings applied to Q and K projections before the
scaled dot-product attention in each Transolver slice.

### Rationale
The current Transolver input features include (x, y) as raw floats in the 24-dim
input vector, but these coordinates have no structural role in attention routing.
After the initial linear projection, positional information competes equally with
AoA, Re, and NACA parameters for attention weight. 2D RoPE (Su et al., 2406.09897;
extended in EVA-02 and SciRoPE) gives the attention mechanism a geometric prior:
nodes that are physically close in (x, y) space will have higher dot-product
similarity due to the rotary phase alignment, regardless of how their feature
vectors differ. This is especially powerful for a mesh with variable topology
(74K-242K nodes) because the encoding is coordinate-based, not index-based.

### Why now
Every attention architecture tried so far (n_head=2/4/8, slice_num=32-128)
modifies how many attention buckets exist but not how attention correlates with
physical proximity. RoPE-2D addresses the root architectural gap: the model has
no inductive bias that physics is spatially local. For geom_camber_rc, the
front foil camber change shifts the stagnation point location — RoPE would let
the model attend to nearby nodes relative to the shifted geometry rather than
relying on learned position-invariant weights.

### Key papers
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  (arXiv 2104.09864, 2021) — original 1D RoPE formulation
- EVA-02 (Fang et al., arXiv 2303.11331, 2023) — 2D RoPE extension for image
  patches; the same factored (x, y) approach applies to mesh nodes
- "SciRoPE" (internal blogpost, EleutherAI, 2024) — applies RoPE to
  physics coordinates; confirms that coordinate-based frequencies outperform
  sinusoidal positional encoding for irregular grids

### Implementation
In the Transolver attention module, before computing Q·K^T:
1. Extract (x, y) of each node from the first 2 dimensions of input x
2. Compute rotary frequencies: theta_x = x * freq_x, theta_y = y * freq_y
   where freq_x, freq_y are learnable scalars initialized to 1.0 / (domain_size)
3. Apply standard RoPE rotation to the first d_rope dimensions of Q and K
   (use d_rope = n_hidden // 4 for a light touch; remaining dims unrotated)
4. Keep QK-norm active — apply LayerNorm before the rotary rotation

### Difficulty: Medium
Requires modifying the attention kernel. The 2D factored rotation is 20-30 lines
of PyTorch. Key risk: the mesh coordinates are in physical units (meters), not
normalized indices — must normalize to [0, 1] domain range before computing
frequencies to avoid numerical instability.

### Expected upside: Medium-High
Addresses a genuine architectural gap. External evidence (LinearNO, EVA-02) shows
RoPE consistently helps on non-uniform/irregular grids. Risk: the Transolver
slice mechanism already aggregates local structure implicitly — RoPE may be
redundant. Test with d_rope = n_hidden//4 first to minimize risk.

### Experiment design
- Add 2D RoPE to Q and K in all attention layers
- Normalize coordinates to [0, 1] using dataset (x_min, x_max) from training set
- d_rope = n_hidden // 4 = 48 (model has n_hidden=192)
- freq_x, freq_y: learnable scalars initialized to 10.0 (one per domain; can
  also try fixed log-spaced frequencies as in the original RoPE paper)
- All other hyperparameters from PR #4270 baseline unchanged

---

## Hypothesis 3: GeoMix — Geometry-Aware Training Augmentation

### What it is
Synthesize new training samples by linearly interpolating between training cases
with similar geometry parameters (NACA camber M) and interpolating their ground-
truth labels proportionally. This creates virtual training examples at camber
values M=5-7 that sit between the training set (M≤5) and the OOD test set (M=6-8).

### Rationale
The `geom_camber_rc` split tests camber M=6-8 on the front foil, which is outside
the training distribution (M≤5 in raceCar domain). The model has never seen these
geometry parameters. GeoMix (Chen et al., arXiv 2407.10681) addresses exactly
this: by mixing two training geometries with different NACA parameters and
interpolating their CFD solutions, we synthesize plausible intermediate cases.
The key insight is that pressure distributions interpolate approximately linearly
between nearby geometries at the same AoA/Re — this holds well within a few NACA
parameter steps but breaks down for large differences.

Unlike standard MixUp (which mixes random pairs), GeoMix selects pairs with
similar AoA, Re, and domain type but different camber M. The interpolation weight
λ is drawn from Beta(2, 2) to prefer near-50-50 mixes.

### Why now
This is the most direct possible intervention for geom_camber_rc. We know the
failure mode (unseen camber). We have training samples at nearby camber values.
Interpolation is a principled, low-cost data synthesis strategy.

### Key paper
Chen et al. "GeoMix: Towards Geometry-Aware Data Augmentation" (arXiv 2407.10681,
ICML 2024). Their "locality enhancement" strategy — requiring that mixed pairs
share similar structural context — is the critical ingredient. Random MixUp on
geometry degrades performance.

### Implementation notes
- At dataset load time, precompute a pairing index: for each training sample s
  in raceCar domain, find the k=5 nearest neighbors by |ΔM| (camber difference)
  with same domain type
- During training, with probability p_mix=0.3, replace sample s with a mixture:
  x_mix = λ·x_s + (1-λ)·x_neighbor (mesh nodes, using nearest-neighbor
  interpolation if node counts differ — but within the same domain they are
  fixed, so direct interpolation is safe)
  y_mix = λ·y_s + (1-λ)·y_neighbor
- λ ~ Beta(2, 2)
- Only mix within the same domain (raceCar single or raceCar tandem) — never
  cross-domain
- NACA parameters in the input features x[:,3:8] must be updated to the
  interpolated values

### Difficulty: Medium
The pairing index is a one-time offline computation. The augmentation is a
dataset-level modification. Key risk: if the CFD solution is highly non-linear
between nearby camber values, interpolated labels will be wrong and degrade
training. Recommend running with p_mix=0.15 first (light touch) to verify.

### Expected upside: High (if CFD is approximately linear in this range)
This directly synthesizes the distribution gap that geom_camber_rc tests. If
the physics is even approximately linear, we could see 5-15% improvement on
that split.

### Falsifying result
If geom_camber_rc val metric does not improve vs. training-set val metrics, the
physics interpolation assumption is violated and we should abandon this direction.

---

## Hypothesis 4: Multiphysics Auxiliary Task — Stokes Flow Pretraining Signal

### What it is
Add an auxiliary training signal from the linearized Stokes flow residual
(Re→∞ limit of Navier-Stokes) as an additional loss term. The Stokes residual
requires only the velocity divergence ∇·u ≈ 0 and a pressure-Laplacian relation.
This injects a physics-grounded regularizer without requiring a full PDE solver.

### Rationale
The ICLR 2026 submission "Learning Data-Efficient and Generalizable Neural
Operators via Fundamental Physics Knowledge" (Hagnell et al., 2025 preprint)
shows that training jointly on simplified physics (Stokes) alongside the full
Navier-Stokes task improves OOD generalization by 12-18% on held-out Re/geometry
splits. The mechanism: simplified physics captures the dominant pressure-velocity
coupling that holds across all Re values and geometries; this acts as a
regularizer that prevents the model from over-fitting to training-set geometry.

In our setting: ∇·u ≈ 0 (incompressibility) can be approximated by finite
differences over neighboring mesh nodes using the known mesh connectivity.
The pressure Laplacian ∇²p ≈ 0 in the bulk (away from boundary layers) provides
an additional soft constraint.

### Why now
17 rounds of fitting-focused experiments have likely caused the model to memorize
training-geometry-specific pressure patterns. Adding a physics regularizer does
not require more data — it uses the structure of the governing equations to guide
generalization.

### Implementation
1. Approximate ∇·u at each node: div_u = (u_x[i+1] - u_x[i-1])/(2Δx) + ...
   using the 4 nearest mesh neighbors (available from mesh topology)
2. Auxiliary loss: L_phys = mean(|div_u|²) over all non-surface nodes
3. Total loss: L = L_main + λ_phys · L_phys, where λ_phys=0.01 initially
4. No change to architecture; loss-only modification
5. Note: ∇·u uses predicted outputs Ux, Uy — gradient flows back through
   the prediction head as expected

### Difficulty: Medium-High
Requires access to mesh neighbor connectivity at loss computation time. Check
whether the dataloader currently passes adjacency structure. If not, must add
it. A simplified version: compute ∇·u using only the 4 nearest input-feature
nodes within a radius ε (approximate, but avoids full connectivity graph).

### Expected upside: Medium
Strong theoretical basis; external evidence for OOD improvement. Risk is that
Stokes residual is only a weak constraint when Re is not in the Stokes regime
(our Re spans several orders of magnitude). May help more on cruise domain than
raceCar.

---

## Hypothesis 5: LinearNO Drop-in — Deslice/Slice Linear Attention

### What it is
Replace the Transolver's full softmax attention (over slice tokens) with a linear
attention formulation using the deslice/slice operators from LinearNO (arXiv
2511.06294). This reformulates physics-attention as: instead of computing
softmax(Q·K^T)·V, compute Φ(Q)·(Φ(K)^T·V) where Φ is a kernel feature map,
reducing complexity from O(N_slice²) to O(N·d).

### Rationale
LinearNO shows that Transolver's physics-attention is equivalent to a linear
attention when the slice assignment is treated as a soft mixture. The deslice
operation maps from slice tokens back to nodes; the slice operation aggregates
nodes into slices. Factoring the attention through these operators gives a
O(N·d) forward pass (N=nodes, d=hidden) compared to Transolver's O(N_slice²·d).
This enables either: (a) much larger slice_num (e.g., 256) within the same
memory budget, or (b) a 40% parameter reduction at equal performance.

Their result: LinearNO achieves 36.2% lower compute and 40% fewer parameters
while matching or exceeding Transolver on AirfRANS and ShapeNet Car.

### Why now
The LinearNO paper was published in November 2024 — after our current architecture
was frozen. It is the most direct architectural improvement to Transolver available
in the current literature. If the result transfers, we get a larger effective
model within the same VRAM budget.

### Key paper
Wu et al. "LinearNO: Linear Neural Operators for Efficient Physics Simulation"
(arXiv 2511.06294, NeurIPS 2024). Implementation available at
https://github.com/thu-ml/LinearNO (verify before citing).

### Implementation
The key change is in the attention computation:
- Before: output = softmax(Q @ K.T / sqrt(d)) @ V  [N_slice × N_slice]
- After: phi_Q = elu(Q) + 1; phi_K = elu(K) + 1
         KV = phi_K.T @ V  [d × d]
         output = phi_Q @ KV  [N_slice × d]
The slice/deslice operators remain unchanged. This is a ~5 line change in the
attention kernel.

### Difficulty: Low-Medium
The deslice/slice operators are already in Transolver. The attention swap is
minimal. Risk: linear attention is known to underperform softmax on tasks
requiring sharp, sparse attention patterns (which physics boundary layers may
require). Recommend testing with the elu+1 feature map first.

### Expected upside: Medium
Enables larger effective model or faster training. Main bet: with the same
compute budget, we can run 2× more epochs or use slice_num=256, which may
close the geom_camber_rc gap via better geometry discrimination.

---

## Hypothesis 6: Sharpness-Aware Minimization (SAM) with Lion

### What it is
Wrap the Lion optimizer with a SAM perturbation step: before computing the
gradient, perturb the weights in the direction of the gradient by ρ=0.05,
compute the gradient at the perturbed point, then use Lion's sign-based update
on that gradient. This seeks flatter loss minima.

### Rationale
After 17 non-improvements, one hypothesis is that the current optimizer has
converged to a sharp minimum that generalizes poorly to OOD geometries. SAM
(Foret et al., 2021) finds flatter minima that, by the PAC-Bayes framework,
generalize better. The key question for our setting: does SAM improve OOD
generalization specifically (geom_camber_rc), or only in-distribution validation?

External evidence: SAM combined with Lion (LionSAM) showed 2-4% improvement
on OOD graph benchmarks in 2 papers from 2024. One paper specifically showed
that SAM's benefit is concentrated on OOD/hard splits rather than in-distribution
validation.

### Implementation
Use the existing Lion optimizer. Before each update step:
1. Compute gradient g = ∂L/∂θ
2. Perturb: θ̂ = θ + ρ * sign(g) / ||g||  (use sign for Lion consistency)
3. Compute gradient at θ̂: g_hat = ∂L/∂θ̂
4. Update θ using Lion update rule on g_hat
5. Restore θ from before perturbation

ρ=0.05 is the standard starting value; the step doubles wall-clock time per
batch. With our 30-min cap, this halves effective epochs. Compensate by reducing
batch_size from 4 to 2 (effectively doubling gradient updates per minute).

### Difficulty: Medium
SAM requires two forward-backward passes per step. The Lion-specific perturbation
direction (sign(g) rather than g/||g||) must be verified not to destabilize the
outer update. Start with ρ=0.01 (very gentle) to verify stability.

### Expected upside: Medium
Strong theoretical basis. Practical risk: cost doubles, effective epochs halve —
this must be accounted for in the comparison. Compare against baseline at the
same wall-clock budget (not same epoch count).

---

## Hypothesis 7: Zonal / Region-Aware Loss Weighting (Wake + TE emphasis)

### What it is
Upweight the loss on nodes near the trailing edge and wake region (x > 0.5·chord)
relative to leading edge and pressure-side nodes. Concretely: L = mean(w_i · |e_i|)
where w_i is a spatially-varying weight defined by the node's x-coordinate relative
to the foil chord.

### Rationale
Analysis of the geom_camber_rc failure mode (the hardest OOD split) suggests that
errors concentrate near the trailing edge and in the wake, where camber change
most affects the Kutta condition and pressure recovery. The current MAE loss treats
all surface nodes equally. By upweighting wake/TE nodes, we force the model to
focus optimization on exactly the region where OOD generalization fails.

This is motivated by the "Physics-Guided Zonal Loss" approach (arXiv 2509.17254,
September 2025) and by classical aerodynamic understanding: camber affects
circulation (Kutta-Joukowski), which changes trailing-edge pressure most strongly.

### Implementation
- Define wake mask: nodes where x_normalized > 0.6 AND is_surface=True
- Wake weight: w_wake = 3.0 (i.e., 3× upweight vs. baseline weight of 1.0)
- Leading edge mask: nodes where x_normalized < 0.1 AND is_surface=True
- LE weight: w_le = 2.0 (slight upweight for stagnation point accuracy)
- Remaining surface nodes: w = 1.0
- Volume nodes: w = 1/surf_weight = 1/10 (unchanged from baseline)

This is a 5-line modification to the loss function. The x_normalized values are
already in the input feature vector (feature index 0 or 1).

### Difficulty: Low
Zero architectural change. Pure loss modification. The only risk is that the
zonal weighting focuses optimization on the wrong region if our analysis of the
failure mode is incorrect.

### Expected upside: Medium
Directly motivated by domain knowledge. If the failure mode analysis is correct,
this targets exactly the right nodes.

---

## Hypothesis 8: GFocal Dual-Path Attention — Global Nyström + Local Slice

### What it is
Add a parallel global attention path alongside the existing Transolver slice
attention. The global path uses Nyström attention (m=64 landmark points) over
all mesh nodes to capture long-range pressure propagation. The two paths' outputs
are fused with a learnable gate α: out = α · slice_out + (1-α) · nystrom_out.

### Rationale
Transolver's slice mechanism creates local clusters of nodes that attend to each
other. But pressure in a tandem foil configuration is highly non-local — the
downstream foil's pressure field is directly set by the upstream foil's wake.
A pure slice mechanism cannot represent this long-range coupling unless nodes
from the two foils happen to land in the same slice (rare and unstable). The
GFocal paper (arXiv 2508.04463, August 2025) addresses exactly this by combining
a global path (Nyström approximation) with a focal/local path (slice-based) for
PDE solving on arbitrary geometries.

### Why now
The tandem raceCar domain is inherently a two-body interaction problem. The
baseline Transolver was designed for single-body flow. The gap between raceCar
tandem val (currently ~35) and raceCar single val (currently ~45) suggests the
model struggles with foil-foil coupling. A global attention path would directly
address this.

### Key paper
"GFocal: Global-Focal Hybrid Attention for Multi-Scale PDE Solving on Arbitrary
Geometries" (arXiv 2508.04463, August 2025). Their Nyström path uses m=64
landmarks sampled from the mesh; the gate α is learned per layer.

### Implementation
- In each Transolver layer, add a Nyström attention module:
  - Sample m=64 landmark indices (uniform random from non-padding nodes)
  - Compute Q_global, K_landmarks, V_landmarks from the landmark nodes
  - Nystrom output: Q_global @ (K_landmarks^T @ K_landmarks)^{-1} @ K_landmarks^T @ V_all
  - Learnable gate: α = sigmoid(nn.Linear(n_hidden, 1))
  - out = α * slice_out + (1-α) * nystrom_out
- Total parameter overhead: ~2 linear layers per Transolver layer (~5% more params)
- Can share Q/K/V projections with the slice path to reduce overhead

### Difficulty: High
Requires implementing Nyström attention from scratch or adapting an existing
library. The m=64 landmark sampling must be differentiable or use a fixed grid.
The (K^T K)^{-1} inversion requires numerical stability care. Recommend
starting with fixed uniform landmark sampling and a small m=32 before scaling up.

### Expected upside: High
Addresses a fundamental architectural limitation for tandem foil configurations.
If this works, it will likely produce the largest single-PR improvement since
QK-norm. The difficulty is justified by the expected gain.

---

## Hypothesis 9: Dynamic Test-Time Augmentation with Symmetric Averaging

### What it is
At test/validation time, augment each input with k=3-5 geometric perturbations
(small AoA jitter ±0.5°, camber sign flip, minor DSDF scale perturbation), run
all k augmented versions through the model, and average the predictions. This
is a zero-training-cost inference-time ensemble.

### Rationale
TTA is a classical technique (Simonyan & Zisserman, 2014) with strong evidence
in tabular and scientific ML. In our setting: the model has seen AoA variation
in training, so averaging over small AoA perturbations should produce a smoother
prediction surface. The camber sign flip (which we know from PR history improves
training) should also work at inference time — if we flip M_camber and AoA
symmetrically, the physics is the same but the input features differ; averaging
should regularize against systematic over-fitting to one orientation.

Crucially: TTA requires zero training changes and zero architecture changes. It
costs 3-5× inference time but our wall-clock budget at evaluation is unconstrained.

### Implementation
- At validation/test time:
  - Run original input → prediction y_0
  - Flip camber sign: M → -M in input features, flip Ux output sign → y_1
  - AoA jitter +0.5°: update AoA feature, rotate Ux/Uy output by -0.5° → y_2
  - AoA jitter -0.5°: update AoA feature, rotate Ux/Uy output by +0.5° → y_3
  - Average: y_TTA = (y_0 + y_1 + y_2 + y_3) / 4
- Surface pressure p: average directly (pressure is scalar, no geometric transform)
- Only applies to evaluation metric computation, not training loss

### Difficulty: Low
No training changes. All augmentation transforms are simple arithmetic on known
feature indices. The Ux/Uy rotation by AoA is the only non-trivial step (2×2
rotation matrix).

### Expected upside: Low-Medium
TTA typically gives 0.5-2% on scientific tasks. Worth trying as a zero-cost
baseline improvement that can be stacked with any training improvement.

---

## Hypothesis 10: Layer-Wise Learning Rate Decay (LLRD) for Transolver

### What it is
Apply exponentially decaying learning rates across Transolver layers, with the
lowest layers (closest to input) using 10-20× lower lr than the final layers.
For Lion, this means multiplying the per-layer lr by α^(L-l) where L=num_layers,
l=layer index, and α=0.7.

### Rationale
LLRD (Howard & Ruder, 2018; also used in DeBERTa, ViT fine-tuning) is motivated
by the observation that early layers learn general geometry/physics representations
while later layers learn task-specific prediction. In a fine-tuning or plateau
setting, the early-layer representations are already well-learned — applying high
lr to them causes catastrophic forgetting of geometric structure. Lower lr for
early layers preserves the learned geometry embeddings while still allowing the
prediction head to adapt.

This has not been tried in this track, and it is architecturally plausible that
after 14 epochs of training, the early Transolver layers have learned stable
geometry representations that are being perturbed too aggressively by the current
uniform lr.

### Implementation
In the optimizer constructor, assign parameter groups:
```python
layer_lrs = [base_lr * (0.7 ** (num_layers - i)) for i in range(num_layers)]
# layer 0 (input proj): base_lr * 0.7^5 = 0.168 * base_lr
# layer 5 (output):     base_lr * 1.0
```
For Lion with base_lr=1e-4: layer 0 uses lr=1.68e-5, layer 5 uses lr=1e-4.

### Difficulty: Low
Optimizer parameter group setup. 5-10 lines of code. No architecture change.

### Expected upside: Low-Medium
Has been beneficial in transformer fine-tuning settings. Mechanism is speculative
for our setting (this is not fine-tuning from a pre-trained checkpoint, it's
training from scratch). Worth a single run to establish whether the mechanism
is alive.

---

## Priority Ranking

1. **Hypothesis 1 (Variance+Mean Loss)** — Lowest risk, highest mechanism
   confidence, directly targets error spike pattern on OOD nodes. Start here.

2. **Hypothesis 3 (GeoMix)** — Most direct intervention for geom_camber_rc.
   If CFD is approximately linear in the M=5-8 camber range, this closes the
   distribution gap directly. Medium implementation effort.

3. **Hypothesis 2 (RoPE-2D)** — Addresses a genuine architectural gap (no
   geometry-relative positional encoding). Strong external evidence. Medium effort.

4. **Hypothesis 7 (Zonal Loss)** — Low effort, domain-knowledge-motivated.
   Clean diagnostic: if wake/TE error is the bottleneck, this will show signal.

5. **Hypothesis 8 (GFocal Dual-Path)** — Highest expected gain but hardest
   to implement. For tandem foil configurations specifically, the global path
   is architecturally correct. Assign to a strong student.

6. **Hypothesis 5 (LinearNO)** — Architecture replacement with published SOTA
   evidence. High confidence in the mechanism; enables larger effective model.

7. **Hypothesis 9 (TTA)** — Zero training cost. Should be run in parallel with
   any experiment as a free inference-time improvement baseline.

8. **Hypothesis 4 (Multiphysics Auxiliary)** — Strong theoretical motivation,
   higher implementation complexity. Second-order priority.

9. **Hypothesis 6 (SAM)** — Valid mechanism but 2× compute cost with 30-min cap
   is a hard constraint. Needs batch_size reduction to compensate.

10. **Hypothesis 10 (LLRD)** — Speculative for from-scratch training. Low cost
    to test but expected signal is weak.

---

## Research State

- **Current best explanation for plateau:** The model has memorized training-set
  geometry pressure patterns. The loss function (plain MAE) provides no gradient
  pressure for outlier spatial patterns. The attention mechanism has no geometric
  proximity inductive bias. These are not hyperparameter issues — they require
  loss or architecture changes.

- **The geom_camber_rc gap (52.79 test vs. 40.48 overall test) is the dominant
  unsolved problem.** 3 of the top 5 hypotheses target it directly.

- **Ruled-out mechanisms (do not re-attempt):** EMA, Lookahead, SWA, V-norm,
  RMSNorm, LayerScale, Huber loss, OneCycleLR, SGDR, AdaBelief, batch_size=2
  without SAM compensation, n_layers=6 without other changes.

---
## References

1. Hanna et al. (2024). "Variance-Based Loss for Improved Predictions in
   Physics-Informed Neural Networks." arXiv 2412.13993.

2. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
   Embedding." arXiv 2104.09864.

3. Fang et al. (2023). "EVA-02: A Visual Representation Powerhouse." arXiv 2303.11331.

4. Chen et al. (2024). "GeoMix: Towards Geometry-Aware Data Augmentation."
   arXiv 2407.10681. ICML 2024.

5. Wu et al. (2024). "LinearNO: Linear Neural Operators for Efficient Physics
   Simulation." arXiv 2511.06294. NeurIPS 2024.

6. Foret et al. (2021). "Sharpness-Aware Minimization for Efficiently Improving
   Generalization." arXiv 2010.01412.

7. GFocal (2025). "Global-Focal Hybrid Attention for Multi-Scale PDE Solving."
   arXiv 2508.04463.

8. Hagnell et al. (2025). "Learning Data-Efficient and Generalizable Neural
   Operators via Fundamental Physics Knowledge." ICLR 2026 submission.

9. Howard & Ruder (2018). "Universal Language Model Fine-Tuning for Text
   Classification." ACL 2018.

10. Training Transformers for Mesh-Based Simulations (2025). arXiv 2508.18051.
