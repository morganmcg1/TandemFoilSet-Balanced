# Round-1 Hypothesis Seeds — TandemFoilSet (2026-04-29)

Primary metric: `val_avg/mae_surf_p` (lower is better).
Baseline: Transolver n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2.
All ideas below are NEW — they do not duplicate any round-1 in-flight assignment.

---

## Theme 1: Mesh / Positional Encodings

### H-01: Random Fourier Features on spatial coordinates

**Hypothesis.** The Transolver processes raw (x, z) node positions in dims 0–1 as
part of the 24-feature input. Neural networks are spectrally biased toward
low-frequency functions; the sharp pressure gradients near foil surfaces and
at the boundary-layer edge are high-frequency. Replacing the raw (x, z) pair
with 32-64 sinusoidal Random Fourier Features (RFF) — `[sin(B·pos), cos(B·pos)]`
where B ~ N(0, sigma²) — gives the model pre-computed high-frequency basis
functions. Expected delta: 3–8% drop in `val_avg/mae_surf_p` based on analogous
results in MARIO (NeurIPS 2025) and F-FNO coordinate encoding on irregular grids.

**Implementation knobs.**
- Prepend an `RFFEncoder(in_dim=2, n_freq=32, sigma=1.0)` module before the
  Transolver stack. Input dim grows from 24 to `24 - 2 + 2*n_freq`.
- `sigma` controls the frequency scale; start with sigma=1.0, also try 0.5, 2.0.
- `n_freq` in {16, 32, 64}; 32 is a good first test.
- Apply RFF only to position dims 0–1; keep dims 2–23 unchanged.
- The RFFEncoder is a fixed (non-learned) random projection; set `requires_grad=False`.

**References.**
- Tancik et al. 2020, "Fourier Features Let Networks Learn High Frequency Functions
  in Low Dimensional Domains." arxiv 2006.10739
- Li et al. 2023, "Geometry-Informed Neural Operator" (GINO). arxiv 2309.00583
  — shows coordinate Fourier encoding consistently helps on unstructured meshes.

**Why it fits here.** TandemFoilSet nodes span O(1) in normalized coordinates but
pressure varies at sub-millimeter resolution near surfaces. RFF directly addresses
spectral bias without changing data contract, architecture, or loss.

---

### H-02: Learned sinusoidal positional embedding shared across domains

**Hypothesis.** Rather than fixed random frequencies, learn a small MLP that maps
(x, z) → 16-dim embedding via `sin/cos` basis (GAOT / NeRF-style). The embedding
captures spatial structure in a task-adapted way. For tandem geometry, the
embedding can implicitly encode proximity to foil 1 vs. foil 2 — information the
current features do not carry explicitly. Expected delta: modest but consistent,
especially on the camber-holdout splits where spatial encoding of novel shapes
matters most.

**Implementation knobs.**
- `SinusoidalPE(in_dim=2, out_dim=16, n_freqs=8)`: frequencies initialized as
  geometric sequence [1, 2, 4, ..., 128], then made learnable.
- Concatenate to input features; Transolver input becomes 24 - 2 + 16 = 38 dims.
- Alternatively, add the PE as a residual to the first projection layer.
- Optionally share a single PE module and also apply it as a bias to each
  attention layer's key/query projection.

**References.**
- Mildenhall et al. 2020, "NeRF." arxiv 2003.08934 — sinusoidal PE fundamentals.
- Herde et al. 2024, "Poseidon: Efficient Foundation Models for PDEs." arxiv
  2408.02168 — sinusoidal PE applied to PDE operator learning.

**Why it fits here.** Unlike RFF (fixed), a learned PE can adapt to the specific
frequency content of pressure fields around NACA foils, and shares weights across
all 74K–242K node meshes without architecture changes.

---

## Theme 2: Loss Reformulations

### H-03: Per-sample scale-normalized loss (instance re-weighting)

**Hypothesis.** The per-sample y_std varies ~10x within each split (RC single:
avg 458, max 2,077). A global MSE or MAE loss is dominated by high-Re, high-std
samples. Dividing each sample's prediction error by its own y_std before averaging
(scale-normalized loss) forces the model to be equally accurate in relative terms
across all Re regimes. This should help low-Re generalization and reduce variance
on `val_geom_camber_cruise` where the std range is narrower.

**Implementation knobs.**
- In the training loop, compute `per_sample_std = y[b, mask[b]].std()` per batch
  element, then weight the per-sample loss as `loss_b / per_sample_std`.
- Use this only for the *training loss*; keep val/test MAE in physical units for
  comparability.
- Also try `per_sample_std.clamp(min=eps)` with `eps=1.0` to prevent division
  instability on near-uniform samples.
- Combine with existing `surf_weight` (e.g. 5 or 10) to keep pressure focus.

**References.**
- Standard technique in multi-task / multi-scale regression; discussed in depth in
  the Transolver paper appendix for heterogeneous PDE datasets.
- Rasp & Thuerey 2021, "Data-Driven Medium-Range Weather Prediction." arxiv
  2008.08626 — scale normalization in NWP surrogates.

**Why it fits here.** The dataset analysis explicitly flags this as a first-order
lever (Section "Magnitudes" in DATASET_ANALYSIS.md). Round-1 Huber addresses
outlier robustness; this directly addresses the cross-sample scale disparity.

---

### H-04: Sobolev-style loss — match spatial gradients of pressure on surface

**Hypothesis.** The primary metric is surface pressure MAE, but surface pressure
gradients (dp/dx, dp/dz) determine aerodynamic lift and drag coefficients. A
Sobolev loss `L = L_val + lambda * L_grad` where L_grad is the MAE of finite-
difference pressure gradients on surface nodes will push the model to produce
physically smooth pressure distributions, not just correct pointwise values.
Analogy from Czarnecki et al. 2017: Sobolev training improved generalization by
~30% on smooth regression problems.

**Implementation knobs.**
- Compute `dp_dx = (p[i+1] - p[i-1]) / 2h` along the surface arc using node
  ordering from the `saf` (signed arc-length) feature (dims 2–3).
- `lambda` in {0.01, 0.1, 0.5}; start with 0.1.
- Apply only to surface nodes (`is_surface` mask); volume gradient is noisy.
- L_grad on the pressure channel only (skip Ux, Uy) to keep alignment with
  primary metric.

**References.**
- Czarnecki et al. 2017, "Sobolev Training for Neural Networks." arxiv 1706.04859
- Wandel et al. 2020, "Learning Incompressible Fluid Dynamics from Scratch."
  arxiv 2006.08762 — Sobolev-style penalty for smooth field learning.

**Why it fits here.** Surface nodes are densely sampled along the foil chord; saf
dims provide a natural arc-length ordering for finite-difference stencils. The
metric is purely pointwise but the physical relevance is in the gradient.

---

### H-05: Pressure-velocity consistency soft constraint (Bernoulli)

**Hypothesis.** Along free-stream streamlines, Bernoulli's equation holds:
`p + 0.5*(Ux² + Uy²) = const`. Adding a soft constraint loss that penalizes
deviations from this relation on volume nodes far from the boundary layer will
couple the three output channels and reduce physically implausible predictions.
The model currently predicts (Ux, Uy, p) independently per node; this loss
introduces cross-channel correlation without changing the architecture.

**Implementation knobs.**
- `bernoulli_residual = (p_pred + 0.5*(Ux_pred² + Uy_pred²)) - bernoulli_const`
  where `bernoulli_const` is estimated per-sample from far-field nodes (first 1%
  by distance from foil center, or nodes where `dsdf` dims 4–11 are all near 1.0).
- Add `lambda_b * MSE(bernoulli_residual, 0)` to the training loss.
- Apply only to volume nodes (`~is_surface`).
- `lambda_b` in {0.01, 0.05, 0.1}; start small.
- Denormalize predictions before computing the quadratic Ux²+Uy² term, or handle
  normalization carefully (add a note in the PR).

**References.**
- Raissi et al. 2020, "Physics-Informed Neural Networks." arxiv 1711.10561
- Mao et al. 2020, "Physics-Informed Neural Networks for High-Speed Flows."
  arxiv 1905.04236 — Euler equation constraints on velocity-pressure fields.

**Why it fits here.** The dataset is steady-state 2D CFD at moderate Re; Bernoulli
is valid away from the boundary layer. The overset mesh has a coarse background
zone that is far from the foils — these nodes are ideal for this constraint.

---

## Theme 3: Training Tricks

### H-06: EMA weight averaging with a slow decay

**Hypothesis.** Exponential moving average (EMA) of model weights with a slow
decay (0.9995–0.999) acts as an implicit ensemble, smoothing the loss landscape
without changing training dynamics. At 50 epochs with 1499 training samples per
epoch, EMA over the final 20 epochs corresponds to ~100 effective checkpoints.
Expected delta: 1–3% on `val_avg/mae_surf_p` based on EMA results in diffusion
models and NLP fine-tuning, at essentially zero extra compute.

**Implementation knobs.**
- Maintain a shadow copy of model parameters: `ema_params = deepcopy(model.state_dict())`.
- After each optimizer step: `ema_params[k] = decay * ema_params[k] + (1-decay) * params[k]`.
- Use `decay=0.9995`; start EMA after epoch 5 (let model warm up first).
- At validation time, temporarily load `ema_params` into the model, evaluate,
  then restore the training params.
- Checkpoint the EMA params as the final model.

**References.**
- Polyak & Juditsky 1992 — original iterate averaging.
- Kaddour et al. 2022, "Stop Wasting My Time! Saving Days of ImageNet and BERT
  Training with Latest Weight Averaging." arxiv 2209.14981
- Ho et al. 2020, DDPM — EMA decay 0.9999 in practice.

**Why it fits here.** 30-min wall clock limits effective epochs; EMA extracts
more from each training run. The surface pressure objective is noisy per-batch,
making EMA smoothing especially beneficial.

---

### H-07: Sharpness-Aware Minimization (SAM) for OOD generalization

**Hypothesis.** The hardest generalization axes are `val_geom_camber_rc` and
`val_geom_camber_cruise` — held-out NACA camber slabs. Standard SGD/Adam finds
sharp minima that generalize well within-distribution but poorly to novel shapes.
SAM explicitly minimizes the maximum loss in an epsilon-ball around current
parameters, finding flatter minima. Flat minima generalize better under
distribution shift. Expected delta: 2–5% on camber-holdout splits.

**Implementation knobs.**
- Replace AdamW with `SAM(base_optimizer=AdamW, rho=0.05)`. Use the
  `sam` package or inline the ~20-line SAM implementation.
- Two forward-backward passes per step: perturb, backward, restore, update.
  This doubles effective compute — halve the number of steps or accept 2x runtime.
- `rho=0.05` is standard; also try 0.01, 0.1.
- Adaptive SAM (ASAM) scales rho per-parameter; often better than vanilla SAM.
- With 30-min timeout: reduce batch accumulation or reduce model size slightly
  to stay within wall clock.

**References.**
- Foret et al. 2021, "Sharpness-Aware Minimization." arxiv 2010.01412
- Kwon et al. 2021, "ASAM: Adaptive Sharpness-Aware Minimization." arxiv 2102.11600

**Why it fits here.** The NACA camber holdouts are a textbook distribution-shift
problem. SAM is specifically designed for generalization under shift, and the
compute overhead (~2x) is manageable given 96 GB VRAM and the small model size.

---

### H-08: Cautious AdamW modifier for stable large-lr training

**Hypothesis.** Round-1 nezuko tests lr=1e-3 (2x the default 5e-4). The risk is
noisy updates destabilizing training. The Cautious optimizer modifier masks update
components where the gradient and Adam momentum disagree in sign — effectively
applying a conservative filter at near-zero overhead. This should allow aggressive
learning rates without divergence, potentially combining the speed of lr=1e-3 with
the stability of smaller lr.

**Implementation knobs.**
- Wrap the existing AdamW: for each parameter group, compute
  `mask = (grad * m_t) > 0` where `m_t` is the EMA first moment.
  Apply update only where `mask=True`; zero-out or scale-down the rest.
- This is ~5 lines added to the AdamW step function.
- Set `lr=1e-3` with cosine annealing; compare against nezuko's plain lr=1e-3.
- Reference implementation: https://github.com/kyleliang919/C-Optim

**References.**
- Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code."
  arxiv 2411.16085 (ICLR 2025 Spotlight)

**Why it fits here.** If round-1 nezuko shows that lr=1e-3 improves early training
but diverges late, Cautious AdamW is the minimal fix with essentially zero overhead.

---

## Theme 4: Architectural Variants

### H-09: Factorized attention over physics-informed slices (F-Transolver)

**Hypothesis.** The current Transolver computes full attention over `slice_num=64`
physics-informed tokens. Factorized attention — alternating row/column attention
as in Axial Transformers — reduces O(N²) to O(N√N) and allows doubling the
number of slices to 128 at the same memory cost as the current 64. More slices
mean finer spatial grouping. Unlike frieren (which just increases slice_num with
the same architecture), this factorizes attention to make the increase
computationally neutral.

**Implementation knobs.**
- Implement two alternating attention layers: one attending within each slice
  (local, O(slice_size²)), one attending across slice representatives (global,
  O(slice_num²)).
- Alternatively: implement "grouped-query attention" where slices are grouped
  into super-tokens, reducing KV size.
- Keep total params comparable to baseline (adjust n_hidden or n_head).
- Try slice_num=96 first (modest increase + factorized), then 128.

**References.**
- Ho et al. 2019, "Axial Attention in Multidimensional Transformers." arxiv 1912.12180
- Wang et al. 2024, "Transolver." arxiv 2402.02366 — baseline; see appendix for
  slice construction.

**Why it fits here.** The mesh has 74K–242K nodes; the current 64 slices group
~1,000–3,500 nodes per slice, which is coarse for the dense foil-surface region.
More slices = finer grouping = better surface pressure resolution.

---

### H-10: Domain-conditioned feature modulation (FiLM over Re + geometry)

**Hypothesis.** The three domains (RC single, RC tandem, Cruise tandem) differ in
Re range, AoA range, and foil count. A Feature-wise Linear Modulation (FiLM) layer
that conditions every Transolver block's LayerNorm on a global context vector
(derived from log(Re), AoA, gap, stagger, NACA params — the global dims 13–23)
should allow the model to adapt its internal representations per-sample. This is
analogous to conditional normalization in image generation, which consistently
improves multi-domain models.

**Implementation knobs.**
- Extract global condition: `cond = x[:, 0, 13:24]` (first node's global features
  are the same for all nodes in a sample; shape [B, 11]).
- Project to `(gamma, beta)` per layer: `film_net = MLP(11 → 2 * n_hidden)`.
- Apply after each LayerNorm: `h = gamma * h + beta`.
- Total added params: `n_layers * 2 * n_hidden * 11 / n_hidden` ≈ tiny.
- Also try conditioning only the first and last Transolver blocks.

**References.**
- Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer."
  arxiv 1709.07871
- Kovachki et al. 2023, "Neural Operator Review." arxiv 2108.08481 — discusses
  conditional operator learning for multi-physics.

**Why it fits here.** The model must simultaneously handle RC single (ground
effect, negative AoA) and cruise (positive AoA, no ground). FiLM adds
cheap but powerful inductive bias for flow-regime conditioning.

---

### H-11: Deeper MLP ratio with residual gating (SwiGLU / GeGLU)

**Hypothesis.** The baseline uses `mlp_ratio=2` (FFN width = 2*n_hidden). Round-1
alphonse increases it to 4. A qualitatively different change: replace the standard
GELU FFN with a gated variant (SwiGLU or GeGLU), which uses two projections and
an element-wise gate. At the same parameter count as mlp_ratio=4, SwiGLU
consistently outperforms vanilla FFN in both LLMs and PDE surrogates by improving
gradient flow through the FFN gating pathway.

**Implementation knobs.**
- In the Transolver FFN: replace `Linear(d, 4d) -> GELU -> Linear(4d, d)` with
  `Linear(d, 8d/3) -> SiLU * Linear(d, 8d/3) -> Linear(8d/3, d)` (SwiGLU
  factor of 2/3 to match param count with mlp_ratio=2).
- Or equivalently set `ffn_dim = int(n_hidden * mlp_ratio * 2 / 3)` for the gate
  version.
- Keep mlp_ratio=2 for fair comparison, just change activation to gated variant.
- Also try mlp_ratio=3 with SwiGLU (matches params of mlp_ratio=4 vanilla).

**References.**
- Noam Shazeer 2020, "GLU Variants Improve Transformer." arxiv 2002.05202
- Touvron et al. 2023, LLaMA 2 — SwiGLU standard in practice; arxiv 2307.09288.

**Why it fits here.** This is a near-zero-cost change (same or fewer params, same
FLOPs) that consistently improves quality. It does not interact with any round-1
experiment, so it can be stacked cleanly.

---

## Theme 5: Augmentations / Inductive Biases

### H-12: AoA sign-flip augmentation for aerodynamic symmetry

**Hypothesis.** For single-foil RC samples, an inverted airfoil at AoA = -5° and
AoA = +5° are geometrically equivalent (mirror reflection about x-axis). Applying
this symmetry as on-the-fly data augmentation — flip z-coordinates, negate Uy,
negate AoA dims — effectively doubles the single-foil training set and teaches the
model about aerodynamic symmetry. For tandem samples, the reflection is valid only
if gap/stagger are also sign-adjusted. Expected delta: most visible on
`val_single_in_dist` and `val_re_rand`.

**Implementation knobs.**
- In the training loop, before normalization: with probability p_aug=0.5,
  apply: `x[:, :, 1] *= -1` (z-coord), `x[:, :, 3] *= -1` (saf_z),
  `x[:, :, 14] *= -1` (AoA foil 1), `x[:, :, 18] *= -1` (AoA foil 2, tandem),
  `y[:, :, 1] *= -1` (Uy channel).
- For tandem samples: also negate `x[:, :, 23]` (stagger) if it has a sign.
- Single-foil check: `is_tandem = (x[:, 0, 22] != 0)`.
- Apply augmentation only during training; disable at val/test.

**References.**
- Belbute-Peres et al. 2020 — symmetry augmentation in mesh-based CFD learning.
- Standard technique in aerodynamic ML; no single canonical reference needed.

**Why it fits here.** The RC domain uses inverted airfoils (negative AoA only).
This augmentation is physically exact (not approximate) and costs nothing at
inference time.

---

### H-13: Re-stratified hard negative sampling (upweight OOD samples)

**Hypothesis.** The current `WeightedRandomSampler` balances across the three
domains equally. Within each domain, all samples are weighted equally. But
high-Re samples are harder (larger flow gradients, harder pressure distributions)
and are underrepresented in training signal if the model fits low-Re samples first.
Upweighting samples in proportion to `log(per_sample_y_std)` (from the stats)
should force more frequent exposure to the hard, high-Re regime, at the cost of
slightly less low-Re coverage.

**Implementation knobs.**
- At dataset load time, compute `per_sample_std` for each training sample.
- Replace the domain-balanced weights with `w_i = domain_weight_i * log(std_i + 1)`.
- Normalize so total expected samples per epoch remains 1499.
- Also try `w_i = domain_weight_i * std_i^0.5` for a softer version.
- Compare `val_re_rand` (benefits from Re coverage) vs. `val_single_in_dist`
  (may regress if low-Re coverage drops).

**References.**
- Curriculum learning literature: Bengio et al. 2009, "Curriculum Learning." ICML.
- Harder-example mining: Shrivastava et al. 2016, "Training Region-based Object
  Detectors with Online Hard Example Mining." CVPR.

**Why it fits here.** The per-sample std disparity is documented as 10x within
each split. This directly targets that disparity without changing the model or loss.

---

## Theme 6: Physics-Aware Approaches

### H-14: Soft incompressibility constraint (∇·u = 0) on volume nodes

**Hypothesis.** For incompressible flow at the Re range of this dataset (100K–5M,
not highly compressible), the continuity equation ∇·u = 0 must hold at every
interior node. Adding a soft penalty `lambda_c * ||div_u||²` — where
`div_u ≈ (Ux[i+1,j] - Ux[i-1,j]) / (2dx) + (Uy[i,j+1] - Uy[i,j-1]) / (2dz)` —
forces the velocity field to be physically consistent. The key challenge is
computing finite-difference divergence on an unstructured mesh; using the `dsdf`
distances (dims 4–11) or k-nearest neighbors as stencil weights is feasible.

**Implementation knobs.**
- For each volume node `i`, find its k=4 nearest neighbors from the batch (or
  precompute neighbor indices). Estimate `du_dx ≈ sum(w_j * Ux_j)` using
  inverse-distance weights.
- `lambda_c` in {0.001, 0.01, 0.1}; start small to avoid over-constraining.
- Apply only to volume nodes (`~is_surface`); surface boundary condition
  is no-slip (u=0), not divergence-free.
- For simplicity in the first test: use a finite-difference on the structured
  background zone only (Zone 0, which has roughly Cartesian structure).

**References.**
- Raissi et al. 2019, "Physics-Informed Neural Networks." arxiv 1711.10561
- Wandel et al. 2021, "Learning Incompressible Fluid Dynamics." arxiv 2006.08762
  — divergence constraint on unstructured velocity fields.

**Why it fits here.** The model predicts Ux, Uy, p jointly; without physical
coupling, nothing prevents (Ux, Uy) from having nonzero divergence. Physically
inconsistent velocity fields reduce pressure accuracy because p satisfies a
Poisson equation driven by the velocity field.

---

### H-15: Multi-task auxiliary head for surface Cp (pressure coefficient)

**Hypothesis.** The pressure coefficient Cp = (p - p_inf) / (0.5 * rho * U_inf²)
is a dimensionless quantity that normalizes pressure by dynamic pressure. At any
given node, Cp is a simple linear function of p, Re, and inflow velocity (which
can be estimated from log(Re) and the domain AoA). Training with an auxiliary
Cp prediction head (in addition to the main (Ux, Uy, p) prediction) adds a
supervision signal that is more uniform across Re values — potentially stabilizing
the learning of pressure at low-Re where raw p values are small. Weight the
auxiliary loss at 0.1–0.5 to avoid distorting the primary prediction.

**Implementation knobs.**
- Add `aux_head = Linear(n_hidden, 1)` to predict Cp from the final hidden state
  on surface nodes.
- Compute `Cp_target = (p_true - p_mean) / (0.5 * U_ref²)` where
  `U_ref = exp(log_Re) * nu / L` (kinematic viscosity nu ≈ 1.5e-5, chord L=1.0).
- `L_aux = lambda_aux * MAE(Cp_pred, Cp_target)` on surface nodes only.
- lambda_aux in {0.05, 0.1, 0.5}.
- The aux head adds < 1K params; negligible overhead.

**References.**
- Caruana 1997, "Multitask Learning." ML — auxiliary tasks improve generalization.
- Belbute-Peres et al. 2020, "Combining Differentiable PDE Solvers and Graph
  Neural Networks for Fluid Dynamics." ICML — dimensionless feature normalization.

**Why it fits here.** The primary metric is surface pressure MAE in physical units,
but the model struggles with cross-Re consistency. Cp normalization is standard
in aerodynamics precisely because it removes the Re-dependent scale factor.

---

## Summary table

| ID   | Theme                      | Mechanism               | Est. delta | Complexity |
|------|---------------------------|-------------------------|-----------|------------|
| H-01 | Positional encoding        | RFF on (x,z)           | 3–8%      | Low        |
| H-02 | Positional encoding        | Learned sinusoidal PE   | 2–5%      | Low        |
| H-03 | Loss reformulation         | Scale-normalized loss   | 2–6%      | Low        |
| H-04 | Loss reformulation         | Sobolev gradient loss   | 2–5%      | Medium     |
| H-05 | Loss reformulation         | Bernoulli constraint    | 1–4%      | Medium     |
| H-06 | Training trick             | EMA weight averaging    | 1–3%      | Low        |
| H-07 | Training trick             | SAM / ASAM              | 2–5%      | Medium     |
| H-08 | Training trick             | Cautious AdamW          | 1–3%      | Low        |
| H-09 | Architecture               | Factorized attention    | 2–4%      | Medium     |
| H-10 | Architecture               | FiLM domain conditioning| 2–5%      | Low        |
| H-11 | Architecture               | SwiGLU FFN              | 1–3%      | Low        |
| H-12 | Augmentation               | AoA sign-flip           | 1–4%      | Low        |
| H-13 | Augmentation               | Re-stratified sampling  | 1–3%      | Low        |
| H-14 | Physics-aware              | Soft ∇·u=0 constraint   | 2–5%      | High       |
| H-15 | Physics-aware              | Cp auxiliary head       | 1–4%      | Low        |

Priority recommendation: H-01 (RFF), H-06 (EMA), H-10 (FiLM), H-11 (SwiGLU),
H-12 (AoA aug), H-03 (scale-norm loss) are all low-complexity and can be assigned
immediately to 6 students as round-2 starters.
