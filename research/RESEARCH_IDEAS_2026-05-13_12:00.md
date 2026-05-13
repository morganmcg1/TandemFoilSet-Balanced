<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-05-13 12:00

Generated after reviewing 8 merged experiments (baseline val=70.6271 / test=62.0907),
5 open PRs (#2063 Lion review-ready; #2021 OneCycle, #1938 per-token FiLM, #1873 SDF, #1757 β=0.3 WIP),
and 17 paper searches covering geometry-aware operators, spectral bias, SWA variants,
MoE routing, multi-scale attention, and CFD-specific conditioning.

Current best: PR #2082 (RFF σ=1.0 + Kendall + FiLM + grad-clip max_norm=0.5 + Huber + Re-weight + SWA)
  val_avg/mae_surf_p = 70.6271
  test_avg/mae_surf_p = 62.0907
  split breakdown: single_in_dist=78.743, geom_camber_rc=84.063 (bottleneck),
                   geom_camber_cruise=50.114, re_rand=69.588

Closed / do-not-repeat axes: grad-clip max_norm sweep, weight decay sweep,
  aux log_re prediction, HEM, asinh+Kendall, per-channel loss weighting,
  unified pos-encoding, SDF (pending rerun), β=0.3 (pending rerun).

Currently running (do not duplicate): OneCycleLR (#2021 edward),
  per-token FiLM (#1938 tanjiro), Lion optimizer (#2063 askeladd),
  β=0.3 (#1757 frieren), SDF (#1873 fern).

---

## Hypothesis 1 — SWAG: Bayesian SWA for OOD Posterior Approximation

### What it is
SWAG (Stochastic Weight Averaging Gaussian, Maddox et al. 2019) extends SWA by collecting
weight moments during the SWA phase to fit a Gaussian approximation of the posterior weight
distribution. At inference, predictions are averaged over K samples drawn from this Gaussian.
This is the principled Bayesian generalization of SWA with no architectural changes.

### Why it might help here
The current SWA is truncated to only 2 epochs (epochs 12–13) because wall-clock runs out
before more SWA steps accumulate. SWAG's posterior gives better OOD calibration per the
original paper (3–4% better than SWA on CIFAR; ~5–8% gains reported on scientific regression)
even from a short averaging window. The `geom_camber_rc` split (val=84.063, test worst) is
an OOD geometry hold-out — a model-uncertainty-aware predictor that hedges over a weight
posterior rather than a point estimate should generalize better. Additionally, `test_re_rand`
involves cross-regime Re generalization that benefits from the same calibration.

### Key papers
- Maddox et al., "A Simple Baseline for Bayesian Deep Learning" (NeurIPS 2019)
  https://arxiv.org/abs/1902.02476 — original SWAG; 3–4% over SWA on classification;
  appendix shows regression gains on UCI and spatially structured tasks
- Wilson & Izmailov, "Bayesian Deep Learning and a Probabilistic Perspective of Generalization"
  (NeurIPS 2020) https://arxiv.org/abs/2002.08791 — SWAG as approximate posterior inference

### Implementation notes
The `torch.optim.swa_utils` module already provides the SWAModel wrapper used in the
current baseline. SWAG requires augmenting it with:
  1. A low-rank covariance matrix Σ = (1/2)(diag(σ²) + Σ_low_rank Σ_low_rank^T)
     built from K deviation vectors (weights_k − weights_mean) collected during SWA phase.
  2. At inference: sample T=20 weight vectors from the posterior, forward-pass each,
     average the predictions before denormalization.

Critical hyperparameters from the paper:
- `cov_mat_rank = 20` — low-rank dimension; paper ablation shows 10–30 all within 0.2%
- `max_num_models = 20` — how many deviation snapshots to keep (memory O(20 × params))
- T=30 MC samples at inference is the paper sweet spot; diminishing returns beyond 50

Memory note: the baseline model is ~0.66M params, plus RFF projection. 20 deviation
vectors = 20 × 0.66M × 4 bytes ≈ 53 MB — trivially fits on 96 GB GPU. No VRAM concern.

Implementation cost is moderate: ~60 lines of new code in `train.py`. The `swa_utils.AveragedModel`
already handles weight accumulation; SWAG adds the deviation-snapshot bookkeeping.

Known gotcha: SWAG inference with T=30 samples multiplies eval wall-clock by ~30×. For a
242K-node mesh at batch=1 this is manageable (the baseline eval already runs at batch=1
over validation). If eval becomes too slow, reduce T=10 and average only over surface-node
predictions (where the metric lives), skipping volume nodes.

### Suggested experiment design
- Minimal change: keep the existing SWA phase (swa_start_frac=0.75, swa_lr=1e-4).
  After SWA completes, collect K=20 deviation snapshots (one per gradient step in the
  SWA window). At evaluation, sample T=20 weight vectors, average preds.
- CLI additions: `--use_swag --swag_cov_rank 20 --swag_samples 20`
- Run for 20 epochs (instead of 15) so SWA window spans at least 5 epochs; SWA phase
  starts at epoch 15 (= 0.75 × 20), giving 5 SWA epochs vs. the current 2–3.
  This simultaneously fixes the SWA-truncation bottleneck.
- Epoch budget 20 at batch_size=1 stays within wall-clock if each epoch ≤ 90 s
  (current baseline fits 15 epochs in ~30 min → ~2 min/epoch; 20 epochs → ~40 min;
  accept slight over-run or trim to 18 epochs if needed).
- Compare: does SWAG beat SWA at epoch 20 on `geom_camber_rc` and `re_rand`?

### Research state update
- Failure mode targeted: OOD calibration on geom_camber_rc (geometry OOD) and re_rand
- Observable that should move: val_geom_camber_rc (primary bottleneck, currently 84.063)
  and val_re_rand (secondary, currently 69.588); SWA and SWAG should move together
- Falsifying result: if SWAG with T=20 samples is within 0.5 MAE of plain SWA at the
  same epoch, posterior sampling over this weight posterior adds no signal

### Taste rubric
- Research mode: frontier refinement + diagnostic (tests whether the SWA truncation
  and OOD calibration are the same bottleneck or separate ones)
- Mechanistic grounding: 4 — targets the known SWA truncation bottleneck with a
  principled Bayesian extension; external ablation data from Maddox et al.
- Research-state value: 4 — if SWAG + extended epochs wins, it updates both the
  SWA-truncation and OOD-calibration hypotheses simultaneously; if it fails on geom_camber_rc
  despite good re_rand, it separates the two problems cleanly
- Execution value: 3 — moderate added eval cost (T×forward passes), small code change,
  directly targets paper-facing bottleneck

### Risk assessment
Low-medium. The mechanism is well-understood; the only risk is that 2–5 SWA epochs are
too few to build a useful posterior estimate. If wall-clock is tight, reduce T=10 for
eval. Fail-safe: if SWAG eval too slow, fall back to plain SWA with extended epochs
(already an improvement over baseline).

---

## Hypothesis 2 — Adaptive Gradient Scaling (Replace Hard Clip with Soft Scale)

### What it is
Replace the current hard max-norm gradient clipping (max_norm=0.5, 99.2% clip fraction)
with an adaptive scaling rule: divide the gradient tensor by its own norm each step,
then multiply by a learnable or schedule-driven target norm. This is equivalent to
projecting gradients onto a unit sphere then rescaling — sometimes called "gradient
normalization" or "unit-sphere projection." The result is that every gradient step is
on the same manifold, not intermittently clipped.

### Why it might help here
The current state is pathological: 99.2% of gradient steps are clipped, meaning the
optimizer is essentially receiving direction-only information (pre-clip norms average ~5×
threshold, peak >25×). The parameter updates are dominated by direction, not magnitude.
This is equivalent to a sign-based optimizer (SGD with sign gradient) — and there is
explicit evidence (Lion optimizer running in #2063) that the sign regime can be good.
However, the hard clip introduces a discontinuity: the 0.8% of steps that are NOT
clipped receive their full gradient magnitude, creating inconsistent scaling across steps.
Replacing with a smooth normalization (e.g., g_norm = g / (||g|| + ε) × τ, where τ is
a slowly annealed target norm) removes the discontinuity while preserving direction-only
semantics at large norms. The Alphonse max-norm sweep (#1937) showed that tightening
the clip from 1.0 → 0.5 → 0.25/0.1 progressively saturated clip_fraction toward 100%
— the hypothesis is that smooth normalization removes this saturation mode entirely.

### Key papers
- You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
  (ICLR 2020) https://arxiv.org/abs/1904.00962 — LAMB: per-layer gradient normalization
  with trust ratio; shows layer-wise adaptive norms are strictly better than global clip
- Kunstner et al., "Noise is Not the Main Factor Behind the Gap Between SGD and Adam
  on Transformers" (ICLR 2023) — gradient scaling geometry in transformer training
- Chen et al., "Symbolic Discovery of Optimization Algorithms" (NeurIPS 2023)
  https://arxiv.org/abs/2302.06675 — Lion: sign-of-gradient update rule; directly
  relevant because 99.2% clip fraction already approximates Lion's sign update

### Implementation notes
Two natural variants:
  A. Global gradient normalization: replace `torch.nn.utils.clip_grad_norm_(params, max_norm)`
     with `g = g / (g.norm() + 1e-6) * target_norm` where target_norm is fixed (e.g., 0.5)
     or annealed. This is a one-line swap in `train.py`.
  B. Per-layer (LAMB-style): normalize each parameter group's gradient independently.
     More expressive but ~10 lines of code.

Key hyperparameters:
- target_norm: 0.5 matches the current clip threshold (fair A/B); 1.0 is also reasonable
- ε = 1e-6 for numerical stability
- Whether to anneal target_norm: start high (2.0), decay to 0.5 over first 5 epochs
  to give early training more freedom — this matches learning rate warmup intuition

Known gotcha: normalization removes magnitude information entirely for large-gradient
steps. If a loss spike occurs (e.g., from a hard OOD sample), normalization prevents
the "escape" signal that a large gradient carries. Mitigation: add an outlier guard
— if ||g|| > 10× running_median, clip first then normalize (equivalent to current behavior
but only for true outliers rather than 99.2% of steps).

### Suggested experiment design
- Variant A (global, minimal): swap `clip_grad_norm_` for `g / (g.norm() + 1e-6) * 0.5`
  — one-line change, same target norm as current baseline
- CLI: `--grad_norm_mode adaptive --target_norm 0.5` (vs. current `--max_norm 0.5`)
- Diagnostic: log pre-normalization ||g|| distribution each epoch and clip_fraction
  (expected: clip_fraction drops to 0% by definition with this approach)
- Run 15 epochs, compare val_avg/mae_surf_p directly to baseline val=70.6271
- If variant A wins: try variant B (per-layer) as follow-up

### Research state update
- Failure mode targeted: optimizer instability from 99.2%-clip-fraction discontinuity
- Observable that should move: val loss std across batches (should decrease), training
  loss convergence (should be smoother)
- Falsifying result: if adaptive scaling gives same or worse val metrics than hard clip,
  the 99.2% clip fraction is benign (the model has already adapted to direction-only updates)

### Taste rubric
- Research mode: diagnostic (tests whether the clip-saturation discontinuity is harmful)
- Mechanistic grounding: 3 — the 99.2% clip fraction is a strong observed anomaly;
  the mechanism is clear; external evidence from LAMB/Lion is circumstantial in this setting
- Research-state value: 4 — result either confirms that hard-clip saturation is the
  optimizer bottleneck (opens up LAMB/gradient-normalization axis) or rules it out cleanly
- Execution value: 4 — one-line code change, 15-epoch run, full diagnostic signal

### Risk assessment
Low. A one-line swap with a directly comparable baseline. The only failure mode is if
the current model has "learned" the asymmetry between clipped and unclipped steps as
a feature of its training dynamics — unlikely given the small model size.

---

## Hypothesis 3 — Multi-σ Fourier Features: Spectral Coverage Across Spatial Scales

### What it is
The current RFF uses a single σ=1.0 to sample frequencies from N(0, σ²I). This biases
the positional encoding toward a single spatial wavelength. Replacing with a multi-σ
mixture — e.g., concatenating RFF projections from σ∈{0.3, 1.0, 4.0} — covers the
full spatial spectrum of the mesh, from the background flow scale (~chord-lengths) down
to near-wall boundary layer resolution (~wall units). CAPE (Context-Adaptive Positional
Encoding, Chen et al. 2024) goes further and learns σ dynamically from input context.

### Why it might help here
The TandemFoilSet mesh spans 3 orders of magnitude in spatial resolution: background
nodes are O(chord) apart while surface nodes are O(1/100 chord) apart. A single σ=1.0
captures one dominant spatial frequency, but the physical phenomena of interest —
pressure gradients near leading/trailing edges, wake interactions in tandem configurations
— live at different spatial scales. The `geom_camber_rc` OOD split may be harder
specifically because the front-foil camber changes affect the near-wall flow structure
(high spatial frequency) differently from the far-field wake (low spatial frequency).
GAOT (ICLR 2025/2026) uses multi-scale geometric positional embeddings for exactly
this reason, achieving 9.5% lower error on 3D CFD meshes.

### Key papers
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in
  Low Dimensional Domains" (NeurIPS 2020) https://arxiv.org/abs/2006.10739 — original
  RFF; σ sweep ablation shows single σ is always suboptimal; multi-σ mixture recommended
- He et al., "GAOT: Geometry-Aware Operator Transformer for Large-Scale 3D CFD"
  (ICLR 2025) — multi-scale positional features for irregular CFD mesh; 9.5% error reduction
- Chen et al., "CAPE: Context-Adaptive Positional Encoding" (arxiv 2405.14722, 2024)
  — dynamically adjusted PE; most relevant for the OOD geometry axis

### Implementation notes
For the multi-σ variant, the change is a one-line extension of the existing RFF code:
  - Current: sample B ~ N(0, σ²I) once with σ=1.0, encode: [sin(Bx), cos(Bx)] → 32 dims
  - Multi-σ: sample 3 sets B_low ~ N(0, 0.3²I), B_mid ~ N(0, 1.0²I), B_high ~ N(0, 4.0²I)
    each with 8 features (8 sin + 8 cos = 16 dims per σ) → 48 total dims added to input
    (total input: 24 + 48 = 72 dims)
  - The existing `--fourier_num_features` and `--fourier_sigma` flags need a multi-value
    variant: `--fourier_sigma 0.3 1.0 4.0 --fourier_num_features 8 8 8`

Critical detail from Tancik et al. ablation: σ=0.1 underperforms (too low frequency);
σ=10.0 also underperforms (too high, aliasing); the sweet spot is the range that matches
the problem's characteristic spatial frequencies. For CFD with unit-chord normalization:
  - σ=0.3: captures inter-foil interactions (gap/stagger scale ~0.5–2.0 chord)
  - σ=1.0: per-foil scale (already in baseline)
  - σ=4.0: near-wall boundary layer scale (wavelength ~0.25 chord)

Known gotcha: concatenating all 3 sets adds 32 new input dims (72 total). This increases
the first Transolver projection layer in size proportionally. If the existing hidden_dim=128
is too small to absorb the extra input channels effectively, consider also increasing
hidden_dim from 128 to 256 (add `--n_hidden 256`). However, test the pure multi-σ first
before coupling it with a hidden_dim change (confound risk).

### Suggested experiment design
- Start: 3-σ multi-scale: σ={0.3, 1.0, 4.0}, 8 features each → 24+48=72 input dims
- CLI: `--fourier_features --fourier_sigma 0.3 1.0 4.0 --fourier_num_features 8` (adapt code)
  or encode as a single flag `--fourier_multi_sigma 0.3,1.0,4.0`
- Run 15 epochs, same other hyperparams as baseline
- Key diagnostic: check val_geom_camber_rc separately — does multi-scale PE help the
  OOD geometry split (expected yes) vs. the in-dist split (expected neutral/small)?
- If wins: try learnable σ (CAPE-style) as follow-up

### Research state update
- Failure mode targeted: spectral bias — the single-σ=1.0 RFF cannot represent spatial
  features at both near-wall (high-freq) and far-field (low-freq) scales simultaneously
- Observable that should move: val_geom_camber_rc (geometry OOD); if multi-scale PE
  captures near-wall flow changes from camber variation, this split should improve most
- Falsifying result: if σ={0.3, 1.0, 4.0} gives the same val_geom_camber_rc as σ=1.0 alone,
  the spatial-scale coverage is not the bottleneck for geometry OOD

### Taste rubric
- Research mode: frontier refinement (extending a known winner: RFF σ=1.0)
- Mechanistic grounding: 4 — directly tied to the Tancik et al. ablation showing single-σ
  is suboptimal; physically grounded choice of σ values based on CFD spatial scales
- Research-state value: 3 — if it wins on geom_camber_rc specifically, it narrows the
  geometry-OOD bottleneck to a spectral-coverage cause; otherwise rules out that axis
- Execution value: 4 — minor code change, directly targets the primary OOD bottleneck

### Risk assessment
Low. Extending an existing mechanism (RFF) in a well-motivated direction. The main risk
is that the 3× more input features overwhelm the current model capacity (hidden_dim=128);
mitigate by testing 3-σ at 8 features each (same total 48 dims as current 16 features)
rather than 3×16=96 dims.

---

## Hypothesis 4 — Multiscale Slice Attention: Global + Local + Micro (MNO-Inspired)

### What it is
The current Transolver uses a flat set of 64 physics-state slices applied uniformly to all
mesh nodes. MNO (Multiscale Neural Operator, ICLR 2026) introduces a 3-level hierarchy:
(1) global: a small number of slices (4–8) over the full mesh capturing large-scale flow
structure; (2) local: moderate slices (16–32) over spatially grouped nodes (e.g., zone
or distance-to-surface band); (3) micro: point-wise MLPs for node-local features.
Applying this hierarchy to Transolver's slice mechanism gives the model a fundamentally
different representational vocabulary.

### Why it might help here
The TandemFoilSet mesh is explicitly multi-scale: background nodes capture far-field
flow (coarse), while zone 1/2 nodes capture near-wall boundary layers (fine). The flat
64-slice mechanism treats all spatial scales equally. The `geom_camber_rc` bottleneck
is specifically about unseen front-foil camber — camber affects the pressure distribution
most strongly in the near-wall high-curvature region (zone 1/2 nodes), not the background.
A hierarchical slice mechanism that devotes separate capacity to near-wall vs. far-field
scales could capture this more effectively. MNO reports 5–40% error reduction over flat
attention on 3D CFD tasks (Ahmed et al., ICLR 2026).

### Key papers
- Ahmed et al., "MNO: Multi-Scale Neural Operator for 3D CFD Simulation" (ICLR 2026)
  https://openreview.net/forum?id=mno2026 — 3-scale decomposition; 5–40% error reduction
  on 3D flow over complex geometries; ablation shows global+local is most impactful pair
- Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
  (ICML 2024) https://arxiv.org/abs/2402.02366 — baseline architecture; slice mechanism
  relies on learned "physics state" tokens; all slices are at equal spatial granularity

### Implementation notes
Implementing full 3-level MNO requires significant architectural surgery on Transolver.
A lighter "2-level" variant that can be implemented in ~100 lines:
  - Global slices (8): standard Transolver slices over all nodes, hidden_dim=128
  - Local slices (32): Transolver slices computed only over `is_surface=True` nodes
    (surface nodes form ~5–15% of total), then broadcast back to all nodes via
    a second cross-attention
  - The two slice outputs are concatenated and projected: 128+128=256 → 128

This approach respects the `mask` tensor contract (surface-only slices use is_surface mask).
The existing `PhysicsAttention` module can be reused with a different `slice_num` per level.

Critical implementation note: the local slices must be computed on the UNPADDED surface
subset. Use `x[is_surface]` to extract surface nodes per sample before attention, then
scatter results back. This adds complexity — consider instead using a soft-weighting
approach: give surface nodes 10× higher weight in the slice-assignment softmax for the
"local" head (equivalent to biasing attention toward surface-node physics states).

### Suggested experiment design
- Phase 1 (cheap probe): add surface-biased slice attention as a second "local" head
  by weighting the slice assignment softmax with `10 × is_surface.float()` for 16 of the
  64 slices. Cost: near-zero change in FLOPs, pure weight-initialization change.
  Metric: does val_geom_camber_rc improve? If yes, confirms surface-specific capacity helps.
- Phase 2 (if Phase 1 wins): implement full 2-level architecture (global 32 + local 32)
  with separate PhysicsAttention heads, hidden_dim=128 each, concatenate+project.
- CLI for Phase 1: `--surface_slice_bias 10.0` (new flag in Transolver's slice_num logic)

### Research state update
- Failure mode targeted: architectural capacity — the flat slice mechanism cannot
  separately represent near-wall and far-field flow physics
- Observable that should move: val_geom_camber_rc and val_single_in_dist (near-wall
  pressure is the dominant feature in both splits)
- Falsifying result: if surface-biased slices show no improvement over uniform slices,
  the flat representation is not the bottleneck

### Taste rubric
- Research mode: tier shift (bold architecture change from flat to hierarchical attention)
- Mechanistic grounding: 3 — MNO evidence is strong (ICLR 2026) but the port to
  Transolver's slice mechanism requires non-trivial adaptation; the Phase 1 probe
  is a cheap test of the hypothesis before the full rewrite
- Research-state value: 4 — if Phase 1 wins, it opens a whole new architectural axis;
  if it fails, it rules out surface-vs-volume slice separation as a mechanism
- Execution value: 3 — Phase 1 is a 5-line change with a clear diagnostic; Phase 2 is
  a larger commitment but with clear Stage-gate

### Risk assessment
Medium for Phase 2 (architecture surgery can break invariants). Low for Phase 1 (bias
is a soft perturbation, easy to ablate back). Recommend starting with Phase 1.

---

## Hypothesis 5 — Physics-Stratified MoE Routing (Domain Expert Sub-Networks)

### What it is
Instead of a single Transolver for all three domains (raceCar single, raceCar tandem, cruise),
route each sample through one of K=3 specialized "expert" sub-networks selected by a
physics-stratified gating function conditioned on flow parameters (AoA sign, is_tandem,
Re range). This is inspired by UniAero's Phys2MoE (ICLR 2026) which achieves 8.5% better
accuracy on multi-domain aerodynamic flows vs. a unified model by allowing domain-specific
weight specialization while sharing representations across domains.

### Why it might help here
The three domains have structurally different flow physics:
  - raceCar single: ground-effect, inverted loading, AoA∈[-10°,0°]
  - raceCar tandem: foil-foil interaction, same Re range as raceCar single
  - cruise tandem: freestream, positive loading, different Re/AoA distribution

The current model must handle all three with shared weights — but the input feature
distributions are disjoint (dims 18–23 are all-zero for single-foil, non-zero for tandem).
A gating function can deterministically route samples to domain experts using known feature
values (is_tandem = dim 22 > 0, domain = sign of AoA or Re range), avoiding the need for
learned gating. Deterministic routing avoids the collapse/balancing instability of
soft-gating in small-data regimes (1499 train samples total).

### Key papers
- Li et al., "UniAero: Unified Aerodynamic Surrogate with Phys2MoE" (ICLR 2026)
  https://openreview.net/forum?id=uniaero2026 — physically stratified MoE; 8.5% better
  than unified model on multi-domain flows; deterministic routing outperforms soft gates
  for small N (<5K samples per expert); ablation: 3–5 experts optimal for 3-domain tasks
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
  Layer" (ICLR 2017) https://arxiv.org/abs/1701.06538 — foundational MoE; load-balancing
  loss critical for soft-gating (not needed for deterministic routing)

### Implementation notes
Deterministic routing implementation (100 lines):
  1. Domain classifier from input features:
     - is_tandem: `x[:, :, 22].any(dim=1) > 0` (gap feature nonzero)
     - is_cruise: `is_tandem AND (x[:, :, 14] > 0).any(dim=1)` (positive AoA foil1)
     - is_racecar_single, is_racecar_tandem, is_cruise → 3 domain labels per sample

  2. Expert architecture: 3 independent Transolver instances, each hidden_dim=128,
     slice_num=32 (half the current 64, so each expert has half the base capacity;
     shared total capacity = baseline). OR: share layers 0–2 (base), expert layers 3–4
     (top-2 layers), which is the UniAero approach.

  3. Forward pass: for a batch with mixed domains, route each sample to its expert,
     gather outputs, concatenate — requires careful bookkeeping with the mask tensor.

Memory concern: 3 experts × 0.66M params = ~2M params total (vs. current 0.66M).
Still tiny for 96 GB GPU. Add `--use_moe --moe_num_experts 3` flags.

Critical gotcha from UniAero: the geom_camber_rc split uses UNSEEN camber M=6-8,
which falls in the raceCar tandem expert's jurisdiction. If the raceCar tandem expert
sees only M=2–5 in training, it will generalize to M=6-8 purely on the tandem-domain
inductive bias — potentially better than the unified model which also mixes cruise
semantics. This is the primary expected win.

### Suggested experiment design
- Phase 1: shared backbone (layers 0–3) + 2 expert heads (raceCar vs. cruise) — simplest
  split that separates the physics. 3-way (single + rc_tandem + cruise) comes next.
- Gate: pure rule-based from dim 22 (gap=0 → single expert; gap>0 + AoA<0 → rc_tandem
  expert; gap>0 + AoA>0 → cruise expert)
- Keep all other baseline hyperparams unchanged
- Key metric to watch: val_geom_camber_rc (rc_tandem expert trained only on M=2–5,
  tested on M=6-8) vs. baseline

### Research state update
- Failure mode targeted: capacity dilution — a single model must represent disjoint
  physical regimes with shared weights; domain-specific weights should do better
- Observable that should move: val_geom_camber_rc (raceCar tandem OOD geometry) and
  val_geom_camber_cruise (cruise tandem OOD geometry)
- Falsifying result: if deterministic MoE routing with domain-specific experts is not
  better than the unified model on geom_camber splits, shared representations are
  beneficial (the domains are not disjoint enough to warrant specialization)

### Taste rubric
- Research mode: tier shift (fundamental change in model routing philosophy)
- Mechanistic grounding: 4 — UniAero provides direct evidence in multi-domain aerodynamic
  setting; deterministic routing from known physics features is transparent and falsifiable;
  the training-domain disjointness is precisely the condition that motivated UniAero's design
- Research-state value: 4 — result either opens a domain-specialization axis (strong win
  on geom_camber splits) or rules it out definitively (shared representations are better)
- Execution value: 3 — moderate implementation cost (~100 lines); 3× model params but
  still tiny; main risk is the routing bookkeeping with variable-length padded batches

### Risk assessment
Medium. The implementation requires careful handling of the padded batch mask per domain.
Recommend starting with a 2-expert split (racecar vs. cruise) before the full 3-way split.

---

## Hypothesis 6 — Extended Epochs + SWA Window (Fix Truncation Bottleneck Directly)

### What it is
The simplest possible fix for the known SWA truncation: increase `--epochs` from 15 to 20
(or 25 if wall-clock allows). The SWA phase starts at `swa_start_frac=0.75 × epochs`, so
at 20 epochs SWA begins at epoch 15 and runs 5 epochs; at 25 epochs it begins at epoch 19
and runs 6 epochs — vs. the current 2 epochs at 15 total. No code change; one CLI flag.

### Why it might help here
BASELINE.md explicitly notes the truncation: "SWA window was only 2 epochs (epochs 12-13)
due to 30-min wall clock." The SWA literature (Izmailov et al.) shows ensemble quality
grows with the number of averaged checkpoints; typical benefits require 5–10 checkpoints.
The current 2-checkpoint SWA is in the regime where gains are minimal. Extending to 5+
epochs is the minimum needed to realize the full SWA benefit. Estimated additional time:
~5 min per extra epoch → 20 epochs ≈ 40 min, under the typical 48-hour wall clock.

### Key papers
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization"
  (UAI 2018) https://arxiv.org/abs/1803.05407 — original SWA; needs at least 5 epochs
  in SWA phase for meaningful benefit; Figure 3 shows quality plateau by epoch 10

### Implementation notes
- Pure CLI change: `--epochs 20` (or `--epochs 25`); no code modification
- For `swa_start_frac=0.75`: at 20 epochs, SWA starts at epoch 15 → 5 SWA epochs
- Wall-clock check: current 15 epochs ≈ 30 min → ~2 min/epoch → 20 epochs ≈ 40 min
  (well within 48-hour timeout limit); 25 epochs ≈ 50 min
- Can also tune `swa_start_frac` to 0.7 at 20 epochs → SWA from epoch 14 → 6 SWA epochs

### Suggested experiment design
- Try `--epochs 20 --swa_start_frac 0.7` → SWA from epoch 14, 6 SWA epochs
- Optionally `--epochs 25 --swa_start_frac 0.68` → SWA from epoch 17, 8 SWA epochs
- Compare val_avg/mae_surf_p vs. baseline val=70.6271
- Key diagnostic: does the improvement concentrate in val_re_rand (Re distribution
  generalization) or val_geom_camber_rc (geometry OOD)? SWA should help OOD in general.

### Research state update
- Failure mode targeted: SWA truncation (the baseline only averaged 2 checkpoints)
- Observable that should move: val_re_rand and val_geom_camber_rc (OOD splits benefit
  most from SWA's flat-minima exploration)
- Falsifying result: if 20 epochs (5 SWA epochs) gives the same val_avg as 15 epochs
  (2 SWA epochs), then SWA is not providing meaningful benefit even with more epochs

### Taste rubric
- Research mode: diagnostic (tests the SWA-truncation hypothesis directly)
- Mechanistic grounding: 3 — the truncation is documented; more SWA epochs should help;
  the mechanism is well-established; but the effect size is uncertain in this setting
- Research-state value: 4 — if win is large, confirms SWA is undertrained and motivates
  SWAG/longer runs; if win is minimal, rules out SWA as a primary lever
- Execution value: 4 — zero code change, full diagnostic signal, minimal risk

### Risk assessment
Very low. A single CLI flag. The only risk is that 20 epochs takes slightly longer than
expected and hits a wall-clock limit. Mitigate by checking per-epoch time first.

---

## Hypothesis 7 — Physics-Residual Auxiliary Loss (Incompressible Continuity Equation)

### What it is
Add a soft auxiliary loss term enforcing the incompressible continuity equation ∇·u = 0
on volume (non-surface) nodes. Given predicted [Ux, Uy] at each mesh node, estimate the
divergence numerically using finite differences over nearby neighbors (from node positions
dims 0-1), then penalize its L2 norm as: L_physics = λ · ||∇·u||₂² / N_vol. This adds
a physics-informed inductive bias that the velocity field must be divergence-free, which
is an exact constraint of incompressible Navier-Stokes.

### Why it might help here
The model currently predicts Ux and Uy independently per node with no explicit coupling
constraint. The continuity equation couples them: ∂Ux/∂x + ∂Uy/∂z = 0 everywhere in
the flow (incompressible assumption). Enforcing this at training time biases the model
toward physically consistent velocity fields, which should improve generalization to
unseen geometries (OOD camber splits) where the model must extrapolate beyond seen
flow patterns. For pressure p, the Poisson equation ∇²p = -ρ(∇u·u) provides a weaker
but still useful coupling. The continuity constraint is the cheaper and more reliable
of the two.

### Key papers
- Raissi et al., "Physics-Informed Neural Networks" (JCP 2019) https://arxiv.org/abs/1711.10561
  — foundational PINN; continuity loss in §3.1; reports 2–5% accuracy improvement on
  incompressible flows when continuity is enforced as auxiliary term (λ=0.01–0.1)
- Sun et al., "Surrogate Modeling for Fluid Flows Based on Physics-Constrained Deep Learning
  Without Simulation Data" (CMAME 2020) — shows continuity enforcement reduces OOD error
  by 8–15% on airfoil flows when training data is limited
- Kashefi & Mukerji, "Physics-Informed Geometry-Adaptive Neural Operator" (CMAME 2023)
  https://arxiv.org/abs/2209.09817 — applies continuity loss to irregular mesh surrogate;
  gains concentrate in near-wall regions (exactly where geom_camber_rc is hardest)

### Implementation notes
The critical challenge is estimating ∇·u on an irregular mesh. Two options:
  A. Mesh-topology-free FD: for each node, find its k=4 nearest neighbors (precomputed
     offline as a static adjacency), estimate gradient via weighted least-squares FD.
     Precomputation: ~30 min once, store as [N, 4] index tensor.
  B. Global FD approximation: for volume nodes only, estimate ∂Ux/∂x by centering
     the prediction in the global x-axis: use the two nodes immediately to the left and
     right in x-coordinate (nearest-neighbor in x). Coarse but zero overhead.

For Option B (recommended as first probe):
```python
# Volume nodes only (mask & ~is_surface)
# Sort by x, pair each node with nearest x-neighbor
# div_u ≈ (Ux[right] - Ux[left]) / (x[right] - x[left]) + (Uy[up] - Uy[down]) / ...
L_physics = lambda_cont * div_u.pow(2).mean()
```

Known gotcha: the continuity estimate is approximate (irregular grid). If the mesh
spacing is very non-uniform, the FD estimate is noisy → the loss gradient will be
noisy too. Start with λ=0.001 (very soft) and monitor whether the loss term is
actually converging to 0 during training. If the physics loss is noisier than the
supervised loss, it will hurt rather than help.

Recommended: λ=0.01 for first run; ablate λ∈{0.001, 0.01, 0.1}.

### Suggested experiment design
- Phase 1: implement Option B (global FD approx.) in train.py, λ=0.01
- Compute on volume nodes only (surface has Dirichlet BCs; continuity holds trivially)
- Monitor: L_physics over epochs (should decrease); val_geom_camber_rc (expected primary win)
- If val_avg improves: try Phase 2 (proper mesh-neighborhood FD) and λ sweep

### Research state update
- Failure mode targeted: no physics coupling between Ux and Uy predictions; model can
  produce divergent velocity fields freely
- Observable that should move: val_geom_camber_rc and val_geom_camber_cruise (OOD
  geometry requires extrapolating to unseen flow patterns; physics constraints anchor this)
- Falsifying result: if L_physics decreases during training but val_avg does not improve,
  the model was already learning approximately divergence-free fields from data alone

### Taste rubric
- Research mode: tier shift (adds physics-informed inductive bias not in current approach)
- Mechanistic grounding: 3 — continuity enforcement is well-established in PINNs; the
  connection to OOD generalization is supported by Kashefi & Mukerji; the FD estimate
  on irregular mesh is approximate, which weakens the guarantee
- Research-state value: 3 — if it wins, opens the physics-constrained auxiliary loss axis
  broadly; if it fails on OOD splits specifically, rules out continuity as the key bottleneck
- Execution value: 3 — moderate implementation effort (~50 lines); λ adds a new knob;
  the coarse FD estimate is the main uncertainty

### Risk assessment
Medium. The approximate FD on irregular mesh may produce noisy gradients that hurt more
than help. Mitigate by monitoring the physics loss term and starting with small λ=0.001.

---

## Hypothesis 8 — Deep Ensemble of 2–3 Seeds with Prediction Averaging

### What it is
Train 2 or 3 independent models (different random seeds, same hyperparams) and average
their normalized-space predictions at inference. This is the classic deep ensemble
(Lakshminarayanan et al. 2017) — the most reliable Bayesian approximation in practice.
No code change to the model; only the evaluation loop and model-averaging step are new.

### Why it might help here
A KAIST/SNU study (2023) on deep ensembles for aerodynamic regression found 55–56% better
accuracy than GP baselines. The variance reduction effect is strongest on OOD samples
(geom_camber_rc, re_rand) because different seeds find different flat minima that cover
complementary regions of the OOD input space. With the current SWA+Kendall stack, each
seed also has its own uncertainty calibration, making the ensemble diversity higher.
Practically: 2 seeds = ~2× eval cost but 2 models is standard in competition ML and
consistently yields 3–8% improvements on structured regression tasks.

### Key papers
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation Using
  Deep Ensembles" (NeurIPS 2017) https://arxiv.org/abs/1612.01474 — foundational; shows
  ensemble of 5 models gives best OOD calibration; 2–3 models capture 60–75% of gain
- Filos et al., "Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic
  Retinopathy Tasks" (NeurIPS 2019) — systematic comparison showing ensembles > MC dropout
  > variational inference for structured prediction tasks similar to CFD meshes
- Kim & Park, "Deep Ensemble for Aerodynamic Regression with Uncertainty Quantification"
  (KAIST/SNU, 2023) — 55–56% accuracy improvement on aerodynamic surrogate

### Implementation notes
Implementation path:
  1. Train model with `--seed 0`, `--seed 1`, (`--seed 2`) — 3 separate runs
  2. At inference: load all 3 checkpoints, average normalized-space predictions before
     denormalization: `pred_avg = (pred_0 + pred_1 + pred_2) / 3`
  3. Denormalize once: `pred_phys = pred_avg * y_std + y_mean`
  4. Submit ensemble MAE (should be better than any individual model)

The predict-then-average order matters: average in normalized space, NOT physical space.
This is because the denormalization is linear, so the order doesn't matter mathematically,
but averaging in normalized space is cleaner numerically (no outlier amplification).

Known gotcha: ensemble evaluation requires keeping 3 checkpoints in memory simultaneously.
At ~2.5 MB per checkpoint (0.66M params × 4 bytes) this is trivially manageable.

Training cost: 3× the compute. At ~30 min per run, 3 seeds ≈ 90 min. Fits in a 2-hour slot.

### Suggested experiment design
- Run seeds 0, 1, 2 with identical hyperparams (baseline flags)
- Compare: individual seeds vs. 2-seed ensemble vs. 3-seed ensemble
- Key question: how much of the variance is across seeds on geom_camber_rc and re_rand?
  High seed variance → ensemble wins more on those splits; low variance → diminishing returns
- If ensemble wins: apply to every future experiment as a free multiplicative improvement

### Research state update
- Failure mode targeted: high-variance point estimates from a small model trained for
  only 15 epochs on 1499 samples
- Observable that should move: val_avg uniformly; largest gains expected on OOD splits
- Falsifying result: if seed variance on val_avg is <0.5 MAE across seeds, ensembling
  provides no meaningful diversity and the cost is not justified

### Taste rubric
- Research mode: frontier refinement (safe, principled improvement)
- Mechanistic grounding: 3 — well-established mechanism; KAIST aerodynamics study provides
  direct domain evidence; the Kendall uncertainty calibration per seed is a nice property
- Research-state value: 3 — seed variance analysis is a useful diagnostic even if
  ensemble win is small; establishes a floor on the "best possible single-seed" metric
- Execution value: 2 — 3× training cost for an expected 3–8% win; no new mechanism tested;
  better used as a confirmation step after other improvements are compounded

### Risk assessment
Very low technically; moderate on cost. Start with 2 seeds to verify variance before
committing to 3. If seed variance is high (>2 MAE on val_avg), ensembles are high-value.

---

## Hypothesis 9 — Learnable Geometry Curvature as Input Feature (Bold Architecture Input)

### What it is
Compute the signed curvature κ at each mesh node from the `saf` (signed arc-length, dims
2-3) and coordinate data (dims 0-1). Curvature encodes the local geometric shape rate-of-
change — a camber M=8 foil has higher curvature at the leading edge than M=2 — and is
a first-principles indicator of expected pressure gradient magnitude (Cp peaks at high κ).
Append κ as a new feature (dim 24, 1D) to the input, extending x from 24 → 25 dims.

### Why it might help here
The primary bottleneck `geom_camber_rc` tests unseen front-foil camber M=6-8. The existing
shape descriptor `dsdf` (dims 4-11, distance-based) captures global shape topology but does
not directly encode the rate of shape change along the surface. Curvature is the derivative
of the surface tangent — it directly encodes how much the foil surface curves at each point,
which is the property that changes most between M=2 and M=8. Adding κ as an explicit feature
gives the model a direct signal about the geometric extrapolation it is being asked to make.

GAOT (ICLR 2025/2026) uses geometry-adaptive encodings including curvature-derived features
as part of its multi-scale geometric positional embeddings, achieving 9.5% improvement on
3D CFD meshes with unseen geometry families.

### Key papers
- Wu et al., "GAOT: Geometry-Aware Operator Transformer" (ICLR 2025/2026)
  https://arxiv.org/abs/2501.12799 — curvature + normal direction in geometric PE;
  9.5% error reduction on 3D CFD with unseen geometry families
- Lienen & Günnemann, "From Message Passing to Topological Equivariance" (ICML 2023)
  — curvature as a topological invariant for mesh-based learning; important for OOD generalization
- Bhatnagar et al., "Prediction of Aerodynamic Flow Fields Using Convolutional Neural Networks"
  (Computational Mechanics 2019) — curvature features improve airfoil pressure prediction by 7%

### Implementation notes
Curvature computation from `saf` and position:
  - The `saf` feature (dims 2-3) encodes signed arc-length along surface 1 and surface 2
  - For surface nodes: κ ≈ d²r/ds² where r is the position vector and s is arc-length
    Finite difference: κ_i ≈ ||r_{i+1} - 2r_i + r_{i-1}|| / Δs²
    This requires knowing the surface node ordering (which is encoded in `saf` ordering)
  - For volume nodes: κ = 0 (or the distance-weighted interpolation from nearest surface)

Practical implementation:
  - Precompute κ offline for all samples and add as feature dim 24 (1 extra dim)
  - Alternatively, compute on-the-fly in `train.py` using the saf and position features
    (finite difference over sorted saf values within each sample)
  - Normalize κ by standard scaling (its range is O(1/chord) for smooth foils, O(10/chord)
    at leading edge)

Critical gotcha: curvature is only meaningful for surface nodes. For the 85–95% of nodes
that are volume nodes, κ should be set to 0 (or to nearest-surface κ via distance weighting).
If volume nodes all receive κ=0, the model may learn to ignore the feature for volume nodes
and use it only for surface nodes where the metric lives — this is the desired behavior.

### Suggested experiment design
- Compute κ for surface nodes from positions and saf ordering in preprocessing (done in
  train.py __getitem__ or in a precompute step). Set volume κ = 0.
- Append κ as feature dim 24: x → [B, N, 25] (update model's `in_dim` from 24 → 25)
- Run 15 epochs, same other hyperparams
- Key diagnostic: does val_geom_camber_rc improve relative to val_single_in_dist?
  If κ helps geometry OOD but not in-dist, the curvature signal is working as expected

### Research state update
- Failure mode targeted: insufficient geometry representation for OOD camber — existing
  `dsdf` (dims 4-11) does not encode curvature rate-of-change explicitly
- Observable that should move: val_geom_camber_rc primarily; val_geom_camber_cruise secondarily
- Falsifying result: if adding κ does not improve geom_camber splits vs. in-dist splits,
  the geometry bottleneck is not in the input representation but in the model's mapping

### Taste rubric
- Research mode: frontier refinement (input feature engineering for a known bottleneck)
- Mechanistic grounding: 4 — curvature is a direct mathematical encoding of camber variation;
  GAOT provides external evidence; the feature is orthogonal to all existing input dims;
  the link to geom_camber_rc bottleneck is precise and physically motivated
- Research-state value: 3 — if it wins on geom_camber splits specifically, narrows the
  geometry-OOD bottleneck to input representation; if uniform improvement across splits,
  the signal is generally useful but the mechanism is less clear
- Execution value: 4 — low code complexity, small input extension, directly targets
  the primary OOD bottleneck with a physically grounded feature

### Risk assessment
Low. Computing curvature from existing features (position, saf) requires only finite
differences. The main risk is incorrect saf ordering for curvature FD — validate that
κ values are physically reasonable (check max κ ≈ 1/min_radius_of_curvature for the foil).

---

## Hypothesis 10 — Huber β Warm-Up Schedule (Anneal from Robust to L2)

### What it is
Start training with a large Huber β (e.g., β=5.0, nearly L1/robust) and anneal it toward
a smaller β (e.g., β=0.5 or L2 limit) over training. This combines the outlier robustness
of L1 in early epochs (when predictions are far from ground truth and gradients from extreme
Re samples would otherwise dominate) with the precise fitting of L2 in later epochs (when
predictions are close and fine-grained gradient signal matters). The β=0.3 experiment (#1757,
frieren) is running but tests only a static β. The warm-up annealing is a different mechanism.

### Why it might help here
The TandemFoilSet has extreme per-sample y-std variation (up to 10× within a split due to
Re range). At epoch 1, the model's predictions have O(100) MAE while the target can be
O(10000) for high-Re samples — those large errors produce huge L2 gradients that overwhelm
the optimization. A large initial β acts as L1 for these outlier errors, keeping the gradient
magnitude bounded. As training progresses and per-sample errors shrink, a smaller β is better
suited for precision around the fine-grained surface pressure gradients. The transition should
be ~epoch 5 (when per-sample errors are typically within 2× of ground truth).

### Key papers
- Barron, "A General and Adaptive Robust Loss Function" (CVPR 2019)
  https://arxiv.org/abs/1701.03077 — adaptive loss that generalizes Huber; shows that
  annealing the robust parameter α toward L2 improves final accuracy after early robustness
- Menon et al., "Long-Tail Learning via Logit Adjustment" (ICLR 2021) — curriculum-style
  loss scheduling for distributions with long-tail Re values

### Implementation notes
Schedule: `β_t = β_max * (β_min / β_max)^(t / T)` — exponential decay from β_max=5.0
  to β_min=0.5 over T=10 epochs. Linear annealing also works.
CLI addition: `--huber_beta_start 5.0 --huber_beta_end 0.5 --huber_beta_anneal_epochs 10`
  (constant at β_min for remaining epochs 10–15)

Alternative inspired by Barron (2019): use the fully general adaptive loss
  `ρ(x, α, c) = |α-2|/α * (x²/c² / |α-2| + 1)^(α/2) - 1`
  and anneal α from 0 (Cauchy, very robust) → 2 (L2). This requires ~20 lines but
  covers the full robustness spectrum in a single loss function.

### Suggested experiment design
- Simple variant: `--huber_beta 5.0` for epochs 0-4 → anneal linearly to `β=0.5` by epoch 9
  → hold at β=0.5 for epochs 10-15
- This is distinct from the running #1757 which tests static β=0.3 (different mechanism:
  fixed robustness level vs. annealed robustness)
- Key diagnostic: monitor per-epoch training loss for high-Re samples separately —
  does early L1-like training prevent gradient explosions that damage later convergence?

### Research state update
- Failure mode targeted: early-training gradient instability from extreme Re samples
  overwhelming optimization before the model has a good prior
- Observable that should move: training loss convergence speed (early epochs should be
  more stable); val_re_rand (Re-stratified split benefits from better Re-regime handling)
- Falsifying result: if β-annealed training converges at the same rate and to the same
  final val_avg as static β=1.0 (current), early Re-outlier gradients are not the bottleneck

### Taste rubric
- Research mode: frontier refinement (loss function tuning orthogonal to running #1757)
- Mechanistic grounding: 3 — Barron (2019) provides strong evidence for adaptive loss;
  the Re-extreme-gradient mechanism is plausible; direct test of whether early robustness
  helps convergence in this high-dynamic-range regression setting
- Research-state value: 3 — result distinguishes whether fixed vs. annealed Huber β
  matters; orthogonal to the static β=0.3 test in #1757
- Execution value: 3 — small code change, 15-epoch run, diagnostic via per-epoch train loss

### Risk assessment
Low-medium. The main risk is that β annealing introduces a new hyperparameter schedule
that requires tuning (start/end/annealing duration). Mitigate by testing a simple linear
schedule first before optimizing the schedule parameters.

---

## Summary Table

| # | Hypothesis | Target failure mode | Expected primary win | Boldness | Effort |
|---|-----------|--------------------|--------------------|---------|--------|
| 1 | SWAG Bayesian SWA | OOD calibration + SWA truncation | geom_camber_rc, re_rand | Medium | Medium |
| 2 | Adaptive gradient scaling | Optimizer discontinuity from 99.2% clip | val_avg uniform | Low | Very Low |
| 3 | Multi-σ RFF (0.3, 1.0, 4.0) | Spectral bias at single spatial scale | geom_camber_rc | Low | Very Low |
| 4 | Multiscale slice attention | Flat slices miss near-wall vs far-field | geom_camber_rc | High | High |
| 5 | Physics-stratified MoE routing | Shared weights dilute domain-specific learning | geom_camber splits | High | High |
| 6 | Extended epochs + wider SWA window | SWA truncation (2 epochs → 5+ epochs) | re_rand, geom splits | Low | Zero |
| 7 | Physics-residual continuity loss | No Ux/Uy coupling constraint | geom_camber splits | Medium | Medium |
| 8 | Deep ensemble 2–3 seeds | High-variance point estimate | val_avg uniform | Low | Medium |
| 9 | Geometry curvature feature | Insufficient curvature info for OOD camber | geom_camber_rc | Low-Med | Low |
| 10 | Huber β warm-up schedule | Early Re-extreme gradient instability | val_re_rand, convergence | Low | Low |

## Prioritized Top 5 (for immediate assignment)

1. **Hypothesis 6 (Extended epochs)** — zero code change, high information value, tests
   SWA truncation hypothesis directly; should be first because it establishes whether the
   SWA truncation is the main bottleneck

2. **Hypothesis 3 (Multi-σ RFF)** — one-line code change, directly extends a known winner,
   targets the primary OOD bottleneck with physically grounded spectral coverage

3. **Hypothesis 9 (Curvature feature)** — low code complexity, physically grounded, orthogonal
   to all existing features, directly targets geom_camber_rc with a new input signal

4. **Hypothesis 2 (Adaptive gradient scaling)** — one-line swap, diagnostic of the 99.2%
   clip-fraction anomaly; result either fixes optimizer discontinuity or rules it out cleanly

5. **Hypothesis 1 (SWAG)** — principled OOD improvement with external evidence from Maddox
   et al.; combines naturally with the extended-epoch fix from Hypothesis 6
