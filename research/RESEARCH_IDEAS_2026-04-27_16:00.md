# RESEARCH IDEAS — 2026-04-27 16:00 (charlie-r3 round 1)

Fresh hypothesis pool for the ICML appendix CFD-surrogate (TandemFoilSet) track.
Primary metric: `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE
across `val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`,
`val_re_rand`. **Lower is better.**

Current merged baseline (from kagent_v_students round): val_avg/mae_surf_p ~49.4
on the recipe (Transolver, nl=3, sn=8, n_head=4, n_hidden=128, mlp_ratio=2,
SwiGLU FFN, fixed Gaussian Fourier PE m=160 sigma=0.7, L1 loss surf_weight=1,
AMP + grad_accum=4, AdamW 5e-4 cosine 50ep).

## What looks under-explored on the new strong base

- **Physics-grounded loss terms.** No prior tried Bernoulli/incompressibility
  residuals on this base; pressure–vorticity coupling untried. asinh
  reparameterization (PR #9) failed on the *old* base — may now be salvageable.
- **Boundary-layer specialists.** Cross-attn surface decoder (PR #18) failed
  on old weak base. Now nl=3+sn=8 leaves ample param budget for a small surface
  refiner head.
- **Equivariance / symmetry.** Nothing tried beyond raw (x,z) Fourier features;
  no SE(2) or reflection equivariance has been imposed. Re scaling (PR #31)
  was post-hoc only — never internalized.
- **Optimization.** Lookahead, SAM, SWA, model-soup snapshot ensembling have
  not been tried at all. Cosine min_lr / warmup is half-finished (PR #40 WIP).
- **Token / slice economy.** Slice count was floor-mapped (sn=8) but the
  slice-token attention has never been augmented with cross-slice gating,
  mixture-of-experts routing, or token-pruning. Multi-resolution pooling
  unattempted.
- **Target representation.** Pressure asinh failed on weak base. p ≈ ½ρU² so
  Cp = (p − p_∞)/(½ U_ref²) is *the* canonical CFD target — never attempted.
- **Augmentation.** H-flip failed on old base, but the previous failure mode
  (under-trained, sn=64) may be moot now. Re-dilation augmentation along the
  Re axis is wholly untried.

The 14 ideas below are ranked by expected impact (rough estimate of % drop
in `val_avg/mae_surf_p`) × probability of success.

---

## Ranked hypothesis pool

### 1. **Pressure-coefficient (Cp) target reparameterization**
- **Hypothesis:** Train the model to predict the dimensionless pressure
  coefficient `Cp = (p − p_∞) / (½ U_ref²)` instead of raw kinematic pressure,
  then invert to physical p before loss/MAE. Use feature-derived
  `U_ref = ν · Re / L_ref` (or constant L_ref=1) so Cp lives in O(1) regardless
  of Reynolds number.
- **Why might it work:** Pressure variance scales like U². Across the train
  corpus Re spans ~50× and y-std varies an order of magnitude *within* one
  domain. The model is currently forced to learn the U² scaling implicitly via
  features 13 (`log Re`) and the AoA inputs. Cp absorbs that scaling
  analytically, so the network only has to learn the *shape* of the pressure
  field. The MAE leaderboard is sensitive to high-Re samples (where errors are
  largest); collapsing them onto the same scale should disproportionately help
  surface-p.
- **Predicted Δ on val_avg/mae_surf_p:** −10% to −20% (best-case −30% if it
  also flattens cross-Re generalization on `val_re_rand`).
- **Implementation sketch:** In `train.py`, augment the y-normalization with a
  per-sample `U_ref²` divisor for the p channel (compute from feature 13:
  `Re = exp(log_Re)`, choose ν=1.5e-5 m²/s and L=1 as constants — they're
  conventional and cancel in MAE differences). Loss in Cp space, then convert
  predicted Cp back via `p_pred = ½ U_ref² · Cp_pred` for MAE. Keep velocity
  channels in normalized-velocity space (no change). Variant: also predict
  `(Ux, Uy)/U_ref` for full nondimensionalization.
- **Risk / failure mode:** Wrong choice of `U_ref` for low-Re samples can blow
  up Cp; clamp or use a learned per-sample scale head on top of `log Re`.
  If the conversion factor is noisy at the surface (`U_ref` is freestream not
  local), surface error may not improve.

### 2. **Bernoulli residual auxiliary loss (steady, incompressible, irrotational
   away from wake)**
- **Hypothesis:** Add an auxiliary loss `L_bern = MSE(p + ½(Ux²+Uy²) −
  H_local)` evaluated at every volume node, where `H_local` is computed from
  the *predicted* fields and supervised against the same combination from the
  *ground-truth* fields. This couples velocity and pressure errors so they
  cannot drift independently.
- **Why might it work:** The current loss treats Ux, Uy, p as independent
  channels. In incompressible CFD, the steady Bernoulli relation links them
  along streamlines. A residual loss on `p + ½|U|²` regularizes the joint
  prediction toward a physically consistent solution; surface pressure errors
  that come from a velocity offset are directly penalized.
- **Predicted Δ:** −5% to −10%. Probably small but consistent.
- **Implementation sketch:** In the training loop, after computing `pred`,
  compute `H_pred = pred[...,2] + 0.5*(pred[...,0]**2 + pred[...,1]**2)` and
  `H_true = y[...,2] + 0.5*(y[...,0]**2 + y[...,1]**2)`, both in **physical**
  space (denormalize first). Loss term `lambda * F.l1_loss(H_pred[mask],
  H_true[mask])` with `lambda` swept over `{0.05, 0.1, 0.25, 0.5}`.
  Apply only to volume nodes (Bernoulli holds outside the boundary layer).
- **Risk / failure mode:** In the boundary layer Bernoulli does not hold, so
  applying it everywhere may drag surface predictions in the wrong direction.
  Mitigate by masking to volume-only nodes or by weighting down close to the
  wall using DSDF feature 4–11.

### 3. **Surface refiner cross-attention head (revival on strong base)**
- **Hypothesis:** Add a small, surface-only cross-attention block after the
  last Transolver block: surface nodes (Q) attend to *all* nodes (K,V) once,
  followed by a 2-layer MLP to refine surface predictions only. Volume
  predictions stay from the trunk.
- **Why might it work:** Surface pressure depends on a *non-local* integral of
  the field (induced velocity from the entire wake/foil), so a global cross-
  attention is well-motivated. PR #18 failed on the old (nl=5, sn=64) wide
  base because it added params on top of an already-saturated model. With
  nl=3, sn=8 we now have spare param budget. With sn=8 slice tokens, the
  effective receptive field of the trunk is *coarser*; a local-to-global
  refiner closes that gap precisely at the surface, where the metric lives.
- **Predicted Δ:** −5% to −15% on surface_p, probably small or zero on volume.
- **Implementation sketch:** In `Transolver.forward`, before the last block's
  `mlp2` projection, compute `surf_idx = is_surface_mask` (passed via `data`).
  Take `q = fx[surf_idx]`, `k = v = fx`, do one head of multihead cross-
  attention with hidden_dim=128, residual + LayerNorm + 2-layer MLP. Replace
  `mlp2(fx)[surf_idx]` with the refined output. Use chunked attention or
  `F.scaled_dot_product_attention` for memory.
- **Risk / failure mode:** Memory: surface nodes can still be 5K-10K, K is
  240K. Use FlashAttention-style chunking; if VRAM blows up, sub-sample K to
  a slice-token-pooled summary. Could overfit on geom-camber splits; use
  dropout p=0.1 in the refiner.

### 4. **Stochastic Weight Averaging (SWA) over the cosine tail**
- **Hypothesis:** After epoch 35 (when LR is small), maintain a running
  average of the model weights and use that average for validation /
  checkpointing. Standard SWA: simple averaging, no extra batch-norm
  recompute (we use LayerNorm).
- **Why might it work:** SWA converges to flatter minima — known to help on
  small dataset surrogate tasks. With cosine annealing, the late-epoch
  weights are sampling around a minimum; the average is closer to the
  centroid of that minimum than any single epoch. **Free** parameter
  improvement (no compute overhead).
- **Predicted Δ:** −2% to −7%.
- **Implementation sketch:** Use `torch.optim.swa_utils.AveragedModel(model)`
  initialized at epoch 35. Update each epoch via `swa_model.update_parameters
  (model)`. Validate **both** `model` and `swa_model`; keep best of either.
  No SWA-LR scheduler — just standard cosine.
- **Risk / failure mode:** If the cosine min isn't deep enough, SWA may mix
  pre-convergence weights and actually degrade. Mitigate by starting SWA at
  the half-way point or once `val_avg/mae_surf_p` stops decreasing 3 epochs
  in a row.

### 5. **Sharpness-Aware Minimization (SAM) for surface MAE generalization**
- **Hypothesis:** Replace the AdamW step with SAM (rho=0.05). Two forward/
  backward passes per step but the perturbed gradient is at the descent step
  ascent in loss space — empirically yields flatter minima and better OOD
  generalization. Critical for the 3 OOD val splits which dominate the metric.
- **Why might it work:** Three of the four val tracks are OOD (geom_camber_rc,
  geom_camber_cruise, re_rand). SAM is well-known to improve OOD by selecting
  flatter minima. Doubling effective compute is offset by smaller batch (we
  already have grad_accum=4 → can go grad_accum=2 with SAM).
- **Predicted Δ:** −5% to −12%, mostly on the OOD splits. May *raise*
  in-distribution slightly.
- **Implementation sketch:** Implement SAM inline (no new package): two
  passes — first computes `epsilon = rho * grad / ||grad||`, applies it,
  recomputes loss/grad, undoes perturbation, applies update. Compare
  `rho ∈ {0.01, 0.05, 0.1}`. Keep grad_accum=2 to fit in time budget.
- **Risk / failure mode:** Doubles per-step time; need to drop grad_accum to
  stay under 30-min cap. Might run out of epochs. Profile first epoch and
  drop epoch count if needed (better SAM on 35 epochs than no-SAM on 50).

### 6. **Cross-slice gated MoE in the FFN of layer 2**
- **Hypothesis:** Replace the SwiGLU FFN in the middle Transolver block with
  a 4-expert mixture-of-experts where expert routing is conditioned on the
  *slice-token assignment weights*, not on the input feature directly. Each
  expert specializes for a different region (boundary layer vs. wake vs.
  outer flow vs. trailing edge).
- **Why might it work:** The slice attention already partitions tokens into
  sn=8 implicit regions; a per-region FFN amplifies that specialization
  without adding many params (only one block has MoE). Conditional compute
  also lets us scale capacity for the boundary layer specifically.
- **Predicted Δ:** −3% to −10%.
- **Implementation sketch:** In layer index `n_layers // 2`, replace the FFN
  with `MoE(num_experts=4, top_k=1, gate_in=slice_weights.mean(slice_dim))`.
  Each expert is a small SwiGLU (hidden_dim, 2*hidden_dim). Add load-
  balancing aux loss `aux_w=0.01`.
- **Risk / failure mode:** Routing collapse to a single expert. Mitigate with
  load-balancing loss and gumbel-softmax exploration in early epochs.

### 7. **Learnable Fourier basis with per-channel frequency adaptation**
- **Hypothesis:** Replace the *fixed* Gaussian Fourier PE (m=160, sigma=0.7)
  with a *learnable* Fourier basis where frequencies are initialized from the
  same Gaussian but become trainable parameters. Optionally use SIREN-style
  modulation: input-dependent amplitude `A(x) · sin(W·x + b)`.
- **Why might it work:** Optimal frequency content for the foil-surface
  pressure differs from the global wake. A fixed bandwidth is a compromise.
  Learnable frequencies adapt sigma to whatever the loss prefers, and SIREN
  amplitude modulation can suppress high-frequency content far from the foil.
- **Predicted Δ:** −2% to −8%. (Risk that learnable PE just learns the same
  basis if regularized poorly.)
- **Implementation sketch:** Replace the registered `gaussian_b` buffer with
  a `nn.Parameter`. Add SIREN-amplitude as a 1-layer MLP from `(saf, dsdf)`
  features. Sweep `lr_pe ∈ {1e-4, 1e-3, 5e-4}` (separate param group with
  100× smaller lr because frequencies are sensitive).
- **Risk / failure mode:** Can drive frequencies into a degenerate
  high-frequency regime causing overfitting. Apply weight decay to PE
  weights and clamp `||W_pe|| < 5*sigma_init`.

### 8. **Asinh pressure target — REVIVAL with surf_weight=1**
- **Hypothesis:** Re-run the asinh(p/scale) target reparameterization (failed
  in PR #9 on old base) on the new strong base. asinh compresses the heavy
  tail of high-Re pressure values without losing sign information, which
  should help with the y-std heterogeneity (max sample std = 2,077 vs mean
  458 in single_in_dist).
- **Why might it work:** PR #9 failed because the old base used surf_weight=10
  and L2 loss — both already amplifying outliers. With L1 + surf_weight=1, the
  loss surface is much more uniform, so asinh's variance reduction may now
  show up cleanly. The MAE in physical space is what we care about and asinh
  is monotonic, so a per-sample inverse is cheap.
- **Predicted Δ:** −2% to −8%. May fail again, but the failure mode has
  changed — worth one slot.
- **Implementation sketch:** y-norm becomes
  `y_norm[..., 2] = asinh(y[..., 2] / scale_p) / asinh_norm` (plus standard
  zero-mean for the velocity channels). At inference, invert via `sinh`. Try
  `scale_p ∈ {500, 1000, 2000}` based on per-domain stats.
- **Risk / failure mode:** Inverse `sinh` is very sensitive to predictions
  near the saturation regime. If the network slightly over-predicts, the
  physical-space error explodes. Mitigate by normalizing to `[-3, 3]` range
  and clamping inversion input.

### 9. **Reflection-equivariant feature lift (vertical flip ↔ AoA sign flip)**
- **Hypothesis:** Add an architectural symmetry: process every sample twice
  (original + vertical flip with sign-flip on Uy and AoA) and average the
  predictions. Train with both copies in the same step. This bakes the
  geometric symmetry of the problem into the model (an inverted foil at +AoA
  is the same physics as a non-inverted foil at -AoA).
- **Why might it work:** The cruise domain (AoA ∈ [-5°, +6°]) and raceCar
  (inverted, AoA ∈ [-10°, 0°]) span very different sign patterns, but the
  underlying physics is the same when reflected. Equivariance halves the
  effective dataset variance and should especially help the geom_camber
  splits.
- **Predicted Δ:** −5% to −12% (mostly geom_camber tracks, modest on others).
- **Implementation sketch:** In the training step, after loading the batch,
  sample a Bernoulli(p=0.5) per sample for "flip mode": negate `x[..., 1]`
  (z-coord), `x[..., 14]` and `x[..., 18]` (AoA) and `y[..., 1]` (Uy). Two
  forward passes through model — one original, one flipped — average loss.
  At validation, do test-time augmentation: predict both, flip back, average.
- **Risk / failure mode:** TTA at val time doubles eval cost and pushes
  against the 30-min budget. Drop training-time aug if too slow and use
  TTA only.

### 10. **Snapshot ensemble / Cosine-Restart model soup**
- **Hypothesis:** Use cosine annealing **with restarts** (T_0=15, T_mult=1)
  to produce 3 snapshot weights at the bottom of each cycle. At inference,
  *uniformly average* their state dicts (model soup) before evaluation.
- **Why might it work:** Snapshot ensembling at near-zero cost in compute,
  and uniform soup of weights from the same training trajectory has been
  shown (Wortsman et al.) to outperform any single snapshot. Three checkpoints
  near three minima of a restart cosine give a good ensemble for free.
- **Predicted Δ:** −3% to −7%.
- **Implementation sketch:** Replace `CosineAnnealingLR` with
  `CosineAnnealingWarmRestarts(T_0=15, T_mult=1)`. Save snapshots at the end
  of each cycle (epochs 15, 30, 45). After training, average state-dicts
  *parameter-wise* (only weights of same shape) and re-evaluate the soup.
- **Risk / failure mode:** Restarts can throw the model out of a deep
  minimum; need to confirm each snapshot still validates well. If any
  snapshot is much worse, exclude it from the soup (greedy soup).

### 11. **Re-conditioned AdaLN: condition every block's LayerNorm on log Re**
- **Hypothesis:** Replace each block's `nn.LayerNorm` with an Adaptive
  LayerNorm whose scale and shift are linear functions of `log Re`. This
  makes Reynolds number a *first-class* conditioning signal instead of just
  another input feature — emulates the conditioning style of DiTs.
- **Why might it work:** The Re axis is the single biggest source of variance
  in the dataset (per-sample y-std varies 10× with Re inside one domain). The
  model currently has to allocate capacity to learn Re-dependent feature
  scaling implicitly. AdaLN gives that scaling for free, parameterized by
  log Re. Should especially help `val_re_rand`.
- **Predicted Δ:** −3% to −10%, weighted toward the re_rand split.
- **Implementation sketch:** Build `class AdaLN(nn.Module)` that takes
  `(x, c)` where c is `log Re` (1-d). Predict `gamma, beta` via 2-layer MLP
  on c, apply `gamma * LN(x) + beta`. Replace `ln_1, ln_2, ln_3` in
  `TransolverBlock`. Plumb `log_Re = x[:, 0, 13]` through forward kwargs.
- **Risk / failure mode:** AdaLN doubles LN params; may overfit. Use weight
  decay 1e-3 on the AdaLN MLPs specifically.

### 12. **Signed-distance enriched pre-processing (geometry encoder lift)**
- **Hypothesis:** Augment the 8-d distance descriptor (features 4–11) with
  a **signed log-distance to nearest foil surface**, computed analytically
  from the NACA profile + AoA + position. This adds a clean, smooth signed
  field that is currently approximated by the raw `dsdf` features.
- **Why might it work:** A signed distance function is the canonical level-
  set representation of a foil surface; it's smoother, sign-aware, and
  asymptotic-friendly (`log|dist|` linearizes the boundary-layer scaling).
  The dataset already has `dsdf` but it's an 8-d distance descriptor without
  sign. Adding a clean signed log-dist as a 9th feature should help the
  model identify near-wall regions.
- **Predicted Δ:** −2% to −7%.
- **Implementation sketch:** In `train.py`, before normalization, compute
  signed distance via a NACA-profile parametric query: sample 256 surface
  points from the NACA-4 formula at AoA, KD-tree nearest-neighbor, sign
  via inside-outside test on a closed polygon. Add as feature 24
  (input dim 24 → 25; preprocess MLP grows by 256 params). Cache per-batch
  on GPU; ~50ms overhead per batch.
- **Risk / failure mode:** Slow if poorly vectorized. NACA-4 parametric form
  must be implemented in pytorch on GPU. If too slow, fall back to caching
  per-sample SDF in a `data/` shim *outside* the loader (still in train.py
  via a one-off precomputation in the train loop init).

### 13. **Two-stage training: pretrain on volume only → fine-tune on surface**
- **Hypothesis:** First 35 epochs use `surf_weight=0.1` (volume-dominated),
  last 15 epochs ramp `surf_weight` to 5.0 (surface-dominated). The volume-
  first phase learns a globally consistent flow field; the surface-finetune
  phase polishes the boundary.
- **Why might it work:** Surface predictions depend on far-field information
  (induced velocity from wake, freestream). If the network's volume
  predictions are noisy, the surface predictions inherit that noise. Pre-
  training the volume forces the network to commit to a coherent flow
  representation before pushing surface accuracy.
- **Predicted Δ:** −3% to −8%.
- **Implementation sketch:** In the training loop, schedule `surf_weight`
  via `cfg.surf_weight = max(0.1, 5.0 * (epoch - 35) / 15)` after epoch 35,
  else 0.1. Sweep cross-over epoch ∈ {25, 30, 35, 40}.
- **Risk / failure mode:** During the volume-dominant phase the surface
  validation will look terrible — checkpoint selection by surface MAE may
  pick an early-phase checkpoint. Mitigate by always checkpointing on a
  blended `0.5 vol + 0.5 surf` MAE in the early phase.

### 14. **POD-prior token initialization (Schmidhuber-style classical revival)**
- **Hypothesis:** Initialize the slice-attention's slice projection
  (`in_project_slice.weight`) with the top-`sn=8` POD modes computed offline
  on the training pressure field, instead of orthogonal random. POD modes
  are the optimal linear basis for variance — a classic CFD reduced-order
  modelling technique going back to Sirovich (1987).
- **Why might it work:** With sn=8, the slice tokens are doing the work of a
  reduced basis. POD modes give the *empirically optimal* 8-d basis on the
  training data. Random init throws this away. Even if the network learns
  past it, the better init means faster convergence and a better local
  minimum within the 50-epoch budget.
- **Predicted Δ:** −2% to −6%. Possibly higher in OOD splits where POD modes
  capture the dominant flow patterns.
- **Implementation sketch:** Pre-compute (one-time, outside train.py via
  inline torch code on first epoch start) the SVD of a `[N_train_total,
  3]` matrix of (Ux, Uy, p) values from the training set, take top-8
  right-singular vectors. Project them back to a 32-d-per-head subspace
  to match `dim_head` and assign to `in_project_slice.weight` of every
  block. Allow further training; this is just init.
- **Risk / failure mode:** SVD on 70M+ nodes is expensive; subsample 1%.
  POD on velocity/pressure is in 3-d output space, not slice-feature space
  — need a projection step. Easier alternative: PCA on `preprocess(x)`
  outputs after one warm-up epoch.

---

## Proposed assignment matrix (8 idle students × 14 ideas)

If 8 students are simultaneously available, prioritize as follows:

| Slot | Hypothesis | Tier |
|------|-----------|------|
| 1 | **Cp reparameterization** (#1) | Tier-1: target re-formulation |
| 2 | **Bernoulli residual loss** (#2) | Tier-1: physics constraint |
| 3 | **Surface refiner cross-attn** (#3) | Tier-1: architecture novelty |
| 4 | **SAM optimizer** (#5) | Tier-1: OOD generalization |
| 5 | **AdaLN on log-Re** (#11) | Tier-1: architectural conditioning |
| 6 | **SWA over cosine tail** (#4) | Tier-2: free improvement |
| 7 | **Reflection equivariance** (#9) | Tier-2: symmetry exploitation |
| 8 | **Snapshot ensemble / soup** (#10) | Tier-2: free improvement |

Reserve ideas #6 (MoE FFN), #7 (learnable Fourier), #8 (asinh revival),
#12 (signed-distance), #13 (two-stage), #14 (POD init) for round 2 once
the first batch reports.

## Combinations worth flagging for round 2

- Cp + Bernoulli residual: complementary (target reframe + joint constraint)
- AdaLN + SWA: both nearly free
- Surface refiner + cross-attn TTA via reflection: surface focus stacks
- SAM + snapshot soup: flatness-of-minima compounds with snapshot averaging
