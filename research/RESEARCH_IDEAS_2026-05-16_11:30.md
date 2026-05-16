<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-16 11:30

Generated for `willow-pai2i-24h-r3` / advisor branch `icml-appendix-willow-pai2i-24h-r3`.

Baseline: val_avg/mae_surf_p = 65.2991, test_avg_nansafe/mae_surf_p = 60.5400 (frieren #3675).
Stack: Lion lr=2e-4, wd=1e-2, Huber δ=2.0, bf16, clip=1.0, eta_min=1e-5, T_max=21.

All ideas below have been verified as NOT tried or in-flight as of this writing.

---

## Ranking summary (1 = highest priority)

| Rank | Slug | Mechanism level | Scope | OOD-targeted |
|------|------|----------------|-------|--------------|
| 1 | slice-num-128-stability | Architecture fix | Medium | Yes |
| 2 | domain-id-feature | Data representation | Small | Yes |
| 3 | swa-post-training | Optimization | Small | No |
| 4 | divergence-free-loss | Physics-informed loss | Medium | Yes |
| 5 | lookahead-lion | Optimizer wrapper | Small | No |
| 6 | layerwise-lr | Optimization | Small | No |
| 7 | re-conditioned-normalization | Data representation | Medium | Yes (val_re_rand) |
| 8 | fft-domain-loss | Loss formulation | Medium | Yes |
| 9 | curriculum-by-re | Data sampling | Small | Yes (val_re_rand) |
| 10 | gumbel-topk-slices | Architecture | Large | Speculative |
| 11 | surf-vol-loss-decoupled-clip | Loss formulation | Small | No |
| 12 | eta-min-ratio-restore | LR schedule | Small | No |

---

## Idea 1: slice-num-128-stability

**Hypothesis:** PhysicsAttention with slice_num=128 produces ±inf because softmax receives large logits in bf16; clamping attention logits and upcasting the softmax computation to fp32 will stabilize it — unlocking the higher slice resolution that showed best-ever cruise val=104.24 before being closed for instability.

**Mechanism / why:** The PhysicsAttention bug at slice_num=128 is numerically induced, not structural. In bf16, large dot-products exceed the representable range, producing inf before softmax normalizes them. A targeted fix — clamp(logits, -80, 80) + softmax in fp32 — is standard practice in bf16 transformers (used in Flash Attention, Llama, and every production LLM). The higher token count (128 vs 64 slices) gives the attention heads a coarser-to-finer view of the mesh and should help especially on the large cruise meshes (~210K nodes).

**Why it hasn't been tried:** The original bug was discovered mid-run (PR #3247) and the decision was to close rather than fix. The fix is a targeted surgery inside PhysicsAttention's forward pass — one clamp + dtype cast — without touching any training logic.

**Code-change scope:** Medium. Requires locating the PhysicsAttention forward pass in the Transolver source, adding a logit clamp and fp32 softmax cast, then re-running the slice_num=128 experiment with the guard in place. Must be paired with the stability guard in the experiment PR.

**Predicted improvement direction:** Largest potential gain on val_geom_camber_cruise and val_re_rand (big meshes where higher slice resolution matters most). Could be negative on val_single_in_dist (smaller meshes, 128 slices may overpartition). Net effect uncertain but cruise val=104.24 at epoch ~5 on the OLD stack suggests very high ceiling once training completes on the SOTA stack.

**OOD targeted:** Yes — cruise meshes are the persistent OOD hard case.

**Suited student:** alphonse (currently on slice_num sweep, familiar with PhysicsAttention and slice_num parameter space).

---

## Idea 2: domain-id-feature

**Hypothesis:** Injecting a one-hot domain identifier (3 dims: single/raceCar/cruise) as extra input features x[:,24:27] will sharpen the model's distribution head for each domain and improve OOD generalization on the camber splits.

**Mechanism / why:** The three training domains have structurally different mesh sizes, Re ranges, AoA ranges, and camber distributions. The model currently infers domain membership implicitly from dims 18–23 (foil 2 AoA/NACA/gap/stagger being zero for single-foil). A one-hot domain tag is a hard, noiseless signal that removes this implicit lookup and lets the model condition each layer's activations on which domain it is processing. This is directly analogous to task-conditioning in multi-task learning and domain adaptation via feature augmentation — both have strong empirical records. The change is entirely in train.py's feature-construction pass before normalization; data/ files are untouched.

**Mechanism — why it should help OOD:** The camber holdout splits (P2 for each domain) have identical domain IDs to the corresponding training data. With domain ID conditioning the model can apply domain-specific activation statistics rather than confounding single-foil and cruise geometry.

**Code-change scope:** Small. Add one-hot domain feature to x in train.py's data loading loop (derivable from dims 18–22), update model's input_channels from 24 to 27, keep normalization stats consistent (one-hot features don't need z-scoring; optionally append zeros to x_mean/x_std). No data/ changes needed.

**Predicted improvement direction:** Moderate improvement on val_geom_camber_cruise and val_geom_camber_rc; neutral or small gain on val_single_in_dist and val_re_rand.

**OOD targeted:** Yes — primary OOD axes are camber (domain-specific).

**Suited student:** tanjiro (has worked with scheduling and input-level changes; small-scope assignment fits a quick confirmation run).

---

## Idea 3: swa-post-training

**Hypothesis:** Applying Stochastic Weight Averaging (SWA) over the last 5–7 checkpoints (epochs 14–19 on the T_max=21 schedule) will smooth the loss landscape and produce a generalization improvement distinct from EMA.

**Mechanism / why:** SWA averages parameter snapshots taken at different points along the trajectory, not an exponential moving average of all steps. EMA (edward's in-flight experiment, PR #3640) and SWA operate differently: EMA tracks a running mean of all parameter updates throughout training (high decay d=0.999 → slow forgetting), while SWA explicitly averages a finite set of late-epoch checkpoints. On the SOTA stack the best checkpoint is epoch 19 (training still descending at timeout), meaning there is a sequence of 5–7 checkpoints near the minimum. SWA over those checkpoints is a cheap, theoretically motivated post-hoc operation that has shown consistent ~0.5–2% gains on image tasks and weather/PDE surrogates. It was independently validated in the original SWA paper (Izmailov et al. 2018) and consistently in downstream CFD surrogate literature.

**EMA vs SWA distinction:** EMA updates continuously during training and requires a separate shadow-parameter store. SWA aggregates finished checkpoint files. They are mechanistically orthogonal and can stack. The round-6 EMA result on T_max=21 stack showed only −0.12 win (shrunk from −5.34 on old stack), suggesting EMA's marginal value is low. SWA from a different epoch window may capture a different basin average.

**Code-change scope:** Small. After training completes, load the last K checkpoint files, average their state_dicts, run a single BN-update pass (not applicable here — no BN), and evaluate. Alternatively, save checkpoints at epochs 14–19 and implement the average in train.py's post-training block.

**Predicted improvement direction:** Small-to-moderate improvement on all splits; by construction it cannot do worse than the best single checkpoint if the sweep uses the correct epoch window.

**OOD targeted:** No — generalization improvement is domain-agnostic.

**Suited student:** edward (familiar with weight-averaging mechanics from EMA work on PR #3640; SWA is a natural extension of his current direction).

---

## Idea 4: divergence-free-loss

**Hypothesis:** Adding a soft incompressibility penalty L_div = mean(|∂Ux/∂x + ∂Uy/∂z|) over volume nodes will inject a physical conservation law as a loss term, acting as a physics-informed regularizer that should improve OOD generalization on cruise geometry.

**Mechanism / why:** Incompressible 2D flow satisfies ∇·u = 0 exactly. The model currently has no mechanism to enforce this — it learns it implicitly from data. On OOD geometry (unseen camber), the physical constraint acts as an anchor: a model that respects mass conservation will extrapolate more faithfully because the conservation law holds regardless of airfoil shape. This is the fundamental motivation behind physics-informed neural networks (PINNs) and their successors (DeepONet + physics constraints, FNO with divergence regularizer). The implementation requires approximate spatial gradients on the irregular mesh — a finite-difference approximation using nearest-neighbor node pairs is sufficient and avoids the need for a full FEM stencil.

**Implementation note:** Mesh nodes are stored with (x, z) coordinates in dims 0–1 of the input. For each volume node i, find its k=4 nearest neighbors (precomputable per sample), estimate ∂Ux/∂x ≈ (Ux_j - Ux_i) / (x_j - x_i) and ∂Uy/∂z ≈ (Uy_j - Uy_i) / (z_j - z_i) via central differences, and add λ_div * mean(|∂Ux/∂x + ∂Uy/∂z|) to the training loss. λ_div = 0.01–0.1 (start at 0.01 to avoid destabilizing current convergence). Must apply mask to exclude padding nodes.

**Code-change scope:** Medium. Nearest-neighbor precomputation adds complexity but is feasible in train.py using a scipy/torch-based k-NN within each sample. The loss term is a small addition to the existing Huber loss.

**Predicted improvement direction:** Primary benefit on cruise val_geom_camber splits and val_re_rand; may degrade vol_p slightly if the penalty gradient conflicts with pressure prediction in early epochs.

**OOD targeted:** Yes — physics conservation is geometry-agnostic and should extrapolate.

**Suited student:** frieren (strong background in loss formulation, has worked with multiple loss variants across rounds 3–6; currently awaiting Arm 2 on PR #3801 so will be free shortly).

---

## Idea 5: lookahead-lion

**Hypothesis:** Wrapping Lion with a Lookahead outer loop (k=5 steps, α=0.5) will smooth Lion's sign-update trajectory — the discrete ±1 gradient signal is particularly susceptible to oscillation, and Lookahead's "slow weights" should act as an additional averaging mechanism orthogonal to momentum.

**Mechanism / why:** Lookahead (Zhang et al. 2019) maintains a slow-weight copy that interpolates toward the fast-weight position every k steps. For SGD/Adam, the benefit is a reduced variance of the final parameter trajectory. For Lion specifically, the sign update discards magnitude information entirely — each step is a discrete jump of exactly LR × 1 in each coordinate. This makes Lion's trajectory inherently more noisy per step than Adam's magnitude-weighted update. Lookahead's slow-weight average is a natural complement: it introduces the averaging that Lion's sign update removes. The combination has not been tested in this codebase. Implementation: a thin wrapper class around the existing Lion optimizer that stores slow-weight state and applies the interpolation every k steps. k=5, α=0.5 are the original paper defaults and have generalized well across tasks.

**Why it's orthogonal to EMA:** EMA operates on model parameters post-update for evaluation. Lookahead operates on optimizer slow weights during training, changing the effective gradient signal seen by the model. They target different points in the compute graph.

**Code-change scope:** Small. ~30-line Lookahead wrapper class in train.py; wrap the existing Lion instantiation; no changes to loss, data, or model.

**Predicted improvement direction:** Small-to-moderate improvement on all splits; primarily targets training stability rather than generalization directly. Expected benefit: 0.5–2 val points.

**OOD targeted:** No (indirect — better optimization may generalize better, but the mechanism is optimizer geometry, not domain structure).

**Suited student:** askeladd (currently on dropout sweep PR #3880; familiar with optimizer hyperparameters from β₁ sweep; this is a natural next assignment once PR #3880 closes).

---

## Idea 6: layerwise-lr

**Hypothesis:** Applying a layer-wise learning rate decay — LR × decay^(L-l) where L=5 layers and l is the layer index from output — will protect early-layer (geometric embedding) weights from overshooting while allowing later-layer (physics-reasoning) weights to update aggressively.

**Mechanism / why:** The Transolver's early layers encode mesh geometry (position, arc-length, shape descriptors) into latent tokens. These representations need to change slowly — they are shared across all physical conditions and camber values. The late layers apply physics-specific transformations. Under Lion with a single lr=2e-4 and clip=1.0 (100% engagement, every coordinate moves ±LR per step), all layers receive identical per-step displacement regardless of which layer is doing more useful work. Layer-wise LR decay (LLRD) was introduced for BERT fine-tuning (Sun et al. 2019) and has been validated in ViTs, FNOs, and scientific ML surrogates. Decay factor 0.8–0.9 per layer from output to input is the standard range.

**Implementation:** In train.py, split model parameters by layer index using named_parameters(), assign a per-layer LR multiplier, and pass param_groups to the Lion optimizer. For L=5, with decay=0.85: layer 5 (output) gets lr=2e-4, layer 4 gets lr=1.7e-4, ..., layer 1 gets lr=0.89e-4. The embedding layer gets the minimum.

**Code-change scope:** Small. ~20 lines of param-group construction; no model or data changes.

**Predicted improvement direction:** Small improvement on camber OOD splits (early-layer geometric encoding protected); modest improvement on val_single_in_dist.

**OOD targeted:** Indirectly — geometric encoding stability should help unseen camber values extrapolate more smoothly.

**Suited student:** nezuko (currently awaiting Arm 2 on H=144 PR #3745; LR architecture is conceptually close to model capacity work).

---

## Idea 7: re-conditioned-normalization

**Hypothesis:** Instead of a single global z-score normalization for all input features, normalize the flow-condition features (dims 13–23: log(Re), AoA, NACA params, gap, stagger) per-Re-decade bucket, while keeping the geometric features (dims 0–12) globally normalized. This should reduce distributional shift for val_re_rand.

**Mechanism / why:** The primary OOD axis for val_re_rand is Reynolds number stratification. log(Re) spans ~5 to ~6.7 in training (100K to 5M). Within each Re decade, the relationship between geometry features and flow fields is more stationary than across decades (boundary layer thickness scales as Re^{-0.5}, so the pressure magnitude distribution changes systematically with Re). A per-decade normalization of flow-condition inputs reduces the input distribution mismatch that the model sees at test time for held-out Re values. This is a data-representation intervention rather than an architectural one.

**Implementation note:** Compute stats.json-style means and stds per Re decade (define buckets: Re < 300K, 300K–1M, 1M–3M, >3M). In train.py, apply the appropriate bucket normalization to dims 13–23 based on each sample's Re value. Alternatively, use a simple continuous normalization: normalize log(Re) to [0,1] range and then apply a small learned affine scaling of dims 13–23 conditioned on the normalized log(Re) — but this makes it medium scope.

**Code-change scope:** Medium (if per-bucket, requires precomputing 4 sets of stats from the training set and adding lookup logic; if continuous affine, requires a small learned module). Start with the per-bucket version for simplicity.

**Predicted improvement direction:** Primary target: val_re_rand. Neutral or small benefit on camber splits. Risk: if the bucket boundaries create discontinuities at test time for held-out Re values near bucket edges, may hurt. A smooth normalization (continuous affine) mitigates this.

**OOD targeted:** Yes — directly targets val_re_rand.

**Suited student:** thorfinn (currently awaiting Arm 2 on wd sweep PR #3751; has experience with normalization-adjacent hyperparameter changes; free slot likely soon).

---

## Idea 8: fft-domain-loss

**Hypothesis:** Adding an auxiliary FFT-domain loss on the surface pressure prediction — penalizing the L1 error between FFT(p_pred|surface) and FFT(p_true|surface) at low spatial frequencies — will bias the model toward capturing large-scale pressure distributions, which are the dominant contributors to lift and drag.

**Mechanism / why:** The MAE on surface pressure aggregates errors uniformly across all surface nodes. In aerodynamics, low-frequency pressure variation (pressure coefficient distribution over the chord) dominates aerodynamic forces. High-frequency node-level noise is penalized equally to low-frequency shape error by the current Huber loss. An FFT loss, evaluated on the 1D ordered sequence of surface nodes (sorted by arc-length dim 2–3), gives higher weight to the dominant spatial modes. This technique has been used successfully in weather prediction (Pathak et al. FourCastNet), seismic modeling, and structural simulation to improve large-scale field predictions.

**Implementation:** In the loss computation, extract surface nodes using is_surface mask, sort them by signed arc-length (dim 2 of x), compute rfft on the pressure prediction and target along the surface sequence, take L1 error between magnitude spectra, and add with weight λ_fft=0.1. The 1D FFT assumption requires approximately ordered surface nodes, which the arc-length encoding provides.

**Code-change scope:** Medium. Requires sorting surface nodes per sample by arc-length in the training loop and computing the FFT loss. The arc-length features (dims 2–3) are already in x.

**Predicted improvement direction:** Primary benefit on val_geom_camber splits (large-scale camber-dependent pressure shape). May reduce test_avg/mae_surf_p more than val_avg because test distribution is broader.

**OOD targeted:** Yes — large-scale pressure distribution should be more robust to unseen camber geometry.

**Suited student:** fern (currently on vol_loss p-weight PR #3747; directly adjacent to loss formulation work; this is a natural extension once that PR resolves).

---

## Idea 9: curriculum-by-re

**Hypothesis:** Ordering training samples by Reynolds number in early epochs (low-Re first, progressively introducing high-Re samples) will give the model a warm-start on the simpler flow regime before confronting high-Re boundary layer complexity, potentially improving val_re_rand OOD performance.

**Mechanism / why:** Low-Re flows have smoother, more laminar pressure fields; high-Re flows develop thinner boundary layers and sharper gradients. Starting with the easier examples and gradually introducing harder ones (curriculum learning, Bengio et al. 2009) has been validated in PDE surrogate literature (e.g., physics-informed curriculum in PINN training). The val_re_rand split is stratified across Re — the model must generalize to held-out Re values within each decade. A curriculum warm-start teaches the model the fundamental geometry-to-pressure mapping before the high-Re perturbations add complexity. Implementation: sort samples by log(Re) in the DataLoader for the first N epochs, then switch to the standard balanced sampler. N=5 is a reasonable start (roughly the first cosine quarter on T_max=21).

**Code-change scope:** Small. Modify the DataLoader instantiation in train.py to use a sorted sampler for epochs 0..N-1, then revert to sample_weights balanced sampling. The existing load_data() return already provides sample_weights for the standard balanced mode.

**Predicted improvement direction:** Moderate improvement on val_re_rand; neutral or small benefit on camber splits; risk of slightly degrading val_single_in_dist if the model over-indexes on low-Re patterns in early epochs.

**OOD targeted:** Yes — directly targets val_re_rand.

**Suited student:** tanjiro (experienced with schedule/ordering interventions from cosine schedule work; this is an orthogonal ordering lever).

---

## Idea 10: gumbel-topk-slices

**Hypothesis:** Replacing PhysicsAttention's fixed uniform slice assignment with a differentiable Gumbel-top-k selection — allowing the model to learn which mesh regions get aggregated into which slice tokens — will improve the model's ability to focus on aerodynamically critical regions (leading edge, trailing edge, suction peak).

**Mechanism / why:** The current slice_num=64 assignment is fixed at initialization (likely based on spatial coordinates or feature clustering). A learnable, differentiable selection using the Gumbel-softmax reparameterization (Jang et al. 2017, Maddison et al. 2017) would allow gradient signal to reshape the slice boundaries during training. In aerodynamics, pressure gradients are highly non-uniform — the suction peak near the leading edge and the pressure recovery toward the trailing edge dominate the surface pressure distribution. A fixed uniform slice may dilute these critical regions by averaging them with lower-gradient neighbors. Gumbel-top-k slice selection is the PhysicsAttention analog of deformable convolution or attention learned to focus on keypoints.

**Implementation challenge:** This is a significant refactor of the PhysicsAttention module. The forward pass currently assigns each node to a slice token via a fixed operation; replacing this with a differentiable assignment matrix adds a learnable routing layer. The Gumbel-top-k implementation is non-trivial and requires careful masking for padded nodes. The temperature annealing schedule (τ: 1.0 → 0.1 over training) is a critical hyperparameter.

**Code-change scope:** Large. Modifies PhysicsAttention internals significantly. High-risk, high-reward.

**Predicted improvement direction:** If it works, could be a step-change improvement on all splits (better token formation). If routing learns degenerate assignments, may be much worse. Requires careful ablation.

**OOD targeted:** Indirectly — learned routing may generalize better to unseen geometries than fixed routing.

**Suited student:** Save for a student who has completed their current assignment and is ready for a complex architectural experiment. Not a first-pass assignment.

---

## Idea 11: surf-vol-loss-decoupled-clip

**Hypothesis:** Applying separate gradient-clipping norms to the surface-loss and volume-loss gradient paths (e.g., clip surface gradients at max_norm=0.5, volume at max_norm=2.0) will prevent the large-magnitude volume loss gradients from drowning out the surface pressure signal — a mechanism orthogonal to vol_p weight tuning.

**Mechanism / why:** The current implementation applies a single global gradient clip at max_norm=1.0 before the Lion optimizer step. Surface pressure errors are the primary metric but the loss is a sum of surface and volume Huber losses (with vol_p weighting). High-Re volume nodes have much larger absolute residuals (|y| up to 29K+), so their gradient contributions dominate the pre-clip gradient norm. After the global clip, the surface gradient may be scaled down more than the volume gradient. A two-path approach — compute surface and volume losses separately, clip each path's gradients independently, then combine — preserves the surface gradient's relative magnitude. This is mechanistically distinct from the vol_p weight sweep (which scales the loss value, not the gradient path).

**Implementation:** Requires calling backward() separately on surf_loss and vol_loss, clipping each separately, then stepping once. Or equivalently, after a single backward, identify surface and volume parameter subsets... actually this is harder than described for shared parameters. A cleaner approach: use a weighted sum with a surf_loss weight that compensates for the gradient norm imbalance, computed dynamically as the ratio of the current surf_loss gradient norm to the vol_loss gradient norm. This is the GradNorm approach (Chen et al. 2018).

**Code-change scope:** Small-to-medium. GradNorm variant is ~20 additional lines.

**OOD targeted:** No — targets training signal quality.

**Suited student:** alphonse (free after slice_num sweep; has worked with gradient mechanics; small assignment).

---

## Idea 12: eta-min-ratio-restore

**Hypothesis:** At lr=2e-4 the ratio eta_min/lr = 1e-5/2e-4 = 0.05, half the 0.1 ratio that was in place when eta_min=1e-5 was set on the lr=1e-4 stack. Testing eta_min=2e-5 ONLY in combination with lr=2e-4 (to restore the original ratio) may recover the "sweet-spot LR floor" that the T_max=21 fix was originally calibrated for.

**Mechanism / why:** The cosine schedule descends from lr to eta_min over T_max cycles. The productive low-LR zone that tanjiro's T_max=21 unlocked is the region where LR ≈ 1–2x eta_min (the final refinement zone). At lr=2e-4 with eta_min=1e-5, the final LR equals 1e-5 — the same absolute floor as before, but now this floor is 20x smaller than the starting LR (vs 10x before). The model may be spending more epochs at an overly small LR. Raising eta_min to 2e-5 restores the original ratio and may rebalance the schedule. CRITICAL NOTE: tanjiro PR #3713 ruled out eta_min raise in ISOLATION (eta_min={2e-5, 3e-5} on the old lr=1e-4 stack were worse). The hypothesis here is specifically that the RATIO matters and the raise is only valid paired with the higher lr=2e-4.

**Code-change scope:** Small — one hyperparameter change paired with the existing lr=2e-4 SOTA stack.

**Predicted improvement direction:** Uncertain. If the sweet-spot floor hypothesis is correct, small improvement. If the absolute floor matters more than the ratio, no improvement. Discriminating experiment between two competing explanations of the T_max/eta_min interaction.

**OOD targeted:** No.

**Suited student:** thorfinn (once wd sweep resolves; familiar with LR schedule mechanics from multiple rounds).

---

## Notes on ordering rationale

Ideas 1–3 are ranked highest because they have the clearest mechanistic grounding and the lowest risk of being confounded by other in-flight experiments:

- Idea 1 (slice_num=128 stability) has a known ceiling signal (cruise val=104.24 on old stack) and a targeted, falsifiable fix.
- Idea 2 (domain-id feature) is a small, clean intervention with strong analogy to task-conditioning literature.
- Idea 3 (SWA) is mechanistically distinct from in-flight EMA and has consistent external validation.

Ideas 4–9 are medium-priority because they require more implementation care and have less direct external validation in this exact setting. Ideas 10–12 are lower priority (large scope, diagnostic, or blocked by prior elimination evidence).

Do not assign Ideas 10 or 12 until Ideas 1–9 have been evaluated — 10 is high-risk architectural, and 12 requires careful setup to avoid repeating a ruled-out direction.
