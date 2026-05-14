# Wave 21 Research Ideas — Second Plateau-Protocol Escalation

**Date**: 2026-05-14 ~05:30 UTC  
**Context**: Two consecutive full washouts (Wave 19: 7/7 closed; Wave 20: 8/8 closed). Baseline fixed at 55.1595 val_avg/mae_surf_p (#2648). All simple hyperparameter axes, the attention-temperature family, standard loss modifications, and first-pass model-class/SSL attempts are closed. Second plateau-protocol escalation: bold, corrected-mechanism hypotheses only.

## Background: Wave 20 Failure Analysis and Corrected Directions

Wave 20 revealed *why* each family failed, not just that it did. These corrected mechanisms are the foundation of Wave 21:

| Wave-20 PR | Root cause of failure | Corrected direction |
|---|---|---|
| H86 relative L1 | Train/eval metric mismatch — normalized gradient, absolute eval | Focal-style hard-sample reweighting that preserves absolute units |
| H87 camber curriculum | Easy-first downweighted camber_rc (the hard target) | Hard-first anti-curriculum: permanently upweight camber_rc at all epochs |
| H88 camber-cond LN | Additive correction on frozen LN; budget too short to unlearn | Fresh Re-conditional γ/β from scratch; lightweight scalar multiplier only |
| H89 Sobolev surface loss | ds_min near-zero causes gradient explosion (up to 8367×) | Equispaced arc-length resampling before gradient computation |
| H90 GeoMPNN | KNN graph construction overhead per forward pass; only 3 epochs | Precomputed static graph + lightweight message-passing correction layer only |
| H91 masked SSL | Masking *input* geometry nodes — irrelevant for pressure prediction | Mask *output* pressure targets; reconstruct as auxiliary supervised task |
| H92 SE(2)-equivariant | 2D SE(2) loses AoA direction; vector capacity d_v=2 too small | AoA-aware: encode α as (cos α, sin α) into equivariant branch with d_v=8 |
| H93 Bernoulli | Physical-space gradients 100-1000× larger than normalized MAE | Normalized-space Bernoulli at λ=1e-5 to 1e-6 |

---

## H94: Hard-First Permanent Upsampling of Worst OOD Split

**Slug**: `hard-first-camber-upsampling`  
**Student**: alphonse  
**Family**: Curriculum B (corrected H87)

**Mechanism**: H87 (camber curriculum) used easy-first weighting during warmup, which *downweighted* camber_rc — the split that needs the most help. The correct intervention is the opposite: permanently set sample weights so camber_rc training samples are drawn at higher frequency throughout all 12 epochs. This is not a curriculum (no schedule); it is a fixed reweighting that forces the model to see more hard-OOD geometry.

**Implementation**:
- In the DataLoader sampler, assign per-sample weights: camber_rc samples get weight `w_hard`, all others get weight 1.0.
- Try `w_hard = 2.0` (arm A) and `w_hard = 3.0` (arm B).
- No warmup, no annealing — fixed weights for all 12 epochs.
- Everything else unchanged from baseline.

**Expected impact**: camber_rc val 68.657 → 62–65. If the gap is density-driven rather than extrapolation, upsampling alone should narrow it. H87's anti-curriculum arm B (val=58.40) showed the right direction but also used schedule-driven logic.

**Risk**: May hurt single_in_dist if over-focusing on camber_rc. w_hard=2 is conservative; w_hard=3 may over-correct. Two-arm sweep gives direct signal.

**Confidence**: Medium-high. The mechanism is clean and the correction to H87's direction is clearly supported by split-level data.

---

## H95: Output-Node Pressure Masking as SSL Auxiliary Loss

**Slug**: `pressure-node-mask-ssl`  
**Student**: edward  
**Family**: SSL B (corrected H91)

**Mechanism**: H91 masked *input* geometry coordinates — the model still had pressure targets for those nodes, so masking geometry gave the model a harder reconstruction problem unrelated to pressure prediction. The correct SSL signal is to mask *output pressure targets* for a random subset of nodes and ask the model to reconstruct them from neighboring pressure values, which directly trains the model to interpolate the pressure field.

**Implementation**:
- Keep the full geometry input unchanged.
- In each forward pass, randomly zero out 20% of target pressure values in the loss computation (not in the architecture — just don't include those nodes in the loss).
- Add a separate reconstruction head (1-layer MLP) that predicts the masked-out pressures from the latent representation.
- Auxiliary reconstruction loss weight λ_ssl = 0.1, added to the primary MAE loss.
- This forces the model to build a spatially coherent pressure field representation.
- All 12 training epochs, no phase split needed — simpler than H91's 2-phase approach.

**Expected impact**: If the model currently overfits to individual node regression without field coherence, masking-and-reconstruct forces better spatial interpolation. Targets re_rand and camber_rc where generalization requires coherent field understanding.

**Risk**: 20% masking may add noise to training signal. Reconstruction head adds ~5K params (well within budget). If val_avg/mae_surf_p rises, try λ_ssl = 0.05.

**Confidence**: Medium. The corrected mechanism is sound; the risk is that 12-epoch budget is tight for two objectives.

---

## H96: Re-Conditional Feature Scaling (Lightweight FiLM Correction)

**Slug**: `re-cond-feature-scale`  
**Student**: nezuko  
**Family**: Conditioning B (corrected H88)

**Mechanism**: H88 (camber-cond LN) tried to add an *additive correction* on top of existing LayerNorm — the model had to unlearn the LN bias to use the conditioning, which is expensive in 12 epochs. The correct intervention is a *multiplicative-only* scalar: after the final Transolver block output, apply a single global Re-conditional gain factor γ(log_Re) ∈ ℝ, identity-initialized (γ=1.0), trained from scratch. This requires no unlearning of existing weights.

**Rationale**: Re-range is the dominant axis of re_rand OOD generalization. A single learned scalar that scales the final representation by Re is the lowest-complexity intervention that could correct the Re-extrapolation gap.

**Implementation**:
- Add a single small MLP: log_Re → γ ∈ ℝ (init output = 1.0 via zero-weight last layer + bias=1).
- After the final transformer block output, multiply all features by γ(log_Re).
- This is ~100 extra parameters.
- Arm A: apply γ after the final block output, before the output projection.
- Arm B: apply γ after the output projection, before the loss.
- Compare to baseline on re_rand split specifically.

**Expected impact**: re_rand val 55.368 → 51–53. The H88 family showed conditioning can help in principle; this is the minimal viable version.

**Risk**: A single scalar may be insufficient. If γ does not converge (output collapses to ~1 everywhere), the mechanism is noise. Two arms give two choices of application point.

**Confidence**: Medium. Very low cost, mechanistically clean.

---

## H97: Equispaced Arc-Length Sobolev Loss

**Slug**: `arclength-sobolev-loss`  
**Student**: fern  
**Family**: Loss B2 (corrected H89)

**Mechanism**: H89 (Sobolev surface loss) collapsed because irregular mesh spacing created near-zero ds values that produced |dp/ds| gradients up to 8367. The fix is to resample the surface pressure profile onto a uniform arc-length grid before computing dp/ds, eliminating the ds_min instability.

**Implementation**:
- At training time, sort surface nodes by cumulative arc-length.
- Interpolate (linear) pressure predictions and targets onto a uniform grid of N=64 arc-length steps.
- Compute dp/ds on the uniform grid (ds = total_arc_length / 63 everywhere).
- L_sobolev = mean(|dp_pred/ds - dp_tgt/ds|) with weight λ = 0.05.
- Primary loss is unchanged absolute MAE.
- This eliminates the blow-up entirely.

**Expected impact**: If the H89 mechanism is correct (pressure-distribution-shape generalization helps OOD), the corrected implementation should show improvement on camber_rc. The raw direction was tested at λ=0.002 (50× below the original spec of λ=0.1) with val=61.78 — this is still a regression. With fixed implementation at λ=0.05, expect something close to or better than baseline.

**Risk**: The interpolation step adds per-batch overhead. N=64 is reasonable. If λ=0.05 is still too large, try λ=0.01.

**Confidence**: Medium. The root cause of H89's failure was implementation-level (instability), not mechanism-level. This is a clean retry.

---

## H98: Precomputed Static KNN Graph + Lightweight GNN Correction Layer

**Slug**: `static-knn-gnn-correction`  
**Student**: tanjiro  
**Family**: Model Class B (corrected H90)

**Mechanism**: H90 (GeoMPNN) was compute-starved because it rebuilt the KNN graph every forward pass — reducing the budget to ~3 effective epochs. The fix is to precompute and cache the static KNN graph (geometry doesn't change within a sample), then add only a *single* GNN message-passing correction layer *on top of* the existing Transolver output, rather than replacing the Transolver entirely.

**Implementation**:
- Keep the full Transolver stack unchanged.
- After the final Transolver block output, add 1 GNN layer:
  - Precompute KNN(k=8) graph for each sample at dataset init time (store as edge indices in the dataset).
  - In the forward pass: aggregate neighbor features via mean-pooling over KNN edges → project → add residual to node features.
  - GNN correction adds ~15K params.
- This adds O(N × k) operations per forward pass with no graph construction overhead at train time.
- The existing Transolver provides strong initial representations; the GNN correction layer adds local geometry awareness on top.

**Expected impact**: If local geometry awareness helps (mechanism from H90), this lightweight correction should provide the signal without the compute cost. The Transolver already provides global attention; GNN adds local edge-level communication.

**Risk**: A single GNN layer may be too shallow. But it is the minimal viable test of the mechanism. If it helps, add more layers in the next iteration.

**Confidence**: Medium. The corrected mechanism removes the practical failure mode of H90; the underlying hypothesis is untested.

---

## H99: Normalized-Space Bernoulli Consistency at λ=1e-5

**Slug**: `normalized-bernoulli-1e5`  
**Student**: frieren  
**Family**: Loss C2 (corrected H93)

**Mechanism**: H93 (Bernoulli physics loss) caused catastrophic gradient amplification because it computed the Bernoulli residual in *physical units* (velocities in m/s, pressures in Pa), producing gradients 100-1000× larger than the normalized-space MAE. The fix is to compute the Bernoulli residual entirely in *normalized space* (the same space the model operates in), and use a very small λ to keep the physics signal as a soft prior rather than a dominant loss term.

**Implementation**:
- All inputs/outputs are already in normalized space. Compute Bernoulli as:
  `R_Bernoulli = P_norm + 0.5 * (Vx_norm^2 + Vy_norm^2) - C_norm`
  where C_norm is the normalized total head (can be estimated as the mean of P + 0.5*V^2 per sample, making R a zero-mean consistency constraint).
- L_bernoulli = mean(R_Bernoulli^2) — penalizes departure from total-head conservation.
- λ = 1e-5 (arm A), λ = 1e-6 (arm B).
- The normalized-space computation ensures gradients are O(1), same order as the MAE loss.

**Expected impact**: A very soft physics prior at λ=1e-5 to 1e-6 may improve re_rand generalization by constraining the model's predictions to be physically consistent across unseen Re values. The mechanism is sound at the right scale.

**Risk**: At λ=1e-5, the signal may be too weak to have any effect. This is acceptable — null result is clean and closes the normalized-physics axis.

**Confidence**: Medium-low. The corrected mechanism is sound but the signal strength is uncertain.

---

## H100: AoA-Decomposed Attention with (cos α, sin α) Direction Encoding

**Slug**: `aoa-decomposed-attention`  
**Student**: thorfinn  
**Family**: Equivariant B (corrected H92)

**Mechanism**: H92 (SE(2)-equivariant attention) failed because the strict 2D SE(2) constraint eliminates AoA direction information — but AoA is one of the most physically important features (determines leading-edge stagnation point, suction peak location). The corrected approach drops the rigid equivariance constraint and instead *explicitly encodes directional awareness*: append (cos α, sin α) to the input features as a 2D direction vector, then add a lightweight 2-head directional cross-attention that can attend to geometry features conditioned on the flow direction.

**Implementation**:
- Add (cos α, sin α) as two additional input channels to every node's feature vector (appended to existing Fourier coords).
- Add a 2-head directional cross-attention layer: Q from node features, K/V from the (cos α, sin α) direction expanded to a learned embedding.
- This is a 2-head attention with d_q=d_k=16, ~8K extra params.
- Apply after the 3rd Transolver block (mid-network), as a residual connection.
- This preserves the full Transolver representation while adding direction-conditioned feature modulation.

**Expected impact**: AoA directly determines which side of the airfoil has suction (upper vs lower), so directional encoding should help both in-dist and OOD splits. Expected improvement on camber_rc (where the model may be confusing angle-of-attack effects across different camber geometries).

**Risk**: The Fourier coordinate encoding already contains implicit AoA information (node positions are in the AoA-rotated frame). Adding explicit direction encoding may be redundant. But the explicit signal is worth testing.

**Confidence**: Medium. The correction to H92 is mechanistically sound; the question is whether the signal is already implicit.

---

## H101: Channel-Adaptive Surface Weight Scheduling

**Slug**: `adaptive-surf-weight-schedule`  
**Student**: askeladd  
**Family**: Loss D (new — not previously tested)

**Mechanism**: The current best surface-channel weight [0.5, 0.5, 2.0] (giving pressure 4× the velocity channels) was found as a fixed hyperparameter. But during training, the pressure channel is learned faster (it's smoother and physically simpler) than the velocity channels. By *scheduling* the surface weight — starting with equal weights [1.0, 1.0, 1.0] in early epochs to ensure velocity channels are properly initialized, then ramping up to [0.5, 0.5, 2.0] by epoch 4 — we may get better multi-channel calibration before the pressure channel dominates.

**This is a new hypothesis**: no previous experiment has tested *scheduled* channel weights (all prior experiments used fixed weights throughout training).

**Implementation**:
- Epoch 1-3: surf_ch_weight = [1.0, 1.0, 1.0] (equal weighting, let velocity channels learn)
- Epoch 4-12: surf_ch_weight = [0.5, 0.5, 2.0] (return to known-best)
- The ramp is a step function (no interpolation needed).
- Arm A: step at epoch 4.
- Arm B: linear interpolation from [1.0,1.0,1.0] to [0.5,0.5,2.0] over epochs 1-4.
- Everything else unchanged.

**Rationale**: The attn-temp annealing mechanism (#2648 merged win) showed that *scheduling* a fixed hyperparameter over epochs can improve over the best fixed value. The channel weight axis was closed as a *fixed* hyperparameter but never tested as a schedule.

**Expected impact**: Modest (1-3% improvement). The mechanism is low-variance and the direction is well-supported by the #2648 precedent that scheduling fixed hyperparameters helps.

**Risk**: Very low. Worst case: val_avg ≈ 55.1595 (neutral). This is a safe test.

**Confidence**: Medium-high. The mechanism is a direct analog of the #2648 precedent applied to a different fixed hyperparameter axis.

---

## Summary Table

| # | Slug | Family | Student | Key mechanism | Primary target |
|---|---|---|---|---|---|
| H94 | hard-first-camber-upsampling | Curriculum B | alphonse | Fixed 2×/3× upsampling of camber_rc | camber_rc ↓ |
| H95 | pressure-node-mask-ssl | SSL B | edward | Mask output pressure targets, reconstruct | re_rand + camber_rc ↓ |
| H96 | re-cond-feature-scale | Conditioning B | nezuko | Scalar Re-conditional gain, identity init | re_rand ↓ |
| H97 | arclength-sobolev-loss | Loss B2 | fern | Uniform arc-length resampling for dp/ds | camber_rc ↓ |
| H98 | static-knn-gnn-correction | Model Class B | tanjiro | Precomputed KNN + 1-layer GNN correction | global geometry ↓ |
| H99 | normalized-bernoulli-1e5 | Loss C2 | frieren | Bernoulli in normalized space at λ=1e-5 | re_rand ↓ |
| H100 | aoa-decomposed-attention | Equivariant B | thorfinn | (cos α, sin α) + 2-head directional cross-attn | camber_rc ↓ |
| H101 | adaptive-surf-weight-schedule | Loss D | askeladd | Schedule surf_ch_weight [1,1,1]→[0.5,0.5,2] | all splits ↓ |

## Ruling-Out Rationale

All hypotheses that repeat previously closed axes are excluded:
- Fixed surf_ch_weight variants (closed as a fixed hyperparameter — H101 tests it as a *schedule*, which is new)
- Any attention-temperature variants (fully closed)
- Any geometry spectral encodings (SDF, Laplacian PE, HF spectral)
- FiLM conditioning identical to #2453 (H96 uses a single scalar multiplier, not full FiLM)
- Slice-token output-side aux losses
- Full GeoMPNN model replacement (H98 is a correction layer only, not model replacement)
- 2-phase SSL with geometry masking (H95 masks pressure targets, not input geometry)
