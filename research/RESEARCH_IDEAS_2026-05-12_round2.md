<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — Round 2+ (2026-05-12)

Tag: `willow-pai2g-48h-r2`
Branch: `icml-appendix-willow-pai2g-48h-r2`

These ideas are orthogonal to Wave 1 (schedule, slice_num, surf_weight, mlp_ratio, SmoothL1, n_hidden, unified_pos, batch_size+lr scaling).

---

## H1: Stochastic Weight Averaging (SWA) for OOD Generalization

**One-liner:** Average model weights over the final training epochs to find a wider, flatter optimum that generalizes better to unseen camber and Re values.

**Rationale for this dataset:**
The primary OOD axes — unseen front-foil camber (M=6–8 raceCar, M=2–4 cruise) and stratified Re holdout — require the model to extrapolate geometry and flow regime. SWA is well-known to find wider minima in the loss landscape, which correlates with better OOD generalization. The foundational paper (Izmailov et al., 2018) showed consistent improvements on domain-shift benchmarks. ICML 2024 provides new generalization bounds connecting SWA's width to transfer error. For a dataset this small (~1.5K samples) with three distinct physical domains, the risk of sharp minima is especially high.

**Concrete change spec:**
In `train.py`, after the optimizer is defined:
```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

swa_model = AveragedModel(model)
swa_start = int(0.75 * MAX_EPOCHS)  # start averaging at 75% through training
swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=5)
```
In the training loop, after each epoch:
```python
if epoch >= swa_start:
    swa_model.update_parameters(model)
    swa_scheduler.step()
```
After training, update BatchNorm stats (not applicable since Transolver uses LayerNorm, skip this step), then evaluate `swa_model` as the final checkpoint. The best-checkpoint selection should still be done on the base `model` during training (for early stopping), but the terminal test evaluation should run with `swa_model`.

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -7% (wider minima help OOD; effect is larger for geometry splits than Re split).

**Risk/failure modes:**
- SWA with CosineAnnealingLR requires careful scheduling: use `SWALR` after `swa_start` rather than continuing cosine decay, otherwise the averaging is over a non-stationary distribution.
- If `swa_start` is too late (last few epochs), not enough checkpoints are averaged.
- Recommended: `swa_start = int(0.75 * MAX_EPOCHS)`, `swa_lr = cfg.lr * 0.2`.

**Time cost (fits 30 min?):** Yes. Overhead is ~5% (parameter averaging is O(1) per epoch). No extra forward passes during training.

---

## H2: Surface-Aware Slice Routing — Learnable Surface/Volume Token Separation

**One-liner:** Give the Physics-Attention slice assignment a learned bias toward separating surface nodes from volume nodes, so surface pressure prediction gets dedicated token capacity.

**Rationale for this dataset:**
The primary metric is surface pressure MAE. The current `PhysicsAttention` does soft-clustering of ALL nodes into `slice_num` tokens without any inductive bias toward the surface/volume distinction. This means surface nodes (which carry the aerodynamic signal) compete equally with interior nodes for token capacity. If we bias the slice assignments to preferentially cluster surface nodes into dedicated slices, the model can learn richer surface representations.

**Concrete change spec:**
In `PhysicsAttention.forward()`, after computing `slice_weights = softmax(in_project_slice(x_mid) / temperature)`, add a surface-bias mask before softmax:
```python
# is_surface_expanded: [B, N, 1] -> [B, H, N, slice_num]
# Reserve first K slices for surface, last (slice_num - K) for volume
K = slice_num // 4  # 25% of slices reserved for surface
surface_bias = torch.zeros_like(logits)
if is_surface is not None:
    surf_flag = is_surface.float().unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
    # positive bias on first K slices for surface nodes
    surface_bias[:, :, :, :K] += surf_flag * 5.0
    # positive bias on last (slice_num-K) slices for volume nodes
    surface_bias[:, :, :, K:] += (1 - surf_flag) * 5.0
slice_weights = F.softmax((logits + surface_bias) / temperature, dim=-1)
```
The `is_surface` tensor must be threaded through `Transolver.forward()` and each `TransolverBlock.forward()` to `PhysicsAttention.forward()`. The bias magnitude (5.0) and K fraction (0.25) should be treated as hyperparameters; 5.0 provides a strong-but-soft separation.

**Predicted delta on `val_avg/mae_surf_p`:** -5% to -12% (direct architectural alignment with the loss objective; surface pressure gets dedicated capacity).

**Risk/failure modes:**
- Threading `is_surface` through the model requires modifying model forward signatures — non-trivial refactor.
- Bias magnitude of 5.0 may be too rigid; consider making it a learnable scalar.
- If slice_num is small, reserving K=16 slices for surface may fragment the volume representation.
- Recommended ablation: first test with K=slice_num//4 and bias=3.0; if unstable, reduce bias.

**Time cost (fits 30 min?):** Yes, no extra compute. Training time unchanged.

---

## H3: Stochastic Weight Averaging with Gaussian Perturbation (SWAG) for Uncertainty

**One-liner:** Maintain a running mean and diagonal variance of model weights to form a Gaussian posterior; sample from it at inference for ensemble-like predictions without training multiple models.

**Rationale for this dataset:**
SWAG (Maddox et al., NeurIPS 2019) captures weight-space uncertainty along the SGD trajectory and provides Bayesian model averaging essentially for free at training time. For CFD surrogates, prediction variance on OOD samples (unseen camber, extreme Re) is a useful reliability signal, and the averaged prediction often beats the MAP estimate. SWAG is particularly effective when the training set is small (~1.5K here), because weight uncertainty is large and averaging over the posterior reduces variance.

**Concrete change spec:**
Use the `swag-pytorch` library or implement the diagonal SWAG update manually:
```python
# After burn-in (e.g., epoch > swa_start):
# Collect mean and variance of each parameter
swag_mean = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
swag_sq_mean = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
n_swag = 0

# After each epoch:
n_swag += 1
for k, v in model.named_parameters():
    swag_mean[k] = (swag_mean[k] * (n_swag - 1) + v.data) / n_swag
    swag_sq_mean[k] = (swag_sq_mean[k] * (n_swag - 1) + v.data ** 2) / n_swag

# At inference (K samples):
preds_list = []
for _ in range(K):
    for k, param in model.named_parameters():
        var = (swag_sq_mean[k] - swag_mean[k] ** 2).clamp(min=1e-30)
        param.data = swag_mean[k] + torch.randn_like(param.data) * var.sqrt()
    preds_list.append(model(data)["preds"])
pred_mean = torch.stack(preds_list).mean(0)
```
For a 30-min budget, use K=5 inference samples. The primary evaluation should use `pred_mean`; the MAP `swag_mean` is also worth logging.

**Predicted delta on `val_avg/mae_surf_p`:** -2% to -5% (Bayesian averaging reduces variance; effect is moderate but reliable).

**Risk/failure modes:**
- SWAG's diagonal approximation ignores weight correlations; full-rank SWAG would be better but costs 2× memory.
- K=5 inference samples adds 5× inference time; for the small val/test sets this is acceptable.
- If the learning rate is very low during the SWA period, the weight distribution is tight and SWAG degenerates to SWA.

**Time cost (fits 30 min?):** Yes. Inference cost is 5× for val, but val sets are 4×100 samples = negligible wall time.

---

## H4: Per-Sample Re-Based Input Normalization (Instance Normalization over Flow Conditions)

**One-liner:** Normalize the node velocity targets per-sample by the freestream velocity scale (derived from Re and geometry), so the model learns to predict dimensionless pressure coefficients rather than dimensional quantities.

**Rationale for this dataset:**
The program.md notes that "per-sample y std varies by an order of magnitude even inside one domain" — specifically, `val_single_in_dist` shows max per-sample y std of 2,077 while mean is 458. This implies the model's MSE loss is dominated by high-Re samples during training, leaving low-Re samples underfit. Global y normalization (with a single y_mean/y_std) partially addresses this, but the residual variance in normalized space is still large. A natural solution is to normalize targets per-sample by the expected dynamic pressure scale q = 0.5 * rho * U_inf^2, where U_inf ~ Re * nu / chord. Since log(Re) is feature dim 13 and chord is encoded in the geometry, this can be approximated as:

```python
# Approximate freestream velocity from log(Re) feature (dim 13)
# Re = rho * U_inf * L / mu => U_inf = Re * nu / L
# For normalization, use Re^1 (or exp(log(Re))) as a per-sample scale
re_scale = x[:, :, 13].exp()  # [B, N] — approximate, log(Re) unnormalized
# Use mean over real nodes as per-sample scale
re_scale_mean = (re_scale * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # [B]
# Scale targets: y_scaled = y / (re_scale_mean[:, None, None] + eps)
```

In practice, implement this as a per-sample `y_scale` factor stored alongside the batch: divide `y_norm` by `re_scale_mean` before loss computation, and multiply predictions by `re_scale_mean` before MAE computation. The `data/scoring.py` contract is preserved because you still denormalize with `y_std * pred + y_mean`; the per-sample rescaling happens inside the training loss only.

**Predicted delta on `val_avg/mae_surf_p`:** -4% to -9% (directly addresses the high-variance low-Re vs. high-Re imbalance; most impactful for `val_re_rand`).

**Risk/failure modes:**
- Re scale is a rough proxy for dynamic pressure; exact freestream velocity requires knowing the chord length, which varies across samples.
- If the scale factor is too coarse, it may over-correct for some samples.
- Interaction with global normalization needs careful handling: the per-sample scale is applied on top of global normalization, not instead of it.
- Recommended: use `log(Re)` directly as a per-sample loss weight (not a target rescaler): `loss = (sq_err * re_weight[:, None, None]) * mask.unsqueeze(-1)` where `re_weight = 1 / log(Re)_normalized`.

**Time cost (fits 30 min?):** Yes. No model changes; pure loss modification.

---

## H5: Geometry-Conditioned Global Conditioning via FiLM Layers

**One-liner:** Extract global aerodynamic context (Re, AoA, NACA params, gap, stagger) and inject it into each Transolver block via Feature-wise Linear Modulation (scale+shift), giving the model an explicit global flow-condition prior at every layer.

**Rationale for this dataset:**
Currently, global flow conditions (Re, AoA, NACA, gap, stagger — dims 13–23) are concatenated into per-node features and processed identically to local geometry features (position, arc-length, dsdf). But these are fundamentally different: global conditions describe the entire flow field, while local geometry describes a specific node's neighborhood. FiLM (Perez et al., 2018) is a lightweight technique from conditional image generation that applies a learned affine transform (gamma * h + beta) to intermediate activations, where gamma and beta are predicted from a conditioning vector. This gives the model an explicit mechanism to modulate its intermediate representations based on global flow conditions.

**Concrete change spec:**
Add a `ConditionNet` that takes the global condition vector (mean of dims 13–23 over all real nodes, since they are constant per sample) and outputs scale/shift for each block:
```python
class FiLMConditioner(nn.Module):
    def __init__(self, cond_dim=11, hidden_dim=64, n_layers=5, n_hidden=128):
        # cond_dim = 11 (dims 13-23)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_layers * n_hidden),  # scale+shift per layer
        )
    def forward(self, x, mask):
        # x: [B, N, 24]; extract global condition as mean over real nodes
        cond = (x[:, :, 13:24] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.net(cond).reshape(B, n_layers, 2, n_hidden)  # [B, L, 2, H]
```
In each `TransolverBlock.forward()`, apply FiLM before the MLP:
```python
gamma, beta = film_params[:, layer_idx, 0, :], film_params[:, layer_idx, 1, :]
fx = gamma[:, None, :] * fx + beta[:, None, :]  # [B, N, H]
```

**Predicted delta on `val_avg/mae_surf_p`:** -4% to -10% (direct injection of global flow conditions into every layer; especially useful for OOD Re generalization and tandem geometry conditionality).

**Risk/failure modes:**
- FiLM scale/shift can destabilize training if initialized poorly; initialize gamma=1, beta=0 (identity mapping).
- If the condition signal is too dominant, the model may ignore local geometry. Use a residual form: `fx = (1 + gamma) * fx + beta` (a.k.a. conditional LayerNorm).
- Requires threading the FiLM parameters through the forward pass.

**Time cost (fits 30 min?):** Yes. The FiLMConditioner is tiny (~50K params). No significant compute overhead.

---

## H6: Log-Scale Pressure Targets for High-Dynamic-Range Pressure Fields

**One-liner:** Transform the pressure target p to sign(p) * log(1 + |p|) (softplus-log) before normalization, then inverse-transform predictions at evaluation time, to reduce the dynamic range that the model must predict.

**Rationale for this dataset:**
The pressure field spans a very large dynamic range: `val_single_in_dist` shows y range (-29,136, +2,692) with per-sample std varying from ~458 mean to ~2,077 max. Log-space targets are standard in other high-dynamic-range prediction tasks (audio, seismology, financial time series). In CFD, surface pressure coefficients Cp vary orders of magnitude between stagnation point (Cp ~ 1) and separation zones (Cp << -1). The signed softplus-log transform `sign(p) * log(1 + |p|)` compresses large magnitudes while preserving the sign and small-value behavior. Since MAE is computed in physical space, this is purely a training-space transformation.

**Concrete change spec:**
In `train.py`, before the loss computation, apply the transform to `y_norm` and `pred`:
```python
def signed_log1p(t):
    return torch.sign(t) * torch.log1p(t.abs())

def signed_expm1(t):
    return torch.sign(t) * torch.expm1(t.abs())

# In training loop, after normalization:
y_norm_log = y_norm.clone()
y_norm_log[..., 2] = signed_log1p(y_norm[..., 2])  # only pressure channel (dim 2)

# Loss uses log-space targets for pressure, linear for Ux, Uy
pred_log = pred.clone()
pred_log[..., 2] = signed_log1p(pred[..., 2])
loss_p = F.mse_loss(pred_log[..., 2][mask], y_norm_log[..., 2][mask])

# For MAE evaluation, inverse-transform:
pred_p_linear = signed_expm1(pred[..., 2])  # already in normalized space
# Then denormalize as usual: pred_p_physical = pred_p_linear * y_std[2] + y_mean[2]
```

Note: this requires careful integration with `data/scoring.py` — the scoring function denormalizes directly. One clean approach is to apply the transform to `y_norm` for the loss only, not to the model output (the model still predicts in linear normalized space), and instead transform `y_norm` to log-space for loss weighting.

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -8% (reduces loss dominance by extreme high-Re pressure values; helps low-Re regime).

**Risk/failure modes:**
- If the model output is in log-space but `scoring.py` expects linear-space, there will be a MAE inflation bug.
- Safer implementation: use the log transform only as a per-sample loss weight, not a target transform: `p_weight = 1 / (1 + |y_norm[..., 2]|)`.
- The transform breaks the existing `y_mean/y_std` normalization contract; test that MAE is unchanged when using the identity transform before deploying the actual transform.

**Time cost (fits 30 min?):** Yes. Pure loss modification, no model changes.

---

## H7: Structured Dropout (DropDim) on Slice Token Dimensions

**One-liner:** Randomly zero out entire dimensions of the slice tokens during training (DropDim), forcing the model to learn redundant representations and improving generalization through a cheap regularization aligned with the attention structure.

**Rationale for this dataset:**
DropDim (Liang et al., 2021) applies dropout not to individual activations but to entire feature dimensions of the intermediate representations, which acts as a structured regularizer that encourages distributed representations. In the Physics-Attention mechanism, slice tokens are the intermediate representation: dropping entire dims of `slice_token` forces each remaining dimension to carry information independently. This is complementary to standard dropout (which drops individual values) and more principled for attention-based architectures. The 2024 IEEE TPAMI paper on implicit regularization via dropout suggests that structured dropout variants often outperform standard dropout in low-data regimes — which applies here (~1.5K training samples).

**Concrete change spec:**
In `PhysicsAttention.forward()`, after computing `slice_token = einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / norm`, add:
```python
if self.training:
    # DropDim: zero out entire dimensions of slice tokens
    dim_mask = (torch.rand(slice_token.shape[-1], device=slice_token.device) > self.drop_dim_rate).float()
    slice_token = slice_token * dim_mask[None, None, None, :]
    # Re-normalize to compensate for dropped dims
    slice_token = slice_token / (1 - self.drop_dim_rate + 1e-8)
```
Add `drop_dim_rate: float = 0.1` to the model config. Values of 0.05–0.15 are typical in DropDim papers. Also consider applying DropDim to the `out_slice` (after attention) rather than `slice_token` (before).

**Predicted delta on `val_avg/mae_surf_p`:** -2% to -5% (regularization effect; most beneficial for OOD splits; modest but reliable).

**Risk/failure modes:**
- Drop rate must be tuned: too high (>0.2) hurts capacity; too low (<0.03) has no effect.
- The re-normalization compensates in expectation but may cause training instability if drop_dim_rate is large.
- Recommended: start with drop_dim_rate=0.1; if val loss is worse after 5 epochs, reduce to 0.05.

**Time cost (fits 30 min?):** Yes. Negligible compute overhead.

---

## H8: Multi-Fidelity Pre-Training on Synthetic XFOIL Data

**One-liner:** Pre-train the model on cheaply generated XFOIL pressure distributions (single foil, inviscid, fast), then fine-tune on the full CFD dataset, giving the model an aerodynamics prior before it sees expensive training samples.

**Rationale for this dataset:**
The training set is only ~1.5K samples. Multi-fidelity training — using cheap low-fidelity data to warm-start the model — is standard in engineering ML (e.g., Kennedy & O'Hagan 2000; modern applications in aerodynamics by Forrester & Keane 2009). XFOIL solves the 2D panel method + viscous boundary layer, giving reasonable Cp distributions on single-foil cases in milliseconds. Pre-training on XFOIL data teaches the model the basic physics of pressure distribution around an airfoil (stagnation point, suction peak, trailing edge recovery) before fine-tuning on the expensive CFD data. For the OOD camber splits, this pre-training would expose the model to the camber range M=6–8 even if the CFD training data doesn't include it.

**Concrete change spec:**
This hypothesis requires generating XFOIL data (Python `pyxfoil` or subprocess calls to the XFOIL binary) and wrapping it in a dataset compatible with the existing interface. The key steps:
1. Generate XFOIL pressure distributions for a grid of NACA 4-digit profiles (M=0–9, T=06–24, P=4), AoA (-10° to +6°), Re (100K–5M) — this produces ~10K synthetic samples cheaply.
2. Map XFOIL Cp to the physical pressure p using `p = q * Cp` where q = 0.5 * rho * U^2.
3. Build a thin dataset wrapper that reads these as `.pt` files with the same `{x, y, is_surface}` schema.
4. Pre-train for N_pretrain epochs (e.g., 5) on XFOIL data, then fine-tune on CFD data with a lower learning rate (lr * 0.1).

**Predicted delta on `val_avg/mae_surf_p`:** -5% to -15% (especially for OOD camber; gives the model a prior over the full NACA parameter space).

**Risk/failure modes:**
- XFOIL data is inviscid/viscous-corrected, not full CFD; the distribution shift may cause negative transfer if the model overfits XFOIL artifacts.
- Generating XFOIL data requires the binary on the worker node; may not be available. Alternative: use analytical thin-airfoil theory (closed-form Cp for flat-plate and parabolic camberlines).
- This is the most complex hypothesis here; budget 2–3x the usual implementation time.
- Recommended staged test: first check if XFOIL binary is available on the node with `which xfoil`.

**Time cost (fits 30 min?):** Tight. The XFOIL data generation needs to be done offline and the `.pt` files placed on the PVC before training. The training itself (pre-train + fine-tune) would need careful epoch budgeting within the 30-min cap.

---

## H9: Auxiliary Boundary-Condition Loss (Zero Pressure Gradient at Outlet)

**One-liner:** Add an auxiliary loss term penalizing non-zero pressure at the outlet nodes (where Cp → 0 at far-field), encoding the boundary condition physically.

**Rationale for this dataset:**
In CFD, the outlet boundary condition imposes zero gauge pressure at the far-field boundary. The current loss is purely data-driven with no physics constraints. Adding a soft auxiliary loss `L_bc = mean(p_pred^2 over outlet nodes)` guides the model to respect this BC without needing labeled data (it is a physics constraint, not a label). B-GNNs (Ogoke et al., 2022) demonstrated 7× generalization improvement by encoding such physical constraints. In the TandemFoilSet mesh, outlet nodes can be identified as nodes in Zone 0 (background zone) that are not surface nodes and are at the far-field boundary — approximately, nodes with |x| > some threshold or nodes labeled by high dsdf values.

**Concrete change spec:**
Since the exact outlet node labeling is not available directly in the preprocessed data (only `is_surface` is provided), use an approximate proxy: nodes in Zone 0 (non-surface, low dsdf) with large distance from the foil. The dsdf features (dims 4–11) provide distance-based information. Define outlet nodes as:
```python
# Approximate outlet nodes: non-surface, large distance from foil
# Use dsdf (dims 4-11) as proxy for distance; large dsdf ≈ far from foil
dsdf_mean = x_norm[:, :, 4:12].mean(-1)  # [B, N]
is_outlet = (~is_surface) & (dsdf_mean > dsdf_percentile_90)  # approximate
outlet_mask = is_outlet & mask

# Auxiliary BC loss: pressure should be near 0 at outlet
# (in normalized space: near -y_mean[2] / y_std[2])
p_ref_norm = (-stats["y_mean"][2] / stats["y_std"][2])  # normalized zero pressure
bc_loss = ((pred[..., 2] - p_ref_norm) ** 2 * outlet_mask.float()).sum() / outlet_mask.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * surf_loss + cfg.bc_weight * bc_loss
```
Set `bc_weight = 0.1` initially; tune from 0.05 to 0.5.

**Predicted delta on `val_avg/mae_surf_p`:** -2% to -6% (physics constraint reduces field-level errors; indirect but consistent benefit for surface pressure via continuity).

**Risk/failure modes:**
- The outlet node identification is approximate; if the proxy is wrong, the BC loss targets the wrong nodes.
- If `bc_weight` is too large, the model may distort its interior predictions to satisfy the BC, worsening MAE.
- Recommended: log `bc_loss` separately to verify it is decreasing before relying on this as a signal.

**Time cost (fits 30 min?):** Yes. Pure loss modification, no model changes.

---

## H10: Learned Positional Encoding via Fourier Features of the Mesh Coordinates

**One-liner:** Replace or augment the raw (x,z) coordinates in the input with Random Fourier Features (RFF) encoding, giving the model a richer multi-scale positional representation of the mesh.

**Rationale for this dataset:**
The current positional encoding uses raw (x,z) coordinates (dims 0–1) and signed arc-length (dims 2–3). These linear encodings cannot represent the multi-scale spatial structure of the mesh efficiently. Random Fourier Features (Rahimi & Recht, 2007) and learnable Fourier features (as in NeRF, Mildenhall et al., 2020) embed coordinates at multiple frequencies, allowing the model to distinguish nearby nodes and capture local mesh topology. For neural operators over irregular meshes, Fourier feature embeddings have been shown to improve convergence rate and generalization (Li et al., FNO series, 2020). The `unified_pos` encoder in the baseline already uses a ref×ref×ref voxel grid, but that is a coarse global encoding; per-node Fourier features provide a complementary local encoding.

**Concrete change spec:**
Add a Fourier feature encoder for the positional dims (0–3 of x: (x,z) and saf):
```python
class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim=4, num_frequencies=16, learnable=False):
        super().__init__()
        if learnable:
            self.B = nn.Parameter(torch.randn(in_dim, num_frequencies) * 10.0)
        else:
            self.register_buffer("B", torch.randn(in_dim, num_frequencies) * 10.0)
        self.out_dim = 2 * num_frequencies  # sin + cos

    def forward(self, x_pos):
        # x_pos: [B, N, in_dim=4]
        proj = x_pos @ self.B  # [B, N, num_frequencies]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B, N, 2*F]
```
In `Transolver.forward()`:
```python
pos_encoded = self.fourier_encoder(x[:, :, :4])  # [B, N, 32]
x_augmented = torch.cat([x, pos_encoded], dim=-1)  # [B, N, 24+32=56]
# Then preprocess takes fun_dim = (24 + 32) - 2 = 54
```
This increases `fun_dim` from 22 to 54; update `model_config["fun_dim"] = 54`. Use `num_frequencies=16` (adds 32 dims) as a starting point; `num_frequencies=32` is worth trying if VRAM allows.

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -7% (better spatial representation; helps with the irregular mesh structure and surface curvature).

**Risk/failure modes:**
- Random Fourier feature scale (10.0) must be tuned to the coordinate range; if coordinates are normalized to [-1,1], scale=5.0 may be more appropriate.
- Adding 32 dims to the input increases preprocess MLP cost modestly; memory impact is negligible.
- Learnable Fourier features may overfit on small datasets; start with fixed random features.

**Time cost (fits 30 min?):** Yes. The encoder is tiny and forward pass cost is negligible.

---

## H11: Domain-Adversarial Training to Force Domain-Invariant Representations

**One-liner:** Add a domain discriminator head that predicts which of the 3 domains (raceCar-single, raceCar-tandem, cruise) a sample comes from, and train the backbone with a gradient reversal layer to make its representations domain-invariant.

**Rationale for this dataset:**
The three domains have very different mesh sizes (~85K vs. ~127K vs. ~210K nodes), flow regimes (raceCar vs. cruise), and physical setups (single vs. tandem). If the model learns domain-specific representations, it will fail to generalize to OOD camber values that span domain boundaries. Domain-Adversarial Neural Networks (Ganin et al., JMLR 2016) use a Gradient Reversal Layer (GRL) to force the feature extractor to learn representations that are uninformative about domain identity. This is especially relevant here because the OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) test generalization to geometry configurations not seen in training but within a specific domain.

**Concrete change spec:**
Add a `GradientReversalLayer` and a `DomainClassifier`:
```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim=128, n_domains=3):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=0.1)
        self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, n_domains))

    def forward(self, fx, mask):
        # Global average pool over real nodes to get domain representation
        pooled = (fx * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.net(self.grl(pooled))
```
The domain label can be inferred from the input features: `gap == 0 & stagger == 0` → raceCar-single (domain 0); `gap != 0 & |AoA_mean| > threshold` → raceCar-tandem (domain 1); otherwise cruise (domain 2).

In the training loop:
```python
domain_logits = domain_classifier(fx_penultimate, mask)  # fx from last block
domain_loss = F.cross_entropy(domain_logits, domain_labels)
total_loss = loss + 0.05 * domain_loss
```

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -8% (domain invariance helps both OOD camber and Re splits; strongest effect on `val_geom_camber_cruise` which crosses the most distinct domain boundary).

**Risk/failure modes:**
- The GRL lambda must be tuned carefully: too large destabilizes training; too small has no effect.
- Start with lambda=0.05 and increase linearly during training (schedule: `lambda = min(0.5, epoch/epochs * 0.5)`).
- Domain labels must be computed correctly; verify with a debugging print before training.

**Time cost (fits 30 min?):** Yes. The domain classifier is tiny. Minor backward-pass overhead from GRL.

---

## H12: Pressure-Only Output Head with Shared Velocity-Pressure Backbone

**One-liner:** Split the output head into two branches — one predicting (Ux, Uy) at all nodes, and a dedicated surface-pressure head that operates only on surface nodes with additional capacity — so the optimization of surface pressure can proceed independently of volume velocity.

**Rationale for this dataset:**
The current model has a single output MLP in the last `TransolverBlock` that predicts all 3 channels simultaneously. Since surface pressure is the primary metric, it may help to give it a dedicated output path with more capacity, while the velocity prediction can use a lighter branch. This is analogous to multi-task learning with task-specific heads (Misra et al., 2016; Vandenhende et al., 2021). For aerodynamic surrogates, pressure and velocity are coupled (via the pressure-velocity Poisson equation) but their prediction difficulty is different on the surface vs. volume.

**Concrete change spec:**
Modify `TransolverBlock` (last layer):
```python
# Current: single mlp2 with out_dim=3
# New: separate heads
self.vel_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 2))
self.surf_p_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
self.vol_p_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

def forward(self, fx, is_surface):
    fx = self.attn(self.ln_1(fx)) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    vel = self.vel_head(fx)  # [B, N, 2]
    p_surf = self.surf_p_head(fx)  # [B, N, 1] — specialized for surface
    p_vol = self.vol_p_head(fx)   # [B, N, 1] — for volume
    # Select pressure branch per node
    p = is_surface.unsqueeze(-1).float() * p_surf + (~is_surface).unsqueeze(-1).float() * p_vol
    return torch.cat([vel, p], dim=-1)  # [B, N, 3]
```

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -8% (dedicated surface pressure head; more capacity for the primary metric).

**Risk/failure modes:**
- The dedicated surface pressure head adds ~2× parameter count for the output; ensure this doesn't cause overfitting on the small surface node subset.
- Volume pressure prediction quality may degrade slightly due to the split.
- The `is_surface` tensor must be threaded to the last block forward.
- Recommended: first test with identical head architectures (no extra capacity) to verify the split alone helps before adding more layers.

**Time cost (fits 30 min?):** Yes. Minor parameter increase in output heads only.

---

## Summary: Top-5 by Expected Impact / Risk Ratio

| Rank | ID | Title | Expected Delta | Risk | Effort |
|------|----|-------|----------------|------|--------|
| 1 | H5 | FiLM Global Conditioning | -4% to -10% | Low | Low |
| 2 | H1 | SWA for OOD Generalization | -3% to -7% | Low | Low |
| 3 | H2 | Surface-Aware Slice Routing | -5% to -12% | Medium | Medium |
| 4 | H11 | Domain-Adversarial Training | -3% to -8% | Medium | Medium |
| 5 | H4 | Per-Sample Re-Based Normalization | -4% to -9% | Medium | Low |

**Notes on orthogonality:** H1 (SWA) and H5 (FiLM) are essentially independent and can be run concurrently. H2 and H12 are complementary and should not be combined in the same run. H4 and H6 (log-scale pressure) are alternatives addressing the same dynamic range problem — pick one first.
