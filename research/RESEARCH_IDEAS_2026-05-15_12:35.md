<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TandemFoilSet Research Ideas — 2026-05-15 12:35

## Preamble

**Target**: beat the vanilla Transolver baseline on `val_avg/mae_surf_p` (surface pressure MAE, lower is better).

**Baseline config** (from `train.py`):
- Model: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Loss: `vol_loss + 10.0 * surf_loss` — MSE on normalized targets, uniform per-node weighting
- Optimizer: AdamW `lr=5e-4, weight_decay=1e-4` + CosineAnnealingLR T_max=50 epochs (no warmup)
- Batch size: 4, variable mesh 74K–242K nodes, padded with boolean mask
- Training cap: 50 epochs / 30 min wall-clock (whichever hits first)

**Key bottlenecks identified**:
1. **Dynamic range**: per-sample `y_std` varies up to ~13× within a single split; uniform MSE gives outsized gradient weight to high-Re samples
2. **Geometry generalization**: val splits 2 and 3 are full NACA camber holdouts (M=6-8 raceCar, M=2-4 cruise) — zero training overlap
3. **Surface pressure priority**: primary metric is surface `p` MAE, but loss treats all 3 output channels and all nodes equally (modulo `surf_weight=10`)
4. **No warmup**: large gradients at epoch 0 when the slice projection is still cold — orthogonal init helps but warmup is free insurance

---

## Hypotheses (prioritized 1=highest expected value)

---

### 1. Per-Sample Scale-Normalizing Loss

**Angle**: `loss`

**Hypothesis**: Replacing the global MSE with a per-sample normalized MSE (divide each sample's squared error by that sample's per-channel variance before averaging over nodes) will reduce gradient dominance from high-Re samples and improve surface pressure accuracy across all four splits, especially the low-Re cruise camber split.

**Mechanism**: The baseline global normalization (`y_mean / y_std` from `stats.json`) is computed over the entire dataset. Per-sample y_std varies by up to 13× within a split. A sample at Re=5M contributes ~169× more gradient signal than one at Re=100K under MSE — the model over-fits to the high-Re regime. Dividing each sample's loss by its own y variance (or y_std per channel) makes all samples equally weighted in normalized gradient space. This is the "scale-free MSE" used in Pathak et al. (2022) FourCastNet and Bonev et al. (2023) Spherical FNO for multi-scale meteorological variables.

**Concrete change in `train.py`**:
```python
# In the training loop, after computing sq_err = (pred - y_norm) ** 2
# Compute per-sample, per-channel variance of y_norm over valid nodes
# shape: [B, 3]
sample_var = (
    (y_norm * mask.unsqueeze(-1)).pow(2).sum(dim=1) /
    mask.sum(dim=1, keepdim=True).clamp(min=1).float()
).detach()  # [B, 3], stop gradient through scale

# Scale sq_err: divide each sample's contribution by its own variance + eps
scale = (sample_var.unsqueeze(1) + 1e-6)  # [B, 1, 3]
sq_err_scaled = sq_err / scale  # [B, N, 3]

# Then use sq_err_scaled in place of sq_err for vol_loss and surf_loss computation
vol_loss = (sq_err_scaled * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_scaled * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

**Risk**: Low. The model still trains in normalized space; the per-sample denominator merely rebalances gradients. Worst case: similar performance to baseline if the dynamic range is not actually the bottleneck. No architectural change.

**Cost**: ~50 epochs / 25 min. Same throughput as baseline.

**Citation/source**:
- Bonev et al. (2023), "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere", ICML 2023. Uses relative L2 loss (`||y_pred - y_true||_2 / ||y_true||_2`) per sample.
- Pathak et al. (2022), "FourCastNet", arXiv:2202.11214. Normalizes loss by per-variable standard deviation.
- Tensor Basis Normalization for turbulence ML: arXiv:2410.12255 (scale normalization for improved Re generalization).

---

### 2. LR Warmup + Cosine Annealing (5-epoch linear warmup)

**Angle**: `optimizer`

**Hypothesis**: Adding a 5-epoch linear warmup before handing off to the existing CosineAnnealingLR will stabilize early training when the PhysicsAttention slice projection is far from a good assignment, and will compound with any other improvement by reducing the chance of a bad local optimum at initialization.

**Mechanism**: The orthogonal initialization of `in_project_slice` is good but the model has ~2–4M parameters with no pre-training; learning rate at epoch 0 is 5e-4 which is large relative to the random initialization loss landscape. Slice tokens at epoch 0 are near-uniform — a high LR can lock them into a degenerate assignment before they sort into physically meaningful clusters. Linear warmup from 1e-5 to 5e-4 over 5 epochs (10% of budget) is standard practice in transformer training (Vaswani et al. 2017, Touvron et al. 2021 for ViT). With a 30 min / 50 epoch budget, 5 warmup epochs costs minimal throughput and is free to combine with any hypothesis.

**Concrete change in `train.py`**:
```python
# Replace the scheduler line:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
# With:
WARMUP_EPOCHS = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.02, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS],
)
```

**Risk**: Very low. The only risk is that 5 warmup epochs eat into the 50-epoch budget, but 45 cosine epochs is essentially the same as 50. This is a known-safe change with essentially no downside.

**Cost**: ~50 epochs / 25 min.

**Citation/source**:
- Vaswani et al. (2017), "Attention is All You Need", NeurIPS 2017.
- Touvron et al. (2021), "Training data-efficient image transformers", ICML 2021 — linear warmup for ViT.
- PyTorch `SequentialLR` documentation.

---

### 3. Pressure-Prioritized Loss: Per-Channel Surface Weights

**Angle**: `loss`

**Hypothesis**: Introducing per-channel loss weights that upweight the pressure channel `p` relative to `Ux` and `Uy` on surface nodes will directly optimize the primary ranking metric and improve `mae_surf_p` without sacrificing velocity accuracy.

**Mechanism**: The baseline uses scalar `surf_weight=10.0` applied uniformly across all 3 output channels. The primary metric is `mae_surf_p` (pressure only on surface nodes). `p` gradients compete equally with `Ux` and `Uy` gradients in the loss. Since pressure is scalar and has different physical units and dynamic range from velocity, equal weighting may not be optimal. Adding a channel weight `[w_Ux, w_Uy, w_p]` with `w_p > w_Ux, w_Uy` on surface nodes directly prioritizes the scored channel. This is analogous to task-specific loss weighting in multi-task learning (Kendall et al. 2018 "Multi-Task Learning Using Uncertainty to Weigh Losses").

**Concrete change in `train.py`**:
```python
# Add to Config dataclass:
# p_surf_weight: float = 3.0  # extra multiplier for pressure on surface nodes

# In the loss block, replace surf_loss computation:
# Channel weights: [Ux, Uy, p]
channel_w = torch.tensor([1.0, 1.0, cfg.p_surf_weight], device=device)  # [3]

sq_err = (pred - y_norm) ** 2  # [B, N, 3]
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)

# Pressure-weighted surface loss
sq_err_surf = sq_err * surf_mask.unsqueeze(-1)  # [B, N, 3]
surf_loss = (sq_err_surf * channel_w[None, None, :]).sum() / surf_mask.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * surf_loss
```
Suggested sweep: `p_surf_weight` in {2.0, 3.0, 5.0}; start with 3.0.

**Risk**: Medium. Upweighting `p` may cause `Ux`/`Uy` to degrade; but since the metric only cares about `p`, a trade-off is acceptable. Could also hurt generalization if `p` and velocity are physically coupled (incompressibility). Monitor `mae_surf_Ux`, `mae_surf_Uy` alongside `mae_surf_p`.

**Cost**: ~50 epochs / 25 min.

**Citation/source**:
- Kendall et al. (2018), "Multi-Task Learning Using Uncertainty to Weigh Losses in Deep Learning", CVPR 2018.
- Shu et al. (2023), "Physics-Embedded Neural Networks", NeurIPS 2023 — pressure-velocity decoupled losses.

---

### 4. Fourier Position Features (Random Fourier / Sinusoidal Encoding of x,z)

**Angle**: `features`

**Hypothesis**: Replacing or augmenting the raw (x, z) node coordinates in dims 0-1 with sinusoidal or random Fourier features (RFF) of those coordinates will help the model resolve high-frequency pressure gradients near the airfoil surface and sharp leading/trailing edge features without requiring deeper layers.

**Mechanism**: The baseline feeds raw (x, z) ∈ [0,1]² into a linear preprocess layer. Gradient magnitudes near leading/trailing edges vary on spatial scales ~100× smaller than the domain. A shallow MLP preprocess cannot learn the needed frequency decomposition from raw coordinates alone. Random Fourier features (Rahimi & Recht 2007) or sinusoidal positional encoding (Mildenhall et al. 2020, NeRF) map coordinates to a higher-dimensional feature space that contains the spatial frequencies needed for sharp boundary layer reconstruction. This technique is standard in neural field / implicit neural representation literature and was applied to PDE solving in Fourier Neural Operators (Li et al. 2021) and MeshGraphNets (Pfaff et al. 2021).

**Concrete change in `train.py`**:
```python
# After loading stats, before training loop — define the encoding:
FOURIER_SCALES = 8  # number of frequency bands, each adds 2 dims (sin+cos)

# In the training loop, before calling model:
# x_norm shape: [B, N, 24]; dims 0-1 are already-normalized x,z coords
pos = x_norm[..., :2]  # [B, N, 2]
freqs = (2.0 ** torch.arange(FOURIER_SCALES, device=device).float()) * math.pi  # [K]
# [B, N, 2, K] -> [B, N, 2K] sin + [B, N, 2K] cos
fourier_feats = torch.cat([
    torch.sin(pos.unsqueeze(-1) * freqs),
    torch.cos(pos.unsqueeze(-1) * freqs),
], dim=-1).reshape(x_norm.shape[0], x_norm.shape[1], -1)  # [B, N, 4*K=32]

x_aug = torch.cat([x_norm, fourier_feats], dim=-1)  # [B, N, 24+32=56]
# Update model to accept space_dim + fourier_dim:
# model_config: fun_dim = X_DIM - 2 + 4*FOURIER_SCALES = 54, space_dim = 2
```
Then update `model_config["fun_dim"] = X_DIM - 2 + 4 * FOURIER_SCALES` and pass `x_aug` as input instead of `x_norm`.

**Risk**: Medium. Increases input dimension from 24 to 56; preprocess MLP grows. Need to ensure VRAM stays within 96GB — with batch_size=4 and 242K nodes this should be fine. If Fourier features of the already-normalized coordinates don't add much (the model already has `saf` and `dsdf` which encode relative position), this may be neutral.

**Cost**: ~50 epochs / 28 min (slightly larger preprocess).

**Citation/source**:
- Rahimi & Recht (2007), "Random Features for Large-Scale Kernel Machines", NeurIPS 2007.
- Mildenhall et al. (2020), "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020.
- Tancik et al. (2020), "Fourier Features Let Networks Learn High Frequency Functions", NeurIPS 2020.
- Li et al. (2021), "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.

---

### 5. Increased Model Capacity: `n_hidden=256, n_head=8, slice_num=128`

**Angle**: `arch-tweak`

**Hypothesis**: Doubling the hidden dimension from 128 to 256, heads from 4 to 8, and slices from 64 to 128 will increase the model's representational capacity and improve accuracy on all splits within the 30-minute wall-clock budget, given that the baseline uses a relatively small model for the 74K–242K-node problem.

**Mechanism**: The baseline Transolver uses `n_hidden=128` — this is the smallest published configuration in the Transolver paper (Table 1 in arXiv:2402.02366, which uses 256 or 512 for the benchmark results). The CFD surrogate problem has large mesh sizes (up to 242K nodes) and 24 input features; a 128-dim hidden space may be capacity-limited especially for the geometry-generalization splits. Doubling to 256 doubles the parameter count from ~2M to ~8M (roughly) while keeping the same O(N + S²) complexity. With 96GB VRAM and batch_size=4, 256-dim should still fit. The 30-minute wall-clock is the binding constraint — need to verify throughput.

**Concrete change in `train.py`**:
```python
# In model_config:
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,      # was 128
    n_layers=5,
    n_head=8,          # was 4
    slice_num=128,     # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

**Risk**: Medium. Larger model may not converge in 50 epochs / 30 min. Risk of OOM for cruise samples (242K nodes × batch_size=4 × 256 hidden). Pre-run VRAM estimate: 4 × 242K × 256 × 4 bytes ≈ 1GB for activations alone — should be fine. Main risk is throughput: if iteration time doubles, only 25 epochs fit in 30 min.

**Cost**: ~40 effective epochs / 30 min (may hit wall-clock before epoch 50).

**Citation/source**:
- Wu et al. (2024), "Transolver: A Fast Transformer Solver for PDEs on General Geometries", ICML 2024, arXiv:2402.02366. Table 1: benchmark uses n_hidden=256-512.

---

### 6. Relative Surface Distance Feature (per-node distance to nearest surface point)

**Angle**: `features`

**Hypothesis**: Adding a pre-computed feature representing the geodesic or Euclidean distance from each volume node to the nearest surface node will give the model an explicit boundary-layer proximity signal, improving pressure prediction in the high-gradient near-wall region that dominates surface MAE.

**Mechanism**: The current features include `dsdf` (dims 4-11, distance-based shape descriptor) which encodes distance to the foil boundaries, and `saf` (dims 2-3, signed arc-length) which encodes position along the surface. But volume nodes only receive implicit proximity information. In CFD, boundary layer thickness scales as `Re^{-0.5}`, so a node at distance 0.001 chord lengths from the wall has completely different physics from one at distance 0.1. Providing `log(d_surf + eps)` as an explicit feature lets the model condition pressure prediction on wall proximity directly, reducing the work the attention layers must do to infer it from spatial position alone.

**Concrete change in `train.py`**:
```python
# In the training loop, after loading batch (x, y, is_surface, mask):
# x dims 0-1 are node (x,z); is_surface is [B, N]
# Compute min Euclidean distance from each node to any surface node:

def compute_surf_dist(x_pos, is_surface, mask):
    # x_pos: [B, N, 2], is_surface: [B, N], mask: [B, N]
    B, N, _ = x_pos.shape
    d_list = []
    for b in range(B):
        pos = x_pos[b][mask[b]]           # [M, 2] valid nodes
        surf = x_pos[b][is_surface[b]]    # [S, 2] surface nodes
        # Brute-force: [M, S] pairwise distances — expensive for large N!
        # Use chunked computation or approximate with kNN
        dists = torch.cdist(pos.unsqueeze(0), surf.unsqueeze(0)).squeeze(0)  # [M, S]
        min_d = dists.min(dim=-1).values  # [M]
        d_full = torch.zeros(N, device=x_pos.device)
        d_full[mask[b]] = min_d
        d_list.append(d_full)
    return torch.stack(d_list, dim=0).unsqueeze(-1)  # [B, N, 1]

log_surf_dist = torch.log(compute_surf_dist(x_norm[..., :2], is_surface, mask) + 1e-6)
x_aug = torch.cat([x_norm, log_surf_dist], dim=-1)  # [B, N, 25]
# Update model_config fun_dim = X_DIM - 2 + 1 = 23
```
**Warning**: Brute-force `cdist` over 242K × ~8K surface nodes is O(N * S) per sample — likely too slow. Use pre-computed approximation: the `dsdf` feature (dims 4-11) already encodes something similar. A cheaper alternative: use `dsdf[:, 0]` (the nearest signed distance field value) as a proxy for `log(d_surf)` and skip the runtime computation. Check whether `dsdf` already encodes this; if so, this hypothesis is already baked in and we should look elsewhere.

**Risk**: Medium-high. The brute-force computation may make training too slow. Pre-compute and cache as part of the `.pt` files (but data loaders are read-only). Best to check whether `dsdf` already captures this before building runtime computation.

**Cost**: ~50 epochs / ~30 min (if fast) or infeasible (if brute-force).

**Citation/source**:
- Pfaff et al. (2021), "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021 — wall distance feature.
- Bhatnagar et al. (2019), "Prediction of aerodynamic flow fields using convolutional neural networks", Computational Mechanics — boundary layer distance feature.

---

### 7. Geometry-Conditioned Slice Assignment (PGOT-Style Geometry Injection)

**Angle**: `arch-tweak`

**Hypothesis**: Injecting geometric conditioning (NACA parameters + gap/stagger from dims 15-23 of x) directly into the slice assignment projection in PhysicsAttention will help the model create geometry-aware slice tokens, improving performance on the camber generalization splits.

**Mechanism**: In the baseline Transolver, `slice_weights = softmax(in_project_slice(x_mid) / temperature)` uses the full node feature to assign nodes to slice tokens. The slice assignment does implicitly see geometry features (NACA params, gap, stagger are in `x`), but they are diluted among 22 features. PGOT (arXiv:2512.23192, Dec 2025) shows that explicitly injecting geometric parameters into the slice assignment ("physics slicing-geometry injection") improves generalization to unseen geometries by 8-15% on standard benchmarks. The insight is that the slice token boundaries should shift with geometry, not just with local flow state.

**Concrete change in `train.py`**:
```python
# In PhysicsAttention.__init__, add a geometry conditioning branch:
# geom_dim = 9  (dims 15-23 of x: NACA foil1 [3], AoA foil2 [1], NACA foil2 [3], gap [1], stagger [1])
self.geom_project = nn.Linear(9, dim_head)  # project geometry to same dim as x_mid

# In PhysicsAttention.forward, before slice_weights:
# Pass global geometry features (per-sample, broadcast to N nodes)
# geom_feats: [B, 1, 9] — take dims 15-23 from first valid node (they're same for all nodes)
geom_feats = x_orig[..., 15:24][:, :1, :]  # [B, 1, 9] global geometry
geom_proj = self.geom_project(geom_feats)   # [B, 1, dim_head]
# Reshape to match x_mid: [B, heads, 1, dim_head]
geom_proj = geom_proj.unsqueeze(1).expand(-1, self.heads, -1, -1)

# Modify slice assignment: add geometry bias
# slice_weights = softmax((in_project_slice(x_mid) + geom_scale * in_project_slice(geom_proj)) / temp)
slice_input = x_mid + geom_proj  # broadcast addition
slice_weights = self.softmax(self.in_project_slice(slice_input) / self.temperature)
```
This requires passing the original `x` (with geometry dims) through to PhysicsAttention, or extracting geometry features in Transolver.forward before calling blocks.

**Risk**: Medium. Requires surgery on PhysicsAttention forward and data flow. Geometry features are already in `x`, so the model theoretically can already do this — the question is whether explicit injection helps. Failure mode: if NACA params are already well-used by the baseline, this adds noise.

**Cost**: ~50 epochs / 27 min.

**Citation/source**:
- Wu et al. (2025), "PGOT: Physics-Geometry Operator Transformer", arXiv:2512.23192. Reports 8-15% improvement on geometry-generalization benchmarks via geometry injection into slice assignment.

---

### 8. Curriculum Learning: High-Re First, Then Mixed (Re-Staged Training)

**Angle**: `training`

**Hypothesis**: Training first on high-Re samples (Re > 1M) for the first 15 epochs, then switching to the full balanced sampler, will prevent the model from spending early capacity on low-Re flow regimes that are simpler and may cause conflicting gradient directions.

**Mechanism**: High-Re flows dominate the loss magnitude under the current uniform normalization (per-sample y_std up to 2,077 vs. 164 for low-Re cruise). Starting with high-Re cases where the flow features are most strongly expressed may give the model a better initialization of the slice tokens and MLP weights before it encounters low-Re samples that require a different feature regime. This is analogous to curriculum learning strategies in NLP (Bengio et al. 2009) and scientific ML (Krishnapriyan et al. 2021 for PINN failure modes, Herde et al. 2024 for Poseidon). The sampler can be swapped in `train.py` without touching the data loaders.

**Concrete change in `train.py`**:
```python
# Add to Config:
# curriculum_warmup_epochs: int = 15  # epochs of high-Re only training

# In training setup, extract high-Re sample indices:
# log(Re) is x dim 13 (normalized). Load raw values:
# Re > 1M corresponds to log(Re) > log(1e6) ≈ 13.8
# After normalization with x_mean[13], x_std[13], threshold = (13.8 - x_mean[13]) / x_std[13]
# But we don't have per-sample Re at sampler creation time without reading all files.
# Alternative: use domain group as proxy — raceCar tandem (Re 1M-5M) and cruise (Re 110K-5M)
# Simpler: use sample_weights already provided, with a custom sampler that phases in all domains

# Phase 1: use weights that only sample raceCar tandem + raceCar single (high Re) — set cruise weights to 0
phase1_weights = sample_weights.clone()
cruise_indices = [i for i, g in idx_to_group.items() if "cruise" in g]
phase1_weights[cruise_indices] = 0.0
# Normalize
phase1_weights = phase1_weights / phase1_weights.sum()

# Phase 2: original balanced sampler (after curriculum_warmup_epochs)
# Switch sampler mid-training in the epoch loop:
for epoch in range(MAX_EPOCHS):
    if epoch == cfg.curriculum_warmup_epochs:
        # rebuild DataLoader with full balanced sampler
        train_loader = DataLoader(..., sampler=WeightedRandomSampler(sample_weights, ...))
```

**Risk**: Medium. Curriculum learning is sensitive to the switch epoch and the definition of "hard" samples. If cruise low-Re samples are actually easier to learn first (smaller values), this curriculum may be backwards. Monitor `val_geom_camber_cruise` specifically — if it degrades vs. baseline, the curriculum is harmful.

**Cost**: ~50 epochs / 25 min.

**Citation/source**:
- Bengio et al. (2009), "Curriculum Learning", ICML 2009.
- Krishnapriyan et al. (2021), "Characterizing possible failure modes in physics-informed neural networks", NeurIPS 2021.
- Herde et al. (2024), "Poseidon: Efficient Foundation Models for PDEs", arXiv:2405.19101 — staged multi-resolution pretraining.

---

### 9. Stochastic Depth (Layer Drop) Regularization

**Angle**: `regularization`

**Hypothesis**: Applying stochastic depth (layer-level dropout with increasing drop probability per layer) to the Transolver blocks will act as an ensemble of architectures of varying depth during training, improving generalization on the camber and Re holdout splits.

**Mechanism**: With 1499 training samples and 5 layers, the model is at moderate risk of over-fitting the training distribution of NACA shapes and Re values. Stochastic depth (Huang et al. 2016) randomly skips each residual block during training with probability `p_l = l / L * drop_rate` (linearly scaled per layer), creating an implicit ensemble. This is standard in Vision Transformers (DeiT, Swin) and was found to be particularly effective for small datasets. For 5 layers with `drop_rate=0.1`, the expected per-block drop probability is {0.02, 0.04, 0.06, 0.08, 0.10} — mild but meaningful regularization.

**Concrete change in `train.py`**:
```python
# In TransolverBlock.forward:
# Add a stochastic depth skip (only active during training):
def forward(self, x):
    if self.training and self.drop_path_prob > 0.0:
        if torch.rand(1).item() < self.drop_path_prob:
            return x  # skip this block (residual identity)
    # ... existing forward logic ...

# In Transolver.__init__, assign drop probabilities:
drop_rates = [i / len(self.blocks) * drop_path_rate for i in range(len(self.blocks))]
for block, rate in zip(self.blocks, drop_rates):
    block.drop_path_prob = rate

# Add to Config:
# drop_path_rate: float = 0.1
```

**Risk**: Low-medium. Stochastic depth with small drop_rate is very conservative. Worst case: negligible effect. With only 50 epochs, the regularization effect may not be large enough to matter — this is more valuable for longer training runs.

**Cost**: ~50 epochs / 25 min.

**Citation/source**:
- Huang et al. (2016), "Deep Networks with Stochastic Depth", ECCV 2016.
- Touvron et al. (2021), "Training data-efficient image transformers", ICML 2021 — DeiT uses stochastic depth with `drop_path_rate=0.1`.

---

### 10. Auxiliary Task: Predict Reynolds Number from Node Features

**Angle**: `aux-task`

**Hypothesis**: Adding an auxiliary head that predicts `log(Re)` from the model's internal representations (after the final Transolver block, using surface-node-averaged features) will force the model to learn Re-aware representations, improving cross-Re generalization on `val_re_rand`.

**Mechanism**: `log(Re)` is feature dim 13 of x — the model already has it as input. But the model may not use it effectively as a conditioning variable deep in the network (it is processed through the preprocess MLP and then mixed into slice tokens). Adding an auxiliary supervised prediction of `log(Re)` from the final block's surface-node average will create a gradient pathway that forces the final representations to be discriminatively Re-aware. This is analogous to auxiliary domain discriminator heads in domain adaptation (Ganin et al. 2016) and multi-task learning auxiliary regression heads in scientific ML (Yin et al. 2022). The auxiliary loss is removed at inference — no inference cost.

**Concrete change in `train.py`**:
```python
# Add auxiliary head to model (or outside model, in train.py):
re_head = nn.Sequential(
    nn.Linear(n_hidden, n_hidden // 2), nn.GELU(),
    nn.Linear(n_hidden // 2, 1)
).to(device)
re_optimizer = torch.optim.AdamW(re_head.parameters(), lr=cfg.lr)

# In training loop, after model forward pass:
# Extract final block's output (fx before the final MLP output layer)
# Pool over surface nodes: [B, n_hidden]
surf_pool = (fx * is_surface.unsqueeze(-1).float()).sum(dim=1)
surf_pool = surf_pool / is_surface.float().sum(dim=1, keepdim=True).clamp(min=1)
re_pred = re_head(surf_pool).squeeze(-1)  # [B]
re_true = x[:, 0, 13]  # log(Re) is same for all nodes in a sample, take from node 0
re_loss = F.mse_loss(re_pred, re_true)
loss_total = loss + 0.1 * re_loss  # small weight so it doesn't dominate
```
This requires modifying Transolver to expose intermediate `fx` before the final projection.

**Risk**: Medium. Requires exposing internal model state; adds an auxiliary optimizer. `log(Re)` is already explicitly in x, so this may be redundant — the model can already "read" it. Benefit is forcing the representations to be useful for Re-conditioned predictions even when Re is indirectly relevant.

**Cost**: ~50 epochs / 27 min.

**Citation/source**:
- Ganin et al. (2016), "Domain-Adversarial Training of Neural Networks", JMLR 2016.
- Yin et al. (2022), "Continuous PDE Dynamics Forecasting with Implicit Neural Representations", arXiv:2209.14855 — auxiliary physics constraint heads.

---

### 11. Data Augmentation: AoA Jitter + Re Interpolation

**Angle**: `augmentation`

**Hypothesis**: Adding per-sample random jitter to the AoA feature and log(Re) feature during training (±0.5° AoA, ±0.1 log-Re units) will act as feature-space augmentation that regularizes the model against exact-parameter memorization and improves Re and AoA generalization.

**Mechanism**: With only 1499 training samples (457 raceCar tandem, 443 cruise tandem, 599 single), the model likely memorizes specific (AoA, Re, NACA) combinations. Adding noise to the conditioning scalar features (not to the mesh positions or is_surface) simulates having more diverse training cases. The jitter must be small enough not to corrupt the physics: ±0.5° out of [-10°, +6°] is ~3% of the range, and ±0.1 log-Re out of [log(1e5), log(5e6)] ≈ 2% of the range. This is analogous to label smoothing and input noise in classification, and to random perturbation of boundary conditions in PDE surrogate training (Herde et al. 2024 Poseidon).

**Concrete change in `train.py`**:
```python
# In training loop, after loading batch (x, y, is_surface, mask) and before normalization:
if model.training:  # only during training, not val/test
    # AoA jitter: dims 14 (foil1) and 18 (foil2), in radians
    # ±0.5° = ±0.00873 radians; apply before normalization
    aoa_noise = (torch.rand_like(x[..., 14:15]) - 0.5) * 0.0175  # ±0.00873
    x[..., 14:15] += aoa_noise
    x[..., 18:19] += aoa_noise * (x[..., 18:19].abs() > 1e-6).float()  # only if foil2 AoA != 0

    # log(Re) jitter: dim 13, add noise before normalization
    re_noise = (torch.rand_like(x[..., 13:14]) - 0.5) * 0.2  # ±0.1 log-Re units
    x[..., 13:14] += re_noise
```

**Risk**: Medium. Feature jitter corrupts the physics if too large. The augmented `x` is used as input only — `y` (the CFD solution) is not changed, so there is a label-input mismatch for the jittered samples. This may hurt if the model learns to be confused by the mismatch rather than to interpolate. Recommend starting with very small noise (half the values above) and monitoring training loss for instability.

**Cost**: ~50 epochs / 25 min.

**Citation/source**:
- Herde et al. (2024), "Poseidon: Efficient Foundation Models for PDEs", arXiv:2405.19101 — boundary condition perturbation augmentation.
- Benton et al. (2020), "Learning Invariances in Neural Networks from Training Data", NeurIPS 2020.

---

### 12. Alternative Architecture: Graph Neural Network Baseline (GCN + Message Passing)

**Angle**: `arch-new`

**Hypothesis**: Replacing Transolver with a simple but well-tuned Graph Convolutional Network (GCN with edge-index from k-nearest-neighbors graph over node positions) will serve as a discriminating diagnostic: if the GNN matches or beats Transolver, the bottleneck is the attention mechanism; if it underperforms, Transolver's global attention is genuinely needed.

**Mechanism**: Transolver's key claim is that its "physics token" attention is better than purely local message passing for global pressure field prediction. Testing a GNN baseline verifies this claim. A 5-layer GCN (or GraphSAGE) with k=16 neighbors (pre-built KNN graph at batch construction time) is a strong local baseline. If it underperforms, we have evidence that global attention matters and should double down on Transolver improvements. If it performs comparably, the bottleneck may be the loss, data, or normalization, not the architecture. This is a discriminating diagnostic, not just another architecture to try.

**Concrete change in `train.py`**:
```python
# Add torch_geometric or use a simple manual message passing (no new packages):
# Manual GCN with pre-built KNN graph using only torch:

class NodeGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=5, k=16):
        super().__init__()
        self.k = k
        self.preprocess = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = data["x"]  # [B, N, in_dim]
        h = self.preprocess(x)  # [B, N, hidden_dim]
        pos = x[..., :2]  # [B, N, 2]
        # Build kNN graph on the fly (expensive but correct):
        # dist_mat = torch.cdist(pos, pos)  # [B, N, N] -- too large for N=242K
        # For N>10K, use random sub-sampling or skip this architecture
        ...
        return {"preds": self.out(h)}
```
**Note**: For N=242K, a dense pairwise graph is infeasible. This architecture requires `torch_geometric` or a custom sparse implementation. Given the "no new packages" constraint, this may be impossible to implement cleanly without adding `torch_geometric` to `pyproject.toml`. The experiment is most valuable as a diagnostic concept — if `torch_geometric` can be added in the same PR, it becomes viable.

**Risk**: High. The "no new packages" constraint may make this architecturally infeasible for large meshes. Implementation complexity is high. Deprioritized to 12 for this reason — include only if a student has strong GNN experience and can add `torch_geometric` to `pyproject.toml`.

**Cost**: ~30 epochs / 25 min (smaller model, potentially faster per-step).

**Citation/source**:
- Kipf & Welling (2017), "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.
- Pfaff et al. (2021), "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021 — MeshGraphNets for CFD.
- Mavriplis (2024), "Graph Transformers for 2D Airfoil Flow Reconstruction", arXiv:2501.17081.

---

## Avoidance Notes

The following directions should **not** be tried without new evidence:

1. **Global y_std re-normalization (changing stats.json)**: Changing the global normalization stats changes the model output contract. The `data/scoring.py` relies on `y_std * pred + y_mean` — do not change `stats.json` or the normalization constants without verifying the full metric pipeline.

2. **Batch size > 4**: With 242K-node cruise samples, batch_size=4 already uses substantial VRAM. Increasing may OOM. Accumulate gradients instead if larger effective batch is needed.

3. **Changing data loaders or `data/` files**: These are explicitly read-only in the constraints.

4. **Attention mask to exclude padding from attention**: The current `pad_collate` uses zeros for padding. The PhysicsAttention forward does not apply an attention mask — adding one could help, but requires verifying that the slice-weight computation correctly handles zero-padded nodes (currently they may contribute small but non-zero slice weights due to zeros being a valid feature value). This is a potential source of subtle error but not obvious enough to prioritize without diagnosis.

5. **Re-scaling `surf_weight` without also changing the loss structure**: Tuning `surf_weight` alone is likely a local minimum — the baseline was tuned to 10.0. Without addressing the dynamic range issue, changing surf_weight is just noise.

---

## Student Assignment Table (8 Students, 1 GPU Each)

| Student | Hypothesis # | Title | Angle | Expected Impact | Screening epochs |
|---------|-------------|-------|-------|-----------------|-----------------|
| 1 | 1 | Per-Sample Scale-Normalizing Loss | loss | High — addresses dynamic range bottleneck directly | 50 |
| 2 | 2 | LR Warmup + Cosine Annealing | optimizer | Medium-High — free insurance, stacks with everything | 50 |
| 3 | 3 | Pressure-Prioritized Loss: Per-Channel Surface Weights | loss | High — directly optimizes primary metric | 50 |
| 4 | 4 | Fourier Position Features | features | Medium — better spatial resolution near surface | 50 |
| 5 | 5 | Increased Model Capacity (n_hidden=256) | arch-tweak | Medium-High — baseline may be underpowered | 40-50 |
| 6 | 7 | Geometry-Conditioned Slice Assignment | arch-tweak | Medium — PGOT-validated approach, helps camber splits | 50 |
| 7 | 8 | Curriculum Learning: High-Re First | training | Medium — targets Re generalization split | 50 |
| 8 | 9 | Stochastic Depth Regularization | regularization | Low-Medium — conservative but clean | 50 |

**Note**: Hypotheses 6 (wall-distance feature), 10 (aux Re prediction), 11 (AoA/Re jitter), and 12 (GNN) are held in reserve as follow-on experiments. Assign hypothesis 6 if hypothesis 4 (Fourier features) returns neutral results and the dsdf feature is confirmed to already encode wall distance. Assign hypothesis 11 if curriculum learning (#8 above) shows promise on the Re split.
