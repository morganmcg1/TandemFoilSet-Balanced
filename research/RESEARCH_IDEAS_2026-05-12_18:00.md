<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-05-12 18:00

Fresh hypotheses for TandemFoilSet CFD surrogate modelling.
Primary target: `val_avg/mae_surf_p` (lower is better).
Baseline architecture: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2.
All proposals respect the 30-minute wall-clock cap, single 96GB GPU, and package constraints.

---

## Idea 1 — `huber-pressure-loss`

**One-line hypothesis.** Replacing the MSE loss with a Huber loss (delta=1.0 in normalized space) will reduce the model's sensitivity to high-Re pressure extremes and improve `val_avg/mae_surf_p` by 3–8%.

**Predicted delta on `val_avg/mae_surf_p`.** −3% to −8% relative (moderate confidence).

**Mechanism.** The current loss is pure MSE in normalized space. Because target magnitudes span roughly an order of magnitude across Re (per-sample y std ranges from ~164 to ~2077 in physical units), a small number of high-Re samples contribute disproportionately large squared gradients and dominate the loss surface. Huber loss transitions from squared to linear behavior for residuals beyond delta, clipping gradient magnitude from outlier samples and re-weighting the effective training signal toward the majority of the distribution. The normalized-space Huber delta of 1.0 corresponds roughly to a 1-sigma residual in the normalized distribution, which is a natural threshold that lets the model learn well from typical samples without being swamped by extremes.

**Implementation sketch.**

In `train.py`, replace the MSE loss computation block (lines ~447–453):

```python
# CURRENT:
sq_err = (pred - y_norm) ** 2
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss

# NEW (add huber_delta: float = 1.0 to Config dataclass):
delta = cfg.huber_delta  # 1.0
residual = pred - y_norm
huber_err = torch.where(
    residual.abs() <= delta,
    0.5 * residual ** 2,
    delta * (residual.abs() - 0.5 * delta),
)
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add to `Config`:
```python
huber_delta: float = 1.0
```

Hyperparameters to hold fixed: `surf_weight=10.0`, `lr=5e-4`, `batch_size=4`, all architecture params unchanged.

**Why now.** The training data contains samples with y std ranging 164–2077 in physical units (documented in program.md value-ranges table). The current MSE loss was not designed for this kind of multi-regime distribution. Huber loss is a direct, minimal-change intervention with no new packages, no architecture changes, and negligible added compute. It is the first thing a Kaggle practitioner would try on a heavy-tailed regression target, and there is strong NeurIPS/ICML precedent for Huber improving PDE surrogate training (Herde et al. 2024, Poseidon paper; McCabe et al. 2023, Multiple Physics Pretraining).

**Risk / failure mode.** If the normalized target distribution is already approximately Gaussian (i.e., normalization by y_std already flattens the tails adequately), Huber will behave identically to MSE for residuals below delta and there will be no effect. The experiment will still be informative: a null result rules out heavy-tailed loss as a bottleneck and confirms normalization is doing its job. A secondary risk is that delta=1.0 is mistuned — if normalized residuals are typically much smaller than 1.0, Huber reduces to MSE and delta is effectively inactive. To diagnose this, log the fraction of residuals exceeding delta during training. If >80% are below delta, try delta=0.2.

---

## Idea 2 — `decoupled-channel-heads`

**One-line hypothesis.** Giving each output channel (Ux, Uy, p) its own independent linear output head, with a pressure-specific surface weight of 20.0 vs. 10.0 for velocity channels, will focus optimization directly on surface pressure MAE.

**Predicted delta on `val_avg/mae_surf_p`.** −4% to −10% relative (moderate confidence).

**Mechanism.** The current architecture uses a single shared MLP output head that maps n_hidden→3 in one step. This means the gradient signal from pressure and velocity errors is pooled before backpropagating through the final layer, giving the model no structural way to prioritize pressure representations. Since the primary metric is `mae_surf_p`, we want the network to allocate capacity specifically toward pressure. Decoupled heads allow (a) independent weight learning per channel, (b) channel-specific surface weighting in the loss without coupling velocity and pressure gradient magnitudes. A `surf_weight_p=20.0` vs `surf_weight_uv=10.0` scheme directly signals that surface pressure accuracy matters twice as much.

**Implementation sketch.**

Replace the output head in `Transolver.__init__`:
```python
# CURRENT (in TransolverBlock last_layer):
self.mlp2 = MLP(n_hidden, n_hidden * mlp_ratio, out_dim, ...)

# NEW: three independent heads in Transolver.__init__
self.head_Ux = nn.Linear(n_hidden, 1)
self.head_Uy = nn.Linear(n_hidden, 1)
self.head_p  = nn.Linear(n_hidden, 1)

# In Transolver.forward, after final block:
fx = self.blocks[-1](fx, return_hidden=True)  # get hidden state before head
pred_Ux = self.head_Ux(fx)
pred_Uy = self.head_Uy(fx)
pred_p  = self.head_p(fx)
preds = torch.cat([pred_Ux, pred_Uy, pred_p], dim=-1)
return {"preds": preds}
```

Add to `Config`:
```python
surf_weight_uv: float = 10.0
surf_weight_p: float = 20.0
```

Update loss in training loop:
```python
sq_err = (pred - y_norm) ** 2  # [B, N, 3]
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
# Per-channel surface weights
ch_surf_weights = torch.tensor(
    [cfg.surf_weight_uv, cfg.surf_weight_uv, cfg.surf_weight_p],
    device=sq_err.device
)
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1) * ch_surf_weights).sum() / (
    surf_mask.sum().clamp(min=1) * 3
)
loss = vol_loss + surf_loss
```

**Why now.** The primary evaluation metric is exclusively surface pressure MAE. Increasing the surface pressure weight is the most direct gradient-level intervention possible. The implementation is simple: three `nn.Linear(128, 1)` layers add negligible parameters. The last block needs a small refactor to expose the hidden state but this is a 5-line change to TransolverBlock.

**Risk / failure mode.** The hidden state before the final head already has implicit channel structure from the shared MLP — decoupling the last layer may not be sufficient if the bottleneck is earlier. In that case, the result will show no improvement and we should consider deeper channel-specific processing. Secondary risk: the pressure head may diverge if the initial learning rate is too high for the new parametrization — if loss explodes in the first few epochs, scale lr to 2e-4.

---

## Idea 3 — `scale-model-256`

**One-line hypothesis.** Scaling the model from n_hidden=128 to n_hidden=256 (with n_head=8) roughly quadruples parameter count from ~1.5M to ~6M and should improve all metrics, especially on the harder OOD camber splits, by 8–15%.

**Predicted delta on `val_avg/mae_surf_p`.** −8% to −15% relative (moderate-high confidence).

**Mechanism.** The baseline is a deliberately small model (~1.5M parameters, n_hidden=128). The original Transolver paper used n_hidden=256 as its standard setting for complex PDE benchmarks. At n_hidden=128, the hidden dimension in PhysicsAttention is 128/4_heads × 4 = dim_head=32, which is very narrow for representing multi-regime physics (100K–5M Re, three domains, both camber and velocity profiles). Doubling n_hidden to 256 increases the expressivity of both the slice-token representations and the per-node MLP feedforward paths. This is a fundamental capacity increase, not a hyperparameter tweak. With 96GB VRAM and batch_size=4, a 6M parameter model fits comfortably — even the largest mesh (242K nodes) at batch_size=4 requires roughly 242K × 4 × 256 × 4 bytes ≈ 1 GB of activations, well within budget.

**Implementation sketch.**

Change `model_config` in `train.py`:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,  # 22
    out_dim=3,
    n_hidden=256,       # was 128
    n_layers=5,
    n_head=8,           # was 4; keeps dim_head=32 consistent
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Keep `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0` unchanged. The AdamW optimizer adapts per-parameter, so lr does not need rescaling for a parameter count change.

**Why now.** The 30-minute timeout may constrain how many epochs complete, but n_hidden=256 does not change the training loop structure — it only changes the forward/backward pass cost per step. A larger model gets fewer epochs in the same wall time but each epoch sees more expressive representations. This is the highest-confidence improvement: scaling laws in neural operators consistently show capacity improvements on complex PDEs (Herde et al. 2024 Poseidon; Alkin et al. 2024 Universal Physics Transformers). The 96GB GPU has ample headroom.

**Risk / failure mode.** If training time per epoch roughly doubles, the 30-minute cap cuts epochs in half (from ~50 to ~25). If the model needs more epochs to converge than available, validation metrics may be worse than the baseline despite higher final potential. To mitigate: consider a warmup-free cosine LR schedule with T_max set to the number of epochs actually completed (available via the early-stop hook). If metrics are improving but truncated, this is evidence to run a longer confirmation run or reduce batch_size to fit more gradient steps per minute.

---

## Idea 4 — `grad-clip-adamw-tuned`

**One-line hypothesis.** Adding gradient clipping (max_norm=1.0) and increasing weight_decay from 1e-4 to 1e-3 will stabilize training on high-Re outlier samples and reduce validation loss variance across the Re-stratified split by 5–10%.

**Predicted delta on `val_avg/mae_surf_p`.** −2% to −6% relative (lower confidence, higher value if training is currently unstable).

**Mechanism.** High-Re samples (up to 5M) produce pressure values up to 29,136 in physical units. After normalization these produce the largest residuals in the training batch and the largest gradient contributions. Without gradient clipping, a single outlier batch can produce gradient norm spikes that shift the entire model state, causing oscillations in the loss curve and preventing convergence to the narrow valley required for accurate surface pressure prediction. Gradient clipping at max_norm=1.0 bounds the per-step perturbation, acting as an adaptive learning rate floor. Increasing weight_decay adds L2 regularization which penalizes large weights that encode domain-specific (Re-specific) mappings rather than generalizable physics, improving OOD performance on `val_re_rand`. Together these changes cost zero compute and require 3 lines of code.

**Implementation sketch.**

In the training loop in `train.py`, after `loss.backward()`:
```python
# ADD after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
optimizer.step()
```

Change `Config`:
```python
weight_decay: float = 1e-3    # was 1e-4
grad_clip: float = 1.0        # new field; set to 0.0 to disable
```

No architecture changes. Keep everything else constant.

**Why now.** This is a diagnostic experiment as much as an improvement experiment. If training loss curves are currently noisy on high-Re batches (which is likely given the 10× variation in target magnitude), gradient clipping will smooth them measurably. The experiment takes essentially zero extra compute and is easy to ablate: set `grad_clip=0.0` to recover current behavior. If it helps, the gain is free. If it does not help, it rules out optimization instability as a bottleneck — which is useful information for prioritizing future architecture or loss experiments.

**Risk / failure mode.** If the gradient norm is already well-behaved (below 1.0 on most steps), clipping will have no effect and the result will be a null. Log the gradient norm before clipping during training to diagnose whether it is ever active. If max gradient norm is consistently below 0.5, the mechanism is not alive and this experiment rules out instability as a cause.

---

## Idea 5 — `mlp-ratio-4-wider-ffn`

**One-line hypothesis.** Increasing the MLP ratio from 2 to 4 in the TransolverBlock feedforward networks, matching the original Transolver paper's default setting, should improve the per-node feature transformation capacity and reduce surface pressure MAE by 4–8%.

**Predicted delta on `val_avg/mae_surf_p`.** −4% to −8% relative (moderate confidence).

**Mechanism.** The Transolver paper uses mlp_ratio=4 as its standard configuration. The current baseline uses mlp_ratio=2, which halves the hidden dimension of the feedforward MLP in each TransolverBlock (from 512 to 256 at n_hidden=128). The feedforward MLP is where per-node nonlinear transformations happen after information is mixed by attention — it is the "compute" phase of the transformer block. At mlp_ratio=2, each block has limited capacity to perform nonlinear feature extraction from the slice-attention output. Restoring mlp_ratio to 4 directly addresses this bottleneck with no change to the attention mechanism or slice structure. Parameter count increases from ~1.5M to ~2.0M — still very compact.

**Implementation sketch.**

Change `model_config` in `train.py`:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=4,        # was 2
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

One-line change. All other config unchanged.

**Why now.** This is the lowest-risk experiment in this list. It is a direct restoration of the paper's intended default configuration. The fact that the baseline uses mlp_ratio=2 instead of 4 may be a conservative tuning decision or an oversight — either way, testing the paper's own recommended setting is the correct first step before more invasive changes. Forward pass cost increases by roughly 30% per layer (since MLP accounts for ~half the computation per TransolverBlock), so expect slightly fewer epochs in 30 minutes — but each epoch should have better gradient quality.

**Risk / failure mode.** If mlp_ratio=2 was intentionally chosen after profiling (e.g., to fit more steps in the wall-clock budget), this experiment may trade epoch count for quality per epoch and net out neutral. If performance is neutral but training loss per step is lower, this is evidence that the model is not underfitting the training set and more capacity is not the bottleneck — in which case the focus should shift to regularization and OOD generalization.

---

## Idea 6 — `more-slices-128`

**One-line hypothesis.** Increasing slice_num from 64 to 128 gives the PhysicsAttention mechanism finer-grained physics partitioning, enabling better separation of near-surface and wake regions, and should reduce surface pressure MAE by 5–10%.

**Predicted delta on `val_avg/mae_surf_p`.** −5% to −10% relative (moderate confidence).

**Mechanism.** In Transolver, `slice_num` controls how many abstract "physics token" slots are learned to represent the full mesh. Each slice captures a soft-cluster of nodes with similar physics behavior (boundary layer, wake, freestream, etc.). At slice_num=64 with meshes of 74K–242K nodes, each slice token represents on average 1,100–3,800 nodes. The surface region is small (a few thousand boundary nodes) and must compete with large interior regions for slice budget. Doubling slice_num to 128 allows the model to dedicate more tokens to the surface and near-surface regions without architectural changes. The PhysicsAttention `in_project_slice` weight goes from shape [dim_head, 64] to [dim_head, 128] — adding ~32K parameters. SDPA cost over slice tokens scales as O(slice_num^2), so memory for the attention on tokens roughly 4× — but since slice tokens are only (batch × heads × 128 × dim_head) in size, this is negligible compared to the node-level tensors.

**Implementation sketch.**

Change `model_config` in `train.py`:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,      # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

One-line change. No other modifications required.

**Why now.** The original Transolver paper evaluates slice_num sensitivity and finds that larger values consistently improve results on complex geometry benchmarks (their best results use slice_num=64 on simpler tasks, up to 128 on complex ones). Our meshes are significantly larger and more complex than the Transolver paper's benchmarks (74K–242K nodes vs. their typical 10K–40K). The baseline slice_num=64 was likely inherited from the original hyperparameters without re-tuning for this mesh scale. This experiment directly tests whether slice capacity is the bottleneck.

**Risk / failure mode.** If the bottleneck is not the number of slices but the expressivity of the per-slice representation (n_hidden), doubling slice_num will have little effect. A null result here, combined with a positive result from idea 3 (scale-model-256), would indicate that hidden dimension matters more than slice count for this dataset. Log the entropy of the slice weight distribution — if slices are already diffuse (low peak weights), adding more slices won't help.

---

## Idea 7 — `re-film-conditioning`

**One-line hypothesis.** Injecting log(Re) as FiLM-style affine conditioning (scale + shift) into each TransolverBlock will give the model an explicit multi-regime pathway and improve `val_re_rand` MAE by 10–15%, with spillover improvements on the camber OOD splits.

**Predicted delta on `val_avg/mae_surf_p`.** −5% to −12% relative (high confidence on val_re_rand, moderate overall).

**Mechanism.** Reynolds number controls the fundamental physics regime: at low Re (~100K), the flow is laminar with thin boundary layers; at high Re (~5M), the pressure distribution is sharply peaked and the wake is turbulent. The current architecture receives log(Re) as a scalar in position 13 of the 24-dimensional input `x`, which is processed identically to spatial coordinates and shape descriptors. This means the model must encode Re-dependent physics entirely within its weights, with no explicit mechanism to switch modes. FiLM (Feature-wise Linear Modulation) — from Perez et al. 2018, well-established in conditional neural fields — injects a conditioning signal as element-wise affine transformation of intermediate features: `h_out = gamma(c) * h + beta(c)` where `c` is the conditioning vector. A tiny 2-layer MLP embeds log(Re) into (2 × n_hidden) parameters (gamma, beta) and modulates the hidden state at each block. This adds ~16K parameters (2 × 128 × 5 blocks × 2) — negligible — but gives the model an explicit route to adapt its computations to the Reynolds regime, which is the most physically meaningful axis of variation in this dataset.

**Implementation sketch.**

Add a `FiLMLayer` class:
```python
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        # Initialize to identity (zero gamma offset, zero beta)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], cond: [B, D_cond]
        out = self.net(cond)  # [B, 2*D]
        gamma, beta = out.chunk(2, dim=-1)  # [B, D] each
        gamma = gamma.unsqueeze(1) + 1.0  # broadcast [B, 1, D], init near 1
        beta = beta.unsqueeze(1)           # broadcast [B, 1, D]
        return gamma * x + beta
```

Add `film: FiLMLayer` to each `TransolverBlock`. Extract log(Re) from input: `log_re = x[:, :, 13].mean(dim=1, keepdim=False)` (all nodes share the same Re per sample, so mean is exact). Pass `cond=log_re.unsqueeze(-1)` (shape [B, 1]) into FiLMLayer after the attention output and before the residual add.

In `Transolver.forward`, extract conditioning before the main loop:
```python
log_re = data["x"][:, :, 13].mean(dim=1)  # [B] — already normalized
cond = log_re.unsqueeze(-1)  # [B, 1]
```

Pass `cond` through each block.

**Why now.** The `val_re_rand` split is explicitly a Re-generalization holdout — it is one of the four equally-weighted val splits. The FiLM mechanism is well-established in conditional neural fields (NeRF variants, NeuralPDE work), requires no new packages (just `nn.Linear` and `nn.SiLU`), and adds minimal parameters. The critical implementation detail is the zero-initialization of the final layer's weights and biases, which ensures the model starts as a copy of the unconditional baseline and can learn Re-dependent modulation progressively. Without zero-init, FiLM conditioning can cause early training instability.

**Risk / failure mode.** If log(Re) is already being used effectively through the shared input embedding (the model may have learned to use it despite no explicit routing), FiLM will add noise during early training without net benefit. The zero-init mitigates this risk significantly. A harder failure mode: if the camber OOD splits are bottlenecked by geometry generalization (not Re), FiLM will help `val_re_rand` but not `val_geom_camber_{rc,cruise}`, and the equal-weight average may show limited movement. Monitor per-split metrics carefully.

---

## Idea 8 — `geometry-aoa-augmentation`

**One-line hypothesis.** Applying light online data augmentation (AoA jitter ±0.5° and NACA parameter jitter ±0.002) on input features 14–21 during training will improve OOD camber generalization (`val_geom_camber_rc`, `val_geom_camber_cruise`) by 5–12% without degrading in-distribution performance.

**Predicted delta on `val_avg/mae_surf_p`.** −3% to −8% relative (moderate confidence; primarily driven by camber OOD splits).

**Mechanism.** The two hardest validation splits test generalization to unseen front-foil camber values (M=6-8 for raceCar, M=2-4 for cruise). The model has never seen these exact camber values during training. The camber and AoA parameters enter as scalar features in x (dims 14–21) that are normalized by x_mean/x_std. By adding small random perturbations to these features at training time, we encourage the model to learn smooth interpolation over the geometry space rather than memorizing the training grid of NACA codes. This is the equivalent of Mixup/CutMix for tabular geometry features — a standard regularization technique for OOD interpolation. The augmentation is applied in `train.py` on the loaded batch before normalization: jitter the AoA dimensions (14, 18) by ±0.5° in radians (~±0.0087 rad) and the NACA camber dims (15, 19) by ±0.002. Since the labels (y) are not changed, this is a form of label-preserving geometric noise injection.

**Implementation sketch.**

In `train.py`, after loading the batch but before normalization, add:
```python
def augment_geometry(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Apply AoA and NACA camber jitter in raw (pre-normalization) feature space."""
    if not cfg.augment or not model.training:
        return x
    x = x.clone()
    aoa_noise = torch.randn(x.shape[0], 1, 1, device=x.device) * cfg.aoa_jitter_rad
    naca_noise = torch.randn(x.shape[0], 1, 1, device=x.device) * cfg.naca_jitter
    # AoA foil 1 (dim 14) and AoA foil 2 (dim 18)
    x[:, :, 14:15] = x[:, :, 14:15] + aoa_noise
    x[:, :, 18:19] = x[:, :, 18:19] + aoa_noise  # tandem foil 2 (zero for single)
    # NACA camber foil 1 (dim 15) and foil 2 (dim 19)
    x[:, :, 15:16] = x[:, :, 15:16] + naca_noise
    x[:, :, 19:20] = x[:, :, 19:20] + naca_noise
    return x
```

Add to `Config`:
```python
augment: bool = True
aoa_jitter_rad: float = 0.00873   # ±0.5 degrees in radians
naca_jitter: float = 0.002        # small perturbation in normalized [0,1] NACA space
```

Call `x = augment_geometry(x, cfg)` before normalization in the training loop. Do NOT apply augmentation during validation or test.

**Important**: The jitter must be applied in raw feature space (before `(x - x_mean) / x_std`), since the jitter values are in the original physical units (radians, NACA [0,1] space). Alternatively, apply after normalization with correspondingly scaled noise: `aoa_jitter_normalized = aoa_jitter_rad / x_std[14]`. Either is correct; pre-normalization is cleaner.

**Why now.** The two OOD camber splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) are part of the four equally-weighted splits that define the primary metric. These are the hardest generalization axes because the model has seen M=2-5 and M=5-9 but not the held-out ranges. Augmentation is the only approach that can improve OOD generalization without changing the model architecture or the training labels — it directly expands the effective training distribution in the geometry parameter space. The implementation requires only tensor operations on the input batch, zero new packages, and adds negligible compute per step.

**Risk / failure mode.** If the camber OOD gap is driven by mesh topology differences (the mesh changes non-smoothly between camber values) rather than smooth function value interpolation, input-space jitter on scalar geometry parameters will not help — the node positions (dims 0-1) and shape descriptors (dims 4-11) encode the actual geometry and would also need perturbation. Perturbing node positions is more complex and risky (requires consistent perturbation of all nodes). If this experiment fails, the follow-up is to analyze whether the dsdf shape descriptor (dims 4-11) already captures sufficient geometric variation and whether augmenting it directly would help.

---

## Summary Table

| Slug | Level | Primary target | Predicted delta | Complexity |
|------|-------|----------------|-----------------|------------|
| `huber-pressure-loss` | Loss | Heavy-tailed Re distribution | −3% to −8% | Very low (1 function) |
| `decoupled-channel-heads` | Architecture (output) | Surface pressure gradient weighting | −4% to −10% | Low (3 linear layers) |
| `scale-model-256` | Architecture (capacity) | All splits, especially OOD | −8% to −15% | Minimal (2 config values) |
| `grad-clip-adamw-tuned` | Optimization | Training stability, val_re_rand | −2% to −6% | Very low (2 lines) |
| `mlp-ratio-4-wider-ffn` | Architecture (FFN) | All splits | −4% to −8% | Minimal (1 config value) |
| `more-slices-128` | Architecture (attention) | Surface nodes, near-surface | −5% to −10% | Minimal (1 config value) |
| `re-film-conditioning` | Architecture (conditioning) | val_re_rand, multi-regime | −5% to −12% | Moderate (new module) |
| `geometry-aoa-augmentation` | Data | val_geom_camber_* splits | −3% to −8% | Low (augment fn) |

## Experiment Tree

Start with the three minimal-change single-config-value ideas in parallel (ideas 3, 5, 6) to establish the capacity/attention frontier quickly. Run idea 4 (grad clip) alongside any of the first three as it is free. Then:

- If `scale-model-256` wins: combine with `huber-pressure-loss` and `decoupled-channel-heads`
- If `more-slices-128` wins: combine with winning capacity change
- If `mlp-ratio-4` wins: combine both mlp-ratio and slices since they are orthogonal
- If all three architecture ideas are neutral: the bottleneck is not capacity → escalate to `re-film-conditioning` and `geometry-aoa-augmentation`
- If `re-film-conditioning` shows strong improvement on `val_re_rand` but not camber: combine with `geometry-aoa-augmentation`
- If all 8 ideas fail or are marginal: revisit the loss surface with a Laplacian/MAE loss instead of MSE/Huber, or investigate the `dsdf` feature representation quality
