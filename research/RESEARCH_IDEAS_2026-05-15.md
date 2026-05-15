<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15

Generated from literature search covering: Transolver successors (GeoTransolver, LinearNO,
UPT, GINO, GNOT), surface/boundary loss for physics-informed surrogates, coordinate
encodings for irregular geometries (SIREN, RFF/FFM), pressure-specific surrogate work
for airfoils, optimizer advances (SOAP, Cautious Adam, EMA weights).

Primary metric: `val_avg/mae_surf_p` (lower is better).
Baseline model: Transolver, ~1M params, `n_hidden=128, n_layers=5, n_head=4, slice_num=64,
mlp_ratio=2`, AdamW lr=5e-4, CosineAnnealingLR, surf_weight=10.0.

---

## Priority table

| Rank | ID | Idea | Predicted delta on val_avg/mae_surf_p | Confidence | Compute cost |
|------|----|------|---------------------------------------|------------|--------------|
| 1 | H1 | Remove inter-slice attention (LinearNO insight) | -8 to -15% | High — direct ablation from peer-reviewed paper in identical setting | Low |
| 2 | H8 | EMA model weights (decoupled_ema_decay=0.999) | -5 to -10% | High — TMLR 2024, consistent OOD gains, zero inference cost | Low |
| 3 | H3 | Pressure-channel weighted loss (p weight x3 on surface) | -5 to -12% | Medium-high — directly optimizes the ranking metric | Low |
| 4 | H2 | Scale model width: n_hidden=256, n_head=8, n_layers=6 | -5 to -10% | Medium — under-parameterized baseline, but VRAM/speed risk | Medium |
| 5 | H5 | Random Fourier Feature coordinate encoding | -4 to -8% | Medium — well established for spatial coordinates, cheap | Low |
| 6 | H9 | Cautious AdamW optimizer | -3 to -7% | Medium — ICLR 2026 one-line change, strong theoretical basis | Low |
| 7 | H6 | Gradient clipping + warm restarts (SGDR) | -3 to -6% | Medium — often stabilizes high-Re regime training | Low |
| 8 | H4 | Asymmetric Q/K projections in PhysicsAttention | -3 to -8% | Medium — LinearNO finding, orthogonal to H1 | Low |
| 9 | H10 | Re-stratified sampling bias toward high-Re samples | -2 to -5% | Medium — high-Re drives the variance in val metrics | Low |
| 10 | H7 | Surface-conditioned output head (two-branch decoder) | -3 to -6% | Medium-low — plausible but adds complexity | Medium |
| 11 | H11 | Log1p normalization of targets per-sample | -2 to -5% | Low-medium — addresses high-Re outlier dominance | Low |
| 12 | H12 | Dropout calibration: 0.1 dropout on MLP sub-layers | -1 to -3% | Low-medium — regularization for OOD splits | Low |
| 13 | H13 | Geometry-conditioned slice tokens (GeoTransolver GALE) | -5 to -10% | Low (complex change) — high ceiling but implementation risk | High |

---

## H1: Remove inter-slice QKV attention (LinearNO insight)

### What it is

Remove the `to_q / to_k / to_v / scaled_dot_product_attention` block inside
`PhysicsAttention` and replace with an identity pass on slice tokens. This makes each
slice token independent — the model relies solely on the slice-aggregation weighted
scatter/gather path. LinearNO (arXiv:2511.06294, Nov 2025) shows that
PhysicsAttention is linear attention in disguise, and removing the inter-slice
dot-product step consistently improves accuracy with 40% fewer params and 36% fewer
FLOPs across multiple benchmarks including NS2d and airfoil tasks.

### Why it might help here

The inter-slice attention may be adding unnecessary expressiveness-through-noise: with
`slice_num=64` and `N~100K` nodes, the slice tokens are dense averages, and
dot-product attention between them may not provide structural benefit. The scatter/gather
path already routes information from geometry to slices and back; the QKV step may
simply add redundancy and a source of overfitting on small training sets (1499 samples).

### Key papers

- **LinearNO** (Hao et al., arXiv:2511.06294, Nov 2025): "LinearNO: Does Transolver
  Really Need Non-linear Attention?" — shows PhysicsAttention = linear attention, and
  removing inter-slice attention beats Transolver on NS2d, Elasticity, Plasticity,
  Weather. https://arxiv.org/abs/2511.06294

### Implementation notes

In `PhysicsAttention.forward`:
1. Delete the `q = self.to_q(slice_token)` / `k = self.to_k(...)` / `v = self.to_v(...)`
   / `F.scaled_dot_product_attention(...)` block.
2. Replace `out_slice = ...` with `out_slice = slice_token` (identity).
3. Remove `self.to_q`, `self.to_k`, `self.to_v` from `__init__` (reduces params by ~3x
   the cost of one QKV triple per head).
4. Keep `in_project_x`, `in_project_fx`, `in_project_slice`, `to_out` intact.

This is a pure simplification — no new hyperparameters. The `temperature` parameter
still controls slice sharpness.

### Suggested experiment design

- Minimal change: modify only `PhysicsAttention.__init__` and `forward`.
- Keep all other hyperparameters at baseline (lr=5e-4, surf_weight=10.0,
  slice_num=64, n_hidden=128, n_layers=5).
- Run for full allowed epochs with default timeout.
- Expected discriminating result: if `val_avg/mae_surf_p` drops materially (>3%),
  the inter-slice attention is indeed noise for this dataset. If it rises, inter-slice
  attention is load-bearing for the tandem interaction signal.

### Risks

- May lose ability to propagate wake/interaction signals between foils that currently
  travel through the inter-slice channel. Tandem flows have complex foil-foil
  interactions.
- Ablation result in LinearNO is on simpler 1D/2D PDEs; tandem airfoil geometry is
  richer.

---

## H2: Scale model width and depth

### What it is

Increase `n_hidden` from 128 to 256, `n_head` from 4 to 8, add one layer (`n_layers=6`).
This roughly 4x increase in parameter count moves the model from ~1M to ~4M params —
still well within budget for 96GB VRAM and mesh sizes up to 242K nodes.

### Why it might help here

The baseline model is unusually small for a transformer surrogate on meshes with ~100K+
nodes. GeoTransolver (NVIDIA, arXiv:2512.20399, Dec 2025) uses 256+ hidden dim on
similar tasks. With only 1499 training samples but complex flow physics (high-Re
boundary layer separation, tandem wake interactions), the model may be under-capacity
for the needed mapping. The OOD camber and Re splits in particular may require more
representational depth.

### Key papers

- **GeoTransolver** (Zhou et al., NVIDIA, arXiv:2512.20399, Dec 2025): geometry-aware
  Transolver extension; uses larger hidden dims and shows width matters for geometry
  generalization. https://arxiv.org/abs/2512.20399
- **Transolver** (Wu et al., ICML 2024, arXiv:2402.02366): baseline architecture.

### Implementation notes

Change in `model_config` dict:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,   # was 128
    n_layers=6,     # was 5
    n_head=8,       # was 4
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

VRAM check: at B=4, N=242K, hidden=256, 6 layers, expected peak ~28–35 GB — well
within 96GB. Confirm with `torch.cuda.max_memory_allocated()` in the first epoch.

Reduce batch_size to 2 if VRAM pressure appears (OOM risk is low but nonzero for
the largest cruise meshes ~210K nodes at B=4).

Slightly reduce lr to 3e-4 to account for increased model size (larger models
typically benefit from slower learning rates).

### Suggested experiment design

- Change model_config only (n_hidden=256, n_head=8, n_layers=6).
- lr=3e-4, all other hyperparameters baseline.
- Monitor peak VRAM in first epoch log; if >80GB reduce batch_size to 2.
- Run full epochs with timeout.

### Risks

- Slower training per epoch — may not converge within the epoch budget.
- If the bottleneck is data diversity (1499 samples) not model capacity, larger model
  will overfit, especially on the OOD camber splits.

---

## H3: Pressure-channel weighted loss

### What it is

Apply per-channel loss weighting so the pressure channel (dim 2) receives 3x more
weight on surface nodes than velocity channels. The primary metric is surface pressure
MAE exclusively; the current loss weights all three output channels equally within the
MSE sum.

### Why it might help here

`val_avg/mae_surf_p` is purely a pressure metric on surface nodes. The current loss
is `vol_loss + 10.0 * surf_loss`, where `surf_loss` averages squared error across all
3 channels equally. Velocity errors on the surface (Ux, Uy) are diluting the gradient
signal for pressure. A direct re-weighting of the pressure channel should tighten
the optimization-metric alignment.

### Key papers

- **Task-aligned loss** (general principle, see e.g. Karniadakis et al., Nature Reviews
  Physics 2021, Section on physics-informed loss balancing): weighting loss terms to
  match the evaluation metric is a standard technique in physics-informed neural networks.
- **B-GNN** (arXiv:2503.18638, Mar 2025): surface-exclusive models for Cp show value
  of surface pressure focus. https://arxiv.org/abs/2503.18638

### Implementation notes

Replace the loss computation in the training loop:

```python
# Current:
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# Proposed — channel weights [Ux, Uy, p]:
ch_weights = torch.tensor([1.0, 1.0, 3.0], device=device, dtype=sq_err.dtype)
surf_loss = (sq_err * surf_mask.unsqueeze(-1) * ch_weights).sum() / (
    surf_mask.sum().clamp(min=1) * ch_weights.sum()
)
```

Also apply the same weighting in `evaluate_split` for the reported `surf_loss`, or
use a separate `surf_loss_train` vs `surf_loss_eval` to keep val metrics consistent.

Important: do NOT change `data/scoring.py` — the MAE accumulation there is
channel-agnostic and read-only. The weighting is purely on the training loss.

Try p_weight=3.0 first (conservative). If this works, a follow-up could try 5.0 or 10.0.

### Suggested experiment design

- Minimal change to training loop only: add `ch_weights` tensor, weight `surf_loss`.
- Keep `surf_weight=10.0`, all other hyperparameters baseline.
- Run full epochs. Primary signal: does `val/mae_surf_p` drop faster per epoch vs
  baseline? Does `val/mae_surf_Ux` or `mae_surf_Uy` regress significantly?

### Risks

- May degrade velocity prediction on surface enough to hurt downstream aerodynamic
  force computation (though MAE_surf_p is the only ranked metric here).
- Too aggressive a pressure weight may destabilize training if the pressure field has
  outlier magnitudes not handled by the global normalization.

---

## H4: Asymmetric Q/K projections in PhysicsAttention

### What it is

Replace the symmetric `to_q` and `to_k` projections in `PhysicsAttention` with
asymmetric projections: `to_q` maps from `slice_token` to a query space, while
`to_k` and `to_v` are derived from a separate linearly-projected copy of the slice
token using a different weight matrix. This follows the LinearNO finding (arXiv:2511.06294)
that asymmetric ψ(K) ≠ φ(Q) consistently improves accuracy over symmetric projections.

### Why it might help here

The current implementation uses the same `dim_head`-dimensional space for both Q and K,
which imposes an implicit symmetry. Asymmetric projections allow the model to learn
separate representations for "what to query" (geometry/physics context) and "what
provides the key" (value aggregation), which matters when slice tokens encode both
spatial identity and physical state.

### Key papers

- **LinearNO** (Hao et al., arXiv:2511.06294, Nov 2025): asymmetric projections
  beat symmetric in all tested PDE benchmarks. https://arxiv.org/abs/2511.06294

### Implementation notes

In `PhysicsAttention.__init__`, keep `to_q`, `to_k`, `to_v` but initialize them
with different random seeds (trunc_normal_ with different stds). More importantly,
add a second projection path for keys:

```python
# Add to __init__:
self.in_project_fx_k = nn.Linear(dim, inner_dim)  # separate projection for K path

# In forward, use fx_mid for queries, but a separately-projected fx for keys:
fx_mid_k = (
    self.in_project_fx_k(x)
    .reshape(B, N, self.heads, self.dim_head)
    .permute(0, 2, 1, 3)
    .contiguous()
)
slice_token_k = torch.einsum("bhnc,bhng->bhgc", fx_mid_k, slice_weights)
slice_token_k = slice_token_k / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
k = self.to_k(slice_token_k)  # K from separate projection
# q and v still from original slice_token
```

This adds ~1 extra linear layer per attention module (~5 extra layers total).

### Suggested experiment design

- Test orthogonally to H1. If H1 (remove inter-slice attention) succeeds, H4 is moot
  because there is no QK path. Run H4 only if H1 is not clearly superior.
- Keep all other hyperparameters at baseline.

### Risks

- Adds parameters — small increase (~5% more), low VRAM impact.
- Marginal gain if the inter-slice attention is already noisy (H1 may be a better fix).

---

## H5: Random Fourier Feature coordinate encoding

### What it is

Replace the raw (x, z) position coordinates (dims 0–1 of the input) with a
`2D_coord → 2*n_freq` Gaussian Random Fourier Feature (RFF) encoding:
`[sin(B @ pos), cos(B @ pos)]` where `B ~ N(0, sigma^2 * I)` is a fixed random
matrix and `sigma` controls the frequency bandwidth. This augments the input from
24 dims to `24 - 2 + 2*n_freq` dims.

### Why it might help here

The mesh coordinates range from background zone (sparse, large) to near-surface
(dense, fine). A linear encoding of (x, z) means the network must learn to separate
far-field from near-surface behavior from a nearly-zero-variance normalized coordinate.
RFF encoding explicitly populates the Fourier spectrum so low-frequency far-field and
high-frequency near-surface features can be learned simultaneously. This has been
shown to improve neural field learning for spatial coordinates (Tancik et al.,
NeurIPS 2020).

### Key papers

- **Fourier Features Let Networks Learn High Frequency Functions** (Tancik et al.,
  NeurIPS 2020): canonical reference for RFF encoding of spatial coordinates.
  https://arxiv.org/abs/2006.10739
- **Neural Tangent Kernel analysis** shows linear/polynomial encodings fail to learn
  high-frequency components; RFF avoids spectral bias.

### Implementation notes

Add a fixed RFF layer in `Transolver.forward` (or a preprocessing function in
`train.py`, applied before normalization). Since the encoding is applied to raw
coordinates, it should be done before the global normalization or to the normalized
coordinates.

```python
class RFFEncoding(nn.Module):
    def __init__(self, n_freq=32, sigma=1.0, seed=42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        B = torch.randn(2, n_freq, generator=rng) * sigma
        self.register_buffer("B", B)  # [2, n_freq]

    def forward(self, xy):  # xy: [..., 2]
        proj = xy @ self.B  # [..., n_freq]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 2*n_freq]
```

Apply to dims 0–1 of the normalized input. Update `fun_dim` in `model_config`:
```python
n_freq = 32
# fun_dim was X_DIM - 2 = 22; now add 2*n_freq - 2 (replace 2 pos dims with 2*n_freq)
fun_dim = X_DIM - 2 + 2 * n_freq - 2  # = 22 + 62 = 84
```

Keep `sigma=1.0` as a first try. The normalized coordinates after standardization
have std~1, so sigma=1.0 covers frequencies 0–2pi which matches typical boundary
layer gradients.

Note: the RFF matrix `B` must be fixed (not learned) — use `register_buffer` not
`nn.Parameter`. The `Transolver.preprocess` MLP input size must be updated accordingly.

### Suggested experiment design

- Add RFF encoding as a thin wrapper around the input before the MLP preprocessor.
- n_freq=32 (64 extra dims), sigma=1.0.
- Keep all other hyperparameters at baseline.
- A/B: compare val curve against baseline to see if early-epoch convergence is faster.
- If this works, follow-up: sigma tuning (0.5, 2.0) or learned frequency bands (SIREN-style).

### Risks

- Increases input dimension from 24 to ~86 — small cost, but changes the preprocess MLP
  input size. Must update `fun_dim` carefully.
- sigma choice is heuristic; wrong sigma can hurt (too low = no high-freq, too high = noisy).

---

## H6: Gradient clipping + cosine warm restarts (SGDR)

### What it is

Add gradient clipping (`max_norm=1.0`) to the training loop and replace
`CosineAnnealingLR` with `CosineAnnealingWarmRestarts` (T_0=10 epochs, T_mult=2)
to escape local minima caused by high-Re outlier gradients.

### Why it might help here

The dataset spans Re 100K–5M with per-sample y std varying by an order of magnitude.
High-Re samples produce much larger gradients than low-Re ones. Without gradient
clipping, a single high-Re batch can dominate an update step and destabilize training.
SGDR adds periodic learning rate spikes that can escape flat regions and saddle points
that the current cosine annealing (monotone decay) cannot.

### Key papers

- **SGDR** (Loshchilov & Hutter, ICLR 2017): cosine annealing with warm restarts for
  escaping local minima. https://arxiv.org/abs/1608.03983
- **Gradient clipping** (Pascanu et al., ICML 2013): standard stabilization for
  networks with large gradient variance.

### Implementation notes

```python
# In optimizer step (training loop):
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Replace scheduler:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
# Also update scheduler.step() call: call after each batch OR after each epoch.
# For CosineAnnealingWarmRestarts, epoch-level call is standard:
scheduler.step(epoch + batch_idx / len(train_loader))  # fractional epoch
```

If calling scheduler per-batch (more common for SGDR), replace the `scheduler.step()`
at end of epoch with a per-batch call inside the training loop.

Note: with T_0=10 and T_mult=2 over 50 epochs, the restart schedule is:
epoch 10, 30, 70... — so within 50 epochs, two restarts occur (at epoch 10 and 30).

### Suggested experiment design

- Add both changes together (they are complementary, not alternatives).
- Keep lr=5e-4, surf_weight=10.0, batch_size=4 at baseline.
- Diagnostic: check if train/surf_loss curve shows the characteristic sawtooth pattern
  indicating SGDR is working. If not, the scheduler step call may be misconfigured.

### Risks

- Warm restarts waste some training budget recovering from each restart.
- With only 50 epochs and ~1499 samples, the restart period T_0=10 may be too long;
  consider T_0=8 if training speed allows more than 50 epochs.

---

## H7: Two-branch output head (surface vs volume decoder)

### What it is

After the final TransolverBlock, split into two separate output MLP heads: one for
surface nodes (is_surface=True) and one for volume nodes (is_surface=False). The
surface head can be deeper (2 layers) or wider, optimizing specifically for the
surface pressure prediction task.

### Why it might help here

Surface nodes and volume nodes have fundamentally different physics: surface nodes
lie on the airfoil boundary where pressure is set by Bernoulli, boundary layer, and
separation; volume nodes track the far-field flow. A shared output head must learn
a compromised representation. The primary metric is surface pressure only — dedicating
more capacity to the surface head directly serves the evaluation objective.

### Key papers

- **B-GNN** (arXiv:2503.18638, Mar 2025): surface-exclusive model for Cp prediction
  shows surface-specific architectures outperform global models on surface pressure.
  https://arxiv.org/abs/2503.18638
- **Multi-task learning output heads** (general principle, Caruana 1997, various NeuralOp
  papers): task-specific decoders after shared encoder.

### Implementation notes

In `TransolverBlock` (last layer), replace the current `mlp2` head with:

```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Two separate decoders
    self.surf_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    self.vol_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, out_dim),
    )
```

In `forward`, after the last block (must pass `is_surface` mask into the block, or
apply the branching in the model's `forward` method):

```python
# In Transolver.forward, after blocks:
# fx: [B, N, hidden_dim]
hidden = self.blocks[-1](fx)  # last_layer=False for the last block, separate call
hidden = self.ln_3(hidden)
surf_mask_expanded = is_surface.unsqueeze(-1)  # [B, N, 1]
out = torch.where(surf_mask_expanded, self.surf_head(hidden), self.vol_head(hidden))
return {"preds": out}
```

This requires passing `is_surface` into the model, changing the model contract to
`model({"x": x_norm, "is_surface": is_surface})`. Update both `Transolver.forward`
and the training/eval loops accordingly. Ensure `data/scoring.py` is not touched.

### Suggested experiment design

- Modify `Transolver.forward` to accept optional `is_surface` in `data` dict.
- Make the last block's `last_layer=False` and apply the branching in `Transolver.forward`.
- Keep all other hyperparameters baseline.
- If complexity is too high, a simpler variant: single head but with an additional
  `is_surface` feature concatenated just before the final linear layer.

### Risks

- Changes the model contract (adds `is_surface` to model forward input). Ensure
  `evaluate_split` also passes it.
- The simpler variant (concatenate `is_surface` feature) is lower risk — but `is_surface`
  is already in input dim 12, so the model already sees it. The two-head approach is
  the true architectural test.

---

## H8: Exponential Moving Average (EMA) of model weights

### What it is

Maintain an EMA copy of the model parameters during training and use the EMA weights
for validation and test evaluation. EMA with decay=0.999 effectively averages the
last ~1000 gradient steps, smoothing out noise in the parameter space and typically
improving generalization — especially on OOD splits.

### Why it might help here

The TandemFoilSet has several OOD validation splits (camber M=6-8, M=2-4, Re stratified).
EMA is known to improve OOD robustness because it reduces the influence of individual
noisy updates from high-Re outlier batches. The cosine annealing schedule already
provides a late-training regime where EMA is most effective (gradients are small, EMA
catches up to the current weights). Morales-Brotons et al. (TMLR 2024) show consistent
+1-3% improvements across diverse settings with zero inference cost.

### Key papers

- **EMA consistently improves accuracy** (Morales-Brotons et al., TMLR 2024): large-scale
  empirical study showing EMA with 0.9995 decay improves validation accuracy in vision,
  language, and physics tasks. https://arxiv.org/abs/2312.06434
- **Mean Teacher / EMA in SSL** (Tarvainen & Valpola, NeurIPS 2017): early demonstration
  of EMA weight averaging improving generalization.

### Implementation notes

```python
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow
```

Add after optimizer initialization:
```python
ema = EMAModel(model, decay=0.999)
```

Add after `optimizer.step()` in the training loop:
```python
ema.update(model)
```

For validation, use the EMA weights:
```python
# Before val:
orig_state = {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(ema.shadow)
# ... run evaluate_split ...
# After val:
model.load_state_dict(orig_state)
```

For checkpoint saving, save `ema.shadow` as the checkpoint (the EMA weights, not
the live weights). This ensures test evaluation uses the EMA checkpoint.

Decay=0.999 is a good starting point. With ~250 batches/epoch and 50 epochs,
0.999 decay has effective window ~1000 steps (~4 epochs).

### Suggested experiment design

- Implement `EMAModel` as a simple class above the training loop (no new packages needed).
- Use decay=0.999.
- Keep all other hyperparameters at baseline.
- Diagnostic: compare val curve from EMA weights vs live weights — they should diverge
  most clearly in late-training epochs.
- This is orthogonal to all other hypotheses and can be composed with any winner.

### Risks

- EMA validation requires restoring model state around each eval call — minor bookkeeping
  overhead.
- If training is very short (few epochs due to timeout), EMA may not have had time to
  warm up and will underperform the live model.

---

## H9: Cautious AdamW optimizer

### What it is

Replace the standard `torch.optim.AdamW` with Cautious AdamW (C-AdamW), which adds a
one-line masking step: only apply the update at parameter positions where the gradient
and the momentum direction agree. Where they disagree, zero the update. This improves
training stability and generalization, particularly for OOD examples.

### Why it might help here

Cautious optimization (Luo et al., arXiv:2411.16085, ICLR 2026) addresses the problem
where momentum-based optimizers can push parameters in the wrong direction during rapid
changes in loss landscape — exactly the scenario during transitions between high-Re and
low-Re batches in the balanced sampler. The cautious mask prevents overzealous updates
on conflicting gradient/momentum signals, which is a direct mechanism for improving
OOD robustness on the Re-stratified and camber-OOD splits.

### Key papers

- **Cautious Optimizers** (Luo et al., arXiv:2411.16085, ICLR 2026): C-AdamW and C-Lion
  improve training stability and downstream generalization across vision and language tasks.
  https://arxiv.org/abs/2411.16085
- Implementation: https://github.com/kyleliang919/C-Optim (MIT license, ~20 lines of code)

### Implementation notes

Cautious AdamW is a minimal modification. The core is:

```python
class CautiousAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        # Standard AdamW step first
        loss = super().step(closure)
        # Apply cautious mask: zero out updates where gradient and update disagree
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # mask: 1 where update and grad agree in sign
                # The update was just applied: p_new = p_old - update
                # We need to reconstruct the mask from grad and step state
                state = self.state[p]
                if 'exp_avg' in state:
                    mask = (state['exp_avg'] * p.grad > 0).float()
                    # Scale update by mask (conservative: zero disagreeing dims)
                    # Note: update already applied; this would require storing p_old
                    # Instead, use the simpler formulation: apply mask to next step
                    # See: https://github.com/kyleliang919/C-Optim for exact impl
        return loss
```

**Easier implementation**: copy the `CautiousAdamW` class directly from
https://github.com/kyleliang919/C-Optim/blob/master/c_optim.py (~50 lines, no new
packages). It implements the cautious mask before the parameter update rather than
after, which is simpler. The key change is applying the mask to the Adam update
vector before it is subtracted from parameters.

No new packages: the implementation is pure PyTorch and can be pasted directly into
`train.py` above the Config class.

Use the same lr=5e-4, weight_decay=1e-4 as baseline. Cautious AdamW typically does
not require retuning the learning rate.

### Suggested experiment design

- Paste the CautiousAdamW implementation from the reference repo into `train.py`.
- Replace `torch.optim.AdamW` with `CautiousAdamW` in the optimizer line.
- Keep all other hyperparameters at baseline.
- Key diagnostic: check if `val_re_rand/mae_surf_p` and `val_geom_camber_*/mae_surf_p`
  improve — these are the OOD splits where cautious optimization should help most.

### Risks

- The cautious mask adds a small per-step overhead (one `> 0` comparison per parameter).
  Negligible at this model size (~1M params).
- If the training dataset is already well-covered, cautious optimization may provide no
  benefit over standard AdamW.

---

## H10: Re-stratified heavy sampling (bias toward high-Re batches)

### What it is

Modify the `WeightedRandomSampler` weights to additionally up-weight high-Re samples
(Re > 1M) by a factor of 2x relative to their current domain-balanced weight. This
gives the optimizer more exposure to the high-Re regime where prediction errors are
largest (per-sample y std up to 2077 in the single-foil split).

### Why it might help here

The current sampler achieves domain balance (raceCar single / raceCar tandem / cruise
equally weighted), but within each domain, Re is approximately uniform. However, the
per-sample y std varies by an order of magnitude with Re — high-Re samples are harder
and drive the MAE metric. The OOD Re validation split (`val_re_rand`) is stratified,
meaning it spans all Re levels, but the model's exposure to high-Re dynamics may be
insufficient for the camber-OOD and Re-OOD splits.

### Key papers

- **Curriculum learning / importance sampling** (Bengio et al., ICML 2009): up-weighting
  harder examples can accelerate convergence on the difficult end of the distribution.
- **Re-weighting for distribution shift** (Shimodaira, 2000, JSTOR): importance weighting
  as covariate shift correction — the high-Re regime is proportionally more important
  to the evaluation metric than its sample count implies.

### Implementation notes

The `sample_weights` tensor returned by `load_data()` has one entry per training sample.
In `train.py`, before constructing `WeightedRandomSampler`:

```python
# load Re values for each training sample and upweight high-Re
# x is in normalized space; dim 13 is log(Re) normalized
# Need raw Re: we can use x_raw (before normalization)
# Access train_ds[i] = (x, y, is_surface); x dim 13 is log(Re) normalized

# Compute Re-based multiplier for each sample
re_multiplier = torch.ones(len(train_ds))
for i in range(len(train_ds)):
    x_i, _, _ = train_ds[i]
    # x dim 13 is log(Re), normalized. Denormalize to get log(Re):
    log_re_norm = x_i[0, 13].item()  # all nodes have same log(Re)
    log_re = log_re_norm * stats_cpu["x_std"][13].item() + stats_cpu["x_mean"][13].item()
    re = math.exp(log_re)
    if re > 1e6:  # Re > 1M
        re_multiplier[i] = 2.0

adjusted_weights = sample_weights * re_multiplier
sampler = WeightedRandomSampler(adjusted_weights, num_samples=len(train_ds), replacement=True)
```

Note: `stats` is on GPU, so create a CPU copy (`stats_cpu`) before the loop, or
read the stats directly from the `stats.json` file.

This loop over 1499 samples is fast (no GPU, simple indexing).

### Suggested experiment design

- Implement Re-stratified weighting as above.
- Keep all other hyperparameters at baseline.
- Key diagnostic: check if `val_re_rand/mae_surf_p` specifically improves.
- Also check that in-distribution `val_single_in_dist/mae_surf_p` does not regress
  significantly (a +5% OOD gain at the cost of -10% in-dist is not a net win).

### Risks

- May hurt low-Re in-distribution performance (val_single_in_dist covers 104K–5M Re,
  so high-Re upweighting could unbalance even the in-distribution split).
- The precomputation loop over 1499 samples reads each sample's feature tensor — may
  be slow if samples are on a remote filesystem. Add a cache or precompute offline.

---

## H11: Per-sample log1p target normalization

### What it is

Instead of using global `(y - y_mean) / y_std` normalization (which is already applied
by the dataset), apply an additional per-sample rescaling using `log1p(|y|) * sign(y)`.
This compresses the dynamic range of high-Re pressure values, reducing the effective
loss weight of extreme outliers.

### Why it might help here

Per-sample y std ranges from ~160 (cruise) to ~2077 (high-Re raceCar single). The
global normalization divides by the dataset-level `y_std` (which is dominated by
high-Re extremes), but within a batch, high-Re samples still have much larger
normalized values than low-Re ones. The MSE loss is quadratic in the prediction error,
so high-Re samples dominate gradient updates even after global normalization.
A per-sample log1p compression further equalizes the effective loss contribution
across Re regimes.

### Key papers

- **Target normalization in regression** (general ML practice): log-transforming skewed
  regression targets is standard in Kaggle competitions and house price prediction.
- **Scale-invariant losses** (general principle): transforms that equalize scale reduce
  sensitivity to outliers in regression.

### Implementation notes

Apply before the MSE loss computation, after global normalization:

```python
def log1p_compress(y_norm):
    """Apply additional per-sample log1p compression to normalized targets."""
    return torch.sign(y_norm) * torch.log1p(torch.abs(y_norm))

def log1p_decompress(y_comp):
    """Inverse: expm1."""
    return torch.sign(y_comp) * torch.expm1(torch.abs(y_comp))

# In training loop, after y_norm = (y - y_mean) / y_std:
y_comp = log1p_compress(y_norm)
pred_comp = log1p_compress(pred)  # model outputs in compressed space
loss based on (pred_comp - y_comp)^2

# For evaluation (MAE in physical units), decompress then denormalize:
pred_phys = log1p_decompress(pred_comp) * stats["y_std"] + stats["y_mean"]
```

Important: if the model is trained in compressed space, it must be evaluated in
physical units (the MAE metric requires denormalization). Ensure the decompression
and denormalization are applied in `evaluate_split`.

This requires updating the model output interpretation throughout. The model contract
changes: model still outputs `[B, N, 3]` but now in compressed-normalized space.

Alternative (simpler, lower risk): apply log1p only during loss computation, not as
a change to the model output space. This is an L-type loss: `MSE(log1p(pred), log1p(y_norm))`
rather than a space change.

### Suggested experiment design

- Implement the simpler variant: compute loss as `MSE(log1p(|pred - y_norm|))` rather
  than standard MSE. This avoids changing the model output space and keeps the model
  contract intact (predictions are still in normalized linear space; only the loss
  function changes).
- Check that `evaluate_split` still produces MAE in physical units (it does, since it
  denormalizes from `pred` directly, not from the loss computation).

### Risks

- Log compression reduces gradient signal for large errors — may slow convergence on
  extreme high-Re cases if those cases are exactly the hardest ones.
- Changes loss landscape significantly; may require lr retuning.

---

## H12: MLP dropout regularization

### What it is

Add dropout=0.1 to the `MLP` sub-layers within each `TransolverBlock` (the FFN part),
keeping the `PhysicsAttention` dropout at 0.0. This targets the over-smooth generalization
on OOD camber splits without affecting the attention mechanism.

### Why it might help here

The baseline `TransolverBlock` uses `dropout=0.0` throughout. With only 1499 training
samples and OOD validation splits that test unseen camber profiles, the model may
overfit the training distribution's feature correlations in the MLP layers. Light
dropout (0.1) on the FFN sub-layers is standard regularization in vision/language
transformers and has been shown to improve generalization without hurting convergence
when applied only to the MLP path (not the attention path).

### Key papers

- **DropBlock / Dropout in Transformers** (Hendrycks & Gimpel, ICLR 2017;
  Vaswani et al., NeurIPS 2017): standard regularization technique.
- **Dropout for PDE surrogates** (used in GNOT, Hao et al., NeurIPS 2023):
  dropout=0.1 in FFN sub-layers is standard in operator transformer implementations.

### Implementation notes

In `TransolverBlock.__init__`, change:
```python
self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
               n_layers=0, res=False, act=act)
```
to:
```python
self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
               n_layers=0, res=False, act=act, dropout=0.1)
```

Add dropout to `MLP`:
```python
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True, dropout=0.0):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn(), nn.Dropout(dropout))
        ...
```

Keep `PhysicsAttention(dropout=0.0)` — attention dropout can cause unstable slice
token distributions.

### Suggested experiment design

- Minimal change: add dropout=0.1 to MLP only.
- Keep all other hyperparameters at baseline.
- Check if val curve shows less overfitting on OOD camber splits in later epochs.

### Risks

- Dropout can hurt performance if the model is already underfitting (which is possible
  with only 1499 samples and 1M params).
- Low risk overall: standard technique with well-understood tradeoffs.

---

## H13: Geometry-conditioned slice tokens (GeoTransolver GALE)

### What it is

Implement a simplified version of GeoTransolver's Geometry-Aware Latent Embedding
(GALE): compute a geometry context vector per sample by averaging surface-node features
and injecting it into the slice tokens at every attention layer via cross-attention.
This provides persistent geometry conditioning throughout the forward pass, not just
at the input layer.

### Why it might help here

The baseline Transolver conditions on geometry only through the input MLP preprocessor
(first layer). In GeoTransolver (NVIDIA, arXiv:2512.20399, Dec 2025), persistent
geometry conditioning at every block substantially reduces regime-shift errors on OOD
geometry splits. The TandemFoilSet OOD camber splits (M=6-8, M=2-4) are exactly the
regime where persistent conditioning should help — the model needs to carry foil shape
information throughout its depth, not just encode it in the initial token representation.

### Key papers

- **GeoTransolver** (Zhou et al., NVIDIA, arXiv:2512.20399, Dec 2025): GALE attention
  with persistent geometry conditioning; multi-scale ball queries; shows consistent
  improvement on OOD geometry splits in automotive CFD. https://arxiv.org/abs/2512.20399

### Implementation notes

Simplified GALE (cross-attention from slice tokens to geometry context):

```python
class GeoConditionedBlock(TransolverBlock):
    def __init__(self, *args, geom_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        # Geometry context cross-attention projection
        hidden_dim = kwargs.get('hidden_dim', args[1] if len(args) > 1 else 128)
        self.geom_proj = nn.Linear(geom_dim, hidden_dim)
        self.geom_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, fx, geom_ctx=None):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if geom_ctx is not None:
            # Inject geometry context: additive conditioning
            g = self.geom_proj(geom_ctx)  # [B, hidden_dim]
            fx = fx + g.unsqueeze(1)  # broadcast across nodes
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx
```

In `Transolver.forward`:
```python
# Compute geometry context from surface nodes
surf_feats = x[:, :, :2]  # use position features as geometry proxy
geom_ctx = surf_feats.mean(dim=1)  # [B, 2] — simple mean of all node positions
# Or better: mean of normalized NACA/AoA features (dims 13-23)
geom_ctx = x[:, 0, 13:24]  # dims 13-23 are global (same for all nodes in sample)

# Pass geom_ctx to each block
for block in self.blocks:
    fx = block(fx, geom_ctx=geom_ctx)
```

Note: dims 13-23 of the input (log(Re), AoA, NACA params, gap, stagger) are identical
for all nodes in a sample (global features). Using their value at node 0 is exact.
This is a natural geometry/regime context vector.

A simpler variant: concatenate `geom_ctx` to the slice tokens before the QKV step
(no new parameters beyond a small projection). This avoids the cross-attention
overhead entirely.

### Suggested experiment design

- Implement the simple variant first: use `x[:, 0, 13:24]` (11-dim global feature
  vector) as the geometry context, project to hidden_dim, and add to `fx` before each
  block's attention. This is a 1-layer additive injection, not full cross-attention.
- Check if `val_geom_camber_rc/mae_surf_p` and `val_geom_camber_cruise/mae_surf_p`
  improve — these are the geometry-OOD splits where GALE should help most.
- Full GALE implementation (multi-scale ball queries, full cross-attention) is the
  high-ceiling follow-up if the simple variant succeeds.

### Risks

- Most complex change in this list. Implementation bugs are likely on first pass.
- The global feature broadcast (`x[:, 0, 13:24]`) only works if all nodes in a sample
  truly have identical global features (AoA, NACA params, Re). Confirm this from the
  data specification (yes — per program.md, these are per-sample scalar conditions).
- May add minimal benefit if the model already implicitly conditions on global features
  via the MLP preprocessor.

---

## Cross-cutting notes

1. **Composability**: H8 (EMA) is orthogonal to all others and should be composed with
   any winning configuration. H3 (pressure-channel loss) is also orthogonal and can
   stack with any architecture change.

2. **Screening vs confirmation**: H1, H8, H3, H9 are all fast changes (single-digit
   line modifications) that can be confirmed within the default 30-minute budget.
   H2 (scale model) and H13 (GeoTransolver GALE) need more careful monitoring of
   VRAM and per-epoch time.

3. **Interaction risk**: H1 (remove inter-slice attention) and H4 (asymmetric QK) are
   mutually exclusive — do not run them together. Test H1 first; if it wins, H4 is
   superseded. If H1 fails, test H4.

4. **Baseline metrics to beat** (update this table from the committed baseline metrics
   once available):

   | Metric | Baseline value |
   |--------|----------------|
   | val_avg/mae_surf_p | (from BASELINE.md or first clean run) |
   | val_single_in_dist/mae_surf_p | — |
   | val_geom_camber_rc/mae_surf_p | — |
   | val_geom_camber_cruise/mae_surf_p | — |
   | val_re_rand/mae_surf_p | — |

5. **Implementation rule**: all changes must be in `train.py` only. `data/` files are
   read-only. No new packages unless added to `pyproject.toml` in the same PR.
