<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 2 Research Ideas — 2026-05-12

Hypotheses for the second wave of experiments on TandemFoilSet.
Round 1 axes (L1 loss, surf_weight 30, wider/deeper Transolver, slice_num 128,
lr warmup, EMA, unified_pos, SwiGLU) are explicitly excluded.

Ranked approximately by expected impact-per-implementation-cost (highest first).

---

## H1 — Ada-Temp: Per-point adaptive slice temperature

**Rationale.**
The current `self.temperature` is a single scalar `[1, heads, 1, 1]` shared
across all points.  Transolver++ (arXiv 2502.02414) identifies this as a root
cause of "homogeneous physical state" collapse: all nodes get nearly equal slice
weights, degenerating toward average pooling.  The fix is a lightweight additive
correction: `τᵢ = τ₀ + Δτᵢ` where `Δτᵢ = Linear(dim, heads)(xᵢ)` is
per-point and per-head.  Points near the foil surface (where pressure gradients
are sharp) will learn to sharpen their slice distribution; far-field nodes will
retain broader averaging.  Transolver++ reports consistent gains on Darcy,
Navier-Stokes, and elasticity benchmarks with this change alone.

**Concrete code change in `train.py`.**

In `PhysicsAttention.__init__`, add one extra projection:

```python
# BEFORE
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

# AFTER
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
self.temp_proj = nn.Linear(dim, heads)   # <-- new
nn.init.zeros_(self.temp_proj.weight)
nn.init.zeros_(self.temp_proj.bias)
```

In `PhysicsAttention.forward`, replace the slice-weight line:

```python
# BEFORE
slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)

# AFTER
# x: [B, N, dim]; need per-node temperature correction
delta_temp = self.temp_proj(x)           # [B, N, heads]
delta_temp = delta_temp.permute(0, 2, 1).unsqueeze(-1)   # [B, heads, N, 1]
local_temp = (self.temperature + delta_temp).clamp(min=1e-3)
slice_weights = self.softmax(self.in_project_slice(x_mid) / local_temp)
```

No other changes.  This adds `dim * heads` = `128 * 4 = 512` parameters, negligible.
Zero-init of `temp_proj` means training starts identically to baseline.

**Expected direction on `val_avg/mae_surf_p`:** decrease (improvement).
Mechanism: sharper slice assignments on boundary-layer nodes give the model
better resolution at pressure peaks that dominate `mae_surf_p`.

**Risk.**
Low.  Zero-init guarantees identical initialization.  The only failure mode is
that the delta collapses to near-zero throughout training (slice collapse was
the original problem, not slice sharpening).  Check `delta_temp.abs().mean()`
as a diagnostic; if it stays near zero, the bottleneck is elsewhere.

---

## H2 — Asymmetric Q/K projections (LinearNO simplification)

**Rationale.**
LinearNO (arXiv 2511.06294) shows that PhysicsAttention's performance comes
almost entirely from the slice/deslice operations, NOT from Q/K/V attention
between slice tokens.  More importantly, it proves that the symmetric design
(`to_q` and `to_k` both act on the same slice_token via the same
`in_project_slice`) limits the diversity of slices the model can learn.
Making `to_q` and `to_k` operate on independently projected representations
improves both slice utilization and final metrics.  LinearNO reports 36% less
compute with equal or better results; here we want only the asymmetric Q/K gain
without full ablation of the attention.

**Concrete code change in `train.py`.**

In `PhysicsAttention.__init__`:

```python
# BEFORE
self.in_project_x = nn.Linear(dim, inner_dim)
self.in_project_fx = nn.Linear(dim, inner_dim)
self.in_project_slice = nn.Linear(dim_head, slice_num)
torch.nn.init.orthogonal_(self.in_project_slice.weight)

# AFTER
self.in_project_x = nn.Linear(dim, inner_dim)
self.in_project_fx = nn.Linear(dim, inner_dim)
self.in_project_slice = nn.Linear(dim_head, slice_num)   # used for value pooling
self.in_project_slice_k = nn.Linear(dim_head, slice_num)  # <-- new, for key path
torch.nn.init.orthogonal_(self.in_project_slice.weight)
torch.nn.init.orthogonal_(self.in_project_slice_k.weight)
```

In `PhysicsAttention.forward`:

```python
# BEFORE
slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
slice_norm = slice_weights.sum(2)
slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
q = self.to_q(slice_token)
k = self.to_k(slice_token)

# AFTER
slice_weights_v = self.softmax(self.in_project_slice(x_mid) / self.temperature)   # for pooling
slice_weights_k = self.softmax(self.in_project_slice_k(x_mid) / self.temperature) # for keys
slice_norm = slice_weights_v.sum(2)
slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_v)
slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
slice_token_k = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_k)
slice_norm_k = slice_weights_k.sum(2)
slice_token_k = slice_token_k / ((slice_norm_k + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
q = self.to_q(slice_token)
k = self.to_k(slice_token_k)   # <-- different input
```

The deslice step at the end still uses `slice_weights_v`.

**Expected direction on `val_avg/mae_surf_p`:** decrease.
Mechanism: richer per-slice representations allow more discriminative
attention between physical states (surface boundary layer vs. wake vs. far field).

**Risk.**
Medium.  Adds `dim_head * slice_num = 32 * 64 = 2048` parameters per attention
layer (5 layers × 4 heads = marginal).  Divergence risk is low since
orthogonal init is preserved.  The failure mode is that the second projection
learns the same thing as the first, giving no gain.

---

## H3 — Remove `in_project_fx` (Transolver++ memory + performance fix)

**Rationale.**
Transolver++ (arXiv 2502.02414) demonstrates that `in_project_fx` is
redundant: the model projects the same input `x` twice
(`in_project_x` for slicing, `in_project_fx` for value pooling), but a single
projection suffices.  Removing `in_project_fx` and re-using `x_mid` as `fx_mid`
halves the memory of that step, opens headroom for larger batch sizes or
architectures, and in Transolver++ leads to no performance degradation
(sometimes improvement) because the tied projection reduces overfitting.
This is a clean ablation with a clear mechanism.

**Concrete code change in `train.py`.**

In `PhysicsAttention.__init__`, remove `in_project_fx`:

```python
# REMOVE this line:
# self.in_project_fx = nn.Linear(dim, inner_dim)
```

In `PhysicsAttention.forward`, remove the `fx_mid` computation:

```python
# REMOVE these lines:
# fx_mid = (
#     self.in_project_fx(x)
#     .reshape(B, N, self.heads, self.dim_head)
#     .permute(0, 2, 1, 3)
#     .contiguous()
# )

# Then change all references to fx_mid -> x_mid:
slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)   # was fx_mid
```

One linear layer removed per attention block: saves `dim * inner_dim` =
`128 * (32 * 4)` = 524,288 parameters per layer across 5 layers (~2.6M params
freed, ~30% of the attention cost).

**Expected direction on `val_avg/mae_surf_p`:** neutral to small decrease.
Mechanism: reduced overfitting from tied projections; more VRAM headroom
to increase batch size or switch to bf16.

**Risk.**
Low.  Strictly subtractive change.  Known to be safe from Transolver++.
The primary risk is a small regression if the dual projections genuinely add
expressivity on this dataset — monitorable in 5 epochs.

---

## H4 — Per-channel loss weighting: up-weight pressure channel

**Rationale.**
The training loss is `MSE(pred, y_norm)` summed equally over all 3 output
channels (Ux, Uy, p).  But the ranking metric `mae_surf_p` cares only about
pressure.  Normalised pressure variance may be lower than velocity variance,
causing the optimizer to ignore it.  A simple fix: multiply the loss on channel
2 (pressure) by a constant `p_weight > 1` so gradient signal is proportionally
reallocated toward the target metric.  This is the task-alignment version of
reweighted multi-task learning, with no added complexity.

**Concrete code change in `train.py`.**

In the training loop, after `sq_err = (pred - y_norm) ** 2`:

```python
# BEFORE
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# AFTER
P_WEIGHT = 3.0   # tunable; start at 3.0
channel_weights = sq_err.new_tensor([1.0, 1.0, P_WEIGHT])   # [3]
weighted_sq_err = sq_err * channel_weights[None, None, :]
vol_loss  = (weighted_sq_err * vol_mask.unsqueeze(-1)).sum()  / vol_mask.sum().clamp(min=1)
surf_loss = (weighted_sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
```

Also apply the same weighting in `evaluate_split` for monitoring consistency
(the physical MAE is unaffected since it operates on denormalized predictions).

**Expected direction on `val_avg/mae_surf_p`:** decrease; possible slight increase
on velocity MAEs as a tradeoff.

**Risk.**
Low.  Zero-complexity change.  P_WEIGHT=3.0 is a reasonable starting point;
if it hurts, try 2.0 or 5.0.  Failure mode: pressure channel is already
well-converged and upweighting merely slows velocity convergence without helping
pressure — visible within 5 epochs.

---

## H5 — Gradient clipping (norm=1.0)

**Rationale.**
The current training loop has no gradient clipping.  Large meshes with sharp
boundary layers can produce occasional gradient spikes that destabilize AdamW,
especially at the beginning of training.  `clip_grad_norm_(..., 1.0)` is
universal practice in transformer training (it appears in virtually every
modern training setup).  On its own it will rarely improve the floor but should
reduce variance across seeds and smoothen convergence, giving experiments with
30-minute budgets a better chance of reaching a good checkpoint.

**Concrete code change in `train.py`.**

After `loss.backward()`, before `optimizer.step()`:

```python
# BEFORE
optimizer.zero_grad()
loss.backward()
optimizer.step()

# AFTER
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Expected direction on `val_avg/mae_surf_p`:** neutral to small decrease.
Mechanism: reduced variance in gradient updates, more stable convergence.

**Risk.**
Very low.  Can only help or be neutral.  The only real failure mode is that
max_norm=1.0 is too aggressive and effectively reduces the learning rate —
use 5.0 or 10.0 as fallback.  Because this interacts with the lr hypothesis in
Round 1 (lr=1e-3 + warmup), this should be run against the baseline lr=5e-4,
not the Round 1 high-lr variant.

---

## H6 — Uncertainty-weighted vol/surf/channel balance (Kendall-style)

**Rationale.**
The current loss `vol_loss + 10 * surf_loss` uses a manually tuned
`surf_weight=10` (Round 1 tests 30).  Kendall et al. (NeurIPS 2018) show that
homoscedastic uncertainty weighting auto-tunes multi-task balancing:
`L = Σᵢ (lᵢ / (2σᵢ²)) + log(σᵢ)`.  Here: three tasks = {vol, surf, vol_p_channel_bonus}
or simpler: two tasks = {vol, surf}.  Learning `σ_vol` and `σ_surf` removes
one major design decision (surf_weight) and may generalize better across the
three domain shifts in validation.  This targets the objective mismatch
hypothesis directly.

**Concrete code change in `train.py`.**

Above the model definition, after Config is parsed:

```python
# Add two learnable log-sigma parameters
log_sigma_vol  = nn.Parameter(torch.zeros(1, device=device))
log_sigma_surf = nn.Parameter(torch.zeros(1, device=device))
# Add to optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) + [log_sigma_vol, log_sigma_surf],
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)
```

In the training loop, replace the loss:

```python
# BEFORE
loss = vol_loss + cfg.surf_weight * surf_loss

# AFTER
# Kendall et al. homoscedastic uncertainty
precision_vol  = torch.exp(-2 * log_sigma_vol)   # = 1/σ²
precision_surf = torch.exp(-2 * log_sigma_surf)
loss = precision_vol * vol_loss + log_sigma_vol + precision_surf * surf_loss + log_sigma_surf
```

Log-sigma initialized at 0 → σ=1 → equivalent to `surf_weight=1` at init.
Training will learn the optimal balance.

**Expected direction on `val_avg/mae_surf_p`:** decrease (improvement on OOD splits,
where a fixed surf_weight=10 may be either too large or too small).

**Risk.**
Medium.  The sigma parameters are global; if the dataset's vol/surf ratio is
stable, they will converge to something close to the original weights.  If
they diverge (one sigma → ∞ kills that task), add a clamp:
`log_sigma_vol.data.clamp_(-3, 3)`.  This must NOT be combined with a changed
surf_weight in the same run — test against surf_weight=10 baseline.

---

## H7 — Fourier positional encoding for (x, z) coordinates

**Rationale.**
The raw node coordinates (dims 0-1) are included directly in the 24-dim
feature vector.  Neural networks have a spectral bias toward low-frequency
functions; sharp gradients in the boundary layer (high-frequency in space)
are systematically underfitted.  Mildenhall et al. (NeRF, NeurIPS 2020) and
Tancik et al. (NeurIPS 2020) show that mapping coordinates through
`[sin(2^k π x), cos(2^k π x)]` for k=0..K-1 before feeding into an MLP
recovers high-frequency detail.  The mesh already contains `dsdf` features
(dims 4-11) as shape descriptors, but those encode proximity to geometry, not
spatial position.  Replacing dims 0-1 with 2L Fourier features (e.g., L=4
frequencies → 8 dims) increases `fun_dim` and lets the model resolve
fine-scale pressure variation near stagnation points.

**Concrete code change in `train.py`.**

Add a Fourier feature transform in the model's `preprocess` step.  The cleanest
approach is to add a `FourierCoordEnc` module and apply it before normalization
(or after, since sin/cos are invariant to scaling):

```python
class FourierCoordEnc(nn.Module):
    """Replace (x, z) coords at dims 0-1 with Fourier features."""
    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        # Fixed frequency schedule: 2^0 ... 2^(n_freqs-1)
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)  # [n_freqs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 24]
        coords = x[..., :2]          # [B, N, 2]
        # [B, N, 2, n_freqs]
        angles = coords.unsqueeze(-1) * self.freqs[None, None, None, :] * torch.pi
        # [B, N, 2*n_freqs*2] = [B, N, 16]
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        feats = feats.reshape(*x.shape[:-1], 2 * 2 * self.n_freqs)
        # Replace original coords with Fourier features; keep dims 2-23 intact
        return torch.cat([feats, x[..., 2:]], dim=-1)   # [B, N, 16+22=38]
```

In `Transolver.__init__`, update `fun_dim`:
The model currently uses `fun_dim = X_DIM - 2 = 22` (everything except
the 2 raw position dims, which go through the `unified_pos` path).
With Fourier encoding of 4 freqs, the encoded position takes 16 dims,
so `fun_dim` rises to `16 + 22 = 38 - 2 = 36`... careful: the Transolver
`preprocess` MLP takes `fun_dim + space_dim` where `space_dim=2`.

Simpler implementation: treat the Fourier expansion as a preprocessing step
that replaces dims 0-1 with 16 dims, giving input size 38.  Update
`model_config["fun_dim"]` from 22 to 36 (since `space_dim` remains 2 and
`preprocess` takes `fun_dim + space_dim = 36 + 2 = 38`).

Apply `FourierCoordEnc(n_freqs=4)` to `x` in the training loop before
normalization (or inside the model).  Re-compute `x_mean`/`x_std` — but since
the stats come from `stats.json` which is fixed, the simplest approach is to
apply the encoding AFTER normalization (normalise the original 24 dims,
then replace dims 0-1 with Fourier features of the normalized coordinates).

**Implementation note:** adjust `model_config["fun_dim"]` to match the new
expanded input.  The `X_DIM` constant from `data/` refers to the raw 24
dimensions; override `fun_dim` in model_config explicitly.

**Expected direction on `val_avg/mae_surf_p`:** moderate decrease on OOD geometry
splits (camber holdouts), where high-frequency spatial detail near the
second foil matters most.

**Risk.**
Medium.  The main risk is a dimension mismatch if the encoding is applied
at the wrong step in the pipeline.  The `dsdf` features (dims 4-11) already
provide some spatial proximity signal, so the gain may be modest.

---

## H8 — Stochastic depth (layer drop) for regularization

**Rationale.**
Stochastic depth (Huang et al., ECCV 2016) randomly drops entire transformer
blocks during training with probability increasing with depth.  It acts as an
implicit ensemble of shallower networks, provides strong regularization, and
is standard in modern ViT training.  With 1499 training samples and a 5-layer
Transolver, overfitting is a genuine risk on the OOD geometry splits
(val_geom_camber_rc, val_geom_camber_cruise).  Stochastic depth with
`drop_rate=0.1` (10% drop at deepest layer) is a near-zero-cost regularizer
requiring only a single forward-pass branch.

**Concrete code change in `train.py`.**

In `TransolverBlock.__init__`:

```python
def __init__(self, ..., stoch_depth_prob: float = 0.0):
    ...
    self.stoch_depth_prob = stoch_depth_prob
```

In `TransolverBlock.forward`:

```python
def forward(self, fx):
    if self.training and self.stoch_depth_prob > 0.0:
        if torch.rand(1).item() < self.stoch_depth_prob:
            return fx   # skip this block
    fx = self.attn(self.ln_1(fx)) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    if self.last_layer:
        return self.mlp2(self.ln_3(fx))
    return fx
```

In `Transolver.__init__`, assign linearly increasing drop rates:

```python
self.blocks = nn.ModuleList([
    TransolverBlock(
        ...,
        stoch_depth_prob=drop_rate * (i / (n_layers - 1)),   # 0 for first layer
    )
    for i in range(n_layers)
])
```

With `drop_rate=0.1` and 5 layers: per-block drop probs = [0.0, 0.025, 0.05, 0.075, 0.10].

**Expected direction on `val_avg/mae_surf_p`:** small decrease, primarily on OOD
geometry splits where regularization matters most.

**Risk.**
Low.  Stochastic depth is a pure training-time regularizer (disabled at eval).
Risk is that it slows convergence within the 30-min budget.

---

## H9 — Gumbel-Softmax slice weights (hard discrete slices)

**Rationale.**
Transolver++ (arXiv 2502.02414) proposes Gumbel-Softmax reparameterization
as an alternative to standard Softmax for slice weights.  The core motivation:
Gumbel-Softmax with a temperature schedule starting at `tau=1` and annealing
to `tau=0.1` produces harder, more discrete assignments, mitigating slice
collapse without the Ada-Temp overhead.  The training signal is more structured
because gradients flow through a sparse assignment rather than a diffuse average.
This is an alternative to H1 (Ada-Temp) — both target the same failure mode
(slice collapse) via different mechanisms.

**Concrete code change in `train.py`.**

In `PhysicsAttention.forward`, replace the slice_weights computation:

```python
# BEFORE
slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)

# AFTER
logits = self.in_project_slice(x_mid) / self.temperature.clamp(min=1e-3)
if self.training:
    slice_weights = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
else:
    slice_weights = self.softmax(logits)   # standard softmax at eval
```

No architecture change; `F.gumbel_softmax` is already in `torch.nn.functional`.

**Temperature schedule** (optional enhancement): anneal `tau` from 1.0 to 0.1
over training.  In the training loop, after `scheduler.step()`:

```python
gumbel_tau = max(0.1, 1.0 * (0.1 ** (epoch / MAX_EPOCHS)))
for block in model.blocks:
    if hasattr(block.attn, '_gumbel_tau'):
        block.attn._gumbel_tau = gumbel_tau
```

For the simple version, fixed `tau=1.0` is sufficient to test the mechanism.

**Expected direction on `val_avg/mae_surf_p`:** decrease, particularly on
in-distribution samples where slice assignment quality directly affects fit.

**Risk.**
Low-Medium.  `F.gumbel_softmax` adds stochastic noise at train time, which can
cause training loss to appear noisy.  The evaluation path (standard softmax)
is deterministic.  The main risk is that Gumbel noise prevents convergence
within the 30-minute budget.  Use `hard=False` (soft relaxation) to reduce noise.

---

## H10 — FiLM conditioning on global flow parameters (Re, AoA, NACA)

**Rationale.**
The current model receives all 24 input features via the `preprocess` MLP and
treats them uniformly.  Physics tells us that global flow parameters (Re,
AoA1, AoA2, NACA1, NACA2) determine the bulk flow regime; local node features
(position, sdf) determine how that regime manifests at each point.  FiLM
(Perez et al., AAAI 2018) proposes separating these: extract global
conditioning `z` from the flow parameters and modulate hidden states at each
transformer block via `γ(z) * hᵢ + β(z)`.  This is standard practice in
physics-informed neural operators when global vs. local conditioning can be
cleanly separated.  Concretely: `z = [log(Re), AoA1, AoA2, NACA1, NACA2, gap, stagger]`
— 11 global dims from the 24-dim input.  This should improve Re and geometry
generalization by decoupling global flow regime from local mesh structure.

**Concrete code change in `train.py`.**

```python
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj  = nn.Linear(cond_dim, hidden_dim)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: [B, N, hidden_dim], z: [B, cond_dim]
        gamma = self.gamma_proj(z).unsqueeze(1)   # [B, 1, hidden_dim]
        beta  = self.beta_proj(z).unsqueeze(1)    # [B, 1, hidden_dim]
        return gamma * h + beta
```

In `TransolverBlock.__init__`, add optional `FiLMLayer`:

```python
def __init__(self, ..., cond_dim: int = 0):
    ...
    self.film = FiLMLayer(cond_dim, hidden_dim) if cond_dim > 0 else None
```

In `TransolverBlock.forward`:

```python
def forward(self, fx, z=None):
    fx = self.attn(self.ln_1(fx)) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    if self.film is not None and z is not None:
        fx = self.film(fx, z)
    ...
```

In `Transolver.forward`, extract the global conditioning vector from `x`:

```python
# Global flow params: dims 13-23 (log Re, AoA1, NACA1*3, AoA2, NACA2*3, gap, stagger)
# Use mean-pooling over nodes to get a single [B, 11] vector
z = x[:, :, 13:24].mean(dim=1)   # [B, 11], same value for all nodes in a sample
for block in self.blocks:
    fx = block(fx, z)
```

Update `Transolver.__init__` to pass `cond_dim=11` to all blocks except last
(or all blocks).

**Expected direction on `val_avg/mae_surf_p`:** moderate decrease on OOD Re and
geometry splits.

**Risk.**
Medium.  FiLM adds `2 * cond_dim * hidden_dim` parameters per block
(2 * 11 * 128 * 5 = ~14K params — negligible).  The main implementation risk is
a signature change to `TransolverBlock.forward` that must be propagated
consistently.  Risk of regression is low since `gamma_proj` is init to ones
(identity) and `beta_proj` to zeros (no shift) so training starts identically.

---

## H11 — Log1p target space: partial log-scale for extreme high-Re values

**Rationale.**
Within every split, high-Re samples drive the target extremes: `val_single_in_dist`
has p values in `(-29,136, +2,692)` and per-sample y std up to 2,077.  MAE in
physical units is therefore dominated by a small fraction of high-Re samples
with large absolute errors.  Log1p scaling (`sign(y) * log(1 + |y|)`) compresses
large values and expands small ones, giving the optimizer a more uniform loss
landscape.  At eval/test time, predictions are un-transformed (`sign(p) * (exp(|p|) - 1)`)
before MAE is computed so the primary metric is unaffected.  This is a
target-space reparameterization that requires only two extra lines.

**Concrete code change in `train.py`.**

After computing `y_norm`, add a signed log1p transform:

```python
# BEFORE (train loop and evaluate_split)
y_norm = (y - stats["y_mean"]) / stats["y_std"]
pred = model({"x": x_norm})["preds"]
sq_err = (pred - y_norm) ** 2

# AFTER
y_norm = (y - stats["y_mean"]) / stats["y_std"]
# Signed log1p in normalized space
y_train = torch.sign(y_norm) * torch.log1p(y_norm.abs())
pred = model({"x": x_norm})["preds"]   # model still predicts in log1p space
# Inverse transform for MAE
pred_norm = torch.sign(pred) * (torch.expm1(pred.abs()))
pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
sq_err = (pred - y_train) ** 2   # loss in log1p space
```

Note: `evaluate_split` must apply the same inverse transform.  The model
contract (predict in normalized space) is preserved at a higher level — this
transform sits between normalization and the model without changing the I/O
signature from the organizer's perspective.

**Expected direction on `val_avg/mae_surf_p`:** directionally unclear —
may help low-Re samples, may hurt high-Re samples where MAE is largest.
Specifically, this hypothesis predicts a lower std across val samples
(less variance in per-sample MAE) but not necessarily a lower mean.

**Risk.**
Medium-High.  Target reparameterization must be applied consistently in both
training and eval paths.  Any mismatch between train and evaluate_split will
silently produce wrong MAEs.  Test with `--debug` first to verify the
inverse is applied correctly.  Consider running a 2-epoch diagnostic to
confirm `mae_surf_p` is computed in original units.

---

## Summary table (ranked by expected impact / implementation cost)

| Rank | ID | Title | Impact | Cost | Risk |
|------|----|-------|--------|------|------|
| 1 | H1 | Ada-Temp per-point temperature | High | Very Low | Low |
| 2 | H4 | Per-channel loss weight for p | High | Very Low | Low |
| 3 | H5 | Gradient clipping (norm=1.0) | Medium | Trivial | Very Low |
| 4 | H2 | Asymmetric Q/K projections | Medium-High | Low | Medium |
| 5 | H6 | Uncertainty-weighted vol/surf | Medium | Low | Medium |
| 6 | H10 | FiLM conditioning on global flow params | Medium | Medium | Medium |
| 7 | H3 | Remove in_project_fx (memory saving) | Medium | Low | Low |
| 8 | H9 | Gumbel-Softmax slice weights | Medium | Trivial | Low-Med |
| 9 | H7 | Fourier coordinate features | Medium | Medium | Medium |
| 10 | H8 | Stochastic depth (layer drop) | Low-Med | Low | Low |
| 11 | H11 | Log1p target space | Unclear | Medium | Med-High |

**Top 5 for immediate assignment:** H1, H4, H5, H2, H6.
H10 (FiLM) is the highest-ceiling idea but requires the most surgical code change.
H3 is worth running as a diagnostic in parallel with H1 (if H3 succeeds, it opens VRAM
for a larger architecture in round 3).
