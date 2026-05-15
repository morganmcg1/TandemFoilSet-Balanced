<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — Round 5 (2026-05-15)

Fresh hypotheses for TandemFoilSet. This is round 5 on the `icml-appendix-charlie-pai2i-48h-r5`
branch, a clean slate with no prior experiment results. All ideas are grounded in
2023–2026 literature and are designed to be implemented entirely in `train.py`.

Primary metric: `val_avg/mae_surf_p` (lower is better).
Baseline: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, surf_weight=10`.
No EMA, no grad clipping, no mixed precision in baseline.

---

## Ranking summary (impact / effort)

| Rank | Slug | Category | Predicted val_avg/mae_surf_p delta | Complexity |
|------|------|----------|------------------------------------|------------|
| 1 | `per-sample-scale-normalization` | Training strategy | -8 to -15% | 2 |
| 2 | `surface-volume-branch-head` | Architecture | -6 to -12% | 3 |
| 3 | `gradient-clipping-heavy-tail` | Training stability | -4 to -10% | 1 |
| 4 | `log-mae-loss-pressure` | Loss formulation | -5 to -10% | 2 |
| 5 | `multiscale-slice-attention` | Architecture | -5 to -10% | 3 |
| 6 | `ema-validation-checkpoint` | Training strategy | -3 to -7% | 1 |
| 7 | `rope-2d-positional-encoding` | Feature engineering | -3 to -7% | 2 |
| 8 | `physics-input-boundary-layer` | Feature engineering | -4 to -8% | 3 |
| 9 | `domain-conditioned-moe-slices` | Architecture | -4 to -8% | 4 |
| 10 | `homoscedastic-uncertainty-weighting` | Loss formulation | -3 to -6% | 2 |
| 11 | `surf-vol-decoupled-loss-schedule` | Loss formulation | -2 to -5% | 1 |
| 12 | `re-stratified-curriculum` | Training strategy | -2 to -5% | 2 |
| 13 | `stochastic-depth-transolver` | Architecture | -2 to -4% | 2 |

---

## 1. Per-sample scale normalization (z-score per sample at forward time)

**Category:** Training strategy / data representation

**What it is.** At inference time, normalize each sample's input features by that
sample's own statistics (mean, std) rather than the global dataset statistics.
The global stats in `stats.json` fold together three domains with very different
pressure magnitudes (max per-sample y-std ranges from 506 in cruise to 2,077 in
raceCar single). A high-Re raceCar sample and a low-Re cruise sample occupy the
same normalized space despite having order-of-magnitude different absolute scales,
making the MSE loss meaningless as a common unit.

**Why it might help.** The core bottleneck is the training loss treating all nodes
uniformly regardless of the dynamic range of the sample they belong to. A high-Re
sample's squared error in normalized space is ~100× larger than a low-Re sample's,
so the optimizer focuses on high-Re regime regardless of surf_weight. Per-sample
normalization (also called instance normalization) forces the model to learn the
flow _shape_ independently of scale, then the prediction is rescaled back. This
matches the approach used in FNO and several follow-up operator learning papers for
multi-regime datasets.

**Specific change to train.py.** Inside the training loop, after loading `y`, compute
per-sample scale stats and normalize:

```python
# In the training loop, after: y = y.to(device, non_blocking=True)
# Compute per-sample y scale (mean over real nodes, std over real nodes)
with torch.no_grad():
    valid_y = y * mask.unsqueeze(-1)  # zero out padding
    n_real = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B,1,1]
    ps_mean = valid_y.sum(dim=1, keepdim=True) / n_real  # [B,1,3]
    ps_var = ((valid_y - ps_mean * mask.unsqueeze(-1))**2).sum(dim=1, keepdim=True) / n_real
    ps_std = ps_var.sqrt().clamp(min=1e-3)  # [B,1,3]

# Replace global normalization with per-sample normalization for loss
y_ps_norm = (y - ps_mean) / ps_std  # [B,N,3]
pred_ps_norm = (pred * stats["y_std"] + stats["y_mean"] - ps_mean) / ps_std
# Compute loss in per-sample normalized space
sq_err = (pred_ps_norm - y_ps_norm) ** 2
```

The model still predicts in global normalized space (contract unchanged). Only the
loss computation switches to per-sample scale. Validation and test evaluation remain
unchanged (uses global stats as required by data/scoring.py).

**Alternative simpler variant.** Per-sample scale-only normalization (divide by
per-sample std, keep global mean subtraction). This is less invasive and avoids
the mean shift complication. Start with this.

**Risk.** The model output contract is global normalized, so ps-norm of the pred
requires the denorm→renorm step shown above. A bug here will silently change the
loss scale without changing the metric. Add an assertion that `ps_std.min() > 1e-4`.
Also watch for training instability in early epochs when ps_std estimates are noisy
on the first few batches.

**Papers:** FNO (Li et al., 2021) uses per-sample normalization in their ablations.
Raonic et al. (2024, "Convolutional Neural Operators") show explicit benefit for
multi-regime datasets. In Kaggle-style competitions for fluid field prediction
(e.g., OpenFoam Kaggle challenges), per-sample std normalization is a standard
first move.

**Complexity:** 2/5. ~20 lines in the training loop.

---

## 2. Separate surface and volume decoder heads

**Category:** Architecture

**What it is.** Replace the single final MLP head in `TransolverBlock` with two
separate decoder heads: one for surface nodes and one for volume nodes. Both
receive the same trunk features from all layers; only the last projection differs.

**Why it might help.** Surface nodes and volume nodes are physically distinct:
surface nodes lie on the foil boundary and carry the no-slip condition and
pressure gradient, while volume nodes carry the free-stream or wake-influenced
velocity. The current single head must learn a single mapping from the same
feature space to both regimes. The loss already distinguishes them (surf_weight=10),
but the model cannot specialize. AB-UPT (TMLR 2025, "Anchor-Based Universal Physics
Transformer") showed that a multi-branch operator for surface vs. volume reduced
automotive CFD error by 18–34% on DrivAerNet++. The B-GNN (2503.18638, 2025) goes
further: using boundary-only representations reduces model size 83% while matching
full-field accuracy, confirming that surface predictions benefit from surface-specific
inductive biases.

**Specific change to train.py.** In `TransolverBlock.__init__` for the last layer:

```python
if self.last_layer:
    self.ln_3_surf = nn.LayerNorm(hidden_dim)
    self.ln_3_vol  = nn.LayerNorm(hidden_dim)
    self.head_surf = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    self.head_vol = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
```

In `TransolverBlock.forward` for the last layer:

```python
if self.last_layer:
    return self.head_surf, self.head_vol, self.ln_3_surf, self.ln_3_vol
```

Then in `Transolver.forward`, apply the routing using `is_surface`:

```python
# After trunk forward pass, final block returns two heads
surf_out = last_block.head_surf(last_block.ln_3_surf(fx))  # [B,N,3]
vol_out  = last_block.head_vol(last_block.ln_3_vol(fx))   # [B,N,3]
# Merge: for surface nodes use surf_out, else vol_out
is_surf = data.get("is_surface", None)  # [B,N] bool
if is_surf is not None:
    preds = torch.where(is_surf.unsqueeze(-1), surf_out, vol_out)
else:
    preds = surf_out  # fallback
return {"preds": preds}
```

The `is_surface` tensor needs to be passed into the model. The cleanest way: pass
it through the `data` dict — `model({"x": x_norm, "is_surface": is_surface})`.
The student must ensure the model contract is respected: output `preds` is still
`[B, N, 3]` in global normalized space.

**Risk.** The surface head receives gradients from the surf_loss term (which is
already 10× weighted), and the vol head from vol_loss. With the separated heads
the surface head could overfit faster. Add a mild weight decay specifically to
the surface head (2× the base wd) to compensate.

**Complexity:** 3/5. ~50 lines, one architecture refactor.

---

## 3. Gradient clipping for heavy-tailed noise (Theoretically motivated)

**Category:** Training stability

**What it is.** Add adaptive gradient clipping with a theoretically motivated
threshold based on the gradient norm distribution, using the `clip-grad-norm_` API
already available in PyTorch.

**Why it might help.** The TandemFoilSet has a heavy-tailed pressure distribution
(max y-std = 2,077, mean = 458 for raceCar single). The MSE loss amplifies
outliers quadratically. Gradient norms on high-Re batches will be orders of
magnitude larger than on low-Re batches, causing the effective learning rate to
vary wildly between steps. Zhang et al. (2024, "Adaptive Gradient Clipping for
Heavy-Tailed Noise", 2406.04443) show theoretically that gradient clipping is
necessary for provable convergence under heavy-tailed gradient noise, and
empirically that a clip threshold at the 95th percentile of recent gradient norms
outperforms fixed clipping. In practice, max-norm clipping with `max_norm=1.0` is
the standard starting point.

**Specific change to train.py.** After `loss.backward()`, before `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This is a single line. Test with `max_norm=1.0` first. If training diverges,
lower to `0.5`. If it is too conservative (loss plateau in early epochs),
raise to `2.0`.

For the adaptive variant described in Zhang et al., track a running 95th percentile
of per-step grad norms and clip to that value. This is ~10 extra lines but more
principled. Start with fixed clipping; adaptive clipping if fixed hurts.

**Risk.** Very low. Gradient clipping is universally safe for transformer training.
The only risk is being too aggressive (clip_norm too small), which would slow
convergence. Monitor `train/surf_loss` curve — if it stops decreasing, raise
`max_norm`.

**Papers:** Zhang et al. (2024) "Gradient Clipping for Non-smooth Convex Optimization" (2406.04443). Also supported by standard transformer training practice (T5, GPT-NeoX all use grad clipping).

**Complexity:** 1/5. One line.

---

## 4. Log-space MAE loss for pressure channel

**Category:** Loss formulation

**What it is.** Apply a log-transform to the pressure channel when computing the
loss, so that errors on low-pressure (low-Re) samples are weighted comparably to
errors on high-pressure (high-Re) samples.

**Why it might help.** The current MSE loss in normalized space is formally correct,
but the global normalization divides by a single y_std that captures the
cross-sample variance (dominated by high-Re samples). Low-Re samples have a much
smaller pressure range, so their squared errors in normalized space are tiny — the
optimizer barely touches them. Yet the val_avg/mae_surf_p metric weights all splits
equally, including `val_geom_camber_cruise` where pressures are smaller. A
log-scale loss effectively re-weights the loss to equalize per-sample contribution.

The specific formulation for the pressure channel (channel index 2):

```python
# Standard loss for Ux, Uy (channels 0, 1)
sq_err_vel = (pred[..., :2] - y_norm[..., :2]) ** 2

# Log-MAE for p (channel 2): sign-aware
p_pred_raw = pred[..., 2:3] * stats["y_std"][2] + stats["y_mean"][2]
p_true_raw = y[..., 2:3]
# Sign-preserving log: sign(x) * log(1 + |x|)
def softlog(t):
    return t.sign() * (t.abs() + 1.0).log()
log_err_p = (softlog(p_pred_raw) - softlog(p_true_raw)) ** 2

# Combine: normalize log_err_p to match scale of sq_err
sq_err = torch.cat([sq_err_vel, log_err_p * cfg.log_p_scale], dim=-1)
```

Add `log_p_scale: float = 0.05` to Config to tune the weighting. Start at 0.05.

**Alternative (simpler):** Huber loss instead of MSE, with `delta=1.0` in
normalized space. This gives linear gradient for large errors instead of
quadratic, reducing the outsized influence of high-Re outlier samples.

```python
sq_err = F.huber_loss(pred, y_norm, reduction="none", delta=1.0)
```

This is a one-line change. Try Huber first; log-MAE second.

**Risk.** The model contract requires predicting in normalized space.
The Huber variant does not break this. The log-MAE variant requires
denormalizing inside the loss, which is slightly more complex but safe.
Watch for gradient explosions with the log variant at very small `|p|` values
near zero (the `+1.0` in softlog prevents this).

**Complexity:** 1/5 for Huber, 2/5 for log-MAE.

---

## 5. Multi-scale physics attention (coarse + fine slice tokens)

**Category:** Architecture

**What it is.** Run PhysicsAttention at two scales simultaneously: a coarse set of
32 slice tokens (global flow structure) and a fine set of 128 slice tokens (local
boundary layer details). Concatenate the two readouts before the output projection.

**Why it might help.** The current single-scale slice_num=64 is a single compromise
between global context (large patterns in the wake, freestream) and local resolution
(boundary layer on the foil surface). The MNO paper (2510.16071, 2025) shows 5–50%
error reduction on 3D CFD by explicitly operating at 3 scales. The MSPT paper
(2512.01738, 2025) uses ball-tree spatial partitioning to create coarse and fine
patches. For TandemFoilSet, the tandem interaction (foil 2 in the wake of foil 1)
is a global effect, while the pressure coefficient peak near the leading edge is
local. A dual-scale attention mechanism directly targets this multi-scale structure.

**Specific change to train.py.** Modify `PhysicsAttention` to accept a list of
slice counts:

```python
class MultiScalePhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 slice_nums=(32, 128)):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.attentions = nn.ModuleList([
            PhysicsAttention(dim, heads=heads, dim_head=dim_head,
                             dropout=dropout, slice_num=s)
            for s in slice_nums
        ])
        # Project concatenated outputs back to dim
        self.merge = nn.Linear(dim * len(slice_nums), dim)

    def forward(self, x):
        outs = [attn(x) for attn in self.attentions]
        return self.merge(torch.cat(outs, dim=-1))
```

Use `MultiScalePhysicsAttention` in `TransolverBlock` in place of `PhysicsAttention`.
Start with `slice_nums=(32, 128)`. Monitor VRAM — this roughly doubles attention
compute. With `dim=128`, this is manageable on 96GB.

**Risk.** Memory: two attention modules per block × 5 blocks. Estimate ~15% VRAM
increase. If OOM, reduce to `slice_nums=(32, 96)`. Also watch for the merge
projection learning to ignore one of the scales entirely (add a small L2 on merge
weights to force both paths to contribute).

**Complexity:** 3/5. ~40 lines, architecture change.

---

## 6. EMA (Exponential Moving Average) checkpoint averaging

**Category:** Training strategy

**What it is.** Maintain an EMA of model weights with decay=0.999, and use the
EMA model for validation checkpoint selection. The training model is updated by
the optimizer; the EMA model is a smoothed version used only for evaluation.

**Why it might help.** The baseline trains with no EMA, which means the best
checkpoint is a single point on the loss trajectory. EMA averaging smooths out
late-training oscillations and is empirically responsible for 0.5–2% gains on
regression benchmarks. It is a standard technique in modern transformer training
(used in ViT-22B, DiT, many Kaggle top solutions for regression). The gain is
free in terms of training compute — it is just a weight update per step.

**Specific change to train.py.** Add after model creation:

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_decay = 0.999
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
```

After each optimizer step:
```python
ema_model.update_parameters(model)
```

For validation and checkpoint selection, use `ema_model.module` instead of `model`.
The EMA model is saved as the checkpoint when it achieves a new best.

Note: `torch.optim.swa_utils` is part of PyTorch core (no new packages needed).

**Risk.** Very low. EMA cannot hurt training loss. The only risk is that for short
runs (debug mode), the EMA has not had time to converge. Check that `ema_model`
predictions equal `model` predictions at initialization.

**Complexity:** 1/5. ~10 lines.

---

## 7. 2D Rotary Positional Encoding (RoPE) for spatial coordinates

**Category:** Feature engineering / positional encoding

**What it is.** Replace the current concatenation of (x, z) coordinates into the
input feature vector with a Rotary Positional Encoding applied in the attention
mechanism, encoding relative spatial distances between slice tokens.

**Why it might help.** The current approach appends raw (x, z) to the input
features and relies on the MLP to learn spatial relationships. This is the
same limitation that RoPE was designed to address for sequence transformers —
the model needs to reason about _relative_ positions between nodes, not
just their absolute coordinates. RoPE-ViT (2403.13298, 2024) shows that 2D
RoPE enables transformers to extrapolate to different spatial resolutions and
outperforms learned absolute positional embeddings across 12 vision benchmarks.
In TandemFoilSet, the mesh has variable node counts (74K–242K) and the tandem
geometry means that relative foil-to-foil distances are structurally important.

The key idea: in `PhysicsAttention`, before computing Q and K on slice tokens,
apply 2D RoPE based on the slice token centroid positions (the weighted average
of (x, z) coordinates assigned to each slice).

**Specific change to train.py.**

Step 1: Compute slice centroids during the PhysicsAttention forward pass.

```python
# In PhysicsAttention.forward, after computing slice_weights [B,H,N,G]:
# Spatial positions are the first 2 dims of the input to the block
# We need to pass pos [B,N,2] through to PhysicsAttention

# Compute slice centroids: [B,H,G,2]
pos_expanded = pos[:, None, :, :].expand(B, self.heads, N, 2)  # [B,H,N,2]
w = slice_weights  # [B,H,N,G]
centroids = torch.einsum("bhng,bhnc->bhgc", w, pos_expanded)  # [B,H,G,2]
centroids = centroids / (w.sum(dim=2, keepdim=True).permute(0,1,3,2) + 1e-5)
```

Step 2: Apply 2D RoPE to Q and K based on centroids.

```python
def apply_rope_2d(q, k, centroids, dim_head):
    """Apply 2D RoPE to q,k using centroid positions."""
    # centroids: [B,H,G,2], q/k: [B,H,G,D]
    half_d = dim_head // 4
    theta = 10000 ** (-torch.arange(half_d, device=q.device) / half_d)
    x_pos = centroids[..., 0:1]  # [B,H,G,1]
    z_pos = centroids[..., 1:2]
    # Frequencies for x and z separately
    freqs_x = x_pos * theta[None, None, None, :]  # [B,H,G,half_d]
    freqs_z = z_pos * theta[None, None, None, :]
    cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
    cos_z, sin_z = freqs_z.cos(), freqs_z.sin()
    # Apply to first half/second half of dim_head
    q1, q2 = q[..., :half_d*2:2], q[..., 1:half_d*2:2]
    q_rot_x = torch.stack([q1*cos_x - q2*sin_x, q1*sin_x + q2*cos_x], dim=-1).flatten(-2)
    # Similar for z... (full implementation ~30 lines)
    return q_rotated, k_rotated
```

The full 2D RoPE implementation is ~50 lines. The key hyperparameter is the
base frequency (10000 is standard from LLaMA/Transformer papers).

**Alternative (simpler):** Use Fourier positional features for spatial coordinates:
instead of raw (x, z) in the input, concatenate sin/cos encodings at multiple
frequencies. This is 10 lines of preprocessing and avoids modifying the attention:

```python
# In Transolver.forward, before preprocess:
def fourier_pos(pos, freqs=[1, 2, 4, 8, 16]):
    encs = [pos]
    for f in freqs:
        encs += [torch.sin(2*math.pi*f*pos), torch.cos(2*math.pi*f*pos)]
    return torch.cat(encs, dim=-1)  # [B,N,2*(1+2*len(freqs))]
```

This expands space_dim from 2 to 22, requiring `preprocess` to accept the
expanded input. Adjust `space_dim` in model_config accordingly.

Start with Fourier features (simpler, no attention modification). Try full 2D RoPE
only if Fourier features show clear gain.

**Risk.** Medium. The RoPE version requires passing positional information through
to the attention module (architecture change). The Fourier version only changes
the input dimension (simpler but requires model_config update).

**Complexity:** 2/5 for Fourier features, 4/5 for full 2D RoPE.

---

## 8. Physics-informed boundary-layer input features

**Category:** Feature engineering / physics priors

**What it is.** Augment the 24-dimensional input features with 3 additional physics-
based features computed from existing inputs: approximate local Reynolds number
(Re_x), approximate inviscid pressure coefficient (Cp_inv), and wall distance
normalized by estimated boundary layer thickness (y_plus proxy).

**Why it might help.** The B-GNN paper (2503.18638, 2025) showed that adding
physics-based features (approximate Re_x from the panel method, inviscid Cp) to
boundary nodes reduced model size by 83% while _matching_ the accuracy of a model
trained without those features. The intuition: these features encode where in the
flow regime each node sits, allowing the model to distinguish attached laminar,
transition, and separated flow without inferring it from geometry alone. For
TandemFoilSet, the `dsdf` distance descriptors (dims 4–11) encode some spatial
context but are purely geometric. Physics-conditioned features would tell the model
something about the local flow state.

**Specific features to add:**

1. `log_re_x`: local chord Reynolds number proxy = `log(Re) + log(normalized_arc_length + eps)`.
   This approximates where on the foil the boundary layer is at a given Re.
   Computed as: `x[:, :, 13:14] + torch.log(x[:, :, 2:3].abs().clamp(min=1e-4))`
   (using dims 13=log(Re), 2=signed arc-length foil 1).

2. `gap_re_interaction`: interaction feature = `gap * log(Re)` for tandem nodes.
   Zero for single-foil samples. This encodes the aerodynamic coupling strength
   between the two foils, which varies strongly with both gap and Re.
   Computed as: `x[:, :, 22:23] * x[:, :, 13:14]` (dim 22=gap, dim 13=log(Re)).

3. `aoa_signed_normalized`: AoA expressed as `sin(AoA)` and `cos(AoA)` instead
   of raw radians. This makes the periodicity of the aerodynamic response explicit.
   Replace dims 14 and 18 with their sin/cos expansions, adding 2 extra dims.

**Implementation.** Add a `PhysicsFeatureAugmenter` module before the `preprocess` MLP:

```python
class PhysicsFeatureAugmenter(nn.Module):
    """Augment raw input features with physics-motivated derived features."""
    def forward(self, x):
        # x: [B, N, 24]
        log_re = x[..., 13:14]           # dim 13
        arc_len = x[..., 2:3]            # dim 2 (signed arc-length)
        gap = x[..., 22:23]             # dim 22
        aoa1 = x[..., 14:15]            # dim 14
        aoa2 = x[..., 18:19]            # dim 18

        log_re_x = log_re + torch.log(arc_len.abs().clamp(min=1e-4))
        gap_re   = gap * log_re
        sin_aoa1 = torch.sin(aoa1 * math.pi)  # aoa already in radians (normalized)
        cos_aoa1 = torch.cos(aoa1 * math.pi)
        sin_aoa2 = torch.sin(aoa2 * math.pi)
        cos_aoa2 = torch.cos(aoa2 * math.pi)

        return torch.cat([x, log_re_x, gap_re, sin_aoa1, cos_aoa1,
                          sin_aoa2, cos_aoa2], dim=-1)  # [B, N, 30]
```

Update `fun_dim` in `model_config` from 22 to 28 (24 original - 2 space dims + 6 new).

**Risk.** Medium. The AoA features are already encoded linearly as radians; the
sin/cos expansion should help for large AoA changes but may not matter for the
small ranges here (-10° to +6°). The log_re_x feature involves the arc-length
which may be zero for interior nodes — the clamp handles this but check for NaN.
Note: `x` at this point has been normalized by `x_mean/x_std`. Work with
_raw_ features (before normalization) or re-check the normalization context.

**Complexity:** 3/5. ~30 lines + model_config update.

---

## 9. Domain-conditioned Mixture-of-Experts slice tokens

**Category:** Architecture

**What it is.** Replace the single fixed `in_project_slice` weight matrix in
`PhysicsAttention` with a domain-conditioned MoE: 3 expert slice projection
matrices (one per domain: raceCar single, raceCar tandem, cruise), with a soft
routing based on the domain indicator inferred from input features.

**Why it might help.** The NESTOR paper (ICLR 2026 submission, "Nested MoE Neural
Operator") introduces image-level MoE (global routing per sample) and token-level
Sub-MoE (local routing per node) for heterogeneous PDE inputs, outperforming
standard operators on multi-domain datasets. TandemFoilSet has 3 structurally
distinct domains: the mesh geometry, AoA range, and Re range all differ across
raceCar single, raceCar tandem, and cruise. The current single slice projection
must learn a single set of "physics slices" that work for all three. Letting each
domain use specialized slices could reduce this compromise.

**Domain indicator.** Infer domain from input features without changing the data
contract:
- Single-foil vs. tandem: check if `x[:, :, 22].max() > 0` (gap > 0 → tandem)
- raceCar vs. cruise: check if AoA of foil 1 (`x[:, :, 14].mean() < 0`) since
  raceCar uses inverted foils (negative AoA).

This gives a 3-way soft routing weight per sample.

**Specific change.** Modify `PhysicsAttention.__init__`:

```python
self.n_experts = 3
self.in_project_slice_experts = nn.ParameterList([
    nn.Parameter(torch.empty(dim_head, slice_num))
    for _ in range(self.n_experts)
])
for p in self.in_project_slice_experts:
    torch.nn.init.orthogonal_(p)
self.domain_router = nn.Linear(dim_head, self.n_experts)
```

In forward, compute soft routing and mix expert projections:

```python
# Domain routing from global average of x_mid [B,H,N,D] -> [B,H,D]
global_feat = x_mid.mean(dim=2)  # [B,H,D]
routing = F.softmax(self.domain_router(global_feat), dim=-1)  # [B,H,3]

# Mix expert projections: [D, G] weighted sum
slice_proj = sum(
    routing[:, :, i, None, None] * self.in_project_slice_experts[i][None, None]
    for i in range(self.n_experts)
)  # [B,H,D,G]

# Apply mixed projection
slice_weights = self.softmax(
    torch.einsum("bhnc,bhcg->bhng", x_mid, slice_proj) / self.temperature
)
```

**Risk.** High complexity; the einsum change is non-trivial to implement correctly.
Monitor for routing collapse (one expert dominating). Add an auxiliary load-balancing
loss (standard MoE technique): `aux_loss = routing.var(dim=0).sum() * 0.01`.

**Complexity:** 4/5. ~60 lines of architecture changes.

---

## 10. Homoscedastic uncertainty weighting for multi-task loss

**Category:** Loss formulation

**What it is.** Automatically learn the relative weight between vol_loss and
surf_loss using the Kendall et al. (2018) homoscedastic uncertainty approach,
where `surf_weight` is replaced by a learnable log-scale parameter per task.

**Why it might help.** The current `surf_weight=10` is a fixed hyperparameter.
It is unclear whether 10 is optimal — it was set heuristically. The Kendall et al.
(CVPR 2018) approach treats each task's loss as having a learned noise scale σ,
and minimizes `L1/(2σ1²) + log(σ1) + L2/(2σ2²) + log(σ2)`. This automatically
adapts the weights as training progresses. For TandemFoilSet, the optimal
vol/surf tradeoff may differ across training phases: early training should
emphasize vol_loss to get the global flow right, then late training should
push surf_loss harder.

**Specific change to train.py.** After model creation, add:

```python
# Learnable log-uncertainty for [vol_loss, surf_loss]
log_sigma = nn.Parameter(torch.zeros(2))  # σ_vol=1, σ_surf=1 initially
# Add to optimizer:
optimizer = torch.optim.AdamW(
    list(model.parameters()) + [log_sigma],
    lr=cfg.lr, weight_decay=0.0  # no wd on log_sigma
)
```

In the training loop, replace the fixed loss:

```python
# Homoscedastic uncertainty weighting (Kendall et al. 2018)
sigma_vol = torch.exp(log_sigma[0])
sigma_surf = torch.exp(log_sigma[1])
loss = (vol_loss / (2 * sigma_vol**2) + log_sigma[0] +
        surf_loss / (2 * sigma_surf**2) + log_sigma[1])
```

Log `sigma_vol.item()` and `sigma_surf.item()` each epoch to verify the weights
are moving in a sensible direction (expect sigma_surf to decrease relative to
sigma_vol over time, as the model improves surface predictions).

**Risk.** Low–medium. The method is well-validated in multi-task learning. The main
risk is that `log_sigma` converges to a regime where vol_loss is heavily
downweighted — monitor this. Initialize `log_sigma[1] = -log(sqrt(10)) ≈ -1.15`
to start near the manually tuned surf_weight=10 baseline.

**Complexity:** 2/5. ~15 lines.

---

## 11. Scheduled surface-pressure loss annealing

**Category:** Loss formulation

**What it is.** Start training with `surf_weight=1` (equal weighting) and linearly
increase it to `surf_weight=20` over the first 60% of training epochs, then hold
at 20. This is a curriculum on the primary metric target.

**Why it might help.** With `surf_weight=10` from epoch 1, the model receives a
strong bias toward surface nodes before it has learned the global flow structure.
Surface pressure is sensitive to the global velocity field (via Bernoulli's
equation), so getting vol predictions right first should improve surf predictions
later. This is analogous to curriculum learning strategies in PDE solvers where
easy targets (smooth fields) are learned before hard targets (sharp boundary layers).

**Specific change to train.py.** Add `surf_weight_schedule: bool = True` and
`surf_weight_final: float = 20.0` to Config. In the training loop:

```python
# Anneal surf_weight from 1 to surf_weight_final over first 60% of epochs
anneal_epochs = int(0.6 * MAX_EPOCHS)
if cfg.surf_weight_schedule and epoch < anneal_epochs:
    current_surf_weight = 1.0 + (cfg.surf_weight_final - 1.0) * (epoch / anneal_epochs)
else:
    current_surf_weight = cfg.surf_weight_final
loss = vol_loss + current_surf_weight * surf_loss
```

**Risk.** Very low. Even if the schedule does not help, it falls back to a higher
surf_weight than the baseline (20 vs. 10), which independently might improve
surface pressure. Test with the schedule first; if it does not help, test
surf_weight=20 fixed as a free ablation.

**Complexity:** 1/5. ~8 lines.

---

## 12. Reynolds-number stratified curriculum sampling

**Category:** Training strategy

**What it is.** Replace the current domain-balanced WeightedRandomSampler with a
two-phase sampler: phase 1 (epochs 1–30%) oversamples low-to-mid Re samples
(Re < 500K), phase 2 (remaining epochs) uses the standard domain-balanced sampler.

**Why it might help.** High-Re samples dominate training gradients due to their
larger absolute pressure values. The OOD val splits include `val_re_rand`, which
stratifies across all Re values. If the model learns the high-Re regime too early,
it may learn the wrong inductive bias for the flow structure (e.g., fully turbulent
assumptions that break at low Re). Starting with low-Re samples (smoother pressure
fields, more regular flow patterns) gives the model a curriculum from easy to hard.

**Specific change.** The `sample_weights` tensor from `load_data()` already balances
by domain. Replace or augment it in `train.py`:

```python
# Additional Re-based weights: boost low-Re samples in early training
def get_re_curriculum_weights(train_ds, epoch, total_epochs, base_weights):
    """Boost low-Re samples (log_Re < log(500K)) in first 30% of training."""
    curriculum_frac = min(epoch / (0.3 * total_epochs), 1.0)
    # Extract log_Re from dataset
    re_weights = torch.ones(len(train_ds))
    for i in range(len(train_ds)):
        x, _, _ = train_ds[i]
        log_re = x[0, 13].item()  # dim 13 is log(Re), from first node
        re_weights[i] = 2.0 - curriculum_frac if log_re < 13.1 else 1.0
        # log(500K) ≈ 13.1
    return base_weights * re_weights / (base_weights * re_weights).sum() * len(train_ds)
```

Note: pre-compute re_weights at epoch 0 (before training) since iterating 1500
samples to get their log_Re is fast (1–2 seconds). Only recompute at the curriculum
phase transition.

**Risk.** Medium. The curriculum requires computing Re values from the dataset,
which adds a small overhead. Also, if the low-Re curriculum phase is too long,
the model may not see enough high-Re samples to converge by the timeout. Start
with `curriculum_frac_end=0.2` (20% of epochs for low-Re phase).

**Complexity:** 2/5. ~30 lines.

---

## 13. Stochastic depth regularization for TransolverBlocks

**Category:** Architecture / regularization

**What it is.** During training, randomly skip entire TransolverBlocks with a
layer-dependent probability (deeper layers skipped more often). At inference, all
layers are used. This is the "DropPath" or "stochastic depth" technique from
DeiT and ViT literature.

**Why it might help.** With 5 layers and `n_hidden=128`, the Transolver is a
relatively small model that may overfit to the training distribution. The OOD
val splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) test geometry
generalization. Stochastic depth is the strongest regularizer found in ViT
ablation studies for small-to-medium models. The `timm.layers` library (already
imported via `from timm.layers import trunc_normal_`) provides `DropPath`.

**Specific change to train.py.** Add to `TransolverBlock.__init__`:

```python
from timm.layers import DropPath
# In __init__, add:
self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
```

In `forward`:
```python
fx = self.drop_path(self.attn(self.ln_1(fx))) + fx
fx = self.drop_path(self.mlp(self.ln_2(fx))) + fx
```

In `Transolver.__init__`, pass linearly increasing drop rates per layer:

```python
dpr = [i / (n_layers - 1) * max_drop_path for i in range(n_layers)]
# max_drop_path = 0.1 as starting point
```

Since `timm` is already in `pyproject.toml` (imported at the top of `train.py`),
this requires no new dependencies.

**Risk.** Low. DropPath from timm is well-tested and the API matches exactly.
The main risk: at drop_path=0.1 with n_layers=5, the effective model depth
varies between 4.5 and 5 layers, which is a mild regularization. If the model
is underfitting (which is possible in 50-epoch runs), increase max_drop_path to 0.2.

**Complexity:** 2/5. ~15 lines using the already-available timm.DropPath.

---

## Implementation priority order for 8 parallel students

Given 8 available GPU slots, the recommended assignment order to maximize
information gain and avoid redundant experiments:

**Wave 1 (independent, high signal-to-cost):**
1. `gradient-clipping-heavy-tail` — 1-line change, definitive answer in 1 run
2. `ema-validation-checkpoint` — 10-line change, orthogonal to all others
3. `surf-vol-decoupled-loss-schedule` — 8-line change, tests surf_weight=20 as a free bonus
4. `homoscedastic-uncertainty-weighting` — 15-line change, replaces the hand-tuned surf_weight

**Wave 2 (conditional on wave 1 results):**
5. `per-sample-scale-normalization` — highest expected impact, needs careful loss contract check
6. `log-mae-loss-pressure` (Huber variant) — 1-line change, combine with #5 if #5 wins
7. `surface-volume-branch-head` — architecture change, run after loss changes settle
8. `stochastic-depth-transolver` — regularization, most useful after architecture settles

**Wave 3 (bold, run if plateau continues):**
9. `multiscale-slice-attention`
10. `rope-2d-positional-encoding` (Fourier variant first)
11. `physics-input-boundary-layer`
12. `domain-conditioned-moe-slices`
13. `re-stratified-curriculum`

---

## Key references

| Paper | Year | Relevance |
|-------|------|-----------|
| Transolver (ICML 2024 Spotlight, arXiv 2402.02366) | 2024 | Baseline architecture |
| MNO: Multiscale Neural Operator (2510.16071) | 2025 | Multi-scale slice attention, #5 |
| AB-UPT: Anchor-Branched UPT (TMLR 2025) | 2025 | Surface-volume branch head, #2 |
| B-GNN Boundary GNNs (2503.18638) | 2025 | Physics features for boundary nodes, #8 |
| MSPT: Multi-Scale Patch Transformer (2512.01738) | 2025 | Ball-tree spatial patches, informs #5 |
| NESTOR: Nested MoE Operator (ICLR 2026) | 2026 | MoE slices, #9 |
| RoPE-ViT (2403.13298) | 2024 | 2D RoPE for spatial transformers, #7 |
| Gradient clipping heavy-tailed noise (2406.04443) | 2024 | Theoretically motivated clipping, #3 |
| Kendall et al. Multi-task uncertainty (CVPR 2018) | 2018 | Homoscedastic weighting, #10 |
| SWA/EMA in PyTorch (swa_utils) | 2021 | EMA model averaging, #6 |
| DropPath / Stochastic Depth (DeiT, timm) | 2021 | Regularization via layer skip, #13 |
