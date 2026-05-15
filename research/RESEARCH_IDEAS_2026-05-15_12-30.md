<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15 12:30

Generated from first principles from `program.md` and the baseline `train.py`.
No prior experiment history consulted (launch isolation rule).

---

## Context: Why the Baseline May Be Leaving Performance on the Table

The baseline Transolver uses:
- **MSE loss in globally normalized space** — global `y_std` does not neutralize
  per-sample dynamic range. High-Re samples (Re → 5M) have roughly 10x larger
  normalized residuals than low-Re samples, so they dominate the gradient signal.
- **`surf_weight=10` flat multiplier** on surface MSE — this is the only surface
  emphasis, but surface nodes are ~5% of the mesh. A poorly-scaled weight either
  under-emphasizes surface or explodes volume gradients.
- **No gradient clipping** — pressure extremes at high Re can produce gradient
  spikes that destabilize early training.
- **No warmup** — cosine annealing from full LR at epoch 1 is fragile when the
  first batches may include high-Re outliers.
- **Slice attention is geometry-blind** — `in_project_slice` operates on the
  hidden representation with no explicit awareness of whether a node is on a
  surface. Surface and volume nodes compete for the same 64 slice tokens.
- **Three output channels (Ux, Uy, p) share one loss term** — pressure has units
  m²/s² and can range ±30K; velocity channels are much smaller. Global
  normalization partially addresses this but does not decouple channel gradients.
- **Model is narrow** — n_hidden=128, n_layers=5. At ~640K parameters, the model
  may be capacity-limited for the hardest OOD splits (unseen camber geometries).

---

## Hypothesis 1: Huber Loss Replaces MSE to Dampen High-Re Outlier Gradients

**Family:** Loss reformulation

**Rationale.**
Global normalization divides by a single `y_std` computed across all samples. A
high-Re sample (Re=5M) has per-sample std ~10x larger than a low-Re sample
(Re=100K). After normalization, the high-Re sample still has residuals ~10x
larger in magnitude, so its squared error is ~100x larger than the low-Re sample.
MSE amplifies this further via squaring. Huber loss with a small delta (e.g.,
delta=0.5 in normalized space) degrades to MAE for these large-residual nodes,
removing the quadratic amplification and stabilizing gradients.

**Predicted mechanism:** Reducing gradient dominance of high-Re pressure extremes
allows the optimizer to learn better representations for mid-Re and
geometry-interpolation cases (the OOD camber splits).

**Exact code change in `train.py`:**
```python
# Replace the sq_err computation in the training loop (line ~490–496):
# OLD:
#   sq_err = (pred - y_norm) ** 2
#   vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
#   surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# NEW: add cfg.huber_delta (default 0.5) to Config dataclass
huber_delta = cfg.huber_delta  # e.g. 0.5
err = pred - y_norm
huber_err = torch.where(
    err.abs() < huber_delta,
    0.5 * err ** 2,
    huber_delta * (err.abs() - 0.5 * huber_delta),
)
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```
Also update `evaluate_split` to use the same Huber for the validation `loss`
metric (not mandatory — MAE metrics are unchanged; only `{split}/loss` changes).

**Hyperparameters to try:**
- `huber_delta=0.5` — primary run
- `huber_delta=1.0` — fallback if 0.5 is too aggressive (nearly pure MAE)
- `surf_weight=10` unchanged initially; re-tune after confirming Huber helps

**Config change:** Add `huber_delta: float = 0.5` to the `Config` dataclass.

**Predicted impact:** -5 to -15% on `val_avg/mae_surf_p`. Largest benefit on
`val_re_rand` and `val_single_in_dist` (highest Re range, most outlier
sensitivity). Modest benefit on camber OOD splits where outlier effect is
secondary.

**Split most helped:** `val_re_rand` (stratified Re holdout — directly tests
cross-Re generalization).

**Risk / failure mode:** If the model is already well-regularized and the high-Re
samples are the hardest but also the most informative, reducing their gradient
weight could slow convergence. Monitor train loss vs. val MAE divergence. If val
MAE stops improving while train loss plateaus, the delta is too small.

---

## Hypothesis 2: Per-Sample Inverse-Std Loss Weighting to Balance Across Re Regimes

**Family:** Loss reformulation

**Rationale.**
Even with Huber loss, samples from different Re regimes contribute unequal
gradient signals. A principled fix is to weight each sample's loss by
`1 / per_sample_y_std`, so a high-Re sample (std=2000) contributes the same
total gradient as a low-Re sample (std=200). This is related to inverse-variance
weighting in Gaussian regression and to per-sample normalization in weather
modelling (WeatherBench, Keisler 2022, GraphCast 2023).

**Exact code change in `train.py`:**
```python
# In the training loop, after y_norm is computed:
# Compute per-sample std over real (non-padding) nodes
with torch.no_grad():
    real_y = y * mask.unsqueeze(-1).float()  # [B, N, 3]
    # y std per sample: mean over nodes and channels
    per_sample_std = real_y.std(dim=1).mean(dim=-1).clamp(min=1.0)  # [B]
    sample_weights_loss = 1.0 / per_sample_std  # [B]
    # Normalize so batch mean weight = 1
    sample_weights_loss = sample_weights_loss / sample_weights_loss.mean()

# Weighted loss: apply [B] weight to each sample's contribution
err = (pred - y_norm) ** 2  # [B, N, 3]
# vol contribution
vol_err_per_sample = (err * vol_mask.unsqueeze(-1)).sum(dim=[1, 2]) / vol_mask.sum(dim=1).clamp(min=1).float()
vol_loss = (vol_err_per_sample * sample_weights_loss).mean()
# surf contribution
surf_err_per_sample = (err * surf_mask.unsqueeze(-1)).sum(dim=[1, 2]) / surf_mask.sum(dim=1).clamp(min=1).float()
surf_loss = (surf_err_per_sample * sample_weights_loss).mean()
loss = vol_loss + cfg.surf_weight * surf_loss
```

**Hyperparameters:** `surf_weight=10` unchanged. No new hyperparameters.

**Predicted impact:** -5 to -10% on `val_re_rand`. Marginal on camber OOD (which
is dominated by geometry generalization, not Re variation). May degrade
`val_single_in_dist` very slightly if high-Re raceCar samples are down-weighted
too aggressively.

**Split most helped:** `val_re_rand`.

**Risk / failure mode:** If per-sample std is unstable (e.g., for samples where
most nodes have near-zero velocity), the weighting can become very large. The
`clamp(min=1.0)` guards against this, but check that `per_sample_std` histograms
look sensible. A hard cap at `clamp(max=std_cap)` (e.g., `std_cap=3000`) may
also be needed.

---

## Hypothesis 3: Surface-MAE Loss With Separate Pressure Channel Weight

**Family:** Loss reformulation

**Rationale.**
The primary metric is surface pressure MAE. The current loss uses MSE over all 3
channels equally, with only `surf_weight=10` separating surface from volume. Two
issues: (a) MSE penalizes large errors quadratically, but MAE is the test metric;
(b) the pressure channel p (channel 2) has much larger magnitude than Ux/Uy,
so it contributes more to MSE even after global normalization (because global
normalization equalizes channel means but not within-sample variation). Switching
surface loss to MAE and adding a p-channel multiplier better aligns training with
the evaluation objective.

**Exact code change:**
```python
# In training loop, replace loss computation:
vol_loss = (((pred - y_norm) ** 2) * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)

# Surface: MAE instead of MSE, with per-channel weights
surf_err = (pred - y_norm).abs()  # [B, N, 3]
# Weight p channel more — channels are [Ux, Uy, p]
channel_weights = torch.tensor([1.0, 1.0, cfg.p_channel_weight],
                                 device=device)  # e.g. p_channel_weight=3.0
surf_err_weighted = surf_err * channel_weights[None, None, :]
surf_loss = (surf_err_weighted * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add `p_channel_weight: float = 3.0` to `Config` dataclass.

**Hyperparameters:**
- `p_channel_weight=3.0`, `surf_weight=10` — primary run
- `p_channel_weight=5.0` as follow-up if 3.0 helps

**Predicted impact:** -5 to -20% on `val_avg/mae_surf_p` because training now
directly targets MAE on the primary metric channel. The MSE→MAE switch for
surface nodes reduces the quadratic penalty on large errors and makes the
training gradient more aligned with the evaluation score.

**Split most helped:** All 4 splits (benefit is on surface pressure prediction,
which is what all splits measure).

**Risk / failure mode:** Combining MAE (surface) and MSE (volume) can produce
gradient scale mismatches. If vol_loss scale is much larger than surf_loss,
`surf_weight` may need re-tuning. Suggest a diagnostic run that logs per-term
gradient norms to see if the magnitudes are well-balanced.

---

## Hypothesis 4: Gradient Clipping + Linear LR Warmup for Training Stability

**Family:** Optimization

**Rationale.**
The baseline has no gradient clipping and starts cosine annealing from full LR.
With 74K–242K node meshes and pressure ranges up to ±30K, early batches that
happen to contain high-Re, high-pressure samples can produce very large
gradients. Without clipping, these can push the model into a bad basin in the
first few epochs. Linear warmup for 5 epochs ramps the LR from near-zero to
`lr=5e-4`, preventing the optimizer from taking large steps before the model
has stabilized. Gradient clipping to `max_norm=1.0` is standard in transformer
training (GPT-2, ViT, Transolver paper's implementation notes).

**Exact code change:**
```python
# In Config dataclass, add:
#   grad_clip: float = 1.0
#   warmup_epochs: int = 5

# Replace scheduler (after optimizer definition):
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1e-6 / cfg.lr,  # effectively start from lr*1e-6
    end_factor=1.0,
    total_iters=cfg.warmup_epochs,
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1),
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[cfg.warmup_epochs],
)

# In training loop, after loss.backward():
if cfg.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

**Hyperparameters:**
- `grad_clip=1.0`, `warmup_epochs=5`
- If training is already stable (no NaN loss spikes), try `grad_clip=5.0` to
  see if the clipping is actually active

**Predicted impact:** Primarily reduces early epoch instability and reduces
variance across seeds. Expected -2 to -8% on `val_avg/mae_surf_p`. Main benefit
is more reliable convergence — the best checkpoint found is less sensitive to
the random ordering of high-Re batches in the first few epochs.

**Split most helped:** `val_single_in_dist` (raceCar single, highest Re range,
most likely to see extreme pressure batches early).

**Risk / failure mode:** Warmup with only 50-epoch budget uses 10% of budget for
warmup. With the 30-minute cap, this may not be enough total epochs to see the
effect. Suggest running at `--epochs 50` with the 30-minute cap and checking
whether training is actually hitting the timeout before epoch 50.

---

## Hypothesis 5: Surface-Biased Slice Token Assignment via is_surface Conditioning

**Family:** Architecture

**Rationale.**
The Transolver's `PhysicsAttention` assigns nodes to 64 soft slice tokens via a
linear projection of the hidden representation. There is no mechanism to ensure
surface nodes cluster together in slice space. Since the evaluation metric is
surface-only, it is suboptimal for surface nodes to compete with the ~95% volume
nodes for slice token assignment. Explicitly conditioning the slice projection on
the `is_surface` boolean allows the model to learn separate slice token
distributions for surface and volume nodes. This is analogous to class-conditional
attention in image generation, or to "landmark" points in FNO-based methods.

**Exact code change in `train.py`:**

In `PhysicsAttention.__init__`, change `in_project_slice` to take 1 extra input
dim (the is_surface flag):
```python
# Old: self.in_project_slice = nn.Linear(dim_head, slice_num)
# New:
self.in_project_slice = nn.Linear(dim_head + 1, slice_num)
```

In `PhysicsAttention.forward`, pass `is_surface` as an argument and concatenate:
```python
def forward(self, x, is_surface=None):
    # ... existing projection code to get x_mid [B, H, N, dim_head] ...
    if is_surface is not None:
        # is_surface: [B, N] → [B, 1, N, 1] → broadcast to [B, H, N, 1]
        surf_flag = is_surface.float().unsqueeze(1).unsqueeze(-1).expand(
            B, self.heads, N, 1
        )
        x_mid_cond = torch.cat([x_mid, surf_flag], dim=-1)
    else:
        x_mid_cond = torch.cat([x_mid, torch.zeros(B, self.heads, N, 1, device=x.device)], dim=-1)
    slice_weights = self.softmax(self.in_project_slice(x_mid_cond) / self.temperature)
    # rest unchanged ...
```

In `TransolverBlock.forward` and `Transolver.forward`, thread `is_surface` down.
The `Transolver.forward` receives `data["is_surface"]` already in the model
contract — just unpack it.

**Hyperparameters:** No new hyperparameters. `slice_num=64` unchanged.

**Predicted impact:** -5 to -15% on `val_avg/mae_surf_p`. The model can now
reserve a subset of slice tokens exclusively for surface nodes, improving the
resolution of surface pressure prediction.

**Split most helped:** `val_geom_camber_rc` and `val_geom_camber_cruise`
(unseen geometry — surface node distribution differs from training; explicit
surface clustering reduces the chance of a surface node being pooled with volume
nodes in the slice token space).

**Risk / failure mode:** Passing `is_surface` through the full block stack
requires careful propagation. The `mask` (padding) and `is_surface` tensors must
be handled separately. If `is_surface` is not available at eval time, the model
will degrade to using all-zero surface flags (equivalent to "no surface"). The
model contract in `program.md` does expose `is_surface` from the batch, so
this should be feasible.

---

## Hypothesis 6: Wider, Shallower Model — n_hidden=256, n_layers=3

**Family:** Architecture (width vs depth tradeoff)

**Rationale.**
The baseline uses n_hidden=128, n_layers=5 (~640K params). For mesh-based PDE
surrogates, the tradeoff between width and depth is often empirically determined.
Wider models have higher expressivity per layer and can represent sharper
spatial features, which matters for pressure at leading/trailing edges. Shallower
models have less information mixing across slices, which may reduce over-smoothing
on the surface. At n_hidden=256 with n_layers=3, the parameter count is
approximately 256*2*256*2 (preprocess MLP) + 3*(256*256*3 + 256*256*2*2)
≈ ~3.3M params — about 5x the baseline but still well within 96GB VRAM for
batch_size=4 with 242K nodes.

**Exact code change (in `model_config` dict):**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,     # was 128
    n_layers=3,       # was 5
    n_head=8,         # was 4; scale heads with width
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

**Hyperparameters:** `lr=5e-4` (unchanged), `surf_weight=10` (unchanged).
If the wider model trains slower (larger gradients per step), consider
`lr=3e-4` as a follow-up.

**Predicted impact:** Uncertain but potentially -10 to -20% if the baseline
is capacity-limited. Main risk is over-fitting since the training set is small
(1499 samples). Monitor train vs. val MAE gap.

**Split most helped:** `val_geom_camber_rc` and `val_geom_camber_cruise`
(hardest OOD splits — may require more representational capacity to interpolate
across unseen camber geometry).

**Risk / failure mode:** VRAM. With batch_size=4 and N_max=242K, each sample
is [242K, 256] float32 ≈ 247MB per layer forward pass. At 3 layers, this is
~750MB just for activations, well within 96GB. But if gradient checkpointing
is needed, add `torch.utils.checkpoint.checkpoint_sequential`. Monitor peak
VRAM reported by the training script.

---

## Hypothesis 7: Deeper, Narrower Model — n_hidden=96, n_layers=8

**Family:** Architecture (width vs depth tradeoff)

**Rationale.**
The complementary experiment to H6. Deeper models process information through
more layers of slice attention, allowing longer-range propagation of boundary
condition information across the mesh. For Navier-Stokes simulations, the
solution at interior nodes depends on the boundary conditions at the foil
surface — a deeper model may be better at propagating this signal. At n_hidden=96,
n_layers=8 the parameter count is comparable to or slightly smaller than the
baseline (~480K params).

**Exact code change:**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=96,      # was 128
    n_layers=8,       # was 5
    n_head=4,         # unchanged; dim_head=96//4=24
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

**Predicted impact:** Uncertain; likely -5 to +5% relative to baseline. If the
bottleneck is propagation depth (boundary signal not reaching volume nodes), this
should help. If the bottleneck is representation width (model can't represent
sharp pressure peaks), this may not help.

**Split most helped:** `val_re_rand` (cross-domain generalization may benefit
from better long-range information mixing in the mesh).

**Risk / failure mode:** With n_head=4 and n_hidden=96, dim_head=24. This is
very small for attention — attention heads may not be expressive enough. If
val loss plateaus early, the model is likely under-capacity. Consider
n_hidden=128, n_layers=8 instead (slightly over baseline params).

---

## Hypothesis 8: More Slice Tokens — slice_num=128

**Family:** Architecture (slice resolution)

**Rationale.**
The tandem-foil mesh has a rich multi-scale structure: coarse background zone,
two dense foil zones, foil surfaces, wake regions. With only 64 slice tokens,
the model may conflate nodes from distinct physical regions (e.g., zone 1 foil 1
surface vs. zone 2 foil 2 interior). Increasing to 128 slices gives more tokens
to partition the ~100-240K node mesh, potentially allowing finer-grained attention
in each region. The slice attention cost is O(slice_num^2) per head, so
doubling from 64 to 128 quadruples the attention FLOPs — still trivial relative
to the O(N) projection cost for N~200K.

**Exact code change:**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,    # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

No other changes. The `in_project_slice` weight shape changes from
`[dim_head, 64]` to `[dim_head, 128]` — already handled by the existing code.

**Predicted impact:** -3 to -10% on `val_avg/mae_surf_p`. Modest but positive
if the current 64 slices are creating unwanted mixing between physical regions.

**Split most helped:** `val_geom_camber_rc` (tandem raceCar — two-foil mesh
with the most distinct spatial sub-regions).

**Risk / failure mode:** With 128 slices and only 1499 training samples, some
slice tokens may rarely activate for unseen camber geometries. Orthogonal
initialization of `in_project_slice` helps spread coverage, but OOD inputs
may cause slice token collapse. If `val_geom_camber_*` metrics worsen while
`val_single_in_dist` improves, over-specialization is the cause.

---

## Hypothesis 9: Fourier Positional Encoding of Node Coordinates

**Family:** Input / Feature Engineering

**Rationale.**
The baseline provides raw (x, z) node coordinates in dims 0-1, already
normalized. For a mesh spanning a ~10-chord domain, normalized coordinates lie
in roughly [-1, 1]. The MLP preprocess layer must learn all position-dependent
features from these two scalars. Random Fourier Features (Rahimi & Recht, 2007)
or learned sinusoidal encodings (Mildenhall et al., NeRF 2020; Tancik et al.,
Fourier Features Let Networks Learn High Frequency Functions, 2020) provide a
richer positional basis that is known to improve neural field predictions of
spatially varying quantities. For pressure field prediction, the sharp gradients
near the foil surface and leading edge are exactly the high-frequency content
that flat coordinate inputs struggle with.

**Exact code change:**
```python
# Add a FourierPositionEncoder at the top of train.py:
class FourierPositionEncoder(nn.Module):
    def __init__(self, n_freqs=16, sigma=1.0):
        super().__init__()
        # Fixed random Fourier features on (x, z): 2D input → 2*n_freqs output
        B_mat = torch.randn(2, n_freqs) * sigma
        self.register_buffer("B", B_mat)

    def forward(self, xy):
        # xy: [..., 2]
        proj = 2 * torch.pi * xy @ self.B  # [..., n_freqs]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 2*n_freqs]

# In Transolver.__init__, augment preprocess input:
# fun_dim stays the same (22), but space_dim doubles from 2 to 2 + 2*n_freqs
# Change model_config:
model_config = dict(
    space_dim=2 + 2*16,   # 2 raw + 32 Fourier = 34
    fun_dim=X_DIM - 2,    # 22 non-position dims
    ...
)

# Before passing to model in training loop:
pos_enc = FourierPositionEncoder(n_freqs=16, sigma=1.0).to(device)
...
x_norm = (x - stats["x_mean"]) / stats["x_std"]
# x_norm[:, :, 0:2] = normalized (x, z); augment with Fourier features
fourier_pos = pos_enc(x_norm[:, :, 0:2])   # [B, N, 32]
x_augmented = torch.cat([fourier_pos, x_norm[:, :, 2:]], dim=-1)  # [B, N, 54]
pred = model({"x": x_augmented})["preds"]
```

Update `space_dim` in model_config to match the augmented input dimensions.
Also update `stats["x_mean"]` and `stats["x_std"]` — Fourier features are
not in stats, so concatenate after normalization.

**Hyperparameters:** `n_freqs=16`, `sigma=1.0` (standard RFF scale for
normalized coordinates). Alternatively use learned sinusoidal embeddings for
fine-tuning later.

**Predicted impact:** -5 to -15% on `val_avg/mae_surf_p`. Largest benefit on
surface pressure where sharp spatial gradients near the foil edge are
difficult to represent from flat normalized coordinates alone.

**Split most helped:** `val_geom_camber_cruise` (the cruise geometry has
the smallest mesh and most distinct leading-edge pressure features).

**Risk / failure mode:** The Fourier features are generated from normalized
coordinates. If `stats["x_mean"][0:2]` and `stats["x_std"][0:2]` are
computed over all mesh nodes including padding, the normalization may not
center the actual mesh domain well. Verify that normalized (x,z) lies
in ~[-2, +2] for the training set before fixing sigma.

---

## Hypothesis 10: Re-Normalized Targets — Per-Sample Z-Score Using log(Re)

**Family:** Input / Feature Engineering (target normalization)

**Rationale.**
The primary challenge to generalization across Re regimes is that pressure
magnitudes scale approximately as Re (dynamic pressure q = 0.5 * rho * U^2, and
for fixed geometry Cp is Re-independent but actual pressure scales with q). The
global normalization `y_norm = (y - y_mean) / y_std` uses dataset-wide constants
and therefore does not neutralize the Re-dependence of the signal magnitude.
A per-sample normalization by the dynamic-pressure proxy (which can be inferred
from `log(Re)` in dim 13 of the input features) would make the model learn
Cp-like fields rather than absolute pressure fields. At test time, the model
predicts in normalized space and the denormalization multiplies back by the
sample's dynamic pressure proxy.

**Exact code change:**
```python
# In training loop, after loading x and y:
# log_Re is at x[:, :, 13] (all nodes share the same flow condition)
# Use the mean across real nodes to get a stable scalar
log_re = (x * mask.unsqueeze(-1).float())[:, :, 13].sum(dim=1) / mask.sum(dim=1).float().clamp(min=1)  # [B]
re_scale = torch.exp(log_re).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

# Optionally also normalize by y_std globally, then further divide by re_scale
# to get a Cp-like normalized target
y_re_norm = y / (re_scale.clamp(min=1e3) * 0.01)  # 0.01 is a tunable scale factor
# Then use standard global normalization on top:
y_norm = (y_re_norm - stats["y_mean"]) / stats["y_std"]
```

Note: this changes the target distribution seen by the model. The `y_mean` and
`y_std` in `stats.json` were computed over un-scaled targets; they would need to
be recomputed over the re-normalized targets, OR the student can compute running
statistics during training. Alternatively, skip the global normalization and use
only the per-sample Re scaling with a fixed reference scale.

This hypothesis requires careful implementation to ensure `data/scoring.py`
still receives predictions in the correct space for denormalization. The model
contract must remain: predict in `(y - y_mean) / y_std` space. Therefore the
re-scaling must be applied inside the loss without changing the model output
space — or a second normalization step must be applied consistently at both
train and eval time.

**Simpler implementation variant:** Instead of changing target normalization,
add `log_re` as an additional conditioning signal injected into the model via
a learned FiLM (Feature-wise Linear Modulation) layer after the preprocess MLP:
```python
# After preprocess MLP output fx [B, N, n_hidden]:
# Condition on log_Re (already in x[:, :, 13] after normalization)
re_cond = x_norm[:, :, 13:14]  # [B, N, 1] — same for all nodes in a sample
re_gamma = nn.Linear(1, n_hidden)(re_cond)   # [B, N, n_hidden]
re_beta  = nn.Linear(1, n_hidden)(re_cond)   # [B, N, n_hidden]
fx = fx * (1 + re_gamma) + re_beta
```
This FiLM conditioning is architecturally cleaner and does not touch the
normalization contract.

**Predicted impact:** -5 to -15% on `val_re_rand`. The FiLM variant is the
safer implementation and should be tried first.

**Split most helped:** `val_re_rand` (stratified Re holdout).

**Risk / failure mode:** FiLM conditioning doubles the parameter count of the
preprocess pathway for the Re dimension — small in absolute terms but adds
two Linear(1, n_hidden) modules. If the model already uses log(Re) implicitly
through the input features, FiLM may not add new information. A diagnostic:
does removing dim 13 from the input worsen Re-holdout performance significantly?

---

## Hypothesis 11: Per-Channel Output Heads with Separate Surface/Volume Decoders

**Family:** Architecture (multi-head decoding)

**Rationale.**
The baseline's last `TransolverBlock` has a single MLP2 output head that jointly
predicts [Ux, Uy, p]. But pressure (p) and velocity (Ux, Uy) obey fundamentally
different PDEs at the surface (p satisfies a Neumann-like condition; velocity
satisfies no-slip). Separate output heads per channel — or a surface-specific
decoder branch — allow the model to learn channel-specific inductive biases.
This is standard practice in multi-task learning (hard parameter sharing in
shared body, separate heads) and was used in Neural Process variants for PDE
fields. The cost is minimal: three extra Linear layers at the output.

**Exact code change in `Transolver` and `TransolverBlock`:**
```python
# In TransolverBlock.__init__ (last_layer=True branch):
# Old: one mlp2 head
#   self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
#                              nn.Linear(hidden_dim, out_dim))
# New: separate heads
self.mlp2_Ux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(),
                               nn.Linear(hidden_dim//2, 1))
self.mlp2_Uy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(),
                               nn.Linear(hidden_dim//2, 1))
self.mlp2_p  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                               nn.Linear(hidden_dim, 1))  # wider head for p

# In forward:
Ux_pred = self.mlp2_Ux(self.ln_3(fx))
Uy_pred = self.mlp2_Uy(self.ln_3(fx))
p_pred  = self.mlp2_p(self.ln_3(fx))
return torch.cat([Ux_pred, Uy_pred, p_pred], dim=-1)
```

Optionally add a surface-specific p decoder that takes `is_surface` as a gate:
```python
# p head with surface conditioning:
surf_gate = is_surface.float().unsqueeze(-1)  # [B, N, 1]
p_pred = self.mlp2_p_vol(h) * (1 - surf_gate) + self.mlp2_p_surf(h) * surf_gate
```

**Predicted impact:** -5 to -10% on `val_avg/mae_surf_p`. The pressure head
benefits most since p has the largest dynamic range and the most distinct
boundary behavior.

**Split most helped:** All splits (pressure head improvement is universal),
but most pronounced on `val_geom_camber_*` where the pressure field near the
surface has the most OOD structure.

**Risk / failure mode:** The surface-gated p decoder requires passing
`is_surface` through `TransolverBlock.forward`, which already requires the H5
thread-through change. These two hypotheses can be combined or run sequentially.
If run alone (without H5), the surface gate is only on the output head, which
is simpler and lower risk.

---

## Hypothesis 12: Dropout Regularization for Unseen-Camber Generalization

**Family:** Data / Regularization

**Rationale.**
The baseline uses `dropout=0.0`. With only 1499 training samples across 3 domains,
the model may memorize the training camber distribution rather than learning a
general pressure-field mapping from geometry. Dropout in the attention output
projection and MLP layers acts as a stochastic ensemble, reducing co-adaptation
of features and improving OOD generalization. This is especially relevant for
the `val_geom_camber_*` splits which test unseen front-foil camber values. The
effect size of dropout for transformers in low-data settings is well-documented
(Dosovitskiy et al. ViT 2020: dropout=0.1 helps on small datasets; He et al.
MAE 2022: no dropout needed at scale — implying dropout matters more when data
is scarce).

**Exact code change:**
```python
# In model_config, change dropout:
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    dropout=0.1,    # was 0.0
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

The existing `PhysicsAttention.__init__` already accepts `dropout` and passes
it to both `self.dropout = nn.Dropout(dropout)` and to `F.scaled_dot_product_attention`.
No other code changes needed.

**Hyperparameters:**
- Primary run: `dropout=0.1`
- Follow-up if no improvement: `dropout=0.05`
- Follow-up if improvement: `dropout=0.15` or `dropout=0.2`

**Predicted impact:** -3 to -10% on the camber OOD splits. Expected to hurt
slightly on `val_single_in_dist` (in-distribution, dropout hurts at test time)
but this is offset by improvement on the harder splits.

**Note:** dropout is only active during training; `model.eval()` disables it
during validation and test evaluation. The existing code in `evaluate_split`
calls `model.eval()` correctly, so no risk of eval-time dropout.

**Split most helped:** `val_geom_camber_rc` and `val_geom_camber_cruise`.

**Risk / failure mode:** With 30-minute wall clock and 50-epoch cap, a dropout
model may not converge as far as the baseline in the same time budget.
If `val_avg/mae_surf_p` is worse at epoch 50 but the learning curve shows
dropout catching up, extend the run or reduce dropout to 0.05.

---

## Summary Table

| # | Family | Hypothesis | Key Change | Primary Split | Expected Delta |
|---|--------|-----------|------------|---------------|----------------|
| 1 | Loss | Huber loss `delta=0.5` | MSE→Huber in normalized space | val_re_rand | -5 to -15% |
| 2 | Loss | Per-sample inv-std weighting | Sample weights by 1/y_std | val_re_rand | -5 to -10% |
| 3 | Loss | Surface MAE + p-channel weight | surf loss = MAE × p_weight | all splits | -5 to -20% |
| 4 | Optim | Gradient clip + LR warmup | clip_norm=1.0, warmup_epochs=5 | val_single_in_dist | -2 to -8% |
| 5 | Arch | Surface-biased slice attention | Concat is_surface to slice proj | val_geom_camber_* | -5 to -15% |
| 6 | Arch | Wider/shallower (256d, 3L) | n_hidden=256, n_layers=3 | val_geom_camber_* | -10 to -20% |
| 7 | Arch | Deeper/narrower (96d, 8L) | n_hidden=96, n_layers=8 | val_re_rand | -5 to +5% |
| 8 | Arch | More slice tokens (128) | slice_num=128 | val_geom_camber_rc | -3 to -10% |
| 9 | Input | Fourier positional encoding | Concat RFF of (x,z) coords | val_geom_camber_cruise | -5 to -15% |
| 10 | Input | FiLM Re-conditioning | FiLM layer on log(Re) | val_re_rand | -5 to -15% |
| 11 | Arch | Per-channel output heads | Separate decoders for Ux,Uy,p | all splits | -5 to -10% |
| 12 | Reg | Dropout 0.1 | dropout=0.0 → 0.1 | val_geom_camber_* | -3 to -10% |

---

## Recommended Priority Order for 8 Idle Students

Given 8 students with 30-minute/50-epoch budget:

1. **H3** (Surface MAE + p-channel weight) — highest expected impact, directly
   aligns training with the primary metric. Low implementation risk.
2. **H4** (Gradient clip + warmup) — zero-cost stability improvement. Should
   be combined with H3 or run alone as a fast diagnostic.
3. **H1** (Huber loss) — orthogonal to H3 (Huber on all nodes, H3 only on surface).
4. **H5** (Surface-biased slice attention) — architecturally motivated, moderate
   implementation complexity.
5. **H6** (Wider/shallower 256d) — capacity bet; run as a 50-epoch screening run.
6. **H9** (Fourier positional encoding) — input augmentation, independent of
   loss and architecture changes.
7. **H10** (FiLM Re-conditioning) — targets Re-holdout split specifically.
8. **H12** (Dropout 0.1) — low-cost regularization; run as diagnostic.

H2, H7, H8, H11 are secondary and should be assigned once initial results from
the primary 8 are in.
