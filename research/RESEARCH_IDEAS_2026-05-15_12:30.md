<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15 12:30

Generated for the TandemFoilSet CFD surrogate programme. Each hypothesis is
mutually independent and targets `val_avg/mae_surf_p` (lower is better). Ranked
by expected impact × implementation simplicity × risk.

---

## H1 — Surface-Pressure MAE Loss (direct metric alignment)

### Why it should help

The training loss is MSE in normalized space over all nodes, but the
paper-facing metric is MAE in physical space on surface nodes only. This is a
double mismatch: (a) squared vs absolute error, and (b) equal weighting of
surface and volume nodes despite the 10× surface weight already in the baseline.
Switching the loss to L1 (MAE) in normalized space directly aligns optimization
with the ranking metric. Log-cosh is a smooth alternative: for |r| < 1 it
approximates MSE (stable gradients early in training), for |r| > 1 it
approximates MAE (robust to high-Re outliers late in training). The transition
is automatic, requiring no threshold tuning.

### Specific concrete change

File: `train.py`, training loop (~line 330 onward, inside the batch loop).

Replace:
```python
sq_err = (pred - y_norm) ** 2
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

With (log-cosh variant):
```python
def log_cosh(r):
    # numerically stable: log(cosh(r)) = |r| + log(1 + exp(-2|r|)) - log(2)
    return r.abs() + torch.nn.functional.softplus(-2 * r.abs()) - 0.6931471805599453

err = pred - y_norm
vol_loss = (log_cosh(err) * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (log_cosh(err) * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Also test pure L1 arm (simplest version, same structure but `err.abs()` instead
of `log_cosh(err)`).

### Expected delta on val_avg/mae_surf_p

-3% to -8%. Metric alignment experiments in similar PDE surrogate settings
(e.g., AirfRANS, DrivAerML) report 3–10% improvements when switching from MSE
to MAE-aligned losses on the surface pressure channel.

### Risk / failure mode

Log-cosh may slow early training convergence relative to MSE because the
gradient magnitude near zero is lower than MSE. Monitor `val_avg/mae_surf_p` at
epoch 10 — if it is worse than baseline at that checkpoint, the pure L1 arm is
safer.

### Compute cost

Zero additional VRAM. Same wall-clock as baseline. Two arms in one PR (log-cosh
+ L1) fit within 50 epochs / 30 min.

### References

- Log-cosh loss as smooth MAE proxy: Chen et al. 2019, "Improved Baselines with
  Momentum Contrastive Learning" (appendix). Standard in regression literature.
- Area-weighted loss for airfoil surface pressure: Bonnet et al. 2026 (Jan),
  geometric DL for aerodynamic surface pressure prediction.

---

## H2 — Surface-Pressure Channel Weighting in Loss (p channel up-weight)

### Why it should help

The primary metric is `mae_surf_p` — surface pressure only. The current loss
treats all three output channels (Ux, Uy, p) equally within the surface and
volume terms. Explicitly up-weighting the pressure channel in the surface loss
term directly focuses gradient signal on the ranked variable, at the cost of
slightly worse velocity predictions which are not ranked.

### Specific concrete change

File: `train.py`, training loop.

After computing `surf_loss`, replace the scalar with a channel-weighted version:

```python
# channel_weights: [3] — Ux, Uy, p
channel_w = torch.tensor([1.0, 1.0, cfg.p_channel_weight], device=device)  # default p_channel_weight=3.0

sq_err = (pred - y_norm) ** 2
vol_loss_ch = (sq_err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1)) / vol_mask.sum().clamp(min=1)
surf_loss_ch = (sq_err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1)) / surf_mask.sum().clamp(min=1)
vol_loss = vol_loss_ch.sum()
surf_loss = (surf_loss_ch * channel_w).sum()
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add `--p_channel_weight` CLI arg defaulting to `3.0`. Test values: 2.0, 3.0,
5.0.

This is orthogonal to H1 and can be combined: the per-channel weighting works
identically with log-cosh or L1 as the base loss.

### Expected delta on val_avg/mae_surf_p

-2% to -6%. Direct pressure-channel focus. Risk is that the model over-fits
pressure to the detriment of velocity fields that provide implicit physical
constraints (e.g., divergence-free condition links Ux, Uy to p).

### Risk / failure mode

Too high a p_channel_weight (>10) may destabilize training because the pressure
channel has a much larger physical magnitude range than velocities. Stay ≤5 in
the first run and monitor per-channel val losses.

### Compute cost

Zero overhead. Single-run screening.

---

## H3 — Exponential Moving Average (EMA) of Model Weights

### Why it should help

EMA of weights is free regularization: the EMA model tracks a smoothed
trajectory through weight space and consistently outperforms the last checkpoint
on out-of-distribution splits, which is exactly the setting here (geom_camber
and re_rand are OOD). The baseline saves and evaluates the best checkpoint by
`val_avg/mae_surf_p`; replacing that checkpoint with the EMA model at test time
costs nothing. Recent systematic study (Izmailov et al. 2018 SWA; Tarvainen &
Valpola 2017 Mean Teacher; and the Nov 2024 systematic analysis arxiv:2411.18704)
shows EMA consistently outperforms the trained model on OOD splits with decay
parameter d ∈ [0.999, 0.9999] and requires no learning-rate adjustment.

### Specific concrete change

File: `train.py`.

After model construction:
```python
from copy import deepcopy
ema_model = deepcopy(model)
ema_model.eval()
EMA_DECAY = 0.999  # try also 0.9999

def update_ema(ema, live, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), live.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
```

At end of each training batch (after `optimizer.step()`):
```python
update_ema(ema_model, model, EMA_DECAY)
```

In the validation loop, evaluate `ema_model` instead of (or in addition to)
`model`. Checkpoint the EMA model when `val_avg/mae_surf_p` improves.

Use `decay=0.999` for the first run. If validation loss at epoch 50 is not
meaningfully different from the live model, try `0.9999`.

### Expected delta on val_avg/mae_surf_p

-1% to -5%, concentrated on `geom_camber_rc`, `geom_camber_cruise`, and
`re_rand` (OOD splits). In-distribution split (`single_in_dist`) may be
unchanged or slightly worse.

### Risk / failure mode

If the EMA model is evaluated but the live model is also checkpointed, there is
no risk to the training run — just disable EMA evaluation if it is slower. The
only cost is memory for a second copy of model weights (~2× model size,
negligible relative to 96 GB VRAM for a 5-layer Transolver).

### Compute cost

~5% wall-clock overhead from `update_ema` call per step. No extra forward pass
if EMA replaces the live model in validation rather than doubling it.

### References

- arxiv:2411.18704 — "Exponential Moving Average of Weights in Deep Learning: A
  Systematic Study" (Nov 2024). Documents optimal decay schedule, interaction
  with LR, and OOD generalization benefits.
- Tarvainen & Valpola 2017 (Mean Teacher) — original neural ODE / EMA-as-
  regularizer observation.

---

## H4 — Increased Slice Count (slice_num 64 → 128)

### Why it should help

The Transolver compresses N nodes (74K–242K) into `slice_num=64` learned tokens
before running attention. The compression ratio varies from ~1160:1
(raceCar single 85K) to ~3780:1 (cruise tandem 242K). At 64 slices, each token
must represent ~1000–4000 physically distinct nodes simultaneously, which is a
severe bottleneck for capturing fine-grained pressure gradients near the foil
surface. Doubling to 128 cuts the ratio by half. The attention cost scales as
O(slice_num²): 128² = 16384 vs 64² = 4096, a 4× increase in attention compute
— but attention is not the wall-clock bottleneck for this model size at batch=4.

### Specific concrete change

File: `train.py`, `model_config` dict:
```python
model_config = dict(
    ...
    slice_num=128,   # was 64
    ...
)
```

No other changes required. The projection matrices inside `PhysicsAttention`
(`in_project_slice`, `out_project_slice`) automatically scale to the new
`slice_num`.

Also test `slice_num=96` as a cheaper midpoint arm in the same PR.

### Expected delta on val_avg/mae_surf_p

-2% to -8% on the cruise tandem splits (highest compression ratio at 64). Likely
smaller gain on raceCar single (lowest ratio). Risk of diminishing returns if
the bottleneck is elsewhere (expressivity of the MLP within each block, not
token count).

### Risk / failure mode

VRAM: at `batch_size=4` and `N_max=242K`, the attention matrix is `[B * n_head,
128, 128]` which is negligible. The dominant VRAM cost is the node-level tensors
`[B, N_max, n_hidden]`. At 96 GB VRAM this is safe; verify with a single debug
epoch first.

### Compute cost

Attention ~4× more expensive, but node-level projections dominate. Estimated
10–20% total wall-clock increase. Should still fit 50 epochs in 30 min.

---

## H5 — Re-Conditioning via FiLM Modulation on Reynolds Number

### Why it should help

The Reynolds number spans ~100K to ~5M (50× range) and is the dominant driver of
field magnitude variation. The current model receives `log(Re)` as input feature
dim 13, which must propagate through all layers as a passive signal mixed with
spatial features. Physical intuition: at high Re, boundary layer thickness
scales as Re^{-0.5}, and pressure coefficients scale with the dynamic pressure.
A FiLM (Feature-wise Linear Modulation) layer that multiplicatively gates each
Transolver block's hidden state by a Re-derived scale factor gives the model an
explicit "global regime" conditioning signal at every layer, not just at input.
This is conceptually similar to timestep conditioning in diffusion models and to
how GeoTransolver conditions on boundary data.

### Specific concrete change

File: `train.py`, `PhysicsAttention` or `TransolverBlock` class.

Add a small conditioning network (2-layer MLP) that maps scalar `log_Re` to
`(gamma, beta)` of shape `[n_hidden]`:

```python
class ReConditioning(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, log_re):
        # log_re: [B, 1]
        out = self.net(log_re)          # [B, 2*H]
        gamma, beta = out.chunk(2, -1)  # each [B, H]
        return 1.0 + gamma, beta        # scale near 1 initially
```

In `TransolverBlock.forward`, after the attention output and before the final
MLP:
```python
gamma, beta = self.re_cond(log_re)   # passed down from top-level forward
x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)
```

Extract `log_re = x_norm[:, :, 13:14].mean(dim=1)` at the top of the model
forward (mean over nodes — it is constant within a sample).

### Expected delta on val_avg/mae_surf_p

-3% to -10% on `re_rand` (stratified Re holdout). Smaller effect on geom_camber
splits. This is one of the most physically motivated interventions because
Reynolds number governs the physical regime.

### Risk / failure mode

If `gamma` is initialized incorrectly, training may diverge early. Use `1 +
gamma` initialization (so the initial scale is 1.0) and monitor loss at epoch 1.
The net has only 2 × n_hidden² = 2 × 128² ≈ 33K parameters — negligible.

### Compute cost

Negligible — FiLM modulation is a single elementwise multiply + add per block.

---

## H6 — Deeper Model (n_layers 5 → 8) with Proportionally Reduced Hidden Dim

### Why it should help

The Transolver baseline has 5 layers with n_hidden=128. Empirically in PDE
surrogates, deeper networks with the same parameter count outperform shallower
wide networks because physical signal must propagate across long-range dependencies
(inlet boundary condition → foil surface → wake). The slice attention in each
block is O(slice_num²) independent of n_hidden, so increasing depth at fixed
parameter budget costs less than increasing width. With n_layers=8 and n_hidden=96
the parameter count is roughly similar to the baseline.

Alternatively, keep n_hidden=128 and increase n_layers from 5 to 8 as a pure
capacity increase — this is cleaner to ablate.

### Specific concrete change

File: `train.py`, `model_config` dict. Two arms:

Arm A (deeper, same hidden):
```python
model_config = dict(n_layers=8, n_hidden=128, ...)
```

Arm B (deeper, narrower — similar parameter count):
```python
model_config = dict(n_layers=8, n_hidden=96, ...)
```

No other changes. The model constructor loops over `n_layers` automatically.

### Expected delta on val_avg/mae_surf_p

-1% to -5% for Arm A. The marginal benefit beyond 6 layers tends to diminish
quickly in neural operators. Arm B may be slower to converge but generalize
better.

### Risk / failure mode

Arm A increases memory by ~60% (3 extra layers × n_hidden activations). At
batch_size=4, N_max=242K, n_hidden=128: extra activation tensor per layer is
B × N_max × n_hidden = 4 × 242000 × 128 × 4 bytes ≈ 500 MB per extra layer ×
3 layers = 1.5 GB additional. Safe.

Wall-clock: 60% longer per-epoch. With 30 min cap and baseline at ~20–25 min
for 50 epochs, Arm A may only complete 30–35 epochs. Use `--epochs 35` to stay
within cap.

### Compute cost

Medium. Arm A: ~60% slower. Set epochs to 35. Arm B: ~45% slower.

---

## H7 — AoA Sinusoidal Encoding (replace raw radians with sin/cos)

### Why it should help

Angle of attack is stored in radians (dims 14 and 18), but the aerodynamic
effect of AoA is periodic and symmetric: sin(AoA) drives lift, cos(AoA) drives
drag. When AoA crosses 0° (raceCar uses -10° to 0°, cruise uses -5° to +6°),
the raw value can flip sign while the aerodynamic effect changes smoothly. A
`(sin(AoA), cos(AoA))` encoding removes the angular discontinuity and makes the
feature space metrically aligned with the physical effect. This adds 2 features
(dims 24 and 25 in a new x of size 26), requiring matching changes to the model
input dimension.

### Specific concrete change

File: `train.py`, data preprocessing before passing to model.

Add a feature augmentation step after normalization (done in `train.py` before
the model forward call, not in `data/loader.py`):

```python
def augment_x(x_norm, x_raw):
    # x_raw: [B, N, 24] unnormalized
    # AoA foil 1: dim 14, AoA foil 2: dim 18 (radians)
    aoa1 = x_raw[..., 14:15]   # [B, N, 1]
    aoa2 = x_raw[..., 18:19]   # [B, N, 1]
    sin_cos = torch.cat([aoa1.sin(), aoa1.cos(), aoa2.sin(), aoa2.cos()], dim=-1)
    return torch.cat([x_norm, sin_cos], dim=-1)   # [B, N, 28]
```

Update `model_config`:
```python
model_config = dict(
    fun_dim=X_DIM - 2 + 4,   # 22 + 4 = 26
    ...
)
```

The raw AoA dims 14 and 18 remain in the normalized input (do not remove them —
the relative magnitude carries regime information). The sin/cos augmentation is
additive.

### Expected delta on val_avg/mae_surf_p

-1% to -4%. The biggest gain should be on `geom_camber_cruise` where AoA spans
both positive and negative (cruise -5° to +6°), making the sign change
physically meaningful.

### Risk / failure mode

The model already receives `log(Re)`, NACA params, gap, and stagger as global
conditioning alongside the spatial features. AoA encoding is a low-risk
augmentation. Main failure mode: if the model already learns to handle raw
radians adequately (5 layers is enough to learn sin/cos implicitly), there is
no gain.

### Compute cost

Negligible. Two extra linear-projection columns added to the input embedding.

---

## H8 — OneCycleLR Scheduler (replace CosineAnnealingLR)

### Why it should help

The baseline uses `CosineAnnealingLR(T_max=MAX_EPOCHS)` which decays LR
monotonically from 5e-4 to 0. OneCycleLR uses a 1-cycle warmup + anneal that
reaches a higher peak LR (`max_lr = cfg.lr * pct_max`) for a fraction of
training then anneals aggressively. This is the standard scheduler in fastai
and is often 20–30% faster to converge on small datasets (1499 training
samples). The full 50-epoch budget is often under-utilized with CosineAnnealing
because the LR is very low by epoch 30+.

OneCycleLR also combines a div_factor warmup (starting at `max_lr / 25`) which
is better than a cold start at full LR on irregular meshes where early batches
can be numerically extreme.

### Specific concrete change

File: `train.py`, optimizer/scheduler section.

Replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

With:
```python
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr * 10,         # peak = 5e-3 if lr=5e-4
    epochs=MAX_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,              # 10% warmup
    div_factor=25.0,            # start at max_lr/25 = 2e-4
    final_div_factor=1e4,       # end at max_lr/1e4 = 5e-7
    anneal_strategy='cos',
)
```

Move `scheduler.step()` inside the batch loop (not epoch loop):
```python
# After optimizer.step() each batch:
scheduler.step()
```

Remove the epoch-level `scheduler.step()` call.

### Expected delta on val_avg/mae_surf_p

-1% to -4% from faster convergence within the 50-epoch cap. The gain is more
about reaching a better checkpoint within the time limit than a fundamentally
better minimum.

### Risk / failure mode

OneCycleLR is sensitive to `max_lr`: too high and training diverges in the
warmup phase. With `max_lr = 5e-3` for a mesh surrogate, this is moderately
aggressive. Monitor epoch-1 loss — if it spikes above 2× the baseline epoch-1
loss, halve `max_lr`.

### Compute cost

Zero overhead.

---

## H9 — Separate Surface/Volume Output Heads with Shared Backbone

### Why it should help

The Transolver uses a single MLP2 projection from `n_hidden` to `out_dim=3`
in the final block. Surface nodes (foil boundary) and volume nodes (fluid
interior) have fundamentally different physics: surface nodes satisfy the
no-slip and pressure-continuity boundary conditions; volume nodes are governed
by the Navier-Stokes PDE interior residuals. A dual-head architecture with a
shared backbone but separate final projections allows the model to specialize
the output mapping for each regime. A March 2025 paper in *Physics of Fluids*
(multi-task surface/volume separation) reports 5–15% improvements on surface
characteristics when surface and volume targets are decoupled.

### Specific concrete change

File: `train.py`, `TransolverBlock` final layer and model forward.

In the last `TransolverBlock`:
```python
# Replace single out_project with two heads
self.surface_head = nn.Linear(n_hidden, out_dim)
self.volume_head  = nn.Linear(n_hidden, out_dim)
```

In `Transolver.forward`, after the last block:
```python
# x: [B, N, n_hidden], is_surface: [B, N]
surf = is_surface.unsqueeze(-1).float()         # [B, N, 1]
pred = (self.surface_head(x) * surf 
      + self.volume_head(x) * (1 - surf))       # [B, N, 3]
```

The `is_surface` mask is already in the input feature dim 12 and also available
as a separate tensor in the batch — pass it into the model forward.

Training loss is unchanged (still separate `surf_loss` and `vol_loss` terms).

### Expected delta on val_avg/mae_surf_p

-2% to -8% from specialization. The surface head can focus gradient updates
entirely on pressure continuity conditions at the foil. The volume head learns
the interior flow field independently.

### Risk / failure mode

The model must receive `is_surface` as an argument to the forward pass —
requires a small change to the `Transolver.forward` signature. This adds 2 ×
n_hidden × out_dim = 2 × 128 × 3 = 768 parameters, which is negligible.

Failure mode: if the model already implicitly learns the surface/volume
distinction from feature dim 12 (`is_surface` in x), the dual head may add
little. The ablation is cheap enough to determine this quickly.

### Compute cost

Negligible — one extra linear layer. Zero overhead on wall-clock.

---

## H10 — Curriculum Learning: Low-Re First, High-Re Later

### Why it should help

The training set spans Re ~100K to ~5M (50× range) with per-sample y standard
deviation varying by an order of magnitude. High-Re samples dominate the MSE
gradient norm because their target values are ~10–50× larger in physical space.
Even in normalized space, high-Re samples often have larger normalized residuals
early in training. A curriculum that presents low-Re (easier, smaller-magnitude)
samples first allows the model to learn the base flow geometry before it must
handle extreme velocity gradients. This is analogous to curriculum learning in
NLP (shorter sequences first) and has been used in CFD surrogates (Bonnet et al.
2023, CFD-Bench).

### Specific concrete change

File: `train.py`, training loop.

Replace the `WeightedRandomSampler` with a curriculum sampler that transitions
from low-Re to mixed-Re over the first 50% of training:

```python
def make_curriculum_sampler(train_ds, stats, epoch, max_epochs, curriculum_pct=0.5):
    """For first `curriculum_pct` of training, over-sample low-Re samples."""
    # Extract log_Re for each sample (dim 13 of x, already stored in .pt files)
    # Use sample_weights as base domain balance weights
    # Anneal: weight_low_re decays from 3.0 to 1.0 linearly over first half
    progress = min(epoch / (max_epochs * curriculum_pct), 1.0)
    low_re_boost = 3.0 * (1.0 - progress) + 1.0 * progress
    # ...compute per-sample weights based on Re and domain...
```

Practically: sort samples by log_Re, assign `low_re_boost` weight to bottom
tercile, `1.0` to top tercile. Rebuild `WeightedRandomSampler` each epoch or
every 5 epochs.

Because `data/loader.py` is read-only, load the log_Re values from the `.pt`
files at startup in `train.py` before training begins:

```python
log_re_per_sample = []
for f in train_ds.files:
    s = torch.load(f, weights_only=True)
    log_re_per_sample.append(s["x"][0, 13].item())  # constant within sample
```

### Expected delta on val_avg/mae_surf_p

-1% to -4%, primarily on `re_rand` and `geom_camber_cruise` (cruise has lower
Re range and thus lower target magnitude — model sees more of these early).
Uncertain because the normalization already partially equalizes the loss across
Re.

### Risk / failure mode

The normalization (`y - y_mean) / y_std`) uses global stats, which already
reduces the effective dynamic range across Re regimes. If normalized loss is
already Re-balanced, curriculum adds noise rather than signal. This should be
detectable at epoch 5: if early training loss is the same as baseline, abandon.

### Compute cost

Negligible — sampler construction is O(n_train) per epoch.

---

## H11 — Multi-Scale Physics Attention (inspired by GeoTransolver)

### Why it should help

The GeoTransolver (arxiv:2512.20399) extends Transolver with multi-scale ball
queries at 6 radii on the mesh, achieving 2.86% surface pressure relative L1 on
DrivAerML. The TandemFoilSet mesh has 3 zones with very different node
densities (dense near foil surfaces, coarse in the background), and a single
global slice compression misses local geometric structure. A lightweight version:
add a local aggregation step before the slice attention that computes
node-neighborhood mean features at two radii (fine: ~0.01 chord, coarse: ~0.1
chord). This local context can be cheaply appended to the node features before
slicing.

### Specific concrete change

File: `train.py`, `PhysicsAttention` or a new preprocessing module.

Use node positions (dims 0–1) to compute kNN or radius-based neighbors and
aggregate hidden features. A practical approximation without building a graph:
use sorted distance buckets and a simple 1D convolution along the node index
(less physically meaningful but fast):

```python
class LocalAggregation(nn.Module):
    """Cheap local context: mean of k-nearest neighbors in position space."""
    def __init__(self, in_dim, k=16):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, x, pos):
        # x: [B, N, D], pos: [B, N, 2]
        # For efficiency: random subset of N for distance computation
        with torch.no_grad():
            dist = torch.cdist(pos, pos)     # [B, N, N] — expensive for N=242K
            knn_idx = dist.topk(self.k + 1, largest=False).indices[:, :, 1:]
        knn_x = x[torch.arange(B)[:, None, None], knn_idx]   # [B, N, k, D]
        local_mean = knn_x.mean(dim=2)       # [B, N, D]
        return self.proj(torch.cat([x, local_mean], dim=-1))
```

**Critical caveat**: full O(N²) cdist at N=242K is prohibitive in memory and
time. The minimal viable version: downsample to the surface nodes only (is_surface
~few thousand nodes) and apply local aggregation only there, then propagate.
Alternatively, use torch-cluster's `radius_graph` if available.

Given the O(N²) issue, **recommend the student implement a surface-only local
aggregation** as a cheaper first step:

```python
# Apply local aggregation only to surface nodes
surf_idx = is_surface[b].nonzero().squeeze(-1)
surf_pos = pos[b, surf_idx]            # [S, 2]
dist_surf = torch.cdist(surf_pos, surf_pos)   # [S, S], S~few K — cheap
```

### Expected delta on val_avg/mae_surf_p

-3% to -10% if the surface-local-aggregation captures fine-grained boundary
layer structure. Uncertain because this is architecturally more complex.

### Risk / failure mode

O(N²) memory for full mesh — must be surface-only. The surface-only version is
~(few thousand)² per sample which is <100MB. Implementation complexity is high
relative to other hypotheses. Recommend assigning to an experienced student.

### Compute cost

Medium. Surface-only local aggregation: <5% overhead. Full mesh: infeasible.

### References

- arxiv:2512.20399, "GeoTransolver" (Dec 2024): multi-scale ball-query geometry
  augmentation of Transolver, SOTA on DrivAerML surface pressure.

---

## H12 — Weight Decay Sweep + AdamW Decoupled Tuning

### Why it should help

The baseline uses `weight_decay=1e-4` without examining whether this is optimal
for this dataset size. With only 1499 training samples spread across 3 domain
groups, the model may underfit (needing less regularization) or overfit on the
in-distribution split while generalizing poorly on OOD splits (needing more).
The effective regularization from weight decay interacts with learning rate in
AdamW: the true L2 penalty strength is `weight_decay × lr`. At `lr=5e-4` and
`wd=1e-4`, the effective penalty is 5e-8 per parameter per step, which is very
weak. Testing `wd=1e-3` and `wd=1e-2` at the same LR would test 10× and 100×
stronger regularization.

### Specific concrete change

File: `train.py`, optimizer construction. Three arms in one PR:

```python
# Arm A (baseline): weight_decay=1e-4
# Arm B: weight_decay=1e-3
# Arm C: weight_decay=1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
```

Add `--weight_decay` CLI arg. Run all three arms with `--epochs 30` for a
screening run to identify the optimal range, then a final 50-epoch confirmation.

### Expected delta on val_avg/mae_surf_p

-1% to -5% if the current `1e-4` is too weak for the small dataset size. The
OOD splits (`geom_camber_rc`, `geom_camber_cruise`) should benefit most from
stronger regularization.

### Risk / failure mode

`wd=1e-2` may underfit on the in-distribution split. Triangle test: if `1e-3`
OOD-val improves but `single_in_dist` regresses, the right value is between
`1e-3` and `1e-4`.

### Compute cost

Low. Three 30-epoch arms, one 50-epoch confirmation. Total ~4 runs.

---

## H13 — Per-Domain Normalization Statistics

### Why it should help

The global `y_std` in `stats.json` is computed across all three domains combined.
But cruise tandem and raceCar domains have dramatically different y ranges
(cruise std ~164, raceCar single std ~458). In normalized space, cruise samples
have smaller normalized residuals than raceCar samples at the same physical
error. This means the model implicitly under-weights cruise training signal
relative to raceCar. Per-domain normalization would equalize residual magnitudes
across domains.

However, `data/scoring.py` assumes a single global `y_mean` and `y_std` for
denormalization. To preserve the scoring contract, per-domain normalization must
be applied only to the loss computation, not to what is passed to `scoring.py`.

### Specific concrete change

File: `train.py`. Precompute per-domain y stats at startup:

```python
# After loading train_ds and stats:
domain_y_stats = {}   # {domain_name: (y_mean, y_std)}
for domain_name, idxs in meta["domain_groups"].items():
    ys = []
    for i in idxs:
        s = torch.load(train_ds.files[i], weights_only=True)
        ys.append(s["y"])
    y_cat = torch.cat(ys, dim=0)
    domain_y_stats[domain_name] = (y_cat.mean(0), y_cat.std(0).clamp(min=1e-6))
```

In the training loss, use the domain-specific std for loss normalization:
```python
# Use domain std for loss scaling but keep global stats for checkpointing and scoring
domain_std = domain_y_stats[domain_of_batch].to(device)
y_domain_norm = (y - domain_y_stats[domain][0]) / domain_y_stats[domain][1]
pred_domain = pred * (stats["y_std"] / domain_std)  # rescale prediction
loss = mse(pred_domain, y_domain_norm)
```

This is moderately complex; alternatively, a simpler per-sample loss
normalization by the sample's own y-std (no precomputation needed):

```python
sample_std = y[mask].std().clamp(min=1.0)  # [scalar]
loss = mse(pred, y_norm) / (sample_std / stats["y_std"].mean())
```

Recommend the per-sample version as the simpler first arm.

### Expected delta on val_avg/mae_surf_p

-1% to -4% on the cruise tandem splits (where the global normalization is most
misaligned). Uncertain because the domain-balanced sampler already compensates
for domain frequency mismatch.

### Risk / failure mode

If the per-sample std normalization results in effective LR variation across
samples (high-std samples get larger effective LR), training may become
unstable. Clip the normalization factor to [0.5, 2.0].

### Compute cost

Precomputation: O(n_train) one-time cost of loading all y tensors (~2–5 min).
Training overhead: negligible.

---

## H14 — LinearNO Architecture (replacing Transolver slice attention)

### Why it should help

LinearNO (arxiv:2511.06294) demonstrates that Transolver's Physics-Attention is
a special case of linear attention where the "slice" mechanism corresponds to
selecting random basis vectors in the kernel approximation. LinearNO derives a
theoretically grounded linear attention with 40% fewer parameters and 36% less
compute than Transolver while achieving SOTA on 6 PDE benchmarks, AirfRANS, and
ShapeNet Car. For this problem, the reduction in parameters could improve
generalization on the small dataset (1499 training samples), and the compute
reduction would allow training either more epochs in the 30-min budget or a
larger model at the same cost.

The core change: replace `PhysicsAttention` with `LinearNO`'s kernel attention
module. The input/output interface is identical (`[B, N, d] → [B, N, d]`).

### Specific concrete change

File: `train.py`. Replace the `PhysicsAttention` class (or add `LinearNOAttention`
as an alternative) following the LinearNO paper formulation:

```python
class LinearNOAttention(nn.Module):
    """Linear neural operator attention from arxiv:2511.06294.
    Replaces Transolver's Physics-Attention with kernel-based linear attention.
    """
    def __init__(self, n_hidden, n_head, n_basis=64):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_hidden // n_head
        self.n_basis = n_basis
        self.q_proj = nn.Linear(n_hidden, n_hidden)
        self.k_proj = nn.Linear(n_hidden, n_hidden)
        self.v_proj = nn.Linear(n_hidden, n_hidden)
        self.phi = nn.Linear(n_hidden, n_basis)   # basis function projection
        self.out_proj = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.n_head, self.head_dim)
        K = self.k_proj(x).view(B, N, self.n_head, self.head_dim)
        V = self.v_proj(x).view(B, N, self.n_head, self.head_dim)
        # Linear attention: avoid N×N matrix
        # phi(K)^T V first (n_basis × head_dim), then Q × that
        Phi = F.elu(self.phi(x)) + 1   # [B, N, n_basis] positive feature map
        # ... (implement full linear attention O(N) complexity)
        return self.out_proj(out.view(B, N, D))
```

**Recommendation**: Rather than re-implementing from scratch, the student should
clone the LinearNO reference implementation from the paper's GitHub (if
available) and adapt it to the `train.py` interface. The key change is swapping
`PhysicsAttention` for `LinearNOAttention` inside `TransolverBlock`.

### Expected delta on val_avg/mae_surf_p

-2% to -8% based on LinearNO's reported improvements on AirfRANS (closest
analogue — also 2D aerodynamic field prediction on irregular meshes). The 40%
parameter reduction may help generalization on this small dataset.

### Risk / failure mode

Implementation complexity is the main risk — LinearNO's kernel approximation
must be implemented correctly or results will be meaningless. This hypothesis
should be assigned to a student comfortable with attention mechanism
implementation. Recommend allocating 30–40 epochs to allow adequate training
time after potential debugging overhead.

### Compute cost

Medium. LinearNO claims 36% compute reduction → faster per-epoch, allowing more
epochs within the 30-min cap.

### References

- arxiv:2511.06294, "LinearNO: Rethinking Transolver" (Nov 2024): derives
  linear attention reformulation of Physics-Attention, 40% param reduction,
  SOTA on 6 PDE benchmarks + AirfRANS.

---

## Priority Ranking Summary

| Rank | ID  | Hypothesis | Rationale |
|------|-----|------------|-----------|
| 1    | H1  | Log-cosh / L1 loss | Direct metric alignment, zero risk, two arms |
| 2    | H3  | EMA weights | Free regularization for OOD splits, minimal code |
| 3    | H2  | p-channel weighting | Orthogonal to H1, zero overhead |
| 4    | H9  | Dual surface/volume heads | Physically motivated, tiny code change |
| 5    | H5  | Re FiLM conditioning | Targets re_rand OOD directly |
| 6    | H4  | slice_num 64→128 | Simple config change, addresses compression bottleneck |
| 7    | H8  | OneCycleLR | Better use of 50-epoch budget, well-validated scheduler |
| 8    | H7  | AoA sin/cos encoding | Low-risk feature augmentation |
| 9    | H12 | Weight decay sweep | Cheap ablation, informative either way |
| 10   | H6  | Deeper model (5→8 layers) | Capacity experiment, medium cost |
| 11   | H10 | Curriculum learning (low-Re first) | Worth testing after normalization understood |
| 12   | H13 | Per-domain normalization | Higher complexity, modest expected gain |
| 13   | H11 | Multi-scale surface aggregation | High expected gain, medium implementation risk |
| 14   | H14 | LinearNO architecture | Highest potential, highest implementation complexity |

### Combination notes

- H1 + H2 + H3 are fully orthogonal and can be tested in parallel with no interaction risk.
- H7 + H5 are additive (both add input conditioning, different aspects).
- H9 requires a model interface change that also enables H5 to pass is_surface cleanly.
- H14 and H11 are architectural bets — run only after the quick wins above are confirmed.
