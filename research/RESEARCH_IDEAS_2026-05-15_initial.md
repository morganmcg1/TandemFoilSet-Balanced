<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15 (Round 3 Fresh Slate)

Branch: `icml-appendix-willow-pai2i-24h-r3`
Generated: 2026-05-15
Context: Clean-slate launch. No prior PRs on this branch. Baseline = published Transolver
with n_layers=5, slice_num=64, n_hidden=128, AdamW lr=5e-4, MSE loss, surf_weight=10,
cosine annealing, no warmup, no grad clip, no EMA, global normalization.
Primary metric: `val_avg/mae_surf_p` (lower is better).

Ordered by confidence × predicted impact, highest first.

---

## 1. deeper-transolver

**Hypothesis:** Increasing Transolver depth from n_layers=5 to n_layers=8 (the depth used
in the original paper) will reduce `val_avg/mae_surf_p` because the baseline is capacity-
limited relative to the published best configuration.

**Predicted delta:** -8% to -15% on `val_avg/mae_surf_p`.

**Mechanism:** The Transolver paper reports L=8 as its standard configuration across all
evaluated PDEs and uses that depth in all published ablations. The baseline uses L=5,
which is 37% shallower. Each additional Transolver layer applies another round of physics-
slice attention followed by a feed-forward block, allowing deeper composition of local
physics features. For complex flows with wake interactions (tandem configuration) and
Reynolds numbers spanning 50x in range, the additional layers provide representational
capacity to capture multi-scale physics. This is the single change most directly supported
by the paper's own ablation evidence.

**Implementation (changes to train.py):**

```python
# In model_config dict, change:
n_layers=8,   # was 5
```

Keep all other hyperparameters identical to baseline. No architectural changes, no new
dependencies. The additional layers increase parameter count by ~60% and per-step compute
by ~60%, but fit well within 96 GB VRAM at batch_size=4.

**Compute notes:** Training time increases roughly proportional to depth (~60% longer per
epoch). At batch_size=4 on a 96 GB GPU, this remains feasible for a 24h window. Expect
~15-20 epochs within a 24h cap depending on mesh size mix per batch. Use
`--wandb_group deeper-transolver` for grouping.

**Risk:** Low. The paper's own ablation supports L=8. The only risk is that the baseline
was tuned for L=5 (lr, wd) and L=8 may need slightly lower lr to avoid instability. If
training loss diverges, reduce lr to 3e-4.

---

## 2. warmup-cosine-grad-clip

**Hypothesis:** Adding a 5-epoch linear LR warmup and gradient clipping (max_norm=1.0)
will reduce `val_avg/mae_surf_p` by stabilizing early training against the extreme pressure
value range (y std up to 2,077; p ranging -29,136 to +2,692 in physical space).

**Predicted delta:** -5% to -10% on `val_avg/mae_surf_p`.

**Mechanism:** The baseline jumps immediately to lr=5e-4 with no warmup, and applies no
gradient clipping. With target values spanning four orders of magnitude and high-Re samples
producing normalized gradients that are ~10x larger than low-Re samples, early SGD steps
are high-variance. Warmup linearly ramps lr from ~5e-6 to 5e-4 over the first 5 epochs,
giving the model time to reach a stable regime before large gradient steps. Gradient
clipping (max_norm=1.0) prevents rare but catastrophic high-Re outlier batches from
destroying early progress. Both techniques are standard practice for training instability
on heterogeneous-magnitude targets. The cosine schedule tail already exists; this adds a
warmup head and a safety rail.

**Implementation (changes to train.py):**

```python
# In Config dataclass, add:
warmup_epochs: int = 5
grad_clip: float = 1.0

# Replace the scheduler creation block with:
def get_lr_lambda(epoch):
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    # cosine after warmup
    progress = (epoch - cfg.warmup_epochs) / max(1, MAX_EPOCHS - cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)

# In training loop, after loss.backward(), add before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
```

Add `import math` if not present. No other changes required.

**Compute notes:** Zero additional VRAM or compute cost. Pure optimizer/scheduler change.
Run for the same number of epochs as baseline; this is a screening experiment.

**Risk:** Very low. Both techniques are well-established. The only risk is that the warmup
is too short (5 epochs) if the model needs more ramp time. If results are inconclusive,
try 10 epochs warmup.

---

## 3. surf-p-weighted-loss

**Hypothesis:** Upweighting the pressure channel (p) relative to velocity channels (Ux, Uy)
in the training loss — using per-channel weights [1, 1, 3] — will improve `val_avg/mae_surf_p`
because the primary metric cares only about surface pressure but the baseline MSE loss treats
all three output channels identically.

**Predicted delta:** -5% to -12% on `val_avg/mae_surf_p`. Velocity metrics may worsen
slightly as a side-effect.

**Mechanism:** The primary evaluation metric is surface pressure MAE (`mae_surf_p`), but
the training objective is unweighted MSE across all three channels. This creates an
objective mismatch: gradients from Ux and Uy errors are equally large as p gradients even
though they do not count toward the primary metric. In normalized space, after dividing by
global y_std, the three channels have roughly comparable variance (by construction), so
the loss is already nominally balanced. However, upweighting p by 3x directly aligns the
gradient direction with the evaluation metric. The loss modification is:

```
loss = vol_loss + surf_weight * surf_loss
sq_err_weighted = sq_err * channel_weight   # [1, 1, 3] broadcast over [B,N,3]
```

This modification increases the effective surface × pressure gradient, which is exactly
what the primary metric measures. The risk is that velocity channels degrade, but the
target does not penalize this.

**Implementation (changes to train.py):**

```python
# In Config dataclass, add:
channel_weights: tuple = (1.0, 1.0, 3.0)   # Ux, Uy, p

# In training loss computation, replace:
#   vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
#   surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
# with:
ch_w = torch.tensor(cfg.channel_weights, device=device)  # [3]
sq_err_w = sq_err * ch_w.unsqueeze(0).unsqueeze(0)       # [B, N, 3]
vol_loss = (sq_err_w * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_w * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

No architecture changes. The ch_w tensor is created once and reused. The normalization
denominator (vol_mask.sum(), surf_mask.sum()) does not need to account for channel weights
because we want the relative weighting, not re-normalized loss.

**Compute notes:** Negligible additional cost. Combine with hypothesis 1 (deeper-transolver)
in a second screening run if hypothesis 1 confirms the value of increased depth.

**Risk:** Medium. Misaligning loss from velocities may create training instability if p is
harder to learn early on. Monitor train loss curves. If instability appears, start with
[1, 1, 2] before trying [1, 1, 3].

---

## 4. larger-slice-num

**Hypothesis:** Increasing slice_num from 64 to 128 will reduce `val_avg/mae_surf_p` by
providing finer physics-slice resolution across the mesh, particularly for large-mesh
(~242K node) cruise tandem samples where each of the 64 slices currently covers an average
of 3,800 nodes.

**Predicted delta:** -4% to -8% on `val_avg/mae_surf_p`.

**Mechanism:** The Transolver's physics-slice attention mechanism projects N mesh nodes into
S learned slice tokens via softmax-weighted aggregation, then applies attention among the S
slices. At S=64 and N=242K, each slice token aggregates ~3,800 nodes on average. For
cruise tandem meshes with large dense zones (zone 1 and 2 near each foil), a single slice
token must represent physics spanning very different boundary-layer regions. Increasing to
S=128 halves the average node-per-slice count to ~1,900, allowing finer localization of
aerodynamic features (wake, boundary layer separation, stagnation point). The paper's
ablation shows that M∈[32, 256] spans its tested range with L=8; M=128 sits squarely in
the middle of this optimal range. Memory cost scales as O(S²) for the attention step:
128²/64² = 4x more attention FLOPs, but the attention is over tokens, not nodes, so
absolute cost is still modest (128² = 16,384 attention pairs per head).

**Implementation (changes to train.py):**

```python
# In model_config dict, change:
slice_num=128,   # was 64
```

No other changes. VRAM increase is modest (~200-300 MB) due to O(S²) attention, well
within the 96 GB budget.

**Compute notes:** ~10-15% longer per-step due to doubled slice attention cost. Acceptable
within a 24h window at n_layers=5. If combining with deeper-transolver (n_layers=8),
verify VRAM headroom with a single test batch before committing to a full run.

**Risk:** Low-medium. The paper shows M=256 still works, so M=128 is in-range. The risk
is that for small-mesh (74K node) raceCar single samples, increasing slices may over-
partition into near-empty slices, slightly hurting in-distribution performance. Monitor
val_single_in_dist separately from geom_camber and re_rand.

---

## 5. huber-robust-loss

**Hypothesis:** Replacing MSE with Huber loss (delta=2.0 in normalized space) will reduce
`val_avg/mae_surf_p` by reducing the disproportionate gradient contribution of high-Re
outlier samples, which have per-sample y std up to 10x larger than low-Re samples even
after global normalization.

**Predicted delta:** -3% to -8% on `val_avg/mae_surf_p`. Expected to most benefit
re_rand and geom_camber splits.

**Mechanism:** After global normalization by y_std, residuals from high-Re samples (Re~5M)
are systematically larger in magnitude than low-Re residuals because global y_std is
dominated by the high-Re tail. The MSE loss squares these residuals, making high-Re errors
contribute O(10x) more gradient weight than low-Re errors. Huber loss transitions from L2
to L1 at a threshold delta, bounding the gradient magnitude for large residuals to delta
instead of the residual magnitude. This effectively down-weights high-Re outlier gradients,
forcing the model to improve on low-Re and medium-Re regimes too. The MAE evaluation
metric is inherently L1-like, so Huber (which is MAE for large residuals) aligns training
objective more closely with evaluation. Delta=2.0 in normalized space means residuals up
to 2 sigma are trained with L2 (full gradient), while extreme residuals are clipped to
L1 gradient.

**Implementation (changes to train.py):**

```python
# In Config dataclass, add:
huber_delta: float = 2.0
loss_type: str = "huber"   # "mse" or "huber"

# Replace sq_err computation in training loop:
if cfg.loss_type == "huber":
    abs_err = (pred - y_norm).abs()
    sq_err = torch.where(
        abs_err < cfg.huber_delta,
        0.5 * abs_err ** 2,
        cfg.huber_delta * (abs_err - 0.5 * cfg.huber_delta)
    )
else:
    sq_err = (pred - y_norm) ** 2
```

The rest of the loss computation (vol_loss, surf_loss, surf_weight masking) is unchanged.
Note: the variable name `sq_err` is now a misnomer for Huber, but renaming it is optional
and cosmetic.

**Compute notes:** Negligible additional cost. A `torch.where` replaces element-wise
squaring. Can be combined with any other hypothesis in the list.

**Risk:** Medium. Delta tuning matters: too small a delta (e.g., 0.5) pushes toward L1
throughout and may cause noisy gradient directions; too large (e.g., 10) reduces to near-
MSE. Delta=2.0 is a reasonable first probe. If inconclusive, try delta=1.0 (tighter clip).
Note that Huber loss changes the loss scale, which may interact with surf_weight=10. Keep
surf_weight fixed to isolate the Huber effect.

---

## 6. ema-model-averaging

**Hypothesis:** Applying Exponential Moving Average (EMA, decay=0.999) over model weights
during training and using the EMA model for validation and test will reduce
`val_avg/mae_surf_p` by producing smoother, lower-variance parameter trajectories that
generalize better to OOD splits.

**Predicted delta:** -2% to -6% on `val_avg/mae_surf_p`, with larger gains on geom_camber
and re_rand OOD splits.

**Mechanism:** At each training step, EMA maintains a shadow copy of parameters as a
weighted average of all past parameter states. With decay=0.999, the EMA model has
effective memory of ~1/(1-0.999)=1000 steps. This acts as an implicit ensemble over
recent parameter states, smoothing over the high-variance updates caused by heterogeneous
batch composition (mixing raceCar single/tandem and cruise in one batch). For OOD
generalization (geom_camber splits), the EMA model avoids sharp local minima that the
training model briefly visits on high-Re raceCar batches, resulting in better mean-field
parameter estimates. EMA is a standard trick in modern ML training pipelines (used in
diffusion models, MAE, MaskFormer) with consistent +1-3% gains and zero inference cost.

**Implementation (changes to train.py):**

```python
# After model creation, add:
from torch.optim.swa_utils import AveragedModel
ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

# At end of each training step (after optimizer.step()), add:
ema_model.update_parameters(model)

# For validation, replace model.eval() calls with ema_model.eval()
# (or use ema_model.module for the underlying parameters)
# For checkpoint saving, save ema_model.state_dict() as the primary checkpoint.
```

Alternatively, use a lightweight custom EMA class:

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        model.load_state_dict(self.shadow)
```

**Compute notes:** EMA adds one copy of model weights in memory (~100-200 MB extra VRAM
for n_hidden=128, n_layers=5-8). Negligible compute overhead per step. No inference cost.

**Risk:** Low. EMA is a known reliable technique. The main risk is that decay=0.999 is too
high for short training (EMA barely catches up to the live model in <20 epochs). If the
training run is short (under ~10 epochs), reduce decay to 0.99. With cosine annealing and
50+ epochs, 0.999 is appropriate.

---

## 7. re-conditioned-loss-weighting

**Hypothesis:** Applying log(Re)-based loss sample weighting — increasing gradient
contribution from low-Re samples relative to high-Re samples — will reduce
`val_avg/mae_surf_p` on `val_re_rand` by improving the model's low-Re regime, which is
currently overwhelmed by high-Re gradients due to unequal value magnitudes under global
normalization.

**Predicted delta:** -3% to -7% on `val_re_rand/mae_surf_p`; modest benefit on other splits.

**Mechanism:** After global normalization by y_mean/y_std, a high-Re sample (Re=5M) with
per-sample y std of ~2,077 produces normalized residuals ~(2,077 / global_y_std) in
magnitude. A low-Re sample (Re=100K) with per-sample y std ~50 produces residuals ~10x
smaller. Under MSE loss, the high-Re sample contributes ~100x more gradient weight.
The model thus learns the high-Re regime well but generalizes poorly to low-Re and mid-Re
samples in val_re_rand (which is stratified across the full range). Re-conditioning
computes a per-sample inverse-variance weight: `w_i = (global_y_std / per_sample_y_std_i)^p`
(with p=0.5 as a soft reweighting) and scales the loss for each sample before averaging
across the batch. This forces equal optimization pressure on all Re regimes. Note: per-
sample y_std must be computed on-the-fly from the true y tensor (before normalization).

**Implementation (changes to train.py):**

```python
# In Config dataclass, add:
re_loss_balance: bool = True
re_balance_power: float = 0.5

# In training loop, after computing sq_err [B, N, 3], add per-sample weighting:
if cfg.re_loss_balance:
    # per-sample std in physical space, using valid (mask) nodes
    with torch.no_grad():
        # y is in physical space before normalization; y_norm is normalized
        # compute per-sample std from y [B, N, 3] using mask [B, N]
        y_masked = y * mask.unsqueeze(-1)           # zero out padding
        n_valid = mask.sum(dim=1, keepdim=True)     # [B, 1]
        y_mean_s = y_masked.sum(dim=1) / n_valid.squeeze(1).unsqueeze(-1)  # [B, 3]
        y_var_s = ((y_masked - y_mean_s.unsqueeze(1)) ** 2 * mask.unsqueeze(-1)).sum(dim=1) / n_valid.squeeze(1).unsqueeze(-1)
        y_std_s = y_var_s.mean(dim=-1).sqrt() + 1e-6   # [B], scalar per sample
        global_std = stats["y_std"].mean().to(device)
        sample_w = (global_std / y_std_s) ** cfg.re_balance_power   # [B]
        sample_w = sample_w / sample_w.mean()   # normalize to unit mean

    # Weight sq_err by sample_w before reducing:
    sq_err_w = sq_err * sample_w.unsqueeze(1).unsqueeze(2)  # [B, N, 3]
    # Then use sq_err_w in place of sq_err for vol_loss and surf_loss
```

This is more principled than explicit log(Re) bucketing because it directly measures the
per-sample value range and compensates for it. It requires no additional data or features.

**Compute notes:** One extra pass over y per batch to compute per-sample std, which is
O(N*B) — negligible compared to forward/backward. No VRAM increase.

**Risk:** Medium-high. The reweighting scheme may interfere with the balanced domain
sampler (which already reweights samples for domain balance). The two reweighting
mechanisms (domain balance via sampler, Re balance via loss) are multiplicative, which
may over-correct toward low-Re samples. Monitor train loss curves and check if raceCar
tandem (high-Re heavy) val metrics degrade. If instability appears, reduce power to 0.25.

---

## 8. naca-camber-fourier-features

**Hypothesis:** Augmenting the NACA camber input features (dims 15 and 19 of x) with
Fourier (sinusoidal) embeddings will improve generalization on `val_geom_camber_rc` and
`val_geom_camber_cruise` splits, where front-foil camber values (M=6-8 for raceCar,
M=2-4 for cruise) are entirely absent from training.

**Predicted delta:** -4% to -10% on geom_camber splits; small or neutral effect on other splits.

**Mechanism:** The NACA camber parameter (M, normalized to [0,1] in the input) is a
scalar that directly controls the foil's lift and pressure distribution. For the geom_camber
OOD splits, the model must extrapolate (or smoothly interpolate across a gap) in camber
space. The baseline represents camber as a single scalar feature, which forces the
downstream MLP to learn smooth functions of raw camber — a difficult inductive bias for
OOD extrapolation. Fourier feature embeddings replace dim 15 (and dim 19 for foil 2) with
a bank of sin/cos features at multiple frequencies:

```
phi(c) = [c, sin(2pi*c), cos(2pi*c), sin(4pi*c), cos(4pi*c), ...]
```

This expands the camber signal into a higher-dimensional smooth representation that
encourages the network to learn Fourier decompositions of the camber-to-flow mapping.
Smooth interpolation in this space between seen (M=2-5 for raceCar P1) and unseen
(M=6-8 for raceCar P2) camber values is more natural in Fourier space than in raw scalar
space. This idea adapts the Fourier Features work (Tancik et al. 2020, NeurIPS) to the
geometry feature encoding problem.

**Implementation (changes to train.py):**

```python
# In Config dataclass, add:
camber_fourier_freqs: int = 4    # number of sin/cos pairs beyond identity

# Add a preprocessing function:
def fourier_encode_camber(x, camber_dim_indices, n_freqs=4):
    """
    x: [B, N, 24]
    camber_dim_indices: list of dim indices to expand (e.g., [15, 19])
    Returns: [B, N, 24 - len(dims) + len(dims) * (1 + 2*n_freqs)]
    """
    parts = []
    skip = set(camber_dim_indices)
    for d in range(x.shape[-1]):
        if d not in skip:
            parts.append(x[..., d:d+1])
        else:
            c = x[..., d:d+1]   # [B, N, 1]
            enc = [c]
            for k in range(1, n_freqs + 1):
                enc.append(torch.sin(2 * math.pi * k * c))
                enc.append(torch.cos(2 * math.pi * k * c))
            parts.append(torch.cat(enc, dim=-1))
    return torch.cat(parts, dim=-1)

# After normalization of x:
x_norm = fourier_encode_camber(x_norm, camber_dim_indices=[15, 19],
                                n_freqs=cfg.camber_fourier_freqs)
```

CRITICAL: The expanded input dimension must be reflected in `fun_dim` in model_config.
With 2 camber dims × (1 + 2*4 - 1) = 8 additional dims (replacing 2 scalars with 9-dim
encoding each), fun_dim increases from 22 to 22 + 2*(8) = 38. Update model_config:

```python
model_config = dict(
    space_dim=2,
    fun_dim=38,   # was 22; update to 22 + camber_fourier_freqs*4 for 2 camber dims
    ...
)
```

Exact fun_dim = 24 - 2 (pos stripped as space_dim) - 2 (camber dims replaced) + 2 * (1 + 2 * n_freqs) = 20 + 2*(1+2*4) = 20 + 18 = 38 for n_freqs=4.

Apply the same encoding to validation and test data. Note: dims 15 and 19 are already
normalized by x_std in x_norm — Fourier features on normalized values are appropriate
since camber is in [0,1] after normalization.

**Compute notes:** Negligible VRAM change (larger input but same architecture width
thereafter). Slight increase in first-layer parameters. No extra training cost.

**Risk:** Medium. Fourier features have no direct evidence in mesh/PDE surrogate settings
for geometry features specifically. The OOD gap (M=6-8 vs trained M=2-5) may be too
large for any smooth feature to bridge without explicit physics supervision. If the geom_camber
splits show no improvement, the problem is likely insufficient physics inductive bias rather
than feature representation. In that case, a physics-residual loss (e.g., divergence of
predicted velocity) would be the right next direction.

---

## Summary Table

| # | Slug | Predicted delta on `val_avg/mae_surf_p` | Confidence | Risk | Key change |
|---|------|-----------------------------------------|------------|------|------------|
| 1 | deeper-transolver | -8% to -15% | High | Low | n_layers 5→8 |
| 2 | warmup-cosine-grad-clip | -5% to -10% | High | Very low | 5ep warmup + grad_clip=1.0 |
| 3 | surf-p-weighted-loss | -5% to -12% | Medium-high | Medium | channel_weights [1,1,3] |
| 4 | larger-slice-num | -4% to -8% | Medium-high | Low | slice_num 64→128 |
| 5 | huber-robust-loss | -3% to -8% | Medium | Medium | Huber delta=2.0 replaces MSE |
| 6 | ema-model-averaging | -2% to -6% | Medium | Low | EMA decay=0.999 on weights |
| 7 | re-conditioned-loss-weighting | -3% to -7% on re_rand | Medium | Medium-high | Per-sample inverse-std weighting |
| 8 | naca-camber-fourier-features | -4% to -10% on geom_camber | Low-medium | Medium | Fourier encoding of camber dims 15,19 |

## Combination Priority (if multiple slots available)

For a 24h window with multiple student slots, the highest-leverage combinations are:

1. **deeper-transolver + warmup-cosine-grad-clip** — complementary, no interaction risk.
   The depth increase benefits from stable early training via warmup. Run first.

2. **surf-p-weighted-loss + deeper-transolver** — once depth is confirmed, add channel
   weighting. Objective alignment with primary metric.

3. **huber-robust-loss** — orthogonal to architecture changes; can run in parallel with #1.

4. **ema-model-averaging** — add to any run as a free improvement; zero cost.

Note: Do NOT combine re-conditioned-loss-weighting with surf-p-weighted-loss in the same
run — both modify the effective loss per sample and their interaction is hard to attribute.
Test them separately.

## Diagnostic Checkpoints

For each run, watch these secondary metrics to diagnose failures:

- **Train loss curve**: should decrease smoothly; instability suggests lr or grad_clip issue.
- **val_single_in_dist/mae_surf_p**: in-distribution sanity check. If this gets worse while
  others improve, the change hurts general accuracy. Merge only if this stays flat or improves.
- **val_re_rand/mae_surf_p**: cross-Re generalization. Sensitive to loss weighting changes.
- **val_geom_camber_rc vs val_geom_camber_cruise**: OOD geometry. The two camber splits
  should move together for architecture changes; if only one improves, investigate whether
  the data distribution for that domain is driving the signal.

## Stop Conditions

- **deeper-transolver fails**: if n_layers=8 does not improve on n_layers=5 by at least
  2%, the model is not capacity-limited — stop pursuing deeper architectures and focus on
  loss formulation and normalization.
- **warmup/grad-clip fails**: if training with warmup produces similar curves to without,
  the instability hypothesis is wrong — the model is training stably already.
- **All of 1-4 fail**: if depth, warmup, channel weighting, and slice count all fail to
  improve, the bottleneck is likely in data representation (global normalization) or in
  the fundamental expressiveness limit of physics-slice attention for this dataset. Move
  to a different architecture family (e.g., mesh transformer with local neighborhoods,
  FNO-style, or hybrid approach).
