# Round-2 Research Hypotheses — TandemFoilSet CFD Surrogate
# Generated: 2026-05-12

These 10 hypotheses are designed to be **distinct from** the 8 round-1 experiments
already in flight, each implementable in <200 lines of `train.py` changes,
runnable in a 30-min / 3–6 epoch window on a single 96 GB GPU.

Primary metric: `val_avg/mae_surf_p` (lower is better).

---

## H1: Per-Sample Adaptive Loss Scaling (Scale-Invariant Loss)

**Mechanism**

The global `y_mean / y_std` normalization is computed over the entire training
corpus, but per-sample y-std varies by an order of magnitude across Re (see
program.md value ranges: avg per-sample y std 164–458, max up to 2077). A
high-Re sample with y_std=2000 contributes loss ~(2000/y_std_global)^2 times
more than a low-Re sample with y_std=100. This creates implicit curriculum
bias: the optimizer is dominated by the hardest (highest Re) samples, leaving
low-Re behaviour under-fitted. Fix: divide each sample's loss contribution by
its *per-sample* empirical y-std (computed from the unmasked nodes), so every
sample contributes roughly unit-variance signal.

**Expected delta on val_avg/mae_surf_p**

-3% to -8%. Should help most on `val_single_in_dist` (wide Re range) and
`val_re_rand`. Low risk of hurting.

**Implementation sketch (train.py changes)**

In the training loop, after computing `y_norm`, add:

```python
# Per-sample std in normalized space, shape [B]
with torch.no_grad():
    per_sample_std = y[mask].reshape(-1, 3).std(dim=0).mean().clamp(min=1.0)
    # per-sample re-weight: mean physical std / this sample's std
    # Actually: compute per-batch mean of per-sample stds for B>1
    B = y.shape[0]
    sample_stds = []
    for b in range(B):
        m_b = mask[b]  # [N]
        if m_b.sum() > 1:
            sample_stds.append(y[b][m_b].std(dim=0).mean())
        else:
            sample_stds.append(torch.tensor(1.0, device=device))
    sample_stds = torch.stack(sample_stds)  # [B]
    # normalize so mean weight = 1 (no scale shift)
    weights = (sample_stds.mean() / sample_stds.clamp(min=1.0)).detach()  # [B]

# Apply weights: weighted average of per-sample losses
vol_loss = 0.0
surf_loss = 0.0
for b in range(B):
    vm = vol_mask[b]
    sm = surf_mask[b]
    vol_loss += weights[b] * (sq_err[b][vm].mean() if vm.sum() > 0 else 0.0)
    surf_loss += weights[b] * (sq_err[b][sm].mean() if sm.sum() > 0 else 0.0)
vol_loss /= B
surf_loss /= B
loss = vol_loss + cfg.surf_weight * surf_loss
```

A cleaner vectorized version: broadcast `weights[:, None, None]` over `[B, N, 3]`.

**Risks**

- The per-sample std computation adds a small overhead (~1% wall clock).
- If sample_stds is noisy (small batch_size=4), might add variance; mitigate by
  using a running EMA of per-sample stds.
- Might hurt if low-Re samples are already well-fit and just need the model to
  focus on high-Re samples; check per-Re diagnostic to confirm direction.

---

## H2: Fourier Positional Encoding for Node Coordinates

**Mechanism**

Transolver's preprocess MLP receives raw (x, z) node coordinates and 22 other
features. Raw coordinates carry spatial frequency information only up to what a
shallow MLP can represent; they are also scale-sensitive. Random Fourier Features
(Rahimi & Recht 2007, also used in NeRF / Neural Tangent Kernel literature) map
coordinates to a high-frequency basis:

  gamma(x) = [sin(2pi * B @ x), cos(2pi * B @ x)]

where B is a fixed random matrix drawn from N(0, sigma^2). This replaces the
2 raw coordinate dims with 2*m features (m=16 or 32 frequencies), giving the
preprocess MLP direct access to multi-scale spatial patterns without needing
many extra layers. The approach is particularly effective for learning smooth
but spatially varying fields (exactly what CFD pressure / velocity look like).

**Expected delta on val_avg/mae_surf_p**

-5% to -12%. Strong prior from NeRF literature and Neural Operator work
(FNO's Fourier layers are the operator-theoretic sibling of this idea). Should
help most on geometry-OOD splits (`val_geom_camber_rc/cruise`) because the
model can better encode the fine boundary-layer structure near foil surfaces.

**Implementation sketch**

```python
# Add at top of train.py, after model_config
FOURIER_DIM = 32  # number of frequencies per axis, total 2*32=64 new dims
torch.manual_seed(42)
FOURIER_B = torch.randn(FOURIER_DIM, 2) * 3.0  # sigma=3 (tunable)
# Keep FOURIER_B on device during training

# In preprocess / forward, replace the 2-coord input:
# Instead of passing x[:, :, :2] as position, pass fourier_encode(x[:, :, :2])
# where:
def fourier_encode(pos, B_matrix):
    # pos: [B, N, 2], B_matrix: [F, 2]
    proj = torch.einsum("bnd,fd->bnf", pos, B_matrix) * 2 * math.pi
    return torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, N, 2F]
```

In `Transolver.__init__`, change `fun_dim + space_dim` to `fun_dim + 2*FOURIER_DIM`.
In `Transolver.forward`, split `x[:, :, :2]` (position) from `x[:, :, 2:]`
(other features), encode position, and concatenate before preprocess MLP.

`sigma` (scale of B) is the key hyperparameter; try 1.0, 3.0, 10.0. sigma=3 is
a good default for normalized coordinates.

**Risks**

- Increases preprocess MLP input dim: `fun_dim + 64` vs `fun_dim + 2`. With
  `n_hidden=128` and `n_layers=0` this is a modest increase.
- sigma selection is domain-sensitive; wrong sigma can hurt (too low = redundant
  with raw coords, too high = aliasing artifacts).
- Should normalize input coordinates to [-1, 1] before encoding (they already
  are after x_norm).

---

## H3: Separate Surface / Volume Output Heads

**Mechanism**

Currently all three output channels (Ux, Uy, p) are predicted by a single final
`mlp2` that runs on all nodes identically. Surface nodes have fundamentally
different statistics from volume nodes: they are on no-slip / slip-wall
boundaries, the pressure gradient is discontinuous normal to the surface, and
p values are the primary metric. Separate heads — one for surface nodes, one for
volume nodes — allow the network to specialize. The surface head can be deeper or
wider than the volume head (budget-neutral if volume head is made smaller).

This is analogous to CLS-token heads in BERT-style models: different head for
different semantic role of the token.

**Expected delta on val_avg/mae_surf_p**

-4% to -10% on surface pressure specifically. The surface head can learn that
the surface feature distribution is a strict subset of the full distribution
and should not fit volume-like features.

**Implementation sketch**

In `TransolverBlock` (last_layer), replace the single `mlp2` with two heads:

```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    self.surf_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    self.vol_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, out_dim),
    )
```

In `Transolver.forward`, after getting hidden states from the last block,
split by `is_surface`:

```python
# data["is_surface"]: [B, N] bool
# Use surf head for surface nodes, vol head for volume nodes
out = torch.zeros(B, N, out_dim, device=fx.device)
out[is_surf] = self.blocks[-1].surf_head(self.blocks[-1].ln_3(fx))[is_surf]
out[~is_surf] = self.blocks[-1].vol_head(self.blocks[-1].ln_3(fx))[~is_surf]
```

Note: `is_surface` needs to be passed through `data` dict (already available
during training; add `data["is_surface"]` to the forward call in train.py).

**Risks**

- Requires passing `is_surface` into the model forward call — a small interface
  change but still within `train.py` since the model is defined there.
- Gradient flow for surface head only passes through ~1–2% of nodes; may need
  a higher `surf_weight` or gradient clipping to compensate.
- If the model has already learned a good surface representation via the
  `surf_weight=10` loss, the marginal gain may be small.

---

## H4: Gradient Clipping + Warm Restarts (Training Stability)

**Mechanism**

Transolver uses AdamW with cosine annealing but no gradient clipping. With
variable mesh sizes (74K–242K nodes) and high-Re samples driving large loss
spikes, per-step gradients can vary by 10–100x. The existing `surf_weight=10`
amplifies surface gradients. Without clipping, a single high-Re cruise sample
(242K nodes, y_std up to 2077) can cause a large update that takes many steps
to recover from, especially in early epochs. Gradient clipping (max_norm=1.0
or 0.5) is the canonical fix and is essentially free in compute.

Additionally, Stochastic Gradient Descent with Warm Restarts (SGDR / Cosine
Annealing with Restarts, Loshchilov & Hutter 2017) can help the model escape
local minima within the tight 6-epoch budget. With T_0=2, T_mult=1 (restart
every 2 epochs), the LR can sweep 3 complete cycles in 6 epochs.

**Expected delta on val_avg/mae_surf_p**

-2% to -6%. Low-risk, near-free change. May compound with other changes.

**Implementation sketch**

```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Optionally: replace CosineAnnealingLR with CosineAnnealingWarmRestarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2, T_mult=1, eta_min=1e-6
)
# Note: step after each batch for CAWR: scheduler.step(epoch + i/n_batches)
```

**Risks**

- Too aggressive clipping (max_norm=0.1) can slow convergence; 1.0 is the
  safe default.
- CAWR with T_0=2 means the LR briefly spikes to `cfg.lr` at epoch 3 and 5,
  which can briefly worsen metrics mid-run; checkpointing is already on best-val
  so no final-epoch degradation risk.
- Combined with a larger LR (from a separate experiment), restarts interact;
  test in isolation first.

---

## H5: Log-Scale Re Feature + Re-Conditional Bias in Attention Temperature

**Mechanism**

The baseline already uses `log(Re)` as input feature dim 13. However, the
attention temperature `self.temperature` in `PhysicsAttention` is a single
learned scalar shared across all samples. High-Re flow has sharper boundary
layers (more concentrated in physical space), suggesting the attention over
slice tokens should also be "sharper" (lower temperature = more concentrated
slice assignments). Conditioning `temperature` on `log(Re)` makes the
slice-assignment sharpness adaptive to the flow regime:

  temperature(x) = base_temp * sigmoid(W_temp @ log_Re + b_temp)

This is a 2-parameter change per attention head per layer.

Additionally: the existing `log(Re)` feature is mixed with geometric features
in the preprocess MLP. A dedicated Re-embedding (e.g., sinusoidal encoding of
log_Re) passed as a global conditioning vector (added to slice tokens after
aggregation) could give the model a cleaner Re-dependent pathway.

**Expected delta on val_avg/mae_surf_p**

-3% to -8%. The `val_re_rand` split (cross-regime generalization) should show
the largest improvement. May also help OOD camber splits since those also span
wide Re.

**Implementation sketch**

In `PhysicsAttention.__init__`, add a small linear mapping:
```python
self.re_temp_proj = nn.Linear(1, heads)  # maps scalar log_Re to per-head temp mod
```

In `PhysicsAttention.forward`, extract log_Re from x (dim 13 after x_norm),
aggregate to batch-level (mean over nodes), compute temperature modulation:
```python
log_re = x[:, :, 13].mean(dim=1, keepdim=True)  # [B, 1]
temp_mod = torch.sigmoid(self.re_temp_proj(log_re))  # [B, heads]
temp_mod = temp_mod[:, :, None, None]  # [B, heads, 1, 1]
# Replace self.temperature with:
effective_temp = self.temperature * temp_mod
slice_weights = self.softmax(self.in_project_slice(x_mid) / effective_temp)
```

Note: x passed to attention is *normalized* x (x_norm), so dim 13 is
`(log_Re - x_mean[13]) / x_std[13]` — still monotone in Re and usable.

**Risks**

- The log_Re value must be consistent across nodes within a sample (it is, it's
  a global flow condition). Mean over nodes is fine.
- If Re variation within a batch (batch_size=4, mixed domains) is large,
  the temperature modulation could conflict across samples; mask-based mean
  (over valid nodes only) is safer.
- Small risk of gradient instability if `effective_temp` approaches 0; clamp
  `temp_mod` to [0.1, 10.0].

---

## H6: Stochastic Depth / LayerDrop for Regularization

**Mechanism**

Transolver uses 5 `TransolverBlock` layers with no dropout beyond the attention
dropout. For a model with ~3.5M parameters training on only ~1499 samples with
~6 epochs, overfitting is a real risk — especially on the OOD geometry splits
(val_geom_camber_rc, val_geom_camber_cruise) where the model has never seen the
specific camber values. Stochastic Depth (Huang et al. 2016, "Deep Networks with
Stochastic Depth") randomly drops entire residual blocks during training with a
linearly increasing probability across layers (survival prob 1.0 at layer 0,
`1 - drop_rate` at last layer). This acts as a strong ensemble regularizer that
is especially effective for transformer-like architectures (used in DeiT, Swin,
etc.).

**Expected delta on val_avg/mae_surf_p**

-3% to -7% on OOD splits, potentially neutral or slightly worse on in-dist.
Net improvement if OOD splits have higher absolute MAE.

**Implementation sketch**

```python
class StochasticDepth(nn.Module):
    def __init__(self, survival_prob: float):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x_skip, residual):
        if not self.training or self.survival_prob == 1.0:
            return x_skip + residual
        # Bernoulli mask per sample
        B = x_skip.shape[0]
        mask = torch.bernoulli(
            torch.full((B,), self.survival_prob, device=x_skip.device)
        ).view(B, 1, 1)
        return x_skip + mask * residual / self.survival_prob  # unbiased estimate
```

In `TransolverBlock.__init__`, add:
```python
self.drop_path = StochasticDepth(survival_prob)  # pass from Transolver
```

In `TransolverBlock.forward`:
```python
fx = self.drop_path(fx, self.attn(self.ln_1(fx)))
fx = self.drop_path(fx, self.mlp(self.ln_2(fx)))
```

In `Transolver.__init__`, set survival_probs linearly from 1.0 to (1-drop_rate)
across n_layers. Recommended drop_rate=0.1 for 5 layers.

**Risks**

- With batch_size=4, the Bernoulli mask may occasionally drop all 4 samples for
  a layer; clamp survival_prob >= 0.5 to avoid this.
- For a 5-layer model, stochastic depth savings are modest. Drop rate 0.1–0.2 is
  safe; 0.3+ risks instability within 6 epochs.
- Must be disabled during evaluation (already handled by `self.training` check).

---

## H7: EMA (Exponential Moving Average) of Weights for Inference

**Mechanism**

Exponential Moving Average of model weights is a near-free "ensemble" technique:
maintain a shadow copy of model weights `theta_ema = beta * theta_ema + (1-beta) * theta`,
and use `theta_ema` at evaluation time rather than the live weights. EMA weights
generalize better than a single checkpoint because they average over the
optimization trajectory. This is used in diffusion models (DDPM, EDM), modern
image classifiers (timm's EMA), and has been shown to improve the test metrics
of physics-informed neural networks (used implicitly in "curriculum denoising"
approaches for PDE surrogates). With only ~6 epochs available, EMA provides
cheap model averaging that would otherwise require multiple restarts.

**Expected delta on val_avg/mae_surf_p**

-2% to -5%. Consistent and reliable improvement in low-epoch regimes. Very low
risk. The gain is typically larger when training is noisy (which it is here due
to variable mesh sizes and the cosine LR hitting minimum near epoch 6).

**Implementation sketch**

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        """Temporarily apply EMA weights to model (for eval)."""
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return orig

    def restore(self, model, orig):
        model.load_state_dict(orig)
```

Usage in training loop:
```python
ema = EMA(model, decay=0.999)
# After optimizer.step():
ema.update(model)

# During validation:
orig = ema.apply(model)
split_metrics = {name: evaluate_split(...) for name, loader in val_loaders.items()}
ema.restore(model, orig)
```

`decay=0.999` is standard. With ~375 steps/epoch and 6 epochs, EMA sees ~2250
updates — enough for convergence. Try also `decay=0.9995`.

**Risks**

- EMA shadow copy doubles model memory footprint (~14 MB for 3.5M params at fp32;
  negligible vs 96 GB VRAM).
- Checkpoint saves should save the EMA weights, not the live weights. Change
  `torch.save(model.state_dict(), model_path)` to
  `torch.save(ema.shadow, model_path)` and load accordingly.
- With very fast LR decay (cosine annealing to near-zero), the EMA will track
  closely to final weights anyway; decay=0.99 may give more averaging benefit.

---

## H8: Asymmetric / Huber Loss for Pressure Outlier Robustness

**Mechanism**

The current MSE (squared error) loss is quadratic in prediction error. For CFD
pressure fields, large outlier errors (common in wake regions and stagnation
points at high Re) contribute disproportionately to the gradient, potentially
causing the model to over-fit to extreme cases at the expense of average
accuracy. Huber loss (smooth L1) interpolates between L1 (for large errors) and
L2 (for small errors), controlled by delta:

  L_huber(e) = 0.5*e^2              if |e| <= delta
             = delta*(|e| - 0.5*delta)   otherwise

For CFD surrogates, Huber loss has been used in DL-based flow field prediction
(e.g., PhysDNet, FNO variants) to improve mean performance by dampening the
influence of extreme pressure spikes.

Alternatively: channel-specific loss weighting. Pressure errors are in different
units from velocity (pressure in m^2/s^2, velocity in m/s). Currently all three
channels are summed equally. A 3-way weight `[w_Ux, w_Uy, w_p]` with `w_p > 1`
directly upweights the primary metric channel.

**Expected delta on val_avg/mae_surf_p**

-2% to -8%. Huber delta is the key hyperparameter; too low delta makes training
too L1-like (slow convergence in early epochs), too high delta ~ L2 (no change).
Try delta=0.5 in normalized space (corresponds to ~0.5 * y_std in physical units,
roughly 250 m^2/s^2 for pressure — a reasonable outlier threshold).

**Implementation sketch**

```python
# Option A: Huber loss (replace sq_err)
delta = 0.5  # in normalized space
err = pred - y_norm
abs_err = err.abs()
sq_err = torch.where(abs_err <= delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta))

# Option B: channel-specific weights (add to Config)
channel_weights = torch.tensor([1.0, 1.0, 3.0], device=device)  # [Ux, Uy, p]
sq_err = (pred - y_norm) ** 2 * channel_weights[None, None, :]

# Option C: combine — Huber on all channels + p-channel upweight
```

**Risks**

- Huber loss changes the metric being optimized from MSE to pseudo-Huber. The
  val metric is MAE in physical space, not MSE, so this change is directionally
  aligned (Huber is between MSE and MAE).
- Channel weights for p must be chosen carefully; too high and Ux/Uy suffer.
  The `surf_weight` already upweights surface nodes; stacking a `p` channel
  weight may over-specialize.
- Test both options separately before combining.

---

## H9: Multi-Resolution Slice Pooling (Coarse + Fine Slices)

**Mechanism**

The current Transolver uses `slice_num=64` fixed-size slice tokens across all
5 layers. Flow fields have multi-scale structure: large-scale pressure gradients
(far field), intermediate-scale separation bubbles, and fine boundary-layer
gradients near surfaces. A single resolution of 64 slices forces a
one-size-fits-all discretization. Inspired by multi-scale attention in Vision
Transformers (Swin Transformer, PVT) and hierarchical graph neural operators
(U-Net-style FNO), we can use **different `slice_num` per layer**:

  Layer 1 (coarse): slice_num=16  — global flow structure
  Layer 2: slice_num=32
  Layer 3 (medium): slice_num=64  — current default
  Layer 4: slice_num=128
  Layer 5 (fine): slice_num=256  — boundary-layer details

This keeps total parameter count similar (attention parameters scale as
O(slice_num * dim_head), not O(slice_num^2)) while giving different layers
different spatial resolution.

**Expected delta on val_avg/mae_surf_p**

-4% to -10%. The OOD geometry splits should benefit most since the fine-grained
slices in later layers can adapt to unseen foil shapes better than a fixed 64.

**Implementation sketch**

In `Transolver.__init__`, change blocks to use per-layer slice_num:
```python
# Instead of a single slice_num=64:
slice_nums = [16, 32, 64, 128, 256]  # or [32, 64, 64, 128, 128]

self.blocks = nn.ModuleList([
    TransolverBlock(
        num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
        act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
        slice_num=slice_nums[i], last_layer=(i == n_layers - 1),
    )
    for i in range(n_layers)
])
```

No other changes needed; `PhysicsAttention` already takes `slice_num` as a
parameter.

**Risks**

- `slice_num=256` at layer 5 increases attention memory (slice tokens are
  [B, heads, 256, dim_head] but slice aggregation is O(N*256) not O(256^2),
  so memory scales linearly in N not quadratically — should be fine for 242K nodes).
- Coarse early layers (slice_num=16) may lose too much spatial detail before
  later layers can recover; try [32, 64, 64, 96, 128] as a conservative default.
- The total number of parameters changes slightly; verify with
  `sum(p.numel() for p in model.parameters())`.

---

## H10: Mixed-Precision + Larger Batch for Faster Convergence

**Mechanism**

The baseline trains at batch_size=4 with full float32. Given 96 GB VRAM and
mesh sizes up to 242K nodes, float16 AMP could reduce memory by ~50% per sample
and increase throughput, allowing batch_size=8 or more. Larger effective batch
sizes provide more gradient signal per step (especially useful for the balanced
domain sampler which draws from 3 domains; a batch of 4 may be dominated by
a single domain in any given batch, while batch=8 is more likely to contain
all 3). In the operator learning literature (FNO, DeepONet), larger batches
improve convergence rate when training is memory-bound rather than compute-bound.

**Expected delta on val_avg/mae_surf_p**

Neutral to -5%. This is primarily a compute-efficiency experiment: the same
number of epochs trains faster, meaning the timeout allows more epochs. The
real test is whether 6 epochs at batch=8 (fewer steps per epoch) beats 6 epochs
at batch=4 (more steps per epoch).

**Implementation sketch**

```python
# Add to imports:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop, replace:
pred = model({"x": x_norm})["preds"]
loss = ...
optimizer.zero_grad()
loss.backward()
optimizer.step()

# With:
optimizer.zero_grad()
with autocast():
    pred = model({"x": x_norm})["preds"]
    # ... same loss computation inside autocast
    loss = vol_loss + cfg.surf_weight * surf_loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Also add `--batch_size 8` or `--batch_size 6` to the run command. Verify no
OOM on a cruise sample (242K nodes * 6 * 24 * 2 bytes = ~70 MB/sample; 6 samples
= ~420 MB, well within 96 GB).

**Risks**

- AMP with float16 can cause NaN in LayerNorm or attention softmax for extreme
  values; mitigate with GradScaler's dynamic loss scaling.
- Larger batch reduces variance of gradient per step but each step is "cheaper"
  in information terms; the optimizer may need a proportionally higher LR
  (linear scaling rule: lr *= batch/4). Test `lr=1e-3` with `batch_size=8`.
- The balanced WeightedRandomSampler still applies; with batch=8, domain balance
  within a batch improves automatically.

---

## Summary Table

| ID  | Mechanism                            | Target split(s)              | Risk  | Compute overhead |
|-----|--------------------------------------|------------------------------|-------|-----------------|
| H1  | Per-sample adaptive loss scaling     | all, esp val_re_rand         | Low   | Minimal         |
| H2  | Fourier positional encoding          | geom-OOD, re_rand            | Med   | Minimal         |
| H3  | Separate surf/vol output heads       | all (surf metric)            | Med   | Minimal         |
| H4  | Grad clipping + warm restarts        | all                          | Low   | None            |
| H5  | Re-conditional attention temperature | val_re_rand                  | Med   | Minimal         |
| H6  | Stochastic depth regularization      | geom-OOD                     | Low   | Minimal         |
| H7  | EMA weight averaging                 | all                          | Low   | ~1% mem         |
| H8  | Huber / channel-weighted loss        | all (surf p)                 | Low   | None            |
| H9  | Multi-resolution slice pooling       | geom-OOD, re_rand            | Med   | ~5% mem         |
| H10 | AMP + larger batch                   | all (throughput)             | Med   | -50% mem/sample |

**Priority order for first round of assignment:**
1. H7 (EMA) — lowest risk, consistent gains in low-epoch regimes, zero compute cost
2. H2 (Fourier pos enc) — strong prior from NeRF/neural operators, ~5-12% expected gain
3. H1 (Per-sample loss scaling) — directly addresses the known Re-variance problem
4. H9 (Multi-resolution slices) — architectural, targets OOD geometry which drives primary metric
