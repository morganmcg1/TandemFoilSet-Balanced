<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet r2 — Willow Research Ideas
Generated: 2026-05-12
Advisor branch: icml-appendix-willow-pai2g-24h-r2

## Context recap

Baseline: Transolver, 128 hidden dim, 5 layers, 4 heads, 64 slices, mlp_ratio=2, GELU.
Loss: `vol_loss + 10.0 * surf_loss` (MSE on normalized targets).
Optimizer: AdamW lr=5e-4, wd=1e-4. Scheduler: CosineAnnealingLR(T_max=epochs).
Batch size: 4, WeightedRandomSampler for 3-domain balance.
Hard 30-min wall-clock cap. Achievable epochs: ~5–15.
Primary metric: `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 val splits, denormalized physical units. Lower is better.

Key diagnostics from the programme spec that constrain hypothesis space:
1. Per-sample y std varies by ~10x across Re range (100K–5M). Global normalization is coarse.
2. Surface pressure (channel 2) is the ranking metric, but the loss treats all three channels equally within each region (vol/surf split).
3. The model gets 5–15 epochs: ideas that show up as a training-dynamics delta in the first few thousand steps win; ideas needing long convergence lose.
4. data/ is read-only; all changes land in train.py or pyproject.toml.
5. Masks must be respected in any custom loss to avoid counting padding.

---

## Hypothesis 1: Per-sample loss normalization (Huber-normalized MAE)

### Category
Loss reformulation

### One-line summary
Replace global-normalized MSE with per-sample Huber MAE, dividing each sample's squared-error map by that sample's empirical y-variance before averaging — so high-Re outlier samples do not dominate the gradient signal.

### Rationale
The current MSE loss is computed in the global-normalization space (`y_norm = (y - y_mean) / y_std`). Because the global y_std is dominated by high-Re samples, low-Re predictions are penalized in units much smaller than their actual error. The model learns preferentially from high-Re samples even with domain-balanced sampling. Per-sample normalization makes every sample contribute roughly equally to the gradient.

A Huber loss (delta=1.0 in normalized space) additionally clips the influence of individual high-residual nodes (e.g., stagnation-point pressure spikes) that can destabilize early training.

Mechanism tested: does equalizing per-sample gradient magnitude improve the model's ability to fit low-Re and OOD samples (the two geometry-OOD val splits include low-Re cruise)?

### Concrete implementation
In the training loop (lines 487–496 of train.py), after computing `y_norm`:

```python
# --- per-sample scale normalization ---
# y_norm: [B, N, 3], mask: [B, N]
# Compute per-sample std of the REAL nodes (exclude padding)
with torch.no_grad():
    m = mask.unsqueeze(-1).float()          # [B, N, 1]
    n_real = m.sum(dim=1).clamp(min=1)      # [B, 1]
    y_mu = (y_norm * m).sum(dim=1) / n_real  # [B, 1, 3]
    y_var = ((y_norm - y_mu) ** 2 * m).sum(dim=1) / n_real  # [B, 1, 3]
    y_scale = (y_var.sqrt() + 1e-4)         # [B, 1, 3]  shape-broadcast

# Normalize residuals per sample before taking loss
res = (pred - y_norm) / y_scale             # [B, N, 3]

# Huber on the normalized residuals (delta=1.0)
delta = 1.0
abs_res = res.abs()
huber = torch.where(abs_res <= delta,
                    0.5 * res ** 2,
                    delta * (abs_res - 0.5 * delta))

vol_loss  = (huber * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

No new packages needed. No model-contract changes.

### Expected delta
Moderate-to-large. The per-sample normalization directly addresses the identified bottleneck (gradient dominated by high-Re samples). Expect improvement particularly on `val_geom_camber_cruise` (low-Re tandem) and `val_re_rand`. Risk: if the global normalization was already well-calibrated, this adds noise.

### Risk / failure mode
If y_scale near zero (very uniform samples at low Re), division could amplify noise. The `1e-4` floor mitigates this but does not eliminate it. Monitor `train/surf_loss` — if it stops decreasing from epoch 1 the floor may be too tight.

### Hyperparameters to keep fixed
surf_weight=10.0, lr=5e-4, epochs=50 (let timeout kill it)

---

## Hypothesis 2: Surface-pressure channel prioritization in the loss

### Category
Surf-pressure prioritization / loss reformulation

### One-line summary
Re-weight the per-channel contribution of pressure (dim 2) inside the surface loss term so that surface pressure errors drive proportionally more gradient, while keeping velocity channels anchored to physics.

### Rationale
`val_avg/mae_surf_p` is the sole ranking metric, yet the current loss treats Ux, Uy, and p identically (unweighted sum of squared errors across all three channels). If pressure and velocity have different convergence rates or loss magnitudes in normalized space, the model may be spending capacity on velocity errors at the expense of pressure accuracy.

The fix is a simple 3-element channel weight vector applied inside the surface term.

### Concrete implementation

Add a CLI parameter `p_weight` (default 1.0, try 3.0 as the first experiment arm):

```python
@dataclass
class Config:
    ...
    p_weight: float = 3.0   # extra weight on pressure channel in surf loss
```

In the loss block:

```python
# channel weights: [Ux, Uy, p] — broadcast over [B, N, 3]
ch_weights = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)

sq_err = (pred - y_norm) ** 2 * ch_weights[None, None, :]

vol_loss  = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Try p_weight in {2.0, 3.0, 5.0}. The student should run p_weight=3.0 as the primary arm.

Note: this is separate from `surf_weight` (which already upweights surface nodes vs. volume nodes). This is an intra-surface channel rebalancing.

### Expected delta
Small-to-moderate. The metric is literally surface pressure MAE, so directing more gradient toward that channel in the surface region is the most targeted loss change possible. Risk: if pressure is already converging faster than velocity, this pushes the model to overfit p at the expense of physically consistent velocity fields.

### Hyperparameters to keep fixed
surf_weight=10.0, lr=5e-4. The channel weight is the only knob.

---

## Hypothesis 3: Higher surface loss weight (surf_weight sweep 20–50)

### Category
Surf-pressure prioritization / loss weight

### One-line summary
Increase `surf_weight` from 10 to 25 or 40 — the surface nodes are a tiny fraction of total nodes (~2–5%), so an order-of-magnitude higher weight may be needed to make surface error dominate the gradient budget proportionally.

### Rationale
With ~85K–210K total mesh nodes and surface nodes being a thin boundary layer, the vol_loss term has far more nodes contributing than surf_loss. Even at surf_weight=10, the gradient from surface nodes may be swamped by the large volume node count. A back-of-envelope: if the surface fraction is 3%, then `surf_weight=33` makes each surface node contribute the same gradient as one volume node in expectation. The current surf_weight=10 under-weights surface nodes.

This is the most direct and cheapest hypothesis to test and is grounded in the node-count imbalance.

### Concrete implementation
No new code logic needed — just change the default:

```python
@dataclass
class Config:
    ...
    surf_weight: float = 30.0   # was 10.0
```

Run two arms: surf_weight=20 and surf_weight=40. Student should run surf_weight=30 as primary.

CLI: `python train.py --surf_weight 30`

### Expected delta
Potentially large and shows up early (epoch 1–3 surface loss scale changes immediately). This is the simplest hypothesis that directly targets the metric. Primary risk: too high a surf_weight causes the vol field to become physically incoherent (velocity divergence etc.) which may hurt generalization on the OOD splits.

### Hyperparameters to keep fixed
lr=5e-4, p_weight=1.0 (default). Let this be the sole change.

---

## Hypothesis 4: Warmup + cosine annealing with restarts (CosineAnnealingWarmRestarts)

### Category
Optimizer / schedule

### One-line summary
Replace `CosineAnnealingLR(T_max=epochs)` with a short linear warmup (5% of budget) followed by `CosineAnnealingWarmRestarts(T_0=10)` so the model can escape local minima within the 30-min wall-clock window.

### Rationale
The baseline schedule decays the LR monotonically. With only 5–15 epochs achievable, the initial LR at epoch 1 is the highest the model will see — but the model is still poorly initialized at epoch 1. The warm restarts pattern (popularized by SGDR, Loshchilov & Hutter 2016) allows the model to spike back to a higher LR mid-training, which in short training regimes has been repeatedly shown to find better minima than a single monotonic decay.

The warmup prevents the high initial LR from causing large gradient instability at epoch 1.

### Concrete implementation

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts

warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))
warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_epochs)
cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                          milestones=[warmup_epochs])
```

Replace the single `scheduler.step()` call (same location, no other changes).

No new packages.

### Expected delta
Small-to-moderate. Schedule changes rarely show up as large deltas in early epochs, but they are free — no extra compute cost. Risk: T_0=10 means the first restart happens at epoch 10, which may be later than the timeout kills training. Student should also try T_0=5.

### Stop condition
If `val_avg/mae_surf_p` after the first restart (epoch 10) is not better than the pre-restart plateau, the restart mechanism is providing no benefit in this regime.

---

## Hypothesis 5: Larger model — wider hidden dim (256) and more heads (8)

### Category
Architecture width

### One-line summary
Double the hidden dimension from 128 to 256 and increase heads from 4 to 8, raising parameter count from ~2M to ~7M, to test whether the baseline is capacity-limited.

### Rationale
The current model (128 hidden, 5 layers) is relatively small for a mesh with 74K–242K nodes. The TandemFoilSet geometry variation (3 domains, wide Re range, OOD camber splits) may require more representational capacity than 128 hidden dims provide. Doubling width is the most standard capacity increase and is independently interpretable.

At 96 GB VRAM with mesh size up to 242K nodes at batch 4, a 256-dim model should still be VRAM-safe — the bottleneck is node count, not parameter count (parameters are reused across nodes in Transolver's architecture).

VRAM estimate (rough): activations per batch for the largest samples at B=4, N=242K, dim=256: 4 * 242K * 256 * 4 bytes ≈ 1.0 GB per layer, 5 layers ≈ 5 GB activations. Safe.

### Concrete implementation

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,      # was 128
    n_layers=5,
    n_head=8,          # was 4 (must divide n_hidden)
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Also reduce batch_size to 2 if VRAM pressure appears (add `--batch_size 2`). The student should first try batch_size=4 and only fall back if OOM.

### Expected delta
Moderate. This is a diagnostic: if the model is capacity-limited, this should help; if it is not, we learn that capacity is not the bottleneck. The wider model processes each batch more slowly — may get 1–2 fewer epochs in the 30-min window, but more steps per epoch with effectively the same total compute.

### Risk / failure mode
Larger model may underfit more in early epochs if the optimizer lr is not adjusted. Consider lr=3e-4 for the wider model (standard practice is to scale lr down slightly with wider models). Student should try lr=3e-4 as an alternative arm.

---

## Hypothesis 6: More Transolver slices (slice_num 128) with reduced batch

### Category
Architecture / slice-attention configuration

### One-line summary
Increase the number of physics-aware slice tokens from 64 to 128 so the model can form finer-grained physical abstractions over the mesh, while reducing batch size to 2 to accommodate the larger attention matrix.

### Rationale
Transolver's slice-attention compresses N mesh nodes into `slice_num` learned tokens, over which full self-attention is applied. With 64 slices, each token averages ~1K–4K nodes (for N=74K–242K). For a tandem airfoil with two separate zones (foil 1, foil 2, background), finer-grained slices may allow the model to separately represent:
- Leading-edge stagnation zone of foil 1
- Trailing-edge wake of foil 1 (which influences foil 2 inlet)
- Surface boundary layer of foil 2
- Background pressure recovery

128 slices also has established precedent in the original Transolver paper (they tested 32–128 on different benchmarks).

### Concrete implementation

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,     # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

CLI: `python train.py --batch_size 2 --slice_num 128` (add `slice_num` to Config dataclass):

```python
@dataclass
class Config:
    ...
    slice_num: int = 64
```

And pass `slice_num=cfg.slice_num` in model_config.

### Expected delta
Small-to-moderate. More slices provide finer spatial decomposition and may help with the tandem geometry where two physically distinct foil zones need to interact through slice tokens.

### Risk / failure mode
More slices means a larger softmax over N nodes (in_project_slice maps to slice_num), increasing attention memory quadratically. At N=242K and slice_num=128, the slice_weights tensor is [B*H, N, 128] = 4*4*242K*128 ≈ 500M float32 = 2 GB — still feasible at batch=2. Monitor for OOM.

---

## Hypothesis 7: Deeper model (7 layers) with layer-drop regularization

### Category
Architecture depth / regularization

### One-line summary
Increase Transolver depth from 5 to 7 layers and apply layer-drop (stochastic depth, p=0.1) to regularize the added depth within the short training window.

### Rationale
Adding depth increases the model's ability to compose hierarchical physical representations (boundary layer → wake → pressure recovery). Without regularization, the extra layers may not receive sufficient gradient in early epochs. Layer-drop (Stochastic Depth, Huang et al. 2016; widely used in ViT and Swin Transformer) randomly drops entire residual blocks during training with linearly increasing probability (0 at layer 0, p at layer L). This regularizes depth without changing inference, and is known to help in regimes where deep models train slowly relative to available budget.

### Concrete implementation

Add a simple stochastic depth wrapper (no new package — pure torch):

```python
import random

class StochasticDepth(nn.Module):
    """Drop an entire residual branch with probability p during training."""
    def __init__(self, module: nn.Module, p: float = 0.1):
        super().__init__()
        self.module = module
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return x   # skip this block
        return self.module(x)
```

Wrap each TransolverBlock (except the last, which has the output projection):

In Transolver.__init__ after the blocks are built:

```python
n_blocks = len(self.blocks)
for i in range(n_blocks - 1):   # don't wrap the last (output) block
    drop_prob = 0.1 * i / max(n_blocks - 2, 1)
    self.blocks[i] = StochasticDepth(self.blocks[i], p=drop_prob)
```

Model config change: `n_layers=7`. No other changes.

### Expected delta
Small-to-moderate. The extra depth is most valuable if the baseline is already near-converged in depth. If 5 layers is already sufficient, the extra 2 layers waste compute. Stochastic depth is cheap (no overhead at inference) and prevents the deeper model from being worse than baseline in early epochs.

---

## Hypothesis 8: SiLU (Swish) activation replacing GELU

### Category
Activation function

### One-line summary
Replace GELU activations in all MLP blocks with SiLU (Swish), which has a stronger gradient near zero and has been shown to improve convergence speed in physics-informed neural networks.

### Rationale
SiLU (f(x) = x * sigmoid(x)) differs from GELU mainly in the tail behavior and gradient near zero. In the neural operator and PDE surrogate literature (Fathony et al. 2021, Raonic et al. 2023 in CNO), smooth non-linearities with strong near-zero gradients have shown consistent improvement over GELU for regression on smooth fields. SiLU is already supported by the `ACTIVATION` dict in the current code (`"silu"` is not there but `nn.SiLU` is a trivial add). This is a zero-risk, single-line test.

### Concrete implementation

In train.py, add `"silu"` to the `ACTIVATION` dict (lines ~53-57):

```python
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,    # add this line
    "relu": nn.ReLU,    # keep for completeness
}
```

Then launch with: `python train.py --act silu`

(The `act` parameter is already threaded through `TransolverBlock` and `MLP`.)

### Expected delta
Small. Activation choice rarely dominates, but it is a free hypothesis — one flag change, completely isolated from other variables, fast to check. The test is a clean ablation.

---

## Hypothesis 9: Gradient clipping + higher LR (aggressive early-training dynamics)

### Category
Optimizer

### One-line summary
Raise the learning rate to 2e-3 (4x baseline) and add gradient norm clipping at 1.0 to allow large parameter updates early in training without divergence — so the model traverses more of parameter space in the first 5–10 epochs.

### Rationale
With only 5–15 epochs in the wall-clock budget, the model must learn quickly. The baseline lr=5e-4 is conservative. In recent work on neural operators (e.g., FNO, GNO) the recommended starting lr is often 1e-3 to 5e-3, with gradient clipping to prevent early instability. Gradient clipping at norm=1.0 is standard in transformer training (CLIP-norm used in ViT, GPT, etc.) and the current Transolver does not clip.

The combination of high LR + clip is safer than high LR alone, as clipping prevents a single large-gradient batch from destabilizing the current optimum.

### Concrete implementation

```python
@dataclass
class Config:
    lr: float = 2e-3          # was 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0    # new
    ...
```

In the training loop, after `loss.backward()` and before `optimizer.step()`:

```python
if cfg.grad_clip > 0:
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

### Expected delta
Could be large positive (faster convergence, more epochs see meaningful progress) or large negative (too much noise, unstable training). This is a discriminating diagnostic: if the model's first-epoch val metrics are much better with lr=2e-3, we know the conservative lr is a bottleneck. If they are worse, we learn that the baseline lr is already well-tuned.

Monitor `train/surf_loss` epoch-by-epoch. A good run should see surf_loss drop steeply in epochs 1–3. An unstable run will oscillate or diverge.

---

## Hypothesis 10: Input feature augmentation — log(dsdf) and Re-scaled position

### Category
Input feature engineering

### One-line summary
Apply log1p transform to the 8-dimensional dsdf distance features (dims 4–11) before passing them to the model, and add a Re-scaled position feature (x * log(Re)) so the model can distinguish near-wall from far-field nodes differently at different Reynolds numbers.

### Rationale
The dsdf features encode distance to foil surfaces. Distance fields are inherently log-scale: a node 0.001m from the surface has qualitatively different physics from a node 0.01m away, even though the raw difference is only 0.009. log1p(dsdf) compresses the far-field range and spreads the near-wall range where physics is most complex. This is standard practice in CFD feature engineering (e.g., SDF-based neural PDE approaches use log-scale distances).

The Re-scaled position `x_pos * log(Re)` (where x_pos = dims 0–1) encodes the fact that at high Re the boundary layer is thinner in physical coordinates — a point at the same physical distance has different local physics at Re=100K vs Re=5M. This is a cheap cross-feature that does not change the model contract.

All transformations applied in-place in train.py after loading x, before computing x_norm.

### Concrete implementation

In the training loop and evaluate_split function, after loading x:

```python
def preprocess_x(x: torch.Tensor, log_re_idx: int = 13,
                 dsdf_start: int = 4, dsdf_end: int = 12,
                 pos_dims: list[int] = [0, 1]) -> torch.Tensor:
    """Apply log-scale transforms to distance features and add Re-scaled position."""
    x = x.clone()
    # log1p on distance fields (they are non-negative)
    x[..., dsdf_start:dsdf_end] = torch.log1p(x[..., dsdf_start:dsdf_end].clamp(min=0.0))
    # Re-scaled position cross-feature: replace dims 0,1 with x_pos * log(Re)
    # Append as extra feature would break 24-dim contract; instead scale in-place
    # (Re is already log-encoded in dim 13 — use it directly)
    log_re = x[..., log_re_idx:log_re_idx+1]  # [B, N, 1]
    for dim in pos_dims:
        x[..., dim] = x[..., dim] * log_re[..., 0]
    return x
```

Call `x = preprocess_x(x)` before normalization in both the training loop and evaluate_split.

Important: after this transform, the x_mean/x_std stats no longer match. Two options: (a) recompute running stats online (adds complexity), (b) skip normalization for the transformed dims and let the model normalize via its LayerNorms. Option (b) is simpler: apply preprocess_x before x_norm, and accept that the stats.json normalization is now an approximation. The model's LayerNorms will adapt.

### Expected delta
Moderate. Log-scale distance is a principled transform; the Re-position cross-feature is more speculative. Risk: the stats.json normalization was computed on the original features, so applying it to log-transformed features will distort the normalization. The student should add a note in W&B on whether val metrics diverge early.

### Stop condition
If `val_avg/mae_surf_p` epoch 1 is significantly worse than baseline (>20%), abandon — the stats mismatch may be too large.

---

## Hypothesis 11: Separate output heads per physical field

### Category
Architecture — output head

### One-line summary
Replace the single shared last-layer MLP (which outputs [Ux, Uy, p] together) with three separate output heads — one per physical field — sharing the same final hidden representation.

### Rationale
The current architecture uses one `mlp2` in the last TransolverBlock to project from hidden_dim to 3 output channels simultaneously:

```python
self.mlp2 = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
    nn.Linear(hidden_dim, out_dim),  # out_dim=3
)
```

Pressure and velocity fields have fundamentally different physical scales and boundary conditions (pressure is a scalar potential; velocities are divergence-related). Separate output heads allow each field to have its own projection with independent weight scaling, which may help the model de-correlate the three channels during gradient descent.

This is a standard trick in multi-task prediction: shared backbone + independent heads.

### Concrete implementation

Replace the last `mlp2` block in `TransolverBlock` when `last_layer=True`:

```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Three separate heads for Ux, Uy, p
    self.head_Ux = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 1),
    )
    self.head_Uy = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 1),
    )
    self.head_p = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 1),
    )
```

Forward pass of last layer:

```python
if self.last_layer:
    h = self.ln_3(fx)
    return torch.cat([self.head_Ux(h), self.head_Uy(h), self.head_p(h)], dim=-1)
```

Model contract unchanged: still outputs [B, N, 3].

### Expected delta
Small-to-moderate. The gain comes from allowing field-specific projections. This is a well-established pattern in multi-task learning, and surface pressure specifically may benefit from having its own head (since the pressure field has different spatial structure from velocity). Minimal compute overhead.

---

## Hypothesis 12: Mixed precision training (bfloat16) + larger effective batch via grad accumulation

### Category
Mixed precision / throughput

### One-line summary
Enable bfloat16 autocast for the forward and loss pass, then compensate for the per-step noise by accumulating gradients over 2 mini-batches before stepping — net effect is 2x effective batch size with the same VRAM, potentially more stable training.

### Rationale
The current baseline uses full float32 throughout. On a 96 GB VRAM H100/A100, bfloat16 AMP halves the activation memory per batch, allowing either larger effective batches or faster epoch throughput — more training steps within the 30-min window. bfloat16 (not float16) is preferred because its exponent range matches float32, avoiding the overflow issues common with float16 in scientific regression tasks.

Gradient accumulation (every 2 steps) compensates for the halved per-batch sample count by effectively using batch_size=8. This can improve gradient SNR for the WeightedRandomSampler whose domain balance is approximate at small batch sizes.

### Concrete implementation

```python
@dataclass
class Config:
    ...
    amp: bool = True             # enable bfloat16 AMP
    grad_accum: int = 2          # gradient accumulation steps
```

Training loop changes:

```python
scaler = torch.amp.GradScaler("cuda", enabled=False)   # bfloat16 doesn't need scaler
accum_steps = cfg.grad_accum
optimizer.zero_grad()

for step_idx, (x, y, is_surface, mask) in enumerate(tqdm(train_loader, ...)):
    ...
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        sq_err = (pred.float() - y_norm.float()) ** 2  # loss in float32
        ...
        loss = (vol_loss + cfg.surf_weight * surf_loss) / accum_steps

    loss.backward()

    if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        wandb.log({"train/loss": loss.item() * accum_steps, "global_step": global_step})
```

### Expected delta
Primarily a throughput hypothesis: the model trains more epochs in 30 min. If the baseline is epoch-starved (only 5–8 epochs), more epochs = better validation. The bfloat16 precision is sufficient for this regression task. Risk: bfloat16 can cause instability in LayerNorm and softmax in edge cases — monitor for NaN in early epochs.

---

## Hypothesis 13: Re-conditioned position encoding (Fourier position encoding scaled by log(Re))

### Category
Input feature engineering / physics-informed encoding

### One-line summary
Replace the raw (x, z) position dims with a Fourier position encoding at frequencies conditioned on log(Re) — so the spatial frequency basis adapts to the expected boundary layer thickness at each Reynolds number.

### Rationale
In CFD, the relevant spatial scales near the airfoil boundary layer depend on Re: the boundary layer thickness scales as ~1/sqrt(Re). A Fourier encoding whose frequencies scale with sqrt(Re) would naturally adapt the model's spatial resolution to the problem's physical scale. This is a physics-informed input representation that the model cannot discover on its own from raw (x, z) coordinates.

This borrows from Neural Implicit Flow (Liu et al. 2022) and Re-dependent FNO, but the key insight is that the position encoding itself should be Re-dependent, not just the features.

### Concrete implementation

In train.py, replace dims 0–1 of x (position) with Re-scaled Fourier features before normalization:

```python
def re_scaled_fourier_pos(x: torch.Tensor, n_freqs: int = 4,
                           re_dim: int = 13, pos_dims: list = [0, 1]) -> torch.Tensor:
    """
    Replace dims 0,1 with Re-scaled Fourier features.
    Output has same shape as input (24 dims) — first 2 dims replaced.
    Actually we replace dim 0 and 1 with 1 frequency each (sin only)
    to stay at 24 dims total.
    """
    x = x.clone()
    log_re = x[..., re_dim]  # [B, N] — already log(Re) encoded
    # Re-scaled frequency: higher Re = higher frequency (finer resolution near wall)
    # We use one frequency per position dim to stay within the 24-dim budget
    for i, pdim in enumerate(pos_dims):
        pos = x[..., pdim]
        freq = torch.exp(log_re * 0.5)   # ~sqrt(Re) scaling
        x[..., pdim] = torch.sin(2 * torch.pi * pos * freq)
    return x
```

Call before normalization in training loop and evaluate_split.

Note: this changes the statistical distribution of dims 0–1, so stats.json normalization for those dims will be approximate. As with Hypothesis 10, the model LayerNorms will partially compensate.

### Expected delta
Speculative. This is a physics-motivated feature that has strong theoretical justification but has not been tested in this exact setting. Could show large gains if the model is currently underfitting the Re-position interaction; could hurt if the stats mismatch is too large. Best run after Hypothesis 10 (simpler log transform) has been validated.

### Taste rating (pre-score)
Mechanistic: 3 (clear physics motivation, specific observable — does the OOD val_re_rand split improve?)
Research-state value: 3 (tests a hypothesis about Re-position interaction that no other experiment covers)
Execution: 2 (stats mismatch risk is real; should be preceded by H10)

---

## Hypothesis 14: Spare — Larger slice_num (256) on single cruise domain diagnostic

### Category
Architecture / diagnostic

### One-line summary
A pure diagnostic: hold everything fixed at baseline and only increase `slice_num` to 256 with `batch_size=1` on cruise domain only (debug=False, but limit epochs to 5) to measure whether the model is token-budget-limited on the large 210K-node cruise meshes.

### Rationale
The cruise domain has ~210K nodes — nearly 3x the raceCar single domain. With 64 slices, each slice token averages ~3K cruise nodes vs ~1.3K raceCar nodes. If the model consistently underperforms on the cruise OOD split (`val_geom_camber_cruise`), insufficient slice resolution may be the cause. This diagnostic tests that hypothesis directly.

### Concrete implementation
Same as H6 but slice_num=256, batch_size=1. Primarily a diagnostic — the student should run this only if H6 shows a positive signal. Spare hypothesis if another slot opens.

---

## Summary Table

| # | Title | Category | Predicted delta | Single change |
|---|-------|----------|-----------------|---------------|
| H1 | Per-sample Huber-normalized loss | Loss | Moderate (improves low-Re/OOD) | Replace sq_err with per-sample-scaled Huber |
| H2 | Pressure channel upweight in surf loss | Loss | Small-moderate | Add `p_weight=3.0` to surf sq_err |
| H3 | surf_weight 10→30 | Loss weight | Potentially large, shows early | `--surf_weight 30` |
| H4 | LR warmup + CosWarmRestarts(T_0=10) | Schedule | Small-moderate | Replace CosAnneal with SequentialLR |
| H5 | Wider model: n_hidden=256, n_head=8 | Architecture width | Moderate | n_hidden=256, n_head=8 |
| H6 | More slices: slice_num=128 | Slice attention | Small-moderate | slice_num=128, batch_size=2 |
| H7 | Deeper model (7 layers) + stochastic depth | Architecture depth | Small-moderate | n_layers=7 + StochasticDepth wrapper |
| H8 | SiLU activation | Activation | Small | `--act silu` (add nn.SiLU to ACTIVATION) |
| H9 | High LR (2e-3) + grad clip (1.0) | Optimizer | Large or large negative | lr=2e-3, grad_clip=1.0 |
| H10 | log1p(dsdf) + Re-scaled position | Input features | Moderate | preprocess_x() before normalization |
| H11 | Separate output heads per field | Architecture output | Small-moderate | Split mlp2 into 3 per-field heads |
| H12 | bfloat16 AMP + grad accum=2 | Throughput | Moderate (more epochs) | autocast bfloat16 + accum 2 steps |
| H13 | Re-conditioned Fourier position | Input features | Speculative / high upside | sin(2π * pos * sqrt(Re)) for dims 0-1 |
| H14 | slice_num=256 cruise diagnostic (spare) | Diagnostic | Diagnostic | slice_num=256, batch_size=1 |

## Assignment priority recommendation

Assign first to the 8 students (priority order, highest-signal first):

1. **H3** (surf_weight=30) — simplest, most direct, shows early. Assign to fastest student.
2. **H1** (per-sample Huber) — most principled loss fix for the Re-scale problem.
3. **H9** (high LR + grad clip) — discriminating diagnostic: pass/fail in epoch 1-2.
4. **H5** (wider model 256) — standard capacity diagnostic.
5. **H2** (p_weight=3.0) — targeted at the exact metric being ranked on.
6. **H12** (bfloat16 + grad accum) — throughput: get more epochs for free.
7. **H11** (separate output heads) — clean architectural change, low risk.
8. **H6** (slice_num=128) — tests slice resolution; moderate risk.

Spares for round 2 or if conflicts arise:
- H4 (schedule warmup + restarts)
- H7 (deeper + stochastic depth)
- H8 (SiLU activation)
- H10 (log1p dsdf + Re-position)
- H13 (Re-Fourier position)
- H14 (slice_num=256 diagnostic)
