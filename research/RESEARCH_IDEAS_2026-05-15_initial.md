<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TandemFoilSet Research Hypotheses — 2026-05-15

Generated from code audit of `train.py` + `program.md` + literature survey.
Target metric: `val_avg/mae_surf_p` (lower is better).
Baseline: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, ~3M params,
AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR, batch=4, surf_weight=10, MSE loss, no warmup, no grad clip.

Hypotheses are ranked by (expected impact x tractability). Each is a single, isolatable change.

---

## H1 — Gradient Clipping + LR Warmup

**One-line statement:** Add gradient norm clipping (max_norm=1.0) and a short linear warmup
(5% of total steps) to the existing cosine schedule, preventing early divergence on high-Re samples.

**Predicted delta on val_avg/mae_surf_p:** -3% to -8% (medium confidence)

**Implementation — `train.py` only:**
1. Replace the scheduler with a linear warmup then cosine decay using a `LambdaLR`:

```python
warmup_steps = max(1, int(0.05 * MAX_EPOCHS * len(train_loader)))
total_steps  = MAX_EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

Call `scheduler.step()` after every optimizer step (not per epoch). Remove the current
`CosineAnnealingLR` line and the per-epoch `scheduler.step()` call.

2. After `loss.backward()` and before `optimizer.step()`, add:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. Add `import math` at the top if not present.

**Why it should work:**
The baseline has no warmup: on epoch 1 the large gradients driven by high-Re samples
(y values up to ±29K) can destabilize the layer-norm and slice-projection weights before
they reach a useful basin. Warmup is standard practice for transformers (see Vaswani 2017,
Chen 2021) and is especially important when the per-sample loss variance is high — as it is
here due to the order-of-magnitude Re spread. Gradient clipping bounds the step size whenever
a single extreme sample dominates the batch, which is likely when 1 of 4 batch samples is
high-Re cruise.

**Risk / failure mode:**
If warmup_steps is too large relative to available epochs (e.g., debug mode with 3 epochs),
the model never reaches peak LR. Validate that warmup_steps < 0.15 * total_steps. LambdaLR
step-level scheduling increases logging complexity slightly — use `scheduler.get_last_lr()[0]`
for the W&B LR log, same as now.

---

## H2 — Huber Loss (Surface Pressure Channel Only)

**One-line statement:** Replace the MSE surf_loss with a Huber (smooth-L1) loss for the
pressure channel only, aligning training signal with the MAE evaluation metric while keeping
MSE for velocities where the baseline is already reasonable.

**Predicted delta on val_avg/mae_surf_p:** -5% to -12% (medium-high confidence)

**Implementation — `train.py` only:**
Replace the loss block inside the training loop (lines ~492-496):

```python
sq_err = (pred - y_norm) ** 2

# Per-channel split: channels 0=Ux, 1=Uy, 2=p
delta = 1.0  # Huber threshold in normalized units; tune if needed

# Pressure channel Huber
p_err  = pred[..., 2:3] - y_norm[..., 2:3]
p_loss = torch.where(p_err.abs() < delta,
                     0.5 * p_err ** 2,
                     delta * (p_err.abs() - 0.5 * delta))

# Velocity channels MSE (unchanged)
vel_sq_err = sq_err[..., :2]

vol_mask_f  = vol_mask.unsqueeze(-1)
surf_mask_f = surf_mask.unsqueeze(-1)

vol_loss  = ((vel_sq_err * vol_mask_f).sum()  / vol_mask.sum().clamp(min=1)
           + (p_loss[..., 0:1] * vol_mask_f).sum()  / vol_mask.sum().clamp(min=1))
surf_loss = ((vel_sq_err * surf_mask_f).sum() / surf_mask.sum().clamp(min=1)
           + (p_loss[..., 0:1] * surf_mask_f).sum() / surf_mask.sum().clamp(min=1))

loss = vol_loss + cfg.surf_weight * surf_loss
```

`delta=1.0` in normalized pressure space corresponds to roughly 1 std deviation of `p`. Start
there; if the model collapses velocities try `delta=0.5`.

**Why it should work:**
The primary metric is MAE (L1), but the baseline trains with MSE (L2). MSE amplifies the
contribution of outlier nodes — in particular the extreme stagnation and suction-peak nodes
near the leading edge at high Re — relative to the dense mid-chord surface where most
evaluation nodes live. Huber is a continuous bridge between L2 (small errors, stable
gradients) and L1 (large errors, outlier-robust) and directly minimizes a loss consistent
with MAE. The asymmetry (Huber only for p) avoids disrupting the already-adequate velocity
predictions. This is the classic "loss-metric alignment" lever and is often the single
largest cheap win in regression surrogates.

**Risk / failure mode:**
Huber gradient magnitude near zero is lower than MSE when errors are sub-threshold, which
can slow early-epoch convergence. If val_avg/mae_surf_p is worse at epoch 5 vs. MSE baseline,
increase delta to 2.0 (makes the transition region larger and gradient magnitude closer to MSE).
Do not use pure L1: near zero it has undefined gradient in float32, causing NaN accumulation
in some torch.autograd paths.

---

## H3 — Per-Channel Pressure-Weighted Surface Loss

**One-line statement:** Upweight the pressure channel specifically on surface nodes by an
additional factor of 3–5x beyond the existing 10x surface weight, since `mae_surf_p` is the
only metric that governs ranking.

**Predicted delta on val_avg/mae_surf_p:** -4% to -10% (medium confidence); likely trades
velocity surface MAE for pressure gains.

**Implementation — `train.py` only:**
Add a `p_surf_weight: float = 3.0` field to `Config`. Modify the loss block:

```python
surf_vel_loss = (sq_err[..., :2] * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
surf_p_loss   = (sq_err[..., 2:3] * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * (surf_vel_loss + cfg.p_surf_weight * surf_p_loss)
```

Recommended sweep: p_surf_weight in {2.0, 3.0, 5.0}. Start with 3.0.

**Why it should work:**
The current loss gives equal weight to Ux, Uy, and p on surface nodes. But the ranking metric
only measures surface pressure. Lifting the gradient signal for the pressure channel directs
the model's capacity toward the performance-critical variable. This is analogous to auxiliary
task weighting in multi-task learning. The risk is that some pressure structure is inferred
through velocity coupling, so extreme overweighting can hurt consistency; 3x is conservative.

**Risk / failure mode:**
Overly aggressive p_surf_weight (>10x) will cause the velocity surface predictions to degrade,
which may indirectly degrade p if the model uses velocity information in the attention to
infer pressure gradients. Monitor mae_surf_Ux and mae_surf_Uy alongside mae_surf_p.

---

## H4 — Scale-Up Model: n_hidden=256, n_layers=8, n_head=8

**One-line statement:** Double the hidden dimension, increase depth to 8 layers and heads to 8
to exploit the 96GB VRAM headroom; the 3M-parameter baseline is severely under-parameterized
relative to the task complexity (3 domains, 74K-242K nodes, OOD geometry + Re).

**Predicted delta on val_avg/mae_surf_p:** -8% to -20% (medium-high confidence)

**Implementation — `train.py` only:**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=8,
    n_head=8,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

This gives ~20M parameters. Memory estimate: at batch=4, N_max~242K, hidden=256, the
dominant cost is the preprocess MLP input projection: 4 * 242000 * 256 * 4 bytes ≈ 1GB per
batch element, well within 96GB. Run with `--batch_size 2` if OOM occurs.

Also lower lr to `3e-4` since wider/deeper models benefit from smaller initial steps.

**Why it should work:**
The Transolver paper uses n_hidden=256-512 for competitive benchmarks. The TandemFoilSet
baseline uses n_hidden=128 — half the paper's standard setting. The task here is harder than
most standard benchmarks (variable mesh, 3 domains, OOD geometry). The literature (scaling
laws for PDE surrogates, e.g. Herde et al. 2024 Poseidon) consistently shows that capacity
scaling improves on all splits when the training set is large enough (1499 samples is
sufficient for this size model). The 96GB VRAM makes this essentially a free lunch.

**Risk / failure mode:**
Larger model is slower per epoch. With a wall-clock timeout, fewer epochs are completed. If
the bigger model has not converged by timeout, the result may be worse than the baseline.
Mitigate by using a slightly higher lr with warmup (H1) to reach a better basin faster.

---

## H5 — Exponential Moving Average (EMA) of Model Weights

**One-line statement:** Maintain an EMA of the model weights (decay=0.999) and use the EMA
model for validation/test, which typically improves generalization without any training cost.

**Predicted delta on val_avg/mae_surf_p:** -2% to -6% (high confidence, very low risk)

**Implementation — `train.py` only:**
Add after model instantiation:

```python
# EMA state: a plain dict of parameter tensors
ema_params = {name: param.data.clone() for name, param in model.named_parameters()}
EMA_DECAY = 0.999

def update_ema(model, ema_params, decay=EMA_DECAY):
    with torch.no_grad():
        for name, param in model.named_parameters():
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
```

Call `update_ema(model, ema_params)` after each `optimizer.step()`.

Before evaluation, swap EMA weights in:

```python
# Before evaluate_split / val loop
orig_params = {name: param.data.clone() for name, param in model.named_parameters()}
for name, param in model.named_parameters():
    param.data.copy_(ema_params[name])

# ... run evaluate_split ...

# Restore training weights
for name, param in model.named_parameters():
    param.data.copy_(orig_params[name])
```

Also use EMA weights for checkpoint saving and test evaluation. The EMA buffer adds ~3M
extra floats (128 hidden, ~3M params) — trivial memory.

**Why it should work:**
EMA acts as an ensemble of model snapshots along the training trajectory. For irregular-mesh
transformers that train with stochastic batching (WeightedRandomSampler introduces high
variance), the EMA provides a smoother, lower-variance estimator at validation time. This is
a near-zero-cost improvement that is universal across ML: used in ImageNet (PyTorch Image
Models), diffusion models, LLMs (DeepSeek, Llama), and specifically reported as beneficial
for operator learning in recent physics surrogate papers (e.g., UNO, GNOT, FactFormer).

**Risk / failure mode:**
If training is highly non-stationary (e.g., oscillating val loss with LR restart), EMA may
lag behind the sharpest descent. Decay=0.999 gives a half-life of ~700 steps — at batch=4
with 1499 samples that is ~2 epochs. This is well-matched to the epoch budget. Start with
decay=0.999; if val MAE with EMA is worse than without (compare at epoch 10), drop to 0.995.

---

## H6 — Physics-Symmetry Data Augmentation: Horizontal Flip + AoA Sign Flip

**One-line statement:** Double the effective training set by flipping each sample left-right
(negate x-coordinate and Ux, negate AoA for both foils), exploiting the exact mirror symmetry
of incompressible flow at zero ground effect.

**Predicted delta on val_avg/mae_surf_p:** -3% to -10% (medium confidence); impact largest
on OOD camber splits where training diversity is sparse.

**Implementation — `train.py` only:**
Add an augmentation function and apply it at the start of the training batch loop:

```python
def flip_augment(x, y, is_surface, mask, p_flip=0.5):
    """Left-right flip: negate x-coord (dim 0), Ux (channel 0), AoA dims (14, 18).
    
    x layout: [0]=pos_x, [1]=pos_z, [2-3]=saf, [4-11]=dsdf, [12]=is_surf,
               [13]=log(Re), [14]=AoA1, [15-17]=NACA1, [18]=AoA2, [19-21]=NACA2,
               [22]=gap, [23]=stagger
    y layout: [0]=Ux, [1]=Uy, [2]=p
    
    Under horizontal flip: x -> -x, Ux -> -Ux, AoA -> -AoA.
    Uy and p are unchanged (Uy is along z, p is scalar).
    saf (arc-length) sign also flips: dims 2-3.
    """
    if torch.rand(1).item() > p_flip:
        return x, y, is_surface, mask

    x = x.clone()
    y = y.clone()
    x[..., 0]  = -x[..., 0]   # pos_x
    x[..., 2]  = -x[..., 2]   # saf_x
    x[..., 3]  = -x[..., 3]   # saf_z (sign depends on orientation; test empirically)
    x[..., 14] = -x[..., 14]  # AoA foil 1
    x[..., 18] = -x[..., 18]  # AoA foil 2
    y[..., 0]  = -y[..., 0]   # Ux
    return x, y, is_surface, mask
```

Call inside the training loop after tensors are moved to device, before normalization:

```python
x, y, is_surface, mask = flip_augment(x, y, is_surface, mask, p_flip=0.5)
x_norm = (x - stats["x_mean"]) / stats["x_std"]
# ... rest unchanged
```

Note: The dsdf features (dims 4-11) encode distances from foil surfaces. Under pure
left-right flip, the distance values are unchanged (distances are unsigned), but the
directionality of the gradient descriptor may flip in sign for some components. Empirically
test whether including dsdf sign flips improves or hurts. Start without flipping dsdf.

**Why it should work:**
The incompressible Navier-Stokes equations are equivariant under reflections of the flow
domain. For the raceCar domain (ground effect with negative AoA), the foil is inverted, but
flipping left-right still produces a valid physical configuration. For cruise (freestream),
the symmetry is exact. This is the same augmentation applied in NeuralFoil and in most
aeroacoustic surrogate papers (e.g., NACA augmentation in Bonfiglioli et al. 2023). The
cruise OOD camber split has only 443 training samples; doubling effective training set
size via valid physical augmentation is low-risk high-reward.

**Risk / failure mode:**
The dsdf features (dims 4-11) are the trickiest — they are computed from the actual mesh
geometry and may encode directional information that is not equivariant under the flip.
If the flip augmentation makes val loss worse at epoch 5, the dsdf sign treatment is the
likely culprit. Test with p_flip=0.1 first (adds only 10% augmented samples) before 0.5.
The stagger feature (dim 23) may also need sign treatment depending on convention — check
program.md: gap ~[-0.8, 1.6], stagger ~[0.0, 2.0]; stagger is likely a signed offset and
should be negated.

---

## H7 — Global Conditioning Tokens: log(Re) + NACA as Cross-Attention Context

**One-line statement:** Prepend 2–4 global "regime tokens" derived from the scalar flow
conditions (log(Re), AoA foil 1, NACA foil 1, gap/stagger) as key-value pairs in a
cross-attention step before the slice tokens, explicitly anchoring each forward pass to its
operating regime.

**Predicted delta on val_avg/mae_surf_p:** -5% to -15% on OOD splits (medium confidence,
higher implementation cost than H1-H6)

**Implementation — `train.py` only:**
This implements the core idea from GeoTransolver (NVIDIA, arxiv 2512.20399, Dec 2025) in a
minimal form. Add a `GlobalCondBlock` that performs one cross-attention step on the slice
tokens conditioned on a small set of global feature vectors:

```python
class GlobalCondBlock(nn.Module):
    """Cross-attention: slice tokens attend to global regime descriptors."""
    def __init__(self, hidden_dim, n_global_tokens=4, n_head=4):
        super().__init__()
        self.n_global_tokens = n_global_tokens
        # Projects scalar features to token space
        self.global_proj = nn.Linear(n_global_tokens, hidden_dim)
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.n_head = n_head

    def forward(self, fx, x_raw):
        # x_raw: [B, N, 24] — raw (unnormalized or normalized) node features
        # Global features: take mean over real nodes (use mask externally if needed)
        # Extract scalar conditions from a representative node (they're uniform per sample)
        global_feats = x_raw[:, 0, [13, 14, 15, 22]].unsqueeze(1)  # [B, 1, 4]
        g_tok = self.global_proj(global_feats)                        # [B, 1, H]
        
        # fx: [B, N, H] — node features before slicing
        # Pool to get per-sample mean representation
        q = self.cross_attn_q(fx.mean(dim=1, keepdim=True))          # [B, 1, H]
        k = self.cross_attn_k(g_tok)                                  # [B, 1, H]
        v = self.cross_attn_v(g_tok)                                  # [B, 1, H]
        
        head_dim = q.shape[-1] // self.n_head
        def split_heads(t):
            B, S, H = t.shape
            return t.view(B, S, self.n_head, head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v)
        )  # [B, heads, 1, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(q.shape)  # [B, 1, H]
        
        # Broadcast delta back to all nodes
        fx = fx + self.out_proj(attn_out)
        fx = self.ln(fx)
        return fx
```

Add one `GlobalCondBlock` after `self.preprocess` in `Transolver.forward`:

```python
def forward(self, data, **kwargs):
    x = data["x"]
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    fx = self.global_cond(fx, x)  # <-- new
    for block in self.blocks:
        fx = block(fx)
    return {"preds": fx}
```

Instantiate in `__init__`: `self.global_cond = GlobalCondBlock(n_hidden, n_global_tokens=4, n_head=n_head)`.

Use the normalized x (x_norm) as input to forward, so global_feats at dims 13, 14, 15, 22
are the normalized log(Re), AoA1, NACA1_camber, gap features.

**Why it should work:**
The baseline Transolver processes every sample identically — the scalar operating conditions
(Re, AoA, geometry) influence the output only through the node feature MLP. But the slice
tokens learn a single global compression of the field regardless of domain. GeoTransolver
(arxiv 2512.20399) showed that adding geometry/BC cross-attention to these tokens at NVIDIA
scale improved generalization to unseen geometries by 15–30% on DrivAerML and SHIFT-Wing.
Our val_geom_camber splits test exactly this axis — unseen front-foil camber. Conditioning
the slice tokens on the scalar flow parameters essentially gives each slice a regime-aware
embedding, making the compression more transferable across domains.

**Risk / failure mode:**
The global feature dims (13=log(Re), 14=AoA1, 22=gap) are already present in every node's
feature vector, so the model has access to them. The benefit is in routing that information
explicitly to the compressed token representation, not in adding new information. If the
model already extracts this information efficiently via the preprocess MLP, the effect will
be small. The cross-attention architecture above is a simplified variant; the full GALE from
GeoTransolver uses ball-query geometry neighbors, which we omit for simplicity.

---

## H8 — Stochastic Depth / DropPath Regularization

**One-line statement:** Add stochastic depth (drop probability 0.1–0.15 linearly increasing
per layer) to the TransolverBlocks, which acts as a cheap ensemble and improves OOD
generalization on the unseen-camber and Re-holdout splits.

**Predicted delta on val_avg/mae_surf_p:** -2% to -6% on OOD splits (medium confidence)

**Implementation — `train.py` only:**
Use `timm.layers.DropPath` (already available via `timm` in `pyproject.toml`):

```python
from timm.layers import trunc_normal_, DropPath
```

Modify `TransolverBlock.__init__` to accept a `drop_path_rate` parameter:

```python
class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 drop_path_rate=0.0):  # <-- add
        super().__init__()
        # ... existing code ...
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, fx):
        fx = self.drop_path(self.attn(self.ln_1(fx))) + fx   # <-- wrap attn
        fx = self.drop_path(self.mlp(self.ln_2(fx))) + fx    # <-- wrap mlp
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx
```

In `Transolver.__init__`, assign linearly increasing drop path rates:

```python
dpr = [x.item() for x in torch.linspace(0, 0.10, n_layers)]
self.blocks = nn.ModuleList([
    TransolverBlock(
        ...,
        drop_path_rate=dpr[i],
    )
    for i in range(n_layers)
])
```

Add `drop_path_rate: float = 0.0` to `model_config` so it is logged.

**Why it should work:**
Stochastic depth (Huang et al. 2016) prevents co-adaptation between layers and is
empirically validated for vision and scientific ML transformers. `timm` implements it
efficiently. The linear schedule (0 at layer 0, 0.10 at last layer) is the standard recipe.
DropPath acts as an ensemble of sub-networks: the model must learn representations that are
valid across multiple sub-paths, which is a form of domain randomization that may help when
the val splits see out-of-distribution geometry or Re. Cost is essentially free — DropPath
adds only a random Bernoulli mask at training time.

**Risk / failure mode:**
At 5 layers and small drop_path_rate=0.10, the effective ensemble effect is modest. The
risk is that on a small training set (1499 samples), dropout-family regularization can slow
convergence, meaning the model needs more epochs to reach the same train loss. With a timeout,
this can hurt. Run at 0.05 first and only go to 0.10 if val OOD splits improve.

---

## H9 — Per-Sample Normalization by sqrt(Re) for Scale Invariance

**One-line statement:** Divide the target y by a per-sample Re-dependent scale factor
(proportional to sqrt(Re) or Re) before loss computation, then undo before MAE, to give
equal gradient contribution to low-Re and high-Re samples regardless of physical value range.

**Predicted delta on val_avg/mae_surf_p:** -3% to -8% on Re-holdout split, potentially mixed on others (medium confidence)

**Implementation — `train.py` only:**
The x features already include dim 13 = log(Re). The Re value is:

```python
# After normalizing x:
log_re_norm = x_norm[..., 0:1, 13:14]  # shape [B, 1, 1] — uniform per sample
log_re = log_re_norm * stats["x_std"][13] + stats["x_mean"][13]  # denorm to log(Re)
re_scale = log_re.exp().sqrt().clamp(min=1.0)  # [B, 1, 1] — sqrt(Re), broadcast-ready
```

Apply before loss:

```python
y_norm_scaled   = y_norm   / re_scale   # scale-normalized targets
pred_scaled     = pred      / re_scale   # scale-normalized predictions
sq_err_scaled   = (pred_scaled - y_norm_scaled) ** 2

vol_loss  = (sq_err_scaled * vol_mask.unsqueeze(-1)).sum()  / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_scaled * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Evaluation (val and test) still uses unscaled pred and y via `evaluate_split` — no changes
needed there since MAE is always computed in physical space.

**Why it should work:**
At high Re (5M), y-std can reach 2077 (from program.md). At low Re (100K), y-std is roughly
an order of magnitude smaller. With global normalization and MSE loss, high-Re samples
contribute ~100x more to the gradient than low-Re samples even after the WeightedRandomSampler
balances domain counts. The Re-dependent rescaling equalizes gradient contribution across the
Re range, which should improve generalization on the val_re_rand split (stratified Re holdout)
and on low-Re samples. This is a form of uncertainty-weighted or heteroscedastic loss
calibration common in multi-scale physical systems (see Weiner et al. 2023 for neural PDE
surrogates).

**Risk / failure mode:**
sqrt(Re) is a heuristic; the actual scaling law of pressure with Re in the RANS equations is
Cp = p / (0.5 * rho * U^2) — i.e., the physically correct normalization is by Re, not
sqrt(Re). The appropriate exponent depends on the flow regime. If the val_avg metric does
not improve but val_re_rand does, that is a useful signal about model calibration even if
it does not help ranking. Test with both Re and sqrt(Re) exponents.

---

## H10 — slice_num Tuning: Increase to 128 for Larger Meshes

**One-line statement:** Increase `slice_num` from 64 to 128 to give the physics-aware
attention more discrete physics regions for the cruise domain's larger meshes (up to 242K
nodes), where 64 slices may under-resolve the flow field.

**Predicted delta on val_avg/mae_surf_p:** -3% to -8% on cruise split; mixed on raceCar (medium confidence)

**Implementation — `train.py` only:**
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,   # <-- changed from 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

This changes `in_project_slice` from [dim_head=32, 64] to [32, 128] and the SDPA from
64×64 to 128×128 — still O(1) in N, essentially no compute increase for large meshes.
The slice-norm denominator may need more careful initialization; the orthogonal init on
`in_project_slice.weight` is preserved.

**Why it should work:**
The Transolver paper (Cao 2024, Table 4) shows that slice_num is the most sensitive
hyperparameter — too few slices under-compress (noisy tokens), too many over-compress
(loss of spatial resolution). The optimal value scales with mesh resolution. With N=242K
(cruise) vs N=85K (raceCar single), the ratio is ~2.8x, suggesting that the optimal
slice_num for cruise is ~1.8x higher than for raceCar. The current 64 is tuned for the
average mesh; increasing to 128 should help cruise at low cost to raceCar.

**Risk / failure mode:**
Increasing slice_num also increases the variance of the slice assignments early in training
(more slices need to differentiate themselves). This can slow convergence. If val loss is
worse at epoch 10 compared to slice_num=64, the model hasn't converged yet — not that
128 is wrong. Also consider a mixed strategy (H11) where slice_num is determined per sample
by mesh size.

---

## H11 — Separate Per-Channel Decoder Heads (Pressure-Specific Last Layer)

**One-line statement:** Replace the shared 3-output MLP head with separate output MLPs for
Ux, Uy, and p, giving the pressure head more capacity and allowing per-channel gradient
scaling without modifying the loss.

**Predicted delta on val_avg/mae_surf_p:** -2% to -5% (medium confidence, low cost)

**Implementation — `train.py` only:**
Modify `TransolverBlock` to have 3 separate output projection MLPs at the last layer:

```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Separate heads per output field
    self.head_Ux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                  nn.Linear(hidden_dim // 2, 1))
    self.head_Uy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                  nn.Linear(hidden_dim // 2, 1))
    self.head_p  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, 1))  # wider for pressure
```

In `forward`:

```python
if self.last_layer:
    h = self.ln_3(fx)
    return torch.cat([self.head_Ux(h), self.head_Uy(h), self.head_p(h)], dim=-1)
```

Remove `self.mlp2` (the current shared output head). Adjust `TransolverBlock.__init__`
to not use `out_dim` directly when `last_layer=True` with separate heads.

**Why it should work:**
The current last-layer head is a 2-layer MLP (hidden_dim → hidden_dim → 3). Sharing
representations across Ux, Uy, and p forces the final projection to encode all three fields
from the same intermediate representation. Pressure and velocity are related (Navier-Stokes)
but pressure depends on second derivatives of velocity — it is the harder prediction task.
Giving the pressure head an extra layer (or simply twice the hidden width) adds capacity
precisely where it matters for the ranking metric. This is a standard multi-task learning
trick (task-specific heads after shared backbone).

**Risk / failure mode:**
Small increase in parameter count (~128K extra params for 128-hidden model). The primary
risk is that shared representations are actually beneficial (shared decoding enforces
physical consistency via Bernoulli/pressure coupling). If the separate head makes mae_surf_Ux
and mae_surf_Uy worse while improving mae_surf_p, it may still be worth keeping if the
ranking metric improves. Check both directions.

---

## H12 — Cosine Learning Rate with Warmup Restarts (SGDR)

**One-line statement:** Replace single-cycle CosineAnnealingLR with 2 or 3 cosine restarts
(CosineAnnealingWarmRestarts with T_0=epochs//3), letting the optimizer escape sharp local
minima and explore a wider loss landscape.

**Predicted delta on val_avg/mae_surf_p:** -2% to -7% (medium confidence); complements H1.

**Implementation — `train.py` only:**
Replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```
With:
```python
T_0 = max(1, MAX_EPOCHS // 3)  # restart every 1/3 of total epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
)
```

Keep `scheduler.step()` at the end of each epoch (not per step).

**Why it should work:**
SGDR (Loshchilov & Hutter 2016) allows the model to explore multiple local minima, and
the checkpointing at the end of each cosine trough often finds flatter minima that
generalize better. For OOD splits (unseen camber, Re holdout), flatter minima are expected
to generalize better than sharp ones. With 3 restarts over a 50-epoch training run, the
model sees 3 full cosine anneals — each restart can settle into a different basin, and the
best validation checkpoint across all basins is selected. This is orthogonal to and
composable with H1 (warmup) and H5 (EMA).

**Risk / failure mode:**
With T_0=epochs//3, each restart cycle is short. If the model needs more than epochs//3
to converge from a warm start, the restarts just add noise. Safe threshold: if T_0 < 10,
don't use restarts and fall back to single cosine (H1 warmup variant). Also: SGDR + EMA
(H5) interact: the EMA should be reset or given a higher decay at each restart to track
the newly-warmed weights. For simplicity, do not combine with EMA in the same experiment
— test H12 standalone.

---

## H13 — AdamW Beta Tuning: (beta1=0.95, beta2=0.999)

**One-line statement:** Change AdamW betas from PyTorch defaults (0.9, 0.999) to (0.95,
0.999), reducing the momentum term so the optimizer reacts faster to batch-to-batch
gradient direction changes driven by high-Re outlier samples.

**Predicted delta on val_avg/mae_surf_p:** -1% to -4% (lower confidence, cheap to test)

**Implementation — `train.py` only:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    betas=(0.95, 0.999)
)
```

**Why it should work:**
The default beta1=0.9 gives a momentum half-life of ~10 gradient steps. With batch_size=4
and a WeightedRandomSampler, adjacent batches often come from different domains (raceCar vs.
cruise), making the gradient direction high-variance. A longer momentum half-life (0.95
gives ~14 steps) smooths across domains, which can help. However, literature on transformers
for scientific computing (specifically the Chinchilla work and subsequent LLM optimizer
studies) suggests that beta1=0.9 can under-smooth for tasks with high loss variance, while
0.95 is a common recommendation for more stable training. This is a 1-line change that can
distinguish between "optimizer momentum is limiting" and "everything else is limiting."

**Risk / failure mode:**
Higher beta1 can slow adaptation to new gradient directions and cause oversmoothing on
well-converged parameters. If the loss plateau appears earlier than baseline, reduce back
to 0.9. The signal here is weak but the experiment cost is essentially zero — pair with
any other hypothesis to get free signal about optimizer sensitivity.

---

## H14 — Mixed Precision Training (bfloat16) for Larger Effective Batch

**One-line statement:** Enable bfloat16 autocast with GradScaler to reduce per-sample
memory by ~50%, allowing batch_size=8 (doubling the effective batch) and faster per-step
compute.

**Predicted delta on val_avg/mae_surf_p:** -2% to -5% (via better gradient estimates from
larger batch); risk of float16 instability on extreme y values.

**Implementation — `train.py` only:**
Use `torch.amp.autocast` and `torch.amp.GradScaler`:

```python
# After optimizer instantiation
scaler = torch.amp.GradScaler("cuda")
```

In the training loop:

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (y - stats["y_mean"]) / stats["y_std"]
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred - y_norm) ** 2
    # ... loss computation unchanged ...

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # after unscale
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

Use `bfloat16` (not float16) because bfloat16 has the same exponent range as float32,
which is critical here given y values up to 29K. float16 would overflow the unnormalized
targets. Keep stats in float32 (already on device).

Also set `cfg.batch_size = 8` in the Config default (from 4), since memory savings from
bf16 justify this.

**Why it should work:**
At N_max=242K, the dominant VRAM cost is the node feature tensor [4, 242K, hidden_dim].
bfloat16 halves this. With 96GB VRAM and the baseline using roughly 20-30GB at batch=4,
bf16 at batch=8 should still fit comfortably. Larger batch size improves gradient SNR,
which is especially helpful when the WeightedRandomSampler produces high-variance batches
(3 domains × 4 batch = possible all-same-domain batches). Throughput increase from bf16
also allows more gradient steps within the timeout.

**Risk / failure mode:**
LayerNorm and softmax can accumulate errors in bf16. PyTorch's autocast handles these by
keeping accumulations in float32 automatically for most ops. Validate that val MAE at
epoch 5 is not degraded vs. full float32. The slice-token normalization (`slice_norm + 1e-5`)
is critical in PhysicsAttention — in bf16, this epsilon may be inadequate. Add a comment
to test slice_token numerical stability.

---

## H15 — Independent Per-Split Validation Weighting via Hard-Split Loss Focus

**One-line statement:** During training, compute an auxiliary loss specifically on high-error
node slices identified by the running val_re_rand error (hardest OOD split), using a thin
second forward pass on a held-aside fraction of the data.

**Predicted delta on val_avg/mae_surf_p:** Too speculative. Not recommended as standalone.
Defer to a later research round after simpler hypotheses are exhausted.

**Note:** This is included for completeness as a meta-learning/curriculum direction. The
implementation requires either an online hard example miner (expensive) or a static
"hard sample" buffer (simpler but stale). Given the 1499-sample training set size and
complex OOD structure, this is a second-order optimization after H1-H6 are validated.
Recommend H7 (global conditioning) as the architectural variant with stronger expected
impact on OOD splits.

---

## Priority Ranking

Ordered by (expected impact × tractability × independence from other hypotheses):

| Rank | Hypothesis | Predicted delta | Complexity | Risk |
|------|-----------|-----------------|------------|------|
| 1 | H2 — Huber surface-pressure loss | -5% to -12% | Very low | Low |
| 2 | H1 — Grad clip + LR warmup | -3% to -8% | Low | Low |
| 3 | H4 — Scale-up (256-hidden, 8-layer) | -8% to -20% | Low | Medium |
| 4 | H5 — EMA (decay=0.999) | -2% to -6% | Low | Very low |
| 5 | H3 — p-channel surf weight boost | -4% to -10% | Very low | Low |
| 6 | H6 — Flip augmentation | -3% to -10% | Low | Medium (dsdf) |
| 7 | H8 — DropPath (0.10) | -2% to -6% | Low | Low |
| 8 | H14 — bfloat16 + batch=8 | -2% to -5% | Low | Medium (numerics) |
| 9 | H10 — slice_num=128 | -3% to -8% (cruise) | Very low | Low |
| 10 | H7 — Global conditioning tokens | -5% to -15% (OOD) | High | Medium |
| 11 | H11 — Per-channel decoder heads | -2% to -5% | Medium | Low |
| 12 | H9 — sqrt(Re) per-sample scaling | -3% to -8% (Re) | Low | Medium |
| 13 | H12 — SGDR restarts | -2% to -7% | Very low | Low |
| 14 | H13 — AdamW beta1=0.95 | -1% to -4% | Very low | Low |
| 15 | H15 — Hard-split curriculum | Unclear | High | High |

## Recommended First Round (Parallel Experiments)

Run these simultaneously on separate students since they are mostly orthogonal:

1. **Student A**: H2 (Huber loss) + H1 (warmup + grad clip) — these compose cleanly
2. **Student B**: H4 (scale-up 256-8-8) + H1 (warmup, essential for deeper model)
3. **Student C**: H5 (EMA) + H3 (p_surf_weight=3.0) — both low-risk, stackable
4. **Student D**: H6 (flip augmentation, p_flip=0.5) — standalone to test augmentation

If all four fail to beat baseline: escalate to H7 (global conditioning / GeoTransolver
architecture variant) and H9 (Re-dependent loss scaling) as the next architectural tier.

## Composability Notes

- H1 (warmup) is universally composable: should be included in any experiment involving
  a larger or modified model (H4, H7, H11).
- H2 (Huber) + H3 (p_surf_weight) are complementary: Huber targets the loss function
  shape, H3 targets the weighting. Both can be active simultaneously.
- H5 (EMA) is composable with everything except H12 (SGDR restarts) — explain why in H12.
- H6 (augmentation) is composable with all others and should be added to any winning
  configuration in round 2.
- H4 (scale-up) + H14 (bf16) are a natural pair: larger model fits in memory via bf16.
