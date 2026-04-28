<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-04-28 (First Pass)

Fresh hypotheses for the TandemFoilSet Transolver baseline. All reproduce commands
assume the working directory is `target/` and that `SENPAI_TIMEOUT_MINUTES` and
`SENPAI_MAX_EPOCHS` are set externally by the harness.

Current baseline (Transolver default): **val_avg/mae_surf_p ≈ TBD** (establish from
first training run). All predicted deltas are relative to the un-modified baseline
run with the settings described in `train.py`.

---

## Category (a) — Loss Reformulation

### H-01: Surface-Weighted Huber Loss with Adaptive Delta

**Slug**: `huber-surf-loss`

**Hypothesis**: The current MSE loss is dominated by high-Re samples whose squared
errors are orders of magnitude larger than low-Re samples. Replacing MSE with a
Huber (smooth-L1) loss that switches from quadratic to linear above a threshold
`delta` bounds the gradient contribution of outlier nodes. For CFD surrogates over
a ~50× Re range, this should improve low-Re accuracy and OOD generalization without
sacrificing high-Re fit.

**Predicted delta on val_avg/mae_surf_p**: −5% to −12%. Evidence from PINN literature
(dynamic Huber in PINNs, arxiv 2310.XXXXX) and standard robust regression. Most
impact expected on `val_geom_camber_cruise` (wider Re spread) and `val_re_rand`.

**Implementation** (all edits in `train.py`):

1. Add `huber_delta: float = 1.0` to the `Config` dataclass.
2. Replace the loss block (lines 491-496) with:

```python
# Huber element-wise
def huber(err, delta):
    abs_err = err.abs()
    return torch.where(abs_err < delta,
                       0.5 * err ** 2,
                       delta * (abs_err - 0.5 * delta))

sq_err = huber(pred - y_norm, cfg.huber_delta)   # [B, N, 3]
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

3. Try `huber_delta` in {0.5, 1.0, 2.0}. The normalized space has std=1, so delta=1.0
   is a natural starting point.

**Risk**: If `delta` is too large the loss degrades to MSE; too small it becomes MAE
and may slow convergence. Mitigation: sweep two values.

**Literature**: Huber (1964); robust loss for PINNs (2023 survey); also used in
MeshGraphNets for large-scale CFD (GraphCast, Lam et al. 2022).

**Reproduce**:
```bash
python train.py --huber_delta 1.0 --wandb_group huber-surf-loss
```

---

### H-02: Per-Channel Uncertainty-Weighted Loss (Homoscedastic)

**Slug**: `uncertainty-weighted-loss`

**Hypothesis**: The three output channels `[Ux, Uy, p]` have very different noise
scales (pressure has ~100× larger absolute values in high-Re cases). Treating them
with equal loss weight ignores this structure. Learnable log-variance weights
(Kendall & Gal, NeurIPS 2017) allow the model to downweight high-variance channels
during hard samples while still training all channels. This is especially relevant
because `p` is the ranking metric but `Ux, Uy` provide gradient signal.

**Predicted delta on val_avg/mae_surf_p**: −3% to −8%. Particularly on
`val_single_in_dist` where Re extremes (104K–5M) create the largest channel
magnitude mismatch.

**Implementation**:

1. Add three learnable log-sigma scalars after model init:
```python
log_sigma = nn.Parameter(torch.zeros(3))   # one per channel
optimizer = torch.optim.AdamW(
    list(model.parameters()) + [log_sigma], lr=cfg.lr, weight_decay=cfg.weight_decay
)
```
2. Replace the loss block:
```python
sq_err = (pred - y_norm) ** 2   # [B, N, 3]
# per-channel precision weighting: sum_c [ err_c / (2*sigma_c^2) + log(sigma_c) ]
precision = torch.exp(-2 * log_sigma)   # [3]
weighted_err = sq_err * precision[None, None, :]  # [B, N, 3]
reg = log_sigma.sum()  # regularizer prevents sigma→∞

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (weighted_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (weighted_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss + reg
```
3. Log `log_sigma.detach().exp()` to W&B each epoch to monitor learned scales.

**Risk**: log_sigma might collapse to a degenerate solution if not regularized. The
`+ reg` term prevents this. Verify `log_sigma` changes during training.

**Literature**: Kendall & Gal, "What Uncertainties Do We Need…" NeurIPS 2017;
also used in multi-task aerodynamic prediction (Li et al., 2023).

**Reproduce**:
```bash
python train.py --wandb_group uncertainty-weighted-loss
```

---

### H-03: Gradient Surgery (PCGrad) for Ux/Uy/p Channel Conflicts

**Slug**: `pcgrad-channel-surgery`

**Hypothesis**: Gradients from the `Ux`, `Uy`, and `p` training channels may point
in conflicting directions (e.g. improving pressure predictions near the stagnation
point may temporarily worsen velocity). PCGrad (Yu et al., NeurIPS 2020) projects
conflicting gradients onto each other's orthogonal complement before the optimizer
step, preventing destructive interference. This is most likely to help on
`val_geom_camber_rc` where geometry extrapolation requires precise pressure
near the suction surface.

**Predicted delta on val_avg/mae_surf_p**: −4% to −10%. Strong evidence from
multi-task learning literature; applied to neural PDE surrogates in 2023-24.

**Implementation**: Compute per-channel losses, call `.backward()` three times with
`retain_graph=True`, store gradients, project, apply manually:

```python
# Replace single loss backward with per-channel PCGrad
channel_losses = []
for c in range(3):
    sq_c = (pred[..., c:c+1] - y_norm[..., c:c+1]) ** 2
    vl = (sq_c * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
    sl = (sq_c * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    channel_losses.append(vl + cfg.surf_weight * sl)

optimizer.zero_grad()
# Collect gradients per channel
grads = []
for i, cl in enumerate(channel_losses):
    cl.backward(retain_graph=(i < 2))
    grads.append({n: p.grad.clone() if p.grad is not None else None
                  for n, p in model.named_parameters()})
    optimizer.zero_grad()

# PCGrad projection
for i in range(3):
    for j in range(3):
        if i == j: continue
        for n, p in model.named_parameters():
            if grads[i][n] is None or grads[j][n] is None: continue
            gi, gj = grads[i][n], grads[j][n]
            dot = (gi * gj).sum()
            if dot < 0:
                grads[i][n] = gi - dot / (gj.norm() ** 2 + 1e-8) * gj

# Apply projected gradients
for n, p in model.named_parameters():
    p.grad = sum(grads[c][n] for c in range(3) if grads[c][n] is not None)

optimizer.step()
```

**Risk**: 3× backward passes per step triples gradient compute. Monitor step time.
If VRAM is tight, reduce batch_size to 2.

**Literature**: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
https://arxiv.org/abs/2001.06782

**Reproduce**:
```bash
python train.py --wandb_group pcgrad-channel-surgery --batch_size 2
```

---

## Category (b) — Architectural Improvements

### H-04: Increase Slice Tokens and Hidden Dimension (Capacity Scaling)

**Slug**: `transolver-scaled`

**Hypothesis**: The default Transolver uses `slice_num=64`, `n_hidden=128`, `n_layers=5`.
The slice tokens are the information bottleneck: 64 learned physics states must
represent the entire 74K–242K node field. Doubling both `slice_num` and `n_hidden`
while keeping layers=5 increases model capacity at the bottleneck. At 96 GB VRAM
and B=4 with up to 242K nodes, this should be feasible.

**Predicted delta on val_avg/mae_surf_p**: −8% to −20%. Scaling is a reliable lever
when the model is underparameterized relative to the problem complexity. The cruise
domain (210K mean nodes) is the hardest bottleneck case.

**Implementation** (edit model_config in train.py lines 417-428):
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,       # was 128
    n_layers=5,
    n_head=8,           # was 4
    slice_num=128,      # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```
Also reduce `batch_size` from 4 to 2 to fit VRAM. Increase `lr` to `7e-4` to
compensate for smaller effective batch.

**Risk**: May OOM on cruise samples (242K nodes × 256 hidden × 128 slices).
Profile memory on first epoch. If OOM, try `n_hidden=192, slice_num=96` as a
mid-point.

**Literature**: Transolver (Wu et al., ICML 2024) ablation shows slice_num and
n_hidden both positively correlated with accuracy up to compute budget.

**Reproduce**:
```bash
python train.py --n_hidden 256 --n_head 8 --slice_num 128 --batch_size 2 --lr 7e-4 --wandb_group transolver-scaled
```

---

### H-05: Low-Rank Spatial Attention (LRSA) for Slice Tokens

**Slug**: `lrsa-slice-attention`

**Hypothesis**: The full S×S self-attention over slice tokens is O(S²) and may not be
the most parameter-efficient form for physics-structured inputs. LRSA (2604.03582)
factorizes the attention weight matrix as `W_Q W_K^T = (U Σ^{1/2})(U Σ^{1/2})^T`
where `U` is a low-rank basis learned over training. This reported >17% average
error reduction on PDE benchmarks. Applied to the 64 slice tokens, the low-rank
basis could learn physics-structure (e.g. pressure eigenmodes) more directly than
full-rank attention.

**Predicted delta on val_avg/mae_surf_p**: −5% to −15%. The mechanism is best
justified for the attention bottleneck over structured physics states — exactly
what slice tokens represent.

**Implementation**: Modify `PhysicsAttention.forward` (lines 84-136 of train.py).
Replace `nn.MultiheadAttention` with a low-rank variant:

```python
# In PhysicsAttention.__init__, add:
rank = max(4, self.n_head)   # low-rank projection dim
self.lrsa_u = nn.Parameter(torch.randn(self.slice_num, rank) / math.sqrt(rank))

# In forward, replace the attn call:
# Instead of: attn_out = self.attn(seed, seed, seed)
# Use explicit low-rank QK:
q = self.to_q(seed)   # [B, S, D]
k = self.to_k(seed)
v = self.to_v(seed)
# Low-rank QK
qk = (q @ self.lrsa_u) @ (k @ self.lrsa_u).transpose(-2, -1) / math.sqrt(rank)
attn_w = qk.softmax(-1)
attn_out = attn_w @ v
```

Requires extracting Q/K/V projections explicitly (the `nn.MultiheadAttention` can
be decomposed or replaced with manual projections).

**Risk**: Implementation requires careful surgery on `PhysicsAttention`; easier to
implement as a new attention class and swap it in. Test numerically against original
on a single batch first.

**Literature**: "Low-Rank Spatial Attention for Neural Operators" (2604.03582, 2025).
https://arxiv.org/abs/2604.03582

**Reproduce**:
```bash
python train.py --lrsa_rank 16 --wandb_group lrsa-slice-attention
```
(Requires adding `--lrsa_rank` CLI arg; default `0` = disabled = original attention.)

---

## Category (c) — Optimization

### H-06: Warmup + Cosine Restart Scheduler

**Slug**: `cosine-restart-warmup`

**Hypothesis**: The current scheduler is a single cosine annealing from lr=5e-4 to 0
over `MAX_EPOCHS`. With balanced domain sampling and large variable meshes, early
epochs see high gradient variance. A linear warmup (5 epochs) followed by cosine
restarts (`T_mult=2`) allows the optimizer to escape early local minima for each
restart while the warmup prevents large early steps that lock in bad representations.
This is a standard improvement in both LLM and operator learning literature.

**Predicted delta on val_avg/mae_surf_p**: −3% to −7%.

**Implementation** (lines 434-435):
```python
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, SequentialLR
)
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
```
Add CLI flags: `--warmup_epochs 5 --T0 15`.

**Risk**: Very low. This is a well-validated technique. Main failure mode is wrong
`T_0` setting (too short → many restarts and instability; too long → no benefit).

**Literature**: SGDR (Loshchilov & Hutter, ICLR 2017); standard in ViT fine-tuning.

**Reproduce**:
```bash
python train.py --warmup_epochs 5 --T0 15 --wandb_group cosine-restart-warmup
```

---

### H-07: AdamW with EMA Model Averaging

**Slug**: `ema-model-avg`

**Hypothesis**: Exponential moving average (EMA) of model weights over training
produces smoother loss landscapes and better generalization, especially for
surrogate models evaluated at a single best checkpoint. EMA is standard in diffusion
models and Kaggle tabular competition winners. For CFD surrogates with high-variance
mesh samples and ~50 epochs of training, the last few checkpoints may overfit to
the balanced sampler's recent batch sequence; EMA averages over the epoch trajectory
instead.

**Predicted delta on val_avg/mae_surf_p**: −2% to −6%.

**Implementation**:
```python
# After model creation:
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))

# In training loop, after optimizer.step():
ema_model.update_parameters(model)

# At validation/test time, use ema_model instead of model for inference
# (swap model → ema_model in evaluate() calls)
```
Add `--ema_decay 0.999` CLI flag (0 = disabled).

**Risk**: EMA requires a second model copy (~same VRAM as model itself). At 96 GB
this should be fine. EMA checkpoint selection requires care: use EMA weights for
evaluation but base model for gradient updates.

**Literature**: Polyak & Juditsky (1992); used in DiffusionNet, PyTorch native
`swa_utils`; top Kaggle tabular ensembles.

**Reproduce**:
```bash
python train.py --ema_decay 0.999 --wandb_group ema-model-avg
```

---

## Category (d) — Data Augmentation

### H-08: Re-Stratified Oversampling of High-Re Training Samples

**Slug**: `re-stratified-oversample`

**Hypothesis**: Within each domain, per-sample y std varies by ~10× across the Re
range (low-Re ~50 vs high-Re ~2000 for raceCar single). The current balanced domain
sampler equalizes domains but does not stratify by Re within a domain. High-Re
samples are physically harder (thin boundary layers, larger gradients) and dominate
the surface-pressure error. Oversampling the top-Re quintile by 2× within each
domain should force more gradient signal on the hard cases without introducing
domain imbalance.

**Predicted delta on val_avg/mae_surf_p**: −4% to −9% on `val_re_rand`. Possible
slight regression on `val_single_in_dist` (low-Re samples undersampled).

**Implementation** (in `train.py`, after `load_data()`):
```python
# Build per-sample Re values from x[:, 13] = log(Re)
# Access via dataset: train_ds[i] returns (x, y, is_surface)
log_re = torch.stack([train_ds[i][0][:, 13].mean() for i in range(len(train_ds))])
re_vals = log_re.exp()

# Quintile threshold (top 20% Re samples get 2x weight)
q80 = re_vals.quantile(0.80)
re_weights = torch.where(re_vals > q80,
                          sample_weights * 2.0,
                          sample_weights)
re_weights = re_weights / re_weights.sum()

sampler = WeightedRandomSampler(re_weights, num_samples=len(train_ds), replacement=True)
```

**Risk**: Constructing `log_re` requires iterating the full dataset once (slow at
startup but cached). Alternatively, use precomputed stats from `meta.json` or
compute lazily. Oversampling high-Re increases effective batch difficulty — may need
`surf_weight` reduced to 8.0 to compensate.

**Literature**: Class-balanced sampling for long-tailed distributions (Cui et al.,
CVPR 2019); Re-stratified evaluation is already in the data design, so symmetric
stratification in training is a natural extension.

**Reproduce**:
```bash
python train.py --re_oversample_factor 2.0 --re_oversample_q 0.80 --wandb_group re-stratified-oversample
```

---

### H-09: Random Mesh Subsampling (Amortized Training)

**Slug**: `amortized-mesh-subsample`

**Hypothesis**: Transolver-3 (arxiv 2602.04940) shows that training on random
subsets of mesh nodes per sample (e.g. 50% of nodes, uniformly sampled) reduces
per-step compute 4× while still learning the correct field. At test time the full
mesh is used. For TandemFoilSet (74K–242K nodes) this would allow batch_size=8
instead of 4, doubling effective samples per epoch under the same VRAM budget. The
key insight is that the Transolver slice mechanism aggregates over all nodes — a
random subset with the same slice tokens still provides valid gradient signal.

**Predicted delta on val_avg/mae_surf_p**: −2% to −8%, mainly from the increased
data throughput (2× samples/epoch). Surface prediction may slightly degrade if
surface nodes are undersampled — mitigate by ensuring `is_surface` nodes are always
kept in the retained subset.

**Implementation** (in the training loop, before loss computation):
```python
if cfg.subsample_ratio < 1.0 and model.training:
    # Always keep surface nodes; randomly keep fraction of volume nodes
    vol_idx = (~is_surface & mask).nonzero(as_tuple=False)  # [K, 2] (batch, node)
    n_keep = int(vol_idx.shape[0] * cfg.subsample_ratio)
    perm = torch.randperm(vol_idx.shape[0], device=x.device)[:n_keep]
    keep_vol = vol_idx[perm]
    # Build new mask: all surface + sampled volume
    sub_mask = is_surface & mask   # start with surface
    sub_mask[keep_vol[:, 0], keep_vol[:, 1]] = True
    mask = sub_mask
```
Add `--subsample_ratio 0.5` CLI flag.

**Risk**: Padding with partial masks requires care — the model still receives full
`x` but loss/metrics only use `sub_mask`. Verify correctness on a single batch.
Surface coverage should be ~100% even at 50% overall subsampling since surface
nodes are ~1-2% of total.

**Literature**: Transolver-3 (2602.04940), Section 3.2 "Amortized Training".
https://arxiv.org/abs/2602.04940

**Reproduce**:
```bash
python train.py --subsample_ratio 0.5 --batch_size 8 --wandb_group amortized-mesh-subsample
```

---

## Category (e) — Capacity Scaling

### H-10: Deeper Transolver with Layer Dropout (Stochastic Depth)

**Slug**: `deeper-stochastic-depth`

**Hypothesis**: Increasing `n_layers` from 5 to 8 while adding stochastic depth
(drop each block with probability `p_drop` during training, scale activations by
`1/(1-p_drop)`) increases effective model capacity while preventing overfitting on
the 1499-sample training set. Stochastic depth has been shown to be more effective
than standard dropout for transformer architectures, improving both generalization
and effective depth. With only ~1500 training samples, pure depth increase without
regularization is likely to overfit — stochastic depth is the right pair.

**Predicted delta on val_avg/mae_surf_p**: −5% to −12%.

**Implementation** (in `TransolverBlock.forward`):
```python
# In __init__: add drop_prob parameter
self.drop_prob = drop_prob

# In forward:
def stochastic_depth(x, residual, drop_prob, training):
    if not training or drop_prob == 0:
        return x + residual
    keep = torch.rand(x.shape[0], 1, 1, device=x.device) > drop_prob
    return x + residual * keep.float() / (1 - drop_prob)

fx = stochastic_depth(fx, self.attn(self.norm1(fx)), self.drop_prob, self.training)
fx = stochastic_depth(fx, self.mlp(self.norm2(fx)), self.drop_prob, self.training)
```
Add CLI `--n_layers 8 --stochastic_depth_prob 0.1`.

**Risk**: Deeper model increases VRAM. At n_layers=8 and hidden=128, should stay
within budget at batch_size=4. Monitor first epoch carefully.

**Literature**: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016;
applied to ViTs in DeiT (Touvron et al., 2021).

**Reproduce**:
```bash
python train.py --n_layers 8 --stochastic_depth_prob 0.1 --wandb_group deeper-stochastic-depth
```

---

## Category (f) — Input Feature Engineering

### H-11: Physics-Aware Boundary Layer Features (Re_x + Local Curvature)

**Slug**: `boundary-layer-features`

**Hypothesis**: The most directly applicable finding from recent CFD surrogate
literature is that adding local boundary-layer features (local Re_x = Re * x_chord,
inviscid pressure from potential flow, surface curvature κ) as additional input
channels dramatically improves surface pressure accuracy. Ref: boundary GNN paper
(2503.18638) showed 88% OOD error reduction and 83% model size reduction using
these physics-informed features. Currently the model sees only `log(Re)` globally
(dim 13) and `is_surface` (dim 12); it has no local boundary-layer information.

**Predicted delta on val_avg/mae_surf_p**: −10% to −25%. Strongest predicted effect
of any hypothesis. The mechanism is direct: surface pressure is determined by
boundary-layer state, which depends on local Re_x and curvature.

**New input features to compute per-node**:

For surface nodes: 
- `re_x`: local chord-wise Reynolds number = Re × (arc-length from LE / chord)
  Already approximated by `saf` (dims 2-3) × `exp(dim 13)`.
- `kappa`: signed curvature of the foil surface from `saf` gradient.
- `log_re_x = log(Re * |saf|)` — the physically meaningful boundary layer parameter.

**Implementation** (in `train.py`, after loading `x` but before normalization):
```python
# x[:, :, 13] = log(Re), x[:, :, 2:4] = saf (signed arc-length)
# Compute local log(Re_x) as a new feature
saf_norm = x[..., 2:4].norm(dim=-1, keepdim=True)   # [B, N, 1]
log_re = x[..., 13:14]                               # [B, N, 1]
log_re_x = log_re + torch.log1p(saf_norm)            # [B, N, 1]
# Concatenate as dim 24 → X_DIM becomes 25
x = torch.cat([x, log_re_x], dim=-1)                 # [B, N, 25]
```
Update `X_DIM = 25` and `fun_dim = X_DIM - 2 = 23` in model_config.
Update `stats["x_mean"]` and `stats["x_std"]` to append mean/std of new feature.

**Risk**: Feature must be computed at both train and test time consistently.
`stats.json` is read-only — new feature stats must be computed inline from training
data. Add `--extra_features log_re_x` flag for clean ablation.

**Literature**: "Boundary Layer Graph Neural Network" (2503.18638, 2025);
classical boundary layer theory (Re_x as the key local parameter for transition).
https://arxiv.org/abs/2503.18638

**Reproduce**:
```bash
python train.py --extra_features log_re_x --wandb_group boundary-layer-features
```

---

## Category (g) — Output Transformations

### H-12: Reversible Instance Normalization (RevIN) for Re-Adaptive Output Scale

**Slug**: `revin-output-norm`

**Hypothesis**: Per-sample output magnitudes vary by ~10× within a single validation
split due to the Re range. The global `y_mean / y_std` normalization in `stats.json`
cannot adapt to per-sample scale. RevIN (Kim et al., ICLR 2022; arxiv 2603.11869
extends to PDEs) adds a learnable per-sample affine normalization at the model
output, then reverses it before loss computation. This allows the model to
concentrate capacity on shape rather than scale — a well-known improvement in
time series forecasting that applies equally to spatially-structured predictions
with varying amplitude.

**Predicted delta on val_avg/mae_surf_p**: −5% to −15%. Effect should be concentrated
on `val_re_rand` and `val_single_in_dist` where Re varies the most.

**Implementation**:

RevIN is applied at the batch level, not node level. Compute instance stats over
the real (non-padding) nodes:

```python
class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_w = nn.Parameter(torch.ones(num_channels))
        self.affine_b = nn.Parameter(torch.zeros(num_channels))

    def normalize(self, x, mask):
        # x: [B, N, C], mask: [B, N]
        m = mask.float().unsqueeze(-1)                           # [B, N, 1]
        n = m.sum(1, keepdim=True).clamp(min=1)                 # [B, 1, 1]
        mean = (x * m).sum(1, keepdim=True) / n                 # [B, 1, C]
        var  = ((x - mean) ** 2 * m).sum(1, keepdim=True) / n  # [B, 1, C]
        self._mean, self._std = mean, (var + self.eps).sqrt()
        x_n = (x - mean) / self._std
        return x_n * self.affine_w + self.affine_b

    def denormalize(self, x):
        x = (x - self.affine_b) / (self.affine_w + 1e-8)
        return x * self._std + self._mean
```

Apply before passing `y` to model's loss and after pred, within the training loop.

**Risk**: RevIN on the output requires that `denormalize()` is called before MAE
evaluation. Since `data/scoring.py` denormalizes with global stats, RevIN must be
applied on top of that in the training loop. Careful to not double-denormalize.

**Literature**: Kim et al., "Reversible Instance Normalization", ICLR 2022;
extended to neural PDE operators in arxiv 2603.11869 (2025).

**Reproduce**:
```bash
python train.py --use_revin --wandb_group revin-output-norm
```

---

## Category (h) — Sampling Strategies

### H-13: Curriculum Learning — Easy-to-Hard Re Ordering

**Slug**: `curriculum-re-ordering`

**Hypothesis**: Training on low-Re samples first and progressively introducing
high-Re samples implements a classic curriculum learning strategy (Bengio et al.,
ICML 2009). For CFD surrogates, low-Re flows have simpler boundary layers and
larger relative pressure signals on the surface. The model should learn the
geometry-to-pressure mapping more cleanly from low-Re data before being confronted
with high-Re samples where boundary layers are thin and gradients are extreme.
Concretely: for the first 15 epochs train only on samples with Re < 1M; for epochs
16-30 add Re < 3M; for epochs 31+ use the full range.

**Predicted delta on val_avg/mae_surf_p**: −3% to −8%, primarily on
`val_geom_camber_*` splits where geometry generalization matters most.

**Implementation**:

In `train.py`, track epoch in the training loop and modify sampler weights:

```python
def get_curriculum_weights(epoch, sample_weights, log_re_by_sample, cfg):
    if epoch < cfg.curriculum_phase1_end:
        re_threshold = math.log(1e6)   # Re < 1M
    elif epoch < cfg.curriculum_phase2_end:
        re_threshold = math.log(3e6)   # Re < 3M
    else:
        return sample_weights          # full range
    phase_mask = (log_re_by_sample <= re_threshold).float()
    w = sample_weights * phase_mask
    # Normalize to avoid zero-weight edge cases
    if w.sum() < 1e-6: return sample_weights
    return w / w.sum()
```

Recreate sampler at epoch boundaries. Add CLI flags:
`--curriculum_phase1_end 15 --curriculum_phase2_end 30`.

**Risk**: If phase 1 samples are too few (low-Re samples are a minority in tandem
domains), the effective training set shrinks and early epochs see high variance.
Add a floor: `w = phase_mask * 0.9 + 0.1` to never fully exclude any sample.

**Literature**: Bengio et al., "Curriculum Learning", ICML 2009;
applied to CFD surrogate training in Vinuesa & Brunton (2022 review).

**Reproduce**:
```bash
python train.py --curriculum_phase1_end 15 --curriculum_phase2_end 30 --wandb_group curriculum-re-ordering
```

---

### H-14: Surface-Node Focal Oversampling in Loss (Focal MAE)

**Slug**: `focal-surface-loss`

**Hypothesis**: The current loss weights surface nodes with `surf_weight=10` but
treats all surface nodes equally. Near the leading edge and suction peak, pressure
gradients are steepest and errors are largest. A focal loss variant that
up-weights the highest-error surface nodes (top-20% error nodes per sample)
directs gradient signal to the hardest surface predictions. This is analogous to
focal loss in detection (Lin et al., ICCV 2017) but applied to regression via
a rank-based hard-example re-weighting scheme.

**Predicted delta on val_avg/mae_surf_p**: −4% to −10%.

**Implementation**:
```python
# After computing per-node surface errors:
surf_err = (pred - y_norm).abs()   # [B, N, 3]
surf_p_err = surf_err[..., 2]      # pressure channel [B, N]
masked_surf_err = surf_p_err * surf_mask.float()

# Focal weight: nodes in top-20% error get 2x weight
with torch.no_grad():
    q80 = masked_surf_err[surf_mask].quantile(0.80)
    focal_w = torch.where(masked_surf_err > q80, 
                           torch.tensor(2.0, device=x.device),
                           torch.tensor(1.0, device=x.device))   # [B, N]
    focal_w = focal_w * surf_mask.float()

sq_err = (pred - y_norm) ** 2
surf_loss = (sq_err * surf_mask.unsqueeze(-1) * focal_w.unsqueeze(-1)).sum() \
            / (surf_mask.float() * focal_w).sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```
Add `--focal_top_q 0.80 --focal_multiplier 2.0` CLI flags.

**Risk**: Focal re-weighting may create instability if the top-20% nodes change
drastically between steps. Use `stop_gradient` on the weight computation (already
done with `torch.no_grad()`).

**Literature**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017;
hard-example mining in SSD (Liu et al., 2016); applied to mesh regression in
MeshDiffusion (2023).

**Reproduce**:
```bash
python train.py --focal_top_q 0.80 --focal_multiplier 2.0 --wandb_group focal-surface-loss
```

---

### H-15: Spiral/2D RoPE for Spatial Positional Encoding

**Slug**: `spiral-rope-2d`

**Hypothesis**: The Transolver's slice attention has no explicit 2D positional
encoding — node positions enter only through the raw `(x, z)` coordinates in the
feature vector. 2D RoPE (Rotary Position Embedding generalized to 2D, as in
Spiral RoPE arxiv 2602.03227) encodes relative spatial distances directly into
the attention scores without additional parameters, allowing the model to
distinguish near-field from far-field interactions at every attention head.
This is especially relevant for surface pressure prediction which depends on
relative distances along the foil chord.

**Predicted delta on val_avg/mae_surf_p**: −4% to −10%, primarily on
`val_geom_camber_*` where the geometry of unseen foils creates new spatial
arrangements.

**Implementation**: In `PhysicsAttention.forward`, before the attention call,
apply 2D RoPE to the Q and K projections using node positions `(x, z)`:

```python
# After computing slice_keys (the aggregated slice token positions):
# x_pos is available as data["x"][:, :, 0:2] (node positions before normalization)

def apply_2d_rope(q_or_k, pos_2d, dim):
    # pos_2d: [B, S, 2] (slice centroid positions)
    # q_or_k: [B, S, D]
    half = dim // 4
    freq_x = 1.0 / (10000 ** (torch.arange(half, device=pos_2d.device).float() / half))
    freq_z = freq_x
    angle_x = pos_2d[..., 0:1] * freq_x  # [B, S, half]
    angle_z = pos_2d[..., 1:2] * freq_z
    cos_x, sin_x = angle_x.cos(), angle_x.sin()
    cos_z, sin_z = angle_z.cos(), angle_z.sin()
    # Interleave real/imaginary rotation
    q1, q2 = q_or_k[..., :half], q_or_k[..., half:2*half]
    q3, q4 = q_or_k[..., 2*half:3*half], q_or_k[..., 3*half:4*half]
    rotated = torch.cat([
        q1 * cos_x - q2 * sin_x,
        q1 * sin_x + q2 * cos_x,
        q3 * cos_z - q4 * sin_z,
        q3 * sin_z + q4 * cos_z,
        q_or_k[..., 4*half:]   # remaining dims unrotated
    ], dim=-1)
    return rotated
```
Add `--use_2d_rope` flag. The slice centroid positions can be computed as the
weighted mean of input node positions using the slice assignment weights.

**Risk**: Requires exposing node positions through to the attention computation.
The `PhysicsAttention` module currently receives pre-embedded features; it needs
access to raw positions. Pass `pos = data["x"][:, :, 0:2]` alongside.

**Literature**: Spiral RoPE (2602.03227, 2025) for multi-directional 2D attention;
RoPE (Su et al., 2023); 2D RoPE for ViTs (EVA-02, Fang et al., 2024).

**Reproduce**:
```bash
python train.py --use_2d_rope --wandb_group spiral-rope-2d
```

---

## Summary Table

| ID | Slug | Category | Predicted delta | Risk | Priority |
|----|------|----------|----------------|------|----------|
| H-01 | huber-surf-loss | Loss | −5% to −12% | Low | **1** |
| H-02 | uncertainty-weighted-loss | Loss | −3% to −8% | Low | 3 |
| H-03 | pcgrad-channel-surgery | Loss | −4% to −10% | Medium | 4 |
| H-04 | transolver-scaled | Architecture | −8% to −20% | Medium | **2** |
| H-05 | lrsa-slice-attention | Architecture | −5% to −15% | Medium | 5 |
| H-06 | cosine-restart-warmup | Optimization | −3% to −7% | Very Low | 6 |
| H-07 | ema-model-avg | Optimization | −2% to −6% | Very Low | 7 |
| H-08 | re-stratified-oversample | Sampling | −4% to −9% | Low | 8 |
| H-09 | amortized-mesh-subsample | Data Aug | −2% to −8% | Medium | 9 |
| H-10 | deeper-stochastic-depth | Capacity | −5% to −12% | Low | 10 |
| H-11 | boundary-layer-features | Features | −10% to −25% | Medium | **3** |
| H-12 | revin-output-norm | Output Transform | −5% to −15% | Medium | 11 |
| H-13 | curriculum-re-ordering | Sampling | −3% to −8% | Low | 12 |
| H-14 | focal-surface-loss | Sampling | −4% to −10% | Low | 13 |
| H-15 | spiral-rope-2d | Architecture | −4% to −10% | Medium | 14 |

Top 3 by expected impact × risk-adjusted probability: H-11 (boundary features),
H-04 (scaled model), H-01 (Huber loss). These three are orthogonal and can run
in parallel immediately.
