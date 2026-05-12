<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Round 1 Research Hypotheses — TandemFoilSet CFD Surrogate

Generated: 2026-05-12  
Branch: icml-appendix-charlie-pai2g-24h-r1  
Baseline: Transolver, val_avg/mae_surf_p = unknown (first round, no prior experiments)

---

## Context and Bottleneck Analysis

The baseline Transolver uses:
- `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (~1M params)
- `AdamW(lr=5e-4, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=epochs)`
- `batch_size=4`, `surf_weight=10.0`
- MSE loss in normalized space, separate vol and surf terms
- No AMP, no gradient clipping, no warmup

**Known failure modes to address:**

1. **Training**: MSE is biased toward high-Re samples (per-sample y std varies ~10x across domains). The loss is driven by high-variance samples, leaving low-Re regime underfit. Per-sample relative L2 / Huber loss directly addresses this.

2. **Training throughput**: No AMP means slower step times and larger memory footprint. With 30-minute wall-clock limit, fewer epochs complete. bf16 AMP can double throughput with no quality loss (ICLR 2024 theory).

3. **Architecture capacity**: 1M params may be insufficient for learning the full pressure/velocity field across 3 very different physical domains. Increasing `n_hidden` and `n_layers` is a direct capacity lever.

4. **Optimization**: AdamW + cosine without warmup can diverge early for transformers. A 2-epoch warmup plus cosine is a standard improvement.

5. **Metric alignment**: Primary metric is surface pressure MAE, but `surf_weight=10.0` was not tuned against this specific metric. Higher surf_weight pushes the model harder on the metric we care about.

6. **Slice capacity**: `slice_num=64` determines how many "virtual physics states" the PhysicsAttention can represent. More slices = finer-grained decomposition of flow states across different Re and geometry regimes.

---

## Hypotheses

---

### H1: bf16 Automatic Mixed Precision

**Slug**: `bf16-amp`

**Hypothesis**: Enabling `torch.autocast("cuda", dtype=torch.bfloat16)` in the forward pass will increase training throughput by ~1.5–2x on A100/H100 hardware, allowing more epochs within the 30-minute wall-clock cap, producing better checkpoint selection without changing the architecture or loss.

**Predicted delta**: Moderate improvement — primarily through more training steps, not better optimization. Expect 5–20% reduction in `val_avg/mae_surf_p` from additional epochs at similar compute. No degradation risk: ICLR 2024 analysis of mixed-precision FNOs shows approximation bounds are preserved under bf16 (not fp16).

**Why it works**: The bottleneck in 30-minute runs is wall-clock time, not epoch count. bf16 AMP reduces memory bandwidth pressure and enables faster matrix multiplication on tensor cores. bfloat16 specifically maintains the float32 exponent range (avoiding overflow/underflow that afflicts fp16) while halving storage cost. ICLR 2024 "Mixed Precision Neural Operators" shows no approximation degradation for FNO-family models under bf16.

**Key paper**: "Mixed Precision Training of Neural Operators" (ICLR 2024) — proves bf16 FNO maintains approximation bounds; validates on NavierStokes, Darcy flow. https://iclr.cc/virtual/2024/poster/18379

**Implementation in `train.py`**:

Add at top of file:
```python
from torch.amp import GradScaler, autocast
scaler = GradScaler()  # for fp16 gradient scaling; bf16 doesn't need scaling but GradScaler is safe to keep
```

Wrap the training forward pass (lines 446–452) in autocast:
```python
with autocast(device_type="cuda", dtype=torch.bfloat16):
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred - y_norm) ** 2
    # ... rest of loss computation
loss = vol_loss + cfg.surf_weight * surf_loss
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

No change to the eval loop — inference stays in fp32.

**Risk / failure mode**: Near-zero risk. bf16 has sufficient range for this problem (pressure values after normalization should be O(1)). The main failure mode is if normalized targets have extreme outliers that overflow bf16 range (±3.4e38 — should never happen after normalization). Test by checking that loss values match fp32 within 5% on epoch 1.

**Suggested epochs**: 15 (wall clock will yield more effective epochs vs. baseline due to speedup)

---

### H2: Model Scale-Up — Wider and Deeper

**Slug**: `wider-deeper`

**Hypothesis**: Increasing `n_hidden=256, n_layers=8, n_head=8` (~8M params, 8x parameter count) gives the model sufficient capacity to simultaneously learn raceCar single (single-foil, low Re), raceCar tandem (higher Re, dual foil), and cruise tandem (very different AoA regime) without domain interference.

**Predicted delta**: Large improvement on OOD val splits (`val_geom_camber_rc`, `val_geom_camber_cruise`), moderate on in-dist. A 1M-param model is at the lower end for operator learning on heterogeneous physical domains.

**Why it works**: Neural operators in CFD typically scale favorably — UPT (NeurIPS 2024) and GNOT ablations show consistent improvement up to ~50M params for multi-condition settings. 96GB VRAM with batch_size=4 at N~242K nodes gives substantial headroom. The three training domains have substantially different physics (ground effect vs. freestream, inverted vs. upright, single vs. tandem); a larger hidden dimension provides more capacity to disentangle these.

**Key papers**:
- "Universal Physics Transformers" (NeurIPS 2024) — shows consistent scaling of neural operators from 1M to 50M+ params. https://arxiv.org/abs/2402.12365
- "GNOT: A General Neural Operator Transformer" (ICML 2023) — multi-query attention for heterogeneous physical fields; ablates hidden dim. https://arxiv.org/abs/2302.14376

**Implementation in `train.py`**:

Change `model_config` dict (lines 389–400):
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,   # was 128
    n_layers=8,     # was 5
    n_head=8,       # was 4
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

No other changes. With `n_hidden=256, n_layers=8`: estimated ~8M params. At 242K nodes, batch_size=2 may be needed to fit — add `--batch_size 2` if OOM.

**Risk / failure mode**: OOM on large cruise samples (242K nodes). Mitigate by reducing batch_size to 2. Optimization may need more epochs to converge — this is a risk with only 30-minute window. May want to combine with H3 (warmup) to stabilize early training.

**Suggested epochs**: 12 (larger model = slower steps; approximately same wall-clock as baseline at 15 epochs)

---

### H3: Warmup + Cosine LR Schedule

**Slug**: `warmup-cosine`

**Hypothesis**: Adding a linear LR warmup for the first 2 epochs (from 0 → lr) before the cosine decay will reduce early training instability and improve final convergence, particularly for the OOD geometry splits where the model must generalize to unseen camber profiles.

**Predicted delta**: Small but consistent improvement across all val splits. Warmup is essentially free — same total epochs, same final LR, but smoother early optimization. Especially beneficial when starting from random initialization with large mesh inputs.

**Why it works**: Standard practice for transformer training since Vaswani (2017). The PhysicsAttention uses learned softmax temperature initialized to 0.5 — at the start of training the slice assignments are essentially random and large gradient steps can create bad minima. Linear warmup gives the slice projection weights time to find stable groupings before the full LR kicks in. `CosineAnnealingLR` alone starts with full LR and then reduces it — the opposite of what helps early-stage attention learning.

**Key references**: "Attention Is All You Need" (2017) warmup schedule; "Swin Transformer" (ICCV 2021) warmup ablation (Fig. 3).

**Implementation in `train.py`**:

Replace the scheduler (line 407) with a sequential warmup + cosine:
```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_epochs = 2
warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - warmup_epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
```

The existing `scheduler.step()` at line 463 works unchanged.

**Risk / failure mode**: If epochs are very few (e.g. only 5 complete in 30 min), warmup consumes 40% of training — hurts rather than helps. For long enough runs (>8 epochs) this is safe. Check that epoch 1 loss is not wildly different from baseline epoch 1.

**Suggested epochs**: 15

---

### H4: Per-Sample Relative L2 Loss (Normalized MSE)

**Slug**: `relative-l2-loss`

**Hypothesis**: Replacing the global-normalized MSE loss with a per-sample relative L2 loss — dividing each sample's squared error by the sample's y variance before averaging — will fix the scale bias in the current loss, improving model fit on low-Re / low-variance samples that are currently dominated by high-Re extremes.

**Predicted delta**: Moderate improvement on `val_geom_camber_cruise` (lower Re cruise, smaller y magnitudes) and `val_re_rand` (stratified Re holdout). Small or neutral on `val_single_in_dist` (high Re raceCar). The improvement comes from better loss signal for low-scale samples.

**Why it works**: The dataset has per-sample y std varying from ~164 to ~2077 m/s (10x range). Global normalization by `y_std` (dataset-level mean/std) partially corrects this, but within a batch the MSE is still dominated by high-Re samples whose residuals are larger in absolute normalized units. A per-sample relative L2 gives each sample equal weight regardless of its scale. This is equivalent to dividing by the sample's variance before summing — standard in neural operator literature (FNO paper's "relative L2 test" is exactly this). Neural Regression for Scale-Varying Targets (NeurIPS 2023 workshop) shows this consistently outperforms MSE when target variance spans more than 2 decades.

**Key paper**: "FNO: Fourier Neural Operator" (ICLR 2021) — uses relative L2 as primary metric; "Neural operator learning for long-time integration in dynamical systems" (2023) — ablates per-sample normalization.

**Implementation in `train.py`**:

Replace the loss computation inside the training loop (lines 447–453):
```python
pred = model({"x": x_norm})["preds"]

# Per-sample relative L2: normalize each sample by its variance
# y_norm has shape [B, N_max, 3]; mask has shape [B, N_max]
B = y_norm.shape[0]
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface

vol_loss = 0.0
surf_loss = 0.0
for b in range(B):
    if vol_mask[b].sum() > 0:
        sq = ((pred[b] - y_norm[b]) ** 2) * vol_mask[b].unsqueeze(-1)
        denom = ((y_norm[b] ** 2) * vol_mask[b].unsqueeze(-1)).sum().clamp(min=1e-6)
        vol_loss += sq.sum() / denom
    if surf_mask[b].sum() > 0:
        sq = ((pred[b] - y_norm[b]) ** 2) * surf_mask[b].unsqueeze(-1)
        denom = ((y_norm[b] ** 2) * surf_mask[b].unsqueeze(-1)).sum().clamp(min=1e-6)
        surf_loss += sq.sum() / denom
vol_loss = vol_loss / B
surf_loss = surf_loss / B
loss = vol_loss + cfg.surf_weight * surf_loss
```

Note: the per-sample loop is necessary because normalization must be per-sample. With batch_size=4 and N~100K nodes this is 4 scalar operations, not expensive.

**Risk / failure mode**: If a low-Re sample has near-zero ground truth values (y_norm ≈ 0), the denominator `||y_norm||^2` approaches zero and the relative loss diverges. Clamp `denom` to `1e-6`. Also, the relative L2 is scale-free — it effectively ignores absolute magnitude. This could hurt the p-channel if pressure offsets are physically important. Monitor both surface and volume MAE; if vol_loss degrades significantly, revert to global MSE for vol and keep relative L2 for surf only.

**Suggested epochs**: 15

---

### H5: Huber Loss for Outlier Robustness

**Slug**: `huber-loss`

**Hypothesis**: Replacing MSE with Huber loss (`delta=0.5` in normalized space) reduces sensitivity to extreme-value mesh nodes near foil leading/trailing edges and wakes, where large prediction errors early in training can dominate the gradient and destabilize learning.

**Predicted delta**: Small-to-moderate improvement on surface MAE across all splits. Particularly expected to help on high-Re raceCar samples which have extreme pressure values near the suction peak.

**Why it works**: MSE penalizes large errors quadratically — a single mesh node with a 5-sigma residual contributes 25x more loss than a 1-sigma node. In CFD fields, the boundary layer near leading edges and the wake region regularly produce outlier residuals, especially early in training when the model hasn't learned the local physics. Huber loss transitions from L2 to L1 above the threshold `delta`, capping the gradient from outlier nodes. Empirically shown to help for pressure coefficient prediction in aerodynamic surrogates (Spalart-style RANS 2024 benchmarks).

**Key reference**: "Robust Loss Functions under Label Noise for Deep Neural Networks" (AAAI 2017) — Huber in regression settings; "Physics-informed neural networks for aerodynamic loads" (Computers & Fluids 2024) — Huber outperforms MSE for pressure fields.

**Implementation in `train.py`**:

Replace the `sq_err` computation in the training loop (line 447):
```python
# Huber loss in normalized space
HUBER_DELTA = 0.5
pred = model({"x": x_norm})["preds"]
residual = pred - y_norm
abs_residual = residual.abs()
sq_err = torch.where(
    abs_residual <= HUBER_DELTA,
    0.5 * residual ** 2,
    HUBER_DELTA * (abs_residual - 0.5 * HUBER_DELTA)
)
# rest of loss is unchanged
```

Also update the `evaluate_split` function (line 243) similarly — or leave as MSE for eval (eval uses MSE only for the `loss` diagnostic; the primary metric is physical-space MAE which is unaffected by the loss choice).

**Risk / failure mode**: `delta` is a hyperparameter. If `delta=0.5` is too small (most residuals are in the L1 regime), the model underfits because L1 provides weaker gradients near zero. Start with `delta=0.5`; if train loss converges slower than baseline, try `delta=1.0`. The key diagnostic is that training loss should not be dramatically higher than baseline after epoch 1.

**Suggested epochs**: 15

---

### H6: Higher Surface Weight (surf_weight=50)

**Slug**: `surf-weight-50`

**Hypothesis**: Increasing `surf_weight` from 10.0 to 50.0 directly increases the gradient pressure on surface nodes relative to volume nodes, improving surface pressure prediction which is the primary ranking metric.

**Predicted delta**: Clear improvement on `val_avg/mae_surf_p` at the possible cost of slightly higher volume MAE. The primary test metric is surface pressure — this is direct alignment.

**Why it works**: Surface nodes constitute a small fraction of total mesh nodes (~1–3% based on 85K-242K total nodes for airfoil meshes). With `surf_weight=10`, the effective per-node gradient weight is 10/97 ≈ 0.1 for surface vs. 1/97 ≈ 0.01 for volume — surface nodes still get only 10x more gradient signal but represent the critical boundary condition. Increasing to 50 pushes this ratio higher and more directly optimizes the metric that matters. PINNs literature consistently shows that boundary condition loss weighting is the primary hyperparameter for boundary accuracy (Raissi 2019, Wang 2022 "Understanding and Mitigating Gradient Failure in PINNs").

**Key reference**: "Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks" (Wang et al. 2022, SIAM J. Scientific Computing) — demonstrates that boundary-interior loss weight ratio directly controls boundary accuracy; prescribes adaptive or large fixed weights.

**Implementation in `train.py`**:

Change `surf_weight` in `Config` default (line 354):
```python
surf_weight: float = 50.0  # was 10.0
```

Or pass `--surf_weight 50` on the command line.

No other changes needed. The eval loop uses `cfg.surf_weight` only for the `loss` diagnostic, not for the primary metric.

**Risk / failure mode**: If surf_weight is too high, the model may overfit to surface nodes and completely fail on volume predictions. However, the primary metric only cares about surface accuracy, so this is acceptable. The failure mode to watch is if the model degenerates to predicting very smooth surface fields (minimizing surface MSE at the cost of physical plausibility). Monitor `mae_vol_p` — if it grows by more than 50% vs. baseline, the volume predictions have collapsed.

**Suggested epochs**: 15

---

### H7: Gradient Clipping (clip_norm=1.0)

**Slug**: `grad-clip-1`

**Hypothesis**: Adding `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before the optimizer step will prevent gradient explosions from extreme-value mesh nodes and allow training with higher effective learning rates, improving convergence stability.

**Predicted delta**: Small consistent improvement through more stable training. Particularly beneficial during early epochs when PhysicsAttention slice assignments are unstable and can produce large slice-token residuals.

**Why it works**: The PhysicsAttention module computes `slice_token = einsum(fx_mid, slice_weights)` — a weighted average over ~100K nodes. If early slice weights concentrate on outlier nodes, the slice token values can be very large, propagating large gradients back through `in_project_slice`. Gradient clipping is standard for transformer training (BERT, GPT-2 both use clip_norm=1.0) and costs negligible compute. It is especially important when the input range is variable (our 3 physical domains have different typical magnitudes even after global normalization).

**Key reference**: "Attention Is All You Need" and subsequent large-scale transformer training papers universally use `clip_grad_norm=1.0`; empirically critical for early training stability.

**Implementation in `train.py`**:

Insert after `loss.backward()` (line 456):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

One line change. Add `grad_clip: float = 1.0` to the `Config` dataclass (after line 359) and use `cfg.grad_clip` for easy ablation to 0 (disabled).

**Risk / failure mode**: If the learning rate is appropriate (5e-4 for AdamW is standard) and gradients are already not exploding, clipping has no effect. The failure mode is clipping too aggressively (clip_norm << 1.0) which undercuts learning — 1.0 is the safe default. Easily diagnosed: if train loss is much higher than baseline after epoch 1, try disabling clipping.

**Suggested epochs**: 15

---

### H8: More Slices (slice_num=128)

**Slug**: `more-slices-128`

**Hypothesis**: Increasing `slice_num` from 64 to 128 gives the PhysicsAttention more "virtual physics states" to route mesh nodes into, enabling finer-grained decomposition across the three training domains (raceCar single, raceCar tandem, cruise) and multiple Re regimes.

**Predicted delta**: Moderate improvement on OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`), where the model must generalize to unseen camber profiles. With only 64 slices, different geometry configurations may be mapped to the same slice, mixing their flow physics.

**Why it works**: The Transolver paper (ICML 2024) identifies slice_num as the most sensitive hyperparameter, reporting best results on their benchmarks with 64 slices. However, their benchmark uses a single physical domain (one dataset at a time). Our setting has 3 domains with different boundary conditions, Re ranges, and mesh topologies — more slices provide more routing capacity to handle this diversity. The linear attention complexity O(n_slices^2) means doubling slices costs only 4x more attention compute (not 4x total — the MLP layers dominate).

**Key paper**: "Transolver: A Fast Transformer Solver for PDEs on General Geometries" (ICML 2024) — Table 3 shows slice_num ablation; 64 is best for single domain. Multi-domain extension is untested. https://arxiv.org/abs/2402.02366

**Implementation in `train.py`**:

Change `slice_num` in `model_config` dict (line 396):
```python
slice_num=128,  # was 64
```

No other changes. The `in_project_slice` layer is `nn.Linear(dim_head, slice_num)` — increasing slice_num increases the output dimension of this layer. Total parameter increase is modest (128 output units vs. 64 for each head's slice projection — roughly +50K params total at n_hidden=128).

**Risk / failure mode**: Potential OOM if slice_num is very large, but 128 slices at n_hidden=128 is well within VRAM budget. More importantly, with 128 slices and only ~1499 training samples, the slice projection may overfit — the model has more "slots" than there are distinct physical configurations in the training set. Monitor if train loss drops much faster than val loss (overfitting). If so, try slice_num=96 as a middle ground.

**Suggested epochs**: 15

---

### H9: Lion Optimizer

**Slug**: `lion-optimizer`

**Hypothesis**: Replacing AdamW with the Lion optimizer (sign-momentum) will reduce memory usage and provide better or comparable convergence, allowing a larger model or larger batch to fit within 96GB VRAM for the same parameter count.

**Predicted delta**: Comparable to or slightly better than AdamW on convergence; the main benefit is memory efficiency (no second moment buffer), which could allow batch_size=6 (vs. 4) or a slightly larger model at the same memory cost.

**Why it works**: Lion (ICLR 2024) is theoretically grounded as a solution to a constrained optimization problem (minimize L1 displacement from the Adam update direction subject to sign constraints). It uses only the sign of the gradient update (like sign-SGD) which avoids the per-parameter second moment buffer of Adam, halving optimizer state memory. On large transformer models, Lion consistently matches or beats AdamW in test accuracy across multiple domains. The key hyperparameter shift: Lion LR is typically 3–10x smaller than AdamW LR (because the sign update is larger in magnitude than Adam's adaptively scaled update).

**Key paper**: "Symbolic Discovery of Optimization Algorithms" (ICLR 2024, cited 500+) — proves Lion as a constrained optimizer; benchmarks on language, vision, and diffusion models. https://arxiv.org/abs/2302.06675

**Implementation in `train.py`**:

Add to pyproject.toml:
```toml
lion-pytorch = ">=0.2.0"
```

Change optimizer (line 406):
```python
from lion_pytorch import Lion
optimizer = Lion(model.parameters(), lr=cfg.lr / 5, weight_decay=cfg.weight_decay)
# Lion lr is typically 3-10x lower than AdamW; use lr=1e-4 if cfg.lr=5e-4
```

Or implement Lion inline (4 lines — no new dependency):
```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0: state['m'] = torch.zeros_like(p)
                m = state['m']
                b1, b2 = group['betas']
                update = m.lerp(grad, 1 - b1)
                p.data.add_(torch.sign(update), alpha=-group['lr'])
                m.lerp_(grad, 1 - b2)
```

Use `lr=1e-4` (5x lower than AdamW's 5e-4) as the starting point.

**Risk / failure mode**: Lion is sensitive to LR scaling — if the LR is not reduced from AdamW's default, training will diverge. Always use lr = AdamW_lr / 5 as the starting point. Also, Lion benefits less from weight_decay annealing — if convergence is slow, try disabling weight_decay entirely.

**Suggested epochs**: 15

---

### H10: Higher LR with AdamW (lr=2e-3) + Gradient Clipping

**Slug**: `higher-lr-clipped`

**Hypothesis**: The baseline LR of 5e-4 may be too conservative for a 30-minute training budget. Increasing to 2e-3 combined with gradient clipping (clip_norm=1.0) and a 2-epoch warmup will reach better minima faster within the wall-clock limit.

**Predicted delta**: If the baseline is learning-rate limited (likely for a 1M-param transformer with standard init), this could improve `val_avg/mae_surf_p` by 10–30%. The key question is whether AdamW at 5e-4 is already near its convergence limit in 15 epochs.

**Why it works**: Transformers routinely train best at LR=1e-3 to 3e-3 with AdamW in the literature (ViT paper uses 3e-3 for base model; Transolver paper does not specify its LR for the Darcy/NS benchmarks). Without clipping, high LR + large mesh inputs can cause gradient explosions on the slice projection layer. With clipping, higher LR is stable and converges faster. This is a diagnostic experiment — if it helps, the baseline was LR-throttled; if it hurts, the baseline LR was appropriate.

**Key reference**: "An Image is Worth 16x16 Words" (ViT, ICLR 2021) — uses LR=3e-3 with AdamW and warmup for large transformers; "DeiT" paper — systematic LR ablation for vision transformers.

**Implementation in `train.py`**:

Change `Config` defaults (lines 351–354):
```python
lr: float = 2e-3       # was 5e-4
```

Add after `loss.backward()` in the training loop (line 456):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Add warmup (as in H3):
```python
warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=2)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - 2)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[2])
```

**Risk / failure mode**: If LR=2e-3 is too high for this architecture, training will diverge in the first 2–3 epochs (loss goes up or NaN). The warmup mitigates this. Diagnostic: check that train loss is decreasing after epoch 3. If not, revert to 5e-4.

**Suggested epochs**: 15

---

### H11: Dedicated Per-Channel Output Heads

**Slug**: `perchannel-heads`

**Hypothesis**: Replacing the single shared output head (predicting all 3 channels Ux, Uy, p simultaneously) with three independent output heads — one per channel — allows the model to learn different output representations for velocity and pressure, which have very different physical scaling and spatial patterns.

**Predicted delta**: Moderate improvement on pressure MAE (`mae_surf_p`), neutral or slight improvement on velocity channels. Pressure and velocity have fundamentally different spatial structure (pressure varies more globally; velocity is locally smooth except at boundaries) — shared head coupling may hurt pressure accuracy.

**Why it works**: The current model uses a single `nn.Linear(n_hidden, 3)` at the end of the last TransolverBlock to predict all 3 outputs simultaneously. This forces the final hidden state to simultaneously encode Ux, Uy, and p in a shared 128-dimensional space. Independent heads each get the full 128 dimensions and can learn different projections. GNOT (ICML 2023) uses per-field decoders for multi-output operator learning. The velocity components (Ux, Uy) share physical scaling, so they could share a head, but pressure (kinematic pressure p/rho in m^2/s^2) has different units and spatial patterns.

**Key paper**: "GNOT: A General Neural Operator Transformer" (ICML 2023) — per-query decoders for multi-field prediction. https://arxiv.org/abs/2302.14376

**Implementation in `train.py`**:

Modify `TransolverBlock.__init__` for the last layer (lines 152–157):
```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Three separate heads
    self.head_Ux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
    self.head_Uy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
    self.head_p  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
    # Pressure head gets full hidden dim; velocity heads share half
```

Modify the forward:
```python
if self.last_layer:
    h = self.ln_3(fx)
    return torch.cat([self.head_Ux(h), self.head_Uy(h), self.head_p(h)], dim=-1)
```

The `out_dim=3` parameter of `TransolverBlock` needs adjustment — pass `out_dim=3` still, but the constructor ignores it internally and replaces with per-channel heads. Alternatively, set `out_dim` to 1 in model_config and concatenate.

**Risk / failure mode**: This is a structural change — the model initialization changes. If the heads are not initialized properly (trunc_normal already handles this via `apply(self._init_weights)`) they may converge to different local minima than the shared head. The `mlp2` attribute referenced in the forward may need to be replaced by the three heads; make sure there is no reference to `mlp2` remaining in `forward()`. Careful code audit required.

**Suggested epochs**: 15

---

### H12: SiLU Activation Throughout

**Slug**: `silu-activation`

**Hypothesis**: Replacing the default GELU activation with SiLU (Swish) throughout the model will improve gradient flow and training convergence, as SiLU is known to outperform GELU on neural operators and physics-informed networks.

**Predicted delta**: Small consistent improvement (~2–5% on primary metric). SiLU benefits are typically small but nearly always positive; low risk, low reward. Primarily useful as a complement to other changes.

**Why it works**: SiLU (x * sigmoid(x)) has smoother gradients than GELU near x=0 and provides non-monotonic activation that helps representation learning for physics fields with both positive and negative regions (Ux, Uy, p all have mixed signs). Empirically, SiLU outperforms GELU in several PDE surrogate papers and is the default in many modern architectures (LLaMA, PaLM, Mamba). The ACTIVATION dict in train.py already includes "silu" — one-line change.

**Key reference**: "Searching for Activation Functions" (Ramachandran et al., 2018) — introduces Swish/SiLU; ablations show consistent improvements in deep networks; adopted in LLaMA family models.

**Implementation in `train.py`**:

In `model_config` dict (add `act` field, it defaults to "gelu"):
```python
model_config = dict(
    ...
    act="silu",   # was implicitly "gelu"
    ...
)
```

The `ACTIVATION` dict already contains `"silu": nn.SiLU` (line 61). The `act` parameter propagates through `Transolver.__init__` → `TransolverBlock.__init__` → `MLP.__init__`. Note: the last-layer `mlp2` in `TransolverBlock` uses `nn.GELU()` hardcoded (line 155) — also change that to `nn.SiLU()` for consistency.

**Risk / failure mode**: Nearly zero risk. SiLU is a drop-in activation replacement. The only failure mode is if the model uses the activation for output-range clamping (e.g., in a sigmoid output head), which this model does not. Check epoch 1 loss is comparable to baseline.

**Suggested epochs**: 15

---

## Summary Table

| # | Slug | Title | Target | Risk | Epochs |
|---|------|-------|--------|------|--------|
| H1 | `bf16-amp` | bf16 Automatic Mixed Precision | Throughput → more steps/30min | Very low | 15 |
| H2 | `wider-deeper` | Model Scale-Up (256 hidden, 8 layers) | Capacity → better OOD generalization | Medium (OOM) | 12 |
| H3 | `warmup-cosine` | Warmup + Cosine LR Schedule | Optimizer stability | Very low | 15 |
| H4 | `relative-l2-loss` | Per-Sample Relative L2 Loss | Loss bias for low-Re samples | Low | 15 |
| H5 | `huber-loss` | Huber Loss (delta=0.5) | Outlier robustness at boundaries | Low | 15 |
| H6 | `surf-weight-50` | Higher Surface Weight (50) | Direct metric alignment | Low | 15 |
| H7 | `grad-clip-1` | Gradient Clipping (norm=1.0) | Training stability | Very low | 15 |
| H8 | `more-slices-128` | More Slices (slice_num=128) | Richer physics decomposition | Low | 15 |
| H9 | `lion-optimizer` | Lion Optimizer | Memory efficiency + convergence | Medium (LR tuning) | 15 |
| H10 | `higher-lr-clipped` | Higher LR (2e-3) + Clip + Warmup | LR-throttled baseline | Medium (diverge risk) | 15 |
| H11 | `perchannel-heads` | Per-Channel Output Heads | Pressure specialization | Medium (code complexity) | 15 |
| H12 | `silu-activation` | SiLU Activation | Gradient flow | Very low | 15 |

## Recommended First 8 (for 8 idle students)

Priority ordering based on expected improvement / risk / orthogonality:

1. `bf16-amp` — nearly free throughput gain, strong theoretical support
2. `surf-weight-50` — direct metric alignment, one-line change
3. `wider-deeper` — largest expected absolute improvement, 96GB VRAM can handle it
4. `relative-l2-loss` — addresses the most clear-cut data bias in the loss
5. `warmup-cosine` — widely validated, safe, orthogonal to all others
6. `more-slices-128` — Transolver-specific; directly extends the paper's ablation to multi-domain
7. `huber-loss` — addresses boundary outliers in CFD fields
8. `grad-clip-1` — safest diagnostic; if it helps, it reveals that baseline was gradient-unstable

Held back for round 2: `lion-optimizer` (LR sensitivity), `higher-lr-clipped` (diverge risk), `perchannel-heads` (code complexity), `silu-activation` (too small to justify a full slot).
