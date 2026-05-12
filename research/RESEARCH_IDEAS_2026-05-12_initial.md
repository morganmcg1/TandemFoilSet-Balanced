<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TandemFoilSet Research Hypotheses — 2026-05-12

Launch: `willow-pai2g-24h-r3` (isolated appendix experiment)
Target metric: `val_avg/mae_surf_p` (minimize)
Baseline: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, lr=5e-4, surf_weight=10.0, MSE loss, no gradient clipping, no AMP, no EMA

---

## H1: `grad-clip-norm1` — Gradient clipping to stabilize high-Re outlier training

### Hypothesis
Adding gradient norm clipping (max_norm=1.0) before optimizer.step() will reduce loss spikes from high-Re samples (y std up to 2077 in val_single_in_dist), leading to a smoother loss trajectory and lower final val_avg/mae_surf_p.

### Concrete code change
In `train.py`, in the training loop just before `optimizer.step()`, add:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Add `--grad_clip` float arg to Config (default=0.0, meaning disabled). When `cfg.grad_clip > 0`, call `clip_grad_norm_` with that value. Suggested run: `--grad_clip 1.0`.

### Literature / empirical reasoning
Gradient clipping is standard in transformer training (Vaswani et al. 2017, GPT-2, PDE-Bench baselines). In regression tasks with heavy-tailed target distributions (which this dataset has — Re spans 100K to 5M), extreme MSE gradients from outlier samples can dominate parameter updates and drive other parameters away from good solutions. The no-clipping baseline is unusually permissive for a transformer on variable-magnitude targets. Clipping at norm=1.0 is well-established as a conservative, widely-applicable default.

### Estimated training cost
Negligible overhead — one extra norm computation per step. No VRAM impact. Fits comfortably in a 30-min run.

### Risk
Low. This is a well-understood regularizer with consistent empirical benefits. The only failure mode is if gradients are rarely large and clipping impedes useful large-step updates, but with MSE loss on high-Re samples this is unlikely. Could try max_norm in {0.5, 1.0, 2.0} if initial run is inconclusive.

---

## H2: `amp-bf16` — Mixed precision training with bfloat16

### Hypothesis
Enabling `torch.autocast("cuda", dtype=torch.bfloat16)` will reduce forward/backward pass time by ~40-60%, allowing more gradient steps (or larger batch size) within the 30-minute wall clock budget, which should translate to lower `val_avg/mae_surf_p` at the same epoch count or equivalent quality at lower cost.

### Concrete code change
Wrap the forward pass and loss computation in `torch.autocast`:

```python
from torch.cuda.amp import GradScaler
scaler = GradScaler()

# in training loop:
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model({"x": x_norm})
    pred = out["preds"]
    # ... loss computation ...

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

Add `--amp` boolean flag to Config (default=False). Note: BF16 is preferred over FP16 for transformers because it has the same exponent range as FP32 (no overflow risk) and does not require loss scaling on A100/H100 hardware. With BF16, GradScaler is technically not needed but does no harm.

### Literature / empirical reasoning
AMP is standard practice for all modern transformer training (Megatron-LM, FasterTransformer, etc.). On A100/H100 GPUs (96GB VRAM matches H100 profile), BF16 tensor cores give ~2x throughput for matmul-heavy workloads. Transolver's PhysicsAttention uses attention and MLP blocks — both matmul-dominated. The 30-min wall clock cap means raw throughput directly converts to more optimizer steps. Several CFD surrogate papers (FNO, GNO, GNOT) report matching or improving baseline metrics after AMP migration.

### Estimated training cost
No extra cost; this reduces cost. Approximately 40-60% wall-clock speedup. Allows doubling epochs within the same timeout, or reducing VRAM pressure to accommodate future larger models.

### Risk
Medium. BF16 reduces precision — rare but possible for accumulated statistics (batch norm running stats, etc.). Transolver uses LayerNorm which is generally AMP-safe. The slice-based soft-assignment in PhysicsAttention involves softmax, which can be numerically sensitive; monitor loss NaN rate. If NaNs appear, try keeping attention logits in FP32 with `autocast` limited to the MLP blocks.

---

## H3: `surf-weight-50` — Increase surface loss weight to 50.0

### Hypothesis
Increasing `surf_weight` from 10.0 to 50.0 will more strongly align the training gradient with surface nodes, which are the exclusive source of the ranking metric (`mae_surf_p`), leading to lower surface pressure error at the cost of marginally higher volume error.

### Concrete code change
In `train.py`, change the default value in Config:

```python
surf_weight: float = 50.0  # was 10.0
```

Or pass `--surf_weight 50.0` as a CLI argument to avoid changing the default. No other changes needed.

### Literature / empirical reasoning
The baseline uses surf_weight=10.0, which was likely set as a reasonable default without direct optimization against the evaluation metric. The paper-facing metric is purely surface-pressure MAE — volume errors do not contribute to ranking at all. In task-aligned loss weighting (a common empirical practice in multi-objective regression), you raise the weight on the components that matter. A 5x increase (10→50) is aggressive but justified because surface nodes are a small fraction of total mesh nodes (airfoil boundary layers), so without strong weighting the surface loss term is overwhelmed by volume loss in absolute gradient magnitude. GNOT and FactFormer papers both demonstrate that domain-specific loss weighting on boundary/surface terms improves boundary metrics substantially.

### Estimated training cost
Zero overhead. This is a single constant change.

### Risk
Low to medium. Overshooting (surf_weight too high) can destabilize training if surface gradients dominate and volume prediction degrades catastrophically. The volume loss still contributes pressure indirectly through the flow field, so complete neglect would be harmful. Monitor `vol_loss` to detect degradation. A safer intermediate would be surf_weight=25.0. If 50.0 diverges, try 25.0.

---

## H4: `smooth-l1-beta01` — Replace MSE with SmoothL1 (Huber) loss, beta=0.1

### Hypothesis
Replacing the MSE loss with SmoothL1 (Huber) loss with beta=0.1 in normalized space will reduce the outsized gradient contribution from high-Re samples, whose normalized residuals can still be large, resulting in more uniform gradient updates across the Re range and lower mean MAE across splits.

### Concrete code change
In `train.py`, replace the squared error computation:

```python
# Current:
sq_err = (pred - y_norm) ** 2  # [B, N, 3]

# New:
import torch.nn.functional as F
# SmoothL1 with beta=0.1: quadratic for |e| < 0.1, linear for |e| >= 0.1
smooth_l1_err = F.smooth_l1_loss(pred, y_norm, reduction="none", beta=0.1)  # [B, N, 3]
```

Then replace all occurrences of `sq_err` with `smooth_l1_err` in the loss computation. Add `--loss_fn {mse,smooth_l1}` and `--smooth_l1_beta 0.1` to Config.

### Literature / empirical reasoning
MSE loss is quadratic in residuals, which means high-Re samples (where normalized errors can reach 5-10x the mean) contribute 25-100x the gradient of typical samples. This is the main mechanism by which extreme outliers dominate training. Huber loss (SmoothL1) transitions to linear behavior above the beta threshold, bounding gradient magnitude per sample. In weather/climate surrogate tasks (GraphCast, Pangu-Weather), Huber loss is standard. For PDE surrogates on variable-Re flows, L1 variants consistently outperform MSE when the target range spans orders of magnitude. Beta=0.1 is chosen to be small relative to the normalized target std (which is approximately 1 by construction), meaning the loss is quadratic for typical residuals and linear only for large outliers.

### Estimated training cost
Negligible. SmoothL1 is essentially the same compute as MSE.

### Risk
Medium. SmoothL1 changes the optimal loss surface. If the dataset is actually well-behaved in normalized space (despite large physical magnitudes), the normalization may already correct for scale differences and MSE may be preferable. The key uncertainty is whether normalized residuals are heavy-tailed. A diagnostic would be to plot the histogram of normalized residuals from a baseline run before committing. If heavy tails are confirmed, this is a strong hypothesis.

---

## H5: `wider-n192` — Increase model width to n_hidden=192, n_head=6

### Hypothesis
Increasing n_hidden from 128 to 192 and n_head from 4 to 6 (maintaining n_hidden/n_head = 32, the standard head dimension) will give the model more representational capacity to simultaneously capture low-Re laminar and high-Re turbulent regimes, reducing underfitting and lowering `val_avg/mae_surf_p`.

### Concrete code change
In `train.py`, modify model_config:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,   # = 22
    out_dim=3,
    n_hidden=192,        # was 128
    n_layers=5,
    n_head=6,            # was 4
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Or expose via CLI args `--n_hidden 192 --n_head 6`.

### Literature / empirical reasoning
Width scaling is the most VRAM-efficient axis for transformers (VRAM scales as O(n_hidden * seq_len) vs O(seq_len²) for depth). The baseline n_hidden=128 is quite small for a problem with 74K-242K nodes and 3 complex output fields across an order-of-magnitude Re range. In the FNO and GNO lineage, increasing channel width reliably improves approximation on flows with multiple distinct regimes. Head dimension 32 (192/6=32) is consistent with standard transformer practice and the original Transolver paper. The n_layers=5 depth stays fixed, keeping per-sample compute moderate. VRAM impact: ~2.25x parameters in attention projections (~(192/128)^2 = 2.25x), which is still within 96GB budget at batch_size=4.

### Estimated training cost
~40-60% slower forward/backward pass (quadratic in hidden dim for attention components). May require reducing batch_size from 4 to 3 if OOM at 242K nodes. Still fits in 30-min run at reduced batch size.

### Risk
Medium. Larger models are slower and may not overcome the 30-min training budget limitation. If the model is not capacity-limited but optimization-limited, width scaling won't help. Consider this after gradient clipping and loss fixes, which address optimization.

---

## H6: `warmup-5ep` — Add 5-epoch linear LR warmup before cosine annealing

### Hypothesis
Adding a 5-epoch linear warmup (lr: 5e-5 → 5e-4) before the cosine decay schedule will prevent large-gradient parameter updates in the first few epochs when the model is far from a good initialization, reducing early overfitting or divergence and resulting in a better final optimum.

### Concrete code change
In `train.py`, replace the single CosineAnnealingLR with a SequentialLR:

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

WARMUP_EPOCHS = 5
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,   # starts at lr * 0.1 = 5e-5
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS,
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS - WARMUP_EPOCHS,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS],
)
```

Add `--warmup_epochs 5` to Config.

### Literature / empirical reasoning
LR warmup is near-universal in modern transformer training (BERT, ViT, Swin, all variants). The mechanism is well-understood: at initialization, gradients are large and noisy because the model has no useful representations; starting at peak LR causes large early parameter jumps that are hard to recover from. The Transolver baseline omits warmup, which is a known oversimplification. In the GNO and FNO training recipes, warmup epochs of 5-10% of total training are standard. At 50 epochs, 5-epoch warmup (10%) is the standard recipe ratio.

### Estimated training cost
Zero overhead. No extra computation; just a different LR trajectory.

### Risk
Low. Warmup is low-risk and widely validated. The only failure mode is if the model is already in a stable initialization region (e.g., from PyTorch defaults) where warmup adds no benefit — in that case the result is neutral. With 5-epoch warmup out of 50 epochs, the cosine decay phase is only slightly shortened.

---

## H7: `ema-decay999` — Exponential Moving Average of model weights (decay=0.999)

### Hypothesis
Maintaining an EMA of model weights with decay=0.999, updated every training step, and using the EMA model for validation/checkpointing will reduce checkpoint noise (high-variance late-training updates) and yield a smoother, better-generalizing parameter estimate, lowering `val_avg/mae_surf_p` without any training compute overhead.

### Concrete code change
In `train.py`, add EMA tracking:

```python
from copy import deepcopy

# After model initialization:
ema_model = deepcopy(model)
ema_model.eval()
EMA_DECAY = 0.999

# In training loop, after optimizer.step():
with torch.no_grad():
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1.0 - EMA_DECAY)

# Use ema_model (not model) for validation and checkpointing
```

Add `--ema_decay 0.999` (default=0.0 meaning disabled) to Config.

### Literature / empirical reasoning
EMA of weights is used in essentially every state-of-the-art vision/ML model (DeiT, EVA, DINO, Bootstrap-Your-Own-Latent, WeatherBench2, Pangu-Weather, GraphCast). The mechanism is that SGD/Adam updates are noisy — each batch pushes parameters in a direction that is optimal only for that mini-batch. EMA averages out this noise across thousands of steps, producing a parameter estimate with lower variance. For CFD surrogates evaluated on surface pressure, where the metric is sensitive to local field accuracy, this smoothing effect is particularly beneficial. Typical improvement is 0.5-3% on validation MAE with zero compute cost during training (only the averaging operation, which is O(params) per step).

### Estimated training cost
Negligible: one O(params) vector operation per step. Doubles memory footprint of model parameters (not activations), which for n_hidden=128 is small relative to 96GB VRAM.

### Risk
Low. EMA is purely additive — it can be enabled/disabled without other changes, and the worst case is no improvement. The main tuning knob is decay: 0.999 is a common default, but higher decay (0.9999) can be better for long training runs; lower decay (0.99) can be better for very short runs. At 50 epochs with ~375 steps/epoch (~18750 total steps), decay=0.999 gives a ~1000-step effective window, which is appropriate.

---

## H8: `p-channel-weight3x` — Per-channel loss weight 3x on pressure channel

### Hypothesis
Multiplying the loss contribution of the pressure channel (index 2) by 3.0 relative to Ux and Uy will increase gradient signal for pressure prediction specifically, directly aligning the training objective with the ranking metric (`mae_surf_p` uses only the p channel), and will lower `val_avg/mae_surf_p`.

### Concrete code change
In `train.py`, after computing squared error, apply per-channel weighting:

```python
# After:
sq_err = (pred - y_norm) ** 2  # [B, N, 3]

# Add channel weights [Ux, Uy, p]:
channel_weights = torch.tensor([1.0, 1.0, 3.0], device=sq_err.device, dtype=sq_err.dtype)
sq_err = sq_err * channel_weights.unsqueeze(0).unsqueeze(0)  # broadcast to [B, N, 3]

# Then continue with existing masking / vol_loss / surf_loss computation
```

Add `--p_weight 3.0` to Config to parameterize the pressure channel multiplier.

### Literature / empirical reasoning
Task-aligned per-output weighting is a basic technique in multi-output regression. The ranking metric uses only the p channel, yet the loss treats Ux, Uy, p equally. This is a misalignment between training and evaluation. Setting p_weight=3.0 triples the gradient signal for pressure relative to velocity. The exact multiplier should be tuned but 3x is a reasonable start — enough to create meaningful bias without completely ignoring velocity (which provides physical context for pressure through Bernoulli). In weather forecasting (GraphCast, Pangu), variable-level weighting is used to emphasize geopotential heights (the most predictable and physically meaningful field) while still training on wind components.

### Estimated training cost
Zero overhead. One element-wise multiply added to the loss path.

### Risk
Low to medium. If the model cannot improve pressure without degrading velocity prediction (e.g., due to Bernoulli coupling), this may not help and could hurt. The coupling in CFD means Ux/Uy and p are not independent — pressure is driven by velocity divergence in incompressible flow. Completely de-emphasizing velocity loss risks incoherent flow predictions. p_weight=3.0 (not ∞) keeps velocity in the training signal. Can also combine with H3 (surf_weight=50) but should test individually first.

---

## Summary Table

| # | Slug | One-liner | Est. delta | Risk |
|---|------|-----------|-----------|------|
| H1 | `grad-clip-norm1` | Add gradient clipping max_norm=1.0 | -1 to -3% | Low |
| H2 | `amp-bf16` | Enable BF16 mixed precision for faster training | -2 to -5% | Medium |
| H3 | `surf-weight-50` | Increase surface loss weight from 10→50 | -3 to -8% | Low-Med |
| H4 | `smooth-l1-beta01` | Replace MSE with SmoothL1(beta=0.1) | -2 to -5% | Medium |
| H5 | `wider-n192` | Widen model to n_hidden=192, n_head=6 | -2 to -6% | Medium |
| H6 | `warmup-5ep` | Add 5-epoch linear LR warmup | -1 to -3% | Low |
| H7 | `ema-decay999` | EMA model weights decay=0.999 for eval | -1 to -3% | Low |
| H8 | `p-channel-weight3x` | 3x per-channel weight on pressure in loss | -2 to -5% | Low-Med |

### Recommended priority order for assignment

If assigning to 8 simultaneous students, order by expected impact / risk ratio:

1. H3 `surf-weight-50` — highest direct alignment with metric, zero cost, strongest prior
2. H1 `grad-clip-norm1` — well-established, minimal risk, likely training instability fix
3. H8 `p-channel-weight3x` — zero cost, direct metric alignment, complements H3
4. H7 `ema-decay999` — zero cost, zero risk, empirically validated across many domains
5. H6 `warmup-5ep` — zero cost, well-validated, combines with any other change
6. H2 `amp-bf16` — speedup enables more gradient steps, moderate implementation risk
7. H4 `smooth-l1-beta01` — loss reformulation, well-motivated, verify normalized residuals first
8. H5 `wider-n192` — capacity increase, run last as it is most expensive

### Combination experiments (second round)

If individual experiments succeed, priority combos to try:
- H1 + H3: grad clip + higher surf weight (both fix training stability/alignment)
- H3 + H8: surf_weight=50 + p_weight=3 (dual pressure emphasis, different levels)
- H1 + H6 + H7: grad clip + warmup + EMA (full training stabilization stack)
- Best individual winner + H2 (AMP for faster iteration on confirmed winner)
