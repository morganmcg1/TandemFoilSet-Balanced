<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — TandemFoilSet Round 1 (2026-05-15)

Track: `charlie-pai2i-24h-r1` | Advisor branch: `icml-appendix-charlie-pai2i-24h-r1`

## TOP PICKS FOR ROUND 1

The three highest expected-value ideas for immediate assignment:

1. **Idea 2 — Surface-Only Loss with Pressure Head (surf_weight=30 + p-channel weight)**
   Reasoning: The primary metric is surface pressure MAE. The baseline trains on vol + 10×surf with equal weighting across Ux/Uy/p. This is the cheapest change with the clearest mechanistic tie to the ranking metric. Every wasted gradient step on volume Uy is a missed step on surface p.

2. **Idea 1 — LR Warmup + Gradient Clipping**
   Reasoning: No warmup and no gradient clipping are both documented training instability sources for transformers. This change costs zero VRAM, zero architecture change, and directly addresses two known weaknesses simultaneously. Should produce clean gains on all splits.

3. **Idea 7 — Wider/Deeper Transolver (n_hidden=256, n_layers=8, slice_num=128)**
   Reasoning: At 1.6M params the baseline is severely under-parameterized relative to the problem complexity (74K–242K nodes, 3 domains, wide Re range). Scaling to ~12M params on a 96GB GPU is trivially feasible. If the bottleneck is model capacity this is the lever; if not, the null result is informative.

---

## Idea 1 — LR Warmup + Gradient Clipping

### What it is
Add a linear warmup schedule over the first 5% of epochs, then hand off to cosine annealing, and clip gradient norms at 1.0.

### Hypothesis
The baseline optimizer starts at lr=5e-4 immediately with no warmup, and there is no gradient clipping anywhere in the training loop. Transformer-style models with attention and residual connections are well-documented to diverge or get stuck in suboptimal basins in early training when gradients are large and LR is already at maximum. Gradient spikes — documented as the primary cause of loss spikes in large transformer training (Molybog et al., 2023) — are especially harmful here because the PhysicsAttention slice softmax can saturate early, leading to degenerate slice assignments that are hard to escape. Adding a 5-epoch warmup gives the slice projection matrix time to develop non-degenerate assignments before full LR is applied. Gradient clipping (norm=1.0) prevents any single high-Re outlier batch from collapsing learned representations.

### Implementation (train.py changes, ~15 lines)

Replace the scheduler setup block (lines 406–407):
```python
# Warmup + cosine annealing via SequentialLR
warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))
warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS - warmup_epochs
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
)
```

Add gradient clipping (after `loss.backward()`, before `optimizer.step()`):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Add `warmup_epochs` to Config for ablation flexibility:
```python
warmup_frac: float = 0.05
grad_clip: float = 1.0
```

### Predicted delta and risk
- Expected: -3% to -8% on `val_avg/mae_surf_p`
- Risk: LOW. Both techniques are standard; failure mode is no improvement (null result), not regression.
- Key failure mode: If the baseline already trains stably (no gradient spikes), this adds near-zero benefit.

### Why now
No warmup on a fresh track is the most easily overlooked training deficiency. The cost is literally 3 lines. This should be in every experiment as a default.

---

## Idea 2 — Surface-Pressure-Focused Loss (higher surf_weight + p-channel upweight)

### What it is
Increase `surf_weight` from 10.0 to 30.0, and add a per-channel weight vector that upweights the pressure channel (p) within the surface loss. Loss becomes: `vol_loss + 30 * (w_Ux * surf_Ux + w_Uy * surf_Uy + w_p * surf_p)` where `w_p=3.0, w_Ux=w_Uy=1.0` (or tunable from Config).

### Hypothesis
The ranking metric is `mae_surf_p` — surface pressure MAE only. The baseline loss weights surface nodes 10× but splits that 10× uniformly across Ux, Uy, and p. One third of the surface gradient signal is pushing toward Ux accuracy, one third toward Uy, and only one third toward p — but the entire competition ranking depends on p alone. Doubling surf_weight to 30 (from 10) and tripling the p-channel weight within that term means surface pressure gets ~9× more gradient signal than the baseline gives it, while volume predictions get ~1/10 of the surface gradient in total. This is the most direct mechanistic alignment between the loss function and the evaluation metric. The physical justification is also clean: in aerodynamics, pressure is the integrated force-bearing quantity, while velocity components near the surface are largely constrained by the no-slip condition — you cannot improve pressure prediction much beyond what accuracy of the velocity field already provides, but the gradient direction for pressure regression is distinct enough that differential weighting matters.

### Implementation (train.py changes, ~10 lines)

Add to Config:
```python
surf_weight: float = 30.0   # was 10.0
p_surf_weight: float = 3.0  # extra weight on pressure channel within surf loss
```

In training loop, replace the loss computation (lines 450–453):
```python
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
# Weighted surface loss: channels [Ux, Uy, p] with p upweighted
chan_weights = torch.tensor([1.0, 1.0, cfg.p_surf_weight],
                             device=device).reshape(1, 1, 3)
surf_sq = sq_err * surf_mask.unsqueeze(-1) * chan_weights
surf_loss = surf_sq.sum() / (surf_mask.sum().clamp(min=1) * (2 + cfg.p_surf_weight))
loss = vol_loss + cfg.surf_weight * surf_loss
```

Also update `evaluate_split` to use same weighted surface loss for monitoring consistency.

### Predicted delta and risk
- Expected: -5% to -15% on `val_avg/mae_surf_p`, potentially larger on OOD splits
- Risk: LOW-MEDIUM. Raising surf_weight too aggressively can destabilize training (volume predictions may diverge, contaminating the physical solution). The 30+3p combination is a moderate first step. Failure mode: volume solution degrades and, through the shared attention representations, pulls surface p along with it.
- Diagnostic: watch `val_*/mae_vol_p` alongside `mae_surf_p` to catch volume degradation early.

### Why now
The loss-metric misalignment is the clearest structural weakness in the baseline. It should be fixed in round 1.

---

## Idea 3 — bf16 Mixed Precision + Larger Effective Batch

### What it is
Enable `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` for forward and backward, with a GradScaler, allowing batch_size to scale from 4 to 8 within the same VRAM budget.

### Hypothesis
The baseline runs in float32 on a 96GB GPU. A cruise-domain sample has ~210K nodes × 24 features × 4 bytes = ~50MB per sample in float32. A batch of 4 padded to the worst-case 242K nodes is ~4 × 242K × 24 × 4 ≈ 93MB just for x — the model activations dwarf this. With bf16, activation memory roughly halves, enabling batch_size=8, which doubles gradient averaging quality and increases per-epoch sample coverage. Unlike fp16, bf16 has the same exponent range as float32, making it numerically safe for the wide value ranges in this dataset (y values up to ±29K). The higher batch size also reduces variance in the balanced sampler, improving domain coverage per step.

### Implementation (train.py changes, ~20 lines)

Add to Config:
```python
use_bf16: bool = True
batch_size: int = 8   # double from 4
```

Wrap forward pass in training loop:
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred - y_norm) ** 2
    ...
    loss = vol_loss + cfg.surf_weight * surf_loss

scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_bf16)
# Then: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
```

Note: keep eval pass in float32 for metric accuracy (just skip the autocast context).

### Predicted delta and risk
- Expected: +10-25% throughput, potentially -2% to -5% on `val_avg/mae_surf_p` from improved batch quality
- Risk: LOW. bf16 is numerically safe for this value range. Failure mode: bf16 denormals in extreme-Re samples causing NaN loss — watchable by monitoring loss curve.
- Key win: if training time is the bottleneck, this compounds every other improvement.

### Why now
Memory efficiency improvements multiply the value of all other techniques by enabling more epochs or larger batches in the same wall-clock budget.

---

## Idea 4 — Per-Sample Normalization by sqrt(Re) Inside the Model

### What it is
Instead of (or in addition to) the global normalization, scale predicted outputs by a per-sample factor derived from `log(Re)` before computing the loss and before passing to scoring. The idea is a learned affine rescaling of predictions per sample, conditioned on Re.

### Hypothesis
The dataset's value ranges span more than an order of magnitude across Re. `val_single_in_dist` shows per-sample y-std from near-zero to 2,077 within one split. The global normalization divides by a single `y_std` across all samples, meaning high-Re samples contribute ~(2077/458)^2 ≈ 20× more to the MSE loss than low-Re samples. The model is effectively trained almost entirely on high-Re signal and evaluated on everything. A learned Re-conditioned output scale — injected as a scalar multiplier on the final prediction — acts like per-sample normalization without breaking the model contract. Concretely: the model predicts in a "Re-normalized" space, then a small MLP maps `log(Re)` → 3 scale factors (one per output channel), and the final prediction is `scale * model_pred`. This is analogous to FiLM conditioning (Perez et al., 2018). The physical basis is that for incompressible flow, velocity scales with U_inf (correlated with Re at fixed geometry) and pressure with `rho * U_inf^2`.

### Implementation (train.py changes, ~25 lines)

Add a `ReScaler` module after the Transolver output:
```python
class ReScaler(nn.Module):
    def __init__(self, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # init: scale=1 (exp(0))

    def forward(self, pred, x_norm):
        log_re = x_norm[:, :, 13:14].mean(dim=1)  # [B, 1] — log(Re) feature
        log_scale = self.net(log_re)  # [B, 3]
        scale = torch.exp(log_scale).unsqueeze(1)  # [B, 1, 3]
        return pred * scale
```

Wrap Transolver with ReScaler in the forward call. Pass `x_norm` through to enable the Re lookup.

### Predicted delta and risk
- Expected: -5% to -12% on `val_avg/mae_surf_p`, especially on `val_re_rand`
- Risk: MEDIUM. If the global normalization already handles Re variation well, this adds noise. The zero-init ensures identity behavior at initialization. Failure mode: scale collapse to near-zero for rare Re values.
- Ablation first: check if Re-stratified val losses are currently imbalanced before implementing.

### Why now
The Re variation is an under-diagnosed bottleneck. The `val_re_rand` split directly tests Re generalization, and the physical scaling laws make this hypothesis mechanistically grounded.

---

## Idea 5 — Huber Loss (delta=1.0) Instead of MSE

### What it is
Replace `(pred - y_norm)**2` with `F.huber_loss(pred, y_norm, delta=1.0)` for both volume and surface terms.

### Hypothesis
MSE loss squares errors, giving quadratic weight to large deviations. In normalized space, high-Re samples still produce larger normalized residuals early in training (the model is undertrained for those extremes), so the first few epochs are dominated by the hardest samples. Huber loss (smooth L1 beyond delta=1.0) is linear for |error| > delta, reducing the gradient from outlier samples by a factor proportional to the error magnitude. This allows the model to fit the typical (lower-Re, moderate-value) samples more efficiently during early training while still being pulled toward the hard cases. The delta=1.0 threshold in normalized space corresponds to a 1-sigma error, so the transition from quadratic to linear happens at a physically meaningful scale. This is analogous to Huber regression in robust statistics and is used in object detection (Girshick 2015) and physics-informed NNs.

### Implementation (train.py changes, ~5 lines)

Add to Config:
```python
loss_type: str = "mse"   # "mse" or "huber"
huber_delta: float = 1.0
```

Replace `sq_err = (pred - y_norm) ** 2` with:
```python
if cfg.loss_type == "huber":
    # Compute per-element Huber loss manually to preserve masking
    abs_err = (pred - y_norm).abs()
    sq_err = torch.where(
        abs_err < cfg.huber_delta,
        0.5 * abs_err ** 2,
        cfg.huber_delta * (abs_err - 0.5 * cfg.huber_delta),
    )
else:
    sq_err = (pred - y_norm) ** 2
```

The rest of the masking and loss aggregation code is unchanged — it already operates on `sq_err`.

### Predicted delta and risk
- Expected: -2% to -8% on `val_avg/mae_surf_p`, more benefit on OOD splits
- Risk: LOW. Huber and MSE are equivalent for small errors; worst case is no improvement. The delta hyperparameter needs one ablation (try 0.5 and 2.0 if 1.0 is flat).
- Failure mode: if the high-error samples are actually the ones containing important physics signal (e.g., separation points at high Re), dampening their gradients could hurt OOD generalization.

### Why now
MSE is rarely the right choice when the error distribution has heavy tails. This dataset's Re range guarantees heavy tails early in training.

---

## Idea 6 — Surface Node Oversampling via Node-Level Reweighting in Loss

### What it is
Within each batch, reweight the surface-node loss by a factor proportional to `n_vol / n_surf` for that sample, ensuring surface and volume nodes contribute proportionally equal gradient per node rather than per count. Additionally, add an auxiliary loss that enforces continuity of surface pressure across adjacent surface nodes.

### Hypothesis
In typical meshes, surface nodes are ~2-5% of total nodes (a 127K-node mesh might have ~3-6K surface nodes). The current `surf_loss` computes `sum(sq_err * surf_mask) / surf_mask.sum()` and `vol_loss` computes `sum(sq_err * vol_mask) / vol_mask.sum()`. These are already normalized per node count, so they are comparable in magnitude. However, the combined loss `vol_loss + 10 * surf_loss` still gives the volume 1/11 of the gradient magnitude despite the volume containing 97% of nodes. This is intentional but the effective model-gradient-per-surface-node is proportional to `10 × (surf_nodes / total_nodes) ≈ 10 × 0.03 = 0.3`, while volume gets `1 × 0.97 = 0.97`. Under this framing, increasing surf_weight to ~32 would make surface nodes contribute as much total gradient as volume nodes. The auxiliary loss idea: consecutive surface nodes share a physical boundary and their pressure should be smooth; an L2 penalty on `||p_i - p_{i+1}||` for ordered surface nodes adds an implicit regularization that smoothness solvers use. This is related to the boundary condition enforcement in physics-informed models.

### Implementation (train.py changes, ~20 lines)

Config addition:
```python
surf_smoothness_weight: float = 0.0  # default off; try 0.1
```

Surface smoothness loss (requires is_surface mask + node ordering, approximated by nearest-neighbor distance in x[:, :, 0:2]):
```python
# Only compute if enabled and surface nodes exist
if cfg.surf_smoothness_weight > 0:
    surf_nodes = pred[surf_mask]  # [K, 3] surface predictions
    surf_pos = x_norm[surf_mask, :2]  # [K, 2] positions
    # simple sequential smoothness: diff of consecutive surface nodes
    if surf_nodes.shape[0] > 1:
        smooth_loss = ((surf_nodes[1:] - surf_nodes[:-1]) ** 2).mean()
        loss = loss + cfg.surf_smoothness_weight * smooth_loss
```

Note: the node ordering assumption (surface nodes are consecutive) is approximately true in CFD meshes but should be validated on one sample first.

### Predicted delta and risk
- Expected (surf_weight only part): same as Idea 2; (-3% to -10%)
- Expected (smoothness): -1% to -3% on `val_avg/mae_surf_p` if surfaces are ordered
- Risk: MEDIUM. The smoothness term requires that surface nodes within a padded batch are meaningfully ordered. This assumption needs empirical validation. Failure mode: random node ordering makes the consecutive difference meaningless.

### Why now
Surface node ordering in CFD meshes is almost always meaningful (they are generated from airfoil panels in order). Worth a quick data inspection before investing here.

---

## Idea 7 — Wider/Deeper Transolver (n_hidden=256, n_layers=8, slice_num=128)

### What it is
Scale the Transolver from (n_hidden=128, n_layers=5, slice_num=64) to (n_hidden=256, n_layers=8, slice_num=128), growing parameter count from ~1.6M to ~12M while fitting comfortably in 96GB VRAM.

### Hypothesis
The baseline Transolver has 1.6M parameters to predict 3D fields over meshes with up to 242K nodes spanning 3 physically distinct domains, 4 Re decades, and 5+ distinct geometric configurations. For comparison, a typical ViT-S has 22M parameters for 1000-class ImageNet; a U-Net++ for medical segmentation uses 10-50M parameters for far simpler problems. The model is almost certainly in an underfitting regime. The main question is whether the Transolver architecture can exploit additional capacity. Doubling n_hidden gives 4× more capacity per layer (quadratic in hidden dim for attention). Increasing slice_num from 64 to 128 doubles the "physics token vocabulary" — each slice token corresponds to a learnable physical mode; 64 may be insufficient to separately represent boundary layer nodes, wake nodes, separation zones, and far-field flow across two foil geometries and a wide Re range. Adding 3 more layers allows deeper composition of these physical modes. VRAM estimate: ~12M params × 4 bytes = 48MB weights; activations for a 242K × 256 batch dominate at ~250MB, well within 96GB budget even at batch_size=4.

### Implementation (train.py changes, ~8 lines)

Modify `model_config` dict (lines 389–400):
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,     # was 128
    n_layers=8,       # was 5
    n_head=8,         # was 4 (keep dim_head=32)
    slice_num=128,    # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Reduce lr to 2e-4 (larger models typically need lower initial LR). Add warmup (Idea 1) as a co-change.

### Predicted delta and risk
- Expected: -8% to -20% on `val_avg/mae_surf_p` if capacity is the bottleneck
- Risk: MEDIUM. If the current model already fits training data well, scaling adds regularization overhead without improving the mapping. Diagnostic: compare train vs. val loss — a large gap indicates overfitting, small gap indicates underfitting (capacity is the bottleneck).
- Note: n_head=8 with n_hidden=256 gives dim_head=32, which is the same as the baseline (128/4=32). Preserving dim_head maintains the same attention expressiveness per head at larger scale.

### Why now
At 1.6M params the model is suspiciously small for the problem. This is the most direct capacity intervention and the result is unambiguous: either train loss drops further (underfitting confirmed) or it doesn't (look elsewhere).

---

## Idea 8 — EMA of Model Weights for Better Checkpointing

### What it is
Maintain an Exponential Moving Average (EMA) of model parameters with decay=0.999, and use the EMA model for validation and checkpointing instead of the live model.

### Hypothesis
EMA weight averaging exploits the ensemble effect of averaging model snapshots across training. A single checkpoint at any given epoch represents a point on a loss surface that may be locally noisy; EMA smooths the trajectory by implicitly averaging over a geometric window of recent checkpoints. For physics surrogate models that see correlated batches (all RaceCar single one epoch, then Cruise the next due to the balanced sampler), EMA provides a more stable prediction surface that does not overfit to the most recently seen domain. This technique is standard in modern vision models (DINOv2, SimCLR, MAE) and in neural weather forecasting (GraphCast, Pangu-Weather). The theoretical justification links to SWA (Izmailov et al., 2018): averaging weights converges to a flatter loss minimum with better generalization. EMA is strictly cheaper than SWA as it requires no learning rate schedule modifications and no separate forward pass.

### Implementation (train.py changes, ~20 lines)

Add a simple EMA class:
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)
```

After each optimizer step: `ema.update(model)`

For validation: swap model to EMA weights, evaluate, then swap back. For checkpointing: save EMA state dict.

Add to Config: `ema_decay: float = 0.999`

### Predicted delta and risk
- Expected: -2% to -6% on `val_avg/mae_surf_p`, primarily on OOD splits
- Risk: LOW. EMA is non-destructive (live model is unchanged). Worst case: EMA performs identically to best checkpoint. Failure mode: if the model is still in early training when checkpointed, the EMA trailing average may lag behind and perform worse.
- Gotcha: EMA decay of 0.999 means the effective window is ~1/(1-0.999) = 1000 steps. For a dataset of 1499 training samples at batch_size=4, one epoch ≈ 375 steps. Effective EMA window ≈ 2.7 epochs — appropriate.

### Why now
EMA is a free improvement that never hurts. Should be a default component of every experiment.

---

## Idea 9 — Fourier Positional Encoding for Node Coordinates

### What it is
Replace the raw (x, z) node coordinates passed to `preprocess` with Fourier positional encodings (FPE): `[sin(2^k * pi * x), cos(2^k * pi * x), sin(2^k * pi * z), cos(2^k * pi * z)]` for k=0,...,L-1, giving 4L features instead of 2.

### Hypothesis
The Transolver `preprocess` MLP receives raw normalized (x, z) coordinates as the positional component of the input (dims 0-1 of x after the train.py `preprocess` split). These coordinates span roughly [-5, 20] × [-10, 10] meters in physical space. A linear layer mapping 2D coordinates to a 128D hidden space must learn to represent both coarse domain structure (background vs. near-foil) and fine surface features (panel curvature, boundary layer thickness) from a 2-number input. Fourier encodings (NeRF-style, Tancik et al. 2020) provide multi-scale basis functions that allow the MLP to efficiently represent both low-frequency (flow regime) and high-frequency (surface geometry) spatial variation. This is directly applicable to 2D PDE solutions on irregular meshes, as shown in FNO variants and Fourier neural operators. The encoding is fixed (no learned parameters), costs near-zero VRAM, and the only design choice is L (typically 6-10).

### Implementation (train.py changes, ~20 lines)

Add a Fourier encoding function:
```python
def fourier_pos_encoding(pos, L=8):
    """pos: [B, N, 2] -> [B, N, 4*L]"""
    freqs = torch.pi * (2 ** torch.arange(L, device=pos.device, dtype=pos.dtype))
    encoded = []
    for d in range(pos.shape[-1]):
        coords = pos[..., d:d+1] * freqs  # [B, N, L]
        encoded.extend([torch.sin(coords), torch.cos(coords)])
    return torch.cat(encoded, dim=-1)  # [B, N, 4*L]
```

In the Transolver `forward`, replace `data["x"][:, :, :2]` with `fourier_pos_encoding(data["x"][:, :, :2], L=8)` before concatenation in `preprocess`. Update `space_dim` from 2 to 32 (4×L=4×8) in model_config.

Note: this changes `fun_dim + space_dim` from 22+2=24 to 22+32=54 in the preprocess MLP input. Verify that the model_config `space_dim=32` propagates correctly through the Transolver `preprocess` MLP.

### Predicted delta and risk
- Expected: -3% to -8% on `val_avg/mae_surf_p`
- Risk: LOW-MEDIUM. FPE is well-validated for NERFs and physics operators. Main risk: the preprocess MLP's input dim change needs careful config propagation in the existing code. A bug here would silently degrade accuracy.
- Failure mode: the dense near-foil mesh already encodes multi-scale geometry through the dsdf features (dims 4-11); FPE on coordinates may be redundant with them.

### Why now
Multi-scale spatial encoding is a known weakness of raw coordinate inputs. The dsdf features handle geometry, but coordinate-based multi-scale encoding handles flow field structure.

---

## Idea 10 — Slice Temperature Annealing (Sharper Slice Assignments Over Training)

### What it is
Initialize the PhysicsAttention `temperature` parameter at a higher value (e.g., 2.0 instead of 0.5) and anneal it down toward 0.1 over training using a cosine or linear schedule, progressively sharpening the soft slice assignments.

### Hypothesis
The Transolver's physics token assignments use a softmax with a learnable temperature: `slice_weights = softmax(in_project_slice(x_mid) / temperature)`. Early in training, with random slice projection weights, all nodes assign nearly uniformly to all slices (high temperature = flat distribution). As training progresses, the slice projections learn to separate physical regions. However, if temperature is a freely learned parameter starting at 0.5, it may stay in a soft-assignment regime that is suboptimal for later training — the slice tokens would be "fuzzy" averages over physically distinct regions rather than clean prototypes for boundary layer, wake, and far-field. Annealing temperature from high (2.0, nearly uniform) to low (0.1, nearly hard) follows a curriculum: start with coarse global information sharing, progressively refine to local physics clusters. This is analogous to slot attention annealings used in object-centric learning and VQ-VAE codebook commitment. The key insight from slot attention normalization research (Biza et al., 2023) is that sharper assignments generalize better to unseen cardinalities — directly applicable to our mesh size variation.

### Implementation (train.py changes, ~25 lines)

Modify PhysicsAttention `__init__` to take `initial_temperature`:
```python
self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * initial_temperature)
```

Add a temperature scheduler in the training loop — after each epoch, set:
```python
# Anneal temperature for all PhysicsAttention modules
progress = (epoch + 1) / MAX_EPOCHS
target_temp = 2.0 * (0.1 / 2.0) ** progress  # cosine-ish from 2.0 to 0.1
for module in model.modules():
    if isinstance(module, PhysicsAttention):
        with torch.no_grad():
            module.temperature.fill_(target_temp)
```

Alternatively: make temperature a buffer (non-learned) and schedule it externally, removing the per-module parameter.

Add to Config: `temp_init: float = 2.0, temp_final: float = 0.1`

### Predicted delta and risk
- Expected: -2% to -6% on `val_avg/mae_surf_p`
- Risk: MEDIUM. Overriding a learned parameter with a schedule may fight the natural learning dynamics. If the model wants temperature=0.8 at epoch 20 but the schedule forces 0.3, this could degrade performance.
- Mitigation: use temperature as a non-learned buffer (remove `nn.Parameter`), making the annealing fully deterministic.

### Why now
The baseline initializes temperature=0.5 which may be a reasonable final value but is suboptimal as an initial value. This directly targets the slice assignment quality in the Transolver's physics tokenization.

---

## Idea 11 — Auxiliary Surface-Normal Gradient Prediction (Physics-Informed Auxiliary Head)

### What it is
Add an auxiliary head that predicts the surface-normal pressure gradient `dp/dn` at surface nodes, trained against a finite-difference approximation computed from the target p field and the node positions. The auxiliary loss is only applied at surface nodes.

### Hypothesis
The surface pressure MAE is the primary metric, but surface pressure is coupled to the near-surface pressure gradient through the Euler/Navier-Stokes equations. At the surface, the normal momentum equation gives `dp/dn = -rho * (v · grad)v + mu * lap(v)`, which for attached flow simplifies to `dp/dn ≈ rho * kappa * U_t^2` where kappa is surface curvature and U_t is tangential velocity. Training the model to also predict `dp/dn` at the surface forces it to develop an internal representation consistent with this physical relationship, acting as an implicit physics constraint. This is analogous to PINN auxiliary losses and to the boundary condition losses used in Modified FNO (Mao et al., 2025). The target labels for `dp/dn` can be estimated from the ground truth: for each surface node i, find neighboring non-surface nodes in the normal direction and compute a finite-difference gradient from the p values.

### Implementation (train.py changes, ~35 lines)
Requires pre-computing approximate surface normal gradients from the y tensor at each batch. The normal direction can be estimated from the `saf` (signed arc-length, dims 2-3) and coordinate features.

An approximate implementation:
```python
# Estimate dp/dn: use closest non-surface neighbor of each surface node
# For simplicity, use difference between surface-adjacent volume node and surface node
# (This requires approximate neighbor lookup — compute on first batch per epoch)
```

Note: full implementation needs a neighbor graph. This idea is higher-complexity and depends on whether `saf` encodes enough information to reconstruct normals.

### Predicted delta and risk
- Expected: -3% to -10% if the physics constraint is informative
- Risk: HIGH. Implementation complexity is significant. The finite-difference approximation of `dp/dn` from padded batches requires careful masking. Wrong approximation → misleading auxiliary signal.
- Recommended: implement as a separate, later-round experiment after simpler ideas are tested.

### Why now
Physics-informed losses have the strongest theoretical justification for fluid mechanics surrogates. But the implementation complexity makes this a round 2+ idea.

---

## Idea 12 — Separate Prediction Heads per Output Field

### What it is
Replace the single `out_dim=3` output MLP in the final TransolverBlock with three separate small MLPs, one per output channel (Ux, Uy, p). The three heads share all Transolver block representations up to the final layer, then diverge.

### Hypothesis
The current model maps from n_hidden=128 directly to 3 output channels via a single 2-layer MLP: `Linear(128, 128) -> GELU -> Linear(128, 3)`. This means the 128D representation must simultaneously encode features for predicting velocity components and pressure. In fluid dynamics, pressure and velocity are coupled but not collinear — pressure satisfies an elliptic Poisson equation while velocities satisfy the hyperbolic advection-diffusion system. Separate heads allow the 128D shared representation to be projected independently to pressure vs. velocity subspaces. Each head can specialize: the pressure head might learn to attend to the entire flow domain (elliptic influence), while velocity heads focus on local streamlines (hyperbolic). This is standard in multi-task learning when outputs have different physical origins and was validated in neural weather models that use separate heads for wind vs. pressure fields.

### Implementation (train.py changes, ~20 lines)

Modify `TransolverBlock` when `last_layer=True`:
```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Separate head for each output field
    self.mlp_Ux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
    self.mlp_Uy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
    self.mlp_p  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
```

Change the forward output for the last layer:
```python
if self.last_layer:
    h = self.ln_3(fx)
    return torch.cat([self.mlp_Ux(h), self.mlp_Uy(h), self.mlp_p(h)], dim=-1)
```

Remove the `output_fields` / `output_dims` parameters from model_config as they are now hardcoded.

### Predicted delta and risk
- Expected: -2% to -5% on `val_avg/mae_surf_p`
- Risk: LOW. More parameters at the head, orthogonal to the backbone changes. Failure mode: shared representation already captures the needed disentanglement.
- Combine with Idea 2 (p-upweight loss) for maximum synergy.

### Why now
Disentangling prediction heads is a cheap structural change that adds <1% parameters and may improve pressure prediction quality meaningfully.

---

## Idea 13 — Domain-Conditional Batch Normalization in MLP Blocks

### What it is
Replace the MLP blocks in TransolverBlock with domain-conditional variants: the normalization and scale/shift parameters of the hidden activations are conditioned on the domain type (raceCar-single, raceCar-tandem, cruise), encoded from the tandem geometry features (dims 22-23 of x: gap and stagger).

### Hypothesis
The three domains have fundamentally different flow physics: raceCar single has ground effect, raceCar tandem has downforce from two inverted foils, and cruise has freestream lift from two normal foils. The global model must use the same MLP weights for all three. Domain-conditional batch normalization (DCBN, or the more general FiLM conditioning) provides a cheap mechanism to adapt the activation statistics of MLP hidden layers per domain, allowing the same backbone to behave differently for different physical regimes. The domain can be inferred from features 18-23 (all-zero for single-foil, nonzero gap/stagger for tandem; AoA direction distinguishes raceCar from cruise). A 3-way domain classifier (trained jointly with a soft classification loss or simply hard-coded from the feature pattern) generates per-domain affine parameters for each MLP LayerNorm.

### Implementation (train.py changes, ~30 lines)

Encode domain from x features before the model:
```python
# Infer domain: 0=raceCar-single (gap=0, stagger=0), 1=raceCar-tandem (gap≠0, AoA_foil1<0), 2=cruise (gap≠0, AoA_foil1≥0)
gap = x_norm[:, :, 22]   # [B, N]
is_tandem = (gap.abs().mean(dim=1) > 0.1).float()  # [B]
aoa1 = x_norm[:, :, 14].mean(dim=1)  # [B]
domain_id = (is_tandem * (1 + (aoa1 > 0).float())).long()  # 0, 1, or 2
```

Add domain embedding to Transolver and inject into each MLP block's LayerNorm as affine shift+scale.

### Predicted delta and risk
- Expected: -3% to -8% on OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`)
- Risk: MEDIUM. Domain ID must be reliably inferred from the (already-normalized) input features. If domain classification is noisy, the conditioning could hurt. The threshold-based approach above should be validated.

### Why now
Domain shift is a documented challenge (3 distinct physical regimes). This targets the OOD generalization splits directly.

---

## Idea 14 — Stochastic Depth (LayerDrop) for Regularization

### What it is
Apply stochastic depth to TransolverBlock during training: each block is independently dropped (replaced by identity) with probability `p_drop = 0.1`, linearly increasing from 0 at the first block to `p_drop` at the last. At test time, all blocks are active with their stochastic depth scale correction.

### Hypothesis
The Transolver processes every node through every block deterministically at training time. Stochastic depth (Huang et al., 2016) provides an implicit ensemble of subnetworks during training, acting as a strong regularizer. For a 5-layer Transolver, each training step uses a random subset of layers, forcing earlier layers to develop self-sufficient representations. This is particularly valuable here because the model must generalize across unseen camber values (val_geom_camber_*) — stochastic depth prevents over-reliance on later layers' memorization of training geometries. Stochastic depth was key to training very deep Vision Transformers stably and is used in DiT, DeiT, and modern ViT variants.

### Implementation (train.py changes, ~15 lines)

Modify `TransolverBlock.forward` to add drop:
```python
def forward(self, fx):
    if self.training and self.drop_prob > 0:
        if torch.rand(1).item() < self.drop_prob:
            return fx  # skip this block entirely
    fx = self.attn(self.ln_1(fx)) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    ...
```

In Transolver `__init__`, assign linearly scaled `drop_prob` to each block:
```python
drop_probs = [i / (n_layers - 1) * stoch_depth_rate for i in range(n_layers)]
```

Add to Config: `stoch_depth_rate: float = 0.1`

### Predicted delta and risk
- Expected: -1% to -4% on OOD splits (val_geom_camber splits)
- Risk: LOW. Stochastic depth cannot hurt in expectation if the model is not already underfit. Worst case: null result.
- Note: do not apply to the last block (last_layer=True) as it contains the output head.

### Why now
Regularization on OOD splits is free and this is a well-tested technique for transformers. Pairs well with Idea 7 (larger model).

---

## Idea 15 — Separate Surface and Volume Encoders with Shared Backbone

### What it is
Split the input processing into two paths: surface nodes go through a dedicated `surface_encoder` MLP before entering the Transolver backbone; volume nodes go through a `volume_encoder` MLP. Both outputs have the same hidden dimension and are concatenated (interleaved by node type) before the first TransolverBlock.

### Hypothesis
Surface nodes and volume nodes have fundamentally different physical roles: surface nodes live on the airfoil boundary (no-slip, known normal pressure gradient), while volume nodes are in the flow interior. The current model treats them identically through the same `preprocess` MLP. A surface-specialized encoder can develop features tuned for boundary conditions (curvature, local panel angle from `saf` features) while the volume encoder focuses on flow field features (local Re, distance to surface from `dsdf`). This architectural prior is analogous to boundary-aware GNNs (B-GNNs) for incompressible flow, which demonstrated 83% parameter reduction over vanilla GNNs by specializing boundary vs. interior processing.

### Implementation (train.py changes, ~30 lines)

Replace the single `preprocess` MLP in Transolver with two:
```python
self.surf_encoder = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
self.vol_encoder  = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
```

In `forward`, split by `is_surface`:
```python
# is_surface is part of x features (dim 12)
surf_mask_inner = x[:, :, 12:13] > 0.5  # [B, N, 1]
fx_surf = self.surf_encoder(x)
fx_vol  = self.vol_encoder(x)
fx = torch.where(surf_mask_inner, fx_surf, fx_vol)
fx = fx + self.placeholder[None, None, :]
```

The rest of the model is unchanged. This adds ~0.2M parameters (~12% increase over baseline).

### Predicted delta and risk
- Expected: -3% to -8% on `val_avg/mae_surf_p`
- Risk: LOW-MEDIUM. The two encoders are initialized identically and will diverge during training only if the gradients from surface vs. volume losses are sufficiently different. If the gradient signal is dominated by volume (which has many more nodes), the encoders may not specialize.
- Mitigation: combine with Idea 2 (higher surf_weight) to ensure surface loss drives surface encoder specialization.

### Why now
Surface/volume specialization is physically motivated and is a first-principles change to the model rather than a hyperparameter tweak.

---

## Idea 16 — Curriculum Learning: Low-Re First, Then High-Re

### What it is
During the first 30% of training epochs, oversample low-Re examples (Re < 1M) by setting a Re-stratified weight in the sampler. In the remaining 70%, return to uniform (or the existing balanced domain) sampling.

### Hypothesis
High-Re flows have much larger absolute values (velocity/pressure) and steeper gradients in the flow field. Starting with high-Re samples causes the loss to be dominated by the hardest examples, potentially leading to poor convergence on the easier low-Re regime. A curriculum (Bengio et al., 2009) that starts with easier samples (low-Re, more laminar, smoother fields) allows the model to learn the basic flow structure first, then refine for turbulent/separated high-Re flows. This mirrors how CFD solvers often initialize with a low-Re solution before ramping Re. In machine learning, curriculum learning has been applied to physics simulations in climate models and protein structure prediction. The `log(Re)` feature in dimension 13 allows straightforward stratification from the input features without modifying data loaders.

### Implementation (train.py changes, ~20 lines)

Add to Config: `curriculum_fraction: float = 0.3`

Modify `sample_weights` in the training setup to include a Re-stratified downweight for high-Re:
```python
# Extract log_Re from dataset files (cached at startup)
log_re_vals = []
for f in train_ds.files:
    s = torch.load(f, weights_only=True)
    log_re_vals.append(s["x"][:, 13].mean().item())  # dim 13 = log(Re)
log_re_tensor = torch.tensor(log_re_vals)

# During training loop, update sampler weights based on epoch
def get_epoch_weights(epoch, max_epochs, curriculum_frac, base_weights, log_re):
    if epoch / max_epochs < curriculum_frac:
        # Phase 1: upweight low-Re samples
        re_weight = torch.exp(-0.5 * log_re)  # lower Re -> higher weight
        re_weight = re_weight / re_weight.mean()
        return base_weights * re_weight
    return base_weights
```

Note: WeightedRandomSampler does not support dynamic weight updates per epoch without creating a new sampler. This requires creating a new DataLoader each epoch during the curriculum phase — feasible but adds ~2s setup overhead per epoch.

### Predicted delta and risk
- Expected: -2% to -5% on low-Re val splits; possibly +1% on `val_re_rand` (mixed benefit)
- Risk: MEDIUM. Loading Re values from all training files at startup adds ~5min overhead. Creating a new DataLoader per epoch adds overhead. The benefit depends on whether low-Re underfitting is currently a bottleneck.
- Diagnostic: check if `val_single_in_dist` (which spans 104K–5M Re) has worse performance at low Re vs. high Re within the split.

### Why now
If Re variation is driving high MAE, curriculum is a low-cost experiment. But requires data inspection first to confirm the hypothesis.

---

## Idea 17 — Aggressive Surface Weight + Small Model Ablation (Diagnostic)

### What it is
A diagnostic ablation: train with `surf_weight=100.0` (extreme surface emphasis) and compare to surf_weight=10.0 baseline on validation curves, to identify whether the current loss is the binding constraint on `mae_surf_p`. This is not intended to be deployed but to inform whether loss reformulation (Ideas 2, 11) will yield significant gains.

### Hypothesis
If increasing surf_weight from 10 to 100 improves `mae_surf_p` monotonically, it means the current weighting is the bottleneck and we should search aggressively in the [10, 100] range. If performance plateaus or degrades, it means the model's representational capacity or the volume loss's role in stabilizing training is the bottleneck instead.

### Implementation (train.py changes, ~2 lines)
```python
surf_weight: float = 100.0
```

That's it. This is purely a diagnostic experiment.

### Predicted delta and risk
- Expected: likely mixed — `mae_surf_p` improves, `mae_vol_*` degrades, training may become unstable
- Risk: LOW (it is explicitly a diagnostic; the insight is worth more than the metric outcome)
- Decision tree: if surf_weight=100 is strictly better than baseline: run surf_weight sweep [20, 30, 50]. If worse: focus capacity/architecture improvements.

### Why now
Before spending GPU time on complex loss formulations, this 1-line change tells us how much headroom the loss weighting has. Cheapest possible diagnostic.

---

## Idea 18 — AdamW + OneCycleLR Replacing CosineAnnealingLR

### What it is
Replace `CosineAnnealingLR` with `OneCycleLR` (super-convergence schedule): a single cycle of LR that rises to `max_lr` in the first 30% of training then decays to near-zero, with `pct_start=0.3, div_factor=25, final_div_factor=1e4`. This embeds warmup, annealing, and aggressive final decay in one schedule.

### Hypothesis
`CosineAnnealingLR` decays from `lr=5e-4` to near-zero over `T_max=MAX_EPOCHS`. There is no warmup and no final "freeze" phase. `OneCycleLR` (Smith & Topin, 2018) was developed for fast convergence and has been shown to outperform cosine annealing on a wide range of tasks by providing a strong LR boost in the middle of training that escapes flat loss regions, followed by an aggressive decay that allows fine-tuning in the sharp minimum. For physics surrogate training, the middle-phase LR boost helps the model cross potential saddle points in the loss surface corresponding to mode switches between physical regimes (e.g., when the model transitions from predicting laminar to turbulent wake structure). Linear decay at the end (Defazio et al., 2023) shows the schedule's end behavior matters more than its peak — `OneCycleLR`'s `final_div_factor=1e4` provides a `5e-8` final LR that enables fine-tuning.

### Implementation (train.py changes, ~5 lines)

Replace scheduler setup:
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.lr,
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    anneal_strategy='cos',
)
```

Move `scheduler.step()` inside the training loop (after each batch step), not after each epoch.

### Predicted delta and risk
- Expected: -3% to -7% on `val_avg/mae_surf_p`
- Risk: LOW-MEDIUM. `OneCycleLR` can cause instability if `max_lr` is too high for the architecture. With the baseline `max_lr=5e-4` this is unlikely.
- Failure mode: if the model already converges well with cosine annealing, `OneCycleLR`'s mid-training LR spike may push it to a worse minimum.

### Why now
Cosine annealing with no warmup is a suboptimal choice. `OneCycleLR` integrates warmup naturally and has stronger empirical backing across model sizes.

---

## Research State Update

### Current best explanation for bottlenecks (fresh track, no prior runs)

**Training stability**: No LR warmup, no gradient clipping. Transformer training without these is known to produce early instabilities that hurt final convergence.

**Loss-metric misalignment**: `surf_weight=10.0` with equal Ux/Uy/p weighting. The metric is surface p only. One third of surface gradient is "wasted" on Ux, one third on Uy.

**Model capacity**: At 1.6M params, the model is almost certainly underfit for the complexity of 3-domain CFD with wide Re range and variable geometry.

**Re-scale variation**: Global normalization does not account for the order-of-magnitude variation in field values across Re. High-Re outliers dominate MSE gradients.

### Ruled-out paths
None yet (fresh track). All ideas are new proposals.

### Open uncertainties
1. Whether model capacity (Idea 7) or loss alignment (Ideas 2, 5) is the primary bottleneck — this determines whether to invest GPU time in larger models or loss engineering first.
2. Whether the three training domains are already well-balanced by the existing weighted sampler or whether some domains have systematically higher train/val loss gaps.
3. Whether the val_geom_camber OOD splits require architectural changes (domain conditioning) or are already addressable by better training setup.

### Next discriminating experiments
1. **Idea 17 (diagnostic surf_weight=100)**: 1-line change; result separates "loss is the bottleneck" from "capacity/architecture is the bottleneck".
2. **Idea 2 (surf_weight=30 + p-upweight)**: applies regardless of the diagnostic result; directly targets the metric.
3. **Idea 1 (warmup + clipping)**: zero-cost training stability improvement; should always be enabled.

### Stop conditions
- If Ideas 1+2+7 combined show no improvement over baseline: loss formulation and capacity are not the bottleneck; investigate data representation (Ideas 9, 15) and physics priors (Idea 11).
- If val_geom_camber splits remain high after ideas 1-7: domain-specific conditioning (Idea 13) becomes the priority.
- If `mae_surf_p` does not improve after 8-10 experiments: investigate evaluation metric assumptions — check if the scoring script's global normalization is computing a fair cross-domain comparison.
