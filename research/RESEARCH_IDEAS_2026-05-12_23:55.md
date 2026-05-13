# SENPAI Research Ideas — 2026-05-12 23:55

## Context

**Plateau Protocol active.** val_avg/mae_surf_p = 101.810 (baseline). Optimizer/schedule/batch/regularization hyperparameter space fully exhausted over 5+ rounds. Meta-diagnosis: system is CONVERGENCE-LIMITED — ~12 epochs in 30 min at ~175 s/epoch (fp32). Only structural changes that (a) increase epoch throughput at same wall-clock or (b) improve per-epoch quality at near-zero extra cost can win.

**Dead ends (do not repeat):** EMA decay=0.999 (cold-start drag), n_layers=7 (too slow), n_hidden=192 (too slow), n_head=8 (too slow), slice_num=128 (too slow), dropout=0.1 (underfit), betas (0.95, 0.99) (worse), all grad clip values (worse), Huber (worse), warm restarts (worse), batch=8 (step-count-limited), T_max tuning (noise), weight_decay extremes (worse).

---

## Hypothesis Ranking (by expected impact at 30-min budget)

---

## 1. bf16 Mixed Precision (HIGHEST PRIORITY)

### What it is

Wrap the forward pass and loss computation in `torch.autocast('cuda', dtype=torch.bfloat16)`. This halves the compute-bound portion of the forward pass and roughly doubles epoch throughput on A100/H100 class hardware.

### Why it might help here

The system is explicitly convergence-limited: ~12 epochs in 30 min. bf16 on A100 delivers 2x FP16/BF16 tensor core throughput vs FP32 for matmul-heavy workloads. Expected epoch time: ~80-100 s/epoch, yielding ~18-22 epochs in 30 min. That is 50-80% more gradient steps at identical wall-clock, which directly attacks the convergence bottleneck. BF16 (vs FP16) is the right choice: same exponent range as FP32, so no loss scaling required, and much safer for transformer attention activations.

**Critical risk and fix:** The `in_project_slice / self.temperature` operation at line ~120 of `PhysicsAttention.forward()` produces values that feed into softmax. Division by a learned scalar `temperature` (initialized to 0.5) can produce large intermediate values that overflow BF16's limited mantissa precision. NVIDIA's PhysicsNeMo engineering notes and the broader transformer-at-bf16 literature both document this. The fix: cast that specific operation to float32 before softmax, then cast back. All other operations (matmul, layernorm, MLP) are bf16-safe.

### Mechanism

More gradient steps at identical wall-clock → more training signal → lower convergence loss. No change to architecture or loss formulation. The only quality risk is numerical precision in the slice attention step, which is mitigated by the targeted float32 cast.

### Per-epoch wall-clock prediction

Baseline: ~175 s/epoch. Expected with bf16: ~85-100 s/epoch (estimated 50-75% speedup on matmul-dominant workload on A100 80GB). Net gain: ~18-22 epochs vs ~12 at 30-min cap.

### Exact code change

```python
# In PhysicsAttention.forward(), replace:
#   slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
# with:
    slice_weights = self.softmax(
        (self.in_project_slice(x_mid).float() / self.temperature.float()).to(x_mid.dtype)
    )

# In the training loop, wrap forward + loss with autocast:
# Before the optimizer step block (replace the current forward call):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        pred = model({"x": x_norm})["preds"]
        abs_err = (pred - y_norm).abs()
        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

No GradScaler needed — BF16 does not require loss scaling (unlike FP16).

### CLI invocation

```bash
python train.py \
  --experiment_name bf16_mixed_precision \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

(All other config is unchanged from current baseline.)

### Falsifying result

If per-epoch time does not drop below ~130 s (less than 25% speedup), the workload is memory-bandwidth-limited not compute-limited. If val_avg/mae_surf_p > 115 (>13% worse), numerical instability in bf16 is dominating despite the fix. Either outcome rules out bf16 as a throughput lever.

---

## 2. Lion Optimizer (SECOND PRIORITY)

### What it is

Replace AdamW with the Lion (EvoLved Sign Momentum) optimizer. Lion uses only the sign of the gradient momentum for the update: `m_t = β₁·m_{t-1} + (1-β₁)·∇L`, `θ_t = θ_{t-1} - η·sign(β₁·m_{t-1} + (1-β₁)·∇L)`. No second moment — half the optimizer state memory.

### Why it might help here

L1 loss already produces constant-magnitude gradients: `sign(pred - target)`. The model is therefore already computing something akin to a sign update for every parameter, but AdamW's adaptive second moment damps this signal by tracking squared magnitudes. Lion drops the second moment entirely and directly uses sign updates, which should pair naturally with L1's constant-gradient dynamics. The memory saving (~30% less optimizer state) also frees VRAM for slightly larger effective batches or sequence lengths, though the primary argument is per-step quality not throughput.

Literature: Chen et al. 2023 (Google Brain) showed Lion matches or exceeds AdamW on language and vision tasks at 2-10x less memory for optimizer state. The sign mechanism is reminiscent of signSGD, which has convergence proofs under L1 loss conditions.

**Key hyperparameter rescaling**: Lion's effective update magnitude is 1.0 (the sign), whereas AdamW's is lr * (gradient / sqrt(v_t + eps)). Typical rescaling: lr_lion ≈ lr_adamw / 3 to lr_adamw / 10. For our lr=5e-4: Lion lr should be 5e-5 to 2e-4. Start at 1e-4.

**Package requirement**: `lion-pytorch` must be added to `pyproject.toml`. Current pyproject.toml does not include it.

### Mechanism

Sign-based updates are more aggressive directional steps with implicit gradient clipping. For L1 loss, this removes the mismatch between AdamW's magnitude-adaptive scaling and the already-constant gradient magnitudes, potentially giving cleaner directional convergence per step.

### Per-epoch wall-clock prediction

Lion has slightly lower per-step cost than AdamW (one fewer running-average buffer, no sqrt/division for second moment). Expected: ~160-170 s/epoch, roughly 3-5% faster. Primarily a quality-per-step improvement, not a throughput improvement.

### Exact code change

```python
# In pyproject.toml, add to dependencies:
#   "lion-pytorch>=0.2.0",

# In train.py, import:
from lion_pytorch import Lion

# Replace AdamW instantiation (current line ~418):
# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
```

Note: Lion's betas have different semantics than AdamW. β₁=0.9 controls momentum weight, β₂=0.99 controls weight decay direction. Keep CosineAnnealingLR unchanged.

### CLI invocation

```bash
python train.py \
  --experiment_name lion_optimizer_lr1e4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

### Falsifying result

If val_avg/mae_surf_p > 110 (>8% worse), Lion's sign update conflicts with this specific optimization landscape. The AdamW betas experiments (0.95,0.99 = +15.4% worse; β1=0.85 = +5.9% worse) suggest this landscape is somewhat sensitive to momentum formulation — Lion may not work here even though L1 arguments suggest it should.

---

## 3. GeGLU Activation in MLP Blocks

### What it is

Replace GELU activations in the MLP feedforward blocks with Gated Linear Unit variants: `GeGLU(x) = (xW₁ ⊙ GELU(xV)) W₂`. The gate `GELU(xV)` modulates which features pass through the feedforward layer, effectively learning a soft feature selection.

### Why it might help here

GeGLU and SwiGLU (the SiLU-gated variant) have become standard in modern transformer LLMs (Llama, Mistral, PaLM, Gemma) precisely because they improve loss at near-zero additional FLOPs when hidden dim is scaled to 2/3 of original to keep parameter count equal. The mechanism is particularly relevant for physics surrogates: the gate can learn to selectively activate features relevant to the current flow regime (high-Re vs. low-Re, surface vs. volume, tandem vs. single), providing implicit conditional computation without explicit routing.

The current `MLP` class in `train.py` uses `nn.GELU` throughout. Adding a `"geglu"` activation key and switching `act="geglu"` in `model_config` is a minimal, surgical change.

### Mechanism

The gate `GELU(xV)` learns which linear combinations of hidden features should propagate. For a physics surrogate with wildly varying magnitudes across Re regimes (per-sample y std varies by 10x within splits), a learned gate provides implicit scale-adaptive routing without explicit normalization changes.

### Per-epoch wall-clock prediction

GeGLU adds one extra linear projection per MLP block but hidden dim can be reduced to 2/3 to match FLOPs. At mlp_ratio=4: hidden_dim=128, MLP inner dim=512. With GeGLU at 2/3: inner dim=341 for gate + 341 for value ≈ equivalent FLOPs. Expected per-epoch time: ~175-185 s/epoch (minimal change). This is a quality-per-step improvement, not throughput.

### Exact code change

```python
# Add GeGLU module before or after the ACTIVATION dict:
class GeGLU(nn.Module):
    """Gated Linear Unit with GELU gate. Input dim → output dim (no hidden parameter)."""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

# In MLP.__init__, handle geglu:
# Change the linear_pre to output 2*n_hidden when act="geglu":
if act == "geglu":
    self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden * 2), GeGLU())
else:
    act_fn = ACTIVATION[act]
    self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())

# In model_config, add act="geglu":
model_config = dict(
    space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
    n_hidden=128, n_layers=6, n_head=4, slice_num=64, mlp_ratio=4,
    output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    act="geglu",
)
# Note: pass act through TransolverBlock → MLP. Verify act kwarg propagates in constructor chain.
# Scale inner dim: change mlp_ratio to 3 with geglu (128*3=384 for gate + 384 for value ≈ 128*4=512 FLOPs-equivalent)
```

Alternative minimal approach — add to ACTIVATION dict with a wrapper that handles the 2x projection internally, and adjust n_hidden passed to MLP:

```python
# Simpler approach: treat mlp_ratio=3, act="geglu" as the experiment
# MLP inner dim: 128*3=384, GeGLU splits → 192 effective, linear_post: 192→128
# vs baseline: mlp_ratio=4, GELU, inner=512
```

### CLI invocation

```bash
python train.py \
  --experiment_name geglu_activation \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

### Falsifying result

If val_avg/mae_surf_p > 107 (>5% worse), the gating overhead or reduced effective hidden dimension cancels any quality benefit. Given n_hidden=128 is already at the width sweet spot, compressing the MLP's inner dim may hurt more than the gate helps.

---

## 4. RMSNorm Replacing LayerNorm

### What it is

Replace the three `nn.LayerNorm` instances in each `TransolverBlock` with `nn.RMSNorm`. RMSNorm skips the mean-centering step of LayerNorm: `RMSNorm(x) = x / RMS(x) * γ` where `RMS(x) = sqrt(mean(x²) + ε)`. Half as many statistics to compute per normalization step.

### Why it might help here

LayerNorm requires computing both mean and variance. RMSNorm skips the mean subtraction, yielding 7-64% wall-clock speedup on normalization operations (measured speedup varies by hardware and sequence length). With 6 TransolverBlocks each having 3 LayerNorm calls = 18 LayerNorm operations per forward pass, plus backward, this is a non-trivial fraction of wall-clock time. The quality argument: T5, Llama, Mistral, and essentially all modern LLMs use RMSNorm over LayerNorm with no observed quality regression and often slight improvements (less gradient noise from centering when inputs are already near-zero mean from Adam updates).

PyTorch 2.4+ ships `nn.RMSNorm` natively — no new package needed.

**`_init_weights` must be updated**: current code checks `isinstance(m, (nn.LayerNorm, nn.BatchNorm1d))` and initializes bias=0, weight=1. RMSNorm has no bias (that's the point), only a weight (γ). Add `nn.RMSNorm` to the isinstance check and only set `m.weight = 1.0` (no bias attribute exists).

### Mechanism

Fewer FLOPs per forward pass → faster iteration → more epochs at same wall-clock. The quality argument (removing centering step) is secondary but does remove one source of representation collapse in deep networks.

### Per-epoch wall-clock prediction

LayerNorm is typically 5-15% of total transformer wall-clock time. Expected epoch time reduction: ~10-20 s/epoch → ~155-165 s/epoch. At 30 min: ~11-12 epochs → ~11-13 epochs (marginal throughput gain). The bigger argument is quality-per-step parity with modern norms.

### Exact code change

```python
# In TransolverBlock.__init__, replace:
#   self.ln_1 = nn.LayerNorm(hidden_dim)
#   self.ln_2 = nn.LayerNorm(hidden_dim)
#   if self.last_layer:
#       self.ln_3 = nn.LayerNorm(hidden_dim)
# with:
    self.ln_1 = nn.RMSNorm(hidden_dim)
    self.ln_2 = nn.RMSNorm(hidden_dim)
    if self.last_layer:
        self.ln_3 = nn.RMSNorm(hidden_dim)

# In Transolver._init_weights, update isinstance check:
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.RMSNorm)):
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
```

No CLI change needed — this is purely a `train.py` architectural change.

### CLI invocation

```bash
python train.py \
  --experiment_name rmsnorm_replace_layernorm \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

### Falsifying result

If per-epoch time does not drop below 160 s (no meaningful speedup), LayerNorm was not a bottleneck. If val_avg/mae_surf_p > 107 (>5% worse), the centering step in LayerNorm was providing useful normalization for this specific physics distribution.

---

## 5. SWA Late-Start (Stochastic Weight Averaging from Epoch 8)

### What it is

Apply Stochastic Weight Averaging starting from epoch 8 (post-convergence), not from epoch 0. Use `torch.optim.swa_utils.AveragedModel` and `SWALR`. The SWA model averages recent optimizer iterates to find a flatter loss minimum with better generalization.

### Why it might help here

Prior EMA experiment (decay=0.999, starting from random init) was +41% worse — but that failure was cold-start drag: the exponential moving average was heavily polluted by poor early-epoch weights. The key difference with late-start SWA: averaging only begins after the model has converged to a good local region (epoch 8 of ~12). The academic SWA literature (Izmailov et al. 2018) consistently shows SWA finds flatter minima with better generalization precisely because it explores the loss basin around the converged solution rather than tracking the volatile early trajectory. This is a mechanism-aware fix of the EMA cold-start failure.

Flatter minima are particularly important here given the extreme distribution shift across val splits (in-dist vs. OOD camber vs. OOD Re). Flat minima generalize better across distribution shifts (Keskar et al. 2017, Foret et al. 2021 SAM).

### Mechanism

SWA averages weights over the final few epochs of training, moving from the sharp converged point toward a wider basin. The `val_single_in_dist` split should maintain quality (flat → still good in-dist), while `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand` may improve (flatter basin → better OOD).

### Per-epoch wall-clock prediction

SWA updates (model parameter averaging) add ~1-2% overhead per epoch and only start at epoch 8. No meaningful impact on throughput. `swa_model.update_parameters(model)` is a lightweight CPU operation.

### Exact code change

```python
# After model creation and optimizer setup, add:
from torch.optim.swa_utils import AveragedModel, SWALR

SWA_START_EPOCH = 8  # start averaging after model has converged
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=3)

# In the training loop, after optimizer.step():
if epoch >= SWA_START_EPOCH:
    swa_model.update_parameters(model)
    swa_scheduler.step()  # replaces main scheduler step when active

# After training loop, update BN stats (no BatchNorm here, skip or call to be safe):
# torch.optim.swa_utils.update_bn(train_loader, swa_model)  # skip if no BN

# For validation after SWA_START_EPOCH, eval swa_model instead of model:
eval_model = swa_model if epoch >= SWA_START_EPOCH else model
# (use eval_model in validation loop)

# Checkpoint the swa_model's underlying module:
# swa_model.module is the averaged parameters model
```

Note: `swa_model.module` contains the averaged weights. Checkpoint `swa_model.module.state_dict()` for the final best checkpoint.

### CLI invocation

```bash
python train.py \
  --experiment_name swa_late_start_epoch8 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

### Falsifying result

If val_avg/mae_surf_p > 107 (>5% worse), the loss basin at epoch 8 is not wide enough for averaging to help — the model converges to a sharp minimum regardless. Check per-split diagnostics: if `val_geom_camber_*` and `val_re_rand` improve but `val_single_in_dist` degrades by more, SWA is trading in-dist for OOD generalization (possibly still a net positive if the average improves).

---

## 6. surf_weight=5 (Lower Surface Emphasis)

### What it is

Change `surf_weight` from 10 to 5. The primary metric is surface pressure MAE. With L1 loss (constant-magnitude gradients), the model allocates gradient signal proportional to the loss weighting. At surf_weight=10, surface nodes receive 10x the per-node gradient signal as volume nodes.

### Why it might help here

The dead-end list includes `surf_weight=25` (channel-weighted loss, counterproductive). But surf_weight=5 has NOT been tested on the current L1 baseline. L1 loss already emphasizes absolute errors, and the volume field may be providing essential geometric context for the surface predictions. Reducing surf_weight from 10 to 5 reallocates gradient budget toward volume accuracy, potentially improving the model's representation of the full flow field and indirectly benefiting surface predictions through better features. This is a 1-line change with no wall-clock cost.

### Mechanism

The model optimizes a weighted combination: `loss = vol_loss + surf_weight * surf_loss`. Lower surf_weight → volume gets relatively more gradient → better volume field representation → better geometric and flow context → potentially improved surface prediction at test time via better intermediate representations.

### Per-epoch wall-clock prediction

Identical to baseline: ~175 s/epoch. No architectural or compute change.

### Exact code change

```bash
# No code change needed — pure CLI argument:
--surf_weight 5
```

Or equivalently in `Config` dataclass default:
```python
surf_weight: float = 5.0  # was 10.0
```

### CLI invocation

```bash
python train.py \
  --experiment_name surf_weight_5 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 5.0 \
  --batch_size 4
```

### Falsifying result

If val_avg/mae_surf_p > 107 (>5% worse), surface nodes genuinely require 10x gradient emphasis with L1. If MAE_surf_p improves but mae_vol_* worsens, the trade-off is favorable for the primary metric but at cost of physical correctness.

---

## 7. Physical-Space L1 (Denormalized Loss)

### What it is

Compute L1 loss in denormalized (physical) units rather than in normalized space. Instead of `|pred_norm - y_norm|`, compute `|pred_phys - y_phys|` where `pred_phys = pred * y_std + y_mean`. Since the primary metric is MAE in physical units, training in physical units creates direct alignment between training objective and evaluation metric.

### Why it might help here

Current training loss: normalized-space L1. The per-channel normalization (`y_std` ≈ [std_Ux, std_Uy, std_p]) equalizes gradient magnitudes across channels. But `val_avg/mae_surf_p` measures only the pressure channel in physical units. Physical-space L1 allows the model to prioritize high-magnitude samples (high-Re flows with large pressure values) naturally, since those dominate the physical-space loss. This may better align gradient signal with what matters for the paper-facing metric.

Risk: physical-space loss may be dominated by extreme high-Re outliers (per-sample y std varies by 10x within splits). This could destabilize training if pressure magnitudes at Re=5M swamp low-Re contributions.

### Mechanism

Direct objective-metric alignment. Training objective equals (or closely approximates) the evaluation metric in units and scale. The channel-weighting that normalization provides is replaced by physical-scale weighting, which naturally emphasizes the dominant physical processes.

### Per-epoch wall-clock prediction

One additional denormalization operation per batch (tensor multiply + add). Completely negligible: ~175 s/epoch unchanged.

### Exact code change

```python
# In the training step, replace normalized-space loss with physical-space loss:
# Current:
#   y_norm = (y - stats["y_mean"]) / stats["y_std"]
#   pred = model({"x": x_norm})["preds"]
#   abs_err = (pred - y_norm).abs()

# Replace with:
    pred_norm = model({"x": x_norm})["preds"]
    pred_phys = pred_norm * stats["y_std"] + stats["y_mean"]
    abs_err = (pred_phys - y).abs()  # y is in physical units (pre-normalization)

# Keep vol_loss, surf_loss, loss computation identical (just using physical-space abs_err)
# Note: stats["y_std"] and stats["y_mean"] are already loaded as tensors/arrays in the
# training loop. Ensure they are on the correct device (move to GPU with .to(device) if needed).
```

### CLI invocation

```bash
python train.py \
  --experiment_name physical_space_l1_loss \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --surf_weight 10.0 \
  --batch_size 4
```

### Falsifying result

If val_avg/mae_surf_p > 115 (>13% worse), high-Re pressure outliers dominate the physical-space loss and destabilize gradient flow. Monitor training loss for divergence or NaN in epoch 1-2. If the training loss drops faster than normalized-space loss but validation MAE stagnates or worsens, the physical-space alignment is overfitting to high-magnitude samples at the cost of low-Re regimes.

---

## Compound Experiments (if individual wins emerge)

If bf16 succeeds (hypothesis 1), immediately compound with:
- bf16 + GeGLU (more epochs + better per-epoch quality)
- bf16 + RMSNorm (more epochs + faster normalization)
- bf16 + SWA late-start (more epochs + weight averaging in final phase)

If Lion succeeds (hypothesis 2), compound with:
- Lion + bf16 (sign updates + throughput)

---

## Experiment Decision Tree

```
Start
├── bf16 mixed precision (#1)
│   ├── SUCCESS (< 101.810): compound with GeGLU or RMSNorm
│   │   ├── bf16 + GeGLU: if both win, compound further
│   │   └── bf16 + RMSNorm: if both win, compound further
│   └── FAIL (> 107): bf16 has overflow or memory-bandwidth bottleneck
│       → Rule out bf16. Fall back to quality-per-step experiments.
│       → Try Lion (#2), GeGLU (#3), RMSNorm (#4) independently
│
├── Lion optimizer (#2) [parallel with bf16]
│   ├── SUCCESS: compound with winning lr (scan 5e-5, 1e-4, 2e-4)
│   └── FAIL: sign-based updates don't suit this landscape. Stop.
│
├── GeGLU activation (#3) [parallel]
│   ├── SUCCESS: compound with bf16 if bf16 also succeeded
│   └── FAIL: MLP gating not beneficial. Try surf_weight=5 (#6) next.
│
├── RMSNorm (#4) [parallel]
│   ├── SUCCESS: compound with any winning throughput experiment
│   └── FAIL: LayerNorm centering was useful. Do not retry.
│
├── SWA late-start (#5) [parallel]
│   ├── SUCCESS: check per-split diagnostics (OOD splits should improve)
│   │   → If in-dist degrades but OOD improves: still merge if avg improves
│   └── FAIL: converged minimum is too sharp for averaging. Close.
│
├── surf_weight=5 (#6) [cheapest — run first in parallel]
│   ├── SUCCESS: compound with winning architecture
│   └── FAIL: surf_weight=10 is correct for L1. Try surf_weight=15.
│
└── Physical-space L1 (#7)
    ├── SUCCESS: direct objective alignment works. Compound with bf16.
    └── FAIL (diverge or NaN): scale physical-space loss by 1/y_std² before summing
        (i.e., implement per-sample normalized physical loss as a hybrid)
```

---

## Summary Table

| Rank | Hypothesis | Mechanism | Expected Epochs (30 min) | Wall-clock Impact | Code Complexity |
|------|------------|-----------|--------------------------|-------------------|-----------------|
| 1 | bf16 Mixed Precision | 2x throughput → more gradient steps | 18-22 (vs 12) | -50% s/epoch | Medium (targeted float32 cast at line 120) |
| 2 | Lion Optimizer | Sign updates pair with L1 gradients | 12-13 | -3-5% s/epoch | Low (1 import, 1 line, pyproject.toml) |
| 3 | GeGLU Activation | Gated feature selection for flow regime | 12-13 | Neutral | Medium (new module class + propagation) |
| 4 | RMSNorm | Fewer norm FLOPs, modern standard | 13-14 | -5-10% s/epoch | Low (3 line replacements + _init_weights) |
| 5 | SWA Late-Start | Flat loss basin, better OOD generalization | 12 | +1-2% s/epoch | Medium (averaging loop after epoch 8) |
| 6 | surf_weight=5 | Rebalance gradient to volume field | 12 | Neutral | Trivial (1 CLI flag) |
| 7 | Physical-space L1 | Direct objective-metric alignment | 12 | Neutral | Low (2 line change in training step) |
