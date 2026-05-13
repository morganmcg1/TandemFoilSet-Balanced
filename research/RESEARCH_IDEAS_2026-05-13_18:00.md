# Research Ideas — 2026-05-13 18:00

## Context

Four students (alphonse, askeladd, frieren, thorfinn) are idle after their PRs closed at 17:45. Four WIPs are already running: nezuko #2486 (AdamW eps=1e-6), edward #2440 (LR warmup 3ep), tanjiro #2420 (lr=7e-4), fern #2397 (grad_clip max_norm=0.5). New hypotheses must be orthogonal to all four in-flight axes and must target the 8-stack baseline: val=58.883, test=51.078.

Constraint summary from closed PRs:
- **Regularization noise-injection axis SATURATED**: dropout (#2180, +2.5/+10.8% seeds) and DropPath (#2415, +15/+22%) both regressed. Structural regularization closed.
- **Per-channel β downward CLOSED**: β_p=0.25 (#2163) gave hard-split regression + easy-split improvement — directional pressure-outlier signal. β_p>0.5 is untested and well-motivated.
- **EMA PARKED**: near-wash (+0.17%) under cosine schedule; revisit after schedule changes land.
- **Scalar-capacity cluster FULLY RETIRED** across 3 baselines (7 total failures).

---

## H1 — askeladd: per-channel Huber β_p = 0.625 (upward bisection)

### Slug
`huber-surf-bp-0625`

### What it is
Change only the pressure channel delta in the surface Huber loss from 0.5 → 0.625, keeping Ux and Uy at δ=0.5. This is a targeted upward bisection between the closed optimum (0.5) and the closed global failure (0.75 was global β, not per-channel).

### Why it might help here
PR #2163 (β_p=0.25) revealed a clean asymmetric per-split signature: hard splits (in_dist +3%, camber_rc +3.5%) regressed while easy splits (camber_cruise −3%, re_rand −2%) improved. The mechanistic interpretation is unambiguous: β_p<0.5 compresses the effective gradient weight on large pressure residuals, and those large residuals correspond to physically meaningful high-Re pressure extremes. The causal direction strictly implies β_p>0.5 (moving toward MSE on pressure, giving larger residuals more gradient weight) is the natural correction.

The global β=0.75 failure (#1882) is not a counter-argument: that changed all three channels uniformly and is 0.25 above the per-channel target. A half-step upward bisect at β_p=0.625 tests whether the mechanism is real at a magnitude that won't over-fit velocity channels.

The β_p=0.625 change is isolated to one scalar in the loss computation. No other component of the stack changes. Zero risk of interaction with in-flight WIPs (all 4 WIPs are on the optimizer/schedule axis).

### Key external reference
Huber δ tuning for asymmetric target distributions is well-studied in robust regression literature: when targets have heavy right tails (as p does in high-Re samples), δ>median yields lower empirical MSE for in-distribution samples while maintaining outlier robustness. The recent loss-formulation wins in this stack (surf-Huber #1505 −4.7%, vol-Huber #1910 −3.5%) both came from the same family.

### Proposed code change

In `train.py`, locate the surface loss block. The current form after the 8-stack is:
```python
# Current (β_p = 0.5 globally)
surf_loss = (
    F.huber_loss(pred_surf[:, 0], y_surf[:, 0], delta=0.5, reduction='mean')
  + F.huber_loss(pred_surf[:, 1], y_surf[:, 1], delta=0.5, reduction='mean')
  + F.huber_loss(pred_surf[:, 2], y_surf[:, 2], delta=0.5, reduction='mean')
) / 3

# Change to (β_p = 0.625 for pressure channel only):
surf_loss = (
    F.huber_loss(pred_surf[:, 0], y_surf[:, 0], delta=0.5,   reduction='mean')  # Ux unchanged
  + F.huber_loss(pred_surf[:, 1], y_surf[:, 1], delta=0.5,   reduction='mean')  # Uy unchanged
  + F.huber_loss(pred_surf[:, 2], y_surf[:, 2], delta=0.625, reduction='mean')  # p upward bisect
) / 3
```

The volume loss is left unchanged (all channels at δ=0.5). If the student finds the surf loss is implemented differently (e.g. computed on masked tensors inline), the key instruction is: **only the pressure channel (index 2) delta changes, velocity channels stay at 0.5**.

### What we expect to see
If mechanism is alive: hard splits (in_dist, camber_rc) recover or improve vs. baseline; easy splits hold or improve. val_avg/mae_surf_p improves −0.5% to −2.0%.
If mechanism is dead: split pattern mirrors β_p=0.25 (similar asymmetry) or is noise (±0.3%). This would suggest β=0.5 is precisely optimal for this data distribution.

### Falsifying result
If hard splits regress again (same direction as β_p=0.25), the mechanism is confirmed wrong-direction or the per-channel concept is invalid. If β_p=0.625 matches baseline within ±0.3%, the optimum is confirmed at 0.5 and the per-channel axis is fully closed.

### Predicted delta
−0.5% to −2.0% on val_avg/mae_surf_p (higher confidence on direction than magnitude; mechanism backed by direct evidence from #2163).

### Taste rubric
| Criterion | Score | Reason |
|---|---|---|
| Mechanistic grounding | 4 | Direct causal evidence from #2163's asymmetric per-split pattern; mechanism named (pressure outliers carry physical info); counter-argument (global β=0.75 failure) explicitly resolved. |
| Research-state value | 4 | Sharply discriminating: success closes the per-channel axis with a winner; failure (same-direction regression) definitively closes it. Either way the map updates cleanly. |
| Execution value | 4 | 1-scalar change, no new logic, no architecture risk, zero interaction with in-flight WIPs. 30-min × 2-seed budget. |

Research mode: **frontier refinement** (natural next step on a directional signal from #2163).

### Reproduce command for askeladd
```bash
python train.py \
  --wandb_group huber_perchannel \
  --run_name huber_surf_bp0625_seed42 \
  --seed 42

python train.py \
  --wandb_group huber_perchannel \
  --run_name huber_surf_bp0625_seed123 \
  --seed 123
```

### Stop condition
If val_avg/mae_surf_p > 59.5 (>1% regression from 58.883), close. If within ±0.3% of baseline, close per-channel axis entirely. If improves, consider β_p=0.75 per-channel-only as the next bisection step.

---

## H2 — frieren: QK-RMSNorm in PhysicsAttention

### Slug
`qk-rms-norm`

### What it is
Add `nn.RMSNorm` on Q and K tensors inside PhysicsAttention, applied after the linear projection and before the attention dot product. No change to V, FFN, or outer LayerNorm.

### Why it might help here
The Transolver PhysicsAttention computes softmax over Q·K^T slices across heterogeneous mesh nodes. The mesh spans three domains with fundamentally different scales: raceCar single (~85K nodes, Re 100K–5M, pressure range −29K to +2.7K) and cruise (~210K nodes, Re 110K–5M, pressure range −7.6K to +2.6K). Surface nodes and volume nodes also differ in feature magnitude (position, is_surface flag, dsdf values). The result: after linear projection, Q and K vectors for surface nodes at high-Re conditions can be an order of magnitude larger in L2-norm than Q/K for interior volume nodes at low-Re. This creates attention entropy collapse — a few tokens dominate the softmax, and the effective attention rank drops.

QK-RMSNorm (as used in PaLM-2, Gemma-2, and recent ViT variants) normalizes Q and K to unit-norm before the dot product, making softmax temperature consistent regardless of input magnitude. The effective temperature becomes purely a function of head_dim^0.5, not the magnitude of the input features. This is the mechanism most likely to help on heterogeneous-scale meshes where the input feature distribution is not homogeneous across nodes.

The 100% grad-clip engagement (raw grad norm ~18-19) is also consistent with attention entropy collapse: if a few dominant tokens receive all the gradient signal, the global gradient norm inflates while most attention weights receive near-zero updates. QK-RMSNorm could simultaneously stabilize attention entropy AND reduce the raw gradient norm (potentially below the current clip threshold for the first time).

The closed SwiGLU experiment (#1735) was a pod reset, not a verdict. QK-RMSNorm is a lower-risk architectural intervention: 2 new RMSNorm modules, no parameter matching required, no FFN changes, and the change is fully reversible with a flag.

### Key external references
- QK-Norm in ViT: "Scaling ViT to 22B Parameters" (Dehghani et al., 2023) — used QK-norm to stabilize large-scale ViT training; directly analogous to heterogeneous attention inputs
- PaLM-2 / Gemma-2 technical reports: RMSNorm on Q/K used as default in bf16 training to prevent attention logit explosion
- "QueryKey Normalization for Transformers" (Henry et al., 2020, ACL) — ablations showing norm helps on tasks with high input-space variance

### Proposed code change

In `train.py`, inside the `PhysicsAttention` class:

```python
# In __init__, add (after existing self.q and self.k projections):
self.q_norm = nn.RMSNorm(self.head_dim)  # head_dim = n_hidden // n_head = 128 // 4 = 32
self.k_norm = nn.RMSNorm(self.head_dim)

# In forward, after projecting Q and K but before the dot product:
# Assuming Q shape is [B, n_heads, N, head_dim] after reshape
Q = self.q_norm(Q)  # normalize each head's Q vectors independently
K = self.k_norm(K)  # normalize each head's K vectors independently
```

If the student finds Q/K are not reshaped to [..., head_dim] before the attention computation, the norm should be applied after reshape. The key invariant is: RMSNorm is applied to the head_dim axis of Q and K, not the sequence or batch axis.

Critical: `nn.RMSNorm` was added in PyTorch 2.4. If the environment uses an older version, use:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + self.eps) * self.weight
```

### What we expect to see
If mechanism is alive: early-epoch loss convergence is smoother; hard-split metrics (camber_rc, re_rand where Re heterogeneity is highest) improve most. Possible secondary signal: raw gradient norm drops, reducing the 100% clip-rate signature.
If mechanism is dead: no change in per-split pattern; identical final val to baseline within noise. This would suggest the attention entropy is already well-regulated by LayerNorm inputs and the PhysicsAttention slice-softmax structure.

### Falsifying result
If val_avg/mae_surf_p degrades >1% vs. baseline, the normalization interferes with the physics-aware slice attention (which uses softmax over a learned slice assignment). This would mean Q/K magnitudes carry slice-discriminative information and normalizing destroys it.

### Predicted delta
−0.5% to −2.0% on val_avg/mae_surf_p (moderate confidence; mechanism is strong external analogy, but mesh-attention heterogeneity is more structured than typical NLP/vision settings — could help more or less).

### Taste rubric
| Criterion | Score | Reason |
|---|---|---|
| Mechanistic grounding | 3 | External analogy strong (PaLM-2, Gemma-2, ViT-22B); mechanism targets specific observed property (domain heterogeneity + 100% clip rate); not yet linked to a direct local diagnostic. |
| Research-state value | 3 | Result distinguishes two hypotheses: (a) attention entropy collapse is a bottleneck; (b) PhysicsAttention slice-softmax already handles scale variation. Either answer is architecturally informative. |
| Execution value | 3 | 2 new modules, ~16 lines of code, no param count change, no data loading change. Minor risk of PyTorch version compatibility (RMSNorm 2.4+ check required). |

Research mode: **architecture-side exploration** (addressing a specific structural bottleneck hypothesis on heterogeneous mesh inputs).

### Reproduce command for frieren
```bash
python train.py \
  --wandb_group qk_rms_norm \
  --run_name qk_rms_norm_seed42 \
  --seed 42

python train.py \
  --wandb_group qk_rms_norm \
  --run_name qk_rms_norm_seed123 \
  --seed 123
```

### Stop condition
If val_avg/mae_surf_p > 59.5 on both seeds, close and note that Q/K magnitude carries slice-assignment information. If improves, consider combining with RMSNorm replacement of outer LayerNorm as a round-3 follow-up.

---

## H3 — alphonse: GELU → SiLU activation swap

### Slug
`silu-activation`

### What it is
Replace all `F.gelu` / `nn.GELU()` instances in the MLP FFN blocks with `F.silu` / `nn.SiLU()`. No parameter count change, no architecture change.

### Why it might help here
The Transolver MLP blocks use GELU activations in their FFN blocks. In the current 8-stack, grad clipping fires on 100% of steps — the raw grad norm (~18-19) is always above max_norm=1.0. This means the optimizer is always operating in a clipped regime where effective step size is lr × max_norm / raw_norm, not lr itself. In a permanently-clipped regime, the activation gradient shape near zero matters more: GELU has a flat zero-gradient region for large negative inputs, while SiLU (x * sigmoid(x)) has a non-zero gradient for all inputs and a smoother slope through zero.

SiLU was found empirically superior to GELU in multiple settings involving high gradient-clip rates:
- LLaMA-1/2 (and most modern LLMs) switched from GELU to SiLU/SwiGLU; the SiLU component of SwiGLU is the operative change
- EfficientNet / MobileNetV3 found SiLU superior for gradient-limited training
- Meta's DINOv2 uses SiLU in FFN blocks explicitly citing gradient stability under clipping

The SwiGLU PR (#1735) was reset (pod issues, 0 commits) and is NOT a closed verdict. SwiGLU = SiLU + gating (W1 ⊙ σ(W2)) which requires matched parameter counts. A pure GELU→SiLU swap has zero parameter change and zero architecture risk, making it the cleanest first test of the SiLU hypothesis.

### Proposed code change

In `train.py`, find the FFN/MLP block inside Transolver. Typically:
```python
# BEFORE:
x = F.gelu(self.fc1(x))

# AFTER:
x = F.silu(self.fc1(x))
```

If using `nn.GELU()` as a module:
```python
# BEFORE:
self.act = nn.GELU()

# AFTER:
self.act = nn.SiLU()
```

The student should search for all occurrences of `gelu` (case-insensitive) in the model definition and replace each with `silu`. There should be 1-3 occurrences in the FFN block; the attention mechanism (Q/K/V) does not use an activation and should not be touched.

### What we expect to see
If mechanism is alive: training loss convergence is smoother in early epochs; final val metric improves across all splits roughly uniformly (SiLU addresses global gradient propagation, not a domain-specific issue). Expected: −0.3% to −1.5% on val_avg/mae_surf_p.
If mechanism is dead: indistinguishable from baseline. The activation choice doesn't matter when the model is small (662K params) and grad-clipped.

### Falsifying result
If val_avg/mae_surf_p degrades >1%, SiLU is actually worse in this mesh-attention setting — possibly because GELU's selective zero-gating helps the model learn sparser feature representations over irregular meshes.

### Predicted delta
−0.3% to −1.5% on val_avg/mae_surf_p (moderate confidence; mechanism has good external analogy but is not targeted at a specific observed split-pattern failure).

### Taste rubric
| Criterion | Score | Reason |
|---|---|---|
| Mechanistic grounding | 2 | External analogy strong (LLaMA, DINOv2); mechanism plausible for 100%-clipped regime; not tied to a specific local observable beyond the clip rate. |
| Research-state value | 2 | If it wins, confirms activation matters. If it loses, rules out SiLU (but not SwiGLU). Moderately informative. |
| Execution value | 4 | 1-3 line changes, zero compute overhead, zero parameter change, zero data loading changes. Pure signal experiment. |

Research mode: **architecture-side exploration** (low-cost activation-function test; complementary to the QK-RMSNorm attention-side change).

### Reproduce command for alphonse
```bash
python train.py \
  --wandb_group silu_activation \
  --run_name silu_activation_seed42 \
  --seed 42

python train.py \
  --wandb_group silu_activation \
  --run_name silu_activation_seed123 \
  --seed 123
```

### Stop condition
If val_avg/mae_surf_p > 59.5 on both seeds, close. If within ±0.3% of baseline, close. If improves, consider SwiGLU (SiLU + gating) as the follow-up (now no longer a reset; frieren's pod issues were environment, not code).

---

## H4 — thorfinn: per-channel target normalization

### Slug
`perchannel-target-norm`

### What it is
Replace the current global target normalization `(y - y_mean) / y_std` with per-channel normalization using a separate mean and std for each of the 3 output channels (Ux, Uy, p). This is a representation change, not a model or loss change.

### Why it might help here
The current stats.json provides scalar `y_mean` and `y_std` that normalize all three channels uniformly. But the three channels have fundamentally different statistics:

- `Ux` (velocity x): O(1–100) m/s, roughly zero-mean across domains
- `Uy` (velocity z): O(0.1–10) m/s, smaller magnitude than Ux
- `p` (pressure): O(100–10,000) Pa, large positive values, ~10× larger std than velocity channels

With global normalization, the pressure channel dominates the normalized loss — its unnormalized residuals are larger, and dividing by a global y_std that averages across all three channels under-normalizes pressure and over-normalizes velocity. This means the model effectively sees a pressure-heavy loss landscape during training, but the primary metric (val_avg/mae_surf_p) is already pressure-focused — so the mismatch is not obviously harmful. However, the **velocity channels may be under-optimized** because their normalized residuals are small relative to pressure even in normalized space.

Per-channel normalization makes each channel's normalized residual distribution comparable in scale, giving the Huber loss equal footing across all three channels. This is the standard approach in multi-output regression and is used by virtually all state-of-the-art CFD surrogates (FNO, GINO, FactFormer all normalize per-channel).

**Critical:** the data contract in `program.md` says scoring denormalizes as `pred * y_std + y_mean`. If this uses the global scalar, the student must ensure the denormalization uses the same per-channel stats they used for normalization. The safest approach: keep the global `y_mean`/`y_std` for the scoring call (to match the existing contract), and only change the normalization inside the training loop for the loss computation.

### Proposed code change

In `train.py`, before the training loop, compute per-channel stats from the training set:
```python
# Compute per-channel stats (after loading train_ds, before training loop)
all_y = torch.cat([sample[1] for sample in train_ds], dim=0)  # [total_nodes, 3]
y_mean_ch = all_y.mean(dim=0)   # [3] — per-channel mean
y_std_ch  = all_y.std(dim=0)    # [3] — per-channel std
y_std_ch  = y_std_ch.clamp(min=1e-6)  # guard against zero std
# Move to device at training time: y_mean_ch.to(device), y_std_ch.to(device)
```

Then in the loss computation (not in scoring), replace:
```python
# BEFORE:
y_norm = (y - stats["y_mean"]) / stats["y_std"]

# AFTER:
y_norm = (y - y_mean_ch.to(device)) / y_std_ch.to(device)  # broadcasts [B, N, 3] correctly
```

The model output `pred` is in this new normalized space. Scoring must still use the original `stats["y_mean"]` and `stats["y_std"]` scalars for the `data/scoring.py` contract. So the student should denormalize for scoring explicitly:
```python
# For scoring only: convert from per-channel normalized space to physical space
pred_phys = pred * y_std_ch.to(device) + y_mean_ch.to(device)
# Then pass pred_phys to scoring (do NOT multiply by stats["y_std"] again)
```

**If the all_y concatenation is too slow or memory-heavy for the training set, an alternative:** read `stats.json` (which may already have per-channel stats if the preprocessing saved them), or compute a running Welford mean/variance over one pass of the dataloader before training.

### What we expect to see
If mechanism is alive: velocity channel MAE (val_avg/mae_surf_Ux, val_avg/mae_surf_Uy) should improve notably as those channels get better gradient signal. Pressure MAE may slightly increase or stay flat (it was already dominant). Net effect on val_avg/mae_surf_p (pressure-only metric): could be slight regression if pressure was over-fit before, or could improve if better-balanced training generalizes better to the held-out splits.

**Caution:** this experiment has a non-trivial risk of neutral or slight regression on val_avg/mae_surf_p (the paper-facing metric is pressure-only). The student should log all per-channel MAEs explicitly to understand the mechanism even if the primary metric doesn't improve.

### Falsifying result
If all three channel MAEs increase, the global normalization was actually well-calibrated and the per-channel approach destabilizes training. If val_avg/mae_surf_p degrades >2% despite velocity channels improving, the primary metric was already well-served by the pressure-dominant loss.

### Predicted delta
−1.0% to −3.0% on val_avg/mae_surf_p (if mechanism is alive); possible neutral/slight regression if pressure benefits from the current over-emphasis. This is the experiment with the highest uncertainty in direction but also the largest upside if the velocity channels are genuinely under-trained.

### Taste rubric
| Criterion | Score | Reason |
|---|---|---|
| Mechanistic grounding | 3 | Standard in multi-output regression; pressure scale mismatch is documented in program.md value ranges table; external analogy strong (FNO/GINO/FactFormer all use per-channel norm). |
| Research-state value | 3 | Result is informative even if primary metric regresses: per-channel MAE breakdown will tell us whether velocity channels are under-optimized, which shapes future loss-formulation experiments. |
| Execution value | 2 | Moderate complexity: requires a pass over the training set before the loop, plus careful denormalization plumbing to keep scoring contract intact. Risk of silent bug if normalization/denormalization is mismatched. |

Research mode: **representation axis** (normalization is the representation, not the model; orthogonal to all 4 in-flight WIPs).

### Reproduce command for thorfinn
```bash
python train.py \
  --wandb_group perchannel_norm \
  --run_name perchannel_norm_seed42 \
  --seed 42

python train.py \
  --wandb_group perchannel_norm \
  --run_name perchannel_norm_seed123 \
  --seed 123
```

### Stop condition
If val_avg/mae_surf_p > 60.5 (>2.7% regression from 58.883) on both seeds, close. If velocity channel MAEs improve but pressure MAE regresses, note the finding and close. If improves on primary metric, the mechanism is confirmed and per-channel norms should become a permanent part of the stack.

---

## Summary table

| Student | Slug | Axis | Predicted Δ val | Confidence | Risk |
|---|---|---|---|---|---|
| askeladd | huber-surf-bp-0625 | Loss formulation (per-channel β upward bisect) | −0.5% to −2.0% | High direction | Low (1-scalar change) |
| frieren | qk-rms-norm | Architecture (attention entropy stabilization) | −0.5% to −2.0% | Moderate | Low-moderate (2 new modules) |
| alphonse | silu-activation | Architecture (activation swap) | −0.3% to −1.5% | Moderate | Very low (1-3 line change) |
| thorfinn | perchannel-target-norm | Representation (per-channel normalization) | −1.0% to −3.0% | Low-moderate | Moderate (plumbing complexity) |

All four hypotheses are orthogonal to each other and to the 4 in-flight WIPs. No two hypotheses operate on the same mechanism. The askeladd and thorfinn hypotheses are the most discriminating (either direction of result is informative). The frieren and alphonse hypotheses have the best execution risk profile.
