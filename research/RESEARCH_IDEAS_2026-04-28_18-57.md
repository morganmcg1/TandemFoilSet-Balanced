<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-04-28 18:57

## Context

Current best: PR #32, `val_avg/mae_surf_p = 40.927` (`test_avg/mae_surf_p = 40.927`).
Configuration: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`.
Baseline (PRs #6/#18): ~81.9.

All hypotheses below build on top of the nl3/sn16/n_head=1 base unless otherwise stated.
Architecture params go in the `model_config` dict in `train.py`; optimizer/scheduler/loss
params are CLI flags via the `Config` dataclass.

Ranking order: expected impact × confidence × execution cost.

---

## Hypothesis 1 — GeGLU activation on nl3/sn16/n_head=1

**Axis:** Architecture (activation function)

**Change:**
Replace the default GELU activation in `TransolverBlock.mlp` with GeGLU:
`GeGLU(x) = gelu(W1·x) ⊗ W2·x`. GeGLU splits the hidden dimension into two halves
(gate and content), so set `mlp_ratio=4` to maintain the same effective capacity
(2/3 × 4 = 2.67 × hidden vs 2 × hidden currently).

**Rationale:**
GeGLU (Noam Shazeer, 2020; "GLU Variants Improve Transformers") consistently outperforms
vanilla GELU by 1–3% on language tasks. The gating mechanism acts as a learned feature
selector — directly relevant here because the mesh spans background nodes (mostly near-zero
gradients) and surface nodes (high-gradient pressure). GeGLU was not in the prior activation
sweep, which only tested the activations already in the `ACTIVATION` dict. SwiGLU was tested
but only bundled with Fourier PE (ranks 9–11, ~58–62), never on the clean nl3/sn16/n_head=1
base. GeGLU is the purer test because it replaces one activation (GELU) with a gated variant
of the same base function, isolating the gate effect from any SiLU vs GELU difference.

**Predicted delta:** −2 to −5 on `val_avg/mae_surf_p` (i.e. 36–39 from 40.927).

**Risk:** Medium. GeGLU requires a custom module not in the current `ACTIVATION` dict.
If mlp_ratio is not compensated the capacity regression may hurt. Keep mlp_ratio=4 with
GeGLU to equalize parameter count.

**Reproduce command:**
```
# Add GeGLU class to train.py:
# class GeGLU(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2)
#     def forward(self, x):
#         x, gate = self.proj(x).chunk(2, dim=-1)
#         return x * F.gelu(gate)
#
# Then use it in TransolverBlock.mlp instead of MLP(..., act="gelu")
# model_config update: mlp_ratio=4 (to compensate halved hidden per path)
# No CLI flag changes needed — architecture change only.
python train.py \
  --wandb_group geglu-activation \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — GeGLU targets the known surface/background heterogeneity;
  SwiGLU+Fourier ran on a different base so this is not a repeat.
- Research-state value: 3 — isolates gate mechanism from base config; fail tells us
  gating hurts on this mesh distribution, which is informative.
- Execution value: 3 — cheap (same batch, same epoch count), directly targets primary metric.

---

## Hypothesis 2 — RMSNorm replacing LayerNorm

**Axis:** Architecture (normalization)

**Change:**
Replace all `nn.LayerNorm` instances in `TransolverBlock` (`ln_1`, `ln_2`, `ln_3`) with
`RMSNorm`: `RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ`. No bias term, no mean subtraction.

**Rationale:**
RMSNorm (Zhang & Sennrich, 2019) drops the re-centering step of LayerNorm, saving ~7%
compute and eliminating bias parameters. LLaMA, PaLM, and Mistral all use RMSNorm.
The key benefit for this task: the per-channel variance normalization is retained while
the mean-subtraction is removed. For CFD predictions where the output range spans several
orders of magnitude across Re, mean subtraction may shift the scale representation in
hidden states. On the nl3/sn16/n_head=1 base, any per-token overhead is proportionally
larger because there are fewer blocks — RMSNorm savings compound across the 3 layers.

**Predicted delta:** −1 to −3 on `val_avg/mae_surf_p`.

**Risk:** Low. RMSNorm is a well-understood drop-in. PyTorch ≥2.1 has
`torch.nn.modules.normalization.RMSNorm` natively. Fallback: one-line custom impl.

**Reproduce command:**
```
# Add to train.py above TransolverBlock:
# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-8):
#         super().__init__()
#         self.scale = nn.Parameter(torch.ones(dim))
#         self.eps = eps
#     def forward(self, x):
#         return x / x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** 0.5) * self.scale / (x.shape[-1] ** 0.5 + self.eps)
#         # Correct form: x * rsqrt(mean(x^2) + eps) * scale
#
# Replace ln_1, ln_2, ln_3 in TransolverBlock with RMSNorm(hidden_dim)
python train.py \
  --wandb_group rmsnorm-sweep \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — targets normalization scale mismatch under multi-order-of-magnitude Re range.
- Research-state value: 3 — clean ablation; failure tells us mean subtraction matters for this mesh distribution.
- Execution value: 4 — near-zero cost change, directly on nl3/sn16/n_head=1 base.

---

## Hypothesis 3 — OneCycleLR with linear warmup

**Axis:** Optimization (scheduler)

**Change:**
Replace `CosineAnnealingLR(T_max=MAX_EPOCHS)` with `OneCycleLR`:
linear warmup for 5% of total steps → cosine anneal to 0. Peak LR = 5e-4 (same as now).
Pct_start = 0.05, anneal_strategy="cos", div_factor=25 (initial LR = 2e-5),
final_div_factor=1e4 (final LR = 5e-8).

**Rationale:**
The current scheduler starts at full LR with no warmup. With batch_size=4 and 1499 training
samples, the first epoch sees ~375 gradient steps at full LR=5e-4 — this is a cold start
into a loss landscape that has large gradients from surface nodes (10× weighted). OneCycleLR's
warmup lets the model enter the sharp loss landscape gradually, reducing the probability of
early over-shooting into a bad basin. Smith & Touvron demonstrated OneCycleLR accelerates
convergence in vision transformers. Given the 50-epoch cap, faster convergence directly
improves the best-checkpoint metric. The warmup cost is 2–3 epochs; the anneal benefit
applies to the remaining 47.

**Predicted delta:** −1 to −4 on `val_avg/mae_surf_p`, primarily via better early convergence.

**Risk:** Low–medium. If the current schedule already converges well by epoch 20, this
adds no benefit. Peak LR must match — don't raise it in this test.

**Reproduce command:**
```
# In train.py, replace:
#   scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
# with:
#   scheduler = OneCycleLR(optimizer, max_lr=cfg.lr,
#                          steps_per_epoch=len(train_loader),
#                          epochs=MAX_EPOCHS, pct_start=0.05,
#                          anneal_strategy='cos', div_factor=25,
#                          final_div_factor=1e4)
# Step the scheduler per batch (not per epoch).
python train.py \
  --lr 5e-4 \
  --wandb_group onecyclelr-warmup \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — cold-start instability is plausible given surf_weight=10 gradient imbalance.
- Research-state value: 3 — scheduler change not tried before; loss curve shape will reveal whether warmup was needed.
- Execution value: 4 — zero additional compute, trivially reversible.

---

## Hypothesis 4 — Huber loss (delta sweep: 0.5, 1.0, 2.0)

**Axis:** Loss formulation

**Change:**
Replace MSE in the training loss with Huber loss (smooth L1):
`L_huber(r, δ) = 0.5·r² if |r|<δ, else δ·(|r|−0.5·δ)`.
Sweep δ ∈ {0.5, 1.0, 2.0} (in normalized residual space). Keep `surf_weight=10`.

**Rationale:**
The target range spans several orders of magnitude across Re (per-sample y std 164–2077).
In normalized space, high-Re samples still produce larger residuals than low-Re ones even
after global normalization with a single y_std. MSE squares these residuals, causing the
loss to be dominated by high-Re outliers. Huber truncates the quadratic to linear above δ,
acting as an adaptive weighting that reduces high-Re dominance without discarding those
samples. The primary metric is MAE not MSE, so moving the training objective closer to
MAE (Huber at small δ) should reduce the train-eval mismatch. δ=1.0 in normalized space
corresponds roughly to a 1σ residual threshold.

**Predicted delta:** −2 to −6 on `val_avg/mae_surf_p` (primarily on `val_re_rand` and OOD splits).

**Risk:** Medium. Huber with large δ approaches MSE; with small δ approaches MAE. If the
model needs to fit very small residuals precisely (low-Re samples), MAE-like loss at δ=0.5
may increase small-residual error. The sweep across 3 values provides the diagnostic.

**Reproduce command:**
```
# In train.py, add --loss_type and --huber_delta flags to Config.
# Replace F.mse_loss with F.huber_loss(pred, target, delta=cfg.huber_delta)
# in the surf and vol loss computations.
python train.py \
  --loss_type huber --huber_delta 1.0 \
  --surf_weight 10.0 \
  --wandb_group huber-loss-sweep \
  --epochs 50
# Run with huber_delta 0.5 and 2.0 separately in the same group.
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 4 — train-eval mismatch (MSE vs MAE) is a well-known failure mode;
  high-Re outlier dominance is directly observable from the per-split y ranges in program.md.
- Research-state value: 4 — the 3-delta sweep will characterize the optimal operating point;
  failure would mean per-sample normalization is the right fix instead.
- Execution value: 3 — 3 runs at 50 epochs each; one of these should become the new base.

---

## Hypothesis 5 — n_hidden width increase to 192 on nl3/sn16/n_head=1

**Axis:** Architecture (capacity)

**Change:**
Increase `n_hidden` from 128 to 192 in `model_config`, keeping all other nl3/sn16/n_head=1
settings. The prior capacity sweep (PR #29, rank 29, 82.739) tested width on the
*baseline* (n_layers=5, sn64, n_head=4) and found it unhelpful. This tests width on
the compressed base, which operates in a very different regime.

**Rationale:**
The nl3/sn16/n_head=1 config is extremely narrow: 128 hidden with 1 head and 16 slice tokens.
The prior width test at the baseline was not informative for the compressed base because
depth and slice changes fundamentally changed how capacity flows through the network.
At nl3/sn16/n_head=1, all representational capacity is concentrated in 16 slice tokens of
dim 128 — adding width to 192 increases the per-slice representation by 50% at a cost of
~(192/128)² ≈ 2.25× parameters (roughly +4M params on a ~3M model). VRAM budget is 96GB
so this is not a constraint. The preprocess MLP (24→256→128) also becomes (24→384→192),
which may help encode the 22 non-spatial features more expressively.

**Predicted delta:** −2 to −5 on `val_avg/mae_surf_p`.

**Risk:** Low–medium. If the model at nl3/sn16/n_head=1 is already overfit to the 1499
training samples, more capacity may regress. Check train vs val loss divergence.

**Reproduce command:**
```
# In train.py, change model_config:
#   n_hidden=192  (was 128)
# All other params: n_layers=3, slice_num=16, n_head=1, mlp_ratio=2
python train.py \
  --wandb_group width-sweep-192 \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — slice token width bottleneck is plausible at nl3/sn16/n_head=1.
- Research-state value: 3 — prior capacity test was on a different base; this is new evidence.
- Execution value: 3 — single run; ~25% slower than nl3/sn16/n_head=1 but well within VRAM.

---

## Hypothesis 6 — slice_num=4 extreme compression (sn4)

**Axis:** Architecture (slice reduction)

**Change:**
Reduce `slice_num` from 16 to 4 in `model_config`, keeping nl3/n_head=1/n_hidden=128.
This is one step past the sn8 tested in PR #39 (rank 2, val=42.426).

**Rationale:**
The leaderboard strongly suggests that fewer slice tokens is better on this dataset:
sn64→32→16→8 shows monotonically improving metrics (81.9→54.6→40.9→42.4). The uptick at
sn8 (42.4 vs 40.9 at sn16) suggests sn16 is near-optimal, but the sn8 result was a
full compound PR that may have had confounders. Testing sn4 directly on the nl3/n_head=1
base gives a clean data point. Physically: 4 slice tokens must encode the entire flow field
(background, surface BL, wake) — this forces extreme compression that may act as strong
regularization, or may lose the ability to distinguish flow zones. Either outcome is
informative.

**Predicted delta:** +2 to +4 regression (i.e. 43–45 from 40.927) OR unexpected improvement.

**Risk:** High for improvement, but the experiment is cheap and informative either way.
If sn4 is worse, it confirms sn16 is the floor and rules out further slice reduction.

**Reproduce command:**
```
# In train.py, change model_config:
#   slice_num=4  (was 16)
python train.py \
  --wandb_group sn4-extreme \
  --epochs 50
```

**Taste scores (Diagnostic):**
- Mechanistic grounding: 3 — the monotonic trend in the leaderboard is a clear signal; sn4 tests whether this extends.
- Research-state value: 4 — if better, opens a new frontier; if worse, closes this axis conclusively.
- Execution value: 4 — single run, faster than baseline (fewer slice tokens), fully interpretable outcome.

---

## Hypothesis 7 — SwiGLU alone on nl3/sn16/n_head=1

**Axis:** Architecture (activation)

**Change:**
Replace GELU in `TransolverBlock.mlp` with SwiGLU (`silu(W1·x) ⊗ W2·x`). Use
mlp_ratio=3 to compensate for the half-hidden split (3 × 2/3 = 2.0 effective ratio,
matching the current mlp_ratio=2).

**Rationale:**
SwiGLU was tested in PR runs ranked 9–11 but always in combination with Fourier PE,
which itself underperforms (ranks 9–22 are all Fourier-related or augmentation-related).
The bundled Fourier PE likely masked or degraded any SwiGLU benefit. This tests SwiGLU
cleanly on the best base. SwiGLU is the activation used in LLaMA 2/3, PaLM, and Gemini —
it consistently outperforms GELU in transformer settings by 0.5–2%. The SiLU gate in SwiGLU
provides a smooth, non-monotonic gating signal that may better handle the bimodal node
distribution (surface vs. background).

**Predicted delta:** −1 to −4 on `val_avg/mae_surf_p`.

**Risk:** Medium. The prior SwiGLU tests with Fourier scored ~58–62, well above the 40.9
baseline. Isolating the activation from Fourier is the critical difference; if clean SwiGLU
also regresses, it confirms that GLU activations genuinely do not help on this mesh task.

**Reproduce command:**
```
# Add SwiGLU module to train.py:
# class SwiGLU(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2)
#     def forward(self, x):
#         x, gate = self.proj(x).chunk(2, dim=-1)
#         return x * F.silu(gate)
#
# Use mlp_ratio=3 with SwiGLU (dim_out = dim_in * 3 // 2 per path)
python train.py \
  --wandb_group swiglu-clean \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — prior SwiGLU evidence is confounded; this isolates the mechanism.
- Research-state value: 4 — either confirms that gating helps (GeGLU/SwiGLU consistent) or rules out GLU activations entirely.
- Execution value: 3 — single run at same cost as baseline.

---

## Hypothesis 8 — EMA model weights for evaluation

**Axis:** Optimization (regularization / checkpoint)

**Change:**
Add exponential moving average (EMA) of model weights with decay=0.999. Use the EMA
model for validation and test evaluation. Keep the live model for gradient updates.
Use `torch_ema` or inline implementation (~10 lines).

**Rationale:**
EMA is standard in high-performing vision models (EfficientNet-B7, ViT-L/16, DeiT).
It reduces the variance of the best-checkpoint metric by smoothing out late-training
oscillations. With a 50-epoch hard cap and cosine annealing, the final epochs see a very
small LR but the model weights may still jitter due to the surface-vs-volume loss imbalance.
EMA with decay=0.999 effectively averages ~1000 gradient steps, providing a more stable
evaluation model. Cost: ~2× model memory (one extra copy), trivial vs 96GB VRAM budget.
The prediction improvement mechanism is purely through checkpoint stability, not a
fundamentally different training signal.

**Predicted delta:** −0.5 to −2 on `val_avg/mae_surf_p` (improvement via reduced variance
in the best-checkpoint selection).

**Risk:** Low. EMA cannot hurt training — it only changes which weights are used for eval.
If no improvement, the current schedule is already stable enough.

**Reproduce command:**
```
# In train.py, after optimizer definition:
# from torch_ema import ExponentialMovingAverage
# ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
#
# After each optimizer.step(): ema.update()
# For evaluation: with ema.average_parameters(): evaluate(...)
# Also add torch-ema to pyproject.toml dependencies.
python train.py \
  --wandb_group ema-weights \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 2 — mechanism is generic; no specific diagnosis of oscillation
  in prior runs (no loss curves available from prior PRs).
- Research-state value: 2 — result would confirm or deny checkpoint instability; doesn't
  address architecture bottleneck directly.
- Execution value: 3 — zero compute overhead, small memory cost, simple implementation.

---

## Hypothesis 9 — Relative MAE loss (log-scale normalization per sample)

**Axis:** Loss formulation

**Change:**
Replace the current global MSE loss with a per-sample relative MAE:
`L_rel = MAE(pred, target) / (|target|.mean() + ε)`.
This normalizes each sample's contribution by its own scale, preventing high-Re samples
from dominating. Compute separately for surface and volume nodes, then combine with
`surf_weight=10`.

**Rationale:**
The primary metric is MAE in physical space, with uniform weight across splits. High-Re
samples have per-sample y std up to 2077 vs 164 for low-Re — a factor of 12×. With global
MSE normalization, the model implicitly weight-averages toward high-Re samples. The relative
loss makes each sample contribute equally to the gradient signal, which should improve
cross-split generalization (especially `val_geom_camber_cruise` which is the low-Re domain).
This is distinct from Huber: Huber clips outlier residuals within a sample; relative MAE
normalizes across samples.

**Predicted delta:** −2 to −5 on `val_avg/mae_surf_p`, especially on `val_geom_camber_cruise`
and `val_re_rand`.

**Risk:** High. Relative loss changes the gradient magnitudes significantly and may destabilize
training. Start with `surf_weight=10` as-is; if diverged, reduce to 5.

**Reproduce command:**
```
# In train.py, define:
# def relative_mae_loss(pred, target, mask):
#     diff = (pred - target).abs()
#     scale = target.abs().mean(dim=-2, keepdim=True).clamp(min=1e-6)
#     return (diff / scale)[mask].mean()
#
# Replace mse_loss calls with relative_mae_loss for surf and vol.
# Add --loss_type relative_mae flag.
python train.py \
  --loss_type relative_mae \
  --surf_weight 10.0 \
  --wandb_group relative-mae-loss \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 4 — per-sample scale heterogeneity is directly measurable from
  program.md statistics; the loss-metric mismatch is a known training objective failure mode.
- Research-state value: 3 — per-split breakdown will tell whether low-Re splits improve at
  cost of high-Re; informative either way.
- Execution value: 3 — single run; relative loss adds negligible compute.

---

## Hypothesis 10 — Gradient clipping (max_norm sweep: 0.5, 1.0)

**Axis:** Optimization (regularization)

**Change:**
Add gradient norm clipping before the optimizer step:
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)`.
Sweep max_norm ∈ {0.5, 1.0}. Currently there is no gradient clipping.

**Rationale:**
With `surf_weight=10` and large high-Re pressure gradients, surface node losses can produce
gradient spikes that destabilize training. Gradient clipping is standard in transformer
training (GPT, T5, BERT all use max_norm=1.0). The nl3/sn16/n_head=1 base is a very narrow
model — a single gradient spike at a surface node could corrupt all 16 slice tokens
simultaneously (since they pool over all nodes via softmax). Clipping at 1.0 provides a
safety net; clipping at 0.5 is more aggressive and may slow convergence but improve stability.

**Predicted delta:** −0.5 to −2 on `val_avg/mae_surf_p`.

**Risk:** Low. If gradients are already well-behaved, clipping has no effect. If they are
spiking, clipping will show clear improvement in loss curve stability.

**Reproduce command:**
```
# In Config dataclass, add: grad_clip: float = 0.0
# In training loop, before optimizer.step():
#   if cfg.grad_clip > 0:
#       torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
python train.py \
  --grad_clip 1.0 \
  --wandb_group grad-clip-sweep \
  --epochs 50
# Also run with --grad_clip 0.5 in same group.
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 3 — slice token pooling makes gradient spikes plausible; surf_weight=10 amplifies surface gradients.
- Research-state value: 2 — informative but low-magnitude expected gain.
- Execution value: 4 — 2 runs, trivial implementation, zero architecture change.

---

## Hypothesis 11 — mlp_ratio=4 on nl3/sn16/n_head=1

**Axis:** Architecture (MLP capacity)

**Change:**
Increase `mlp_ratio` from 2 to 4 in `TransolverBlock`, keeping all other nl3/sn16/n_head=1
settings. This doubles the MLP hidden width from 256 to 512.

**Rationale:**
MLP ratio=2 is below standard transformer ratios (typically 4). The prior MLP ratio sweep
(PR #38, rank 6, 53.417) was run at sn16 but likely on a different n_layers/n_head config.
At nl3/sn16/n_head=1, the entire model has 3 blocks with 1 head — all pairwise interactions
are handled by a single attention head over 16 slice tokens. The per-node FFN capacity
(the MLP) is the only other place where non-linear transformations happen. Increasing
mlp_ratio from 2 to 4 at n_hidden=128 costs 128×(512−256)×2=65K additional parameters
per layer × 3 layers = ~200K params (small). This may recover some capacity lost by
reducing n_head from 4 to 1.

**Predicted delta:** −1 to −3 on `val_avg/mae_surf_p`.

**Risk:** Low–medium. If the bottleneck is attention expressivity (not MLP), this won't help.

**Reproduce command:**
```
# In train.py, change model_config:
#   mlp_ratio=4  (was 2)
# All other params: n_layers=3, slice_num=16, n_head=1, n_hidden=128
python train.py \
  --wandb_group mlp-ratio-4 \
  --epochs 50
```

**Taste scores (Frontier Refinement):**
- Mechanistic grounding: 2 — prior mlp_ratio sweep was not on this exact base; connection is indirect.
- Research-state value: 2 — result constrains MLP vs. attention capacity tradeoff.
- Execution value: 3 — minimal cost change.

---

## Hypothesis 12 — Compound: GeGLU + RMSNorm + OneCycleLR

**Axis:** Architecture + Optimization (compound)

**Change:**
Apply hypotheses 1 + 2 + 3 simultaneously: GeGLU activation with mlp_ratio=4, RMSNorm,
and OneCycleLR with 5% warmup. This compound should be run AFTER individual components
are validated, or immediately if GPU slots are available.

**Rationale:**
All three changes target orthogonal components (activation, normalization, optimizer) and
each has strong theoretical and empirical backing from large-scale transformer research.
If each delivers ~2 points individually, the compound may deliver ~4–6 points total.
The compound is only justified if the individual components have been tested — otherwise
it is uninterpretable. Run this in parallel as a higher-variance bet: if individual tests
are slower to return, this might be the fastest path to a new best metric.

**Predicted delta:** −4 to −8 on `val_avg/mae_surf_p`.

**Risk:** High interpretability cost but low execution risk. The components are individually
low-risk; their combination should not diverge.

**Reproduce command:**
```
# Apply all three changes from H1, H2, H3 simultaneously.
# GeGLU mlp (mlp_ratio=4), RMSNorm, OneCycleLR pct_start=0.05.
python train.py \
  --lr 5e-4 \
  --wandb_group compound-geglu-rmsnorm-onecycle \
  --epochs 50
```

**Taste scores (Tier Shift — compound):**
- Mechanistic grounding: 3 — each component is individually grounded; compound assumes additivity.
- Research-state value: 2 — interpretability is low; only useful if individual runs are slow.
- Execution value: 2 — should come after individual components return results.

---

## Priority Ranking for Student Assignment

| Rank | Hypothesis | Expected delta | Risk | Run cost |
|------|-----------|----------------|------|---------- |
| 1 | H4 — Huber loss delta=1.0 | −2 to −6 | Medium | 1 run |
| 2 | H9 — Relative MAE loss | −2 to −5 | High | 1 run |
| 3 | H1 — GeGLU activation | −2 to −5 | Medium | 1 run |
| 4 | H3 — OneCycleLR warmup | −1 to −4 | Low | 1 run |
| 5 | H2 — RMSNorm | −1 to −3 | Low | 1 run |
| 6 | H5 — n_hidden=192 | −2 to −5 | Med | 1 run |
| 7 | H6 — sn4 extreme | Unknown | High | 1 run |
| 8 | H7 — SwiGLU clean | −1 to −4 | Med | 1 run |
| 9 | H10 — Grad clip | −0.5 to −2 | Low | 2 runs |
| 10 | H11 — mlp_ratio=4 | −1 to −3 | Low | 1 run |
| 11 | H8 — EMA weights | −0.5 to −2 | Low | 1 run |
| 12 | H12 — GeGLU+RMSNorm+OneCycle compound | −4 to −8 | Med-interp | 1 run |

## Ruled-out paths (do not repeat without new evidence)

- Fourier positional encoding (standalone or per-block) — ranks 9–22
- FiLM conditioning on Re/AoA — rank 20
- Attention temperature annealing — rank 23
- Near-surface volume weighting — rank 21–22
- Horizontal flip augmentation — rank 30
- Cross-attention decoder — rank 28
- Pressure reparameterization — rank 24–27
- SwiGLU bundled with Fourier PE (confounded result)
- Width increase on the baseline (n_layers=5, sn64) — different regime

## Open uncertainties

1. Is the nl3/sn16/n_head=1 base undertrained or converged? No loss curves from PR #32 are available — the training dynamics are unknown. OneCycleLR (H3) + EMA (H8) will reveal this.
2. Is the bottleneck the attention mechanism or the per-node MLP? GeGLU (H1) vs mlp_ratio=4 (H11) will provide evidence.
3. Does the train/val objective mismatch (MSE vs MAE) explain the remaining ~41 error? Huber (H4) and relative MAE (H9) directly test this.
