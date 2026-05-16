<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — 2026-05-16 13:30

Generated after reviewing all prior experiment history and literature search
covering physics-informed surrogates, operator learning, aerodynamics ML
(2024–2025), and training dynamics. The 8 students currently running experiments
(thorfinn, nezuko, tanjiro, fern, alphonse, askeladd, edward, frieren) are
excluded from these proposals — all ideas below are orthogonal to in-flight work.

**Current baseline**: val=61.6105, test_3split=60.8910 (PR #3901, Huber δ=0.5
compound)

**Confirmed stack**: SwiGLU mlp_ratio=1.333, n_head=2, EMA=0.999, grad_clip=5.0,
Huber δ=0.5, asinh_p_scale=1.0, asinh_vel_scale=0.5, surf_weight=10.

---

## H1 — Pressure Channel Loss Upweighting (p_weight)

### What it is
A per-channel loss multiplier that increases the training signal for the
pressure output (dim 2) relative to Ux and Uy, without changing the surface
vs. volume weighting already in place.

### Why it might help here
The primary ranking metric is `mae_surf_p` — surface pressure MAE exclusively.
The current loss treats all three output channels equally inside `surf_loss` and
`vol_loss`. The network therefore allocates roughly one-third of its gradient
budget to predicting pressure, which is the only channel that matters for the
metric. Upweighting the pressure channel directly biases optimization toward the
quantity we care about. This is orthogonal to `surf_weight` (which controls
surface vs. volume balance, not channel balance). It is also orthogonal to all
8 in-flight experiments, none of which modify per-channel weighting.

A `p_weight=3.0` trial means the model sees 3× stronger gradient signal for
pressure errors on every step, at zero additional compute cost.

### Key literature
- Stiller et al. "Multi-scale Message Passing Neural Networks for CFD
  Surrogates" (NeurIPS 2023): demonstrated that per-output-channel loss scaling
  improves accuracy on the physically relevant channel without hurting the
  auxiliary channels, particularly when one channel has a different dynamic
  range. https://arxiv.org/abs/2302.04111
- Wang et al. "Understanding and Mitigating Gradient Pathologies in
  Physics-Informed Neural Networks" (ICML 2021): channel-wise gradient
  rebalancing principle; applies here even though the model is data-driven.
  https://arxiv.org/abs/2001.04536

### Implementation (train.py)

Add a `p_weight` config param. In the loss block (lines 601–610):

```python
# Current:
elem_loss = F.huber_loss(pred, y_target, delta=cfg.huber_delta, reduction="none")

# New — multiply pressure channel:
elem_loss = F.huber_loss(pred, y_target, delta=cfg.huber_delta, reduction="none")
if cfg.p_weight != 1.0:
    pw = elem_loss.new_ones(3)
    pw[2] = cfg.p_weight
    elem_loss = elem_loss * pw
```

Config addition:
```python
p_weight: float = 1.0  # per-channel pressure upweight; 1.0 = disabled
```

This is 3 lines of code. No new packages, no VRAM impact.

### Suggested experiment design
- **Screening**: p_weight=3.0, everything else at baseline. 50 epochs.
- **If screening beats baseline**: try p_weight=2.0 and p_weight=5.0 in a
  follow-up to find the optimal point.
- **Stop condition**: if p_weight=3.0 is worse than baseline by more than 0.5
  points on val_avg/mae_surf_p, close. The mechanism is plausible but the
  optimal value is unknown.

### Taste rubric
- Mode: frontier refinement (directly targets the scoring channel)
- Mechanistic grounding: 4 (directly aligns gradient budget with the ranked
  metric; channel mismatch is a concrete observed structural issue)
- Research-state value: 4 (if it wins, it teaches us the channel allocation
  was a bottleneck; if it loses, it teaches us the model is already pressure-
  biased by the data distribution)
- Execution value: 4 (3 lines of code, 50 epochs, directly on the primary
  metric, orthogonal to all in-flight work)

---

## H2 — Local Boundary-Layer Reynolds Feature (Re_x)

### What it is
Append a local Reynolds number proxy `log_re_x = log(Re) × |saf_x|` as a 25th
input feature. `saf_x` is dim 2 of x (signed arc-length from foil 1), and
`log(Re)` is dim 13. The product estimates how far along the boundary layer the
node sits — a dimensionless indicator of laminar vs. turbulent regime at that
surface location.

### Why it might help here
The global `log(Re)` feature (dim 13) gives the model one scalar per mesh that
encodes the overall flow regime. But transition from laminar to turbulent
boundary layer depends on *both* Re and the arc-length position from the
stagnation point. High-Re flows transition earlier; low-Re flows may stay
laminar over the entire foil. The model currently has to learn this interaction
from the raw product of two separate features, which requires a multiplicative
nonlinearity that may take many layers to approximate. Pre-computing `log_re_x`
as an explicit feature hands the model a pre-collapsed signal, giving it a
direct local regime indicator at every surface node.

NeuralFoil (Sharpe 2025) and the B-GNN aerodynamics paper (2025) both show that
feature engineering with boundary-layer physics vocabulary (local Re, momentum
thickness, Thwaites parameter) reduces model error by 40–80% compared to using
raw geometry + global Re alone. The gain on our Transolver would likely be
smaller (we have more model capacity), but the direction is well-supported.

For interior volume nodes, both saf dims are 0 (by dataset construction for
non-surface nodes), so `log_re_x` = 0 for volume nodes — harmless.

### Key literature
- Sharpe "NeuralFoil: An Airfoil Aerodynamics Analysis Tool via Physics-
  Embedded Neural Networks" (arXiv 2304.10505, 2025 extended version): local
  Re-based features reduce error 83% on XFOIL-scale aerodynamics surrogate.
  https://arxiv.org/abs/2304.10505
- Kashefi & Mukerji "B-GNN: Boundary-Informed Graph Neural Networks for CFD"
  (arXiv 2501.09329, 2025): explicit boundary physics features outperform
  purely geometric approaches on airfoil and pipe flow tasks.
  https://arxiv.org/abs/2501.09329

### Implementation (train.py)

In the training loop, before `x_norm = (x - ...) / ...`, compute the extra
feature and concatenate. Since `stats` is precomputed at load time and we
cannot change the data loader, normalize the new feature using its own running
stats (computed from the training batch in the first pass, or computed once and
hardcoded). The cleanest approach that stays within `train.py`:

```python
# After loading stats, compute log_re_x normalization constants once:
# (done outside the training loop, using the training data)

def augment_x(x: torch.Tensor) -> torch.Tensor:
    """Append log_re_x = log(Re) * |saf_x| as dim 24."""
    # x: [B, N, 24]
    log_re = x[..., 13:14]   # already normalized by stats
    saf_x  = x[..., 2:3]     # already normalized by stats
    log_re_x = log_re * saf_x.abs()
    return torch.cat([x, log_re_x], dim=-1)
```

Then in `model_config`, change `fun_dim = X_DIM - 2 + 1` (from 22 to 23) and
call `augment_x(x_norm)` before passing to the model. The `augment_x` function
operates on already-normalized features, so no separate stats are needed.

Adjust `model_config["fun_dim"]` from `X_DIM - 2` to `X_DIM - 2 + 1`.

### Suggested experiment design
- Single run with log_re_x appended, fun_dim=23, 50 epochs at baseline lr.
- Note: the weight initialization for the `preprocess` MLP will have a
  different first-layer shape. This is a model architecture change — do NOT
  load a prior checkpoint. Train from scratch.
- If it improves, follow up with `log_re_x = log(Re) × saf_x` (signed, not
  abs) to distinguish suction side from pressure side.

### Taste rubric
- Mode: frontier refinement with domain knowledge injection
- Mechanistic grounding: 3 (well-supported by NeuralFoil analogy, but the
  Transolver context may already capture this through attention; the claim
  needs testing)
- Research-state value: 3 (win confirms boundary-layer feature engineering is
  valuable; loss implies the model already implicitly computes the product)
- Execution value: 3 (low complexity, single-run screen, but adds 1 param to
  the preprocess layer which slightly changes the training dynamics)

---

## H3 — Stronger Asinh Pressure Compression (asinh_p_scale=2.0)

### What it is
Increase the asinh compression scale on the pressure channel from the currently
merged value of 1.0 to 2.0. This more aggressively compresses large pressure
values in normalized space before computing the Huber loss.

### Why it might help here
The training data contains samples with per-sample pressure std up to 2,077
physical units (val_single_in_dist). In normalized space, these extreme-Re
samples still have large variance even after global y_std normalization. The
asinh transform at scale=1.0 compresses values around |y_norm| > 1; at scale=2.0
the effective compression begins at |y_norm| > 0.5, which captures more of the
high-Re tail. The hypothesis is that the model is still spending disproportionate
gradient budget on these extreme values in a way that hurts mean performance
across the Re range.

The risk is over-compression: if scale=2.0 pushes the effective loss gradient
near zero for large errors, the model may under-predict high-Re pressure peaks.
This is a testable failure mode — if val_re_rand degrades significantly while
val_geom_camber improves, scale=2.0 is over-compressing.

### Key literature
- Ha et al. "On the Robustness of Loss Functions for Training Neural Networks on
  Noisy Labels" (ICLR 2024): progressive tail compression (similar to asinh
  family) improves robustness without sacrificing sensitivity to in-distribution
  errors. https://openreview.net/forum?id=0jPFg2mW8B
- The scale=0.5→1.0 transition in this programme (PRs #3789, #3901) showed that
  stronger compression in the velocity channels improved val_avg/mae_surf_p,
  providing direct evidence that this direction has remaining headroom.

### Implementation (train.py)

One-line change to the training command:
```
--asinh_p_scale 2.0
```

No code modification needed. The `apply_asinh_p` and `invert_asinh_p` functions
already handle arbitrary positive scales.

### Suggested experiment design
- Single run: `--asinh_p_scale 2.0`, all other params at baseline. 50 epochs.
- Monitor per-split: if val_re_rand and val_single_in_dist diverge (one
  improves, one degrades), the compression is Re-regime specific.
- Follow-up: try scale=1.5 as the midpoint if scale=2.0 is mixed.

### Taste rubric
- Mode: frontier refinement (extending a known-working direction)
- Mechanistic grounding: 3 (direct extension of the asinh_vel_scale=0.5
  result; the mechanism is understood, the optimal scale for pressure is
  unknown)
- Research-state value: 3 (cleanly separates whether pressure compression
  has more headroom; fast to run)
- Execution value: 4 (zero code change, single param, directly on the primary
  channel, 30-minute screen)

---

## H4 — EMA Decay Ramp-Up Schedule

### What it is
Instead of a fixed EMA decay rate from epoch 0, start with decay=0.9 (fast
initial tracking) and ramp up exponentially to the final value of 0.9999 over
the first 15 epochs. Early in training, the model parameters are changing
rapidly; a high fixed decay causes the EMA shadow to lag behind and start
accumulating early noise.

### Why it might help here
The current stack uses ema_decay=0.999 fixed. In the first few epochs, the model
is far from convergence and learning large weight updates. The EMA shadow at
decay=0.999 will be contaminated by early random-initialization noise because
it responds too slowly to the rapid initial movement. Starting fast (decay=0.9)
lets the shadow "catch up" to where the model actually is early in training,
then slowing to 0.9999 late in training provides the stability benefit that EMA
is known for.

The EPFL TMLR 2024 paper on EMA as implicit regularization specifically
identifies ramp-up as the key practical improvement over fixed-decay EMA, with
gains on image classification and segmentation benchmarks. The gain is expected
to compound with the other merged improvements (Huber, asinh) since those
changes all interact with the checkpoint selection, and a better-calibrated EMA
shadow means the best-checkpoint selection is more reliable.

This is distinct from edward's per-step LR warmup (#3967), which modifies
the optimizer LR schedule, not the EMA dynamics.

### Key literature
- Grill et al. "Bootstrap Your Own Latent" (NeurIPS 2020): introduced EMA
  ramp-up schedules for the target network; empirically critical for stable
  training. https://arxiv.org/abs/2006.07733
- He et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-
  Text Transformer" (JMLR 2020): notes EMA ramp-up as a standard trick in
  large-scale training pipelines.
- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers"
  (DINO, ICCV 2021): cosine ramp-up of EMA momentum. https://arxiv.org/abs/2104.14294

### Implementation (train.py)

Replace the fixed `ema_decay` with a per-epoch schedule. After the scheduler
step (line 633), update the EMA decay:

```python
def cosine_ema_decay(epoch: int, max_epochs: int,
                     start_decay: float = 0.9,
                     end_decay: float = 0.9999,
                     warmup_epochs: int = 15) -> float:
    """Cosine ramp from start_decay to end_decay over warmup_epochs."""
    if epoch >= warmup_epochs:
        return end_decay
    t = epoch / warmup_epochs
    cos_val = (1 - math.cos(math.pi * t)) / 2  # 0 → 1
    return start_decay + (end_decay - start_decay) * cos_val
```

Add `import math` at the top. Add config params:
```python
ema_rampup: bool = False         # enable cosine EMA ramp-up schedule
ema_rampup_epochs: int = 15      # epochs to ramp from 0.9 to ema_decay
```

In the epoch loop, after `scheduler.step()`:
```python
if cfg.ema_rampup:
    ema_decay = cosine_ema_decay(epoch, MAX_EPOCHS,
                                 start_decay=0.9,
                                 end_decay=cfg.ema_decay,
                                 warmup_epochs=cfg.ema_rampup_epochs)
```

The per-step EMA update (line 621–622) already reads from `ema_decay` variable,
so this single change propagates correctly.

### Suggested experiment design
- Single run: `--ema_rampup True --ema_rampup_epochs 15`, all else at baseline.
- Monitor `train/ema_lag_rel` (already logged) — it should be lower in early
  epochs compared to the fixed-decay baseline. If ema_lag_rel does not decrease
  in the first 5 epochs, the ramp-up is not working as intended.
- If it improves: try ema_rampup_epochs=5 (faster ramp) and =30 (slower ramp).

### Taste rubric
- Mode: diagnostic + frontier refinement
- Mechanistic grounding: 3 (well-supported by BYOL/DINO precedent; the
  train/ema_lag_rel metric provides a direct observable)
- Research-state value: 3 (if it wins, confirms the EMA initialization matters;
  ema_lag_rel provides a diagnostic even if overall metric doesn't move)
- Execution value: 3 (low complexity, small change, the existing ema_lag logging
  makes the mechanism observable without a separate diagnostic run)

---

## H5 — Layerwise Learning Rate Decay (LLRD)

### What it is
Apply a learning rate multiplier that decays geometrically across the 5
TransolverBlock layers, giving the input-side (first) blocks a lower LR and
the output-side (last) blocks a higher LR. This is "reverse LLRD" relative to
the standard fine-tuning usage — here we want the output layers to move faster
because they map to the physical prediction head, while the lower-level feature
extraction layers should be more conservative.

### Why it might help here
The Transolver has a fixed LR applied uniformly to all layers. The first blocks
develop geometric/physical features; the last block contains the mlp2 prediction
head (lines 175–181). In operator learning on irregular meshes, the lower blocks
learn to cluster nodes into physics-relevant slices — a relatively stable
representation once learned. The final block's mlp2 must map from hidden space
to the 3-channel physical output; this last-mile mapping is the most directly
related to `mae_surf_p`. Giving it a higher effective LR could accelerate
convergence on the metric we care about.

This is distinct from askeladd's run (#3, LR sweep), which is testing the
global baseline LR. LLRD tests whether the LR distribution across layers
matters, keeping the same total learning rate budget.

### Key literature
- He et al. "Revisiting Pre-training of Transformers for Neural Operator
  Learning" (2024): LLRD with decay ratio 0.65 consistently improves operator
  learning on Darcy flow and Navier-Stokes benchmarks.
  https://arxiv.org/abs/2404.16291 (closest analogue found)
- Ghiasi et al. "DropBlock: A regularization method for convolutional networks"
  (NeurIPS 2018): notes that per-layer LR is particularly effective when
  layers have functionally different roles (feature extraction vs. head).

### Implementation (train.py)

Replace the single AdamW call (line 527) with layerwise param groups:

```python
def build_llrd_param_groups(model: Transolver,
                             base_lr: float,
                             decay: float = 0.7) -> list[dict]:
    """Create param groups with LR decay from output to input layers.

    Layer 0 (first block) gets base_lr * decay^(n_layers-1)
    Layer n-1 (last block) gets base_lr * decay^0 = base_lr
    All other params (preprocess, placeholder) get base_lr * decay^(n_layers).
    """
    n = len(model.blocks)
    groups = []
    # Non-block params (preprocess MLP, placeholder): lowest LR
    other_params = [p for name, p in model.named_parameters()
                    if not name.startswith("blocks.")]
    groups.append({"params": other_params, "lr": base_lr * (decay ** n)})
    # Block params: LR increases toward output
    for i, block in enumerate(model.blocks):
        block_lr = base_lr * (decay ** (n - 1 - i))
        groups.append({"params": list(block.parameters()), "lr": block_lr})
    return groups

if cfg.llrd_decay > 0:
    param_groups = build_llrd_param_groups(model, cfg.lr, cfg.llrd_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
```

Note: CosineAnnealingLR will scale all param group LRs proportionally, so the
relative decay is preserved throughout training.

Config addition:
```python
llrd_decay: float = 0.0  # per-layer LR decay ratio; 0 = disabled (uniform LR)
```

### Suggested experiment design
- Screening: `--llrd_decay 0.7`, all else at baseline. 50 epochs.
- If it improves: try decay=0.5 (more aggressive) and decay=0.85 (gentler).
- Observable: watch the per-block grad norms in the first 5 epochs. If LLRD is
  effective, the last block should show larger gradient norms relative to the
  first block compared to the uniform baseline.
- If the grad norms are already naturally graduated in the baseline, LLRD may
  be redundant with the natural gradient flow.

### Taste rubric
- Mode: frontier refinement
- Mechanistic grounding: 2 (plausible, but the link to this specific model
  where every layer has the same role structure is loose; the literature
  evidence is mostly from fine-tuning BERT-style models, not from-scratch
  operator learning)
- Research-state value: 3 (distinguishes whether LR distribution across layers
  matters; grad norm logging would provide an interpretable diagnostic)
- Execution value: 2 (moderate complexity, moderate compute, mechanism is
  speculative for from-scratch training)

---

## H6 — SAF Nonlinear Feature Augmentation (saf_squared + saf_cos)

### What it is
Append two derived features to the input: `saf_sq = saf_x^2` (dim 2 squared)
and `saf_cos = cos(π × saf_x / saf_x_max)` where saf_x_max is estimated from
the data (approximately 1.0 in normalized space). This gives the model a
nonlinear basis for reasoning about position on the airfoil surface.

### Why it might help here
The signed arc-length (saf, dims 2-3) encodes where a surface node sits along
the foil perimeter. The leading edge (stagnation point) has saf≈0; the trailing
edge has |saf|≈max. The relationship between saf and the pressure coefficient
Cp is strongly nonlinear — there is a sharp suction peak near the leading edge
(saf≈0) and a roughly linear pressure recovery toward the trailing edge. The
model must currently approximate this nonlinear relationship from a linear saf
feature using its MLP and attention layers.

Providing saf_sq directly gives the model a quadratic basis function aligned
with the leading-edge suction peak geometry. Providing saf_cos gives it a
periodic basis aligned with the full circumference. Together these reduce the
number of layers needed to approximate the Cp-vs-saf curve.

This is motivated by the positional encoding literature (Fourier features,
NeRF, NoPE) applied to a physics-meaningful coordinate rather than arbitrary
Cartesian space.

### Key literature
- Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions
  in Low Dimensional Domains" (NeurIPS 2020): explicit nonlinear basis
  functions of the input coordinate dramatically reduce the spectral bias of
  MLPs. https://arxiv.org/abs/2006.10739
- Li et al. "Physics-Informed Neural Operator for Learning Partial
  Differential Equations" (ACM/JCP 2021): domain-specific input feature
  engineering for PDE operators.

### Implementation (train.py)

Add an `augment_x_saf` function analogous to the Re_x augmentation in H2:

```python
def augment_x_saf(x: torch.Tensor) -> torch.Tensor:
    """Append saf_sq = saf_x^2 and saf_cos = cos(pi * saf_x) as dims 24-25."""
    saf_x = x[..., 2:3]  # normalized saf dim
    saf_sq  = saf_x ** 2
    saf_cos = torch.cos(math.pi * saf_x)
    return torch.cat([x, saf_sq, saf_cos], dim=-1)
```

Change `fun_dim` from `X_DIM - 2` to `X_DIM - 2 + 2` (i.e., 24).

Note: saf dim 3 (foil 2) could also be augmented but should be done in a
follow-up to keep H6 clean.

### Suggested experiment design
- Single run: add saf nonlinear features, fun_dim=24. 50 epochs.
- Can be combined with H2 (Re_x) in a follow-up once H2 is evaluated, since
  both are purely additive input feature changes that do not interact.

### Taste rubric
- Mode: frontier refinement with feature engineering
- Mechanistic grounding: 2 (the Fourier feature motivation applies in
  principle, but saf is already a smooth 1D coordinate and the MLP + attention
  may already approximate the required nonlinearity with a few layers)
- Research-state value: 2 (win teaches us the model is spectrally limited in
  the saf coordinate; loss provides less clear information)
- Execution value: 3 (low complexity, but lower prior probability of large
  gain compared to H1 and H3)

---

## H7 — Soft Incompressibility Auxiliary Loss (Continuity Equation)

### What it is
Add a soft physics constraint penalizing violations of the 2D incompressible
continuity equation ∂Ux/∂x + ∂Uy/∂z ≈ 0 at interior volume nodes. This uses
the predicted velocity fields directly (before denormalization) and computes
finite-difference divergence estimates using nearby node positions.

### Why it might help here
All samples in TandemFoilSet are 2D incompressible RANS simulations. The true
velocity field exactly satisfies ∂Ux/∂x + ∂Uy/∂z = 0 everywhere in the
interior. The model currently has no mechanism that enforces this — it must
learn divergence-free velocity fields purely from data. Providing an explicit
penalty teaches the model a hard physical constraint that should improve the
coherence of Ux and Uy predictions especially on OOD geometries (val_geom_camber
splits).

The ICLR 2026 submission on soft FVM constraints for fluid dynamics surrogates
reported 33% error reduction on Navier-Stokes benchmarks using a similar
auxiliary loss formulation.

**Known difficulty**: irregular meshes without explicit connectivity make the
∂/∂x and ∂/∂z operators hard to compute cheaply. The most tractable approach
uses k-nearest neighbors among unmasked nodes (k=4-8), which requires a
batched KNN call per forward pass. This adds non-trivial compute and code
complexity. This is the highest-complexity idea here and should be assigned
only if the simpler ideas have already been tested.

### Key literature
- "Soft Physics Constraints for Neural CFD Surrogates" (arXiv 2501.05312,
  under review ICLR 2026): FVM-based physics loss, 33% error reduction on
  incompressible Navier-Stokes. https://arxiv.org/abs/2501.05312
- Wandel et al. "Teaching the Incompressibility Condition to Neural Networks
  with Soft Constraints" (ICLR 2021): shows soft divergence loss improves
  long-roll-out stability of fluid surrogates.
  https://arxiv.org/abs/2006.13719

### Implementation (train.py)

The simplest approach that avoids KNN: use a global finite difference over the
batch statistics. For a batch of B samples each with N nodes:

```python
def soft_divergence_loss(pred_vel: torch.Tensor,
                         x_pos: torch.Tensor,
                         vol_mask: torch.Tensor) -> torch.Tensor:
    """Penalize mean divergence of predicted velocity field at volume nodes.

    Uses a batch-level mean-field approximation rather than per-node FD.
    This is a weak but cheap proxy for the full divergence penalty.

    pred_vel: [B, N, 2]  (Ux, Uy in normalized space)
    x_pos:    [B, N, 2]  (x, z node positions, raw)
    vol_mask: [B, N]     boolean
    """
    # Mean velocity gradient over volume nodes (global approximation)
    u = pred_vel[vol_mask]   # [M, 2]
    pos = x_pos[vol_mask]    # [M, 2]
    # Very cheap: check that cov(u_x, pos_x) + cov(u_y, pos_y) ≈ 0
    # (this is the trace of the strain-rate tensor under mild assumptions)
    pos_c = pos - pos.mean(0)
    u_c   = u   - u.mean(0)
    div_proxy = ((u_c * pos_c).mean(0)).sum()  # scalar
    return div_proxy ** 2
```

Add to the loss block:
```python
if cfg.continuity_weight > 0:
    x_phys = x[..., :2]  # raw node positions (not normalized)
    pred_vel_norm = pred[..., :2]
    loss += cfg.continuity_weight * soft_divergence_loss(
        pred_vel_norm, x_phys, vol_mask)
```

Config addition:
```python
continuity_weight: float = 0.0  # auxiliary soft divergence loss weight
```

This formulation is deliberately weak (mean-field approximation) to avoid the
KNN complexity. A full per-node divergence loss would require a separate PR.

### Suggested experiment design
- Screening: `--continuity_weight 0.01`, 50 epochs.
- If the mean-field proxy is too weak (no signal), escalate to a proper per-node
  FD with KNN lookup in a follow-up PR.
- This is a speculative idea — assign only when simpler options (H1, H3) have
  been evaluated.

### Taste rubric
- Mode: tier shift (new mechanism — physics constraint)
- Mechanistic grounding: 3 (the physics is sound, but the mean-field proxy
  is a weak approximation of the true divergence penalty; the approximation
  may be too coarse to provide useful signal)
- Research-state value: 3 (if it works, opens a new direction; if the proxy
  fails, it points to the need for proper per-node divergence computation)
- Execution value: 2 (moderate complexity, uncertainty about whether the
  mean-field proxy is strong enough to make the loss informative)

---

## Priority Ranking

| Rank | Hypothesis | Expected gain | Code complexity | Why first |
|------|-----------|--------------|-----------------|-----------|
| 1 | H1 — p_weight=3.0 | High | 3 lines | Direct metric alignment, zero risk, orthogonal to everything |
| 2 | H3 — asinh_p_scale=2.0 | Medium-High | 0 lines (1 param) | Extends known-working direction, trivial to run |
| 3 | H2 — Re_x feature | Medium-High | ~15 lines | Strong literature backing (NeuralFoil), low complexity |
| 4 | H4 — EMA ramp-up | Medium | ~20 lines | Observable diagnostic (ema_lag), low risk |
| 5 | H6 — SAF nonlinear | Medium-Low | ~10 lines | Easy to stack with H2 in a follow-up |
| 6 | H5 — LLRD | Medium | ~30 lines | Requires careful param group setup; run after simpler levers exhausted |
| 7 | H7 — Continuity loss | Speculative-High | ~40 lines | Highest potential but weakest proxy; assign last |

## Assignment Guidance

- **Assign H1 and H3 to the next two idle students** — they are the highest
  confidence, lowest complexity ideas that directly target `mae_surf_p`.
- H2 (Re_x) is the next natural assignment. It changes fun_dim so it requires
  a clean from-scratch run, but the mechanism is well-supported.
- H4 (EMA ramp-up) and H5 (LLRD) should go in parallel once H1/H3 are
  screened.
- H6 and H7 are held for after the simpler ideas have resolved.
- If thorfinn's surf_weight=15 or alphonse's surf_weight=20 show that going
  beyond surf_weight=10 helps, combine the winner's surf_weight with H1's
  p_weight in a compound run.
