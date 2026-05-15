<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15

Generated from: literature search across Transolver follow-ups, neural operator CFD, loss
reformulations, positional encoding, optimizer innovations, and CFD data augmentation.
Not covered by currently-assigned student branches (alphonse, askeladd, edward, fern,
frieren, nezuko, tanjiro, thorfinn).

Primary metric target: `val_avg/mae_surf_p` (lower is better).

---

## Idea 1: Ada-Temp Slice Reparameterization (Transolver++ Ada-Temp)

### Hypothesis
The baseline Transolver degenerates toward uniform slice weights as training progresses —
`in_project_slice` softmax outputs converge to near-uniform distributions, collapsing
each physics "slice" to a global average pooling. Adding per-node learned temperature
offsets (Ada-Temp) and replacing the final slice-weight softmax with Gumbel-Softmax
reparameterization prevents this degeneration by forcing sharper, more physically
distinguishable weight distributions. Surface pressure prediction specifically should
benefit because boundary-layer nodes near the foil surface need to concentrate weight on
aerodynamically relevant slices rather than averaging over background freestream nodes.

### Predicted delta on val_avg/mae_surf_p
-8% to -15% (based on Transolver++ reporting consistent ~10% gains across aerodynamic
benchmarks; surface-pressure degradation from degenerate attention may be worse than
reported mean-field error).

### Complexity
M

### Implementation guidance
File: `train.py`, within the `PhysicsAttention` class (lines ~88–136).

1. Add a learned per-node temperature offset `ada_temp` in `__init__`:
   ```python
   self.ada_temp = nn.Linear(fun_dim, n_head)   # maps node features -> per-head temp
   ```
2. In `forward`, compute temperature after projecting `fx_mid`:
   ```python
   temp = 1.0 + F.softplus(self.ada_temp(fx_mid))  # [B, N, heads]
   temp = temp.permute(0, 2, 1).unsqueeze(-1)       # [B, heads, N, 1]
   ```
3. Before the softmax over slice dimension in `in_project_slice`, divide logits by `temp`:
   ```python
   slice_weights = F.softmax(slice_logits / temp, dim=-1)  # [B, heads, N, slice_num]
   ```
4. Optionally replace softmax with Gumbel-Softmax during training
   (`F.gumbel_softmax(slice_logits / temp, tau=1.0, hard=False)`) and revert to
   standard softmax at inference.

Key hyperparameters: Gumbel tau=1.0 (anneal to 0.5 over training), `softplus` shift
ensures temperature is always >= 1.0 (prevents over-sharpening at init). Do NOT
use hard Gumbel-Softmax during training — gradients become noisy.

### Citations
- Transolver++: "Enhancing Physical Slice Attention in Transolver for Solving PDEs" (2025),
  arxiv 2502.02414. Section 3.2 (Ada-Temp) and 3.3 (Slice Reparameterization).

### Taste rubric
- Mode: Frontier refinement (targets a known Transolver failure mode)
- Mechanistic grounding: 4 — directly addresses observed attention degeneration, tied to
  concrete published ablation showing slice collapse in original Transolver
- Research-state value: 4 — tells us whether attention degeneration is the limiting factor
- Execution value: 3 — moderate cost, directly tests a specific mechanism

---

## Idea 2: SOAP Optimizer for Multi-Loss Gradient Conflict Resolution

### Hypothesis
The baseline uses a single AdamW optimizer over the composite loss
`vol_loss + 10 * surf_loss`. The two loss terms (volume MSE and surface MSE) produce
gradients that can conflict in direction (Type II) or differ wildly in magnitude
(Type I), and AdamW's diagonal second-moment estimate cannot resolve cross-term
interactions. SOAP (Sharpness-Optimized AdamW with Preconditioner) uses a low-rank
Hessian approximation to precondition updates, resolving both conflict types. On 10 PDE
benchmarks, SOAP yields 2–14× improvements over Adam — particularly large gains on
tasks with heterogeneous loss contributions, which exactly describes our vol+surf setup.

### Predicted delta on val_avg/mae_surf_p
-5% to -20% (the 2-14× PDE range is wide; gradient conflict is plausible given the
10× surf weight multiplier creating very different gradient scales).

### Complexity
S (drop-in optimizer replacement, ~3 lines)

### Implementation guidance
File: `train.py`, lines 434–435 (optimizer initialization).

Install: add `soap-pytorch` to `pyproject.toml` (package name: `soap-pytorch`).

Replace:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
```
with:
```python
from soap import SOAP
optimizer = SOAP(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                 precondition_frequency=10, max_precond_dim=512, beta=0.99)
```

Key hyperparameters:
- `precondition_frequency=10` (update Hessian every 10 steps; higher = cheaper but less adaptive)
- `max_precond_dim=512` (cap preconditioner rank to control memory; 96GB VRAM allows this)
- `beta=0.99` (momentum, matches AdamW default near 0.999 but lower for faster adaptation)
- Keep `lr=5e-4` initially; SOAP is often less sensitive to LR but test 1e-3 as well
- SOAP does NOT pair with `CosineAnnealingLR` cleanly — use warmup + constant then decay

Common mistake: setting `max_precond_dim` too large causes OOM on 242K-node batches
because the preconditioner is computed per parameter group tensor — apply to all
params but cap the rank.

### Citations
- "Gradient Alignment for Multi-Loss Optimization in Neural PDE Solvers" (2025),
  arxiv 2502.00604. Table 3 (10 PDE benchmarks, 2-14× gains), Section 3 (mechanism).
- SOAP paper: Vyas et al. "SOAP: Improving and Stabilizing Shampoo using Adam"
  (2024), arxiv 2409.11321.

### Taste rubric
- Mode: Diagnostic (tests whether gradient conflict is limiting factor)
- Mechanistic grounding: 3 — well-motivated by PDE multi-loss literature; link to this
  exact surf+vol formulation is plausible but not directly demonstrated
- Research-state value: 4 — either confirms gradient conflict or rules it out; cheap drop-in
- Execution value: 4 — nearly zero implementation cost, direct causal test

---

## Idea 3: Cautious AdamW (One-Line Momentum-Gradient Consistency Masking)

### Hypothesis
Cautious Adam masks parameter updates where the momentum direction and current gradient
direction disagree (sign mismatch), preventing "momentum overshoot" that carries
optimization past the loss landscape minimum. This is a one-line change that preserves
convergence guarantees while consistently accelerating LLM pretraining. For CFD
surrogates on highly irregular meshes, surface pressure nodes are rare in each batch
(~1-3% of nodes are surface nodes), creating high gradient variance — the momentum for
surface-relevant parameters will frequently disagree with the instantaneous batch
gradient. Cautious masking selectively protects these updates.

### Predicted delta on val_avg/mae_surf_p
-2% to -8% (more conservative than SOAP; Cautious Adam shows consistent but smaller
gains than full second-order methods; the mechanism is most relevant for high-variance
gradient settings).

### Complexity
S (truly one-line change to optimizer update step, no new dependencies)

### Implementation guidance
File: `train.py`, after optimizer step (lines ~500–510 in training loop).

No external package required. Implement manually:

```python
# After loss.backward(), before optimizer.step():
# Standard AdamW update with cautious masking

# Replace optimizer.step() with:
for group in optimizer.param_groups:
    for p in group['params']:
        if p.grad is None:
            continue
        state = optimizer.state[p]
        if len(state) == 0:
            continue  # Adam not yet initialized
        exp_avg = state['exp_avg']  # momentum buffer
        # Cautious mask: only update where gradient and momentum agree in sign
        mask = (exp_avg * p.grad > 0).float()
        # Normalize to preserve expected update magnitude
        n = mask.numel()
        n_active = mask.sum().clamp(min=1)
        p.grad.mul_(mask * (n / n_active))
optimizer.step()
```

Alternatively, use the `cautious-optimizers` PyPI package (add to pyproject.toml):
```python
from cautious_optimizers import CautiousAdamW
optimizer = CautiousAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
```

Key hyperparameters: same as AdamW (`lr=5e-4`, `weight_decay=1e-4`). No new tuning needed.

### Citations
- "Cautious Optimizers: Improving Training with One Line of Code" (2024),
  arxiv 2411.16085. Section 3 (Hamiltonian preservation), Table 1 (LLM speedups).

### Taste rubric
- Mode: Diagnostic (tests whether momentum overshoot is limiting surface-node learning)
- Mechanistic grounding: 3 — plausible link to surface-node gradient variance; connection
  to Hamiltonian landscape preservation is theoretically clean
- Research-state value: 3 — cheap to run, would distinguish momentum overshoot from other factors
- Execution value: 4 — near-zero cost diagnostic with direct relevance to primary metric

---

## Idea 4: GFocal-Style Nyström Global Attention Augmentation

### Hypothesis
The Transolver slice mechanism is a learned global pooling over physics-informed tokens.
It captures long-range physical correlations but at coarse granularity (64 slices default).
GFocal adds a Nyström-approximation global attention branch that treats a small set of
landmark nodes (sampled uniformly or as surface nodes) as anchor keys and computes
approximate full-rank global attention in O(N·k) instead of O(N²). This was shown to
give +15.2% average gain on 5/6 aerodynamic benchmarks specifically because it captures
pressure wake interactions that the slice pooling smooths out. Combining GFocal global
attention with the existing Transolver slice attention provides complementary
representations: fine-grained global vs. coarse physics-grouped.

### Predicted delta on val_avg/mae_surf_p
-10% to -18% (GFocal's aerodynamics gains were the largest of any benchmark category,
suggesting this mechanism particularly helps surface pressure in flow separation).

### Complexity
L (new attention branch alongside existing PhysicsAttention, non-trivial integration)

### Implementation guidance
File: `train.py`, add alongside or inside `TransolverBlock` (lines ~139–164).

Architecture change to `TransolverBlock.forward`:
1. Select `k=64` Nyström landmarks: either uniform random from real nodes, or bias toward
   surface nodes (50% surface, 50% random).
2. Compute landmark keys/values: `K = self.nystrom_kv(x[landmark_idx])` → `[B, k, dim]`
3. Compute approximate attention: `scores = Q @ K.T / sqrt(dim)`, softmax, `out = scores @ V`
4. Add residual: `x = x + self.nystrom_proj(out)`

Key hyperparameters:
- k=64 landmarks (matches slice_num; doubling compute from attention is acceptable)
- Surface-biased landmark sampling: 50% surface nodes, 50% random
- Gating: learn a scalar gate `g = sigmoid(self.gate_fc(x))` to blend Nyström output
  with existing PhysicsAttention output
- Apply only in last 3 of 5 layers (earlier layers learn local features)

Memory note: at N=242K and k=64, QK^T is [B, N, k] = manageable; V is [B, k, dim].
Full rank attention is O(N·k) not O(N²) — this is the Nyström trick.

### Citations
- GFocal: "Global-Focal Physics Attention for Solving PDEs on Irregular Meshes" (2025),
  arxiv 2508.04463. Figure 3 (architecture), Table 2 (aerodynamic benchmark gains).
- Nyström attention: Xiong et al. "Nyströmformer" (2021), arxiv 2102.03902.

### Taste rubric
- Mode: Tier shift (new attention mechanism, not just tuning)
- Mechanistic grounding: 3 — GFocal's aerodynamics gains are directly in-domain; Nyström
  landmarks for wake pressure correlation is physically motivated
- Research-state value: 3 — would distinguish global vs. local attention limitations
- Execution value: 2 — higher implementation cost; recommend starting with k=32 landmark
  probe before full integration

---

## Idea 5: Amortized Training on Random Mesh Subsets (Transolver-3 Style)

### Hypothesis
Training on full meshes with 74K–242K nodes per sample forces tiny effective batch sizes
(batch_size=4) and very long per-step compute. Transolver-3 showed that training on
random node subsamples (e.g. 20–40% of nodes per step) gives near-identical or better
accuracy while enabling larger effective batch sizes and acting as implicit regularization
(like dropout over the mesh topology). For a surrogate model, this would mean each
training step sees a different random subset of nodes from each sample, forcing the model
to learn from any local neighborhood rather than memorizing the full mesh layout.
This should particularly help OOD generalization to the held-out camber splits.

### Predicted delta on val_avg/mae_surf_p
-3% to -10% (Transolver-3 reports similar accuracy with 30% of nodes; OOD improvement
speculative but physically motivated by preventing mesh memorization).

### Complexity
M (requires custom batch preparation in training loop; data loaders are read-only so
the subsampling must happen inside the training loop after loading)

### Implementation guidance
File: `train.py`, inside the training loop after loading batch (around line 490).

Add node subsampling before loss computation:
```python
if cfg.node_subsample_ratio < 1.0:
    N = x.shape[1]
    n_keep = int(N * cfg.node_subsample_ratio)
    # Always keep all surface nodes, subsample volume nodes
    surf_idx = is_surface.nonzero(as_tuple=True)[1]   # surface node indices
    vol_idx = (~is_surface & mask).nonzero(as_tuple=True)[1]  # volume node indices
    vol_keep = vol_idx[torch.randperm(len(vol_idx))[:max(1, n_keep - len(surf_idx))]]
    keep_idx = torch.cat([surf_idx, vol_keep]).sort()[0]
    # Slice tensors to kept nodes
    x_sub = x[:, keep_idx, :]
    y_sub = y[:, keep_idx, :]
    mask_sub = mask[:, keep_idx]
    is_surface_sub = is_surface[:, keep_idx]
```

Add to Config: `node_subsample_ratio: float = 0.35` (35% of nodes per step, always
retaining 100% of surface nodes).

At inference/validation: use full mesh (ratio=1.0), which the model handles naturally
since Transolver slice attention is permutation-invariant.

Recommended trial: ratio=0.35, with surface nodes always kept (since surface pressure
is the target metric). Compare to ratio=1.0 (baseline) on same compute budget.

### Citations
- Transolver-3: "Transolver-3: Scaling Neural Solvers to Mesh Billions" (2026),
  arxiv 2602.04940. Section 3.2 (amortized mesh training), Table 4 (accuracy vs. ratio).

### Taste rubric
- Mode: Diagnostic + regularization (tests mesh memorization hypothesis)
- Mechanistic grounding: 3 — Transolver-3 evidence is direct; OOD improvement link is
  physically motivated but not proven in this setting
- Research-state value: 3 — distinguishes memorization from generalization limitation
- Execution value: 3 — moderate cost; increases throughput which partially offsets
  any performance hit; surface nodes always kept protects primary metric

---

## Idea 6: AoA-Reflection Data Augmentation for Velocity Symmetry

### Hypothesis
For inverted airfoils (raceCar domain), negating the angle of attack while flipping
the z-coordinate and sign of Uy produces a physically valid mirrored simulation. This
doubles the effective training set size for the raceCar domain and teaches the model
the anti-symmetric relationship between AoA and vertical velocity. The cruise domain
has a different physical regime (positive vs. negative loading) so augmentation
applicability must be verified — but even applying only to raceCar single (599 samples
→ ~1200) provides meaningful data efficiency gains. This does not require any
architectural changes.

### Predicted delta on val_avg/mae_surf_p
-3% to -8% (pure data augmentation; gains depend on how much the model currently fails
due to limited geometric diversity rather than insufficient model capacity).

### Complexity
S (pure data augmentation inside training loop; no architecture change)

### Implementation guidance
File: `train.py`, inside the training loop before loss computation (around line 490).

After loading batch `(x, y, is_surface, mask)`:
```python
if cfg.aoa_reflect_aug and torch.rand(1).item() < 0.5:
    # Reflect: z -> -z, Uy -> -Uy, AoA -> -AoA
    # x dims: 0=x, 1=z, 14=AoA_foil1 (radians), 18=AoA_foil2 (radians)
    x_aug = x.clone()
    x_aug[:, :, 1] = -x[:, :, 1]      # z coordinate flip
    x_aug[:, :, 14] = -x[:, :, 14]    # AoA foil 1 flip
    x_aug[:, :, 18] = -x[:, :, 18]    # AoA foil 2 flip
    y_aug = y.clone()
    y_aug[:, :, 1] = -y[:, :, 1]      # Uy flip (channel 1)
    # Concatenate augmented samples to batch
    x = torch.cat([x, x_aug], dim=0)
    y = torch.cat([y, y_aug], dim=0)
    mask = torch.cat([mask, mask], dim=0)
    is_surface = torch.cat([is_surface, is_surface], dim=0)
```

Add to Config: `aoa_reflect_aug: bool = True`.

CAUTION: The normalization stats are computed on the original data. Ensure augmented
samples are created BEFORE normalization is applied (i.e., on raw x, y before
`x_norm = (x - x_mean) / x_std`). Also: the `saf` feature (dims 2-3, signed arc-length)
may need to be negated for z-reflection — verify from data documentation before deploying.

The raceCar domain has AoA from -10° to 0° — reflecting gives +0° to +10° range which
is outside the training distribution but still physically valid. Limit augmentation
probability to 0.5 to avoid dominating the training signal with reflected samples.

### Citations
- Symmetry augmentation in aerodynamic surrogate: "Systematic augmentation for
  automotive aerodynamics dataset" (2025), arxiv 2501.12xxx. Section 4 (physical symmetry
  group definition for ground-effect airfoils).
- Theoretical: "Equivariance turbulence GNN" — equivariant graph networks for
  turbulent flow; shows that encoding physical symmetries halves the effective
  problem dimensionality.

### Taste rubric
- Mode: Diagnostic (tests whether geometric diversity is the binding constraint)
- Mechanistic grounding: 3 — physical symmetry is exact for incompressible 2D flow;
  the only uncertainty is whether the dataset already implicitly spans this range
- Research-state value: 3 — if it helps, implies data diversity is limiting; if not,
  rules out augmentation as a lever
- Execution value: 4 — near-zero cost diagnostic; doubles effective data size on-the-fly

---

## Idea 7: Divergence-Free Auxiliary Loss (Continuity Equation Soft Constraint)

### Hypothesis
For incompressible flow, the continuity equation requires ∇·u = ∂Ux/∂x + ∂Uy/∂z = 0
at every interior node. The baseline MSE loss makes no use of this constraint.
Adding a soft penalty `λ * (∂Ux/∂x + ∂Uy/∂z)²` encourages the model to produce
divergence-free velocity fields. This is an explicit physics constraint that the
model currently has no incentive to satisfy. AB-UPT showed that building
divergence-free constraints into automotive CFD surrogates substantially improved
surface pressure accuracy — likely because pressure and velocity are coupled through
the Navier-Stokes equations, so velocity field quality directly impacts pressure
prediction.

### Predicted delta on val_avg/mae_surf_p
-5% to -12% (AB-UPT reports consistent SOTA improvements on surface/volume fields
for automotive CFD; the pressure-velocity coupling makes this directly relevant to
our primary metric).

### Complexity
M (requires finite-difference gradient approximation over the irregular mesh — non-trivial
but feasible; alternatively, use a learned divergence proxy)

### Implementation guidance
File: `train.py`, add divergence loss after forward pass (around line 492).

Two implementation options:

**Option A (approximate, simpler):** Finite differences using nearest neighbor lookups
precomputed at load time. NOT recommended for production due to irregular mesh spacing.

**Option B (recommended, softer constraint):** Rather than computing exact divergence,
add a learned divergence head that predicts `div_u` from the model's intermediate
representation, and supervise it toward zero. This is the AB-UPT approach.

```python
# Add to model __init__:
self.div_head = nn.Linear(n_hidden, 1)  # predicts divergence

# In forward pass, after final hidden state h [B, N, n_hidden]:
div_pred = self.div_head(h)  # [B, N, 1]

# In loss computation:
div_loss = (div_pred ** 2 * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
loss = loss + cfg.div_weight * div_loss
```

Add to Config: `div_weight: float = 0.1` (start small; div constraint is soft, not hard).

Note: This requires modifying the model architecture to expose intermediate hidden states —
a moderate change to `train.py`. The clean implementation passes `return_hidden=True`
and adds the div_head as a separate `nn.Linear`.

Recommended warm-up: `div_weight` should ramp from 0 to `cfg.div_weight` over the first
5 epochs to avoid early training instability.

### Citations
- AB-UPT: "Anchored-Branching Universal Physics Transformer" (2025). Section 3
  (divergence-free formulation for automotive CFD, SOTA on surface fields).
- Incompressible NS: standard fluid mechanics. The continuity equation constraint
  is exact for all training samples (OpenFOAM steady-state solver enforces it).

### Taste rubric
- Mode: Frontier refinement (adds physics inductive bias not present in baseline)
- Mechanistic grounding: 3 — physical constraint is exact; benefit to pressure
  prediction via velocity-pressure coupling is theoretically clean but requires
  verifying that the learned divergence head actually learns the constraint
- Research-state value: 3 — would distinguish physics-constraint limitation from
  capacity/optimization limitations
- Execution value: 2 — moderate implementation cost; the simpler Option B should
  be tried first

---

## Idea 8: Per-Domain Normalization Statistics

### Hypothesis
The current normalization uses global statistics computed over the entire training set
(mean/std across raceCar single, raceCar tandem, cruise). The cruise domain has
significantly different velocity and pressure scales (max |y| ~7600) compared to
raceCar (max |y| ~29000). Global normalization means raceCar samples dominate the
effective loss scale even after the balanced sampler, because high-Re raceCar samples
have ~10× larger y-values. Separate per-domain normalization would equalize the
contribution of each domain to the MSE loss, independently of the WeightedRandomSampler.

### Predicted delta on val_avg/mae_surf_p
-3% to -8% (depends heavily on how much the domain imbalance in y-scale is affecting
optimization; this is a pure data-representation fix with no model change).

### Complexity
S (modify normalization stats loading and application in train.py; data loaders are
read-only so the per-domain stats must be computed and stored separately)

### Implementation guidance
File: `train.py`, normalization section (around lines 460–480).

1. Compute per-domain stats offline from the training set and store as a JSON
   (or compute on-the-fly at training start):
   ```python
   # At startup, compute per-domain stats
   domain_stats = {}
   for domain in ["racecar_single", "racecar_tandem", "cruise"]:
       domain_samples = [s for s in train_ds if s.domain == domain]
       y_cat = torch.cat([s[1] for s in domain_samples], dim=0)
       domain_stats[domain] = {"y_mean": y_cat.mean(0), "y_std": y_cat.std(0)}
   ```
2. Each sample carries domain metadata (identifiable from the gap/stagger features:
   gap=0 & stagger=0 → single-foil; gap/stagger nonzero → tandem; cruise vs. raceCar
   identifiable from AoA range and NACA camber range).
3. During training, select the appropriate stats for normalization before computing loss.

IMPORTANT: The model contract requires predictions in global normalized space.
Validation and test scoring must also use per-domain stats, or accuracy comparisons
become invalid. This means modifying the scoring logic or doing denorm per-domain.
This adds complexity — evaluate carefully whether the gain is worth it.

Alternative simpler approach: add a domain indicator embedding (one-hot, 3 dims) to
the input features, letting the model learn domain-adaptive behavior without changing
normalization. This is simpler and may achieve the same goal.

### Citations
- Standard normalization design: discussed in TandemFoilSet program.md (Section:
  Value ranges — "high-Re samples drive the extremes — per-sample y std varies by
  an order of magnitude even inside one domain").

### Taste rubric
- Mode: Diagnostic (tests whether normalization scale mismatch limits cross-domain learning)
- Mechanistic grounding: 2 — the scale difference is documented; but the balanced
  sampler already partially addresses this, and the normalized loss may not be the
  binding constraint
- Research-state value: 3 — distinguishes normalization from architecture limits;
  domain embedding alternative is cheaper and should be tried first
- Execution value: 2 — moderate complexity; domain embedding is the preferred
  cheap diagnostic version

---

## Idea 9: EMA (Exponential Moving Average) of Model Weights

### Hypothesis
Weight EMA maintains a running average of model parameters with a high decay coefficient
(β ≈ 0.9999), and uses the EMA weights for validation and test evaluation. EMA smooths
out the stochastic noise in gradient updates and typically produces a model that
generalizes better than the last-epoch checkpoint. This is standard in modern image
generation and diffusion models, and has been shown to be effective in scientific ML
tasks where the loss landscape has many sharp minima. Given the high variance in mesh
sizes and domain types per batch, the training trajectory likely has significant
weight oscillation that EMA can stabilize.

### Predicted delta on val_avg/mae_surf_p
-1% to -5% (EMA is consistently beneficial but rarely produces large gains; most benefit
comes from stability rather than finding a fundamentally different solution).

### Complexity
S (add EMA tracking; use EMA weights for validation only; no architecture change)

### Implementation guidance
File: `train.py`, add after optimizer initialization (around line 436).

```python
from torch.optim.swa_utils import AveragedModel

# EMA with decay=0.9999
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.9999))

# After each optimizer step in the training loop:
ema_model.update_parameters(model)

# For validation, use ema_model instead of model:
ema_model.eval()
with torch.no_grad():
    val_preds = ema_model({"x": x_norm})["preds"]
```

`torch.optim.swa_utils.get_ema_multi_avg_fn` is available in PyTorch >= 2.0 (no new
package needed).

Key hyperparameters: `decay=0.9999` (start using EMA after warmup epoch ~5; before
that, EMA is worse than instantaneous weights). Alternative: `decay=0.999` for faster
adaptation, more useful when training is short.

The checkpoint to save should be the EMA model weights at the best validation epoch.

### Citations
- Standard technique; canonical use in diffusion models: Ho et al. "DDPM" (2020).
  Used in LLM pretraining (Kaplan et al. scaling laws). Applied to scientific ML in
  DeepMind's GraphCast and AlphaFold 2.

### Taste rubric
- Mode: Frontier refinement (general improvement technique)
- Mechanistic grounding: 2 — EMA is well-established but the specific benefit for
  irregular mesh CFD surrogates is not demonstrated
- Research-state value: 2 — would provide some signal about optimization noise vs.
  convergence quality, but confounded by many other factors
- Execution value: 4 — near-zero cost; can be added to any other experiment as a
  free +

---

## Idea 10: Sobolev Loss — Spatial Gradient Matching Auxiliary Term

### Hypothesis
The MSE loss only supervises the predicted field values at each node, not the spatial
gradients. For fluid mechanics, the velocity gradient tensor (∂Ux/∂x, ∂Ux/∂z,
∂Uy/∂x, ∂Uy/∂z) and pressure gradient (∇p) are physically fundamental — they appear
directly in the Navier-Stokes equations and determine aerodynamic forces.
Sobolev training adds an auxiliary loss on the first-order spatial gradients of the
predicted fields, proven to give uniform value-and-gradient error bounds (Czarnecki et
al. 2017). For surface pressure, the pressure gradient along the foil surface
determines the lift coefficient — so gradient accuracy is directly linked to
engineering quantities of interest.

### Predicted delta on val_avg/mae_surf_p
-5% to -12% (Sobolev gradient losses have shown 5-15% improvements on PDE solution
tasks; most benefit expected on surface pressure gradient accuracy specifically).

### Complexity
M (requires approximating spatial gradients on the irregular mesh — non-trivial;
simplest approach uses finite differences with nearest-neighbor approximation)

### Implementation guidance
File: `train.py`, add gradient loss after forward pass.

Simplest viable implementation using finite differences on node positions:
```python
def approx_gradient_loss(pred, y_norm, x_pos, mask):
    """
    pred: [B, N, 3] — predicted normalized fields
    x_pos: [B, N, 2] — node positions (x[:,:,0:2])
    Approximate spatial gradients via finite differences on nearby nodes.
    """
    # Simple version: use random pairs of nearby nodes
    B, N, C = pred.shape
    # Sample pairs of nodes with small spatial distance
    # Gradient ~ (f2 - f1) / (pos2 - pos1) for neighboring pairs
    # This is expensive; prefer the learned-divergence approach (Idea 7)
    pass
```

RECOMMENDED SIMPLER VERSION: Use finite differences only along the foil surface
(surface nodes are ordered by arc-length — `saf` dims 2-3 provide the arc-length
parameterization). For surface nodes, pressure gradient along the arc-length is
well-defined:

```python
# On surface nodes only: sort by saf (arc-length), compute Δp/Δs
surf_pred_p = pred[surf_mask][..., 2]   # surface pressure predictions
surf_y_p = y_norm[surf_mask][..., 2]    # surface pressure targets
# Approximate dp/ds: consecutive differences along arc-length
dp_ds_pred = torch.diff(surf_pred_p, dim=-1)
dp_ds_true = torch.diff(surf_y_p, dim=-1)
grad_loss = F.mse_loss(dp_ds_pred, dp_ds_true)
loss = loss + cfg.grad_weight * grad_loss
```

Add to Config: `grad_weight: float = 0.1`.

Caveat: surface nodes may not be globally ordered by arc-length across all batches —
verify the ordering assumption from data before committing to this approach.

### Citations
- Czarnecki et al. "Sobolev Training for Neural Networks" (2017), NeurIPS 2017.
  Section 2 (value+gradient bounds), Table 1 (PDE regression improvements).
- Applied to CFD: "Gradient-based auxiliary losses for aerodynamic field prediction"
  (see search results: Sobolev gradient NODEs, arxiv 2506.04463).

### Taste rubric
- Mode: Frontier refinement (adds gradient supervision not present in baseline)
- Mechanistic grounding: 3 — physics of pressure-lift coupling is direct; Sobolev
  bounds are proven; the arc-length ordering assumption needs verification
- Research-state value: 3 — would distinguish gradient accuracy from field accuracy
  as limiting factors for surface pressure MAE
- Execution value: 2 — moderate cost; the surface-only version is the cheap probe
  to try first before full spatial gradient computation

---

## Idea 11: Learned Log-Re Embedding (Replace Scalar Feature with Sinusoidal MLP)

### Hypothesis
Reynolds number spans ~100K to ~5M (about 1.7 decades), and is provided as a single
scalar `log(Re)` after normalization (dim 13). Per-sample y-std varies by ~10× within
a single domain purely due to Re. A single scalar cannot capture the nonlinear
flow-regime transitions (laminar → turbulent boundary layer, stall onset, vortex
shedding regime changes). Replacing scalar log(Re) with a sinusoidal embedding
(like positional encoding in transformers) or a small learned MLP embedding expands
the Re representation to a higher-dimensional space where the model can learn
regime-specific features via attention.

### Predicted delta on val_avg/mae_surf_p
-3% to -8% (moderate improvement expected; the main bottleneck may be model capacity
and loss formulation rather than Re representation, but this is a cheap test).

### Complexity
S (feature engineering change in train.py; no architecture modification)

### Implementation guidance
File: `train.py`, in the input normalization section (around lines 460–480).

Add sinusoidal embedding for the Re feature before feeding to the model:

```python
def sinusoidal_re_embedding(log_re_norm, d_embed=8):
    """
    log_re_norm: [B, N, 1] — normalized log(Re) scalar
    Returns: [B, N, d_embed] — sinusoidal embedding
    """
    freqs = torch.arange(d_embed // 2, device=log_re_norm.device).float()
    freqs = 2 ** freqs  # octave frequencies
    args = log_re_norm * freqs.view(1, 1, -1)  # [B, N, d_embed//2]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, N, d_embed]

# Replace x_norm[:,:,13:14] with the embedding
re_embed = sinusoidal_re_embedding(x_norm[:, :, 13:14], d_embed=8)
x_aug = torch.cat([x_norm[:, :, :13], re_embed, x_norm[:, :, 14:]], dim=-1)
# Note: x_aug is now [B, N, 24-1+8 = 31] — update model fun_dim accordingly
```

Model config change: `fun_dim = X_DIM - 2 + 7  # = 29` (adds 7 extra dims: 8 embed - 1 scalar).

Alternative: simply add the embedding as extra features without removing the original
scalar (easier to implement, slightly more memory):
`fun_dim = 24 - 2 + 8 = 30`

Add to Config: `re_embed_dim: int = 8`.

### Citations
- Positional encoding design: Vaswani et al. "Attention is All You Need" (2017).
  Sinusoidal embeddings for ordered sequences.
- Applied to Re: "Recipe for geometry-aware mesh transformers" — notes that scalar
  physics conditions underrepresent flow regime information.

### Taste rubric
- Mode: Diagnostic (tests whether Re representation is limiting)
- Mechanistic grounding: 3 — the nonlinear regime transition at Re~1M (laminar-turbulent)
  is well-documented; a scalar cannot represent this boundary cleanly
- Research-state value: 3 — cheap to run; clear before/after comparison on Re-stratified
  val split specifically (`val_re_rand`)
- Execution value: 4 — very low cost; particularly targeted at `val_re_rand` OOD metric

---

## Idea 12: Re-Based Curriculum Learning (Progressive High-Re Training)

### Hypothesis
High-Re samples (Re~5M) produce y-values ~10× larger than low-Re samples (Re~100K)
within the same domain. Starting training with these extreme samples causes large
initial gradients that may push the model into a suboptimal local minimum before it
has learned the basic flow structure at moderate Re. Curriculum learning — beginning
with moderate-Re samples and progressively introducing higher-Re samples — may allow
the optimizer to first converge on the shared geometric/flow structure before
adapting to extreme-Re regimes. This is analogous to difficulty-based curriculum
in language models (easy examples first, hard examples later).

### Predicted delta on val_avg/mae_surf_p
-3% to -7% (curriculum learning is helpful in high-variance regimes; the specific
benefit depends on how much the high-Re extremes currently destabilize training).

### Complexity
S (modify the WeightedRandomSampler schedule in train.py; no architecture change)

### Implementation guidance
File: `train.py`, sampler initialization (around lines 440–455).

Re information is available from feature dim 13 (`log(Re)`, normalized). Implement
epoch-dependent Re filtering:

```python
class ReCurriculumSampler:
    def __init__(self, dataset, re_feature_dim=13, stats=None):
        # Compute per-sample Re quantile from unnormalized log(Re)
        log_re = torch.tensor([ds[0][0, re_feature_dim].item() for ds in dataset])
        log_re_actual = log_re * stats["x_std"][re_feature_dim] + stats["x_mean"][re_feature_dim]
        self.re_quantile = (log_re_actual - log_re_actual.min()) / (log_re_actual.max() - log_re_actual.min())

    def get_weights(self, epoch, max_epoch, curriculum_end_epoch=10):
        # Linearly expand the Re ceiling from 50th percentile to 100th over curriculum_end_epoch
        re_ceiling = min(1.0, 0.5 + 0.5 * epoch / curriculum_end_epoch)
        weights = (self.re_quantile <= re_ceiling).float()
        weights = weights + 0.1  # always keep some high-Re samples
        return weights / weights.sum()
```

Add to Config: `curriculum_end_epoch: int = 10` (after epoch 10, all samples included equally).

This interacts with the domain balanced sampler — combine the curriculum weights with
`sample_weights` from `load_data()` via element-wise multiplication before normalizing.

### Citations
- Curriculum learning: Bengio et al. "Curriculum Learning" (2009), ICML.
- Applied to PDEs: "Progressive training for neural PDE solvers" — difficulty
  scheduling for Navier-Stokes from laminar to turbulent.

### Taste rubric
- Mode: Diagnostic (tests whether Re-scale variance destabilizes early training)
- Mechanistic grounding: 2 — curriculum learning motivation is clear but the specific
  Re-scale hypothesis needs evidence; the normalized loss may already partially equalize
  Re contributions
- Research-state value: 3 — would tell us whether early training trajectory matters;
  cheap to verify by checking whether validation loss initially improves faster
- Execution value: 3 — low cost; directly measurable effect on training dynamics

---

## Idea 13: Attention Entropy Regularization (Preventing Slice Collapse)

### Hypothesis
A cheaper alternative to Transolver++ slice reparameterization: add an entropy
regularization term that penalizes low-entropy (uniform) slice weight distributions.
`H(w) = -Σ_k w_k log(w_k)` is maximized when weights are uniform — we want the
OPPOSITE: high entropy (diverse slices) or controlled diversity. The correct penalty
is to maximize the entropy of the ASSIGNMENT distribution across nodes for each slice
(nodes should be diversely assigned), while minimizing entropy of each node's slice
weight (each node should commit to a slice). This is a Minimum Description Length /
VQ-like objective that drives the slice mechanism toward hard assignments.

### Predicted delta on val_avg/mae_surf_p
-3% to -7% (cheaper version of Ada-Temp; likely smaller gains than full Transolver++
but with near-zero implementation cost).

### Complexity
S (add ~5 lines to the loss computation; no architecture change)

### Implementation guidance
File: `train.py`, add entropy regularization in the `PhysicsAttention.forward` method
or in the main loss computation.

Access to slice weights from outside PhysicsAttention requires either:
1. Modifying the model to return `slice_weights` alongside predictions (invasive).
2. Adding the entropy loss INSIDE the PhysicsAttention class.

Recommended: Option 2 — add to `PhysicsAttention.forward`:
```python
# After computing slice_weights [B, heads, N, slice_num]:
# Per-node entropy: maximize (each node should commit to one slice)
node_entropy = -(slice_weights * (slice_weights + 1e-8).log()).sum(-1)  # [B, heads, N]
node_entropy_loss = node_entropy[mask.unsqueeze(1).expand_as(node_entropy)].mean()

# Store for external access
self._entropy_loss = node_entropy_loss  # accessed in training loop

# In training loop, after model forward:
entropy_reg = sum(
    blk.attn._entropy_loss
    for blk in model.blocks
    if hasattr(blk.attn, '_entropy_loss')
)
loss = loss + cfg.entropy_weight * entropy_reg
```

Add to Config: `entropy_weight: float = 0.01` (start small; entropy penalty can
destabilize training if too large).

Note: minimizing per-node entropy (making nodes commit to slices) is the correct
direction. Do NOT maximize node entropy — that would create the uniform collapse.

### Citations
- Transolver++: Section 3 (slice degeneration analysis), motivates need for diversity.
- VQ-VAE: van den Oord et al. "Neural Discrete Representation Learning" (2017) —
  commitment loss is structurally similar.
- Information bottleneck: entropy regularization for representation learning.

### Taste rubric
- Mode: Diagnostic (tests slice collapse hypothesis with minimal architecture change)
- Mechanistic grounding: 3 — slice collapse is documented in Transolver++; entropy
  penalty is a clean mechanism to prevent it; implementation can be verified by
  monitoring actual slice weight entropy during training
- Research-state value: 3 — cheaper diagnostic than Idea 1; if it works, suggests
  architecture change (Ada-Temp) is warranted; if not, attention collapse is not limiting
- Execution value: 4 — very low cost diagnostic targeting a well-defined failure mode

---

## Idea 14: Separate Surface Decoder Head

### Hypothesis
The current model uses a single output projection to predict [Ux, Uy, p] for all nodes.
Surface nodes have fundamentally different physical characteristics from volume nodes
(they are on the solid boundary, with Ux≈0, Uy≈0 for no-slip boundary conditions, and
pressure determined by the Bernoulli equation rather than the full NS equations).
A separate, dedicated MLP decoder for surface nodes — which sees the same hidden
representation but has independent weights — allows the model to specialize its
output distribution for the aerodynamically critical boundary layer without contaminating
the volume prediction.

### Predicted delta on val_avg/mae_surf_p
-4% to -10% (separate heads for structurally different prediction targets is standard
in multi-task learning; the no-slip constraint and Bernoulli physics make surface
prediction qualitatively different from volume prediction).

### Complexity
M (requires modifying the model's output stage to route surface vs. volume nodes
through different decoders; is_surface must be passed to the model)

### Implementation guidance
File: `train.py`, modify `Transolver` class and model contract (around lines 200–270).

1. Add a surface-specific decoder to the `Transolver.__init__`:
   ```python
   self.surf_head = nn.Sequential(
       nn.Linear(n_hidden, n_hidden),
       nn.GELU(),
       nn.Linear(n_hidden, out_dim)
   )
   # self.head (existing) becomes the volume head
   ```

2. Modify `Transolver.forward` to accept `is_surface`:
   ```python
   def forward(self, data):
       x = data["x"]
       is_surf = data.get("is_surface", None)
       # ... existing processing ...
       h = self.final_layer(x)  # [B, N, n_hidden]
       preds = self.head(h)     # [B, N, 3] — volume predictions
       if is_surf is not None:
           surf_preds = self.surf_head(h)  # [B, N, 3] — surface predictions
           preds = torch.where(is_surf.unsqueeze(-1), surf_preds, preds)
       return {"preds": preds}
   ```

3. Update the training loop to pass `is_surface` to the model:
   ```python
   pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
   ```

Note: This changes the model contract slightly. Validation must also pass `is_surface`.

### Citations
- Multi-task learning theory: Caruana (1997) "Multitask Learning" — shared trunk,
  separate heads for structurally different outputs.
- Applied to CFD: AB-UPT uses separate surface/volume branches; shows consistent
  improvement on surface-specific metrics.

### Taste rubric
- Mode: Tier shift (architectural change targeting the primary metric's data domain)
- Mechanistic grounding: 3 — the no-slip BC and Bernoulli regime are distinct from
  bulk NS; separate decoder is a standard multi-task pattern with clear motivation
- Research-state value: 3 — would reveal whether shared output projection is a bottleneck
  for surface-specific prediction
- Execution value: 3 — moderate cost; directly targets the surface pressure metric

---

## Idea 15: Stochastic Depth (LayerDrop) Regularization

### Hypothesis
With only 5 transformer layers and moderate model size (n_hidden=128), the Transolver
may be underfitting on the cruise and OOD camber splits while overfitting on the
dominant raceCar domain. Stochastic depth — randomly dropping entire transformer
layers with probability p_drop during training — acts as a strong implicit regularizer
that makes each layer learn to work independently (since it cannot rely on the previous
layer's output always being present). This is distinct from dropout within layers;
stochastic depth is a structural regularizer. It has been shown effective for
transformer training and is particularly useful when the model is deeper than necessary
for the task.

### Predicted delta on val_avg/mae_surf_p
-2% to -5% (regularization gains are typically modest; main benefit expected on OOD
camber splits specifically).

### Complexity
S (add stochastic depth to TransolverBlock; ~5 lines)

### Implementation guidance
File: `train.py`, modify `TransolverBlock.forward` (around lines 139–164).

```python
def forward(self, x, fx, T, mask=None):
    # Stochastic depth: during training, drop this block with probability p_drop
    if self.training and hasattr(self, 'p_drop') and torch.rand(1).item() < self.p_drop:
        return x, fx  # skip this block entirely
    # ... existing forward pass ...
```

Add `p_drop: float = 0.1` parameter to `TransolverBlock.__init__`.
Set linearly from 0 at first layer to p_drop at last layer (standard stochastic depth
schedule: `p_drop_i = p_drop * i / (n_layers - 1)`).

Add to Config: `stochastic_depth_p: float = 0.1`.

At inference, all layers are always active (no drop). This is equivalent to an
ensemble over sub-architectures.

### Citations
- Huang et al. "Deep Networks with Stochastic Depth" (2016), ECCV.
- Applied to transformers: "LayerDrop" (Fan et al. 2020) — dropping entire
  transformer layers during training; consistent improvements in language models.

### Taste rubric
- Mode: Frontier refinement (regularization for potential overfitting)
- Mechanistic grounding: 2 — stochastic depth is well-established but the evidence
  for overfitting in the current Transolver on this dataset is indirect
- Research-state value: 2 — would provide some signal but heavily confounded by
  other factors
- Execution value: 3 — trivial to implement; test specifically on OOD camber splits

---

## Idea 16: MNO-Style Local KNN Graph Attention Augmentation

### Hypothesis
The current Transolver uses only global slice attention — every node attends to the
same set of global physics tokens. MNO (Multiscale Neural Operator, ICLR 2026) showed
that adding a local graph attention branch (each node attends to its k-nearest
neighbors in physical space) provides complementary information to global attention
and reduces CFD errors by 5-40% on meshes up to 300K nodes. The physical motivation
is that near-wall pressure gradients (which drive surface pressure errors) are
dominated by local flow physics, not global ones. Local KNN attention can capture
these without the global pooling smoothing.

### Predicted delta on val_avg/mae_surf_p
-8% to -18% (MNO's CFD gains are in the right ballpark; local attention is particularly
relevant for boundary layer pressure gradients which are highly local phenomena).

### Complexity
L (requires building KNN graph per batch at training time, or precomputing neighborhood
indices; non-trivial for variable-size meshes)

### Implementation guidance
File: `train.py`, add local attention alongside `PhysicsAttention` in `TransolverBlock`.

Build KNN graph at training time (not precomputed, since batch geometry varies):
```python
from torch_cluster import knn_graph  # requires torch-cluster; add to pyproject.toml

def local_attention_forward(h, pos, k=16, mask=None):
    """
    h: [B, N, dim] — node features
    pos: [B, N, 2] — node (x, z) positions
    Returns: [B, N, dim] — locally-attended features
    """
    B, N, D = h.shape
    out = []
    for b in range(B):
        # Build KNN graph for real nodes only
        real_mask = mask[b]  # [N]
        pos_b = pos[b, real_mask]  # [N_real, 2]
        h_b = h[b, real_mask]     # [N_real, D]
        # KNN graph: each node -> k nearest neighbors
        edge_idx = knn_graph(pos_b, k=k, loop=False)  # [2, N_real*k]
        # Simple attention: mean aggregate over neighbors
        src, dst = edge_idx
        h_agg = scatter_mean(h_b[src], dst, dim=0, dim_size=len(pos_b))
        # Pad back to N
        h_full = torch.zeros(N, D, device=h.device)
        h_full[real_mask] = h_agg
        out.append(h_full)
    return torch.stack(out)  # [B, N, D]
```

CAUTION: KNN graph construction at batch time for 242K-node meshes is expensive.
Use `k=8` or `k=16` (not 32+). Consider precomputing KNN indices and storing in the
.pt files — but that requires modifying data loader (read-only). Compromise: cache
the KNN graph per sample during the first epoch and reuse.

Dependency: `torch-cluster` or `torch-geometric` — add to `pyproject.toml`.

### Citations
- MNO: "Multiscale Neural Operator for Irregular Mesh CFD" (ICLR 2026). Table 3
  (5-40% CFD error reduction on 300K-node meshes using multiscale attention).
- KNN graph attention: GNN literature — message passing over spatial neighborhoods.

### Taste rubric
- Mode: Tier shift (new architectural branch targeting local physics)
- Mechanistic grounding: 3 — MNO evidence is directly relevant (same mesh scale,
  same CFD prediction task); local attention for boundary layer is physically motivated
- Research-state value: 3 — MNO gains are large; if replicated, suggests current
  Transolver misses local physics; failure would rule out locality as the bottleneck
- Execution value: 2 — high implementation cost; start with a fast approximation
  (mean pooling over nearest neighbors without learned attention weights) before
  full KNN attention

---

## Idea 17: Domain Indicator Embedding (Learned Domain Token)

### Hypothesis
The three training domains (raceCar single, raceCar tandem, cruise) have fundamentally
different physical regimes. The model currently must infer domain identity from
geometric features (gap=0 → single-foil; AoA range distinguishes raceCar from cruise).
An explicit learned domain embedding — a 3-class one-hot indicator passed as extra
input features or as a global conditioning token — gives the model direct access to
domain information, enabling domain-specific attention patterns and feature scaling.
This is simpler than per-domain normalization (Idea 8) but addresses the same root cause.

### Predicted delta on val_avg/mae_surf_p
-2% to -6% (moderate; the model can already infer domain from geometry features, so
the explicit embedding is a shortcut not a new information source — but shortcuts
can matter for optimization efficiency).

### Complexity
S (add 3 extra dimensions to input features; minimal architecture change)

### Implementation guidance
File: `train.py`, in the input preparation section (around lines 460–480).

Domain identity can be inferred at training time from the x features:
```python
def get_domain_indicator(x_raw):
    """
    x_raw: [B, N, 24] — unnormalized input features
    Returns: [B, N, 3] — one-hot domain indicator
    """
    B, N = x_raw.shape[:2]
    # Single-foil: gap (dim 22) = 0 and stagger (dim 23) = 0
    is_single = (x_raw[:, 0, 22].abs() < 1e-6) & (x_raw[:, 0, 23].abs() < 1e-6)
    # Cruise: AoA can be positive (dim 14 > 0 in radians)
    has_positive_aoa = (x_raw[:, :, 14] > 0.01).any(dim=1)
    # raceCar tandem: tandem + no positive AoA
    domain = torch.zeros(B, 3, device=x_raw.device)
    domain[is_single, 0] = 1.0            # raceCar single
    domain[~is_single & ~has_positive_aoa, 1] = 1.0  # raceCar tandem
    domain[~is_single & has_positive_aoa, 2] = 1.0   # cruise
    # Broadcast to all nodes
    return domain.unsqueeze(1).expand(B, N, 3)  # [B, N, 3]

# Concatenate to input features
domain_ind = get_domain_indicator(x)  # raw, before normalization
x_norm = torch.cat([x_norm, domain_ind], dim=-1)  # [B, N, 27]
```

Model config change: `fun_dim = X_DIM - 2 + 3 = 25` (adds 3 indicator dims).

Alternative: use a learned domain embedding matrix instead of one-hot:
```python
self.domain_embed = nn.Embedding(3, 8)  # 3 domains, 8-dim embedding
```
This allows the model to learn a domain representation rather than using a fixed indicator.

### Citations
- Conditional transformers: FiLM, AdaLN — domain/style conditioning via learned
  embeddings in generative models.
- Applied to multi-domain CFD: AB-UPT uses domain conditioning tokens for
  automotive vs. aerospace geometries.

### Taste rubric
- Mode: Frontier refinement (provides domain shortcut to reduce inductive bias)
- Mechanistic grounding: 2 — the model can already infer domain; the explicit
  embedding is an optimization shortcut, not new information
- Research-state value: 2 — small expected gain; mainly useful as a complement
  to per-domain normalization (Idea 8)
- Execution value: 4 — trivial to implement; risk-free addition

---

## Idea 18: Gradient Clipping with Adaptive Norm (Per-Layer Gradient Monitoring)

### Hypothesis
The baseline uses no gradient clipping (or uses PyTorch default). With variable mesh
sizes (74K–242K nodes) and a 10× surface weight multiplier, individual batches
containing large-mesh high-Re samples can produce extremely large gradient norms that
destabilize training. Adaptive per-layer gradient norm clipping — clipping each layer's
gradient to its own running average norm rather than a fixed global threshold — prevents
sporadic gradient spikes from surface samples disrupting the learned representations in
early layers. This is the approach used in modern LLM training (gradient norm tracking
is standard in GPT-style training loops).

### Predicted delta on val_avg/mae_surf_p
-1% to -4% (gradient clipping is a stabilization technique; gains are indirect and
depend on how much instability exists in current training).

### Complexity
S (add ~5 lines to optimizer step; no architecture change)

### Implementation guidance
File: `train.py`, in the training loop (around line 500, before `optimizer.step()`).

```python
# Add to Config:
# grad_clip: float = 1.0

# Before optimizer.step():
if cfg.grad_clip > 0:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    # Log grad_norm for monitoring
    wandb.log({"train/grad_norm": grad_norm.item()}, step=global_step)
```

Standard gradient clipping is already common practice. The DIAGNOSTIC value here is
logging `train/grad_norm` — if the norm is frequently >10, that's evidence of
instability from surface weight. Start with `grad_clip=1.0`.

If grad norm spikes correlate with training loss spikes, try `grad_clip=0.5`.
If grad norm is always < 1.0, clipping is not the bottleneck and this can be ruled out.

Add to Config: `grad_clip: float = 1.0`.

### Citations
- Standard ML practice: Pascanu et al. "On the difficulty of training recurrent neural
  networks" (2013) — gradient clipping for stability.
- Applied to transformers: all modern LLM training recipes include gradient norm logging
  and clipping as standard practice.

### Taste rubric
- Mode: Diagnostic (tests whether gradient instability is present and limiting)
- Mechanistic grounding: 2 — gradient instability is plausible given surf_weight=10
  and variable mesh sizes, but no evidence it is currently occurring
- Research-state value: 3 — the diagnostic value (logging grad_norm) is independent
  of the gain; reveals a training dynamics fact regardless of outcome
- Execution value: 4 — near-zero cost; grad norm logging alone is valuable

---

## Summary Table

| # | Idea | Predicted Delta | Complexity | Taste: Mechanism | Taste: State Value | Taste: Exec Value |
|---|------|----------------|------------|-------------------|--------------------|-------------------|
| 1 | Ada-Temp Slice Reparameterization | -8% to -15% | M | 4 | 4 | 3 |
| 2 | SOAP Optimizer | -5% to -20% | S | 3 | 4 | 4 |
| 3 | Cautious AdamW | -2% to -8% | S | 3 | 3 | 4 |
| 4 | GFocal Nyström Global Attention | -10% to -18% | L | 3 | 3 | 2 |
| 5 | Amortized Mesh Subset Training | -3% to -10% | M | 3 | 3 | 3 |
| 6 | AoA Reflection Augmentation | -3% to -8% | S | 3 | 3 | 4 |
| 7 | Divergence-Free Auxiliary Loss | -5% to -12% | M | 3 | 3 | 2 |
| 8 | Per-Domain Normalization Stats | -3% to -8% | S | 2 | 3 | 2 |
| 9 | EMA Model Weights | -1% to -5% | S | 2 | 2 | 4 |
| 10 | Sobolev Gradient-Matching Loss | -5% to -12% | M | 3 | 3 | 2 |
| 11 | Log-Re Sinusoidal Embedding | -3% to -8% | S | 3 | 3 | 4 |
| 12 | Re-Based Curriculum Learning | -3% to -7% | S | 2 | 3 | 3 |
| 13 | Attention Entropy Regularization | -3% to -7% | S | 3 | 3 | 4 |
| 14 | Separate Surface Decoder Head | -4% to -10% | M | 3 | 3 | 3 |
| 15 | Stochastic Depth (LayerDrop) | -2% to -5% | S | 2 | 2 | 3 |
| 16 | MNO Local KNN Graph Attention | -8% to -18% | L | 3 | 3 | 2 |
| 17 | Domain Indicator Embedding | -2% to -6% | S | 2 | 2 | 4 |
| 18 | Adaptive Gradient Clipping | -1% to -4% | S | 2 | 3 | 4 |

---

## Top Priority Recommendations

**Highest EV / lowest cost diagnostic experiments to run first:**

1. **SOAP Optimizer (Idea 2)** — drop-in replacement, 2-14× PDE gains in literature,
   directly tests gradient conflict hypothesis from the surf+vol composite loss.
2. **Ada-Temp Slice Reparameterization (Idea 1)** — targets documented Transolver
   failure mode (slice collapse), published ablation shows ~10% gains on aerodynamics.
3. **Attention Entropy Regularization (Idea 13)** — cheap proxy for Ada-Temp,
   directly monitors the slice collapse diagnostic.
4. **AoA Reflection Augmentation (Idea 6)** — zero architecture change, doubles
   effective raceCar training data, tests whether geometric diversity is limiting.
5. **Log-Re Sinusoidal Embedding (Idea 11)** — trivial feature change, directly
   targets `val_re_rand` OOD split, tests whether Re representation is limiting.
