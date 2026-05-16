# Research Ideas — 2026-05-16 01:30

Generated after reviewing 13 PRs on `icml-appendix-charlie-pai2i-24h-r3`, sibling-branch history
(~56 experiments in pai2g/pai2h), yesterday's 8 hypotheses, and targeted web searches on neural
operators, physics-informed losses, and transformer scaling for CFD surrogates.

**Current baseline:** `val_avg/mae_surf_p = 87.62` (PR #3513, cosine T_max=20 + BF16 + Huber).
**Primary metric:** `val_avg/mae_surf_p` (lower is better).

---

## 1. Softmax Temperature Annealing in PhysicsAttention [HIGHEST PRIORITY]

**What it is:** Anneal the slice-assignment softmax temperature from τ=1.0 (soft, diffuse) down to
τ=0.1 (hard, crisp) over training, controlled by a cosine schedule synchronized with the LR cosine.

**Mechanism targeting observed bottleneck:** PhysicsAttention's quality depends on how sharply
nodes are clustered into physics-meaningful slices. At τ=1.0 each node smears across all G=64
slices — this softness helps early in training (exploration) but hurts late (prevents the model from
learning sharp boundary-layer vs. wake vs. freestream decompositions). Hard clustering at the end
enforces a clean physics partitioning. This directly targets the slice-token quality, not
downstream MLP or attention weights — it is a uniquely low-risk change (one float schedule, no
architectural surgery).

**External evidence:** Temperature annealing in discrete-latent models (VQ-VAE, DINO) consistently
improves the quality of learned discrete representations. The analogous mechanism here is
differentiable but the principle is identical: start soft for gradient flow, end hard for
representational quality.

**Not tried:** The baseline fixes `self.temperature = nn.Parameter(torch.ones(1) * temperature)`
as a learned scalar (initialized to 1.0 and trained by backprop). Annealing it externally as a
schedule — bypassing the learned parameter — has not been attempted in any PR.

**Implementation:**
```python
# In train loop, after optimizer step:
tau = 1.0 - 0.9 * (1 - math.cos(math.pi * epoch / cfg.cosine_t_max)) / 2
for m in model.modules():
    if hasattr(m, 'temperature'):
        m.temperature.data.fill_(tau)
```

**Taste rubric:** Mechanistic grounding 4 | Research-state value 4 | Execution value 4
**Mode:** Tier shift (targets PhysicsAttention core mechanism, not a hyperparameter sweep)
**Stop condition:** If val_avg/mae_surf_p does not improve vs. fixed-temperature baseline, and
learned temperature has already saturated near 0.1 anyway (check with a logging hook), rule out
and move on.

---

## 2. mlp_ratio 2→4 with n_hidden=128 [QUICK SCREEN]

**What it is:** Increase the FFN expansion ratio in each TransolverBlock from 2 to 4, matching
standard transformer practice (GPT-2, ViT, etc.). This doubles the width of each MLP block from
256 to 512 units, with no change to attention heads, layers, or hidden dim.

**Mechanism:** The model's MLP bottleneck (ratio=2) is non-standard — most transformer literature
uses ratio=4. The FFN is the primary function-approximation component per block; the attention
aggregates geometry but the MLP transforms features. With n_hidden=128, ratio=4 gives 128×4=512
intermediate dim, still well within VRAM.

**External evidence:** Multiple ablation papers (e.g., Liu et al. 2022 on Swin Transformer V2,
PDE-Transformer 2025) show FFN ratio=4 consistently outperforms ratio=2 at the same parameter
budget, suggesting the current model is MLP-underpowered relative to its attention capacity.

**Not tried:** sibling branches tried 256 hidden/8 layers/8 heads as a bundle but never isolated
mlp_ratio at the current n_hidden=128 baseline.

**Config change:**
```python
model_config = dict(..., mlp_ratio=4, ...)
```

**VRAM check:** 128 hidden, ratio=4 adds ~(128×4×128 - 128×2×128) × 5 layers × 2 (in/out) =
~655K params — negligible VRAM impact, well within the 63 GB headroom.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 4
**Mode:** Frontier refinement (trivial config change, high information density per GPU-hour)
**Stop condition:** If val loss moves <1% after 10 epochs vs. baseline at same epoch, abandon.

---

## 3. n_head 4→8 with n_hidden=128 [QUICK SCREEN]

**What it is:** Double attention heads from 4 to 8 while holding n_hidden=128 fixed (dim_head goes
from 32 to 16). Each head specializes in a narrower feature subspace; 8 heads may better decompose
the physical fields (Ux, Uy, p each have different spatial structure).

**Mechanism:** Head diversity hypothesis: with 4 heads at dim_head=32, the model may be unable to
simultaneously track boundary-layer gradients (fine spatial scale), wake structure (medium scale),
and far-field pressure (global scale) across its G=64 slices. More heads with narrower dim_head
allow specialization without changing total parameter count.

**Key distinction from sibling-branch attempt:** sibling branches tried n_head=8 only with
n_hidden=256 (as a bundle). At n_hidden=128 with dim_head=16, this is an untested point in the
design space — the head count effect at the current baseline size is unknown.

**Not tried:** Confirmed distinct from all 13 current PRs and sibling history.

**Config change:**
```python
model_config = dict(..., n_head=8, ...)
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 4
**Mode:** Frontier refinement
**Stop condition:** If val_avg/mae_surf_p does not improve after 20 epochs vs. baseline, close.

---

## 4. Stochastic Weight Averaging (SWA) over Cosine Plateau [DISTINCT FROM EMA #3241]

**What it is:** At epoch 15 (after the cosine annealing main cycle), restart LR to 10% of peak
and collect uniform weight averages every 2 epochs through epoch 25-30, then evaluate the SWA
model. Unlike EMA (#3241, in-flight), SWA averages a small number of discrete checkpoints rather
than maintaining a continuous exponential trace.

**Mechanism:** SWA finds flatter minima that generalize better under distribution shift (Izmailov
et al. 2018). The val_geom_camber splits are geometry-OOD; flatter minima help OOD generalization.
After cosine annealing reaches near-zero LR, the model sits at a sharp minimum — SWA perturbs it
with small LR restarts and averages the resulting trajectory, trading slight training-set accuracy
for generalization.

**Key distinction from #3241 (EMA):** EMA is a continuous exponential trace (decay=0.9999) of
all training steps. SWA takes a small number of snapshots (3-8) on a high-LR plateau after the
main cosine cycle ends. The two methods live in different parts of the optimization trajectory.
Both could be run; SWA is more appropriate here because the cosine T_max=20 already provides the
main optimization trajectory, and SWA leverages the remaining budget (epochs 20-50) differently.

**External evidence:** PyTorch's `torch.optim.swa_utils.AveragedModel` is battle-tested. SWA gave
+0.5-1.5% accuracy on CIFAR-10/100 in the original paper; it was later shown to improve neural
operator generalization by ~2% on Darcy flow in PINO experiments.

**Implementation sketch:**
```python
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
# After epoch 15, switch to SWA:
swa_model.update_parameters(model)
swa_scheduler.step()
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 4 | Execution value 3
**Mode:** Diagnostic (tests whether the current sharp minimum is the bottleneck for OOD splits)
**Stop condition:** If val_geom_camber_rc and val_geom_camber_cruise do not improve while
val_single_in_dist degrades, SWA overfits to flatter but wrong basin — close.

---

## 5. Incompressibility Soft Constraint Loss (Divergence-Free Auxiliary)

**What it is:** Add a soft physics constraint loss term penalizing ∇·u ≠ 0 at interior nodes.
For 2D incompressible flow: ∂Ux/∂x + ∂Uy/∂z = 0. Estimated via finite differences using the
node position coordinates (x[:,0], x[:,1]) available in the input features.

**Mechanism:** The primary metric is surface pressure MAE, but accurate pressure requires accurate
velocity fields (via the Bernoulli/pressure-velocity coupling in RANS). If the model's Ux/Uy
predictions violate incompressibility, the pressure field will have systematic error. A soft
divergence penalty guides the model to learn physically self-consistent velocity fields without
requiring any changes to the architecture or data loader.

**Implementation notes:**
- FD stencil: for each node i, find nearest neighbors in x-space using precomputed kNN indices
  (or approximate via batch-level scatter with node coordinates). Simpler: use only surface nodes
  where the mesh is regular, or approximate ∂Ux/∂x ≈ Ux difference / position difference for
  adjacent nodes within each sample.
- Alternatively, use spectral estimation via the existing slice tokens: compute divergence in the
  slice-averaged latent space as a proxy.
- Loss weight λ_div=0.01 to 0.1 recommended; start at 0.01.

**Key risk:** Variable unstructured meshes make exact FD stencils non-trivial to batch. A simpler
approximation: penalize mean |pred_Ux - pred_Ux.detach()| correlation with nearby Uy gradients
using a soft proxy. Alternatively, the arxiv 2502.09692 (AB-UPT) approach uses a divergence-free
parameterization rather than a penalty loss — this is harder but architecturally cleaner.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 4 | Execution value 2
**Mode:** Tier shift (introduces physics constraint not currently in any PR)
**Stop condition:** If the divergence loss does not decrease within 5 epochs (meaning the model is
ignoring it), or if the primary metric regresses despite lower divergence (constraint fights data),
close. Implementation complexity is the primary risk here.

---

## 6. Cosine T_max Extension: 25 or 30

**What it is:** Increase `cosine_t_max` from 20 to 25 or 30. With a 30-minute wall clock and
BF16, the model typically reaches 25-30 epochs. T_max=20 means the LR approaches zero around
epoch 20 while training continues — the model coasts on near-zero LR for the remaining epochs.

**Mechanism:** Near-zero LR means no learning in the final 5-10 epochs. Extending T_max to 25-30
maintains a small but nonzero LR through the full training budget, allowing continued improvement
on the hard OOD samples (val_geom_camber splits reach their best later in training due to slower
convergence on unseen geometry).

**Key question:** Does the training run actually hit epoch 25-30 under the 30-min budget? If yes,
T_max=25 should strictly improve on T_max=20. If no (run terminates at epoch 20 due to time), then
this is a no-op. The student should log `epoch_count` to disambiguate.

**Implementation:**
```python
@dataclass
class Config:
    cosine_t_max: int = 25  # or 30
```

**Taste rubric:** Mechanistic grounding 2 | Research-state value 3 | Execution value 4
**Mode:** Frontier refinement (clean single-parameter change with clear hypothesis)
**Stop condition:** If training terminates before epoch 25 (budget), the hypothesis was moot.
If training runs past epoch 25 and val does not improve vs. T_max=20, close.

---

## 7. Scale-Consistency Regularization for Re Generalization

**What it is:** Inspired by arxiv 2507.18813 (Scale-Consistent Learning). For each training batch,
sample a random Re-scaling factor α∈[0.5, 2.0], construct a "rescaled" input by multiplying
log(Re) (dim 13) by α and rescaling targets by α^2 (kinematic pressure scales as Re^{-2} for
attached boundary layers), and add a consistency loss penalizing the difference between
`model(x_scaled)` and `α²·model(x)`.

**Mechanism:** val_re_rand is a stratified holdout across all Re values — the model struggles on
Re combinations it hasn't seen precisely. Re-consistency training forces the model to learn
Re-equivariant representations: if you scale Re, the solution must scale accordingly. This is a
form of self-supervised augmentation that does not require new data.

**Key caveat:** The exact scaling law (α^2 for pressure) is only exact for laminar flows with
Re-invariant geometry. For turbulent flows (Re > ~100K) the scaling is approximate. Treating this
as a soft regularizer (weight 0.01-0.1) rather than a hard constraint handles the approximation.

**Implementation sketch:**
```python
alpha = torch.empty(B, 1, 1).uniform_(0.5, 2.0).to(device)
x_scaled = x_norm.clone()
# Rescale log(Re) feature (dim 13)
x_scaled[..., 13] = x_norm[..., 13] * alpha.squeeze(-1)
with torch.no_grad():
    pred_scaled = model({"x": x_scaled})["preds"]
pred_ref = model({"x": x_norm})["preds"]
# Pressure (dim 2) should scale as alpha^2; velocity (dims 0-1) as alpha
scale_vec = torch.cat([alpha, alpha, alpha**2], dim=-1)  # [B, 1, 3]
consistency_loss = F.mse_loss(pred_scaled, pred_ref * scale_vec)
loss = main_loss + 0.05 * consistency_loss
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 4 | Execution value 3
**Mode:** Tier shift (novel physics-informed regularizer not in any prior PR)
**Stop condition:** If val_re_rand does not improve while other splits hold, the Re-equivariance
prior is not the binding constraint. If consistency_loss fails to decrease (<5 epochs), the
scaling law assumption is too inaccurate for BF16 training — close.

---

## 8. Gradient Accumulation (effective batch size 8 or 16)

**What it is:** Accumulate gradients over N_accum=2 or 4 steps before calling `optimizer.step()`,
giving effective batch size 8 or 16 without increasing VRAM. This requires adjusting LR scaling
accordingly (LR ∝ sqrt(effective_batch_size) per linear scaling rule, so lr=5e-4 × sqrt(2)≈7e-4
for N_accum=2).

**Mechanism:** Variable mesh sizes (74K-242K nodes) cause gradient variance to be high when
batch_size=4 and sample sizes vary 3×. Larger effective batch smooths gradient estimates and
reduces variance-driven oscillations, potentially improving convergence on the harder OOD samples.

**Key detail:** Must zero_grad only every N_accum steps, and divide loss by N_accum before
backward:
```python
loss = loss / cfg.grad_accum_steps
loss.backward()
if (step + 1) % cfg.grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

Must also adjust LR: `lr = 5e-4 * math.sqrt(cfg.grad_accum_steps)`.

**Taste rubric:** Mechanistic grounding 2 | Research-state value 3 | Execution value 3
**Mode:** Frontier refinement
**Stop condition:** If val_avg/mae_surf_p does not improve by >1% vs. batch_size=4 baseline, the
gradient variance hypothesis was wrong for this problem — close.

---

## 9. Pre-LayerNorm Architecture (Pre-LN)

**What it is:** Move LayerNorm before the attention and FFN sublayers (Pre-LN) rather than after
(Post-LN, current default). Modern transformers (GPT-3, LLaMA, all frontier LLMs) universally use
Pre-LN for training stability and gradient flow.

**Mechanism:** Post-LN (used in original Transolver) suffers from vanishing gradients in deep
layers because the normalization is applied after the residual add — the gradient must pass through
the normalization before reaching earlier layers. Pre-LN normalizes the residual branch only,
leaving the skip connection unnormalized, which stabilizes gradient magnitudes across depth.

**Why it might help here:** The model's 5-layer depth is modest, but with BF16 and variable mesh
sizes, gradient scale varies significantly across batches. Pre-LN should reduce this variance and
potentially allow a slightly higher LR.

**Implementation:** In each TransolverBlock, move `self.norm1` and `self.norm2` calls to wrap
the sublayer inputs, not outputs. If Transolver already implements this, it is a no-op — student
should verify by reading the TransolverBlock forward pass.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 3
**Mode:** Diagnostic (tests whether gradient normalization placement is limiting convergence)
**Stop condition:** If val loss trajectory is not smoother or faster in first 10 epochs vs.
Post-LN baseline, the normalization placement is not the bottleneck — close.

---

## 10. AdamW β2 Reduction (0.999→0.99) for High-Re Extremes

**What it is:** Reduce AdamW's β2 from the default 0.999 to 0.99. β2 controls how quickly the
second-moment estimate adapts to gradient magnitude changes.

**Mechanism:** High-Re samples produce outlier gradients (y std up to 2077 vs. 164 for low-Re
cruise). With β2=0.999, the second-moment estimator has a time constant of ~1000 steps — it
adapts very slowly to sudden high-Re batch gradients, causing the effective LR to be artificially
suppressed after a high-Re sample. β2=0.99 reduces the time constant to ~100 steps, allowing
faster adaptation to gradient scale changes between domains. This is directly analogous to why
RMSprop typically uses β=0.99 for non-stationary objectives.

**Key risk:** Lower β2 increases optimizer noise; the LR may need a small reduction (5e-4→4e-4)
to compensate.

**Config change:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.99),  # default is (0.9, 0.999)
)
```

**Taste rubric:** Mechanistic grounding 3 | Research-state value 3 | Execution value 4
**Mode:** Frontier refinement (trivial one-line change with clear mechanism)
**Stop condition:** If val_re_rand (the high-Re-sensitive split) does not improve while others
hold, the second-moment adaptation hypothesis was wrong — close.

---

## 11. DSDF Feature Clipping (Preprocessing Outlier Reduction)

**What it is:** Before passing features through the model, clip the DSDF shape descriptor dims
4-11 at ±3σ (using the precomputed stats.json x_mean and x_std). DSDF values for surface-adjacent
nodes can have long tails that survive normalization.

**Mechanism:** The baseline normalizes inputs as `(x - x_mean) / x_std`. If dims 4-11 have
skewed distributions with outliers at ±5σ or beyond, the normalized features still contain extreme
values that the model's linear projections see as noise. Clipping at ±3σ is standard preprocessing
(e.g., used in many Kaggle tabular competition winners) and is a zero-parameter change.

**Implementation:**
```python
# In train.py, after x_norm = (x - stats["x_mean"]) / stats["x_std"]:
x_norm = x_norm.clamp(-3.0, 3.0)  # or clamp only dims 4-11
```

**Key risk:** Clipping may discard physically meaningful extremes for high-Re boundary layer nodes.
Student should log the fraction of values clipped per dim to verify this is not too aggressive.

**Taste rubric:** Mechanistic grounding 2 | Research-state value 3 | Execution value 4
**Mode:** Diagnostic (tests whether feature outliers are hurting normalization)
**Stop condition:** If the fraction of clipped values in dims 4-11 is <0.1% per dim, the tails
are not meaningful — the change is effectively a no-op. If val_avg/mae_surf_p does not improve
after 10 epochs, close.

---

## 12. Multi-Scale Slice Hierarchy (G_fine=64 + G_coarse=16)

**What it is:** Add a second PhysicsAttention branch per TransolverBlock that uses G_coarse=16
slices (global structure) alongside the existing G_fine=64 slices (local structure). Merge the
two branches via a learned scalar gate. Inspired by MNO (Multiscale Neural Operator, ICLR 2026)
which showed 5-40% improvement on 3D CFD with up to 300K points.

**Mechanism:** G=64 slices capture fine-grained physics partitioning (boundary layer, wake, etc.)
but may not capture global flow structure (stagnation point, circulation, far-field pressure).
G=16 provides a complementary coarse representation. The two views merged with a gate give the
model access to both scales without forcing attention to choose.

**Key complexity:** This is an architectural change requiring a second `PhysicsAttention` instance
per block. VRAM impact is ~2× per-block attention VRAM (still within budget). The gate can be
a single learnable scalar `σ(α)` per block.

**Taste rubric:** Mechanistic grounding 3 | Research-state value 4 | Execution value 2
**Mode:** Tier shift (architectural change, higher complexity)
**Stop condition:** If OOM occurs, reduce n_layers from 5 to 4. If val improvement is <2% after
full training, the complexity cost is not justified — close.

---

## 13. slice_num 32 (Reduce from 64)

**What it is:** Reduce the number of slice tokens from G=64 to G=32. This is the opposite
direction from previously-failed G=48 and G=12 trials, but at G=32 the focus is on regularization
rather than capacity.

**Mechanism:** With G=64 slices and N~100K nodes per sample, each slice gets ~1600 nodes on
average. In the context of mesh sizes up to 242K, some slices may become degenerate (very few
nodes) while others are over-crowded. G=32 gives ~3000 nodes per slice on average — a more
uniform distribution. This regularizes the attention and may reduce overfitting to training mesh
structures.

**Key distinction from prior attempts:** slice_num=48 was tried in sibling branches — but that
was likely paired with larger hidden dim. slice_num=32 at n_hidden=128 is an untested combination.

**Taste rubric:** Mechanistic grounding 2 | Research-state value 2 | Execution value 4
**Mode:** Frontier refinement (low risk, fast to screen)
**Stop condition:** If val_avg/mae_surf_p worsens vs. G=64 after 15 epochs, close.

---

## Experiment Decision Tree

```
Start: val_avg/mae_surf_p = 87.62

Round 1 (run first — quick screens, high expected information/compute):
  A: Temperature annealing [highest priority]
  B: mlp_ratio 4         [quick config change]
  C: n_head 8            [quick config change]
  D: AdamW β2=0.99       [quick config change]
  E: Cosine T_max=25     [quick config change]

  If A wins → merge, run multi-scale hierarchy (idea 12) as follow-up
  If A fails → rule out slice-sharpness as bottleneck; move to architecture
  If B+C both win → try n_hidden=192 (#3567) with ratio=4, n_head=8 bundled
  If B or C wins individually → merge and hold the other as follow-up
  If D wins → try further reduction to β2=0.95 as follow-up
  If D fails → optimizer adaptation is not the bottleneck

Round 2 (conditional on Round 1 results):
  F: SWA over plateau (if OOD splits are still lagging after Round 1 merges)
  G: Scale-consistency loss (if val_re_rand is still lagging)
  H: Pre-LN (if training instability observed — loss spikes, slow convergence)
  I: Divergence-free loss (only if implementation risk is manageable)

Round 3 (if plateau persists after Round 2):
  J: Multi-scale hierarchy (big architectural bet)
  K: DSDF clipping (cheap diagnostic to rule out feature preprocessing)
  L: slice_num=32 (quick screen)
```

---

## Ruled Out / Do Not Reproduce

- Re-curriculum (#3242): +60% regression, mechanism falsified
- surf_weight=50 (#3303): +3.5% regression
- surf_p_weight_extra=4 (#3393): not merged, direction unclear — wait for student follow-up
- 256 hidden + 8 layers + 8 heads (bundled): tried in sibling branches, no improvement over 128/5/4
- GeGLU/SwiGLU activations: tried in sibling branches
- LayerScale, DropPath: tried in sibling branches
- Per-channel decoder heads: tried in sibling branches (note: idea 7 in this file is different —
  it isolates just the pressure head, not all channels)
- Lion optimizer: tried in sibling branches
- Inter-block multiplicative/additive scaling: tried in sibling branches

## Active (Do Not Duplicate)

- #3567: n_hidden 128→192
- #3241: EMA (decay=0.9999)
- #3240: Z-reflection augmentation
- #3239: Fourier positional encoding
- #3177: Per-sample Re-scale normalization
