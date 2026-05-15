<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 2 Hypothesis Menu — TandemFoilSet Transolver
**Round 2 hypothesis menu generated after Round 1 leader (nezuko 102.67) confirmed**
**Date:** 2026-05-15
**Branch:** icml-appendix-willow-pai2i-48h-r5
**Current merged best:** val_avg/mae_surf_p = 130.46 (PR #3123, Fourier PE n=16 sigma=10)
**Empirical baseline (no changes):** val_avg = 135.23

---

## Context and Round 1 Gaps

Round 1 covers: Huber loss (β=0.1), n_hidden scale-up (192/256), slice_num scaling (96/128/192),
LR warmup+cosine, bf16+batch, grad-clip+EMA(0.999), per-channel p-weighting.

Merged: Fourier PE (n=16, sigma=10) — dominant effect on camber_cruise OOD (−25.5%).

Round 2 targets mechanisms NOT covered in Round 1. Every arm-A is baseline-equivalent
(Fourier PE n=16 already on the merged codebase).

Wall-clock constraint: 30 min binds at ~epoch 14/50. SAM adds ~1.5–1.8× wall-clock.
Test metrics blocked on NaN fix (PR #3296).

---

## H1 — Compound Winners Stack: EMA + Grad-Clip + Huber + Fourier PE (Priority 1)

**Mechanism:** Combine the four individually promising Round 1 changes into a single PR:
EMA(decay=0.999) for OOD flat-minima inference, grad-clip(max_norm=1.0) for gradient
spike suppression, Huber loss(beta=0.1) for heavy-tail robustness, and the already-merged
Fourier PE(n=16, sigma=10). These four are expected to be orthogonal — each targets a
different failure mode (inference sharpness, gradient variance, loss scale, positional
encoding) — so gains should stack additively.

**Why now:** Round 1 ran these individually; we need to confirm stack orthogonality before
spending more compute on further R&D. If the compound PR wins, it becomes the new baseline
for all subsequent work.

**Expected delta:** 5–12% over current merged best (130.46 → ~115–123). This is the
single highest-confidence bet in Round 2.

**Interaction risk:** Per-channel p-weighting (tanjiro #3118) is explicitly excluded — it
targets the same objective direction as Huber but with a different mechanism; if both are
included without ablation they confound the result.

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline (Fourier PE n=16 only) | Establishes per-run reference on the stacked codebase |
| B | EMA(0.999) + grad-clip(1.0) | Optimization stack only — isolates the opt improvements |
| C | EMA(0.999) + grad-clip(1.0) + Huber(beta=0.1) | Full compound stack |

**Implementation notes (~30 LOC total in train.py):**

EMA (after model init):
```python
ema_model = torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
)
```
After each batch update:
```python
ema_model.update_parameters(model)
```
Use `ema_model` (not `model`) in validation loop.

Gradient clipping (immediately before optimizer.step()):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Huber loss swap (replace MSE surf/vol terms):
```python
criterion = torch.nn.HuberLoss(reduction='none', delta=0.1)
# surf_loss = criterion(pred_surf, y_surf).mean()
# vol_loss  = criterion(pred_vol,  y_vol).mean()
# total_loss = vol_loss + 10.0 * surf_loss
```
Note: PyTorch HuberLoss delta is in normalized space. beta=0.1 with normalized targets
is aggressive (transitions to L1 at 10% of std). This is correct for heavy-tailed CFD.

Checkpoint saving: save `ema_model.module.state_dict()` for portability.

**Implementation complexity:** ~30 LOC. Low risk. All components individually tested in
Round 1 code. Student must be careful to use ema_model only in eval, never for
optimizer.step().

**Reproduce command:**
```bash
python train.py \
  --use_ema true --ema_decay 0.999 \
  --grad_clip 1.0 \
  --loss huber --huber_beta 0.1 \
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-compound
```

---

## H2 — FiLM Conditioning on log(Re) (Priority 2)

**Mechanism:** Add Feature-wise Linear Modulation in each Transolver MLP block, conditioned
on the per-sample global log(Re) scalar. A small 3-layer MLP maps log_re_global → (gamma,
beta) of shape [B, n_hidden]; each Transolver block applies `h = gamma * h + beta` after
its first linear layer. This gives the model explicit regime-switching capability —
cross-Re generalization currently relies only on Re appearing as one of 24 input channels
mixed with local geometry features.

**Why now:** `val_re_rand` is the weakest remaining split. Fourier PE helped camber_cruise
but left re_rand essentially unchanged (120.44 → 123.13, slightly worse). FiLM is the
canonical approach from AeroDiT (arXiv:2412.17394) and Universal Physics Transformers
(NeurIPS 2024) for conditioning neural operators on global flow parameters.

**Expected delta:** 5–10% on val_re_rand. Possible ~3–5% on val_avg through re_rand
improvement. Limited effect on camber splits (those are geometry OOD, not regime OOD).

**Per-split prediction:**
- in_dist: neutral to slight improvement (~0–2%)
- camber_rc: neutral (~0%)
- camber_cruise: neutral to slight improvement (~0–2%)
- re_rand: largest improvement (+5–10%)

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline (Fourier PE n=16 only) | Reference |
| B | FiLM on output MLP only (log_re → γ,β applied to last MLP block) | Lowest-risk entry point |
| C | FiLM on all 5 MLP blocks + output MLP | Full conditioning strength |

**Implementation notes (~50 LOC in model.py):**

```python
# In Transolver __init__:
self.film_mlp = nn.Sequential(
    nn.Linear(1, 64), nn.SiLU(),
    nn.Linear(64, 64), nn.SiLU(),
    nn.Linear(64, 2 * n_hidden)  # outputs (gamma, beta)
)

# In Transolver forward:
log_re_global = x_raw[:, :, 13].mean(dim=1, keepdim=True)  # [B, 1]
# Use raw/unnormalized log(Re) — better numerical properties than normalized
# log_re_global = log_re_raw; Re range [100K, 5M] → log range [11.5, 15.4]
film_params = self.film_mlp(log_re_global)  # [B, 2*n_hidden]
gamma, beta = film_params.chunk(2, dim=-1)   # each [B, n_hidden]
gamma = gamma.unsqueeze(1)                   # [B, 1, n_hidden]
beta  = beta.unsqueeze(1)                    # [B, 1, n_hidden]

# In MLP block (for Arm C, apply to all blocks):
h = gamma * h + beta
```

Critical gotcha: `x[:, :, 13]` in the model receives the NORMALIZED version of log(Re).
Either extract the raw log(Re) before normalization and pass it separately, or remap
the normalized dim 13 back to physical scale using stored stats. The easiest approach:
pass `log_re_raw` as a separate 1D tensor in the batch dict rather than re-extracting
from the normalized x. Check train.py for how the batch dict is constructed.

Second gotcha: Arm B (output MLP only) — ensure `film_mlp` outputs are applied only to
the final output projection, not to the Physics Attention slice weights. Conditioning the
slice weights can cause training instability.

**Implementation complexity:** ~50 LOC. Medium risk. The model forward pass must be
modified. Recommend starting with Arm B to verify the mechanism before Arm C.

**Reproduce command:**
```bash
python train.py \
  --use_film true --film_mode output_only \   # Arm B
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-film
```

---

## H3 — Per-Sample Relative L2 Loss (Priority 2)

**Mechanism:** Divide each sample's MSE loss contribution by that sample's per-channel
y-standard-deviation before averaging across the batch. This makes the training objective
scale-invariant: a low-Re sample (y_std~50) and a high-Re sample (y_std~2000) contribute
equally to the gradient, rather than the high-Re sample dominating by 40×. This is the
standard loss in FNO, GNOT, and most NeurIPS neural operator benchmarks, and is distinct
from Huber — Huber clips large residuals within a sample, while relative L2 equalizes
across samples.

**Why now:** With per-sample y-std varying 5–10× within each split, the baseline optimizer
is heavily biased toward high-Re samples. The val_re_rand split likely includes both
extremes, and the OOD camber splits have shifted Re distributions. Equalizing gradient
contributions should improve calibration across the Re spectrum.

**Expected delta:** 3–7% on val_avg, concentrated on re_rand and camber_cruise. Risk
of slight regression on in_dist if the model currently over-fits to high-Re cases.

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline | Reference |
| B | Relative L2 on surf term only | Target the metric directly; conservative first step |
| C | Relative L2 on both surf and vol terms | Full objective change |

**Implementation notes (~15 LOC in train.py):**

```python
# Compute per-sample scale from the OUTPUT (in normalized space use stored y_std or
# compute from the batch y directly):
y_scale = y.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, C] or [B, 1]

# In the loss:
sq_err = (pred - y) ** 2          # [B, N, C]
rel_sq_err = sq_err / y_scale**2   # scale-normalized
surf_loss = (rel_sq_err * surf_mask).sum() / surf_mask.sum()
```

Note: if y is already normalized (divided by y_std at dataset load time), the scale
equalization is already partially done by dataset normalization. Check what normalization
`train.py` applies. If y is fully standardized (zero mean, unit std), per-sample
relative loss is less impactful — but within-batch variance across samples still exists.
The mechanism is most useful when samples have heterogeneous absolute scales.

Critical gotcha: `y.std(dim=1)` computes std over mesh nodes. Ensure dim=1 is the node
dimension and not the channel or batch dimension. Verify shapes carefully.

**Implementation complexity:** ~15 LOC. Very low risk. This is a well-tested recipe in
the neural operator community.

**Reproduce command:**
```bash
python train.py \
  --loss relative_l2 --relative_l2_mode surf_only \  # Arm B
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-relative-l2
```

---

## H4 — Fourier Sigma Sweep (Priority 3)

**Mechanism:** Keep the winning Fourier PE architecture (n=16 frequencies) but sweep the
projection matrix scale sigma over {4, 8, 20, 40} to find the optimal frequency content
for TandemFoilSet geometry. The current sigma=10 was not swept — it was chosen from the
initial literature (Tancik et al. 2020 uses sigma=10 for NeRF). Different values encode
different spatial frequency ranges: small sigma (4) = coarse features (wing chord scale);
large sigma (20–40) = fine features (leading-edge curvature, boundary layer).

**Why now:** The merged PR found sigma=10 works well for camber_cruise (OOD geometry) but
slightly hurt camber_rc. A sigma sweep costs minimal LOC change and may recover the
camber_rc regression while preserving or improving camber_cruise.

**Physical intuition:** Airfoil leading-edge pressure gradients have spatial scale ~0.01c
(chord lengths). At sigma=10, the Fourier features encode spatial periods down to ~1/10
of the domain scale. The optimal sigma aligns with the dominant spatial frequency of the
pressure field, which differs between RaceCar (ground-effect, large-scale) and Cruise
(freestream, sharp leading-edge) configurations.

**Expected delta:** 1–4% on val_avg. Higher probability of recovery on camber_rc.

**Arms:**

| Arm | sigma | Expected winner |
|-----|-------|----------------|
| A | sigma=10, n=16 (current merged baseline) | Reference |
| B | sigma=4, n=16 | Hypothesis: coarser features may help camber_rc |
| C | sigma=20, n=16 | Hypothesis: finer features may help leading-edge pressure |

**Implementation notes (~1 LOC change):**

The Fourier projection matrix B is drawn from N(0, sigma^2). To change sigma, change the
initialization in the FourierFeatures module. Verify that B is sampled at runtime or
stored as a buffer — if stored as a buffer, the checkpoint will fix sigma, which is correct.

If results show sigma=4 and sigma=20 both lose to sigma=10, the current value is near-
optimal. If sigma=20 wins, consider a further sweep {20, 30, 40} in Round 3.

**Implementation complexity:** ~1 LOC. Near-zero risk. The cleanest diagnostic in Round 2.

**Reproduce command:**
```bash
python train.py \
  --n_fourier 16 --fourier_sigma 4 \   # Arm B
  --wandb_group round2-sigma-sweep
```

---

## H5 — 1st-Order SAM for OOD Generalization (Priority 3)

**Mechanism:** Replace the standard AdamW optimizer step with a SAM (Sharpness-Aware
Minimization) step that perturbs parameters toward high-loss regions, then computes the
actual update from that perturbed point. This finds parameters in flat regions of the
loss landscape, which generalize better under distribution shift. 1st-order SAM
(arXiv:2411.01714) achieves ~95% of full SAM benefit with only 1 extra forward pass
instead of 2 (1.5× vs 2× wall-clock).

**Why now:** 3 of 4 val splits are OOD. Schapiro & Zhao (arXiv:2412.05169) show SAM
improves OOD generalization by +4.76%–+8.01% over Adam across standard benchmarks.
The effect is orthogonal to architecture and loss — it is a pure optimizer-level
intervention. With the 30-min wall clock binding at epoch 14, 1st-order SAM will
bind at ~epoch 9–10, so this experiment must be understood as testing mechanism
viability at under-training, not final convergence.

**Expected delta:** 3–8% on val_avg if the mechanism is active at epoch 9–10. Uncertain
because the under-training regime may not fully expose OOD gap.

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline (Fourier PE n=16, standard AdamW) | Reference |
| B | 1st-order SAM, rho=0.05 | Conservative rho; standard from Kaddour 2024 |
| C | 1st-order SAM, rho=0.1 | More aggressive perturbation |

**Implementation notes (~40 LOC in train.py):**

```python
# 1st-Order SAM step (inline, no external dependency needed):
# Standard backward pass:
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()

# Compute gradient norm and perturbation:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
eps = rho / (grad_norm + 1e-12)

# Add perturbation to parameters:
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p.data.add_(p.grad * eps)

# Forward pass at perturbed point (no backward needed for 1st-order):
# NOTE: 1st-order SAM re-uses the SAME gradient from the original point
# (the key difference from 2nd-order SAM which recomputes gradient here)

# Remove perturbation:
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p.data.sub_(p.grad * eps)

# Apply update using the original gradient:
optimizer.step()
```

Wall-clock gotcha: 1st-order SAM as above adds only a second forward pass overhead
equivalent, not a full backward. Measure wall-clock per epoch and confirm the 30-min
limit is not hit at fewer than 7 epochs. If epochs<7 are the result, reduce batch
size or slice_num to recover iteration throughput.

Combination note: do NOT stack SAM with EMA in the same PR — they interact in
non-obvious ways (EMA averages the perturbed-then-restored trajectory). Test SAM
standalone first; combine in Round 3 if both survive.

**Implementation complexity:** ~40 LOC. Medium risk. The perturbation logic is simple
but easy to implement incorrectly (the 2nd forward pass must happen at the perturbed
point, then parameters must be exactly restored). Test with a unit test: verify
`before_params == after_params` after one SAM step.

**Reproduce command:**
```bash
python train.py \
  --optimizer sam_1st_order --sam_rho 0.05 \
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-sam
```

---

## H6 — AoA Reflection Symmetry Data Augmentation (Priority 4)

**Mechanism:** For RaceCar single-foil samples, flip the mesh coordinates (z → -z),
negate AoA (dim 14 → -dim 14), and negate Uy (output dim 1 → -dim 1) to obtain a
physically valid second sample at the reflected angle of attack. This doubles the
effective training set for RaceCar single (599 → 1,198 samples) at zero additional
data cost. The symmetry follows directly from the incompressible Navier-Stokes equations:
if u(x,z) solves the equations at AoA θ, then the reflected field u'(x,-z) = [Ux, -Uy, p](x,-z)
solves the equations at AoA -θ.

**Why now:** RaceCar single is the smallest training domain (599 samples) and the
in_dist val split is from this domain. Doubling it is free and should reduce in_dist
overfitting without requiring any model change.

**Caution:** The dsdf feature (dims 4–11) is distance-based and may not transform simply
under the z-flip. Signed distance fields transform as sdf(x,-z) = -sdf(x,z) for signed
variants, and sdf(x,-z) = sdf(x,z) for unsigned variants. Verify the exact semantics of
dims 4–11 before applying the flip — flip only after confirmed safety. If uncertain,
exclude dsdf dims from the flip and replace them with 0 (indicating "unknown" state).

**Expected delta:** 2–5% on val_in_dist, possible slight improvement on camber_rc (also
RaceCar domain). Neutral on camber_cruise and re_rand.

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline | Reference |
| B | AoA reflection aug on RaceCar single only, dsdf dims zeroed | Safe conservative version |
| C | AoA reflection aug on RaceCar single + tandem (AoA1 and AoA2 both negated) | Full augmentation |

**Implementation notes (~40 LOC in dataset.py or train.py data loading):**

```python
# Apply to a batch or single sample:
def reflect_aoa(x, y):
    # x: [N, 24], y: [N, 3]
    x_aug = x.clone()
    y_aug = y.clone()
    
    # Flip z-coordinate:
    x_aug[:, 1] = -x[:, 1]  # dim 1 = z position
    
    # For Arm B: zero out dsdf dims (uncertain transform):
    x_aug[:, 4:12] = 0.0    # dims 4-11 = dsdf
    
    # Negate AoA for foil 1 (and foil 2 for Arm C):
    x_aug[:, 14] = -x[:, 14]  # dim 14 = AoA foil 1
    # (Arm C: x_aug[:, 18] = -x[:, 18])  # dim 18 = AoA foil 2
    
    # Negate Uy output:
    y_aug[:, 1] = -y[:, 1]  # dim 1 = Uy
    
    return x_aug, y_aug
```

This must be applied AFTER normalization, or coordinate adjustments must account for
how the dataset normalizes dims 1, 14, and the outputs. Check if the dataset class
normalizes in __getitem__ or in a collate function.

For RaceCar single, AoA range is -10° to 0°. The reflection produces AoA 0° to +10°,
which is outside the training distribution — the augmentation adds genuinely new
regime coverage, not just copies.

**Implementation complexity:** ~40 LOC. Medium risk due to dsdf uncertainty. Must verify
feature transform semantics before trusting Arm C results.

**Reproduce command:**
```bash
python train.py \
  --aoa_reflection_aug true --aug_domains racecar_single \
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-aoa-aug
```

---

## H7 — Stochastic Depth (DropPath) Regularization (Priority 4)

**Mechanism:** Randomly drop entire Transolver transformer blocks during training with
probability p_drop, creating an implicit ensemble of shallower subnetworks. At inference,
all layers are active (with expected value scaling). This is the "stochastic depth"
technique (Huang et al. 2016) adapted to the Transolver block structure. Recent work
(NeurIPS 2025: "Training Transformers for Mesh-Based Simulations") shows stochastic
depth provides consistent generalization improvement for CFD mesh transformers at
essentially zero inference cost.

**Why now:** The Transolver has only 5 layers and ~1.5M parameters, trained on 1,499
samples — a classic under-regularized small-data regime. With the 30-min wall clock
limiting epochs to ~14, explicit regularization may substitute for the missing epochs.
Stochastic depth is strictly orthogonal to all other Round 1/2 changes.

**Expected delta:** 2–5% on OOD splits (camber_rc, re_rand). The mechanism
acts as implicit ensemble, which tends to help most on out-of-distribution data.

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline | Reference |
| B | DropPath p=0.1 on all 5 Transolver blocks | Conservative; standard for shallow transformers |
| C | DropPath p=0.2 on layers 3–5 only (increasing with depth) | Standard stochastic depth schedule |

**Implementation notes (~20 LOC in model.py):**

```python
# PyTorch does not include DropPath natively; use timm's implementation:
from timm.models.layers import DropPath

class TransolverBlock(nn.Module):
    def __init__(self, ..., drop_path_rate=0.0):
        ...
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x, ...):
        # Standard: h = x + self.attn(self.norm1(x))
        h = x + self.drop_path(self.attn(self.norm1(x)))
        h = h + self.drop_path(self.mlp(self.norm2(h)))
        return h
```

For Arm C (depth-scaled), set rates per layer:
```python
drop_rates = [0.0, 0.0, 0.1, 0.15, 0.2]  # increasing with depth for 5-layer Transolver
```

`timm` is likely already a dependency (Transolver uses it for positional embeddings in
some implementations). If not, `pip install timm` or implement DropPath directly
(~8 lines using `torch.bernoulli`).

**Implementation complexity:** ~20 LOC. Low risk. DropPath is a well-tested, stable
regularizer with no known pathological interactions with PhysicsAttention.

**Reproduce command:**
```bash
python train.py \
  --drop_path_rate 0.1 \
  --n_fourier 16 --fourier_sigma 10 \
  --wandb_group round2-stochastic-depth
```

---

## H8 — Sobolev Loss on Surface Nodes (Priority 5)

**Mechanism:** Add a gradient-matching term to the surface loss that penalizes the
discrepancy between predicted and target spatial gradients of pressure along the foil
surface (∂p/∂s, where s is arc-length). Standard MSE/MAE matches pointwise values;
Sobolev loss additionally matches the derivative, which encodes physics of pressure
distribution (e.g., leading-edge suction peak curvature, Kutta condition at trailing
edge). The gradient can be approximated via finite differences over sorted surface nodes.

**Why now:** The primary metric is surface pressure MAE. The hardest cases are OOD
geometry camber splits where the model must interpolate to unseen foil shapes. Pressure
gradient matching provides a physically-motivated inductive bias for smooth, physically
consistent solutions — this is particularly valuable for extrapolating to unseen camber.

**Expected delta:** 3–6% on camber splits. Risk of slight regression on in_dist if the
gradient signal is noisy (surface node ordering matters for the finite difference).

**Arms:**

| Arm | Change | Rationale |
|-----|--------|-----------|
| A | Baseline | Reference |
| B | Sobolev weight λ=0.1 on surface ∂p/∂s term | Conservative; gradient signal as soft regularizer |
| C | Sobolev weight λ=0.5 on surface ∂p/∂s term | Stronger gradient pressure |

**Implementation notes (~50 LOC in train.py):**

```python
# Sort surface nodes by arc-length (dim 2-3 = signed arc-length):
# Assumes x[:, :, 2] contains the arc-length parameterization of surface nodes.
# surf_mask: [B, N] boolean; x: [B, N, 24]; pred/y: [B, N, 3]

# Extract surface nodes:
surf_x    = x[surf_mask]    # [S, 24], S = total surface nodes in batch
surf_pred = pred[surf_mask] # [S, 3]
surf_y    = y[surf_mask]    # [S, 3]

# Arc-length along surface (dim 2 = saf):
saf = surf_x[:, 2]  # [S]

# Sort by arc-length (per-sample, requires looping or segment sort):
# For simplicity, use finite diff over sorted arclength:
dp_pred = torch.diff(surf_pred[:, 2], dim=0) / (torch.diff(saf, dim=0).clamp(min=1e-6))
dp_true  = torch.diff(surf_y[:, 2],   dim=0) / (torch.diff(saf, dim=0).clamp(min=1e-6))

sobolev_loss = F.mse_loss(dp_pred, dp_true)
total_loss = base_loss + lambda_sobolev * sobolev_loss
```

Critical gotcha: `torch.diff` is applied globally across all surface nodes in the batch,
including across sample boundaries. Must segment by sample (use batch index mask) to
avoid computing finite differences between the last node of sample i and first node of
sample i+1. This is the most common implementation bug for Sobolev loss in batched
settings.

**Implementation complexity:** ~50 LOC. Medium risk due to surface node sorting and
boundary handling. If this is too complex to implement cleanly in one PR, defer to
Round 3 when more wall-clock budget may be available.

---

## Ranking and Prioritization

**Top 3 for Round 2 (ranked by expected EV):**

1. **H1 — Compound Winners Stack** (EMA + grad-clip + Huber + Fourier PE)
   - Highest confidence. Four independently-motivated mechanisms, orthogonal failure
     modes, linear combination expected. If Round 1 results confirm individual
     components, this stack should definitively improve the baseline by 5–12%.
   - Priority: assign immediately upon confirming at least one Round 1 component win.

2. **H2 — FiLM on log(Re)**
   - Directly targets val_re_rand, the only split that Fourier PE did NOT improve.
   - Mechanism is well-supported by AeroDiT and Universal Physics Transformers.
   - 50 LOC, medium complexity, but uniquely addresses cross-regime generalization.
   - Priority: assign alongside H1 to a second student.

3. **H3 — Per-Sample Relative L2 Loss**
   - Simplest new mechanism (15 LOC), targets the core heteroskedasticity problem
     from a different angle than Huber (cross-sample vs. within-sample scale).
   - Well-validated in the neural operator literature (FNO, GNOT).
   - Priority: assign to a third student as a fast, low-risk complement to H1+H2.

**H4 (Fourier sigma sweep):** Assign when 3+ students are otherwise idle. Near-zero
implementation risk, useful as a diagnostic to determine whether sigma=10 is truly
optimal or accidentally close.

**H5 (SAM):** Assign when a student is available and wall-clock budget is understood.
Must confirm 1st-order SAM implementation is correct with a quick unit test before
trusting results.

**H6–H8:** Reserve for Round 3 or when H1–H5 plateau.

---

## Stop Conditions

- If H1 (compound stack) shows no improvement over any individual Round 1 winner:
  implies negative interaction between components. Ablate pairwise (EMA alone, grad-clip
  alone) in Round 3 before further stacking.

- If H2 (FiLM) shows no improvement on val_re_rand: implies log(Re) information is
  already adequately captured by dim 13 in the existing architecture. Deprioritize
  further conditioning approaches.

- If H3 (relative L2) shows no improvement: implies heteroskedasticity is not the
  binding constraint (possibly the 14-epoch training regime is the real bottleneck).
  Consider using relative L2 only as part of a longer training run rather than standalone.

---

*Generated from: Round 1 PR survey (9 PRs, 1 merged), literature scan (FiLM/AeroDiT,
SAM/OOD, Sobolev/CFD, stochastic depth/mesh transformers), TandemFoilSet dataset
analysis, Fourier PE ablation results (PR #3123).*
