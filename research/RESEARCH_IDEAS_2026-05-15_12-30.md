<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-15 12:30

Generated for the TandemFoilSet CFD surrogate programme.
Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE
across 4 val splits). Lower is better.
All ideas are implementable as edits to `train.py` only (data/* is read-only).

---

## Orientation: bottleneck diagnosis

Before listing hypotheses, here is a causal map of what the baseline may be
getting wrong:

**Loss formulation.** The baseline MSE loss in normalized space weights every
squared error equally. But per-sample y-std varies from ~50 to ~2077 even
within a single domain split. After global z-score normalization, a
high-Re sample with large physical error may contribute orders-of-magnitude
more gradient signal than a low-Re sample. The model likely learns to fit
high-Re examples well and ignores the structure of low-Re pressure fields —
but MAE is reported equally across all samples.

**Surface vs. volume imbalance.** `surf_weight=10.0` adds emphasis on
surface nodes, but the loss is still symmetric over all three output channels
(Ux, Uy, p). The primary metric is *surface pressure only*, yet the model
spends equal loss budget on Ux and Uy. An extra pressure channel weight on
surface nodes would focus gradient more precisely.

**Optimizer state.** No gradient clipping. No EMA. Cosine annealing with no
warmup. With variable-length sequences (74K–242K nodes), gradient norms can
spike severely on large meshes. Missing warmup means the early gradients
(while the model is randomly initialized) set a poor trajectory.

**Capacity.** n_hidden=128, 5 layers, 4 heads, slice_num=64 — 1.5M params.
This is small. The Transolver paper uses n_hidden=256 in its stronger configs.
There is likely headroom from width alone, since the model must simultaneously
handle 3 domains, 2 foil types, a 50x Re range, and OOD camber generalization.

**Conditioning.** `log(Re)` is provided as one of the 24 input features, but
the model processes it identically to spatial coordinates. FiLM or AdaLN
conditioning from the global flow parameters (log(Re), AoA, gap, stagger)
could let the model adapt its internal representations to the flow regime
without having to learn it from position alone.

**EMA absent.** Weight averaging at test time is free generalization; the
baseline does not use it.

---

## Hypotheses (ordered by expected impact / implementation cost)

---

### H1: Per-channel surface-pressure loss upweighting

**What it is.** Add a separate multiplier for the pressure channel (dim 2) on
surface nodes, on top of the existing `surf_weight`. The baseline applies
`surf_weight=10.0` equally to all three output channels. Since the primary
metric is surface pressure MAE, over-weighting the pressure channel on surface
nodes should directly reduce it.

**Hypothesis.** The model divides its surface-node learning capacity equally
between Ux, Uy, and p. By up-weighting the gradient contribution from surface-p
errors by an additional factor of 2–5, the model will develop sharper
pressure representations on the foil boundary, improving `val_avg/mae_surf_p`
without significantly hurting the other channels.

**Predicted delta.** -3% to -8% on `val_avg/mae_surf_p`.

**Implementation sketch (train.py only).**
Add a `p_surf_weight: float = 3.0` field to `Config`. In the loss block,
split sq_err by channel after computing `surf_mask`:

```python
# existing baseline loss:
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# replacement:
channel_weights = torch.tensor([1.0, 1.0, cfg.p_surf_weight],
                                device=device).reshape(1, 1, 3)
weighted_sq_err = sq_err * channel_weights
surf_loss = (weighted_sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
```
Vol loss is unchanged. Surf_weight remains 10.0.

**Recommended hyperparameters.** p_surf_weight=3.0 (first arm),
p_surf_weight=5.0 (second arm). lr=5e-4, all other defaults.

**Risk.** Low. If Ux/Uy on surfaces degrade badly and drag the model off a
useful internal representation, restore p_surf_weight=1.0. Failure mode is
pressure improves but velocity diverges, causing the preprocess MLP to
learn a pressure-only embedding.

---

### H2: EMA of model weights

**What it is.** Maintain an exponential moving average of model parameters
during training; evaluate and checkpoint using the EMA parameters instead of
the live optimizer parameters.

**Hypothesis.** The baseline uses the raw optimizer parameters for
checkpointing and evaluation. EMA smooths out the noise introduced by
mini-batch gradients and the large variance in per-batch mesh sizes (74K to
242K nodes). A TMLR paper (Izmailov et al. 2018 follow-ons) demonstrates that
EMA consistently improves out-of-distribution generalization, which is critical
for val_geom_camber_rc and val_geom_camber_cruise — the two splits that test
unseen camber geometry.

**Predicted delta.** -2% to -6% on `val_avg/mae_surf_p`, with stronger benefit
on the OOD geometry splits.

**Implementation sketch (train.py only).**
Use `torch.optim.swa_utils.AveragedModel` (ships with PyTorch, no new package
needed):

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

# in the training loop, after optimizer.step():
ema_model.update_parameters(model)

# use ema_model.module for evaluation and checkpointing:
# replace model.eval() → ema_model.eval() in evaluate_split calls
# save ema_model.module.state_dict() to checkpoint
```

Start EMA from epoch 1 (not delayed). Decay=0.999. Use the EMA copy for all
val/test evaluations and checkpoint saves.

**Recommended hyperparameters.** decay=0.999. Same lr=5e-4, cosine scheduler.

**Risk.** Low. EMA cannot hurt training loss (the live model still updates
normally). The only failure mode is that the EMA copy lags the optimizer
parameters for too long at the start — start from epoch 1 to avoid this.

---

### H3: Gradient clipping + LR warmup

**What it is.** Add `torch.nn.utils.clip_grad_norm_` and a short linear LR
warmup before the cosine schedule.

**Hypothesis.** Mesh sizes vary 3x across batches (74K to 242K nodes). A
single large-mesh batch can produce gradient norms that are an order of
magnitude larger than a small-mesh batch, causing training instability and
biasing early convergence toward the dominant mesh regime. Clipping at max_norm=1.0
and warming up the LR over 5 epochs prevents the random-init parameters from
being driven into poor local structure during the first steps, when gradients
are largest and least informative.

**Predicted delta.** -2% to -5% on `val_avg/mae_surf_p`. Larger expected
improvement on val_re_rand (stratified Re holdout, most diverse meshes) than
on in-distribution splits.

**Implementation sketch (train.py only).**
In the Config dataclass, add:
```python
grad_clip: float = 1.0
warmup_epochs: int = 5
```
Replace the scheduler construction:
```python
def lr_lambda(epoch):
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(1, MAX_EPOCHS - cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```
Add `import math` at top. In the training loop after `loss.backward()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

**Recommended hyperparameters.** grad_clip=1.0, warmup_epochs=5.

**Risk.** Low-Medium. If warmup slows early convergence too much, reduce
warmup_epochs=3. Gradient clipping at 1.0 is standard; only risk is if the
model needs large gradient norms to escape initialization.

---

### H4: Huber loss to handle heavy-tailed pressure errors

**What it is.** Replace the MSE loss with a Huber (smooth L1) loss that behaves
like L2 near the origin but like L1 for large residuals.

**Hypothesis.** In the normalized target space, a few extreme high-Re samples
(Re=5M, pressure excursions to -29,136 m²/s²) produce normalized residuals
that are far outside the typical range. Under MSE, these squared residuals
dominate the gradient and cause the model to over-fit the extreme-value
structure at the expense of typical-value precision. Huber loss clips the
gradient contribution from outlier residuals, letting the model distribute
learning capacity more evenly and improving median-case surface pressure
accuracy. This directly targets the metric: MAE, not MSE, is the ranking
criterion.

**Predicted delta.** -3% to -8% on `val_avg/mae_surf_p`, with strongest
improvement on val_re_rand (widest Re range).

**Implementation sketch (train.py only).**
In Config, add:
```python
loss_fn: str = "huber"   # "mse" | "huber"
huber_delta: float = 1.0
```
Replace the sq_err block in the training loop:
```python
if cfg.loss_fn == "huber":
    sq_err = F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)
else:
    sq_err = (pred - y_norm) ** 2
```
All downstream masking and weighting code is unchanged because `sq_err` has the
same shape [B, N, 3]. Note: Huber loss reduces to 0.5*(pred-y)^2 for
|pred-y|<delta and delta*(|pred-y|-0.5*delta) otherwise, so the units shift
but the gradient direction is the same. The evaluation is still MAE in physical
space — this only changes the loss.

**Recommended hyperparameters.** huber_delta=1.0 (normalized space; roughly 1
std in normalized target). Try delta=0.5 as a second arm if first arm shows
improvement — lower delta is more robust but slower for inliers.

**Risk.** Low. Gradient magnitude changes but not direction. Primary failure mode:
model converges to a different local optimum that favors median-fitting over
exact-surface fitting; monitor surf_loss (which uses the training loss) vs.
mae_surf_p (which uses MAE) to detect this.

---

### H5: Wider model — n_hidden 128→256

**What it is.** Double the hidden dimension of the Transolver from 128 to 256.
This quadruples the attention inner dimension and doubles MLP width.

**Hypothesis.** At n_hidden=128 with 4 heads, dim_head=32. The physics
attention is operating in a very compressed latent space while needing to
distinguish 3 domains, 50x Re range, 2 foil configurations, and unseen
camber geometries. Doubling n_hidden to 256 (dim_head=64) is the standard
Transolver configuration from the original paper and provides significantly
more representational capacity for the OOD generalization splits.

**Predicted delta.** -5% to -15% on `val_avg/mae_surf_p`. Strongest improvement
on OOD geometry splits (val_geom_camber_rc, val_geom_camber_cruise) where the
compressed baseline likely cannot capture the pressure gradient patterns of
unseen camber profiles.

**Implementation sketch (train.py only).**
In the `model_config` dict:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,       # was 128
    n_layers=5,
    n_head=8,           # was 4, so dim_head stays 32 → or keep n_head=4 for dim_head=64
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```
Two sub-arms: (a) n_hidden=256, n_head=8 (dim_head=32, same as baseline);
(b) n_hidden=256, n_head=4 (dim_head=64, larger per-head capacity).

VRAM check: largest mesh is 242K nodes. At n_hidden=256 with B=2,
N=242K, the feature tensor is 2*242K*256*4 bytes = ~500MB. Attention over
S=64 slices is negligible. Safe.

**Recommended hyperparameters.** n_hidden=256, n_head=8, batch_size=2 (reduce
from 4 to stay under VRAM limit with doubled activations), lr=5e-4. Or keep
batch_size=4 and verify VRAM empirically.

**Risk.** Medium. VRAM pressure with B=4 and 242K nodes at n_hidden=256 —
reduce batch_size to 2 if needed. Training time per epoch increases ~2x.
With 50-epoch cap and timeout, may not fully converge; use best-checkpoint
evaluation to capture early improvement.

---

### H6: Pressure-only decoder head with shared backbone

**What it is.** Add a separate MLP decoder head for the pressure channel (p)
attached to the last hidden state of the Transolver, rather than predicting all
three outputs from the same final linear layer.

**Hypothesis.** Ux and Uy are smooth, approximately potential-flow quantities
near the surface. Pressure p has a very different structure: sharp stagnation
point gradients, suction peaks, and Kutta condition enforcement that require
capturing high-frequency spatial patterns. A shared output head means the model
must learn one representation that is simultaneously optimal for both structures.
A separate p head allows the model to specialize the last few layers of the
decoder for pressure without constraint from velocity.

**Predicted delta.** -3% to -8% on `val_avg/mae_surf_p`.

**Implementation sketch (train.py only).**
In the Transolver model, add a second output head for pressure. The cleanest way
to do this without modifying TransolverBlock (to keep architecture minimal):
Modify the last TransolverBlock to output hidden state instead of predictions,
then split the output:

In the `Transolver` class, change `out_dim=3` to `out_dim=4` (3 for velocity +
1 intermediate) OR, more cleanly, add an additional `p_head` submodule:

```python
class Transolver(nn.Module):
    def __init__(self, ..., separate_p_head=False):
        ...
        self.separate_p_head = separate_p_head
        if separate_p_head:
            # last block outputs hidden_dim, not out_dim
            # add dedicated p head with extra MLP layer
            self.p_head = nn.Sequential(
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, n_hidden),
                nn.GELU(),
                nn.Linear(n_hidden, 1),
            )
            self.vel_head = nn.Sequential(
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, 2),
            )
```
This requires making the last TransolverBlock not use `last_layer=True` (so it
returns hidden state). Add a `last_layer=False` option on the final block and
attach heads manually in `Transolver.forward()`.

**Recommended hyperparameters.** n_hidden=128 (same as baseline), separate_p_head=True.
Add to `Config`: `separate_p_head: bool = True`.

**Risk.** Medium. Requires touching the Transolver class (not just Config). The
split point (where the backbone ends and heads begin) affects gradient flow
through all prior layers. Keep the existing `last_layer` MLP in the final block
but detach it — or simply add the p_head as an extra output on top of the
existing prediction, using it as a residual correction.

---

### H7: FiLM conditioning on global flow parameters (log Re, AoA, gap/stagger)

**What it is.** Add Feature-wise Linear Modulation (FiLM) layers that condition
each TransolverBlock's LayerNorm on a global context vector derived from the
per-sample flow parameters (log(Re), AoA foil 1, AoA foil 2, gap, stagger).

**Hypothesis.** The baseline feeds all 24 input features (including log(Re),
AoA, gap, stagger) into a per-node MLP that projects them into the hidden
space. This means the global flow parameters are mixed with node-local spatial
information from the very first layer. FiLM/AdaLN conditioning extracts the
global parameters as a separate conditioning signal and applies per-layer
scale/shift transformations, letting the model cleanly separate "where is this
node in space?" from "what flow regime are we in?". This is particularly
valuable for the OOD camber and Re-holdout generalization splits, where the
global condition (unseen camber, different Re) must modulate the spatial
patterns the model has already learned.

**Predicted delta.** -4% to -12% on `val_avg/mae_surf_p`, primarily from
val_geom_camber_rc and val_geom_camber_cruise where the global geometry
parameters shift the most from training.

**Implementation sketch (train.py only).**
Add a global context extractor and FiLM layers to the TransolverBlock:

```python
# Global condition dims: log(Re)=1, AoA1=1, AoA2=1, gap=1, stagger=1 → 5 dims
# These are per-sample global features (same value for all nodes in a sample)
# Extract from x[:, 0, [13, 14, 18, 22, 23]] (indices: log(Re), AoA1, AoA2, gap, stagger)

class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        # x: [B, N, D], cond: [B, D_cond]
        scale = self.scale(cond).unsqueeze(1)  # [B, 1, D]
        shift = self.shift(cond).unsqueeze(1)  # [B, 1, D]
        return x * (1 + scale) + shift
```
Add `FiLMLayer` to each `TransolverBlock`, applied after `ln_1` (before attn)
and after `ln_2` (before MLP). Pass the global condition from `Transolver.forward()`:

```python
# In Transolver.forward():
cond_indices = [13, 14, 18, 22, 23]
cond = x[:, 0, cond_indices]  # [B, 5] — same for all nodes in sample
# Then pass cond to each block
```
The global condition features are already normalized by stats["x_mean/std"].

**Recommended hyperparameters.** cond_dim=5, hidden_dim=128, n_hidden=128
(unchanged). No increase in depth.

**Risk.** Medium-High. Requires passing a conditioning tensor through the block
hierarchy. The conditioning vector uses only 5 of the 24 input dims (those that
are genuinely global per-sample), extracted from `x[:, 0, :]` assuming the
first node has the same global features as all others. This assumption holds
since global features (log Re, AoA, gap, stagger) are identical for all nodes
in a sample. Failure mode: if the model over-fits the conditioning signal and
under-fits the spatial structure, OOD performance degrades. Use weight decay
on the FiLM parameters to mitigate.

---

### H8: Per-sample normalization (Z-score by sample statistics)

**What it is.** Normalize each sample's target values by that sample's own
mean and standard deviation (computed from the training set's range), rather
than relying solely on the global z-score from `stats.json`.

**Hypothesis.** Per-sample y std varies from ~50 to ~2077 (a 40x range) even
within a single domain split. After global normalization, a Re=5M sample has
a normalized target std of ~4.5, while a Re=100K sample has a normalized std
of ~0.1. The MSE loss then assigns 1600x more gradient weight per node to
high-Re samples than low-Re samples — the model learns to predict high-Re flows
precisely and rounds off low-Re pressure fields. By dividing each sample's
normalized targets by its own std (computed from the training statistics, not
leaking test info), the loss treats all samples equally. Predictions are
denormalized by the per-sample std before MAE is computed, preserving physical
units in the metric.

**Predicted delta.** -3% to -10% on `val_avg/mae_surf_p`, with the largest
improvement on val_re_rand (widest Re range, most imbalanced gradient signal).

**Implementation sketch (train.py only).**
The per-sample std must be computed from `y` (the target tensor) on the CPU
before normalization. Since `data/*` is read-only, implement this entirely in
the training loop. The key insight: compute per-sample std from the REAL nodes
only (exclude padding), then divide normalized targets by that std:

```python
# In the training loop, after computing y_norm:
y_norm = (y - stats["y_mean"]) / stats["y_std"]
# Per-sample scale: std of y_norm over real nodes
# shape: [B, N, 3] → [B, 1, 1]
real_y = y_norm * mask.unsqueeze(-1)
sample_std = real_y.std(dim=1, keepdim=True).clamp(min=0.1)  # [B, 1, 3]
y_scaled = y_norm / sample_std
pred_scaled = pred / sample_std   # scale predictions the same way
sq_err = (pred_scaled - y_scaled) ** 2
# Loss uses scaled targets; MAE computation uses original y_norm → pred
```
For MAE computation in evaluate_split, skip the per-sample scaling (MAE is
already computed from denormalized predictions in the original physical space).
This change only affects the gradient signal, not the metric.

**Recommended hyperparameters.** min_std_clamp=0.1 (prevents division by zero
for near-constant samples). Keep surf_weight=10.0.

**Risk.** Medium. The per-sample scaling changes the magnitude of the loss
across epochs in a non-stationary way. Early in training, sample_std will be
large (model predictions are far from targets); the effective loss will be
smaller than baseline. LR may need to be increased (try lr=1e-3 with this
change). Also: during evaluation, sample_std should NOT be applied (evaluation
uses unscaled y_norm), so the scaling must be strictly confined to the training
loop. Verify that evaluate_split still uses the baseline unscaled path.

---

### H9: Warmup-Stable-Decay (WSD) schedule with AdamW beta2 annealing

**What it is.** Replace cosine annealing with a three-phase WSD schedule
(warmup → stable LR plateau → aggressive LR decay), and increase AdamW beta2
to 0.98 during the stable phase and the initial decay.

**Hypothesis.** The WSD paper (arXiv:2508.01483, 2025) shows that keeping LR
high for longer (stable plateau) before rapid decay recovers most of the
generalization benefit of extended training, and that higher beta2 during the
decay phase consistently improves final metrics by smoothing the gradient
variance estimate. On the TandemFoilSet's noisy, domain-imbalanced training
signal, a longer stable phase at high LR lets the model explore the loss
landscape before decay locks in the final point.

**Predicted delta.** -2% to -5% on `val_avg/mae_surf_p`.

**Implementation sketch (train.py only).**
Add to Config:
```python
warmup_epochs: int = 5
stable_epochs: int = 30    # total = warmup + stable + decay
# decay fills remaining epochs automatically
```
Replace scheduler construction:
```python
import math
warmup_e = cfg.warmup_epochs
stable_e = cfg.stable_epochs
decay_e = max(1, MAX_EPOCHS - warmup_e - stable_e)

def wsd_lr_lambda(epoch):
    if epoch < warmup_e:
        return (epoch + 1) / warmup_e
    elif epoch < warmup_e + stable_e:
        return 1.0
    else:
        t = (epoch - warmup_e - stable_e) / decay_e
        return 0.5 * (1.0 + math.cos(math.pi * t))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lr_lambda)
```
Beta2 annealing: use a single higher beta2=0.98 for the full run (simpler and
validated by the WSD paper):
```python
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=cfg.lr, weight_decay=cfg.weight_decay,
                               betas=(0.9, 0.98))
```

**Recommended hyperparameters.** warmup_epochs=5, stable_epochs=30,
betas=(0.9, 0.98), lr=5e-4.

**Risk.** Low. The schedule change is isolated to the LR lambda; all other code
is unchanged. Failure mode: if 50 epochs are not enough for the stable plateau
to converge, the decay phase starts before the model reaches a good attractor.
Monitor train/surf_loss during the stable phase — it should plateau before
decay begins.

---

### H10: Increased slice count — slice_num 64→128

**What it is.** Double the number of latent "slice tokens" in the Transolver's
physics attention from 64 to 128.

**Hypothesis.** The Transolver aggregates N mesh nodes (up to 242K) into S=64
slice tokens via weighted softmax, then runs S×S attention. With S=64 slice
tokens spanning meshes with 3 zones (background, foil 1, foil 2) and two
distinct physical regimes (surface boundary layer and volume field), many
slices are likely over-worked. Doubling S=128 increases representational
capacity of the latent space without changing depth or width, costs only
O(S²) additional FLOPs in attention (256 vs. 64 tokens is negligible vs.
the N-to-S scatter), and has O(S) additional VRAM for the slice tokens
themselves.

**Predicted delta.** -2% to -6% on `val_avg/mae_surf_p`, primarily from
better resolution of the pressure gradients near the leading edge and
trailing edge of each foil.

**Implementation sketch (train.py only).**
In `model_config`:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=128,     # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```
One-line change. Increased VRAM is minimal (slice tokens are [B, h, S, d] =
[4, 4, 128, 32] = 65K elements; negligible vs. 242K-node feature tensors).

**Recommended hyperparameters.** slice_num=128, batch_size=4 (unchanged),
lr=5e-4. May combine cleanly with H5 (wider model).

**Risk.** Low. The orthogonal initialization of `in_project_slice.weight` will
need 64→128 orthogonal rows in 32-dimensional space; this is not possible
(orthogonal init requires nrows ≤ ncols). PyTorch's `orthogonal_` truncates to
a semi-orthogonal (QR-based) init when nrows > ncols — this is fine. Primary
failure mode: with more slices, each slice receives fewer nodes on average;
if the slice-weight distribution becomes too diffuse, each slice token carries
less meaningful signal. Monitor the slice weight entropy (optional diagnostic).

---

### H11: More Transolver layers — n_layers 5→7

**What it is.** Add two more TransolverBlock layers to the Transolver, going
from 5 to 7 layers.

**Hypothesis.** With 3 distinct physical domains, tandem foil interactions,
and unseen camber OOD generalization, the 5-layer baseline may be insufficient
to compose the necessary sequence of abstractions: raw geometry → local pressure
gradients → boundary layer effects → far-field coupling → output. Adding 2
layers at the same width costs ~40% more FLOPs per forward pass but may yield
disproportionate improvements on the hardest val splits (OOD camber and Re
holdout), where more compositional depth is needed to generalize.

**Predicted delta.** -3% to -8% on `val_avg/mae_surf_p`.

**Implementation sketch (train.py only).**
In `model_config`:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=7,         # was 5
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```
VRAM: two additional TransolverBlock forward passes at [B=4, N=242K, D=128]
→ ~100MB additional activation memory per block. With B=4, still well within
96GB. Reduce batch_size=2 if needed.

**Recommended hyperparameters.** n_layers=7, batch_size=2 if VRAM is tight,
lr=5e-4.

**Risk.** Low-Medium. Additional layers slow training; with timeout and 50-epoch
cap, fewer epochs complete per second. Use --epochs=40 to reserve eval time
or trust the timeout will trigger a clean checkpoint save. Note that the final
block still uses `last_layer=True` — the code handles this via the loop index
`i == n_layers - 1`, so changing n_layers automatically shifts the last-layer
designation.

---

### H12: Compound baseline — H1 + H2 + H3 (p-weight + EMA + warmup/clip)

**What it is.** Combine the three lowest-risk improvements into a single
enhanced baseline: per-channel surface-p upweighting, EMA weight averaging,
and LR warmup with gradient clipping.

**Hypothesis.** H1, H2, and H3 target orthogonal failure modes: gradient
focus (H1), model averaging noise (H2), and optimization stability (H3). They
do not share parameters or interact negatively (EMA runs on top of any optimizer;
gradient clipping is compatible with any loss; p-weight only changes the
gradient magnitude, not direction). Together, they should compound additively
or super-additively.

**Predicted delta.** -6% to -15% on `val_avg/mae_surf_p` (assuming approximate
additivity of H1+H2+H3 individual deltas).

**Implementation sketch.** Implement H1 (p_surf_weight=3.0), H2 (EMA
decay=0.999), and H3 (grad_clip=1.0, warmup_epochs=5) simultaneously in
train.py. No other architectural changes.

**Recommended hyperparameters.** p_surf_weight=3.0, ema_decay=0.999,
grad_clip=1.0, warmup_epochs=5, lr=5e-4, all other defaults.

**Risk.** Medium. Combining changes makes it impossible to attribute which
component drove improvement if results are ambiguous. Only use this as
an expedient "get a strong baseline quickly" run if students are idle and
individual arms have already been submitted. If this compound PR lands first
and wins, follow up with individual ablations to understand attribution.

---

## Priority order

| Rank | Hypothesis | Expected impact | Implementation cost | Risk |
|------|-----------|-----------------|--------------------|----|
| 1 | H5: Wider model (n_hidden 128→256) | High | Trivial | Med (VRAM) |
| 2 | H1: Per-channel p surf upweighting | Med-High | Trivial | Low |
| 3 | H2: EMA weight averaging | Med | Low | Low |
| 4 | H4: Huber loss | Med | Low | Low |
| 5 | H3: Gradient clip + LR warmup | Low-Med | Low | Low |
| 6 | H7: FiLM Re conditioning | High (if OOD) | High | Med-High |
| 7 | H10: slice_num 64→128 | Low-Med | Trivial | Low |
| 8 | H11: n_layers 5→7 | Med | Trivial | Low-Med |
| 9 | H6: Separate p decoder head | Med | Med | Med |
| 10 | H9: WSD schedule + beta2 | Low-Med | Low | Low |
| 11 | H8: Per-sample normalization | Med | Med | Med |
| 12 | H12: Compound H1+H2+H3 | High | Med | Med |

---

## Research state summary

**Current best explanation for what limits the baseline.**
The dominant bottleneck is likely a combination of (a) MSE gradient imbalance
across the 40x range of per-sample y-std, which under-trains the model on
low-to-medium Re samples that make up the bulk of the evaluation distribution,
and (b) insufficient model capacity (n_hidden=128 is small for a 3-domain,
OOD-generalizing task). Both of these can be attacked cheaply.

**Evidence.**
- Per-sample y std varies 50–2077 (from program.md). Global z-score does not
  correct this. The loss is MSE, which gives 40x more gradient weight per node
  to the highest-std sample.
- Baseline n_hidden=128 with 5 layers = ~1.5M params. The original Transolver
  paper uses 256-hidden in its best reported configuration.
- No gradient clipping means one 242K-node batch can produce a gradient spike
  that biases the optimizer trajectory. This is untested but common in
  variable-mesh settings.
- EMA is free generalization that is known to improve OOD robustness (TMLR
  evidence); it is absent from the baseline.

**Ruled out.**
Nothing is yet ruled out (fresh branch, zero prior experiments).

**Open uncertainties.**
1. Is model capacity (width/depth/slices) or training stability (clipping/EMA)
   the primary limiting factor?
2. Does the global flow conditioning (log Re, AoA) need explicit FiLM
   treatment, or is the existing per-node feature embedding sufficient?
3. Does Huber loss help or hurt for the pressure channel specifically
   (surface pressure peaks are "real" physics, not noise — Huber might
   suppress exactly the signal we care about near the trailing edge)?

**Next discriminating experiments.**
Run H5 (wider model) and H1 (p surf weight) in parallel to separate
capacity-limited from gradient-focus-limited explanations. If H5 wins by a
large margin, capacity is the bottleneck; prioritize H11, H10, and H7 next.
If H1 wins, loss formulation is the bottleneck; prioritize H4 and H8 next.
