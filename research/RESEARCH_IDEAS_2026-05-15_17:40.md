# Research Ideas — 2026-05-15 17:40

Generated after Round-1 experiments and initial Round-2 retries. All 10 original
`RESEARCH_IDEAS_2026-05-15_init.md` hypotheses and all current Round-2 assignments
are excluded from this file.

Baseline: `val_avg/mae_surf_p = 121.69` (EMA decay=0.999, fern/PR#3186).
Current code reference: `train.py` as it stands on branch `icml-appendix-willow-pai2i-48h-r2`
after EMA was merged. Students edit `train.py` only. `data/` is read-only.
Realized epoch budget: ~14 epochs (30-min wall clock).

---

## H-01: `swa-plateau-average`

### Mechanism

Stochastic Weight Averaging (SWA) maintains a running average of the optimizer
trajectory rather than a running average of model weights. EMA tracks the
parameter centroid via exponential decay; SWA collects snapshots on a cyclic
schedule and averages them uniformly. These two averages explore different regions
of weight space. EMA smooths the trajectory — SWA samples multiple basin floors.
Under cosine annealing the LR oscillates back up at each restart, revisiting
different parameter sub-regions; SWA averages across restarts and is known to
find wider, flatter minima that generalize better than the single best checkpoint.
Because EMA already works here, SWA is an independent orthogonal bet on a
complementary mechanism.

### Implementation recipe

```python
# After optimizer definition, add:
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

swa_model = AveragedModel(model)
# SWA phase starts at epoch 8 (roughly halfway through ~14 realized epochs)
SWA_START_EPOCH = 8
# Cycle length = 2 epochs; LR anneal from base lr to swa_lr
swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=2, anneal_strategy="cos")

# Inside epoch loop, after optimizer.step():
if epoch >= SWA_START_EPOCH:
    swa_model.update_parameters(model)

# After scheduler.step() at end of epoch, replace cosine step:
if epoch < SWA_START_EPOCH:
    scheduler.step()
else:
    swa_scheduler.step()

# At end of training, before test eval, update BN stats:
# (no BN in Transolver — skip update_bn; just evaluate swa_model directly)
# Evaluate swa_model instead of ema_model for val checkpointing
```

Key detail: Transolver has no BatchNorm so `update_bn` is not needed. The SWA
model should be evaluated on the val set at each epoch (same as ema_model is
currently). Replace the EMA eval with the SWA eval from epoch `SWA_START_EPOCH`
onward; keep the EMA checkpoint active before that epoch.

Alternatively, run BOTH and take the better of the two. This is cheap and
informative.

**Exact CLI:** `python train.py --wandb_group swa-plateau-average`

### Expected value: Medium-High

SWA has strong evidence in standard deep learning (Izmailov et al., 2018) and has
been used in NeurIPS-winning ensembles. The main risk is that 14 realized epochs
gives a short SWA averaging window. If SWA_START_EPOCH=8 only 6 snapshots are
collected. Consider SWA_START_EPOCH=6 for more averaging.

### Risk notes

- Short epoch budget limits SWA window. If val metrics plateau quickly then SWA
  is collecting from a converged basin, which is less useful.
- SWA and EMA are correlated; if EMA already finds the flat basin, SWA adds
  marginal value. The experiment still teaches us whether the minima structure
  matters beyond EMA.
- SWA is from torch.optim.swa_utils (standard PyTorch — no new dependency).

---

## H-02: `weight-decay-sweep`

### Mechanism

Current AdamW weight_decay=1e-4. This is very light regularization. The model
has ~1.3M parameters but only sees ~200 training samples (cross-domain balanced
via sample_weights). This is a heavily under-constrained regime. Heavier weight
decay (1e-3 to 1e-2) encourages smaller-norm weights and generalization, which
is especially valuable for OOD splits (val_geom_camber_rc, val_re_rand).
EMA already acts as a regularizer, but weight decay targets a different axis:
parameter scale rather than parameter trajectory smoothing.

Round-1 structural finding: the val_single_in_dist vs OOD tradeoff in
loss-redirection experiments suggests the model overfits in-distribution. Higher
weight decay may reduce in-dist overfitting without hurting OOD.

### Implementation recipe

Test three values in separate runs within the same `--wandb_group`:

```python
# Change in Config dataclass and/or at optimizer creation:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay  # try 1e-3, 5e-3, 1e-2
)
```

Run three arms: `weight_decay=1e-3`, `weight_decay=5e-3`, `weight_decay=1e-2`.

Add `weight_decay` to the `Config` dataclass so it's a CLI arg:
```python
weight_decay: float = 1e-4  # already there — just pass different values
```

**Exact CLI:**
```
python train.py --weight_decay 1e-3 --wandb_group weight-decay-sweep
python train.py --weight_decay 5e-3 --wandb_group weight-decay-sweep
python train.py --weight_decay 1e-2 --wandb_group weight-decay-sweep
```

### Expected value: Medium

Weight decay sweeps are cheap and well-studied. Given the small training set and
OOD emphasis of this benchmark, there is real plausibility here. The risk is that
EMA already provides enough smoothing and weight decay primarily hurts in-dist
performance without OOD benefit.

### Risk notes

- If weight_decay is too high it interacts with the cosine LR schedule and can
  destabilize training in later epochs.
- At weight_decay=1e-2, effective LR is heavily penalized — monitor for early
  convergence stall.
- This is a single-hyperparameter scan; quick to run and easy to interpret.

---

## H-03: `asinh-pressure-output`

### Mechanism

Surface pressure in this dataset spans a wide dynamic range — from stagnation
point values (large positive) to suction peak values (large negative), with the
range scaling nonlinearly with Re. In the current formulation predictions are in
normalized space via z-score (subtract mean, divide std). But if the pressure
distribution is heavy-tailed, MSE loss in z-score space still overweights the
large-magnitude samples. The `asinh` transform (inverse hyperbolic sine) is a
smooth, differentiable log-like compression that handles negative values:
`asinh(x) ≈ log(2x)` for large |x|, linear near zero. Predicting `asinh(p)`
instead of `p` compresses high-pressure magnitudes, making the loss more
sensitive to low-pressure errors (which matter physically for lift/stall).

The scoring function in `data/scoring.py` is read-only and always evaluates in
original space. But if the model predicts `asinh(p_normalized)` during training,
we can invert via `sinh()` before calling `accumulate_batch`.

### Implementation recipe

This change is applied only to the pressure channel (channel index 2):

```python
# After y_norm = (y - stats["y_mean"]) / stats["y_std"]
# Apply asinh to pressure channel in training and eval
import torch

ASINH_SCALE = 1.0  # try 1.0 first; controls the "knee" of the compression

def apply_asinh_p(y_norm):
    y_t = y_norm.clone()
    y_t[..., 2] = torch.asinh(y_norm[..., 2] * ASINH_SCALE) / ASINH_SCALE
    return y_t

def invert_asinh_p(pred_t):
    pred = pred_t.clone()
    pred[..., 2] = torch.sinh(pred_t[..., 2] * ASINH_SCALE) / ASINH_SCALE
    return pred

# In training loop and evaluate_split:
y_t = apply_asinh_p(y_norm)
pred = model({"x": x_norm})["preds"]
sq_err = (pred - y_t) ** 2  # loss in transformed space

# For metric computation (accumulate_batch, etc.):
pred_inv = invert_asinh_p(pred)  # back to normalized space
pred_orig = pred_inv * stats["y_std"] + stats["y_mean"]  # back to original space
```

The `evaluate_split` function must be updated to apply the same transform to
`y_norm` and invert predictions before calling `accumulate_batch`. The
normalization stats remain unchanged — we compose `asinh` on top of z-score.

**Exact CLI:** `python train.py --wandb_group asinh-pressure-output`

### Expected value: Medium

This is a well-motivated output representation change. The asinh transform has
been used successfully in precipitation prediction (NWP ML) and in financial
time-series models where the target is heavy-tailed. For pressure prediction
at high Re (large pressure gradients near the leading edge), this should reduce
loss domination by stagnation-point samples and improve suction-peak accuracy
which is physically critical for lift.

### Risk notes

- The inversion `sinh()` can produce large values if the model outputs large
  predictions in transformed space. Clamp `pred_t[..., 2]` to e.g. ±10 before
  inversion during eval.
- Requires consistent application in both the training loop AND `evaluate_split`.
  Missing the inversion in one place is a silent bug — per-split MAE will be
  inflated without error.
- ASINH_SCALE controls the "knee": smaller scale = more linear behavior, larger
  scale = more compression. Start with ASINH_SCALE=1.0 (operates on z-scored
  values which are already O(1)).

---

## H-04: `dropout-regularization`

### Mechanism

The current `PhysicsAttention` and `TransolverBlock` already accept a `dropout`
parameter, but `model_config` passes `dropout=0.0` (default). With only ~200
training samples and a moderately-sized transformer, adding dropout targets
co-adaptation of attention heads. This is a classical regularization lever for
transformers that is orthogonal to EMA (trajectory smoothing) and weight decay
(norm constraint). In vision transformer literature, even small dropout rates
(0.1–0.2) can improve generalization when the training set is small relative to
model capacity.

The key observation: `dropout=0.0` in the attention module means
`PhysicsAttention.dropout` is a `nn.Dropout(0.0)` no-op. Setting `dropout=0.1`
enables `F.scaled_dot_product_attention(dropout_p=0.1)` during training AND
`self.to_out = nn.Sequential(..., nn.Dropout(0.1))` in the output projection.

### Implementation recipe

```python
# In model_config dict, change:
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    dropout=0.1,   # was 0.0 — try 0.1 first, then 0.2 if promising
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Test two arms: `dropout=0.1` and `dropout=0.2`.

**Exact CLI:**
```
# Arm 1: dropout=0.1 (edit model_config directly in train.py)
python train.py --wandb_group dropout-regularization

# Arm 2: dropout=0.2 (edit model_config)
python train.py --wandb_group dropout-regularization
```

Note: `dropout` is not in the `Config` dataclass — the student should add it as
a config field if they want to pass it as a CLI arg, or simply hard-code two
separate runs by editing `model_config["dropout"]` directly in `train.py`.

### Expected value: Medium

Very low implementation risk. Dropout is likely to help in the low-data OOD
regime. The risk is that with EMA already in the stack, dropout may interact
negatively — EMA already acts as an implicit regularizer and adding dropout on
top may reduce training signal too much at 14-epoch budget.

### Risk notes

- Dropout and EMA: during training, dropout introduces noise; EMA smooths across
  the noisy trajectory. The combination can work well (it does in LLMs) but may
  need a slightly higher LR to compensate.
- `dropout=0.2` with only 14 realized epochs may under-converge. Prefer 0.1 first.
- `PhysicsAttention.dropout` is used as `dropout_p` in `F.scaled_dot_product_attention`
  during training but is set to 0 during `.eval()` mode — this is correct behavior.

---

## H-05: `foil-id-token`

### Mechanism

The current 24-dimensional node features include NACA params for both foils, plus
a `is_surface` flag, but there is NO explicit "which foil am I on?" feature for
surface nodes. For interior (volume) nodes there may be ambiguity about which foil
a node's flow field is governed by. More importantly, for surface nodes, foil-1
and foil-2 surface nodes are structurally identical in feature space (same NACA
params format, same AoA encoding) — the model must infer foil identity from
spatial position alone.

Adding a learned foil-identity token (or a simple one-hot/integer feature)
as an additional input dimension breaks this symmetry explicitly. This is
analogous to positional encoding or segment tokens in NLP. For single-foil
samples, the token is 0 for the only foil. For tandem samples, the token is
1/0 or learned embedding distinguishing foil-1 (front) from foil-2 (rear).

The `data/` module is read-only, but `x` arrives as a `[B, N, 24]` tensor in
`train.py`. We can augment it in `train.py` with a per-node learned embedding.

### Implementation recipe

In `train.py`, add a learnable foil-identity embedding that maps a foil-index
integer to a `d_foil`-dimensional vector, then concatenate to `x` before
normalization:

```python
# Foil index: 0 = single-foil node, 1 = foil-1 in tandem, 2 = foil-2 in tandem
# We derive foil index from existing features.
# According to program.md, feature dim 10 is `is_foil1_surface` and dim 11
# is `is_foil2_surface` (exact indices may vary — verify against X_DIM layout).

# Simpler approach: augment model's preprocess MLP input dim without changing data
# Add a binary feature derived from existing features:
# The is_surface flag is dim 4. The foil-1 vs foil-2 distinction is inferred
# from spatial position (z coordinate, dim 1) or from NACA shape params.

# Cleanest approach that works without data/ changes:
# Add an additional input head that processes a subset of features as "foil context"
# and concatenates to the preprocessed representation.

# Concretely, increase fun_dim by 1 and add a binary foil-which indicator
# derived in train.py from the sign of the x-coordinate (foil-1 is upstream,
# foil-2 is downstream for most tandem configs).
# feature_index = 0 is x (chordwise), feature_index = 1 is z (normal)

# In training/eval loop, after loading x:
# foil_flag = (x[..., 0:1] > 0.5).float()  # rough spatial heuristic
# x_aug = torch.cat([x, foil_flag], dim=-1)  # [B, N, 25]
# Then update fun_dim = X_DIM - 2 + 1 = 23 in model_config

# Better: use learned embedding from a discrete foil-id
# But requires knowing which nodes belong to which foil from the data
```

**Recommended minimal implementation:**
Since we cannot inspect data layout, use a learned continuous embedding on
the chord-normalized x-position to create a foil-context feature:

```python
# Add after model definition:
foil_context_proj = nn.Linear(1, 4).to(device)  # project 1 spatial cue to 4 dims

# Update model_config fun_dim:
model_config["fun_dim"] = X_DIM - 2 + 4  # 22 + 4 = 26

# In training loop, before x_norm:
with torch.no_grad():
    # Use gap/stagger features (last two dims) as foil context cue
    # gap is feature dim 22, stagger is dim 23 (0-indexed, verify in program.md)
    foil_cue = x[..., -2:]  # [B, N, 2] gap + stagger
foil_emb = foil_context_proj(foil_cue[..., :1])  # [B, N, 4]
x_aug = torch.cat([x, foil_emb], dim=-1)  # [B, N, 28]
x_norm = (x_aug - stats_aug["x_mean"]) / stats_aug["x_std"]
```

**Important:** This approach requires re-computing normalization stats for the
augmented input. Alternatively, keep the foil embedding in a separate branch and
add it post-normalization to the preprocessed features.

**Simplest clean implementation:** Add a `foil_embed = nn.Embedding(3, 8)` where
node foil-id (0=single, 1=foil1, 2=foil2) is passed as a separate integer tensor
alongside x, and add the embedding to the preprocessed hidden state. This avoids
changing normalization stats entirely.

**Exact CLI:** `python train.py --wandb_group foil-id-token`

### Expected value: Medium

The structural pattern across Round-1 is that RC-camber OOD gains most from
any change that helps the model process pressure on novel geometry. Explicit foil
disambiguation may help the model specialize its attention to the correct foil
surface rather than averaging across both. Risk is implementation complexity.

### Risk notes

- The biggest risk is implementation correctness — if the foil-id derivation from
  features is wrong, this adds noise rather than signal.
- Verify the exact feature layout from `program.md` or by inspecting a batch
  tensor before implementing. The X_DIM=24 feature layout must be understood
  precisely.
- If this changes `fun_dim`, the normalization stats in `stats.json` no longer
  cover the new dims — the student must handle augmented dims with appropriate
  normalization (e.g., standardize foil-id embedding output to zero mean unit std).

---

## H-06: `physics-continuity-loss`

### Mechanism

Incompressible 2D flow satisfies continuity: `∂Ux/∂x + ∂Uy/∂z = 0` at every
interior (volume) node. The current MSE loss does not enforce this; it treats
Ux and Uy as independent outputs. Adding a weak continuity penalty on volume
nodes encodes known physics directly into the loss, potentially reducing the
hypothesis space the model needs to search.

This is a soft physics-informed constraint — not a hard constraint like imposing
it exactly, but a penalty term that nudges predictions toward physical consistency.
The implementation uses finite differences on the irregular mesh. For each volume
node, compute an approximate `∂Ux/∂x + ∂Uy/∂z` using the nearest neighbors.

Simpler alternative: use the predicted divergence as a proxy loss term computed
from the attention output itself (no neighbor lookup needed). But the cleanest
approach uses the spatial coordinates which are already in the input tensor.

### Implementation recipe

Simple version using finite-difference approximation on a structured sub-sample:

```python
# In training loop, after forward pass (pred shape: [B, N, 3])
# pred[..., 0] = Ux, pred[..., 1] = Uy, pred[..., 2] = p
# x[..., 0] = x-coord (normalized), x[..., 1] = z-coord (normalized)

# Divergence penalty on volume nodes only
# Use a simple spatial gradient approximation:
# For each batch, find pairs of adjacent nodes and compute finite differences.
# This is expensive for 85K-242K nodes. Use random subsampling instead.

CONTINUITY_WEIGHT = 0.1  # start low — this is a weak regularizer
N_DIV_SAMPLES = 512  # sample this many node pairs per batch item

def continuity_penalty(pred, x_orig, vol_mask, n_samples=N_DIV_SAMPLES):
    """Approximate divergence penalty via random node pairs."""
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        # Get volume node indices for this batch item
        vol_idx = vol_mask[b].nonzero(as_tuple=True)[0]
        if len(vol_idx) < 2:
            continue
        # Subsample n_samples random pairs
        n = min(n_samples, len(vol_idx) - 1)
        perm = torch.randperm(len(vol_idx) - 1, device=pred.device)[:n]
        idx_a = vol_idx[perm]
        idx_b = vol_idx[perm + 1]  # simple sequential pair approximation
        
        dx = x_orig[b, idx_b, 0] - x_orig[b, idx_a, 0]  # delta x
        dz = x_orig[b, idx_b, 1] - x_orig[b, idx_a, 1]  # delta z
        dUx = pred[b, idx_b, 0] - pred[b, idx_a, 0]
        dUy = pred[b, idx_b, 1] - pred[b, idx_a, 1]
        
        # Approximate div via directional derivative
        r2 = dx**2 + dz**2 + 1e-10
        div = (dUx * dx + dUy * dz) / r2  # approximate divergence
        total += (div**2).mean()
    return total / B

# In training loss computation:
div_penalty = continuity_penalty(pred, x_norm, vol_mask)
loss = vol_loss + cfg.surf_weight * surf_loss + CONTINUITY_WEIGHT * div_penalty
```

**Exact CLI:** `python train.py --wandb_group physics-continuity-loss`

### Expected value: Medium

Physics-informed losses are a well-studied direction for CFD surrogates (PINN
literature, Raissi et al. 2019). The main question is whether a soft penalty adds
real value when the training data already implicitly encodes continuity. The
penalty is especially likely to help in OOD generalization (the model has to
satisfy physics regardless of training distribution).

### Risk notes

- The finite-difference divergence approximation on irregular meshes is noisy.
  Random node pairs are especially noisy unless nodes are sorted spatially.
  Consider using only nodes within the same structured block if the mesh has
  block structure, or use the k-nearest neighbors approach.
- The continuity loss is computed on normalized Ux/Uy. The normalization stats
  for Ux and Uy have different scales, which affects the gradient magnitudes of
  the divergence term. Consider computing div in original space: invert
  normalization before computing the penalty.
- CONTINUITY_WEIGHT=0.1 is a starting point. If the divergence loss dominates
  the main MSE loss, lower to 0.01.
- The inner loop over batch items is slow for large meshes. Use vectorized ops
  or limit to B=1 batch subsets.

---

## H-07: `lr-one-cycle`

### Mechanism

The current LR schedule is CosineAnnealingLR with T_max=MAX_EPOCHS=50. Under
the 30-min wall-clock cap only ~14 epochs land, meaning the cosine schedule
operates on an effective T_max=50 but sees only 14/50 = 28% of its period. The
LR starts at 5e-4 and only decays to ~4.6e-4 by epoch 14 — nearly flat. The
cosine tail (which provides the final fine-tuning benefit) never activates.

OneCycleLR resolves this by scaling the schedule to the actual realized
iterations, with a rapid rise to max_lr and an aggressive decay to near-zero
by the final epoch. This has two effects: (1) the warmup is short and automatic,
(2) the tail decay actually fires within the realized budget. FastAI research
(Smith & Topin 2018) showed OneCycleLR consistently outperforms fixed cosine
on the same budget in image classification and regression tasks.

This is different from the `lr-warmup-cosine` hypothesis that was already tried:
that hypothesis added a separate warmup phase before cosine. OneCycleLR uses a
single unified schedule with a `pct_start` warmup fraction.

### Implementation recipe

```python
# Replace the CosineAnnealingLR line:
# OLD: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
# NEW:

# Estimate total_steps from training set size and batch_size
# train_ds size: ~200 samples, batch_size=4, so ~50 steps/epoch, ~700 steps total at 14 epochs
# Use a conservative estimate; OneCycleLR adapts to actual step count
ESTIMATED_STEPS = len(train_loader) * MAX_EPOCHS  # upper bound on steps

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,          # peak LR — 2x the current base lr
    total_steps=ESTIMATED_STEPS,
    pct_start=0.1,        # 10% of steps for warmup (not 5 full epochs)
    anneal_strategy="cos",
    div_factor=25.0,      # initial_lr = max_lr / 25 = 4e-5
    final_div_factor=1e4, # final_lr = initial_lr / 1e4 ≈ 4e-9
)

# Change scheduler.step() to be per-batch (inside the training loop):
# Move scheduler.step() from after the epoch loop to inside the batch loop:
# optimizer.step()
# scheduler.step()   # <-- inside batch loop
```

Key: OneCycleLR steps per batch, not per epoch. The `scheduler.step()` call
moves from after the epoch to inside the training batch loop.

**Exact CLI:** `python train.py --wandb_group lr-one-cycle`

Also try `max_lr=5e-4` (same peak as current) and `max_lr=2e-3` (4x current):
```
python train.py --wandb_group lr-one-cycle  # with max_lr=1e-3 (default)
python train.py --wandb_group lr-one-cycle  # with max_lr=5e-4 (conservative)
```

### Expected value: High

This directly targets the identified LR schedule mismatch. The root cause
(cosine schedule not reaching its tail within realized budget) is confirmed by
experimental evidence (lr-warmup-cosine failure analysis). OneCycleLR is the
canonical fix for this type of budget mismatch. High EV, low implementation risk.

### Risk notes

- OneCycleLR expects `scheduler.step()` to be called after EVERY batch, not
  after every epoch. Calling it per-epoch instead of per-batch is a common
  mistake that will mis-calibrate the schedule.
- `total_steps` must be set to the correct total number of expected batches.
  Use `len(train_loader) * MAX_EPOCHS` as the upper bound. If training hits
  wall-clock timeout before MAX_EPOCHS, the schedule is "ahead of time" but
  continues working — the LR just reaches its minimum earlier.
- EMA update should remain per-step as it currently is — no change needed there.
- Do NOT use a warmup-cosine schedule on top of OneCycleLR. Choose one or the other.

---

## H-08: `surface-tangential-smoothness`

### Mechanism

Surface pressure on an aerodynamic foil should vary smoothly along the chord
except at the trailing edge (sharp corner) and the suction peak. Large discontinuities
in predicted pressure along the surface arc-length indicate non-physical
oscillations. Adding a surface tangential smoothness penalty penalizes large
differences in predicted pressure between adjacent surface nodes.

This targets a specific known failure mode for ML surrogates: they can predict
accurate individual node values but with high-frequency oscillations along the
surface that are physically implausible. The smoothness penalty provides a soft
inductive bias toward physically realistic pressure distributions.

The surface arc-length (`saf`, signed arc-length from leading edge) is already
encoded in the input features (one of the 24 dims), so surface node ordering
is accessible from the input tensor without changes to `data/`.

### Implementation recipe

```python
# In training loop, after computing surf_loss:
# Apply surface tangential smoothness on pressure channel only

SMOOTH_WEIGHT = 0.05  # start low; this is a soft inductive bias

def surface_tangential_smooth(pred, x_orig, surf_mask, smooth_weight):
    """
    Penalize large differences in pred pressure between adjacent surface nodes.
    Uses signed arc-length (saf) feature to approximate adjacency.
    saf is feature dim 2 in x (0=x_coord, 1=z_coord, 2=saf).
    """
    # x_orig: [B, N, 24], pred: [B, N, 3]
    # surf_mask: [B, N] bool
    B = pred.shape[0]
    total = 0.0
    count = 0
    for b in range(B):
        sidx = surf_mask[b].nonzero(as_tuple=True)[0]
        if len(sidx) < 2:
            continue
        # Sort surface nodes by arc-length (dim 2)
        saf_vals = x_orig[b, sidx, 2]  # signed arc-length
        sort_order = saf_vals.argsort()
        sidx_sorted = sidx[sort_order]
        
        # Pressure channel = dim 2 of pred
        p_surf = pred[b, sidx_sorted, 2]  # [n_surf]
        # Penalize finite differences
        dp = p_surf[1:] - p_surf[:-1]
        total += (dp**2).mean()
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=pred.device)
    return smooth_weight * total / count

# In training loss:
smooth_penalty = surface_tangential_smooth(pred, x_norm, surf_mask, SMOOTH_WEIGHT)
loss = vol_loss + cfg.surf_weight * surf_loss + smooth_penalty
```

**Exact CLI:** `python train.py --wandb_group surface-tangential-smoothness`

Also try SMOOTH_WEIGHT=0.1 as a second arm.

### Expected value: Medium

Surface smoothness penalties are used in geometry optimization and CAD-ML
literature. For aerodynamics, a smooth Cp curve is both a physical prior and
a practical desideratum. The risk is that sorting by arc-length per batch item
has O(N log N) cost for N_surf nodes per item, which may slow down the training
loop noticeably at batch_size=4 with large meshes.

### Risk notes

- Arc-length feature dim must be verified against `program.md`. The description
  says `saf` (signed arc-length from leading edge) is in the input features — but
  the exact index in the 24-dim vector must be confirmed.
- Sorting per batch item per epoch step is O(N_surf log N_surf). For N_surf ~5K
  nodes per item at batch_size=4, this is ~80ms per step — acceptable but not free.
- The smoothness penalty is computed in normalized space. Pressure normalization
  compresses large variations, so the penalty signal may be weaker than in original
  space. Consider applying it in a demi-normalized space (scale only, no offset).
- Trailing-edge nodes have a physical discontinuity. To avoid penalizing this,
  either ignore nodes within `|saf| > 0.45` (near TE) or clip the penalty at a
  threshold: `min(dp**2, clip_val**2).mean()`.

---

## H-09: `longer-budget-ema-winner`

### Mechanism

The EMA winner (fern, decay=0.999, val_avg=121.69) ran under the same 30-min /
50-epoch cap as all Round-1 experiments, landing ~14 epochs. The model may not be
fully converged — the loss curve is still declining at epoch 14 for all four splits.
A simple experiment: rerun the exact EMA configuration with `SENPAI_TIMEOUT_MINUTES=55`
(near the hardware limit for a single GPU) and `MAX_EPOCHS=50` (unchanged), giving
the same architecture significantly more wall clock. This is not an architecture
or loss change — it is a compute scaling test.

This tests whether the EMA plateau is optimization-limited (more epochs help) or
architecture-limited (more epochs do not help). A 5-10% additional gain would
indicate under-convergence. No gain would indicate the current architecture/loss
is at its capacity under the given budget.

### Implementation recipe

Run the exact EMA baseline configuration (decay=0.999, surf_weight=10,
lr=5e-4, weight_decay=1e-4, batch_size=4) but with:

```bash
SENPAI_TIMEOUT_MINUTES=55 python train.py \
    --epochs 50 \
    --wandb_group longer-budget-ema-winner
```

The `SENPAI_TIMEOUT_MINUTES` env var controls `MAX_TIMEOUT_MIN` in `train.py`
(line: `MAX_TIMEOUT_MIN = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))`).
Setting it to 55 gives ~22 epochs instead of ~14 (epochs land at ~2.5 min each).

No code change to `train.py` is needed — this is a pure environment variable override.

**Exact CLI:**
```
SENPAI_TIMEOUT_MINUTES=55 python train.py --wandb_group longer-budget-ema-winner
```

### Expected value: High

If the model is under-converged (very likely given the loss curves), extending
the wall-clock budget is the highest-leverage lever available. This experiment
has near-zero implementation risk and directly tests whether the current EMA
winner can be pushed further before the next round of architectural experiments.

If val_avg improves significantly (say to <115), this becomes the new baseline for
all subsequent experiments and changes the interpretation of what is achievable in
the 30-min budget.

### Risk notes

- This is not "training longer" in a different configuration — it must replicate
  the exact EMA winner configuration to be a clean apples-to-apples comparison.
- The 55-min timeout is an estimate; the actual hardware limit may be different.
  If the GPU pod has a hard job time limit, the student should confirm before
  submitting.
- If val_avg does not improve beyond epoch 14 with more budget, this rules out
  under-convergence as the main bottleneck and redirects attention to architecture
  and loss.

---

## H-10: `geometry-augmentation-vertical-mirror`

### Mechanism

Single-foil configurations are symmetric under vertical mirror (z → -z, AoA → -AoA,
Uy → -Uy). For tandem configurations, the stagger geometry (one foil above the other)
makes full vertical mirror valid only if foil positions are also mirrored. This
augmentation is well-studied for airfoil ML surrogates — it doubles the effective
dataset size for single-foil samples and introduces the pressure-symmetric counterpart
of each training case.

This is a data augmentation applied in the training loop (not requiring changes to
`data/`). The augmentation is: flip sign of z-coordinate, flip sign of AoA inputs,
flip sign of Uy output.

The input feature layout is critical. From `program.md`:
- Feature 0: x (chordwise) — unchanged under mirror
- Feature 1: z (normal/vertical) — sign flip
- Feature 3: AoA (angle of attack) — sign flip
- Feature 12-13: foil gap, stagger — for tandem, stagger sign flips

Output:
- Channel 0: Ux — unchanged under mirror
- Channel 1: Uy — sign flip
- Channel 2: p — unchanged under mirror

### Implementation recipe

```python
# Define augmentation function (apply in training loop)
# IMPORTANT: verify exact feature indices against program.md before running

Z_COORD_DIM = 1      # z-coordinate feature index
AOA_DIMS = [5, 9]    # AoA dims for foil-1 and foil-2 (verify in program.md)
UY_CHANNEL = 1       # Uy output channel index

AUGMENT_PROB = 0.5   # apply augmentation to half the batch

def vertical_mirror_augment(x, y, prob=AUGMENT_PROB):
    """
    Apply vertical mirror augmentation with probability prob.
    x: [B, N, 24], y: [B, N, 3]
    Returns augmented (x, y).
    """
    if torch.rand(1).item() > prob:
        return x, y
    
    x_aug = x.clone()
    y_aug = y.clone()
    
    # Flip z-coordinate
    x_aug[..., Z_COORD_DIM] = -x[..., Z_COORD_DIM]
    
    # Flip AoA for all foils
    for dim in AOA_DIMS:
        x_aug[..., dim] = -x[..., dim]
    
    # Flip Uy output
    y_aug[..., UY_CHANNEL] = -y[..., UY_CHANNEL]
    
    return x_aug, y_aug

# In training loop, after loading x, y:
x, y = vertical_mirror_augment(x, y)
# Then proceed with normalization as usual
```

**Exact CLI:** `python train.py --wandb_group geometry-augmentation-mirror`

### Expected value: Medium-High

Dataset augmentation for airfoil symmetry is a well-established technique.
The main risk is that the feature indices for AoA, z-coordinate, and stagger
in the 24-dim input must be verified precisely. An incorrect augmentation
will corrupt the inputs rather than help.

For single-foil samples: vertical mirror is exact (z → -z, AoA → -AoA, Uy → -Uy).
For tandem samples: the augmentation is valid if stagger is also flipped.
Start with single-foil-only augmentation (apply only when a single-foil indicator
feature can be detected, e.g., when foil-2 NACA params are zero).

### Risk notes

- **Feature index verification is critical.** Before implementing, print out a
  sample batch and inspect the 24 feature dimensions against `program.md`. An
  off-by-one error here produces corrupted training data and a misleading result.
- For tandem samples, the mirror transform requires flipping stagger sign. Applying
  the augmentation to tandem samples without flipping stagger is physically incorrect
  (it would create a different tandem configuration, not the mirror image).
- Start with AUGMENT_PROB=0.5 (augment half the batch). If the training loss
  oscillates more than baseline, reduce to 0.3.
- The normalization stats (stats.json) are computed on the original (un-augmented)
  training set. The augmented features (flipped z, flipped AoA) will be outside the
  "normal" range for those features after z-score normalization. For z-coordinate
  (symmetric around zero) this is fine. For AoA, which may span an asymmetric range
  in the training set, a negative AoA may produce an out-of-distribution normalized
  value. Confirm AoA range in the training data first.
