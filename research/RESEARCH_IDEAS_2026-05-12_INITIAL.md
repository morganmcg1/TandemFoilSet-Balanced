<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — TandemFoilSet CFD Surrogate (2026-05-12, Initial Round)

Generated from first principles + literature search. Zero prior round results consulted.

Baseline: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0, CosineAnnealingLR T_max=epochs.
Primary target: val_avg/mae_surf_p (lower is better).

---

## Rank 1 — `pressure-channel-prioritized-loss`

**Slug:** `pressure-channel-prioritized-loss`

**Predicted delta:** −5% to −12% on val_avg/mae_surf_p

### Mechanism

The training objective is MSE of all three output channels (Ux, Uy, p) equally weighted in
normalized space. But the primary ranking metric is surface pressure MAE only. Pressure (channel 2)
has different gradient signal characteristics from velocity channels. By up-weighting the pressure
channel in the loss, we explicitly align the optimization objective with the evaluation metric.

The normalized-space loss is:

    loss = vol_loss + surf_weight * surf_loss
         = mean_over_nodes(sq_err) + 10 * mean_over_surface(sq_err)

where sq_err is averaged over all 3 channels equally. Replacing this with channel-specific weights
means the pressure channel receives a stronger gradient signal on surface nodes specifically — the
exact nodes and channel that determine the ranking metric.

This is a targeted objective realignment, not a change to the architecture or optimizer. It is
among the simplest possible changes (two scalar hyperparameters) with strong mechanistic grounding.

### Concrete changes to train.py

In the training loop (line ~493-496), replace:

```python
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

with:

```python
# Channel weights: [Ux_weight, Uy_weight, p_weight]
# Applied to sq_err [B, N, 3] before aggregation
chan_weights = torch.tensor([1.0, 1.0, cfg.pressure_weight], device=device).float()
sq_err_w = sq_err * chan_weights[None, None, :]
vol_loss = (sq_err_w * vol_mask.unsqueeze(-1)).sum() / (vol_mask.sum().clamp(min=1) * 3)
surf_loss = (sq_err_w * surf_mask.unsqueeze(-1)).sum() / (surf_mask.sum().clamp(min=1) * 3)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add `pressure_weight: float = 3.0` to the Config dataclass.

Also apply the same channel weighting in `evaluate_split` for the `loss`/`surf_loss`/`vol_loss`
monitoring keys (optional — these are just logged, not used for checkpoint selection).

### Hyperparameters

- `pressure_weight = 3.0` (primary setting; also try 5.0 in a follow-up arm)
- `surf_weight = 10.0` (unchanged)
- All other baseline HPs unchanged

### Wall-clock cost

Negligible overhead (one tensor multiply per step). Same cost as baseline (~25–28 min for 50
epochs on a 96GB GPU).

### Risk

Low. The mechanism is direct and falsifiable: if surface pressure MAE improves, the channel
up-weighting worked. Risk is that velocity channels degrade enough to affect downstream use, but
that is irrelevant to the ranking metric. Worst case: pressure improves, velocity degrades; the PR
metric still improves. The only failure mode is if the optimizer destabilizes, which is unlikely
for a 3x scalar multiplier.

---

## Rank 2 — `warmup-cosine-lr`

**Slug:** `warmup-cosine-lr`

**Predicted delta:** −4% to −10% on val_avg/mae_surf_p

### Mechanism

The baseline uses `CosineAnnealingLR(T_max=MAX_EPOCHS)` starting immediately at lr=5e-4. On a
dataset with variable-scale inputs (Re varies 50x across samples; y std varies 10x within a
single domain), the early gradient signal is dominated by the high-Re, high-variance samples. A
warm-up phase (linear ramp from a small lr to peak) allows the model's slice-assignment projections
— which are initialized orthogonally but untrained — to stabilize before the optimizer takes large
steps. This is particularly important for the `in_project_slice` layer (init: orthogonal), whose
temperature parameter (init: 0.5) shapes the slice softmax distribution.

LR warmup is near-universal in modern transformer training and has been shown to matter most for
models with softmax-gated aggregation (the slice attention is exactly this class). Without warmup,
early large-step updates can push the temperature or slice weights into saturation before the
representations are meaningful.

Additionally, the current CosineAnnealingLR does not restart. Adding a warmup prefix effectively
makes the first phase a conservative initialization pass before the cosine decay begins.

### Concrete changes to train.py

Replace the scheduler definition (line ~435):

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

with:

```python
warmup_epochs = cfg.warmup_epochs  # default: 5
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(max(1, MAX_EPOCHS - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

Add `import math` at the top. Add `warmup_epochs: int = 5` to Config.

### Hyperparameters

- `warmup_epochs = 5` (linear ramp over first 5 epochs; ~10% of budget)
- `lr = 5e-4` (unchanged; this is the peak)
- All other baseline HPs unchanged

### Wall-clock cost

Zero overhead. Same duration as baseline.

### Risk

Low. The scheduler change does not touch the model or loss. Failure mode: if the warm-up is too
long it wastes training budget; 5 epochs of 50 is a standard ratio. A shorter warmup (3 epochs)
is a safe fallback if 5 underperforms.

---

## Rank 3 — `wider-mlp-ratio`

**Slug:** `wider-mlp-ratio`

**Predicted delta:** −3% to −8% on val_avg/mae_surf_p

### Mechanism

The baseline uses `mlp_ratio=2`, meaning the feed-forward hidden dimension in each TransolverBlock
is 2 × n_hidden = 256. Standard transformers (ViT, GPT-2, BERT) use mlp_ratio=4. The MLP in
each block is the primary point-wise nonlinear transformation that maps slice-aggregated
representations into the final per-node features. A wider MLP enables richer per-node feature
interaction without changing the attention complexity.

For CFD surrogates, the MLP is especially important because it must represent the mapping from
aggregate flow statistics (captured by slice attention) back to local field values at individual
nodes. Increasing from 2 to 4 roughly doubles the MLP parameter count while leaving the
attention computation (the expensive O(S²) term) unchanged. This is a capacity increase in the
exact sub-module responsible for spatial detail recovery.

With n_hidden=128 and mlp_ratio=4, MLP hidden dim becomes 512. Model goes from ~2.4M to ~3.8M
parameters — still well within VRAM budget.

### Concrete changes to train.py

In model_config (line ~425):

```python
model_config = dict(
    ...
    mlp_ratio=4,   # was 2
    ...
)
```

That's the only change. The `MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, ...)` call
in TransolverBlock expands automatically.

### Hyperparameters

- `mlp_ratio = 4` (doubled from baseline)
- All other model and training HPs unchanged

### Wall-clock cost

Marginal increase (~5–10% extra forward/backward FLOPS from the MLP layers). Should remain within
the 30-min cap.

### Risk

Low-medium. The model simply becomes wider. The only risk is slight overfitting on the smaller
val splits, but the training set is large enough (~1500 samples × ~100K nodes each) that this
is unlikely. The baseline mlp_ratio=2 is unusually low for a transformer; mlp_ratio=4 is the
natural first correction.

---

## Rank 4 — `gradient-clipping-and-higher-lr`

**Slug:** `gradient-clipping-and-higher-lr`

**Predicted delta:** −3% to −8% on val_avg/mae_surf_p

### Mechanism

The baseline has no gradient clipping. With samples spanning y std from ~50 (low-Re cruise) to
~2,000 (high-Re raceCar single), the gradient norms from different samples are wildly
heterogeneous within the same batch. High-variance samples produce large loss values that generate
large gradients and can destabilize the temperature parameters and slice weights. Gradient clipping
(max_norm=1.0) caps the worst-case updates and allows training at a higher base learning rate
without divergence.

Combined: clip gradients to max_norm=1.0 and increase lr from 5e-4 to 1e-3. The clipping prevents
the large-gradient samples from taking oversized steps, while the higher lr allows faster
convergence on the majority of steps that remain within the clip threshold. This combination is
standard practice in transformer training (e.g., GPT training uses clip=1.0 universally).

### Concrete changes to train.py

In the training loop, replace (line ~499-500):

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

with:

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
optimizer.step()
```

Change `lr = 5e-4` to `lr = 1e-3` in Config defaults. Add `grad_clip: float = 1.0` to Config.

### Hyperparameters

- `lr = 1e-3` (2x baseline)
- `grad_clip = 1.0`
- All other baseline HPs unchanged

### Wall-clock cost

Negligible (one norm computation per step).

### Risk

Medium. Higher lr could diverge in the first few epochs if the gradient clips are ineffective.
Monitoring the first 5 epochs' train loss is sufficient to detect divergence. If train loss
spikes, lr=7e-4 with same clipping is the fallback.

---

## Rank 5 — `larger-hidden-dim`

**Slug:** `larger-hidden-dim`

**Predicted delta:** −3% to −7% on val_avg/mae_surf_p

### Mechanism

The baseline uses n_hidden=128 with 5 layers, giving ~2.4M parameters. For a dataset with 74K–
242K mesh nodes per sample and 3 output channels (Ux, Uy, p) with large-scale Re-driven variance,
the model capacity may be the active bottleneck. Doubling the hidden dimension to 256 quadruples
the attention capacity (attention dim scales as n_hidden²) and doubles the MLP capacity. This is
the most direct capacity increase.

n_hidden=256 with n_head=8 (dim_head=32) and mlp_ratio=2 gives ~9.4M parameters — a standard
"small transformer" size. The 96GB GPU can comfortably hold this with batch_size=4 and meshes up
to 242K nodes.

VRAM estimate: with n_hidden=256, the slice attention operates on [B, H, S, C] = [4, 8, 64, 32]
tensors, which is negligible. The dominant VRAM cost is the node-level activation tensors
[B, N, n_hidden] = [4, 242K, 256] ≈ 1.0 GB per layer — within budget even for 5 layers.

### Concrete changes to train.py

In model_config (lines ~417-428):

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,    # was 128
    n_layers=5,
    n_head=8,        # was 4 (keep dim_head=32)
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Also reduce batch_size from 4 to 2 as a precaution (memory doubles):

```python
batch_size: int = 2  # was 4
```

And increase lr slightly to compensate for smaller batch: `lr = 6e-4` (linear scaling rule for
batch size halved suggests lr = 5e-4 * sqrt(2) ≈ 7e-4; 6e-4 is conservative).

### Hyperparameters

- `n_hidden = 256`
- `n_head = 8` (maintains dim_head=32)
- `batch_size = 2`
- `lr = 6e-4`
- All other baseline HPs unchanged

### Wall-clock cost

~1.5–1.8x slower per epoch (larger activations, more FLOPS). May hit 30-min cap before completing
50 epochs; expect ~30–35 epochs to complete. Still sufficient for meaningful signal.

### Risk

Medium. VRAM usage is the main concern with large meshes (242K nodes, batch=2, n_hidden=256).
Estimate: [2, 242K, 256] × 4 bytes × 5 layers ≈ 2.5 GB activation + gradient ≈ 5 GB — safe on
96GB. The risk is that the effective batch size reduction (from 4 to 2) slows convergence. This
is mitigated by the lr adjustment.

---

## Rank 6 — `more-slices`

**Slug:** `more-slices`

**Predicted delta:** −2% to −6% on val_avg/mae_surf_p

### Mechanism

The Transolver uses slice_num=64 slices. Each slice aggregates a weighted set of nodes and then
attends to other slices. With 3 distinct mesh zones (background + up to 2 dense foil zones) and
complex flow topology (boundary layers, wake regions, stagnation zones), 64 slices may not be
sufficient to partition the mesh into physically meaningful groups.

Transolver++ (Luo et al., ICML 2025, arXiv 2505.02107) demonstrates that richer slice
representations with local adaptive mechanisms significantly improve prediction quality. While the
full Transolver++ architecture requires more invasive changes, the cheapest approximation is simply
to increase slice_num from 64 to 128. This gives the model twice as many "physical state" groups
to work with, enabling finer partitioning of the flow field (e.g., separate slices for the front-
foil suction side vs. pressure side, wake behind foil 1, stagnation near foil 2).

The computational cost of attention in slice space is O(slice_num²), so doubling slice_num from
64 to 128 quadruples the S×S attention cost — but since S is tiny (64 or 128) compared to N
(100K+), this is negligible in absolute terms.

### Concrete changes to train.py

In model_config (line ~424):

```python
model_config = dict(
    ...
    slice_num=128,   # was 64
    ...
)
```

Only this one parameter changes.

### Hyperparameters

- `slice_num = 128`
- All other baseline HPs unchanged (batch_size=4, lr=5e-4, n_hidden=128)

### Wall-clock cost

Near-zero overhead. The S×S attention is tiny relative to the N-level operations.

### Risk

Low. This is a pure capacity increase with trivial cost. The risk is that more slices do not
capture better physical structure — if the existing 64 slices already saturate the representation,
64 more will be redundant. But the cost is so low that even a null result is informative.

---

## Rank 7 — `per-sample-re-normalized-loss`

**Slug:** `per-sample-re-normalized-loss`

**Predicted delta:** −3% to −9% on val_avg/mae_surf_p

### Mechanism

The most important structural problem with the current training regime: global normalization
(y_mean, y_std from stats.json) cannot account for the order-of-magnitude variation in per-sample
y std driven by Re. A low-Re (Re≈100K) sample has y std ≈ 50 in normalized space; a high-Re
(Re≈5M) sample has y std ≈ 2000. After global normalization, high-Re samples still have large
normalized residuals and dominate the MSE loss. The model learns to minimize error on the rare
but high-variance high-Re samples at the expense of the much more numerous low-Re samples.

Fix: compute per-sample inverse-variance weights and normalize each sample's loss contribution
by its batch-estimated variance. In batch terms, for each sample i in the batch, compute
var_i = y_i.var() and weight the loss contribution by 1/var_i.

This is the core idea of Batch Inverse-Variance Weighting (BIVW, Aglin et al. 2020,
arXiv:2009.02165): when samples have heterogeneous variance, uniform MSE loss over-penalizes
high-variance samples and under-trains on low-variance ones. IVW loss re-weights to give each
sample equal importance independent of its absolute scale.

### Concrete changes to train.py

In the training loop, after computing sq_err (line ~490), add per-sample normalization:

```python
# Per-sample inverse-variance weighting for heteroscedastic Re distribution
# y: [B, N, 3]; compute per-sample std from valid (masked) nodes
with torch.no_grad():
    y_var = []
    for b in range(x.shape[0]):
        valid_y = y_norm[b][mask[b]]  # [N_valid, 3]
        y_var.append(valid_y.var().clamp(min=1e-4))
    y_var = torch.stack(y_var)          # [B]
    sample_weights_ivw = 1.0 / y_var    # [B]
    sample_weights_ivw = sample_weights_ivw / sample_weights_ivw.mean()  # normalize mean=1
    sw = sample_weights_ivw[:, None, None]  # [B, 1, 1] for broadcasting

sq_err_w = sq_err * sw  # [B, N, 3], up-weighted for low-variance (low-Re) samples

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err_w * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_w * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

No other changes required.

### Hyperparameters

- All baseline HPs unchanged
- The IVW is parameter-free; the weight normalization (mean=1) ensures the absolute loss scale
  is unchanged

### Wall-clock cost

Minimal. The per-sample variance computation is O(N) and done in a small Python loop over batch
size (≤4 elements). Negligible overhead.

### Risk

Medium. The IVW may cause instability if a single high-variance sample dominates a batch, but
the `clamp(min=1e-4)` and mean normalization guard against this. The main risk is that the
IVW over-rotates the loss toward low-Re samples that are already well-fitted by the baseline,
providing diminishing returns. Observable: check whether low-Re val splits (`val_single_in_dist`,
`val_geom_camber_cruise`) improve while `val_re_rand` stays flat or regresses.

---

## Rank 8 — `surface-aware-output-head`

**Slug:** `surface-aware-output-head`

**Predicted delta:** −2% to −5% on val_avg/mae_surf_p

### Mechanism

The current model uses a single shared output head (mlp2 in the last TransolverBlock) for both
surface and volume nodes. The pressure field behaves qualitatively differently at the surface
(boundary-condition pressure, directly related to aerodynamic forces) vs. in the volume (far-field
pressure recovery, wake dynamics). A separate output MLP for surface nodes vs. volume nodes,
conditioned on the `is_surface` flag, allows the model to specialize its final projection for
these two regimes.

Concretely: after the last TransolverBlock produces the hidden representation fx_out [B, N, d],
apply two separate linear heads, one for surface nodes and one for volume nodes, then combine
with the is_surface mask. The `is_surface` flag is already in the input features (dim 12), so the
model can in principle learn this routing implicitly — but an explicit routing forces it.

### Concrete changes to train.py

In the Transolver.forward method, change the final block to output the hidden representation
(not the final prediction), then add a dual-head projection:

Option A (minimal surgery — preferred): Add a second output head in the Transolver class:

```python
class Transolver(nn.Module):
    def __init__(self, ...):
        ...
        # Dual output heads for surface vs. volume nodes
        self.surf_head = nn.Sequential(
            nn.LayerNorm(n_hidden), nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, out_dim)
        )
        self.vol_head = nn.Sequential(
            nn.LayerNorm(n_hidden), nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, out_dim)
        )
```

Modify the last TransolverBlock to not apply mlp2 (set `last_layer=False` for the final block
in the ModuleList), and have the final block instead return the hidden representation. Then in
`Transolver.forward`, call both heads and blend:

```python
def forward(self, data, **kwargs):
    x = data["x"]
    is_surf = x[..., 12:13]  # [B, N, 1], the is_surface feature (unnormalized 0/1)
    is_surf_bin = (is_surf > 0.5).float()

    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks:
        fx = block(fx)  # last block returns hidden repr (not prediction)

    surf_pred = self.surf_head(fx)   # [B, N, out_dim]
    vol_pred = self.vol_head(fx)     # [B, N, out_dim]
    preds = is_surf_bin * surf_pred + (1 - is_surf_bin) * vol_pred
    return {"preds": preds}
```

This requires changing the last block's `last_layer=False` and removing its mlp2.

### Hyperparameters

- No new HPs; parameter count increases ~0.2M (two extra small MLPs)
- All training HPs unchanged

### Wall-clock cost

Near-zero overhead (two small MLPs applied after the main transformer).

### Risk

Medium. The surgery to the last block's `last_layer` flag is the only tricky part. A
simpler implementation option: keep the existing last_layer head as-is and add a small learned
residual on top of its output conditioned on is_surface. This avoids modifying TransolverBlock:

```python
# After model({"x": x_norm}) returns preds [B, N, 3]:
surf_residual = surf_correction_mlp(x_norm * is_surface.unsqueeze(-1))  # [B, N, 3]
preds = base_preds + surf_residual
```

This is safer and adds only ~0.1M parameters. Recommended implementation for the student.

---

## Summary ranking table

| Rank | Slug | Mechanism level | Predicted delta | Cost | Risk |
|------|------|----------------|-----------------|------|------|
| 1 | pressure-channel-prioritized-loss | Loss formulation | −5% to −12% | Negligible | Low |
| 2 | warmup-cosine-lr | Optimizer/schedule | −4% to −10% | Negligible | Low |
| 3 | wider-mlp-ratio | Architecture (MLP) | −3% to −8% | +5–10% FLOPS | Low-medium |
| 4 | gradient-clipping-and-higher-lr | Optimizer + stability | −3% to −8% | Negligible | Medium |
| 5 | larger-hidden-dim | Architecture (scale) | −3% to −7% | +1.5–1.8x/epoch | Medium |
| 6 | more-slices | Architecture (capacity) | −2% to −6% | Negligible | Low |
| 7 | per-sample-re-normalized-loss | Loss formulation | −3% to −9% | Negligible | Medium |
| 8 | surface-aware-output-head | Architecture (head) | −2% to −5% | Negligible | Medium |

### Rationale for top ranking

Ranks 1–2 are ranked highest because they directly target the mismatch between the training
objective and the evaluation metric with zero cost and low risk. The baseline trains with equal-
weight MSE across 3 channels; the metric measures only pressure on surface nodes. Rank 1 closes
that gap directly. Rank 2 addresses a known transformer training failure mode (early large-step
instability in softmax-gated models) with no downside. These should be the first two assignments.

Ranks 3–4 are standard transformer improvements that are robust across settings. Ranks 5–8 are
higher-risk or higher-cost but still well-grounded in the data properties and architecture.

### Recommended first-round assignment (8 students)

Assign all 8 in parallel. The top 4 are the most likely winners. Ranks 5–8 provide
complementary coverage in case the optimizer-level changes (1, 2, 4) are already near-optimal
or the architecture (3, 5, 6) has more headroom than expected.

**Critical note:** All hypotheses are independent single-variable changes from baseline. If
multiple win, they should be composed in order of improvement magnitude in subsequent rounds.
The most valuable composition is likely: pressure-channel-prioritized-loss + warmup-cosine-lr +
wider-mlp-ratio, as these three are orthogonal changes to loss, optimizer, and architecture.
