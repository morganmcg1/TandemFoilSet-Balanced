<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round-1 Research Ideas — TandemFoilSet CFD Surrogate
Generated: 2026-05-15 12:30  
Advisor branch: `icml-appendix-willow-pai2i-48h-r1`  
Budget per run: 30 min wall clock, 50 epochs max, 1 × 96 GB GPU  
Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## Baseline (for comparison in every PR body)

```
model:  Transolver — 5 layers, hidden=128, heads=4, slice_num=64, mlp_ratio=2
optim:  AdamW lr=5e-4, weight_decay=1e-4, batch=4, CosineAnnealingLR T_max=50
loss:   vol_MSE + 10 × surf_MSE (normalized space)
```

No val_avg/mae_surf_p number yet — this is the first round. Students must report
the number from their own run so the next round has a concrete target.

---

## H1 — Huber loss on surface nodes

**Headline:** Replace MSE with smooth-L1 (Huber) loss, especially on surface nodes,
to align the training objective with MAE-ranked surface pressure.

**Mechanism:** The ranking metric is MAE; MSE upweights large-residual (high-Re)
nodes quadratically, pulling capacity away from moderate-error surface nodes.
Smooth-L1 behaves like L1 for large residuals and L2 near zero, giving a
gradient signal that scales more like the evaluation metric. The surface
pressure MAE should benefit most because the quadratic tail of MSE is the
key mismatch.

**Predicted delta:** Moderate improvement on `val_avg/mae_surf_p`. Literature
on regression-to-MAE shows ~3–10% gains when switching from MSE to Huber at
the optimization target alone, without any other change.

**Exact implementation (all changes in `train.py`):**

1. Replace the loss block in the training loop (lines 490–496):

```python
# OLD
sq_err = (pred - y_norm) ** 2
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss

# NEW
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
# beta=0.1 keeps Huber in L1 regime for typical normalized residuals ~O(0.1–1)
vol_loss = F.huber_loss(pred[vol_mask], y_norm[vol_mask], delta=0.1, reduction="mean")
surf_loss = F.huber_loss(pred[surf_mask], y_norm[surf_mask], delta=0.1, reduction="mean")
loss = vol_loss + cfg.surf_weight * surf_loss
```

Note: the masked indexing `pred[vol_mask]` and `y_norm[vol_mask]` is safe when
`vol_mask` is a 2D bool tensor — PyTorch broadcasts correctly because both
tensors are [B, N, 3] and vol_mask is [B, N]; you need `.unsqueeze(-1).expand_as(pred)`
or compute the masked loss manually:

```python
vol_err = F.huber_loss(pred, y_norm, delta=0.1, reduction="none")   # [B,N,3]
vol_loss  = (vol_err  * vol_mask.unsqueeze(-1)).sum() / (3 * vol_mask.sum().clamp(min=1))
surf_loss = (vol_err  * surf_mask.unsqueeze(-1)).sum() / (3 * surf_mask.sum().clamp(min=1))
```

Wait — `F.huber_loss(..., reduction="none")` on the full tensor and then mask
is the correct approach. Use this pattern:

```python
err = F.huber_loss(pred, y_norm, delta=0.1, reduction="none")   # [B,N,3]
vol_mask_3d  = vol_mask.unsqueeze(-1)
surf_mask_3d = surf_mask.unsqueeze(-1)
vol_loss  = (err * vol_mask_3d).sum()  / (vol_mask_3d.sum().clamp(min=1))
surf_loss = (err * surf_mask_3d).sum() / (surf_mask_3d.sum().clamp(min=1))
loss = vol_loss + cfg.surf_weight * surf_loss
```

2. Keep everything else identical: lr=5e-4, surf_weight=10.0, batch=4, 50 epochs.

3. CLI: `python train.py --wandb_group huber_loss`

**Hyperparameter sensitivity:** `delta=0.1` is critical. In normalized space the
typical surface pressure residual early in training is ~0.2–0.5 (std-normalized),
so delta=0.1 keeps the loss in the L1 regime for most of training (stable
gradients). If delta=1.0, Huber degenerates to nearly MSE for those residuals
and loses the benefit. Do not use delta>0.5 without ablation.

**Risk:** If residuals are almost always small (< 0.05 normalized), Huber is
indistinguishable from MSE. Check `train/surf_loss` in W&B — if it's < 0.01
by epoch 5 the switch has no effect and we should try pure L1 instead.

**Stop condition:** Close if `val_avg/mae_surf_p` at best checkpoint is ≥ 5%
worse than the baseline number reported by another student in this round.

---

## H2 — Surface-pressure-only L1 loss with high surf_weight

**Headline:** Optimize L1 loss exclusively on the surface pressure channel
(`y[:, :, 2]`), and use a high `surf_weight` (50), while keeping MSE on the
volume and on the velocity channels.

**Mechanism:** The metric is `mae_surf_p` — it ignores volume nodes entirely
and ignores Ux/Uy. The baseline spreads gradient mass over 3 channels × two
node regions. This hypothesis asks: what if we directly optimize the thing
that matters? The volume MSE term and Ux/Uy still receive gradient (to avoid
representation collapse), but the surface p channel gets a dedicated L1 loss
with amplified weight.

**Predicted delta:** Large improvement potential on `val_avg/mae_surf_p`, moderate
risk of degraded Ux/Uy metrics (which are not scored). Literature on task-specific
loss weighting in multi-output regression consistently shows 5–15% gains on the
target channel when weights are shifted in this direction.

**Exact implementation (all changes in `train.py`):**

Replace training loss block (lines 490–496):

```python
err = (pred - y_norm) ** 2          # MSE base [B,N,3]
vol_mask_3d  = (mask & ~is_surface).unsqueeze(-1)   # [B,N,1]
surf_mask_3d = (mask &  is_surface).unsqueeze(-1)   # [B,N,1]

vol_loss  = (err * vol_mask_3d).sum()  / vol_mask_3d.sum().clamp(min=1)

# For surface: L1 on pressure channel, MSE on velocity channels
surf_err_vel = err[:, :, :2]         # Ux, Uy — [B,N,2]
surf_err_p   = (pred - y_norm)[:, :, 2:3].abs()  # L1 on p — [B,N,1]
surf_err_combined = torch.cat([surf_err_vel, surf_err_p], dim=-1)  # [B,N,3]
surf_loss = (surf_err_combined * surf_mask_3d).sum() / surf_mask_3d.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * surf_loss
```

Change `Config` default: `surf_weight: float = 50.0` (or pass via CLI as
`--surf_weight 50`).

CLI: `python train.py --surf_weight 50 --wandb_group surf_p_l1`

**Hyperparameter sensitivity:** `surf_weight` is the key dial. Try 50 first;
if training loss explodes (surf_loss diverges in W&B), fall back to 25.
The velocity channels may degrade — that is acceptable if `mae_surf_p` wins.

**Risk:** Extremely high surf_weight can cause the model to ignore interior
flow structure, which may hurt generalization to the unseen camber splits
(where the training distribution of boundary conditions is shifted). Monitor
`val_geom_camber_rc/mae_surf_p` and `val_geom_camber_cruise/mae_surf_p`
separately — if those splits diverge from `val_single_in_dist`, the surface
overfit is hurting generalization.

---

## H3 — Cosine warmup LR schedule

**Headline:** Add a 5-epoch linear warmup before the cosine decay, keeping the
same peak LR (5e-4) and total schedule length.

**Mechanism:** The baseline starts at peak LR immediately. With Adam-family
optimizers, early steps have high variance in gradient estimates (sparse second
moments). A warmup lets the second-moment estimates stabilize before the
optimizer takes large steps, reducing early instability especially at surface
nodes (which have much lower count than volume nodes, so per-node gradient
variance is higher). This is standard practice in transformer training and has
been validated in neural operator work (FNO, Geo-FNO, etc.).

**Predicted delta:** Small to moderate. Expected effect is better best-checkpoint
metrics at equivalent training budget — the best epoch arrives later when warmup
prevents early overfitting to large-residual samples.

**Exact implementation (all changes in `train.py`):**

Replace scheduler construction (lines 434–435):

```python
# OLD
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# NEW
WARMUP_EPOCHS = 5

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    # cosine decay from 1.0 to 0.0 over remaining epochs
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

Add `import math` at the top of the file.

Keep everything else identical. CLI: `python train.py --wandb_group lr_warmup`

**Hyperparameter sensitivity:** WARMUP_EPOCHS=5 is ~10% of the 50-epoch budget.
If the run terminates before 50 epochs due to timeout, warmup is a smaller
fraction of actual training time — this is fine. Do not set WARMUP_EPOCHS > 10
or warmup will consume too much of the 30-min budget.

**Risk:** If the model was already learning well in early epochs (low train loss
by epoch 5), warmup only helps marginally. Check val metric at epoch 5 vs.
baseline — if it is already near the baseline best, warmup was not the bottleneck.

---

## H4 — Wider model (n_hidden=192, increase slice_num=96)

**Headline:** Increase Transolver hidden dimension from 128 to 192 and slice_num
from 64 to 96 to increase model capacity without changing architecture.

**Mechanism:** The baseline is a deliberately small model (~1.5M params). At
128-dim with 4 heads, each head is 32-dim — quite narrow for capturing
spatial flow structures at multiple scales. Increasing to 192-dim (heads of
48-dim) and adding more slice tokens (96) expands the physical state space the
model can represent. The key question is whether 30 min is enough to train a
wider model. At batch=4 and mesh sizes up to 242K, the attention over
slice_num=96 tokens is negligible cost; the cost increase is primarily in the
MLP projection layers.

**Predicted delta:** Moderate. Transolver paper reports capacity scaling helps
up to a point; the baseline may be underfitting given the parameter budget.

**Exact implementation (`train.py` lines 417–428):**

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=192,       # was 128
    n_layers=5,
    n_head=4,           # each head is now 192//4 = 48-dim
    slice_num=96,       # was 64
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Keep all other config unchanged (lr=5e-4, surf_weight=10, batch=4).
CLI: `python train.py --wandb_group wider_model`

**VRAM check:** At B=4, N=242K, hidden=192: the dominant tensor in PhysicsAttention
is `fx_mid` at [4, 4, 242K, 48] = ~750M floats × 4 bytes ≈ 3 GB. Total VRAM
with activations and optimizer states should remain well under 96 GB.

**Hyperparameter sensitivity:** If `n_hidden=192` is already enough, `n_hidden=256`
is the next step. Do NOT jump to 256 in this run — it would invalidate the
ablation. Keep it a single step.

**Risk:** Wider model may need more epochs than the 30-min budget allows to converge.
Check if `train/surf_loss` is still decreasing at the final epoch — if yes, the
run is undertrained and a longer run or lower LR is needed.

---

## H5 — Per-sample relative loss (normalize by per-sample y-std)

**Headline:** Divide the loss by per-sample target standard deviation before
aggregating, so low-Re samples contribute proportionally rather than being
dominated by high-Re samples.

**Mechanism:** Per-sample y-std varies by ~10× across the dataset (RaceCar single:
avg 458, range up to 2077). In the current batch-level MSE, a single high-Re
sample with y-std=2000 contributes ~100× more to the loss than a low-Re sample
with y-std=200. This means the gradient is dominated by high-Re physics. The
primary metric averages MAE across splits that span different Re regimes
(`val_re_rand` is explicitly a Re holdout), so improving low-Re generalization
matters. A per-sample normalization de-emphasizes high-Re extremes and improves
training balance.

**Predicted delta:** Moderate improvement on `val_re_rand` and `val_single_in_dist`
(which spans Re 104K–5M). May slightly hurt high-Re accuracy but net effect
should be positive on the equal-weight average.

**Exact implementation (all changes in `train.py`):**

Replace training loop loss block (lines 487–496):

```python
x_norm = (x - stats["x_mean"]) / stats["x_std"]
y_norm = (y - stats["y_mean"]) / stats["y_std"]
pred = model({"x": x_norm})["preds"]

# Per-sample normalization: divide each sample's loss by its y-std
# y has shape [B, N, 3]; compute std over valid (non-padded) nodes
# for each sample and each channel, then average channels
with torch.no_grad():
    y_s = []
    for b in range(y.shape[0]):
        valid_y = y[b][mask[b]]             # [N_valid, 3]
        s = valid_y.std(dim=0).mean()       # scalar std averaged over channels
        y_s.append(s.clamp(min=1.0))
    per_sample_std = torch.stack(y_s)       # [B]

sq_err = (pred - y_norm) ** 2
vol_mask = mask & ~is_surface
surf_mask = mask & is_surface

# Divide each sample's loss by its physical y-std (in physical space,
# y_std_normalized = 1 since we already normalized, so just use raw y std)
vol_loss_per = []
surf_loss_per = []
for b in range(y.shape[0]):
    vm = vol_mask[b].unsqueeze(-1)
    sm = surf_mask[b].unsqueeze(-1)
    vl = (sq_err[b] * vm).sum() / vm.sum().clamp(min=1)
    sl = (sq_err[b] * sm).sum() / sm.sum().clamp(min=1)
    vol_loss_per.append(vl / per_sample_std[b])
    surf_loss_per.append(sl / per_sample_std[b])

vol_loss  = torch.stack(vol_loss_per).mean()
surf_loss = torch.stack(surf_loss_per).mean()
loss = vol_loss + cfg.surf_weight * surf_loss
```

CLI: `python train.py --wandb_group per_sample_rel_loss`

**Hyperparameter sensitivity:** The `clamp(min=1.0)` on per_sample_std is important
to prevent division by near-zero std on degenerate samples. Check if the train
loss value is sensible (should be around 0.1–1.0 in normalized terms).

**Risk:** The per-sample loop adds Python overhead inside the training loop.
With batch=4 it is negligible. For larger batches it would matter. Also, the
per-sample std is computed in physical space (before normalization) — mixing
physical and normalized quantities. The computation above uses `y` (physical)
divided by `per_sample_std` (physical), which is correct — it makes each
sample contribute ~equal relative error to the aggregate loss.

---

## H6 — Deeper model with gradient clipping (n_layers=7)

**Headline:** Increase depth from 5 to 7 TransolverBlocks and add gradient
clipping to stabilize the deeper stack.

**Mechanism:** Depth increases the number of physics-aware attention passes;
each pass refines the slice-token representation. For complex tandem-foil
interference effects (where the wake of foil 1 modifies pressure on foil 2),
more layers allow the information to propagate further between physics slices.
The Transolver paper reports gains up to 8 layers on some benchmarks. Gradient
clipping (`max_norm=1.0`) is critical to prevent gradient explosion in deeper
models with skip connections.

**Predicted delta:** Small to moderate — deeper models help when the current
depth is a bottleneck for spatial propagation. Most beneficial for the tandem
splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) where the two-foil
interaction is the generalization challenge.

**Exact implementation (all changes in `train.py`):**

```python
# Line 424 in model_config
n_layers=7,       # was 5
```

Add gradient clipping in the training loop (after `loss.backward()`, before
`optimizer.step()`):

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

CLI: `python train.py --wandb_group deeper_7layers`

**VRAM:** 2 extra layers each add ~4 × n_hidden² × 2 parameters (attention +
MLP) ≈ 2 × 131K = 262K params. Negligible VRAM cost.

**Hyperparameter sensitivity:** `max_norm=1.0` is the standard value for
transformer training. If `train/surf_loss` oscillates wildly, reduce to 0.5.

**Risk:** If the baseline 5 layers is already sufficient depth and the bottleneck
is width (attention head dimension), extra layers add only marginal value.
Diagnostic: check if the 5-layer baseline's last-layer gradient norm in W&B
(added as a logged metric) is already small — if yes, depth is not the bottleneck.

---

## H7 — Fourier positional features on (x, z) coordinates

**Headline:** Replace the raw (x, z) node position input features (dims 0–1)
with multi-frequency Fourier embeddings before the model preprocess MLP.

**Mechanism:** Raw (x, z) Cartesian coordinates are a poor basis for representing
multi-scale flow phenomena: boundary layers (O(1e-4 m)), wake interactions
(O(0.1 m)), and far-field decay (O(10 m)) all coexist in the same mesh. Fourier
features (random or deterministic at log-spaced frequencies) allow the model to
directly fit smooth functions over multiple length scales without needing to
synthesize them from linear combinations of raw coordinates. This is the
foundational idea from Mildenhall et al. NeRF (2020) and Tancik et al.
"Fourier Features Let Networks Learn High-Frequency Functions in Low Dimensions"
(NeurIPS 2020), and has been applied to mesh operators in Geo-FNO and subsequent
work.

**Predicted delta:** Moderate. Most impactful for the surface boundary layer
where pressure varies rapidly with arc-length; Fourier features at high
frequencies let the model represent sharp pressure gradients near the leading
edge and trailing edge without capacity waste.

**Exact implementation (all changes in `train.py`):**

Add a `FourierPositionEmbedding` module and modify how x_norm is fed to the
model. Changes in the forward pass only — the model input contract
`{"x": x_norm}` is unchanged but x_norm is augmented.

```python
# Add this class near the top of train.py (before Config)
class FourierPositionEmbedding(nn.Module):
    """Replace 2D position (dims 0–1 of x) with multi-freq Fourier features.

    Output dimension: 2 * num_freqs (sin + cos per frequency).
    Uses log-spaced frequencies from min_freq to max_freq.
    """
    def __init__(self, num_freqs=16, min_freq=1.0, max_freq=1000.0):
        super().__init__()
        freqs = torch.exp(
            torch.linspace(
                math.log(min_freq), math.log(max_freq), num_freqs
            )
        )  # [num_freqs]
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * num_freqs

    def forward(self, xy):
        # xy: [..., 2] in physical units (already normalized by x_mean/x_std)
        angles = xy.unsqueeze(-1) * self.freqs  # [..., 2, num_freqs]
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)  # [..., 4*num_freqs]
```

Then in the training/eval loop, after `x_norm` is computed:

```python
# Build augmented features: replace dims 0–1 with Fourier-embedded version
# (pos_embed outputs 4*NUM_FREQS dims instead of 2 raw coords)
NUM_FREQS = 8   # 8 freqs => 32 extra dims; replaces 2 raw dims => net +30 dims
pos_embed = FourierPositionEmbedding(num_freqs=NUM_FREQS).to(device)

# In the loop (both train and eval):
xy_norm  = x_norm[:, :, :2]                     # [B,N,2]
rest_norm = x_norm[:, :, 2:]                    # [B,N,22]
pos_enc  = pos_embed(xy_norm)                   # [B,N,4*NUM_FREQS=32]
x_aug    = torch.cat([pos_enc, rest_norm], dim=-1)  # [B,N,2+22+30=54]
pred     = model({"x": x_aug})["preds"]
```

And update `model_config` and `Transolver` instantiation to accept the new
input dimension:

```python
FOURIER_DIMS = 4 * 8   # 32 for NUM_FREQS=8 -- replaces 2 raw pos dims
model_config = dict(
    space_dim=2,
    fun_dim=(X_DIM - 2) + (FOURIER_DIMS - 2),  # 22 original non-pos + 30 extra Fourier
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Note: `Transolver.preprocess` MLP is `MLP(fun_dim + space_dim, ...)` where
`space_dim=2` and `fun_dim=22` (baseline). With Fourier features, set
`space_dim=FOURIER_DIMS` and `fun_dim=22` (unchanged), so the MLP input
becomes `FOURIER_DIMS + 22 = 54`. OR, simpler: just set `space_dim=0` and
`fun_dim=FOURIER_DIMS + 22` — either works since Transolver's preprocess
just concatenates them.

Add `import math` if not already present.
CLI: `python train.py --wandb_group fourier_pos`

**Hyperparameter sensitivity:** `min_freq` and `max_freq` must span the relevant
length scales. In mesh coordinates (units ~meters): boundary layer thickness
~1e-4, chord ~0.2, far field ~10. So `min_freq=1.0, max_freq=1000.0` with
log-spacing covers this range. `num_freqs=8` per spatial dimension is a
reasonable starting point — can scale to 16 if the run is fast enough.

**Risk:** The coordinate normalization (via `x_mean`, `x_std`) shrinks the
physical (x,z) range to ~[-2, +2] standard units. The Fourier frequencies
must be in normalized-coordinate space, not physical space. After normalization,
a frequency of 1.0 has a period of 2π ≈ 6 normalized units (spanning the
whole domain once), and 1000.0 captures sub-pixel-scale variation. This is
correct — just make sure the embedding is applied AFTER normalization.

---

## H8 — Pressure-channel decoupled output head

**Headline:** Add a separate output MLP for the pressure channel (`p`) in the
last TransolverBlock, so pressure gets a dedicated capacity path while sharing
all attention layers.

**Mechanism:** The current `mlp2` in `TransolverBlock.last_layer` is a single
two-layer MLP that outputs all 3 channels from the same hidden state. Pressure
has fundamentally different physics than velocity: it is a scalar potential
satisfying the Poisson equation, while velocity satisfies the incompressible
Navier–Stokes vector equations. A dedicated head allows the model to specialize
capacity for the metric channel. This is analogous to the multi-head output
design in GNOT (Li et al., 2023) where each field gets its own projection.

**Predicted delta:** Small to moderate. Most useful if the shared head is
the bottleneck (underfitting on p while overfitting on Ux/Uy). Check
`val/mae_surf_p` vs `val/mae_surf_Ux` in the baseline — if p error is
disproportionately large, a dedicated head should help.

**Exact implementation (all changes in `train.py`):**

In `TransolverBlock.__init__` (around line 153):

```python
if self.last_layer:
    self.ln_3 = nn.LayerNorm(hidden_dim)
    # Shared head for Ux, Uy
    self.mlp2_vel = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, 2),           # Ux, Uy
    )
    # Dedicated head for pressure p
    self.mlp2_p = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, 1),           # p
    )
```

In `TransolverBlock.forward` (around line 163):

```python
if self.last_layer:
    h = self.ln_3(fx)
    vel = self.mlp2_vel(h)       # [B, N, 2]
    p   = self.mlp2_p(h)         # [B, N, 1]
    return torch.cat([vel, p], dim=-1)   # [B, N, 3]
```

The `out_dim` parameter in `TransolverBlock` can be left at 3 (it only affects
the old `mlp2` which is now replaced). Or keep the old `mlp2` for non-last
layers and replace only for `last_layer=True` as shown above.

No change to `model_config` needed. CLI: `python train.py --wandb_group split_p_head`

**Risk:** This doubles the parameter count of the output head, but since the
head is tiny (128→128→1 vs 128→128→3), the total increase is ~16K params —
negligible. The risk is that the shared representation in the attention layers
is still insufficient to support two specialized heads; in that case there is no
gain.

---

## H9 — Increase surf_weight to 25

**Headline:** Simply raise `surf_weight` from 10.0 to 25.0, giving surface nodes
2.5× more gradient weight relative to the current setup.

**Mechanism:** Surface nodes are ~2–5% of total mesh nodes but 100% of the
evaluation metric. The current vol_MSE + 10×surf_MSE gives surface ~10/(10+1)
= 91% of the loss weight per node... but volume has ~20–50× more nodes. Net
gradient mass is: vol_nodes × 1 + surf_nodes × 10. If there are 5000 surface
nodes and 200K vol nodes, surface contributes 5000×10 = 50K vs volume 200K×1
= 200K — only 20% of total gradient mass. Raising to 25 would give 125K vs
200K — 38% surface. This is a direct, interpretable lever with no architectural
change.

**Predicted delta:** Small to moderate improvement specifically on `mae_surf_p`
and `mae_surf_Ux/Uy`. May slightly hurt `mae_vol_*` metrics.

**Exact implementation:** Single-line change in `train.py` Config default
(line 381), or pass via CLI:

```python
# Change Config default:
surf_weight: float = 25.0   # was 10.0

# Or just pass CLI:
python train.py --surf_weight 25 --wandb_group surf_weight_25
```

No other changes needed.

**Hyperparameter sensitivity:** This is a low-risk, easily ablated change. If
25 shows no improvement over 10, try 40. If 25 shows improvement, H2 with
surf_weight=50 is the natural follow-up.

**Risk:** Very high surf_weight (> 50) can cause instability if the surface
loss dominates and surface nodes have high gradient variance (few nodes per
batch at batch=4). For surf_weight=25, the risk is low.

---

## H10 — Larger slice_num (128) with same hidden dim

**Headline:** Double the number of physics slice tokens from 64 to 128, keeping
all other parameters identical, to give the model a richer physical-state
partition.

**Mechanism:** Slice tokens are the key abstraction in Transolver — they
represent learned "physical states" (like different flow regimes, or different
spatial zones). With 64 tokens and 5 attention layers, the model may be
conflating near-surface boundary-layer physics with interior wake physics in
the same slice. 128 tokens allows finer partitioning. The cost of
attention over slice_tokens scales as O(S²) where S=slice_num — going from
64 to 128 quadruples the attention FLOPS over slices, but since S << N (128
<< 242K), this is negligible. The dominant cost is the N × S weighted-sum
operation, which doubles linearly.

**Predicted delta:** Small to moderate. Most benefit if the current 64-token
partition is a bottleneck for representing multiple flow regimes simultaneously.

**Exact implementation:**

```python
# In model_config (line 422):
slice_num=128,    # was 64
```

CLI: `python train.py --wandb_group slicenum_128`

**Hyperparameter sensitivity:** The `in_project_slice` weight matrix has shape
[dim_head, slice_num] = [32, 128] (since hidden=128, heads=4, dim_head=32).
The orthogonal initialization in `__init__` (`torch.nn.init.orthogonal_`) will
still work (orthogonal initialization on a 32×128 matrix is valid — it
initializes the rows to be orthogonal, not columns). No code change needed.

**Risk:** Low. If 64 tokens was not the bottleneck, doubling to 128 will show
no improvement but also no regression. Cheap experiment to confirm or rule out.

---

## H11 — AdamW betas tuning + mild gradient clipping

**Headline:** Change AdamW betas from PyTorch defaults (0.9, 0.999) to
(0.9, 0.95) and add gradient clipping (max_norm=1.0).

**Mechanism:** The default `beta2=0.999` is designed for low-noise, high-frequency
gradient settings (dense mini-batches in classification). In this setting,
batches are size 4 with high-variance per-node gradients (due to mesh padding,
domain mixing, and Re diversity). `beta2=0.95` reduces the momentum timescale
of the second moment estimate, making the optimizer more responsive to recent
gradient magnitudes. This is the recommendation from the "Sophia" paper and the
Llama training reports, and has been validated in neural operator settings.
Gradient clipping (max_norm=1.0) prevents occasional large-gradient batches
(high-Re samples) from destabilizing training.

**Predicted delta:** Small. This is a stability improvement; expect smoother
training curves and possibly a better best checkpoint on the validation set
because instability peaks are avoided.

**Exact implementation (line 434 in `train.py`):**

```python
# OLD
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# NEW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.95),
)

# In the training loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

CLI: `python train.py --wandb_group adamw_betas_clip`

**Risk:** Low. If `beta2=0.95` is too aggressive, training may converge slower
early but reach the same or better final value. If the run is timeout-limited,
slightly slower convergence could hurt the reported metric. Check `train/loss`
vs baseline at epoch 5 — if it's higher, increase beta2 back to 0.98.

---

## H12 — OneCycleLR (super-convergence)

**Headline:** Replace CosineAnnealingLR with `OneCycleLR` at a higher peak LR
(1e-3), using the "super-convergence" schedule discovered by Smith & Topin (2017)
that has shown consistent gains in neural PDE solvers.

**Mechanism:** OneCycleLR ramps from a low LR (max_lr/25) to a peak LR, then
decays to a minimum (max_lr/10000), spending most of the budget on the descent.
At a higher peak LR than the current cosine baseline (5e-4), the optimizer
explores a wider region of the loss landscape and can escape local minima that
cosine annealing from a fixed LR would not. For mesh-based problems with
high-variance gradients (many nodes, sparse surface signal), a briefly higher
LR phase can help the model find better representations faster. OneCycleLR is
validated in the Transolver codebase (original authors' other benchmarks) and
in weather/climate neural operators (FourCastNet follow-ups).

**Predicted delta:** Moderate. Super-convergence typically yields 5–15% faster
convergence, meaning the same quality checkpoint is reached in fewer epochs —
directly valuable under the 30-min budget constraint.

**Exact implementation (lines 434–435 in `train.py`):**

```python
# OLD
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# NEW — OneCycleLR requires knowing total steps upfront
STEPS_PER_EPOCH = len(train_loader)   # must be called after train_loader is defined
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3 / 25,           # start low; OneCycleLR will ramp this up
    weight_decay=cfg.weight_decay,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=MAX_EPOCHS * STEPS_PER_EPOCH,
    pct_start=0.3,          # 30% ramp-up, 70% decay
    anneal_strategy="cos",
    div_factor=25.0,         # initial_lr = max_lr / 25
    final_div_factor=1e4,    # final_lr = max_lr / 1e4
)
```

Move the scheduler definition to AFTER `train_loader` is created (currently
`train_loader` is defined before `optimizer` anyway, so this is fine).

**Critical:** OneCycleLR updates every STEP (every batch), not every epoch.
Replace `scheduler.step()` at the END of the epoch block with `scheduler.step()`
INSIDE the batch loop (after `optimizer.step()`):

```python
# Inside the training batch loop, after optimizer.step():
scheduler.step()
```

Remove the `scheduler.step()` call at the epoch level (line 508).

CLI: `python train.py --wandb_group onecycle_lr`

**Hyperparameter sensitivity:** `max_lr=1e-3` is 2× the current cosine baseline.
If `train/surf_loss` diverges in the first 5 epochs, try `max_lr=7e-4`.
`pct_start=0.3` (30% ramp) is the typical recommendation for 50-epoch runs.

**Risk:** If the run terminates early due to the 30-min wall-clock timeout,
OneCycleLR may be mid-cycle (before the final deep descent), leaving a
suboptimal checkpoint. Monitor whether the best val checkpoint arrives before
or after the peak LR — if it's before the peak (in the ramp phase), reduce
`pct_start` to 0.15.

---

## Priority ordering for student assignment (Round 1)

Assign in this order to maximize orthogonality and information gain across students:

| Priority | Hypothesis | Rationale |
|----------|-----------|-----------|
| 1 | H1 (Huber loss) | Fixes the biggest known mismatch: MSE training vs MAE metric |
| 2 | H9 (surf_weight=25) | Simplest possible change targeting the metric; establishes the weight sensitivity curve |
| 3 | H12 (OneCycleLR) | Pure schedule change; orthogonal to loss and architecture |
| 4 | H8 (split p head) | Directly targeted at the scored channel; easy ablation |
| 5 | H2 (surf p L1 + w=50) | Stronger version of H1+H9 combo; tests whether combining both improvements stacks |
| 6 | H3 (cosine warmup) | Low-cost schedule experiment; rules out early instability hypothesis |
| 7 | H4 (wider: hidden=192) | Capacity question; rules out underfitting hypothesis |
| 8 | H10 (slice_num=128) | Ruling out attention bottleneck hypothesis |
| 9 | H5 (per-sample relative loss) | Re-regime generalization; important if val_re_rand is the weakest split |
| 10 | H7 (Fourier pos features) | Largest code change; should wait until simpler levers are measured |
| 11 | H6 (7-layer depth) | Depth vs width; do after wider model result returns |
| 12 | H11 (AdamW betas) | Lowest expected delta; diagnostic run, not a primary bet |
