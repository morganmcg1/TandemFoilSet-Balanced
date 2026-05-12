<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-12 (Willow pai2g-48h-r3, round 1)

Generated after a literature survey for Transolver follow-ups, masked attention,
loss/metric alignment, positional encodings, and surface-anchored cross-attention.

Scope note: this launch (`icml-appendix-willow-pai2g-48h-r3`) is research-isolated.
Cross-launch results and prior compound baselines are out of scope; the only
reference for round 1 is the unmodified `train.py` on this branch (Transolver
n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; AdamW lr=5e-4,
wd=1e-4, surf_weight=10, MSE in normalized space, cosine schedule). Any
"already tried" or "ruled out" notes below refer to general literature precedent,
not to this branch's empirical history.

Primary target metric: `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p`
(paper-facing). All experiments are designed for a 30-minute wall-clock budget
on a single 96 GB GPU.

---

## Ranked Hypotheses

---

### 1. Mask-Aware PhysicsAttention (Fix Padding Noise in Slice Formation)

**Hypothesis.** PhysicsAttention's soft slice-assignment (the temperature-scaled softmax
over node-to-slice similarity) currently operates on ALL batch positions, including
zero-padded dummy nodes. Zero-padded positions contribute real weight to slice tokens,
biasing every slice for every sample in a variable-length batch. Masking out padded nodes
before the softmax normalization should reduce this noise and sharpen slice semantics,
improving the quality of the slice-token representations without any additional compute.

**Mechanistic target.** Padding noise in slice formation --> corrupted slice tokens -->
degraded attention output for all real nodes. The falsifying result: no change in
val_avg/mae_surf_p after masking, meaning slice bias from padding is negligible.

**Implementation sketch.**

In `train.py`, locate `PhysicsAttention.forward` (~line 100). The soft assignment is:

```python
# current (no mask):
attn = torch.softmax(fx_tmp / self.temp, dim=1)   # [B, N, slice_num]
```

Change to set padding positions to -inf before softmax:

```python
# proposed:
if mask is not None:
    # mask: [B, N], True = real node
    fx_tmp = fx_tmp.masked_fill(~mask.unsqueeze(-1), float('-inf'))
attn = torch.softmax(fx_tmp / self.temp, dim=1)   # [B, N, slice_num]
```

Propagate `mask` down through `TransolverBlock.forward` and `Transolver.forward`.
`Transolver.forward` already receives `x` which carries the mask; add `mask` as an
explicit argument and pass it through each `TransolverBlock` and into each
`PhysicsAttention` call.

The scatter-back step (weighted sum from slice tokens to nodes) is fine as-is; the fix
only touches the forward (nodes-to-slices) softmax direction.

**Risk.** Low. This is a correctness fix, not a speculation. The only question is
magnitude of improvement; if batch sizes are small (batch_size=4) and the largest sample
dominates, the bias may be modest. No new hyperparameters.

**Time impact.** Near-zero additional compute (one masked_fill per attention layer).
Fits comfortably in 30 minutes.

**Taste scores (diagnostic, mechanistic grounding=4, research-state value=4, execution=4).**

---

### 2. Per-Sample Re-Normalization of Training Loss

**Hypothesis.** The current MSE loss in normalized space uses global `y_std` for
normalization. Within a single training batch, high-Re samples have per-sample y-std
roughly 10x larger than low-Re samples. In normalized space, a 10x y-std amplification
still leaves high-Re prediction errors dominating the gradient signal. Dividing each
sample's squared error by its own per-sample y-std (computed from the real-node ground
truth) would equalize the gradient contribution across the Re spectrum and force the model
to learn the low-Re regime as carefully as high-Re.

**Mechanistic target.** Re-imbalanced gradients --> undertrained low-Re predictions -->
val_re_rand and val_geom_camber_cruise OOD degradation. Falsifying result: per-sample
normalization hurts high-Re accuracy faster than it helps low-Re.

**Implementation sketch.**

In the training loop, after computing `y_norm` and `pred`:

```python
# Per-sample std over real nodes, shape [B, 1, 3]
y_phys = y  # [B, N, 3] in physical space, already available
per_sample_std = torch.zeros(y_phys.shape[0], 1, 3, device=y.device)
for b in range(y_phys.shape[0]):
    real = y_phys[b, mask[b]]          # [n_real, 3]
    per_sample_std[b, 0] = real.std(dim=0).clamp(min=1.0)  # avoid div-by-zero

sq_err = (pred - y_norm) ** 2
# Weight each sample inversely by its own per-sample std (normalized)
global_std = stats["y_std"].to(y.device)                  # [3]
rel_std = per_sample_std / global_std.unsqueeze(0)         # [B, 1, 3]
sq_err_weighted = sq_err / rel_std                         # broadcast over N

vol_loss = (sq_err_weighted * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err_weighted * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add a new `Config` field `per_sample_norm: bool = True` to control this.

**Risk.** Medium. Per-sample normalization changes the effective loss landscape globally.
If low-Re samples are easy, this will under-weight them in the right way; if they're hard
in different ways, weighting them up may hurt. The interaction with the existing global
normalization needs care -- do not double-normalize.

**Time impact.** One extra std() call per batch, negligible overhead. 30-minute budget fine.

**Taste scores (frontier refinement, mechanistic grounding=3, research-state value=3, execution=3).**

---

### 3. Surface-Anchored Cross-Attention (AB-UPT-Style)

**Hypothesis.** AB-UPT (Anchored-Branched Universal Physics Transformers, TMLR 2025)
demonstrated that treating surface/boundary nodes as "anchor" tokens and using them as
queries in cross-attention against volume tokens produces significantly better surface
pressure predictions than treating all nodes uniformly. For TandemFoilSet, where
`mae_surf_p` is the primary metric, anchoring on surface nodes should directly improve
the metric by giving the model a dedicated pathway for surface-to-volume communication,
rather than relying on surface nodes competing for slice token representation against
~200K volume nodes.

**Mechanistic target.** Surface nodes are vastly outnumbered by volume nodes in slice
token formation; AB-UPT's anchored branch gives them a dedicated representation.
Falsifying result: cross-attention cost makes training too slow for 30 minutes, or
surface MAE degrades due to decoupling from volume context.

**Implementation sketch.**

Add a `SurfaceCrossAttentionBlock` after the Transolver backbone:

```python
class SurfaceCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))

    def forward(self, surf_feats, vol_feats, surf_mask, vol_mask):
        # surf_feats: [B, N_surf, D], vol_feats: [B, N_vol, D]
        # cross-attend: surface queries, volume keys/values
        key_padding_mask = ~vol_mask  # True = ignore
        out, _ = self.cross_attn(surf_feats, vol_feats, vol_feats,
                                  key_padding_mask=key_padding_mask)
        surf_feats = self.norm1(surf_feats + out)
        surf_feats = self.norm2(surf_feats + self.ffn(surf_feats))
        return surf_feats
```

In `Transolver.forward`, after the final TransolverBlock output `z` [B, N, D]:
- Extract surface subset: `surf_idx`, `vol_idx` via `is_surface` mask
- Run `SurfaceCrossAttentionBlock` on those subsets
- Merge back: write updated surface features back into `z` at surf positions
- Then apply output projector

To avoid variable-length gather cost, use fixed-size padded extraction with a secondary
surface mask. Keep `n_head=4`, `d_model=128` to match existing hidden dim.

Add `Config` field `surf_cross_attn: bool = True`.

**Risk.** Medium-high. Variable-length surface/volume extraction inside the batch is
fiddly; padding the inner cross-attention adds a second padded dimension. If cruise
samples have very few surface nodes relative to volume (they have ~210K nodes total),
the benefit may be diluted. Consider limiting to 1 cross-attention layer initially.

**Time impact.** Cross-attention over surface nodes only (~1-3K surface nodes vs 74-242K
total); much cheaper than full mesh attention. Should fit in 30 minutes at batch_size=2.

**Taste scores (tier shift, mechanistic grounding=3, research-state value=3, execution=2).**

---

### 4. Multiplicative Re Conditioning (MFN-Style)

**Hypothesis.** FiLM conditioning (dead-ended in charliepai2f) only applies an affine
transformation (gamma*h + beta) per layer using Re as conditioning signal. The Caltech
MFN (Multiplicative Filter Network) approach instead computes elementwise products
between Re-derived features and hidden state activations -- a fundamentally richer
interaction than additive/affine shift. The dead-ending of FiLM may reflect the
inadequacy of the affine parameterization rather than Re conditioning being unhelpful per
se. MFN-style conditioning adds a learnable frequency-domain interaction that affine FiLM
cannot express.

**Mechanistic target.** FiLM dead-end was affine parameterization weakness, not Re
conditioning futility. Falsifying result: MFN conditioning also fails to improve over
unconditioned baseline, ruling out Re-conditioning as a lever entirely.

**Implementation sketch.**

Replace FiLM gamma/beta with a multiplicative gate:

```python
class MFNReConditioner(nn.Module):
    def __init__(self, re_dim, hidden_dim, n_freqs=8):
        super().__init__()
        # Map log(Re) to Fourier features, then to hidden_dim gate
        self.freq_proj = nn.Linear(1, n_freqs)
        self.gate_proj = nn.Linear(n_freqs * 2, hidden_dim)  # sin + cos
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)  # init as identity (gate=1)

    def forward(self, h, log_re):
        # h: [B, N, D], log_re: [B] scalar per sample
        log_re = log_re[:, None]  # [B, 1]
        freqs = self.freq_proj(log_re)  # [B, n_freqs]
        fourier = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [B, 2*n_freqs]
        gate = self.gate_proj(fourier).unsqueeze(1)  # [B, 1, D]
        return h * gate  # elementwise multiplicative modulation
```

Extract `log_re` from `x[:, :, 13]` (dim 13 = log(Re), constant per sample) inside
`Transolver.forward`. Apply `MFNReConditioner` after each TransolverBlock's output,
with a single shared conditioner instance (not per-layer) to keep param count low.

Add `Config` field `mfn_re_cond: bool = True`.

**Risk.** Medium. The FiLM dead-end is real evidence against this direction. However,
the MFN gate mechanism is qualitatively different and the init-to-identity trick means
it can start as a no-op. The key question is whether Re is truly informative beyond what
the normalized features already provide.

**Time impact.** Tiny overhead (one multiply per layer per sample). 30-minute budget fine.

**Taste scores (frontier refinement, mechanistic grounding=3, research-state value=3, execution=3).**

---

### 5. Non-NACA Specials Disambiguation Feature

**Hypothesis.** File 3 (raceCar tandem P3) encodes 150 non-NACA special foils
(CH10, E423, FX74, LA5055, S1210) as NACA (0,0,0). This creates a collision: single-foil
samples that are genuinely tandem-free also have (0,0,0) for NACA foil 2 features
(dims 19-21). The model cannot distinguish "this is a non-NACA special foil" from "this
is a single-foil sample" from dim 15-17 alone. Adding a single boolean `is_special_foil`
feature should resolve this ambiguity and help the model attend to the correct geometry
branch for File 3 samples.

**Mechanistic target.** Feature aliasing between non-NACA specials and no-foil-2 slots
causes systematic error on File 3 samples. Falsifying result: val_single_in_dist and
val_re_rand both show no change, meaning the model already learned to resolve the
ambiguity from other cues (e.g., gap/stagger dims 22-23, which are 0 for single-foil).

**Implementation sketch.**

In `train.py`, add a feature augmentation step after loading the batch (do NOT modify
`data/`). The flag is 1 if dims 15-17 are all exactly 0 AND dims 22-23 are nonzero
(tandem sample with non-NACA foil), else 0:

```python
def add_special_foil_flag(x):
    # x: [B, N, 24]
    naca_zero = (x[..., 15:18].abs().sum(-1) < 1e-6)   # [B, N] NACA (0,0,0)
    is_tandem = (x[..., 22:24].abs().sum(-1) > 1e-6)    # [B, N] gap or stagger nonzero
    flag = (naca_zero & is_tandem).float().unsqueeze(-1) # [B, N, 1]
    return torch.cat([x, flag], dim=-1)                  # [B, N, 25]
```

Update `model_config` to `fun_dim=23` (space_dim=2 + fun_dim=22 becomes space_dim=2 +
fun_dim=23, total=25). Update Transolver input linear accordingly. Also update
`x_norm` stats: the new dim 24 is binary so no normalization needed; append
`stats["x_mean"]` and `stats["x_std"]` with 0.0 and 1.0 respectively.

Add `Config` field `special_foil_flag: bool = True`.

**Risk.** Low-medium. dims 22-23 (gap, stagger) already encode tandem vs single implicitly.
The model may already be resolving the ambiguity. If it is, this is a no-op with small
positive side-effect of cleaner features. The main risk is off-by-one in normalization
if stats slicing is done carelessly.

**Time impact.** Negligible. 30-minute budget fine.

**Taste scores (diagnostic, mechanistic grounding=3, research-state value=3, execution=4).**

---

### 6. Temperature Annealing for PhysicsAttention Softmax

**Hypothesis.** The slice-assignment softmax in PhysicsAttention uses a fixed temperature
`self.temp = nn.Parameter(torch.ones(1) * 0.5)` (learned scalar). Early in training,
high temperature produces diffuse, uniform slice assignments that give the optimizer a
dense gradient signal everywhere; later, lower temperature sharpens slice boundaries.
Initializing with high temperature (e.g., 2.0) and annealing it toward 0.1 over training
mirrors the technique used in vector quantization / Gumbel-softmax annealing and should
improve the quality of learned slice semantics compared to a single learned scalar.

**Mechanistic target.** Premature slice crystallization in early training traps the model
in a suboptimal partitioning. Falsifying result: no change in val metrics, meaning slice
temperature is already adaptive enough via learned scalar.

**Implementation sketch.**

Replace `self.temp = nn.Parameter(torch.ones(1) * 0.5)` with a non-learnable buffer
that is updated by the trainer:

```python
# In PhysicsAttention.__init__:
self.register_buffer("temp", torch.tensor(2.0))  # start hot

# In Transolver class, add method:
def set_slice_temperature(self, temp_val):
    for m in self.modules():
        if isinstance(m, PhysicsAttention):
            m.temp.fill_(temp_val)
```

In the training loop, add a cosine anneal from T_start=2.0 to T_end=0.1 over epochs:

```python
T_start, T_end = 2.0, 0.1
temp_val = T_end + 0.5 * (T_start - T_end) * (1 + math.cos(math.pi * epoch / cfg.epochs))
model.set_slice_temperature(temp_val)
```

Add `Config` fields `temp_anneal: bool = True`, `temp_start: float = 2.0`,
`temp_end: float = 0.1`.

**Risk.** Low-medium. The existing learned scalar temperature already adapts; the question
is whether schedule matters more than magnitude. The annealing schedule is standard and
robust. Risk is primarily that the existing temp scalar converges to a similarly good
value anyway.

**Time impact.** One `fill_` call per epoch. Negligible. 30-minute budget fine.

**Taste scores (diagnostic, mechanistic grounding=3, research-state value=3, execution=4).**

---

### 7. OOD Geometry Curriculum (Progressive Camber Upweighting)

**Hypothesis.** The hardest val splits are `val_geom_camber_rc` and
`val_geom_camber_cruise` (fully OOD front-foil camber). Training sees adjacent cambers
(M=2-5 and M=9 for raceCar; M=0-2 and M=4-6 for cruise tandem). The camber sweep in
training is not uniform; certain camber values are structurally closer to the OOD range.
A curriculum that starts with balanced-domain sampling and progressively up-weights the
"boundary camber" samples closest to the held-out range should improve camber
interpolation generalization, inspired by NOCL (NeurIPS 2025) which uses NTK theory to
guide curriculum order but here uses a simpler distance-to-holdout heuristic.

**Mechanistic target.** The model over-fits to camber regimes far from OOD boundary;
curriculum shifts training distribution toward the generalization frontier. Falsifying
result: val_geom_camber_rc and val_geom_camber_cruise show no improvement, meaning the
model already generalizes camber smoothly and the bottleneck is elsewhere.

**Implementation sketch.**

In `train.py`, after `load_data()`, compute camber distance for tandem samples:

```python
def camber_proximity_weight(dataset, holdout_cambers_rc=(6,7,8), holdout_cambers_cr=(2,3,4)):
    weights = []
    for x, y, is_surf in dataset:
        # Dim 15 = NACA camber M (normalized 0-1, raw M/10)
        # Dim 22 = gap (0 for single-foil)
        is_tandem = x[0, 22].abs() > 0.01
        if not is_tandem:
            weights.append(1.0)
            continue
        m_norm = x[0, 15].item()  # first real node's camber feature
        m_raw = m_norm * 10       # approx M digit
        # raceCar tandem: closest to holdout M=6-8 is M=5 (dist=1) and M=9 (dist=1)
        # cruise tandem:  closest to holdout M=2-4 is M=2 (dist=0) and M=4 (dist=0)
        dist = min(abs(m_raw - h) for h in list(holdout_cambers_rc) + list(holdout_cambers_cr))
        proximity = 1.0 / (1.0 + dist)  # 1.0 if adjacent, 0.5 if 2 away
        weights.append(proximity)
    return torch.tensor(weights, dtype=torch.float)
```

Add `Config` field `camber_curriculum: bool = True`, `camber_curr_start_epoch: int = 5`.
Before `camber_curr_start_epoch`, use the original `sample_weights`. After, blend:
`blended = (1 - alpha) * sample_weights + alpha * camber_weights` where alpha ramps
from 0 to 1 linearly from epoch 5 to epochs//2. Re-create `WeightedRandomSampler` at
each epoch boundary (or every N steps if epoch-level is too coarse).

**Risk.** Medium. The camber feature read from normalized `x[:,15]` assumes correct
stats application; verify against `stats.json`. Re-creating the sampler per epoch adds
loop overhead but is cheap. The proximity calculation is a heuristic -- may need tuning.

**Time impact.** Sampler recreation per epoch: ~1s per epoch. Negligible. 30-minute fine.

**Taste scores (frontier refinement, mechanistic grounding=2, research-state value=3, execution=3).**

---

### 8. Deeper Backbone with Fewer Slices (n_layers=7, slice_num=32)

**Hypothesis.** The current config (n_layers=5, slice_num=64) may be suboptimal in its
depth-breadth tradeoff. Slice self-attention cost scales as O(slice_num^2); reducing
slice_num from 64 to 32 frees enough compute to add 2 more TransolverBlocks within the
same VRAM and time budget. More layers allow the model to compose more complex physical
features; fewer slices may force more discriminative physical partitioning. This is
inspired by the Transolver++ paper's finding that local adaptive mechanisms (effectively
richer per-layer slice formation) outperform simply wider global attention.

**Mechanistic target.** Current depth (5 layers) is insufficient to compose multi-scale
physics features; width (64 slices) wastes capacity on redundant slice directions.
Falsifying result: deeper-narrower performs worse or equal to baseline config.

**Implementation sketch.**

Change `model_config` in `train.py`:

```python
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=7, n_head=4,
    slice_num=32, mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

No other changes required. Verify VRAM budget: n_layers=7 + slice_num=32 adds 2 layers
of self-attention on 32 tokens (trivial), and 2 extra MLP passes over N nodes (linear
in N). Total param increase is minimal (~300K extra). Estimated VRAM impact: negligible.

**Risk.** Low. This is a pure architecture ablation with no new code paths. The
interaction with torch.compile is benign. The risk is that depth helps less than width
in this problem or that the 32-slice constraint causes coarser physical partitioning.

**Time impact.** 7 layers vs 5: ~40% more forward pass cost per batch. May need to
reduce `batch_size` from 4 to 3 for largest cruise samples. 30-minute budget still fine.

**Taste scores (frontier refinement, mechanistic grounding=2, research-state value=2, execution=4).**

---

### 9. Pressure-Only Output Head with Dedicated Capacity

**Hypothesis.** The current model uses a single shared output projector predicting
[Ux, Uy, p] jointly from the same hidden representation. Surface pressure (the ranking
metric) may benefit from a dedicated prediction head with more capacity, while the
velocity components can share a lighter head. This is architecturally similar to
multi-task learning with asymmetric heads for tasks of different importance/difficulty.

**Mechanistic target.** The shared output head is a capacity bottleneck for pressure
prediction specifically. Falsifying result: a deeper pressure head adds no improvement,
meaning the bottleneck is in the encoder not the decoder.

**Implementation sketch.**

In `Transolver.forward`, replace the shared output projector with:

```python
# Existing: self.out = nn.Linear(n_hidden, out_dim)  # predicts [Ux, Uy, p]
# Replace with:
self.out_vel = nn.Linear(n_hidden, 2)             # Ux, Uy
self.out_p = nn.Sequential(
    nn.Linear(n_hidden, n_hidden // 2),
    nn.GELU(),
    nn.Linear(n_hidden // 2, 1)                   # p only
)
# In forward:
vel = self.out_vel(z)       # [B, N, 2]
pres = self.out_p(z)        # [B, N, 1]
preds = torch.cat([vel, pres], dim=-1)   # [B, N, 3]
```

The pressure MLP adds n_hidden * n_hidden/2 + n_hidden/2 = ~12K parameters (~9% of
total model). No changes to loss or metric computation needed.

Add `Config` field `dedicated_p_head: bool = True`.

**Risk.** Low. Minimal code change; fully backwards-compatible with existing scoring
contract. Risk is that the bottleneck is in encoder capacity, not decoder capacity, in
which case this is a no-op. Interaction with surf_weight loss: the pressure head still
receives the same surf_weight=10 loss gradient for surface nodes.

**Time impact.** Negligible overhead. 30-minute budget fine.

**Taste scores (frontier refinement, mechanistic grounding=2, research-state value=2, execution=4).**

---

### 10. AdamW Beta1/Beta2 Tuning (beta1=0.85, beta2=0.95)

**Hypothesis.** The default AdamW beta1=0.9, beta2=0.999 is calibrated for language
model training at large batch sizes with many parameters. For mesh-based physics
surrogates with small batch sizes (batch_size=4) and variable-length sequences, a lower
beta2 (0.95) provides faster adaptation of the second moment estimate and reduces the
"stale variance" problem for rare mesh configurations. A lower beta1 (0.85) increases
gradient noise tolerance. Together these may speed up convergence within the 30-minute
window. This is inspired by Karpathy's recommendations for transformer training and
confirmed by ablations in the Transolver++ paper.

**Mechanistic target.** Current beta2=0.999 causes slow variance estimate adaptation for
small-batch variable-mesh training. Falsifying result: metrics are unchanged or worse,
meaning variance estimate adaptation rate is not a bottleneck in this regime.

**Implementation sketch.**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.85, 0.95),  # changed from (0.9, 0.999)
    eps=1e-8,
)
```

Add `Config` fields `adam_beta1: float = 0.9`, `adam_beta2: float = 0.999`, then use
`betas=(cfg.adam_beta1, cfg.adam_beta2)`.

**Risk.** Low. One-line change. beta1=0.85 and beta2=0.95 are well-motivated for small
batches and have been used in many physics surrogate papers. The main risk is that lower
beta2 makes training noisier late in training, which the cosine LR schedule should
mitigate naturally.

**Time impact.** No overhead. 30-minute budget fine.

**Taste scores (diagnostic, mechanistic grounding=2, research-state value=2, execution=4).**

---

### 11. Spectral Slice Initialization (PCSM-Inspired)

**Hypothesis.** PCSM (arXiv 2410.11382) showed that using calibrated spectral basis
functions (Chebyshev or Fourier modes of the spatial coordinate domain) as the initial
slice-token centers produces consistently better neural operator performance than random
initialization. In PhysicsAttention, the slice query matrix `W` is randomly initialized.
Replacing this initialization with principal Fourier modes of the (x,z) spatial
coordinates of the training set should give the attention a physically meaningful
starting point from which to refine via gradient descent.

**Mechanistic target.** Random initialization of slice queries leads to redundant or
poorly distributed physical partitions; spectral init breaks symmetry in a
domain-aware way. Falsifying result: converged performance is the same regardless of
init, meaning optimization escapes bad inits equally well.

**Implementation sketch.**

Precompute the top-k Fourier modes from training node positions as a one-time step
before the training loop:

```python
def compute_spectral_init(train_ds, n_modes, space_dim=2, device='cpu'):
    # Collect all node (x,z) positions from a subsample of training data
    all_pos = []
    for i in range(min(50, len(train_ds))):
        x, y, is_surf = train_ds[i]
        all_pos.append(x[:, :2])   # [N, 2] -- node positions (x, z)
    all_pos = torch.cat(all_pos, dim=0)  # [total_N, 2]
    # Fit PCA modes as initial slice directions
    U, S, V = torch.linalg.svd(all_pos.T @ all_pos / all_pos.shape[0], full_matrices=False)
    # Return top n_modes direction vectors, repeated to fill slice_num
    return V[:n_modes]  # [n_modes, 2]
```

In `PhysicsAttention.__init__`, after creating `self.slice_q`, overwrite its first
`min(n_modes, slice_num)` rows with the spectral init vectors (mapped to hidden_dim
via a fixed random projection).

Add `Config` field `spectral_slice_init: bool = True`.

**Risk.** Medium. The PCA/SVD of spatial coordinates is a reasonable proxy for spectral
modes but is not the full PCSM formulation. The init is overwritten by gradient descent;
if training is long enough the difference may wash out. For short 30-minute runs the
init matters more.

**Time impact.** One SVD of a 2x2 matrix (after aggregation). Negligible. 30-minute fine.

**Taste scores (frontier refinement, mechanistic grounding=2, research-state value=2, execution=3).**

---

### 12. Gradient Clipping Tuning (max_norm=0.5 from default 1.0)

**Hypothesis.** The current gradient clipping uses `max_norm=1.0` (standard). For
irregular mesh batches with highly variable mesh sizes (74K to 242K nodes), large mesh
samples produce proportionally larger gradient norms when their loss contributions are
not normalized by node count. Although vol_loss and surf_loss are per-node normalized,
the final loss scale still varies. A tighter clip at 0.5 may stabilize training on high-
Re cruise samples (largest meshes, most extreme values) without significantly slowing
convergence for easier samples.

**Mechanistic target.** Large-mesh, high-Re samples generate spikes in gradient norm
that destabilize optimizer state; tighter clipping reduces variance at the cost of
slightly slower large-mesh learning. Falsifying result: val_geom_camber_cruise (largest
mesh domain) shows no improvement, meaning gradient spikes are not a limiting factor.

**Implementation sketch.**

```python
# Current:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Proposed:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
```

Add `Config` field `grad_clip_norm: float = 0.5`.

**Risk.** Low. Well-understood technique. The main risk is slowing convergence for
high-Re samples disproportionately, but the cosine LR schedule should absorb this.

**Time impact.** No overhead. 30-minute budget fine.

**Taste scores (diagnostic, mechanistic grounding=2, research-state value=2, execution=4).**

---

### 13. Warmup LR Schedule (500-Step Linear Warmup + Cosine)

**Hypothesis.** The current schedule uses CosineAnnealingLR with no warmup. For neural
operators on irregular meshes, the first few hundred steps see highly variable batch
contents (different mesh sizes, domains, Re values) — jumping to lr=9e-4 immediately
risks overshooting the attention weight initialization. A 500-step linear warmup from
lr=1e-6 to cfg.lr, followed by cosine anneal, is standard in transformer training and
may stabilize early optimization. This was NOT tried in the existing PR history (warmup
experiments targeted different schedules -- WarmRestart, OneCycle).

**Mechanistic target.** Unstable early training (high LR on cold attention weights)
corrupts early slice assignments; warmup prevents this. Falsifying result: warmup+cosine
performs identically to cold cosine, meaning early LR is not a limiting factor.

**Implementation sketch.**

```python
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

warmup_steps = 500
def warmup_lambda(step):
    return min(1.0, step / warmup_steps)

warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                         milestones=[warmup_steps])
```

Requires switching from epoch-level to step-level scheduler stepping. Change
`scheduler.step()` call in training loop to per-step. Add `Config` field
`lr_warmup_steps: int = 500`.

**Risk.** Low-medium. The warmup_steps value of 500 may be too few or too many
depending on total training duration (30 min). At batch_size=4 and ~2s/batch for large
meshes, 500 steps ~ 17 minutes -- this may cut into useful cosine annealing time.
Consider `warmup_steps=200` as a safer default.

**Time impact.** No overhead. 30-minute budget fine.

**Taste scores (frontier refinement, mechanistic grounding=2, research-state value=2, execution=3).**

---

## Priority Order for Assignment

Recommended assignment order for 8 idle students (best diagnostic/ROI first):

1. **Mask-Aware PhysicsAttention** (#1) -- correctness fix, highest information value
2. **Per-Sample Re-Normalization** (#2) -- addresses known Re-imbalance in gradients
3. **Non-NACA Specials Disambiguation** (#5) -- cheap, high-precision diagnostic
4. **Temperature Annealing** (#6) -- zero-cost, directly tests slice formation quality
5. **Surface-Anchored Cross-Attention** (#3) -- bigger bet, directly targets primary metric
6. **MFN-Style Re Conditioning** (#4) -- tests whether FiLM failure was mechanism-specific
7. **Deeper Backbone (n_layers=7, slice_num=32)** (#8) -- clean architecture ablation
8. **Dedicated Pressure Head** (#9) -- minimal code, directly tests decoder bottleneck

Ideas #7, #10, #11, #12, #13 are secondary; assign if additional slots open up or if
primary eight show mixed results.

---

## Ruled-Out Directions (Do Not Repeat)

- **FiLM affine Re conditioning** (charliepai2f): dead-ended; MFN multiplicative is the
  next logical test of the Re-conditioning hypothesis
- **SWA (Stochastic Weight Averaging)**: closed dead-end
- **Warm restarts / OneCycleLR**: closed dead-ends
- **Aux surface-pressure head** (separate loss term): closed dead-end
- **Richer Re features (5-D)**: closed dead-end
- **Scheduled surf_weight warmup**: closed dead-end
- **Lion optimizer**: closed dead-end (or not improved enough to merge)
- **Lookahead optimizer**: no improvement logged
- **Sobolev loss / divergence-free penalty**: closed dead-ends
- **BF16 training**: closed dead-end or superseded

---

## Current Open Uncertainties

1. **Is the PhysicsAttention padding noise actually significant?** Batch size=4 means
   at most 3 padded-to samples per batch; worst case is one 242K-node cruise sample
   padding a 85K-node raceCar sample. The magnitude of contamination is unknown.
   Mask-aware PhysicsAttention (#1) is the diagnostic.

2. **Is the primary bottleneck in the encoder (slice quality) or decoder (output head)?**
   #9 (dedicated pressure head) vs #1/#6 (slice quality) is the diagnostic pair.

3. **How much of the OOD camber gap is featurization vs generalization?** The non-NACA
   disambiguation (#5) and OOD curriculum (#7) probe different ends of this question.
