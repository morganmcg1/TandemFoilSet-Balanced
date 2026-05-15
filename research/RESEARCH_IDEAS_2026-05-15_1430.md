# Research Ideas — TandemFoilSet CFD Surrogate
Generated: 2026-05-15 14:30
Branch: icml-appendix-charlie-pai2i-24h-r3

## Context

Baseline: Transolver with 5 layers, 128 hidden, 4 heads, 64 slices, mlp_ratio=2.
Primary metric: val_avg/mae_surf_p (surface pressure MAE, equal-weight across 4 val splits).
Hard constraints: SENPAI_TIMEOUT_MINUTES=30, SENPAI_MAX_EPOCHS=50.
Editable: train.py only. Data interface (data/) is read-only.
Available packages: torch, einops, timm, numpy, simple-parsing, tqdm, rich, pyyaml.

## Already tried (do not repeat these)

From sibling branches (pai2g, pai2h series, ~56 PRs):
- surf_weight sweeps (7, 25, 50)
- Model size: 256 hidden, 8 layers, 8 heads
- slice_num 12, 48
- LR warmup schedules
- Gradient clipping
- Per-channel decoder heads
- Lion optimizer
- Weight decay sweeps
- GeGLU, SwiGLU activations
- RMSNorm, LN hybrids
- LayerScale
- DropPath
- Bias-free Linear
- Per-channel surface loss weights
- 2-layer output projection MLP
- Attention dropout
- Re input jitter augmentation
- trunc_normal_ init std=0.01
- asinh pressure transform
- OneCycleLR
- Inter-block additive/multiplicative scaling

---

## Hypothesis 1: Learnable Fourier Positional Encoding (Architecture)

**Slug**: fourier-pos-enc
**Direction**: Reduction in val_avg/mae_surf_p
**Predicted delta**: 3-8% improvement

### What to change in train.py

Add a `FourierPosEnc` module before the `Transolver.preprocess` MLP. This replaces the raw (x,z) coordinates (dims 0-1 of x) with multi-frequency sinusoidal features, giving the model an explicit inductive bias that nearby mesh nodes have correlated flow values.

In `Transolver.__init__`, change the input expansion:

```python
# Add this class above Transolver:
class FourierPosEnc(nn.Module):
    """Learnable Fourier features for 2-D spatial coordinates."""
    def __init__(self, n_freqs: int = 16):
        super().__init__()
        # Log-spaced base frequencies; learned scale per frequency
        self.register_buffer("freqs", torch.logspace(-1, 2, n_freqs))
        self.scale = nn.Parameter(torch.ones(n_freqs))  # per-freq learnable amplitude

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 2] — raw (x, z) in physical space
        # Returns [B, N, 4*n_freqs]
        angles = coords.unsqueeze(-1) * (self.freqs * self.scale)[None, None, None, :]  # [B,N,2,F]
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)  # [B,N,4F]
```

Then in `Transolver.__init__`, replace `fun_dim + space_dim` with `fun_dim + 4*n_freqs` as the MLP input, and in `Transolver.forward`:

```python
def forward(self, data, **kwargs):
    x = data["x"]                        # [B, N, 24]
    pos = x[..., :2]                     # raw (x, z) coords
    feats = x[..., 2:]                   # remaining 22 features
    pos_enc = self.pos_encoder(pos)      # [B, N, 4*n_freqs]
    x_in = torch.cat([feats, pos_enc], dim=-1)
    fx = self.preprocess(x_in) + self.placeholder[None, None, :]
    ...
```

Set `n_freqs=16` (adds 4*16=64 positional dims instead of raw 2). Update `model_config`:
- `fun_dim = X_DIM - 2` (unchanged, feats)
- Change preprocess input size: `MLP(fun_dim + 4*n_freqs, n_hidden*2, n_hidden, ...)` inside `Transolver.__init__`

### Why it should work

The Transolver's slice routing currently sees raw normalized (x,z) coordinates embedded through a single linear projection. Fourier features let the model represent high-frequency spatial variations (boundary layer gradients, wake patterns) without needing many layers to compose them. This is the core insight of NeRF and Neural Tangent Kernel theory: random/learned Fourier features dramatically accelerate learning of high-frequency mappings. In CFD, pressure and velocity gradients are sharpest near surfaces — exactly the nodes where we care most (mae_surf_p). The log-spaced frequencies match the multi-scale nature of overset meshes (coarse background, dense foil zones).

### Risk / failure mode

If the raw (x,z) coordinates are already being used for slice routing in a way that is tightly tuned, remapping them could disrupt the learned slice assignments. Mitigate: keep the raw coords as additional channels alongside the Fourier features (concatenate all 26+64 dims rather than replacing). If the preprocess MLP struggles with the much larger input, reduce n_freqs to 8.

---

## Hypothesis 2: Huber Loss for Robustness to High-Re Outliers (Loss)

**Slug**: huber-loss
**Direction**: Reduction in val_avg/mae_surf_p
**Predicted delta**: 2-6% improvement

### What to change in train.py

Replace the squared-error loss in the training loop (lines 447-453) with smooth-L1 (Huber) loss. This is a one-line change plus a new config parameter:

```python
# In Config dataclass, add:
huber_delta: float = 1.0   # transition point; 1.0 in normalized space ≈ 1 std unit

# In training loop, replace:
sq_err = (pred - y_norm) ** 2
# With:
sq_err = F.huber_loss(pred, y_norm, reduction='none', delta=cfg.huber_delta)
```

The variable `sq_err` is then used unchanged for vol_loss and surf_loss computation (lines 450-453), so all downstream masking stays identical. Set `huber_delta=1.0` as default.

### Why it should work

MSE squares large residuals, meaning high-Re samples (which have physical values 5-10x larger than low-Re) dominate the gradient signal even after global normalization, because normalization by a single global y_std does not eliminate per-sample scale variation. Per `program.md`, per-sample y_std varies by an order of magnitude within each split. The Huber loss caps the gradient for large outliers at delta, reducing the bias toward fitting extreme high-Re regimes at the expense of moderate-Re samples where surface pressure is most relevant to the OOD camber splits. The `val_geom_camber_rc` and `val_geom_camber_cruise` splits (geometry OOD) are likely dominated by moderate-Re regimes where Huber's MAE-like tail should reduce relative error.

### Risk / failure mode

Too-small delta (e.g., 0.1) turns the loss purely MAE-like and slows early convergence when large residuals are actually informative. Too-large delta (e.g., 10.0) recovers MSE behavior. The safe operating range is 0.5-2.0 in normalized space. The known failure mode: if the per-sample scale variance is so large that no fixed delta works well across the full training distribution, the improvement may be marginal. In that case, a per-sample normalization scheme (Hypothesis 4) is the complementary fix.

---

## Hypothesis 3: EMA (Exponential Moving Average) of Model Weights (Optimization)

**Slug**: ema-weights
**Direction**: Reduction in val_avg/mae_surf_p
**Predicted delta**: 1-4% improvement; primarily reduced variance across checkpoints

### What to change in train.py

Add an EMA shadow of model parameters that updates each training step. Use the EMA model for validation and checkpoint selection. No new packages needed — pure PyTorch.

```python
# After model definition (line ~402), add:
class ModelEMA:
    """Exponential moving average of model weights for inference stability."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Load EMA weights into model for evaluation."""
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, original_state: dict) -> None:
        """Restore original (non-EMA) weights after evaluation."""
        pass  # caller holds original_state

ema = ModelEMA(model, decay=0.9999)

# In the training loop, after optimizer.step() (line ~457):
ema.update(model)

# Before validation (line ~469), replace model.eval() block with:
original_state = {k: v.clone() for k, v in model.state_dict().items()}
ema.apply(model)
model.eval()
# ... run validation ...
model.load_state_dict(original_state)

# For checkpoint saving, save ema.shadow rather than model.state_dict()
```

Set decay=0.9999. For 50-epoch runs with ~375 steps/epoch (~18750 steps), decay^18750 ≈ 0.16 — meaning the oldest weights have < 16% influence, appropriate for a 50-epoch run.

### Why it should work

EMA weights are standard practice in modern ML (Stable Diffusion, DINO, DeiT) because they smooth out the high-curvature trajectory of AdamW near the end of cosine annealing. With short 50-epoch runs and variable-mesh batches, individual validation checkpoints can be noisy. EMA systematically gives a better "average" model that generalizes better than any single checkpoint. It costs one additional copy of model weights in memory (~4MB for 128 hidden Transolver) and negligible compute per step.

### Risk / failure mode

With very small dataset and short training, the model may not converge enough for EMA to diverge from the raw checkpoint. Decay=0.9999 may be too slow for 50 epochs; try decay=0.999 or even 0.9990 if EMA barely differs from the final checkpoint. Failure mode: if training diverges early, EMA accumulates a bad model silently.

---

## Hypothesis 4: Per-Sample Scale Normalization in the Loss (Data/Normalization)

**Slug**: per-sample-scale-norm
**Direction**: Reduction in val_avg/mae_surf_p, especially on OOD splits
**Predicted delta**: 5-12% on val_geom_camber_* splits; 2-5% overall

### What to change in train.py

In the training loop, compute a per-sample scale factor from the target values and apply it to normalize the loss contribution. Do NOT change the model's output space — normalization stays in the global (y - y_mean)/y_std space. This only rescales the loss gradient weighting.

```python
# Replace the loss computation block (lines 447-453):

y_norm = (y - stats["y_mean"]) / stats["y_std"]
pred = model({"x": x_norm})["preds"]

# Per-sample scale: std of valid target nodes in normalized space
# Shape: [B], computed per sample to equalize gradient contribution
with torch.no_grad():
    per_sample_std = (y_norm * mask.unsqueeze(-1)).std(dim=(1, 2)).clamp(min=0.1)  # [B]
    loss_scale = 1.0 / per_sample_std  # [B] — upweight low-variance (low-Re) samples
    loss_scale = loss_scale / loss_scale.mean()  # normalize to unit mean

err = (pred - y_norm) ** 2 * loss_scale[:, None, None]  # broadcast over N, C

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

No config changes needed (uses existing surf_weight).

### Why it should work

`program.md` explicitly flags that per-sample y_std varies by an order of magnitude across Re regimes (e.g., val_single_in_dist max y_std ~2077 vs typical ~458). After global normalization, a high-Re sample still has ~3-5x higher normalized variance than a low-Re sample — so MSE disproportionately gradients toward fitting high-Re. The OOD geometry splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) likely include the full Re range; equalizing per-sample loss scale should improve generalization to unseen camber values because low-Re and moderate-Re shapes get equal training signal. This is equivalent to importance sampling the loss by the inverse of each sample's difficulty in the current normalized space.

### Risk / failure mode

If the model genuinely needs more emphasis on high-Re samples to learn the full solution manifold, this reweighting could hurt high-Re performance (val_re_rand split). Monitor both the averaged metric and the per-split breakdown. Failure mode: if all samples in a batch happen to be the same Re (possible with weighted domain sampler), per-batch normalization is close to no-op. True per-sample (not per-batch) scaling avoids this.

---

## Hypothesis 5: Horizontal Flip Augmentation (Augmentation)

**Slug**: hflip-augment
**Direction**: Reduction in val_avg/mae_surf_p
**Predicted delta**: 2-6% improvement, especially on OOD camber splits

### What to change in train.py

Add a physical symmetry augmentation in the training loop. For NACA airfoils, the flow is symmetric under horizontal reflection (x → -x, Ux → -Ux, Uy unchanged, p unchanged) when the AoA is mirrored. However, since AoA is non-zero and asymmetric, a safer augmentation is a stochastic geometric jitter that exploits the near-symmetry.

Actually, the most principled and safe augmentation for this dataset is to flip the flow along z (the z-velocity): since Uy is the vertical component and many raceCar samples have near-symmetric upper/lower surface, we can reflect z → -z, Uy → -Uy, AoA → -AoA, saf → -saf.

```python
# In the training loop, before the forward pass (after x/y are on device):

# Stochastic z-reflection augmentation (50% probability)
if torch.rand(1).item() < 0.5:
    # Reflect z-coordinate: dim 1 of x (z-position)
    x = x.clone()
    y = y.clone()
    x[..., 1] = -x[..., 1]          # z-coord → -z
    x[..., 3] = -x[..., 3]          # saf z-component → -saf (dim 3)
    x[..., 14] = -x[..., 14]        # AoA foil 1 → -AoA (dim 14)
    x[..., 18] = -x[..., 18]        # AoA foil 2 → -AoA (dim 18)
    y[..., 1] = -y[..., 1]          # Uy → -Uy
```

Apply AFTER moving to device, BEFORE normalization. The augmentation must happen in physical space before normalization so statistics are correct.

### Why it should work

The raceCar domain uses inverted airfoils generating downforce; the cruise domain uses positive-lift foils. A z-reflection of the entire flow field maps a downforce case to an upforce case — but since both are present in training, this augmentation actually creates valid physical examples that may bridge the distribution between the two AoA sign conventions. Crucially, it doubles the effective training set size for the geometry OOD splits which use the same reflection symmetry. Data augmentation via physical symmetries is a standard technique in scientific ML (e.g., rotation invariance in molecular dynamics) and is especially powerful when the dataset is limited (~1500 training samples).

### Risk / failure mode

The augmentation requires careful feature mapping. Errors in which features to flip (e.g., incorrectly flipping NACA camber position or gap) would introduce corrupted training examples. The key risk is that AoA, saf, and position transforms must be verified against the physical meaning of each feature. Feature dims 14 and 18 are AoA in radians — these should negate under z-flip. Feature dim 3 is "signed arc-length saf" — the z-component should negate. Feature dim 1 is z-position — negates. Features 4-11 (dsdf) are distance-based and likely magnitude-only — do NOT flip these. Run a quick sanity check: after augmentation, the signed features should have the correct mean shift.

---

## Hypothesis 6: Freestream-Condition Curriculum Learning (Sampling/Curriculum)

**Slug**: re-curriculum
**Direction**: Reduction in val_avg/mae_surf_p, especially val_re_rand
**Predicted delta**: 3-8% improvement

### What to change in train.py

Implement a two-phase curriculum: first half of training sorts by log(Re) ascending (easy → hard), second half uses random sampling. This matches the TandemFoilSet paper's reported "training schemes based on freestream conditions."

```python
# In Config, add:
curriculum_frac: float = 0.5   # fraction of epochs to use curriculum

# After load_data, build a sorted index for curriculum phase:
# Extract log(Re) for each training sample (feature dim 13 in x, unnormalized)
# We need to sort train_ds by Re — add this after train_ds is loaded:

from torch.utils.data import Subset

# Build sorted indices by Re (ascending)
re_vals = []
for idx in range(len(train_ds)):
    xi, _, _ = train_ds[idx]      # x: [N, 24]; dim 13 is log(Re)
    re_vals.append(xi[:, 13].mean().item())   # mean over nodes (constant per sample)
sorted_indices = sorted(range(len(train_ds)), key=lambda i: re_vals[i])

# In the epoch loop, decide loader:
curriculum_epochs = int(MAX_EPOCHS * cfg.curriculum_frac)
if epoch < curriculum_epochs:
    # Curriculum: sequential pass through Re-sorted samples
    curr_loader = DataLoader(
        Subset(train_ds, sorted_indices),
        batch_size=cfg.batch_size,
        shuffle=False,   # preserve Re order
        **loader_kwargs
    )
    train_iterator = curr_loader
else:
    train_iterator = train_loader   # back to weighted random sampler
```

The curriculum_frac=0.5 means first 25 epochs are Re-sorted, last 25 epochs are standard weighted random.

### Why it should work

The TandemFoilSet ICLR 2026 paper reports that curriculum learning from freestream conditions (Re ordering) is one of the key improvements over vanilla training — described as "training schemes based on freestream conditions." Starting from low-Re (laminar, smooth) solutions and progressing to high-Re (complex wakes, separation) follows the physical difficulty gradient: low-Re flows are smoother and easier to represent, so the model builds a good prior before encountering turbulent examples. This prevents the high-Re gradient spike problem described under Hypothesis 4, but through data ordering rather than loss reweighting.

### Risk / failure mode

With only 50 epochs, 25 epochs of non-random sampling may hurt domain balance — the Re-sorted sequence will mix raceCar and cruise samples in a different ratio than the weighted sampler. Domain imbalance could hurt the geometry OOD splits. Mitigate: within the curriculum phase, still apply the weighted sampler logic (only sort within each domain separately) — this is more complex but safer. If curriculum_frac=0.5 hurts, try 0.25. Stop condition: if val_geom_camber splits diverge while val_re_rand improves, curriculum is hurting geometric generalization.

---

## Hypothesis 7: Local Running Reynolds Number as Physics Feature (Physics-Aware)

**Slug**: local-re-feature
**Direction**: Reduction in val_avg/mae_surf_p, especially surface nodes
**Predicted delta**: 4-10% improvement on surface metrics

### What to change in train.py

Add a derived physics feature: local running Reynolds number Re_x = Re * |x_chord|, where x_chord is the streamwise position of each node. This feature encodes where in the boundary layer development the node sits — a key quantity for predicting local wall pressure and velocity.

```python
# In train.py, add a feature augmentation function:

def add_local_re_feature(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Append local Re_x = Re * |x_coord| as a 25th input feature.
    
    x: [B, N, 24] in PHYSICAL (unnormalized) space
    Returns: [B, N, 25]
    """
    # Dim 0: x-coord (streamwise position, physical)
    # Dim 13: log(Re), physical
    log_re = x[..., 13:14]                    # [B, N, 1]
    re = torch.exp(log_re)                     # [B, N, 1]
    x_coord = x[..., 0:1].abs()               # [B, N, 1] streamwise |x|
    re_x = torch.log1p(re * x_coord)          # log(1 + Re_x) to compress range
    return torch.cat([x, re_x], dim=-1)       # [B, N, 25]
```

Update the training loop to call this before normalization:

```python
# After x.to(device), before normalization:
x_aug = add_local_re_feature(x, stats)   # [B, N, 25]
# Extend stats to handle the new feature (normalize re_x separately)
re_x_mean = x_aug[..., 24].mean()   # or precompute from training set
re_x_std = x_aug[..., 24].std().clamp(min=1e-6)
x_norm = torch.cat([
    (x_aug[..., :24] - stats["x_mean"]) / stats["x_std"],
    (x_aug[..., 24:25] - re_x_mean) / re_x_std
], dim=-1)
```

Update `model_config`:
```python
model_config = dict(
    ...
    fun_dim = X_DIM - 2 + 1,   # 22 + 1 = 23 (the new re_x feature)
    ...
)
```

Precompute re_x_mean/std from a pass over train_ds before training starts.

### Why it should work

The B-GNN paper (arXiv 2503.18638) demonstrates that adding local boundary-layer physics features (local Re_x = Re * x_chord) reduces model size by 83% while maintaining the same accuracy — the feature directly encodes the viscous length scale at each point. In the TandemFoilSet, surface pressure variation is driven by the boundary layer state (attached, transitional, separated), which Re_x captures at the node level. The model currently sees only global Re (log(Re) in dim 13), which is constant across all nodes in a sample. Re_x gives the model a spatially-varying physics prior that varies along the chord — particularly useful for separating the leading-edge stagnation pressure from the trailing-edge separated flow, which are the hardest nodes to predict. Log1p compression prevents the feature from dominating at high Re on long chords.

### Risk / failure mode

The normalization of the new feature must be precomputed from the training set (not computed batch-by-batch) to avoid leakage and instability. If re_x_mean/std are computed only from the first batch, the normalization will be wrong. Use a pre-training pass over 100 samples to estimate mean/std. Failure mode: the re_x feature has very different distributions for surface nodes (chord-aligned) vs. volume nodes (arbitrary position) — the model may confuse these. Mitigate by multiplying re_x by is_surface (dim 12) so it is zero for volume nodes, keeping it a surface-only signal.

---

## Hypothesis 8: Dual-Branch Surface/Volume Prediction Heads (Architectural Alternative)

**Slug**: dual-branch-heads
**Direction**: Reduction in val_avg/mae_surf_p
**Predicted delta**: 4-10% improvement, especially on surface pressure

### What to change in train.py

Replace the single output head in the last TransolverBlock with two separate prediction heads: one for surface nodes, one for volume nodes. The shared backbone (4 layers) learns a common representation; the 5th layer splits into two specialized decoders.

```python
# Replace TransolverBlock with a modified final-layer version:
class DualHeadTransolverBlock(nn.Module):
    """Final layer with separate surface/volume output heads."""
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, out_dim=1, slice_num=32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        # Separate output heads for surface vs. volume
        self.ln_surf = nn.LayerNorm(hidden_dim)
        self.head_surf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ln_vol = nn.LayerNorm(hidden_dim)
        self.head_vol = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, fx, is_surface=None):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        # Select head based on is_surface
        surf_out = self.head_surf(self.ln_surf(fx))   # [B, N, out_dim]
        vol_out = self.head_vol(self.ln_vol(fx))       # [B, N, out_dim]
        if is_surface is not None:
            # Blend: use surf_out for surface nodes, vol_out for volume nodes
            s = is_surface.float().unsqueeze(-1)       # [B, N, 1]
            return s * surf_out + (1 - s) * vol_out
        return surf_out  # fallback (eval without is_surface)
```

Modify `Transolver.forward` to pass `is_surface` to the last block:

```python
def forward(self, data, **kwargs):
    x = data["x"]
    is_surface = data.get("is_surface")   # [B, N], optional
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for i, block in enumerate(self.blocks):
        if i == len(self.blocks) - 1 and hasattr(block, 'head_surf'):
            fx = block(fx, is_surface=is_surface)
        else:
            fx = block(fx)
    return {"preds": fx}
```

Pass is_surface through the training/eval loops:
```python
pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
```

### Why it should work

Surface and volume nodes have fundamentally different physics: surface nodes sit on no-slip walls with sharp pressure gradients and boundary layer profiles; volume nodes sit in smooth free-stream or wake regions. A single output head must simultaneously learn both mappings, which is a harder multi-task problem. Dual heads let the model specialize: the surface head can learn the sharp pressure coefficient profile (Cp), while the volume head learns the smoother far-field velocity decay. This is inspired by AB-UPT (TMLR 2025) which uses decoupled surface/volume branches, and B-GNN which processes boundary nodes with specialized graph operations. The parameter overhead is minimal: one additional 3-layer MLP of size hidden→hidden→3 (~49K params on top of ~1.6M base), negligible compute.

### Risk / failure mode

The is_surface tensor is already in x[:, 12] as a binary feature — the model already has access to this signal. If the baseline has already learned to use this implicitly, the dual head may be redundant. The critical distinction is that this approach forces DIFFERENT weight matrices for surface vs. volume predictions, not just gating. Failure mode: gradient mixing at the blend boundary (nodes where is_surface transitions) may cause artifacts. Mitigate: apply a soft blend rather than hard gating, or add a small residual from the shared representation to each head.

---

## Summary Table

| # | Slug | Direction | Research Mode | Predicted Delta |
|---|------|-----------|---------------|-----------------|
| 1 | fourier-pos-enc | Architecture | Tier Shift | 3-8% |
| 2 | huber-loss | Loss | Diagnostic | 2-6% |
| 3 | ema-weights | Optimization | Frontier Refinement | 1-4% |
| 4 | per-sample-scale-norm | Data/Normalization | Diagnostic | 5-12% OOD |
| 5 | hflip-augment | Augmentation | Tier Shift | 2-6% |
| 6 | re-curriculum | Sampling/Curriculum | Tier Shift | 3-8% |
| 7 | local-re-feature | Physics-Aware | Tier Shift | 4-10% surface |
| 8 | dual-branch-heads | Architectural Alternative | Tier Shift | 4-10% surface |

## Taste Rubric Scores

| # | Slug | Mechanistic Grounding | Research-State Value | Execution Value | Total |
|---|------|-----------------------|----------------------|-----------------|-------|
| 1 | fourier-pos-enc | 3 | 3 | 3 | 9/12 |
| 2 | huber-loss | 4 | 3 | 4 | 11/12 |
| 3 | ema-weights | 3 | 2 | 4 | 9/12 |
| 4 | per-sample-scale-norm | 4 | 4 | 4 | 12/12 |
| 5 | hflip-augment | 3 | 3 | 3 | 9/12 |
| 6 | re-curriculum | 3 | 3 | 3 | 9/12 |
| 7 | local-re-feature | 4 | 4 | 3 | 11/12 |
| 8 | dual-branch-heads | 3 | 3 | 3 | 9/12 |

**Top priorities by rubric**: per-sample-scale-norm (12/12), huber-loss (11/12), local-re-feature (11/12).

## Research State Update

**Current best explanation of bottleneck**: The primary limitation is the mismatch between global normalization (single y_mean/y_std across all domains) and the order-of-magnitude variance in per-sample target scale across Re regimes. This creates biased gradients toward high-Re samples in MSE loss. The secondary limitation is that position encoding in the Transolver is purely linear — a single projection from raw (x,z) — which may not capture multi-scale spatial structure needed to predict boundary layer behavior.

**Open uncertainties**:
1. Does global normalization + MSE cause measurable gradient bias toward high-Re, or is the WeightedRandomSampler already correcting for this implicitly?
2. Is the Transolver's slice routing finding physically meaningful "slices" (e.g., surface vs. wake) or arbitrary data-driven partitions?
3. Does the model genuinely benefit from surface-specific inductive biases, or does the is_surface feature (dim 12) already give the model sufficient signal?

**Next discriminating experiment**: `per-sample-scale-norm` is the highest-value diagnostic — it directly tests the gradient bias hypothesis and requires a minimal code change (< 10 lines). If it improves val_geom_camber splits while leaving val_re_rand neutral, the gradient bias hypothesis is confirmed.

**Stop condition**: If any of the top-3 hypotheses (per-sample-scale-norm, huber-loss, local-re-feature) show > 5% regression on val_avg/mae_surf_p, the hypothesis family is falsified and attention should shift to Hypothesis 1 (Fourier pos enc) or Hypothesis 8 (dual-branch heads) as the next tier.
