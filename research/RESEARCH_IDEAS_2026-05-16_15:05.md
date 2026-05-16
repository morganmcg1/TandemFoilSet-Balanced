# Round-7 Research Ideas — 2026-05-16 15:05

**Track:** willow-pai2i-48h-r4  
**Base config:** SwiGLU + mlp_ratio=3 + epochs=12-14 + Fourier PE (num_freq=4) + coord_noise_std=0.01 + L1 loss + lr=5e-4  
**Current baseline:** val_avg/mae_surf_p = 59.0038, test_avg/mae_surf_p = 50.7368 (PR #3908)  
**Per-split test:** single_in_dist=57.19, geom_camber_rc=62.63, geom_camber_cruise=33.66, re_rand=49.46

All 8 hypotheses build on the merged baseline stack. mlp_ratio=3 is the default unless noted.

---

## Hypothesis 1: Multi-scale Fourier PE (num_freq sweep + wider freq range)

**Title:** Multi-scale Fourier PE — widen frequency range

**Mechanism:**  
The current Fourier PE uses 4 log-spaced frequencies from base 2^0=1 to 2^3=8, encoded as `2.0 ** arange(num_freq)`. This covers spatial scales from ~1 mesh unit to ~8 mesh units. For 2D aerodynamic flows on meshes spanning the full chord (~O(1) in normalized coords) down to near-wall boundary layer structure (~O(0.01) in normalized coords), the current range may miss fine boundary-layer structure and/or large-scale wake patterns. A wider range — e.g., 8 frequencies spanning 2^(-2) to 2^5 (0.25 to 32) — doubles the spectral coverage and should help the model learn both near-wall pressure gradients and far-field flow simultaneously. The encoded dimension grows from 4*4=16 to 4*8=32, adding 16 non-trainable features, which is low-cost.

**Code change:**  
In `Config`, change `num_freq` default or add CLI flag. The `fourier_pos_encode` function already supports arbitrary `num_freq`. The key change is also shifting the frequency range. In the current code:
```python
freqs = 2.0 ** torch.arange(num_freq, dtype=coords.dtype, device=coords.device)
```
Modify `encode_inputs` / `fourier_pos_encode` to accept a `freq_min_exp` and `freq_max_exp` argument, using `torch.linspace(freq_min_exp, freq_max_exp, num_freq)` instead of `arange`. Then test two variants:
- Arm A: `num_freq=8, freq_min=0, freq_max=7` (same octaves, more resolution)  
- Arm B: `num_freq=8, freq_min=-2, freq_max=5` (wider range, covers finer scales)

`ENCODED_X_DIM` adjusts automatically since it's `4 * cfg.num_freq + (X_DIM - 2)`. `model_config["fun_dim"]` stays correct.

**Concrete diff:**
1. Add `num_freq_min_exp: float = 0.0` and `num_freq_max_exp: float = None` to Config (None = use arange as before)  
2. In `fourier_pos_encode`: if `freq_max_exp is not None`, use `torch.linspace(freq_min_exp, freq_max_exp, num_freq)` for exponents  
3. Run `--num_freq 8` as simplest first screen; if regress, try arm B with explicit range

**Risk:** Low-medium. Increasing `num_freq` widens the model input, which requires no architecture changes (just `fun_dim` auto-recalculates). The only risk is that extra high-frequency components add noise rather than signal on coarse mesh regions.

**Expected observable:** Improvement on `val_geom_camber_cruise` (larger meshes, stronger geometric variation) and `val_geom_camber_rc` (OOD geometry). The current num_freq=4 was established on the pre-SwiGLU stack; with SwiGLU's improved expressivity, the model may benefit from finer-grained spatial encoding.

**CLI command:**
```bash
python train.py --epochs 12 --mlp_ratio 3 --num_freq 8 \
  --wandb_group willow-r7-fourier-pe --agent <student>
```

---

## Hypothesis 2: Surface-Curvature Feature Augmentation

**Title:** Add local surface curvature as input feature

**Mechanism:**  
The model currently has no explicit information about the local geometry curvature at surface nodes. Pressure peaks in 2D airfoil flow are driven by leading-edge curvature (suction peak), trailing-edge curvature (separation), and camber-line curvature (loading distribution). The model must infer these implicitly from the raw (x,z) coordinates and DSDF features. Adding a pre-computed local curvature estimate at each surface node directly provides the physics signal the model needs to predict where pressure extremes occur.

Curvature can be computed from the `saf` (signed arc-length, dims 2-3) and `(x,z)` position (dims 0-1) in the input features: finite-difference second derivatives along the surface arc (`d^2x/ds^2`, `d^2z/ds^2`) give the signed curvature `kappa = (x'z'' - z'x'') / (x'^2 + z'^2)^{3/2}`. However, since `saf` is already normalized and the spatial coordinates come pre-processed, the simplest approach is to approximate curvature from the sample's `x` feature tensor inside the training loop: for surface nodes sorted by arc-length, compute finite differences of (x,z) in the batch.

Simpler alternative (lower implementation risk): use the magnitude of the DSDF gradient as a proxy for curvature — high |grad(DSDF)| at surface nodes corresponds to tight curvature. DSDF dims 4-11 are already distance-based shape descriptors; their finite differences in 2D already approximate curvature-related quantities.

**Recommended implementation (start simple):**  
Rather than computing curvature on-the-fly (which is tricky with the padded irregular mesh), add a pre-computed feature. Since `data/` is read-only, compute it inside `train.py` from the input features:

For each batch sample, at surface nodes, compute:
```python
# Inside the training loop, after x_norm is computed
# Approximate surface curvature from saf and position change
saf = x_norm[..., 2:4]       # (B, N, 2) — signed arc-length  
pos = x_norm[..., 0:2]       # (B, N, 2) — (x, z)
# Use DSDF magnitude as curvature proxy: sum of squared DSDF values
dsdf = x_norm[..., 4:12]     # (B, N, 8)
curvature_proxy = dsdf.norm(dim=-1, keepdim=True)  # (B, N, 1)
x_enc = torch.cat([encode_inputs(x_norm, cfg.num_freq), curvature_proxy], dim=-1)
```
Then increment `ENCODED_X_DIM` by 1. Update `model_config["fun_dim"]` accordingly (`fun_dim = ENCODED_X_DIM - 2`).

**Risk:** Low-medium. The DSDF magnitude proxy is quick to compute and adds 1 feature. The main risk is that this is redundant information already encoded in the DSDF dims themselves, providing no new signal.

**Alternative (higher-signal, more involved):** Pre-compute arc-length-sorted curvature offline and store as a new feature, but that requires changes to data loading which is read-only. Stick with the in-training proxy.

**CLI command:**
```bash
python train.py --epochs 12 --mlp_ratio 3 \
  --wandb_group willow-r7-curvature --agent <student>
```
(With the inline curvature proxy code change in train.py)

---

## Hypothesis 3: Stochastic Depth (DropPath) Regularization

**Title:** DropPath stochastic depth in TransolverBlocks

**Mechanism:**  
Stochastic depth randomly drops entire residual branches during training, acting as a strong regularization that prevents co-adaptation between layers. It was first introduced in Huang et al. (2016) for ResNets and later became standard in DeiT/ViT vision transformers (Touvron 2021). With 5 TransolverBlocks, each contributing attention + FFN residuals, stochastic depth can regularize the depth dimension without reducing capacity at inference time.

The mechanism: with probability `p_drop` (scheduled linearly from 0 to `drop_path_rate` across layers), the entire residual of the block is zeroed at training time. At test time all paths are active and the residual is scaled by `(1 - p_drop)`. This is equivalent to randomly sampling a shallower network at each training step.

For a 5-layer Transolver with SwiGLU FFN (already showing signs of underfitting since best_epoch=12/12 on all seeds), stochastic depth may help regularize rather than restrict capacity. The benefit is most likely on OOD splits (geom_camber_rc, geom_camber_cruise) where generalization is the bottleneck.

**Code change:**
Add a `DropPath` module to `train.py`:
```python
class DropPath(nn.Module):
    """Stochastic depth drop-path for residual networks."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # (B, 1, 1) — same mask applied to all positions in the sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob
```

In `TransolverBlock.__init__`, add:
```python
self.drop_path = DropPath(drop_path_rate)
```

In `TransolverBlock.forward`, change residual additions:
```python
fx = self.drop_path(self.attn(self.ln_1(fx))) + fx
fx = self.drop_path(self.mlp(self.ln_2(fx))) + fx
```

In `Transolver.__init__`, pass linearly-scheduled drop rates per block:
```python
dpr = [drop_path_rate * i / (n_layers - 1) for i in range(n_layers)]  # linear schedule
self.blocks = nn.ModuleList([
    TransolverBlock(..., drop_path_rate=dpr[i], last_layer=(i == n_layers-1))
    for i in range(n_layers)
])
```

Add `drop_path_rate: float = 0.1` to `Config` and pass through `model_config`.

**Risk:** Low. Standard technique with well-understood behavior. The only risk is that with 5 layers (shallow), the drop rates need to be kept modest (0.05-0.15). Start at 0.1.

**Expected observable:** Improvement on OOD val splits (geom_camber_rc, geom_camber_cruise). Potentially small regression on in-dist split if it over-regularizes.

**CLI command:**
```bash
python train.py --epochs 12 --mlp_ratio 3 --drop_path_rate 0.1 \
  --wandb_group willow-r7-droppath --agent <student>
```

---

## Hypothesis 4: Surface-Weighted Loss (Curvature-Scaled surf_weight)

**Title:** Curvature-adaptive surface loss weighting

**Mechanism:**  
The current loss equally weights all surface nodes within the surface term (`surf_loss`). However, pressure prediction error is dominated by a small fraction of high-curvature surface locations — the leading-edge suction peak and trailing-edge pressure recovery — because these nodes have the largest pressure gradients and thus the largest absolute errors. If the model under-fits these critical regions, a uniform surface loss will not punish those errors proportionally.

A curvature-weighted surface loss down-weights smooth surface sections and up-weights leading-edge and trailing-edge nodes, focusing the gradient signal where prediction is hardest and physically most important.

The implementation uses the DSDF norm as a curvature proxy (same as Hypothesis 2, computed inline):
```python
# After x_norm and x_enc are computed, inside the training loop
dsdf_norm = x_norm[..., 4:12].norm(dim=-1)          # (B, N) — curvature proxy
# Normalize per-sample so weights sum to 1 within each sample's surface nodes
surf_weights = dsdf_norm * surf_mask.float()          # zero out non-surface
surf_weights = surf_weights / surf_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
surf_weights = surf_weights * surf_mask.sum(dim=-1, keepdim=True).float()  # re-scale to n_surf
# Use in loss
surf_loss = (err.mean(-1) * surf_weights).sum() / surf_mask.sum().clamp(min=1)
```

This is a purely loss-side change with no architectural modifications. It is especially relevant given that `geom_camber_rc` is our hardest split (test=62.63), and camber changes primarily affect the leading-edge suction peak geometry.

**Risk:** Low. Pure training signal change. The risk is that DSDF-norm is a poor proxy and shifts weight toward unimportant nodes. Can be validated cheaply: if `val_geom_camber_rc` improves but `val_single_in_dist` regresses, the proxy may be mis-calibrated.

**Expected observable:** Primary improvement expected on `val_geom_camber_rc` (hardest OOD split, camber geometry variation). Secondary improvement on `val_geom_camber_cruise`.

**Falsifying result:** If `val_geom_camber_rc` doesn't improve but val_single_in_dist regresses, the curvature proxy is not tracking the right nodes and this direction should be closed.

**CLI command:**
```bash
python train.py --epochs 12 --mlp_ratio 3 --use_curvature_weight \
  --wandb_group willow-r7-curvature-loss --agent <student>
```
(Add `use_curvature_weight: bool = False` to Config)

---

## Hypothesis 5: Camber Symmetry Data Augmentation

**Title:** Camber flip augmentation (x→x, z→-z, AoA→-AoA)

**Mechanism:**  
For symmetric airfoil flow (zero camber, zero AoA), the pressure distribution is symmetric about the chord line: `p(x, z) = p(x, -z)`. For cambered airfoils with non-zero AoA, flipping the z-coordinate and negating the AoA gives a physically equivalent flow from a mirrored geometry.

This augmentation doubles the effective training data size for free. More importantly, it directly expands the coverage of front-foil camber M values, which is exactly the OOD axis being tested in `val_geom_camber_rc` and `val_geom_camber_cruise`. The held-out camber ranges are M=6-8 (raceCar) and M=2-4 (cruise). The augmentation does not inject those exact values but does add a regularizing bias toward symmetric behavior, improving interpolation across camber values.

**Implementation (inside train.py, applied in the training loop before encoding):**
```python
# Data augmentation: random z-flip + AoA negation (50% probability per sample)
# Applied to x_norm BEFORE coord noise and BEFORE Fourier encoding
if training and torch.rand(1).item() < 0.5:
    x_aug = x_norm.clone()
    # Flip z-coordinate (dim 1) for all nodes
    x_aug[..., 1] = -x_aug[..., 1]
    # Negate AoA foil 1 (dim 14) and AoA foil 2 (dim 18)
    x_aug[..., 14] = -x_aug[..., 14]
    x_aug[..., 18] = -x_aug[..., 18]
    # Also flip saf z-component (dim 3): saf is (arc-x, arc-z) dims 2-3
    x_aug[..., 3] = -x_aug[..., 3]
    # Flip y targets: Uy flips sign (dim 1), Ux and p unchanged
    y_aug = y_norm.clone()
    y_aug[..., 1] = -y_aug[..., 1]
    x_norm, y_norm = x_aug, y_aug
```

Note: SAF features (dims 2-3) are signed arc-length along the surface; the z-component (dim 3) flips sign under z-reflection. The DSDF (dims 4-11) are distance-based shape descriptors — under z-reflection these distances are preserved (DSDF is symmetric about the flip by construction since it measures distances to geometry). The NACA params (dims 15-17, 19-21) encode M, P, T — none change sign under z-flip. Gap (dim 22) and stagger (dim 23) are unsigned and don't change.

**Risk:** Medium. The flip logic must be correct for all features. A subtle sign error (e.g., wrong handling of SAF or DSDF) will inject noisy training signal. Validate by checking: after the flip, the augmented sample should yield identical `mae_surf_p` if passed through a baseline checkpoint (up to re-normalization effects). The key correctness check is that Uy changes sign and Ux, p do not.

**Expected observable:** Improvement primarily on `val_geom_camber_rc` and `val_geom_camber_cruise`, since those test OOD geometry generalization. The augmentation most directly expands the effective camber coverage.

**CLI command:**
```bash
python train.py --epochs 12 --mlp_ratio 3 --camber_flip_aug \
  --wandb_group willow-r7-camber-aug --agent <student>
```
(Add `camber_flip_aug: bool = False` to Config; apply inside training loop with the flip logic above)

---

## Hypothesis 6: AdamW Decoupled Weight Decay Tuning + Cosine LR Floor

**Title:** Tune weight_decay + cosine eta_min on mlp_ratio=3

**Mechanism:**  
The current weight_decay=1e-4 was established on an earlier stack (pre-SwiGLU). With SwiGLU inner_dim=320 (+0.25M params vs mlp_ratio=2), the model has more parameters and may benefit from stronger regularization. SwiGLU's gated structure (w1, w2 for gate, w3 for projection, all bias=False) has a different effective parameter geometry than a standard MLP, and the optimal weight decay for bias-free gated layers is often higher than for standard MLPs with bias.

This is a focused hyperparameter experiment: test `weight_decay=1e-3` (10x increase) with two learning rate floors:
- Arm A: `weight_decay=1e-3, eta_min=1e-6` (default cosine floor)
- Arm B: `weight_decay=1e-3, eta_min=1e-5` (slightly raised floor, less aggressive LR decay)

The mechanism: higher weight decay acts as L2 regularization across all parameters, reducing overfitting on the training samples. With epochs=12-14 and best_epoch=12/12 on all seeds (model still training at cutoff), the model may be slightly overfit — stronger WD could improve generalization.

Note: `eta_min` sweep was listed as exhausted (pre-SwiGLU), but this is on the new mlp_ratio=3 stack with different parameter count and potentially different optimal regularization.

**Code change:** Add `eta_min: float = 0.0` to Config. Change cosine lambda:
```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
    cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
    # Scale to [eta_min/lr, 1.0] range
    return cfg.eta_min / cfg.lr + (1.0 - cfg.eta_min / cfg.lr) * cosine_val
```

And pass `weight_decay=cfg.weight_decay` from Config to optimizer (already done via `cfg.weight_decay`).

**Risk:** Low. These are well-understood hyperparameters with narrow sensitivity ranges. The risk of regression is low; the upside is modest (1-3%).

**Expected observable:** Small but consistent improvement across all val splits. If only OOD splits improve, it's a regularization effect; if in-dist improves too, it's better optimization.

**CLI command:**
```bash
# Arm A
python train.py --epochs 14 --mlp_ratio 3 --weight_decay 1e-3 \
  --wandb_group willow-r7-wd-tuning --agent <student>
```

---

## Hypothesis 7: SAM (Sharpness-Aware Minimization) Optimizer

**Title:** SAM optimizer for flatter minima + better generalization

**Mechanism:**  
SAM (Foret et al., 2021) seeks parameters in neighborhoods with uniformly low loss, rather than just locally low loss. It has consistently improved generalization in image classification and more recently in scientific ML settings (PDE surrogates, graph neural networks). The key insight is that flat minima generalize better — SAM makes this explicit in the optimizer.

SAM requires two gradient steps per training step: a forward step to compute the perturbation direction (normalize the gradient and perturb parameters), then a backward step at the perturbed point to compute the true gradient. This doubles training compute per step but often improves val metrics by 2-5% in settings with limited training data (our setting: 1499 training samples is small for a mesh-based PDE surrogate).

For this setting, the OOD val splits (geom_camber_rc at test=62.63, re_rand at test=49.46) are exactly where flat minima should help — smoother loss landscape means better interpolation to unseen geometry/Re.

**Implementation:** SAM does not require a new package; implement inline in train.py:
```python
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021)."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "sharpness"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "sharpness-aware" minimum
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)
```

Training loop changes (replace the current optimizer step):
```python
# First forward-backward pass
pred = model({"x": x_enc})["preds"]
loss = compute_loss(pred, ...)
loss.backward()
optimizer.first_step(zero_grad=True)

# Second forward-backward pass at perturbed point
pred = model({"x": x_enc})["preds"]
loss = compute_loss(pred, ...)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
optimizer.second_step(zero_grad=True)
```

Add `use_sam: bool = False` and `sam_rho: float = 0.05` to Config. When `use_sam=True`, create `SAM(model.parameters(), torch.optim.AdamW, rho=cfg.sam_rho, lr=cfg.lr, weight_decay=cfg.weight_decay)`.

Important: SAM doubles the compute per step. With epochs=12 and 30-minute timeout, this may hit the wall clock limit. Start with epochs=10 to screen viability, then extend if promising.

**Risk:** Medium-high. Doubles training compute (two forward passes per step). May hit timeout with epochs=12. Start with epochs=10 as a viability screen. The optimizer is well-established in the literature but implementation complexity is higher than standard hyperparameter changes.

**Expected observable:** Improvement on OOD splits (geom_camber_rc, geom_camber_cruise, re_rand). Potentially neutral on in-dist. If it's worse everywhere, the compute overhead lost more epochs than the flat-minima benefit gained.

**CLI command:**
```bash
python train.py --epochs 10 --mlp_ratio 3 --use_sam --sam_rho 0.05 \
  --wandb_group willow-r7-sam --agent <student>
```

---

## Hypothesis 8: n_layers=6 Depth Scaling on SwiGLU+mlp_ratio=3

**Title:** n_layers=6 depth scaling on SwiGLU+mlp_ratio=3 base

**Mechanism:**  
Adding a 6th TransolverBlock increases the model's depth and representational capacity. Previous depth experiments failed pre-SwiGLU at epoch=8 (under-converged), but those runs used the old vanilla FFN (mlp_ratio=4, standard MLP). With SwiGLU:
1. The model converges faster per-epoch (SwiGLU gradient signal is cleaner)
2. `mlp_ratio=3` has already proven that the model benefits from more capacity
3. `best_epoch=12/12` on all recent seeds suggests the model is not yet overfit — more depth should not hurt

The additional layer adds ~0.95M params (one TransolverBlock with n_hidden=160, n_head=4, SwiGLU inner=320): `PhysicsAttention (~0.3M) + SwiGLU FFN (~0.4M) + LayerNorm (small) ≈ 0.7M`. Total model grows from ~4.6M to ~5.3M params.

The key change in VRAM: each additional layer adds ~(B * N_max * n_hidden) = (4 * 242000 * 160) ≈ 155M float32 activations per forward pass. That's ~620MB per batch, well within 96GB VRAM budget.

**Code change:** Add `n_layers: int = 5` to Config (already implicitly set in model_config). Pass `n_layers=cfg.n_layers` through:
```python
model_config = dict(
    ...
    n_layers=cfg.n_layers,
    ...
)
```
Then run with `--n_layers 6`.

**Risk:** Low-medium. The pre-SwiGLU failure (epoch=8, under-converged) is explained by the training budget, not by a fundamental capacity mismatch. On the SwiGLU+mlp_ratio=3 base with epochs=14, there should be sufficient budget. Main risk: may need epochs=14-16 to fully converge, meaning the 30-min wall clock could cut off training before the model peaks.

**Expected observable:** Improvement across all splits, primarily in-dist and re_rand (generalization across Re regimes). If the model needs more epochs than the budget allows, it will show continued descent at final epoch without clear improvement over n_layers=5.

**Falsifying result:** If val curve is still descending at epoch 12 and val is worse than n_layers=5 best, this is a budget problem, not a capacity problem. Fix: extend to epochs=16 or increase timeout.

**CLI command:**
```bash
python train.py --epochs 14 --mlp_ratio 3 --n_layers 6 \
  --wandb_group willow-r7-depth --agent <student>
```

---

## Summary Table

| # | Student | Title | Risk | Primary expected gain | n_params delta |
|---|---------|-------|------|----------------------|----------------|
| 1 | TBD | Multi-scale Fourier PE (num_freq=8) | Low | val_geom_camber splits | +0 (feature only) |
| 2 | TBD | DSDF-norm curvature feature | Low-med | val_geom_camber_rc | +tiny (1 feature) |
| 3 | TBD | DropPath stochastic depth (p=0.1) | Low | OOD splits | +0 (no new params) |
| 4 | TBD | Curvature-adaptive surface loss | Low | val_geom_camber_rc | +0 (loss only) |
| 5 | TBD | Camber flip augmentation | Medium | geom_camber splits | +0 (augmentation) |
| 6 | TBD | weight_decay=1e-3 + eta_min tuning | Low | All splits (regularization) | +0 (optim only) |
| 7 | TBD | SAM optimizer (rho=0.05) | Med-high | OOD splits | +0 (optim only) |
| 8 | TBD | n_layers=6 depth scaling | Low-med | In-dist + re_rand | +~0.7M |

---

## Priority Ordering

1. **Hypothesis 8 (n_layers=6)** — depth scaling on SwiGLU is mechanistically clean and the pre-SwiGLU failure is well-explained. High upside, straightforward implementation.
2. **Hypothesis 5 (camber flip aug)** — directly targets the hardest OOD split (geom_camber_rc=62.63). Free data doubling if correct.
3. **Hypothesis 1 (multi-scale Fourier PE)** — established technique, low risk, targets both OOD splits.
4. **Hypothesis 3 (DropPath)** — standard regularization with consistent ViT-era results; easy to implement.
5. **Hypothesis 6 (WD tuning)** — conservative but grounded; complements the SwiGLU + mlp_ratio=3 stack.
6. **Hypothesis 4 (curvature-weighted loss)** — novel and directly targets worst-case nodes; medium implementation complexity.
7. **Hypothesis 2 (curvature feature)** — redundant with Hyp 4 in mechanism but tests a different intervention point.
8. **Hypothesis 7 (SAM)** — highest potential reward on OOD, but also highest compute risk; screen at epochs=10 first.

---

## Notes on Base Config Uncertainty

WIP PRs #3969 (epochs=14, mlp_ratio=2) and #4002 (epochs=14, mlp_ratio=3) will determine whether mlp_ratio=2 or mlp_ratio=3 is the better extended-training base. All round-7 hypotheses above are written for mlp_ratio=3 (current merged baseline). If #3969 wins with mlp_ratio=2, swap `--mlp_ratio 2` in the CLI commands above. The hypothesis logic is independent of this choice.
