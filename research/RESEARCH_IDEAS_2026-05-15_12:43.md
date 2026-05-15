<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — Round 5 (charlie-pai2i-24h-r5)
Generated: 2026-05-15 12:43

Launch context: clean slate on branch `icml-appendix-charlie-pai2i-24h-r5`. No prior PRs in this round. Bold, structural ideas prioritized — only `train.py` (and `pyproject.toml` when a new dependency is needed) are editable. Target: `val_avg/mae_surf_p` lower is better. Current baseline TBD (Transolver default config).

---

## Idea 1: FiLM Re/AoA Conditioning in Every Block

**Title slug:** film-flow-condition-every-block

**Predicted delta on `val_avg/mae_surf_p`:** -8% to -15%

**Why it should work:**
The current Transolver mixes the scalar flow conditions (log(Re), AoA, NACA, gap, stagger — dims 13–23) into the 24D input via a single flat MLP preprocessor and then never explicitly re-injects them. Physics intuition and the ML literature on conditional neural operators both say that Re and AoA are global modifiers of the entire field, not just the input encoding — the difference between Re=100K and Re=5M is a full order of magnitude in velocity and pressure gradients. FiLM (Feature-wise Linear Modulation) injects learned scale and shift at each transformer block, ensuring the conditioning signal reaches every layer of the computation. This directly targets the Re-rand and unseen-camber splits where the model must generalize across regime and geometry space outside training distribution.

**Implementation in `train.py`:**

1. Add a `FiLMConditioner` module that takes the per-sample global flow vector (log(Re), AoA1, AoA2, NACA1, NACA2, gap, stagger = 10 dims from x[:, 13:]) and outputs `(scale, shift)` pairs for each block:
   ```python
   class FiLMConditioner(nn.Module):
       def __init__(self, cond_dim, n_hidden, n_blocks):
           super().__init__()
           # small MLP: cond_dim -> n_hidden -> 2 * n_hidden * n_blocks
           self.net = nn.Sequential(
               nn.Linear(cond_dim, n_hidden),
               nn.SiLU(),
               nn.Linear(n_hidden, 2 * n_hidden * n_blocks),
           )
           self.n_blocks = n_blocks
           self.n_hidden = n_hidden
       def forward(self, cond):  # cond: [B, cond_dim]
           out = self.net(cond)  # [B, 2 * n_hidden * n_blocks]
           out = out.view(cond.shape[0], self.n_blocks, 2, self.n_hidden)
           return out[:, :, 0, :], out[:, :, 1, :]  # scale, shift each [B, n_blocks, n_hidden]
   ```

2. In `TransolverBlock.forward`, accept optional `film_scale` and `film_shift` ([B, n_hidden]) and apply after the attention residual:
   ```python
   # After: fx = fx + self.attn(self.norm1(fx), ...)
   if film_scale is not None:
       fx = fx * (1 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)
   ```

3. In `Transolver.forward`, extract the per-sample condition vector as the mean of flow-condition dims across all nodes (they are identical per sample), run it through `FiLMConditioner`, and pass block-specific scale/shift to each `TransolverBlock`.

4. Extract condition from x: `cond = x[:, :, 13:].mean(dim=1)` (flow conditions are constant across nodes per sample, so mean is exact).

5. Keep `n_hidden=128`, add `cond_dim=11` (dims 13–23), no new packages needed (pure PyTorch).

**Risk / failure mode:**
If the flow condition dims are already well-captured by the flat MLP preprocessor (because the model memorized the training distribution), FiLM may just add noise and slow convergence. The unseen-camber splits are the real test: if NACA dims 15-17 don't carry enough geometric signal, conditioning won't help. Also, the condition vector is extracted from input x which is already normalized — ensure the FiLM MLP gets the raw normalized condition dims, not an additional re-normalization.

**References:**
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018. https://arxiv.org/abs/1709.07871
- Herde et al., "Poseidon: Efficient Foundation Models for PDEs," NeurIPS 2024. https://arxiv.org/abs/2408.02810
- Rahman et al., "U-NO: U-shaped Neural Operators," TMLR 2023. https://arxiv.org/abs/2204.11127 (uses global conditioning injection)

---

## Idea 2: EMA Weights for Checkpoint Selection

**Title slug:** ema-weights-checkpoint

**Predicted delta on `val_avg/mae_surf_p`:** -3% to -7%

**Why it should work:**
A recent TMLR survey (Morningstar et al. 2024) shows EMA of model weights is a near-universal free improvement for generalization, calibration, and OOD robustness across diverse deep learning settings — with essentially no computational overhead beyond a second copy of weights. The current baseline uses raw (non-EMA) weights for checkpoint selection and test evaluation. For the unseen-camber and Re-rand splits — exactly the OOD generalization axes — EMA should systematically reduce variance from the noisy tail of training and produce smoother, better-calibrated predictions. This is the kind of change that compounds with everything else.

**Implementation in `train.py`:**

1. After model creation, initialize an EMA model:
   ```python
   import copy
   ema_model = copy.deepcopy(model)
   ema_model.requires_grad_(False)
   EMA_DECAY = 0.999  # tunable; 0.9999 for very long runs
   ```

2. After each optimizer step in the training loop, update EMA:
   ```python
   with torch.no_grad():
       for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
           p_ema.mul_(EMA_DECAY).add_(p_model, alpha=1.0 - EMA_DECAY)
       for b_ema, b_model in zip(ema_model.buffers(), model.buffers()):
           b_ema.copy_(b_model)
   ```

3. Replace validation calls to use `ema_model` instead of `model`:
   ```python
   split_metrics = {name: evaluate_split(ema_model, val_loaders[name], ...) for name in VAL_SPLIT_NAMES}
   ```

4. Save `ema_model.state_dict()` (not `model.state_dict()`) to `checkpoint.pt`.

5. At test time, load the EMA checkpoint into `ema_model` for final evaluation.

6. No new packages. EMA decay of 0.999 is standard for ~50 epoch runs; if training runs are shorter, use 0.99.

**Risk / failure mode:**
EMA is low risk but gives diminishing returns if training converges cleanly (EMA ~= final model). The benefit is largest early in the convergence plateau and for noisy OOD metrics. If training is extremely short (debug mode), EMA may lag badly. The decay constant needs to be calibrated to training length: `EMA_DECAY = exp(-1 / (T/10))` where T is expected number of optimizer steps to convergence is a good heuristic.

**References:**
- Morningstar et al., "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits," TMLR 2024. https://arxiv.org/abs/2411.18704
- Cai et al., "EfficientViT," CVPR 2023 (uses EMA for detection/segmentation). https://arxiv.org/abs/2205.14756
- Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models," CVPR 2024. https://arxiv.org/abs/2312.02696

---

## Idea 3: Cautious AdamW Optimizer

**Title slug:** cautious-adamw-optimizer

**Predicted delta on `val_avg/mae_surf_p`:** -2% to -5%

**Why it should work:**
C-AdamW (Cautious AdamW, arxiv 2411.16085, ICLR 2026) adds a single masking line to AdamW: gradient updates are gated to zero wherever the momentum vector and raw gradient disagree in sign (`m = (u * g > 0).float()`). This prevents the optimizer from taking steps in directions where recent history and current gradient are in conflict — a regime that is especially common in OOD generalization tasks where the loss landscape is rougher. Empirically tested on LLM pre-training, vision, and RL, it consistently accelerates effective convergence and improves final metrics at zero extra compute cost and near-zero code complexity. The unseen-camber splits in particular have a rough loss landscape (the geometry is outside training distribution), making cautious masking valuable.

**Implementation in `train.py`:**

Replace the AdamW optimizer with a drop-in `CautiousAdamW`:

```python
class CautiousAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])
            # Standard AdamW update
            torch.optim.adamw(params_with_grad, grads, exp_avgs, exp_avg_sqs,
                              max_exp_avg_sqs, state_steps, amsgrad=False,
                              beta1=beta1, beta2=beta2, lr=group["lr"],
                              weight_decay=group["weight_decay"], eps=group["eps"],
                              maximize=False, foreach=None, capturable=False,
                              differentiable=False, fused=False)
            # Cautious mask: only update where momentum and grad agree
            for p, g, m in zip(params_with_grad, grads, exp_avgs):
                mask = (m * g > 0).to(dtype=p.dtype)
                mask = mask * (mask.numel() / (mask.sum() + 1))  # normalize to preserve update scale
                p.data.mul_(mask + (1 - mask))  # no-op to preserve structure
                # Actually mask the delta: recompute needed
        return loss
```

**Simpler alternative (recommended for clarity):** Just subclass and post-hoc apply the mask after the parent step by tracking the param deltas. See the reference implementation at https://github.com/kyleliang919/C-Optim for the clean 10-line version. Add `cautious_adamw.py` as a small helper in the repo root (not in `data/`), import it, swap optimizer:

```python
from cautious_adamw import CautiousAdamW
optimizer = CautiousAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
```

No new packages needed — pure PyTorch.

**Risk / failure mode:**
The mask normalizes step size to preserve expected update magnitude. If implemented incorrectly (without the normalization factor), the effective learning rate drops, which masquerades as underfitting. The reference implementation is the safest starting point. Also, C-AdamW has been validated on much larger models and longer runs; benefit on a 50-epoch ~35M param model may be modest.

**References:**
- Liang et al., "Cautious Optimizers: Improving Training with One Line of Code," ICLR 2026. https://arxiv.org/abs/2411.16085
- Reference implementation: https://github.com/kyleliang919/C-Optim

---

## Idea 4: Separate Surface Decoder Head

**Title slug:** separate-surface-decoder-head

**Predicted delta on `val_avg/mae_surf_p`:** -5% to -12%

**Why it should work:**
The primary ranking metric is `mae_surf_p` — pressure MAE only on surface nodes. The current architecture uses a single shared output MLP for all nodes (surface + volume), so the model must balance competing objectives: accurate surface pressure and accurate volume velocity. Surface nodes are a small fraction of total nodes (~1-2% by count, since the mesh has ~100K+ volume nodes and only the foil boundary is surface). Even with 10x surface loss weighting, the shared decoder must compress both regimes through the same final projection. A dedicated surface decoder head — a separate MLP that reads the same transformer features but produces the surface prediction independently — allows the model to specialize its surface prediction capacity and can be trained with a higher effective weight on the metric that matters. This is standard in multi-task networks (e.g., panoptic segmentation with separate heads for semantic and instance) and consistently improves the higher-priority task.

**Implementation in `train.py`:**

1. Modify `TransolverBlock` (or `Transolver`) to produce a separate surface head output on the final layer:
   ```python
   # In Transolver.__init__:
   self.surface_head = MLP(n_hidden, n_hidden, out_dim, n_layers=1, res=False)

   # In Transolver.forward, after the last block:
   vol_preds = fx  # [B, N, 3] from existing head
   surf_preds = self.surface_head(fx)  # [B, N, 3] separate prediction
   return {"preds": vol_preds, "surf_preds": surf_preds}
   ```

2. In the training loop, use `surf_preds` for the surface loss and `preds` for the volume loss:
   ```python
   preds = model({"x": x_norm})
   vol_pred = preds["preds"]
   surf_pred = preds.get("surf_preds", vol_pred)  # fallback for compat

   sq_err_vol = (vol_pred - y_norm) ** 2
   sq_err_surf = (surf_pred - y_norm) ** 2

   vol_loss = (sq_err_vol * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
   surf_loss = (sq_err_surf * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
   loss = vol_loss + cfg.surf_weight * surf_loss
   ```

3. For metrics/evaluation: denormalize `surf_pred` (not `vol_pred`) for surface nodes when computing `mae_surf_p`. Update `evaluate_split` to pass through the surface prediction.

4. For final output: combine `vol_pred` for volume nodes and `surf_pred` for surface nodes before scoring, or keep them separate and only use surface head output for surface metrics.

**Risk / failure mode:**
If the shared features from the transformer body are already well-suited for surface prediction, the separate head may overfit the training surface distribution without improving OOD generalization. The key question is whether the OOD failure comes from the shared representation or from the shared output head — this experiment tests the latter. Also, this changes the scoring path in `evaluate_split`, so verify mask logic carefully.

**References:**
- He et al., "Mask R-CNN," ICCV 2017. https://arxiv.org/abs/1703.06870 (separate heads for segmentation/box)
- Li et al., "GeoTransolver: Geometry-Aware Transformers for PDE Solving," 2024. https://arxiv.org/abs/2512.20399 (geometry-BC-aware attention for surface accuracy)
- Bonnet et al., "AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions," NeurIPS 2022. https://arxiv.org/abs/2212.07569 (surface vs. volume accuracy tradeoffs in aerodynamic surrogates)

---

## Idea 5: Per-Sample Instance Normalization of Targets

**Title slug:** per-sample-instance-norm-targets

**Predicted delta on `val_avg/mae_surf_p`:** -6% to -14%

**Why it should work:**
The training data spans three orders of magnitude in velocity/pressure scale — per-sample y std varies from ~50 (low-Re cruise) to ~2000 (high-Re raceCar). The global normalization `(y - y_mean) / y_std` using dataset-level stats shrinks high-Re samples to a moderate range and leaves low-Re samples extremely small in normalized space. The mean squared error loss then effectively ignores low-Re samples because their normalized residuals are tiny. Per-sample instance normalization — subtracting the per-sample mean and dividing by the per-sample std before computing loss — places every sample on equal footing in the loss landscape regardless of Re. At inference, the model predicts in the per-sample-normalized space, and we recover the physical prediction by storing the per-sample mean/std and rescaling. This is equivalent to "predicting the shape of the field" rather than its absolute scale, which is a much easier and more transferable regression target. This should particularly help the Re-rand split.

**Implementation in `train.py`:**

1. In the training loop and evaluation, after unpacking `(x, y, is_surface, mask)`:
   ```python
   # Per-sample normalization of targets
   # Compute stats over real (unmasked) nodes only
   # y: [B, N, 3], mask: [B, N]
   mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
   y_s_sum = (y * mask_f).sum(dim=1, keepdim=True)   # [B, 1, 3]
   y_s_count = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
   y_s_mean = y_s_sum / y_s_count  # [B, 1, 3]
   y_s_var = ((y - y_s_mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / y_s_count
   y_s_std = y_s_var.sqrt().clamp(min=1e-6)  # [B, 1, 3]

   # Normalize y for loss computation (in ADDITION to global normalization)
   y_norm_global = (y - stats["y_mean"]) / stats["y_std"]
   y_norm_instance = (y_norm_global - y_s_mean_norm) / y_s_std_norm
   # where y_s_mean_norm, y_s_std_norm are the per-sample stats of y_norm_global
   ```

2. Model still predicts in the instance-normalized space. For evaluation/MAE, undo the per-sample normalization and then the global normalization to recover physical units.

3. Simpler alternative: just divide the loss for each sample by that sample's y_std (loss reweighting rather than input normalization), which avoids changing the model output contract:
   ```python
   # Compute per-sample normalization factor
   y_s_scale = y_s_std.squeeze(1).mean(dim=-1)  # [B] scalar per sample
   # Weight each sample's loss inversely by its scale
   loss_per_sample = loss / y_s_scale.detach()
   loss = loss_per_sample.mean()
   ```

   This "scale-invariant loss" variant is cleaner and doesn't require changing the evaluation path. Recommend starting with this.

**Risk / failure mode:**
Instance normalization of targets breaks the global normalization contract that `data/scoring.py` expects. The simpler scale-invariant loss reweighting approach avoids this. However, if the per-sample std is dominated by a few extreme nodes (e.g., leading edge stagnation point), the normalization may over-inflate the importance of smooth low-Re samples. Check that per-sample std is computed over the full field, not just the surface. This idea requires careful bookkeeping in evaluation.

**References:**
- McClenny & Braga-Neto, "Self-Adaptive Physics-Informed Neural Networks," J. Comput. Phys. 2023. https://arxiv.org/abs/2203.07557 (adaptive per-sample loss balancing)
- Snoek et al., "Input Warping for Bayesian Optimization of Non-stationary Functions," ICML 2014. (instance normalization for heteroscedastic targets)
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR 2021. https://arxiv.org/abs/2010.08895 (discusses normalization choices for multi-regime PDE problems)

---

## Idea 6: Geometry Interpolation Augmentation via NACA Mixing

**Title slug:** naca-camber-geometry-augmentation

**Predicted delta on `val_avg/mae_surf_p`:** -5% to -12% on unseen-camber splits

**Why it should work:**
Files 2 (M=6-8 raceCar) and 5 (M=2-4 cruise) are the two fully held-out geometry splits with zero training samples. The model must interpolate between camber values it has seen (M=2-5 and M=9 for raceCar; M=0-2 and M=4-6 for cruise). A simple data augmentation that linearly interpolates the NACA feature dims (15-17 for foil 1) between two training samples with different camber values creates synthetic in-distribution-like data for the gap region. Since the NACA params enter as continuous 24D features (the model never sees raw geometry), we can interpolate in feature space: pick two training samples a and b, sample alpha ~ U[0,1], create x_aug = alpha * x_a + (1-alpha) * x_b and y_aug = alpha * y_a + (1-alpha) * y_b. This is a form of Mixup restricted to the camber dimension, and it directly teaches the model to extrapolate/interpolate across camber values.

**Implementation in `train.py`:**

1. Add a `CamberMixup` augmentation class that operates on batches:
   ```python
   class CamberMixup:
       def __init__(self, alpha=0.4, camber_dims=(15, 16, 17), prob=0.5):
           self.alpha = alpha
           self.camber_dims = camber_dims
           self.prob = prob

       def __call__(self, x, y, mask):
           # x: [B, N, 24], y: [B, N, 3], mask: [B, N]
           if random.random() > self.prob:
               return x, y
           B = x.shape[0]
           lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
           perm = torch.randperm(B)
           x_mix = lam * x + (1 - lam) * x[perm]
           y_mix = lam * y + (1 - lam) * y[perm]
           # Only mix along camber dims (dims 15-17) and flow conditions
           # to avoid mixing incompatible mesh geometries
           # Safer: only mix the flow condition + NACA features, keep node positions fixed
           x_out = x.clone()
           x_out[:, :, 13:] = lam * x[:, :, 13:] + (1 - lam) * x[perm, :, 13:]
           return x_out, y_mix, mask
   ```

2. Safer variant: rather than full Mixup (which mixes incompatible meshes), only interpolate the GLOBAL condition features (dims 13-23, the ones that are constant across nodes for a sample), and leave the node-local features (dims 0-12) unchanged. Then interpolate y proportionally. Since x dims 0-12 (node position, arc-length, DSDF, is_surface) are mesh-local and cannot be meaningfully mixed, this is the principled approach.

3. Apply in the training loop after collation:
   ```python
   if augmentor is not None:
       x, y, mask = augmentor(x, y, mask)
   ```

4. Only apply during training, not validation/test. Set alpha=0.4, prob=0.5.

**Risk / failure mode:**
The fundamental issue is that y_mix = lam * y_a + (1-lam) * y_b is only physically correct if the two samples have compatible mesh topologies. Since meshes are padded to the same size per batch but differ in node count, the interpolation may be meaningless for padding positions. This risk is manageable by only interpolating the flow condition features (dims 13-23) and checking that both samples' real node masks agree (use the intersection of the two masks as the effective mask for the mixed sample). The supervision signal for the mixed sample is soft but physically plausible for interpolated NACA values.

**References:**
- Zhang et al., "Mixup: Beyond Empirical Risk Minimization," ICLR 2018. https://arxiv.org/abs/1710.09412
- Guo et al., "Generalizing to Unseen Geometries in Neural PDE Surrogates," ICLR 2024 Workshop. (geometry interpolation for OOD generalization in PDE surrogates)
- Allen-Zhu & Li, "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning," ICLR 2023. https://arxiv.org/abs/2012.09816 (interpolation-based augmentation theory)

---

## Idea 7: Larger Transolver (n_hidden=256, n_layers=8)

**Title slug:** transolver-capacity-scale-up

**Predicted delta on `val_avg/mae_surf_p`:** -5% to -10%

**Why it should work:**
The current model uses `n_hidden=128, n_layers=5, n_head=4` (~3-5M parameters). The Transolver paper and follow-up works consistently show that increasing width and depth improves accuracy on PDE surrogate benchmarks, and the dataset (1499 training samples, 74K-242K nodes each) is large enough to support a significantly larger model. Doubling to `n_hidden=256, n_layers=8` (~15-20M params) keeps the model well within GPU VRAM limits (the attention bottleneck is the slice dimension, fixed at 64, so memory scales with n_hidden, not N). This is not a blind capacity increase — it targets the representation bottleneck: the unseen-camber and Re-rand generalization failures are plausibly caused by insufficient capacity to encode the joint geometry-condition manifold across the full training distribution.

**Implementation in `train.py`:**

Change the `model_config` dict:
```python
model_config = dict(
    space_dim=2, fun_dim=X_DIM-2,
    out_dim=3, n_hidden=256, n_layers=8, n_head=8,  # was 128/5/4
    slice_num=64, mlp_ratio=2,
    output_fields=["Ux","Uy","p"], output_dims=[1,1,1],
)
```

Also lower the learning rate slightly to account for larger model: `lr=3e-4` (from 5e-4).

VRAM check: at batch_size=4, N_max~242K nodes, n_hidden=256, the attention is on 64 slice tokens per head (8 heads) — `64x64x8` SDPA = trivial. The main memory cost is the node features `[4, 242K, 256]` = ~1GB fp32 per batch, well within 96GB.

**Risk / failure mode:**
A larger model may overfit the raceCar single in-dist split (599 training samples) while improving the tandem splits. Watch the single-foil val metric. Also, if training is time-limited, a larger model may not converge fully in the allocated wall-clock. Consider also trying `n_hidden=256, n_layers=6` as an intermediate point.

**References:**
- Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General Geometries," ICML 2024. https://arxiv.org/abs/2402.02366 (Table 3 shows consistent improvement with depth)
- Yang et al., "GNOT: A General Neural Operator Transformer for Operator Learning," ICML 2023. https://arxiv.org/abs/2302.14376 (scaling analysis for neural operators)
- Herde et al., "Poseidon: Efficient Foundation Models for PDEs," NeurIPS 2024. https://arxiv.org/abs/2408.02810 (foundation model-scale PDE transformer)

---

## Idea 8: Multi-Scale Slice Attention (Hierarchical Slice Counts)

**Title slug:** multi-scale-slice-attention

**Predicted delta on `val_avg/mae_surf_p`:** -6% to -12%

**Why it should work:**
The current Transolver uses a fixed `slice_num=64` in every attention block — each block collapses all nodes into 64 learned physics slices, runs attention on those 64 tokens, and scatters back. This is a single-resolution compression. In fluid dynamics, the relevant scales vary dramatically: the near-wall boundary layer has features at scales of millimeters, while the wake and freestream vary over meters. Using different slice counts at different layers — e.g., coarse (32 slices) → fine (128 slices) → coarse (64 slices) — allows early blocks to capture global flow topology and later blocks to refine local gradients. This is the spatial-hierarchy principle from U-Net and multi-scale GNNs, applied to the Transolver slice mechanism without requiring a true hierarchical graph or mesh.

**Implementation in `train.py`:**

1. Allow per-layer `slice_num` configuration:
   ```python
   # In PhysicsAttention.__init__, slice_num is already a parameter
   # Modify TransolverBlock and Transolver to accept per-layer slice_num
   ```

2. Change `model_config` to pass a list of slice counts:
   ```python
   # For n_layers=5, try: [32, 64, 128, 64, 32] (hourglass) or [64, 64, 128, 128, 64] (expanding)
   model_config = dict(
       ...,
       slice_num=[32, 64, 128, 64, 32],  # per-layer; modify Transolver to accept list
       ...
   )
   ```

3. In `Transolver.__init__`, when `slice_num` is a list:
   ```python
   if isinstance(slice_num, int):
       slice_nums = [slice_num] * n_layers
   else:
       slice_nums = slice_num
   self.blocks = nn.ModuleList([
       TransolverBlock(n_hidden, n_head, slice_nums[i], ...)
       for i in range(n_layers)
   ])
   ```

4. Note: the `in_project_slice` weight shapes will differ per block, which is expected. The shared `preprocess` MLP output flows into all blocks independently.

5. Recommended starting point: `[32, 64, 128, 64, 32]` for the default 5-layer model, or `[64, 128, 256, 128, 64, 64, 64, 64]` for the 8-layer model from Idea 7. This can be combined with the capacity scale-up.

**Risk / failure mode:**
Fewer slices in early blocks may lose fine-grained surface information before it can be processed. The hourglass pattern helps here but is not guaranteed to converge to better solutions. Also, the `orthogonal_` initialization of `in_project_slice` may need adjustment for smaller slice counts (fewer orthogonal vectors available). If results are worse, try the expanding pattern `[32, 64, 128, 128]` instead, which is closer to coarse-to-fine and more physically motivated.

**References:**
- Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General Geometries," ICML 2024. https://arxiv.org/abs/2402.02366 (slice mechanism reference)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015. https://arxiv.org/abs/1505.04597 (multi-scale hierarchy principle)
- Cao et al., "Choose a Transformer: Fourier or Galerkin," NeurIPS 2021. https://arxiv.org/abs/2105.14995 (multi-scale attention analysis for PDE operators)

---

## Idea 9: Log-Transform Pressure Target

**Title slug:** log-transform-pressure-target

**Predicted delta on `val_avg/mae_surf_p`:** -4% to -10%

**Why it should work:**
The pressure field `p` (kinematic pressure, m²/s²) varies by ~4 orders of magnitude across the dataset (range -29K to +2.7K, with per-sample std up to 2077). Velocity components `Ux, Uy` are similarly large for high-Re cases. The global Z-normalization compresses this dynamic range but does not remove the fundamental issue: the MSE loss is dominated by high-Re samples with extreme pressure gradients, leaving the model undertrained on low-Re and cruise cases. A log-space (or sinh-space) transform of the pressure target before the global normalization creates a more uniform loss landscape. Specifically, `p_log = sign(p) * log(1 + |p|)` (the signed log transform) maps the full range to a compact interval while preserving sign and physical ordering, without destroying the near-zero pressure region. This is standard practice in atmospheric modeling and financial time-series regression.

**Implementation in `train.py`:**

1. Define the forward and inverse transforms:
   ```python
   def signed_log(x, eps=1.0):
       return torch.sign(x) * torch.log1p(torch.abs(x) / eps)

   def signed_log_inv(x, eps=1.0):
       return torch.sign(x) * (torch.expm1(torch.abs(x)) * eps)
   ```

2. Apply to the pressure channel (dim 2 of y) in the training loop after loading, before global normalization:
   ```python
   y_transformed = y.clone()
   y_transformed[..., 2] = signed_log(y[..., 2])
   # Then apply global normalization as usual (but recompute stats on transformed space if needed)
   ```

3. The cleanest approach: apply the transform as a preprocessing step and use modified stats (computed on the transformed training targets). Since we cannot modify `data/`, we re-derive transformed stats in `train.py` from a pass through the training data at startup.

4. At evaluation time, apply the inverse transform to convert predictions back to physical pressure before MAE computation. Update `evaluate_split` to include the inverse transform for the pressure channel.

5. Alternative simpler approach: use a smooth Huber-like loss on the pressure channel instead of MSE, which reduces sensitivity to extreme values without requiring a target transform. This requires no stats recomputation.

**Risk / failure mode:**
The global `stats.json` (y_mean, y_std) was computed on the untransformed targets, so the model output contract and `data/scoring.py` both expect predictions in the original normalized space. Applying a target transform requires careful bookkeeping: the scoring path must inverse-transform the pressure dimension before physical MAE is computed. This is doable but risky to get wrong. The Huber alternative is much safer and worth trying first.

**References:**
- Rasp & Thuerey, "Data-Driven Medium-Range Weather Prediction with a Resnet Pretrained on Climate Simulations," J. Advances in Modeling Earth Systems 2021. https://arxiv.org/abs/2008.08626 (discusses log-transform for atmospheric field prediction)
- Wang et al., "On the Eigenvector Bias of Fourier Feature Networks," CVPR 2021. https://arxiv.org/abs/2006.10739 (spectral/scale analysis in neural PDE approximators)
- Chen et al., "HiGNN: Hierarchical Informative Graph Neural Networks," TNNLS 2024. (scale-aware predictions for irregular physical fields)

---

## Idea 10: Weighted Huber Loss (Robust to Extreme Re)

**Title slug:** huber-loss-pressure-weighted

**Predicted delta on `val_avg/mae_surf_p`:** -4% to -8%

**Why it should work:**
The current MSE loss (`(pred - y_norm)^2`) is dominated by the squared residual, which means a single node with a large error (common at leading/trailing edges for high-Re cases) contributes quadratically more than 100 nodes with small errors. Huber loss (`L_delta`) behaves like MSE for small errors and like MAE for large ones, effectively down-weighting outlier nodes. This is directly aligned with the evaluation metric, which is MAE (not MSE), so reducing the train-eval metric gap by switching from MSE-dominated to MAE-closer loss should help. Additionally, increase the surface weight from 10 to 20 or 25 to further focus capacity on the primary metric. The combination of Huber loss + higher surface weight is a single coherent change targeting the train-to-metric alignment problem.

**Implementation in `train.py`:**

1. Replace the squared error with Huber:
   ```python
   # Replace: sq_err = (pred - y_norm) ** 2
   delta = 1.0  # Huber threshold in normalized space; tune if needed
   abs_err = (pred - y_norm).abs()
   huber_err = torch.where(abs_err < delta, 0.5 * abs_err ** 2, delta * (abs_err - 0.5 * delta))
   # Use huber_err everywhere sq_err was used
   ```

2. Simultaneously raise the surface weight:
   ```python
   cfg.surf_weight = 20.0  # was 10.0
   ```

3. The Huber delta in normalized space needs calibration. A value of `delta=1.0` in the z-score normalized space corresponds to "errors larger than 1 std are treated as outliers." Given that high-Re samples can have normalized errors of 3-5 sigma, this is appropriate.

4. Alternative: smooth L1 loss (PyTorch's `F.smooth_l1_loss`) with `beta=0.5` is equivalent to Huber with delta=0.5 in PyTorch's convention.

**Risk / failure mode:**
Huber loss is more conservative near zero than MSE, which may slow convergence for the many nodes with small errors (volume interior). If vol_loss increases significantly while surf_loss decreases, the model may be underfitting the volume field. Monitor vol vs. surf losses separately. The surf_weight increase may also need to be tuned jointly with the Huber delta — start with surf_weight=15 rather than 20 to avoid collapse.

**References:**
- Huber, "Robust Estimation of a Location Parameter," Annals of Statistics 1964. (original Huber loss)
- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," Int. J. Forecasting 2021. https://arxiv.org/abs/1912.09363 (robust loss for regression with extreme values)
- Li et al., "Physics-Informed Neural Networks for High-Reynolds-Number Flows," J. Comput. Phys. 2022. (loss formulation for high-Re CFD surrogates)

---

## Idea 11: Surface-Aware Positional Encoding via Signed Arc-Length

**Title slug:** surface-arclen-positional-encoding

**Predicted delta on `val_avg/mae_surf_p`:** -4% to -9%

**Why it should work:**
The input already contains signed arc-length features (dims 2-3), but these are mixed flat into the 24D input and projected through the same MLP as all other features. There is a principled reason to treat surface arc-length as a positional encoding rather than a raw feature: on the foil surface, the flow solution is predominantly determined by the arc-length-parameterized boundary layer, and attention should be aware of "where on the foil" a node sits relative to others. Adding a sinusoidal or learned Fourier positional encoding derived from the arc-length (similar to how sequence Transformers use position embeddings) gives the surface nodes an explicit relational structure. The key difference from the current setup: the PE is additive at the embedding level (before the PhysicsAttention), ensuring the arc-length signal is injected into the query/key space of the slice assignment, not just the value space.

**Implementation in `train.py`:**

1. Add a `SurfacePositionalEncoding` module:
   ```python
   class SurfacePositionalEncoding(nn.Module):
       def __init__(self, n_hidden, n_freqs=16):
           super().__init__()
           # Learnable frequencies for arc-length encoding
           self.freqs = nn.Parameter(torch.randn(n_freqs) * 0.01)
           self.linear = nn.Linear(2 * n_freqs + 2, n_hidden)  # +2 for raw saf

       def forward(self, saf, is_surface):
           # saf: [B, N, 2] — signed arc-length dims 2-3
           # is_surface: [B, N] bool
           freqs = self.freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, n_freqs]
           saf_expanded = saf.unsqueeze(-1)  # [B, N, 2, 1]
           pe_sin = torch.sin(saf_expanded * freqs)  # [B, N, 2, n_freqs]
           pe_cos = torch.cos(saf_expanded * freqs)  # [B, N, 2, n_freqs]
           pe = torch.cat([pe_sin, pe_cos, saf], dim=-1).view(saf.shape[0], saf.shape[1], -1)  # [B, N, 4*n_freqs+2]
           pe = self.linear(pe)  # [B, N, n_hidden]
           # Only apply to surface nodes; zero for volume
           pe = pe * is_surface.unsqueeze(-1).float()
           return pe
   ```

2. In `Transolver.forward`, add the surface PE to `fx` after `preprocess`:
   ```python
   saf = x[:, :, 2:4]  # raw (before normalization, or use normalized saf)
   is_surface = x[:, :, 12].bool()
   fx = self.preprocess(x) + self.placeholder[None, None, :]
   fx = fx + self.surface_pe(saf, is_surface)  # additive surface encoding
   ```

3. Keep volume nodes' PE contribution zero — this is a targeted injection that only enriches surface node representations.

**Risk / failure mode:**
Arc-length features (dims 2-3) are already in the input, so this encoding is partially redundant. The benefit comes from injecting the PE in frequency space (allowing the model to learn resonances of the boundary layer) rather than as a raw linear feature. If the preprocessing MLP has already learned to extract the relevant frequencies, this will have no effect. Check whether saf dims are redundant with position dims 0-1 — they should be complementary (saf measures boundary-following distance, position measures Euclidean).

**References:**
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017. https://arxiv.org/abs/1706.03762 (sinusoidal positional encoding)
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions," NeurIPS 2020. https://arxiv.org/abs/2006.10739 (Fourier feature encoding for coordinate networks)
- Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D PDEs," NeurIPS 2023. https://arxiv.org/abs/2309.00583 (geometry-aware encoding for mesh-based PDE surrogates)

---

## Idea 12: Warmup + Longer Cosine Annealing Schedule

**Title slug:** lr-warmup-long-cosine-schedule

**Predicted delta on `val_avg/mae_surf_p`:** -2% to -5%

**Why it should work:**
The current optimizer uses `CosineAnnealingLR(T_max=MAX_EPOCHS)` with no warmup, starting at lr=5e-4. For transformer models on irregular mesh data, a linear warmup period (5-10% of total epochs) prevents early instability from large initial gradient steps through the scatter/gather einsum operations in PhysicsAttention. Furthermore, `T_max=MAX_EPOCHS` means the cosine schedule bottoms out exactly at the last epoch — but if training is cut by the timeout before `MAX_EPOCHS`, the schedule is in the middle of its descent and the model is trained with a suboptimal lr trajectory. Setting `T_max = MAX_EPOCHS * 1.5` (or using a one-cycle schedule) ensures the model always sees the low-lr fine-tuning phase regardless of actual training duration. Additionally, a 5-epoch linear warmup provides stability for the larger models proposed in Ideas 1, 4, and 7.

**Implementation in `train.py`:**

```python
# Replace:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# With warmup + cosine:
WARMUP_EPOCHS = max(3, MAX_EPOCHS // 10)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS],
)
```

This is a 2-line change beyond the original scheduler. `LinearLR` and `SequentialLR` are in `torch.optim.lr_scheduler` (no new packages).

Also, consider adding gradient clipping `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()` for early-training stability with the warmup.

**Risk / failure mode:**
Warmup delays convergence on short runs. If the wall-clock budget is tight and the model is well-behaved with the current scheduler, this may waste the first few epochs on suboptimal lr. Risk is low but the gain may also be small — treat this as a "polish" experiment rather than a bold bet. Best combined with a larger model (Idea 7) where warmup is more important.

**References:**
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017. https://arxiv.org/abs/1706.02677 (linear warmup for large-scale training)
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017. https://arxiv.org/abs/1608.03983 (cosine annealing reference)
- Liu et al., "Transformers as Gaussian Processes," ICLR 2023. https://arxiv.org/abs/2302.10009 (warmup importance for attention-based models)

---

## Priority Ranking for Assignment

Higher priority = larger expected delta on `val_avg/mae_surf_p`, targets all four splits, especially OOD ones.

| Rank | Slug | Expected delta | Risk | Best for |
|------|------|----------------|------|----------|
| 1 | film-flow-condition-every-block | -8 to -15% | Medium | Re-rand + unseen-camber |
| 2 | per-sample-instance-norm-targets | -6 to -14% | Medium-high | Re-rand |
| 3 | separate-surface-decoder-head | -5 to -12% | Low-medium | All splits (surf_p focus) |
| 4 | naca-camber-geometry-augmentation | -5 to -12% | Medium | Unseen-camber only |
| 5 | multi-scale-slice-attention | -6 to -12% | Medium | All splits |
| 6 | transolver-capacity-scale-up | -5 to -10% | Low | All splits |
| 7 | log-transform-pressure-target | -4 to -10% | Medium-high | Re-rand |
| 8 | surface-arclen-positional-encoding | -4 to -9% | Low | Surf-p focus |
| 9 | huber-loss-pressure-weighted | -4 to -8% | Low | All splits |
| 10 | ema-weights-checkpoint | -3 to -7% | Very low | OOD splits |
| 11 | cautious-adamw-optimizer | -2 to -5% | Very low | All splits |
| 12 | lr-warmup-long-cosine-schedule | -2 to -5% | Very low | All splits (polish) |
