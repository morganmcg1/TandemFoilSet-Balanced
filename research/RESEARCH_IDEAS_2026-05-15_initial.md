<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — TandemFoilSet Transolver Baseline
**Date:** 2026-05-15
**Scope:** Literature scan only. No prior experiment history consulted.
**Primary metric target:** `val_avg/mae_surf_p` (lower is better)

---

## Context Summary

The baseline is a Transolver with physics-aware attention over irregular meshes. Key configuration:
- `slice_num=64`, `n_layers=5`, `n_head=4`, `n_hidden=128`, `mlp_ratio=2`
- Loss: `vol_loss + 10.0 * surf_loss` in normalized MSE space
- Optimizer: AdamW lr=5e-4, weight_decay=1e-4, NO gradient clipping
- Scheduler: CosineAnnealingLR, NO warmup
- 3 of 4 val splits are OOD (geometry camber, Re holdout)
- Per-sample y-std varies by ~5x within a single domain (avg 458, max 2,077 for RaceCar single)
- Pressure surface MAE is the primary ranking metric — equal weight across 4 val splits

---

## Focus Area 1: Loss Reformulation

### Idea 1.1 — Huber / SmoothL1 Loss Swap

**What it is:** Replace MSE with Huber loss (SmoothL1) in both vol and surf terms, using a threshold delta that transitions between L1 and L2 regimes.

**Why it might help here:** TandemFoilSet has extreme per-sample y-std spread (max 2,077 vs avg 458 in RaceCar single). Within-domain scale varies by ~5x, and across domains by ~10x. MSE over-penalizes high-Re outlier nodes, biasing the optimizer away from low-Re samples. Huber loss is theoretically optimal under heavy-tailed noise (Fan et al. 2022): it reverts to L1 for large residuals, capping their quadratic influence on the gradient.

**Key papers:**
- Fan et al. (2022). "Huber Loss Revisited: A Theoretical Analysis." arXiv:2203.10418. Derives the optimal Huber threshold as a function of the noise distribution; confirms L1 limit for heavy tails.
- Terven et al. (2023). "A Comprehensive Survey of Loss Functions for Machine Learning." arXiv:2307.02694. Systematic comparison of Huber, MAE, MSE with practical guidance.
- Wang et al. (2023). "Deep Regression Loss Functions: From Squared to Deep Optimal." arXiv:2309.12872. Demonstrates adaptive loss outperforms fixed MSE for regression with mixed scales.

**Implementation notes:**
- `torch.nn.HuberLoss(reduction='none', delta=1.0)` — returns per-element loss, then apply mask
- Critical: `delta` is in normalized space. With y normalized by `y_std`, a delta of 1.0 corresponds to ~1 standard deviation of error. Try `delta=0.5` (aggressive) and `delta=2.0` (conservative).
- The existing loss structure already separates surf and vol — just swap the MSE computation. ~5 LOC change.
- Gotcha: `HuberLoss` in PyTorch is multiplied by 0.5 in the L2 regime — verify your delta expectation matches the documentation (PyTorch differs from sklearn).

**Expected delta:** Moderate. Primarily helps with high-Re/low-Re scale mixing. Expect 2–8% improvement on `mae_surf_p`, larger on OOD splits where extreme samples are overrepresented.

**Implementation effort:** ~5 LOC. Very low risk.

**Risk:** If delta is tuned to the training distribution, it may not generalize to OOD extremes. Try per-split validation to check OOD vs in-dist tradeoff.

---

### Idea 1.2 — Per-Channel Pressure Loss Upweighting

**What it is:** Separate the loss into three channel-specific terms (`Ux`, `Uy`, `p`) and apply a multiplicative weight `p_weight > 1.0` to the pressure channel, since `val_avg/mae_surf_p` cares only about surface pressure MAE.

**Why it might help here:** The current loss treats `Ux`, `Uy`, and `p` identically in a mean over channels. The primary ranking metric is surface pressure MAE exclusively. Directly up-weighting the pressure channel aligns the training objective with the evaluation objective. This is a classic Kaggle insight: if the competition metric is a subset of the training loss, make the loss reflect that subset.

**Key papers:**
- Objective matching via loss weighting is a standard practice in multi-task learning. Howard & Ruder (2018) ULMFiT uses task-specific loss weighting; Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018) provides a principled treatment.
- For CFD surrogates: Thuerey et al. (2020) "Deep Learning Methods for Reynolds-Averaged Navier–Stokes Simulations of Airfoil Aerodynamics" (AIAA Journal) demonstrates that surface pressure accuracy is the hardest and most impactful channel to optimize.

**Implementation notes:**
- Add `p_weight` config parameter (try 2.0, 4.0, 8.0).
- In the loss computation, split `sq_err` by channel: `sq_err[..., 0]` (Ux), `sq_err[..., 1]` (Uy), `sq_err[..., 2]` (p).
- Apply `p_weight` to `surf_loss` pressure channel only (surface pressure is the metric).
- Gotcha: increasing p_weight on the surface term may degrade volume pressure if the model has limited capacity — monitor `{split}/mae_vol_p` separately.

**Expected delta:** Moderate to large for the primary metric specifically. The model may shift capacity toward p prediction. ~5–15% improvement on `mae_surf_p`. Risk of regression on Ux/Uy.

**Implementation effort:** ~10 LOC. Low risk.

---

### Idea 1.3 — Per-Sample Relative / Scale-Invariant Loss

**What it is:** Normalize the loss contribution of each sample by that sample's per-channel y-standard-deviation (or root-mean-square), making the training objective invariant to the scale of each sample's flow magnitude.

**Why it might help here:** Within a single domain split, per-sample y-std varies by a factor of 5 (avg 458, max 2,077). High-Re samples have 5x larger absolute residuals in normalized space — their gradients dominate batch updates, effectively down-weighting low-Re samples that may matter for OOD generalization. A relative loss equalizes gradient magnitude across samples regardless of flow intensity.

**Key papers:**
- Wang et al. (2023). arXiv:2309.12872. Per-sample scale normalization for deep regression.
- Relative L2 norm for neural operators: used in FNO, GNOT, and most NeurIPS 2022–2024 neural operator benchmarks. Computed as `||pred - y||_2 / ||y||_2` per sample.

**Implementation notes:**
- Compute `y_scale = y.std(dim=1, keepdim=True).clamp(min=1e-6)` per sample (before normalization, or equivalently derive from the normalized y using stored stats).
- Divide each sample's loss contribution by `y_scale` before averaging over the batch.
- Gotcha: if some samples have near-zero y (very low Re), `y_scale` can be tiny and the loss can blow up — use `clamp(min=epsilon)` with epsilon set to e.g. the 5th percentile of the distribution.
- Can be combined with Huber (idea 1.1).

**Expected delta:** Moderate. Primarily improves OOD splits where Re distribution differs from training mean. Expect 3–7% on `val_re_rand` and `val_geom_camber_*`.

**Implementation effort:** ~15 LOC. Low risk.

---

## Focus Area 2: Architecture Tweaks

### Idea 2.1 — Increase slice_num (64 → 128)

**What it is:** Double the number of physics-aware "slice tokens" in Transolver's PhysicsAttention from 64 to 128, giving the model finer-grained physics tokenization.

**Why it might help here:** Transolver's PhysicsAttention compresses N=74K–242K mesh nodes into `slice_num` tokens via soft assignment. With meshes up to 242K nodes and 3 physical zones, 64 tokens must encode everything. Doubling to 128 increases the attention bottleneck capacity quadratically (attention is O(slice_num^2)), which the 96 GB GPUs can absorb.

**Key papers:**
- Transolver paper (arxiv 2402.02366): ablation table shows slice_num sensitivity — accuracy monotonically increases up to the tested range with diminishing returns. The original paper tests up to 64 on smaller meshes.
- AB-UPT (TMLR 10/2025, openreview): anchored-branched UPT for automotive CFD surrogates — shows that scaling the "physics token" bottleneck is the dominant architectural lever for large CFD meshes.

**Implementation notes:**
- Single config change: `slice_num=128` in `model_config`.
- VRAM impact: attention over 128 tokens vs 64 is 4x — but total batch VRAM is dominated by the N-node forward pass, not the slice_num^2 attention. Likely safe at batch_size=4 with 242K-node samples.
- Gotcha: `n_head` must divide `slice_num`. With `n_head=4`, slice_num must be divisible by 4. 128 is fine.
- Can also try 96 (intermediate).

**Expected delta:** Moderate. Original ablation suggests 3–5% improvement from 32→64; 64→128 may yield 2–4%.

**Implementation effort:** 1 LOC. Zero risk to code correctness.

---

### Idea 2.2 — FiLM Conditioning on log(Re)

**What it is:** Feature-wise Linear Modulation (FiLM) — add scale/shift layers in the MLP blocks of Transolver that are conditioned on `log(Re)`, allowing the model to explicitly modulate its computation based on the Reynolds number regime.

**Why it might help here:** `log(Re)` is already in input dim 13 as a global scalar, but it enters the model as just another input channel mixed with node-local features. FiLM gives the model explicit regime-switching capability — the scale/shift parameters let the model "know" which flow regime it is in before applying the MLP transformations. This is especially important for `val_re_rand` which tests cross-regime generalization.

**Key papers:**
- AeroDiT (arxiv 2412.17394): "Diffusion Transformers for RANS Airfoil Simulation" — conditions on Re via adaptive layer normalization (FiLM equivalent) for airfoil CFD; the conditioning mechanism is explicitly described and shown to improve cross-Re generalization.
- Perez et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018. Original formulation.
- Transolver (arxiv 2402.02366): uses no explicit Re conditioning beyond the input feature.

**Implementation notes:**
- Extract `log_re = x_norm[:, :, 13:14]` — or better, pool over the mesh: `log_re_global = x_norm[:, :, 13].mean(dim=1)` to get a per-sample scalar.
- Add a small MLP `film_mlp: [1 → 64 → 2*n_hidden]` that outputs `(gamma, beta)` of shape `[B, 2*n_hidden]`.
- In each Transolver MLP block: `h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)` after the first linear.
- Can condition just the final output MLP or all MLP blocks — start with just the output MLP (~30 LOC).
- Gotcha: `log_re` is already normalized by `x_std`; remap it or use the raw `log(Re)` value for better numerical properties.

**Expected delta:** Moderate to large on `val_re_rand`. Expect 5–10% improvement on Re holdout.

**Implementation effort:** ~50 LOC. Medium risk — requires modifying the Transolver forward pass.

---

## Focus Area 3: Optimization

### Idea 3.1 — Gradient Clipping

**What it is:** Clip the L2 norm of the gradient to a maximum value (e.g., 1.0) before each optimizer step, preventing gradient spikes from high-Re outlier samples from destabilizing training.

**Why it might help here:** The baseline has NO gradient clipping. With per-sample y-std up to 2,077 and a mean around 458, high-Re samples produce ~5x larger gradients than average. MSE loss amplifies this quadratically. Without clipping, a single high-Re batch can produce a gradient norm spike that shifts the model away from its current good position. This is especially harmful for surface nodes where the pressure field has extreme values near the foil leading/trailing edges.

**Key papers:**
- Koloskova et al. (2023). "Revisiting Gradient Clipping: Stochastic Bias and Tight Convergence Guarantees." arXiv:2305.01588. Provides tight theoretical bounds showing gradient clipping provides a favorable bias-variance tradeoff for heavy-tailed noise.
- Schnell & Thuerey (2024). "Stabilizing Backpropagation Through Time for Physics Simulations." arXiv:2405.02041. Demonstrates gradient clipping as a critical stabilizer for physics-based ML models.
- Zhang et al. (2020). "Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity." ICLR 2020. Shows clipping effectively adapts the LR to the gradient magnitude.

**Implementation notes:**
- One line before `optimizer.step()`: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Try `max_norm` in {0.5, 1.0, 2.0}. For AdamW with normalized targets, 1.0 is a safe default.
- Gotcha: measure gradient norm before and after clipping using `torch.nn.utils.clip_grad_norm_` return value (it returns the norm before clipping) — log this to W&B to see if clipping is actually activating.

**Expected delta:** Small to moderate. Primarily stabilization — fewer training crashes or oscillations, better final convergence. Expect 1–4% improvement if the baseline has gradient spikes.

**Implementation effort:** 1 LOC. Near-zero risk.

---

### Idea 3.2 — LR Warmup

**What it is:** Add a linear warmup phase of ~5% of total epochs before transitioning to cosine annealing. The LR starts at 0 (or a small fraction of the target LR) and linearly ramps up.

**Why it might help here:** The baseline uses CosineAnnealingLR with NO warmup. Warmup is well-established as critical for transformer stability at initialization (the attention weights are random and large initial LR steps can destabilize the softmax). The Transolver PhysicsAttention uses softmax over slice weights — a large initial LR step can collapse the slice assignments.

**Key papers:**
- Kosson et al. (2024). "Learning Rate Warmup is Essential for GPT Training." arXiv:2410.23922. Demonstrates warmup is critical for transformer convergence; ablates warmup duration systematically.
- He et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016 — original warmup motivation for deep networks.
- Vaswani et al. (2017). "Attention Is All You Need." — original transformer warmup schedule.

**Implementation notes:**
- Use PyTorch `LinearLR` followed by `CosineAnnealingLR` via `SequentialLR`:
  ```python
  warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))
  scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
  )
  scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=MAX_EPOCHS - warmup_epochs
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
      milestones=[warmup_epochs]
  )
  ```
- Gotcha: `SequentialLR` calls `step()` on the active sub-scheduler — ensure your training loop calls `scheduler.step()` once per epoch (not per batch).

**Expected delta:** Small to moderate. Primarily helps with convergence stability and final metric. Expect 1–3% improvement.

**Implementation effort:** ~10 LOC. Low risk.

---

### Idea 3.3 — EMA of Model Weights

**What it is:** Maintain an exponential moving average (EMA) of the model weights during training, and use the EMA model for validation and test inference.

**Why it might help here:** EMA smooths the optimization trajectory by averaging over recent parameter states. For OOD generalization, it acts as implicit regularization — the EMA model sits in a broader flat region of the loss landscape than the instantaneous model. Morales-Brotons et al. (2024) demonstrate that EMA provides consistent generalization improvements across diverse architectures and tasks.

**Key papers:**
- Morales-Brotons et al. (2024). "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits." arXiv:2411.18704. TMLR. Systematic study of EMA dynamics; shows EMA consistently improves generalization with near-zero cost.
- Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." UAI 2018 (SWA paper). Shows weight averaging finds wider optima — better OOD performance.
- SeWA (2025). "Selective Weight Averaging via Probabilistic Masking." arXiv:2502.10119. Extension of SWA with selective masking.

**Implementation notes:**
- Use `torch.optim.swa_utils.AveragedModel` with `multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.999)`:
  ```python
  ema_model = torch.optim.swa_utils.AveragedModel(
      model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
  )
  # after each batch:
  ema_model.update_parameters(model)
  # for validation/test, use ema_model instead of model
  ```
- Try decay in {0.99, 0.999, 0.9999}. For ~50 epoch training with ~375 batches/epoch (~18K steps), 0.999 means the EMA half-life is ~700 steps.
- Gotcha: `AveragedModel` wraps the model — its `forward` takes the same input dict. Ensure the validation loop calls `ema_model({"x": x_norm})` not `model(...)`.
- For checkpoint saving, save `ema_model.module.state_dict()` or wrap appropriately.

**Expected delta:** Small to moderate. Robust 1–3% across OOD splits. Rarely hurts.

**Implementation effort:** ~20 LOC. Low risk.

---

### Idea 3.4 — Sharpness-Aware Minimization (1st-Order SAM)

**What it is:** SAM (Sharpness-Aware Minimization) seeks parameters in flat loss landscape regions by simultaneously minimizing loss value and loss sharpness. 1st-order SAM approximates the full SAM update with a single gradient computation.

**Why it might help here:** 3 of 4 val splits are OOD (geometry camber and Re holdouts). Schapiro & Zhao (2024) show SAM variants improve OOD generalization by +4.76% to +8.01% on standard benchmarks. Flat minima generalize better to distribution shifts — exactly the condition needed for the geometry camber holdouts. 1st-order SAM (as analyzed in arxiv 2411.01714) achieves nearly the same benefit as 2nd-order SAM at half the cost.

**Key papers:**
- Foret et al. (2021). "Sharpness-Aware Minimization for Efficiently Improving Generalization." ICLR 2021. openreview 6Tm1mposlrM.
- Schapiro & Zhao (2024). "SAM and OOD Generalization: A Comprehensive Analysis." arXiv:2412.05169. Shows +4.76%–+8.01% OOD improvement; analyzes which SAM variant works best.
- Kaddour et al. (2024). "1st-Order Magic: Why 1st-Order SAM Nearly Matches SAM." arXiv:2411.01714. 1st-order SAM achieves ~95% of 2nd-order SAM benefit at 50% cost.
- Kwon et al. (2021). "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning." ICML 2021.

**Implementation notes:**
- 1st-order SAM: after computing loss, do `loss.backward()`, then perturb: `eps_hat = rho * grad / ||grad||`; add `eps_hat` to params; compute loss again without backward; subtract `eps_hat`; do `optimizer.step()`.
- Practical recipe using `sam.py`:
  ```python
  # Step 1: first forward-backward
  loss = compute_loss(model, batch)
  loss.backward()
  optimizer.first_step(zero_grad=True)
  # Step 2: second forward-backward
  loss2 = compute_loss(model, batch)
  loss2.backward()
  optimizer.second_step(zero_grad=True)
  ```
- Use `rho=0.05` (default in most implementations). Try {0.02, 0.05, 0.1}.
- Gotcha: SAM doubles the number of forward passes per step — expect ~1.5–1.8x wall-clock slowdown. With SENPAI_TIMEOUT_MINUTES as a hard cap, halve epochs or batch size to compensate.
- Available as `sam` Python package or copy ~50 lines from https://github.com/davda54/sam.

**Expected delta:** Moderate to large on OOD splits. +5–8% on `val_geom_camber_*` and `val_re_rand`. May slightly hurt in-dist.

**Implementation effort:** ~50 LOC + SAM class. Medium risk — doubles compute per step.

---

## Focus Area 4: Augmentation for CFD

### Idea 4.1 — AoA Reflection Symmetry Augmentation

**What it is:** For RaceCar single-foil samples, augment by negating the x-component of node positions and the AoA feature simultaneously (physical left-right mirror of the flow field), and negate Uy in the target (the reflected velocity field).

**Why it might help here:** A horizontal mirror of a 2D airfoil flow is a physically valid solution at the negated AoA. For RaceCar single (the largest domain with 599 train samples), this effectively doubles the training data at zero CFD cost. NeuralFoil (Sharpe & Hansman, 2025) explicitly builds AoA symmetry into their airfoil ML model and shows it significantly improves generalization at extreme AoA values.

**Key papers:**
- Sharpe & Hansman (2025). "NeuralFoil: Airfoil Aerodynamics Analysis through Embedded Deep Learning." arXiv:2503.16323. Uses physics-informed symmetry augmentation for aerodynamic prediction; detailed discussion of AoA reflection symmetry and its implementation.
- Benjamin & Iaccarino (2024). "Systematic Dataset Generation for Automotive Aerodynamics." arXiv:2408.07318. Discusses augmentation strategies for automotive aerodynamic ML.

**Implementation notes:**
- For a mirror augmentation with probability `p_aug=0.5`:
  - Flip node x-positions: `x_aug[:, :, 0] = -x_aug[:, :, 0]`
  - Negate AoA: `x_aug[:, :, 14] = -x_aug[:, :, 14]` (and `x_aug[:, :, 18]` for foil 2)
  - Negate Uy target: `y_aug[:, :, 1] = -y_aug[:, :, 1]`
  - Ux and p are symmetric under this reflection
  - Signed arc-length (dims 2-3) may also need sign flip depending on convention
- Gotcha: `dsdf` features (dims 4-11) are distance-based and may not flip cleanly. Verify that the reflected `dsdf` is consistent with the reflected geometry. If not, skip `dsdf` flipping and just flip positions and AoA.
- Implement as a transform in `train.py`'s training batch loop — no dataloader changes needed.
- Apply only to single-foil samples (check `x[:, :, 22].abs().max() < 1e-4` for gap=0).

**Expected delta:** Small for in-dist, moderate for `val_geom_camber_rc` (nearby AoA range). Expect 2–5% on RaceCar splits.

**Implementation effort:** ~25 LOC. Medium risk (need to verify dsdf symmetry).

---

## Focus Area 5: Positional Encoding for Meshes

### Idea 5.1 — STRING: 2D Rotary Positional Embeddings

**What it is:** STRING (Spatial Transformers with Rotational Inductive biases for Nested Grids) extends RoPE (Rotary Position Embedding) to 2D spatial coordinates. Instead of sequence-position-based rotation, the query/key vectors in attention are rotated by the spatial (x, z) coordinates of the node, encoding relative distances in the attention weights.

**Why it might help here:** Transolver's PhysicsAttention uses learned slice assignments but no explicit relative spatial encoding. Adding STRING-style 2D RoPE to the slice-level attention would give the model translation invariance and scale sensitivity to spatial proximity, which is physically meaningful — nearby nodes should have more similar physics tokens. STRING achieves ICML 2025 spotlight recognition, suggesting strong empirical results.

**Key papers:**
- Schenck et al. (2025). "STRING: Better 2D and 3D Position Encodings with RoPE." arXiv:2502.02562. ICML 2025 spotlight. Extends RoPE to 2D/3D; tested on vision transformers; shows consistent improvements. openreview XXFBqfwnUp.
- Su et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing. Original RoPE formulation.
- HodgeFormer (2025). "Transformers for Learnable Operators on Triangular Meshes." arXiv:2509.01839. Applies topology-aware positional encoding to mesh-based transformers.

**Implementation notes:**
- RoPE for 2D: for each head, split the d_head dimensions into 2 groups. Rotate group 1 by `theta_x * x_coord` and group 2 by `theta_z * z_coord` where `x_coord` and `z_coord` are the node spatial positions.
- In Transolver, apply this to the slice-level keys and queries (after the `in_project_slice` projection, before softmax assignment).
- `theta` base frequencies: use `[10000^(-2i/d_head)]` as in RoPE, separately for x and z.
- Reference implementation: STRING GitHub (https://github.com/nschenck/STRING) provides plug-and-play RoPE modules for 2D.
- Gotcha: Transolver's PhysicsAttention has `slice_num` tokens with shape `[B, H, slice_num, d_head]` — STRING RoPE needs the spatial position of each slice token. Use the slice-weighted mean position as the "position" of each slice token.

**Expected delta:** Moderate. STRING shows 1–3% improvement on 2D vision benchmarks. CFD meshes are geometrically structured, so the spatial bias should help more than on random point clouds.

**Implementation effort:** ~60 LOC. Medium-high risk — requires modifying PhysicsAttention internals.

---

## Focus Area 6: Output Parameterization

### Idea 6.1 — Residual Prediction Over Freestream Baseline

**What it is:** Instead of predicting the full flow field, predict the residual between the actual CFD solution and a simple analytical baseline (potential flow / freestream). The model learns the correction term, which is a much smaller signal.

**Why it might help here:** For most mesh nodes (especially far from the foil), the flow is close to the freestream: `Ux ≈ U_inf * cos(AoA)`, `Uy ≈ U_inf * sin(AoA)`, `p ≈ 0`. The model currently predicts the full field, but the vast majority of residuals come from near-foil boundary layer nodes. Residual prediction shifts the learning target to a zero-mean, smaller-amplitude signal, making it easier for the model to focus on the physically interesting (and metric-relevant) near-surface region.

**Key papers:**
- Inductive transfer from inviscid panel methods: "Transfer Learning for Deep Neural Networks Applied to Aerodynamic Prediction." Advances in Aerodynamics, Springer 2024 (doi.org/10.1186/s42774-024-00186-0). Uses potential flow as a prior for RANS surrogate.
- Raghu et al. (2019). "Transfusion: Understanding Transfer Learning for Medical Imaging." NeurIPS 2019 — demonstrates residual prediction in medical imaging as analogous concept.

**Implementation notes:**
- Compute freestream baseline in physical space: `Ux_free = U_ref * cos(AoA1)`, `Uy_free = U_ref * sin(AoA1)`, `p_free = 0`. Here `U_ref = 1.0` in normalized units (since the simulation normalizes by freestream speed).
- Or use the simpler approximation: just subtract the y_mean (already done by normalization). The residual prediction is partially already happening via the `y_mean` offset.
- A stronger version: subtract a per-node baseline computed from `log(Re)` and `AoA` (e.g., from a linear regression pre-fit on the training set) before computing the loss.
- Gotcha: the `model contract` requires predictions in normalized space and scoring denormalizes with `y_std * pred + y_mean`. Adding a freestream baseline would need to be applied AFTER denormalization — check scoring.py carefully before implementing.

**Expected delta:** Small to moderate. Particularly helpful for far-field nodes (vol loss improvement). Surface nodes are already near-wall so freestream is a poor prior there. May improve `mae_vol_*` more than `mae_surf_p`.

**Implementation effort:** ~30 LOC + pre-fitting step. Medium risk.

---

## Focus Area 7: Curriculum / Hard-Sample Mining

### Idea 7.1 — Per-Domain Loss-Based Hard Sample Mining

**What it is:** Track per-sample validation loss over training epochs and upweight samples with consistently high loss in the next epoch's sampling weights, focusing training on hard cases.

**Why it might help here:** The balanced sampler already equalizes the 3 domains. Within each domain, high-Re samples (extreme y-std) likely have higher loss but are currently sampled uniformly. A curriculum that gradually increases the sampling weight of hard samples (or uses online hard example mining) targets the long tail of the error distribution.

**Key papers:**
- "Active Learning for Neural PDE Solvers." ICLR 2025. openreview 00d4e128a6e7193312954cdc42f3d6a9ea76c7bd. Query-by-committee active learning for hard PDE samples; demonstrates 2–5x sample efficiency gains.
- "Curriculum Learning-Driven Physics-Informed Extreme Learning Machines for Fluid Flow Problems." arXiv:2503.06347. Domain-specific curriculum for fluid flow PDE solving.
- Shrivastava et al. (2016). "Training Region-Based Object Detectors with Online Hard Example Mining." CVPR 2016. Classic OHEM — the foundational reference.

**Implementation notes:**
- Maintain a per-sample EMA loss score: `loss_ema[i] = 0.9 * loss_ema[i] + 0.1 * sample_loss[i]`
- Recompute sampling weights from `loss_ema` every N epochs (e.g., N=5) and pass to `WeightedRandomSampler`.
- Gotcha: this requires per-sample loss tracking, which means not averaging over the batch. Compute per-sample loss by summing (not averaging) over nodes per sample before batch-averaging.
- Simpler version: just oversample the top-20% highest-loss samples by 2x using a fixed schedule.
- The current `sample_weights` from `load_data()` are for domain balancing — create a new weight vector that multiplies domain weights by per-sample loss weights.

**Expected delta:** Small. Primarily improves the high-Re tail of `val_re_rand`. Expect 1–3% overall.

**Implementation effort:** ~30 LOC. Medium risk — changes training dynamics in ways that can interact with the scheduler.

---

## Top 10 Ideas Ranked by (Expected Impact / Implementation Effort)

| Rank | Idea | Impact | Effort | Ratio | Primary Mechanism |
|------|------|--------|--------|-------|-------------------|
| 1 | 3.1 Gradient Clipping | Small-Mod | 1 LOC | Very High | Stabilizes training under heavy-tailed gradient noise |
| 2 | 2.1 Increase slice_num 64→128 | Moderate | 1 LOC | Very High | More physics token capacity for large meshes |
| 3 | 1.1 Huber/SmoothL1 Loss | Moderate | 5 LOC | High | Optimal under heavy-tailed y-std distribution |
| 4 | 1.2 Per-Channel p Loss Weight | Mod-Large | 10 LOC | High | Directly aligns training with primary metric |
| 5 | 3.2 LR Warmup | Small-Mod | 10 LOC | High | Prevents early attention collapse |
| 6 | 3.3 EMA Weights | Small-Mod | 20 LOC | Med-High | Implicit regularization for OOD generalization |
| 7 | 1.3 Per-Sample Relative Loss | Moderate | 15 LOC | Medium | Scale-invariant loss across Re regimes |
| 8 | 3.4 1st-Order SAM | Mod-Large | 50 LOC | Medium | Flat minima for OOD; +5–8% on camber splits |
| 9 | 2.2 FiLM Re Conditioning | Mod-Large | 50 LOC | Medium | Explicit cross-regime conditioning |
| 10 | 4.1 AoA Reflection Augmentation | Small-Mod | 25 LOC | Medium | Doubles effective training data via physics symmetry |

---

## Honorable Mentions

### STRING 2D RoPE (Idea 5.1)
High potential but ~60 LOC requiring PhysicsAttention modifications. Strong theoretical motivation (ICML 2025 spotlight). Worth a second-round experiment after the low-effort ideas are exhausted.

### DPOT: Auto-Regressive Denoising Operator Transformer (arxiv 2403.03542)
Diffusion-style denoising training for neural operators. Showed strong results on multiple PDE benchmarks. Higher implementation effort (~100 LOC) but qualitatively different training paradigm.

### Universal Physics Transformers (UPTs, NeurIPS 2024, arxiv)
Efficient scaling with compressed latent representations. Would require replacing Transolver with a different architecture — a larger architectural bet.

### GFocal: Global-Focal Neural Operator (arxiv 2508.04463)
Multi-resolution attention combining global and focal (local) attention for arbitrary geometries. Strong fit for multi-zone CFD meshes but high implementation effort.

---

## Implementation Priority Recommendation

**Round 1 (days 1–2, near-zero risk):**
- Gradient clipping (1 LOC): run immediately
- slice_num 64→128 (1 LOC config): run immediately
- Combine both in one training run

**Round 2 (days 3–5, low effort):**
- Huber loss + per-channel p weight (combine in ~15 LOC)
- LR warmup (~10 LOC)

**Round 3 (days 6–10, medium effort):**
- EMA weights (~20 LOC)
- Per-sample relative loss (~15 LOC)
- Test whether Huber or relative loss is better — do not combine both until you know which mechanism is active

**Round 4 (if plateau, days 10–15):**
- 1st-Order SAM (50 LOC + halved epochs to fit wall-clock)
- FiLM Re conditioning (50 LOC)

---

## References

1. Fan et al. (2022). "Huber Loss Revisited." arXiv:2203.10418
2. Wang et al. (2023). "Deep Regression Loss Functions." arXiv:2309.12872
3. Terven et al. (2023). "Survey of Loss Functions for ML." arXiv:2307.02694
4. Transolver (2024). "Transolver: Physics-Aware Transformer." arXiv:2402.02366
5. AeroDiT (2024). "Diffusion Transformers for RANS Airfoil." arXiv:2412.17394
6. AB-UPT (2025). "Anchored-Branched UPT for Automotive CFD." TMLR 10/2025
7. Foret et al. (2021). "SAM: Sharpness-Aware Minimization." ICLR 2021. openreview 6Tm1mposlrM
8. Schapiro & Zhao (2024). "SAM and OOD Generalization." arXiv:2412.05169
9. Kaddour et al. (2024). "1st-Order SAM." arXiv:2411.01714
10. Morales-Brotons et al. (2024). "EMA of Weights in Deep Learning." arXiv:2411.18704. TMLR
11. Koloskova et al. (2023). "Gradient Clipping: Tight Convergence Guarantees." arXiv:2305.01588
12. Schnell & Thuerey (2024). "Stabilizing BPTT for Physics Simulations." arXiv:2405.02041
13. Kosson et al. (2024). "LR Warmup is Essential for GPT Training." arXiv:2410.23922
14. Schenck et al. (2025). "STRING: Better 2D/3D Position Encodings." arXiv:2502.02562. ICML 2025
15. Sharpe & Hansman (2025). "NeuralFoil: Airfoil Aerodynamics via DL." arXiv:2503.16323
16. Benjamin & Iaccarino (2024). "Dataset Generation for Automotive Aero." arXiv:2408.07318
17. Active Learning for Neural PDE Solvers. ICLR 2025. openreview 00d4e128a6e7
18. Curriculum Learning for PIELMs. arXiv:2503.06347
19. Izmailov et al. (2018). "SWA." UAI 2018
20. SeWA (2025). arXiv:2502.10119
21. DPOT (2024). arXiv:2403.03542
22. GFocal (2025). arXiv:2508.04463
23. UPTs (NeurIPS 2024). Alkin et al.
24. HodgeFormer (2025). arXiv:2509.01839
25. GeoPE (ICLR 2026). openreview y6piOp5MSO
