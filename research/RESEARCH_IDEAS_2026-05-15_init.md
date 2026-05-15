# TandemFoilSet Research Hypotheses — 2026-05-15 (Initial Launch)

**Branch:** `icml-appendix-willow-pai2i-48h-r2`  
**Constraints:** train.py only; no new packages; SENPAI_MAX_EPOCHS=50; SENPAI_TIMEOUT_MINUTES=30; 96 GB VRAM; batch_size=4 default.  
**Primary metric:** `val_avg/mae_surf_p` (lower is better), equal-weight mean surface-pressure MAE across 4 val splits.  
**Baseline model:** Transolver — space_dim=2, fun_dim=22, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2; AdamW lr=5e-4, wd=1e-4; cosine annealing T_max=MAX_EPOCHS; surf_weight=10.0; no gradient clipping.

---

## Hypothesis 1: pressure-channel-weight — Per-channel surface loss weighting to boost pressure accuracy

**Slug:** `pressure-channel-weight`

**Motivation.** The primary ranking metric is surface pressure MAE (`val_avg/mae_surf_p`), yet the current loss treats Ux, Uy, and p equally inside both `vol_loss` and `surf_loss`. In normalized space the three channels have comparable variance (normalization removes the scale difference), but the model's gradient signal is spread one-third to pressure and two-thirds to velocity components that are not scored. Per-channel loss weighting is a standard technique in multi-output regression: by up-weighting the pressure channel in the surface loss, we redirect gradient signal toward the scored objective without any architectural change. This is the cheapest possible intervention — one hyperparameter — and should be tried first. Conceptually related to task-specific loss weighting in multi-task learning (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018), though here we hand-tune rather than learn the weights. The key risk is that over-weighting pressure during training may degrade velocity prediction and indirectly worsen geometry generalization splits where velocity and pressure are tightly coupled through the Navier-Stokes momentum equation.

**Implementation.**
- In `train.py`, locate the loss computation block inside the training loop.
- Currently: `loss = vol_loss + cfg.surf_weight * surf_loss` where both terms are mean-squared-error averaged uniformly over the 3 output channels (Ux, Uy, p at indices 0, 1, 2).
- Change the surface loss to apply a per-channel weight vector `[1.0, 1.0, w_p]` before summing:
  ```python
  ch_weight = torch.tensor([1.0, 1.0, cfg.p_surf_weight], device=sq_err.device)
  surf_loss = (sq_err * surf_mask.unsqueeze(-1) * ch_weight).sum() / surf_mask.sum().clamp(min=1)
  ```
- Add `p_surf_weight: float = 5.0` to the `Config` dataclass. This multiplies the pressure channel's gradient contribution within the surface loss by 5x relative to velocity channels.
- The volume loss remains uniform (pressure accuracy in the volume is less directly scored but is coupled to surface accuracy via boundary conditions).
- Run two arms: `p_surf_weight=3.0` and `p_surf_weight=5.0`. Keep `surf_weight=10.0` unchanged.
- Use `--wandb_group pressure-channel-weight`.

**Predicted delta:** −5 to −15% relative on `val_avg/mae_surf_p`. The channel is directly in the metric; re-directing gradient should produce a clean win.

**Failure modes.**
- If the normalized pressure residuals are already the dominant loss term (because |p| >> |Ux|, |Uy| in normalized space at high Re), extra weighting may push the optimizer into a degenerate regime where velocity predictions collapse.
- If the three channels are already near-optimally balanced by the normalization, the effect may be small (< 2% improvement).

**Wall-clock estimate:** ~20–25 min per arm at default batch_size=4. Run both arms in the same job by iterating over `p_surf_weight` values, or submit as two separate 25-min runs.

**Multi-arm note:** Two arms recommended — `p_surf_weight=3.0` and `p_surf_weight=5.0`. Report both in the SENPAI-RESULT.

---

## Hypothesis 2: grad-clip-huber — Gradient clipping + Huber loss to stabilize high-Re training

**Slug:** `grad-clip-huber`

**Motivation.** The training set spans Reynolds numbers from ~100K to ~5M. In physical units, per-sample `y` standard deviation ranges from tens to thousands, meaning high-Re samples produce normalized residuals much larger than low-Re samples even after global normalization. The baseline uses MSE loss in normalized space, which squares these large residuals and produces gradient spikes proportional to the squared normalized error. Without gradient clipping, these spikes can dominate parameter updates and destabilize training, pushing the model toward fitting the extreme-value region at the cost of the moderate-Re regime where most test samples live. Two complementary interventions address this: (1) gradient clipping (Pascanu et al., "On the difficulty of training recurrent neural networks", ICML 2013) bounds the parameter update norm and prevents runaway updates from isolated high-Re batches; (2) Huber loss (smooth L1) with δ=1.0 in normalized space transitions from L2 to L1 beyond one standard deviation of normalized residual, reducing the weight given to the most extreme values while remaining quadratic in the well-predicted regime. Both interventions are standard in robust regression and have been validated in physics-informed neural network settings (Wang et al., 2022; Krishnapriyan et al., 2021).

**Implementation.**
- Add `grad_clip: float = 1.0` to `Config`. In the training loop, after `loss.backward()` and before `optimizer.step()`, add:
  ```python
  if cfg.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
  ```
- Replace MSE with Huber: change `sq_err = (pred_norm - y_norm).pow(2)` to:
  ```python
  delta = 1.0
  abs_err = (pred_norm - y_norm).abs()
  sq_err = torch.where(abs_err < delta, 0.5 * abs_err.pow(2), delta * (abs_err - 0.5 * delta))
  ```
  (This is `F.huber_loss` semantics with `reduction='none'`; you can also use `F.smooth_l1_loss(reduction='none')` which has δ=1 by default.)
- Keep `surf_weight=10.0` unchanged. Keep the scoring.py MAE accumulation unchanged (it operates on denormalized predictions).
- Run as a single arm: grad_clip=1.0 + Huber δ=1.0.
- Use `--wandb_group grad-clip-huber`.

**Predicted delta:** −3 to −10% relative on `val_avg/mae_surf_p`, primarily through more stable training on high-Re samples. Secondary benefit: better `val_re_rand` (the stratified Re holdout), which should be tracked separately.

**Failure modes.**
- If the model is already well-converged and gradient spikes are rare, this intervention will have minimal effect.
- Huber loss with δ=1.0 may be too aggressive in reducing the cost of large residuals if the model is still far from convergence at epoch 1; the loss shape changes during training.
- Clipping at 1.0 may be too conservative if the baseline gradients are well-behaved; a too-tight clip can slow convergence.

**Wall-clock estimate:** ~20–22 min. Minimal overhead versus baseline.

---

## Hypothesis 3: surf-weight-scan — Higher surface weight to drive surface accuracy at the cost of volume

**Slug:** `surf-weight-scan`

**Motivation.** The baseline `surf_weight=10.0` was not derived from principled tuning — it is a round-number default. Surface nodes are a small fraction of all mesh nodes (roughly 2–5% by count, given a 74K–242K node mesh with O(1K–5K) surface nodes per foil), so without upweighting they would contribute only 2–5% of the loss signal. At surf_weight=10, their effective contribution is still only 20–50% of the total loss. The primary metric is surface-only MAE, so there is a direct argument for increasing surface weight substantially. At surf_weight=50 or 100, the model would be almost entirely optimizing surface accuracy in normalized space. The risk is that the volume field becomes unconstrained, which may matter for generalization (the volume solution constrains the surface solution through the PDE coupling). This is a pure ablation of an existing hyperparameter — cheap to run and directly informative.

**Implementation.**
- No code changes except the value passed to `--surf_weight`.
- Run two arms: `surf_weight=25.0` and `surf_weight=50.0`.
- Keep all other hyperparameters at baseline defaults: lr=5e-4, wd=1e-4, batch_size=4.
- Use `--wandb_group surf-weight-scan`.
- Compare both arms against baseline `surf_weight=10.0`. The winner (if any) can be combined with other improvements in later PRs.

**Predicted delta:** −5 to −20% relative. If the surface fraction is small, the upside could be large. At surf_weight=100 there is risk of divergence (the volume field becomes ill-conditioned); 25–50 is a reasonable first range.

**Failure modes.**
- If volume-field inaccuracy causes the model to extrapolate badly on OOD geometry splits (`val_geom_camber_rc`, `val_geom_camber_cruise`), higher surf_weight may hurt generalization while improving in-distribution surface accuracy.
- If surface nodes are already well-fitted and the bottleneck is volume accuracy (which propagates to surface via physics), this intervention will not help.

**Wall-clock estimate:** ~20 min per arm. Both arms fit in a single 30-min job if run sequentially, or two separate 25-min jobs.

**Multi-arm note:** Run `surf_weight=25` and `surf_weight=50`. Report both in SENPAI-RESULT with best as primary metric.

---

## Hypothesis 4: lr-warmup-cosine — Linear warmup + cosine annealing for more stable early training

**Slug:** `lr-warmup-cosine`

**Motivation.** The baseline uses CosineAnnealingLR starting from the maximum learning rate at epoch 0, which means the first few epochs receive the largest gradient steps on randomly initialized weights. In transformer models this is known to cause instability: before the attention mechanism has learned meaningful slice assignments, large parameter updates can push the softmax-based slice weights into degenerate configurations (e.g., all mass on one slice, or perfectly uniform mass). Linear warmup over 3–5 epochs ramps the learning rate from near zero to `lr_max`, allowing the model to first reach a stable basin before aggressive parameter updates begin. This is standard practice in transformer pre-training (Vaswani et al. 2017; Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond", ICLR 2020) and has been validated in physics-based neural operators (Li et al., FNO, 2021). The warmup is cheap to implement using `torch.optim.lr_scheduler.LinearLR` + `SequentialLR` and adds zero architectural complexity.

**Implementation.**
- Replace the existing `CosineAnnealingLR` scheduler setup with a `SequentialLR` combining a linear warmup and cosine decay:
  ```python
  warmup_epochs = 5
  warmup_sched = torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
  )
  cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=MAX_EPOCHS - warmup_epochs, eta_min=1e-6
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
  )
  ```
- Add `warmup_epochs: int = 5` to `Config` (with `warmup_epochs=0` reproducing baseline).
- Keep lr=5e-4, wd=1e-4, surf_weight=10.0 at baseline.
- Single arm: warmup_epochs=5.
- Use `--wandb_group lr-warmup-cosine`.

**Predicted delta:** −3 to −8% relative, primarily from more stable convergence toward lower minima. The benefit is expected to be consistent across all val splits.

**Failure modes.**
- With only 50 epoch budget, using 5 epochs for warmup leaves 45 epochs of cosine annealing. If the effective learning rate is already low by epoch 30 and the model has not converged, warmup may not help enough to offset the reduced cosine budget.
- If the baseline optimizer is already behaving stably (inspectable by looking at training loss curves in W&B), warmup may add negligible benefit.

**Wall-clock estimate:** ~22 min. No overhead vs. baseline.

---

## Hypothesis 5: slice-num-128 — Double the number of physics attention tokens from 64 to 128

**Slug:** `slice-num-128`

**Motivation.** The Transolver's PhysicsAttention mechanism assigns each of N mesh nodes to `slice_num` learnable "physics tokens" via soft membership. The 64 default means the entire flow field is summarized into 64 representative states. For our dataset, this is potentially a bottleneck: the TandemFoilSet meshes contain up to 242K nodes spanning 3 distinct flow zones (background + 2 foil-local zones). The overset mesh topology means nodes near each foil experience qualitatively different flow physics than background nodes. With only 64 tokens, the model may be unable to simultaneously represent boundary-layer gradients near the foil surfaces, the wake region, and the far-field. Doubling to 128 tokens increases representational capacity within the attention bottleneck without changing the architecture — the `in_project_slice` layer projects from `dim_head=32` to `slice_num`, so changing `slice_num` changes only that projection's output size and the subsequent attention weight matrices. Quadratic attention cost is in token space (128² vs 64²), which is trivial compared to the N=200K+ node projection cost. (Cao et al., "Don't be so dense", NeurIPS 2022; Wu et al., "Transolver", ICML 2024).

**Implementation.**
- In `model_config`, change `slice_num=64` to `slice_num=128`:
  ```python
  model_config = dict(
      space_dim=2, fun_dim=X_DIM - 2,
      out_dim=3, n_hidden=128, n_layers=5, n_head=4,
      slice_num=128, mlp_ratio=2,
      output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
  )
  ```
- This is the only change needed. The `Transolver.__init__` uses `slice_num` to size `in_project_slice` and the attention mask matrix.
- Verify that memory usage does not spike significantly — the 128² attention is computed in token space (128×128 per head per batch), not in node space, so VRAM impact is minimal.
- Single arm.
- Use `--wandb_group slice-num-128`.

**Predicted delta:** −5 to −15% relative, primarily on OOD geometry splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) where capturing the diverse flow physics of unseen airfoil shapes benefits most from richer token representations.

**Failure modes.**
- If slice_num=64 is already sufficient to represent the flow diversity in training data, doubling may not help — this indicates the bottleneck is elsewhere (architecture depth, training dynamics, or data coverage).
- The `orthogonal_` init on `in_project_slice.weight` was designed for a 64-column matrix; the 128-column case may benefit from different initialization, but orthogonal_ still applies (the weight matrix is `[hidden_dim, 128]` rather than `[hidden_dim, 64]`).
- More tokens may cause slower convergence in early training (more assignments to learn), potentially requiring more epochs to reach comparable quality.

**Wall-clock estimate:** ~22 min. Token-space attention is negligible cost vs. node-space projection.

---

## Hypothesis 6: re-sinusoidal-embed — Replace scalar log(Re) feature with sinusoidal Reynolds number embedding

**Slug:** `re-sinusoidal-embed`

**Motivation.** Dimension 13 of the input feature vector is `log(Re)`, a single scalar spanning `log(100K)≈11.5` to `log(5M)≈15.4`. In the normalized input space this becomes a scalar in roughly [−2, +2] after (x − x_mean) / x_std. The model must learn to condition all flow field predictions on this single scalar, which is the most physically important parameter (Re governs whether the flow is laminar or turbulent, the thickness of boundary layers, the separation point, wake structure, etc.). Replacing this scalar with a sinusoidal positional encoding — i.e., projecting `log(Re)` into a d-dimensional embedding using sin/cos at multiple frequencies — provides the network with a richer, multi-scale representation of the Re regime without requiring any new learned parameters if we fix the frequencies (or with O(d) parameters if we learn them). This is inspired by the sinusoidal position encoding from transformers (Vaswani et al. 2017), Fourier feature embeddings (Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020), and conditioning in diffusion models. The intuition: different frequency components of the Re embedding allow the model to easily separate low-Re from high-Re regimes while maintaining smooth interpolation within each regime.

**Implementation.**
- Precompute a sinusoidal embedding of `log(Re)` in the training loop (before normalization), replacing dimension 13 with a d=8 embedding (adding 7 dimensions to the feature vector, total X_DIM_NEW = 24 + 7 = 31). Alternative: replace dimension 13 in-place with the first sine component and append the remaining components, ensuring the model `fun_dim` is updated.
- Concretely in `train.py`, after loading a batch, before normalizing:
  ```python
  def sinusoidal_re_embed(x, re_dim=13, embed_dim=8):
      log_re = x[..., re_dim:re_dim+1]  # [B, N, 1]
      freqs = torch.exp(torch.linspace(0, 3, embed_dim // 2, device=x.device)) # 4 freqs
      angles = log_re * freqs.unsqueeze(0).unsqueeze(0)  # [B, N, 4]
      embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, N, 8]
      x_new = torch.cat([x[..., :re_dim], embed, x[..., re_dim+1:]], dim=-1)  # [B, N, 31]
      return x_new
  ```
- Update `model_config`: `fun_dim = 31 - 2 = 29` (node_pos is space_dim=2, remainder is fun_dim).
- Update the normalization: the stats.json `x_mean` and `x_std` have 24 dimensions; the new dimensions (sin/cos of log(Re)) are already "normalized" to [−1, +1] by construction. Replace the stats normalization for dimensions 13 onwards: only normalize dims 0-12 and 14-30 using stats, or simply zero-center the new embedding dimensions manually.
- Simpler alternative (lower risk): keep X_DIM=24, replace dim 13 with `sin(log(Re)/5.0)` and add `cos(log(Re)/5.0)` as dim 24 (expanding x from 24 to 25), update fun_dim=23, update stats normalization to handle the new dim with mean=0, std=1.
- Use `--wandb_group re-sinusoidal-embed`.

**Predicted delta:** −3 to −8% relative, especially on `val_re_rand` (cross-regime generalization) where the model must interpolate across the Re range.

**Failure modes.**
- The sinusoidal embedding changes the effective input distribution, which interacts with the pre-computed normalization stats (stats.json was computed for the original 24-dim input). Careful handling of the new dimensions is required.
- If the model has already learned an adequate internal Re representation from the single scalar (e.g., the first-layer weights have implicitly learned a periodic mapping), this may add complexity without gain.
- Implementation complexity: the `fun_dim` change propagates into the `preprocess` MLP inside Transolver, which must be updated consistently with the new input dimension.

**Wall-clock estimate:** ~22 min. No VRAM impact.

---

## Hypothesis 7: ema-weights — Exponential moving average of model weights for better generalization

**Slug:** `ema-weights`

**Motivation.** Stochastic gradient descent with a learning rate scheduler converges to a neighborhood of a local minimum but oscillates around it in late training. Averaging model weights over the trajectory (Polyak averaging, or exponential moving average) produces a point near the center of this neighborhood that often generalizes better than any single checkpoint, particularly on out-of-distribution samples. This is empirically well-established: SWA (Stochastic Weight Averaging; Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018) and EMA (used in diffusion models and modern vision transformers) consistently improve generalization on geometry-shifted or distribution-shifted evaluation sets. In our setting, `val_geom_camber_rc` and `val_geom_camber_cruise` are full file holdouts on unseen NACA camber values — exactly the kind of distribution shift where weight averaging is most likely to help. EMA is simpler than SWA to implement (no scheduling required) and adds zero training overhead (it runs during validation, not forward/backward passes).

**Implementation.**
- After `model = Transolver(...).to(device)`, create an EMA shadow:
  ```python
  ema_decay = 0.999
  ema_model = copy.deepcopy(model)
  for p in ema_model.parameters():
      p.requires_grad_(False)
  ```
- After each optimizer step in the training loop:
  ```python
  with torch.no_grad():
      for ema_p, p in zip(ema_model.parameters(), model.parameters()):
          ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
  ```
- For validation and checkpoint selection, evaluate `ema_model` instead of `model`. For saving the best checkpoint and W&B artifact, save `ema_model.state_dict()`.
- Add `ema_decay: float = 0.999` to `Config`. Add `import copy` at the top of train.py (already likely present, but verify).
- Use `--wandb_group ema-weights`.

**Predicted delta:** −3 to −10% relative, concentrated on OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`). In-distribution split (`val_single_in_dist`) may show smaller improvement.

**Failure modes.**
- EMA with decay=0.999 takes ~1000 steps to converge to a stable average (burn-in period). With batch_size=4 and ~1499 training samples, one epoch ≈ 375 steps, so 1000 steps ≈ 3 epochs. With only 50 epochs, the EMA should be stable well before validation starts.
- If training is noisy and the model oscillates between good and bad basins, EMA will smooth these out but may converge to a suboptimal average. A too-high decay (0.9999) would slow adaptation.
- The EMA model must be used consistently for both validation scoring and checkpoint saving; using the live model for one and EMA for the other will corrupt the baseline comparison.

**Wall-clock estimate:** ~22 min. EMA update is a few tensor operations per step, negligible overhead.

---

## Hypothesis 8: hidden-256-depth6 — Wider hidden dimension and deeper model

**Slug:** `hidden-256-depth6`

**Motivation.** The baseline Transolver has n_hidden=128 and n_layers=5. For a dataset spanning 2D CFD over multiple physical domains with variable geometry and Re, 128 dimensions may be a representational bottleneck — insufficient hidden size to simultaneously encode boundary-layer structure, wake dynamics, and far-field effects. Increasing to n_hidden=256 doubles the model's per-node representational width and quadruples the number of parameters in the attention projection matrices. n_layers=6 adds one additional cross-attention + MLP block, deepening the hierarchical feature extraction. This follows the standard scaling hypothesis: for a fixed compute budget, neural operators on mesh data typically benefit from width scaling up to the point where VRAM or throughput becomes limiting (Li et al. FNO 2021; Herde et al. "Poseidon", 2024). At n_hidden=256, batch_size=4, and mesh sizes up to 242K nodes, VRAM usage should remain within 96 GB (the dominant cost is the `[B, N, hidden]` activations: 4 × 242K × 256 × 4 bytes ≈ 1 GB, plus attention matrices, well within budget).

**Implementation.**
- In `model_config`, change `n_hidden=128` to `n_hidden=256` and `n_layers=5` to `n_layers=6`:
  ```python
  model_config = dict(
      space_dim=2, fun_dim=X_DIM - 2,
      out_dim=3, n_hidden=256, n_layers=6, n_head=4,
      slice_num=64, mlp_ratio=2,
      output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
  )
  ```
- Keep batch_size=4 (default). Verify no OOM at the start of training on a large cruise sample (242K nodes × 256 dim × 4 batch).
- Keep lr=5e-4. Larger models sometimes benefit from lower lr; if the student sees instability, they can halve to 2.5e-4 in a follow-up.
- Single arm.
- Use `--wandb_group hidden-256-depth6`.

**Predicted delta:** −5 to −15% relative. Wider models often show compounding improvements when combined with other changes (e.g., surf_weight tuning on a wider model). Expect the most gain on `val_re_rand` and `val_geom_camber_rc` where more capacity helps generalize across regimes.

**Failure modes.**
- VRAM usage may be higher than estimated if the PhysicsAttention stores full `[B, heads, N, slice_num]` weight matrices (4 × 4 × 242K × 64 × 4 bytes ≈ 1 GB per layer, ×6 layers ≈ 6 GB plus gradients ≈ 12 GB); still within 96 GB.
- Convergence may be slower — a wider model may need more epochs to reach the quality of the narrow baseline. With a 30-minute wall-clock cap, this is a real risk: the wider model may not train long enough to fully leverage its capacity.
- Weight initialization: orthogonal_ init on `in_project_slice` produces an orthogonal-ish matrix for the 128→64 projection; for 256→64 the matrix is no longer square, but `torch.nn.init.orthogonal_` handles non-square matrices (orthonormal columns), so this is fine.

**Wall-clock estimate:** ~25 min. Expect ~20–30% slower per-epoch due to wider matrices.

---

## Hypothesis 9: per-channel-output-heads — Separate output MLPs for velocity and pressure channels

**Slug:** `per-channel-output-heads`

**Motivation.** The baseline Transolver uses a single output head `mlp2 = [hidden→hidden→GELU→3]` that maps the final hidden state to all three output channels simultaneously. This forces the same learned representation to be decoded into two conceptually different physical quantities: velocity (Ux, Uy — smooth, near-divergence-free in most of the domain) and pressure (p — driven by the Bernoulli/Poisson relationship with the velocity magnitude, exhibiting suction peaks near the leading edge). Pressure near the surface is determined by a different combination of upstream flow features than velocity. Separating the output into two independent heads — one for velocity `[hidden→hidden→2]` and one for pressure `[hidden→hidden→1]` — allows the model to learn different projections from the shared hidden state to each quantity, potentially improving pressure accuracy without sacrificing velocity accuracy. This is standard practice in multi-task learning and has precedent in physics-informed neural networks where separate decoders for each PDE variable consistently outperform joint decoders (Raissi et al. 2019; Wang et al. 2021).

**Implementation.**
- In the `Transolver.__init__`, replace the single `self.mlp2` in the last `TransolverBlock` with two separate heads:
  ```python
  # In Transolver.__init__, after creating blocks:
  # The last block's mlp2 is accessed via self.blocks[-1].mlp2
  # Override the last block's mlp2 and add a pressure head:
  last_block = self.blocks[-1]
  last_block.mlp2_vel = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
      nn.Linear(hidden_dim, 2),  # Ux, Uy
  )
  last_block.mlp2_p = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
      nn.Linear(hidden_dim, 1),  # p
  )
  # Remove the original mlp2 from the last block
  del last_block.mlp2
  ```
- Override the forward pass of the last `TransolverBlock` (or subclass it) to concatenate the two head outputs:
  ```python
  # In TransolverBlock.forward, when last_layer=True:
  feat = self.ln_3(fx)
  vel = self.mlp2_vel(feat)  # [B, N, 2]
  prs = self.mlp2_p(feat)    # [B, N, 1]
  return torch.cat([vel, prs], dim=-1)  # [B, N, 3]
  ```
- Simplest implementation: add a `split_output_heads: bool = False` flag. When True, replace `mlp2` in `Transolver.__init__` with two heads and update `forward` accordingly.
- Use `--wandb_group per-channel-output-heads`.

**Predicted delta:** −2 to −8% relative on `val_avg/mae_surf_p`. The benefit is modest in expectation because the shared hidden state can in principle learn to disentangle velocity and pressure internally; the separate heads help but are not a fundamental change.

**Failure modes.**
- The structural change to the last TransolverBlock requires careful implementation to avoid bugs (e.g., forgetting to delete the original `mlp2`, or not properly forwarding through both heads).
- This adds two new `nn.Sequential` modules to the model; the total parameter count increases by roughly 2 × (hidden²) compared to 1 × (hidden×3), which is negligible at n_hidden=128.
- If the bottleneck is in the attention mechanism (not the output head), this will have no effect.

**Wall-clock estimate:** ~22 min. No architectural overhead.

---

## Hypothesis 10: cosine-lr-restart — Cosine annealing with warm restarts (SGDR) to escape local minima

**Slug:** `cosine-lr-restart`

**Motivation.** The baseline cosine annealing schedule decays the learning rate monotonically over all 50 epochs, which means the model can get trapped in a local minimum early in training and never escape. Stochastic Gradient Descent with Warm Restarts (SGDR; Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017) periodically resets the learning rate to its maximum, allowing the optimizer to jump out of local minima and explore the loss landscape more thoroughly. For physics neural operators on diverse datasets, the loss landscape contains multiple local minima corresponding to different physical regimes (low Re vs. high Re, single-foil vs. tandem). SGDR with T_0=10 (restart every 10 epochs) and T_mult=2 (doubling the period each restart) would give restarts at epochs 10, 30, with the final phase running epochs 30–50, giving the model time to converge in the last phase. This is a drop-in replacement for `CosineAnnealingLR` using `CosineAnnealingWarmRestarts` from PyTorch.

**Implementation.**
- Replace the scheduler:
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer, T_0=10, T_mult=2, eta_min=1e-6
  )
  ```
- Add `T_0: int = 10` and `T_mult: int = 2` to `Config`.
- Keep all other parameters at baseline defaults.
- Single arm: T_0=10, T_mult=2.
- Use `--wandb_group cosine-lr-restart`.

**Predicted delta:** −3 to −8% relative. Expected improvement concentrated on OOD splits where the model benefits from exploring multiple basins. The final convergence phase (epochs 30–50) is the longest, giving the model the most time to refine after the last restart.

**Failure modes.**
- SGDR is most beneficial when the training run is long enough to see multiple full cycles; with MAX_EPOCHS=50 and T_mult=2, there are only 2 restarts (at 10 and 30), which may be too few for the benefit to manifest.
- If the model reaches a good basin by epoch 10 and the restart pushes it away, performance may temporarily regress during the restart phase (visible in W&B as a loss spike).
- With a 30-minute wall-clock cap, actual epoch count may be lower than 50; if training terminates after the first restart (epoch 10–30), the effective schedule is a single cosine cycle similar to baseline.

**Wall-clock estimate:** ~22 min. Scheduler swap is essentially free.

---

## Priority Order

1. **surf-weight-scan** (Hypothesis 3) — Pure hyperparameter ablation; directly targets the primary metric objective; near-zero implementation risk; highest information-per-compute.
2. **pressure-channel-weight** (Hypothesis 1) — One hyperparameter on top of surf_weight; directly targets `mae_surf_p`; clean mechanism.
3. **grad-clip-huber** (Hypothesis 2) — Addresses a known failure mode (high-Re gradient spikes); cheap to implement; should improve `val_re_rand` as a secondary signal.
4. **ema-weights** (Hypothesis 7) — Well-established technique for OOD generalization; no architectural change; expected to help geometry holdout splits.
5. **slice-num-128** (Hypothesis 5) — Architectural capacity increase with minimal implementation change; addresses the physics-token bottleneck hypothesis.
6. **hidden-256-depth6** (Hypothesis 8) — Scaling; larger expected impact but higher risk (convergence time, VRAM).
7. **lr-warmup-cosine** (Hypothesis 4) — Optimization stability; lower expected gain than loss/architecture changes in this regime.
8. **per-channel-output-heads** (Hypothesis 9) — Structural change with modest expected gain; worth testing but lower priority than loss and optimization levers.
9. **re-sinusoidal-embed** (Hypothesis 6) — Higher implementation risk due to stats normalization interaction; medium expected gain.
10. **cosine-lr-restart** (Hypothesis 10) — Lowest priority; SGDR benefit may not materialize within the 30-min wall-clock cap.

---

## Notes on Combinations

If any of the top-3 hypotheses produce improvements, the following combinations are promising follow-up PRs:
- **surf-weight-scan winner + pressure-channel-weight**: orthogonal changes to the loss formulation.
- **grad-clip-huber + ema-weights**: complementary stability improvements (loss-side + weight-side).
- **slice-num-128 + hidden-256-depth6**: capacity scaling both in token count and hidden width.

Do not combine unvalidated changes in a single PR — each combination should be a separate PR using the best available checkpoint as the new baseline.
