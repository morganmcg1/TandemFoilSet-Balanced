<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Research Ideas — 2026-05-13 22:30
# Launch: willow-pai2g-24h-r3 | Advisor branch: icml-appendix-willow-pai2g-24h-r3

## Context and Constraints

**Current best**: val_avg/mae_surf_p = 40.2741, test_avg/mae_surf_p = 33.6017 (PR #2192, n_head=2 + Lion, seed `gd934e9l`)

**Plateau protocol triggered**: 3 confirmed regressions — n_layers=3 (+3.34%), n_head=1 (+9.5%), wd=3e-4 (+2.6%). All hyperparameter tweaks on n_head=2+Lion stack are exhausted.

**Merge bar**: val ≤ 36.2 (≥10% gain) → merge directly; 36.2–40.3 → second seed needed; ≥40.3 → close.

**Closed axes** (do not repeat without new evidence):
- Architecture hyperparameters: n_layers=3, n_head=1, n_head≥4, n_hidden=192/n_head=6
- Loss shape (residual-conditional): MSE, Pure L1, Truncated L1 (SmoothL1/Huber β=0.1 is optimum)
- Loss weighting: surf_weight>10, p-channel ×3/5×, Per-Re WeightedRandomSampler, Re-level reweighting
- Regularization: DropPath, AdamW wd≠1e-4, coord-jitter/input-noise, wd=3e-4 on Lion
- Feature engineering: Fourier K>12 (all regress), mlp_ratio≠2
- Schedule: Cosine tail compression (T_max=15/20/25)

**Mechanistic lessons encoded in these hypotheses**:
1. Bernoulli coupling: Ux/Uy/p globally coupled via pressure-Poisson; channel reweighting starves other channels.
2. Lion+noise amplification: sign-based momentum amplifies corrupted gradient directions; any noisy perturbation hurts more than with AdamW.
3. EMA(0.999) already subsumes DropPath-style variance reduction.
4. Sample-conditional gradient shaping is unexplored (residual-conditional exhausted).
5. Under-budgeted training: all seeds hit best_epoch=final_epoch, model still descending at 30-min cap.
6. SmoothL1 win was gradient capping on outlier tails, not quadratic-near-zero smoothing.

---

## Category 1: Loss Reformulation (Sample-Conditional)

### H1: Adaptive Per-Weight Curriculum (APW) via Loss-Based Weighting

**What it is**: Per-sample loss weighting that starts flat (equal weight) and progressively upweights high-loss samples as training proceeds, implementing an easy-to-hard curriculum via the sample's own running loss value.

**Why it might help**: The sample-conditional gradient shaping axis is completely unexplored. The current SmoothL1 shapes gradient magnitude as a function of residual magnitude (residual-conditional), but treats all samples equally. With 3 domains spanning order-of-magnitude differences in y-std (raceCar single 458 avg std vs. cruise 164 avg std), high-Re samples dominate raw-MAE while low-Re samples dominate normalized-space loss magnitude. APW addresses this by letting the model discover which samples it finds hard and upweighting those — without the manual camber/Re groupings that already failed.

**Mechanism**: Maintain an exponential moving average of per-sample loss `ℓ̄_i`. At each step, sample weight `w_i = (ℓ̄_i / ℓ̄_mean)^α` where α anneals from 0 → 1 over training. Loss is `mean(w_i * SmoothL1(pred_i, y_i))`. This is implemented entirely in `train.py` — no data loader changes.

**Key paper**: APW Curriculum (arxiv 2505.01665) — "Adaptive Point-Weighting for Imbalanced Data in Scientific ML"; reports consistent improvement on PDE tasks with domain imbalance.

**Implementation notes**:
- Use a `dict` keyed by sample index (from dataloader) to store EMA loss. Batch indices come from `DataLoader` with `__index__` passthrough.
- α schedule: linear warmup from 0 to 1 over first 50% of epochs; freeze at 1 thereafter.
- Normalize weights per-batch to avoid scale drift: `w_i = w_i / w_i.mean()`.
- Critical: compute weight from the **previous step's** loss (detach), not the current step. Otherwise you get gradient-through-weight instability.
- Start with α_max = 0.5 (mild curriculum). α_max = 1.0 can destabilize if rare high-loss samples get extreme weights.
- Compatible with current Lion + EMA + AMP stack.

**Suggested experiment**: 30-min screening run, n_head=2, Lion, SmoothL1 base, APW with α_max=0.5, linear anneal 0→0.5 over first half of training. Compare val_avg/mae_surf_p against 40.27 baseline.

**Taste scores**: Mechanistic grounding: 3 (targets specific identified gap — sample-conditional shaping), Research-state value: 4 (either confirms new axis worth pursuing or closes it decisively), Execution value: 3 (cheap screening, minimal code change, directly targets paper metric).

---

### H2: Quantile Huber Loss (Expectile Regression)

**What it is**: Replace SmoothL1 with quantile Huber (expectile) loss at τ=0.6–0.7, which asymmetrically penalizes over-predictions more than under-predictions (or vice versa).

**Why it might help**: Surface pressure errors are physically asymmetric — suction-side pressure underestimation causes more aerodynamic force error than overestimation. The current symmetric SmoothL1 treats both directions equally. An asymmetric loss that harder-penalizes underprediction on negative-p extremes could improve surface-p accuracy specifically.

**Mechanism**: `L_τ(r) = τ·max(r,0) + (1-τ)·max(-r,0)` where r = pred - target. At τ=0.5 this is MAE. At τ=0.6 it penalizes underprediction 1.5× harder. Combine with Huber smoothing near zero: `L_τ,β(r) = τ·H_β(r) if r≥0 else (1-τ)·H_β(r)`.

**Key reference**: Standard distributional RL loss (QR-DQN, Dabney et al. 2018); adapted for regression in Expectile Regression networks.

**Implementation notes**:
- τ should be applied in normalized space (model output space), not physical space, so it's not confounded by y_std scaling.
- Key risk: the asymmetry must be calibrated to physical error direction. Test τ=0.5 (baseline), τ=0.55, τ=0.6 — do not go above 0.7.
- This cannot reweight channels individually (Bernoulli coupling lesson). Apply same τ to all 3 channels.
- β = 0.1 (same as current SmoothL1 β) is the right starting point.

**Suggested experiment**: Single screening run with τ=0.6, β=0.1 vs. current τ=0.5 (SmoothL1). If τ=0.6 wins, try τ=0.55 to bracket optimum.

**Taste scores**: Mechanistic grounding: 2 (plausible but the asymmetry direction is not validated), Research-state value: 2 (easy to run but tells us less than H1), Execution value: 3 (trivial code change, fast screen).

---

## Category 2: Architecture Beyond Hyperparameters

### H3: Cautious Lion Optimizer

**What it is**: One-line modification to Lion — mask the update step for each parameter where `sign(gradient) ≠ sign(momentum)`, preventing the signed momentum from overriding the current gradient signal when they disagree.

**Why it might help**: Lion's confirmed amplification of gradient noise (coord-jitter regression, +4.86%) is because the sign operation discards magnitude and applies full momentum in the signed direction even when the gradient signal is weak or reversed. Cautious Lion adds a binary gate that requires current gradient and momentum to agree before applying the update. This directly addresses the mechanism that hurt coord-jitter without abandoning Lion's overall efficiency gain (−19.8% val over AdamW).

**Mechanism**: In Lion update rule `θ_{t+1} = θ_t - lr * sign(β₁*m_{t-1} + (1-β₁)*g_t)`, Cautious variant additionally multiplies by `I[sign(g_t) == sign(β₁*m_{t-1} + (1-β₁)*g_t)]`. This is equivalent to setting the step to zero whenever gradient and interpolated momentum disagree in sign.

**Key paper**: Cautious Optimizers (arxiv 2411.16085, ICLR 2026) — 1 line of code, reports 1.5–3× speedup on LLM/vision tasks; mechanistically motivated by sign-compression instability.

**Implementation notes**:
- Implementation is literally 3 lines added to Lion's `step()` method:
  ```python
  # After computing update = sign(interp)
  cautious_mask = (g * update > 0).float()
  cautious_mask = cautious_mask / cautious_mask.mean().clamp(min=1e-3)
  update = update * cautious_mask
  ```
- The normalization `/ mean()` preserves expected update magnitude so lr doesn't need retuning.
- Keep all other Lion hyperparameters identical to current best (lr, β₁, β₂, wd).
- This is orthogonal to EMA — EMA still runs on top.

**Suggested experiment**: Drop-in replacement of Lion with CautiousLion in train.py. 30-min screening run identical to current best config except optimizer class. If it wins, it opens the door to re-testing coord-jitter augmentation (which failed under standard Lion due to noise amplification).

**Taste scores**: Mechanistic grounding: 4 (directly addresses the confirmed Lion+noise failure mode with a mechanistically precise fix, strong external evidence at ICLR 2026), Research-state value: 4 (if it wins, reopens the augmentation axis; if it loses, confirms noise amplification is not the bottleneck), Execution value: 4 (trivial code change, 30-min screen, highest leverage per unit compute of any idea here).

---

### H4: SwitchEMA — Periodic EMA-to-Weights Swap

**What it is**: Periodically replace the trained model weights with the current EMA snapshot, then continue training from there, effectively restarting exploration from the flat basin that EMA occupies.

**Why it might help**: The current EMA(0.999) tracks a smoothed version of the weights but the live weights continue descending in loss curvature. SwitchEMA periodically resets the live weights to EMA's position, which lives in a flatter, more generalizable basin. This provides "free" exploration-exploitation cycling without changing learning rate or model architecture. The under-budgeting finding (best_epoch=final_epoch always) suggests the model has not converged, and SwitchEMA could accelerate convergence by periodically restarting from smoother parameter territory.

**Mechanism**: Every K steps (K=500 is recommended in the paper), copy `ema_weights → model.weights`, then reset the EMA buffer from the new model weights. This costs zero FLOPs and requires no architectural change.

**Key paper**: SwitchEMA (arxiv 2402.09240) — "Enhancing Sharpness-Aware Minimization via SwitchEMA"; reports consistent improvement over standard EMA across vision transformers and language models.

**Implementation notes**:
- Implementation requires modifying the EMA update loop in `train.py`:
  ```python
  if global_step % switch_interval == 0:
      # Copy EMA weights to model
      for (name, param), ema_param in zip(model.named_parameters(), ema_model.parameters()):
          param.data.copy_(ema_param.data)
      # Reset EMA buffer to current model (which is now == EMA)
      ema_model.load_state_dict(model.state_dict())
  ```
- switch_interval: 500 steps recommended; with 30-min budget and ~200 steps/epoch, this is ~2-3 switches per run.
- After the switch, EMA momentum will be high (0.999) so it will take ~1000 steps to diverge significantly again. This is expected behavior.
- Validation and checkpointing continue to use EMA weights (no change to eval protocol).
- Key risk: if switch_interval is too small, the model never explores beyond EMA; if too large, it misses the benefit. 500 steps is conservative.

**Suggested experiment**: 30-min screening run with SwitchEMA switch_interval=500. No other changes vs. current best.

**Taste scores**: Mechanistic grounding: 3 (mechanism is clear; paper evidence is strong; but the benefit depends on under-budgeting being the core bottleneck), Research-state value: 3 (if it wins, tells us the model was trapped in sharpness artifacts; if it loses, suggests EMA is already sufficient), Execution value: 3 (minimal code, 30-min screen, zero compute overhead).

---

### H5: GeoTransolver — Cross-Attention to Multi-Scale Geometry Context

**What it is**: Augment Transolver's physics-slice self-attention with a cross-attention branch that reads from a learned multi-scale geometry encoding (foil shape, boundary conditions, global flow parameters), inspired by GeoTransolver's GALE mechanism.

**Why it might help**: The current Transolver self-attention pools over all mesh nodes globally, treating geometry purely as part of node features (dims 0-23). The GALE mechanism in GeoTransolver extracts a separate geometry/BC summary via a learnable encoder and injects it as additional keys/values in the physics attention. This gives the model explicit access to global aerodynamic context (Re, AoA, NACA codes, gap/stagger) as structured context, not just per-node features. This targets the OOD geometry splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) where the model must generalize to unseen foil shapes.

**Mechanism**: 
1. Extract a global context vector from the sample-level features (Re, AoA, NACA, gap, stagger) via a small MLP → pooled representation c_global ∈ R^d.
2. Extract surface-node summary via masked average pooling of surface node features → c_surface ∈ R^d.
3. Concatenate [c_global, c_surface] → project to K_cross key/value pairs.
4. In each Transolver attention block, compute cross-attention from slice queries Q to [K_self; K_cross], [V_self; V_cross].

**Key paper**: GeoTransolver / GALE (arxiv 2512.20399) — "Geometry-Aware Local-global Encoder for Neural Operators on Irregular Meshes"; reports improvement on airfoil and cavity benchmark tasks.

**Implementation notes**:
- The cross-attention adds O(S × K_cross) compute per layer where S=slice_num=32. With K_cross=8, overhead is ~25% per attention block.
- Critical: the geometry context must be extracted from the **raw (unnormalized?) or normalized** features consistently. Use normalized x as input to the geometry encoder.
- Foil-2 features (dims 18-23) are all zero for single-foil samples — the geometry encoder must handle this gracefully (it will if we use a shared MLP that simply produces near-zero context for zero inputs).
- Start with K_cross=4 (very cheap) and only scale up if the mechanism proves effective.
- This requires architectural changes to Transolver's PhysicsAttention class in train.py.

**Suggested experiment**: Screening run with K_cross=4 geometry cross-attention injected at all Transolver layers. 30-min budget. If val improves, scale to K_cross=8 and add multi-scale (zone-level) geometry encoding.

**Taste scores**: Mechanistic grounding: 3 (targets specific failure mode on OOD geometry splits with a mechanistically motivated architectural change), Research-state value: 3 (would confirm whether global geometry context is missing from current model), Execution value: 2 (requires non-trivial architectural changes; higher implementation risk than H1-H4).

---

### H6: Transolver++ Local Adaptive Token Merging

**What it is**: Replace Transolver's static physics-slice assignment with a learnable local token merging mechanism that groups nearby mesh nodes into adaptive tokens, capturing local spatial structure that the global slice pooling misses.

**Why it might help**: Transolver's PhysicsAttention assigns nodes to slices by learned dot-product routing, but this is purely feature-based — there is no locality bias. Transolver++ (arxiv 2502.02414) adds a learnable local token merging step before slice assignment, which groups spatially adjacent nodes together. For CFD, the boundary layer (thin region near foil surface with high gradients) is critical, and these nodes are sparsely distributed across slices without locality. Local merging ensures the boundary layer is represented coherently.

**Mechanism**: Before slice assignment, apply a GNN-style local aggregation (k-nearest neighbors in mesh space, k=4) to produce locally-smoothed node tokens. These smoothed tokens go into the slice router, not the raw node features. The rest of Transolver is unchanged.

**Key paper**: Transolver++ (arxiv 2502.02414) — "Local Adaptive Mechanism for Million-Scale Geometries"; reports 13% gain on 6 PDE benchmarks relative to Transolver baseline.

**Implementation notes**:
- The k-NN graph must be precomputed per sample (mesh topology varies). Cost is O(N log N) per sample — acceptable for 74K–242K nodes but cannot be done naively on GPU.
- Use spatial coordinates (dims 0-1 of x) to build k-NN. Do this in the DataLoader collation or as a precompute step at the start of each epoch.
- The local aggregation itself is a simple message passing: `h_i = MLP(cat(x_i, mean_j∈NN(x_j)))`. MLP can be very small (2-layer, d=32).
- Since data loaders are read-only, this must be implemented entirely in train.py's forward pass, computing the k-NN graph on-the-fly per batch.
- Risk: computing k-NN for 242K nodes on-the-fly every forward pass may be too slow. Profile first. If it is, fall back to a coarser approach using the zone structure (zone ID is implicitly encoded in the node positions).

**Suggested experiment**: Profile k-NN computation cost first (single forward pass timing). If <10% overhead, proceed with full 30-min screening run. If >20% overhead, pivot to zone-based aggregation using implicit zone structure.

**Taste scores**: Mechanistic grounding: 3 (targets boundary layer coherence via locality; paper reports 13% gain; local structure is plausibly missing), Research-state value: 3 (architectural change with clear mechanism; failure would tell us global slicing is sufficient), Execution value: 2 (implementation risk is moderate-to-high; k-NN compute overhead uncertain).

---

## Category 3: Training Schedule

### H7: Cosine Warm Restarts (SGDR) with Lion

**What it is**: Replace the current flat-LR + linear-warmup schedule with cosine annealing with warm restarts (SGDR), where LR cycles between a high and low value on a fixed period, allowing the optimizer to repeatedly escape sharp minima.

**Why it might help**: The under-budgeting finding (best_epoch=final_epoch) suggests the model has not converged — it is still on the descent slope at budget exhaustion. The previous cosine tail compression experiments (T_max=15/20/25) failed because they tried to compress the schedule into a fixed decay. SGDR is different: it uses full cosine cycles, resetting periodically rather than decaying monotonically. The critical difference from the closed cosine experiments is that SGDR does not decay to near-zero LR — it restarts from a high LR, which has been shown to improve exploration in flat loss landscapes.

**Mechanism**: LR follows `η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(π*t/T_i))` where T_i is the cycle length, doubling each restart (T_mult=2). Starting with T_0=10 epochs and T_mult=2 gives cycle lengths 10, 20, 40 — well-matched to the 30-min budget.

**Implementation notes**:
- PyTorch's `CosineAnnealingWarmRestarts` scheduler implements this directly.
- Set η_min = lr/100 (not 0; Lion with lr=0 is degenerate).
- T_0=10 epochs, T_mult=2 is the standard "SGDR" recipe from the original paper.
- The previous cosine experiments failed at T_max=15/20/25 with a **monotone decay** — SGDR is fundamentally different (periodic restart from high LR). This is not a repeat of a closed axis.
- Checkpoint selection must handle the cycling: save best across all cycles, not just at cycle endpoints.

**Suggested experiment**: 30-min screening run with SGDR (T_0=10, T_mult=2, η_min=lr/100) on current best config. Check that best_epoch is not always final_epoch (if so, warm restarts are helping exploration).

**Taste scores**: Mechanistic grounding: 2 (mechanism is well-known but the connection to this specific bottleneck is loose; under-budgeting ≠ sharp minima necessarily), Research-state value: 2 (hard to distinguish from "another LR schedule tweak" unless best_epoch shifts), Execution value: 3 (trivial code change, 30-min screen).

---

### H8: Lookahead Wrapper on Lion

**What it is**: Wrap the Lion optimizer with Lookahead (k=5, α=0.5), which maintains a "slow weights" copy that interpolates toward the fast weights every k steps, implementing an inner/outer loop dynamic that stabilizes sign-based momentum.

**Why it might help**: Lion's sign compression discards gradient magnitude, which can cause the fast weights to oscillate in directions where the gradient is noisy. Lookahead's slow-weights average over k Lion steps, reducing the sensitivity to individual noisy updates. This is a different noise-stabilization mechanism from Cautious Lion (H3): Cautious masks updates where signs disagree, while Lookahead averages over multiple steps' updates. Both target Lion's noise amplification but via different mechanisms.

**Key papers**: Lookahead optimizer (Zhang et al. 2019); Lion+Lookahead has been empirically explored in several vision transformer papers.

**Implementation notes**:
- Lookahead is a wrapper: `optimizer = Lookahead(Lion(params, lr=...), k=5, alpha=0.5)`.
- Standard k=5, α=0.5 from the original paper is the right starting point.
- EMA is maintained on top of Lookahead's slow weights, not Lion's fast weights — this requires careful implementation to ensure EMA sees the slow-weights updates.
- Alternatively, maintain EMA on the model's fast weights as currently, and let Lookahead operate independently. This is simpler and likely fine.
- Compute overhead: ~0% (Lookahead step is cheap, k=5 means 1 extra backward step per 5).

**Suggested experiment**: Run H3 (Cautious Lion) first. If it wins, H8 becomes lower priority (both target the same mechanism via different paths). If H3 fails, H8 tests whether the noise problem is addressable via averaging rather than gating.

**Taste scores**: Mechanistic grounding: 2 (mechanism is plausible but less precisely targeted than Cautious Lion; both address sign-compression noise), Research-state value: 2 (mostly interpretable only in context of H3 result), Execution value: 3 (trivial code change, 30-min screen).

---

## Category 4: Data-Side

### H9: Masked Node Pre-training as Self-Supervised Warm-Start

**What it is**: Before the main supervised training, run a short pre-training phase where 40% of mesh node features are randomly masked (set to zero), and the model must reconstruct the masked nodes from the visible ones. Then fine-tune on the supervised (x→y) task.

**Why it might help**: The model's current poor OOD geometry performance (`val_geom_camber_rc`, `val_geom_camber_cruise` are hardest splits) suggests it has not learned a robust representation of mesh topology — it is overfitting to the specific node-feature patterns in the training distribution. Masked pre-training forces the model to fill in missing spatial context, learning spatially consistent representations of flow fields. This is directly analogous to BERT masking for language and MAE masking for vision.

**Mechanism**: During pre-training, for each sample, randomly mask 40% of node feature vectors (set x_i = 0) and add a mask token (learned embedding). The model predicts the original x values for masked positions via an auxiliary decoder head. After pre-training converges, remove the decoder head and fine-tune normally.

**Key paper**: Masked GNN Pre-training (arxiv 2501.08738) — "Self-Supervised Pre-training for Irregular Mesh Neural Operators"; reports significant improvement on OOD geometry generalization for airfoil CFD. Also: GiNOT (arxiv 2504.19452) uses masked pre-training for neural operator generalization.

**Implementation notes**:
- Pre-training task: predict x_masked from visible nodes. This is purely self-supervised — no y labels needed.
- Mask ratio 40% is from the Masked GNN paper recommendation for mesh settings (vs. 75% for image MAE).
- Pre-training epochs: 10-15 epochs with a smaller LR (1/10 of fine-tuning LR). Then reduce to standard schedule.
- Decoder head: a single linear layer projecting from d_model → 24 (input feature dim) is sufficient.
- Important: do NOT mask during fine-tuning, only during pre-training. Clean inputs during supervised phase.
- Risk: if pre-training loss converges to trivially predicting zero (since masked nodes are set to zero), add a positional indicator to distinguish masked vs. visible nodes — use a binary mask token added to x.
- Budget concern: pre-training consumes some of the 30-min wall-clock budget. If pre-training takes >10 min, fine-tuning quality suffers. Profile first with 5 epochs of pre-training.

**Suggested experiment**: 5-epoch masked pre-training (40% mask, LR=1e-4) → fine-tune with current best config for remaining wall-clock budget. Compare val_avg/mae_surf_p and specifically check `val_geom_camber_rc`, `val_geom_camber_cruise` for disproportionate improvement (that would confirm OOD geometry hypothesis).

**Taste scores**: Mechanistic grounding: 3 (targets specific OOD geometry failure mode with established pre-training mechanism; strong external evidence), Research-state value: 4 (if split-specific improvement confirms OOD geometry hypothesis, it opens an entire pre-training direction; if not, closes it), Execution value: 2 (requires two-stage training, careful budget allocation, and a decoder head — more engineering complexity than H1-H4).

---

### H10: Zone-Conditional Normalization

**What it is**: Instead of global `(y - y_mean) / y_std` normalization using dataset-level statistics, normalize each sample by its local per-zone statistics (background zone vs. foil zones separately), then denormalize for loss computation.

**Why it might help**: The current global normalization creates a massive dynamic range problem: raceCar single samples at Re=5M have y_std≈2077, while cruise samples at low Re have y_std≈164 — a 13× ratio. In normalized space, the loss is dominated by low-std samples (which are already small magnitude in physical space). Zone-conditional normalization would allow the model to learn a more uniform-scale prediction task across the three domains.

**Mechanism**: At training time, compute per-sample normalization stats from the current batch sample (sample-level standardization). Use `y_i_norm = (y_i - mean(y_i)) / std(y_i)` instead of global `y_mean`/`y_std`. The model predicts in this per-sample normalized space. At inference, denormalize with the sample's own stats.

**Critical constraint**: `data/scoring.py` uses global `y_std * pred + y_mean` to denormalize. If we change normalization in train.py, we must also pass per-sample stats to the scoring function. Since `data/scoring.py` is read-only, the per-sample normalization must be inverted back to global stats before scoring, or we must compute MAE in physical space directly in the loss computation step and skip the scoring module's normalization.

**Implementation notes**:
- The safest path: train in per-sample normalized space, but before calling `scoring.py`, re-apply the global normalization: `pred_global_norm = (pred_phys - y_mean) / y_std`. Then `scoring.py` denormalizes correctly.
- Per-sample `mean(y_i)` and `std(y_i)` must be computed from unmasked nodes only (use mask tensor).
- Risk: per-sample normalization removes the mean of each sample, which may eliminate information about absolute pressure level. This could hurt physically meaningful output.
- Alternative: use per-domain normalization (3 sets of stats, one per domain). This is less aggressive and retains inter-domain structure.

**Suggested experiment**: Per-domain normalization (3 sets of stats computed offline from training data). This is simpler than per-sample and avoids the mean-removal problem.

**Taste scores**: Mechanistic grounding: 2 (dynamic range is a known issue but the effect in normalized space is less clear — the model should learn to handle this via loss weighting implicitly), Research-state value: 2 (result is ambiguous without additional ablations), Execution value: 2 (requires careful handling of scoring.py interface; higher bug risk).

---

## Category 5: Multi-Task / Auxiliary Losses

### H11: Continuity Equation Auxiliary Loss (Divergence-Free Constraint)

**What it is**: Add an auxiliary loss term that penalizes violations of the incompressible continuity equation ∇·u = ∂Ux/∂x + ∂Uy/∂z = 0 at mesh nodes, computed from finite differences of predicted velocity fields.

**Why it might help**: The current model predicts Ux, Uy, p independently at each node with no explicit physical constraints. The Bernoulli coupling lesson showed that reweighting channels individually breaks the pressure-Poisson coupling. However, the continuity equation (∇·u = 0 for incompressible flow) is purely a velocity constraint — it does not involve pressure, so it does not trigger Bernoulli coupling issues. Adding this constraint could regularize the velocity field and indirectly improve pressure accuracy through the physics coupling.

**Mechanism**: Estimate ∂Ux/∂x ≈ (Ux_i - Ux_j) / (x_i - x_j) for neighboring node pairs (i,j). Penalize `||∂Ux/∂x + ∂Uy/∂z||²_2` as an auxiliary loss with weight λ. Start with λ=0.01 (small perturbation to main loss).

**Implementation notes**:
- Requires k-NN graph (same as H6) to identify neighboring nodes for finite differences. Same compute concern applies.
- Alternative: use the zone structure. Background zone has a regular grid — finite differences are easier there.
- Critical: compute the divergence in physical space (denormalized predictions), not normalized space. Normalized space mixing Ux and Uy via different std values makes the divergence physically meaningless.
- The divergence loss applies only to interior (non-surface) nodes — surface nodes have a no-slip condition (u=0), not divergence-free.
- λ must be very small (0.001–0.01) to avoid dominating the main loss. Monitor that main val loss does not regress.
- Budget: k-NN graph computation per step is the bottleneck. If it causes >20% slowdown, this idea is not viable in the current 30-min budget.

**Suggested experiment**: First, profile the k-NN cost. If feasible, run with λ=0.01, divergence loss on interior nodes only.

**Taste scores**: Mechanistic grounding: 3 (physics-motivated, avoids Bernoulli coupling pitfall; continuity is a genuine constraint), Research-state value: 3 (if it helps, confirms physics-informed regularization is a live axis; if not, tells us incompressible constraint is already implicit), Execution value: 2 (k-NN compute overhead is the key unknown; feasibility uncertain without profiling).

---

### H12: Global Flow Property Prediction Auxiliary Task

**What it is**: Add auxiliary prediction heads that predict global aerodynamic quantities — lift coefficient CL and drag coefficient CD — from the predicted pressure and velocity fields via integration, and train these heads with a secondary loss alongside the main nodal-MAE loss.

**Why it might help**: CL and CD are integrated quantities over the foil surface (pressure integration + skin friction). By explicitly training the model to predict these via integration from its own node-level predictions, it is incentivized to get the surface pressure field right in a globally-consistent way — not just minimizing local errors. This targets the primary metric (surface-pressure MAE) directly, because CL/CD regression pulls the surface-p accuracy in a physically meaningful direction.

**Mechanism**: After model forward pass, compute `CL_pred = ∫ p_pred · n_z ds` over surface nodes (discrete sum with arc-length weights from saf features, dims 2-3). Similarly for CD. Add `λ * (|CL_pred - CL_gt| + |CD_pred - CD_gt|)` to the main loss. CL/CD ground truth is computed at training time from the y labels — no additional data needed.

**Implementation notes**:
- Arc-length weights are available from dims 2-3 of x (saf features). Use these as ds approximation.
- CL/CD require surface node selection (is_surface mask). Ensure padding mask is applied.
- CL/CD will be in physical units — normalize by chord length (implicit in the geometry; use max(x_z) - min(x_z) of surface nodes as proxy).
- λ=0.1 is a starting point — CL/CD are order-1 quantities in normalized form, so the scale relative to node-MAE needs empirical tuning.
- Risk: CL/CD are available only for training samples where y is known. This is fine — the auxiliary loss only applies during training.
- Risk: For single-foil samples (no tandem coupling), CL/CD derivation is straightforward. For tandem samples, you get CL/CD per foil or total. Total is simpler to implement.

**Suggested experiment**: Compute CL/CD from ground truth y at batch time (cheap, no precompute needed). Add auxiliary loss with λ=0.05. If val_avg/mae_surf_p improves, try λ=0.1.

**Taste scores**: Mechanistic grounding: 3 (CL/CD integration directly incentivizes surface-p accuracy; the physical chain is clear), Research-state value: 3 (addresses whether global physics integration helps local accuracy), Execution value: 3 (no k-NN needed; just masked surface summation; low implementation complexity).

---

## Category 6: Ensembling and SWA

### H13: Multi-Seed Ensemble (2–3 Seeds)

**What it is**: Train 2-3 independent runs with different random seeds and average their predictions at inference time.

**Why it might help**: The standard deviation across seeds is unknown for this setup — one seed has been explored per configuration. If the per-seed variance is high (which is plausible given the small dataset size: 599+457+443=1499 training samples), ensembling could reduce variance error substantially. Test-time cost is linear in ensemble size, but there is no constraint on inference cost.

**Mechanism**: Train n=2 or 3 runs with seeds from `random.randint(0, 2^31)`. Average denormalized predictions before computing MAE. No architecture or training changes.

**Implementation notes**:
- This is the cheapest possible experiment: literally run the current best config twice more.
- Ensemble averaging can be done post-hoc: save each run's test predictions to disk, average, then score.
- The current train.py already saves model artifacts to W&B — retrieve the best checkpoints from each seed, load all at test time, average predictions.
- Expected gain: if per-seed std on test_avg/mae_surf_p is σ, ensemble of n seeds reduces variance by 1/n. For n=3, expected improvement is ~1/√3 ≈ 42% variance reduction. Whether this translates to mean shift depends on seed bias.
- This is a confirmation experiment, not a discovery experiment — it tells us about seed variance but doesn't open new directions.

**Suggested experiment**: Run 2 additional seeds of current best config (n_head=2, Lion, SmoothL1, AMP, warmup). Average test predictions. Compare test_avg/mae_surf_p of ensemble vs. best single-seed 33.60.

**Taste scores**: Mechanistic grounding: 2 (mechanism is clear but generic; no specific bottleneck addressed), Research-state value: 2 (tells us about variance but doesn't constrain architecture or training hypotheses), Execution value: 3 (extremely cheap, zero risk of regression, directly improves test metric if seeds are diverse).

---

## Category 7: Test-Time Techniques

### H14: SCaSML — Defect Correction with Coarse Residual

**What it is**: At inference time, run a coarse traditional solver (or linearized CFD estimate) to produce a rough flow field, then use the neural surrogate to predict the residual (correction) between the coarse solution and the true field, rather than predicting the full field from scratch.

**Why it might help**: The largest errors in the current model are concentrated in high-gradient regions (boundary layer, wake, interference zones in tandem configurations). A coarse solver gets the bulk flow right but struggles with these regions. Training the model to predict corrections to a known approximate solution is a much easier regression task — the corrections are smaller-magnitude and more spatially concentrated than the full field.

**Key paper**: SCaSML (arxiv 2504.16172) — "Simulation-Calibrated Scientific Machine Learning"; demonstrates neural surrogate accuracy improves substantially when predicting defect corrections to a coarse solver rather than full-field predictions.

**Implementation notes**:
- This requires access to a coarse solver at both training and inference time. The dataset does not include coarse-solver outputs.
- Without a coarse solver, an approximate version: use the global mean flow (inlet conditions) as the "coarse solution" and predict the deviation from that. This is essentially removing the mean flow and predicting perturbations.
- Simpler version: predict `y - y_mean_per_sample` instead of `y - y_mean_global`. This is a form of per-sample mean subtraction that makes the prediction task easier. (Connects to H10.)
- Full SCaSML as in the paper requires running OpenFOAM or a linearized Euler solver for each test sample — this is not feasible in the current training pipeline without significant infrastructure work.
- Verdict: full SCaSML is out of scope given infrastructure constraints. The per-sample mean-subtraction version overlaps with H10 and has ambiguous benefit.

**Assessment**: Not recommended as a standalone experiment in this launch given infrastructure constraints. Partial implementations overlap with H10 (zone-conditional normalization). Defer unless H10 is confirmed effective.

**Taste scores**: Mechanistic grounding: 2 (the full mechanism requires infrastructure not available), Research-state value: 1 (partial implementation is confounded with H10), Execution value: 1 (infrastructure cost too high for this budget).

---

### H15: ATCA — Adaptive Spatial Compute Allocation at Inference

**What it is**: At test time, allocate more Transolver forward passes (additional refinement iterations) to spatially complex regions — boundary layer, wake, foil-foil interference zone — identified by high predicted gradient magnitude.

**Why it might help**: The current model does a single forward pass for all nodes equally. In fluid dynamics, the interesting physics is concentrated in thin boundary layers and interaction zones representing <5% of the mesh nodes. Running 2-3 additional refinement iterations specifically on these nodes (identified cheaply from the first pass's gradient magnitude) could improve accuracy on the critical surface-p nodes without proportional compute cost.

**Key paper**: ATCA (OpenReview xLrA907jVi) — "Adaptive Test-Time Compute Allocation for Neural Operators"; demonstrates spatial compute allocation improves accuracy-compute tradeoff.

**Implementation notes**:
- ATCA requires a spatial decomposition mechanism to identify "hard" regions. For Transolver, the slice assignments can serve as this: identify which slices have high loss (from calibration set) and run those slices through more attention layers at inference.
- Implementation: after first forward pass, compute `||∂pred/∂x||` (gradient of prediction w.r.t. spatial coordinates) as a proxy for complexity. Select top-10% highest-gradient nodes for a second forward pass with those nodes' embeddings refined.
- High implementation complexity — requires a non-standard inference loop.
- Inference-time only: does not affect training, so cannot improve val metric (which uses standard single-pass inference). Would only help if test inference uses ATCA. The current test evaluation in train.py uses standard forward pass.

**Assessment**: ATCA as described would not improve the validation metric (which is what checkpoint selection uses) and would require non-trivial changes to the evaluation harness. Defer.

**Taste scores**: Mechanistic grounding: 2 (mechanism is real but can't improve val metric with current eval protocol), Research-state value: 1 (does not address the bottleneck as defined by checkpoint selection metric), Execution value: 1 (high implementation cost for zero val improvement).

---

## Priority Ranking and Recommended Launch Order

Based on mechanistic grounding, research-state value, and execution value:

| Rank | Hypothesis | Mode | Score (M/R/E) | Why Prioritize |
|------|-----------|------|----------------|----------------|
| 1 | H3: Cautious Lion | Diagnostic | 4/4/4 | Directly addresses confirmed Lion+noise failure; 1-line change; opens augmentation axis if it wins |
| 2 | H4: SwitchEMA | Frontier refinement | 3/3/3 | Zero-overhead EMA cycling; under-budgeting is the #1 confirmed bottleneck |
| 3 | H1: APW Curriculum | Frontier refinement | 3/4/3 | First test of sample-conditional shaping; gap clearly identified in EXPERIMENTS_LOG.md |
| 4 | H12: CL/CD Auxiliary Task | Tier shift | 3/3/3 | Global physics integration directly targets surface-p accuracy; low implementation cost |
| 5 | H9: Masked Node Pre-training | Tier shift | 3/4/2 | High research-state value; OOD geometry hypothesis; 2-stage training is the main complexity |
| 6 | H5: GeoTransolver Cross-Attn | Tier shift | 3/3/2 | Architecture-level attack on OOD geometry; non-trivial implementation |
| 7 | H6: Transolver++ Local Merging | Tier shift | 3/3/2 | Boundary layer coherence; 13% external evidence; k-NN overhead uncertain |
| 8 | H7: SGDR Warm Restarts | Frontier refinement | 2/2/3 | Easy to test; related closed axis (cosine decay) differs in mechanism |
| 9 | H11: Continuity Aux Loss | Diagnostic | 3/3/2 | Physics-motivated; k-NN overhead is the key risk |
| 10 | H13: Multi-Seed Ensemble | Diagnostic | 2/2/3 | Cheap; tells us about variance but not mechanism |
| 11 | H8: Lookahead+Lion | Frontier refinement | 2/2/3 | Secondary to H3; only run if H3 fails |
| 12 | H2: Quantile Huber | Frontier refinement | 2/2/3 | Asymmetry direction unvalidated |
| 13 | H10: Zone-Conditional Norm | Diagnostic | 2/2/2 | Implementation risk; scoring.py interface constraint |
| 14 | H14: SCaSML | Tier shift | 2/1/1 | Infrastructure not available |
| 15 | H15: ATCA | Tier shift | 2/1/1 | Cannot improve val metric with current eval protocol |

---

## Experiment Decision Tree

```
Start: 40.27 val baseline
│
├── Run H3 (Cautious Lion) — 30 min
│   ├── WIN (< 40.27): Merge + re-test H9 coord-jitter (noise was the blocker)
│   │   └── Also run H1 (APW) and H4 (SwitchEMA) in parallel
│   └── FAIL (≥ 40.27): Lion noise is not the bottleneck
│       └── Run H1 (APW) — sample-conditional shaping
│           ├── WIN: APW opens curriculum axis; try α_max=0.7, try combining with H4
│           └── FAIL: Neither noise nor curriculum is the issue
│               └── Run H4 (SwitchEMA) + H12 (CL/CD aux) in parallel
│                   ├── Either WIN: continue down that branch
│                   └── Both FAIL: Move to architecture tier (H5, H6, H9)
│                       └── Run H9 (Masked Pre-training) as highest R-value arch change
│                           ├── WIN on OOD geo splits: confirms pre-training direction
│                           └── FAIL: Run H5/H6 architectural changes
│                               └── FAIL all: Consider full model swap (not Transolver)
```

---

## Stop Conditions

- **H3 (Cautious Lion)**: Stop if val ≥ 42 (worse than baseline by >4%). Continue and run second seed if 38-40.
- **H1 (APW)**: Stop if val ≥ 42 with α_max=0.5. If 38-40, try α_max=0.3 (less aggressive).
- **H9 (Masked Pre-training)**: Stop if val ≥ 41 AND no disproportionate improvement on `val_geom_camber_*` splits. If the OOD splits improve even while val_avg doesn't, continue — it's telling us something.
- **Full direction stop**: If H3, H1, H4, and H12 all fail (all ≥ 40.27), the n_head=2+Lion stack is exhausted. Switch to a fundamentally different model architecture (not a Transolver variant).

---

## Research State Update

**Current best explanation for plateau**: The n_head=2+Lion stack has converged to a local optimum in the hyperparameter space accessible within the 30-min budget. Three mechanisms may be responsible:

1. **Lion noise floor**: Sign-based momentum amplifies gradient noise. Cautious Lion (H3) tests this directly.
2. **Sample-conditional gap**: All loss formulations so far treat samples equally in normalized space. APW (H1) tests this.
3. **Architecture expressivity ceiling**: Current 2-layer, n_head=2 Transolver may lack capacity for the OOD geometry splits. The architecture changes (H5, H6, H9) test this.

**Ruled out**: All hyperparameter variants on Transolver (n_layers, n_head, mlp_ratio, wd, Fourier features K, loss shape in residual-conditional formulation, channel reweighting, surf_weight, Per-Re sampling, cosine tail compression, DropPath).

**Top uncertainties**:
1. Whether the Lion noise amplification is addressable at the optimizer level (H3) or requires a different optimizer entirely.
2. Whether the OOD geometry splits are failing due to missing architectural inductive bias (locality, H6) or missing pre-training diversity (H9).
3. Whether the 30-min budget is the true bottleneck — if best_epoch is always final_epoch, the model is still learning. A longer budget (if `SENPAI_TIMEOUT_MINUTES` allows) might move the needle without any other changes.
