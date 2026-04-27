# Research Ideas — Round 0 — 2026-04-27 22:25

**Summary.** Round 0 baseline is a vanilla Transolver (1M params, AdamW lr=5e-4, MSE in normalized space, surf_weight=10, cosine schedule with no warmup). The single biggest exposed weakness is **scale heterogeneity**: per-sample y-std varies by 40x across the corpus (from ~50 in low-Re cruise to ~2000 in high-Re raceCar), so a global y_std normalization makes high-Re samples dominate the gradient. Combined with no warmup, no AMP, no gradient clipping, no augmentation, and a small slice_num for >200K-node meshes, the low-hanging fruit lies firmly in the loss/normalization/training-recipe layer rather than in exotic architecture swaps. The 8 ideas below front-load that low-hanging fruit and reserve the architectural changes for the back half.

---

### H1: Per-sample y-std loss normalization
- **Bucket:** Loss reformulation
- **Predicted Δ on val_avg/mae_surf_p:** -8% to -18%
- **Why it should work:** The global `y_std` used for normalization is computed once across the entire train corpus, but per-sample y-std varies by ~40x (from ~50 in low-Re cruise to ~2000 in high-Re raceCar single). After normalization with a single `y_std`, the high-Re samples still have squared error magnitudes ~1600x larger than low-Re samples, so each gradient step is dominated by a few extreme-Re cases — exactly the cases the OOD splits do not test on. By rescaling the per-sample loss by that sample's own std (or a Re-conditional std), every sample contributes equally to the gradient, which directly attacks the cross-Re generalization gap. This is the standard "scale-aware loss" lever in regression with heteroscedastic targets and is almost free to implement.
- **Concrete recipe:**
  - In `train.py`, after `y_norm = (y - y_mean) / y_std`, compute per-sample, per-channel std over valid masked nodes: `s = (y_norm * mask.unsqueeze(-1)).std(dim=1, keepdim=True).clamp(min=0.1)` shape `[B, 1, 3]`.
  - Replace `sq_err = (pred - y_norm) ** 2` with `sq_err = ((pred - y_norm) / s) ** 2`.
  - Apply identically to `vol_loss` and `surf_loss`. Validation MAE is unchanged (still in physical space).
  - Keep all other hyperparameters at baseline.
- **Failure-mode signature:** If low-Re samples are now over-weighted, val_avg MAE will improve but val_re_rand will get worse on the high-Re tail. We'd see test/val disagreement on `val_re_rand` and the surface pressure histogram on raceCar would show systematic underprediction.
- **Cross-split prediction:** Largest improvement on `val_geom_camber_cruise` (lowest-Re, currently ignored by the loss); modest improvement on `val_re_rand`; minimal improvement on `val_single_in_dist`.

---

### H2: Linear warmup + cosine to zero, with explicit T_max accounting for timeout
- **Bucket:** Optimization
- **Predicted Δ on val_avg/mae_surf_p:** -3% to -7%
- **Why it should work:** Two compounding training-recipe bugs in the baseline. (1) `CosineAnnealingLR(T_max=MAX_EPOCHS=50)` but the 30-min wall clock typically cuts training at ~20-30 epochs, so the LR never reaches zero — the schedule effectively becomes "5e-4 with mild decay", losing the fine-tuning value of cosine's tail. (2) No warmup means the first few hundred steps are at full lr=5e-4, which is risky on small batches (B=4) with extreme y values, possibly harming the slice-token init. Linear warmup for 5% of steps + cosine to ~0 over the *actually-run* epochs gives the model a better start and a meaningful end. Standard recipe, low risk, compounds with everything.
- **Concrete recipe:**
  - Add `--epochs 25` (so T_max matches what the timeout actually allows; tune to whatever empirically fits in 30 min).
  - Replace `CosineAnnealingLR` with `torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, total_steps=len(train_loader)*MAX_EPOCHS, pct_start=0.05, anneal_strategy='cos', div_factor=25, final_div_factor=1e3)` and call `scheduler.step()` per-batch instead of per-epoch.
  - Alternatively, keep epoch-step and use `LambdaLR` with `min(step/warmup, 0.5*(1+cos(pi*(step-warmup)/(total-warmup))))`.
- **Failure-mode signature:** First-epoch loss is unstable or NaN → warmup is too short. If late-epoch loss diverges, warmup max_lr is too high.
- **Cross-split prediction:** Uniform improvement across all four tracks; slightly larger on `val_single_in_dist` (sanity track tracks training quality most directly).

---

### H3: Smooth L1 (Huber) loss on surface pressure
- **Bucket:** Loss reformulation
- **Predicted Δ on val_avg/mae_surf_p:** -2% to -6%
- **Why it should work:** Surface pressure has heavy tails — leading-edge stagnation points and trailing-edge separation produce spikes orders of magnitude larger than the local mean. MSE squares those errors, so the optimizer becomes obsessed with them at the expense of the bulk of surface nodes. Since the *evaluation metric is MAE*, training with MSE introduces a systematic mismatch: MSE-optimal predictions are not MAE-optimal on heavy-tailed targets. Smooth L1 (Huber with β small) is MAE-like on large errors and quadratic on small ones, aligning training and validation metrics on surface pressure specifically. Volume loss can stay MSE since volume MAE isn't ranked.
- **Concrete recipe:**
  - In `train.py`, replace surface loss with `surf_loss = F.smooth_l1_loss(pred_surf, y_norm_surf, beta=0.1, reduction='sum') / surf_mask.sum().clamp(min=1)` where `pred_surf, y_norm_surf` are gathered with `surf_mask`.
  - Keep `vol_loss` as MSE.
  - Run a 2-cell mini-matrix: `--smooth_l1_beta 0.1` and `--smooth_l1_beta 0.5` (configurable as a CLI flag).
- **Failure-mode signature:** Volume MAE regresses while surface MAE improves — acceptable since we only rank surface_p, but if surface MAE *also* regresses then the bulk-error term was actually carrying useful gradient for surface stagnation.
- **Cross-split prediction:** Largest improvement on splits with high-Re extremes (`val_re_rand`, `val_single_in_dist`); smaller on `val_geom_camber_cruise`.

---

### H4: Surface-only learnable y-mean/y-std + signed surface-distance feature
- **Bucket:** Target/feature engineering
- **Predicted Δ on val_avg/mae_surf_p:** -4% to -10%
- **Why it should work:** Two-pronged. (1) The current normalization uses one `y_mean`, `y_std` for *all* nodes, but surface and volume nodes have very different value distributions (surface pressure has higher variance, lower mean). A separate normalization for surface nodes lets the network specialize on the surface field without the volume distribution dragging the surface predictions toward the volume mean. (2) The model knows which nodes are surface (dim 12) but doesn't know how *far* away from the surface other nodes are — yet boundary-layer physics says distance-to-surface is the dominant geometric variable for the wake. Adding a precomputed distance-to-surface feature (or using saf/dsdf more explicitly) gives the attention richer geometric grounding.
- **Concrete recipe:**
  - Compute per-split `y_mean_surf, y_std_surf, y_mean_vol, y_std_vol` from the train set (one-shot, save in stats.json or compute online on first batch).
  - Predict in a "selectable normalization" space: train two separate output heads (one for surface, one for volume) on their respective normalized targets. At loss time, use the matching normalization. Simplest: split the last block's `mlp2` into `mlp2_surf` and `mlp2_vol` and gate with `is_surface`.
  - Add a "distance to nearest surface node" feature computed per-batch on GPU: `d = torch.cdist(pos, pos[is_surface]).min(dim=-1)`. Append to x as a 25th channel; update model_config `fun_dim = X_DIM - 2 + 1`.
- **Failure-mode signature:** If the two heads are imbalanced (one over- or under-trains), volume MAE will regress dramatically while surface MAE either improves modestly or regresses too. If distance feature has no effect, drop it.
- **Cross-split prediction:** Uniform improvement; biggest on geometry-OOD splits (`val_geom_camber_*`) where new front-foil cambers reshape the boundary layer.

---

### H5: Random Fourier features on (x, z)
- **Bucket:** Sequence/position handling
- **Predicted Δ on val_avg/mae_surf_p:** -2% to -8%
- **Why it should work:** The current model takes raw (x, z) as the only spatial signal. Transolver's slice tokens implicitly group nodes via attention, but the slicing has to learn from scratch what regions look like spatially because positions enter only through a single linear projection. Random Fourier features (NeRF/SIREN-style) inject high-frequency spatial information that lets the network represent sharp pressure gradients near leading edges, separation lines, and the wake-foil interaction zone — all of which involve sub-mesh-resolution structure. This is well-established in implicit neural representations of fields and has been shown to help neural-operator models on irregular meshes specifically (e.g., GNOT, MeshfreeFlowNet).
- **Concrete recipe:**
  - Add a Fourier feature module: `B = torch.randn(num_freq, 2) * sigma` (fixed buffer, not learnable). At forward, replace `x[:, :, :2]` with `[sin(2π x[:,:,:2] @ B^T), cos(2π x[:,:,:2] @ B^T)]`. Concatenate with the original (x, z) so we don't lose them.
  - Run a 3-cell matrix: `(num_freq=16, sigma=2.0), (32, 4.0), (64, 8.0)`.
  - Update `space_dim` in model_config to `2 + 2*num_freq` and `fun_dim` accordingly.
- **Failure-mode signature:** If `sigma` too large, model overfits to local high-freq noise: train loss drops fast but val_geom_camber_* regresses (model memorized training meshes). If `sigma` too small, no effect.
- **Cross-split prediction:** Strongest on `val_geom_camber_*` (where geometry is novel and high-freq spatial reasoning matters); minimal on `val_single_in_dist`.

---

### H6: bf16 + torch.compile + larger effective batch via gradient accumulation
- **Bucket:** Throughput / compounding
- **Predicted Δ on val_avg/mae_surf_p:** -3% to -9%
- **Why it should work:** The baseline runs in fp32, no compile, B=4 — leaving 2-3x throughput on the table on H100/A100-class GPUs. With 96GB VRAM and meshes up to 242K nodes, the current ~1M-param model uses a small fraction of memory. Faster steps mean more epochs in the same wall clock (currently ~25; target 50+), and a larger batch (B=8 effective via accumulation) reduces gradient noise on the variable mesh sizes. More compute per dollar — the most important compounding lever and orthogonal to every other hypothesis.
- **Concrete recipe:**
  - Wrap forward/loss in `torch.cuda.amp.autocast(dtype=torch.bfloat16)`. Note: bf16 doesn't need `GradScaler`. Cast `pred` back to fp32 before MAE accumulation (already in fp64 in scoring).
  - Add `model = torch.compile(model, mode='reduce-overhead')` after instantiation. Disable for debug mode.
  - Increase `--batch_size 8` if it fits, else `--batch_size 4 --grad_accum_steps 2` (compute loss on each microbatch, divide by accum, only call `optimizer.step()` every N).
  - Re-tune cosine T_max for the new step count.
- **Failure-mode signature:** bf16 loss is NaN early → fall back to fp16 with GradScaler or fp32. If `torch.compile` triggers recompiles every batch (variable N_max), use `dynamic=True` or pad to fixed N. Larger batch can over-smooth gradients; if val_avg gets worse, drop to B=4 and rely on AMP only.
- **Cross-split prediction:** Uniform improvement (more epochs = more learning); marginal extra benefit on geom-OOD splits since geometric generalization typically benefits from more iterations.

---

### H7: Z-axis mirror augmentation with output sign flips and tandem-aware filters
- **Bucket:** Regularization / Sampling
- **Predicted Δ on val_avg/mae_surf_p:** -3% to -8%
- **Why it should work:** 2D CFD over airfoils is symmetric about z=0 if you flip the geometry, the AoA sign, the camber sign, and the Uy sign. The dataset only contains *negative* AoA on raceCar (-10° to 0°) and a mix of negative/positive on cruise — so the model never sees an inverted raceCar configuration, but z-mirror would synthesize one for free. Mirror augmentation effectively doubles training data without paying any data-generation cost and forces the model to learn the underlying physics rather than memorize the AoA-sign convention. Critically, it also breaks the spurious correlation between (camber sign × AoA sign) that may help on `val_geom_camber_*` where unseen cambers appear.
- **Concrete recipe:**
  - In the train loop, with probability 0.5 per sample (do it batch-wise for simplicity): flip `z = x[:, :, 1] *= -1`; flip surface arc-length sign in `saf` (`x[:, :, 2] *= -1` — verify the convention from `data/loader.py` or `prepare_splits.py`); flip AoA sign for foil 1 and foil 2 (`x[:, :, 14] *= -1; x[:, :, 18] *= -1`); flip camber sign for both foils (`x[:, :, 15] *= -1; x[:, :, 19] *= -1`); flip stagger sign (`x[:, :, 23] *= -1`); flip y[:, :, 1] (`Uy *= -1`); leave Ux and p alone.
  - Be careful: if NACA camber feature is normalized to [0,1], a sign flip needs to act on the un-normalized value or the normalization has to be re-applied. Apply augmentation *before* x normalization, or apply on raw x and re-normalize.
  - Toggle via `--augment_zmirror 0.5` flag; run with and without to confirm signal.
- **Failure-mode signature:** If saf/camber sign convention is wrong, surface pressure on mirrored samples will be systematically off and val MAE *regresses*. If the model has memorized AoA-direction priors, training loss may briefly increase before recovering.
- **Cross-split prediction:** Strongest improvement on `val_geom_camber_*` (more diverse geometry); modest on `val_re_rand` and `val_single_in_dist`.

---

### H8: Slice_num scaling with width compensation (slice_num=128 or 256)
- **Bucket:** Architecture levers
- **Predicted Δ on val_avg/mae_surf_p:** -2% to -7%
- **Why it should work:** Transolver's `slice_num=64` is the bottleneck for representing distinct flow regions on a 200K-node mesh — that's ~3000 nodes per slice on average. The architecture amortizes computation by attention across slices, but if slices can't isolate features like leading-edge stagnation, separation bubble, wake from foil 1 hitting foil 2, etc., the network is forced to share representations. Doubling slice_num to 128 (or 256) gives the model finer-grained partitioning at minimal parameter cost (slice projection is dim_head × slice_num — roughly 32×64 → 32×128, ~2K extra params). Since attention is over slices not nodes, time/memory overhead is small. With 96GB VRAM available, we can also try a slightly wider model (`n_hidden=192`, `n_head=6`) to increase capacity for the additional slices.
- **Concrete recipe:**
  - Run a 3-cell matrix all in the same PR (one job each): (a) `slice_num=128` with default n_hidden=128; (b) `slice_num=256` with default; (c) `slice_num=128, n_hidden=192, n_head=6`.
  - Add a CLI flag `--slice_num` and pass into model_config; same for n_hidden/n_head if doing (c).
  - Use `--wandb_group slice_num_scan` to group runs.
- **Failure-mode signature:** If slice_num=256 has higher loss than 128 but more params, the partition collapses (slices become redundant). If the wider variant (c) overfits, val_single_in_dist improves but val_geom_camber_* regresses. Watch slice_weight entropy in attention if loggable.
- **Cross-split prediction:** Largest on `val_re_rand` (more flow regimes need more slices); modest on geom-OOD; smallest on single_in_dist.

---

### H9: Pressure-gradient penalty along the surface (∇_s p smoothness)
- **Bucket:** Physics-aware
- **Predicted Δ on val_avg/mae_surf_p:** -2% to -5%
- **Why it should work:** Surface pressure on an airfoil is not arbitrary — it's a smooth function along the arc-length except at the trailing edge (where the Kutta condition forces a jump). The current MSE loss treats every surface node independently, so the model can produce predictions that are correct in MAE but spatially wiggly. A small TV-style or Laplacian penalty on neighboring-node pressure differences encourages locally smooth predictions, which tends to improve MAE on smooth regions (the bulk of the surface) without harming the trailing-edge spike. Crucially, we can compute neighbors cheaply via the signed arc-length feature (saf, dim 2) — sort surface nodes by saf within each foil and penalize first-differences of predicted p.
- **Concrete recipe:**
  - For each batch, gather surface nodes with `is_surface`, sort by `x[..., 2]` (saf) within each sample, compute `Δp_pred = p_pred[1:] - p_pred[:-1]`, `Δp_gt = p_gt[1:] - p_gt[:-1]`. Compute `loss_grad = ((Δp_pred - Δp_gt)**2).mean()`.
  - Add `loss = vol_loss + surf_weight * surf_loss + grad_weight * loss_grad` with `--grad_weight 0.5` (tune).
  - Avoid scoring across foil boundaries: split surface nodes by saf-sign or by another foil-id heuristic. Or use a small `k=5` window of nearest spatial neighbors via `torch.cdist` instead of saf-sorting.
- **Failure-mode signature:** Surface MAE doesn't improve and the trailing-edge prediction gets blurry: penalty is too high. Pressure-gradient histogram on val shows artificially flat predictions.
- **Cross-split prediction:** Modest uniform improvement; slightly larger on cruise (smoother flow regime) than raceCar.

---

### H10: surf_weight sweep with annealing (start at 5, ramp to 30)
- **Bucket:** Loss reformulation / Curriculum
- **Predicted Δ on val_avg/mae_surf_p:** -1% to -4%
- **Why it should work:** `surf_weight=10` is a fixed compromise. Early in training, the model is learning gross flow structure and a high surf_weight forces it to overfit to surface details before the volume field is even coherent. Late in training, the volume field is approximately right but surface pressure still has the most informative gradient — a higher weight at this stage pushes the optimizer where it matters. A simple linear or cosine ramp from 5 → 30 across training matches this intuition. Cheap to run, low risk, compounds with everything.
- **Concrete recipe:**
  - Run a 3-cell matrix: (a) constant `surf_weight=20`; (b) constant `surf_weight=5`; (c) linear ramp `surf_weight = 5 + (30-5) * epoch/MAX_EPOCHS`.
  - Add `--surf_weight_end 30` flag (defaulting to `--surf_weight` if not set) and interpolate per epoch.
  - Group: `--wandb_group surf_weight_sweep`.
- **Failure-mode signature:** With (a) high constant, volume MAE may regress hard, dragging surface predictions in compensation; with (b) low, surface MAE improves slowly and underperforms. Best result will be (c) ramp if the hypothesis holds.
- **Cross-split prediction:** Uniform; slightly stronger on `val_single_in_dist` where in-distribution gradient quality matters most.

---

### H11: Re-conditional FiLM modulation between blocks
- **Bucket:** Target/feature engineering / Architecture
- **Predicted Δ on val_avg/mae_surf_p:** -3% to -7%
- **Why it should work:** Re (dim 13) is currently a single scalar fed in alongside 23 other features and embedded once in `preprocess`. But Re fundamentally controls the *style* of flow — laminar vs turbulent, attached vs separated — and should modulate every layer's behavior, not just appear in the input. FiLM (Feature-wise Linear Modulation) adds two small MLPs that produce per-block (γ, β) from `log(Re)` and apply `γ ⊙ x + β` after each LayerNorm. This is a tiny parameter cost (~2 × n_layers × n_hidden) and gives the network a clean conditional pathway. Closely related techniques are dominant in conditional generative models and recent neural operators (e.g., FNO with channel modulation).
- **Concrete recipe:**
  - In `Transolver.__init__`, add `self.film_re = MLP(1, 64, 2*n_hidden*n_layers, n_layers=1, res=False)`.
  - In `forward`, extract `re = data['x'][:, 0, 13:14].mean(...)` (or take from `data['x'][:, :, 13:14]` since it's constant per sample), compute `gamma_beta = self.film_re(re).reshape(B, n_layers, 2, n_hidden)`.
  - Modify `TransolverBlock` to accept `(gamma, beta)` and apply after `ln_1`: `fx = gamma * ln_1(fx) + beta`.
  - Single run; if signal is positive, scale to multi-feature FiLM (Re + AoA + camber).
- **Failure-mode signature:** If FiLM is too aggressive, low-Re samples get over-modulated and val_geom_camber_cruise regresses. If too weak, no effect — typical of FiLM when init is poor; ensure `gamma` initializes near 1 (`fc.bias.data[:n_hidden] = 1, fc.bias.data[n_hidden:] = 0` and weights small).
- **Cross-split prediction:** Strongest on `val_re_rand` (the test it was designed for); modest elsewhere.

---

### H12: EMA of model weights for evaluation
- **Bucket:** Optimization
- **Predicted Δ on val_avg/mae_surf_p:** -1% to -4%
- **Why it should work:** With B=4 and ~1500 train samples, each epoch sees ~375 noisy gradient steps. EMA (exponential moving average) of weights at decay 0.999 averages out optimization noise and gives a smoother, often-better validation model — at zero parameter cost and trivial compute. Common practice in modern ML training (used in DDPM, semi-supervised learning, distillation). Works especially well in late training when LR is low and the model is oscillating around a local minimum. Compounds with every other hypothesis.
- **Concrete recipe:**
  - Maintain a shadow `ema_model` initialized as a deep copy of `model`. After each `optimizer.step()`: `for p, p_ema in zip(model.parameters(), ema_model.parameters()): p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)`.
  - Validate using `ema_model` instead of `model` (or run validation on both and log both).
  - Save the EMA checkpoint as the artifact if it's better.
  - Run a 2-cell sweep: `decay=0.999` and `decay=0.9995`.
- **Failure-mode signature:** EMA val is worse than raw val for the entire training run → decay too high (EMA is lagging), or model is making large beneficial updates that EMA dampens. Try lower decay (0.99).
- **Cross-split prediction:** Uniform; the gain depends on training-noise level, not split type.
