<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TandemFoilSet Research Ideas — 2026-05-15

Fresh hypotheses for beating the Transolver baseline on `val_avg/mae_surf_p`.
All ideas are feasible within 30-min wall clock / 50-epoch / 96GB VRAM constraints.
No new packages required unless explicitly noted (and pyproject.toml edit called out).
Ranked by estimated impact × probability of success (highest first).

---

## Idea 1 — Surface-Priority Huber Loss with Per-Sample Re-Normalization

**Axis:** Loss reformulation

**Hypothesis:**
The baseline MSE loss in normalized space has two structural problems: (a) the global y_std normalization does not account for per-sample dynamic range, so high-Re samples (y_std ~2000) dominate gradients relative to low-Re samples (y_std ~160), even with domain-balanced sampling; (b) squared-error amplifies outlier nodes disproportionately, degrading surface pressure prediction at separation/reattachment lines. Replacing MSE with Huber loss (delta tuned to ~1.0 in normalized space) while adding a per-sample scale correction factor based on the sample's estimated y_std (derived from `log(Re)` in x[:,13]) will equalize gradient contribution across the Re range and reduce outlier sensitivity. Primary metric direction: lower surface pressure MAE.

**Implementation outline:**
1. In the training loop replace `sq_err = (pred - y_norm)**2` with Huber: `huber_err = F.huber_loss(pred, y_norm, reduction='none', delta=1.0)`.
2. Compute a per-sample Re-based scale weight: `re_scale = (x_norm[:, :, 13].mean(dim=1, keepdim=True).detach().exp().clamp(1, 50))`. Use this to upweight low-Re samples: `huber_err = huber_err / re_scale.unsqueeze(-1)`.
3. Keep the same `vol_loss + surf_weight * surf_loss` aggregation pattern but on the Huber errors.
4. Tune `delta` in {0.5, 1.0, 2.0} — start with 1.0. No other changes to model or optimizer.
5. Add `huber_delta` to the `Config` dataclass with default 1.0 so results are reproducible.

**Expected ROI:** High
**Risk:** Low
**Why now:**
Huber loss is the standard response to heteroscedastic regression targets in CFD surrogate literature (Kashefi & Muller, 2023; Pfaff et al. GNS, 2021). The per-sample Re-scale addresses the specific dataset heterogeneity documented in `program.md` (10× range in per-sample y_std). No architecture changes needed, so the run cost is identical to baseline. Falsification: if Huber with per-sample Re-scale does not improve surface p MAE over all 4 val splits vs. MSE baseline, the normalization/outlier hypothesis is wrong — the problem is elsewhere (architecture capacity or data diversity).

---

## Idea 2 — Larger Transolver Capacity: n_hidden=192, n_layers=6, slice_num=96

**Axis:** Architecture / capacity

**Hypothesis:**
The baseline uses n_hidden=128, n_layers=5, slice_num=64 giving ~2.5M parameters. Given 96GB VRAM and meshes of 74K–242K nodes with batch_size=4, we are severely underfitting in hidden dimension. For a dataset spanning 3 distinct physical domains (raceCar single, raceCar tandem, cruise) with different AoA regimes, Re ranges, and mesh topologies, a larger token representation space should allow more distinct physical mode capture per slice. Specifically: n_hidden=192 (50% wider), n_layers=6 (+1 block for more mixing), slice_num=96 (+50% tokens) should trade ~1.5-2× compute for a meaningful accuracy gain without any algorithmic change. The VRAM budget is the constraint, not algorithmic novelty.

**Implementation outline:**
1. Change `model_config` in `train.py`: `n_hidden=192, n_layers=6, slice_num=96, mlp_ratio=2`.
2. Verify VRAM usage does not exceed ~80GB: estimate ~5M params → forward peak ~40-50GB with batch=4 at N=242K. If OOM, reduce to n_hidden=160, n_layers=6, slice_num=80.
3. Keep optimizer, lr, weight_decay, and surf_weight identical to baseline. Do not change learning rate — a larger model needs at most the same LR with cosine annealing.
4. Run 50 epochs. Check peak_memory_gb in the JSONL log and confirm no OOM on the large-mesh cruise domain batches.
5. If not OOM, this is a pure capacity test with zero hyperparameter uncertainty beyond VRAM.

**Expected ROI:** High
**Risk:** Medium (VRAM risk at B=4 with N=242K; fallback to n_hidden=160 if needed)
**Why now:**
The original Transolver paper (Wu et al., 2024) demonstrated consistent gains scaling hidden dim from 64→256. The dataset has ~1500 training samples across large meshes — the model is almost certainly underfitting. This is the fastest win that doesn't require any research: run baseline + bigger model, compare curves. If the bigger model also underfits (val loss still falling at epoch 50), that points to data or longer training as the bottleneck. Falsification: if val_avg/mae_surf_p does not improve vs. baseline, the architecture is not the constraint — the bottleneck is data diversity or loss quality.

---

## Idea 3 — EMA (Exponential Moving Average) of Model Weights

**Axis:** Optimization / regularization

**Hypothesis:**
Model weights at training termination are a noisy last-iterate — the best checkpoint selection already captures some of this (we save best val MAE), but EMA weight averaging smooths the trajectory and consistently produces a better generalization-vs.-sharpness tradeoff than the last or best single iterate, particularly under limited data regimes. Izmailov et al. (SWA, 2018) and Tarvainen & Valpola (Mean Teacher, 2017) both demonstrate this; a 2024 systematic study (arXiv:2411.18704) confirms EMA is superior to SGD last iterate across generalization, calibration, and noisy-label robustness. With only 1500 training samples and 50 epochs, each checkpoint is noisy. EMA decay ~0.999 should track the model but suppress per-batch noise. At validation time, use the EMA weights instead of the raw weights for checkpoint selection.

**Implementation outline:**
1. After model instantiation, create `ema_model = copy.deepcopy(model)` with `ema_model.requires_grad_(False)`.
2. After each optimizer step, update: `for p_ema, p in zip(ema_model.parameters(), model.parameters()): p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)`. Use `ema_decay=0.999`.
3. For validation and checkpoint selection, evaluate `ema_model` (not `model`) and save `ema_model.state_dict()` when best val metric improves.
4. Add `ema_decay: float = 0.999` to Config. Add `import copy` at top.
5. Final test evaluation also uses `ema_model.state_dict()`.

**Expected ROI:** High
**Risk:** Low
**Why now:**
EMA is a near-zero-cost, drop-in improvement with no architecture changes, no new hyperparameters beyond the decay rate (0.999 is universally robust), and no VRAM increase. The 1500-sample regime is exactly where EMA helps most. This is arguably the highest-leverage/lowest-risk experiment in the list. If EMA doesn't help here, the model is not in the high-variance training noise regime — which is informative. Falsification: if best-val EMA checkpoint is not better than best-val raw checkpoint, training variance is not the bottleneck.

---

## Idea 4 — Signed Distance Field (SDF) Surface Normal Features

**Axis:** Input features / encoding

**Hypothesis:**
The current input x has coordinates (dims 0-1), signed arc-length (dims 2-3), and 8 distance-based DSDF descriptors (dims 4-11), but does not include explicit surface normal direction or distance-to-surface gradient direction. Surface pressure `p` is most sensitive to normal stress at the wall, and Ux/Uy boundary conditions enforce tangential velocity = 0 at the surface. Adding the 2D surface normal vector (estimated from the arc-length gradient direction, or approximated from the spatial position of the nearest surface node) as 2 extra features gives the model an explicit physical cue about wall-normal vs. wall-tangential directions. The Geometric-DeepONet paper (arXiv:2503.17289) showed 32% boundary-layer accuracy improvement from adding SDF derivative constraints — the feature version of this should be cheaper and directionally consistent.

**Implementation outline:**
1. At training time (in the training loop, before normalization), compute a proxy surface-normal feature: for each node in the batch, the signed-distance gradient direction from the arc-length features (dims 2-3) gives an approximate normal direction `(dsdf_x, dsdf_z)` from dims 4-5 of x. Extract and normalize to unit vector: `nx = x[:, :, 4] / (x[:, :, 4:6].norm(dim=-1, keepdim=False) + 1e-8)`, `nz = x[:, :, 5] / (x[:, :, 4:6].norm(dim=-1, keepdim=False) + 1e-8)`.
2. Concatenate these 2 features to x before normalization, expanding X_DIM=24 to 26. Update model `fun_dim=24` (since `fun_dim = X_DIM - 2 = 24`). Important: also update the stats normalization slices to handle the 26-dim input — simplest is to compute running stats for the new features inline before training starts.
3. Add `--use_surface_normal` flag to Config (bool, default False). When True, apply the feature augmentation.
4. Verify that the surface normal proxy is zero for non-surface nodes (i.e., multiply by `is_surface.unsqueeze(-1).float()`).
5. Alternative simpler approach: use x dims 2-3 (saf) derivative proxy — no stats change needed, just pass as additional un-normalized feature with its own normalization entry.

**Expected ROI:** Medium
**Risk:** Medium (feature engineering; proxy quality matters)
**Why now:**
Physics-aware features have strong precedent in CFD surrogates (Boundary GNN paper, Geometric-DeepONet). The current DSDF features encode distance-based geometry but not directional/orientational information relative to the wall. Surface pressure is fundamentally a wall-normal quantity — the model must infer normal direction from position + DSDF, which is indirect. This is a cheap feature addition with clear physical motivation. Falsification: if adding surface normal features does not improve surface p MAE, the model already adequately infers orientation from position + DSDF.

---

## Idea 5 — Domain-Adaptive Normalization: Per-Domain y Statistics

**Axis:** Loss reformulation / data representation

**Hypothesis:**
The global y_mean/y_std normalization bakes in a single scale for all three domains despite their wildly different physical ranges: raceCar single (max |y| ~29K), cruise tandem (max |y| ~7.6K), raceCar tandem (max |y| ~10K). A model trained with global normalization sees cruise samples as roughly 4× smaller signal than raceCar single, making them harder to fit accurately — but cruise is one of the 4 val splits. The balanced domain sampler equalizes sample counts, but not the loss scale per sample. Per-domain normalization (separate y_mean and y_std for each of the 3 domains, identified by the tandem/single indicator in x and by the Re/AoA ranges) would equalize gradient magnitudes across domains. The domain of each sample is inferable from dims 18-23 of x: if x[:,18] == 0 (no second foil AoA), it is single-foil; otherwise check AoA sign and camber range for raceCar vs. cruise.

**Implementation outline:**
1. Before training, compute per-domain y stats from train_ds. Loop through train_ds.files and identify domain by loading each sample's x tensor (dims 18-23 zero → single; dims 14 AoA < 0 → raceCar tandem; else cruise tandem).
2. Compute domain_y_mean and domain_y_std for each of 3 domains (mean/std over all y tensors in that domain).
3. At training time, apply per-domain normalization: for each sample in the batch, determine its domain from x[:,18-23], then normalize y with domain-specific stats. At eval time, denormalize with domain-specific stats before MAE computation.
4. Add `--domain_norm` flag to Config. If False, use global stats (baseline behavior).
5. Important: the scoring path in `data/scoring.py` denorms with global `y_std * pred + y_mean`. With domain-specific norms, you must denorm BEFORE passing to `accumulate_batch`. Make sure `pred_orig` uses the per-domain denorm stats, not global stats.

**Expected ROI:** Medium
**Risk:** Medium (inference-time domain ID is required; mis-identification = wrong normalization)
**Why now:**
The 10× per-sample y_std range documented in `program.md` is a strong signal that global normalization is suboptimal. Domain-adaptive normalization is standard in multi-domain surrogate learning. The domain identification from input features is straightforward (tandem indicator dims 18-23, AoA sign). The cruise tandem split is the hardest OOD split — improving its normalized loss scale should directly help `val_geom_camber_cruise`. Falsification: if per-domain norm does not improve cruise val split vs. global norm, the scale mismatch hypothesis is wrong or the domain ID inference is corrupted.

---

## Idea 6 — Gradient Clipping + Warmup LR Schedule

**Axis:** Optimization

**Hypothesis:**
The baseline uses AdamW with cosine annealing from lr=5e-4, no gradient clipping, and no warmup. With highly variable mesh sizes (74K–242K nodes) and domain-balanced sampling, the early training batches can produce large gradient spikes when a high-Re cruise sample (N~210K) follows several low-Re raceCar single samples. Gradient clipping (max_norm=1.0) is the standard remedy for gradient variance in variable-resolution settings. Additionally, a 5-epoch linear warmup from 1e-5 to 5e-4 before cosine decay stabilizes the PhysicsAttention slice assignment parameters, which use orthogonal initialization and can collapse early. This is a free win that costs nothing in terms of model capacity or VRAM, and addresses a known training instability pattern in attention models over irregular meshes.

**Implementation outline:**
1. Add gradient clipping in the training loop: after `loss.backward()` and before `optimizer.step()`, add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
2. Add a linear warmup scheduler: replace the single CosineAnnealingLR with a `LambdaLR` warmup for 5 epochs followed by cosine:
   ```python
   warmup_epochs = 5
   def lr_lambda(epoch):
       if epoch < warmup_epochs:
           return (epoch + 1) / warmup_epochs
       progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
       return 0.5 * (1 + math.cos(math.pi * progress))
   scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   ```
3. Add `import math` at top of train.py.
4. Add `warmup_epochs: int = 5` and `grad_clip: float = 1.0` to Config.
5. Log the current LR each epoch for diagnostics.

**Expected ROI:** Medium
**Risk:** Low
**Why now:**
Gradient clipping is universally applied in modern transformer training. The absence in the baseline is a gap. Warmup is standard for attention models with learned slice assignment (unstable at high LR without warmup). This is purely an optimization improvement with no model changes — run time is identical. The combination is worth testing as one unit since warmup+clip interact. Falsification: if train loss curves show no spike reduction and val metric does not improve, training stability is not the bottleneck in this setting.

---

## Idea 7 — Multi-Head Output with Separate Pressure Head

**Axis:** Architecture / multi-task

**Hypothesis:**
The baseline uses a single last-layer MLP projecting from n_hidden=128 to out_dim=3, treating (Ux, Uy, p) as a uniform prediction problem. However, pressure satisfies the incompressibility constraint p ∝ -(1/ρ) ∇·(ρu²/2 + ...) while velocity satisfies continuity ∇·u=0. The primary metric is surface pressure MAE — explicitly dedicating a separate, slightly deeper output head for `p` and a shared head for (Ux, Uy) should allow the model to weight pressure features more heavily in its last-layer representation. This mirrors the FUSE paper's (NeurIPS 2024) separate heads architecture and the Transolver paper's `output_fields` parameter which is already in the code but currently unused.

**Implementation outline:**
1. The `Transolver.__init__` already has `output_fields` and `output_dims` parameters but uses a single head in `TransolverBlock(last_layer=True)` which projects to `out_dim=3`. Modify `TransolverBlock` to optionally output multiple streams: add a second `self.mlp_p` head in the last-layer block that takes `fx` and produces `[1]` output.
2. Change `model_config`: set `out_dim=2` for Ux/Uy and add a separate MLP `nn.Sequential(nn.Linear(n_hidden, n_hidden//2), nn.GELU(), nn.Linear(n_hidden//2, 1))` for pressure. Concatenate outputs: `torch.cat([vel_pred, p_pred], dim=-1)`.
3. In the last-layer TransolverBlock, before projecting, add a residual pressure branch: `p_residual = self.pressure_head(self.ln_p(fx))`. Return `{"preds": torch.cat([vel_out, p_residual + vel_base[:, :, 2:3]], dim=-1)}`.
4. Simplest version (lowest risk): just double the last-layer hidden dim for pressure. Change TransolverBlock last-layer MLP for pressure channel to `nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden, 1))` vs `nn.Linear(n_hidden, 1)` for Ux/Uy.
5. Surface weight loss stays at 10.0 but add an additional `pressure_weight=2.0` multiplier on the pressure channel of surf_loss to bias gradient toward p.

**Expected ROI:** Medium
**Risk:** Medium (architecture change, interaction with slice assignment)
**Why now:**
The primary metric is exclusively pressure MAE — dedicating model capacity to pressure is directly aligned with the objective. The output_fields architecture is already scaffolded in the codebase. This is a minimal surgery to the last block. Falsification: if separate pressure head + higher pressure loss weight does not improve `mae_surf_p`, pressure capacity is not the limiting factor — the bottleneck is the shared representation before the output layer.

---

## Idea 8 — Surface-Node Subsampling: Train on Surface-Dense Mini-Meshes

**Axis:** Data / sampling

**Hypothesis:**
With meshes up to 242K nodes and batch_size=4, the effective per-batch node count is up to ~968K nodes. Yet surface nodes are a small fraction (~1% of total — a 2D mesh with foil surface perimeter ~10K nodes out of 242K). The model spends most of its attention and memory budget on interior (volume) nodes. A training pass that subsamples volume nodes (keep 100% of surface nodes, sample K% of volume nodes) would: (a) allow larger effective batch sizes, (b) increase the relative weight of surface nodes in attention computation without changing the explicit surf_weight loss parameter, (c) potentially improve surface pressure accuracy by forcing the model to process more surface-dense batches. At test time, evaluate on full mesh (no subsampling). This is similar to point-cloud subsampling used in mesh-based learning (Pfaff et al. MeshGraphNets, Brandstetter et al. MAGNET).

**Implementation outline:**
1. In the training loop, after loading batch, add a volume subsampling step:
   ```python
   if cfg.vol_subsample < 1.0:
       for b in range(B):
           vol_idx = (~is_surface[b] & mask[b]).nonzero(as_tuple=True)[0]
           keep_n = max(1, int(len(vol_idx) * cfg.vol_subsample))
           drop_idx = vol_idx[torch.randperm(len(vol_idx))[keep_n:]]
           mask[b, drop_idx] = False  # exclude from loss
   ```
2. This does NOT change x or y tensors — just zeros out the mask for dropped volume nodes. The model still processes all nodes (same VRAM), but loss and MAE computation ignores dropped nodes.
3. Add `vol_subsample: float = 1.0` to Config. Test with `vol_subsample=0.3` (keep 30% of volume nodes in loss).
4. Validation always uses full mask (vol_subsample does not apply to eval loops).
5. Expected side effect: surf_loss effectively increases its relative contribution vs. vol_loss since vol nodes are sparser. May need to reduce surf_weight from 10.0 to 5.0 to compensate.

**Expected ROI:** Medium
**Risk:** Medium (subsampling changes gradient signal from volume; may hurt volume MAE)
**Why now:**
Surface MAE is the primary metric; volume MAE is a diagnostic. The current loss equally weights all volume nodes — which are dominated by the wake/far-field where gradients are smoother. Focusing gradient on surface-adjacent nodes is physically motivated. Falsification: if vol_subsample=0.3 improves surface p MAE but hurts volume MAE, this represents a real accuracy/cost tradeoff and should be evaluated on whether the primary metric is net positive.

---

## Idea 9 — Position-Induced Transformer (PiT) Attention Variant

**Axis:** Beyond-Transolver architecture

**Hypothesis:**
The PiT architecture (arXiv:2405.09285, Chen et al. 2024) introduces "position-attention" where the attention matrix is computed from spatial coordinates only (not function values). This creates a purely geometric attention that acts as a spatial filter independent of the current solution, analogous to a learned Green's function. The Transolver PhysicsAttention mixes both geometry (via spatial projection) and function values (via fx_mid and x_mid projections). Splitting these two roles — one geometric attention path and one function-value attention path, combined additively — should give the model more interpretable and generalizable representations, particularly for the geometry-OOD test splits (val_geom_camber_rc, val_geom_camber_cruise).

**Implementation outline:**
1. Add a `PositionAttention` module that computes attention weights from spatial coordinates only: take x[:,0:2] (node position) and produce an attention map over tokens. Use the same slice assignment scheme but from coordinates only: `pos_slice_weights = softmax(linear(pos_embedding) / temp)`.
2. Modify `PhysicsAttention.forward`: run the standard fx-based attention path AND a position-only path. Combine outputs: `out = alpha * fx_attn_out + (1-alpha) * pos_attn_out` where `alpha` is a learnable scalar per head initialized to 0.5.
3. Key implementation detail: the position-only path uses the same slice mechanism but projects from a 2D or enriched position embedding (e.g. Fourier features of position at 4 frequencies, giving 18 dims) rather than the full hidden state.
4. This doubles the number of attention projections in PhysicsAttention but not the depth or number of layers — VRAM impact is ~20%.
5. Keep n_hidden=128, n_layers=5, slice_num=64 (same as baseline) to isolate the architectural effect.

**Expected ROI:** Medium
**Risk:** Medium (architecture change; may need careful init of alpha)
**Why now:**
PiT demonstrated SOTA on Darcy flow and Navier-Stokes benchmarks. The geometry-OOD splits are the hardest val tracks — a model that learns to attend based on spatial position rather than learned function values should generalize better to unseen foil cambers. The combination of physics-attention (function values) + position-attention (geometry) is more principled than function-only attention for operator learning. Falsification: if position-attention does not improve the geometry OOD splits specifically, the generalization bottleneck is not attention geometry.

---

## Idea 10 — Fourier Positional Encoding of Node Coordinates

**Axis:** Input features / encoding

**Hypothesis:**
The baseline passes raw node coordinates (x, z) as the first 2 input dimensions. Neural networks benefit from sinusoidal/Fourier feature encodings of continuous coordinates because MLPs struggle to represent high-frequency functions from raw coordinates (Tancik et al. 2020, "Fourier Features Let Networks Learn High Frequency Functions"). For CFD boundary layers, the solution has high spatial frequencies near the wall (pressure gradients steepest at the surface, velocity changes from 0 to freestream over thin boundary layer). Replacing raw (x, z) with Fourier features at K=8 frequencies (giving 32 dims) should give the model better spatial resolution near the wall. The preprocess MLP `fun_dim = X_DIM - 2 = 22` would become `fun_dim = 22 + (32 - 2) = 52`, with the Fourier features replacing the raw (x, z) coordinates.

**Implementation outline:**
1. Define a Fourier feature encoder: `B_matrix = torch.randn(2, K) * sigma` (fixed, not learned; sigma=1.0). Feature: `[sin(2π x B), cos(2π x B)]` where x is the 2D coordinate. This gives 2*K = 16 features for K=8 frequencies.
2. In train.py, before normalization, compute Fourier features from `x[:, :, 0:2]` and replace (or append to) the raw coordinates. If appending: new input dim = 24 + 2*K = 40 for K=8; update `model_config["fun_dim"] = 38` and normalize only the non-coordinate dims.
3. Alternative: replace raw (x,z) with Fourier features only, keeping X_DIM=22+2*K=38 for K=8.
4. Key: do NOT apply z-score normalization to Fourier features (they are already bounded to [-1,1]); handle by masking those dims in the x_std division.
5. Add `fourier_k: int = 0` to Config (0 = disabled, same as baseline). Test with `fourier_k=8`.

**Expected ROI:** Medium
**Risk:** Low
**Why now:**
Fourier features are standard in neural operator literature (FNO, Galerkin Transformers) and address a well-documented limitation of MLPs on coordinate regression. The boundary layer high-frequency structure is exactly where this helps. Zero architecture change; just a feature preprocessing step. Falsification: if Fourier coordinate features do not improve surface p MAE, the model already encodes high-frequency spatial structure through the DSDF features.

---

## Idea 11 — Auxiliary Lift/Drag Coefficient Regression Head

**Axis:** Multi-task / auxiliary

**Hypothesis:**
Adding an auxiliary head that predicts global aerodynamic coefficients (Cl, Cd) computed from the surface pressure field acts as a regularizer that forces the model to learn physically consistent surface pressure distributions. Cl and Cd are computed as integrals over the surface pressure — if the model predicts them correctly, the surface pressure distribution must be correct at an aggregate level. This is related to the FUSE (NeurIPS 2024) approach of joint field + parameter prediction. The pseudo-labels for Cl/Cd can be computed from the training data's ground-truth y (pressure channel on surface nodes): `Cl = mean(p_surf * nz) * 2 / Re^(1/2)` or more simply just `mean(p[is_surface])` as a surrogate lift proxy.

**Implementation outline:**
1. After the main Transolver output, add a global pooling step: `surf_pool = (fx * surf_mask.unsqueeze(-1)).sum(dim=1) / surf_mask.sum(dim=1, keepdim=True).clamp(min=1)` where `fx` is the pre-output hidden state of the last block.
2. Add a small MLP head: `self.coeff_head = nn.Sequential(nn.Linear(n_hidden, 32), nn.GELU(), nn.Linear(32, 2))` predicting (mean_surf_p, std_surf_p) of the pressure field.
3. Compute target: `mean_surf_p_true = (y[:, :, 2] * surf_mask).sum(dim=1) / surf_mask.sum(dim=1).clamp(min=1)` and similarly for std.
4. Add aux loss: `aux_loss = F.mse_loss(coeff_pred[:, 0], mean_surf_p_true_norm)` with weight 0.1.
5. Total loss: `loss = vol_loss + surf_weight * surf_loss + aux_weight * aux_loss`. Add `aux_weight: float = 0.1` to Config.
6. The coeff_head attaches to the last block's hidden state before the output projection — need to expose `fx` from `TransolverBlock.forward` when `last_layer=True`.

**Expected ROI:** Medium
**Risk:** Medium (requires accessing intermediate hidden state; minor architecture surgery)
**Why now:**
Physics-integrated auxiliary supervision is a strong prior in CFD surrogates. Mean and variance of surface pressure are low-dimensional quantities that integrate over the full surface — predicting them correctly forces the model to have a globally consistent pressure distribution. This is particularly valuable for the geometry OOD splits where the local solution can drift. Falsification: if the aux loss reduces train loss but surface p MAE does not improve, the auxiliary supervision is not adding useful signal — the model already learns consistent global features.

---

## Idea 12 — Curriculum Learning: Easy-to-Hard Re Ordering

**Axis:** Data / training

**Hypothesis:**
The dataset has strong Re-dependent difficulty: low-Re flows (laminar, smooth gradients) are easier to fit than high-Re flows (turbulent-like, steep boundary layers, large value ranges). Starting training with low-Re samples (Re < 500K) and gradually introducing high-Re samples (Re up to 5M) should allow the model to learn the flow structure before needing to handle extreme value ranges. The TandemFoilSet dataset paper (ICLR 2026) explicitly tested curriculum learning as one of their approaches — worth reproducing and potentially improving. Re is directly available in the input features as `log(Re)` at dim 13 of x.

**Implementation outline:**
1. Pre-sort train samples by Re: load all train sample files, extract `x[0, 13]` (log Re at any node — it's constant per sample), and sort indices from low to high Re.
2. Implement a curriculum sampler: for epoch e of E total, use samples with log(Re) < percentile(e/E). Specifically: for the first 20% of epochs, use only the bottom 40% of Re; for 20-60% of epochs, use bottom 70%; for 60-100% of epochs, use all samples.
3. Implement as a custom `SubsetWeightedRandomSampler` in train.py that updates its index pool each epoch. Keep domain balancing within each epoch's active pool.
4. Alternative simpler version: start with batch-level soft curriculum — add a curriculum weight per sample proportional to `exp(-k * log_re / max_log_re)` where k decays from 2 to 0 over training. No need to change the sampler, just multiply sample_weights.
5. Add `curriculum: bool = False` to Config. When enabled, apply the curriculum schedule.

**Expected ROI:** Medium
**Risk:** Medium (curriculum schedule is a hyperparameter; wrong pacing may hurt)
**Why now:**
The dataset authors tested curriculum learning in their ICLR 2026 paper, suggesting it was worth attempting. The Re range is an order of magnitude — without curriculum, the model sees extreme-Re samples from epoch 1. The gradient signal from high-Re samples (loss ~10×) can overwhelm low-Re samples early in training, causing instability. Falsification: if curriculum learning does not improve vs. random sampling (which already has domain balance), the Re-based difficulty ordering is not the bottleneck.

---

## Idea 13 — SpiderSolver-Style Multi-Scale Tokenization (Coarse + Fine Slices)

**Axis:** Beyond-Transolver architecture

**Hypothesis:**
The SpiderSolver (OpenReview 2025) uses a two-level tokenization: coarse tokens over the full domain + fine tokens localized near boundaries. For the TandemFoilSet with overset mesh structure (coarse background zone + fine foil zones), this mirrors the physical mesh design. The current Transolver uses a single slice_num=64 across all nodes — all physical scales compete for the same 64 token slots. A two-level token pool (32 coarse tokens from all nodes + 32 fine tokens from surface-adjacent nodes with `is_surface==True`) would give the model dedicated capacity for boundary-layer representation without losing global domain context.

**Implementation outline:**
1. Modify `PhysicsAttention` to accept an `is_surface` mask. Compute two separate slice assignments: `coarse_weights` from all nodes (softmax over 32 slices), `fine_weights` from surface nodes only (softmax over 32 slices, zero for non-surface nodes).
2. Compute two token pools: `coarse_tokens = einsum(fx_mid, coarse_weights)` (from all N nodes), `fine_tokens = einsum(fx_mid * is_surface_mask, fine_weights)` (from surface nodes only).
3. Concatenate token pools: `tokens = cat([coarse_tokens, fine_tokens], dim=-2)` → shape [B, H, 64, D]. Run attention over all 64 tokens jointly.
4. Broadcast back: `out = einsum(tokens[:, :, :32], coarse_weights) + einsum(tokens[:, :, 32:], fine_weights)`.
5. Pass `is_surface` to `PhysicsAttention.forward` by threading it through `TransolverBlock` and `Transolver.forward`. This requires model signature change: `model({"x": x_norm, "is_surface": is_surface})`.
6. Keep slice_num=64 total (32+32) to match baseline token budget. VRAM impact is negligible since token count is the same.

**Expected ROI:** Medium
**Risk:** High (significant architecture surgery; is_surface threading through model; debugging effort)
**Why now:**
The physical motivation is strong: boundary-layer physics lives in the fine tokens, far-field in coarse tokens. But the implementation requires non-trivial surgery to the attention module and model forward pass. Best run after the lower-risk ideas (1-8) have been tried. Falsification: if coarse+fine tokenization does not improve surface p MAE over flat 64-token baseline, the model's bottleneck is not in token specialization.

---

## Idea 14 — Label Smoothing / Soft Targets via Laplacian Smoothing of Ground Truth

**Axis:** Loss reformulation / physics-informed

**Hypothesis:**
CFD solutions are smooth functions — the velocity and pressure fields satisfy PDEs and are differentiable almost everywhere (except at shocks, but we have low-speed incompressible flow). The ground truth is not perfectly smooth at the discrete mesh level due to solver discretization and numerical diffusion. Applying mild Laplacian smoothing to the target y before computing MSE loss acts as a physics-consistent regularizer: it biases the model toward smoother predictions that respect the underlying PDE character of the solution. This is related to the concept used in PDE-constrained learning (Raissi et al. PINN style, but as a target smoother rather than loss term). For the surface pressure channel specifically, smooth priors reduce noise in high-curvature regions of the foil.

**Implementation outline:**
1. Implement a simple per-sample 1D Laplacian smoother on surface nodes ordered by arc-length. Since arc-length saf (dims 2-3) orders surface nodes: sort surface nodes by `x[:, :, 2]` to get arc-length order, apply 1D Gaussian smoothing with kernel width 3-5 nodes.
2. Only smooth the pressure channel (channel 2) of y for surface nodes; leave volume nodes and Ux/Uy unchanged.
3. Blend with original: `y_smooth_surf_p = alpha * y_smooth + (1-alpha) * y_original` where `alpha=0.1` (gentle smoothing). At alpha=0 this is exactly baseline.
4. Apply only during training loss computation (not during val/test MAE accumulation, which always uses original y).
5. Add `label_smooth_alpha: float = 0.0` to Config. This is a pure training regularizer — adds ~5% compute overhead for the sort operation.

**Expected ROI:** Low-Medium
**Risk:** Medium (wrong kernel width or alpha could bias predictions toward over-smoothed solutions)
**Why now:**
Smooth targets in PDE settings help prevent the model from fitting high-frequency numerical noise. For CFD with finite-volume solvers, the ground truth has mesh-resolution-dependent numerical diffusion. The surface pressure in particular is sensitive to local geometry curvature — the solver introduces local discretization artifacts that smooth targets would de-emphasize. Falsification: if label smoothing degrades surface p MAE (model needs fine details), the solution fields have meaningful high-frequency information the smoothing destroys.

---

## Idea 15 — Learnable Per-Channel Output Scaling (Scale-and-Shift Head)

**Axis:** Architecture / optimization

**Hypothesis:**
The baseline output is a single linear projection from n_hidden=128 to 3 channels, where all channels share the same final linear transformation structure. The three physical quantities (Ux, Uy, p) have fundamentally different scales and behaviors: pressure p is a scalar, Ux is the streamwise component (large magnitude), Uy is the cross-flow (small magnitude near zero). After the shared hidden representation, learning separate scale/shift parameters per channel — analogous to conditional layer normalization or AdaIN — would allow the model to explicitly calibrate its output scale per field. This is a zero-cost augmentation of the final projection: replace `linear(n_hidden, 3)` with `linear(n_hidden, 3)` + learnable `(scale, shift)` per channel initialized to (1, 0).

**Implementation outline:**
1. In the last-layer `TransolverBlock.mlp2` sequential, after the final linear, add `nn.Parameter` tensors: `self.out_scale = nn.Parameter(torch.ones(out_dim))` and `self.out_shift = nn.Parameter(torch.zeros(out_dim))`.
2. In the last-layer `forward`, apply: `return mlp2_out * self.out_scale + self.out_shift`.
3. Initialize out_scale=1.0, out_shift=0.0 (same as no-op). The parameters learn to adjust during training.
4. Add an L2 regularization on `out_scale` towards 1.0 and `out_shift` towards 0.0 in the loss: `scale_reg = 0.01 * ((out_scale - 1)**2 + out_shift**2).sum()`.
5. This adds only 6 parameters (3 scales + 3 shifts). No VRAM or compute impact.

**Expected ROI:** Low-Medium
**Risk:** Low
**Why now:**
Scale miscalibration is the simplest explanation for systematic bias in channel predictions. The global y_mean/y_std normalization removes first-order bias, but learned output scale/shift can correct residual calibration drift that the shared linear head introduces. This is the cheapest possible fix for systematic over/under-prediction per channel. Falsification: if learned out_scale/shift converges to near (1,0) at the best checkpoint, the output layer is already well-calibrated and this is not the bottleneck.

---

## Idea 16 — Incompressibility Penalty: Soft Divergence-Free Loss

**Axis:** Physics-informed

**Hypothesis:**
For incompressible 2D flow, the continuity equation requires ∇·u = ∂Ux/∂x + ∂Uy/∂z = 0. The baseline model has no mechanism to enforce this — it predicts Ux and Uy independently with no coupling constraint. Adding a soft divergence-free penalty using finite-difference approximation from neighboring node positions would inject this physics prior directly into training. The AB-UPT paper (TMLR 2025) showed a divergence-free constraint improved accuracy on CFD benchmarks by forcing physically consistent predictions. For an irregular mesh, approximate ∇·u at each node using the K=4 nearest neighbors in the batch (identified from node positions x[:,0:2]).

**Implementation outline:**
1. After computing predictions `pred`, approximate the divergence for surface and near-surface nodes. For each node i, find its K=4 nearest neighbors j1..j4 using `torch.cdist(x[:, :, 0:2], x[:, :, 0:2])` on the unpadded nodes.
2. Approximate: `div_u = (pred[:, :, 0].unsqueeze(2) - pred[nbrs, :, 0]) / (pos[:, :, 0].unsqueeze(2) - pos[nbrs, :, 0] + 1e-8)` + similar for Uy/z. This is expensive for large N — limit to surface nodes only (is_surface mask), where incompressibility matters most.
3. Penalize: `div_loss = (div_u * surf_mask.float()).pow(2).mean()`.
4. Total loss: `loss = vol_loss + surf_weight * surf_loss + div_weight * div_loss`. Add `div_weight: float = 0.01` to Config.
5. Critical: the `cdist` computation at N=242K is O(N²) — absolutely must be limited to surface nodes only (typically ~5K nodes). Use `is_surface` mask to extract surface node subset before computing neighbors.

**Expected ROI:** Medium
**Risk:** High (O(N²) neighbor computation even on surface; careful masking needed; may not converge smoothly)
**Why now:**
Physics constraints directly encode domain knowledge. The incompressibility condition is exact for the governing equations — violations in predicted (Ux, Uy) indicate the model is not learning the coupled physics. The surface-only approximation keeps computation feasible. The Geometric-DeepONet paper found velocity gradient penalties improved boundary accuracy by 32%. However, implementing discrete divergence on irregular meshes is tricky and the gradient signal from the penalty may be noisy. Best run after lower-risk ideas are exhausted. Falsification: if div_loss decreases during training but surface p MAE does not improve, incompressibility of velocity is not the bottleneck for pressure accuracy.

---

## Idea 17 — Separate Train/Val Normalization per Split Domain (Test-Time Domain Adaptation)

**Axis:** Data / evaluation

**Hypothesis:**
The 4 val splits correspond to different flow regimes with very different physical scales. `val_single_in_dist` has y values up to ±29K, `val_geom_camber_cruise` up to ±7.6K. When the model predicts in the global normalized space and the predictions are denormalized with global y_std (which is dominated by high-Re raceCar single), the cruise domain predictions are systematically under-scaled. A test-time affine correction: after generating predictions with the model, shift and scale the predictions per-split using the split's own validation-set statistics (computed once before training from val_splits data). This is a post-processing step that does not change the model or training.

**Implementation outline:**
1. Before training, compute `val_y_stats` for each val split: mean and std of y over the split's samples (loading from val_splits, using is_surface mask to get surface-only stats for the p channel).
2. After model inference at val time, apply per-split affine correction: `pred_corrected = (pred - global_y_mean_norm) * (split_y_std / global_y_std) + split_y_mean_norm`.
3. This is equivalent to using the split's y statistics for denormalization instead of the global statistics.
4. Important: this correction must ONLY be applied at val/test time, not during training (that would create train/val normalization mismatch in the loss).
5. Add `domain_adapt_eval: bool = False` to Config. When True, compute split-specific correction factors at startup and apply during evaluate_split.

**Expected ROI:** Low-Medium
**Risk:** Low (pure post-processing; cannot hurt training)
**Why now:**
This is the cheapest possible test of whether normalization scale mismatch is hurting val metrics on the cruise splits. No model changes. If it helps, it validates the domain-adaptive normalization hypothesis (Idea 5) for training, and the gain from Idea 5 should be larger. Falsification: if per-split affine correction at eval time does not change val metrics, the normalization scale hypothesis is wrong.

---

## Idea 18 — AdamW with Schedule-Free Optimization (No LR Schedule)

**Axis:** Optimization

**Hypothesis:**
The Schedule-Free optimizer (Defazio et al. 2024, arXiv:2405.15682, presented at NeurIPS 2024) eliminates the need for a LR schedule by using a primal-dual averaging trick that implicitly averages iterates. Under a fixed iteration budget (50 epochs), schedule-free Adam typically matches or beats cosine-annealed Adam because it does not need to guess the right schedule length. The baseline uses cosine annealing with T_max=50 — which is optimal only if training runs exactly 50 epochs and doesn't time out early. With the 30-min wall clock cap, the number of actual epochs can vary. Schedule-free removes this dependency. The implementation is available in pure PyTorch with no external dependency.

**Implementation outline:**
1. Implement schedule-free AdamW directly in train.py (no new package needed — it's ~50 lines of PyTorch):
   ```python
   class ScheduleFreeAdamW(torch.optim.Optimizer):
       # Primal-dual averaging: maintain x (primal), z (checkpoint), y (gradient)
       # Full implementation from arXiv:2405.15682 Algorithm 2
   ```
2. Replace `torch.optim.AdamW` + `CosineAnnealingLR` with `ScheduleFreeAdamW(lr=5e-4, weight_decay=1e-4)`. Remove the `scheduler.step()` call.
3. At eval time (before val loop), call `optimizer.eval()` to swap to the averaged weights. After eval, call `optimizer.train()` to swap back.
4. Add `schedule_free: bool = False` to Config to toggle this vs. baseline cosine schedule.
5. Key hyperparameter: `warmup_steps` (schedule-free warmup) = 5% of total steps ≈ 50 * 375/4 * 0.05 ≈ 234 steps.

**Expected ROI:** Medium
**Risk:** Low (well-validated at NeurIPS 2024; pure PyTorch implementation)
**Why now:**
Schedule-free optimization is one of the most practically significant optimization advances in 2024. The 30-min wall clock + 50-epoch hard cap means the actual run length is uncertain — schedule-free removes the schedule-length dependency. The implementation is ~50 lines of PyTorch and requires no new packages. This directly addresses the "is cosine annealing optimal under variable run length?" uncertainty. Falsification: if schedule-free does not improve vs. cosine, the optimizer schedule is not the bottleneck — training dynamics are dominated by the batch composition or model capacity.

---

## Summary Rankings (by impact × probability)

| Rank | Idea | Axis | ROI | Risk | Rationale |
|------|------|------|-----|------|-----------|
| 1 | EMA weights (#3) | Optimization | High | Low | Drop-in, zero cost, robust across settings |
| 2 | Larger capacity (#2) | Architecture | High | Medium | Baseline is almost certainly underfitting |
| 3 | Huber + Re-scale loss (#1) | Loss | High | Low | Directly targets known heterogeneity issue |
| 4 | Gradient clip + warmup (#6) | Optimization | Medium | Low | Free stability fix; broad applicability |
| 5 | Schedule-Free optimizer (#18) | Optimization | Medium | Low | NeurIPS 2024; no schedule tuning |
| 6 | Per-domain norm (#5) | Loss/Data | Medium | Medium | 10× scale mismatch is documented |
| 7 | Fourier pos encoding (#10) | Features | Medium | Low | Standard in operator learning; cheap |
| 8 | Multi-head pressure output (#7) | Architecture | Medium | Medium | Primary metric = pressure; direct alignment |
| 9 | Surface-dense sampling (#8) | Data | Medium | Medium | Surface focus aligns with metric |
| 10 | SDF surface normals (#4) | Features | Medium | Medium | Physical motivation; cheap feature addition |
| 11 | Auxiliary coeff head (#11) | Multi-task | Medium | Medium | FUSE-style global consistency supervision |
| 12 | Curriculum learning (#12) | Data | Medium | Medium | Dataset authors tested this |
| 13 | Per-split eval correction (#17) | Evaluation | Low-Med | Low | Cheap diagnostic; can be merged free |
| 14 | Output scale/shift (#15) | Architecture | Low-Med | Low | 6 extra params; calibration fix |
| 15 | PiT position attention (#9) | Architecture | Medium | Medium | Geometry OOD motivation |
| 16 | SpiderSolver multi-scale (#13) | Architecture | Medium | High | Strong physics; complex implementation |
| 17 | Incompressibility penalty (#16) | Physics | Medium | High | O(N²) risk; nontrivial gradient |
| 18 | Laplacian label smooth (#14) | Loss | Low-Med | Medium | Soft physics prior; hard to calibrate |

## First Batch of 8 for Immediate Assignment

Based on diversity across axes, high ROI, and low-medium risk:

1. **Idea #3** (EMA weights) — alphonse — axis: optimization
2. **Idea #2** (Larger capacity) — askeladd — axis: architecture
3. **Idea #1** (Huber + Re-scale loss) — edward — axis: loss
4. **Idea #6** (Gradient clip + warmup) — fern — axis: optimization
5. **Idea #18** (Schedule-Free optimizer) — frieren — axis: optimization
6. **Idea #10** (Fourier positional encoding) — nezuko — axis: features
7. **Idea #7** (Multi-head pressure output) — tanjiro — axis: architecture
8. **Idea #5** (Per-domain normalization) — thorfinn — axis: data/loss
