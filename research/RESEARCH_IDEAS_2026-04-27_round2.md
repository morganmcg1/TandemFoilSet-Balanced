# Research Ideas — Round 2 (Fresh, Beyond Round-1 Standard Moves)

**Date:** 2026-04-27
**Branch:** icml-appendix-willow-pai2c-r3
**Target:** TandemFoilSet — minimize `val_avg/mae_surf_p` (mean surface pressure MAE across 4 splits)
**Baseline (Transolver):** 5L × 128h × 4heads × slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, batch=4, surf_weight=10, AdamW + cosine, 50 epochs, 30-min cap.

Round-1 already covers: width capacity, surf-weight sweep, Huber loss, slice_num, depth, Fourier PE, Re-FiLM, BF16+batch.
**This document proposes 12 ideas that go BEYOND those.** Each is grounded in (a) a heavy-tail pressure distribution, (b) tandem-vs-single geometry asymmetry, (c) physics-aware inductive biases that have not been tested, and (d) generalization mechanisms specific to the M-camber holdout splits.

---

## 1. Per-sample pressure rescaling (predict in y_std-normalized output space, then auto-rescale)

**Mechanism.** Per-sample pressure std varies 10× within a split (e.g. val_single_in_dist: avg 458, max 2077). Global stats fold all that variance into one denominator, so high-Re samples dominate the loss in normalized space and low-Re samples are under-weighted in the optimization. A per-sample affine rescaling predicted by the model — or a normalization that conditions on Re (since Re monotonically explains most of the std) — gives the network an explicitly factorized magnitude vs. shape decomposition.

**Implementation.** Two variants to A/B:
  (a) Add 2 scalar heads `(log_scale, bias)` predicted from a global pooled token; multiply normalized prediction by `exp(log_scale)` and add bias before denormalization. Initialize both to 0.
  (b) Replace global `y_std` with a Re-conditioned `y_std(Re)`: precompute `y_std_p(Re_bin)` from training data in 6 Re bins, look up at runtime; this is a closed-form rescaling, no extra params.

**Expected delta.** -8% to -15% on `mae_surf_p` (very high signal — directly attacks the heavy-tail problem the global stats hide).
**Risk.** Medium. (a) can over-fit if scale head is unstable; mitigate with detach + small init. (b) is essentially free.
**Compute.** ~1.0× baseline (negligible new params).

---

## 2. Log-magnitude + sign decomposition of pressure target

**Mechanism.** Pressure in this dataset is heavy-tailed and signed (range ~(-29K, +2.7K)). MSE in the original space rewards getting the magnitude right and ignores small-but-physically-meaningful pressure variations near zero. Predicting `(sign(p), log(|p|+epsilon))` — then reconstructing — gives equal effective resolution to small and large pressures. This is the same trick used in audio (mu-law) and astrophysics (log-flux).

**Implementation.** In `train.py` only. Replace 3-channel target with `[Ux, Uy, sign(p), log(|p|+1.0)]` for loss; reconstruct `p = sign * (exp(log_p) - 1)` for MAE. Use BCE-with-logits for sign and Huber for log-mag. Keep `y_std` normalization on Ux/Uy unchanged.

**Expected delta.** -5% to -12% on `mae_surf_p`. Especially helpful on cruise splits where p has smaller dynamic range.
**Risk.** Medium-high. Discontinuity at p=0 hurts gradients; soft-sign via tanh(p/eps) helps. Requires care in MAE reconstruction.
**Compute.** 1.0× baseline.

---

## 3. Surface-token cross-attention head (decoder over surface points conditioned on volume context)

**Mechanism.** Currently the model has one shared trunk + a single decoder that treats all nodes equivalently. But ~95% of nodes are volume, only ~5% are surface, and surface pressure is the primary metric — surface nodes are starved of capacity. Adding a small cross-attention head where surface tokens query the volume slices gives surface predictions a dedicated computation path with all the physical context.

**Implementation.** After the last `TransolverBlock`, branch:
- Volume head: existing MLP2 over all tokens (predicts Ux, Uy, p for volume nodes).
- Surface head: extract surface tokens (`fx[is_surface]`), pass through 1 cross-attention layer where Q=surface tokens, K=V=slice_tokens from the last block (or all volume tokens), then a 2-layer MLP → 3 channels.
The surface-head prediction overrides the trunk prediction at surface nodes.

**Expected delta.** -7% to -15% on `mae_surf_p` (this is THE metric).
**Risk.** Medium. Needs careful handling of variable surface counts across batch; pad/mask. Adds ~0.3-0.5M params.
**Compute.** 1.05× baseline.

---

## 4. Symmetry augmentation: x-axis (vertical) flip with channel-correct sign flips

**Mechanism.** 2D Navier–Stokes is equivariant under reflection: flipping x→-x maps `(Ux, Uy, p)` to `(-Ux, Uy, p)`. The cruise domain has freestream BCs and is genuinely symmetric. The raceCar domain is NOT symmetric (ground at z=0 breaks vertical reflection — the raceCar foils are inverted, ground beneath). BUT a horizontal flip about the streamwise direction perpendicular to the inflow IS valid for cruise. Augmenting cruise samples with random x-flip + AoA negation + Ux negation effectively doubles cruise training data and forces a smoother manifold around the cruise camber holdout (M=2-4, the hardest split).

**Implementation.** In `train.py` collate or right after loading: detect cruise (gap !=0 and ground absent — use AoA-foil1 sign or a ground-effect feature). With p=0.5: flip node positions x→-x (dim 0), negate Ux (dim 0 of y), negate AoA features (dims 14, 18 of x), negate stagger (dim 23). Verify with a single-sample sanity check.

**Expected delta.** -5% to -10% on `val_geom_camber_cruise`, -2% to -4% on `val_avg/mae_surf_p`.
**Risk.** Low. Easy to gate by domain; if the ground-effect detection is wrong, augmentation degrades. Cruise has freestream so reflection-symmetric.
**Compute.** 1.0× baseline.

---

## 5. Multi-task gradient balancing via PCGrad / GradNorm / uncertainty weighting

**Mechanism.** Loss = vol_loss + 10 × surf_loss is a fixed multi-task scalarization. The 10× ratio is a guess. Worse: within `vol_loss`, Ux/Uy/p compete in normalized space (and the 3 channels have very different optimization landscapes — pressure has multimodal pockets, velocity is smoother). A learned per-task weight (Kendall's homoscedastic uncertainty: `L = sum_i (L_i / 2*sigma_i^2 + log sigma_i)`) lets the optimizer balance gradients automatically. PCGrad goes further: project conflicting gradients before summing.

**Implementation.** Replace fixed weighting with 5 learned log-sigmas (vol-Ux, vol-Uy, vol-p, surf-Ux+Uy combined, surf-p). Initialize `log_sigma_surf_p = 0`, others to `log(sqrt(0.1))`. Loss: `sum_i exp(-2*log_sigma_i) * L_i + log_sigma_i`. Treats surf_p as primary; gradients balance themselves. (PCGrad is a heavier add-on if needed later.)

**Expected delta.** -3% to -8% on `mae_surf_p`. Bigger if current 10× is far from optimal.
**Risk.** Low. Standard technique. Requires monitoring sigma curves for sanity.
**Compute.** 1.0× baseline (5 extra scalars).

---

## 6. Foil-1 / foil-2 / volume token-type embeddings

**Mechanism.** The `is_surface` boolean lumps both foils together, but tandem dynamics depend critically on which foil a surface node belongs to (front foil sheds wake; rear foil receives it). The model could in principle learn this from `dsdf` (distance-to-surface descriptor) features, but giving it explicit identity tokens accelerates convergence. Three token types: `volume`, `foil1_surf`, `foil2_surf` — discriminate surfaces via SAF (signed arc-length) sign or `dsdf`-min-distance to each foil's BBox.

**Implementation.** In `train.py` preprocessing: compute `foil_id = 0 (volume) | 1 (foil1) | 2 (foil2)` from `is_surface` + `dsdf` (the foil with smaller dsdf min is the home foil). Add a 3-row `nn.Embedding(3, n_hidden)` and add it to `fx` after `preprocess`. For single-foil samples, foil2 channel is unused (still works because gap==0).

**Expected delta.** -4% to -10% on tandem splits (`val_geom_camber_*`, `val_re_rand`); -2% to -5% overall.
**Risk.** Medium. If foil-id detection is buggy, hurts. Need a verifier on a few samples first.
**Compute.** 1.0× baseline (~0.5K extra params).

---

## 7. Geometry-aware contrastive pretext: reconstruct a perturbed wake region

**Mechanism.** The hardest splits are camber holdouts (M=6-8, M=2-4): the model has to interpolate over front-foil shape. A self-supervised auxiliary objective — randomly mask 30% of volume nodes near the foil surface (within dsdf < 0.1) and ask the model to predict their `(Ux, Uy, p)` from the rest of the field — pushes the model toward learning a generative model of the local flow, which extrapolates better to unseen camber.

**Implementation.** During training, with p=0.5, drop the mask of 30% volume nodes near surface (dsdf-min < 0.15) by zeroing their non-position features and adding a `[masked]` flag in input. Compute auxiliary loss only on those masked nodes. Loss = main_loss + 0.3 * masked_recon_loss. Acts like MAE-style pretraining mixed with main task.

**Expected delta.** -3% to -7% on camber-holdout splits. Slower convergence within 50 epochs is the risk.
**Risk.** Medium-high. Self-supervised objectives can compete with the main loss.
**Compute.** 1.0× baseline.

---

## 8. EMA weight averaging (Polyak) + SWA

**Mechanism.** With 30-minute wall-clock and high LR (5e-4), the loss surface around the optimum is noisy. Maintaining an exponential moving average of weights with decay 0.999 (or 0.9995) gives a smoother point in parameter space — typically 1-3% lower validation error in a few hundred lines of code. SWA (Stochastic Weight Averaging) extends this: in the last 30% of training, swap to a constant LR and average weights every epoch.

**Implementation.** Add `ema = AveragedModel(model)` in `torch.optim.swa_utils`. After each `optimizer.step()`, call `ema.update_parameters(model)`. Validate using EMA model. On final epoch, run BN-update pass. Save EMA checkpoint instead of raw model.

**Expected delta.** -2% to -5% on `mae_surf_p`. Cheap, reliable.
**Risk.** Low. Standard. EMA decay needs tuning (0.999 typical for 50-epoch runs).
**Compute.** 1.0× baseline (2× memory for EMA copy — fine at 96GB).

---

## 9. Mixture-of-Experts decoder gated by Re + foil regime

**Mechanism.** Re ranges 4.5 orders of magnitude (100K to 5M), and the flow regime transitions: low-Re is Stokes-like with thick boundary layers, high-Re is turbulent with thin layers and steep pressure gradients. A single decoder fits one regime well at the expense of the other. A 4-expert mixture, each expert a small MLP, gated by `softmax(W * [log_Re, gap, stagger, AoA1, AoA2])`, lets the network specialize without forcing it to. Top-2 routing keeps compute the same.

**Implementation.** Replace the `mlp2` decoder with a `MoEHead(n_experts=4, top_k=2)`. Each expert is a 2-layer MLP from n_hidden→3. Gate from a linear over Re+gap+stagger+AoA. Add load-balancing aux loss (0.01 weight). Total decoder params 4× a single decoder, but only 2 active per token.

**Expected delta.** -3% to -8% on `val_re_rand` specifically; -2% to -4% overall.
**Risk.** Medium. MoE training instability; needs warmup-style scaling on aux loss.
**Compute.** 1.05× baseline (~0.2M extra params).

---

## 10. Curriculum on Re: train on low-Re first, then expand

**Mechanism.** Optimization geometry of high-Re samples (sharp boundary layers, large pressure spikes) is much harder than low-Re. Following classical curriculum learning + numerical homotopy in CFD, start training on low-Re samples only (where flow is smooth), then gradually admit higher-Re samples over the first 30-50% of training. This regularizes the early optimization landscape and is well-known to help on heavy-tailed regression.

**Implementation.** Modify the WeightedRandomSampler. Track training progress as `frac = epoch / max_epochs`. Sample weight = base_weight * sigmoid((frac - log_Re_norm) / 0.1) where log_Re_norm is sample's normalized log_Re in [0,1]. At frac=0 only low-Re is sampled; by frac=0.5 all Re sampled uniformly.

**Expected delta.** -3% to -6% on `mae_surf_p`. Particularly helps high-Re extremes.
**Risk.** Medium. If curriculum is too aggressive, late-Re convergence is rushed.
**Compute.** 1.0× baseline.

---

## 11. Energy/divergence physical regularizer on velocity field

**Mechanism.** Incompressible flow satisfies `div(U) = ∂Ux/∂x + ∂Uy/∂y = 0`. The model predicts Ux,Uy at irregular nodes — we cannot directly compute analytic divergence, but we CAN compute it via a graph Laplacian over k-NN neighbors using the local spatial coordinates. Adding a small soft constraint `lambda * |graph_div(U_pred)|^2` regularizes Ux/Uy predictions toward physical realism, and pressure (related via momentum balance) inherits the smoothness benefits.

**Implementation.** Precompute k=8 NN graph per sample at preprocessing or during collate (cache to disk). At training, after prediction, compute approximate divergence via `Σ_j (Ux_j - Ux_i) * (x_j - x_i) / |x_j-x_i|^2 + Uy similar`. Add `lambda * mean(div^2)` to loss with `lambda=0.01` initially.

**Expected delta.** -2% to -6% on volume Ux/Uy — small impact on surface_p directly, but improves flow consistency near surface (where p depends on velocity gradients via Bernoulli/momentum equations).
**Risk.** Medium-high. KNN graph computation overhead is non-trivial; needs careful implementation.
**Compute.** 1.05-1.1× baseline (graph compute).

---

## 12. Predict residual from a simple potential-flow initialization

**Mechanism.** The main difficulty is regressing values across 4+ orders of magnitude. If we provide the model with an analytic potential-flow estimate as a "prior" — even just `p ~ -0.5 * |U_inf|^2` (Bernoulli upper bound, scales with Re) — and predict `delta = p_true - p_prior`, the residual has a much narrower distribution, which is far easier to fit. A common trick in physics ML.

**Implementation.** Compute `p_prior(x) = -0.5 * U_inf^2 * (1 - (x_dist/r)^2)` for each node, where `U_inf` is implied by Re and reference length, `x_dist` is dsdf-min, `r` is a characteristic length. Stack into x as a 25th feature. Train the model to predict `(Ux, Uy, p - p_prior)` (subtract from y before normalizing). Add `p_prior` back at output.

**Expected delta.** -5% to -10% on `mae_surf_p` (residual targets are inherently smaller, easier to fit).
**Risk.** Medium. The prior must be smooth and not introduce artifacts. Simple choices like `p_prior = 0` are safe but uninformative; smarter priors (panel method) introduce engineering complexity.
**Compute.** 1.0× baseline.

---

## Bonus / Stretch Ideas

### B1. Test-time augmentation (TTA) with x-flip on cruise samples
Run prediction on (x, flipped_x), un-flip the prediction, and average. Free at training time, only test-time cost. Expected: -1% to -3% on cruise. Compute: 2× test eval time.

### B2. Sliced-Wasserstein loss on per-sample pressure distribution
Replace MSE on surface p with a 1D sliced-Wasserstein distance between predicted and true pressure distributions on each foil. Captures distribution shape, not just pointwise error. Risky but potentially high-payoff for heavy-tailed targets.

### B3. ScanNet-style coarse-to-fine refinement
Predict at low resolution (subsample 1/4 nodes), refine with cross-attention from low-res prediction back to high-res. This is essentially a 2-stage Transolver. Memory benefits from short attention path.

### B4. Replace softmax-slicing with hard top-k slice attention
Current `slice_weights = softmax(...)` is dense. Hard top-k (k=8 of 64 slices, with straight-through gradient) makes attention sparse, allows much larger slice_num (256, 512) at same compute. May give the model a much richer global token vocabulary.

---

## Recommended Top-5 to assign to Round 2

1. **Per-sample pressure rescaling** (#1) — directly attacks heavy-tail, low risk variant (b) is essentially free.
2. **Surface-token cross-attention head** (#3) — gives the primary metric its own decoder path, expected high impact.
3. **Multi-task uncertainty weighting** (#5) — eliminates fixed surf_weight magic number with principled learning.
4. **EMA + SWA** (#8) — simple, reliable, almost-always wins on noisy training.
5. **x-flip augmentation on cruise samples** (#4) — almost-free augmentation that targets the hardest split.

Each is a clean, single-hypothesis PR. Each tests a fundamentally different mechanism (output param, architecture, loss balance, optimization, augmentation), so wins compose orthogonally with round-1 results.
