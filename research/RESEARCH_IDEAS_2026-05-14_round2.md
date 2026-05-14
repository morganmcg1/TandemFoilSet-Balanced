# Research Ideas — Round 2 (2026-05-14)

Generated after: PR #1405 merge (val_avg=73.295), all 8 student slots active.
Baseline: `--epochs 25 --lr 2e-3 --loss l1` + bf16 + OneCycleLR, ~19 epochs realized.

## What is NOT to repeat

Dead ends (confirmed or suspected):
- SAM optimizer: 2x compute cost, too few realized epochs
- n_hidden=256 or n_layers=8: VRAM / throughput too costly
- EMA on OneCycleLR: no gradient noise to exploit (closed PR #1601)
- Peak LR > 2e-3: frieren sweep confirmed saturation
- slice_num=128: VRAM / attention cost too high

In-flight (do NOT duplicate):
- frieren #2913: epochs 30/40 sweep
- askeladd #2914: n_layers 6/7
- thorfinn #2915: EMA decay 0.999/0.9999
- tanjiro #2916: bs=8 + extended schedule
- fern #1602: gc=2.0 re-run on bf16 baseline
- edward #1605: asinh-p680 re-run on bf16 baseline
- nezuko #1625: surf_channel_weight cw=2 re-run
- alphonse #1582: surf_weight=5 re-run

---

## Hypothesis 1: Geometric symmetry augmentation (z-flip + AoA negation)

**Predicted Δval_avg:** -3 to -7 pts (effective 2x training set, free)

**Mechanism:**
The raceCar domain uses negative AoA (inverted airfoil, -10° to 0°). The flow field for angle α is the mirror image of the field for -α, reflected across z=0. Specifically:
- Node z → -z
- AoA (dims 14 and 18) → negated
- Uy (target dim 1) → negated
- p and Ux are unchanged by this symmetry

During each training batch, with probability p=0.5, apply this transform to the entire sample (in the normalized input and target tensors) before forward pass. No new data, no new parameters, no architecture change. Effective training set doubles. Particularly beneficial for the geom_camber_rc split which covers the inverted-foil regime.

This is a well-established technique in geometric ML and CFD surrogate work (data-space equivariance exploitation). The constraint: verify that the cruise domain AoA (+/- mixed) also satisfies this symmetry exactly — if cruise foils have asymmetric geometry (e.g., camber M=2-4), the reflection might not perfectly reproduce the target field. A safe implementation applies augmentation only to samples where `x[:, 22] != 0` OR limits it to the raceCar portion (identifiable by dims 19-21 for foil 2 being near -10° to 0° range).

**Implementation sketch (in train.py, zero data/ changes):**
```python
# Inside the training loop, after loading (x, y, is_surface, mask) from loader
# Apply with 50% probability
if torch.rand(1).item() < 0.5:
    # flip z-coordinate (dim 1 of x)
    x = x.clone(); x[:, :, 1] = -x[:, :, 1]
    # negate AoA foil 1 and foil 2 (dims 14, 18)
    x[:, :, 14] = -x[:, :, 14]
    x[:, :, 18] = -x[:, :, 18]
    # negate dsdf dims that encode signed z-distance (dims 4-11, spot-check needed)
    # negate Uy target (dim 1)
    y = y.clone(); y[:, :, 1] = -y[:, :, 1]
```
Note: the normalization stats x_mean/x_std are already applied before the loop. The flip must happen BEFORE normalization, or equivalently the z-flip in normalized space is: `x_norm[:,:,1] = -x_norm[:,:,1]` only if z_mean=0 (which it may not be for the mesh). Safer to do the flip in raw unnormalized space before `x_norm = (x - stats['x_mean']) / stats['x_std']`. The current training loop normalizes inside the epoch loop — insert the flip before that step.

**Risk:** Medium. The symmetry must be physically exact for every domain. If cruise geometry breaks the symmetry (non-symmetric camber line at M=3), augmentation injects corrupted training signal. Ablation: compare with augmentation applied only to raceCar samples (identifiable by foil-2 AoA dim 18 being non-zero and in [-10, 0] range).

**Priority:** High — this is one of the cheapest free improvements in the CFD surrogate literature. No hyperparameters, no extra compute, no architecture change.

---

## Hypothesis 2: OneCycleLR pct_start warmup tuning (0.1 → 0.05 or 0.3)

**Predicted Δval_avg:** -1 to -4 pts

**Mechanism:**
The current schedule spends `pct_start=0.1` of total steps (= 0.1 × 25 × n_batches ≈ 2.5 epochs) warming up to peak_lr=2e-3. With only ~19 epochs realized, those 2.5 warmup epochs consume 13% of the productive budget.

Two competing hypotheses:
- **Shorter warmup (pct_start=0.05, ~1.25 epochs):** Spend less budget on warmup, more on the high-LR productive phase. Net effect: reach peak sooner, get more steps near peak before the cosine tail. Expected gain if the current warmup is excessive.
- **Longer warmup (pct_start=0.3, ~7.5 epochs):** With bf16 potentially causing gradient noise in early training, a longer warmup may stabilize the initial phase and allow the schedule tail (high-LR phase) to be more productive. Risk: wastes budget.

The schedule tail (ep14→19 in PR #1405) contributed ~12 pts improvement. If the warmup is shortened, the tail starts earlier with respect to wall-clock, potentially squeezing more epochs into the high-LR phase before the 30-min cap.

**Implementation:** Change single line in OneCycleLR setup: `pct_start=0.05` (or `0.3`). Run arm A (0.05) and arm B (0.3) in parallel. Can be combined with the epochs=30 sweep from frieren #2913 if that completes first.

**Risk:** Low-medium. The pct_start sensitivity in OneCycleLR is moderate; results can be noisy over 1-2 pts. Two arms in one PR removes ambiguity.

---

## Hypothesis 3: Variance-penalized loss (mean + std of absolute errors)

**Predicted Δval_avg:** -2 to -5 pts, particularly on geom_camber_rc

**Mechanism:**
Standard L1 minimizes the mean of |errors| globally. In fluid mechanics, high-error nodes cluster spatially: stagnation points, wake regions, foil suction peaks. A predictor can achieve low mean error while leaving these localized pathological nodes essentially unfit.

The variance-penalized loss adds a std term:
```
L = mean(|err_surf|) + λ * std(|err_surf|)
```
This penalizes inconsistency in prediction quality across surface nodes. From the arxiv paper on variance-based loss applied to 2D Navier-Stokes (2412.13993), λ=0.5 or λ=1.0 worked best. The effect: reduces error variance at the expense of slightly higher mean, but the net primary metric (MAE) often improves because the high-error outlier nodes that were ignored by pure mean loss now pull the gradient.

The geom_camber_rc split sees M=6-8 front-foil cambers (unseen during training). High-camber foils produce sharper suction peaks — exactly the spatial-outlier pattern this loss targets. The rc split's persistent gap (87.82 vs 54.42 for cruise) suggests localized high-error regions that the current loss treats as low-weight outliers.

**Implementation sketch:**
```python
# In the loss computation block, replace the L1 surf_loss:
surf_abs_err = (pred - y_norm).abs()[:, :, 2]  # pressure channel only for surf
surf_abs_err_surf = surf_abs_err * surf_mask
# mean over surface nodes
n_surf = surf_mask.sum().clamp(min=1)
mean_err = surf_abs_err_surf.sum() / n_surf
# std over surface nodes (biased estimator, fast)
mean_sq = (surf_abs_err_surf**2).sum() / n_surf
variance = mean_sq - mean_err**2
std_err = variance.clamp(min=0).sqrt()
surf_loss = mean_err + 0.5 * std_err  # λ=0.5 as default
```
Note: the variance must be masked properly — only over `surf_mask` nodes, not padding. Apply the same vol_loss as before unchanged.

**Risk:** Medium. The std penalty is noisy for small batches (bs=4). With padding-masked variance, there may be gradient instability early in training. Use a warm-up delay: apply the variance term only after epoch 5.

---

## Hypothesis 4: Temperature annealing in PhysicsAttention slice routing

**Predicted Δval_avg:** -2 to -6 pts

**Mechanism:**
The PhysicsAttention uses a learnable `temperature` parameter (init=0.5) to control the sharpness of softmax slice routing:
```python
slice_weights = softmax(in_project_slice(x_mid) / temperature)
```
With learned temperature=0.5 throughout training, routing stays at a fixed entropy. Early in training, high-entropy (soft) routing helps exploration; late in training, low-entropy (sharp) routing improves discrimination between slice regions.

Temperature annealing: start with a large temperature (soft assignments, many nodes contribute to each slice → stable gradients) and anneal toward a smaller temperature (sharper assignments, more specialized slices). This mirrors the gumbel-softmax and VQ-VAE training schedules in discrete representation learning.

**Implementation:**
In the Transolver model, replace the static learnable temperature with a scheduled temperature that decays linearly or cosine from `temp_start=1.0` to `temp_end=0.1` over training:
```python
# In PhysicsAttention, instead of self.temperature as a leaf parameter:
# Pass current_epoch/max_epochs as a scalar to the model forward
# temp = temp_start * (1 - progress) + temp_end * progress
# slice_weights = softmax(in_project_slice(x_mid) / temp)
```
The `temperature` parameter becomes a schedule variable, not a learnable scalar. Alternatively, anneal the initial value via a LR-like schedule while keeping it learnable (i.e., set learning rate for temperature parameter separately).

This is inspired by the progressive discretization in discrete VAEs and the slot attention literature (Locatello et al. 2020, Locatello et al. 2022 SA-DINO) where soft → hard annealing improves binding.

**Risk:** Medium-high. Requires modifying the model forward pass to accept a schedule parameter. The current model takes only `{"x": ...}` as input — need to add a `temperature_override` kwarg. Implementation is ~15 lines. Key hyperparameters: temp_start, temp_end, and annealing schedule shape (cosine vs linear).

---

## Hypothesis 5: Per-split normalization statistics via domain index feature

**Predicted Δval_avg:** -3 to -8 pts, mostly from rc split

**Mechanism:**
The three domains (raceCar single, raceCar tandem, cruise tandem) have dramatically different pressure ranges. Stats.json uses global y_mean/y_std, which compresses all domains into one normalized space. The cruise domain has max |p| ~7,648 vs raceCar single at ~29,136 — a 4x difference. Global normalization forces the model to handle both regimes with a single output scale.

Alternative: compute per-domain normalization constants and apply them based on the domain indicator derivable from x:
- Single foil: x[:, 18]==0 and x[:, 19:22]==(0,0,0) approximately
- RaceCar tandem: x[:, 18] in [-10°, 0°] rad, foil2 NACA active, x[:, 22]>0
- Cruise tandem: x[:, 14] in [-5°, +6°], foil2 NACA active

During training, apply per-domain (y - y_domain_mean) / y_domain_std for the loss computation, and undo with the correct per-domain stats for the MAE evaluation. This is purely a training-time normalization change — the model still predicts a 3D vector.

Evidence basis: NeuralFoil (2503.16323) explicitly uses separate normalizations per flow regime. The val_single_in_dist (79.89) vs val_geom_camber_rc (87.82) gap partly reflects normalization mismatch.

**Implementation:** Can be done entirely in train.py — read the existing stats.json to initialize global stats, then compute per-domain stats from the training set at startup:
```python
# Compute per-domain y stats from training set samples
# domain_ids = {0: single, 1: racecar_tandem, 2: cruise_tandem}
# Map each sample to its domain via x features
# Store y_mean_domain[d], y_std_domain[d]
# In training loop: look up domain from batch x, apply domain-specific normalization
```
The domain detection is cheap: `is_single = (x[:, 18].abs() < 0.01)` and `is_cruise = (x[:, 14] > 0)` for batch-level detection. Since each sample comes from one domain, this is well-defined.

**Risk:** Medium. Requires 30-50 lines of code to compute per-domain stats at startup. The larger risk is train/eval consistency: the val/test scoring (in data/scoring.py, read-only) uses global stats from stats.json for denormalization. This means per-domain normalization must be applied symmetrically for the MAE to be correct. Since scoring.py re-denormalizes using `pred * y_std + y_mean` with global stats, the model must predict in global normalized space at eval time. Per-domain normalization would only apply to the loss signal, not the output scale — which is actually fine: train on per-domain-normalized loss, but eval with global-denormalized preds. This asymmetry requires careful implementation.

---

## Hypothesis 6: torch.compile for per-epoch time reduction

**Predicted Δval_avg:** -3 to -9 pts (via more realized epochs, not quality)

**Mechanism:**
Current per-epoch time is ~97 s, yielding ~18-19 epochs in the 30-min cap. The CPU/dataloader bottleneck limits GPU utilization. `torch.compile` (PyTorch 2.x) reduces Python overhead and kernel launch latency, typically giving 10-25% throughput improvement on transformer workloads even when the dataloader is the bottleneck — because the GPU side finishes faster, more of each epoch is "GPU-idle waiting for data" → fewer total wall-clock seconds.

If `torch.compile` yields 15% speedup: ~97 s/epoch → ~82 s/epoch → 22 epochs in 30 min (vs 19 currently). That is 3 additional epochs in the schedule tail where each epoch historically contributes 3-7 val pts.

**Implementation:**
```python
# After model creation, before training loop:
model = torch.compile(model, mode="reduce-overhead")
```
`reduce-overhead` mode is best for transformer workloads with fixed shape inner loops. `fullgraph=True` would be faster but may fail on dynamic control flow in the model.

Key concern: the variable-mesh padding (74K–242K nodes) means batch shape changes every step — `torch.compile` will recompile on first new shape. Solution: use `mode="reduce-overhead"` (allows shape recompilation with caching) rather than `mode="max-autotune"` (too slow on shape changes). Alternatively, set `dynamic=True`:
```python
model = torch.compile(model, dynamic=True)
```

Second concern: bf16 autocast + compile. The `torch.autocast` context manager interacts with compile — wrap the compile inside the autocast context in the training loop, or use `torch.compile` with the model wrapped in `torch.cuda.amp.autocast`. Current code already uses `with torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — this is compatible.

**Risk:** Low-medium. `torch.compile` is a compile-time investment of ~60-120 s on first batch, but this is within the 30-min window if triggered at epoch 0 step 0. If it produces slower-than-expected results (CUDA graphs breaking on variable shapes), the fallback is trivial (remove the one line). Worth a dedicated screening run.

---

## Hypothesis 7: Surface-conditioned decoder with separate surface head

**Predicted Δval_avg:** -4 to -10 pts

**Mechanism:**
The current Transolver predicts [Ux, Uy, p] with a single output head for ALL nodes (surface + volume). But the pressure field on foil surfaces obeys different physics than interior volume pressure: boundary layer equations, no-slip, Kutta condition. The same 3-channel output head tries to fit both regimes.

A surface-conditioned decoder uses two separate final projection heads:
- `head_vol`: for volume nodes (mask=True, is_surface=False)
- `head_surf`: for surface nodes (mask=True, is_surface=True)

Both heads take the same latent representation from the last TransolverBlock output, but have independent learnable parameters. The loss is split accordingly:
- vol_loss uses head_vol predictions on volume nodes
- surf_loss (10x weighted) uses head_surf predictions on surface nodes

This is architecturally similar to the multi-task head designs in FNO variants and the protein structure prediction heads in AlphaFold2 (separate backbone and sidechain torsion heads).

**Implementation sketch:**
Add to the Transolver class:
```python
self.head_surf = nn.Sequential(
    nn.LayerNorm(n_hidden),
    nn.Linear(n_hidden, n_hidden),
    nn.GELU(),
    nn.Linear(n_hidden, out_dim)
)
# Keep existing head as head_vol
```
In forward, compute two outputs and recombine:
```python
preds_vol = self.head_vol(fx)    # existing head
preds_surf = self.head_surf(fx)  # new head
# recombine using is_surface mask (passed through data dict or as separate arg)
preds = torch.where(is_surface.unsqueeze(-1), preds_surf, preds_vol)
return {"preds": preds}
```
Note: `is_surface` must be passed into the model. Current model takes only `{"x": ...}`. Add `is_surface` to the data dict: `model({"x": x_norm, "is_surface": is_surface})["preds"]`.

**Risk:** Medium. The is_surface flag is already in `x` (dim 12), so the model already has implicit access. A separate head may not outperform the current implicit conditioning. Also, the surface head has fewer surface training examples per batch (surface nodes are ~1-2% of total), so the head may underfit. To mitigate: use a smaller head (single linear layer) and higher weight_decay for head_surf.

---

## Hypothesis 8: Gradient accumulation as schedule horizon extender

**Predicted Δval_avg:** -2 to -6 pts

**Mechanism:**
The CPU/dataloader bottleneck means the per-step time is dominated by data loading, not the forward/backward pass. Gradient accumulation (accumulate_steps=2 or 4) runs N micro-steps before optimizer.step(), which:
1. Reduces the number of optimizer steps per epoch by N (fewer LR schedule updates)
2. Effectively increases batch size by N (more diverse gradients)
3. DOES NOT reduce per-epoch wall-clock time (data loading is still the bottleneck)

But the OneCycleLR scheduler is configured with `total_steps = MAX_EPOCHS * len(train_loader)`. With accumulate_steps=2, optimizer steps = MAX_EPOCHS * len(train_loader) / 2. For the same epochs, the LR schedule decays slower — the schedule is now calibrated to take 2x as many micro-steps per "effective step", meaning we could run to epochs=50 and only use half the schedule budget that 50 epochs would normally consume.

Net effect: OneCycleLR configured for 50 epochs but only completing ~18 epochs of forward passes → LR stays in the high-productive warmup/decay region for longer.

**Implementation:**
```python
# In train.py, add accumulate_steps to Config
# Adjust OneCycleLR total_steps:
scheduler = OneCycleLR(
    optimizer,
    max_lr=cfg.lr,
    total_steps=MAX_EPOCHS * len(train_loader) // cfg.accumulate_steps,
    ...
)
# In training loop:
for step, batch in enumerate(train_loader):
    loss = compute_loss(batch) / cfg.accumulate_steps
    loss.backward()
    if (step + 1) % cfg.accumulate_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```
`accumulate_steps=2, epochs=50`: schedule horizon extends to ~36 effective optimizer steps (vs 19 at current), while the 30-min cap still limits actual epochs.

**Risk:** Medium. Gradient accumulation in PyTorch with bf16 requires care: ensure loss.backward() is called within autocast context, and gradient scaling is consistent across micro-steps. The mask handling (variable-length sequences) means the effective batch size is variable — gradient accumulation of variable-length batches changes the normalization semantics. Use loss scaling based on number of valid nodes (already done via `.sum() / mask.sum()`) to avoid this.

---

## Hypothesis 9: Re-weighting training domains toward OOD-hard splits

**Predicted Δval_avg:** -2 to -5 pts on rc split; possible regression on in-dist

**Mechanism:**
The current `WeightedRandomSampler` gives equal weight to the 3 training domain groups. The val results show a persistent gap: geom_camber_rc=87.82 vs geom_camber_cruise=54.42 (-33 pts). The rc split evaluates on unseen camber M=6-8, trained on M=2-5 + M=9 (the held-out camber range has no training examples at those values).

The cruise split trains on M=0-2 and M=4-6, and evaluates M=2-4 — the camber interpolation is easier because the training data brackets the eval range on both sides. The rc split evaluation is extrapolation (M=6-8 not bracketed below by M=5 and above by M=9 may still be interpolation in higher dimensions... but the val signal says it's harder).

Reweighting: increase the sampling probability of cruise_tandem (which informs the rc split's geometry distribution more than racecar_single) and/or downweight racecar_single (which contributes less to the tandem OOD splits).

**Implementation:** In train.py, modify sample_weights computation after `load_data()`:
```python
# sample_weights is a tensor of per-sample weights from load_data()
# Identify which samples belong to each domain group via the data loader
# (domain group membership is implicit in the tandem geometry features)
# Increase cruise weight by 1.5x, decrease racecar_single by 0.5x
# This effectively increases tandem-domain fraction from ~60% to ~75%
```
The challenge: `sample_weights` is returned by `load_data()` which is read-only. But the returned tensor can be modified in train.py after the call: `sample_weights = sample_weights * domain_multiplier_tensor`.

**Risk:** Medium. The primary metric is an equal-weight average of 4 splits. Reweighting toward tandem domains may improve rc and cruise while hurting single_in_dist (racecar single is the sanity check). Worst case: rc improves 5 pts, single_in_dist regresses 8 pts, net negative. Include single_in_dist tracking in the experiment report.

---

## Hypothesis 10: Progressive slot count (slice_num) warm-up

**Predicted Δval_avg:** -2 to -5 pts

**Mechanism:**
The Transolver uses `slice_num=64` softmax-routing slots throughout training. In early epochs, the model initializes with random weights and the 64 slots all collapse to similar representations (random, diffuse routing). Training wastes early gradient steps on routing disambiguation.

Progressive warm-up: start with `slice_num=16` (or even 8) for the first few epochs, then expand to 32, then 64. The early-phase smaller slot count forces coarse-grained physics grouping (which the model can learn quickly) before fine-grained 64-slot specialization. This is analogous to progressive growing of GANs and the multi-resolution training in ViTs.

However, in the Transolver implementation, `slice_num` is a fixed architecture parameter (the weight matrices `in_project_slice` have shape `[heads, hidden//heads, slice_num]`). Changing slice_num requires re-initializing these weights.

Simpler implementation: use a **temperature warm-up** instead (see Hypothesis 4). Temperature=10.0 (very soft) at epoch 0 → temperature=0.5 (normal) at epoch 10. The soft routing at early epochs mimics few effective slots. This tests the same hypothesis without requiring weight re-initialization.

**Implementation:** Combined with Hypothesis 4 — test this hypothesis via temperature annealing as the mechanism.

**Risk:** High for progressive architecture resizing. Low-medium for temperature proxy approach. Recommend testing via temperature annealing (Hypothesis 4) first.

---

## Hypothesis 11: Reduce eval frequency to every 2 epochs

**Predicted Δval_avg:** -1 to -4 pts (via more training steps in wall-clock budget)

**Mechanism:**
Each eval pass evaluates 4 validation splits × 100 samples = 400 forward passes. At ~97 s/epoch, the eval time is not explicitly reported but is likely ~15-25 s per eval call (100 samples × ~10 KB nodes × fast forward). Reducing from eval-every-epoch to eval-every-2-epochs saves ~8-12 eval calls in the 30-min window.

Saved eval time ≈ 10 eval calls × 20 s = 200 s → roughly 2 additional training epochs. At 3-7 pts/epoch in the productive tail, this is worth 6-14 val pts in expectation — which would be the single largest gain per line of code.

**Implementation:**
```python
# In the epoch loop, add a condition:
if epoch % cfg.eval_every == 0 or epoch == MAX_EPOCHS - 1:
    # run validation
```
Add `eval_every: int = 1` to Config. Run with `--eval_every 2` for the experiment.

Checkpoint selection: with eval_every=2, best-val checkpoint is selected from a coarser grid of evaluation epochs. In the schedule tail (ep14→19), skipping one eval may miss the true best. Add mandatory eval at epoch MAX_EPOCHS (already implied by `or epoch == MAX_EPOCHS - 1`).

**Risk:** Low. Single-line change, easily reversible. Main risk: missing the best checkpoint by 1-2 epochs (best might be at ep17 if we only eval at ep16, ep18). Mitigate by also evaling at the final epoch before timeout.

---

## Hypothesis 12: Curriculum learning by mesh complexity (small → large batches first)

**Predicted Δval_avg:** -2 to -5 pts

**Mechanism:**
Current training uses random shuffled samples with domain-balanced sampling. Mesh sizes range from 74K to 242K nodes (3x variation). Larger meshes dominate per-step compute time. A curriculum that starts with small meshes (raceCar single, ~85K nodes) and progressively introduces larger meshes (cruise, ~210K nodes) has two benefits:
1. Early training steps are faster (more steps per wall-clock second → more warmup coverage)
2. The model learns basic physics on simpler geometries before complex tandem interactions

This is a form of curriculum learning (Bengio et al. 2009) adapted to mesh-based neural operators. Similar approaches are used in molecular dynamics simulators (start with small molecules, add larger).

**Implementation:**
Sort training samples by mesh size (available from dataset stats or computable at load time via `x.shape[0]`). In the first 30% of epochs, sample preferentially from small-mesh samples; in the last 70%, use the standard balanced sampler.

Since `load_data()` returns `sample_weights` (read-only), the curriculum must be implemented by replacing the `WeightedRandomSampler` with a custom epoch-aware sampler in train.py. This requires ~20 lines of code.

**Risk:** Medium. The domain-balance property of the current sampler is important — curriculum learning may unbalance domains during early training (raceCar single is small, cruise is large). The interaction between curriculum and domain balance is not guaranteed to be positive.

---

## Hypothesis 13: Multi-resolution auxiliary loss on subsampled mesh

**Predicted Δval_avg:** -2 to -6 pts

**Mechanism:**
The current loss is computed over all nodes. Add an auxiliary loss over a coarsely subsampled mesh (e.g., every 10th surface node) at a different loss scale. This forces the model to capture coarse-grained pressure patterns (global lift coefficient analog) before fine-grained local accuracy.

Specifically: subsample surface nodes to N_coarse = is_surface.sum() // 10, compute mean pressure prediction error at coarse scale, add as auxiliary loss with weight α=0.1.

Physical motivation: the pressure distribution integral (lift coefficient) is determined by coarse spatial structure. A model that gets global pressure shape right first will have easier gradients for the fine details.

**Implementation:** ~10 lines in the loss block:
```python
# Coarse auxiliary loss
coarse_idx = torch.randperm(surf_mask.sum())[::10]  # every 10th surface node
surf_nodes = pred[surf_mask]
surf_nodes_true = y_norm[surf_mask]
coarse_loss = (surf_nodes[coarse_idx, 2] - surf_nodes_true[coarse_idx, 2]).abs().mean()
loss = loss + 0.1 * coarse_loss
```
**Risk:** Medium. The coarse loss may not add information beyond the existing surf_loss — it's just a subsample of the same quantity. The benefit only materializes if the coarse-grained gradient provides a "cleaner" signal that helps convergence on the fine-grained version.

---

## Hypothesis 14: Stochastic depth regularization

**Predicted Δval_avg:** -1 to -3 pts (via regularization)

**Mechanism:**
With n_layers=5 and ~1.1M parameters, the Transolver is modestly sized. Stochastic depth (DropPath, Huang et al. 2016) randomly skips entire transformer blocks during training, acting as a strong regularizer. Each block is kept with probability `keep_prob` (typically 0.8-0.9 for 5 layers).

The OOD generalization gap (geom_camber_rc worst) suggests some overfitting to training geometry patterns. Stochastic depth is one of the most effective and cheapest regularizers for transformer architectures — it's already in DeiT, ViT, and most modern ViT variants.

**Implementation:**
```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device)
        return x * (random_tensor >= self.drop_prob).float() / (1 - self.drop_prob)

# In TransolverBlock.forward:
out = x + self.drop_path(self.attn(self.norm1(x)))
out = out + self.drop_path(self.ff(self.norm2(out)))
```
Assign linearly increasing drop rates: layer 0 gets 0.0, layer 4 gets 0.1 (stochastic depth linear schedule).

**Risk:** Low-medium. Standard technique; well understood. Risk: with only 19 epochs of training, additional regularization may hurt (underfitting is already a concern since the model is still improving at epoch 19). Recommend low drop_prob=0.05 as a first test.

---

## Hypothesis 15: Sinusoidal position encoding injection for boundary nodes

**Predicted Δval_avg:** -2 to -5 pts

**Mechanism:**
The Transolver currently encodes node position via the raw (x, z) coordinates in dim 0-1 of the input features. These are then processed by `preprocess(x)` (a linear layer). The model has no explicit way to distinguish "same arc-length position on different foils" or "same position in different geometric configurations".

Sinusoidal encodings for the arc-length feature (dims 2-3, `saf`) add a rich positional signal that captures the periodic nature of foil surface geometry. Arc-length from 0→1 traces the foil surface, and its periodic structure (leading edge, suction side, trailing edge, pressure side) has natural sinusoidal decomposition.

Add Fourier features of arc-length to the input:
```python
# For dims 2-3 (arc-length saf), add k=1..K sinusoidal embeddings
# Resulting in 4K additional features (sin/cos × 2 arc-length dims × K freqs)
# Concatenate to x before feeding to model (requires fun_dim adjustment)
K = 4  # 4 frequencies → 16 extra dims; fun_dim → 22+16=38
for k in range(1, K+1):
    sin_feats = torch.sin(2 * pi * k * x[:, :, 2:4])
    cos_feats = torch.cos(2 * pi * k * x[:, :, 2:4])
    x = torch.cat([x, sin_feats, cos_feats], dim=-1)
```
This is a standard technique from NeRF (Mildenhall et al. 2020) and Neural Operator papers (Li et al. FNO). With `K=4`, adds 16 features, changing fun_dim from 22 to 38. The Transolver `preprocess` MLP handles arbitrary fun_dim — only `fun_dim` parameter needs updating.

**Risk:** Low-medium. The arc-length features (dims 2-3) are already in the model. The question is whether explicit frequency decomposition provides useful signal beyond the linear preprocessing. The dsdf features (dims 4-11) already capture multi-scale geometry. Risk: the extra dimensions increase the input-to-hidden mapping in `preprocess`, adding ~3K parameters, negligible. The real risk is train/eval distribution mismatch if the Fourier features of arc-length interact poorly with the normalized stats.

---

## Summary Table

| # | Hypothesis | Predicted Δval | Complexity | Risk | Priority |
|---|-----------|---------------|-----------|------|---------|
| 1 | Geometric symmetry augmentation (z-flip) | -3 to -7 | Low | Medium | **HIGH** |
| 6 | torch.compile epoch speedup | -3 to -9 | Very Low | Low | **HIGH** |
| 11 | Eval every 2 epochs (more training time) | -1 to -4 | Very Low | Low | **HIGH** |
| 3 | Variance-penalized L1 loss | -2 to -5 | Low | Medium | High |
| 4 | Temperature annealing in slice routing | -2 to -6 | Medium | Medium | High |
| 7 | Separate surface/volume decoder heads | -4 to -10 | Medium | Medium | High |
| 2 | OneCycleLR pct_start tuning | -1 to -4 | Very Low | Low | Medium |
| 5 | Per-domain normalization | -3 to -8 | Medium | Medium | Medium |
| 8 | Gradient accumulation (slow LR schedule) | -2 to -6 | Medium | Medium | Medium |
| 9 | Domain re-weighting toward tandem | -2 to -5 rc | Low | Medium | Medium |
| 14 | Stochastic depth (DropPath) | -1 to -3 | Low | Low | Medium |
| 15 | Fourier arc-length features | -2 to -5 | Low | Low | Medium |
| 12 | Curriculum by mesh size | -2 to -5 | Medium | Medium | Low |
| 13 | Coarse auxiliary pressure loss | -2 to -6 | Low | Medium | Low |
| 10 | Progressive slice count warmup | -2 to -5 | High | High | Low (use H4 instead) |

## Recommended First Wave (when student slots open)

Given current 8 students all active on WIP PRs, these are the candidates for the next assignment wave, ranked by expected information-per-compute:

1. **torch.compile** (H6): One line, no risk, directly tests the "more epochs = better" hypothesis with a clean mechanism. If it works, every future experiment benefits.
2. **Geometric symmetry augmentation** (H1): Free 2x data, no parameters, directly addresses the OOD gap. Highest expected return per implementation line.
3. **Eval-every-2-epochs** (H11): Trivially safe, single-line, tests whether eval overhead is the binding constraint.
4. **Variance-penalized loss** (H3): Directly targets the rc-split high-error-region pathology. Well-motivated by Navier-Stokes literature.
5. **Separate surface/volume decoder heads** (H7): Medium complexity, targets the fundamental physics-architecture mismatch. Good tier-shift candidate.
6. **OneCycleLR pct_start tuning** (H2): Two arms in one PR (0.05 and 0.3), fast to run, directly tests schedule sensitivity.

Hypotheses 4, 5, 8, 9 are good second-wave candidates once the first wave settles.
