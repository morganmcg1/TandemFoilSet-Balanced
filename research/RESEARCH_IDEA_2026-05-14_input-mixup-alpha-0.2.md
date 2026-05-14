# Round 129 — Input-mixup (α=0.2) data augmentation

## Hypothesis

Add **input-mixup augmentation** (Zhang et al. 2018, "mixup: Beyond Empirical Risk Minimization") with `λ ~ Beta(0.2, 0.2)` applied to **50% of training batches**. Mixes both input tensors `x` and target tensors `y` linearly via `λ`. Tests whether **data-level interpolation** can break the cruise/in_dist over-specialization meta-signal observed across 5 consecutive losing experiments.

## Why this might WIN

1. **Directly targets over-specialization, but at the DATA level.** Five consecutive losing experiments (#2889 mlp_ratio=4, #2890 additive geo-FiLM, #2899 asym mlp, #2903 RMSNorm, #2905 wd=5e-4) all showed in_dist LOSS + camber_cruise WIN — meta-signal of in_dist over-specialization. All five attacks were on the model/training-side. Mixup is fresh: it attacks at the DATA level by creating interpolated samples that lie between training distributions.

2. **Mixup is well-established for the exact problem we have.** Original mixup paper: "improves model robustness ... to corrupt labels ... reduces sensitivity to adversarial examples". Zhang et al. show it specifically helps when models over-confidently fit individual training samples. Our 5-experiment meta-signal is textbook over-confident in_dist fitting.

3. **CFD data is unusually well-suited to mixup.** PDE solutions are LINEAR in boundary conditions for the LAPLACE/HELMHOLTZ part of the operator. Mixing two airfoil geometries linearly with mixed targets (which is mathematically wrong for full NS) is a soft form of physics-informed augmentation — the model learns to handle "blended" boundary conditions, which is what camber_cruise generalization requires.

4. **α=0.2 is the canonical mixup parameter.** Original paper found 0.2 robust across CIFAR/ImageNet. Symmetric Beta(0.2, 0.2) is bimodal — most mixes lean toward one or the other source sample. We are NOT setting α=1 (which produces uniform mixing) because that's too aggressive for a CFD surrogate that needs precise local features.

5. **Zero new parameters, zero new modules.** Pure training-time data transformation. Implementation is ~10 lines.

## Why this might LOSS

1. **CFD targets may not interpolate linearly.** Navier-Stokes is non-linear; mixing two flow solutions linearly is generally NOT a valid flow solution. The targets we'd be giving the model are physically inconsistent with the mixed inputs, which could confuse the model.

2. **Point cloud structure may be incompatible with simple mixup.** Each sample has N=cell-count points in a different spatial layout. Mixing positions linearly produces a non-physical mesh. We mitigate by mixing only the FEATURE channels (the input embedding) and keeping spatial positions fixed; this should be done at the embedded feature level, not the raw input.

3. **50% application rate may be too high.** Mixup typically helps when applied to a fraction of batches. We mitigate by mixing only some batches.

4. **In_dist is the easy split; mixup may regularize it AWAY.** Could perversely make in_dist worse if it shifts the model's "default" representation away from the simple-NACA regime.

## Mitigation: mix at the EMBEDDED feature level, not the raw input

Because point clouds have variable spatial layouts per sample, raw-input mixup is not well-defined. Apply mixup AFTER the preprocess MLP (where `fx` is `[B, N, n_hidden=96]`) and BEFORE slicing. Targets `y` are mixed at the corresponding sample level.

This is "**manifold mixup**" (Verma et al. 2019), a stronger variant proven to work better than input-mixup on structured data. The mechanism is identical to input-mixup but operates in the learned feature space.

## Falsifiable predictions

- **WIN** (val < 30.5605): Mixup decouples the cruise/in_dist trade-off; both effects retained or improved. Try α=0.4, manifold mixup at multiple layers.
- **PARTIAL** (in_dist within ±1% AND camber_cruise improved): meta-signal partially broken. Try variant α or application rate.
- **WASH** (val ≈ 30.5605 ± 0.5%): No effect at α=0.2. Try α=0.5 or 80% application.
- **LOSS** (val > 31.0): Mixup at this scale + this data hurts; physical inconsistency dominates. Close mixup axis.

## Implementation

### Step 1: Add mixup logic in the training loop

In `train.py`, find the training step (around line 695-710 per summary, where `loss = vol_loss + cfg.surf_weight * surf_loss` is computed). Add mixup BEFORE the forward pass:

```python
# In training loop, AFTER batch unpacking but BEFORE forward pass
# ... x, y already loaded ...

apply_mixup = (
    model.training and 
    torch.rand(1, device=x.device).item() < 0.5  # 50% application rate
)

if apply_mixup:
    lam = float(np.random.beta(0.2, 0.2))  # mixup parameter
    perm = torch.randperm(x.size(0), device=x.device)
    # Mix inputs and targets in parallel
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y + (1.0 - lam) * y[perm]
    # Replace
    x, y = x_mixed, y_mixed
```

**IMPORTANT:** Apply this BEFORE the forward pass `pred = model(x)`. Apply ONLY during training (`model.training`); never during eval. The `device=x.device` argument on randperm avoids host-device sync.

### Step 2: Add startup diagnostic

```python
print(f"Mixup: α=0.2, application_rate=0.5 (50% of batches)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # 407,940 unchanged
```

### Step 3: Optional — manifold mixup at the embedded feature level

If raw input-mixup proves to LOSS (test 1 first), the manifold-mixup variant is:
- Inside `Transolver.forward`, after `fx = self.preprocess(x)`, BEFORE flow-FiLM and slicing
- Mix `fx` and `y` per-sample at the same `lam`/`perm`
- Run for an additional arm if input-mixup succeeds

Skip step 3 for the first run; only do input-mixup.

## Baseline (PR #2879)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **30.5605** |
| test_avg/mae_surf_p | **26.5160** |
| val_single_in_dist | 23.3997 |
| val_geom_camber_rc | 46.0708 |
| val_geom_camber_cruise | 17.8657 |
| val_re_rand | 34.9057 |
| Param count | 407,940 |

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/input-mixup-alpha-0.2" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. No new CLI flag — mixup is hardcoded with α=0.2 and 50% application rate. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Meta-signal table:** add this row to the 5-experiment table — does mixup retain camber_cruise improvement WITHOUT in_dist regression?
4. Param count confirmation (~407,940 — zero new params)
5. Epochs completed (target 60), sec/epoch (expect ~30s, no FLOPs increase), peak GPU memory
6. Train-loss vs val-loss gap (mixup should shrink this gap by reducing memorization)
7. Optionally log fraction of batches that received mixup (should be ~50%) and lam distribution histogram
8. **Verdict on meta-signal:** did data-level interpolation break the cruise/in_dist coupling?

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
