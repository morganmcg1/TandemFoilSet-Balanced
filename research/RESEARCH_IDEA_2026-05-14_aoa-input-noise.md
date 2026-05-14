# Round 138 — AoA input noise σ=0.05 (conditioning-variable data augmentation, angle-of-attack)

## Hypothesis

Add Gaussian noise σ=0.05 to the ANGLE-OF-ATTACK input features (x[:, :, 14] = AoA foil 1, x[:, :, 18] = AoA foil 2) during TRAINING ONLY (no noise at eval). Completes the conditioning-variable data-augmentation triplet alongside #2973 edward (log_Re noise) and #2976 fern (coordinate noise).

## Motivation

The data-augmentation-at-input-channels axis now has three orthogonal experiments in-flight or planned:
1. **log_Re noise** (edward #2973, in-flight) — CONDITIONING, Reynolds number
2. **Coordinate noise** (fern #2976, in-flight) — POSITIONAL, mesh x/y coordinates
3. **AoA noise** (this PR) — CONDITIONING, angle of attack

Per alphonse #2961 student insight: "Worth a data-distribution probe rather than another model intervention." AoA directly controls the camber/cruise OOD split geometry. camber_cruise uses foils at positive AoA; camber_rc uses different AoA regime. Regularizing AoA sensitivity may help geometric OOD.

**Feature locations in input tensor `x` (shape [B, N, 24]):**
- `x[:, :, 14]` = AoA foil 1 (radians) — normalized
- `x[:, :, 18]` = AoA foil 2 (radians, 0 for single-foil) — normalized

All input features are NORMALIZED. σ=0.05 on normalized AoA is ~5% of a standard deviation — conservative, analogous to log_Re noise σ=0.05.

## Why this might WIN

1. **AoA directly determines split membership.** camber_cruise (positive AoA regime) and camber_rc (negative AoA) differ primarily in AoA conditioning. Noise on AoA during training = training on broader AoA distribution = better OOD generalization.

2. **Physically meaningful: small AoA perturbations produce small flow changes.** Aerodynamics is continuous in AoA — model should be robust to ±0.05 rad shifts.

3. **Different channel than #2973 (log_Re) and #2976 (coords).** Strict complementary axis.

4. **Zero new params, ~5 lines.** Minimal cost.

## Why this might LOSS

1. **AoA conditioning is already sharp in the model.** FiLM gate uses AoA (one of the 3 conditioning scalars). Noise disrupts the conditioning signal.
2. **σ may be too large or too small.**
3. **In_dist LOSS.** Training on perturbed AoA moves distribution away from in-dist.

## Implementation

```python
# In the training forward pass, before AoA reaches FiLM/routing:
if model.training:
    x = x.clone()
    aoa_noise = torch.randn(x.shape[0], 1, 1, device=x.device) * 0.05
    x[:, :, 14:15] = x[:, :, 14:15] + aoa_noise
    x[:, :, 18:19] = x[:, :, 18:19] + aoa_noise  # same noise for both foils in tandem
```

CRITICAL: Use the SAME noise value for both foil AoA channels (physically: both foils experience the same perturbation). Use per-SAMPLE noise (broadcast across nodes), not per-node noise.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-frieren \
    --experiment_name "charliepai2g48h5-frieren/aoa-input-noise" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```
