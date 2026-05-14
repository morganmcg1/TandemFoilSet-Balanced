# Round 138 — Mesh coordinate Gaussian noise σ=0.01 (data augmentation at positional channel)

## Hypothesis

Add Gaussian noise σ=0.01 to mesh COORDINATES (`pos_x`, `pos_y`) during TRAINING ONLY (no noise at eval). Tests whether positional robustness improves geometric OOD generalization (camber_rc, camber_cruise splits) — a fresh DATA-AXIS experiment at the positional channel level.

Distinct from edward's #2973 log_Re-INPUT-noise (conditioning-variable channel). Both are data-augmentation experiments but target ORTHOGONAL feature channels.

## Motivation

Five structural surface-targeting interventions all LOSS (#2933, #2946, #2952, #2956, #2961). Alphonse's headline student insight from #2961 closure: *"The meta-signal lives somewhere we haven't probed — three structural interventions all reproduce or worsen the in_dist regression. Suggests it's not a surface-level architectural lever but something about data-balance. Worth a data-distribution probe rather than another model intervention."*

This PR is the FIRST positional-channel data-augmentation experiment in this launch. Direct test of: do CFD point-cloud surrogates benefit from positional jitter regularization?

The geometric OOD splits (camber_rc, camber_cruise) are by definition tests of geometric robustness. If the model memorizes specific coordinate populations during training, it should regress on shifted geometries. Adding noise on x, y during training breaks this memorization.

## Why this might WIN

1. **Targets geometric OOD splits directly.** camber_rc has been the WORST OOD split across many recent LOSS results (#2956 +4.66%, #2958 +7.25%, #2961 +7.18%). camber_cruise is the meta-signal beneficiary. Coordinate noise tests whether shifting positions during training improves geometric generalization.

2. **CFD physics-aware: small coordinate jitter is meaningful.** Real CFD meshes have remeshing variance; σ=0.01 on normalized coordinates simulates this naturally. The pressure/velocity field should be CONTINUOUS in space, so small position shifts should produce small output shifts — model should be robust.

3. **Different mechanism than EVERY prior data experiment.** #2918 input-mixup CATASTROPHIC, #2973 log_Re-noise IN-FLIGHT. Coordinate-channel noise is a strict third axis.

4. **Conservative magnitude: σ=0.01.** If coordinates are normalized to [-1, 1], this is 1% jitter. Tiny enough to be physically valid, large enough to break exact-position memorization.

5. **Zero new params, ~5 lines of code.** Lowest-cost positional regularizer this round.

## Why this might LOSS

1. **σ may be too small or too large.** σ=0.01 is a guess; could need σ=0.001 or σ=0.05.

2. **Coordinates may already implicitly carry useful per-position information** (FiLM modulation depends on Re/AoA, but ATTENTION depends on relative positions). Noise on coords breaks attention's spatial pattern.

3. **In_dist might LOSS.** Adding noise to train coords moves train distribution AWAY from test distribution at the positional level. May hurt in_dist.

4. **Cruise might LOSS.** Cruise's meta-signal WIN may depend on sharp positional specialization that noise breaks.

5. **Underfit at 60ep.** Adding noise slows convergence further.

## Falsifiable predictions

- **WIN** (val < 30.5605): positional robustness helps geometric OOD. Most likely improvement on camber_rc and camber_cruise splits. Try σ=0.005 and σ=0.02 sweeps.
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): mild regularization, marginal effect. Sweep σ magnitude.
- **LOSS** (val > 31.0): coordinate noise hurts — either σ too large, or geometric OOD is NOT driven by positional memorization. Closes positional-channel data-augmentation axis at this magnitude.

## Implementation

### Step 1: Locate where coordinates enter the model

Find the line that extracts mesh coordinates from the batch (probably in dataloader or model forward). The Transolver expects positional channels embedded in the feature tensor `fx` along with other features.

```python
# In the forward pass, find where pos_x, pos_y are accessed:
# Common pattern (verify with the actual code):
pos_x = batch['pos_x']   # or part of x[:, ?, pos_x_channel]
pos_y = batch['pos_y']
```

NOTE: Student should verify the EXACT location and shape via `print` statements at startup. The coordinates may be inside `fx` as the first 2 channels.

### Step 2: Add Gaussian noise during training

```python
# Just BEFORE coordinates enter the model (or just inside model.forward when training=True):
if self.training:
    pos_x_noise = torch.randn_like(pos_x) * 0.01
    pos_y_noise = torch.randn_like(pos_y) * 0.01
    pos_x = pos_x + pos_x_noise
    pos_y = pos_y + pos_y_noise
# else: no change at eval
```

NOTE: Eval/test should NEVER see noise. Confirm via `model.training` flag (False at eval, True at train).

### Step 3: Startup diagnostics

```python
print(f"Input noise: pos_x, pos_y += N(0, 0.01^2) during training only")
print(f"pos_x shape/range: {pos_x.shape} [{pos_x.min():.3f}, {pos_x.max():.3f}], mean={pos_x.mean():.3f}, std={pos_x.std():.3f}")
print(f"pos_y shape/range: {pos_y.shape} [{pos_y.min():.3f}, {pos_y.max():.3f}], mean={pos_y.mean():.3f}, std={pos_y.std():.3f}")
print(f"Effective noise magnitude: σ=0.01 relative to coord scale ~{max(abs(pos_x.min()), abs(pos_x.max())):.3f} = {0.01 / max(abs(pos_x.min()), abs(pos_x.max())):.1%}")
print(f"Total params: {sum(p.numel() for p in model.parameters())}")  # expect 407,940 unchanged
```

### Step 4: Per-epoch logging

Same as baseline. Watch for:
- Slower convergence in early epochs (expected: noise should slow ep1-5)
- Geometric splits (camber_rc, camber_cruise) — does either improve faster than baseline?
- val_avg trajectory — does it converge to a lower point, even if slower?

### Step 5: Stability monitoring

No NaN/Inf expected — noise σ=0.01 is small. Watch ep1-3 for any divergence.

## Baseline (PR #2879) and recent comparable closures

| Metric | Baseline | #2918 (input-mixup α=0.2) | #2973 (log_Re noise IN-FLIGHT) | This PR target |
|---|---|---|---|---|
| val_avg/mae_surf_p | **30.5605** | 37.0259 (+21.16% LOSS) | TBD | beat baseline |
| Mechanism | — | full input + mask mixup across samples | log_Re scalar noise | mesh coord noise |
| Param count | 407,940 | 407,940 | 407,940 | **407,940** |
| Data channel | — | ALL channels mixed | CONDITIONING channel | **POSITIONAL channel** |

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-fern \
    --experiment_name "charliepai2g48h5-fern/coord-noise-sigma-0.01" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Reporting

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Especially close attention to camber_rc and camber_cruise (geometric OOD splits)** — are either most-improved?
4. Param count (unchanged 407,940)
5. Epochs completed, sec/epoch, peak GPU memory
6. Train→val gap at convergence (does noise reduce gap = better regularization, or grow it = under-fit?)
7. **Meta-signal check:** does cruise WIN / in_dist LOSS pattern repeat, attenuate, or invert?
8. **Coordinate statistics at startup** (shape, range, distribution) — for verifying noise magnitude is reasonable relative to coord scale
9. **Plain-language verdict:** WIN / PARTIAL / LOSS
