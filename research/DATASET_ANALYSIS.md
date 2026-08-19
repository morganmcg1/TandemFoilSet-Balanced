# TandemFoilSet Dataset Analysis

## Scope and evaluation target

This launch studies surrogate prediction of the full `(Ux, Uy, p)` field on irregular overset meshes. The primary ranking quantity is lower `val_avg/mae_surf_p`, the equal-weight mean of global surface-pressure MAE over four validation splits. Final paper-facing evidence is the corresponding `test_avg/mae_surf_p` from the checkpoint selected by the validation metric. Every candidate must report all four split values and the four test values from the best validation-pressure checkpoint; a normalized loss or a single favorable split is not sufficient.

No launch-scoped terminal experiment or W&B run was available when this analysis was written. Historical runs outside this launch are intentionally excluded. The first exact-code run must therefore establish a reproducible numerical anchor before any later PR claims an improvement.

## Corpus and split design

The training corpus has 1,499 samples across three physical domains:

- RaceCar single: 599 samples, roughly 85K nodes per mesh, inverted airfoil with ground effect.
- RaceCar tandem: 457 samples, roughly 127K nodes per mesh, dual inverted foils.
- Cruise tandem: 443 samples, roughly 210K nodes per mesh, dual freestream foils.

The loader uses inverse-domain-size weights so the three domains are sampled equally by sample count. However, the baseline loss averages over all valid nodes in a batch. Consequently, a cruise sample contributes more gradient terms than a raceCar-single sample, and the existing sampler is not equivalent to node-balanced or per-sample-balanced optimization.

There are four validation tracks, each with 100 samples, and four paired test tracks, each with 200 samples:

1. `single_in_dist`: random single-foil holdout; a sanity check.
2. `geom_camber_rc`: raceCar tandem front-foil camber M=6-8 held out by file; tests geometry interpolation.
3. `geom_camber_cruise`: cruise tandem front-foil camber M=2-4 held out by file; tests geometry interpolation.
4. `re_rand`: stratified Reynolds-number holdout across tandem domains; this is principally interpolation because nearby Reynolds values remain in training.

The tandem geometry split shares rear-foil families between train and validation while holding out front-foil files. Thus it tests front-camber interpolation, not fully compositional rear-geometry generalization. There is no official AoA-tail holdout, rear-foil holdout, or joint geometry/Re extrapolation split in this launch.

## Features and physical conditioning

Each node has 24 input features: position `(x,z)`, signed arc-length features, eight distance-based shape descriptors, a surface flag, `log(Re)`, AoA and three NACA parameters for foil 1, AoA and three NACA parameters for foil 2, gap, and stagger. Single-foil samples use zeros for foil-2 metadata, gap, and stagger. The targets are `[Ux, Uy, p]`.

The metadata is useful but has two limitations. First, the model must learn smooth dependence on Reynolds number, AoA, gap, stagger, and continuous NACA parameters from scalar inputs. Second, invalid/non-NACA profiles can share zero-valued NACA metadata with absent foil-2 fields, so the model may need to infer the distinction from node-local descriptors and other features. The collapsed boolean surface flag also does not distinguish front from rear foil surfaces.

Target magnitudes vary strongly with Reynolds number and domain. High-Re samples drive extreme values, while per-sample target standard deviations vary by roughly an order of magnitude. Global training statistics and global normalized MSE can therefore favor high-magnitude regimes even though the official metric is physical-space absolute error and gives each validation split equal weight.

## Batching and masking risk

Meshes range from approximately 74K to 242K nodes. `pad_collate` pads a batch to its largest mesh and returns `mask` along with `x`, `y`, and `is_surface`. The loss and scorer correctly exclude invalid padded nodes. The model currently receives only normalized `x`, however. Because normalization is applied after zero padding, padded features become `-x_mean/x_std`, not neutral zeros. Physics attention still includes those positions when computing slice weights and slice tokens. This creates a concrete batch-composition-dependent contamination path: valid predictions can depend on how much padding is present in the batch.

The first implementation test should pass the mask into the model and exclude padded nodes from attention aggregation while preserving the existing output and scoring contracts. A batch-size-1 diagnostic can independently estimate the magnitude of the contamination, but it should be treated as a diagnostic rather than a model improvement unless its implementation differs only in batching.

## Baseline training contract

The exact current Transolver is a 24-to-128 model with five physics-attention blocks, four heads, 64 latent slices, MLP ratio 2, and three output channels. Training uses normalized targets, masked volume and surface squared error, `loss = vol_loss + 10 * surf_loss`, AdamW with learning rate `5e-4` and weight decay `1e-4`, and cosine decay over 50 epochs. Regular runs use batch size 4, replacement sampling, and the authoritative one-GPU, 55-minute, 50-epoch limits.

The model must continue to output normalized `[B,N,3]` predictions. Any custom loss or pooling must use the real-node mask and surface mask. Validation checkpoint selection and test evaluation must remain unchanged so comparisons are apples-to-apples.

## Initial falsifiable hypotheses

1. **Attention masking:** padded normalized values currently enter physics attention; mask-aware slice aggregation should reduce batch-size sensitivity and improve surface pressure, especially on mixed mesh-size batches.
2. **Per-sample/domain weighting:** global node averaging overweights large cruise meshes; averaging valid-node losses per sample, or explicitly equalizing domain contributions, should improve the equal-weight split objective without changing the sampler.
3. **Pressure-focused objective:** the official objective is surface pressure MAE, while the baseline surface MSE weights Ux, Uy, and p equally in normalized space. A pressure-channel multiplier or robust surface-pressure term should improve the primary metric if velocity gradients are consuming capacity.
4. **Reynolds conditioning:** a continuous embedding of `log(Re)` or Fourier features may improve smooth interpolation across the wide dynamic range; gains must hold across the Re-random and both geometry tracks, not only in-distribution single foil.
5. **Capacity/resolution:** modest changes to latent-slice resolution, MLP ratio, or learning rate can test whether the five-block baseline is under-capacity or under-updated, but each PR must vary one lever only and monitor VRAM/time.
6. **Geometry conditioning:** explicit interactions among NACA metadata, AoA, gap, and stagger may improve held-out front-camber interpolation; report raceCar and cruise geometry splits separately because their parameter ranges differ.

## Required evidence and interpretation

A terminal result is valid only when every primary validation and test metric is finite and present, the best checkpoint is identified by `val_avg/mae_surf_p`, and the result includes direct W&B run links. Compare the four split values before the equal-weight average, inspect surface velocity and volume diagnostics for regressions, and record wall time, epoch, VRAM/OOM status, and exact command. A small primary improvement with no disproportionate complexity is meaningful; a gain confined to one split or accompanied by a severe robustness regression requires follow-up rather than unconditional adoption.
