# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-28 (round 1 kickoff)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`

## Current research focus

Establish a working baseline on `val_avg/mae_surf_p` and `test_avg/mae_surf_p`
by exploring the **stock-Transolver knobs that are most likely to help out of
the box** before committing to architectural changes. The stock config is
visibly under-tuned along several axes:

- The Transolver is small (~512K params) while VRAM is ~96 GB — capacity is
  available.
- Loss is MSE while the metric is MAE — there is a known loss/metric mismatch.
- `surf_weight` and per-channel weights have not been swept; surface pressure
  is the only thing the headline metric measures.
- Default `lr=5e-4` with no warmup and `batch_size=4` are conservative.

Round 1 covers eight orthogonal levers so we can read the gradient of the
landscape after a single round of merges.

## Round 1 hypotheses (one per student)

| Student | Lever | Predicted edge |
|---------|-------|----------------|
| alphonse | Capacity scale-up (`n_hidden=256, n_layers=8`) | Headroom from underused VRAM |
| askeladd | L1 loss in normalized space | Loss/metric alignment |
| edward | `surf_weight` sweep up (10 → 30) | Direct upweight of headline metric |
| fern | Per-channel loss weight on `p` (3×) | Pressure is the only ranked channel |
| frieren | More physics slices (`slice_num=64 → 128`) | Finer slice decomposition |
| nezuko | LR warmup (5% linear) + cosine | Adam stability early in training |
| tanjiro | Higher peak LR (1e-3) with 10% warmup | Default 5e-4 is conservative |
| thorfinn | Larger batch (`batch_size=8`) | Better gradient estimates, VRAM available |

## Potential next research directions (round 2 onwards)

These are not assigned yet — they are queued for after we have round 1 data:

1. **Loss reformulation continued** — Charbonnier / Huber, log-cosh, or
   physics-aware blends (e.g. weighted MAE in physical-units space).
2. **Output target reparameterization** — predict `Cp` (`p / (0.5 ρ U_inf²)`)
   instead of raw `p`; should regularize the high-Re extremes.
3. **Conditioning improvements** — separate embedding for `log(Re)`, AoA,
   gap/stagger; FiLM-style conditioning of slice tokens.
4. **Mesh/domain-aware features** — explicit foil-1 vs foil-2 marker, distance
   to nearest surface as a feature, signed distance for a richer geometry
   prior.
5. **Architecture replacements** — GINO/Geo-FNO style spectral kernel, neural
   operator alternatives, or a Transformer with rotary position encoding over
   `(x, z)` instead of slice attention.
6. **Sampler / class-balancing** — stratify by Re bin in addition to domain;
   downweight low-Re samples that dominate the easy metric, upweight high-Re
   that drive the cross-Re holdout.
7. **Data augmentation** — vertical mirror flip (sign-flip `Uy`, `z`, AoA),
   small mesh-node dropout, Re jitter.
8. **Test-time tricks** — TTA (mirror), seed/checkpoint ensembling.
9. **Distillation / self-training** on the unlabeled test inputs (legitimate
   given inputs are in the public test set).

## Notes / constraints

- `SENPAI_MAX_EPOCHS=50` and `SENPAI_TIMEOUT_MINUTES=30` are hard caps. Don't
  override them — design experiments that fit in 30 minutes wall-clock.
- All four val splits are weighted equally in the headline metric. Prefer
  changes that travel across splits over hacks that only help one.
- Test metrics are computed at the end of every run from the best-val
  checkpoint — they are the ranking quantity for the paper.
