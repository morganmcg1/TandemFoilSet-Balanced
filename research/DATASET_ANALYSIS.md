# TandemFoilSet — Dataset Analysis (advisor reference)

This is the advisor-side analytical reference for round 5. The authoritative
data contract is in `target/program.md` and `target/data/SPLITS.md`; this file
adds aggregations and observations useful for experiment design.

## Corpus structure

| Domain | Train | Mean nodes | Re range | AoA range |
|---|---|---|---|---|
| RaceCar single (file 0) | 599 | ~85K | 104K–5M | -10° to 0° |
| RaceCar tandem (files 1, 3) | 457 | ~127K | 1.0M–5M | -10° to 0° |
| Cruise tandem (files 4, 6) | 443 | ~210K | 110K–5M | -5° to +6° |

- **Balanced sampler** equalizes the three domains via
  `WeightedRandomSampler(sample_weights, replacement=True)`.
- **Variable mesh size** (74K to 242K nodes); `pad_collate` pads to the
  per-batch max. **`mask` must always be used** in any custom loss.
- **24 input features** (positions, signed arc-length, distance shape descriptor,
  is_surface, log(Re), AoAs, NACA M/P/T, gap, stagger). 18-23 are zero for
  single-foil samples — those carry the tandem distinction.
- **3 outputs** `[Ux, Uy, p]`, all in the original physical space. Model
  predicts in normalized space; `data/scoring.py` denormalizes before MAE.

## Target magnitude profile (val splits)

| Split | y range | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|
| `val_single_in_dist` | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` | (-7,648, +2,648) | 164 | 506 |

High-Re samples drive the extremes — per-sample y std varies by an order of
magnitude even **within** a single split. This is the main numerical headache:
the same model has to predict both small-magnitude and very-large-magnitude
fields.

## Splits — what they actually test

- `val_single_in_dist` / `test_single_in_dist`: in-distribution single-foil
  sanity check (train sees this domain).
- `val_geom_camber_rc` / `test_geom_camber_rc`: held-out front foil camber
  M=6-8 for raceCar tandem — geometry interpolation between M=2-5 (P1) and
  M=9+specials (P3).
- `val_geom_camber_cruise` / `test_geom_camber_cruise`: held-out front foil
  camber M=2-4 for cruise tandem — geometry interpolation between M=0-2 (P1)
  and M=4-6 (P3).
- `val_re_rand` / `test_re_rand`: stratified Re holdout across all tandem
  training domains — cross-Reynolds-regime generalization.

**Primary ranking metric:** equal-weight mean across the four splits of
`mae_surf_p` (surface pressure MAE in physical units, global accumulation
over surface nodes, float64). Surface MAE is `sum|err| / n_surf_nodes`, not a
per-sample average.

## Implications for experiment design

1. **Surface vs volume loss balance is the main lever.** The training loss is
   `vol_loss + surf_weight * surf_loss`. Surface nodes are a small fraction of
   nodes per sample, so without weighting the model under-prioritizes the very
   thing we score on. Default `surf_weight=10.0`; sweeping it is a cheap, direct
   intervention.
2. **Channel imbalance matters.** The ranking metric is surface **pressure**
   only, but the loss treats all three channels equally. Reweighting per
   channel is a one-line, high-leverage change.
3. **Loss shape matters at high-Re extremes.** MSE in normalized space puts
   disproportionate weight on the tail of the y distribution. Swapping to
   Smooth-L1 (Huber) is a simple robustness move that aligns better with the
   eventual MAE metric.
4. **Architecture scale is a moderate lever.** With 96 GB VRAM, moderate
   scale-ups (n_hidden 128→192, n_layers 5→7, slice_num 64→128) all fit and
   are reasonable bets.
5. **Generalization split disagreement = signal.** When wins on
   `val_single_in_dist` don't carry over to `val_geom_camber_*` or `val_re_rand`,
   the change is overfitting in-distribution. Always check per-split deltas,
   not just the average.

## Operational reminders

- **No remote experiment logging** for this arm (no W&B / wandb).
- **30-minute wall-clock cap per training execution.** Don't design hypotheses
  that require longer runs; choose changes whose signal shows within ~15-30
  epochs.
- **No new packages** outside of `pyproject.toml` without adding them in the
  same PR.
- **Data loaders are read-only.** Custom samplers / feature transforms must
  live in `train.py`.
- **Mask discipline.** Any custom loss or pooling that does not respect
  `mask` will silently corrupt metrics.
