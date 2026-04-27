# SENPAI Research State

- 2026-04-27 — fresh launch on `icml-appendix-willow-pai2d-r2`
- No prior PRs on this advisor branch. Baseline = unmodified `train.py`.
- Primary metric: `val_avg/mae_surf_p` (equal-weight surface pressure MAE
  across 4 val tracks). Paper-facing: `test_avg/mae_surf_p`.

## Round 1 focus

This first round is broad, low-cost coverage of the largest standard levers on
a Transolver. Each student gets one orthogonal axis so we can attribute deltas
cleanly. We expect at least 2–3 winners; merge them sequentially, then in
round 2 stack the orthogonal wins and explore the next layer (loss
reformulation, target representation, schedulers, regularization).

### Round 1 axes (one per student, all targeting val_avg/mae_surf_p)

| Student   | Axis                       | Change                                          |
|-----------|----------------------------|-------------------------------------------------|
| alphonse  | model width                | n_hidden 128 → 192                              |
| askeladd  | model depth                | n_layers 5 → 8                                  |
| edward    | FFN ratio                  | mlp_ratio 2 → 4                                 |
| fern      | physics-token count        | slice_num 64 → 128                              |
| frieren   | loss-objective alignment   | MSE → Huber (smooth L1, β=1) in normalized space|
| nezuko    | surface-vs-volume weight   | surf_weight 10 → 25                             |
| tanjiro   | LR schedule                | lr 5e-4 → 1e-3 + 5-epoch linear warmup + cosine |
| thorfinn  | batch + LR scaling         | batch_size 4 → 8, lr 5e-4 → 7e-4 (sqrt scaling) |

## Potential next research directions (post-round-1)

- **Combine winners.** Architecture wins (width/depth/MLP/slice) usually
  stack with optimizer wins (LR/batch/warmup). Pick the strongest of each
  bucket and merge.
- **Target-space reformulation.** Per-sample y std varies by 10× even within
  a domain — try asinh / log1p on the pressure channel, or per-sample
  normalization. Could decouple high-Re extreme samples from dominating
  gradients.
- **Per-channel loss weighting.** Metric is surface pressure only, but loss
  weights all 3 channels equally. Try 2–3× weight on `p` channel.
- **Domain conditioning.** Single-foil vs tandem signal already in features
  18–23. Consider explicit gating or domain embeddings.
- **Bigger models.** If width/depth/MLP each give clean gains, push capacity
  further with an EMA + gradient clipping for stability.
- **Augmentation.** Geometry-preserving x-flip (mirror y-coord and corresponding
  flow components) for the ground-effect raceCar domain.
- **Pretrain on Re only / curriculum.** Sort batches by per-sample y std,
  warmup the model on low-magnitude samples first.

## Constraints we're respecting

- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` are not overridden.
- Capacity-scaling hypotheses (alphonse, askeladd) may hit timeout before
  50 epochs; we accept that and select on best-checkpoint, but instruct the
  student to drop epochs to 40 if validation looks unconverged at the
  timeout cutoff.
- VRAM is 96 GB so BS=8 and wider models are safe even on 242K-node meshes.
