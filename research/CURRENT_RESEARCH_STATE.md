# SENPAI Research State

- **Date:** 2026-05-12
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none yet — controlled 24h/48h Charlie-vs-Willow
  logging ablation. Local JSONL metrics only.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).

## Current focus

Round 1 — seeding eight diverse, high-leverage single-variable hypotheses on
the stock Transolver baseline. The primary ranking metric is
`val_avg/mae_surf_p` and the test-time metric is `test_avg/mae_surf_p`.

Hypotheses span five distinct axes so each result is informative regardless of
sign:

| Axis | Idea | Student |
|------|------|---------|
| Loss alignment | L1 (MAE) loss in normalized space — directly optimize the ranking metric | alphonse |
| Generalization | EMA-of-weights for validation/test evaluation | askeladd |
| Width capacity | `n_hidden` 128 → 192 | edward |
| Physics attention bottleneck | `slice_num` 64 → 128 | fern |
| Surface focus | `surf_weight` 10 → 25 | frieren |
| Depth capacity | `n_layers` 5 → 8 | nezuko |
| LR schedule | Linear warmup (3 ep) + cosine, peak `lr=1e-3` | tanjiro |
| MLP capacity | `mlp_ratio` 2 → 4 | thorfinn |

## Why these eight

The 30-minute per-run cap rewards cheap, single-knob changes that produce a
clear val signal in 4–8 epochs. Splitting the round across loss / generalization
/ capacity / schedule / architecture gives us the most information per GPU
hour, and avoids "two-of-the-same-flavor" outcomes where one regression refutes
multiple hypotheses at once. Each PR tests **one** change vs. stock baseline.

## Potential next directions

After round 1 we will know:

1. Whether the baseline is undersized (width vs. depth vs. mlp_ratio).
2. Whether the loss/schedule is leaving improvement on the table (L1, warmup,
   surface weight).
3. Whether cheap inference-time tricks (EMA) help generalization.

Probable round-2 directions, conditional on round-1 signal:

- **Compound winners.** Stack the orthogonal winners (e.g. L1 loss + EMA +
  larger slice_num) into a single follow-up.
- **Input feature engineering.** Fourier positional features on `(x,z)`,
  `unified_pos=True`, log(Re) embedding into slice token conditioning.
- **Loss reformulation.** Channel-aware (down-weight Ux/Uy vol, up-weight p
  surf), Huber on p, gradient-smoothness regularizer.
- **Optimization.** EMA-only, SWA at end of training, gradient clipping, lr
  re-tuning if width/depth wins.
- **Augmentation / sampling.** Mesh-coarsening as augmentation, mesh-size-aware
  batching to cut padding waste.
- **Bigger architecture swings.** If we hit a plateau, move to a different
  backbone family (GNO, PointNet++, GraphCast-style message passing, or a
  hybrid GNN-attention).

## Open questions

- What does 30 minutes of training actually buy in epochs at baseline? (Round 1
  will answer.)
- Is the loss/metric mismatch (MSE-train / MAE-eval) costing >2% on the primary
  metric?
- Are 64 slices enough to model 242K-node cruise meshes?
