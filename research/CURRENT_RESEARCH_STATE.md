# SENPAI Research State

- **As of:** 2026-05-12 (UTC)
- **Track:** `willow-pai2g-24h-r4` (round 4 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — branch and PRs are scoped strictly to `icml-appendix-willow-pai2g-24h-r4`; do not cross-reference other rounds. 30-min hard cap on every training run (`SENPAI_TIMEOUT_MINUTES=30`).
- **Primary metric:** `test_avg/mae_surf_p` (validation analogue: `val_avg/mae_surf_p`).

## Current research focus

R4 begins with no merged improvements on this branch. The first round attacks
"easy levers" — single-knob changes that should land signal even inside a
30-min training window where the baseline only completes a handful of epochs.
The goal is to identify which orthogonal directions (optimization, loss
formulation, capacity) move `val_avg/mae_surf_p` most efficiently per training
minute, so subsequent rounds can stack them.

## Round-1 hypothesis families (one student each)

| Student | Hypothesis |
|---------|------------|
| alphonse | Higher peak LR (1e-3) with 3-epoch linear warmup + cosine |
| askeladd | Replace MSE with Smooth L1 (Huber, beta=1.0) — aligns loss with MAE |
| edward | Channel-weighted loss: p:3, Ux:1, Uy:1 — focus gradient on primary metric |
| fern | Higher surf_weight=25 (up from 10) |
| frieren | More slice tokens: slice_num=128 (up from 64) |
| nezuko | OneCycleLR (max_lr=1e-3, pct_start=0.1) — fast convergence in fixed budget |
| tanjiro | Bigger model: n_hidden=192, n_head=6 |
| thorfinn | bfloat16 mixed precision + grad_clip=1.0 — fits more epochs in 30 min |

## Potential next research directions

- Stacking winners from round 1 (e.g. higher-LR + smooth-L1 + channel weights).
- Architecture: more slice tokens with depth-vs-width tradeoffs; SwiGLU MLP; rotary positional embeddings for node coords.
- Loss: relative-MAE / log-domain pressure regression to handle the dynamic
  range across Re (single std varies up to 10x); per-domain or per-Re reweighting.
- Sampler: stratify minibatches by Re or by domain to reduce gradient variance.
- Data augmentation: x-mirror, AoA sign flip for cruise foils, coordinate jitter.
- Mixed precision (if not already deployed by thorfinn) — universal speedup.
- Test-time augmentation: average predictions across mirrored geometries.
- Better positional features: explicit signed-distance to closest foil surface.

This is a living doc — prune entries once they're tried or superseded.
