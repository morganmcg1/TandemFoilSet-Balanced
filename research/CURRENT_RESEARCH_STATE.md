# SENPAI Research State

- **As of:** 2026-05-12 ~19:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 4 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`).
- **Current best:** val_avg/mae_surf_p = **146.2510** (PR #1396, slice_num=128)
- **test_avg blocked by scoring bug** — `data/scoring.py` NaN propagation (fix in PR #1521)

## Current research focus

First round of experiments completed. Key findings:
1. **slice_num=128 works** (val 146.25). Merged as new baseline. Model still descending at 30-min cutoff — more epochs or faster training would help.
2. **Scoring bug blocking all test metrics** — NaN GT in test_geom_camber_cruise sample 20 leaks through `err*mask`. Fix (`nan_to_num`) assigned to frieren (PR #1521).
3. **OneCycleLR needs correct total_steps** — sized for 50 epochs but only ~12 achievable. Retry assigned to nezuko.
4. **Bigger hidden (n_hidden=192) needs retry** on new baseline (slice_num=128) — original run only got 9 epochs. Retry assigned to tanjiro (PR #1522).

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1373 | lr-warmup-1e-3 | WIP |
| askeladd | #1379 | smooth-l1-loss | WIP |
| edward | #1383 | p-channel-weight | WIP |
| fern | #1390 | higher-surf-weight | WIP |
| frieren | #1521 | **fix-scoring-nan** (data/scoring.py) | WIP |
| nezuko | #1404 | onecycle-lr (corrected total_steps) | Sent back / WIP |
| tanjiro | #1522 | hidden192-on-slice128 | WIP |
| thorfinn | #1415 | bf16-amp | WIP |

## Potential next research directions (after scoring fix lands)

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
