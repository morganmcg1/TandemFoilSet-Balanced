# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 2)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 126.0751`** — PR #1502 (BIVW, tanjiro), merged.
`test_avg/mae_surf_p = NaN` due to known cruise-split data corruption (fix in progress via PR #1527).

## Current research focus and themes

**Cycle 2.** Two results are in; BIVW (per-sample inverse-variance loss weighting) is the round-4 baseline at 126.08. The standalone surface-correction head (PR #1503) was 6% worse at the same budget and closed. Both results surfaced a pre-existing infrastructure NaN in `test_avg/mae_surf_p` that must be fixed before paper metrics are reliable.

Primary focus: **can we beat BIVW's 126.08?** Six hypotheses from the opening fan-out are still WIP (architecture and schedule families). Two new PRs target pipeline robustness and composition.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Pressure channel x3 in MSE loss |
| 1497 | askeladd | warmup-cosine-lr | WIP | 5-epoch linear LR warmup |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP | Doubled MLP width |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP | grad_clip=1 + lr=1e-3 |
| 1500 | frieren | larger-hidden-dim (128 to 256) | WIP | Bigger transformer |
| 1501 | nezuko | more-slices (64 to 128) | WIP | More slice tokens |
| 1527 | tanjiro | fix-test-nan-scoring | WIP | **Pipeline fix**: guard evaluate_split |
| 1528 | thorfinn | surf-head-on-bivw | WIP | **Composition**: BIVW + surface head |

## Working hypotheses

1. **BIVW wins by re-balancing Re-driven gradient heterogeneity** — confirmed (PR #1502). Low-Re cruise split now at 97.2. All future comparisons beat 126.0751.
2. **Pressure-channel emphasis is orthogonal to BIVW** — PR #1496 tests pure channel weighting. If it wins, compose `BIVW + channel-weight` next.
3. **Architecture capacity (MLP width, hidden, slices) may be the next bottleneck** — PRs #1498, #1500, #1501 cover three independent axes on top of BIVW.
4. **Schedule warmup helps softmax-gated attention** — PR #1497 is the cleanest test.
5. **Test NaN is a pipeline bug** — both students independently diagnosed the root cause. PR #1527 fixes it in train.py (scoring.py is read-only).

## Potential next directions

- **Compose BIVW + channel-weight** (if #1496 wins)
- **Compose BIVW + warmup-cosine** (if #1497 wins)
- **BIVW + wider-MLP + warmup** triple composition (if all three win)
- **SmoothL1 surface loss** (Huber on surface — aligns with L1 metric)
- **Per-channel BIVW** — current pools variance across all 3 channels; per-channel variant
- **Re-bin sampler** (stratified Re sampling, complement to loss-level IVW)
- **Transolver++ slice variants** (if #1501 wins, follow arXiv:2505.02107)
- **BF16/AMP** for ~1.5x epochs/30 min (free win on any winning recipe)

## Known issues

- **test_avg/mae_surf_p = NaN** on all current runs: `test_geom_camber_cruise` sample 20 has 761 `-inf` GT p-values; scoring.py `0 x inf = NaN` bug poisons the accumulator. Fix pending in PR #1527.
- Until #1527 merges: use the 3-split test mean (excluding cruise) as proxy paper metric.
