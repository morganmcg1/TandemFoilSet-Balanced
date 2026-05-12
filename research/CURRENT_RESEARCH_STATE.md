# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 3)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 119.2987`** — PR #1528 (BIVW + zero-init surface correction head, thorfinn), merged.
`test_avg/mae_surf_p = NaN` due to known cruise-split data corruption (fix in progress via PR #1527).
3-split test mean ≈ 119.63 (indicative proxy).

## Improvement trajectory

| Round | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| R4 start | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| R4 cycle 2 | #1528 | BIVW + zero-init surf-head | **119.2987** | −5.37% |

## Current research focus and themes

**Cycle 3.** Two wins confirmed: BIVW (126.08) then BIVW+surf_head (119.30). The composition of a sample-level loss rebalancer (BIVW) and an architectural surface specialiser (surf-head) validated the orthogonality hypothesis.

Fern's clip+LR experiment (#1499) reached 113.15 on BIVW alone (before surf-head was merged) — the strongest individual result so far — but had merge conflicts. Sent back for rebase onto the new baseline with additional `--grad_clip 10.0` arm to separate outlier suppression from per-step renormalisation. Key finding: 100% of gradient steps were clipped at max_norm=1.0, meaning the clip was acting as a near-uniform step rescaler, not an outlier guard.

Primary focus: **can we beat 119.30 by stacking more orthogonal improvements?**

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Pressure channel x3 in MSE loss |
| 1497 | askeladd | warmup-cosine-lr | WIP | 5-epoch linear LR warmup |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP | Doubled MLP width |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP (rebase) | Rebase on new baseline; add clip=10.0 arm |
| 1500 | frieren | larger-hidden-dim (128 to 256) | WIP | Bigger transformer |
| 1501 | nezuko | more-slices (64 to 128) | WIP | More slice tokens |
| 1527 | tanjiro | fix-test-nan-scoring | WIP | **Pipeline fix**: guard evaluate_split |
| 1558 | thorfinn | smooth-l1-surface-loss | WIP | Huber delta=1.0 and 0.5 arms |

## Working hypotheses

1. **BIVW wins by re-balancing Re-driven gradient heterogeneity** — confirmed (PR #1502). Low-Re cruise split now at 97.2.
2. **Surf-head specialisation composes with BIVW** — confirmed (PR #1528). Zero-init MLP head at surface nodes adds +5.37% on top of BIVW.
3. **Grad-clip + higher LR substantially beats the BIVW-only baseline** — 113.15 on BIVW alone (PR #1499). Strong signal but needs rebase onto BIVW+surf-head baseline to confirm stacking.
4. **SmoothL1 surface loss aligns opt objective with MAE metric** — testing in PR #1558. The MSE objective–metric gap may be penalising large-error surf nodes too aggressively.
5. **Architecture capacity (MLP width, hidden, slices) may be the next bottleneck** — PRs #1498, #1500, #1501 still WIP.
6. **Schedule warmup helps softmax-gated attention** — PR #1497 still WIP.
7. **Test NaN is a pipeline bug** — PR #1527 still WIP.

## Potential next directions

- **Compose BIVW + clip+LR + surf-head** (if #1499 rebase confirms clip+LR still wins on new baseline — most likely next merge)
- **Compose BIVW + channel-weight** (if #1496 wins vs new 119.30 baseline)
- **Compose BIVW + warmup-cosine** (if #1497 wins vs new 119.30 baseline)
- **Decoupled LR for surf-head** — train backbone at 5e-4, surf-head at 5x LR to accelerate head specialisation
- **Larger surf-head** (64→256 hidden) — current 26K-param head may be capacity-limited
- **Per-channel BIVW** — current BIVW pools variance across all 3 channels; per-channel variant may help p vs Ux/Uy balance
- **Re-bin sampler** (stratified Re sampling, complement to loss-level IVW)
- **BF16/AMP** for ~1.5x epochs/30 min (free win on any winning recipe)
- **Transolver++ slice variants** (if #1501 wins)
- **Address val_geom_camber_rc regression** — this OOD camber split regressed +6.84% in #1528; root-cause unknown; geometry conditioning or head regularization may fix it

## Known issues

- **test_avg/mae_surf_p = NaN** on all current runs: `test_geom_camber_cruise` sample 20 has 761 `-inf` GT p-values; scoring.py `0 x inf = NaN` bug poisons the accumulator. Fix pending in PR #1527.
- Until #1527 merges: use the 3-split test mean (excluding cruise) as proxy paper metric.
- **val_geom_camber_rc regression (+6.84%)** in PR #1528: the surf-head improved 3 of 4 splits but hurt the OOD raceCar camber split. Net avg improvement still positive. Worth monitoring in future PRs.
