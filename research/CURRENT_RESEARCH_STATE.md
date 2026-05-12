# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 4)
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
| R4 cycle 3 | #1528 | BIVW + zero-init surf-head | **119.2987** | −5.37% |

## Current research focus and themes

**Cycle 4.** Current best is BIVW + surf-head at 119.30. Key open questions:

1. Does grad-clip + higher-LR (#1499) stack on top of surf-head? Fern is rebasing; we saw 113.15 on BIVW-alone — if it holds on the new baseline this would be a major improvement.
2. Does SmoothL1 surface loss (#1558) help? Aligns MSE optimization with MAE evaluation metric; thorfinn running delta=1.0 and 0.5 arms.
3. Does BF16/AMP (#1559) unlock capacity? Frieren showed n_hidden=256 is 4× slower than expected — BF16 should halve VRAM and speed (~1.5×), enabling capacity experiments to be fairly evaluated.
4. What do the capacity/schedule experiments (#1496 alphonse, #1497 askeladd, #1498 edward, #1501 nezuko) show? All 4 are still WIP (pods healthy, running). Results expected soon.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Pressure channel x3 in MSE loss |
| 1497 | askeladd | warmup-cosine-lr | WIP | 5-epoch linear LR warmup |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP | Doubled MLP width |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP (rebase) | Rebasing on new baseline (119.30); adding clip=10.0 arm |
| 1501 | nezuko | more-slices (64 to 128) | WIP | More slice tokens |
| 1527 | tanjiro | fix-test-nan-scoring | WIP | **Pipeline fix**: guard evaluate_split |
| 1558 | thorfinn | smooth-l1-surface-loss | WIP | Huber delta=1.0 and 0.5 arms |
| 1559 | frieren | bf16-mixed-precision | WIP | BF16 for n_hidden=128 and n_hidden=256 arms |

## Working hypotheses

1. **BIVW wins by re-balancing Re-driven gradient heterogeneity** — confirmed (PR #1502).
2. **Surf-head specialisation composes with BIVW** — confirmed (PR #1528, −5.37%).
3. **Grad-clip + higher LR beats BIVW-only** — strong signal at 113.15 (PR #1499). Rebasing on new baseline now.
4. **SmoothL1 surface loss aligns opt objective with MAE metric** — testing in PR #1558.
5. **BF16/AMP unlocks wall-clock for capacity experiments** — testing in PR #1559. n_hidden=256 cannot be fairly evaluated at FP32 under 30-min cap.
6. **Architecture capacity (MLP width, slices) may be bottleneck** — PRs #1498, #1501 still WIP.
7. **Schedule warmup helps softmax-gated attention** — PR #1497 still WIP.
8. **Pressure-channel emphasis is orthogonal to BIVW** — PR #1496 still WIP.
9. **Test NaN is a pipeline bug** — PR #1527 still WIP.

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head without BIVW) — 6.2% worse than BIVW. Composition (#1528) succeeded.
- **PR #1500** (n_hidden 128→256 at FP32) — budget failure, only 8/50 epochs in 30 min. n_hidden=256 needs BF16 to be competitive.

## Potential next directions

- **Compose BIVW + clip+LR + surf-head** — highest priority if #1499 rebase beats 119.30 (likely strongest composition)
- **Compose BIVW + SmoothL1 + surf-head** — if #1558 wins
- **Capacity scaling with BF16** — once #1559 validates AMP, n_hidden=256 can be retested
- **Per-channel BIVW** — pool per-channel variance rather than across all 3 channels; complements sample-level BIVW
- **Warmup + BIVW composition** — if #1497 wins vs 119.30
- **Re-bin sampler** — stratified Re sampling, orthogonal to loss-level IVW
- **Decoupled LR for surf-head** — backbone at 5e-4, surf-head at 5× LR for faster specialisation
- **Address val_geom_camber_rc OOD regression** — currently +6.84% vs BIVW-alone in #1528

## Known issues

- **test_avg/mae_surf_p = NaN** on all runs: cruise split bug (PR #1527 in progress).
- **val_geom_camber_rc regression (+6.84%)** in #1528: surf-head improves overall avg but hurts OOD raceCar camber split.
- **Slice-attention VRAM scaling**: at N=242K nodes, actual VRAM for n_hidden=256 is ~42GB (not ~5GB). BF16 required for capacity experiments.
