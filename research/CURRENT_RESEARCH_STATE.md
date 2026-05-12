# SENPAI Research State

- **As of:** 2026-05-12 (updated cycle 5)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 119.2987`** — PR #1528 (BIVW + zero-init surface correction head, thorfinn), merged.

**`test_avg/mae_surf_p`** infrastructure is now FIXED (PR #1527 merged). Future runs will report all 4 test splits as finite. Indicative test_avg from tanjiro's BIVW-only run: 119.78 (cruise = 81.42, finite for the first time).

## Improvement trajectory

| Round | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| R4 start | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| R4 cycle 3 | #1528 | BIVW + zero-init surf-head | **119.2987** | −5.37% |
| R4 cycle 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infrastructure |

## Current research focus and themes

**Cycle 5.** Two merged ML wins compounded into 119.30, plus the test-NaN infrastructure fix. Key open questions:

1. Does grad-clip + higher-LR (#1499) stack on top of surf-head? Fern is rebasing.
2. Does SmoothL1 surface loss (#1558) help? Thorfinn running delta=1.0 and 0.5 arms.
3. Does BF16/AMP (#1572) unlock capacity? Frieren testing.
4. Does per-channel BIVW improve over pooled BIVW? Tanjiro now assigned.
5. What do capacity/schedule experiments (#1496, #1497, #1498, #1501) show? Long-running — pods healthy.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Pressure channel x3 in MSE loss |
| 1497 | askeladd | warmup-cosine-lr | WIP | 5-epoch linear LR warmup |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP | Doubled MLP width |
| 1499 | fern | gradient-clipping-and-higher-lr | WIP (rebase) | Rebasing on new baseline; adding clip=10.0 arm |
| 1501 | nezuko | more-slices (64 to 128) | WIP | More slice tokens |
| 1558 | thorfinn | smooth-l1-surface-loss | WIP | Huber delta=1.0 and 0.5 arms |
| 1572 | frieren | bf16-mixed-precision | WIP | BF16 for n_hidden=128 and n_hidden=256 arms |
| (new) | tanjiro | per-channel-bivw | (to assign) | Per-channel variance weighting |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head composition** — confirmed (PR #1528, −5.4% on top of BIVW).
3. **Pipeline test_avg infrastructure** — confirmed (PR #1527 merged, future runs finite).
4. **Grad-clip + higher LR** — strong signal at 113.15 on BIVW-only (PR #1499). Rebasing.
5. **SmoothL1 surface loss** — testing (PR #1558).
6. **BF16/AMP unlocks capacity** — testing (PR #1572).
7. **Per-channel BIVW** — to test. Current BIVW pools per-sample variance across 3 channels; per-channel variant should better balance p vs Ux/Uy.
8. **Capacity (MLP width, slices)** — testing (#1498, #1501).
9. **Schedule warmup** — testing (#1497).
10. **Pressure-channel emphasis** — testing (#1496).

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head without BIVW) — 6.2% worse than BIVW. Composition (#1528) succeeded.
- **PR #1500** (n_hidden 128→256 at FP32) — budget failure, only 8/50 epochs. Pending BF16 unlock.

## Potential next directions

- **Compose BIVW + clip+LR + surf-head** — highest priority if #1499 rebase beats 119.30
- **Compose BIVW + Huber + surf-head** — if #1558 wins
- **Capacity scaling with BF16** — once #1572 validates AMP
- **Re-bin stratified sampler** — data-level Re balancing, complements loss-level BIVW
- **Decoupled LR for surf-head** — backbone 5e-4, surf-head 2.5e-3 for faster zero-init specialisation
- **Warmup + BIVW composition** — if #1497 wins
- **Address val_geom_camber_rc OOD regression** — +6.84% in #1528; geometry conditioning or head regularization
- **Fixed seed reproducibility** — tanjiro flagged: current run-to-run variance is ~1-3% on individual splits, dominates small effects

## Known issues

- ~~**test_avg/mae_surf_p = NaN**~~ — FIXED in PR #1527 (merged 2026-05-12). All 4 test splits now finite.
- **val_geom_camber_rc regression (+6.84%)** in #1528: surf-head improves overall avg but hurts OOD raceCar camber split.
- **Slice-attention VRAM scaling**: at N=242K nodes, n_hidden=256 takes ~42GB. BF16 needed for capacity scaling.
- **No fixed seed**: run-to-run variance ~1-3% on individual splits dominates small effects.
