# SENPAI Research State

- **Last updated**: 2026-05-15 ~22:50 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline

- `val_avg/mae_surf_p` = **97.55** (PR #3300, edward, `bf16-mixed-precision`, epoch 17)
- **MERGED 2026-05-15 22:45 UTC**
- Change from Huber baseline: BF16 autocast on forward pass → 1.3x throughput, 5 more epochs in 30-min cap (14→19), VRAM 42.1→32.95 GB. No model changes.

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 114.41 | 1.387 | 0.674 |
| val_geom_camber_rc | 104.96 | 2.060 | 0.851 |
| val_geom_camber_cruise | 79.72 | 1.135 | 0.531 |
| val_re_rand | 91.09 | 1.532 | 0.678 |
| **val_avg** | **97.55** | 1.529 | 0.684 |

## Key observations
1. **BF16 is THE throughput unlock**: 14→19 epochs in 30-min cap, -17.1% on val_avg. This is now baked into the baseline — all future experiments start here.
2. **Budget mismatch still persists**: T_max=50 but we only reach ~19 epochs. At best-val epoch (17), LR is still 74% of initial (not annealed). This is the next lever.
3. **VRAM headroom opened**: 32.95 GB used out of 96 GB = 63 GB free. Enables n_hidden=192 or batch_size=8 without memory concerns.
4. **NaN bug persists** in `test_geom_camber_cruise`: `inf` in GT of sample 20. Workaround: rank on val_avg/mae_surf_p; report test_avg as clean-3 mean.
5. **Stale_wip is still pandemic** for 6 of 7 remaining PRs (alphonse, askeladd, frieren, nezuko, tanjiro, thorfinn). Push-flow has worked for fern (rebase pushed) and thorfinn (result pushed). Pattern: training completes but commits not pushed before harness heartbeat reset.

## Active PRs

| # | Student | Slug | Status | Note |
|---|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP (stale) | no commits since assign; Huber heads-up posted |
| #3235 | askeladd | `local-re-feature` | WIP (stale) | sendback posted; no rerun pushed |
| #3238 | fern | `dual-branch-heads` | WIP — rebased on Huber | branch healthy (70cf8a6), awaiting fresh training run |
| #3239 | frieren | `fourier-pos-enc` | WIP (stale) | no commits since assign; Huber heads-up posted |
| #3240 | nezuko | `hflip-augment` | WIP (stale) | no commits since assign; Huber heads-up posted |
| #3241 | tanjiro | `ema-weights` | WIP (stale) | confirmed rebase direction, pod restarted; needs redo |
| #3393 | thorfinn | `surf-p-channel-weight` | WIP — sent back | extra=4 neutral; trying extra=2 next |
| TBD | edward | `cosine-schedule-match` | **NEW — assigning** | T_max=20 to match realistic epoch horizon |

## Human research direction
None received yet.

## Current research themes

**Budget efficiency** (edward new):
- Cosine schedule match: T_max=20 instead of 50, so LR fully anneals within the ~19-epoch budget
- (BF16 merged — baseline now includes it)

**Loss formulation** (thorfinn #3393, alphonse #3177):
- per-channel surface pressure weighting (surf_p_weight_extra=2, calibrated from extra=4 result)
- per-sample-scale-norm + Huber: balance Re-regime gradient magnitudes

**Architecture** (fern #3238, frieren #3239):
- Dual surface/volume heads (re-running with Huber+BF16 after rebase)
- Fourier positional encoding (multi-scale spatial features)

**Features** (askeladd #3235):
- Local-Re feature + Huber (needs Huber rebase)

**Augmentation / Optimization** (nezuko #3240, tanjiro #3241):
- z-reflection symmetry
- EMA weight averaging (rebased onto Huber)

## Potential next directions (round 3, after current PRs land)
1. **Larger model under BF16**: n_hidden=192 or n_layers=6 (63 GB VRAM free — lots of room)
2. **Larger batch (batch_size=8)**: with BF16, activation memory halved — batch=8 fits comfortably (~55 GB estimated vs 96 GB available)
3. **Cosine schedule match (edward)**: T_max=20 — the most immediate leverage after BF16
4. **Huber delta sweep**: δ ∈ {0.5, 2.0} to test sensitivity around δ=1.0
5. **Per-channel pressure-only auxiliary loss**: extra loss term on dim 2 (p) only (related to thorfinn's work)
6. **Warmup-cosine schedule**: 3-epoch warmup → cosine to zero by epoch 20
7. **Mesh-aware sampler**: weight training samples by inverse squared mesh size
8. **Per-domain stats**: separate (y_mean, y_std) for raceCar/cruise/single domains

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. `data/scoring.py::accumulate_batch` correctly masks these but does `err = abs(pred - y)` *before* applying the per-sample mask, and `inf - finite = inf`, `inf × 0 = NaN`. The accumulator becomes NaN globally.

Affects: All `test_avg/mae_surf_p` numbers on this branch are NaN.
Fix requires modifying `data/scoring.py` (marked read-only). Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits in the paper.
