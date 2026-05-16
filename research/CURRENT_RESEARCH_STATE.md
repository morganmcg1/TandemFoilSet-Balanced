# SENPAI Research State

- **Last updated**: 2026-05-16 ~00:45 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline

- `val_avg/mae_surf_p` = **87.62** (PR #3513, edward, `cosine-schedule-match`, epoch 19)
- **MERGED 2026-05-16 00:40 UTC**
- Change from BF16 baseline (97.55): set `cosine_t_max=20` so LR fully anneals within 19-epoch budget. Zero overhead — same epochs, same VRAM, same throughput. Pure scheduling gain.

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 98.44 | 1.174 | 0.616 |
| val_geom_camber_rc | 96.95 | 1.923 | 0.837 |
| val_geom_camber_cruise | 71.27 | 0.782 | 0.493 |
| val_re_rand | 83.83 | 1.356 | 0.647 |
| **val_avg** | **87.62** | 1.230 | 0.608 |

## Key observations
1. **Cosine schedule mismatch was a major leak**: T_max=50 with only 19 epochs means the LR never anneals properly. T_max=20 → LR at 0.62% of initial by epoch 19. Free 10% gain.
2. **BF16 + Huber + cosine T_max=20 is now the baseline stack** — all three improvements merged.
3. **VRAM headroom**: 32.94 GB used out of 96 GB = 63 GB free. Enables n_hidden=192/256 or batch_size=8.
4. **Budget still binding**: best epoch is 19/19 (hit the cap). More epochs → more improvement. Model not saturated.
5. **NaN bug persists** in `test_geom_camber_cruise`: workaround rank on val_avg; report test_avg as clean-3 mean.

## Active PRs

| # | Student | Slug | Status | Note |
|---|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP (stale) | no commits since assign |
| #3235 | askeladd | `local-re-feature` | WIP (stale) | sendback posted; no rerun |
| #3238 | fern | `dual-branch-heads` | WIP — sent back | sent back for rebase onto new baseline (87.62); had been 95.56 on 97.55 baseline |
| #3239 | frieren | `fourier-pos-enc` | WIP (stale) | no commits since assign |
| #3240 | nezuko | `hflip-augment` | WIP (stale) | no commits since assign |
| #3241 | tanjiro | `ema-weights` | WIP (stale) | needs redo after pod restart |
| #3393 | thorfinn | `surf-p-channel-weight` | WIP — sent back | needs rebase onto BF16+cosine then rerun extra=1.0 |
| TBD | edward | NEW | **ASSIGNING** | idle after cosine merge |

## Human research direction
None received yet.

## Current research themes

**Schedule / budget efficiency** (all experiments now stacked):
- BF16 mixed-precision (PR #3300, merged): +5 epochs, 1.3x throughput, −22% VRAM
- Cosine T_max=20 (PR #3513, merged): −10.18% on val_avg; LR fully anneals within budget
- **Next**: Warmup-cosine (3-epoch warmup → cosine to zero by epoch 20), or restart schedules

**Architecture** (fern #3238):
- Dual surface/volume heads — sent back to rebase onto new 87.62 baseline; mechanism orthogonal, expected stacked val_avg ~85-86

**Loss formulation** (thorfinn #3393, alphonse #3177):
- Per-channel surface pressure weighting (extra=1.0 most promising, needs rebase onto full stack)
- Per-sample-scale-norm + Huber (stale)

**Features** (askeladd #3235):
- Local-Re feature + Huber (stale, needs rebase)

**Augmentation / Optimization** (nezuko #3240, tanjiro #3241, frieren #3239):
- Fourier positional encoding (stale)
- z-reflection augmentation (stale)
- EMA weight averaging (stale)

## Potential next directions (round 4)
1. **Larger model**: n_hidden=192 or n_hidden=256 (63 GB VRAM free — lots of room). With cosine T_max=20 the model may benefit more from capacity.
2. **Batch_size=8**: BF16 halved activation memory, batch=8 likely fits (~50-55 GB estimated). Larger batch can stabilize training and allow slightly higher LR.
3. **Warmup-cosine**: 2-3 epoch linear warmup → cosine to near-zero by epoch 20. Prevents early noisy gradients from locking in bad solutions.
4. **Cosine warm restarts (SGDR)**: T_0=10, T_mult=1, T_max=20 — two full cosine cycles in the budget. May find a sharper minimum.
5. **More aggressive Huber delta sweep**: δ ∈ {0.3, 0.5, 2.0} now that we have a cleaner baseline.
6. **Per-domain normalization**: separate (y_mean, y_std) for raceCar/cruise/single domains — the single_in_dist split (98.44) is hardest, likely has different pressure statistics than camber splits.
7. **Deeper network**: n_layers=6 or n_layers=7 — same 63 GB headroom, adds depth without width cost.
8. **Attention head expansion**: n_head=8 (from 4) to improve slice-token mixing at same n_hidden=128.

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits. Fix requires modifying `data/scoring.py` (marked read-only).
