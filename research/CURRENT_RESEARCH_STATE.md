# SENPAI Research State

- **Last updated**: 2026-05-16 ~04:50 UTC
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
6. **Seed variance at current performance level**: fern's 3-replicate study shows σ=0.79 on val_avg/mae_surf_p. A single-seed Δ < ~1.5 should not be treated as a reliable improvement — we need >2σ gains to be confident.
7. **n_hidden=192 fails in fixed budget**: wider model uses 15 epochs (vs 19) and converges slower. Raw scale doesn't help here. Focus on same-width quality improvements.

## Active PRs (assignments)

| # | Student | Slug | Status | Note |
|---|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP (stale) | no commits since assign |
| #3235 | askeladd | `local-re-feature` | WIP — awaiting rebase | sent back 01:27 UTC for rebase onto 87.62 baseline; no new push yet |
| #3239 | frieren | `fourier-pos-enc` | WIP (stale) | no commits since assign |
| #3240 | nezuko | `hflip-augment` | WIP (stale) | no commits since assign |
| #3241 | tanjiro | `ema-weights` | WIP (stale) | needs redo after pod restart |
| #3393 | thorfinn | `surf-p-channel-weight` | WIP — sent back | needs rebase onto BF16+cosine then rerun extra=1.0 |
| TBD | edward | `physattn-temperature-anneal` | **ASSIGNING** | branches ready; PR pending REST rate limit reset (~05:20 UTC) |
| TBD | fern | `mlp-ratio-4` | **ASSIGNING** | branches ready; PR pending REST rate limit reset (~05:20 UTC) |

## Just closed

| # | Student | Slug | Outcome |
|---|---|---|---|
| #3238 | fern | `dual-branch-heads` | Closed — parity (87.35 best of 3 runs, mean 88.10, σ=0.79; baseline inside variance range) |
| #3567 | edward | `wider-model-hidden192` | Closed — failure (93.25, +6.4% vs baseline; underconverges in fixed budget) |

## Human research direction
None received yet.

## Current research themes

**Mechanism / architecture quality** (new — high priority):
- **PhysicsAttention temperature annealing** (edward, next PR): anneal softmax τ from 1.0→0.1 over cosine schedule. Directly targets slice-token quality — soft early for gradient flow, crisp late for physics partitioning. Zero compute overhead.
- **mlp_ratio 2→4** (fern, next PR): double FFN width per block from 256→512. Standard transformer practice. Isolated change at current n_hidden=128; expected ~2-5% gain.

**Schedule / budget efficiency** (all experiments now stacked):
- BF16 mixed-precision (PR #3300, merged): +5 epochs, 1.3x throughput, −22% VRAM
- Cosine T_max=20 (PR #3513, merged): −10.18% on val_avg; LR fully anneals within budget
- **Next**: n_head 4→8 (idea #3 from round-4), warmup-cosine, or cosine warm restarts

**Loss formulation** (thorfinn #3393, alphonse #3177):
- Per-channel surface pressure weighting (extra=1.0 most promising, needs rebase onto 87.62 baseline)
- Per-sample-scale-norm + Huber (stale)

**Features** (askeladd #3235):
- Local-Re via saf_norm: 106.28 on old 117.66 baseline (pre-BF16/cosine). Sent back for rebase onto 87.62 — needs one more run. High potential if BF16+cosine stacks with the feature.

**Augmentation / Optimization** (frieren #3239, nezuko #3240, tanjiro #3241):
- Fourier positional encoding (stale — 4+ hours no commits)
- z-reflection augmentation (stale)
- EMA weight averaging (stale)

## Round-4 ideas (ranked, from RESEARCH_IDEAS_2026-05-16_0130.md)

1. **PhysicsAttention temperature annealing** — ASSIGNED to edward ✓
2. **mlp_ratio 2→4** — ASSIGNED to fern ✓
3. **n_head 4→8**: same param count as baseline (dim_head 32→16); head diversity
4. **Stochastic Weight Averaging (SWA)**: after cosine annealing, SWA averages checkpoints at high LR plateau — flatter minima, better OOD generalization
5. **Incompressibility soft constraint loss**: penalize ∇·u ≠ 0 — physically principled; execution risk from unstructured mesh FD
6. **Cosine T_max 25/30**: extend annealing horizon if training reaches more epochs than expected
7. **Scale-consistency Re loss**: additional loss term on Re-invariance
8. **Gradient accumulation**: simulate larger batch
9. **Pre-LN**: move LayerNorm before attention/MLP (Pre-LN vs Post-LN)
10. **AdamW β2=0.99**: slower second-moment decay for heavy-tail gradient noise

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits. Fix requires modifying `data/scoring.py` (marked read-only).
