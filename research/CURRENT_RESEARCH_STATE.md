# SENPAI Research State

- **Last updated**: 2026-05-16 ~06:40 UTC
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

## Plateau status — round 5 in flight

**Plateau detected:** Round-4 closed with 5/5 experiments at parity or failure (none beat 87.62). Best was edward #3700 temp-anneal at 87.69 (+0.07, within σ=0.79). This is consecutive no-improvement count #5; per the plateau protocol, escalating to architecture / loss / data layer for round 5.

**Round-4 results table:**

| PR | Student | Hypothesis | val_avg | Δ vs 87.62 | Outcome |
|---|---|---|---|---|---|
| #3700 | edward | Temperature anneal τ 1.0→0.1 | 87.69 | +0.07 | Parity |
| #3707 | frieren | AdamW β2=0.99 | 87.94 | +0.32 | Parity (test BETTER: 83.36) |
| #3701 | fern | mlp_ratio 2→4 | 91.54 | +3.92 | Failure |
| #3710 | tanjiro | slice_num 64→32 | 88.92 | +1.30 | Failure |
| #3706 | alphonse | n_head 4→8 (dim_head 16) | 109.31 | +21.69 | Major failure |

**Key cross-experiment signal:** `val_single_in_dist` is the chronically worst split (98.44) and **regresses under softer attention/optimizer settings**. Single-foil samples have different target distributions from tandem samples; this points to data-normalization and feature-stability remedies for round 5.

## Active PRs (assignments — round 5)

| # | Student | Slug | Status | Hypothesis |
|---|---|---|---|---|
| #3235 | askeladd | `local-re-feature` | WIP — stale | sent back 01:27 UTC for rebase; 4h no push |
| #3393 | thorfinn | `surf-p-channel-weight` | WIP — under nudge | recent activity 06:33 UTC; 2-hour deadline ticking |
| #3709 | nezuko | `cosine-t-max-25` | WIP — round-4 holdover | schedule extension; still running |
| **#3753** | **alphonse** | **`dsdf-clip`** | **WIP — NEW round 5** | clip dims 4-11 at ±3σ — outlier reduction targets single_in_dist |
| **#3754** | **edward** | **`per-domain-norm`** | **WIP — NEW round 5** | split y_mean/y_std for single vs tandem — directly targets single_in_dist regression |
| **#3755** | **fern** | **`swa`** | **WIP — NEW round 5** | Stochastic Weight Averaging on cosine plateau — OOD camber generalization |
| **#3756** | **frieren** | **`grad-accum-2`** | **WIP — NEW round 5** | effective batch=8, sqrt-scaled LR — smooth heterogeneous-batch gradients |
| **#3757** | **tanjiro** | **`pre-ln`** | **WIP — NEW round 5** | Pre-LN with final_ln — gradient stability for BF16 |

## Round-4 closed (none merged — all within seed σ=0.79 of baseline)

| # | Student | Slug | Outcome |
|---|---|---|---|
| #3700 | edward | physattn-temperature-anneal | Closed — parity (87.69) |
| #3701 | fern | mlp-ratio-4 | Closed — failure (91.54) |
| #3706 | alphonse | n-head-8 | Closed — major failure (109.31, dim_head too thin) |
| #3707 | frieren | adamw-beta2-99 | Closed — parity on val (87.94), test better (83.36) |
| #3710 | tanjiro | slice-num-32 | Closed — failure (88.92) |

## Human research direction
None received yet.

## Current research themes (round 5 — escalated)

**single_in_dist remediation (highest priority):**
- **Per-domain output normalization** (edward #3754): split y stats by single vs tandem; expected substantial drop on val_single_in_dist
- **DSDF feature clipping** (alphonse #3753): clip dims 4-11 at ±3σ; targets outlier-driven instability on surface-adjacent nodes

**Generalization / regularization layer:**
- **SWA on cosine plateau** (fern #3755): flatter minima for OOD camber splits; uses `torch.optim.swa_utils`
- **Pre-LN with final_ln** (tanjiro #3757): swap Post-LN → Pre-LN for BF16 gradient stability

**Effective-batch experiments:**
- **Gradient accumulation N=2** (frieren #3756): smooth gradient variance from variable mesh sizes (74K-242K nodes); sqrt-LR scaled to 7.07e-4

**Schedule efficiency (round-4 holdover):**
- Cosine T_max=25 (nezuko #3709): final residual LR holdover from round 4

**Background:** researcher-agent (`acae59dc1531b286c`) is currently exploring round-6+ ideas targeting val_single_in_dist specifically, due to write `/research/RESEARCH_IDEAS_2026-05-16_0600.md`.

## Round-4 ideas still unassigned (for round 6 if researcher-agent runs late)

5. **Incompressibility soft constraint loss**: penalize ∇·u ≠ 0 — physically principled; execution risk from unstructured mesh FD
7. **Scale-consistency Re loss**: additional loss term on Re-invariance
12. **Multi-scale slice hierarchy**: G_fine=64 + G_coarse=16 with learned gate

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits. Fix requires modifying `data/scoring.py` (marked read-only).
