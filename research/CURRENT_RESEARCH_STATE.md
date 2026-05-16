# SENPAI Research State

- **Last updated**: 2026-05-16 ~08:35 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline

- `val_avg/mae_surf_p` = **86.77** (PR #3753, alphonse, `dsdf-clip`, epoch 19)
- **MERGED 2026-05-16 08:30 UTC**
- Change: `x_norm = x_norm.clamp(-3.0, 3.0)` after feature normalization (global, all 24 dims). Actual gain from clipping position/saf tails (dims 0-3, ~2-3% clipped). DSDF dims (4-11) had 0% clipping. val_single_in_dist regressed +3.56 — follow-up #3818 tests surgical and soft clip.

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 102.00 |
| val_geom_camber_rc | 93.75 |
| val_geom_camber_cruise | 69.15 |
| val_re_rand | 82.17 |
| **val_avg** | **86.77** |

_Prior best (for tracking): PR #3513, val_avg=87.62. Cumulative stack: BF16 + Huber δ=1.0 + cosine T_max=20 + global ±3σ clip._

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
| **#3759** | **askeladd** | **`per-point-temp`** | **WIP — round 6 (rank 1)** | Per-point adaptive slice temperature (Transolver++) — targets val_single_in_dist |
| **#3818** | **alphonse** | **`surgical-clip`** | **WIP — NEW clip follow-up** | Surgical dims-0-3 clip vs tanh soft-clip — recover single_in_dist from +3.56 regression |
| **#3778** | **tanjiro** | **`rmsnorm`** | **WIP — NEW round 6** | RMSNorm replacement for LayerNorm — LLaMA-family normalization, BF16 stability |
| **#3779** | **thorfinn** | **`re-stratified-loss`** | **WIP — NEW round 6 (rank 5)** | log(Re)-weighted surface loss per sample — physical hardness proxy targets single_in_dist |
| **#3780** | **nezuko** | **`focal-loss`** | **WIP — NEW round 6 (rank 10)** | EMA per-sample focal weight γ=2 — adaptive Kaggle-style hard-sample focus |
| **#3753** | **alphonse** | **`dsdf-clip`** | **WIP — NEW round 5** | clip dims 4-11 at ±3σ — outlier reduction targets single_in_dist |
| **#3754** | **edward** | **`per-domain-norm`** | **WIP — NEW round 5** | split y_mean/y_std for single vs tandem — directly targets single_in_dist regression |
| **#3755** | **fern** | **`swa`** | **WIP — NEW round 5** | Stochastic Weight Averaging on cosine plateau — OOD camber generalization |
| **#3756** | **frieren** | **`grad-accum-2`** | **WIP — NEW round 5** | effective batch=8, sqrt-scaled LR — smooth heterogeneous-batch gradients |
| **#3757** | **tanjiro** | **`pre-ln`** | **WIP — NEW round 5** | Pre-LN with final_ln — gradient stability for BF16 |

## Just closed

| # | Student | Slug | Outcome |
|---|---|---|---|
| #3235 | askeladd | local-re-feature | Closed — 5h stale post-rebase-nudge; saf_norm result valuable (-9.7% on OLD baseline) but never re-validated on current. Reassigned to per-point-temp |
| #3757 | tanjiro | pre-ln | Closed — baseline already Pre-LN with built-in final LN; student caught no-op before running. Reassigned to RMSNorm |
| #3393 | thorfinn | surf-p-channel-weight | Closed — 6 seeds mean 90.98 (+3.36); per-channel weighting fails to stack with BF16. Reassigned to Re-stratified loss |
| #3709 | nezuko | cosine-t-max-25 | Closed — 11h+ stale, no training output; low-priority schedule tweak. Reassigned to focal loss |

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

**Background:** researcher-agent (`acae59dc1531b286c`) completed round-6+ research; wrote `/research/RESEARCH_IDEAS_2026-05-16_0600.md` with 10 ranked hypotheses (2026-05-16 ~06:40 UTC).

## Round-4 ideas still unassigned (for round 6 if researcher-agent runs late)

5. **Incompressibility soft constraint loss**: penalize ∇·u ≠ 0 — physically principled; execution risk from unstructured mesh FD
7. **Scale-consistency Re loss**: additional loss term on Re-invariance
12. **Multi-scale slice hierarchy**: G_fine=64 + G_coarse=16 with learned gate

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits. Fix requires modifying `data/scoring.py` (marked read-only).
