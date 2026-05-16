# SENPAI Research State

- **Last updated**: 2026-05-16 ~10:45 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline

- `val_avg/mae_surf_p` = **86.77** (PR #3753, alphonse, `dsdf-clip`)
- Stack: BF16 + Huber δ=1.0 + cosine T_max=20 + global ±3σ clip

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 102.00 |
| val_geom_camber_rc | 93.75 |
| val_geom_camber_cruise | 69.15 |
| val_re_rand | 82.17 |
| **val_avg** | **86.77** |

## High-value in-flight result

**PR #3759 askeladd (per-point-τ)**: val_avg=85.479 on OLD 87.62 baseline (−2.45%). Branch rebased onto current 86.77 baseline (commit 9530cc9) at ~10:35 UTC; student needs to RE-RUN training and post new SENPAI-RESULT. If the mechanism stacks, expected val_avg ≈ 84.5-85.5.

## Active PRs (round 7)

| # | Student | Slug | Status | Hypothesis |
|---|---|---|---|---|
| **#3759** | **askeladd** | **`per-point-temp`** | **Rebased; awaiting new training result — HIGH PRIORITY** | Per-point adaptive slice temperature |
| **#3846** | **edward** | **`stream-fn`** | **WIP — round 6** | Stream function ψ aux head with autograd L_div |
| **#3844** | **fern** | **`surf-attn`** | **WIP — round 6** | Surface-only all-to-all cross-attention |
| **#3778** | **tanjiro** | **`rmsnorm`** | **WIP — round 6** (GPU 100% as of 10:33) | RMSNorm replacement for LayerNorm |
| **#3892** | **alphonse** | **`mlp-head`** | **WIP — NEW round 7** | 2-layer MLP output head (Linear→GELU→Linear); +33K params |
| **#3893** | **nezuko** | **`ema-eval`** | **WIP — NEW round 7** | EMA model weights (decay=0.999) for evaluation; Kaggle classic |
| **#3894** | **frieren** | **`lr-warmup`** | **WIP — NEW round 7** | 5-epoch linear warmup + cosine T_max=15 |
| **#3895** | **thorfinn** | **`geglu-ffn`** | **WIP — NEW round 7** | GeGLU FFN inside TransolverBlock; +25% FFN params |

## Just closed (round 6 part-2)

| # | Student | Slug | Outcome |
|---|---|---|---|
| #3848 | frieren | dual-scale | Closed — compute budget failure at 1.09M params (101.44, +14.67) |
| #3847 | thorfinn | re-consistency | Closed — loss contribution negligible + 2× forward halves epochs (128.08, +41.31) |
| #3818 | alphonse | surgical-clip | Closed — both arms underperform (88.13-89.95); clipping family exhausted |
| #3780 | nezuko | focal-loss | Closed — EMA difficulty correlates with target magnitude (98.95, +12.18) |

## Refuted approaches (do NOT re-assign)

- **Per-channel output weighting**: fails to stack with BF16 (3.36 worse)
- **Per-domain output normalization**: backfires on low-sample splits
- **SWA**: incompatible with heterogeneous-mesh training
- **Re-stratified sample weighting**: Re range too narrow
- **Gradient accumulation N=2**: stale_wip pattern
- **Cosine T_max=25**: stale_wip
- **Pre-LN swap**: already in baseline
- **n_head=8 with dim_head=16**: too thin
- **mlp_ratio=4**: FFN not bottlenecked
- **slice_num=32**: G=64 binding
- **DualScalePhysicsAttention** (G_fine=64 + G_coarse=16): compute budget failure
- **Re-consistency loss**: loss contribution negligible + 2× forward fatal
- **Surgical/tanh clip**: clipping family exhausted at global ±3σ
- **Focal surface loss (EMA Huber)**: difficulty correlates with target magnitude
- **AdamW β2=0.99**: parity on val (test slightly better; not actionable)
- **Temperature annealing τ 1.0→0.1**: parity

## Round 7 research themes

**Output capacity (NEW lever):**
- 2-layer MLP head (alphonse #3892) — head expressivity for sharp surface features

**Weight-space averaging (NEW family — different from SWA):**
- EMA model weights for eval (nezuko #3893) — continuous trajectory averaging vs SWA's discrete snapshot averaging

**Schedule shape (NEW within schedule family):**
- LR warmup + cosine (frieren #3894) — stabilize early training for outlier-magnitude samples

**FFN structure (NEW within FFN family):**
- GeGLU FFN (thorfinn #3895) — gated information routing for heterogeneous mesh content

**Mid-flight architecture (round 6):**
- Stream function ψ aux head (edward #3846)
- Surface-only cross-attention (fern #3844)
- RMSNorm (tanjiro #3778)

## Pending high-impact verification

**PR #3759 askeladd**: rebased, needs RE-RUN to confirm per-point τ stacks with DSDF clip. Expected val_avg < 86.0 if mechanism is fully orthogonal.

## Round-7+ candidates (for next round if needed)

- **Saf-channel normalization** (askeladd's old result, -9.7% on OLD 117.66 baseline; never re-validated on current)
- **Channel-wise pressure-only Huber delta** (separate δ_p, δ_u from current δ=1.0 global)
- **Per-domain conditional norm γ** (learnable per-split scale at output)
- **Test-Time Augmentation** (predict on rotated/flipped meshes at val time, average)
- **AdamW β2=0.97 + LR=4e-4** (β2=0.99 frieren #3707 showed test asymmetry; revisit at different LR)

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits.
