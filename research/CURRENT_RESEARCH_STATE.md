# SENPAI Research State

- **Last updated:** 2026-05-17 ~02:00 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 01:10 UTC)

## Current best baseline — Lookahead (PR #4142 nezuko, merged 00:35 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **54.2986** | PR #4142 nezuko Lookahead k=5 α=0.5 slice=8 `qhphlg41` |
| `test_3split/mae_surf_p` | **52.8790** | PR #4142 nezuko |

Per-split val (PR #4142, best seed `qhphlg41`):

| Split | mae_surf_p | Δ vs prior alphonse baseline (56.426) |
|---|---|---|
| val_single_in_dist | 63.937 | −1.9% |
| val_geom_camber_rc | **68.753** | **+2.4% ← REGRESSED — dominant residual** |
| val_geom_camber_cruise | 31.954 | **−15.7% ← best improvement** |
| val_re_rand | 52.552 | −5.2% |
| **val_avg** | **54.299** | **−3.8%** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --agent <student>
```

## Stack progression

| Merge | val | test | Δ val |
|---|---|---|---|
| PR #4062 fern (slice=8) | 56.895 | 55.982 | — |
| PR #4067 alphonse (β2=0.95, slice=16) | 56.426 | 55.339 | −0.83% |
| **PR #4142 nezuko (Lookahead k=5 α=0.5, slice=8)** | **54.299** | **52.879** | **−3.77%** |

Total improvement since raw seed: **−60.3%** on val.

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | On baseline | Status |
|----|---------|-----------|-------------|--------|
| **#4249** | **nezuko** | Lookahead + β2=0.95 compound | NEW Lookahead baseline | WIP — HIGHEST PRIORITY (β2 specifically improved camber_rc) |
| **#4251** | **edward** | Lookahead + lr=1e-3 | NEW Lookahead baseline | WIP — tests if Lookahead stability enables higher LR |
| **#4151** | **thorfinn** | Lookahead + LLRD=0.95 (gentler decay, retest) | NEW Lookahead baseline | WIP — SENT BACK after LLRD=0.85 result mixed (val −0.59%, test +0.88%); testing gentler decay |
| **#4266** | **alphonse** | Lookahead k-bracket sweep (k=3 vs k=10) | NEW Lookahead baseline | WIP — characterizes optimal k; k=5 from paper defaults, not swept |
| **#4267** | **fern** | Physics-consistent AoA rotation aug ±5° | NEW Lookahead baseline | WIP — targets camber_rc; fully coupled rotation (positions+velocities+AoA scalar) |
| **#4283** | **askeladd** | Lookahead + slice=16 + β2=0.95 triple compound | NEW Lookahead baseline | WIP — NEW: motivated by #4250 closure (slice=16 owns camber_rc); unexplored 3-axis cell |
| **#4284** | **tanjiro** | weight_decay sweep (5e-4, 1e-3) | NEW Lookahead baseline | WIP — NEW: post-AdamW EMA closure; fresh axis under Lookahead |
| **#4204** | **frieren** | Per-sample surf-loss reweighting by peak \|p\| (α=1.0) | Old alphonse baseline | WIP — actively training (GPU 97%); if positive, retest on Lookahead |

**Key note**: #4204 was submitted against OLD alphonse baseline (56.426). Even if it beats the OLD baseline, it needs retesting against the NEW Lookahead baseline (54.299) before merging.

## Round 18/19/20 closures

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✓ **#4142** | **nezuko** | **Lookahead k=5 α=0.5 (rebased)** | **54.299** | ✓ **MERGED — NEW BASELINE** |
| ✗ #4250 | askeladd | Lookahead + slice=16 compound | 54.645 | ✗ Closed — val_avg substitutive (+0.65%), BUT val_geom_camber_rc=66.52 BEST EVER; slice=16 owns camber_rc resolution mechanism. Reassigned to triple compound |
| ✗ #4219 | tanjiro | AdamW β1=0.95 (slower momentum) | 60.51 | ✗ Closed — bracket closure; β1 axis fully characterized; default 0.9 optimal; β1 vs β2 asymmetric directions confirmed |
| ↻ #4151 | thorfinn | LLRD=0.85 + Lookahead | 53.98 / test 53.34 | ↻ SENT BACK — val win 0.59% within noise, test regressed 0.88%; retest with LLRD=0.95 (gentler) |
| ✗ #4226 | fern | Per-channel surf weights (p=2.0, Ux=0.5, Uy=0.5) | 57.476 | ✗ Closed — camber_rc REGRESSED +4.76%; shared-latent damage from velocity down-weighting |
| ✗ #4162 | alphonse | β2=0.95 + slice=8 compound | 55.304 (best seed), 57.09 (3-seed mean) | ✗ Closed — 3-seed mean doesn't beat either baseline; nezuko #4249 tests this compound under Lookahead |
| ✗ #4218 | askeladd | AGC λ=0.01 | 56.52 | ✗ Closed — neutral val, worse test; clips load-bearing gradient signal |
| ✗ #4184 | edward | EMA decay=0.995 | 56.66 | ✗ Closed — slow Polyak hurts camber_rc |
| ✗ #4194 | askeladd | --grad_clip=1.0 | 56.46 | ✗ Closed — global scalar clip wrong instrument |
| ✗ #4171 | tanjiro | AdamW β1=0.85 | 57.65 | ✗ Closed — faster momentum hurts convergence |
| ✗ #4193 | frieren | Welsch biweight c=1.0 | 60.22 | ✗ Closed — loss-shape axis FULLY closed |
| ✗ #4163 | fern | mesh aug ±15° | 70.38 | ✗ Closed — physics inconsistency + budget-bound |

## Key mechanism insight — Lookahead split signature

Lookahead improved cruise (−15.7%) and re_rand (−5.2%) but WORSENED camber_rc (+2.4%). The slow-weight averaging reduces optimization variance, which helps splits with high inter-batch variance but doesn't fix structural extrapolation to high-camber geometries.

**Dominant residual: val_geom_camber_rc=68.75** (highest it's been since early rounds). Top priority target.

Two active experiments directly targeting camber_rc:
1. **#4249 nezuko**: Lookahead+β2=0.95 — β2 specifically improved camber_rc on old stack by 4.2%
2. **#4250 askeladd**: Lookahead+slice=16 — slice=16 improved camber_rc on old stack by 4.2%

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP, n_head=2, vel-asinh=0.5
- Huber δ=0.5 (tighter quadratic transition)
- slice_num=8 (Lookahead stack) or slice_num=16 (question open under Lookahead via #4250)
- **AdamW β2=0.95 (PR #4067): fast 2nd-moment EMA — paper-quality finding**
- **Lookahead k=5 α=0.5 (PR #4142): slow-weight averaging — biggest single-axis win**

## What does NOT work

- β1=0.85 (faster momentum hurts), β1=0.95 (pending)
- EMA=0.995 (substitutes with Lookahead, hurts camber_rc)
- AGC λ=0.01 (clips load-bearing gradient signal), grad_clip=1.0 (global clip wrong instrument)
- bs=8 (step starvation), DropPath, dropout, SWA
- All loss-shape axes closed (Huber δ, asym Huber, log-cosh, Welsch)
- Mesh aug ±15° (physics inconsistency + budget-bound)
- Per-channel surface loss reweighting (shared-latent damage from velocity down-weighting)
- β2=0.95 + slice=8 compound without Lookahead (high seed variance, no robust mean improvement)

## Strategic outlook

**Target**: val < 52.0, test < 51.0. Current: 54.30 / 52.88. Need −4.3% val more.

Priority order:
1. **#4249 nezuko Lookahead+β2=0.95**: highest compound probability (β2 specifically improved camber_rc)
2. **#4283 askeladd Lookahead+slice=16+β2=0.95 triple**: motivated by #4250's camber_rc=66.52 best-ever; tests if slice+β2 compound under Lookahead
3. **#4267 fern AoA rotation aug ±5°**: first physics-consistent data aug; targets camber_rc
4. **#4151 thorfinn LLRD=0.95+Lookahead**: gentler decay retest; either merges or closes LLRD axis
5. **#4266 alphonse Lookahead k-bracket**: characterizes Lookahead k for paper; may find k=3 or k=10 better than k=5
6. **#4284 tanjiro weight_decay sweep**: fresh axis after AdamW EMA closure
7. **#4251 edward Lookahead+lr=1e-3**: bolder bet; Lookahead stability may unlock higher LR
8. **#4204 frieren per-sample reweight**: on OLD baseline; needs Lookahead retest if positive

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-17 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 01:10 UTC
- **scripts/test_eval_only.py**: in repo from alphonse #4162 — recovers test metrics from saved EMA checkpoints at batch_size=1 when OOM during test eval
