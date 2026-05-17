# SENPAI Research State

- **Last updated:** 2026-05-17 00:40 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 00:40 UTC)

## Current best baseline — NEW (Lookahead merge at 00:35 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **54.2986** | PR #4142 nezuko Lookahead k=5 α=0.5 slice=8 `qhphlg41` |
| `test_3split/mae_surf_p` | **52.8790** | PR #4142 nezuko |

Per-split val (PR #4142, best seed `qhphlg41`):

| Split | mae_surf_p | Δ vs prior alphonse baseline (56.426) |
|---|---|---|
| val_single_in_dist | 63.937 | −1.9% |
| val_geom_camber_rc | **68.753** | **+2.4% ← REGRESSED** |
| val_geom_camber_cruise | 31.954 | **−15.7% ← best improvement** |
| val_re_rand | 52.552 | −5.2% |
| **val_avg** | **54.299** | **−3.8%** |

Per-split test (PR #4142):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 54.230 |
| test_geom_camber_rc | 60.693 |
| test_geom_camber_cruise | NaN (fleet-wide bug) |
| test_re_rand | 43.715 |
| **test_3split** | **52.879** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --agent <student>
```

**NOTE**: β2=0.999 (default). β2=0.95+Lookahead compound is the highest-priority open experiment (#4249 nezuko).

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
| **#4249** | **nezuko** | Lookahead + β2=0.95 compound | NEW Lookahead baseline | WIP — zero code change, HIGHEST PRIORITY |
| **#4250** | **askeladd** | Lookahead + slice=16 compound | NEW Lookahead baseline | WIP — does slice=16's camber_rc benefit persist under Lookahead? |
| **#4251** | **edward** | Lookahead + lr=1e-3 (Lookahead enables higher LR) | NEW Lookahead baseline | WIP — tests Lookahead paper's "larger inner LR" claim |
| **#4204** | **frieren** | Per-sample surf-loss reweighting by peak |p| (α=1.0) | Old alphonse baseline | WIP — if positive vs old baseline, retest on Lookahead stack |
| **#4226** | **fern** | Per-channel surf weights (p=2.0, Ux=0.5, Uy=0.5) | Old alphonse baseline | WIP — if positive vs old baseline, retest on Lookahead stack |
| **#4219** | **tanjiro** | AdamW β1=0.95 (slower momentum, bracket closure) | Old alphonse baseline | WIP — if positive vs old baseline, retest on Lookahead stack |
| **#4162** | **alphonse** | β2=0.95 + slice=8 compound | Old alphonse baseline | WIP — 2 seeds split; 3rd seed pending |
| **#4151** | **thorfinn** | **LLRD=0.85 + Lookahead compound (REBASE NEEDED)** | NEW Lookahead baseline | **Sent back for 3rd rebase** — must add Lookahead to LLRD stack |

**Key note**: #4204, #4226, #4219 were submitted against OLD alphonse baseline (56.426). Even if they beat the OLD baseline, they need retesting against the NEW Lookahead baseline (54.299) before merging.

## Key mechanism insight — Lookahead split signature

Lookahead improved cruise (−15.7%) and re_rand (−5.2%) but WORSENED camber_rc (+2.4%). The slow-weight averaging reduces optimization variance, which helps splits with high inter-batch variance but doesn't fix structural extrapolation to high-camber geometries.

**Dominant residual: val_geom_camber_rc=68.75** (highest it's been since early rounds). Top priority target.

## Round 17/18 closures and merges

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✓ **#4142** | **nezuko** | **Lookahead k=5 α=0.5 (rebased)** | **54.299** | ✓ **MERGED — NEW BASELINE** |
| ✗ #4218 | askeladd | AGC λ=0.01 | 56.52 | ✗ Closed — neutral val, worse test; per-param clip removes load-bearing gradient signal |
| ✗ #4184 | edward | EMA decay=0.995 | 56.66 | ✗ Closed — slow Polyak hurts camber_rc split |
| ✗ #4194 | askeladd | --grad_clip=1.0 | 56.46 | ✗ Closed — global scalar clip wrong instrument |
| ✗ #4171 | tanjiro | AdamW β1=0.85 | 57.65 | ✗ Closed — faster momentum hurts convergence |
| ✗ #4193 | frieren | Welsch biweight c=1.0 | 60.22 | ✗ Closed — loss-shape axis FULLY closed |
| ✗ #4163 | fern | mesh aug ±15° | 70.38 | ✗ Closed — physics inconsistency + budget-bound |

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP, n_head=2, vel-asinh=0.5
- Huber δ=0.5 (tighter quadratic transition)
- slice_num=8 (Lookahead stack) or slice_num=16 (question open under Lookahead)
- **AdamW β2=0.95 (PR #4067): fast 2nd-moment EMA — paper-quality finding**
- **Lookahead k=5 α=0.5 (PR #4142): slow-weight averaging — biggest single-axis win**

## What does NOT work

- β1=0.85 (faster momentum hurts), β1=0.95 (pending)
- EMA=0.995 (substitutes with Lookahead, hurts camber_rc)
- AGC λ=0.01 (clips load-bearing gradient signal), grad_clip=1.0 (global clip wrong instrument)
- bs=8 (step starvation), DropPath, dropout, SWA
- All loss-shape axes closed (Huber δ, asym Huber, log-cosh, Welsch)
- Mesh aug ±15° (physics inconsistency + budget-bound)

## Strategic outlook

**Target**: val < 52.0, test < 51.0. Current: 54.30 / 52.88. Need −4.3% val more.

Priority order:
1. **#4249 nezuko Lookahead+β2=0.95**: highest compound probability (β2 specifically improved camber_rc)
2. **#4151 thorfinn LLRD+Lookahead**: LLRD targets per-layer LR asymmetry, orthogonal to Lookahead
3. **#4250 askeladd Lookahead+slice=16**: directly tests if slice=16's camber_rc benefit persists
4. **#4251 edward Lookahead+lr=1e-3**: bolder bet; Lookahead stability may unlock higher LR

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test/test_single_in_dist + test/test_geom_camber_rc + test/test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-17 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 00:40 UTC
