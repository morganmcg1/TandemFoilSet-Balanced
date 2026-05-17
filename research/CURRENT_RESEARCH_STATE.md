# SENPAI Research State

- **Last updated:** 2026-05-17 ~04:00 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 04:00 UTC)

## Current best baseline — Lookahead+β2=0.95 (PR #4249 nezuko, merged ~02:35 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **52.9444** | PR #4249 nezuko Lookahead k=5 α=0.5 + β2=0.95 slice=8 `5qg8ex1g` |
| `test_3split/mae_surf_p` | **52.7523** | PR #4249 nezuko |

Per-split val (PR #4249, run `5qg8ex1g`):

| Split | mae_surf_p | Δ vs prior Lookahead-only baseline (54.299) |
|---|---|---|
| val_single_in_dist | 63.8415 | −0.15% |
| val_geom_camber_rc | **64.6348** | **−5.99% ← best ever, dominant residual** |
| val_geom_camber_cruise | 32.6315 | +2.12% |
| val_re_rand | 50.6698 | −3.58% |
| **val_avg** | **52.9444** | **−2.49%** |

Per-split test (PR #4249):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 56.4277 |
| test_geom_camber_rc | 58.5654 |
| test_geom_camber_cruise | NaN (fleet-wide bug) |
| test_re_rand | 43.2638 |
| **test_3split** | **52.7523** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --adamw_beta2 0.95 \
  --agent <student>
```

## Stack progression

| Merge | val | test | Δ val |
|---|---|---|---|
| PR #4062 fern (slice=8) | 56.895 | 55.982 | — |
| PR #4067 alphonse (β2=0.95, slice=16) | 56.426 | 55.339 | −0.83% |
| PR #4142 nezuko (Lookahead k=5 α=0.5, slice=8) | 54.299 | 52.879 | −3.77% |
| **PR #4249 nezuko (Lookahead + β2=0.95, slice=8)** | **52.944** | **52.752** | **−2.49%** |

Total improvement since raw seed: **−62.7%** on val.

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | On baseline | Status |
|----|---------|-----------|-------------|--------|
| **#4307** | **nezuko** | Lookahead α-bracket sweep (α=0.3, α=0.7) | Lookahead+β2 baseline | WIP — characterizes Lookahead α; complement to alphonse's k-bracket |
| **#4309** | **askeladd** | n_head=4 architecture sweep | Lookahead+β2 baseline | WIP — first architecture test on new optimizer stack |
| **#4347** | **fern** | Camber-bridging mixup within racecar_tandem (Beta(0.4,0.4)) | Lookahead+β2 baseline | NEW — feature-space interpolation bridges discrete-camber gap |
| **#4313** | **frieren** | n_hidden=192 model capacity (+50%) | Lookahead+β2 baseline | WIP — tests if model is capacity-limited post-optimizer |
| **#4266** | **alphonse** | Lookahead k-bracket sweep (k=3 vs k=10) | Lookahead+β2 baseline | WIP — characterizing optimal k (default k=5 never swept) |
| **#4151** | **thorfinn** | LLRD=0.95 + Lookahead (gentler decay retest) | Lookahead-only baseline | WIP — LLRD=0.85 mixed; 0.95 targets test regression recovery; actively training 85GB |
| **#4251** | **edward** | Lookahead + lr=1e-3 | Lookahead-only baseline | WIP — actively training 89GB |
| **#4334** | **tanjiro** | LR linear warmup (360 steps) on Lookahead+β2=0.95 | Lookahead+β2 baseline | WIP — schedule-shape axis; first warmup test on new stack |

**Key note**: #4151, #4251 were started against the Lookahead-only baseline (54.30). They need to beat the NEW Lookahead+β2 baseline (52.94) to merge.

## Recent closures

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✓ **#4249** | **nezuko** | **Lookahead + β2=0.95 compound** | **52.944** | ✓ **MERGED — NEW BASELINE** |
| ✗ #4311 | fern | Camber-stratified sampler 3× | 52.982 | ✗ Closed — val_camber_rc improved (−3.38%) but test_camber_rc REGRESSED (+5.55%); discrete-distribution frequency reweighting can't bridge gaps |
| ✗ #4284 | tanjiro | weight_decay sweep (5e-4, 1e-3) | 53.931 / 54.413 | ✗ Closed — both arms +1.87%/+2.78% WORSE vs new baseline on val AND test; Lookahead×wd substitutive |
| ✗ #4283 | askeladd | Lookahead+slice=16+β2=0.95 triple | 55.006 | ✗ Closed — β2=0.95 anti-compounds with slice=16 on camber_rc |
| ✗ #4267 | fern | AoA rotation aug ±5° | 56.069 | ✗ Closed — cruise +17.77%; AoA aug axis fully closed |
| ✗ #4204 | frieren | Per-sample peak-\|p\| reweight α=1.0 | 64.396 | ✗ Closed — catastrophic; backbone starvation |

## Key mechanism insights

- **Lookahead × β2=0.95 are orthogonal**: trajectory-averaging (Lookahead) and step-size adaptation (β2) operate at different abstraction levels → additive compounding
- **slice=16 × β2=0.95 are NOT orthogonal under Lookahead**: the triple compound failed; β2=0.95 anti-compounds with slice=16 on camber_rc specifically
- **val_geom_camber_rc=64.63** is now the floor (best ever). Still the dominant residual — 14% above single_in_dist.
- **Lookahead split signature preserved**: Lookahead still helps cruise/re_rand (variance reduction) but camber_rc is now attacking via β2 axis

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP, n_head=2, vel-asinh=0.5
- Huber δ=0.5 (tighter quadratic transition)
- slice_num=8 (Lookahead+β2 stack; slice=16 only partially beneficial and not under Lookahead)
- **AdamW β2=0.95 (PR #4067 original, now confirmed under Lookahead in PR #4249)**
- **Lookahead k=5 α=0.5 (PR #4142, #4249): slow-weight averaging — biggest single-axis win**

## What does NOT work

- β1 axis: β1=0.85 or β1=0.95 (default 0.9 is local optimum)
- EMA=0.995 (substitutes with Lookahead, hurts camber_rc)
- AGC λ=0.01 (clips load-bearing gradient signal)
- grad_clip=1.0 (global scalar clip wrong instrument)
- bs=8 (step starvation), DropPath, dropout, SWA
- All loss-shape axes: Huber δ, asym Huber, log-cosh, Welsch
- Per-sample peak-|p| reweighting (backbone starvation)
- Per-channel surface loss reweighting (shared-latent damage)
- AoA rotation augmentation ±5°/±15° (cruise regression)
- Lookahead+slice=16+β2=0.95 triple (anti-compounding on camber_rc)
- weight_decay > 1e-4 under Lookahead (substitutive with slow-weight regularization; wd=1e-4 is optimal)
- Camber-stratified frequency oversampling 3× (val_camber_rc improves but test_camber_rc regresses; discrete distribution can't bridge held-out gap)

## Strategic outlook

**Target**: val < 52.0, test < 51.0. Current: 52.94 / 52.75. Need −1.8% val, −3.3% test.

Priority order for currently running experiments:
1. **#4347 fern camber-bridging mixup**: feature-space interpolation directly synthesizes held-out M=6,7,8 region; targets dominant residual
2. **#4313 frieren n_hidden=192**: model capacity test; optimizer may have eliminated optimization bottleneck
3. **#4309 askeladd n_head=4**: architecture axis; richer attention subspaces for camber geometry
4. **#4151 thorfinn LLRD=0.95+Lookahead**: gentler LLRD retest; could compound with new baseline
5. **#4307 nezuko α-bracket**: characterizes Lookahead α on new baseline
6. **#4266 alphonse k-bracket**: characterizes Lookahead k
7. **#4334 tanjiro LR warmup (360 steps)**: schedule-shape axis; β2=0.95 early-step instability fix
8. **#4251 edward lr=1e-3**: bolder bet; actively training

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-18 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 04:00 UTC
- **scripts/test_eval_only.py**: in repo — recovers test metrics from saved EMA checkpoints at batch_size=1 when OOM during test eval
