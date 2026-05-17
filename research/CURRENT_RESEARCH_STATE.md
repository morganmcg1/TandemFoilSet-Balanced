# SENPAI Research State

- **Last updated:** 2026-05-17 ~05:30 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 05:30 UTC)

## Current best baseline — Lookahead k=3 (PR #4266 alphonse, merged ~04:50 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **51.3066** | PR #4266 alphonse Lookahead k=3 α=0.5 β2=0.999 slice=8 `0aj92l9d` |
| `test_3split/mae_surf_p` | **51.8862** | PR #4266 alphonse |

Per-split val (PR #4266, run `0aj92l9d`):

| Split | mae_surf_p | Δ vs prior baseline (52.9444) |
|---|---|---|
| val_single_in_dist | 57.803 | **−9.59%** |
| val_geom_camber_rc | 63.854 | −1.21% |
| val_geom_camber_cruise | 32.409 | −0.67% |
| val_re_rand | 51.159 | +0.97% |
| **val_avg** | **51.3066** | **−3.09%** |

Per-split test (PR #4266):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 52.578 |
| test_geom_camber_rc | 60.074 |
| test_geom_camber_cruise | NaN (fleet-wide bug) |
| test_re_rand | 43.007 |
| **test_3split** | **51.8862** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --agent <student>
```

**Key finding**: k=3+β2=0.999 BEATS k=5+β2=0.95 (previous compound baseline 52.94). k-axis is a stronger lever than β2-axis at the 30-min budget.

## Stack progression

| Merge | val | test | Δ val |
|---|---|---|---|
| PR #4062 fern (slice=8) | 56.895 | 55.982 | — |
| PR #4067 alphonse (β2=0.95, slice=16) | 56.426 | 55.339 | −0.83% |
| PR #4142 nezuko (Lookahead k=5 α=0.5, slice=8) | 54.299 | 52.879 | −3.77% |
| PR #4249 nezuko (Lookahead + β2=0.95, slice=8) | 52.944 | 52.752 | −2.49% |
| **PR #4266 alphonse (Lookahead k=3, β2=0.999, slice=8)** | **51.307** | **51.886** | **−3.09%** |

Total improvement since raw seed: **−64.3%** on val.

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | On baseline | Status |
|----|---------|-----------|-------------|--------|
| **#4369** | **alphonse** | k=3 + β2=0.95 compound | NEW k=3 baseline | WIP — expected val ≈ 50.0 if fully additive; highest expected value in flight |
| **#4370** | **tanjiro** | k=2+k=4 fine bracket | NEW k=3 baseline | WIP — characterizes k-axis limit; does trend continue below k=3? |
| **#4387** | **frieren** | slice_num bracket (4, 12) | NEW k=3 baseline | NEW — geometry-resolution axis untested under k=3 dynamics |
| **#4347** | **fern** | Camber-bridging mixup within racecar_tandem (Beta(0.4,0.4)) | k=5+β2=0.95 baseline | WIP — data-side camber_rc intervention |
| **#4309** | **askeladd** | n_head=4 architecture sweep | k=5+β2=0.95 baseline | WIP — needs to beat NEW baseline (51.31) |
| **#4307** | **nezuko** | Lookahead α-bracket sweep (α=0.3, α=0.7) | k=5+β2=0.95 baseline | WIP — needs to beat NEW baseline (51.31) |
| **#4151** | **thorfinn** | LLRD=0.95 + Lookahead (gentler decay retest) | Lookahead-only baseline | WIP — very unlikely to beat 51.31 |
| **#4251** | **edward** | Lookahead + lr=1e-3 | Lookahead-only baseline | WIP — very unlikely to beat 51.31 |

**Key note**: All in-flight PRs (#4307, #4309, #4347, #4387) must beat the NEW k=3 baseline (val=51.31, test=51.89). PRs #4151 and #4251 were started on OLD Lookahead-only baseline (54.30) — they almost certainly cannot clear the new bar without a rerun on k=3 stack.

## Recent closures

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✗ **#4313** | **frieren** | **n_hidden=192 capacity (+123% params)** | **60.24** | ✗ **Closed — compute-budget regression. epoch time +100%, only 12/22 epochs complete. Capacity axis CLOSED: camber_rc is training-time-limited, not capacity-limited.** |
| ✓ **#4266** | **alphonse** | **Lookahead k=3 (β2=0.999)** | **51.307** | ✓ **MERGED — NEW BASELINE** |
| ✓ **#4249** | **nezuko** | **Lookahead + β2=0.95 compound** | **52.944** | ✓ **MERGED** |
| ✗ #4334 | tanjiro | LR warmup 360 steps | 53.046 | ✗ Closed — val_camber_rc REGRESSED +4.22% (mechanism falsified) |
| ✗ #4311 | fern | Camber-stratified sampler 3× | 52.982 | ✗ Closed — test_camber_rc REGRESSED; discrete-distribution gap can't be bridged by frequency reweighting |
| ✗ #4284 | tanjiro | weight_decay sweep (5e-4, 1e-3) | 53.931 | ✗ Closed — Lookahead×wd substitutive |
| ✗ #4283 | askeladd | Lookahead+slice=16+β2=0.95 triple | 55.006 | ✗ Closed — β2=0.95 anti-compounds with slice=16 |
| ✗ #4267 | fern | AoA rotation aug ±5° | 56.069 | ✗ Closed — cruise +17.77%; AoA aug axis fully closed |
| ✗ #4204 | frieren | Per-sample peak-\|p\| reweight | 64.396 | ✗ Closed — catastrophic; backbone starvation |

## Key mechanism insights

- **k-axis is the dominant Lookahead parameter on this budget**: k=3 (−3.09% val) beats β2=0.95 (−2.49% val) as a single axis; monotone trend k=3 < k=5 << k=10 confirmed
- **Lookahead × β2=0.95 were orthogonal at k=5**: compounding verified in #4249 (−2.49%). Untested at k=3 — alphonse testing now
- **slice=16 × β2=0.95 are NOT orthogonal under Lookahead**: triple compound failed; anti-compounding on camber_rc specifically
- **val_geom_camber_rc=63.85** is new floor (k=3 baseline). Dominant residual — 10% above val_single_in_dist=57.8
- **k-axis mechanism**: 30-min/~6300-step budget heavily favors higher slow-weight update frequency; k=3 → 2100 updates vs k=5 → 1260. Still descending at timeout — k-axis not yet saturated
- **Capacity axis closed upward**: n_hidden=192 (PR #4313) confirms camber_rc is compute-budget-limited, NOT capacity-limited. 2× epoch time means only 12 epochs in 30-min budget vs 22 at baseline. No value in wider models without more wall-clock time
- **Slice axis open on k=3**: slice_num has only been validated at 8 on the new k=3 baseline. The coarse bracket {4, 12} tests geometry-resolution axis; if camber_rc is sensitive to slice granularity, this is a zero-code-change win

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP, n_head=2, vel-asinh=0.5
- Huber δ=0.5 (tighter quadratic transition)
- slice_num=8 (Lookahead+β2 stack; slice=16 only partially beneficial and not under Lookahead)
- **AdamW β2=0.95 (PR #4067 original, now confirmed under Lookahead in PR #4249)**
- **Lookahead k=3 α=0.5 (PR #4266): slow-weight averaging at k=3 — biggest recent win, dominant lever on 30-min budget**

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
- weight_decay > 1e-4 under Lookahead (substitutive with slow-weight regularization)
- Camber-stratified frequency oversampling 3× (val_camber_rc improves but test_camber_rc regresses)
- LR warmup (360 steps): consumes budget, camber_rc regression
- n_hidden=192 (2.23× params → 2× epoch time → only 12/22 epochs; compute-budget regression, NOT representation failure)

## Strategic outlook

**Target**: val < 50.0, test < 50.0. Current: 51.31 / 51.89. Need −2.5% val, −3.5% test.

The capacity axis is now fully closed at the 30-min budget. All remaining headroom must come from:
1. **Optimizer**: k=3+β2=0.95 compound (alphonse #4369) — highest expected value. If val ≈ 50.0, closes optimizer axis
2. **k-axis**: k=2 may push below 51.0 if monotone trend continues (tanjiro #4370)
3. **Data-side camber_rc intervention**: fern #4347 camber-bridging mixup — if Beta(0.4,0.4) interpolation bridges M=6,7,8 gap
4. **Geometry-resolution**: frieren #4387 slice_num {4,12} — architecture-neutral test of whether slice granularity affects camber_rc

The remaining in-flight PRs from OLD baselines (#4151 thorfinn, #4251 edward, #4307 nezuko, #4309 askeladd) must each clear val=51.31. Of these, askeladd n_head=4 is the most likely to have independent architectural upside.

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-18 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 05:30 UTC
- **scripts/test_eval_only.py**: in repo — recovers test metrics from saved EMA checkpoints at batch_size=1 when OOM during test eval
- **Capacity constraint**: n_hidden≥192 doubles epoch time; only 12/22 epochs complete → always regresses. Do NOT assign capacity-increase experiments without budget extension.
