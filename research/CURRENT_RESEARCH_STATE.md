# SENPAI Research State

- **Last updated:** 2026-05-17 ~05:55 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 05:55 UTC)

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

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --agent <student>
```

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
| **#4369** | **alphonse** | k=3 + β2=0.95 compound | NEW k=3 baseline | WIP — highest expected value; could reach val≈50.0 if fully additive |
| **#4370** | **tanjiro** | k=2+k=4 fine bracket | NEW k=3 baseline | WIP — characterizes k-axis limit |
| **#4387** | **frieren** | slice_num bracket (4, 12) on k=3 | NEW k=3 baseline | WIP — geometry-resolution axis untested under k=3 |
| **#4400** | **fern** | Cosine eta_min floor (5e-5, 1e-5) | NEW k=3 baseline | NEW — prevent near-zero LR at timeout |
| **#4401** | **edward** | AdamW eps bracket (1e-7, 1e-9) | NEW k=3 baseline | NEW — numerical stability denominator, never tested under Lookahead |
| **#4404** | **thorfinn** | mlp_ratio bracket (1.0, 1.667) | NEW k=3 baseline | NEW — smaller FFN = more epochs in 30-min budget |
| **#4309** | **askeladd** | n_head=4 architecture sweep | k=5+β2=0.95 baseline | WIP — needs to beat 51.31 |
| **#4307** | **nezuko** | Lookahead α-bracket sweep (α=0.3, α=0.7) | k=5+β2=0.95 baseline | WIP — needs to beat 51.31 |

All 8 GPUs occupied. Zero idle students.

## Recent closures (this round)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✗ **#4347** | **fern** | **Camber-bridging feature-space mixup Beta(0.4,0.4)** | **58.05** | ✗ **Closed — paper-quality null result. Data-side axis FULLY CLOSED. Only 16.5% of mixes hit held-out range; all splits regress.** |
| ✗ **#4251** | **edward** | **Lookahead+lr=1e-3** | **55.15** (best) | ✗ **Closed — 7× variance, converges to worse minimum. lr-axis closed under Lookahead.** |
| ✗ **#4151** | **thorfinn** | **LLRD=0.85/0.95 + Lookahead** | **53.98** (on old baseline) | ✗ **Closed — substitutive with Lookahead on test; new baseline jumped beyond reach.** |
| ✗ **#4313** | **frieren** | **n_hidden=192** | **60.24** | ✗ **Closed — compute-budget regression. Capacity axis CLOSED.** |
| ✓ **#4266** | **alphonse** | **Lookahead k=3** | **51.307** | ✓ **MERGED — CURRENT BASELINE** |

## Key mechanism insights

- **k-axis dominant lever**: k=3 delivers −3.09% val by maximizing slow-weight update frequency (2100 updates vs 1260 at k=5)
- **β2=0.95 × Lookahead orthogonal at k=5** (PR #4249); untested at k=3 — alphonse testing now
- **slice=16 × β2=0.95 anti-compound under Lookahead** (PR #4283 closed)
- **camber_rc residual (63.85) is the dominant residual** — 10% above val_single_in_dist=57.8
- **DATA-SIDE AXIS CLOSED for camber_rc** (PRs #4311, #4347): frequency reweighting AND feature-space mixup BOTH fail to bridge held-out M∈{6,7,8} gap. Architecture-level conditioning is the remaining path.
- **Capacity axis closed at 30-min budget** (PR #4313): n_hidden=192 doubles epoch time, only 12/22 epochs complete
- **LLRD substitutive with Lookahead on test**: compounds with slice/β2 but partially redundant under slow-weight averaging
- **lr-axis closed under Lookahead**: lr=1e-3 (7× variance, worse minimum); lr=5e-4 is optimal
- **LR warmup axis closed** (PR #4334): consumes budget, camber_rc regression
- **weight_decay axis closed** (PR #4284): Lookahead×wd substitutive at wd>1e-4

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP (mlp_ratio=1.333), n_head=2, vel-asinh=0.5
- Huber δ=0.5, slice_num=8
- AdamW β2=0.999 (default), lr=5e-4, wd=1e-4
- **Lookahead k=3 α=0.5 — dominant lever on 30-min budget**

## What does NOT work

- β1 axis: β1=0.85 or β1=0.95 (default 0.9 is local optimum)
- EMA=0.995 (substitutes with Lookahead)
- AGC, grad_clip=1.0 (wrong instrument)
- bs=8, DropPath, dropout, SWA
- All loss-shape axes: Huber δ bracket, asym Huber, log-cosh, Welsch
- Per-sample peak-|p| reweighting (backbone starvation)
- Per-channel surface loss reweighting
- AoA rotation augmentation
- Lookahead+slice=16+β2=0.95 triple (anti-compounding)
- weight_decay > 1e-4 under Lookahead
- Camber-stratified frequency oversampling (val improves, test regresses)
- Camber-bridging mixup (both val AND test regress — data-side axis CLOSED)
- LR warmup 360 steps (consumes budget, camber_rc regression)
- n_hidden=192 (compute-budget regression, capacity axis closed)
- lr=1e-3 under Lookahead (7× variance, converges to worse minimum)
- LLRD under Lookahead (substitutive on test)

## Strategic outlook

**Target**: val < 50.0, test < 50.0. Current: 51.31 / 51.89. Need −2.5% val, −3.5% test.

All simple single-axis levers have been explored. Remaining open axes:

1. **Optimizer compound (alphonse #4369, k=3+β2=0.95)** — highest expected value. If fully additive at k=3 (as it was at k=5), val ≈ 50.0 or below. Defines the current-stack optimizer optimum.
2. **k-axis limit (tanjiro #4370, k=2+k=4)** — k=2 could push below 51.0 if trend continues
3. **Geometry resolution (frieren #4387, slice=4/12)** — architectural but zero-code-change test; camber_rc floor depends on slice granularity
4. **Scheduler tail (fern #4400, eta_min floor)** — recover late-epoch training that's wasted at near-zero LR
5. **Optimizer stability (edward #4401, eps bracket)** — never tested; interaction with Lookahead k=3's faster trajectory
6. **FFN width vs epochs (thorfinn #4404, mlp_ratio=1.0)** — counterintuitive: smaller model → more epochs in budget
7. **Architecture attention (askeladd #4309, n_head=4)** — still in flight on old baseline; interesting if it clears new bar
8. **Lookahead α-axis (nezuko #4307, α=0.3/0.7)** — still in flight on old baseline; probably won't clear 51.31

**Next plateau protocol**: if k=3+β2=0.95 compound and k=2 both come back negative (or minimal), the optimizer axis is exhausted. At that point, shift to: (a) architectural conditioning on camber M (FiLM/cross-attention), (b) different model family (e.g. PointNet++, FNO), or (c) loss reformulation with geometric priors.

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-18 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 05:55 UTC
- **scripts/test_eval_only.py**: in repo — recovers test metrics from saved EMA checkpoints at batch_size=1 when OOM during test eval
- **Capacity constraint**: n_hidden≥192 doubles epoch time — do NOT assign capacity-increase experiments without budget extension
