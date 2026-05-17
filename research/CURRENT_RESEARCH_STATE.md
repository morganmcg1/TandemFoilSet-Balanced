# SENPAI Research State

- **Last updated:** 2026-05-17 ~07:50 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 07:50 UTC)

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
| **#4461** | **nezuko** | Lookahead α=0.7 multi-seed (3 seeds) on k=3 | NEW k=3 baseline | NEW — direct follow-up from #4307 asymmetry finding; compound with k=3's higher sync frequency |
| **#4468** | **askeladd** | FiLM conditioning on camber M channel | NEW k=3 baseline | NEW — paper-direction architectural intervention targeting camber_rc=63.85 residual |
| **#4469** | **frieren** | surf_weight bracket (5, 20) on k=3 baseline | NEW k=3 baseline | NEW — surf_weight=10.0 never tuned; first loss-formulation axis on k=3 |
| **#4453** | **alphonse** | n_layers depth bracket (3, 7) | NEW k=3 baseline | WIP — depth vs epochs tradeoff; same mechanism as k=3 winning k=5 |
| **#4370** | **tanjiro** | k=2+k=4 fine bracket | NEW k=3 baseline | WIP — k=4 arm still pending; k=2 trials showed regression |
| **#4400** | **fern** | Cosine eta_min floor (5e-5, 1e-5) | NEW k=3 baseline | WIP |
| **#4401** | **edward** | AdamW eps bracket (1e-7, 1e-9) | NEW k=3 baseline | WIP |
| **#4404** | **thorfinn** | mlp_ratio bracket (1.0, 1.667) | NEW k=3 baseline | WIP |

All 8 GPUs occupied. Zero idle students.

## Recent closures (rounds 29–30)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| ✗ **#4387** | **frieren** | **slice_num bracket (4, 12) on k=3** | **51.81 (4) / 53.45 (12)** | ✗ **Closed — val_geom_camber_rc monotonically regresses both directions. Slice axis CLOSED at 8.** |
| ✗ **#4309** | **askeladd** | **n_head=4 on k=5+β2=0.95** | **52.65 (best/5 seeds)** | ✗ **Closed — camber_rc REGRESSED (+1.40); high seed variance (2.16 spread). n_head axis closed at n_head=2 for n_hidden=128.** |
| ✗ **#4307** | **nezuko** | **Lookahead α-bracket (0.3, 0.7) on k=5+β2=0.95** | **51.72 (α=0.7 best seed)** | ✗ **Closed — cannot beat 51.31. But α=0.7 beats α=0.5 with 2/2 seeds (real asymmetry mechanism). Follow-up: #4461 on k=3.** |
| ✗ **#4369** | **alphonse** | **k=3+β2=0.95 compound** | **51.18 seed-1 / 52.65 seed-2** | ✗ **Closed — 2-seed spread 11× win margin; single_in_dist regression mechanism. β2 closed at k=3.** |

## Key mechanism insights

- **k-axis dominant lever**: k=3 delivers −3.09% val by maximizing slow-weight update frequency
- **α-axis asymmetric at k=5**: α=0.7 beats α=0.5 (2/2 seeds), α=0.3 uniformly worse — stronger slow-weight pull compounds with β2=0.95 adaptation. Mechanism amplified at k=3 (more frequent syncs).
- **β2=0.95 × Lookahead k-dependent**: compounding at k=5 (long excursion window), substitutive at k=3 (short window). β2 closed at k=3.
- **Slice axis optimum at 8**: monotonic degradation in both directions under k=3 Lookahead. Bracket {4, 8, 12, 16} exhausted.
- **n_head axis: n_head=2 optimal at n_hidden=128**: n_head=4 (32-dim/head) below expressive threshold; camber_rc specifically regresses.
- **camber_rc residual (63.85) dominant**: data-side axis (frequency, mixup) both closed. Slice axis closed. Architecture-level M conditioning (FiLM, #4468) is the primary remaining path.
- **Capacity axis closed at 30-min budget**: n_hidden≥192 = compute-budget regression.
- **LLRD substitutive with Lookahead on test**.
- **lr-axis closed under Lookahead**: lr=5e-4 optimal.
- **~8 consecutive closures since last merge** — shifting to architecture-level and loss-formulation interventions.

## What works on the full stack

- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0, SwiGLU gated MLP (mlp_ratio=1.333), n_head=2, vel-asinh=0.5
- Huber δ=0.5, slice_num=8
- AdamW β2=0.999 (default), lr=5e-4, wd=1e-4
- **Lookahead k=3 α=0.5 — dominant lever on 30-min budget**

## What does NOT work

- β1 axis: β1=0.85 or β1=0.95 (default 0.9 is local optimum)
- β2=0.95 under Lookahead k=3 (substitutive at short excursion window)
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
- n_hidden=192 (compute-budget regression)
- lr=1e-3 under Lookahead (7× variance)
- LLRD under Lookahead (substitutive on test)
- slice_num ≠ 8 under k=3 (monotonic degradation away from 8)
- n_head=4 at n_hidden=128 (32-dim/head below expressive threshold; camber_rc regresses)
- Lookahead α=0.3 (fast/slow weights decouple; uniform +3-5% regression)

## Strategic outlook

**Target**: val < 50.0, test < 50.0. Current: 51.31 / 51.89. Need −2.5% val, −3.5% test.

**Plateau status**: 8+ consecutive closures since last merge. Per Plateau Protocol, now in active escalation:

**Tier 1 (in flight — optimizer fine-tuning):**
1. α=0.7 on k=3 (#4461 nezuko) — highest-EV open item; mechanism confirmed at k=5 with 2/2 seeds
2. k=2 bracket (#4370 tanjiro) — k-axis limit, k=2 trials so far showed regression
3. eta_min floor (#4400 fern) — scheduler tail
4. AdamW eps (#4401 edward) — numerical stability
5. n_layers depth (#4453 alphonse) — depth vs epochs tradeoff
6. mlp_ratio width (#4404 thorfinn) — FFN width vs epochs

**Tier 2 (in flight — loss formulation):**
7. surf_weight bracket (#4469 frieren) — surf_weight=10.0 never tuned

**Tier 3 (in flight — architecture escalation per Plateau Protocol):**
8. FiLM conditioning on M (#4468 askeladd) — first paper-direction architectural intervention; targets camber_rc=63.85 directly

**Next escalation if all Tier 1–3 come back negative:**
- (a) FNO or PointNet++ model family swap — philosophical shift from Transolver
- (b) Geometric priors in loss: surface curvature-weighted residual
- (c) Multi-task Ux/Uy auxiliary loss with pressure as primary

## Operational notes

- **cruise NaN**: fleet-wide; test_3split = (test_single_in_dist + test_geom_camber_rc + test_re_rand)/3
- **W&B test namespace**: `test/test_*/mae_surf_p` (not bare `test_*`)
- **Per-run budget**: 30 min, ~15-18 epochs at slice=8 (~108s/epoch)
- **GPU utilization**: 100% — all 8 students assigned as of 07:50 UTC
- **scripts/test_eval_only.py**: in repo — recovers test metrics from saved EMA checkpoints at batch_size=1 when OOM during test eval
- **Capacity constraint**: n_hidden≥192 doubles epoch time — do NOT assign capacity-increase experiments without budget extension
- **β2-axis diagnostic**: +3–5% single_in_dist regression across both seeds = per-split substitutive signature for optimizer-internal compounds at k=3
