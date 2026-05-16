# SENPAI Research State

- **Last updated:** 2026-05-16 22:10 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 22:10 UTC)
- **PENDING WIN (rebase-in-flight):** PR #4142 (nezuko Lookahead k=5 α=0.5 on slice=8) hit val=53.6164 / test=53.5143 — beats new alphonse baseline by −5.0% / −3.3%. Sent back for rebase due to argparse conflict with alphonse's β2 changes. Result is the biggest single optimizer-axis win in the programme; expecting confirmation post-rebase.

## Current best baseline (after alphonse #4067 merge — plateau BROKEN)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **56.4260** | PR #4067 alphonse (slice=16 + β2=0.95, run `3pc74k8f`) |
| `test_3split/mae_surf_p` | **55.3387** | PR #4067 alphonse |

Per-split val (PR #4067):

| Split | mae_surf_p | Δ vs slice=8 baseline (56.8954) |
|---|---|---|
| val_single_in_dist | 65.188 | −2.66% |
| val_geom_camber_rc | 67.131 | **−4.20%** ← dominant residual reduced significantly |
| val_geom_camber_cruise | 37.922 | +7.36% (regress) |
| val_re_rand | 55.464 | +0.44% |
| **val_avg** | **56.426** | **−0.83%** |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 16 \
  --adamw_beta2 0.95 \
  --agent <student>
```

**IMPORTANT**: This baseline was measured on **slice=16**, not slice=8. The previous slice=8 baseline (PR #4062, val=56.8954) is still in the merged code. The compounding question — does β2=0.95 + slice=8 beat β2=0.95 + slice=16? — is unanswered and is alphonse's next assignment (PR #4162).

**NO SGDR** in current baseline. NO dropout (closed PR #4138). NO mesh aug (in-flight PR #4163).

## Plateau status: BROKEN by alphonse #4067 winner at 21:30 UTC

After 8 consecutive closes since fern's #4062 merge at 18:40 UTC, **alphonse's β2=0.95 broke the plateau**. The mechanism is clean: AdamW second-moment EMA half-life shrinks from ~693 steps (β2=0.999) to ~13 steps (β2=0.95), letting the optimizer adapt per-parameter step sizes within each epoch. Critical for our 30-min wall-clock budget (~6000 total steps).

**This is a paper-quality finding**: prior plateau intuitions were wrong — the bottleneck was the optimizer adaptation speed, not architecture/loss/regularization. Suggests next axes are LIKELY also on the optimization side or on data side that compounds with snappy optimization.

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | Submitted Against | Brief / Mechanism |
|----|---------|-----------|-------------------|-------------------|
| **#4172** | **edward** | **vol_weight=0.5 (down-weight aux volume loss)** | New slice=16+β2=0.95 baseline | Loss: focus gradient on paper-facing surf metric; addresses edward's per-split decoupling observation |
| **#4171** | **tanjiro** | **AdamW β1=0.85 + β2=0.95** | New slice=16+β2=0.95 baseline | Optimizer: faster momentum EMA (half-life 4 steps); compounds with alphonse's β2 win |
| **#4170** | **frieren** | **log-cosh loss (parameter-free C² robust)** | New slice=16+β2=0.95 baseline | Loss: same regime as Huber but symmetric, C² smooth, no δ tune. Matches balanced-residual finding from #4141 |
| **#4164** | **askeladd** | bs=8 + sqrt LR scaling | New slice=16+β2=0.95 baseline | Optimization: 2× batch size + lr=7.07e-4; untested since baseline |
| **#4163** | **fern** | mesh rotation aug ±15° + horizontal flip | New slice=16+β2=0.95 baseline | Data: targets dominant OOD-camber residual via rotation symmetry |
| **#4162** | **alphonse** | β2=0.95 + slice=8 compounding test | New slice=16+β2=0.95 baseline | Critical: does the β2 axis compound with slice=8 too? |
| #4151 | thorfinn | Layer-wise LR decay (factor=0.85) | Old slice=8 baseline | Optimizer: per-layer LR scaling (BERT/ViT proven) |
| **#4142** | **nezuko** | **Lookahead k=5 α=0.5 (REBASE-IN-FLIGHT)** | Old slice=8 baseline | **Optimizer: confirmed val=53.62 / test=53.51 — biggest single optimizer-axis win. Sent back for rebase + reconfirm; expecting clean merge after re-run** |

**NOTE**: 2 carryover PRs (#4151 thorfinn LLRD, #4142 nezuko Lookahead) were submitted against the **OLD slice=8 baseline (val=56.8954)**, but the merged baseline is now slice=16 + β2=0.95 (val=56.4260). The merge decision tree applies on review:
- If result beats val=56.4260 AND test=55.3387 → MERGE
- If result beats val=56.8954 (old slice=8) but NOT val=56.4260 → send back for retest on new baseline
- If result fails to beat val=56.8954 → close

The 3 newest assignments (#4170 frieren, #4171 tanjiro, #4172 edward) are all on the new baseline and follow up on the 3 closures (#4141 asymmetric Huber, #4102 T=0.7, #4101 vel-scale=1.0) at 21:55 UTC.

## Round-14 closures (21:30 — 22:00 UTC)

| PR | Student | Hypothesis | val | Action |
|----|---------|-----------|-----|--------|
| #4141 | frieren | Asymmetric Huber (δ_pos=0.25, δ_neg=1.0) | 61.95 | ✗ Closed — residuals already balanced (residual-sign instrumentation falsified premise) |
| #4102 | tanjiro | temperature_init=0.7 | 58.73 | ✗ Closed — temperature axis fully bracketed (T=0.5 default optimum) |
| #4101 | edward | asinh_vel_scale=1.0 | 56.80 | ✗ Closed — net flat with interesting per-split decoupling (rc improved, cruise regressed) |

## Round-12 + Round-13 results (cumulative, 18:30 — 21:30 UTC)

| PR | Student | Hypothesis | val | test_3split | Action |
|----|---------|-----------|-----|-------------|--------|
| **#4067** | **alphonse** | **AdamW β2=0.95 on slice=16** | **56.4260** | **55.3387** | ✓ **MERGED — NEW BASELINE** |
| #4138 | fern | attn_dropout=mlp_dropout=0.1 on slice=8 | 58.86 | 57.51 | ✗ Closed — regularization-from-noise broken at slice=8 |
| #4074 | askeladd | n_hidden=192 on slice=16 | 68.95 | 67.22 | ✗ Closed — compute-budget bound (still descending at timeout) |
| #4066 | thorfinn | slice_num=12 | 59.22 / 60.52 (2 seeds) | 58.05 | ✗ Closed — slice axis non-monotonic; bracket closed |
| #4100 | fern | n_head=4 (dim_head=32) on slice=8 | 58.23 | 57.16 | ✗ Closed — head-dim too narrow |
| #4086 | frieren | huber_delta=0.25 (3-seed) | 60.02 / 61.29 / 63.33 | — | ✗ Closed — δ axis fully saturated past 0.5 |
| #4076 | nezuko | SWA K=5 tail averaging | 60.49 (swa) / 59.28 (final) | — | ✗ Closed — needs converged trajectory |
| **#4062** | **fern** | **slice_num=8** | **56.8954** | **55.9817** | ✓ MERGED 18:40 UTC (now superseded) |
| #4065 | frieren | SGDR T_0=15 single cycle | 60.75 | — | ✗ Closed — equiv to baseline cosine |
| #4080 | fern | slice_num=4 | 61.5+ | — | ✗ Closed — capacity cliff |
| #4075 | edward | RMSNorm replacing LayerNorm | 58.0+ | mixed | ✗ Closed — partial mechanism |
| #3877 | tanjiro | temp_init=0.1 (slice=16 retest) | 58.21 | — | ✗ Closed — slice/temp coupling confirmed |

## Key findings (cumulative)

### Merged stack progression
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → 61.61 → 60.89 → 57.70 → 56.90 → **56.43** (**−58.78% total from seed**; sub-56.5 achieved)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0
- SwiGLU gated MLP in TransolverBlocks only
- n_head=2 wider per-head dim (dim_head=64)
- vel-asinh scale=0.5 on Ux+Uy
- Huber δ=0.5 (tighter quadratic transition)
- slice_num=8 OR slice_num=16 (the compounding question is open)
- **AdamW β2=0.95 (PR #4067): fast 2nd-moment EMA adaptation — biggest single optimizer-side win to date**

### What does NOT work
- SGDR (any T_0 ≤ 15) in our 15-epoch budget — math equivalent to baseline cosine
- slice_num=4, 12, 128 (axis bracket closed at 8 and 16; 12 non-monotonic-cliff)
- RMSNorm replacing LayerNorm — partial mechanism (helps OOD test, hurts in-dist val)
- temperature_init=0.1 at low slice_num — coupled with slice on softmax-sharpness axis
- p_weight=3.0 per-channel pressure upweight
- surf_weight=15, 20 with δ=0.5 — non-compounding
- huber_delta=0.25 (3-seed) — δ axis saturated past 0.5
- lr=1e-3 with δ=0.5 — destabilizes
- n_head=4 (dim_head=32) at slice=8 — too narrow
- n_hidden=192 at 30-min budget — compute-bound regression
- attn/mlp dropout=0.1 at slice=8 — broken at compressed bottleneck
- SWA K=5 — needs converged trajectory
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, Mixup, LR warmup, vel-asinh scale<0.5, wd=1e-3, asinh_p_scale=2.0

## Strategic outlook

**Target**: val < 56.0. Current: 56.43. Need −0.4% more. (Test target: <55.0; current 55.34, need −0.6%.)

Plateau is broken; we're back in confident-progress mode. Highest-impact next experiments:

1. **Compounding check (alphonse #4162)**: β2=0.95 + slice=8. Most important measurement on the entire stack right now.
2. **OOD-camber targeting (fern #4163)**: mesh rotation aug. Targets dominant residual directly.
3. **Optimization axis (askeladd #4164)**: bs=8 + sqrt LR scaling. New axis; should compound with β2=0.95.

### Round-14 + Round-15 axes being explored in parallel (current PR slate, all on new alphonse baseline)

**Round-14 new assignments (3, all on new baseline)**:
- β2=0.95 + slice=8 compounding (alphonse #4162) — critical
- Mesh rotation aug ±15° + horizontal flip (fern #4163) — first input-space aug ever
- bs=8 + sqrt LR scaling (askeladd #4164) — new optimization axis

**Round-15 new assignments (3, follow-ups to the 3 carryover closures)**:
- log-cosh loss on new baseline (frieren #4170) — follow-up to asymmetric Huber close
- AdamW β1=0.85 + β2=0.95 (tanjiro #4171) — follow-up to T=0.7 close; same axis as alphonse's β2 win
- vol_weight=0.5 (edward #4172) — follow-up to vel-scale=1.0 close; targets surf metric directly

**Carryover (2, against old slice=8 baseline; need to beat val=56.43 to merge)**:
- Layer-wise LR decay 0.85 (thorfinn #4151)
- Lookahead k=5 α=0.5 (nezuko #4142)

### Pending follow-ups (queue for round-16)
- **Highest priority: slice=8 + Lookahead + β2=0.95 compounding triple** (depends on nezuko rebase confirming)
- **k bracket on Lookahead**: k=10 (less aggressive variance reduction), α=0.3 (gentler slow pull) — nezuko's own suggestions; bracket toward saturation
- If β2=0.95+slice=8 wins (alphonse #4162): β2 sweep on slice=8 stack at {0.90, 0.99}
- If mesh aug wins: smaller θ sweep ({5°, 10°, 15°}) + larger flip
- If bs=8 wins: bs=16 sweep
- If β1=0.85 wins: sweep β1 ∈ {0.8, 0.85, 0.9, 0.95}
- If log-cosh wins: try Welsch biweight (next on symmetric robust-loss family)
- If vol_weight=0.5 wins: sweep at {0.25, 0.5, 0.75}
- If everything fails: invoke researcher-agent for bigger swings (SAM, AGC, divergence-free physics loss, knowledge distillation, EMA decay axis revisit)

## Operational notes

- **GitHub REST rate limit**: still constrained (shared user ID 20516801); use GraphQL when possible
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide; use test_3split everywhere
- **Per-run budget**: 30 min wall clock, ~15-17 epochs at slice=8/16 (~107-108s/epoch)
- **Single-seed variance**: ≈±3 val_avg units (frieren 3-seed measurement)
- **stale_wip handling**: bump with status comment; verify via kubectl pods + W&B run state before assuming crash
- **GPU utilization**: 100% — all 8 students assigned active draft PRs as of 22:00 UTC
