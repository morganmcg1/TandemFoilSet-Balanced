# SENPAI Research State

- **Date:** 2026-05-16 23:35
- **Launch:** willow-pai2i-48h-r1 (round 11 — Lookahead k=3 era; programme best val=55.97)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead k=3/α=0.5 + triple-stack (PROGRAMME BEST)** | **55.9681** | **53.4423** | PR #4158, W&B `oeb54ela`, seed=0 | Merged 2026-05-16 23:35 |
| Lookahead k=5/α=0.5 + triple-stack | 57.2203 | 54.0468 | PR #4132, W&B `d9ujr4oe`, seed=0 | 3-seed k=5: median=57.05, seed=1=78.50 outlier |
| Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 | 60.4338 | 57.4381 | PR #3995, W&B `insf46p8`, seed=0 | 3-seed μ̂=61.66 ± 1.32 (closed) |

**Win threshold: val < 55.97 (Lookahead k=3 seed=0).** 3-seed canonical for k=3 in progress.

## k-sweep finding (definitive)

| k | val_avg | Δ vs k=5 | Decision |
|---|---|---|---|
| **k=3** | **55.97** | **−1.25** | **MERGED — new best** |
| k=5 | 57.22 | — | superseded |
| k=8 | 60.09 | +2.87 | regression |

Monotone: smaller k → more sync events → lower val. k=3 gives 2125 syncs/run; k=8 gives 797. Optimal k at α=0.5 is ≤3. k=2 in flight (tanjiro #4203).

## α-sweep finding (partial, at k=5)

| α | val_avg (at k=5) | Δ vs α=0.5 |
|---|---|---|
| 0.3 | 61.58 | +4.36 (worse) |
| **0.5** | **57.22** | — |
| **0.7** | **56.92** | **−0.30** |

Monotone: larger α → lower val at k=5. Optimum α is >0.5. α sweep at k=3 now underway (#4211 α∈{0.6,0.7}, #4213 α=0.8).

## Lookahead 3-seed canonical picture (k=5 era, complete; k=3 era, in progress)

**k=5, α=0.5:**
| Seed | val | Source |
|---|---|---|
| 0 | 57.220 | nezuko #4132 |
| 1 | **78.503** (OUTLIER, best_ep=10) | thorfinn #4160 |
| 2 | 57.046 | alphonse #4174 |

μ̂=64.26 ± 12.4, median=57.05. Paper-facing: report median with outlier footnote.

**k=3, α=0.5 (in progress):**
| Seed | val | Source |
|---|---|---|
| 0 | 55.968 | nezuko #4158 |
| 1 | (running: alphonse #4202) | — |
| 2 | (running: nezuko #4210) | — |

## Pending high-magnitude verification

### PR #4123 (edward, Lion) — Pure Lion verified, Lookahead-Lion missing

**Pure Lion Arm 1 verified:** W&B `ux8amr59` (rebased) reproduces original `rv8hjgtx` to 4dp: val=49.0721, test=47.0707, best_epoch=17. The Lion win is REAL at seed=0 (Δ=−6.27 vs NEW Lookahead k=3 best).

**Arm 2 (Lookahead-Lion) NOT YET RUN.** No W&B trace exists. Edward sent detailed comment directing: (1) push rebase commits, (2) run Arm 2 with lookahead_k=5/α=0.5 + use_lion=True, (3) post terminal SENPAI-RESULT.

Decision rules: Arm 2 > Arm 1 → Lookahead-Lion merges (new best); Arm 2 ≈ Arm 1 → Pure Lion merges (val~49); Arm 2 < Arm 1 → Pure Lion merges.

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4123 | edward | Lion (rebased) — Pure Lion verified; Arm 2 Lookahead-Lion pending | Pending Arm 2 |
| #4182 | fern | Lookahead k=3 + higher LR ({7e-4, 1e-3}) | Running |
| #4183 | frieren | Lookahead k=3 + β2 fine scan ({0.93, 0.97}) | Running |
| #4202 | alphonse | Lookahead k=3 seed=1 (3-seed canonical verify) | Running |
| #4203 | tanjiro | Lookahead k=2 (extend k-sweep below k=3) | Running |
| #4210 | nezuko | Lookahead k=3 seed=2 (3-seed canonical verify) | **NEW** |
| #4211 | askeladd | Lookahead k=3 + α sweep (α∈{0.6,0.7}) | **NEW** |
| #4213 | thorfinn | Lookahead k=3 + α=0.8 (push saturation boundary) | **NEW** |

## Round-11 closures / merges

- **#4158 nezuko (Lookahead k=3)** MERGED: val=55.97, new programme best (Δ=−1.25 vs k=5)
- **#4160 thorfinn (Lookahead k=5 seed=1)** closed: val=78.50 OUTLIER canonical data
- **#4175 askeladd (Lookahead α sweep at k=5)** closed: α=0.7 won old baseline but not new; reassigned to k=3 α sweep

## Key mechanistic findings to date

### Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null |
| LR=1e-3 under T_max=15 | Divergence |
| Head+embed LR boost (1.5–2.5×) | All null/worse |
| T_max < 17 | Suboptimal (PyTorch Gotcha #3) |
| RMSNorm (vs LayerNorm) | −5.18 val regression |
| slice_num=128 (2× attention) | −10.92 val regression |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail | Regression + kick-out |
| SWA tail4 at T_max=17 cosine | Regression + non-stationary tail |
| EMA(0.999) on fast weights | Regression — same root cause |
| SWA on slow (Lookahead) weights | No-op — same root cause |
| β1=0.95 + β2=0.95 | +2.86 val regression |
| Lookahead k=8 | +2.87 val regression |
| Lookahead α=0.3 | +4.36 val regression |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()`
3. T_max must equal total_epochs — T_max<total causes hard-zero LR

## Key mechanistic findings this round

### Lookahead mechanism (PR #4132 nezuko — current programme best)

| Config | val_avg | Δ |
|---|---|---|
| Triple-stack (β2=0.95+GeGLU+T_max=17) | 60.43 | — |
| Triple-stack + Lookahead (k=5, α=0.5) | **57.22** | **−3.21** |

Online basin-averaging (every k=5 steps: θ_slow ← θ_slow + α·(θ_fast − θ_slow)) delivers flat-minima benefits during training. Dominant OOD gains: val_geom_camber_cruise −6.12, val_re_rand −3.96, test_re_rand −4.63. In-distribution barely moves — flat-minima → generalization story confirmed.

### Post-hoc weight averaging is exhausted at T_max=17

| Experiment | Failure mode |
|---|---|
| PR #3644 (SWA + constant-LR tail) | LR kick-out: constant tail jumps LR ~25× before basin floor |
| PR #4089 (SWA tail4 at T_max=17 cosine) | Budget-limited: cosine still descending in SWA window |
| PR #4121 (EMA decay=0.999) | Same: EMA half-life 1.87ep drags in higher-loss epochs |

**All three failed for the same reason: T_max=17 cosine has no stationary tail window for the FAST trajectory.** Online averaging (Lookahead) and averaging on the SLOW trajectory (Lookahead + SWA-of-slow, PR pending for tanjiro) are the only avenues that can work.

### Triple-stack decomposition (3-seed canonical complete)

| Config | val_avg (single seed) | μ̂ (3-seed) |
|---|---|---|
| T_max=17 SwiGLU | 62.10 | 63.06 ± 0.93 |
| T_max=17 GeGLU (β2=0.9 default) | 62.47 | — (single seed, but ablation arm) |
| T_max=17 GeGLU + β2=0.95 (triple-stack) | 60.43 | **61.66 ± 1.32** (closed) |

β2=0.95 was −2.04 dominant lever in triple-stack. GeGLU contributed −0.59 alone.

### Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null |
| LR=1e-3 under T_max=15 | Divergence |
| Head+embed LR boost (1.5–2.5×) | All null/worse |
| T_max < 17 | Suboptimal (PyTorch Gotcha #3) |
| RMSNorm (vs LayerNorm) | −5.18 val regression |
| slice_num=128 (2× attention) | −10.92 val regression |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail | Regression + kick-out (#3644) |
| SWA tail4 at T_max=17 cosine | Regression + non-stationary tail (#4089) |
| EMA(0.999) on fast weights | Regression (#4121) — same root cause as SWA failures |
| β1=0.95 + β2=0.95 (compound momentum) | +2.86 val regression (#4118) |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds (#3934)
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()` (#3993)
3. T_max must equal total_epochs — T_max<total causes hard-zero LR (#3994)

## Next research directions

### Priority 1 (Lion verification — pending edward's rebased PR)

If pure Lion val=49.07 reproduces post-rebase, this is the largest single result in the programme. Lookahead-Lion composition is the natural next-step compose test.

### Priority 2 (Lookahead 3-seed canonical + α/k scan)

- seed=1 (#4160 thorfinn), seed=2 (#4174 alphonse) underway
- α sweep at k=5 (#4175 askeladd) underway
- k sweep at α=0.5 (#4158 nezuko) underway

After all 4 land, we'll know Lookahead's σ̂ and optimal (k, α).

### Priority 3 (composition with Lookahead)

- **Lookahead-Lion** (edward Arm 2) — both target small-batch gradient variance, may be orthogonal
- **Lookahead + mlp_ratio=3** (if fern #4124 shows orthogonal gain)
- **Lookahead + SWA-of-slow-weights** (tanjiro) — averaging on the smoothed slow trajectory
- **Lookahead + larger model (h=192)** — VRAM headroom + Lookahead's no v_t savings

### Priority 4 (speculative)

- Physics-informed loss (continuity equation soft constraint)
- SAM (sharpness-aware, 2× compute — budget-limited to ~8-9ep, gotcha #3 risk)
- Cross-slice attention (between-slice heads)
- Layer-wise LR decay
