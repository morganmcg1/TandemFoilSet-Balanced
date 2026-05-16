# SENPAI Research State

- **Date:** 2026-05-16 23:00
- **Launch:** willow-pai2i-48h-r1 (round 10 — Lookahead era; programme best val=57.22, potential k=3 winner pending)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-AdamW k=5/α=0.5 + triple-stack (PROGRAMME BEST)** | **57.2203** | **54.0468** | PR #4132, W&B `d9ujr4oe`, seed=0 | 3-seed canonical: μ̂=64.26 ± 12.4, median=57.05 (seed=1 outlier) |
| Lookahead k=3 (POTENTIAL BEST, pending SENPAI-RESULT) | 55.968 | 53.442 | PR #4158, W&B `oeb54ela`, seed=0 | Δ=−1.25 vs k=5; needs 3-seed verification (#4202 alphonse k=3 s1 assigned) |
| Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 | 60.4338 | 57.4381 | PR #3995, W&B `insf46p8`, seed=0 | 3-seed μ̂=61.66 ± 1.32 (closed) |
| h=128+SwiGLU+T_max=17 (prior best) | 62.1023 | 59.5529 | PR #3994, W&B `5q47ozlp`, seed=0 | 3-seed μ̂=63.06 ± 0.93 |

**Win threshold:** val < 57.22 (Lookahead k=5 seed=0). k=3 single-seed at 55.97 pending 3-seed verification.

## Lookahead-AdamW 3-seed canonical (round 10 closed)

| Seed | val_avg/mae_surf_p | test_avg/mae_surf_p | Source |
|---|---|---|---|
| 0 | 57.220 | 54.047 | nezuko #4132 (`d9ujr4oe`) |
| **1** | **78.503** | 74.185 | thorfinn #4160 (`pjvhrh4f`, 2 reruns bit-identical) — **OUTLIER** |
| 2 | 57.046 | (similar) | alphonse #4174 (`a6l7j8ec`) |

**Statistics:** μ̂=64.26, σ̂=12.4, median=57.05. Seed=1 is a clear outlier — 2-of-3 seeds at ~57, one at ~78. best_epoch=10 on seed=1 (vs 17 for seeds 0/2) suggests it lands in a worse basin early. For paper-facing reporting, the median (57.05) or explicit-outlier-noted mean is the appropriate statistic.

## Pending high-magnitude verification

### PR #4123 (edward, Lion) — Pure Lion verified, Lookahead-Lion missing

**Pure Lion Arm 1 verified:** W&B `ux8amr59` (rebased) reproduces original `rv8hjgtx` to 4dp: val=49.0721, test=47.0707, best_epoch=17. The Lion win is REAL at seed=0 (Δ=−8.15 vs Lookahead k=5).

**Arm 2 (Lookahead-Lion, lookahead_k=5/α=0.5 + use_lion=True) NOT YET RUN.** No W&B trace exists. Edward's currently-running `bny8b2mi` is another Pure Lion variant (`lookahead_k=0`), not Arm 2.

**Edward's PR has 2 commits only — no rebase pushed.** Posted detailed comment on #4123 directing: (1) push rebase commits, (2) run Arm 2 with exact config, (3) post terminal SENPAI-RESULT.

### Decision rules

- Arm 1 reproduces → Pure Lion at seed=0 is real (DONE: yes, val=49.07)
- Arm 2 > Arm 1 → Lookahead-Lion composition merges as new best (could reach ~46-48)
- Arm 2 < Arm 1 → Pure Lion alone merges (antagonistic)
- Arm 2 ≈ Arm 1 → Pure Lion merges (composition is no-op or null-improvement)

**Seed sensitivity concern:** Given Lookahead-AdamW's seed=1 outlier behavior, both Lion arms may also have high seed variance. Will address with seed-scan after Arm 2 completes.

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4123 | edward | Lion (rebased) — Pure Lion verified val=49.07; Arm 2 Lookahead-Lion pending | Pending Arm 2 + rebase push |
| #4158 | nezuko | Lookahead k sweep — k=3 finished val=55.97 (POTENTIAL WIN), k=8 running | Awaiting SENPAI-RESULT |
| #4175 | askeladd | Lookahead α sweep — α=0.3 finished val=61.58 (worse), α=0.7 running | Running |
| #4182 | fern | Lookahead + higher LR sweep ({7e-4, 1e-3}) on triple-stack | Running (lr=7e-4 in flight) |
| #4183 | frieren | Lookahead + β2 fine scan ({0.93, 0.97}) on triple-stack | Running (β2=0.93 in flight) |
| #4202 | alphonse | Lookahead k=3 seed=1 verification (checks if k=3 win is seed-stable) | **NEW** (assigned this round) |
| #4203 | tanjiro | Lookahead k=2 (k-sweep below k=3) | **NEW** (assigned this round) |

## Round-10 closures

- **#4160 thorfinn (Lookahead seed=1)** closed: val=78.50 OUTLIER, key finding for paper reporting
- **#4174 alphonse (Lookahead seed=2)** closed: val=57.05, clean reproduction
- **#4176 tanjiro (Lookahead + SWA-of-slow)** closed: val=57.22 NO-OP, confirms T_max=17 no-stationary-tail story (any post-hoc averaging fails on fast OR slow trajectory)

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
