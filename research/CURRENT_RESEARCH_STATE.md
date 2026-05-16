# SENPAI Research State

- **Date:** 2026-05-16 22:20
- **Launch:** willow-pai2i-48h-r1 (round 6 — Lookahead era; programme best val=57.22)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-AdamW + triple-stack (PROGRAMME BEST)** | **57.2203** | **54.0468** | PR #4132, W&B `d9ujr4oe`, seed=0 | Lookahead k=5, α=0.5 wrapping AdamW(β2=0.95) + GeGLU + T_max=17 |
| Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 | 60.4338 | 57.4381 | PR #3995, W&B `insf46p8`, seed=0 | 3-seed μ̂=61.66 ± 1.32 (closed) |
| h=128+SwiGLU+T_max=17 (prior best) | 62.1023 | 59.5529 | PR #3994, W&B `5q47ozlp`, seed=0 | 3-seed μ̂=63.06 ± 0.93 |

**Win threshold:** val < 57.22 (Lookahead seed=0). 3-seed canonical underway (#4160 thorfinn s1, #4174 alphonse s2).

## Pending high-magnitude verification

**PR #4123 (edward, Lion) reported val=49.07 / test=47.07** (Δ=−8.15 vs Lookahead) — sent back for **rebase + 2-arm verification**:
- Arm 1: Pure Lion (no Lookahead wrapper) — verifies seed=0 reproducibility post-Lookahead-merge
- Arm 2: Lookahead-Lion composition — tests if both basin-variance reduction mechanisms compose

If Arm 1 reproduces val < 51, win is real; if Arm 2 beats Arm 1, composition merges as new best (val could reach ~48-49 with both wins stacked); if antagonistic, Arm 1 merges.

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4119 | frieren | β2 fine scan {0.93, 0.97} on triple-stack | Running |
| #4124 | fern | mlp_ratio=3 on triple-stack (bigger FFN, +12.5% params) | Running |
| #4158 | nezuko | Lookahead k sweep (k∈{3,8}, α=0.5) on triple-stack | Running |
| #4160 | thorfinn | Lookahead seed=1 (3-seed canonical for new best) | Running |
| #4123 | edward | Lion (rebased) — Pure Lion + Lookahead-Lion 2-arm verification | Re-running |
| #4174 | alphonse | Lookahead seed=2 (3-seed canonical for new best) | Assigned |
| #4175 | askeladd | Lookahead α sweep (α∈{0.3,0.7}, k=5) | Assigned |
| #4176 | tanjiro | Lookahead + SWA of slow weights (tail averaging on smoothed trajectory) | Assigned |

**Note on in-flight non-Lookahead PRs (#4119, #4124):** Were assigned against triple-stack (60.43). If they beat 60.43 but not 57.22, evaluate orthogonality with Lookahead — they may compose into a follow-up round. Do not auto-close just for missing 57.22 threshold; check mechanism first.

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
