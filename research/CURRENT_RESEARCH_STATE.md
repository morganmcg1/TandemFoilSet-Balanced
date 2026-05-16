# SENPAI Research State

- **Date:** 2026-05-16 23:45
- **Launch:** willow-pai2i-48h-r1 (round 11 — Lookahead-Lion era; programme best val=47.97)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=5/α=0.5) + triple-stack (PROGRAMME BEST)** | **47.9735** | **46.4900** | PR #4123, W&B `rx3negp7`, seed=0 | Merged 2026-05-16 23:45 |
| Lookahead-AdamW k=3/α=0.5 + triple-stack | 55.9681 | 53.4423 | PR #4158, W&B `oeb54ela`, seed=0 | Superseded by Lion |
| Lookahead-AdamW k=5/α=0.5 + triple-stack | 57.2203 | 54.0468 | PR #4132, W&B `d9ujr4oe`, seed=0 | 3-seed: median=57.05, seed=1=78.50 outlier |
| Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 | 60.4338 | 57.4381 | PR #3995, W&B `insf46p8`, seed=0 | 3-seed μ̂=61.66 ± 1.32 |

**Win threshold: val < 47.97 (Lookahead-Lion seed=0).** Seed=1 verification in progress (#4224 edward).

## Lookahead-Lion mechanism decomposition

| Intervention | Δ val | Mechanism |
|---|---|---|
| AdamW → Lion | −8.15 | Sign-based: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | −1.10 | Slow-weight averaging: reduce per-basin commitment variance |
| **Total: AdamW → Lookahead+Lion** | **−9.25** | Orthogonal-additive composition |

Lookahead's smaller gain on top of Lion (−1.10) vs on top of AdamW (−3.21) suggests partial mechanism overlap but no antagonism. Composition is genuinely additive.

## Lookahead sweep findings (context for Lookahead-Lion era)

**k-sweep at α=0.5 (Lookahead-AdamW, now informing Lookahead-Lion):**
| k | val (AdamW) | Δ vs k=5 |
|---|---|---|
| **k=3** | **55.97** | **−1.25** |
| k=5 | 57.22 | — |
| k=8 | 60.09 | +2.87 |

**α-sweep at k=5 (Lookahead-AdamW):**
| α | val (AdamW) | Δ vs α=0.5 |
|---|---|---|
| 0.3 | 61.58 | +4.36 |
| **0.5** | **57.22** | — |
| 0.7 | 56.92 | −0.30 |

These suggest optimal (k, α) is k≤3, α≥0.6 for AdamW. Likely transfers to Lookahead-Lion but unverified.

## Lookahead seed-1 outlier picture

**Lookahead-AdamW k=5:** seed=0=57.22, seed=1=78.50 (outlier, best_ep=10), seed=2=57.05.
**Lookahead-Lion k=5:** seed=0=47.97, seed=1=**pending (#4224 edward)**, seed=2=pending.

If Lion has same seed=1 outlier: 3-seed mean ~(47.97 + 78?? + 47??) = high variance, but median still competitive.
If Lion avoids the outlier (due to sign-based step reducing initialization sensitivity): 3-seed mean ~48-50.

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4224 | edward | Lookahead-Lion seed=1 (verify seed robustness of new best) | **NEW — highest priority** |
| #4202 | alphonse | Lookahead-AdamW k=3 seed=1 (3-seed canonical) | Running |
| #4203 | tanjiro | Lookahead-AdamW k=2 (k-sweep extension) | Running |
| #4210 | nezuko | Lookahead-AdamW k=3 seed=2 (3-seed canonical) | Running |
| #4211 | askeladd | Lookahead-AdamW k=3 + α∈{0.6,0.7} sweep | Running |
| #4213 | thorfinn | Lookahead-AdamW k=3 + α=0.8 | Running |
| #4182 | fern | Lookahead-AdamW + LR sweep ({7e-4, 1e-3}) | Running |
| #4183 | frieren | Lookahead-AdamW + β2 scan ({0.93, 0.97}) | Running |

**Note on #4202/#4203/#4210/#4211/#4213/#4182/#4183:** All are on Lookahead-AdamW (now superseded by Lookahead-Lion). Their k/α findings still inform the Lookahead-Lion hyperparameter space — let them run to completion. Results will guide Lookahead-Lion k/α sweep decisions.

## Round-11 merges / closures

- **#4123 edward (Lookahead-Lion)** MERGED: val=47.97 — new all-time best (Δ=−9.25 vs AdamW triple-stack)
- **#4158 nezuko (Lookahead k=3)** MERGED: val=55.97 — superseded by Lion, but k=3 finding persists
- **#4160 thorfinn (Lookahead k=5 seed=1)** closed: val=78.50 OUTLIER
- **#4175 askeladd (Lookahead α sweep at k=5)** closed: superseded; reassigned to k=3 α sweep

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
