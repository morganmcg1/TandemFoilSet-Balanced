# SENPAI Research State

- **Date:** 2026-05-16 21:45
- **Launch:** willow-pai2i-48h-r1 (round 6 — Lookahead era; NEW programme best val=57.22)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-AdamW + triple-stack (NEW PROGRAMME BEST)** | **57.2203** | **54.0468** | PR #4132, W&B `d9ujr4oe`, seed=0 | Lookahead k=5, α=0.5 wrapping AdamW(β2=0.95) + GeGLU + T_max=17 |
| Triple-stack: h=128+GeGLU+β2=0.95+T_max=17 | 60.4338 | 57.4381 | PR #3995, W&B `insf46p8`, seed=0 | 2-seed mean ≈60.99 (seed=1 was 61.54) |
| h=128+SwiGLU+T_max=17 (prior best) | 62.1023 | 59.5529 | PR #3994, W&B `5q47ozlp`, seed=0 | 3-seed μ̂=63.06 ± 0.93 |

**Win threshold:** val < 57.22 (Lookahead seed=0). Seed variance unknown — Lookahead canonical underway.

## Active WIP experiments

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #4158 | nezuko | Lookahead k sweep (k∈{3,8}, α=0.5) on triple-stack | Assigned |
| #4160 | thorfinn | Lookahead seed=1 (3-seed canonical for new best) | Assigned |
| #4117 | alphonse | Triple-stack seed=2 (3-seed canonical for triple-stack) | Running |
| #4118 | askeladd | β1=0.95 on triple-stack (compound momentum) | Running |
| #4119 | frieren | β2 fine scan {0.93, 0.97} on triple-stack | Running |
| #4121 | tanjiro | EMA model weights (decay=0.999) on triple-stack | Running |
| #4123 | edward | Lion optimizer on triple-stack (lr/3, sign-based updates) | Running |
| #4124 | fern | mlp_ratio=3 on triple-stack (bigger FFN, +12.5% params) | Running |

**Note on in-flight PRs (#4117–#4124):** These compare against triple-stack baseline (60.43). With new best = 57.22, any result that beats 60.43 but not 57.22 is still informative — the lever may compose orthogonally with Lookahead in a next-round stack experiment. Do not auto-close on arriving results below 57.22 threshold — evaluate mechanism.

## Key mechanistic findings this round

### Lookahead mechanism (PR #4132 nezuko — NEW)

| Config | val_avg | Δ |
|---|---|---|
| Triple-stack (β2=0.95+GeGLU+T_max=17) | 60.43 | — |
| Triple-stack + Lookahead (k=5, α=0.5) | **57.22** | **−3.21** |

**Lookahead is the dominant new lever.** Online basin-averaging (every k=5 steps: θ_slow ← θ_slow + α·(θ_fast − θ_slow)) delivers the flat-minima benefits that post-hoc SWA (PRs #3644, #4089) could not — because SWA needs a stationary tail window, while Lookahead works continuously during descent.

OOD splits drove the gain: val_geom_camber_cruise −6.12, val_re_rand −3.96, test_re_rand −4.63. In-distribution barely moved (test_single_in_dist +0.42). Flat-minima → generalization story confirmed.

### SWA failure modes (now closed)

| Experiment | Failure mode | Why Lookahead avoids it |
|---|---|---|
| PR #3644 SWA + constant-LR tail | LR kick-out: constant tail jumps LR ~25× before basin floor | Lookahead doesn't touch LR schedule |
| PR #4089 SWA tail4 at T_max=17 | Budget-limited: cosine still descending in SWA window (68→62 in last 4ep) | Lookahead accumulates during descent, not after |

SWA-at-T_max=17 avenue is exhausted (both failure modes locked in). Lookahead is the right basin-averaging tool for our trajectory shape.

### Triple-stack decomposition (PR #4032 ablation — still relevant foundation)

| Config | val_avg | Δ |
|---|---|---|
| T_max=17 SwiGLU μ̂ (3-seed) | 63.06 | — |
| T_max=17 + GeGLU (default β2=0.9) | 62.47 | −0.59 |
| T_max=17 + GeGLU + β2=0.95 (triple-stack) | 60.43 | **−2.04** |
| Triple-stack + Lookahead | **57.22** | **−3.21 more** |

Cumulative gain from T_max=17 SwiGLU to Lookahead stack: **−5.84 val** over 4 merged experiments.

### Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null |
| LR=1e-3 under T_max=15 | Divergence |
| Head+embed LR boost (1.5–2.5×) | All null/worse |
| T_max < 17 | Confirmed suboptimal (PyTorch Gotcha #3) |
| RMSNorm (vs LayerNorm) | −5.18 val regression |
| slice_num=128 (2× attention) | −10.92 val regression |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail | Regression + kick-out (PR #3644) |
| SWA tail4 at T_max=17 cosine | Regression + budget-limited trajectory (PR #4089) |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds (#3934)
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()` (#3993)
3. **T_max must equal total_epochs** — T_max<total causes hard-zero LR before training ends (#3994)

## Next research directions

### Priority 1 (Lookahead seed confirmation + k sweep)

Lookahead val=57.22 is single-seed. σ̂ unknown. Win threshold wobbles without canonical.
- **seed=1, seed=2** — establish σ̂ under Lookahead, confirm gain is not a lucky draw
- **k sweep (k∈{3,8})** — characterize sync-frequency optimum. k=5 was first-pass default; k=3 adds more averaging, k=8 adds more exploration.

### Priority 2 (Lookahead hyperparameter scan)

- **α sweep (α∈{0.3, 0.7})**: slow-step size. α=0.5 is the paper default; α=0.3 is more conservative, α=0.7 is more aggressive.
- **Lookahead-AdamW + SWA of slow weights**: once Lookahead slow trajectory exists, the slow-weight trajectory itself may have a flatter tail amenable to post-hoc averaging (orthogonal to PRs #3644/#4089 which averaged fast weights).

### Priority 3 (stacking in-flight results onto Lookahead)

In-flight PRs (#4117–#4124) were assigned against triple-stack baseline. Results that beat triple-stack (60.43) but not Lookahead (57.22) should still be stacked onto Lookahead in a follow-up experiment — the levers may be orthogonal.
- Candidates: β1=0.95 (askeladd), β2 scan (frieren), mlp_ratio=3 (fern), EMA (tanjiro), Lion (edward)
- If any in-flight PR beats 57.22 directly, merge immediately.

### Priority 4 (architecture and loss on Lookahead baseline)

- **Physics-informed loss**: continuity equation soft constraint λ·|∇·u|² on predicted (Ux, Uy)
- **mlp_ratio=3 + Lookahead stack** (after fern #4124 result in)
- **SAM (sharpness-aware)**: ρ=0.05, 2× compute — budget-limited to ~8-9ep (gotcha #3 risk), may not be feasible
- **Cross-slice attention**: between-slice heads in Transolver
- **Lookahead + larger model (h=192)**: retest advisor-default h=192 with Lookahead (may now fit in budget if Lookahead helps convergence speed)
