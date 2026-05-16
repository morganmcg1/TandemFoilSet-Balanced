# SENPAI Research State

- **Date:** 2026-05-16 (updated 19:22 — #3497 tanjiro MERGED (grad_clip=1.0, new canonical val=47.1000); #4099 tanjiro assigned (clip lower bound); all 8 students notified)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 47.1000`, `test_avg/mae_surf_p (excl cruise) = 46.2590`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + **Huber beta=0.1 (PR #3868)** + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)**
  - Full stack: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.1**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split`. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Tracked config issue: precondition_frequency default

`train.py` default is `precondition_frequency=10` but canonical uses `precondition_frequency=5`. Always pass `--precondition_frequency 5` explicitly. Fixed in BASELINE.md.

## Tracked hardware drift

~1.7 val drift between GPU machines on identical config/seed (SOAP eigendecomposition non-determinism). Within-PR relative deltas are reliable; absolute BASELINE.md numbers are reference targets. Tanjiro's baseline-noclip arm reproduced canonical 48.4191 exactly — confirming determinism on that hardware.

## Merged winners (cumulative)

| PR | Student | Hypothesis | Δ vs previous canonical | Cumulative val |
|---|---|---|---|---|
| #3147 | askeladd | LR warmup + peak 1e-3 | **−8.9%** | 123.20 |
| #3155 | fern | Huber loss (beta=1.0) | **−18.1%** | 110.83 |
| #3283 | alphonse | SOAP optimizer | **−31.7%** | 75.70 |
| #3430 | nezuko | EMA of model weights (decay=0.999) | **−18.8%** | 61.43 |
| #3495 | askeladd | SOAP precond_freq=5 | **−1.78%** | 60.33 |
| #3591 | nezuko | EMA decay=0.99 | **−3.85%** | 58.005 |
| #3316 | fern | Huber beta=0.5 | **−6.05%** | 54.494 |
| #3612 | edward | Cauchy loss c=1.0 | **−3.67%** | 52.494 |
| #3868 | fern | Huber beta=0.1 | **−3.77%** | 50.5133 |
| #3947 | alphonse | Lookahead k=5 on SOAP (freq=5) | **−4.14%** | 48.4191 |
| **#3497** | **tanjiro** | **grad_clip=1.0 on Lookahead canonical** | **−2.72%** | **47.1000** |

Old launch baseline: 135.30. Total gain: **−65.2%** over 11 compounding improvements.

## Closed hypotheses (complete)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192) | +18.7% — wall-clock penalty |
| #3161 | frieren | Per-sample loss normalization | +13.0% |
| #3165 | nezuko | Depth scaling (5→8 layers) | +25.4% — wall-clock penalty |
| #3169 | tanjiro | MLP ratio 2→4 | crashed |
| #3172 | thorfinn | Fourier (x,z) + slice_num=96 | +14.3% — dead end |
| #3319 | askeladd | LR warmup duration sweep | flat region |
| #3322 | frieren | AoA reflection aug | +15.5% |
| #3323 | nezuko | Entropy reg (PhysicsAttn) | +4-7% |
| #3152 | edward | p×3 surface upweight | regressed on SOAP |
| #3703 | askeladd | SOAP precond_freq {3,2} vs 5 | U-shape, closed |
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 | worse, closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw, closed |
| #3728 | nezuko | EMA decay lower sweep {0.97, 0.95} | Monotone worse, closed |

## Active WIP experiments (target: val < 47.1000, full canonical stack)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) + Lookahead + grad_clip** | **Inputs** | **WIP — rebased, training. New canonical notified.** |
| **#4070** | **alphonse** | **Lookahead alpha sweep: α ∈ {0.3, 0.5, 0.7}, k=5 fixed** | **Optimization** | **WIP — training.** |
| **#4037** | **fern** | **Huber beta lower bound: {0.05, 0.025, 0.01} vs β=0.1** | **Loss tuning** | **WIP — rebased on Lookahead canonical; needs grad_clip rebase.** |
| **#4021** | **nezuko** | **SWA: uniform late-epoch averaging on top of EMA+Lookahead** | **Training** | **WIP — training with SWA code.** |
| **#4099** | **tanjiro** | **Grad-clip lower bound: {0.5, 0.1} vs clip=1.0** | **Optimization** | **WIP — just assigned (follow-up to merged #3497).** |
| **#3975** | **askeladd** | **bfloat16 rerun on Lookahead+grad_clip canonical** | **Throughput** | **WIP — needs rebase onto new grad_clip canonical.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1) on new canonical** | **Loss tuning** | **WIP — rebased; needs grad_clip canonical.** |
| **#3736** | **thorfinn** | **surf_weight {10,5} rerun on Lookahead canonical** | **Loss weighting** | **WIP — training (lost time to GH rate limit earlier).** |

Zero idle students.

## Key learnings (cumulative)

1. **grad_clip=1.0 compounds with Lookahead — −2.72% val.** Huber β=0.1 L1-dominant gradients have explosive dynamic range (p50=112 vs Cauchy p50=17). clip=1.0 active on 100% of steps. SOAP preconditioner is direction-sensitive — global magnitude rescaling doesn't destroy signal.
2. **Lookahead k=5 compounds — −4.14% val.** k=5 aligns with precondition_frequency=5, slow weights absorb stale-curvature noise between SOAP refreshes.
3. **Huber β=0.1 beats Cauchy c=1.0 by −3.77%.** L1-dominant loss produces noisier gradients that benefit more from both Lookahead and grad_clip.
4. **EMA decay=0.99 is global optimum.** Lower decay monotonically hurts.
5. **SOAP preconditioner is direction-sensitive.** Both Lookahead (slow-weight sync) and grad_clip (magnitude normalization) add value on top of SOAP's curvature adaptation.
6. **Log-Re sinusoidal embedding: −1.20% within-PR on Huber β=0.1 stack.** Input-side, orthogonal to all optimizer/loss changes. Awaiting full Lookahead+clip rerun.
7. **bf16: 1.30× throughput, −22% VRAM, quality-neutral at matched epochs.** Once measured on new canonical, unlocks more effective training for all future runs.
8. **Wall-clock cap is binding.** Best epoch = 14. bf16 gives 17 epochs in same 30-min cap.
9. **GitHub API rate limit (~5000/hr shared).** Causes brief pod stalls when rate is exhausted. Self-recovering within ~5 min.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re + full canonical (#3415).** −1.20% within-PR on older stack. Input-side, orthogonal. **Highest-EV pending result** — expected val ≈ 46.5 if compounding holds.
- **Fern Huber β lower bound (#4037).** β={0.05, 0.025, 0.01} — if monotone trend continues to β=0.01, pure MAE next.
- **Alphonse Lookahead alpha sweep (#4070).** α ∈ {0.3, 0.5, 0.7} — tuning merged #3947.
- **Nezuko SWA (#4021).** Epoch-scale averaging on top of EMA+Lookahead+clip.
- **Tanjiro grad-clip lower bound (#4099).** {0.5, 0.1} — does aggressive clip continue to improve?
- **Thorfinn surf_weight (#3736).** sw=5 expected to beat sw=10 on full canonical.
- **Askeladd bf16 (#3975).** Needs rebase onto grad_clip canonical. After merge: unlocks wider/deeper Transolver.
- **Edward log-pressure (#3952).** Needs rebase onto grad_clip canonical. Modest within-PR signal, testing if it compounds.

### Post-sweep stack
1. **Clip lower bound** — if clip=0.1 still wins, try normalized-gradient optimizers (sign-SGD-with-momentum, NGD).
2. **Pure MAE (L1 loss):** if Huber β=0.01 still improves.
3. **AGC (adaptive gradient clipping per-parameter):** scale-invariant alternative.
4. **SAM on SOAP:** only viable after bf16 (more epochs).
5. **Wider Transolver (n_hidden=192):** only after bf16 confirms VRAM headroom.
6. **Batch size sweep:** 33 GB VRAM headroom after bf16 → try batch_size=6 or 8.
