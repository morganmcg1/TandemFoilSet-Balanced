# SENPAI Research State

- **Date:** 2026-05-16 (updated 23:10 — #4099 tanjiro CLOSED (grad-clip lower bound — monotone worse on both stacks); #4200 tanjiro assigned (Lookahead k sweep))
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 45.9199`, `test_avg/mae_surf_p (excl cruise) = 45.1094`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + Huber beta=0.1 (PR #3868) + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)** + **Huber beta=0.01 (PR #4037)**
  - Full stack: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.01**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split`. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Tracked config issue: precondition_frequency default

`train.py` default is `precondition_frequency=10` but canonical uses `precondition_frequency=5`. Always pass `--precondition_frequency 5` explicitly. Fixed in BASELINE.md.

## Tracked hardware drift

~1.7 val drift between GPU machines on identical config/seed (SOAP eigendecomposition non-determinism). Within-PR relative deltas are reliable; absolute BASELINE.md numbers are reference targets.

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
| #3497 | tanjiro | grad_clip=1.0 on Lookahead canonical | **−2.72%** | 47.1000 |
| **#4037** | **fern** | **Huber beta=0.01 (near-L1 regime)** | **−2.51%** | **45.9199** |

Old launch baseline: 135.30. Total gain: **−66.0%** over 12 compounding improvements.

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
| #4021 | nezuko | SWA on EMA+Lookahead+clip | +8.6% — non-plateaued training |
| #4099 | tanjiro | Grad-clip lower bound {0.5, 0.1} | monotone worse on both stacks — 1.0 is sweet spot |
| #3152 | edward | p×3 surface upweight | regressed on SOAP |
| #3703 | askeladd | SOAP precond_freq {3,2} vs 5 | U-shape, closed |
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 | worse, closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw, closed |
| #3728 | nezuko | EMA decay lower sweep {0.97, 0.95} | Monotone worse, closed |

## Active WIP experiments (target: val < 45.9199, full canonical stack)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) on full canonical** | **Inputs** | **WIP — CONFLICTING (needs rebase). New canonical notified.** |
| **#4070** | **alphonse** | **Lookahead alpha sweep: α ∈ {0.3, 0.5, 0.7}, k=5 fixed** | **Optimization** | **WIP — training on β=0.1+grad_clip canonical; new β=0.01 canonical notified.** |
| **#4161** | **nezuko** | **AGC: adaptive gradient clipping (λ=0.01) vs fixed max_norm=1.0** | **Optimization** | **WIP — just assigned.** |
| **#4200** | **tanjiro** | **Lookahead k sweep: {3, 5, 10} with α=0.5 fixed** | **Optimization** | **WIP — just assigned.** |
| **#3975** | **askeladd** | **bfloat16 rerun on full canonical** | **Throughput** | **WIP — training; new β=0.01 canonical notified.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1)** | **Loss tuning** | **WIP — training; new β=0.01 canonical notified.** |
| **#3736** | **thorfinn** | **surf_weight {10,5} rerun on full canonical** | **Loss weighting** | **WIP — training; new β=0.01 canonical notified.** |
| **#4139** | **fern** | **Huber β near-L1 sweep: {0.005, 0.001, 0.0001}** | **Loss tuning** | **WIP — just assigned (follow-up to merged #4037).** |

Zero idle students.

## Key learnings (cumulative)

1. **Huber β=0.01 compounds with grad_clip=1.0 — −2.51% val (12th win).** Near-pure L1 regime (quadratic zone <1% of residuals). Monotone trend: β=0.10→0.05 (−0.60%) →0.025 (−1.90%) →0.01 (−2.51%). β=0.001/0.0001 testing next.
2. **grad_clip=1.0 compounds with Lookahead — −2.72% val.** Huber β=0.1 L1-dominant gradients have explosive dynamic range (p50=112 vs Cauchy p50=17). clip=1.0 active on 100% of steps. SOAP preconditioner is direction-sensitive — global magnitude rescaling doesn't destroy signal.
3. **Lookahead k=5 compounds — −4.14% val.** k=5 aligns with precondition_frequency=5, slow weights absorb stale-curvature noise between SOAP refreshes.
4. **Huber β=0.1 beats Cauchy c=1.0 by −3.77%.** L1-dominant loss produces noisier gradients that benefit more from both Lookahead and grad_clip.
5. **EMA decay=0.99 is global optimum.** Lower decay monotonically hurts.
6. **SOAP preconditioner is direction-sensitive.** Both Lookahead (slow-weight sync) and grad_clip (magnitude normalization) add value on top of SOAP's curvature adaptation.
7. **Log-Re sinusoidal embedding: −1.20% within-PR on Huber β=0.1 stack.** Input-side, orthogonal to all optimizer/loss changes. Awaiting full canonical rerun (frieren #3415).
8. **bf16: 1.30× throughput, −22% VRAM, quality-neutral at matched epochs.** Once measured on new canonical, unlocks more effective training for all future runs.
9. **Wall-clock cap is binding.** Best epoch = 14. bf16 gives 17 epochs in same 30-min cap.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re + full canonical (#3415).** −1.20% within-PR on older stack. Input-side, orthogonal. **Highest-EV pending result** — expected val ≈ 44.8 if compounding holds on new canonical.
- **Fern near-L1 sweep (#4139).** β={0.005, 0.001, 0.0001} — closing the monotone β trend. If still improving, pure MAE is the endpoint.
- **Alphonse Lookahead alpha sweep (#4070).** α ∈ {0.3, 0.5, 0.7} — on β=0.1+grad_clip baseline, needs rerun on β=0.01.
- **Nezuko AGC (#4161).** Adaptive per-parameter gradient clipping (λ=0.01) — natural follow-on to global clip=1.0 success.
- **Tanjiro Lookahead k sweep (#4200).** k={3, 5, 10} — orthogonal to alphonse's α sweep, closes the {k, α} Lookahead hyperparameter space.
- **Thorfinn surf_weight (#3736).** sw=5 expected to beat sw=10 on full canonical.
- **Askeladd bf16 (#3975).** Needs measurement on β=0.01 canonical. After merge: unlocks wider/deeper Transolver.
- **Edward log-pressure (#3952).** Modest within-PR signal, testing if it compounds with β=0.01.

### Post-sweep stack
1. **Pure MAE (L1 loss):** if β=0.0001 still improves — natural endpoint of β family.
2. **AGC (adaptive gradient clipping per-parameter):** scale-invariant alternative to fixed max_norm.
3. **Clip lower bound continuation:** if tanjiro finds {0.5, 0.1} wins, try normalized-gradient optimizers.
4. **SAM on SOAP:** only viable after bf16 (more epochs).
5. **Wider Transolver (n_hidden=192):** only after bf16 confirms VRAM headroom.
6. **Batch size sweep:** 33 GB VRAM headroom after bf16 → try batch_size=6 or 8.
