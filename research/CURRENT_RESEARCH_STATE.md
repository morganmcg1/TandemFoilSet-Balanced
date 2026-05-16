# SENPAI Research State

- **Date:** 2026-05-16 (updated 15:40 — #3736 thorfinn sent back (surf_weight rerun on Huber β=0.1); #3728 nezuko CLOSED (EMA decay=0.99 is optimal floor); nezuko assigned #4021 SWA)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 50.5133`, `test_avg/mae_surf_p (excl cruise) = 49.8493`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)** + SOAP precond_freq=5 (PR #3495, −1.78%) + **EMA(0.999) (PR #3430, −18.8%)** + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy c=1.0 (PR #3612, −3.67%) + **Huber beta=0.1 (PR #3868, −3.77%)**
  - Full stack config: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.1 (cauchy_c=0.0)**

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Tracked config issue: precondition_frequency default

`train.py` default is `precondition_frequency=10` but canonical uses `precondition_frequency=5`. Always pass `--precondition_frequency 5` explicitly. Fixed in BASELINE.md reproduce commands as of 2026-05-16 14:55.

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
| **#3868** | **fern** | **Huber beta=0.1** | **−3.77%** | **50.5133** |

Old launch baseline: 135.30. Total gain: **−62.7%** over 9 compounding improvements.

## Pending winner (awaiting rebase onto Huber β=0.1 canonical)

| PR | Student | Hypothesis | Best result so far | Notes |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4)** | val=51.0991, test=50.9922 on Huber β=0.5 stack | **Needs rebase onto Huber β=0.1 canonical. Updated instructions sent.** |

- Expected post-rebase: val ≈ 47.5 if compounding holds (log-Re is input-side, orthogonal to loss)

## Closed hypotheses (complete)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192) | +18.7% — wall-clock penalty |
| #3161 | frieren | Per-sample loss normalization | +13.0% — destabilizes gradients |
| #3165 | nezuko | Depth scaling (5→8 layers) | +25.4% — wall-clock penalty |
| #3169 | tanjiro | MLP ratio 2→4 | crashed — wall-clock penalty |
| #3172 | thorfinn | Fourier (x,z) + slice_num=96 | +14.3% consistently — dead end |
| #3319 | askeladd | LR warmup duration sweep | flat region, seed variance > signal |
| #3322 | frieren | AoA reflection aug | +15.5% — camber breaks symmetry |
| #3323 | nezuko | Entropy reg (PhysicsAttn) | +4-7% — slice specialization is functional |
| #3152 | edward | p×3 surface upweight | regressed on SOAP stack |
| #3703 | askeladd | SOAP precond_freq {3,2} vs 5 | U-shape; freq=3 +5.5%, freq=2 +6.6% worse — closed |
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 vs 1e-3 | lr=2e-3 +1.0% worse — closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw: T_max=47, LR never reaches floor. Student caught pre-launch. |
| **#3728** | **nezuko** | **EMA decay lower sweep {0.97, 0.95}** | **Monotone worse: 0.99 is global optimum. 0.97: +4.5%, 0.95: +8.2%. Clean negative.** |

## Active WIP experiments (all on Huber β=0.1+EMA+SOAP canonical, target <50.5133)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) — rebase onto Huber β=0.1** | **Inputs** | **WIP — updated rebase instructions sent** |
| **#3947** | **alphonse** | **Lookahead k=5 rerun with freq=5 + huber_beta=0.1** | **Optimization** | **WIP — 2-arm rerun on new canonical** |
| **#4010** | **fern** | **Huber beta lower bound: {0.05, 0.025, 0.01} vs β=0.1** | **Loss tuning** | **WIP — running 4-arm sweep** |
| **#4021** | **nezuko** | **SWA: uniform late-epoch averaging on top of EMA** | **Training** | **WIP — just assigned** |
| **#3736** | **thorfinn** | **surf_weight {10,5} rerun on Huber β=0.1 canonical** | **Optimization** | **WIP — sent back, 2-arm rerun on new canonical** |
| **#3497** | **tanjiro** | **Grad-clip {no, 5, 10} — BIGGEST within-PR signal (−12.1%)** | **Optimization** | **WIP — running on Cauchy stack, notified of new canonical** |
| **#3952** | **edward** | **Log-pressure aux loss (weight 0.05 vs 0.1)** | **Loss tuning** | **WIP — 3 arms on Cauchy stack** |
| **#3975** | **askeladd** | **bfloat16 autocast: more epochs in 30-min cap** | **Throughput** | **WIP — implementing** |

Zero idle students.

## Key learnings (cumulative)

1. **Huber β=0.1 beats Cauchy c=1.0 by −3.77%.** Monotone trend extends through full range {2.0, 1.0, 0.5, 0.25, 0.1}. Pure L1 regime (β→0) may be optimal — next sweep tests β={0.05, 0.025, 0.01}.
2. **Cauchy loss was superseded.** Huber β=0.1 outperforms Cauchy c=1.0 — possibly because SOAP's adaptive preconditioning handles curvature, making Cauchy's redescending influence redundant.
3. **Log-Re sinusoidal embedding works (within-PR).** freqs=4 gives −6.23% val on Huber β=0.5 stack. Awaiting rebase onto Huber β=0.1 canonical.
4. **EMA decay=0.99 is global optimum (confirmed).** Lower decay (0.97, 0.95) monotonically hurts. Upper (0.999) also hurts. 0.99 ≈ 100-step EMA horizon is optimal for our 14-epoch budget.
5. **surf_weight=5 beats sw=10 within-PR (−1.31%) on Cauchy stack.** Awaiting confirmation on Huber β=0.1 canonical. Mechanism: SOAP preconditioner already balances scale, sw=10 over-weights surface loss.
6. **SOAP precond_freq=5 is optimal. Default in train.py is 10 — always pass `--precondition_frequency 5` explicitly.**
7. **Lookahead k=5 shows real within-PR signal (−4.81%) but ran with freq=10.** Rerunning on canonical stack.
8. **Wall-clock cap is binding.** Best epoch consistently = 14. CosineAnnealingLR uses T_max=47, LR only at 87% of peak at cap. bf16 is primary lever to increase effective epoch count.
9. **Cosine LR floor (eta_min) doesn't apply.** With T_max=47 and ~14-epoch cap, LR never decays near zero.
10. **SOAP + small Huber β handles noisier gradient signal.** SOAP's adaptive preconditioning appears to handle the L1-dominant gradient noise from small β robustly — the gradient direction is still useful even if magnitudes vary.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re + Huber β=0.1 rebase (#3415).** Strong signal (−6.23% within-PR). Expected val ≈ 47.5 if compounding holds. **Highest-EV pending result.**
- **Fern Huber β lower bound (#4010).** Maps remaining L1-floor headroom. If β=0.01 still wins, next step is pure MAE.
- **Tanjiro grad-clip rebase (#3497).** −12.1% within-PR on SOAP-only. Must now beat 50.5133.
- **Alphonse Lookahead rerun (#3947).** Clean within-PR signal. Rerunning with freq=5 + Huber β=0.1.
- **Thorfinn surf_weight rerun (#3736).** sw=5 expected to beat sw=10 on new canonical; could be a small win.
- **Nezuko SWA (#4021).** First test of SWA on top of EMA. Expected 1-3% if flat-minima mechanism holds.
- **Askeladd bf16 autocast (#3975).** Throughput diagnostic — if ≥1.2x, unlocks more effective training.

### Post-sweep stack (after pending WIPs land)
1. Pure MAE (L1 loss): if Huber β=0.01 still improves, switch to `nn.L1Loss` directly.
2. Adaptive Barron loss: learn α and c per-output-channel. Removes loss hyperparameter.
3. SAM on SOAP: sharpness-aware minimization. Doubles forward cost — only viable after bf16 (more epochs).
4. Divergence-free auxiliary loss (incompressibility penalty on velocity field).
5. Longer schedule: if bf16 gives more epochs, retry experiments with 20+ effective epochs.
