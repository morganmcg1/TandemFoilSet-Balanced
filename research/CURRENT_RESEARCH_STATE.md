# SENPAI Research State

- **Date:** 2026-05-16 (updated 10:10 — **PR #3316 MERGED**: new canonical val=54.494 test=52.837; Huber beta=0.5 now default)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 54.494`, `test_avg/mae_surf_p (excl cruise) = 52.837`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)** + SOAP precond_freq=5 (PR #3495, −1.78%) + **EMA(0.999) (PR #3430, −18.8%)** + EMA decay=0.99 (PR #3591, −3.85%) + **Huber beta=0.5 (PR #3316, −6.05%)**
  - Full stack config: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.5**

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Merged winners (cumulative)

| PR | Student | Hypothesis | Δ vs previous canonical | Cumulative val |
|---|---|---|---|---|
| #3147 | askeladd | LR warmup + peak 1e-3 | **−8.9%** | 123.20 |
| #3155 | fern | Huber loss (beta=1.0) | **−18.1%** | 110.83 |
| #3283 | alphonse | SOAP optimizer | **−31.7%** | 75.70 |
| #3430 | nezuko | EMA of model weights (decay=0.999) | **−18.8%** | 61.43 |
| #3495 | askeladd | SOAP precond_freq=5 | **−1.78%** | 60.33 |
| #3591 | nezuko | EMA decay=0.99 | **−3.85%** | 58.005 |
| **#3316** | **fern** | **Huber beta=0.5** | **−6.05%** | **54.494** |

Old launch baseline: 135.30. Total gain: **−59.7%** over 7 compounding improvements.

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

## Active WIP experiments (all on EMA+SOAP+Huber0.5 canonical, target <54.494)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3728** | **nezuko** | **EMA decay lower sweep {0.97, 0.95} vs 0.99** | **Training** | **WIP — notified of new canonical (54.494)** |
| **#3493** | **alphonse** | **SOAP LR sweep on full canonical stack** | **Optimization** | **WIP — notified of new canonical** |
| **#3703** | **askeladd** | **SOAP precond_freq finer {3, 2} vs freq=5** | **Optimization** | **WIP — notified of new canonical** |
| **#3497** | **tanjiro** | **Grad-clip {no, 5, 10} — BIGGEST signal (−12.1% within-PR)** | **Optimization** | **WIP — notified of new canonical** |
| **#3736** | **thorfinn** | **surf_weight finer sweep {10,5,3} on full canonical** | **Optimization** | **WIP — notified of new canonical** |
| **#3415** | **frieren** | **Log-Re freqs=4 on Huber0.5+EMA+SOAP (rebase)** | **Inputs** | **WIP — sent back; within-PR −2.18% → expect ~53.3 val** |
| **#3868** | **fern** | **Huber beta finer sweep {0.5, 0.25, 0.1}** | **Loss tuning** | **WIP (new — monotone confirms lower is better)** |
| **#3612** | **edward** | **Cauchy c=1.0 vs Huber0.5 on full canonical** | **Loss tuning** | **WIP — needs rebase** |

Zero idle students.

## Key learnings (cumulative)

1. **EMA of model weights + SOAP compound.** EMA provides ~19% additional gain on top of SOAP. Mechanism: SOAP finds flat minima via curvature-aware steps; EMA averages across the flat basin.
2. **Optimizer is the dominant single lever.** SOAP −31.7%, EMA −18.8%, Huber(beta=1.0) −18.1%, Huber(beta=0.5) −6.05%, LR warmup −8.9%.
3. **Huber beta: monotone, smaller is better.** {2.0, 1.0, 0.5} showed strict monotone improvement. beta=0.5 now canonical; exploring {0.1, 0.25} next.
4. **Capacity scaling blocked.** Width/depth/MLP-ratio all fail under 30-min cap. Family is closed.
5. **SOAP generalizes strongly on OOD.** Largest gains on re_rand and geom_camber_rc splits.
6. **Channel upweighting dead end.** p×3 upweight fails on both MSE and SOAP stacks.
7. **AoA reflection augmentation inapplicable.** Camber breaks z-symmetry.
8. **Log-Re sinusoidal confirmed (within-PR).** freqs=4 gives −2.18% val vs paired baseline. Waiting for rebase onto Huber0.5 canonical to quantify full compound.

## Next directions (priority order)

### Immediate (active)
- **Huber beta finer sweep (#3868, fern).** Monotone confirmed — expect optimum at or below 0.25.
- **Tanjiro grad-clip rebase (#3497).** BIGGEST within-PR signal (−12.1% on SOAP-only). Must verify on new canonical.
- **Frieren log-Re rebase (#3415).** Within-PR −2.18% should compound to ~53.3 on Huber0.5 stack.
- **EMA decay lower sweep (#3728, nezuko).** Does 0.95-0.97 beat 0.99?

### Post-sweep stack
1. Combine best SOAP LR + precond_freq + grad_clip + surf_weight winners with full canonical.
2. Cauchy/Welsch vs Huber0.5 (edward #3612).
3. Combine log-Re sinusoidal + Huber0.5 (frieren #3415 rebase).

### Further-future
1. Ada-Temp slice reparameterization (per-point softmax temperature).
2. SWA (Stochastic Weight Averaging) — complementary to EMA but cycle-LR based.
3. Divergence-free auxiliary loss (incompressibility penalty).
4. Physical-units scale-aware loss (edward's round-1 suggestion).
