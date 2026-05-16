# SENPAI Research State

- **Date:** 2026-05-16 (updated 12:05 — PR #3612 MERGED: Cauchy c=1.0 new canonical val=52.494; PR #3493 closed; alphonse #3947 Lookahead-SOAP assigned; edward #3952 log-pressure assigned)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 52.494`, `test_avg/mae_surf_p (excl cruise) = 51.220`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)** + SOAP precond_freq=5 (PR #3495, −1.78%) + **EMA(0.999) (PR #3430, −18.8%)** + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + **Cauchy c=1.0 (PR #3612, −3.67%)**
  - Full stack config: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **cauchy_c=1.0**

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
| #3316 | fern | Huber beta=0.5 | **−6.05%** | 54.494 |
| **#3612** | **edward** | **Cauchy loss c=1.0** | **−3.67%** | **52.494** |

Old launch baseline: 135.30. Total gain: **−61.2%** over 8 compounding improvements.

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
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 vs 1e-3 | lr=2e-3 +1.0% worse; lr=1e-3 stays canonical — closed |

## Active WIP experiments (all on full Cauchy+EMA+SOAP canonical, target <52.494)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3868** | **fern** | **Huber beta finer sweep {0.5, 0.25, 0.1}** | **Loss tuning** | **WIP — arm 1 running (Huber stack, result may not beat new Cauchy canonical)** |
| **#3497** | **tanjiro** | **Grad-clip {no, 5, 10} — BIGGEST signal (−12.1% within-PR)** | **Optimization** | **WIP — stale, needs rebase onto Cauchy canonical** |
| **#3415** | **frieren** | **Log-Re freqs=4 — rebase onto Cauchy canonical** | **Inputs** | **WIP — within-PR −2.18%, rebase in progress** |
| **#3728** | **nezuko** | **EMA decay lower sweep {0.97, 0.95} vs 0.99** | **Training** | **WIP — notified of new canonical** |
| **#3736** | **thorfinn** | **surf_weight finer sweep {10,5,3} on full canonical** | **Optimization** | **WIP — rebased, arms running** |
| **#3926** | **askeladd** | **Cosine LR floor (eta_min ∈ {0, 1e-5, 1e-4})** | **Optimization** | **WIP — just assigned, implementing** |
| **#3947** | **alphonse** | **Lookahead wrapper on SOAP (k=5 vs k=10)** | **Optimization** | **WIP — just assigned** |
| **#3952** | **edward** | **Log-pressure aux loss (weight 0.05 vs 0.1)** | **Loss tuning** | **WIP — just assigned** |

Zero idle students.

## Key learnings (cumulative)

1. **Cauchy loss beats Huber across all stacks.** Redescending influence function downweights outliers more aggressively than Huber's linear tail. Compounded with EMA+SOAP+Huber β reduction: −6.46% within-group, −3.67% vs Huber β=0.5 canonical.
2. **EMA of model weights + SOAP compound.** EMA provides ~19% additional gain on top of SOAP. Mechanism: SOAP finds flat minima via curvature-aware steps; EMA averages across the flat basin.
3. **Optimizer is the dominant single lever.** SOAP −31.7%, EMA −18.8%, Huber(beta=1.0) −18.1%, LR warmup −8.9%.
4. **Huber beta: monotone, smaller is better.** {2.0, 1.0, 0.5} showed strict monotone improvement. beta=0.5 canonical; Cauchy c=1.0 is now the loss function.
5. **SOAP precond_freq=5 is optimal (U-shape).** Going more frequent (freq=3, freq=2) injects preconditioner-frame noise, causing +5-7% regressions.
6. **Capacity scaling blocked.** Width/depth/MLP-ratio all fail under 30-min cap. Family is closed.
7. **LR=1e-3 is optimal for SOAP.** lr=2e-3 loses by +1.0% val; lr=5e-4 untested but deprioritized.
8. **Log-Re sinusoidal confirmed (within-PR).** freqs=4 gives −2.18% val vs paired baseline. Rebase in progress.
9. **Wall-clock cap is binding.** Best epoch consistently = 14 out of ~50. LR decays to near-zero by epoch 12-13. Cosine LR floor (eta_min) under test.

## Next directions (priority order)

### Immediate (active)
- **Tanjiro grad-clip rebase (#3497).** BIGGEST within-PR signal (−12.1% on SOAP-only). Must verify on Cauchy canonical.
- **Frieren log-Re rebase (#3415).** Within-PR −2.18% should compound to ~52 on full canonical.
- **Fern Huber beta finer sweep (#3868).** Note: Huber β sweeps on the Huber stack — may not beat Cauchy canonical. Result still informative for understanding loss landscape.
- **Askeladd cosine LR floor (#3926).** Near-zero-risk 2-line change; ~15% of epochs wasted at lr≈0.
- **Alphonse Lookahead on SOAP (#3947).** Slow-weight sync targeting SOAP preconditioner noise; orthogonal to EMA.
- **Edward log-pressure aux loss (#3952).** Relative-error pressure loss; targets OOD Re generalization.

### Post-sweep stack
1. Adaptive Barron loss (Barron 2019): learn α and c per-output-channel (Ux/Uy vs p). Removes loss hyperparameter and targets channel-specific tail behavior.
2. SAM on SOAP: sharpness-aware minimization for OOD generalization. High expected gain (2–4% OOD) but doubles forward cost.
3. bf16 autocast: free epoch budget increase within wall-clock cap. Diagnostic value — tells us if wall-clock is the binding constraint.
4. SWA (Stochastic Weight Averaging): complementary to EMA but cycle-LR based.
5. Divergence-free auxiliary loss (incompressibility penalty on velocity field).
