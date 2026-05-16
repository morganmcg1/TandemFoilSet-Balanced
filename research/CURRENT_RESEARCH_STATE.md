# SENPAI Research State

- **Date:** 2026-05-16 (updated 04:28 — W&B audit found nezuko #3591 variant-decay0.99=58.005, −5.6% vs canonical; awaiting terminal SENPAI-RESULT)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 61.43`, `test_avg/mae_surf_p (excl cruise) = 60.92`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)** + **EMA of model weights decay=0.999 (PR #3430, −18.8%)**
  - Full stack config: SOAP precondition_frequency=10, lr=1e-3, warmup_epochs=3, ema_decay=0.999

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Merged winners (cumulative)

| PR | Student | Hypothesis | Δ vs previous canonical | Cumulative val |
|---|---|---|---|---|
| #3147 | askeladd | LR warmup + peak 1e-3 | **−8.9%** | 123.20 |
| #3155 | fern | Huber loss (beta=1.0) | **−18.1%** | 110.83 |
| #3283 | alphonse | SOAP optimizer | **−31.7%** | 75.70 |
| **#3430** | **nezuko** | **EMA of model weights (decay=0.999)** | **−18.8%** | **61.43** |

Old launch baseline: 135.30. Total gain: −54.6% over 4 compounding improvements.

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

## Active WIP experiments (all on EMA+SOAP canonical, target <61.43)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3591** | **nezuko** | **EMA decay sweep {0.99, 0.999, 0.9999}** | **Training** | **WIP — 2/3 arms done (W&B): variant-decay0.99=58.005 (−5.6%), awaiting arm 3 + SENPAI-RESULT** |
| **#3493** | **alphonse** | **SOAP LR (lr=2e-3 winner) on EMA+SOAP** | **Optimization** | **WIP — rebase + 2-arm compounding test (sent back 02:40, within-PR: −3.2% val on SOAP-only)** |
| **#3495** | **askeladd** | **SOAP precond_freq (freq=5 winner) on EMA+SOAP** | **Optimization** | **WIP — rebase + 2-arm compounding test (sent back 02:40, within-PR: −5.3% val on SOAP-only)** |
| **#3497** | **tanjiro** | **Grad-clip {no, 5, 10} on EMA+SOAP (clip5 winner)** | **Optimization** | **WIP — rebase + 3-arm compounding test (sent back 03:55, within-PR: −12.1% val on SOAP-only — BIGGEST round-3 signal)** |
| **#3501** | **thorfinn** | **SOAP surf_weight sweep {5, 10, 20}** | **Optimization** | **WIP — running SOAP-baseline arms; notify when done** |
| **#3415** | **frieren** | **Log-Re sinusoidal (SOAP stack, seed=42)** | **Inputs** | **WIP — arm1 done (77.88), variants in progress** |
| **#3316** | **fern** | **Huber beta sweep (0.5/1.0/2.0) on SOAP stack** | **Loss tuning** | **WIP — rebased, arms running** |
| **#3612** | **edward** | **Cauchy robust loss sweep (c=0.5, 1.0) on EMA+SOAP** | **Loss tuning** | **WIP (new)** |

Note: PRs #3493, #3495, #3497, #3501 are running against the SOAP-without-EMA baseline. Advisor has notified all 4 students about the new EMA canonical. Within-PR comparison is still valid for identifying the best hyperparameter setting; winners will be asked to rebase onto EMA+SOAP stack.

Zero idle students.

## Key learnings (cumulative)

1. **EMA of model weights + SOAP compound.** EMA provides ~19% additional gain on top of SOAP. Mechanism: SOAP finds flat minima via curvature-aware steps; EMA averages across the flat basin, finding a more central (lower-variance) point. Monotone val curve vs oscillating online weights confirms regularization effect.
2. **Optimizer is the dominant single lever.** SOAP −31.7%, EMA −18.8%, Huber −18.1%, LR warmup −8.9% (in order of impact).
3. **Capacity scaling blocked.** Width/depth/MLP-ratio all fail under 30-min cap. Family is closed.
4. **Slice specialization functional.** Entropy reg to destroy it fails. Layer-0 collapse is a soft cluster-head mechanism.
5. **SOAP generalizes strongly on OOD.** Largest gains on re_rand and geom_camber_rc splits.
6. **Channel upweighting dead end.** p×3 upweight fails on both MSE and SOAP stacks.
7. **AoA reflection augmentation inapplicable.** Camber breaks z-symmetry.

## Next directions (priority order)

### Immediate (active)
- **EMA decay sweep (PR #3591, nezuko).** Find optimal decay in {0.99, 0.999, 0.9999}.
- **Winners from SOAP-stack sweeps** (alphonse/askeladd/tanjiro/thorfinn) → rebase winning arm onto EMA+SOAP stack.
- **Log-Re sinusoidal (PR #3415, frieren).** Does it compound with EMA+SOAP on test_re_rand?
- **Huber beta (PR #3316, fern).** Is beta=1.0 optimal with SOAP? Does EMA change the answer?

### Post-sweep stack
1. Combine best SOAP LR + precond_freq + grad_clip + surf_weight winners with EMA+SOAP.
2. SOAP × Huber-beta re-optimization on EMA+SOAP stack.
3. Alternative robust losses (Cauchy/Welsch) on EMA+SOAP stack.

### Further-future
1. Ada-Temp slice reparameterization (per-point softmax temperature).
2. SWA (Stochastic Weight Averaging) — complementary to EMA but cycle-LR based.
3. Divergence-free auxiliary loss (incompressibility penalty).
4. Physical-units scale-aware loss (edward's round-1 suggestion).
