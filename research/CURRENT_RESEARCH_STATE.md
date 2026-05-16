# SENPAI Research State

- **Date:** 2026-05-16 (updated 13:30 — PR #3926 closed (design flaw, cosine LR); #3415 frieren winner on Huber stack, sent back for Cauchy rebase; #3975 askeladd bf16 assigned)
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

## Pending winner (awaiting Cauchy-stack rebase)

| PR | Student | Hypothesis | Result on Huber β=0.5 stack | Δ vs that canonical |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal embedding (freqs=4)** | val=51.0991, test=50.9922 | **−6.23% val within-PR** |

- Already beats Cauchy canonical (52.494, 51.220) by −2.66% val / −0.44% test
- Needs rebase onto current advisor branch + 2-arm confirmation (Cauchy stack)
- Expected post-rebase on Cauchy: val ≈ 49.2 if fully compounding (input encoding vs loss function are orthogonal mechanisms)

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
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw: T_max=47, only 14 epochs at cap → floor never active. Student caught pre-launch. |

## Active WIP experiments (all on Cauchy+EMA+SOAP canonical, target <52.494)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) + Cauchy rebase** | **Inputs** | **WIP — rebasing, 2-arm confirmation on Cauchy needed** |
| **#3868** | **fern** | **Huber beta finer sweep {0.5, 0.25, 0.1}** | **Loss tuning** | **WIP — Huber stack, result may not beat Cauchy canonical** |
| **#3497** | **tanjiro** | **Grad-clip {no, 5, 10} — BIGGEST within-PR signal (−12.1%)** | **Optimization** | **WIP — stale, nudged, needs rebase onto Cauchy** |
| **#3728** | **nezuko** | **EMA decay lower sweep {0.97, 0.95} vs 0.99** | **Training** | **WIP — notified of new Cauchy canonical** |
| **#3736** | **thorfinn** | **surf_weight finer sweep {10,5,3} on EMA+SOAP+Cauchy canonical** | **Optimization** | **WIP — arms running** |
| **#3947** | **alphonse** | **Lookahead wrapper on SOAP (k=5 vs k=10)** | **Optimization** | **WIP — implementing** |
| **#3952** | **edward** | **Log-pressure aux loss (weight 0.05 vs 0.1)** | **Loss tuning** | **WIP — implementing** |
| **#3975** | **askeladd** | **bfloat16 autocast: more epochs in 30-min cap** | **Throughput** | **WIP — just assigned** |

Zero idle students.

## Key learnings (cumulative)

1. **Cauchy loss beats Huber across all stacks.** Redescending influence function. Compounded with full canonical: −3.67% vs Huber β=0.5.
2. **Log-Re sinusoidal embedding works (within-PR).** freqs=4 gives −6.23% val on Huber β=0.5 stack. Awaiting Cauchy-stack rebase to confirm compound.
3. **EMA of model weights + SOAP compound.** EMA ~19% on top of SOAP.
4. **Optimizer is the dominant single lever.** SOAP −31.7%, EMA −18.8%, Huber(beta=1.0) −18.1%, LR warmup −8.9%.
5. **SOAP precond_freq=5 is optimal (U-shape).** More frequent (freq=3, freq=2) injects frame noise.
6. **Capacity scaling blocked.** Width/depth/MLP-ratio all fail under 30-min cap.
7. **LR=1e-3 is optimal for SOAP.** lr=2e-3 loses by +1.0%.
8. **Wall-clock cap is binding.** Best epoch consistently = 14. CosineAnnealingLR uses T_max=47, so LR is only at 87% of peak at cap — NOT near-zero as initially hypothesized. bf16 autocast is the primary lever to increase effective epoch count.
9. **Cosine LR floor (eta_min) doesn't apply.** With T_max=47 and ~14-epoch cap, LR never decays near zero. Student caught this pre-launch — excellent analytical work.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re + Cauchy rebase (#3415).** Strong signal (−6.23% within-PR). Expected to push val well below 50 if compounding holds.
- **Tanjiro grad-clip rebase (#3497).** −12.1% within-PR on SOAP-only. Highest-EV untested. Stale — needs action.
- **Askeladd bf16 autocast (#3975).** Diagnostic + potential free epochs. If throughput improves ≥1.3x, unlocks more effective training for all future experiments.
- **Fern Huber beta {0.25, 0.1} (#3868).** On Huber stack — may not beat Cauchy canonical; still informative for understanding loss landscape boundaries.

### Post-sweep stack (after pending WIPs land)
1. Adaptive Barron loss: learn α and c per-output-channel. Removes loss hyperparameter.
2. SAM on SOAP: sharpness-aware minimization. OOD gains 2-4%, but doubles forward cost.
3. SWA (Stochastic Weight Averaging): cycle-LR based, complementary to EMA.
4. Divergence-free auxiliary loss (incompressibility penalty on velocity field).
5. Longer schedule: if bf16 gives more epochs, retry experiments with 20+ effective epochs.
