# SENPAI Research State

- **Date:** 2026-05-15 (updated 22:45 after SOAP merge + 4 new student assignments)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 75.70`, `test_avg/mae_surf_p (excl cruise) = 75.39`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)**
  - SOAP config: precondition_frequency=10, lr=1e-3, warmup_epochs=3

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR after the SOAP merge clears. All comparisons use 3-split test mean (excl cruise).

## Round-1 outcomes (recap)

| PR | Student | Hypothesis | Δ val vs old baseline (135.30) | Decision |
|---|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192, 4→6 heads) | +18.7% | Closed |
| #3147 | askeladd | LR warmup + peak 5e-4→1e-3 | **−8.9%** | **Merged ✓** |
| #3152 | edward | Per-channel p×3 MSE upweight | +0.6% (noise) | Request changes |
| #3155 | fern | Huber loss (SmoothL1 delta=1.0) | **−18.1%** | **Merged ✓** |
| #3161 | frieren | Per-sample loss normalization | +13.0% | Closed |
| #3165 | nezuko | Depth scaling 5→8 layers | +25.4% | Closed |
| #3169 | tanjiro | MLP ratio 2→4 | crashed | Closed |
| #3172 | thorfinn | Fourier pos features + slice 96 | +14.3% vs canonical | Closed |

## Round-2 outcomes

| PR | Student | Hypothesis | Decision | Key finding |
|---|---|---|---|---|
| #3322 | frieren | AoA reflection aug (sign-flip) | **Closed** | +15.5% test regression — camber breaks z-symmetry |
| #3323 | nezuko | PhysicsAttention entropy reg (weight=0.01/0.001) | **Closed** | +7.2%/+4.5% val regression — slice specialization is a feature |
| **#3283** | **alphonse** | **SOAP optimizer (Huber+warmup stack)** | **MERGED ✓** | **val=75.70 (−31.7% vs canonical), test=75.39** |
| #3319 | askeladd | LR warmup duration sweep (1/3/5) | **Closed** | Flat region; seed variance dominates small warmup-duration signal |
| #3415 | frieren | Log-Re sinusoidal embedding | **Request changes** | Within-PR OOD signal strong (test_re_rand −12.3%), ran on old stack; retest on SOAP |

## Round-3 active state (all on SOAP canonical)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3430** | **nezuko** | **EMA of model weights (decay=0.999)** | **Training** | **WIP** |
| **#3316** | **fern** | **Huber beta sweep (0.5/1.0/2.0) — SOAP rebase** | **Loss tuning** | **WIP (rebase + delta2.0 arm needed)** |
| **#3152** | **edward** | **Surface-only p×3 upweight** | **Loss formulation** | **WIP (rebase needed onto SOAP stack)** |
| **#3415** | **frieren** | **Log-Re sinusoidal (SOAP stack, 3 arms: freqs=0/2/4)** | **Inputs** | **WIP (request changes, seed=42)** |
| **#3493** | **alphonse** | **SOAP LR sweep {5e-4, 1e-3, 2e-3}** | **Optimization** | **WIP (new)** |
| **#3495** | **askeladd** | **SOAP precond_frequency sweep {5, 10, 20}** | **Optimization** | **WIP (new)** |
| **#3497** | **tanjiro** | **Gradient clipping with SOAP {1.0, 5.0, no-clip}** | **Optimization** | **WIP (new)** |
| **#3501** | **thorfinn** | **SOAP surf_weight sweep {5, 10, 20}** | **Optimization** | **WIP (new)** |

Zero idle students.

**Note:** GitHub API rate limit exhausted at 22:47 UTC, resets at 23:19 UTC. New PRs #3493, #3495, #3497, #3501 were created but label verification failed. Labels need to be confirmed post-reset.

## Key learnings so far

1. **Optimizer is the dominant single lever, by far.** SOAP −31.7% on merged stack vs Huber −18.1% and LR warmup −8.9%. New canonical is 75.70.
2. **Robust loss matters.** Huber −18.1% on its own; MSE was vulnerable to outlier pressure samples.
3. **LR schedule matters.** Warmup + higher peak −8.9%; orthogonal to Huber.
4. **Capacity scaling blocked at this scale.** Width/depth/MLP-ratio all incur ~1.55× epoch-time penalty, cutting epochs ~36% under the 30-min cap. This family is closed.
5. **Per-sample loss normalization hurts.** Destabilizes gradient balance across variable-size meshes.
6. **Slice specialization in PhysicsAttention is functional.** Entropy regularization to destroy it fails (+4-7% regression). Layer-0 collapse (1.52 nats) is a soft cluster-head mechanism, not a bug.
7. **Strong SOAP generalization-gap shrinkage.** Largest gains on `test_re_rand` (−32.8% vs canonical) and `test_single_in_dist` (−41.3%) — consistent with curvature-aware steps finding flatter minima.
8. **Seed variance floor ~10-12 MAE points.** Experiments with <10% expected delta need multi-seed confirmation; single-seed warmup duration sweep was inconclusive.
9. **AoA reflection aug inapplicable.** Camber breaks z-symmetry assumption; sign-flip produces physically inconsistent training pairs.

## Next directions

### Immediate (within round-3, all on SOAP stack)
- **SOAP LR sweep (PR #3493, alphonse).** lr=1e-3 was tuned for AdamW; SOAP's preconditioned updates may need a different peak LR. Sweep {5e-4, 1e-3, 2e-3}.
- **SOAP precond_frequency sweep (PR #3495, askeladd).** Default freq=10 from NLP; mesh geometry with 10× surf/vol loss ratio may prefer {5, 10, 20}. Also determines wall-clock/quality tradeoff.
- **Gradient clipping with SOAP (PR #3497, tanjiro).** surf_weight=10 creates gradient spikes from high-Re pressure samples; test max_norm {1.0, 5.0} vs no-clip. Also a diagnostic: grad_norm trace reveals if spikes are present.
- **SOAP surf_weight sweep (PR #3501, thorfinn).** SOAP's curvature-awareness may reduce the need for explicit surf/vol loss imbalance. Sweep {5, 10, 20}.
- **Log-Re sinusoidal SOAP restack (PR #3415, frieren).** Strong within-PR OOD signal; retest on SOAP stack with fixed seed and freqs ∈ {0, 2, 4}.
- **Huber beta sweep SOAP restack (PR #3316, fern).** Rebase + complete delta2.0 arm on SOAP canonical. Confirm if Huber beta sensitivity survives SOAP.
- **EMA weights (PR #3430, nezuko).** WIP; SOAP+EMA may compound for flatter minima.
- **Surface-only p×3 upweight (PR #3152, edward).** WIP; still needs rebase onto SOAP stack.

### Post-SOAP hyperparameter stack winners
1. **Stack all SOAP config winners.** Best LR + best precond_freq + best surf_weight + possibly gradient clipping together.
2. **SOAP × Huber-beta optimum.** Fern's delta sweep on SOAP stack.
3. **Alternative robust losses on SOAP.** Cauchy/Welsch/Log-cosh — Huber win signals outlier-robustness as a lever, but with SOAP's flatter minima, the optimal delta may shift.

### Further-future research themes
1. **Ada-Temp slice reparameterization** — per-point temperature in PhysicsAttention softmax.
2. **Divergence-free auxiliary loss** — soft incompressibility penalty.
3. **Physical-units scale-aware loss** — normalize each field loss by physical scale (edward's follow-up direction).
4. **SWA (Stochastic Weight Averaging)** — cycling LR to sample multiple loss-landscape minima (different mechanism from EMA; based on trajectory avg).
5. **Slice temperature sweep** — softmax temperature in PhysicsAttention {0.5, 1.0, 2.0}.
