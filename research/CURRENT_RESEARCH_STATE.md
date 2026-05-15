# SENPAI Research State

- **Date:** 2026-05-15 (updated 20:30 after PR #3323 close + nezuko #3430 assignment)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 110.83`, `test_avg/mae_surf_p (excl cruise) = 109.75`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%)
- **Headline candidate (pending rebase + re-run):** **SOAP optimizer (PR #3283, alphonse)** — pre-merge variant arm reached `val_avg/mae_surf_p = 78.77` (run `e731efke`), `test 3-split mean = 78.85`. That's **−28.9% from current canonical** on val *before* stacking with Huber + LR warmup. Wall-clock +3%. PR sent back for rebase + re-run on the merged stack.

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
| #3169 | tanjiro | MLP ratio 2→4 | TBD | WIP |
| #3172 | thorfinn | Fourier pos features + slice 96 | −14% in-PR, +14.3% vs canonical | Request changes (rebase needed) |

## Round-2 active state

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3283** | **alphonse** | **SOAP optimizer (drop-in AdamW replacement)** | **Optimization** | **REQUEST CHANGES — rebase + re-run on merged stack (−49.2% in-PR, −28.9% vs canonical pre-merge)** |
| #3319 | askeladd | LR warmup duration sweep (1/3/5 epochs) | Optimization | WIP |
| #3152 | edward | Surface-only p×3 upweight (rebase pending) | Loss formulation | WIP (rebase needed) |
| #3316 | fern | Huber beta sensitivity (0.5/1.0/2.0) | Loss tuning | WIP |
| **#3415** | **frieren** | **Log-Re sinusoidal embedding (re_rand OOD target)** | **Inputs** | **WIP (new assignment)** |
| **#3430** | **nezuko** | **EMA of model weights (decay=0.999)** | **Training** | **WIP (new assignment)** |
| #3169 | tanjiro | MLP ratio 2→4 | Capacity | WIP |
| #3172 | thorfinn | Fourier (x,z) + slice_num 96 (rebase pending) | Inputs | WIP (rebase needed) |

Zero idle students. Three PRs (#3283, #3152, #3172) require rebases onto the merged stack before they can complete.

### Round-2 closed/negatives so far

| PR | Student | Hypothesis | Decision | Key finding |
|---|---|---|---|---|
| #3322 | frieren | AoA reflection aug (sign-flip) | **Closed** | +15.5% test regression — camber breaks z-symmetry |
| #3323 | nezuko | PhysicsAttention entropy reg (weight=0.01/0.001) | **Closed** | +7.2%/+4.5% val regression — slice specialization is a feature, not a bug |

## Key learnings so far

1. **Optimizer is the dominant single lever, by far.** SOAP −49.2% in-PR (pre-merge) vs Huber −18.1% and LR warmup −8.9%. If the merged-stack rebase confirms, this resets the strategic landscape.
2. **Robust loss matters.** Huber −18.1% on its own is the second-largest gain; signals MSE was vulnerable to outlier pressure samples.
3. **LR schedule matters.** Warmup + higher peak −8.9%; orthogonal to Huber.
4. **Capacity scaling blocked at this scale.** Width/depth/MLP-ratio all incur ~1.55× epoch-time penalty, cutting epochs ~36% under the 30-min cap. Fourier PE is intermediate (1.15× — acceptable).
5. **Per-sample loss normalization hurts.** Destabilizes gradient balance across variable-size meshes.
6. **Strong generalization-gap shrinkage with SOAP.** Largest gains are on `test_re_rand` (−50.5%) and `val_geom_camber_cruise` (−55.5%) — consistent with curvature-aware steps finding flatter minima rather than just lower train loss.

## Next directions

### Immediate (within round-2)
- **SOAP rebase + re-run (PR #3283).** Confirm orthogonality with Huber + LR warmup. Highest-EV experiment of the entire round.
- **Fourier PE rebase (PR #3172).** Re-test on merged stack to see if the −14% in-PR signal survives.
- **p×3 upweight rebase (PR #3152).** Re-test channel weighting layered on top of Huber.

### Post-SOAP-merge (assuming it lands)
1. **SOAP LR sweep.** SOAP's effective LR may differ from AdamW's. Sweep {2e-4, 5e-4, 1e-3, 2e-3} with the new optimizer.
2. **SOAP preconditioner frequency.** Default is 10 steps; try {1, 5, 10, 20} to find the wall-clock/quality knee.
3. **SOAP × surf_weight re-balance.** With SOAP handling gradient conflict natively, is surf_weight=10 still optimal?
4. **SOAP × AoA augmentation × entropy reg stacking.** Test whether regularizers compound or saturate against the flatter-minimum SOAP regime.

### Further-future research themes
1. **Alternative robust losses on SOAP.** Cauchy/Welsch/Tukey biweight — Huber win signals outlier-robustness as a lever, but with SOAP's flatter minima, MSE-with-SOAP might already match Huber-with-AdamW.
2. **Ada-Temp slice reparameterization** — per-point temperature in PhysicsAttention softmax.
3. **Log-Re sinusoidal embedding** — 8-dim sinusoidal on log(Re); targets re_rand OOD (currently the strongest SOAP gain — synergy possible).
4. **Divergence-free auxiliary loss** — soft incompressibility penalty.
5. **Physical-units scale-aware loss** — normalize each field loss by physical scale (edward's follow-up direction).
6. **Stack winners after SOAP confirms.** Huber-optimal-delta + best-warmup + AoA-aug + entropy-reg + SOAP all together.
