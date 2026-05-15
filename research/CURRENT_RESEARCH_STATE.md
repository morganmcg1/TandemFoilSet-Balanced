# SENPAI Research State

- **Date:** 2026-05-15 (updated 16:30 after round-1 merges + round-2 assignments)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline:** `val_avg/mae_surf_p = 110.83`, `test_avg/mae_surf_p (excl cruise) = 109.75`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%)

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR after round-2 clears. All comparisons use 3-split test mean (excl cruise).

## Round-1 outcomes

| PR | Student | Hypothesis | Δ val vs old baseline (135.30) | Decision |
|---|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192, 4→6 heads) | +18.7% | Closed |
| #3147 | askeladd | LR warmup + peak 5e-4→1e-3 | **−8.9%** | **Merged ✓** |
| #3152 | edward | Per-channel p×3 MSE upweight | +0.6% (noise) | Request changes |
| #3155 | fern | Huber loss (SmoothL1 delta=1.0) | **−18.1%** | **Merged ✓** |
| #3161 | frieren | Per-sample loss normalization | +13.0% | Closed |
| #3165 | nezuko | Depth scaling 5→8 layers | +25.4% | Closed |
| #3169 | tanjiro | MLP ratio 2→4 | just started (delayed by pod issue) | TBD |
| #3172 | thorfinn | Fourier pos features + slice 96 | just started (delayed by pod issue) | TBD |

## Round-2 current assignments (all 8 students active)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| #3283 | alphonse | SOAP optimizer (drop-in AdamW replacement) | Optimization | WIP |
| #3319 | askeladd | LR warmup duration sweep: 1-epoch vs 3-epoch vs 5-epoch | Optimization | WIP |
| #3152 | edward | Surface-only p×3 upweight (follow-up) | Loss formulation | WIP |
| #3316 | fern | Huber delta sensitivity: beta=0.5 vs 1.0 vs 2.0 | Loss tuning | WIP |
| #3322 | frieren | AoA reflection augmentation (sign-flip AoA + Uy, p=0.5) | Data aug | WIP |
| #3323 | nezuko | PhysicsAttention entropy reg (slice collapse prevention) | Architecture | WIP |
| #3169 | tanjiro | MLP ratio 2→4 | Capacity | WIP (just started) |
| #3172 | thorfinn | Fourier (x,z) features + slice_num 64→96 | Inputs | WIP (just started) |

Zero idle students.

## Key learnings so far

1. **Robust loss is the dominant lever.** Huber −18.1% is the largest single gain. MSE was vulnerable to outlier pressure samples.
2. **LR warmup + higher peak compounds with loss changes.** −8.9% from warmup + peak; orthogonal to Huber.
3. **Capacity scaling blocked.** Width/depth/MLP-ratio all incur ~1.55× epoch-time penalty, cutting epochs by ~36% under the 30-min cap.
4. **Per-sample loss normalization hurts.** Destabilizes gradient balance across variable-size meshes.

## Next directions (post round-2)

After round-2 results:
1. **Alternative robust losses** — Cauchy/Welsch/Tukey biweight (Huber win signals outlier-robustness is a key lever)
2. **Ada-Temp slice reparameterization** — per-point temperature in PhysicsAttention softmax
3. **Log-Re sinusoidal embedding** — 8-dim sinusoidal on log(Re); targets re_rand OOD
4. **Divergence-free auxiliary loss** — soft incompressibility penalty
5. **Per-domain normalization** — domain-conditioned y stats
6. **Physical-units scale-aware loss** — normalize each field loss by physical scale (edward's analysis)
7. **Stack winners** — run Huber-optimal-delta + best-warmup + AoA-aug + entropy-reg all together
