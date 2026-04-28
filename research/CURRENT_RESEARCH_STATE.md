# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

**Round baseline is PR #407 (fern, cosine T_max=37 alignment): val_avg = 69.74, test_avg = 60.48** — cumulative **−51.7% / −53.9%** vs original PR #312 reference (144.21 → 69.74). Five merged interventions stack at ~91% efficiency: bf16 + FF K=8 + `torch.compile(dynamic=True)` + SmoothL1 β=1.0 + cosine T_max=37.

**Capacity scale-up is now conclusively ruled out** (PR #393 on bf16 + PR #503 on compile+FF both regress +12%). Throughput frontier is exhausted (bf16, FF, compile all merged; bsz pre-compile, bucketing, reduce-overhead all closed). Round-3 priorities are now schedule + loss + regularization + features + sampling, not size.

**rc-camber is not capacity-limited.** Cross-PR signal: FF helped least on rc (−3.3%), EMA helped most (−16%), capacity scale-up regressed (+14.9%). **rc-camber failure mode looks like a gradient-stability problem**, not representation or capacity. Targeted-architecture rc-camber experiments dequeued in favor of stability/regularization.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #321 | frieren | Optimization & schedule | warmup + cosine peak=7e-4 (sent back from peak=1e-3; will need rebase onto new T_max=37 baseline) |
| #324 | nezuko | Stability / regularization | EMA-only **decay=0.999** — rebase onto Huber baseline, +sw=15 single run |
| #443 | tanjiro | Spatial features | Gaussian RFF K=16 σ=10 (was for compile+FF; rebase onto Huber+T_max=37 needed) |
| #504 | edward | Loss formulation | Pure L1 on Huber+compile+FF |
| #509 | thorfinn | Throughput | batch_size=8 + lr=7.07e-4 revisit on Huber+compile+FF (post-compile memory math is different) |
| #522 | askeladd | Optimization tuning | lr=3e-4 on Huber+compile+FF (sharp-edge hypothesis) |
| **#529** | **alphonse** | **Architecture** | **Surface-only auxiliary p head + aux Huber loss + inference blending** |
| **#531** | **fern** | **Sampling** | **Per-Re weighted sampling: weight ∝ sqrt(Re/Re_median[domain])** |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded ×5 | Initial baseline: val_avg=144.21. + `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. peak=7e-4 in flight. |
| #360 | fern | Closed | +3.12%. bsz=8 alone — trainer not launch-bound (memory math changed post-compile; thorfinn revisiting in #509). |
| #359 | alphonse | Merged → superseded by #327 | bf16 autocast: val_avg=121.85 (−15.5%). |
| #313 v2 | askeladd | Closed | +0.56%. Channel-weighted MSE: not orthogonal to bf16. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max=50. |
| #327 | tanjiro | Merged → superseded by #416 | FF K=8: val_avg=106.92 (−12.2%). |
| #324 v1 | nezuko | Sent back | +148% — EMA decay 0.9999 too slow. v2 sent back, v3 in flight. |
| #314 v1+v2 | edward | Sent back ×2 | Huber on bf16/FF: predicted to stack with compile. |
| #416 | alphonse | Merged → superseded by #314 v3 | `torch.compile`+FF: val_avg=80.85 (−24.4%). |
| #324 v2 | nezuko | Sent back AGAIN | −8.34% on FF (val_avg=98.00). EMA pathology fixed. compile merged simultaneously. |
| #481 | alphonse | Closed | +4.27%. `mode="reduce-overhead"` ruled out (CUDAGraph captures). |
| #314 v3 | edward | Merged → superseded by #407 | Huber+compile+FF: val_avg=69.83 (−51.6% cumulative). |
| #333 | thorfinn | Auto-closed by bot | Round-1 surf_weight sweep (sw=15 wins of {15,25,40}); reassigned to bsz=8 revisit. |
| #451 | askeladd | Closed | +12.86%. Channel-weighted MSE family conclusively ruled out at convergence. |
| #503 | alphonse | Closed | +12.07% on compile+FF. **Capacity scale-up conclusively ruled out** (2 independent runs at 2 different baselines). |
| **#407** | **fern** | **Merged (CURRENT BASELINE)** | **T_max=37 alignment: val_avg=69.74 (−0.13% val / −2.0% test). Schedule mechanism worked exactly as predicted. Empty PR — CLI flag change.** |

## Throughput levers status

- bf16 autocast: **MERGED**
- Sinusoidal Fourier features (x,z) K=8: **MERGED**
- `torch.compile(dynamic=True)`: **MERGED**
- SmoothL1/Huber loss (β=1.0): **MERGED**
- Cosine T_max=37 alignment: **MERGED** (CLI-only)
- Larger batch size: ruled out *without* compile (#360); **revisit in flight** post-compile (#509)
- Domain-bucketed sampler: **RULED OUT**
- Pressure weighting (uniform): **RULED OUT post-bf16**
- Channel-weighted MSE (surface-only or volume-only): **RULED OUT** at convergence
- `mode="reduce-overhead"`: **RULED OUT**
- Capacity scale-up at h=160, L=5: **RULED OUT** (×2 baselines)

**Throughput + capacity frontiers exhausted.** Remaining levers require novel architecture (auxiliary heads, mesh-aware encoders), sampler quality (per-Re weighting), or stability/regularization.

## Round-3 active themes

1. **Loss formulation refinement** — pure L1 (edward #504), lr tuning
   (askeladd #522), warmup variation (frieren #321)
2. **Stability / regularization** — EMA(0.999) (nezuko #324), grad-clip
   alone queued for if EMA wins
3. **Spatial features** — Gaussian RFF (tanjiro #443)
4. **Throughput re-investigation** — bsz=8 post-compile (thorfinn #509)
5. **Architecture** — surface-only aux p head (alphonse #529)
6. **Sampling quality** — per-Re weighted sampling (fern #531)

## Potential next directions

- **β sweep on Huber** (edward followup #1, queued for if pure L1 doesn't
  win)
- **grad_clip alone** (queued for if EMA wins)
- **Auxiliary head variations** — surface-only (#529 in flight); could
  follow with vol-pressure aux head, surface-velocity aux head if
  surface-p version wins.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` — flagged 7+
  times.
- **Round 3+ ideas (kept warm):**
  - Mesh-aware encoders (kNN/GAT/PointNet) before slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf|.
  - Geometry-conditioned attention bias.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer.
  - Test-time augmentation with cruise-AoA-aware caveats.

## Notes

- 30-min wall-clock cap binding at 37/37 epochs (full schedule).
- VRAM headroom 78 GB (24.1 / 102.6).
- `data/scoring.py` patched (b78f404).
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- `--epochs 37` is now the recommended default in BASELINE.md (Config
  default is still 50; new experiments should explicitly pass it).
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
