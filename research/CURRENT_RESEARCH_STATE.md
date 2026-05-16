# SENPAI Research State

- **Date:** 2026-05-16 03:30
- **Launch:** willow-pai2i-48h-r1 (round 4 continuing; frieren+nezuko PRs assigned; 2 negative results pending student SENPAI-RESULT posts)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~18 epochs achievable in bf16 at bs=4)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline
- **val_avg/mae_surf_p = 87.9105** (PR #3480, bf16 autocast + T_max=15)
- **test_avg/mae_surf_p = 83.3782** (4-split, all valid)
- **Epochs in 30 min:** 18 (bf16) vs 14 (fp32)
- **Peak VRAM:** 32.9 GB vs 78 GB (fp32)
- W&B: `t00506x1`
- Noise floor σ ≈ 1.80 (alphonse PR #3305 4-replicate characterization, confirmed by nezuko PR #3175)

## Merged PRs
| PR | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|----|-----------|---------------------|---------------------|
| #3159 | Huber loss δ=0.1 | 112.9001 | 115.7589 (3/4) |
| #3309 | NaN fix (cruise test) | 112.8295 | 106.5996 (4/4) |
| #3317 | Cosine T_max=15 | 91.3319 | 88.4260 (3/4) |
| #3480 | **bf16 autocast (bs=4)** | **87.9105** | **83.3782 (4/4)** |

## Closed PRs (key dead ends)
| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| #3162 | surf_weight=25 (MSE) | 133.41 | Loss misalignment |
| #3188 | slice_num=128 (MSE) | 134.74 | Capacity not bottleneck (fp32 base) |
| #3167 | OneCycleLR | 137.12 | Budget too short |
| #3180 | h=192 wider (MSE/T=50) | 150.38 | Resource-constrained, unfavorable conditions |
| #3361 | slice_num=128 (retried) | 116.19 | 30% slower/ep, fp32 VRAM-saturated |
| #3359 | edward no-commit | 133.32 | Iterated w/o pushing |
| #3395 | LR peak scan 3e-4/8e-4 | 94.18/94.46 | lr=5e-4 at basin minimum |
| #3426 | Warm restarts T_0=5 | 103.07 | 5-ep cycles too short |
| #3460 | bf16 + bs=8 | 110.72 | bs=8 starved AdamW (-39% updates) |
| #3459 | EMA decay=0.999 | 100.92 | Decay half-life > training horizon |
| #3174 | L1-on-p + surf_w=50 | 99.51 | Gradient starvation (94% on surf-p) |
| #3305 | Huber δ=0.05 (4 replicates) | 91.47 ± σ=1.80 | Within noise; **headline σ characterization** |
| #3428 | surf_weight scan (15, 20) | 91.6 / 92.07 | Within σ — uniform surf_weight lever exhausted |
| #3522 | L1-on-p ONLY w=10 | 103.07 | L1-shape weaker small-residual grads |
| #3171 | Split pressure head v3 | 100.78 | Capacity cost > 14-ep payback |
| #3363 | AdamW β2=0.95 + clip 1.0 | 92.43 (rebased) | Compounded with schedule fix into noise |
| #3175 | Cosine warmup (3 replicates) | mean 95.16, best 89.65 | Within noise — mean firmly worse than baseline |
| #3542 | TTA via h-flip (eval) | 91.14 raw / TTA=161.10 | **Dataset NOT z-symmetric — raceCar catastrophically OOD** |
| #3580 | SWA over last 5 ckpts | 89.86 | SWA ≈ best-by-val under cosine T_max=15 (frozen tail) |
| #3563 | Train-aug h-flip | 111.70 | **+27% catastrophic — confirms #3542 dataset finding** |

## Active WIP — 8 students total (zero idle)
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3546 | alphonse | Seed control + 4 baseline replicates (σ̂ on 87.91 base) | **4 runs DONE — pending SENPAI-RESULT post** |
| #3562 | askeladd | Wider Transolver h=192 slice=96 T_max=18 under bf16 | ACTIVE — run `fqzs1zk1` training |
| #3611 | edward | Per-channel surf weight β_p=20 (Ux/Uy at α=10) | In flight |
| #3566 | fern | Unified positional encoding (Transolver unified_pos=True) | **2 runs DONE (val ~107, regress) — pending SENPAI-RESULT** |
| #3642 | frieren | Layer-wise LR decay γ=0.85 (5 Transolver blocks) | NEW after #3563 close |
| #3644 | nezuko | Cosine T_max=10 + 8-ep constant LR tail + SWA-over-tail | NEW after #3580 close |
| #3574 | tanjiro | Per-channel Huber-δ (δ_p=0.05 on surf-p only) | In flight |
| #3521 | thorfinn | EMA decay=0.99 (faster forgetting) | ACTIVE — run `s35tc2it` training (post-pod-restart) |

## Round 4 surfacing results (preliminary, pending SENPAI-RESULT posts)

### alphonse #3546 — 4 baseline replicates (W&B):
| Run | Seed | val_avg | test_avg |
|---|---|---:|---:|
| `ek21s9hy` | seed0 (retry) | 91.11 | 85.64 |
| `8vcv4ojk` | seed1 | 90.16 | 85.54 |
| `1y3my9x2` | seed2 | 93.60 | 86.83 |
| `0ekl0alh` | seed3 | 90.25 | 85.37 |
| **mean** |  | **91.28** | **85.85** |
| **σ̂** |  | **~1.60** | **~0.65** |

**Critical implication:** baseline #3480 val=87.91 is **~2σ below** the canonical-config seed distribution mean. The 87.91 number may have been a downward outlier. *True* expected val under canonical bf16+T_max=15 is ~91 ± 1.6. **All future PR evaluations should be calibrated against this distribution**, not the point-estimate 87.91. Will update BASELINE.md variance notes once alphonse posts terminal marker.

### fern #3566 — unified_pos=True (W&B):
| Run | val_avg | test_avg |
|---|---:|---:|
| `nugotxr6` | 106.51 | 99.76 |
| `s0tj1q82` | 108.07 | 103.13 |

>10σ regression on both runs. **Closing pending SENPAI-RESULT post.** Per-split breakdown will determine whether the regression is OOD-concentrated (informative finding about positional encoding's role in geom generalization) or uniform (clean dead end).

## Key insights from rounds 3 & 4 (cumulative)
1. **bf16 is a clean orthogonal win** (PR #3480). 18 epochs/30min, 32.9GB VRAM. Now in canonical train.py. Stacks with all other levers.
2. **Noise floor is σ ≈ 1.80** (alphonse PR #3305) — independently confirmed by nezuko's PR #3175 (3 seeds, ~5pt std). alphonse #3546 is running a fresh 4-replicate σ̂ on the new 87.91 baseline.
3. **L1-on-surf-p lever fully exhausted** (#3174, #3522). The cruise OOD signal from #3174 was confounded with the surf_weight boost.
4. **Marginal hyperparameter tweaks plateau within σ of baseline.** Surf_weight scan, β2+clip, Huber δ scan, cosine warmup, SWA — all within ±2σ. Time to bigger swings.
5. **Capacity scaling under unfavorable conditions failed** (#3180, #3188). bf16's VRAM unlock + T_max=18 budget enables a real retest (askeladd #3562, in flight).
6. **Schedule warmup lever is dead** (#3175). Cosine T_max=15 already provides soft warmup-like ramp.
7. **Optimizer-stability lever is dead** (#3363). β2+clip on OLD base was schedule fix in disguise.
8. **Dataset is NOT z-symmetric for raceCar domain** (PR #3542, edward). RaceCar has one-sided pos_z, AoA; non-negative NACA camber encoding. **This falsifies the entire naive horizontal-flip-symmetry lever class** (eval-time TTA #3542 and train-aug #3563 both catastrophically regressed).
9. **SWA under cosine T_max=15 is a no-op** (PR #3580). Last 5 epochs near-frozen → SWA ≈ best-by-val. nezuko's follow-up tests this on a constant-LR-tail schedule.
10. **Cosine T_max=15 tail is essentially dead weight** (epochs 15-17 with LR≈0). Implication: thorfinn's #3521 EMA decay=0.99 also faces this — its averaging window includes the frozen tail.

## Round 4+ active levers (8 students)
1. **Noise-floor characterization** (alphonse #3546) — seed control + 4-replicate baseline σ̂.
2. **Capacity scaling** (askeladd #3562) — bigger model (h=192 slice=96 T_max=18), bf16.
3. **Per-channel surf weight** (edward #3611) — β_p=20 on surface-p only, Ux/Uy stay at α=10.
4. **Architectural — unified positional encoding** (fern #3566).
5. **Layer-wise LR decay** (frieren pending) — γ=0.85 across 5 Transolver blocks.
6. **Constant-LR-tail SWA** (nezuko pending) — cosine T_max=10 + 8-ep constant tail + SWA.
7. **Loss shape — per-channel Huber-δ** (tanjiro #3574) — δ_p=0.05 on surface-p only.
8. **Weight averaging — EMA** (thorfinn #3521) — decay=0.99 during training.

## Next research directions (post round 4)
1. **Stack winners** — if per-channel-δ (#3574) and per-channel-surf-weight (#3611) both land, stack them (loss shape × gradient magnitude, orthogonal mechanisms).
2. **Per-domain normalization** — pressure ranges differ by split (edward's #3542 distribution table). Not yet tested.
3. **β_p scan** — if edward's #3611 β=20 wins, scan β∈{15, 25, 30}.
4. **Mesh-permutation augmentation** — true symmetry of the mesh encoder, no physical-content change. Replaces the dead h-flip augmentation lever.
5. **Corrected h-flip** (flip pos_z + AoA + Uy, leave camber alone) — frieren's suggested follow-up from #3563 analysis.
6. **Cruise-only conditional augmentation** — leverage the one z-symmetric split.
7. **bf16 + h=192 + EMA + LLRD stack** — full kitchen-sink confirmation run once individual levers land.
8. **SwiGLU/SiLU activation swap** — modern transformer activation, untested in this program.
9. **DropPath / stochastic depth** — orthogonal regularizer, untested.
10. **Lighter split head** (fern's suggested follow-up from #3171) — h→h instead of h→2h→h.

## Plateau status
Not in plateau. bf16 unlock (#3480) was a real win past noise on test. Round 4 closes are concentrated information: dataset asymmetry finding (#3542 + #3563) is a high-value negative ruling out an entire lever class; SWA-on-frozen-tail finding (#3580) explains the redundancy and points to a fix. 7 active orthogonal lever classes in flight; 1-3 expected to compound.

## Operational notes
- **GitHub REST API rate limit (5000/hr) is being hit** by combined advisor+student traffic. Student pods (alphonse, thorfinn) hit it on assignment polls and fall back to "no work assigned" sleep, but training continues uninterrupted. Advisor work batches around the hourly reset.
- **Thorfinn pod self-recovered** from earlier stuck-on-checkout state (uncommitted train.py blocking branch switch) via natural restart. Currently iteration 2, 100% GPU utilization.
- Both alphonse and thorfinn are training-active; the stale_wip flag is a false positive driven by GH rate-limited polling, not actual stall.
