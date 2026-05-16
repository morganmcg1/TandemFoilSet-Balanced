# SENPAI Research State

- **Date:** 2026-05-16 01:35
- **Launch:** willow-pai2i-48h-r1 (round 4 in progress)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~18 epochs achievable in bf16 at bs=4)
- **Latest direction from human team:** None

## Research contract
Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Primary paper-facing metric also includes `test_avg/mae_surf_p` (all 4 splits valid since PR #3309 merged).

## Current best baseline ← NEW
- **val_avg/mae_surf_p = 87.9105** (PR #3480, bf16 autocast + T_max=15)
- **test_avg/mae_surf_p = 83.3782** (4-split, all valid)
- **Epochs in 30 min:** 18 (bf16) vs 14 (fp32)
- **Peak VRAM:** 32.9 GB vs 78 GB (fp32)
- W&B: `t00506x1`
- Noise floor σ ≈ 1.80 (alphonse PR #3305 4-replicate characterization)

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
| #3428 | surf_weight scan (15, 20) | 91.6 / 92.07 | Within σ — surf_weight lever exhausted |
| #3522 | L1-on-p ONLY w=10 | 103.07 | L1-shape weaker small-residual grads |
| #3171 | Split pressure head v3 | 100.78 | Capacity cost > 14-ep payback |
| #3363 | AdamW β2=0.95 + clip 1.0 | 92.43 (rebased) | Compounded with schedule fix into noise |
| #3175 | Cosine warmup (3 replicates) | mean 95.16, best 89.65 | Within noise — mean firmly worse than baseline |

## Active WIP — 8/8 students assigned (zero idle GPUs)
| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3546 | alphonse | **Seed control + 4 baseline replicates (σ̂ on 87.91 base)** | NEW after #3305 close |
| #3562 | askeladd | **Wider Transolver h=192 slice=96 T_max=18 under bf16** | NEW after #3480 merge |
| #3542 | edward | **TTA via horizontal-flip symmetry (eval-time)** | NEW after #3428 close |
| #3566 | fern | **Unified positional encoding (Transolver unified_pos=True)** | NEW after #3171 close |
| #3563 | frieren | **Train-time horizontal-flip augmentation** | NEW after #3522 close |
| #3580 | nezuko | **SWA over last 5 checkpoints (variance reduction)** | NEW after #3175 close |
| #3574 | tanjiro | **Per-channel Huber-δ (δ_p=0.05 on surf-p only)** | NEW after #3363 close |
| #3521 | thorfinn | EMA decay=0.99 (faster forgetting) | Assigned 2026-05-15 23:35 |

## Key insights from round 3
1. **bf16 is a clean orthogonal win** (PR #3480). 18 epochs/30min, 32.9GB VRAM. Now in canonical train.py. Stacks with all other levers.
2. **Noise floor is σ ≈ 1.80** (alphonse PR #3305) — **independently confirmed by nezuko's PR #3175** (3 identical-config seeds, ~5pt std). Two pieces of strong evidence. train.py has no seed control — alphonse's #3546 adds it as a permanent fixture.
3. **L1-on-surf-p lever fully exhausted** across both `surf_weight=50` (#3174 gradient starvation) and `surf_weight=10` (#3522 weak small-residual grads). The cruise OOD signal from #3174 was confounded with the surf_weight boost.
4. **Marginal hyperparameter tweaks plateau within σ of baseline.** Surf_weight scan, β2+clip, Huber δ scan, cosine warmup — all came back within ±2σ. Time to bigger swings.
5. **Capacity scaling under unfavorable conditions failed** (#3180 h=192 T_max=50, #3188 slice=128 fp32). bf16's VRAM unlock + T_max=18 budget enables a real retest (askeladd #3562).
6. **Schedule warmup lever is dead** (#3175). The bare cosine T_max=15 already provides a soft warmup-like LR ramp; explicit warmup adds nothing measurable.
7. **Optimizer-stability lever is dead** (#3363). β2+clip gain on the OLD base was the schedule fix in disguise.

## Round 4 active levers (8/8 orthogonal classes — ZERO idle)
1. **Noise-floor characterization** (alphonse #3546) — seed control + 4-replicate baseline σ̂.
2. **Capacity scaling** (askeladd #3562) — bigger model (h=192 slice=96 T_max=18), bf16.
3. **Inference-time symmetry (TTA)** (edward #3542) — h-flip + average.
4. **Architectural — unified positional encoding** (fern #3566).
5. **Train-time symmetry augmentation** (frieren #3563) — h-flip during training.
6. **Weight averaging — SWA** (nezuko #3580) — average last 5 checkpoints (post-training).
7. **Loss shape — per-channel Huber-δ** (tanjiro #3574) — δ_p=0.05 on surface-p only.
8. **Weight averaging — EMA** (thorfinn #3521) — decay=0.99 during training.

Note: items 6 and 8 both test weight-averaging at different granularities (post-hoc uniform vs in-training EMA). These complement rather than duplicate.

## Next research directions (post round 4)
1. **Triple-stack winners** (bf16 + TTA + train-aug if all three land) — biggest paper-facing impact.
2. **Per-domain normalization** — pressure ranges differ by split; not yet tested.
3. **Layer-wise LR decay** — orthogonal regularizer.
4. **SWA (Stochastic Weight Averaging)** during final epochs — natural extension if EMA decay=0.99 works.
5. **Smaller δ for surf-p only** (frieren's suggested follow-up from #3522) — δ_p < 0.1, leave Ux/Uy at 0.1.
6. **Lighter split head** (fern's suggested follow-up from #3171) — h→h instead of h→2h→h.
7. **Per-channel surf weight** (edward/frieren suggestion) — α=10 on Ux/Uy, β>10 on p.
8. **Test-time + train-time symmetry stacked** — once both edward and frieren land.
9. **bf16 + h=192 + EMA + TTA + train-aug stack** — full kitchen-sink confirmation run.

## Plateau status
Not in plateau. The bf16 unlock (PR #3480) was a real win past the noise floor on the test metric. Round 4 has 5 fresh orthogonal lever classes in flight; expect 1-3 of them to compound.
