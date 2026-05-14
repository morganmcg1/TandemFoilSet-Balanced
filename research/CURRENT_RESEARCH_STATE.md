# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 01:50
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.

## Current baseline (10th shift)

**PR #2562 (Lion lr=7.5e-5)** merged 2026-05-13 22:30:
- **`val_avg/mae_surf_p` = 45.433** (seed 2 `srveevtx`)
- **`test_avg/mae_surf_p` = 39.509**
- Per-split test: single_in_dist=42.56, geom_camber_rc=53.48, geom_camber_cruise=24.00, re_rand=37.99
- **New merge bar: val < 45.43, test < 39.51, all four test splits finite**

## Baseline progression

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 start |
| PR #1505 (Huber β=0.5 surf) | 2026-05-13 00:00 | 113.794 | 101.782 | −4.7% / −7.2% |
| PR #1715 (bf16 AMP) | 2026-05-13 02:00 | 89.597 | 79.907 | −21.3% / −21.5% |
| PR #1810 (torch.compile dynamic=True) | 2026-05-13 05:15 | 67.831 | 59.784 | −24.3% / −25.2% |
| PR #1910 (vol-Huber β=0.5) | 2026-05-13 07:30 | 65.469 | 57.837 | −3.5% / −3.3% |
| PR #1692 (grad_clip max_norm=1.0) | 2026-05-13 12:00 | 60.093 | 53.370 | −8.2% / −7.7% |
| PR #1589 (AdamW betas 0.9, 0.95) | 2026-05-13 16:03 | 59.970 | 52.363 | −0.2% / −1.9% |
| PR #2017 (weight_decay 1e-4 → 2e-4) | 2026-05-13 16:10 | 58.883 | 51.078 | −1.8% / −2.4% |
| PR #2516 (Lion optimizer) | 2026-05-13 20:05 | 50.193 | 43.501 | −14.8% / −14.8% |
| **PR #2562 (Lion lr=7.5e-5)** | **2026-05-13 22:30** | **45.433** | **39.509** | **−9.5% / −9.2%** |

**Cumulative: −62.0% val, −64.0% test from round-1 start.** Still compute-bound (best=last on both seeds at all 10 merges).

## Current research focus

**Lion optimizer hyperparameter tuning and schedule exploration.** The 10-stack baseline is compute-bound at every merge. The remaining exploration axes are:

1. **Beta1 sweep**: β1=0.85 (edward #2700) — bracketing β1 axis; β1=0.95 confirmed variance-reduction but slowed convergence
2. **SWA (thorfinn #2712)**: average last-10-epoch checkpoints (epochs 26-35) — free improvement from checkpoint averaging; zero compute overhead
3. **Beta2=0.999 (frieren #2713)**: longer momentum-buffer EMA window; opposite direction from closed β2=0.95 failure; untested bracket
4. **Gradient Centralization (nezuko #2564)**: GC inside Lion step — zero-mean gradient constraint before momentum update
5. **max_norm=0.5 (fern #2565)**: rebasing onto new baseline (sent back); OOD camber improved −3.8 on old baseline
6. **CosineAnnealingWarmRestarts T_0=12 (tanjiro #2693)**: multi-cycle schedule; different shape from retired cosine flat/long-T_max
7. **Charbonnier loss ε=0.5 (askeladd #2694)**: smooth L1 alternative to Huber; loss-family change
8. **SiLU activation (alphonse #2505)**: GELU→SiLU in FFN; rebasing onto new baseline

## Round 1 portfolio (live)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1504–#2017 | various | 8 stacked improvements | **MERGED** (baseline history above) |
| #2516 | edward | Lion optimizer | **MERGED** 2026-05-13 20:05 (val=50.19) |
| #2562 | tanjiro | Lion lr=7.5e-5 | **MERGED** 2026-05-13 22:30 (val=45.43) — 10th baseline shift |
| #2561 | edward | Lion beta2=0.95 | **CLOSED** (+14.8%, beta2 analogy wrong) |
| #2520 | thorfinn | n_head 4→8 | **CLOSED** (+24%, capacity loss) |
| #2504 | frieren | QK-RMSNorm | **CLOSED** (+14%, Q/K magnitude signal) |
| #2628 | tanjiro | Lion lr=1e-4 | **CLOSED** (+1.9% val, overshoot — sweet spot at 7.5e-5) |
| #2501 | askeladd | β_p=0.625 | **CLOSED** (+6.8% val — per-channel β axis fully closed) |
| #2565 | fern | max_norm=0.5 | WIP — rebasing onto new baseline |
| #2564 | nezuko | Gradient Centralization | WIP — running |
| #2505 | alphonse | SiLU activation | WIP — running on new baseline |
| #2633 | edward | Lion beta1=0.95 | **CLOSED** (+4.83 pt val; variance −85% but convergence slowed) |
| #2631 | thorfinn | Lion warmup 5ep | **CLOSED** 2026-05-14 01:45 (+4.44% val; variance −67% but 5 epochs eat compute budget) |
| #2629 | frieren | Lion wd=3e-3 | **CLOSED** 2026-05-14 01:45 (+1.68 pt val; all 4 splits regress; wd axis monotonic-worse) |
| #2700 | edward | Lion beta1=0.85 | **WIP** |
| #2693 | tanjiro | CosineAnnealingWarmRestarts T_0=12 | **WIP** |
| #2694 | askeladd | Charbonnier loss ε=0.5 | **WIP** |
| **#2712** | **thorfinn** | **SWA (average epochs 26-35)** | **WIP NEW 2026-05-14 01:50** |
| **#2713** | **frieren** | **Lion beta2=0.999** | **WIP NEW 2026-05-14 01:50** |

**Merged:** 10 | **Closed:** 36 | **WIP:** 8 | **Idle:** 0

## Key meta-findings from round 1

1. **Compute is permanently binding** — best=last at every merge. The 30-min cap has been the dominant constraint since bf16 (PR #1715).
2. **Lion composes cleanly with grad_clip** — no "double normalization" fight; both operate on orthogonal mechanisms.
3. **LR=7.5e-5 seed variance** — seed std increased 4-6× (3.60 pt vs 0.97 pt). β1=0.95 reduces variance 85% but slows convergence 5pt. Warmup reduces variance 67% but consumes compute budget. SWA and β1=0.85 are the remaining variance-reduction candidates.
4. **Warmup costs too much at 35-epoch budget** — every epoch counts; 5-epoch warmup shortens effective cosine phase from 35→30 epochs.
5. **wd axis monotonic-worse upward** — wd=3e-3 regressed all 4 splits; any convergence slowdown at compute-bound cap shows as worse val. wd=2e-3 is confirmed optimal or near-optimal.
6. **Lion beta2 axis**: β2=0.95 regressed +14.8% (too-short EMA). β2=0.999 (longer EMA) is the untested upper bracket — currently in flight.
7. **Architecture axis CLOSED for round 1** — n_head=8 (+24% regression) joins the scalar-capacity cluster of failures.
8. **Q/K magnitude carries physics-discriminative signal** — QK-RMSNorm regressed because per-domain log(Re) and dsdf scales propagate through Q/K projections.

## Potential next research directions

### Immediate (follow from current results)

1. **β1=0.85 result** — if improves: confirmed faster adaptation helps; try β1=0.8 next. If regresses: β1 axis fully closed, keep β1=0.9.
2. **SWA result** — if improves: extend averaging window or try different aggregation (median-of-weights). If no gain: model's late-training checkpoints are less diverse than hypothesized.
3. **β2=0.999 result** — if improves: try β2=0.9995 to find optimal EMA window. If regresses: Lion β2 axis fully retired in both directions.
4. **fern max_norm=0.5 on new baseline** — the OOD-camber −3.8 pt improvement was the most promising single-split signal this round; needs to hold on 7.5e-5 baseline.
5. **Combine best axes** — once β1 and β2 axes close, try joint (β1=0.85, β2=0.999) if both improve individually.
6. **max_norm=0.25** — continue downward clip scan from 0.5 if fern's rebase wins.

### Medium-term (needs researcher-agent exploration)

6. **Per-sample Re embedding** — re_rand and cruise are still the easiest splits; Re-normalized input features may unlock OOD generalization on the harder re-variation axis
7. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens; directly addresses the "surface inherits from volume" structural relationship
8. **Quantile/Pinball loss** — more aggressive median-targeting than Huber for the physically-meaningful pressure channel
9. **Data augmentation** — y-flip + Uy-negation for cruise samples (flow-symmetric BCs admit clean mirror augmentation)
10. **Larger model with Lion** — scalar-capacity failed on AdamW; Lion's faster convergence might enable n_hidden=192 at 35 epochs

### Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines
- **Cosine LR shape** (T_max, eta_min, warmup-then-flat) — 3 negative results; warm restarts (#2693) currently in flight as different shape
- **Noise injection** (dropout, DropPath) — both regressed; regularization stack already saturated
- **Lion beta2=0.95** — analogy to AdamW wrong; keep (0.9, 0.99)
- **Lion LR (upper bracket)** — 1e-4 overshoots; 7.5e-5 is the sweet spot
- **Per-channel Huber β** — both directions failed under both optimizer baselines; global β=0.5 robust
- **n_head=8** — capacity loss + per-epoch overhead
- **QK-RMSNorm** — Q/K magnitudes carry physics-discriminative signal
- **surf_weight** — fully bracketed (5/10/20); convex, 10 is optimal
- **EMA weights** — EMA-lag on cooling cosine cancels the smoothing
- **Lion warmup 5ep** — 5 warmup epochs eat compute budget; variance reduction insufficient to compensate
- **Lion wd=3e-3** — monotonic-worse; wd=2e-3 confirmed optimal
