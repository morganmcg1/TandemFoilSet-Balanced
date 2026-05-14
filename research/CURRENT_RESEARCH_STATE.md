# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 00:35
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

**Lion optimizer LR and hyperparameter tuning** — the primary axis of active exploration. The Lion merger delivered −14.8% (the 3rd-largest round-1 win). The LR raise to 7.5e-5 delivered another −9.5%. The model is still compute-bound; the convergence curve continues to descend at the 30-min timeout.

Active sub-axes:
1. **Warmup (thorfinn):** 5-epoch linear warmup 0→7.5e-5 then cosine (PR #2631) — addresses 4-6× seed variance increase at 7.5e-5
2. **Weight decay (frieren):** wd=3e-3 at lr=7.5e-5 (PR #2629) — stronger L2 at higher LR
3. **Beta1 (edward):** beta1=0.95 vs default 0.9 (PR #2633) — reduces per-batch gradient noise in Lion's sign update
4. **Gradient Centralization (nezuko):** GC inside Lion step (PR #2564) — zero-mean gradient constraint before momentum update
5. **max_norm=0.5 (fern):** sent back for rebase onto new baseline (PR #2565)
6. **CosineAnnealingWarmRestarts (tanjiro):** T_0=12, 3 restart cycles in 35 epochs (PR #2693) — schedule axis fresh direction
7. **Charbonnier loss (askeladd):** ε=0.5 smooth L1 alternative to Huber (PR #2694) — loss-family change

**Independent axes still in flight:**
8. **SiLU activation (alphonse, PR #2505):** GELU→SiLU in FFN; rebasing onto new baseline

**Closed this round (post-Lion lr=7.5e-5):**
- **Lion lr=1e-4 (tanjiro #2628):** +1.9% val regression. Three diagnostic signals (ep-15 didn't improve, final val regressed, s2 destabilized at end) confirm overshoot. **LR sweet spot at 7.5e-5; further upward exploration retired.**
- **Per-channel β_p=0.625 (askeladd #2501):** +6.8% val regression, all 4 splits regressed. **Per-channel β axis FULLY CLOSED in both directions, under both AdamW and Lion baselines.** Global β=0.5 is robust.

## Key meta-findings from round 1

1. **Compute is permanently binding** — best=last at every merge. The 30-min cap has been the dominant constraint since bf16 (PR #1715).
2. **Lion composes cleanly with grad_clip** — no "double normalization" fight; both operate on orthogonal mechanisms. The early-epoch trajectory is identical to AdamW; Lion's advantage opens in late-training.
3. **LR=7.5e-5 seed variance signal** — seed std increased 4-6× (3.60 pt vs 0.97 pt). Warmup and beta1 tuning are the primary variance-reduction candidates.
4. **Lion beta2 axis CLOSED** — (0.9, 0.95) regressed +14.8%. Analogy to AdamW beta2=0.95 was wrong (different parameter semantics). Keep (0.9, 0.99).
5. **Architecture axis CLOSED for round 1** — n_head=8 (+24% regression) joins the scalar-capacity cluster of failures. PhysicsAttention's per-head QKV layout makes n_head a capacity axis, not a redistribution axis.
6. **Q/K magnitude carries physics-discriminative signal** — QK-RMSNorm regressed because per-domain log(Re) and dsdf scales propagate through Q/K projections. Unit-norming destroys this. Pre-attention LayerNorm is already sufficient.

## Round 1 portfolio (live)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1504–#2017 | various | 8 stacked improvements | **MERGED** (baseline history above) |
| #2516 | edward | Lion optimizer | **MERGED** 2026-05-13 20:05 (val=50.19) |
| #2562 | tanjiro | Lion lr=7.5e-5 | **MERGED** 2026-05-13 22:30 (val=45.43) — 10th baseline shift |
| #2561 | edward | Lion beta2=0.95 | **CLOSED** 22:50 (+14.8%, beta2 analogy wrong) |
| #2520 | thorfinn | n_head 4→8 | **CLOSED** 22:42 (+24%, capacity loss) |
| #2504 | frieren | QK-RMSNorm | **CLOSED** 22:43 (+14%, Q/K magnitude signal) |
| #2628 | tanjiro | Lion lr=1e-4 | **CLOSED** 2026-05-14 00:30 (+1.9% val, overshoot — sweet spot at 7.5e-5) |
| #2501 | askeladd | β_p=0.625 | **CLOSED** 2026-05-14 00:30 (+6.8% val — per-channel β axis fully closed) |
| #2565 | fern | max_norm=0.5 | WIP — sent back for rebase onto new bar |
| #2564 | nezuko | Gradient Centralization | WIP — running on lr=7.5e-5 |
| #2505 | alphonse | SiLU activation | WIP — running on new baseline |
| #2631 | thorfinn | Lion warmup 5ep | WIP — running |
| #2629 | frieren | Lion wd=3e-3 | WIP — running |
| #2633 | edward | Lion beta1=0.95 | WIP — running |
| **#2693** | **tanjiro** | **CosineAnnealingWarmRestarts T_0=12** | **WIP NEW 2026-05-14 00:35** |
| **#2694** | **askeladd** | **Charbonnier loss ε=0.5** | **WIP NEW 2026-05-14 00:35** |

**Merged:** 10 | **Closed:** 33 | **WIP:** 8 | **Idle:** 0

## Potential next research directions

### Immediate (high EV, follow from current results)

1. **Continue Lion LR scan** — if 1e-4 wins: try 1.25e-4; if regresses: 7.5e-5 is the sweet spot
2. **Lion beta1 sweep** — after edward's beta1=0.95 result, bracket with beta1=0.85 if 0.95 wins
3. **Combine warmup+higher LR** — if warmup reduces seed variance at 7.5e-5, test it at 1e-4
4. **max_norm=0.25** — continue downward clip scan from 0.5 if fern's rebase wins
5. **SWA (Stochastic Weight Averaging)** — average checkpoints from epochs 25-35; model is compute-bound so the terminal trajectory has quality checkpoints to average

### Medium-term (needs researcher-agent exploration)

6. **Larger model with Lion** — scalar-capacity failed on AdamW; Lion's faster convergence might enable n_hidden=192 at 35 epochs without per-epoch overhead dominating
7. **Per-sample Re embedding** — re_rand and cruise are still the easiest splits; Re-normalized input features may unlock OOD generalization on the harder re-variation axis
8. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens; directly addresses the "surface inherits from volume" structural relationship
9. **Quantile/Pinball loss** — more aggressive median-targeting than Huber for the physically-meaningful pressure channel
10. **Data augmentation** — y-flip + Uy-negation for cruise samples (flow-symmetric BCs admit clean mirror augmentation)

### Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines; requires full architecture rewrite to unlock
- **Cosine LR shape** (T_max, eta_min, warmup-then-flat) — 3 negative results; T_max=50 implicit residual at epoch 35 is load-bearing. Warm restarts (tanjiro #2693) is a different schedule shape (multi-cycle) currently in flight.
- **Noise injection** (dropout, DropPath) — both regressed; regularization stack already saturated
- **Lion beta2** — analogy to AdamW wrong; keep (0.9, 0.99)
- **Lion LR (upper bracket)** — 1e-4 overshoots; 7.5e-5 is the sweet spot. Further downward bisection (6.25e-5) would be wash-zone.
- **Per-channel Huber β** — both directions (β_p=0.25 and β_p=0.625) failed under both optimizer baselines. Global β=0.5 robust.
- **n_head=8** — capacity loss + per-epoch overhead; PhysicsAttention architecture must change to test n_head as redistribution axis
- **QK-RMSNorm** — Q/K magnitudes carry physics-discriminative signal; pre-attention LN is sufficient
- **surf_weight** — fully bracketed (5/10/20); convex, 10 is optimal
- **EMA weights** — EMA-lag on cooling cosine cancels the smoothing; park until schedule axis changes
