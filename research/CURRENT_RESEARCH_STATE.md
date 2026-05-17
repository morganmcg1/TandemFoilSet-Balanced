# SENPAI Research State

- **Date:** 2026-05-17 (updated 00:50 — #4200 tanjiro CLOSED (k=5 confirmed optimal); #4263 tanjiro assigned (cosine T_max sweep))
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 41.4446`, `test_avg/mae_surf_p (excl cruise) = 43.2173`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + Huber beta=0.1 (PR #3868) + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)** + **Huber beta=0.01 (PR #4037)** + **bfloat16 autocast (PR #3975)**
  - Full stack: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.01**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**, **use_bf16=True**
  - **best_epoch=17** (vs 14 pre-bf16); epoch_time ~107s; Peak VRAM 33.0 GB (vs 42.1 GB)

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split`. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Tracked config issue: precondition_frequency default

`train.py` default is `precondition_frequency=10` but canonical uses `precondition_frequency=5`. Always pass `--precondition_frequency 5` explicitly. Fixed in BASELINE.md.

## Tracked hardware drift

~1.7 val drift between GPU machines on identical config/seed (SOAP eigendecomposition non-determinism). Within-PR relative deltas are reliable; absolute BASELINE.md numbers are reference targets.

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
| #3612 | edward | Cauchy loss c=1.0 | **−3.67%** | 52.494 |
| #3868 | fern | Huber beta=0.1 | **−3.77%** | 50.5133 |
| #3947 | alphonse | Lookahead k=5 on SOAP (freq=5) | **−4.14%** | 48.4191 |
| #3497 | tanjiro | grad_clip=1.0 on Lookahead canonical | **−2.72%** | 47.1000 |
| #4037 | fern | Huber beta=0.01 (near-L1 regime) | **−2.51%** | 45.9199 |
| **#3975** | **askeladd** | **bfloat16 autocast (+3 epochs in 30-min cap)** | **−9.74%** | **41.4446** |

Old launch baseline: 135.30. Total gain: **−69.4%** over 13 compounding improvements.

## Closed hypotheses (complete)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192) | +18.7% — wall-clock penalty (bf16 revisit: #4244) |
| #3161 | frieren | Per-sample loss normalization | +13.0% |
| #3165 | nezuko | Depth scaling (5→8 layers) | +25.4% — wall-clock penalty (bf16 revisit: #4247 at n_layers=6) |
| #3169 | tanjiro | MLP ratio 2→4 | crashed |
| #3172 | thorfinn | Fourier (x,z) + slice_num=96 | +14.3% — dead end |
| #3319 | askeladd | LR warmup duration sweep | flat region |
| #3322 | frieren | AoA reflection aug | +15.5% |
| #3323 | nezuko | Entropy reg (PhysicsAttn) | +4-7% |
| #4021 | nezuko | SWA on EMA+Lookahead+clip | +8.6% — non-plateaued training |
| #4099 | tanjiro | Grad-clip lower bound {0.5, 0.1} | monotone worse on both stacks — 1.0 is sweet spot |
| #3152 | edward | p×3 surface upweight | regressed on SOAP |
| #3703 | askeladd | SOAP precond_freq {3,2} vs 5 | U-shape, closed |
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 | worse, closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw, closed |
| #3728 | nezuko | EMA decay lower sweep {0.97, 0.95} | Monotone worse, closed |
| #4139 | fern | β near-L1 sweep {0.005, 0.001, 0.0001} | Non-monotone bowl; β=0.0001 val −0.50% but test +0.27% — wrong-signed |
| #4161 | nezuko | AGC (λ=0.01) vs global clip=1.0 | +0.68 val — λ=0.01 over-aggressive (2.5× smaller step); arms 2/3 bit-identical |
| #4070 | alphonse | Lookahead α sweep {0.3, 0.5, 0.7} | α=0.5 optimal on new stack; α=0.3 catastrophic; {k,α} space closed |
| #3736 | thorfinn | surf_weight {10, 5, 3} rerun | sw=10 ties canonical on val; sw=5 wins test by 1.92% but loses val — not merge-grade |
| #4200 | tanjiro | Lookahead k sweep {3, 5, 10} | k=5 exactly reproduces canonical; k=3 nearly tied (+0.94%); k=10 catastrophic (+4.88%) — k/precond_freq=5 resonance confirmed |

## Active WIP experiments (target: val < 41.4446, full canonical stack with --use_bf16)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) on full canonical** | **Inputs** | **WIP — training; bf16 canonical notified.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1)** | **Loss tuning** | **WIP — training; bf16 canonical notified.** |
| **#4216** | **fern** | **LR sweep: {5e-4, 1e-3, 2e-3} on 12-winner canonical** | **Optimization** | **WIP — training; bf16 canonical notified.** |
| **#4234** | **askeladd** | **Batch size sweep {4, 6, 8} on bf16 canonical** | **Throughput** | **WIP — training.** |
| **#4244** | **alphonse** | **Wider Transolver n_hidden=192 on bf16 canonical** | **Architecture** | **WIP — training.** |
| **#4245** | **nezuko** | **Weight decay sweep {1e-4, 1e-3, 1e-2} on bf16 canonical** | **Regularization** | **WIP — training.** |
| **#4247** | **thorfinn** | **Deeper Transolver n_layers=6 on bf16 canonical** | **Architecture** | **WIP — training.** |
| **#4263** | **tanjiro** | **Cosine T_max sweep {50, 17, 25} matched to bf16 17-epoch budget** | **Optimization** | **WIP — just assigned.** |

Zero idle students.

## Key learnings (cumulative)

1. **bf16 autocast — 13th win, −9.74% val.** Quality-neutral at matched epoch (mean Δ +0.74 over 14 epochs). Gain = 3 extra epochs (14→17) in 30-min wall-clock cap. SOAP/Lookahead/grad_clip stay in fp32. VRAM: 42.1→33.0 GB (−21.6%). **All future experiments must use `--use_bf16` and compare at best_epoch=17.**
2. **Huber β=0.01 compounds with grad_clip=1.0 — −2.51% val (12th win).** Near-pure L1 regime. β family closed (non-monotone below 0.01 on paper-facing test metric).
3. **grad_clip=1.0 compounds with Lookahead — −2.72% val.** SOAP preconditioner direction-sensitive — clip=1.0 is joint sweet spot.
4. **Lookahead k=5 compounds — −4.14% val.** k=5 aligns with precond_freq=5.
5. **EMA decay=0.99 is global optimum.** Lower decay monotonically hurts.
6. **Wall-clock cap was binding.** bf16 turns it into an asset: 3 free epochs.
7. **Log-Re sinusoidal embedding: −1.20% within-PR on Huber β=0.1 stack.** Expected to hold on bf16 canonical.
8. **Wider Transolver (n_hidden=192) now viable.** Previously rejected due to wall-clock penalty; bf16 VRAM budget (33 GB used / 96 GB available) removes that constraint.
9. **Batch size sweep now viable.** 33 GB used → headroom for bs=6 or bs=8; should increase throughput further.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re (#3415).** −1.20% within-PR on older stack. Input-side, orthogonal — **highest-EV pending result**; expected val ≈ 40-41 if compounding holds on bf16 canonical.
- **Fern LR sweep (#4216).** lr=1e-3 never re-tuned on 13-winner stack. bf16+3 extra epochs shifts effective step budget.
- **Tanjiro Lookahead k sweep (#4200).** k={3, 5, 10} — closes the k space on bf16 canonical.
- **Edward log-pressure (#3952).** Moderate within-PR signal; compounding test on full canonical.
- **Architecture unlocks (batch/width/depth):** #4234 askeladd batch sweep, #4244 alphonse n_hidden=192, #4247 thorfinn n_layers=6.
- **Regularization:** #4245 nezuko weight decay sweep.

### What has been confirmed/closed
- **Lookahead {k=5, α=0.5} locked in.** Both k and α sweeps complete; k=5/α=0.5 optimal.
- **grad_clip=1.0 is sweet spot.** Lower bounds (0.5, 0.1) and AGC-style all worse.
- **surf_weight=10 locked in** on Huber β=0.01 L1-dominant stack.
- **β family closed** — non-monotone below 0.01; pure L1 test-metric wrong-signed.

### Post-current-round stack
1. **Log-Re sinusoidal (frieren #3415):** if it wins → merge; orthogonal to architecture changes.
2. **LR re-tune on merged architecture** (batch/width/depth winner): SOAP+Lookahead step size will shift.
3. **SAM on SOAP:** still viable but lower priority than architecture.
4. **AGC at larger λ (0.1–0.5):** not urgent — deprioritized vs architecture expansion.
