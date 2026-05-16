# SENPAI Research State

- **Date:** 2026-05-16 (updated 17:45 — #3415 frieren sent back (log-Re rerun on Lookahead canonical); #4070 alphonse assigned (Lookahead alpha sweep); tanjiro #3497 rebase nudge sent)
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 48.4191`, `test_avg/mae_surf_p (excl cruise) = 47.8034`
  - Achieved via: Huber loss (PR #3155, −18.1%) + LR warmup 1e-3 (PR #3147, −8.9%) + **SOAP optimizer (PR #3283, −31.7%)** + SOAP precond_freq=5 (PR #3495, −1.78%) + **EMA(0.999) (PR #3430, −18.8%)** + EMA decay=0.99 (PR #3591, −3.85%) + Huber beta=0.5 (PR #3316, −6.05%) + Cauchy c=1.0 (PR #3612, −3.67%) + **Huber beta=0.1 (PR #3868, −3.77%)** + **Lookahead k=5 (PR #3947, −4.14%)**
  - Full stack config: SOAP **precondition_frequency=5**, lr=1e-3, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.1 (cauchy_c=0.0)**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**

## Tracked infrastructure issue: cruise-test NaN

`test_geom_camber_cruise/mae_surf_p` returns NaN on any unchanged model. Fix target: `train.py:evaluate_split` — mask samples with non-finite predictions. Deferred to a dedicated small PR. All comparisons use 3-split test mean (excl cruise).

## Tracked config issue: precondition_frequency default

`train.py` default is `precondition_frequency=10` but canonical uses `precondition_frequency=5`. Always pass `--precondition_frequency 5` explicitly. Fixed in BASELINE.md reproduce commands as of 2026-05-16 14:55. Edward #3952 ran freq=10 accidentally — sent back.

## Tracked hardware drift: SOAP eigendecomposition non-determinism

Alphonse's pod reproduces identical config (seed=42, same args) at val=48.823 vs the fern canonical run at val=50.5133 — ~1.7 val drift between GPU machines. Likely SOAP eigendecomposition is non-deterministic across hardware. The within-PR relative delta (e.g. Lookahead −0.83%) remains hardware-controlled and reliable. Absolute BASELINE.md numbers are the reference point, but ±1-2 val drift is expected across pods.

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
| **#3947** | **alphonse** | **Lookahead k=5 on SOAP (freq=5)** | **−4.14%** | **48.4191** |

Old launch baseline: 135.30. Total gain: **−64.2%** over 10 compounding improvements.

## Pending winner (awaiting rebase onto new Lookahead canonical)

| PR | Student | Hypothesis | Best result so far | Notes |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4)** | val=51.0991, test=50.9922 on Huber β=0.5 stack | **Needs rebase onto Lookahead+Huber β=0.1 canonical. Notified.** |

- Expected post-rebase: val ≈ 45 if compounding holds (log-Re is input-side, orthogonal to Lookahead)

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
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 vs 1e-3 | lr=2e-3 +1.0% worse — closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw: T_max=47, LR never reaches floor. Student caught pre-launch. |
| **#3728** | **nezuko** | **EMA decay lower sweep {0.97, 0.95}** | **Monotone worse: 0.99 is global optimum. 0.97: +4.5%, 0.95: +8.2%. Clean negative.** |

## Active WIP experiments (all on Huber β=0.1+EMA+SOAP+Lookahead k=5 canonical, target <48.4191)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) — rerun on Lookahead canonical** | **Inputs** | **WIP — sent back. −1.20% within-PR on Huber β=0.1 (no Lookahead). CONFLICTING — needs rebase + Lookahead add.** |
| **#4070** | **alphonse** | **Lookahead alpha sweep: α ∈ {0.3, 0.5, 0.7}, k=5 fixed** | **Optimization** | **WIP — just assigned. Tuning α of #3947 canonical.** |
| **#4037** | **fern** | **Huber beta lower bound: {0.05, 0.025, 0.01} vs β=0.1** | **Loss tuning** | **WIP — running on Lookahead canonical** |
| **#4021** | **nezuko** | **SWA: uniform late-epoch averaging on top of EMA** | **Training** | **WIP — running on Lookahead canonical** |
| **#3736** | **thorfinn** | **surf_weight {10,5} rerun on Lookahead canonical** | **Loss weighting** | **WIP — running, Lookahead added** |
| **#3497** | **tanjiro** | **Grad-clip {none, clip=1} rerun on Lookahead canonical** | **Optimization** | **WIP — CONFLICTING (needs rebase). Rebase + rerun instructions sent.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1) rerun on Lookahead canonical** | **Loss tuning** | **WIP — sent back. CONFLICTING — needs rebase + freq=5 + Lookahead.** |
| **#3975** | **askeladd** | **bfloat16 autocast: more epochs in 30-min cap** | **Throughput** | **WIP — throughput diagnostic, orthogonal to Lookahead** |

Zero idle students. All 8 pods active.

## Key learnings (cumulative)

1. **Lookahead k=5 compounds with Huber β=0.1 — −4.14% val.** k=5 aligns with precondition_frequency=5 — slow weights average exactly one SOAP preconditioner refresh window. OOD splits benefit most (re_rand, camber_cruise). EMA and Lookahead non-redundant (different timescales).
2. **Huber β=0.1 beats Cauchy c=1.0 by −3.77%.** Monotone trend extends through full range {2.0, 1.0, 0.5, 0.25, 0.1}. Pure L1 regime (β→0) may be optimal — next sweep tests β={0.05, 0.025, 0.01}.
3. **Cauchy loss was superseded.** Huber β=0.1 outperforms Cauchy c=1.0 — SOAP's adaptive preconditioning handles curvature, making Cauchy's redescending influence redundant.
4. **Log-Re sinusoidal embedding works (within-PR).** freqs=4 gives −6.23% val on Huber β=0.5 stack. Awaiting rebase onto new canonical.
5. **EMA decay=0.99 is global optimum (confirmed).** Lower decay monotonically hurts. 0.99 ≈ 100-step EMA horizon is optimal for 14-epoch budget.
6. **surf_weight=5 beats sw=10 within-PR on Cauchy stack.** Awaiting confirmation on new Lookahead canonical.
7. **Grad-clip=1.0 has real within-PR signal on Cauchy stack (−3.79%).** Awaiting Huber β=0.1 + Lookahead rerun.
8. **Log-pressure aux loss: modest signal on wrong canonical (−1.13% within-PR).** Awaiting rerun on Huber+Lookahead. Mechanism partially confirmed (val_re_rand improves, but val_camber_rc regresses).
9. **SOAP + small Huber β handles noisier gradient signal robustly.** Lookahead further smooths stale-curvature noise between preconditioner refreshes.
10. **Hardware drift: ~1.7 val across GPU machines on identical seed/config.** Likely SOAP eigendecomposition. Relative within-PR deltas are reliable; absolute BASELINE.md numbers are reference targets.
11. **Wall-clock cap is binding.** Best epoch consistently = 14. bf16 is primary lever to increase effective epoch count.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re + Lookahead canonical rebase (#3415).** −6.23% within-PR on Huber β=0.5. Log-Re is input-side, orthogonal to all optimizer/loss changes. **Highest-EV pending result** — expected val ≈ 45 if compounding holds.
- **Fern Huber β lower bound (#4037).** β={0.05, 0.025, 0.01} on full new canonical (Lookahead + Huber β=0.1). If β→0 still wins, next step is pure MAE.
- **Nezuko SWA (#4021).** First test of SWA on top of EMA + Lookahead. Three timescales: EMA (per-step), Lookahead (k-step), SWA (epoch-scale). Expected 1-3% if flat-minima mechanism holds.
- **Thorfinn surf_weight (#3736).** sw=5 expected to beat sw=10 with full Lookahead canonical.
- **Tanjiro grad-clip (#3497).** Huber β=0.1's noisier L1-dominant gradients should amplify clip=1's value.
- **Edward log-pressure (#3952).** −1.13% within-PR on wrong canonical. Rerunning on new canonical with Lookahead.
- **Askeladd bf16 (#3975).** Throughput diagnostic — if ≥1.2x speedup, unlocks more epochs for everyone.

### Post-sweep stack (after pending WIPs land)
1. **Lookahead k sweep (k ∈ {3, 7, 10})** — verify U-shape, confirm k=5 is optimal. After alpha sweep resolves.
2. **Pure MAE (L1 loss):** if Huber β=0.01 still improves, switch to `nn.L1Loss` directly.
3. **Adaptive Barron loss:** learn α and c per-output-channel. Removes loss hyperparameter.
4. **SAM on SOAP:** sharpness-aware minimization. Doubles forward cost — only viable after bf16.
5. **Divergence-free auxiliary loss:** incompressibility penalty on velocity field.
6. **Longer schedule:** if bf16 gives more epochs, retry experiments with 20+ effective epochs.
