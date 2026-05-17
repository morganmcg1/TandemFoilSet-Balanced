# SENPAI Research State

- **Date:** 2026-05-17 10:05 — #4296 thorfinn MERGED (slice_num=32, val=31.9978, −7.42%, **17th winner**); #4421 fern CLOSED (warmup=1 beats T_max=25 canonical but superseded); #4234 askeladd CLOSED (batch size failed on old stack); #3952 edward CLOSED (log-p aux negative on canonical); #4502 tanjiro assigned LR retune, #4504 nezuko EMA sweep. **4 idle students need assignments.**
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 31.9978`, `test_avg/mae_surf_p (excl cruise) = 32.017`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + Huber beta=0.1 (PR #3868) + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)** + **Huber beta=0.01 (PR #4037)** + **bfloat16 autocast (PR #3975)** + **cosine T_max=25 (PR #4263)** + **lr=2e-3 (PR #4336)** + cosine T_max=20 (PR #4447) + **slice_num=32 (PR #4296)**
  - Full stack: SOAP **precondition_frequency=5**, **lr=2e-3**, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.01**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**, **use_bf16=True**, **cosine_t_max=25**, **slice_num=32**
  - **best_epoch=21** (slice_num=32 fits more epochs in 30-min cap); epoch_time ~87s; Peak VRAM ~33.0 GB
  - Note: T_max reverts to 25 (was 20). At 21 epochs, T_max=20 over-cools; T_max=25 gives LR≈17% of peak at epoch 21. T_max retune at slice_num=32 is a high-priority follow-up.

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
| **#4263** | **tanjiro** | **cosine T_max=25 (schedule aligned to bf16 budget)** | **−8.47%** | **37.9354** |
| **#4336** | **tanjiro** | **lr=2e-3 on T_max=25 canonical** | **−6.33%** | **35.5322** |
| **#4447** | **tanjiro** | **cosine T_max=20 (more aggressive cooldown)** | **−2.72%** | **34.5662** |
| **#4296** | **thorfinn** | **slice_num=32 (coarser attention, +4 free epochs)** | **−7.42%** | **31.9978** |

Old launch baseline: 135.30. Total gain: **−76.3%** over 17 compounding improvements.

## Closed hypotheses (complete)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192) | +18.7% — wall-clock penalty |
| #4244 | alphonse | n_hidden=192 on bf16 canonical | +5.55% val — wall-clock binding |
| #3161 | frieren | Per-sample loss normalization | +13.0% |
| #3165 | nezuko | Depth scaling (5→8 layers) | +25.4% — wall-clock penalty |
| #4247 | thorfinn | Deeper Transolver n_layers=6 on bf16 canonical | +9.78% val — schedule/LR bottleneck |
| #3169 | tanjiro | MLP ratio 2→4 | crashed (fp32 OOM) |
| #3172 | thorfinn | Fourier (x,z) + slice_num=96 | +14.3% — dead end |
| #3319 | askeladd | LR warmup duration sweep | flat region |
| #3322 | frieren | AoA reflection aug | +15.5% |
| #3323 | nezuko | Entropy reg (PhysicsAttn) | +4-7% |
| #4021 | nezuko | SWA on EMA+Lookahead+clip | +8.6% — non-plateaued training |
| #4099 | tanjiro | Grad-clip lower bound {0.5, 0.1} | monotone worse — 1.0 is sweet spot |
| #3152 | edward | p×3 surface upweight | regressed on SOAP |
| #3703 | askeladd | SOAP precond_freq {3,2} vs 5 | U-shape, closed |
| #3493 | alphonse | SOAP LR sweep: lr=2e-3 | worse, closed |
| #3926 | askeladd | Cosine LR floor (eta_min) | Design flaw, closed |
| #3728 | nezuko | EMA decay lower sweep {0.97, 0.95} | Monotone worse, closed |
| #4139 | fern | β near-L1 sweep {0.005, 0.001, 0.0001} | Non-monotone bowl; β=0.0001 wrong-signed |
| #4161 | nezuko | AGC (λ=0.01) vs global clip=1.0 | +0.68 val — over-aggressive |
| #4070 | alphonse | Lookahead α sweep {0.3, 0.5, 0.7} | α=0.5 optimal; {k,α} space closed |
| #3736 | thorfinn | surf_weight {10, 5, 3} rerun | sw=10 ties val; sw=5 wins test but loses val |
| #4200 | tanjiro | Lookahead k sweep {3, 5, 10} | k=5 exactly canonical; k=3 near-tied; k=10 catastrophic |
| #4305 | fern | mlp_ratio revisit {3, 4} on bf16 canonical | Both regress val; mlp_ratio=2 stays canonical |
| #4245 | nezuko | weight_decay sweep {1e-4, 1e-3, 1e-2} | val/test divergence — wd=1e-4 stays canonical |
| #4359 | fern | warmup_epochs sweep {1, 5} at lr=1e-3 | Strong signal: warmup=1 wins; re-tested at lr=2e-3 |
| #4388 | tanjiro | LR push above 2e-3 at T_max=25 | Both arms regress — ceiling was T_max-dependent |
| #4423 | nezuko | Dropout sweep {0.05, 0.1} | Both regress consistently; dropout closed |
| #4234 | askeladd | Batch size sweep {4, 6, 8} | Failed on old stack; bs=4 optimal without LR scaling |
| #3952 | edward | Log-pressure aux loss (logp_weight=0.1) | Negative on β=0.01 canonical; log-p aux closed for this formulation |
| #4421 | fern | Warmup=1 retest at lr=2e-3 | Beats T_max=25 canonical (−3.18%) but superseded by slice_num=32 |

## Active WIP experiments (target: val < 31.9978, new canonical: --use_bf16 --lr 2e-3 --cosine_t_max 25 --slice_num 32)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) on full canonical** | **Inputs** | **WIP — notified of new canonical; should use slice_num=32 + T_max=25.** |
| **#4348** | **alphonse** | **Attention head sweep {2, 8} on bf16 canonical** | **Architecture** | **WIP — stale stack (T_max=20, slice_num=64); instructed to finish and report.** |
| **#4502** | **tanjiro** | **LR re-tune at T_max=20 {2.5e-3, 3e-3}** | **Optimization** | **WIP — stale stack; instructed to finish arms and report relative signal.** |
| **#4504** | **nezuko** | **EMA decay sweep {0.995, 0.999} at T_max=20** | **Regularization** | **WIP — stale stack; instructed to finish arms and report relative signal.** |

## Newly assigned experiments (this tick, ~10:10 UTC)

| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| **#4538** | **askeladd** | **warmup=1 + slice_num=32** (compound warmup × architecture) | ~−3% (confirmed warmup mechanism) |
| **#4539** | **edward** | **T_max retune {25, 30, 35} at slice_num=32** (21-epoch calibration) | ~−1-3% (T_max sweet spot) |
| **#4540** | **fern** | **LR push {2.5e-3, 3e-3} at slice_num=32** (LR ceiling re-test) | Unknown — previously failed at slice_num=64 |
| **#4541** | **thorfinn** | **slice_num=16/24 finer sweep** (probe lower boundary) | ~−2-5% (direction confirmed <32 potential) |

## Key learnings (cumulative)

1. **bf16 autocast — 13th win, −9.74% val.** Quality-neutral at matched epoch. Gain = 3 extra epochs (14→17) in 30-min wall-clock cap. VRAM: 42.1→33.0 GB. **All future experiments must use `--use_bf16`.**
2. **Huber β=0.01 compounds with grad_clip=1.0 — −2.51% val (12th win).** Near-pure L1 regime. β family closed.
3. **grad_clip=1.0 compounds with Lookahead — −2.72% val.** SOAP preconditioner direction-sensitive — clip=1.0 is sweet spot.
4. **Lookahead k=5 compounds — −4.14% val.** k=5 aligns with precond_freq=5.
5. **EMA decay=0.99 is global optimum.** Lower decay monotonically hurts.
6. **Wall-clock cap was binding.** bf16 turns it into an asset: 3 free epochs; slice_num=32 adds 4 more.
7. **Log-Re sinusoidal embedding: −1.20% within-PR on Huber β=0.1 stack.** Expected to hold on new canonical (input-side, orthogonal).
8. **Width (n_hidden=192) closed under 30-min cap.** Wall-clock binding; VRAM headroom insufficient to overcome epoch penalty.
9. **Batch size sweep requires linear LR scaling.** bs=4 optimal at lr=1e-3; larger batch without proportional LR under-trains in 30-min.
10. **Cosine T_max=25 — 14th win, −8.47% val.** T_max=25 gives 22-epoch cosine window; at epoch 17 LR ≈ 29% of peak.
11. **LR=2e-3 — 15th win, −6.33% val.** T_max=25 unlocked LR ceiling: monotone 1e-3→1.5e-3→2e-3.
12. **LR ceiling is T_max-dependent.** At T_max=25, ceiling is 2e-3. At T_max=20, ceiling may differ. After slice_num=32 (T_max=25, 21 epochs), LR re-tune is high priority.
13. **cosine T_max=20 — 16th win, −2.72% val.** But reverted after slice_num=32 merge: at 21 epochs, T_max=20 over-cools. **T_max retune at slice_num=32 (try T_max ∈ {25, 30, 35}) is high-priority.**
14. **Dropout closed (#4423).** Both 0.05 and 0.10 regress. 17-epoch budget too short; consistent pattern: capacity-reducing regularization fails under 30-min cap.
15. **slice_num=32 — 17th win, −7.42% val (MASSIVE).** Over-segmentation at 64 slices. Coarser 32-slice grouping better for 0.66M model. Key compounding effect: slice_num=32 fits 21 epochs (not 17) in 30-min cap, adding 4 free cosine cooldown epochs. Direction confirmed: 96 >> 64 >> 32 for this model size. **All future experiments use `--slice_num 32`.**
16. **warmup=1 beats warmup=3 by −3.18% at T_max=25/slice_num=64.** Orthogonal to architecture; likely to compound with slice_num=32. Mechanism: 2 extra peak-LR epochs absorbed by grad_clip+SOAP. The validated gain exists but was superseded before merging.

## Next directions (priority order, new canonical: T_max=25 + slice_num=32)

### Immediate (active)
- **Frieren log-Re (#3415):** Input-side feature, orthogonal — highest-EV pending WIP if run on new canonical.
- **Alphonse n_head sweep (#4348):** {2, 8} on old stack; within-PR signal still informative.
- **Tanjiro LR retune (#4502):** LR ceiling at T_max=20/slice64; relative signal guides follow-up.
- **Nezuko EMA sweep (#4504):** EMA {0.995, 0.999} at T_max=20/slice64; relative signal guides follow-up.

### New assignments needed (4 idle students, ~2h to launch end)

1. **warmup=1 on slice_num=32 + T_max=25** (askeladd) — confirmed −3.18% mechanism, likely compounds. Target: val ~30-31.
2. **T_max retune at slice_num=32** (edward) — student suggestion: T_max ∈ {25, 30, 35}. At 21 epochs, T_max=30 may be optimal (LR at epoch 21 = ~5% of peak). Target: val ~30-31.
3. **slice_num=16/24 on T_max=25 canonical** (thorfinn) — student's own suggestion: direction confirmed (<64 > 64 > 96), may be even lower optimum. Quick 2-arm sweep.
4. **LR push on slice_num=32 + T_max=25** (fern) — LR=2e-3 was ceiling at T_max=25/slice64. With slice_num=32 + 21 epochs, the LR ceiling may shift. Try lr ∈ {2.5e-3, 3e-3}.

### Post-round high-priority ideas

- **Activation function sweep (GELU/SiLU/SwiGLU):** Never tested. GELU is a common improvement over ReLU for physics-informed Transformers. Low-cost, orthogonal.
- **n_hidden retune at slice_num=32:** slice_num=32 changes the attention capacity; n_hidden=192 may now fit in VRAM without wall-clock penalty (33 GB headroom).
- **Cosine T_max after architecture settles:** T_max retune at slice_num=32 (in-flight via edward).
- **LR after warmup=1 settles:** warmup=1 + LR push likely compound further.
- **Physical-space log-pressure aux loss:** log(|p_phys|+ε) rather than normalized — different from the closed experiment, may avoid the redundancy with Huber β=0.01.

### Confirmed closed
- **Lookahead {k=5, α=0.5} locked in.** Both sweeps complete.
- **grad_clip=1.0 is sweet spot.** Lower and AGC worse.
- **surf_weight=10 locked in** on Huber β=0.01 L1-dominant stack.
- **β family closed** — non-monotone below 0.01.
- **Depth (n_layers=6) closed** under 30-min cap.
- **LR push above 2e-3 at T_max=25 closed (#4388).** Ceiling is T_max-dependent.
- **Dropout closed (#4423).** Budget too short.
- **Batch size (no LR scaling) closed (#4234).** bs=4 optimal without proportional LR.
- **Log-pressure aux loss closed (#3952).** Redundant with Huber β=0.01 on current stack.
