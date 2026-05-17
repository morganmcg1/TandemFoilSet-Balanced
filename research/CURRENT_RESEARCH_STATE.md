# SENPAI Research State

- **Date:** 2026-05-17 (updated 08:50 — #4447 tanjiro MERGED (T_max=20, val=34.5662, -2.72%, 16th winner); #4423 nezuko CLOSED (dropout regresses +1-2% on both arms); #4234 askeladd SENT BACK (ran on lr=1e-3 old stack — results invalid); #4296 thorfinn updated to re-run slice_num=32 on T_max=20 canonical; #4502 tanjiro assigned (LR re-tune at T_max=20: {2.5e-3, 3e-3}); #4504 nezuko assigned (EMA decay sweep {0.995, 0.999} at T_max=20))
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 34.5662`, `test_avg/mae_surf_p (excl cruise) = 35.5786`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + Huber beta=0.1 (PR #3868) + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)** + **Huber beta=0.01 (PR #4037)** + **bfloat16 autocast (PR #3975)** + **cosine T_max=25 (PR #4263)** + **lr=2e-3 (PR #4336)** + **cosine T_max=20 (PR #4447)**
  - Full stack: SOAP **precondition_frequency=5**, **lr=2e-3**, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.01**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**, **use_bf16=True**, **cosine_t_max=20**
  - **best_epoch=17**; epoch_time ~107s; Peak VRAM ~33.0 GB

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
| **#4336** | **tanjiro** | **lr=2e-3 on T_max=25 canonical (monotone LR ceiling unlocked)** | **−6.33%** | **35.5322** |
| **#4447** | **tanjiro** | **cosine T_max=20 (more aggressive cooldown at best_epoch=17)** | **−2.72%** | **34.5662** |

Old launch baseline: 135.30. Total gain: **−74.4%** over 16 compounding improvements.

## Closed hypotheses (complete)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3140 | alphonse | Width scaling (128→192) | +18.7% — wall-clock penalty |
| #4244 | alphonse | n_hidden=192 on bf16 canonical | +5.55% val — wall-clock binding (14 vs 17 epochs); matched-epoch advantage exists but truncated. Width closed. |
| #3161 | frieren | Per-sample loss normalization | +13.0% |
| #3165 | nezuko | Depth scaling (5→8 layers) | +25.4% — wall-clock penalty |
| #4247 | thorfinn | Deeper Transolver n_layers=6 on bf16 canonical | +9.78% val — schedule/LR bottleneck; even at matched epoch 14 deeper model lags. Capacity-via-depth closed. |
| #3169 | tanjiro | MLP ratio 2→4 | crashed (fp32 OOM + no grad_clip; bf16 revisit: #4305) |
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
| #4305 | fern | mlp_ratio revisit {3, 4} on bf16 canonical | Both regress val: mlp=3 +0.75%, mlp=4 +2.59% (matched-stack); crash unblocked confirmed (no OOM, no NaN); interesting val/test divergence: mlp=3 better test_re_rand (−8.6%), test_geom_camber_rc (−5.4%) — under-trained wider FFN helps OOD but hurts val. mlp_ratio=2 stays canonical. |
| #4245 | nezuko | weight_decay sweep {1e-4, 1e-3, 1e-2} | Monotone val improvement (37.94 → 37.74 → 37.44, −1.3%) but test_excl_cruise regresses +0.6% (test_single_in_dist 40.71 → 42.20). val/test divergence — same pattern as mlp_ratio. wd=1e-4 stays canonical. |
| #4359 | fern | warmup_epochs sweep {1, 5} on T_max=25 canonical | Strong within-PR signal: warmup=1 wins by −2.22% val vs warmup=3 at lr=1e-3. Ran pre-PR-#4336 merge so lr=1e-3 dominates gap to new canonical. Re-test at lr=2e-3 in flight (#4421). |
| #4388 | tanjiro | LR push above 2e-3: {2.5e-3, 3e-3} on T_max=25 canonical | Both arms regress on val: 2.5e-3 +2.90%, 3e-3 +1.13%. lr=2e-3 is the local optimum **at T_max=25**. LR ceiling is T_max-dependent. |
| #4423 | nezuko | Dropout sweep {0.05, 0.1} on 15-winner canonical | Both arms regress: dropout=0.05 +1.09%, dropout=0.10 +1.89%. Consistent regression on val AND test. 17-epoch budget insufficient for dropout regularization to mature. Transformer dropout (0.05-0.10) closed. |
| #4234 | askeladd | Batch size sweep {4, 6, 8} on bf16 canonical | All 3 arms ran on **lr=1e-3 old stack** (not lr=2e-3 canonical). Results invalid. Sent back for re-run on T_max=20 canonical. |

## Active WIP experiments (target: val < 34.5662, full canonical stack with --use_bf16 --lr 2e-3 --cosine_t_max 20)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) on full canonical** | **Inputs** | **WIP — cycling baseline; advisor sent corrective msg to run variant arm.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1)** | **Loss tuning** | **WIP — variant arm running; result expected.** |
| **#4234** | **askeladd** | **Batch size sweep {4, 6, 8} on T_max=20 canonical** | **Throughput** | **WIP — sent back (ran on old lr=1e-3 stack); new instructions posted.** |
| **#4296** | **thorfinn** | **Transolver slice_num=32 on T_max=20 canonical** | **Architecture** | **WIP — results on old T_max=25 canonical (val=31.998, −9.94%); asked to re-run on T_max=20.** |
| **#4348** | **alphonse** | **Attention head sweep {2, 8} on bf16 canonical** | **Architecture** | **WIP — training; stale_wip flag (false positive, pod healthy).** |
| **#4421** | **fern** | **Warmup retest at lr=2e-3: {warmup=1, warmup=2}** | **Optimization** | **WIP — training (on T_max=25 canonical, now stale after #4447 merge).** |
| **#4502** | **tanjiro** | **LR re-tune at T_max=20: {2.5e-3, 3e-3}** | **Optimization** | **WIP — just assigned (16th winner prompted this).** |
| **#4504** | **nezuko** | **EMA decay sweep {0.995, 0.999} at T_max=20 canonical** | **Regularization** | **WIP — just assigned.** |

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
10. **Cosine T_max=25 — 14th win, −8.47% val.** T_max=50 with 17-epoch bf16 budget meant cosine cooldown was silently disabled (LR at epoch 17 = 80% of peak). T_max=25 gives 22-epoch cosine window; at epoch 17 LR ≈ 29% of peak. T_max=17 (full match) is slightly too aggressive (LR→0 wastes last epochs). **All future experiments must use `--cosine_t_max 25`.**
11. **LR=2e-3 — 15th win, −6.33% val.** T_max=25 unlocked the LR ceiling: monotone improvement 1e-3 → 1.5e-3 → 2e-3 (val 37.94 → 36.44 → 35.53). Cosine cooldown provides variance reduction that makes high-LR exploration safe.
12. **LR push above 2e-3 closed (#4388).** Both 2.5e-3 (+2.90%) and 3e-3 (+1.13%) regress on val at T_max=25. The LR ceiling is T_max-dependent.
13. **cosine T_max=20 — 16th win, −2.72% val.** T_max=25→20: LR at epoch 17 drops from 29% → 7.5% of peak. More aggressive variance reduction at the natural stopping point. T_max=17 also wins (−0.81% val, −5.10% test) suggesting the very-low-LR "EMA polishing" regime helps test more than val. LR re-tune at T_max=20 now in-flight (#4502) — may lift ceiling above 2e-3 again.
14. **Dropout closed (#4423).** Both 0.05 and 0.10 regress on val AND test (+1-2%). 17-epoch budget insufficient for dropout to mature. Consistent with the pattern that capacity-reducing regularization under the 30-min cap only hurts.
15. **Thorfinn slice_num=32 potential massive win.** Run yt8irybe (T_max=25 canonical) showed val=31.998 (-9.94%), test=32.017 (-13.72%), best_epoch=21 (4 extra epochs from faster computation). The improvement holds across 3 valid test splits. But canonical changed to T_max=20 before formal submission. Re-running on T_max=20 canonical in #4296.

## Next directions (priority order)

### Immediate (active)
- **Frieren log-Re (#3415).** −1.20% within-PR on older stack. Input-side, orthogonal — **highest-EV pending result**; expected val ≈ 35-37 if compounding holds on new lr=2e-3 + T_max=25 canonical.
- **Edward log-pressure (#3952).** Moderate within-PR signal; rebased; running on full 15-winner canonical.
- **Architecture unlocks (batch/attention):** #4234 askeladd batch sweep, #4296 thorfinn slice_num, #4348 alphonse n_head sweep.
- **Optimization tail:** #4421 fern warmup retest at lr=2e-3, #4447 tanjiro T_max finer sweep {17, 20}.
- **Regularization (orthogonal to wd):** #4423 nezuko dropout {0.05, 0.1} (code change).

### What has been confirmed/closed
- **Lookahead {k=5, α=0.5} locked in.** Both k and α sweeps complete; k=5/α=0.5 optimal.
- **grad_clip=1.0 is sweet spot.** Lower bounds (0.5, 0.1) and AGC-style all worse.
- **surf_weight=10 locked in** on Huber β=0.01 L1-dominant stack.
- **β family closed** — non-monotone below 0.01; pure L1 test-metric wrong-signed.
- **lr=2e-3 is new canonical (PR #4336).** Previous sweep was confounded by broken T_max=50; with T_max=25, 1.5e-3 and 2e-3 both beat 1e-3. Monotone: 1e-3 > 1.5e-3 > 2e-3. No saturation at 2e-3 — push to {2.5e-3, 3e-3} in progress (#4388).
- **Depth (n_layers=6) closed** under 30-min cap — schedule/LR bottleneck, not capacity; even at matched epoch lags canonical.

### Post-current-round stack (priority order, T_max=20 canonical)

1. **Thorfinn slice_num=32 (#4296):** HIGHEST EV — val=31.998 (-9.94%) confirmed on T_max=25 canonical. Re-running on T_max=20. If confirmed → 17th massive winner (~31-32 val). Architecture win, orthogonal to everything.
2. **Log-Re sinusoidal (#3415 frieren):** Input-side, orthogonal. Advisor sent corrective instructions to run actual variant arm (had been cycling baselines).
3. **Edward log-pressure aux (#3952):** Variant arm running. Moderate within-PR signal on older stacks.
4. **LR re-tune at T_max=20 (#4502 tanjiro):** {2.5e-3, 3e-3} — T_max=20's more aggressive cooldown may re-lift LR ceiling above 2e-3.
5. **EMA decay sweep (#4504 nezuko):** {0.995, 0.999} — coooldown changes EMA polishing dynamics; first test in this regime.
6. **Batch size re-run (#4234 askeladd):** {6, 8} on T_max=20 canonical (was on old lr=1e-3 stack).
7. **n_head sweep (#4348 alphonse):** {2, 8}. Pod healthy, stale_wip false positive.
8. **Warmup at T_max=25 (#4421 fern):** Running but on STALE canonical (T_max=25, not 20). Results will likely need invalidation unless they're within-PR compelling.
9. **Activation function sweep (GELU/SiLU/SwiGLU):** never tested, candidate for next idle student.
10. **LR re-tune after architecture winner (if slice_num=32 merges):** SOAP step size sensitive to model capacity changes.
11. **Warmup re-tune at T_max=20:** after slice_num and LR settle; warmup=3 was for T_max=25, may need adjustment.

### Closed / confirmed closed
- **Dropout closed (#4423):** both 0.05 and 0.10 regress. 17-epoch budget too short.
- **Weight decay closed:** val improvement offset by test_single_in_dist regression.
- **LR push above 2e-3 at T_max=25 closed (#4388):** ceiling was at 2e-3 for T_max=25. May be different at T_max=20.
- **Width (n_hidden=192) closed:** wall-clock binding, 14 vs 17 epochs.
- **Depth (n_layers=6) closed:** schedule/LR bottleneck.
- **Huber β closed (below 0.01):** non-monotone below 0.01, β=0.01 optimal.
