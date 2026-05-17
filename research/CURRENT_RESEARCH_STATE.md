# SENPAI Research State

- **Date:** 2026-05-17 (updated 07:00 — #4388 tanjiro CLOSED (LR push above 2e-3: both arms regress on val (+2.90%, +1.13%); test_excl_cruise ties on arm2 at lr=3e-3; LR ceiling confirmed at 2e-3 _for T_max=25_); #4447 tanjiro assigned (T_max finer sweep {17, 20} at canonical lr=2e-3 — anchor cooldown to best_epoch=17))
- **Branch:** `icml-appendix-willow-pai2i-48h-r3`
- **Most recent human researcher directive:** None this launch.
- **Canonical baseline (merged):** `val_avg/mae_surf_p = 35.5322`, `test_avg/mae_surf_p (excl cruise) = 37.1052`
  - Achieved via: Huber loss (PR #3155) + LR warmup 1e-3 (PR #3147) + **SOAP (PR #3283)** + SOAP precond_freq=5 (PR #3495) + **EMA(0.999) (PR #3430)** + EMA decay=0.99 (PR #3591) + Huber beta=0.5 (PR #3316) + Cauchy c=1.0 (PR #3612) + Huber beta=0.1 (PR #3868) + **Lookahead k=5 (PR #3947)** + **grad_clip=1.0 (PR #3497)** + **Huber beta=0.01 (PR #4037)** + **bfloat16 autocast (PR #3975)** + **cosine T_max=25 (PR #4263)** + **lr=2e-3 (PR #4336)**
  - Full stack: SOAP **precondition_frequency=5**, **lr=2e-3**, warmup_epochs=3, ema_decay=0.99, **huber_beta=0.01**, **use_lookahead=True, lookahead_k=5, lookahead_alpha=0.5**, **grad_clip=1.0**, **use_bf16=True**, **cosine_t_max=25**
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

Old launch baseline: 135.30. Total gain: **−73.7%** over 15 compounding improvements.

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
| #4388 | tanjiro | LR push above 2e-3: {2.5e-3, 3e-3} on T_max=25 canonical | Both arms regress on val: 2.5e-3 +2.90%, 3e-3 +1.13%. test_excl_cruise: 2.5e-3 +0.90%, 3e-3 −0.01% (~tied within hardware drift). lr=2e-3 is the local optimum **at T_max=25**. Pre-clip grad_norm rules out preconditioner failure. Compounding insight: LR ceiling is T_max-dependent → follow-up: T_max finer sweep (#4447). |

## Active WIP experiments (target: val < 35.5322, full canonical stack with --use_bf16 --lr 2e-3 --cosine_t_max 25)

| PR | Student | Hypothesis | Family | Status |
|---|---|---|---|---|
| **#3415** | **frieren** | **Log-Re sinusoidal (freqs=4) on full canonical** | **Inputs** | **WIP — training; new canonical notified.** |
| **#3952** | **edward** | **Log-pressure aux loss (logp_weight=0.1)** | **Loss tuning** | **WIP — rebased; new canonical notified.** |
| **#4234** | **askeladd** | **Batch size sweep {4, 6, 8} on bf16 canonical** | **Throughput** | **WIP — training; new canonical notified.** |
| **#4296** | **thorfinn** | **Transolver slice_num sweep {32, 96} on bf16 canonical** | **Architecture** | **WIP — rebased; new canonical notified.** |
| **#4348** | **alphonse** | **Attention head sweep {2, 8} on 14-winner canonical** | **Architecture** | **WIP — training; new canonical notified.** |
| **#4421** | **fern** | **Warmup retest at lr=2e-3: {warmup=1, warmup=2}** | **Optimization** | **WIP — training.** |
| **#4423** | **nezuko** | **Dropout sweep {0.05, 0.1} on 15-winner canonical** | **Regularization** | **WIP — training (code change required).** |
| **#4447** | **tanjiro** | **Cosine T_max finer sweep {17, 20} at canonical lr=2e-3** | **Optimization** | **WIP — just assigned.** |

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
12. **LR push above 2e-3 closed (#4388).** Both 2.5e-3 (+2.90%) and 3e-3 (+1.13%) regress on val at T_max=25. The plateau between 2e-3 and 3e-3 is shallow but real. Compounding insight: **the LR ceiling is T_max-dependent** — the 1e-3→2e-3 lift came from T_max=50→25. Smaller T_max may re-lift the LR ceiling at the cost of training time at high LR; T_max sweep #4447 tests this.

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

### Post-current-round stack
1. **Log-Re sinusoidal (frieren #3415):** if it wins → merge; orthogonal to all optimization changes. **Highest-EV pending**.
2. **Warmup retest at lr=2e-3 (#4421 fern):** {1, 2} — verifying #4359's within-PR signal at new canonical LR; could compound to ~val 34.69 if delta transfers.
3. **T_max finer sweep (#4447 tanjiro):** {17, 20} — tests whether bringing cooldown to best_epoch=17 yields tighter variance reduction; if either wins, follow-up LR re-tune at new T_max could push LR above 2e-3 again.
4. **Dropout sweep (#4423 nezuko):** {0.05, 0.1} — orthogonal regularization to wd. Code change required (add `--dropout` CLI flag).
5. **Architecture results (#4234 batch, #4296 slice_num, #4348 n_head):** any winner → merge, update canonical stack.
6. **LR re-tune on merged T_max winner (if #4447 wins):** SOAP step size sensitive to schedule; the LR ceiling lifted in #4336 may lift again with shorter T_max.
7. **LR re-tune on merged architecture winner:** SOAP step size sensitive to model capacity.
8. **Activation function sweep (GELU/SiLU/SwiGLU):** never tested on this stack — small architectural change with broad ML literature support.
9. **Huber β fine sweep around 0.01:** β family declared closed, but the merged 15-winner stack may shift the optimum subtly; low-risk follow-up if all else fails.
10. **EMA decay revisit at higher tail (0.995, 0.999):** never re-tuned after cosine T_max=25 merged; cooldown changes effective tracking timescale.
11. **Weight decay closed:** monotone val improvement offset by test_single_in_dist regression — wd=1e-4 stays canonical.
12. **LR push above 2e-3 closed (#4388):** both 2.5e-3 and 3e-3 regress on val at T_max=25. Pre-clip grad_norm ruled out preconditioner failure. lr=2e-3 is local optimum at this T_max.
