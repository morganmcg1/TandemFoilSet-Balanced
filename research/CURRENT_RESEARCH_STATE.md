# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-15 01:00
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.


## Current baseline (15th shift) ← UPDATED

**PR #2948 (2× FiLM-Re γ MLP width, film_re_hidden=256)** merged 2026-05-14 19:15:
- **`val_avg/mae_surf_p`** = 33.7062 (mean 2 seeds); best seed 33.5660 (s1 `94flg3ls`)
- **`test_avg/mae_surf_p`** = 28.6525 (mean 2 seeds); best seed 28.4010 (s1 `94flg3ls`) — **NEW BEST**
- Per-split test surf_p (mean): single_in_dist=32.221, geom_camber_rc=41.458, geom_camber_cruise=14.909, re_rand=26.022
- Default: `--init_std 0.07 --film_re_hidden 256` (baked into trunk)
- **New merge bar (15th shift): mean val < 33.71, mean test < 28.65, all four test splits finite**

**Previous (14th shift, PR #2865 γ-only FiLM-Re + σ=0.07):** val=34.55, test=28.95 — superseded.

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
| PR #2562 (Lion lr=7.5e-5) | 2026-05-13 22:30 | 45.433 | 39.509 | −9.5% / −9.2% |
| PR #2801 (Pinball τ=0.55 pressure) | 2026-05-14 07:15 | 43.092 | 37.194 | −5.1% / −5.9% |
| PR #2817 (σ=0.05 init) | 2026-05-14 09:21 | 40.820 | 35.247 | −5.3% / −5.4% |
| PR #2882 (σ=0.07 init) | 2026-05-14 12:15 | 36.575 | 30.644 | −10.4% / −13.1% |
| **PR #2865 (FiLM-Re + σ=0.07)** | **2026-05-14 14:45** | **34.554** | **28.953** | **−5.4% / −5.6% ← 14th shift** |
| **PR #2948 (2× FiLM-Re γ width)** | **2026-05-14 19:15** | **33.706** | **28.653** | **−2.45% / −1.04% ← 15th shift** |

**Cumulative: −71.8% val, −73.9% test from round-1 start.** Still compute-bound (best=last on all 15 merges).

## Current research focus (rounds 13–15)

**Active compounding strategy.** 14th shift merged (FiLM-Re + σ=0.07). In-flight experiments probe 8 orthogonal axes against the new baseline.

**Hardest remaining target:** geom_camber_rc (test=41.997, mean; 40.59, best seed). This OOD split is 40% harder than single_in_dist (32.53) and is the primary differentiator.

**Key insight from round-13 closes:**
- Lion lr scan (#2942): lr=7.5e-5 is optimal; lr=9e-5 shows a clear OOD-vs-IID trade-off (wins camber_rc, loses single_in_dist). Single global lr cannot resolve this — motivates per-block lr scaling (#2959).
- SwiGLU (#2902): NOT orthogonal to FiLM-Re. FiLM-Re already provides the Re-conditional routing that SwiGLU added on the σ=0.05 baseline. Closed — frieren now tests conditioning-variable Mixup (#2960).
- σ-axis: fully bracketed, σ=0.07 peak confirmed.
- PP-loss (#2909): closed — h⁴ weighting kills boundary-layer signal.

**Current working model of the improvement space:**
- **Per-block lr scaling (#2959 alphonse):** Late blocks do OOD work (FiLM-Re γ_w_L2 grows with depth) — give them higher lr without overshooting IID. Direct resolution of the trade-off from #2942.
- **Conditioning Mixup (#2960 frieren):** Interpolate (Re, AoA) + targets during training to regularize across conditioning manifold. Direct OOD gap attack — model never saw interpolated conditions during training.
- **FiLM-AoA (#2886 thorfinn):** Sent back for σ=0.07+FiLM-Re compound. AoA-conditional γ_w permutation (uniform across blocks, different mechanism from FiLM-Re depth-monotone γ_bias). Orthogonal compound potential.
- **Fourier-Re-FiLM (#2965 fern):** Replace scalar log(Re) input to γ MLP with Fourier features (K=2, K=4) — combats MLP low-frequency bias on Re-conditioning. Orthogonal to tanjiro #2948 (capacity vs input information).
- **Output head depth (#2943 edward):** 3-layer and 4-layer MLP head to decode richer FiLM-Re feature manifold.
- **FiLM-Re γ MLP capacity (#2948 tanjiro):** 2× and 4× γ MLP hidden dim to test conditioning bottleneck.
- **Slice softmax temperature (#2953 askeladd):** τ=0.5 (sharper) and τ=2.0 (smoother) on PhysicsAttention. Fundamental Transolver knob, never touched.
- **DropPath (#2926 nezuko):** Stochastic depth (rates 0.1/0.2) as regularizer.

## Active WIPs (8 students, 8 PRs) — updated 2026-05-15 01:00

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3057 | alphonse | Gradient-clipping max_norm bracket scan (0.5/2.0 vs baseline 1.0) — only-affects-Lion-via-sign-flip axis | ASSIGNED 2026-05-15 00:30 |
| #3042 | nezuko | Polyak weight averaging over last K epochs at eval time (K=5 and K=3) | ASSIGNED 2026-05-14 23:00 |
| #3046 | edward | Full FiLM-Re (scale+shift): add additive β output alongside γ in film_gamma | ASSIGNED 2026-05-14 23:50 |
| **#3067** | **tanjiro** | **FiLM-Re joint conditioning: camber channels [log_re, camber_1, camber_2] — direct attack on geom_camber_rc OOD split** | **ASSIGNED 2026-05-15 01:00** |
| #3028 | askeladd | FiLM-Re γ at output decoder (Re-blind decoder injection, target: re_rand OOD) | ASSIGNED 2026-05-14 21:55 |
| #3041 | frieren | Re-jitter: Gaussian noise on FiLM-Re γ-MLP input at train time (σ=0.10/0.30) | ASSIGNED 2026-05-14 23:00 |
| #3035 | thorfinn | FiLM-Re γ on PhysicsAttention routing layer (slice_proj logits — Re-conditional routing) | ASSIGNED 2026-05-14 22:15 |
| #3038 | fern | slice_num bracket scan: 48 (0.75×) and 96 (1.5×) vs baseline 64 | ASSIGNED 2026-05-14 22:25 |

**Closed this round (rounds 12–15):**
- #2908 (tanjiro σ interior) — σ=0.06/0.09 regress +17-22%. σ-axis fully bracketed at peak σ=0.07.
- #2909 (askeladd PP-loss) — all 4 splits regress +11-29%. h⁴ weighting kills boundary-layer signal.
- #2902 (frieren SwiGLU compound) — val=34.72 misses by +0.49%. NOT orthogonal to FiLM-Re.
- #2942 (alphonse Lion-lr) — lr=7.5e-5 confirmed optimal. Motivates per-block lr.
- #2960 (frieren cond-mixup) — 76-88% worse. Mesh-aligned target mixup physically meaningless.
- #2886 (thorfinn FiLM-AoA compound) — test=29.54 (+2.04% over 14th-shift bar). FiLM-AoA gains subsumed by σ=0.07+FiLM-Re. Mechanism orthogonality confirmed (paper-worthy ablation). Closed.
- #2895 (fern y-flip compound) — test=30.86. FiLM-Re γ specialization disrupted by y-flip.
- #2953 (askeladd slice-temperature) — τ=0.5 all OOD splits regress; learnable τ already near optimum.
- #2943 (edward head-depth) — depth=3 single_in_dist −6.90% but all OOD splits regress. 4th OOD-vs-IID trade-off.
- **#2972 (edward LayerScale)** — init=0.1: val +5.3%, test +6.0%; init=0.01: val +9.4%, test +9.8%. Near-identity residual init wastes early-epoch learning at 30-min cap. All 4 splits regress. Closed.
- **#2971 (askeladd slice dropout)** — drop_p=0.1: test +3.41%; drop_p=0.2: test +5.53%. Monotonic OOD regression; IID improves (classic dropout-as-regularizer for in-dist). Four-axis routing pattern now complete. Closed.
- **#2926 (nezuko DropPath)** — drop_path=0.1: val +3.65/test +4.70; drop_path=0.2: val +4.81/test +4.12. test_geom_camber_rc regress +11-13% (opposite of OOD-help hypothesis). 5-block Transolver too shallow for stochastic depth (literature regime depth ≥ 12). Closed.
- **#2959 (alphonse per-block lr)** — Closed after 3 send-backs without successful rebase. Meta-finding #14 (OOD signal in early blocks) preserved; inverted-scaling follow-up axis now covered by askeladd #3002. Alphonse reassigned to per-block wd (#3012) — orthogonal optimizer-side axis on the same param-group split.
- **#2990 (tanjiro γ-MLP depth-2)** — depth-2 at width=256: val=33.7942 (+0.26% REGRESS), test=28.5276 (−0.44% pass). Primary metric fails merge bar. γ_w_L2 depth-monotone pattern FLATTENED (3.97→5.75 → ~4.1–4.6 flat). Seed variance widens to 1.68% (vs 0.4%). γ-MLP depth redistributes capacity at width=256 without adding it. Depth axis closed. Meta-finding #18 added. Tanjiro reassigned to joint Re+AoA γ-MLP input (#3019).
- **#3002 (askeladd inverted lr)** — 0.7×/0.5× late-block lr: val=35.19/36.59 (+4.4%/+8.6% REGRESS), test=29.61/30.92. γ_w_L2 inverted at 0.5× (late drops below early). Combined with #2959 boost results: geom_camber_rc monotone-bad in BOTH directions from 1.0×. Per-block lr fully retired. Meta-finding #14 ("early blocks drive OOD") RETRACTED (misattribution). Meta-finding #19 added. Askeladd reassigned to FiLM-Re decoder injection (#3028).
- **#2984 (frieren input-only cond-mixup)** — α=0.2/0.4: val=55.95/58.79 (+66%/+74% REGRESS), test=50.19/52.87 (+75%/+85% REGRESS). All 4 splits catastrophic. Mechanism: per-batch λ + per-sample targets = conditioning label noise; model is forced to ignore conditioning. The 76-86% test_re_rand regression is the smoking gun (encoder learned to IGNORE Re). Meta-finding #20 added. Cond-mixup axis closed; clean alternative is single-sample Gaussian Re-jitter with target pairing preserved.
- **#2991 (thorfinn head-decoder-width)** — 2×/3× head_hidden: val=36.22/36.08 (+7.4%/+7.0% REGRESS), test=30.83/30.23 (+7.6%/+5.5% REGRESS). All 4 splits uniformly regress (no OOD-selective signal). Mirror of edward's #2943 head-depth result. 128-d single-hidden-layer head is at capacity sweet spot. Meta-finding #21 added. Output decoder head capacity (width AND depth) fully retired as a free axis.
- **#2965 (fern Fourier-Re K=4 on 15th-shift)** — K=4 compound with width=256 γ MLP: val=35.38/36.66 (+4.94%/+8.76% REGRESS), test=30.24/30.94 (+5.56%/+8.01% REGRESS). All 4 splits regress; test_single_in_dist worst at +14.5%. γ_w_L2 trajectory stays flat at ~4.2 (no further relief beyond what width=256 alone provides). Meta-finding #22 added. **Conditioning-encoder bottleneck-relief axes (γ-width, γ-depth, Fourier input) are NOT orthogonal — width=256 paid the bottleneck once and K=4 became redundant.** Fern reassigned to next plateau-protocol bet.
- **#3001 (edward FiLM-Re γ MLP init std)** — film_re_init_std=0.05/0.03 (hidden W1 only; output identity preserved per student deviation correctly flagged): val=34.95/34.67 (+3.7%/+2.8% REGRESS), test=28.67/29.78 (+0.07%≈tie/+3.9% REGRESS). Both miss merge bar on val. s2 γ_w_L2 non-monotone (block 0=5.09 > block 1=4.73) — smaller W1 forces W2 over-travel at early blocks (degraded health signal). Meta-finding #25 added. **Identity output init (W2=0, b2=1) is at or near optimal at this basin** — hidden init scale is not a viable independent axis. Edward reassigned to full FiLM-Re scale+shift.
- **#3012 (alphonse per-block wd 0.25×/4×)** — s1 (0.25× late): val=35.34 (+4.8% REGRESS), test=30.05 (+4.9% REGRESS); s2 (4× late): val=34.88 (+3.5% REGRESS), test=29.66 (+3.5% REGRESS). Clean IID-vs-OOD wedge (s1 hurts IID +10.3%, s2 hurts cruise +10.5%). **16× per-block wd ratio produces ≤0.15 L2 difference on late-block params under Lion** — Lion's sign() update neuters magnitude-based regularization. γ_w_L2 depth-monotone pattern is robust to 16× wd perturbation (block 4 ≈ 5.78 across all arms vs baseline 5.75). Meta-finding #26 added. **Per-block magnitude regularization (lr + wd) fully retired under Lion.** Alphonse reassigned to gradient-clip max_norm bracket-scan (the only remaining optimizer axis that does affect Lion's sign() via sign-flip on clipped grads).
- **#3019 (tanjiro FiLM-Re joint [log_re, AoA_1, AoA_2])** — s1 (`mozcqjv2`): val=33.13 (−1.72% ✓), test=27.87 (−2.73% ✓), camber_rc=40.04 (−3.43% ✓) — clean win; s2 (`iqi0eqc4`): val=35.11 (+4.16% ❌), test=29.89 (+4.30% ❌) — clean loss. 2-seed mean: val=34.12 (+1.22% ❌), test=28.88 (+0.79% ❌), camber_rc=41.83 (+0.89% ❌). Variance-dominated (~2pt seed spread vs baseline ~0.4%). Per-block input-L2 diagnostic: AoA_1 dominates block 0 (1.27 vs log_re 0.97) but decays to 0.78 by block 3; log_re ramps monotonically (0.97→1.57). γ_w_L2 depth-monotone FLATTENED (same redistribute-not-add signature as #2990, #2965). Meta-finding #27 added. Conditioning-surface-area axis with AoA channels retired; camber channels ([log_re, camber_1, camber_2]) are the natural follow-up since geom_camber_rc is geometry-defined. Tanjiro reassigned to #3067.

## Key meta-findings

1. **Compute is permanently binding** — best=last at every merge. 30-min cap dominant constraint since bf16.
2. **Variance-vs-mean decoupling (10 instances)** — any mechanism reducing step frequency, representation capacity, or initial activation scale trades mean improvement for variance reduction. At 35-ep cap, mean cost never recovered.
3. **Lion betas FULLY BRACKETED** — β1=0.90, β2=0.99 confirmed optimal.
4. **σ-axis: init scale and wd are substitutes, not complements** — σ=0.07 + wd=2e-4 wins; σ=0.07 + wd=1e-3 HURTS (over-regularizes already-regularized basin). Characterized by #2897.
5. **FiLM-Re mechanism confirmed orthogonal to σ-axis** — identical relative improvement (−5.4%/−5.6%) across σ=0.02 and σ=0.07 bases.
6. **γ(Re) depth-gradience pattern** — late blocks (3-4) develop stronger Re-dependent gain modulation than early blocks; consistent with deeper blocks doing more task-specific processing. Motivates per-block lr scaling.
7. **geom_camber_rc is structural OOD** — responds to conditioning + physical regularization axes; Re/AoA FiLM + conditioning Mixup are the two most promising direct interventions.
8. **SwiGLU mechanism insight** — SwiGLU gained on σ=0.05 by compensating under-conditioning that FiLM-Re now provides. FFN-capacity axes not orthogonal when Re-conditioning is already rich.
9. **Global lr cannot resolve OOD-vs-IID split trade-off** — lr=9e-5 wins geom_camber_rc but hurts single_in_dist. Per-block lr scaling is the natural resolution.
10. **Mechanism overlap surfacing at 14th-shift basin** — both SwiGLU (#2902) and y-flip (#2895) helped on σ=0.05 but regress on σ=0.07+FiLM-Re. Both mechanisms add "Re-regime feature diversity" that FiLM-Re now provides explicitly. Pattern suggests aug/capacity axes that helped at σ=0.05 are increasingly likely to overlap with FiLM-Re at 14th shift; emphasis must shift to genuinely orthogonal axes (input enrichment, optimizer geometry, decoder capacity).
11. **OOD-vs-IID trade-off pattern (now 6 confirmed instances, 1 exception)** — Lion lr=9e-5, head_depth=3, slice_temp=0.5, late_block_lr×1.5, LayerScale init<1, slice_dropout ALL improve IID but regress OOD. The **one exception: FiLM-Re γ MLP capacity (#2948)** lifts all four splits simultaneously — conditioning capacity is the only axis tested that breaks the trade-off.
12. **Learnable per-head slice softmax τ is already present in baseline** — init=0.5. Slice softmax is not at τ=1.0 default. Future axis scans on PhysicsAttention must account for this learnable temperature.
13. **FiLM-Re γ MLP capacity breaks OOD-vs-IID trade-off (#2948, 15th shift MERGED)** — 2× γ width: val=33.706 (−2.45%), test=28.653 (−1.04%), ALL 4 splits improve simultaneously. Conditioning capacity is orthogonal to IID-specific capacity. 4× width regresses → optimal at 2×. Compound prediction: 2× width + depth-2 (tanjiro #2990) + Fourier input (fern #2965) + smaller init (edward #3001) could stack.
14. **Per-block lr OOD signal lives in EARLY layers, not late (#2959 alphonse)** — late_block_lr×1.5 improves IID by −4.0% but regresses geom_camber_rc by +2.4%; 2.0× amplifies (+6.0% on camber_rc). 5th OOD-vs-IID instance. Inverted scaling (late_block_lr_scale=0.7/0.5) is now the active test (askeladd #3002).
15. **FiLM-Re γ MLP is input-bottlenecked (γ_w_L2 evidence, #2965 fern)** — K=4 Fourier input flattens γ_w_L2 depth gradient (3.4→5.2 monotone → ~3.6 flat). Validates the capacity+input-expressivity axis pair. Now being retested on 15th-shift baseline (compound with 2× γ width).
16. **Slice routing perturbation reliably trades OOD for IID (four-axis closed)** — sharpen τ (#2953), soften τ, slice dropout (#2971), late-block lr boost (#2959) all show the same OOD-IID wedge. Only conditioning-capacity expansion (FiLM-Re γ width) breaks the wedge. Future routing-layer interventions expected to show the same pattern.
17. **Block-level stochastic regularizers retired at depth=5 (#2926)** — DropPath at rates 0.1/0.2 over-regularizes a 5-block Transolver; literature regime is depth ≥ 12 (ViT-L, Swin-L, MAE). The depth-scaled per-block residual is doing structural work that random zeroing destroys faster than ensemble benefit can compensate. Combined with #11/#16, **only conditioning-capacity expansion has worked**. Two regularization tiers retired: block-level over-regularizes; routing-perturbation trades OOD/IID. Path forward: conditioning-capacity-side interventions only (γ width #2948, γ depth #2990, γ Fourier input #2965, γ component-init #3001).
18. **γ-MLP depth does NOT add capacity at 35-ep compute-bound budget (#2990)** — depth-2 at width=256: γ_w_L2 depth-monotone pattern FLATTENS (3.97→5.75 → ~4.1–4.6 flat); depth redistributes the depth-dependent modulation from output-Linear weights into a new hidden-layer, without expanding what the model can express. Seed variance widens 4× (0.4%→1.68%). **Width=256 saturates the conditioning-capacity headroom at 35 epochs.** Further γ-MLP internal capacity expansion exhausted — gains must come from **input expressivity** (Fourier-Re #2965) or **conditioning surface area** (joint Re+AoA input #3019).
19. **Per-block lr scaling is fully retired — meta-finding #14 RETRACTED (#3002, #2959)** — both boost (1.5×/2.0×, #2959) and reduction (0.7×/0.5×, #3002) hurt all splits monotonically; geom_camber_rc is monotone-bad in BOTH directions from 1.0×. The per-block lr optimum is centered at 1.0× and any deviation hurts OOD. **Meta-finding #14 ("early blocks drive OOD signal") was a misattribution of confounded run variance.** γ_w_L2 depth-monotone diagnostic confirmed as health-signal: runs that fail to grow the early~4.2/late~5.6 pattern consistently regress (0.5× late-lr inverts pattern, maps perfectly to worst performance).
20. **Per-batch λ Mixup on conditioning inputs without per-sample target preservation = conditioning label noise (#2984)** — α=0.2 and α=0.4 both regress >66% val, >75% test, all 4 splits catastrophic. Implementation samples one λ per batch and pairs `(x[perm[i]] cond, y[i] target)` for ~half of batches (Beta-bimodal mass at λ≈0/1). Model's optimal response: down-weight conditioning channels. test_re_rand +76-86% confirms encoder learned to IGNORE Re. **Conditioning Mixup has no clean variant on non-corresponding meshes.** Single-sample Gaussian Re-jitter with target pairing preserved is the clean replacement axis if conditioning-smoothing is desired (already-explored axis worth revisiting at 15th-shift with FiLM-Re in place).
21. **Output decoder head capacity is fully retired (#2991 width, #2943 depth)** — 2×/3× head_hidden uniformly regresses 5-9% across all 4 test splits (no OOD trade-off); mirrors #2943 head-depth (depth=3 closed). At the 15th-shift basin, output head capacity is NOT orthogonal to trunk — additional head params rob optimization budget from trunk + γ-MLP within the 30-min/35-epoch cap. 128-d single-hidden-layer head sits at the capacity sweet spot. Future capacity moves: trunk-internal width, slice_num, or genuinely new injection points (e.g. #3028 decoder-FiLM-Re).
22. **Conditioning-encoder bottleneck-relief axes are NOT orthogonal (#2965 K=4 Fourier on 15th-shift)** — γ-width (#2948 width=256), Fourier input expressivity (#2965 K=4), and γ-MLP depth (#2990) all relieve the same scalar→γ information bottleneck. γ_w_L2 trajectory flattens to ~4.2 with EITHER width=256 OR K=4-on-width=128; combining both gives flat ~4.2 too — no further relief. K=4 on 15th-shift baseline regresses 5-9% on all 4 splits (test_single_in_dist hit hardest at +14.5%, consistent with over-parameterized γ MLP memorizing IID patterns through redundant Fourier features). **Pay the bottleneck relief once at the cheapest axis** (γ-width was the winner). Further conditioning-encoder capacity expansion exhausted — gains must come from injection-point expansion (#3028 decoder, #3035 routing), conditioning surface area (#3019 joint Re+AoA), or data-side distribution levers.
23. **Re-distribution rebalancing is NOT the lever for OOD generalization on TandemFoilSet (#3034 frieren)** — Per-split log_re mean+std analysis on the train/val splits: `val_re_rand` is co-distributed with train (mean 14.65 ≈ train 14.59) → re_rand is not Re-coverage-bound. `val_geom_camber_rc` is shifted toward HIGHER Re than train (mean 14.86 vs 14.59). Quantile binning is structurally a no-op (equal bin_counts by construction → uniform weights). Equal-width binning would up-weight low-Re samples, but camber_rc lives at HIGH Re — equal-width balancing would HURT camber_rc. Conclusion: training-data Re-distribution is the wrong intervention for OOD on this dataset; conditioning gains must come from MODEL-side axes (FiLM-Re width/decoder/routing/joint-input). The per-split log_re statistics are paper-appendix material.
24. **Y-flip TTA is fundamentally incompatible with TandemFoilSet (#3007 nezuko)** — Two compounding failure modes: (a) 70% of train is ground-bound (raceCar single+tandem, z ≥ 0); y-flipping → below-ground OOD; (b) cruise tandem subset (30%) has cambered foils at non-zero AoA → FLOW is non-y-equivariant even when the MESH is z-symmetric. Per-sample sym-gate (z_min ≥ -0.5) fixes (a) but NOT (b) → 3.4× test regression on cruise even when correctly gated. 3.6× uniform TTA test regression overall. Symmetry-free eval-time ensembling (multi-checkpoint averaging, Polyak weight averaging over last K epochs) is the clean alternative — no symmetry assumption needed.
25. **Identity output init for FiLM-Re γ-MLP is locked at optimum (#3001 edward)** — hidden W1 std=0.05/0.03 (preserving identity output init): val regresses +3.7%/+2.8%, test ties/regresses +0.07%/+3.9%. s2 (std=0.03) shows non-monotone γ_w_L2 depth trajectory (block 0=5.09 > block 1=4.73, vs monotone baseline 3.97→5.75) — smaller W1 forces W2 over-travel at early blocks. The hypothesis premise ("γ ≠ 1 at init → over-conditioning") doesn't hold because identity output init (W2=0, b2=1) enforces γ ≡ 1 at epoch 0 regardless of hidden init. **FiLM-Re γ-MLP init is locked at hidden std=0.07, output identity-init (W2=0, b2=1).** Future γ init experiments must target the output layer carefully (high risk of trunk collapse if b2 ≠ 1 at init). The next clean axis on the γ-MLP itself is the OUTPUT FORMULATION — γ-only (scale) vs full FiLM (scale+shift, β additive) — which is what edward's next assignment tests.
26. **Per-block magnitude regularization fully retired under Lion (#3012 alphonse, with #2959/#3002)** — 0.25× and 4× per-block wd (16× ratio between arms) produces ≤0.15 L2 difference on late-block params and ≤0.07 total param L2 difference. Both arms regress 3.5-4.9% on val/test and show the OOD-IID wedge (s1 hurts IID +10.3% single_in_dist, s2 hurts cruise +10.5%). **Lion's sign(m) update direction is independent of gradient magnitude**, so wd's pull (which scales with parameter magnitude) competes against a unit-vector update — the differential framing has no traction. γ_w_L2 depth-monotone pattern (4.0→5.8 baseline) is fully **robust to 16× wd perturbation** (block 4 ≈ 5.78 across all arms) — confirms the pattern is a training-signal property, not a hyperparameter. Combined with retired per-block lr (#2959/#3002), the per-block magnitude-based optimization axes are fully exhausted. Future per-block axes must operate on a non-magnitude dimension (gradient direction via sign-flip noise injection, attention masking, structural connectivity). The clipping max_norm axis remains untouched at this basin — it's the only optimizer-side axis that affects Lion's sign() via grad sign-flip when clipping rotates the momentum vector across the sign boundary.
27. **Joint conditioning input is variance-dominated at 35-ep budget (#3019 tanjiro)** — [log_re, AoA_1, AoA_2] joint input: s1 clean win (test −2.73%, all 4 splits improve, camber_rc −3.43%), s2 clean loss (test +4.30%). 2-seed mean misses all bars (+1.22%/+0.79% val/test). Key mechanism: AoA_1 dominates γ-MLP at block 0 (col-L2 1.27 vs log_re 0.97) but decays by block 2+ where log_re ramps to 1.57. γ_w_L2 depth-monotone pattern flattens (3.97→5.75 → 4.9–5.4 flat) — same redistribute-not-add signature as #2990 depth-2 and #2965 K=4 Fourier. **Joint input perturbs the conditioning landscape locally but doesn't expand total conditioning capacity. The correct OOD lever for geom_camber_rc is foil geometry (camber), not Re×AoA.** Follow-up: [log_re, camber_1, camber_2] (#3067) is the natural successor — camber is the structural variable that defines the OOD split by construction.

## Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines; 6th failure was slice_num=128 compute-bound (PR #1507)
- **EMA weights, SWA** — tested PR #2399 (ema-0p999): wash mechanism at cosine-cooling schedule; PR #2712 (SWA): modest variance reduction but test mean misses bar. Both retired. Re-test condition: schedule change with higher eta_min.
- **LayerScale** — PR #2972: near-identity residual init (0.01/0.1) wastes early-epoch learning at 30-min cap; all 4 splits regress uniformly.
- **Schedule shape** — T_max, eta_min, warmup, warm restarts — all retired
- **Per-neuron Dropout** — regularization stack already saturated (stochastic depth DropPath has NOT been tested — #2926 tests this)
- **Lion betas** — β1=0.90, β2=0.99 confirmed optimal, fully bracketed
- **Lion LR (global)** — 7.5e-5 confirmed optimal at 14th-shift basin (#2942). Per-block scaling in progress (#2959).
- **Weight-decay (wd axis at σ=0.07)** — wd=2e-4 wins; wd=1e-3 regresses. Axis closed.
- **σ-axis (init_std)** — σ=0.07 confirmed PEAK. Non-monotonic: parameter-scale alone insufficient.
- **Pressure-Poisson aux loss** — h⁴ stencil weighting kills boundary-layer signal; PP loss adds +70% wall-clock and gradients conflict with surf_p. Physics-informed aux loss axis retired at 30-min cap.
- **SwiGLU FFN** — NOT orthogonal to FiLM-Re at σ=0.07 baseline; redundant conditioning path.
- **Per-channel Huber β, surf_weight, per-channel amplitude weighting** — fully bracketed/retired
- **n_head=8, QK-RMSNorm, RMSNorm (hidden)** — various capacity/normalization failures
- **EMA weights, SWA** — variance reduction works but mean misses bar
- **SiLU activation, Charbonnier loss, CosineAnnealingWarmRestarts** — retired
- **Lookahead, Gradient Accumulation** — momentum/step-count destructive under 35-ep cap
- **Per-channel amplitude weighting** — Lion sign() discards gradient magnitude
- **Conditioning-variable jitter (log(Re))** — supervised inconsistency
- **Gradient Centralization on Lion** — sign-incompatible
- **Pinball τ > 0.55 (pressure)** — τ=0.55 optimal; τ-axis fully bracketed
- **Pinball on velocity channels (Ux/Uy)** — unbiased channels; τ≠0.5 regresses
- **Re-Fourier input features** — scalar Re aliasing; FiLM-style trunk conditioning supersedes
- **AoA-Fourier input features** — K=8 frequency aliasing on narrow AoA range
- **Divergence-free auxiliary loss (∇·u=0)** — λ calibration 3 OOM off; all λ range tested

## Potential next research directions

### Near-term (queue for next idle slots)

1. **Per-block separate β init scan** — if edward's scale+shift wins, test β-init variants (0.0 identity vs small noise)
2. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens. Direct attack on mae_surf_p; never tried at this basin.
3. **Sophia or Adan optimizer** — bold plateau-protocol switch from Lion. Compute-trade aware: must fit in 35-ep cap.
4. **Distance-to-leading-edge weighted surf loss** — geometric pressure-gradient prior. Different from h⁴ Poisson weighting (retired).
5. **Per-block FiLM-AoA on top of FiLM-Re** — only if tanjiro's #3019 joint Re+AoA fails (confirms they need to be separate paths).

### Medium-term

7. **Token-mixing alternative** — replace PhysicsAttention with gated linear attention or MLP-mixer block (plateau-protocol escalation)
8. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
9. **Pretrain-then-finetune at higher Re** — explicit OOD curriculum for geom_camber_rc
10. **Compound FiLM + DropPath** — if both win independently, combine as mutually orthogonal regularizers
