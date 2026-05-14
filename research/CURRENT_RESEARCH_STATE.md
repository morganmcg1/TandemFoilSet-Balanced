# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 20:40
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

## Active WIPs (8 students, 8 PRs, 0 idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #2959 | alphonse | Per-block lr scaling: 1.5× 2-seed rerun on 15th-shift baseline | WIP (rebase+new-baseline sent 2026-05-14 19:50) |
| #2984 | frieren | Input-only conditioning Mixup: Re/AoA inputs mixed, targets unchanged | ASSIGNED 2026-05-14 18:50 |
| #2991 | thorfinn | Output decoder head MLP width scan: 2× (256) and 3× (384) | ASSIGNED 2026-05-14 19:25 |
| #2965 | fern | Fourier-Re K=4: 2-seed rerun on 15th-shift baseline (compound with 2× γ width) | WIP (rebase+new-baseline sent 2026-05-14 19:50) |
| #2926 | nezuko | Stochastic depth DropPath (depth-scaled rates 0.1/0.2) | WIP (GPU-active, pod iter 12 @ 98%) |
| #3001 | edward | FiLM-Re γ MLP init std scan: film_re_init_std=0.05/0.03 vs global 0.07 | ASSIGNED 2026-05-14 20:40 |
| #2990 | tanjiro | FiLM-Re γ MLP depth-2 at width=256 (extend #2948 win) | ASSIGNED 2026-05-14 19:20 |
| #3002 | askeladd | Inverted late-block lr: 0.7×/0.5× reduction tests early-block OOD signal | ASSIGNED 2026-05-14 20:40 |

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

1. **Early-block lr boost (inverted late-block)** — ACTIVE as askeladd #3002 (0.7×/0.5× late-block lr). Direct test of meta-finding #14.
2. **FiLM-Re γ MLP component-specific init** — ACTIVE as edward #3001 (film_re_init_std=0.05/0.03). New axis.
3. **Y-flip TTA at inference** — apply y-flip on eligible (cruise) test samples and average predictions in physical space. Free at training (training-time y-flip retired per #2895). Paper-facing finishing move.
4. **Conditioning Mixup with geometric features** — if input-only Re/AoA mixup (#2984) wins, extend to include foil shape parameters (camber, chord) to directly target geom_camber_rc
5. **Per-block lr early_block_start=0 (all 5 blocks different)** — finer granularity than 2-group binary split. If #3002 or #2959 show any monotonic pattern across block depth, a 5-group ramp is the natural follow-up.
6. **Deeper Fourier-Re input (K=8)** — if #2965 rerun on 15th-shift confirms K=4 compound gain, try K=8 for richer Re encoding. Risk: AoA-Fourier-K=8 already failed (narrow AoA range aliases); Re has wider dynamic range so K=8 may be viable.
7. **FiLM depth-2 compound with Fourier-Re** — once #2990 and #2965 land, the 3-way compound (2× γ width + depth-2 + K=4 Fourier) is the natural summit.

### Medium-term

7. **Token-mixing alternative** — replace PhysicsAttention with gated linear attention or MLP-mixer block (plateau-protocol escalation)
8. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
9. **Pretrain-then-finetune at higher Re** — explicit OOD curriculum for geom_camber_rc
10. **Compound FiLM + DropPath** — if both win independently, combine as mutually orthogonal regularizers
