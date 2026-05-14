# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 18:15
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.


## Current baseline (14th shift)

**PR #2865 (γ-only FiLM-Re + σ=0.07 init)** merged 2026-05-14 14:45:
- **`val_avg/mae_surf_p`** = 34.5536 (mean 2 seeds); best seed 33.5570 (s2 `vt8acm18`)
- **`test_avg/mae_surf_p`** = 28.9528 (mean 2 seeds); best seed 28.2333 (s2 `vt8acm18`)
- Per-split test surf_p (mean): single_in_dist=32.53, geom_camber_rc=41.997, geom_camber_cruise=15.19, re_rand=26.09
- Default init_std=0.07 + FiLM-Re conditioning in train.py
- **New merge bar: mean val < 34.55, mean test < 28.95, all four test splits finite**

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

**Cumulative: −71.1% val, −73.6% test from round-1 start.** Still compute-bound (best=last on all 14 merges).

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
| #2959 | alphonse | Per-block lr scaling: late blocks (2-4) get 1.5× or 2.0× lr | ASSIGNED 2026-05-14 16:40 |
| #2960 | frieren | Conditioning-variable Mixup: α=0.2 (s1), α=0.4 (s2) | ASSIGNED 2026-05-14 16:45 |
| #2886 | thorfinn | γ-only FiLM-AoA: per-block AoA conditioning (sent back for σ=0.07+FiLM-Re compound) | WIP (sent back) |
| #2965 | fern | Fourier-encoded Re input to FiLM-Re γ MLP (K=2, K=4) | ASSIGNED 2026-05-14 17:15 |
| #2926 | nezuko | Stochastic depth DropPath (depth-scaled rates 0.1/0.2) | WIP |
| #2972 | edward | LayerScale: per-block channel-wise learnable gain (init=0.1, 0.01) | ASSIGNED 2026-05-14 17:40 |
| #2948 | tanjiro | FiLM-Re γ MLP capacity scan: 2× ALL-4-SPLITS-WIN single seed, awaiting 2nd seed | WIP (sent back 2026-05-14 18:00) |
| #2971 | askeladd | Slice attention dropout: drop_p=0.1 (s1), drop_p=0.2 (s2) | ASSIGNED 2026-05-14 17:38 |

**Closed this round (rounds 12–13):**
- #2908 (tanjiro σ interior) — σ=0.06/0.09 regress +17-22%. σ-axis fully bracketed at peak σ=0.07. L2 monotone in σ but val non-monotonic — parameter-scale alone insufficient.
- #2909 (askeladd PP-loss) — all 4 splits regress +11-29%. h⁴ weighting kills boundary-layer signal; +70% wall-clock overhead binding.
- #2902 (frieren SwiGLU compound) — val=34.72 misses by +0.49%. NOT orthogonal to FiLM-Re; redundant conditioning path.
- #2942 (alphonse Lion-lr) — lr=6e-5 ≈ baseline; lr=9e-5 trade-off (wins camber_rc, loses single_in_dist). lr=7.5e-5 confirmed optimal. Axis closed; motivates per-block lr.
- #2895 (fern y-flip compound on σ=0.07+FiLM-Re) — val=35.41, test=30.86. NOT orthogonal: directly-augmented cruise split regresses +27% (15.19→19.28). FiLM-Re's per-sample γ specialization is diluted by mirrored cruise inputs. Y-flip training-time aug retired at FiLM-Re baseline.
- #2953 (askeladd slice-temperature) — τ=0.5 val +0.07%/test +2.86% (single_in_dist −1.0%, all OOD regress); τ=2.0 val +4.0%/test +7%. CRITICAL: baseline has learnable per-head τ initialized to 0.5, not 1.0. Default sits near per-head optimum. Slice-temp axis closed.
- #2943 (edward head-depth) — depth=3 val −0.16%/test +1.04% (single_in_dist −6.90%, all 3 OOD splits regress); depth=4 strictly worse. Head depth NOT the bottleneck at FiLM-Re baseline. 4th OOD-vs-IID trade-off instance.

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
11. **OOD-vs-IID trade-off pattern (4 instances)** — Lion lr=9e-5 (#2942), head_depth=3 (#2943), slice_temp=0.5 (#2953) ALL improve single_in_dist but regress all 3 OOD splits. The 14th-shift basin saturates IID-side capacity. Future axes targeting IID-side capacity are predicted to fail similarly. The path forward is mechanisms that increase **redundancy / regularization** on OOD splits without adding IID capacity — slice dropout (#2971), DropPath (#2926), cond-mixup (#2960), per-block lr (#2959).
12. **Learnable per-head slice softmax τ is already present in baseline** — init=0.5. Slice softmax is not at τ=1.0 default. Future axis scans on PhysicsAttention must account for this learnable temperature.
13. **First experiment to break OOD-vs-IID trade-off pattern (#2948 tanjiro 2× FiLM-Re γ width)** — single-seed `94flg3ls`: val=33.566 (−2.86%), test=28.401 (−1.91%), ALL 4 splits improve simultaneously (single_in_dist=32.32, geom_camber_rc=41.51, geom_camber_cruise=14.84, re_rand=24.93). The mechanism is **γ MLP capacity**, not IID-side capacity — widens the conditioning bottleneck without adding capacity in a regime-specific manner. Confirms FiLM-Re γ MLP is input/representation-bottlenecked, validates tanjiro #2948 and fern #2965 (Fourier-Re γ input) as the right axis pair. Awaiting 2nd seed for merge confirmation. 4× width regresses → capacity has finite optimum. Compound prediction: tanjiro 2× + fern Fourier-Re (capacity + input information) could stack.

## Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines
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

1. **FiLM dual conditioning (Re + AoA compound)** — natural compound if thorfinn #2886 (FiLM-AoA) wins; combine into single joint (Re, AoA) conditioning per block
2. ~~**Slice softmax temperature bracketing**~~ — closed (#2953). Learnable per-head τ near per-head optimum.
3. **Per-block lr sharpening** — if #2959 wins, try late_block_start=3 (only block 4 boosted) for sharper targeting
4. ~~**Fourier Re embedding into γ MLP**~~ — ACTIVE as #2965 (fern)
5. **Y-flip TTA at inference** — apply y-flip on eligible (cruise) test samples and average predictions in physical space. Free at training (training-time y-flip retired per #2895). Paper-facing finishing move.
6. **SWA revisit at new baseline** — variance reduction may pair better with σ=0.07+FiLM-Re stronger base
7. **Conditioning Mixup with geometric features** — extend cond-mixup to include foil shape parameters (camber, chord) to directly target geom_camber_rc

### Medium-term

7. **Token-mixing alternative** — replace PhysicsAttention with gated linear attention or MLP-mixer block (plateau-protocol escalation)
8. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
9. **Pretrain-then-finetune at higher Re** — explicit OOD curriculum for geom_camber_rc
10. **Compound FiLM + DropPath** — if both win independently, combine as mutually orthogonal regularizers
