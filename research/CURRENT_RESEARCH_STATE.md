# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 15:35
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

## Current research focus (rounds 12–13)

**Active compounding strategy.** 14th shift merged (FiLM-Re + σ=0.07). In-flight experiments probe 7 orthogonal axes against the new baseline.

**Hardest remaining target:** geom_camber_rc (test=41.997, mean; 40.59, best seed). This OOD split is 40% harder than single_in_dist (32.53) and is the primary differentiator.

**Current working model of the improvement space:**
- σ=0.07 init: better basin (already merged)
- FiLM-Re: better Re-conditional representations (already merged)
- FiLM-AoA (thorfinn #2886): better AoA-conditional representations → directly targets camber_rc
- SwiGLU (frieren #2902): richer FFN → potential capacity uplift on top of FiLM
- Y-flip aug (fern #2895): ~2× effective data → regularization + OOD coverage
- DropPath (nezuko #2926): block-level stochastic ensemble → regularization
- Pressure-Poisson (askeladd #2909): physics-informed aux loss → strong OOD coupling signal
- Lion lr scan (alphonse #2942): optimizer re-tune after basin shift
- Output head depth (edward #2943): richer per-channel decoder → potential OOD decoding improvement
- FiLM-Re γ MLP capacity scan (tanjiro #2948): widen γ branch — direct extension of 14th-shift mechanism

## Active WIPs (8 students, 8 PRs, 0 idle)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #2886 | thorfinn | γ-only FiLM-AoA: per-block AoA conditioning (targets camber_rc OOD) | WIP |
| #2895 | fern | Y-flip data augmentation (flow y-equivariance, 2× data free) | WIP (sent back for σ=0.07+FiLM-Re rerun) |
| #2902 | frieren | SwiGLU FFN: replace GELU+Linear with gated FFN | WIP (sent back for σ=0.07+FiLM-Re rerun) |
| #2909 | askeladd | Pressure-Poisson auxiliary loss (λ=0.01) | WIP |
| #2926 | nezuko | Stochastic depth DropPath (depth-scaled rates 0.1/0.2) | WIP |
| #2942 | alphonse | Lion lr re-tune at 14th-shift baseline (lr=6e-5, lr=9e-5) | WIP (assigned 15:10) |
| #2943 | edward | Output head depth scan (head_depth=3/4 on 14th-shift baseline) | WIP (assigned 15:10) |
| **#2948** | **tanjiro** | **FiLM-Re γ MLP capacity scan: 2× and 4× γ width** | **ASSIGNED 2026-05-14 15:32** |

**Closed this round:** #2908 (tanjiro σ interior) — both σ=0.06/0.09 regress hard; σ-axis fully characterized with peak at σ=0.07. Mechanism: parameter-scale alone does NOT explain σ-axis (L2 monotone in σ, val sharply non-monotonic with peak at σ=0.07).

## Context for in-flight PRs

PRs assigned before 14th shift (#2886, #2895, #2902, #2908, #2909, #2926) must compare against the **new 14th-shift bar** (mean val<34.55, mean test<28.95). Any result that beats the OLD bar (val<36.58) but misses 14th-shift: evaluate mechanism strength and send back for compound test with σ=0.07+FiLM-Re.

## Key meta-findings

1. **Compute is permanently binding** — best=last at every merge. 30-min cap dominant constraint since bf16.
2. **Variance-vs-mean decoupling (10 instances)** — any mechanism reducing step frequency, representation capacity, or initial activation scale trades mean improvement for variance reduction. At 35-ep cap, mean cost never recovered.
3. **Lion betas FULLY BRACKETED** — β1=0.90, β2=0.99 confirmed optimal.
4. **σ-axis: init scale and wd are substitutes, not complements** — σ=0.07 + wd=2e-4 wins; σ=0.07 + wd=1e-3 HURTS (over-regularizes already-regularized basin). Characterized by #2897.
5. **FiLM-Re mechanism confirmed orthogonal to σ-axis** — identical relative improvement (−5.4%/−5.6%) across σ=0.02 and σ=0.07 bases.
6. **γ(Re) depth-gradience pattern** — late blocks (3-4) develop stronger Re-dependent gain modulation than early blocks; consistent with deeper blocks doing more task-specific processing.
7. **geom_camber_rc is structural OOD** — this split responds most to conditioning + physical regularization axes; Re/AoA FiLM + Pressure-Poisson are the two most promising direct interventions.

## Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines
- **Schedule shape** — T_max, eta_min, warmup, warm restarts — all retired
- **Per-neuron Dropout** — regularization stack already saturated (stochastic depth DropPath has NOT been tested — #2926 tests this)
- **Lion betas** — β1=0.90, β2=0.99 confirmed optimal, fully bracketed
- **Lion LR** — 1e-4 overshoots; 7.5e-5 sweet spot historically; re-scan in #2942 at new basin
- **Weight-decay (wd axis at σ=0.07)** — wd=2e-4 wins; wd=1e-3 regresses. Axis closed.
- **σ-axis (init_std)** — σ=0.07 confirmed PEAK (PR #2882 won; PR #2908 closed). σ=0.06 (+17% val), σ=0.09 (+16% val), σ=0.10 (memory OOM risk). Non-monotonic: parameter-scale alone does NOT explain σ-axis. Axis fully bracketed.
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
- **Divergence-free auxiliary loss (∇·u=0)** — λ calibration 3 OOM off; all λ range tested; ∇·u constraint doesn't help surf_p

## Potential next research directions

### Near-term (queue for next idle slots)

1. ~~**γ MLP capacity scan**~~ — ACTIVE as #2948 (tanjiro)
2. **FiLM dual conditioning (Re + AoA)** — natural compound if thorfinn #2886 (FiLM-AoA) wins; combine into single joint (Re, AoA) conditioning per block
3. **Lion lr bracket winner** — if #2942 wins at lr=9e-5, test lr=1.1e-4; if wins at lr=6e-5, test lr=5e-5
4. **Fourier Re embedding into γ MLP** — richer conditioning input for FiLM: log(Re) + sin/cos(πk·log(Re)) for K=4
5. **Slice softmax temperature** — Transolver-specific, untouched; learnable τ or fixed τ ∈ {0.5, 2.0}
6. **Mixup on conditioning (Re, AoA) + targets** — augment effective conditioning range; OOD-targeted regularizer
7. **SWA revisit at new baseline** — variance reduction may pair better with σ=0.07+FiLM-Re stronger base

### Medium-term

6. **Token-mixing alternative** — replace PhysicsAttention with gated linear attention or MLP-mixer block (plateau-protocol escalation)
7. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
8. **Pretrain-then-finetune at higher Re** — explicit OOD curriculum for geom_camber_rc
9. **Mixup on conditioning variables (Re, AoA)** — augment effective conditioning range; helps OOD coverage
10. **Compound FiLM + DropPath** — if both win independently, combine as mutually orthogonal regularizers
