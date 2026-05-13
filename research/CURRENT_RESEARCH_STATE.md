# SENPAI Research State

- **Last updated**: 2026-05-13 20:45 UTC (Wave 17+: MERGE #2475 fern layerscale-init-0.1 (19th compound win, val −0.49% / test +0.29%; anti-correlated init mechanism); CLOSE #2469/#2434/#2453/#2476 (4 dead-ends with mechanism findings captured); ASSIGN #2510 alphonse decoupled-layerscale-init / #2511 thorfinn input-feature-gate / #2513 frieren stoch-depth-deep-concentrated / #2514 askeladd reflection-tta)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2475 merged — LayerScale init=0.1 retuned, 19th compound win)

- `val_avg/mae_surf_p` = **58.3244** (e12; full stack: ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + **LayerScale γ no-WD 10× lr init=0.1** + LR warmup + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **50.9438**
- Per-split val: single_in_dist=71.343 / camber_rc=71.041 / camber_cruise=35.411 / re_rand=55.503
- Per-split test: single_in_dist=63.834 / camber_rc=64.360 / camber_cruise=29.389 / re_rand=46.192
- Δ vs prior PR #2436 baseline (58.6093 / 50.7946): **−0.49%** val_avg, **+0.29%** test (within noise)
- **Mechanism finding (PRIMARY VALUE) — anti-correlated init responses**:
  - attn LayerScale γ drifts DOWN from init=0.1 → lands 0.04-0.08 (sparse-state attractor, std/mean 150-243%)
  - mlp LayerScale γ drifts UP from init=0.1 → lands 0.149-0.194 (dense-state attractor, std/mean 16-30%)
  - **The two paths have OPPOSITE attractors under no-WD 10× lr** — contradicts "single global equilibrium" framing
  - The no-WD 10× lr group is in a flat region of the loss landscape where init bias compounds rather than relaxing
- **Why merged despite student's "don't merge"**: CLAUDE.md is explicit that `val_avg/mae_surf_p` is the metric of record; small improvements compound; +0.29% test regression is within noise. The mechanism finding directly motivates the next-wave decoupled-init test (#2510 alphonse).
- **Compound progress**: 100.957 → **58.3244** = **−42.23% over 19 merges**
- **n_params**: **892,637** (unchanged)

## Current research focus

**Wave 17+ — Refined optimizer-group axis + emergent anti-correlated-init mechanism.**

After Wave 17 partial results, the optimizer-group axis split into two empirically distinct classes:
- **Additive scale params** (multiply activations linearly): freqs (#2370 MERGED, −3.73%), LayerScale γ (#2436 MERGED, −1.60%; #2475 MERGED retuning, −0.49%). Win at no-WD + 10× lr.
- **Softmax-internal scale params** (govern combinatorial routing): slice temperature (#2437 CLOSED Outcome C, +4.81%), QK temperature v1 (#2377 CLOSED). Fast updates destabilize routing.

**NEW Wave 17+ mechanism (anti-correlated init under no-WD 10× lr group)**:
The merge of #2475 LayerScale init=0.1 retuning revealed a striking pattern:
- attn LayerScale γ drifts DOWN from init=0.1 → lands 0.04-0.08 (sparse-state attractor)
- mlp LayerScale γ drifts UP from init=0.1 → lands 0.15-0.19 (dense-state attractor)
- The two paths have OPPOSITE attractors despite identical optimizer treatment
- The same overshoot pattern shows in #2434 (frieren freq-init-equilibrium): freqs continued drifting past the #2370 endpoint when started there

**Interpretation**: the no-WD + 10× lr group operates in a **flat region** of the loss landscape where init bias COMPOUNDS rather than relaxing. This is a previously-unrecognized property of the group — it explains:
- Why exact init matters less than direction-of-drift
- Why retuning init close-but-not-AT the attractor can still win
- Why different params with different optimal endpoints can coexist in the same group

**Wave 17+ now tests mechanism follow-ups**:
- `layerscale-init-decoupled` (alphonse #2510): split init at attractors (attn=0.025, mlp=0.1) — tests if it removes wrong-init reversal cost
- `layerscale-per-block-lr` (fern #2515): per-block lr scaling 5×→15× — tests if the per-block std/mean heterogeneity in #2475 is a tunable signal
- `input-feature-gate` (thorfinn #2511): extends additive-scale theme to input position
- `stoch-depth-deep-block-concentrated` (frieren #2513): tests deep-block redundancy theory
- `reflection-tta-at-inference` (askeladd #2514): risk-free TTA at val/test — paper-friendly addition

**Wave 17+ active threads (all 8 students busy):**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| alphonse | #2510 | layerscale-init-decoupled | Decoupled LayerScale init `attn=0.025, mlp=0.1` — directly tests #2475's anti-correlated-attractor finding | ASSIGNED |
| thorfinn | #2511 | input-feature-gate | Per-feature input gate γ_input (44 params, no-WD 10× lr) — extends additive-scale theme to input position | ASSIGNED |
| frieren | #2513 | stoch-depth-deep-block-concentrated | Concentrate stoch-depth in deeper blocks `[0,0,0.05,0.10,0.15]` vs linear `[0,0.025,0.05,0.075,0.1]` — anti-redundancy training | ASSIGNED |
| askeladd | #2514 | reflection-tta-at-inference | Y-axis reflection test-time augmentation at val/test eval; predict on (x,y) AND (x,-y) with Uy sign flip; average outputs | ASSIGNED |
| nezuko | #2465 | norm-bias-no-wd | LayerNorm γ/β + Linear biases + placeholder + attn.temperature → wd=0 (standard transformer recipe, orthogonal to fern #2436) | IN FLIGHT |
| edward | #2441 | hybrid-rff-plus-learned-freqs | Additive GaussianRFF σ=3 on top of learned-freqs stack (m=6 fixed RFF concatenated after existing FourierCoordEnc output) | IN FLIGHT |
| tanjiro | #2488 | rmsnorm-qk-gamma | QK-norm v3 — RMSNorm-Q/K with learnable per-head per-channel γ (1280 new params, no-WD 10× lr group); preserves magnitude while controlling variance (Gemma/DeepSeek-V3 formulation) | IN FLIGHT |
| fern | #2515 | layerscale-per-block-lr | Per-block-separate LayerScale lr scaling (5×→15× linearly from block 0 to 4) — directly tests #2475's per-block heterogeneity finding (deep blocks have higher std/mean) | ASSIGNED |

## Key findings from Wave 13/14/15/16/17

**Gate activation axis (FULLY CLOSED at ReLU):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win)
- GELU → ReLU: **−1.92% val, −4.07% test** (15th compound win)
- ReLU → Squared ReLU: **+5.19%** (CLOSED) — catastrophic
- ReLU → AbsGLU: **+10.35%** (#2385 CLOSED) — bidirectional gate destabilizes; one-sided dead zone is essential
- **Gate axis closed at ReGLU = max(0,x)**; triple-confirmed on negative side (AbsGLU, squared-ReLU both fail)

**Capacity axis (CLOSED):**
- inner_dim=256 (baseline) → 288 (MERGED, 16th win) → 320 (CLOSED #2386: +6.0%, overfit despite full budget)
- **inner_dim axis closed at 288**; bias-variance frontier confirmed between 288 and 320
- n_hidden: closed at 128 (#2371 +19.18%, quadratic scaling under 30-min budget)

**Encoder / Fourier axis:**
- Fixed dyadic freqs: closed at L=6
- Learnable freqs (#2370 MERGED, 17th win): −3.73% — bottom 3 freqs adapted strongly, top 3 gradient-limited
- **Wave 17 follow-ups**: freq init at equilibrium (frieren), 50× lr to unlock top freqs (thorfinn)

**Optimizer-group insight (Wave 17 new axis — refined after #2437):**
- Original framing: "scale/frequency parameters are systematically under-trained at default WD=1e-4 + lr=5e-4"
- **Refined framing (post-#2437)**: the no-WD + high-lr treatment WORKS for **additive scale params** (freqs, LayerScale γ — multiply activations) but FAILS for **softmax-internal scale params** (slice temperature, QK temperature — govern combinatorial routing)
- Freq insight (#2370 MERGED): +17th win (−3.73%)
- Slice-temp at 10× lr (#2437 CLOSED Outcome C): +4.81%; block 0 collapsed uniformly to 0.185, destabilizing token-to-slice routing
- Still testing: LayerScale γ at 10× lr (fern #2436 — additive scale, should win), QK temperature at no-WD + init=0 (tanjiro #2427 — softmax-internal but with proper init), norm/bias at default lr no-WD (nezuko #2465 — broad standard-transformer recipe)

**Sampling axis (CLOSED for camber_rc upsampling):**
- OOD upsampling (#2391): +8.22% — camber_rc bottleneck is geometric extrapolation, NOT data density; upsampling collapses in-dist

## Permanently closed axes (do not re-test)

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum |
| Cosine T_max | T_max=14 (per-batch) | #2308: T_max=12 +3.24% |
| LR warmup | epoch-1 linear ramp | Merged |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L (fixed) | L=6 dyadic | L=8 plateau; now learned, see #2370 |
| LayerScale init (symmetric) | γ_l=0.025 | Sweep complete; asymmetric test in #2414 |
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | slice_num=96 +8.82%, dilution |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal |
| Gate: all except ReGLU | ReGLU (ReLU) | SiLU<GELU<ReLU<AbsGLU<SqReLU all tested; ReLU optimum |
| n_hidden (residual stream) | 128 | #2371: quadratic scaling, compute-bound |
| inner_dim | 288 | #2386: 320 over-fits; 256 under-fits |
| OOD domain upsampling (camber_rc) | equal weights | #2391: extrapolation gap, not density gap |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Per-sample scalar Fourier | concat | #2286 class falsified |
| Hybrid dyadic+RFF σ=1.0 | dyadic L=6 | #2309 redundant low-freq overlap |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| Slice temperature at 10× lr | n/a (don't raise lr) | #2437 CLOSED Outcome C: +4.81%; softmax-internal routing destabilized |
| Learned freqs at 50× lr | n/a (lr ≥ 10× is plenty) | #2435 CLOSED: top freqs are gradient-magnitude-limited, NOT lr-limited |
| Asymmetric LayerScale init (attn>mlp) | symmetric 0.025 | #2414 CLOSED Outcome C: +7.92% vs current; optimizer inverts the asymmetry (final mlp γ > attn γ in all 5 blocks) — the hypothesized direction was wrong |
| F.normalize-based QK-norm | not used | #2377 + #2427 BOTH CLOSED: v1 init confound + v2 magnitude collapse (Q/K unit-sphere destroys per-query magnitude variance needed for attention sharpening); RMSNorm variant tested in #2488 |
| Freqs-xy-separate | unified 6 freqs | #2469 CLOSED Outcome B: directional asymmetry IS real at low freqs (37% asymmetry at freq[1]) but top freqs [8,16,32] stayed pinned in both directions; **top-freq gradient-pin NOT caused by directional cancellation** (mesh-scale aliasing remains the leading hypothesis) |
| Freq-init at equilibrium | dyadic init | #2434 CLOSED Outcome C +4.16%; equilibrium init OVERSHOOTS rather than relaxes — same anti-correlated-init pattern as #2475 LayerScale γ; the no-WD 10× lr group has a flat landscape where init bias compounds |
| FiLM with zero-init γ/β heads | no FiLM | #2453 CLOSED Outcome C +4.45%; FiLM IS active but generalizes in wrong direction — helps single_in_dist (−4.86%), hurts all OOD splits (camber_cruise +16.46%); zero-init heads can't build robust per-cond representations in 12 epochs |
| SWA (last-N epochs averaging) | no SWA | #2476 CLOSED Outcome C +7.79%; cosine T_max=14 + 30-min cap means val_avg drops 9.7% in single epoch (e11→e12); model still mid-descent at training cut-off; no plateau to average over |

## Prioritized open research themes (Wave 17+)

**Newly active (assigned in this iteration)**:
1. **Decoupled LayerScale init** (alphonse #2510 NEW): `attn=0.025, mlp=0.1` — directly tests #2475's anti-correlated-attractor finding (does starting each path at its natural attractor avoid wrong-init reversal cost?)
2. **Per-block LayerScale lr scaling** (fern #2515 NEW): block 0→5× lr, block 4→15× lr — tests #2475's per-block std/mean heterogeneity (deep blocks have more variable γ)
3. **Per-feature input gate** (thorfinn #2511 NEW): 44 learnable params at input position in no-WD 10× lr group — extends additive-scale theme to a new architecture position
4. **Stoch-depth deep-concentrated** (frieren #2513 NEW): `[0,0,0.05,0.10,0.15]` vs current linear `[0,0.025,0.05,0.075,0.1]` — anti-redundancy theory; deeper blocks should tolerate more drop
5. **Reflection TTA** (askeladd #2514 NEW): y-axis reflection test-time augmentation at val/test eval; risk-free (no retraining, doubles eval cost only)

**In flight**:
6. **Norm-bias no-WD** (nezuko #2465 IN FLIGHT): broad standard-transformer recipe — LN γ/β + Linear bias + placeholder + attn temp → wd=0
7. **Hybrid RFF + learned freqs** (edward #2441 IN FLIGHT but stale, may need re-check): additive Gaussian σ=3 RFF ON TOP of current learned-freqs stack
8. **RMSNorm-Q/K + learnable γ** (tanjiro #2488 IN FLIGHT): QK-norm v3 — preserves magnitude while controlling variance (Gemma/DeepSeek-V3 formulation)

**Future ideas (queued)**:
9. **All-param optimizer sweep**: if LayerScale + norm-bias both win, test joint no-WD config for all 1D params simultaneously
10. **Per-block learned freqs (30 params)**: natural escalation if direction-separation doesn't unlock — block-specific freq pools allow each block its own scale
11. **Slice temp init sweep (not lr sweep)**: if Wave 18 reopens this axis, try init ∈ {0.3, 0.7} at default lr — gradient signal in #2437 suggested sharper is preferred but lr was too aggressive
12. **Different conditioning architecture** (post-#2453): FiLM PER-BLOCK instead of at embedding; or explicit OOD regularization on conditioning head
13. **Auxiliary self-supervised loss**: predict reconstructed coords from internal representations (Kaggle-staple OOD-improver)
14. **Weight standardization on Linear layers**: standardize weight rows before forward pass (used in CIFAR/ImageNet recipes)
15. **Interpolation init sweep for LayerScale**: γ_init=0.05 (halfway between 0.025 and 0.1) — only tested 0.025 and 0.1, the curve shape between them is unknown
