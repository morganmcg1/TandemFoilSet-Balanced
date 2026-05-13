# SENPAI Research State

- **Last updated**: 2026-05-13 18:40 UTC (Wave 17: MERGE #2436 fern layerscale-lr-10x (18th compound win, −1.60% val, −1.59% test); CLOSE #2435 thorfinn 50× lr (mechanism falsified, top freqs gradient-limited; lift was n_params architectural confound); ASSIGN #2469 thorfinn freqs-xy-separate (direction-separated learnable freqs))
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2436 merged — LayerScale γ no-WD + 10× lr, 18th compound win)

- `val_avg/mae_surf_p` = **58.6093** (ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + **LayerScale γ no-WD 10× lr** + LR warmup + LayerScale init 0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 12)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **50.7946**
- Per-split val: single_in_dist=70.160 / camber_rc=71.104 / camber_cruise=36.251 / re_rand=56.922
- Per-split test: single_in_dist=63.342 / camber_rc=63.135 / camber_cruise=29.692 / re_rand=47.009
- Δ vs prior PR #2370 baseline (59.5645 / 51.6141): **−1.60%** val_avg, **−1.59%** test_avg
- **Mechanism (LayerScale γ)**: no-WD + 10× lr for 1280 LayerScale γ params unblocks MLP-path amplification (γ shifted 4.6–6.2× off init=0.025 to 0.114–0.156) and attn-path per-channel sparse gating (std/mean 247–319%). Default WD=1e-4 was suppressing both. Camber_cruise val −7.49%, test −8.29%. Net OOD gains; test_single_in_dist +3.94% trade-off but net positive.
- **Wave 17 "additive scale" theme confirmed**: 2nd win after #2370 (freqs). Multiplicative-on-activations scale params benefit from no-WD + 10× lr. Softmax-internal scale (slice temp #2437, qk-norm v1 #2377) does NOT.
- **Compound progress**: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→#2266→#2304→#2360→#2370→**#2436** → val_avg has improved from 100.957 to **58.6093** = **−41.94% over 18 merges**.
- **n_params**: **892,637** — corrected (prior baseline entry's 831,197 was stale; pre-dated the inner_dim=288 merge but was claimed on the merged stack)

## Current research focus

**Wave 17+ — Refined optimizer-group axis + attention mechanisms + conditioning.**

After Wave 17 partial results, the optimizer-group axis split into two empirically distinct classes:
- **Additive scale params** (multiply activations linearly): freqs (#2370 MERGED, −3.73%), LayerScale γ (#2436 MERGED, −1.60%). Both win at no-WD + 10× lr.
- **Softmax-internal scale params** (govern combinatorial routing): slice temperature (#2437 CLOSED Outcome C, +4.81%), QK temperature v1 (#2377 CLOSED). Fast updates destabilize routing.

Still-pending Wave 17 tests:
1. ~~freq-init-equilibrium (frieren)~~ — still in flight #2434
2. ~~learned-freqs-50x-lr (thorfinn)~~ — **CLOSED #2435** (mechanism falsified; top freqs gradient-limited)
3. ~~layerscale-lr-10x (fern)~~ — **MERGED #2436** (18th win)
4. ~~slice-temp-lr-10x (nezuko)~~ — **CLOSED #2437** (softmax-internal scale falsifier)

New Wave 17+ threads testing refined hypotheses:
- `norm-bias-no-wd` (nezuko #2465): LayerNorm γ/β + biases + placeholder + attn temp → wd=0 (standard transformer recipe)
- `freqs-xy-separate` (thorfinn #2469): direction-separated learnable freqs — tests whether top-freq gradient cancellation was between x and y directions
- `flow-cond-film-v2` (askeladd #2453): FiLM γ/β = MLP(log_Re, AoA0, AoA1) — global conditioning

**Wave 17+ active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| frieren | #2434 | freq-init-equilibrium | Start Fourier freqs at #2370 equilibrium [0.75,1.46,3.44,8,16,32] + keep 10× lr — architectural vs dynamic mechanism test | IN FLIGHT |
| thorfinn | #2469 | freqs-xy-separate | Direction-separated learnable freqs (6 for x, 6 for y) — tests whether top-freq gradient cancellation was between x and y directions | ASSIGNED |
| nezuko | #2465 | norm-bias-no-wd | LayerNorm γ/β + Linear biases + placeholder + attn.temperature → wd=0 (standard transformer recipe, orthogonal to fern #2436) | ASSIGNED |
| askeladd | #2453 | flow-cond-film-v2 | FiLM γ/β = MLP(log_Re,AoA0,AoA1) — refreshed (v1 stale, never picked up); zero-init heads AFTER apply, cond_indices=[35,36,40] | ASSIGNED |
| edward | #2441 | hybrid-rff-plus-learned-freqs | Additive GaussianRFF σ=3 on top of learned-freqs stack (m=6 fixed RFF concatenated after existing FourierCoordEnc output) | ASSIGNED |
| alphonse | #2414 | attn-layerscale-0.05 | Dual LayerScale init — attn γ=0.05, mlp γ=0.025 asymmetric | IN FLIGHT |
| tanjiro | #2427 | qk-norm-temp-init-0 | QK-norm v2 — log_temp=0 (qk_scale=1.0) + exclude tau from WD | IN FLIGHT |
| fern | TBD | (idle after #2436 merge — new assignment pending) | — | IDLE |

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

## Prioritized open research themes (Wave 17+)

1. ~~LayerScale lr-10x~~ **MERGED #2436** (18th win, −1.60% val) — additive-scale theme confirmed
2. ~~Slice temperature lr-10x~~ **CLOSED #2437 Outcome C** (softmax-internal scale ≠ additive scale)
3. ~~Freq 50× lr~~ **CLOSED #2435** (top freqs gradient-limited, not lr-limited)
4. **Norm-bias no-WD** (nezuko #2465 NEW): broad standard-transformer recipe — LN γ/β + Linear bias + placeholder + attn temp → wd=0
5. **Direction-separated freqs** (thorfinn #2469 NEW): x and y get independent 6-freq vectors — tests gradient cancellation hypothesis for top freqs
6. **Freq equilibrium init** (frieren #2434 IN FLIGHT): was the #2370 win architectural or dynamic?
7. **FiLM conditioning** (askeladd #2453): proper global conditioning for Re/AoA scalars (refreshed from stale #2368 on post-#2370 stack)
8. **Hybrid RFF + learned freqs** (edward #2441): additive Gaussian σ=3 RFF ON TOP of current learned-freqs stack — tests orthogonality of mechanisms
9. **QK-norm v2** (tanjiro #2427 IN FLIGHT): with corrected init=0 and tau no-WD
10. **Asymmetric LayerScale** (alphonse #2414 IN FLIGHT): attn=0.05, mlp=0.025
11. **All-param optimizer sweep**: if LayerScale + norm-bias both win, test joint no-WD config for all 1D params simultaneously
12. **Per-block learned freqs (30 params)**: natural escalation if direction-separation doesn't unlock — block-specific freq pools allow each block its own scale
13. **Slice temp init sweep (not lr sweep)**: if Wave 18 reopens this axis, try init ∈ {0.3, 0.7} at default lr — gradient signal in #2437 suggested sharper is preferred but lr was too aggressive
14. **fern next assignment** (idle after #2436 merge): top candidates — (a) per-block LayerScale γ no-WD/no-share group (analog to per-block freqs), (b) MLP-only LayerScale 10× lr (test whether attn-path or MLP-path is dominant), (c) Wave 17+ additive-scale extension
