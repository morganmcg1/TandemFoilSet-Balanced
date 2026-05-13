# SENPAI Research State

- **Last updated**: 2026-05-13 18:00 UTC (Wave 17: CLOSE #2437 nezuko slice-temp-lr-10x (Outcome C, +4.81% val; load-bearing finding — softmax-internal scale ≠ additive scale under no-WD/high-lr regime); ASSIGN #2465 nezuko norm-bias-no-wd (standard transformer recipe))
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2370 merged — Learned Fourier freqs no-WD + 10× lr, 17th compound win)

- `val_avg/mae_surf_p` = **59.5645** (ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + LR warmup + LayerScale 0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 12)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **51.6141**
- Per-split val: single_in_dist=70.235 / camber_rc=71.466 / camber_cruise=39.185 / re_rand=57.372
- Per-split test: single_in_dist=60.940 / camber_rc=64.131 / camber_cruise=32.376 / re_rand=49.009
- Δ vs PR #2360 baseline (61.875 / 54.117): **−3.73%** val_avg, **−4.63%** test_avg
- **Mechanism (learned freqs)**: no-WD + 10× lr for 6-param Fourier freq vector unblocks bottom freqs [1,2,4] → [0.75,1.46,3.44] (−15 to −27% drift). Top freqs [8,16,32] stayed within ±1.5% (gradient-magnitude limited). Camber_cruise −14.62%, re_rand −7.73%. Orthogonal to MLP capacity axis (#2360).
- **Compound progress**: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→#2266→#2304→#2360→**#2370** → val_avg has improved from 100.957 to **59.5645** = **−41.0% over 17 merges**.
- **n_params**: 831,197 (+6 freq params over prior 831,191)

## Current research focus

**Wave 17 — Optimizer-group axis + attention mechanisms + conditioning.**

The 17th compound win (learned-freqs-no-WD-10×lr) revealed that default optimizer settings systematically under-train scale/frequency parameters. This opens a broad new axis: **which other model parameters benefit from the same treatment?** Wave 17 tests 4 orthogonal hypotheses:
1. `freq-init-equilibrium` (frieren): start freqs at #2370 equilibrium, keep 10× lr — tests architectural vs dynamic component of the win
2. `learned-freqs-50x-lr` (thorfinn): test if top freqs [8,16,32] are strictly gradient-magnitude limited or lr-limited at 50×
3. `layerscale-lr-10x` (fern): LayerScale γ params in 10× lr no-WD group — same optimizer insight
4. `slice-temp-lr-10x` (nezuko): slice attention temperature params in 10× lr no-WD group

Parallel to these, still-in-flight experiments cover conditioning (FiLM #2368), hybrid Fourier (#2369), attention QK-norm (#2427), and asymmetric LayerScale (#2414).

**Wave 17 active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| frieren | #2434 | freq-init-equilibrium | Start Fourier freqs at #2370 equilibrium [0.75,1.46,3.44,8,16,32] + keep 10× lr — architectural vs dynamic mechanism test | ASSIGNED |
| thorfinn | #2435 | learned-freqs-50x-lr | Keep dyadic init, raise lr from 10× to 50× — tests gradient-magnitude limit for top freqs | ASSIGNED |
| fern | #2436 | layerscale-lr-10x | LayerScale params (10 tensors, 1280 scalars) in 10× lr no-WD group — same optimizer insight as #2370 | ASSIGNED |
| nezuko | #2465 | norm-bias-no-wd | LayerNorm γ/β + Linear biases + placeholder + attn.temperature → wd=0 (standard transformer recipe, orthogonal to fern #2436) | ASSIGNED |
| askeladd | #2453 | flow-cond-film-v2 | FiLM γ/β = MLP(log_Re,AoA0,AoA1) — refreshed (v1 stale, never picked up); zero-init heads AFTER apply, cond_indices=[35,36,40] | ASSIGNED |
| edward | #2441 | hybrid-rff-plus-learned-freqs | Additive GaussianRFF σ=3 on top of learned-freqs stack (m=6 fixed RFF concatenated after existing FourierCoordEnc output) | ASSIGNED |
| alphonse | #2414 | attn-layerscale-0.05 | Dual LayerScale init — attn γ=0.05, mlp γ=0.025 asymmetric | IN FLIGHT |
| tanjiro | #2427 | qk-norm-temp-init-0 | QK-norm v2 — log_temp=0 (qk_scale=1.0) + exclude tau from WD | IN FLIGHT |

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

## Prioritized open research themes (Wave 17+)

1. **LayerScale lr-10x** (fern #2436): does the optimizer-group insight generalize from freqs to LayerScale γ?
2. ~~Slice temperature lr-10x~~ **CLOSED #2437 Outcome C** (softmax-internal scale ≠ additive scale); replaced with **norm-bias no-WD** (nezuko #2465)
3. **Freq 50× lr** (thorfinn #2435): can top freqs be unlocked with higher lr?
4. **Freq equilibrium init** (frieren #2434): was the #2370 win architectural or dynamic?
5. **FiLM conditioning** (askeladd #2453): proper global conditioning for Re/AoA scalars (refreshed from stale #2368 on post-#2370 stack)
6. **Hybrid RFF + learned freqs** (edward #2441 NEW): additive Gaussian σ=3 RFF ON TOP of current learned-freqs stack — tests orthogonality of mechanisms
7. **QK-norm v2** (tanjiro #2427): with corrected init=0 and tau no-WD
8. **Asymmetric LayerScale** (alphonse #2414): attn=0.05, mlp=0.025
9. **All-param optimizer sweep**: if LayerScale + norm-bias both win, test joint no-WD config for all 1D params simultaneously
10. **Per-block learned freqs (30 params)**: natural escalation if 50× lr doesn't unlock top freqs
11. **Slice temp init sweep (not lr sweep)**: if Wave 18 reopens this axis, try init ∈ {0.3, 0.7} at default lr — gradient signal in #2437 suggested sharper is preferred but lr was too aggressive
