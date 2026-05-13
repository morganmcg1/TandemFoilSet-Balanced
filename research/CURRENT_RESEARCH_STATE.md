# SENPAI Research State

- **Last updated**: 2026-05-13 16:55 UTC (Wave 16: CLOSE #2377 tanjiro qk-norm v1 (Outcome B/init confound — log_temp=-1.733 too cold); ASSIGN #2427 tanjiro qk-norm-temp-init-0 (v2 with log_temp=0 + no WD on tau))
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2360 merged — ReGLU + inner_dim=288, 16th compound win)

- `val_avg/mae_surf_p` = **61.875** (ReGLU MLP gate + SwiGLU inner_dim=288 + LR warmup + LayerScale init=0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 12)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **54.117**
- Per-split val: single_in_dist=67.276 / camber_rc=72.143 / camber_cruise=45.901 / re_rand=62.181
- Per-split test: single_in_dist=60.873 / camber_rc=65.103 / camber_cruise=37.112 / re_rand=53.380
- Δ vs PR #2304 baseline (62.949 / 54.221): **−1.71%** val_avg, **−0.19%** test_avg
- **Mechanism (inner_dim=288)**: extra 32 channels per SwiGLU gate/up/down projection compensates for ReGLU's dead-channel sparsity. Epoch-budget hypothesis confirmed: +4.7% sec/epoch preserved 12-epoch window, val_single_in_dist IMPROVED −3.79% (vs prior +11.2% regression at 320). Wider capacity at right compute cost.
- Compound progress: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→#2266→#2304→**#2360** → val_avg has improved from 100.957 to **61.875** = **−38.7% over 16 merges**.

## Current research focus

**Wave 16 — Capacity + conditioning + spectral + attention + gate + LayerScale-asymmetry + OOD-sampling.**

The compound stack has 16 merged wins (100.957 → 61.875 = **−38.7%**). Key Wave 16 closures: #2359 squared-relu FAILED (+5.19% vs baseline — gate axis confirmed closed at ReLU); #2371 n_hidden=144 FAILED (+19.2% vs baseline — width axis closed at 128, quadratic param scaling makes residual-stream width untenable under 30-min cap); #2360 inner_dim=288 MERGED (16th win, −1.71%). Fresh assignment #2414: alphonse dual LayerScale (asymmetric attn=0.05/mlp=0.025 init) — tests whether the deeply-augmented MLP path (SwiGLU+ReGLU+inner=288) has miscalibrated the symmetric γ=0.025 optimum.

**Wave 16 active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| thorfinn | #2385 | abs-glu-gate | AbsGLU: replace relu→abs in SwiGLUMLP gate — symmetric threshold, no dead zone | ASSIGNED |
| fern | #2386 | reglu-inner-dim-320 | inner_dim=288→320 on ReGLU stack (epoch-budget mechanism confirmed at 288) | ASSIGNED |
| nezuko | #2391 | ood-upsampling | 2.5× camber_rc + 2× re_rand sampling weights — targets hardest OOD splits | ASSIGNED |
| askeladd | #2368 | flow-cond-film | FiLM γ/β = MLP(log_Re,AoA0,AoA1) modulation of TransolverBlock activations | IN FLIGHT |
| edward | #2369 | hybrid-fourier-sigma-3 | Hybrid dyadic L=6 + Gaussian RFF m=6 σ=3.0 (winning σ from #2225) | IN FLIGHT |
| frieren | #2370 | learned-freqs-no-wd-10x-lr | learned freqs in no-wd group, 10× lr multiplier, post-step clamp(0.1, 100) | IN FLIGHT |
| alphonse | #2414 | attn-layerscale-0.05 | Dual LayerScale init — attn γ=0.05, mlp γ=0.025 asymmetric; tests whether deep MLP augmentation miscalibrated symmetric γ optimum | ASSIGNED |
| tanjiro | #2427 | qk-norm-temp-init-0 | QK-norm v2 — log_temp=0 (qk_scale=1.0) + exclude tau from WD (v1 #2377 closed: cold init −1.733 made attention near-uniform, mechanism never activated) | ASSIGNED |

## Key findings from Wave 13/14/15/16

**Gate activation axis (FULLY CLOSED at ReLU):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win)
- GELU → ReLU: **−1.92% val, −4.07% test** (15th compound win; test gain > val gain)
- ReLU → Squared ReLU: **+5.19% val** (#2359 CLOSED) — quadratic amplification of positive activations catastrophic in continuous-regression regime; grad-norm peaked 120.6 in ep4; val_single_in_dist +13.83%
- ReLU = max(0, x) is exactly right: maximum sparsity + identity slope for positive survivors
- **AbsGLU** (thorfinn NEW): tests whether absolute-value (no dead zone, bidirectional sharpness) could win vs one-sided gate

**Capacity axis (inner_dim active, n_hidden CLOSED):**
- inner_dim=256 (baseline): val=62.949 (#2304 ReGLU)
- inner_dim=288 (fern #2360 MERGED): val=61.875 (−1.71%) — epoch budget confirmed; +4.7% sec/epoch
- inner_dim=320 (fern NEW): natural follow-up; ReGLU's simpler gate may keep 320 in 12-epoch window
- n_hidden=144 (alphonse #2371 CLOSED): +19.18% — width is quadratic-cost, lost 3 epochs to wall-clock cap; residual-stream width axis closed at 128

**Encoder axis findings (#2225, #2286, #2309, #2312):**
- Hybrid σ=1.0 (#2309) FAILED — redundant low-freq overlap
- Per-sample scalar Fourier (#2286) FAILED — class falsified
- Learned freqs (#2312) borderline — under-trained (freqs barely moved)
- **Wave 16 follow-ups**: σ=3.0 hybrid (edward), no-wd+10×lr learned-freqs (frieren), FiLM conditioning (askeladd)

**Schedule/optimizer axes (all closed):**
- T_max=12 (#2308 CLOSED) — +3.24% vs new baseline; runtime jitter makes T_max=14 more robust
- T_max=14 confirmed optimal; LR=5e-4 confirmed optimal; AdamW confirmed

## Permanently closed axes (do not re-test)

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum confirmed on ReGLU (#2361: 0.05 max +0.18% worse) |
| Cosine T_max | T_max=14 (per-batch) | #2308 alphonse: T_max=12 +3.24%, runtime jitter risk |
| LR warmup | epoch-1 linear ramp | Merged |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L | L=6 dyadic | L=8 plateau, L=4 baseline |
| LayerScale init | γ_l=0.025 | Sweep 0.1→0.05→0.025→0.0125 complete |
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Vol-loss ch-weight | off | Conflicts with surf-ch-weight |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | slice_num=96 +8.82%, dilution |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal; CLOSED |
| Lion optimizer | AdamW (post-SwiGLU) | Redundant + schedule mismatch |
| AdamW betas | (0.9, 0.999) default | β₂=0.95 non-uniform regression |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Output-side calibration | off | log1p, γ-bias, per-channel all regressed |
| Gumbel/Ada-Temp slices | off | 3 tests, mechanism learned, outcome rejected |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| SmoothL1 / Huber | off | Absorbed by LayerScale |
| Adaptive grad-clip | off | Over-clips on LayerScale-attenuated stack |
| Gaussian RFF (pure σ=1.0) | dyadic L=6 | Low-pass filter, in-dist regression |
| Hybrid dyadic+RFF σ=1.0 | dyadic L=6 | #2309 redundant low-freq overlap; retry at σ=3.0 |
| Per-sample scalar Fourier | concat | #2286 class falsified — no spectral structure |
| Gate: Squared ReLU | ReGLU (ReLU) | #2359 +5.19%; quadratic amplification breaks high-magnitude CFD regression |
| Gate: SiLU, GELU | ReGLU (ReLU) | SiLU<GELU<ReLU monotonic; axis closed at ReLU (pending AbsGLU test) |
| n_hidden (residual stream width) | 128 | #2371 +19.18%: quadratic param scaling (n_hidden²) drove n_params +26.07%, lost 3 epochs to wall-clock cap; width is wrong knob under 30-min compute budget |

## Prioritized open research themes (Wave 16+)

1. **AbsGLU gate** (thorfinn NEW): F.abs(x) gate — symmetric threshold, no dead zone; tests if exact-zero threshold vs absolute-value matters in gate design
2. **inner_dim=320 on ReGLU** (fern NEW): natural follow-up to 288 win; ReGLU simpler than GELU, may stay in 12-epoch window
3. **OOD-upweighted sampling** (nezuko #2391 NEW): 2.5× camber_rc + 2× re_rand training weight — directly targets the val_geom_camber_rc bottleneck (72.143, 16% above val_avg)
4. **FiLM-style global conditioning** (askeladd #2368): γ/β = MLP(log_Re,AoA0,AoA1) — proper mechanism for per-sample scalars
5. **Hybrid Fourier σ=3.0** (edward #2369): retest hybrid with the actual #2225 winning σ; high-freq RFF complement
6. **Learned freqs no-wd + 10× lr** (frieren #2370): unblock the under-trained 6-freq vector
7. **Dual LayerScale init** (alphonse #2414 NEW): asymmetric attn γ=0.05 / mlp γ=0.025 — tests whether the augmented MLP path (SwiGLU+ReGLU+inner=288) miscalibrated the symmetric LayerScale optimum
8. **QK-norm v2** (tanjiro #2427): log_temp init=0 (qk_scale=1.0) + tau excluded from WD — fixes v1 cold-init confound where max logit was 0.18 across 64 slice tokens (near-uniform softmax). v1 still showed mild test gain (−0.41%) → strong motivation
9. **inner_dim=304** bisect: if 320 is compute-bound again, test midpoint between 288 (win) and 320
10. **Per-block learned freqs**: 5 × 6 = 30 freq params — escalation if frieren no-wd still under-trains
