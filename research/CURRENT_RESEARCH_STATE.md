# SENPAI Research State

- **Last updated**: 2026-05-13 14:00 UTC (Wave 14: MERGE #2266 thorfinn GeGLU (−4.75%, 14th compound win); CLOSE #2262/#2244/#2239/#2225/#2098; Assigning new work to 6 students)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2266 merged — GeGLU gate, 14th compound win)

- `val_avg/mae_surf_p` = **64.182** (GeGLU MLP gate + SwiGLU inner_dim=256 + LR warmup + LayerScale init=0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 13)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **56.523**
- Per-split val: single_in_dist=67.894 / camber_rc=76.235 / camber_cruise=47.790 / re_rand=64.808
- Per-split test: single_in_dist=60.676 / camber_rc=70.778 / camber_cruise=39.001 / re_rand=55.636
- Δ vs PR #2175 baseline (67.381 / 57.800): **−4.75%** val_avg, **−2.21%** test_avg
- **Mechanism (GeGLU)**: single-char `F.silu → F.gelu` in SwiGLUMLP.forward(). GELU's harder switch for x<0 suppresses cross-channel contamination at stagnation/wake pressure-peak regions. Under L1+surf-ch-weight [0.5,0.5,2.0], gradient is dominated by extreme p values → gate sharpness matters most at those features. Zero param cost; same best epoch (13).
- Compound progress: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→**#2266** → val_avg has improved from 100.957 to **64.182** = **−36.4% over 14 merges**.

## Current research focus

**Wave 14 — Gate activation axis + Fourier encoder variants + orthogonal probes.**

The compound stack has 14 merged wins (100.957 → 64.182 = **−36.4%**). The GeGLU win (−4.75%) is the largest single-PR gain from a 1-character change, confirming that gate activation sharpness is an important lever on the current stack. Two axes remain open: (1) gate activation sharpness (SiLU < GELU < ReLU — need ReGLU to close the monotonicity question), and (2) Fourier encoder structure (dyadic wins over pure-Gaussian RFF, but hybrid dyadic+RFF and learned frequencies untested).

**Wave 14 active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| askeladd | #2286 | flow-cond-fourier-re-aoa | Fourier encode log_Re+AoA0+AoA1 dims (n_freqs=2); +12 features, fun_dim 44→56 | IN FLIGHT |
| tanjiro | #2281 | swiglu-inner-dim-320 | SwiGLU inner_dim 256→320: bisect between won-256 and lost-384. NOTE: branched pre-GeGLU; compare vs old SwiGLU baseline 67.381, then evaluate vs new 64.182 | IN FLIGHT |
| thorfinn | #2304 | reglu-gate | ReGLU gate: F.gelu → F.relu in SwiGLUMLP. Closes gate-sharpness monotonicity (SiLU<GELU<ReLU). | WIP |
| alphonse | #2308 | cosine-tmax-12 | T_max=14→12: calibrate cosine schedule to actual ~12-13 epoch budget post-SwiGLU/GeGLU | WIP |
| edward | #2309 | hybrid-fourier-dyadic-rff | Keep dyadic L=6 + concatenate small Gaussian RFF block (m=6, σ=1.0): combine high-freq precision + OOD-cruise generalization | WIP |
| fern | #2306 | geglu-inner-dim-320 | GeGLU + inner_dim=320: does wider inner_dim compound with GeGLU gate win? | WIP |
| nezuko | #2310 | lr-bracket-up-7e-4 | LR=5e-4→7e-4: close upper side of LR bracket (only lower side tested) | WIP |
| frieren | #2312 | learned-fourier-freqs | FourierCoordEnc.freqs as nn.Parameter (dyadic init): let network learn optimal frequency spectrum | WIP |

## Key findings from Wave 13/14

**Gate activation axis (#2266 win + context):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win). Largest single-character gain.
- Mechanism: GELU < SiLU for x<0 → sharper gate suppresses cross-channel contamination at high-magnitude p features
- Gate-sharpness monotonicity hypothesis: SiLU < GELU — next test is ReLU (harshest: zero for x<0)
- If ReGLU also wins → activate monotonic, harder is better, try `abs(x)*x/2` or learned gates
- If ReGLU regresses → GELU is the optimal operating point (Goldilocks gate sharpness)

**Encoder axis (from #2225 Gaussian RFF + in-flight #2286):**
- Pure Gaussian RFF acts as low-pass filter vs dyadic broadband — misses high-freq pressure structure
- But RFF *did* improve OOD-cruise splits (+1-2%) across both σ values consistently — signal is real
- Hybrid dyadic+RFF could capture both (in-dist high-freq + OOD smooth coverage)
- Learned frequencies (B as Parameter) is the most principled approach to discover optimal spectrum

**Lion optimizer (#2098 close):**
- Lion + SwiGLU: partially redundant (both address per-channel grad heterogeneity) + schedule mismatch (12 vs 14 epochs)
- **T_max schedule adjustment (T_max=11/12) is a clean follow-up** — frieren's student identified this correctly
- Assigning alphonse to test T_max adjustment (schedule, not optimizer)

**Depth/width/slice axes (all closed in this batch):**
- n_layers=6: closed (+5.43%), compute-bound + capacity-overfitting at 1499 samples
- slice_num=96: closed (+8.82%), representation dilution
- lr=3e-4: closed (+5.95%), schedule mismatch; upper bracket (7e-4) still open
- inner_dim=320 (tanjiro, SwiGLU): in-flight, compare vs old SwiGLU baseline
- inner_dim=320 (fern, GeGLU): now assigned — higher priority given GeGLU is the current stack

## Permanently closed axes (do not re-test)

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum |
| Cosine T_max | T_max=14 (per-batch) | T_max=15/50 both worse |
| LR warmup | epoch-1 linear ramp | Merged; T_max adjustment still open |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L | L=6 dyadic | L=8 plateau, L=4 baseline |
| LayerScale init | γ_l=0.025 | Sweep 0.1→0.05→0.025→0.0125 complete |
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum; vol-pressure-weight conflict |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Vol-loss ch-weight | off | Conflicts with surf-ch-weight; merged axis |
| Normalization | LayerNorm + β | RMSNorm +20.2%; mean-centering essential for SwiGLU |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | slice_num=96 +8.82%, dilution |
| LR bracket-down | lr=5e-4 | lr=3e-4 +5.95%, schedule mismatch |
| Lion optimizer | AdamW (with SwiGLU) | Partially redundant + schedule mismatch |
| AdamW betas | (0.9, 0.999) default | β₂=0.95 non-uniform regression |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Output-side calibration | off | log1p, γ-bias, per-channel all regressed |
| Gumbel/Ada-Temp slices | off | 3 tests, mechanism learned, outcome rejected |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| SmoothL1 / Huber | off | Absorbed by LayerScale |
| Adaptive grad-clip | off | Over-clips on LayerScale-attenuated stack |
| Gaussian RFF (pure) | dyadic L=6 | Low-pass filter, in-dist regression |

## Prioritized open research themes (Wave 15+)

1. **ReGLU gate** (thorfinn, assigned): closes gate-sharpness axis; if monotonic, try harder gates
2. **GeGLU inner_dim=320** (fern, assigned): does inner_dim expansion compound with GeGLU win?
3. **Hybrid dyadic+RFF encoder** (edward, assigned): combine structural high-freq + smooth OOD coverage
4. **Learned Fourier frequencies** (frieren, assigned): B as nn.Parameter — most principled frequency learning
5. **LR bracket-up 7e-4** (nezuko, assigned): close LR axis from upper direction
6. **T_max adjustment** (alphonse, assigned): calibrate cosine schedule to actual ~12-epoch budget
7. **FlowCond Fourier** (askeladd, in-flight): non-spatial Fourier for log_Re/AoA — could open val_re_rand
8. **SwiGLU inner_dim=320** (tanjiro, in-flight): if beats SwiGLU-256, also need GeGLU-320 comparison
9. **Width increase n_hidden=144**: only if inner_dim hits saturation
10. **Data augmentation**: MixUp across foil geometries, or Re/AoA perturbation (after architecture plateau)
