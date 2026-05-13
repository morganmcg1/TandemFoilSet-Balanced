# SENPAI Research State

- **Last updated**: 2026-05-13 14:20 UTC (Wave 15: MERGE #2304 thorfinn ReGLU (−1.92%, 15th compound win); CLOSE #2306/#2310; Assigning thorfinn/fern/nezuko)
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2304 merged — ReGLU gate, 15th compound win)

- `val_avg/mae_surf_p` = **62.949** (ReGLU MLP gate + SwiGLU inner_dim=256 + LR warmup + LayerScale init=0.025 + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth; best @ ep 12)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **54.221**
- Per-split val: single_in_dist=69.925 / camber_rc=74.845 / camber_cruise=44.262 / re_rand=62.765
- Per-split test: single_in_dist=61.108 / camber_rc=66.196 / camber_cruise=36.305 / re_rand=53.276
- Δ vs PR #2266 baseline (64.182 / 56.523): **−1.92%** val_avg, **−4.07%** test_avg
- **Mechanism (ReGLU)**: `F.gelu → F.relu` in SwiGLUMLP.forward(). ReLU's exact-zero gate for x<0 maximally suppresses cross-channel contamination. Gate-sharpness monotonicity confirmed: SiLU<GELU<ReLU wins. OOD splits gain more than in-dist (test −4.07% vs val −1.92%). Run hit 30-min cap at ep 12 with val still descending — full potential likely higher.
- Compound progress: #1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896→#2018→#1754→#2105→#2175→#2266→**#2304** → val_avg has improved from 100.957 to **62.949** = **−37.7% over 15 merges**.

## Current research focus

**Wave 15 — Gate-sharpness continuation + Fourier encoder variants + capacity bisect.**

The compound stack has 15 merged wins (100.957 → 62.949 = **−37.7%**). The ReGLU win (−1.92% val, −4.07% test) confirmed the gate-sharpness monotonicity: SiLU→GELU→ReLU each improved, with OOD splits benefiting most from harder gates. The next question: does gate sharpness continue past ReLU (Squared ReLU = F.relu²)? Simultaneously, Fourier encoder axis (hybrid dyadic+RFF, learned frequencies) and capacity bisect (ReGLU+inner_dim=288) are in-flight. The LR and depth axes are now closed.

**Wave 15 active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| askeladd | #2286 | flow-cond-fourier-re-aoa | Fourier encode log_Re+AoA0+AoA1 dims; +12 features, fun_dim 44→56 | IN FLIGHT |
| tanjiro | #2281 | swiglu-inner-dim-320 | SwiGLU inner_dim 256→320 (pre-ReGLU branch; compare vs old SwiGLU baseline 67.381) | IN FLIGHT |
| alphonse | #2308 | cosine-tmax-12 | T_max=14→12 cosine schedule recalibration | IN FLIGHT |
| edward | #2309 | hybrid-fourier-dyadic-rff | Dyadic L=6 + Gaussian RFF m=6 σ=1.0 concatenated Fourier encoder | IN FLIGHT |
| frieren | #2312 | learned-fourier-freqs | FourierCoordEnc.freqs as nn.Parameter (dyadic init) | IN FLIGHT |
| thorfinn | #2359 | squared-relu-gate | F.relu(x)^2 gate in SwiGLUMLP — Primer-style monotonic continuation of ReGLU win | ASSIGNED |
| fern | #2360 | reglu-inner-dim-288 | ReGLU + inner_dim=288: bisect between 256 (win) and 320 (B) on current ReGLU stack | ASSIGNED |
| nezuko | #2361 | stoch-depth-0.05-reglu | Reduce max stoch-depth from 0.10 to 0.05 — ReGLU sparsity may reduce need for explicit drop | ASSIGNED |

## Key findings from Wave 13/14/15

**Gate activation axis (fully confirmed monotonic):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win)
- GELU → ReLU: **−1.92% val, −4.07% test** (15th compound win; test gain > val gain)
- Gate-sharpness monotonicity: SiLU < GELU < ReLU — each harder gate wins
- OOD splits benefit more from harder gates than in-dist (camber_cruise, camber_rc, re_rand all improved most)
- val_single_in_dist small regression (+2.03) with ReGLU — harder gate may slightly over-suppress in-dist features
- **Next: Squared ReLU** (F.relu²) — Primer (So et al. 2021) — tests whether monotonicity continues past ReLU
- If squared ReLU wins → axis still open; if not → ReLU is the gate optimum

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
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum (re-testing on ReGLU stack: nezuko #stoch-depth-0.05) |
| Cosine T_max | T_max=14 (per-batch) | T_max adjustment: alphonse #2308 in-flight |
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
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal near-tie; CLOSED |
| Lion optimizer | AdamW (post-SwiGLU) | Redundant + schedule mismatch |
| AdamW betas | (0.9, 0.999) default | β₂=0.95 non-uniform regression |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Output-side calibration | off | log1p, γ-bias, per-channel all regressed |
| Gumbel/Ada-Temp slices | off | 3 tests, mechanism learned, outcome rejected |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| SmoothL1 / Huber | off | Absorbed by LayerScale |
| Adaptive grad-clip | off | Over-clips on LayerScale-attenuated stack |
| Gaussian RFF (pure) | dyadic L=6 | Low-pass filter, in-dist regression |

## Prioritized open research themes (Wave 16+)

1. **Squared ReLU gate** (thorfinn, NEW): F.relu(x)^2 — Primer-style gate; tests whether monotonicity continues past ReLU
2. **ReGLU inner_dim=288** (fern, NEW): bisect between 256 (win) and 320 (B) on current ReGLU stack
3. **Stoch-depth on ReGLU** (nezuko, NEW): reduce max drop 0.10→0.05; ReGLU sparsity may reduce need for explicit drop
4. **FlowCond Fourier** (askeladd, in-flight #2286): non-spatial Fourier for log_Re/AoA — targets val_re_rand
5. **Hybrid dyadic+RFF Fourier** (edward, in-flight #2309): combine high-freq + smooth OOD coverage
6. **Learned Fourier frequencies** (frieren, in-flight #2312): discover optimal frequency spectrum
7. **Cosine T_max=12** (alphonse, in-flight #2308): schedule recalibration
8. **SwiGLU inner_dim=320** (tanjiro, in-flight #2281): pre-ReGLU branch; compare vs old SwiGLU baseline 67.381
9. **Harder gates**: Cube ReLU, or learned per-channel gate (nn.PReLU); only if Squared ReLU wins
10. **Width n_hidden=144**: only if gate/capacity axes saturate
