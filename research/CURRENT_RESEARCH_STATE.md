# SENPAI Research State

- **Last updated**: 2026-05-13 15:45 UTC (Wave 16: CLOSE #2312/#2309/#2286/#2308; ASSIGN #2368 askeladd FiLM, #2369 edward hybrid-σ-3, #2370 frieren freqs-no-wd, #2371 alphonse n-hidden-144)
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

**Wave 16 — Gate-sharpness continuation + global conditioning + spectral adaptation refinement.**

The compound stack has 15 merged wins (100.957 → 62.949 = **−37.7%**). Three Wave 15 Fourier-axis experiments closed: (#2312 learned-freqs borderline +1.00%, freqs under-trained; #2309 hybrid σ=1.0 redundant +8.14%; #2286 flow-cond Fourier on per-sample scalars falsified +11.47%). Three reglu-axis experiments remain in-flight (thorfinn squared-relu, fern reglu-288, nezuko stoch-0.05). Wave 16 reassigns the 3 Fourier-axis follow-ups: (a) FiLM-style global conditioning to replace failed flow-cond Fourier, (b) hybrid Fourier retest with σ=3.0 (the actual winning σ from #2225), (c) per-block learned freqs OR no-wd+higher-lr on freqs to unblock the under-training. The LR and depth axes are closed.

**Wave 16 active threads:**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| tanjiro | #2281 | swiglu-inner-dim-320 | SwiGLU inner_dim 256→320 (pre-ReGLU branch; compare vs old SwiGLU baseline 67.381) | IN FLIGHT |
| alphonse | #2371 | n-hidden-144 | n_hidden 128→144 width bump on ReGLU stack (replaces closed cosine-tmax-12) | ASSIGNED |
| thorfinn | #2359 | squared-relu-gate | F.relu(x)^2 gate in SwiGLUMLP — Primer-style monotonic continuation of ReGLU win | IN FLIGHT |
| fern | #2360 | reglu-inner-dim-288 | ReGLU + inner_dim=288: bisect between 256 (win) and 320 (B) on current ReGLU stack | IN FLIGHT |
| nezuko | #2361 | stoch-depth-0.05-reglu | Reduce max stoch-depth from 0.10 to 0.05 — ReGLU sparsity may reduce need for explicit drop | IN FLIGHT |
| askeladd | #2368 | flow-cond-film | FiLM γ/β = MLP(log_Re,AoA0,AoA1) modulation of TransolverBlock activations | ASSIGNED |
| edward | #2369 | hybrid-fourier-sigma-3 | Hybrid dyadic L=6 + Gaussian RFF m=6 σ=3.0 (winning σ from #2225) | ASSIGNED |
| frieren | #2370 | learned-freqs-no-wd-10x-lr | learned freqs in no-wd group, 10× lr multiplier, post-step clamp(0.1, 100) | ASSIGNED |

## Key findings from Wave 13/14/15/16

**Gate activation axis (fully confirmed monotonic):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win)
- GELU → ReLU: **−1.92% val, −4.07% test** (15th compound win; test gain > val gain)
- Gate-sharpness monotonicity: SiLU < GELU < ReLU — each harder gate wins
- OOD splits benefit more from harder gates than in-dist (camber_cruise, camber_rc, re_rand all improved most)
- val_single_in_dist small regression (+2.03) with ReGLU — harder gate may slightly over-suppress in-dist features
- **Next: Squared ReLU** (F.relu²) — Primer (So et al. 2021) — thorfinn #2359 in-flight
- If squared ReLU wins → axis still open; if not → ReLU is the gate optimum

**Encoder axis findings (#2225, #2286, #2309, #2312):**
- Pure Gaussian RFF (#2225 σ=3.0) acts as low-pass filter vs dyadic broadband — misses high-freq pressure structure, but improved OOD-cruise +1-2%
- Hybrid σ=1.0 (#2309) FAILED (+8.14%) — σ=1.0 RFF freq band overlaps dyadic's low octaves → redundancy, capacity dilution
- Per-sample scalar Fourier (#2286) FAILED (+11.47%) — falsified class; Fourier needs per-node spatial variation
- Learned freqs (#2312) borderline (+1.00% val, 3/4 OOD splits improved, in-dist regressed +5.35%); high-freq freqs stayed pinned at init; mechanism is real but under-trained
- **Wave 16 follow-ups**: σ=3.0 hybrid (edward), no-wd+10×lr learned-freqs (frieren), FiLM conditioning for scalars (askeladd)

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
| Gaussian RFF (pure σ=1.0) | dyadic L=6 | Low-pass filter, in-dist regression |
| Hybrid dyadic+RFF σ=1.0 | dyadic L=6 | #2309 redundant low-freq overlap; retry at σ=3.0 |
| Per-sample scalar Fourier | concat | #2286 class falsified — no spectral structure |

## Prioritized open research themes (Wave 16+)

1. **Squared ReLU gate** (thorfinn #2359 in-flight): F.relu(x)^2 — Primer-style gate; tests whether monotonicity continues past ReLU
2. **ReGLU inner_dim=288** (fern #2360 in-flight): bisect between 256 (win) and 320 (B) on current ReGLU stack
3. **Stoch-depth on ReGLU** (nezuko #2361 in-flight): reduce max drop 0.10→0.05; ReGLU sparsity may reduce need for explicit drop
4. **FiLM-style global conditioning** (askeladd, NEW): γ/β = MLP(log_Re,AoA0,AoA1) modulating block activations — proper mechanism for per-sample scalars
5. **Hybrid Fourier σ=3.0** (edward, NEW): retest hybrid with the actual #2225 winning σ; high-freq RFF complement
6. **Learned freqs no-wd + 10× lr** (frieren, NEW): unblock the under-trained 6-freq vector; let high-freq freqs actually move
7. **n_hidden=144 width** (alphonse #2371 NEW): residual-stream width bump (was cosine T_max=12 closed; schedule axis confirmed closed at T_max=14)
8. **SwiGLU inner_dim=320** (tanjiro #2281 in-flight): pre-ReGLU branch; compare vs old SwiGLU baseline 67.381
9. **Harder gates**: Cube ReLU, or learned per-channel gate (nn.PReLU); only if Squared ReLU wins
10. **Per-block learned freqs**: 5 × 6 = 30 freq params for spatial spectral carving — if frieren no-wd hits limits
11. **Width n_hidden=144**: only if gate/capacity axes saturate
