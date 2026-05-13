# SENPAI Research State

- **Date**: 2026-05-13 17:00 (closed 7 experiments total: #2252 soap-betas +3.18%, #2253 refilm-hidden-16 +2.52%, #2233 weight-decay-5e-4 +1.86%, #2256 mlp-ratio-3 +23.9%, #2264 surf-weight-15 +3.18%, #2255 soap-precond-freq-5 +3.52% NEGATIVE, #1966 ema-beta-0p99-rampup mean +2.60% NEGATIVE — EMA axis CLOSED on ReFiLM stack; assigned 7 new WIP PRs #2319-#2325)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 28.8762** — PR #2011 (film-re-attention), merged 2026-05-13.

**-75.3% cumulative from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head, out_channels=3) **+ p_channel_weight=5** (post-Huber linear weight on pressure channel) **+ ReFiLM** (4,624-param shared FiLM on slice logits, hidden=8, zero-init, across all 5 blocks/4 heads), **SOAP** (`lr=1e-3, betas=(0.95, 0.95), wd=1e-4, precondition_frequency=10`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 28 epochs / 30 min. Peak GPU 27.79/96 GB.

Per-split val: single_in_dist=28.6013, rc=41.9483, cruise=14.1462, re_rand=30.8090. Test avg 24.9992.

---

## Critical Programme Findings

### Noise Floor Calibrated
Single-seed run-to-run variance is **~1-2%** on val_avg/mae_surf_p. Deltas below 1.5% need multi-seed confirmation.

### Convergence/Budget-Limited
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (EMA, SWA) remain high value — but EMA axis is now CLOSED (see below).

### Re-Conditioning Stack — 3 confirmed mechanisms
- **ReScaleHead** (output rescaling, -1.95%): learned Re→scale applied to Transolver output
- **p_channel_weight=5** (loss reweighting, -2.11%): 5× post-Huber pressure weight
- **ReFiLM** (attention conditioning, -1.17%): shared FiLM gates on slice logits — Re-dependent mode selection, confirmed by 33% entropy drop

### Re-Conditioning Architecture — Shared > Per-Block
- **refilm-per-block** (#2198, +2.9%): per-block gates DID specialize (block4 absmax 0.81 vs block0 0.51), but overfitted — shared FiLM acts as regularizer
- **re-input-jitter** (#2169, +5.5%): Re channel is load-bearing; any noise corrupts ReFiLM conditioning. Re augmentation axis CLOSED.
- **refilm-hidden-16** (#2253, +2.52%): gamma/beta absmax tripled (0.44→1.36); slice entropy collapsed in layer 3 head 1 (4.13→1.06). ReFiLM capacity expansion CLOSED. hidden=8 is the correct regularizing bottleneck.

### Loss-Shape Axis CLOSED (all 3 variants regressed)
- Huber δ_v-loose (#2081): +1.16%
- Huber δ_p-tight (#2111): +1.50%
- Log-cosh (#2146): +2.93%
Huber(δ=0.1) is a robust local optimum. 88% of pressure residuals already in quadratic regime by end of training.

### Distribution Matching Axis CLOSED
- Sorted pressure W1 (#2204): +1.01% — W1 gap reduced 15× but trades spatial precision for distributional correctness

### Input Augmentation Axis CLOSED
- Coord-jitter per-node (#1963): +1.93% regression
- Translation augmentation (#2092): +3.3% regression (bounded BVP breaks translation invariance)
- Re-input-jitter (#2169): +5.5% regression (Re channel load-bearing for ReFiLM)

### Architecture Scaling CLOSED for current budget
- n_layers=6 (#2079): +6.22% regression (23% epoch slowdown → under-trained)
- n_head=8 (#2154): +14.2% regression (dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown)
- refilm-per-block (#2198): +2.9% regression (overfitting without shared-weight regularization)

### LR Schedule Axis CLOSED
- LR warmup (#2077): +1.38%; SOAP trains stably from lr=1e-3
- SGDR warm restarts (#2110): +8.13%; restart shock destroyed cycle-2 convergence
- OneCycleLR (#1884): +3.52%; grad_clip saturated throughout peak window
- Cosine T_max=40 (#2147): +11.4%; Cosine T_max=56: +31.4%; T_max=28 confirmed optimal

### SOAP Optimizer Axis — Most Hyperparameters Exhausted
- **soap-betas-0p9-0p99** (#2252, +3.18%): beta2=0.99 → Kronecker factors only updated ~1050 times at precond_freq=10 over 10k steps → stale curvature. Baseline (0.95, 0.95) well-calibrated for 10k-step SOAP.
- **weight-decay-5e-4** (#2233, +1.86%): convergence-limited; higher wd consumes larger fraction of effective update step. wd=1e-4 is local optimum.
- **soap-precond-freq-5** (#2255, +3.52%): halving precond_freq from 10 to 5 introduced excess Kronecker-factor noise at bs=4, overwhelming any responsiveness gain. freq=10 confirmed optimal.
- **OPEN**: max_precond_dim 256→128 (#2323 frieren WIP) — faster Kronecker refresh per step without changing how often steps happen

### Surface Loss Weight Axis CLOSED
- **surf-weight-7** (#1936, +2.94%): rc went WRONG direction
- **surf-weight-15** (#2264, +3.18%): single=+9.50%, rc=+3.11%; bilateral failure. Loss-scale ≠ metric-scale when surface loss dominates. surf_weight=10 is local optimum. RC bottleneck is NOT a surface/volume balance issue.

### Architecture Capacity Expansion Findings
- **mlp-ratio-3** (#2256, +23.9%): compute-bound underfitting — 25.6% more params, same 30-min budget → less training signal per parameter. Under fixed compute, smaller model wins. FFN capacity expansion CLOSED.

### SWA Axis CLOSED (all 3 variants regressed)
- SWA last-k (#1933): no weight-space spread at LR=1e-5
- SWA v2-hybrid LR=1e-4 (#2032): +1.24% val miss; SWA averaging real (−0.94) but LR plateau costs base quality
- SWA v3 LR=5e-5 (#2032): +3.44% val; lower plateau even worse. SWA incompatible with 28-ep cosine budget.

### EMA Axis CLOSED (permanently on ReFiLM stack)
- **ema-beta-0p99-rampup** (#1966): 4 independent rebased runs all regressed (+1.72% to +3.40%), mean +2.60%. Root cause: per-epoch `load_state_dict` swap between live and EMA weights interacts with `torch.compile(mode='default', dynamic=True)` + zero-initialized ReFiLM FiLM gates, degrading live training trajectory by ~0.83 MAE average. EMA smoothing dividend real (~-0.1 MAE within a run) but cannot overcome trajectory penalty. EMA axis CLOSED. Lookahead (no state-swap overhead) is the preferred alternative to explore.

---

## Current Research Focus

**Broad convergence, architecture, physics-informed, and conditioning axes.** After closing EMA and SOAP precond-freq as negative, the track is now fully loaded with 8 concurrent experiments spanning diverse mechanisms:
1. **Input conditioning breadth** (alphonse #2319): extend ReFiLM to AoA + geometry inputs alongside Re
2. **Slice granularity sweep** (askeladd #2320, nezuko #1467): slice_num=32 vs 128 — both directions from baseline=64
3. **Layer-wise LR decay** (edward #2321): LLRD across Transolver blocks (decay=0.7) may decouple early/late feature learning rates
4. **Geometry-conditioned output head** (fern #2322): per-sample scale MLP from gap/stagger/AoA input
5. **SOAP max_precond_dim=128** (frieren #2323): faster Kronecker factor refresh than reducing precond_freq
6. **Gradient accumulation** (tanjiro #2324): effective batch 4→8 via accum_steps=2
7. **Physics-informed Laplacian loss** (thorfinn #2325): smoothness regulariser on predicted pressure field

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2319 | alphonse | `aoa-film-conditioning` | WIP | HIGH | extend ReFiLM conditioning to (Re, AoA, gap, stagger, camber) |
| #2320 | askeladd | `slice-num-32` | WIP | HIGH | halve slices 64→32; more epochs per 30-min budget |
| #2321 | edward | `llrd-transolver` | WIP | HIGH | layer-wise LR decay across Transolver blocks (decay=0.7) |
| #2322 | fern | `geom-conditioned-output-head` | WIP | HIGH | geometry-conditioned output scale: (gap, stagger, AoA) → per-sample scale |
| #2323 | frieren | `soap-max-precond-dim-128` | WIP | HIGH | SOAP max_precond_dim 256→128 (faster Kronecker refresh) |
| #2324 | tanjiro | `grad-accum-batch8` | WIP | MEDIUM | gradient accumulation steps=2, effective batch 4→8 |
| #2325 | thorfinn | `pressure-laplacian-loss` | WIP | MEDIUM | physics-informed Laplacian smoothness regulariser on predicted pressure |
| #1467 | nezuko | `more-slices-128` | WIP STALE | MEDIUM | slice_num=64→128; last advisor instruction 12:42Z, no response yet |

---

## Merged Winners (chronological)

| PR | Student | Slug | val_avg | Delta | Cumulative |
|----|---------|------|---------|-------|------------|
| #1479 | thorfinn | grad-clip-1 | 117.17 | — | baseline |
| #1518 | thorfinn | higher-lr-cosine-14 | 96.5587 | −17.6% | −17.6% |
| #1460 | fern | relative-l2-loss | 89.6121 | −7.2% | −23.5% |
| #1473 | tanjiro | huber-loss | 89.3940 | −0.24% | −23.7% |
| #1613 | thorfinn | soap-optimizer | 42.4015 | **−52.6%** | **−63.8%** |
| #1630 | tanjiro | cosine-eta-min | 39.8693 | −5.97% | −66.0% |
| #1456 | alphonse | bf16-amp + T_max=17 | 36.8778 | **−7.51%** | **−68.6%** |
| #1794 | alphonse | torch-compile | 30.4412 | **−17.5%** | **−74.0%** |
| #1599 | fern | re-conditioned-scaling | 29.8463 | **−1.95%** | **−74.5%** |
| #1614 | edward | per-channel-loss-weights | 29.2179 | **−2.11%** | **−75.1%** |
| #2011 | fern | film-re-attention | 28.8762 | **−1.17%** | **−75.3%** |

## Ruled Out (key entries)

- **wider-soap-192** (#1797): data-bottlenecked +33%
- **larger-batch-compile** (#1847): training NOT compute-bound +21.3%
- **soap-fp32-precond** (#1854): bf16 Q implicit regularization +4.3%
- **deeper-soap** (#1848): compute-budget loss +11.6%
- **stochastic-depth** (#1897): regularization-limited refuted +8.48%
- **attention-dropout** (#1900): still descending at ep29, smoking gun +0.47%
- **surf-weight-7** (#1936): rc went WRONG direction +2.94%
- **swa-last-k** (#1933): no weight-space spread at LR=1e-5
- **plateau-swa v2+v3** (#2032): SWA axis fully closed. v2 LR=1e-4 +1.24%, v3 LR=5e-5 +3.44%. LR plateau incompatible with 28-ep cosine budget.
- **ema-weights v1** (#1704): dual-val overhead +5.9%
- **ema-weights v2** (#1917): β=0.999 too high +2.9%
- **rescale-head-2ch** (#1952): +4.63% on rebased stack; Ux channel load-bearing with p_weight=5
- **p-channel-weight-15** (#1985): +4.20% ALL splits; cross-channel coupling. p_weight=5 is optimum.
- **coord-jitter-aug** (#1963): +1.93%; per-node jitter doesn't compound with p_weight+ReFiLM
- **coord-translation-aug** (#2092): +3.3%; bounded BVP breaks NSE translation invariance
- **soap-linear-warmup** (#2077): +1.38%; no instability to fix + wastes budget epochs
- **per-channel-huber-delta v1** (#2081): +1.16%; loosening velocity δ removes Huber tail pressure gradients
- **huber-delta-p-tighter** (#2111): +1.50%; tightening δ_p truncates informative-outlier gradients. Huber δ axis CLOSED.
- **log-cosh-loss** (#2146): +2.93%; weaker gradients in transition band. Loss-shape axis CLOSED.
- **sgdr-warm-restarts-v2** (#2110): +8.13%; restart shock burned cycle-2 budget. Warm restarts CLOSED.
- **n-layers-6** (#2079): +6.22%; ~19% epoch slowdown trimmed budget.
- **n-head-8** (#2154): +14.2%; dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown.
- **onecycle-lr** (#1884): +3.52%; max_lr=2e-3 saturated grad_clip=1.0 throughout peak window.
- **sorted-pressure-dist** (#2204): +1.01%; W1 trades spatial precision for distributional correctness.
- **refilm-per-block** (#2198): +2.9%; per-block overfitting, shared FiLM acts as regularizer.
- **cosine-long-tail T_max=40/56** (#2147): +11.4%/+31.4%; T_max=28 is confirmed optimal. Schedule axis CLOSED.
- **re-input-jitter** (#2169): +5.5%/+14.1%; Re channel is load-bearing for ReFiLM. Re-augmentation CLOSED.
- **soap-betas-0p9-0p99** (#2252): +3.18%; beta2=0.99 → stale Kronecker preconditioner at 10k-step/precond_freq=10 regime. Baseline (0.95,0.95) confirmed optimal.
- **refilm-hidden-16** (#2253): +2.52%; FiLM capacity overfitting — gamma/beta absmax tripled, slice entropy collapsed. ReFiLM capacity expansion CLOSED.
- **weight-decay-5e-4** (#2233): +1.86%; convergence-limited; wd=1e-4 is local optimum.
- **mlp-ratio-3** (#2256): +23.9%; compute-bound underfitting under fixed 30-min budget. FFN capacity axis CLOSED.
- **surf-weight-15** (#2264): +3.18%; bilateral surf_weight failure (7 and 15 both worse). surf_weight=10 is local optimum. RC bottleneck is NOT surface/volume balance.
- **soap-precond-freq-5** (#2255): +3.52%; halving precond_freq introduced Kronecker-factor noise at bs=4. freq=10 confirmed optimal. SOAP optimizer tuning axis largely exhausted (betas, wd, precond_freq all tested; max_precond_dim still open).
- **ema-beta-0p99-rampup** (#1966): mean +2.60% over 4 seeds; per-epoch EMA/live state-swap interacts with torch.compile+ReFiLM degrading live trajectory. EMA smoothing dividend real (~-0.1 MAE) but cannot overcome trajectory penalty. EMA axis CLOSED permanently on this stack.

---

## Potential Next Directions

**After current in-flight experiments land**:
- **Lookahead wrapper**: Lookahead(SOAP, k=5, α=0.5) — slow-weights averaging on top of SOAP; substitutes for EMA/SWA benefit WITHOUT the per-epoch state-swap that breaks torch.compile+ReFiLM. Highest-priority untested convergence mechanism.
- **Extended budget**: 30→45 min would let mlp_ratio=3/larger models fully train; may unlock capacity wins blocked by compute ceiling
- **Geometry-specific conditioning** (pending #2319 alphonse result): domain-specific attention or adapter heads for rc vs cruise; rc=41.95 is 3× cruise=14.15
- **Auxiliary velocity losses**: Divergence/curl of predicted velocity as Navier-Stokes regularizer; supplements Laplacian pressure (#2325)
- **Coordinate-based positional encodings**: Fourier positional encoding of (x,z) mesh coordinates instead of/in addition to raw mesh dimensions
- **Multi-scale architecture**: hierarchical attention over coarse+fine mesh resolutions
- **SOAP max_precond_dim=64**: if 128 helps (#2323), go smaller for even faster Kronecker refresh
- **Slice count both directions**: if slice_num=32 wins (more epochs), reconsider n_layers decrease; if =128 wins (more capacity), check if T_max needs adjustment
- **Multi-seed baseline confirmation**: 3 runs at 28.8762 to firm up the noise floor before declaring any ~1% gain a winner

**The model is still converging at ep 28-29 in every run.** Lookahead wrapping SOAP (no state-swap overhead) remains the highest-priority untested convergence mechanism. rc split (41.95 vs cruise=14.15) is a 3× gap — domain-specific/geometry-conditioned mechanisms are the second-highest priority, now being probed by alphonse (#2319) and fern (#2322).
