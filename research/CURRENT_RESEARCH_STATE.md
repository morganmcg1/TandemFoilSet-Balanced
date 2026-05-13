# SENPAI Research State

- **Date**: 2026-05-13 11:40 (reviewed #1884 onecycle-lr CLOSED +3.52% clip-saturated; assigned weight-decay-5e-4 #2233 alphonse)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 28.8762** — PR #2011 (film-re-attention), merged 2026-05-13.

**-75.3% cumulative from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head, out_channels=3) **+ p_channel_weight=5** (post-Huber linear weight on pressure channel) **+ ReFiLM** (4,624-param shared FiLM on slice logits, zero-init, across all 5 blocks/4 heads), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 28 epochs / 30 min. Peak GPU 27.79/96 GB.

Per-split val: single_in_dist=28.6013, rc=41.9483, cruise=14.1462, re_rand=30.8090. Test avg 24.9992.

---

## Critical Programme Findings

### Noise Floor Calibrated
Single-seed run-to-run variance is **~1-2%** on val_avg/mae_surf_p. Deltas below 1.5% need multi-seed confirmation.

### Convergence/Budget-Limited
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (EMA, SWA, OneCycleLR) remain high value.

### Re-Conditioning Stack — 3 confirmed mechanisms
- **ReScaleHead** (output rescaling, -1.95%): learned Re→scale applied to Transolver output
- **p_channel_weight=5** (loss reweighting, -2.11%): 5× post-Huber pressure weight
- **ReFiLM** (attention conditioning, -1.17%): FiLM gates on slice logits — Re-dependent mode selection, confirmed by 33% entropy drop

### Loss-Shape Axis CLOSED (all 3 variants regressed)
- Huber δ_v-loose (#2081): +1.16%
- Huber δ_p-tight (#2111): +1.50%
- Log-cosh (#2146): +2.93%
Huber(δ=0.1) is a robust local optimum. 88% of pressure residuals already in quadratic regime by end of training.

### Input Augmentation Axis CLOSED
- Coord-jitter per-node (#1963): +1.93% regression
- Translation augmentation (#2092): +3.3% regression (bounded BVP breaks translation invariance)

### Architecture Scaling CLOSED for current budget
- n_layers=6 (#2079): +6.22% regression (23% epoch slowdown → under-trained)
- n_head=8 (#2154): +14.2% regression (dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown, budget cut at epoch 23/28)

### LR Warmup for SOAP CLOSED
3-epoch LinearLR warmup regressed +1.38%. SOAP already trains stably from lr=1e-3.

### SGDR Warm Restarts CLOSED for 30-min budget
Restart shock destroyed cycle-2 convergence (#2110 +8.13%).

---

## Current Research Focus

**Re-conditioning axis expansion, distribution matching, and regularization**. Programme is now exploring:
1. **Budget-aware training**: EMA β=0.99 (frieren, rebasing), SWA plateau v3-lower-lr (edward, in-flight)
2. **Re-conditioning depth**: ReFiLM per-block (fern #2198 — 5 independent FiLMs vs shared)
3. **Cosine schedule extension**: T_max=40/56 (askeladd #2147)
4. **Re input robustness**: Gaussian noise on log(Re) (tanjiro #2169)
5. **Distribution matching**: Sorted pressure W1 loss (thorfinn #2204)
6. **Regularization**: SOAP weight decay 5× (alphonse #2233)
7. **Capacity**: slice_num=128 (nezuko #1467)

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2204 | thorfinn | `sorted-pressure-dist` | NEW | **HIGH** | W1 regularizer on sorted surface pressure quantiles; per-sample per-split pressure distribution matching |
| #2198 | fern | `refilm-per-block` | NEW | **HIGH** | 5 independent FiLMs (one per Transolver block) vs current single shared; +18K params; depth-specific Re gating |
| #2169 | tanjiro | `re-input-jitter` | WIP | **HIGH** | Gaussian noise σ=0.05/0.10 on log(Re) channel; targets re_rand OOD-Re |
| #2147 | askeladd | `cosine-long-tail` | WIP | **HIGH** | T_max=40/56 so cosine never completes within 28-ep budget; higher final LR |
| #2032 | edward | `plateau-swa` | WIP REBASE | **HIGH** | SWA over 1e-4 plateau; needs rebase onto 28.8762 |
| #1966 | frieren | `ema-beta-0p99-rampup` | WIP REBASE | **HIGH** | EMA β=0.99; needs rebase onto 28.8762 |
| #2233 | alphonse | `weight-decay-5e-4` | NEW | **HIGH** | SOAP wd 1e-4→5e-4 (5×); OOD generalization via stronger L2 regularization |
| #1467 | nezuko | `more-slices-128` | WIP STALE | MEDIUM | slice_num=128; baseline update sent |

All 8 students active.

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
- **ema-weights v1** (#1704): dual-val overhead +5.9%
- **ema-weights v2** (#1917): β=0.999 too high +2.9%
- **rescale-head-2ch** (#1952): +4.63% on rebased stack; Ux channel load-bearing with p_weight=5
- **p-channel-weight-15** (#1985): +4.20% ALL splits; cross-channel coupling. p_weight=5 is optimum.
- **coord-jitter-aug** (#1963): +1.93% on rebased stack; per-node jitter doesn't compound with p_weight+ReFiLM
- **coord-translation-aug** (#2092): +3.3% val; bounded BVP breaks NSE translation invariance
- **soap-linear-warmup** (#2077): +1.38% val; no instability to fix + wastes budget epochs
- **per-channel-huber-delta v1** (#2081): +1.16% val; loosening velocity δ removes Huber tail pressure gradients
- **huber-delta-p-tighter** (#2111): +1.50% val; tightening δ_p truncates informative-outlier gradients. Huber δ axis CLOSED.
- **log-cosh-loss** (#2146): +2.93% val; weaker gradients in transition band (tanh(1)≈0.76 < Huber 1.0). Loss-shape axis CLOSED.
- **sgdr-warm-restarts-v2** (#2110): +8.13% val; restart shock burned cycle-2 budget. Warm restarts CLOSED.
- **n-layers-6** (#2079): +6.22% val; ~19% epoch slowdown trimmed budget. n_layers=6 closed for current budget.
- **n-head-8** (#2154): +14.2% val; dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown → epoch 23/28 cutoff.
- **onecycle-lr** (#1884): +3.52% val; max_lr=2e-3 saturated grad_clip=1.0 throughout peak window (clip_frac=1.0 ep2-15); effective LR = max_lr/grad_norm_mean ≈ 2e-3/8 ≈ 2.5e-4. OneCycleLR closed.

---

## Potential Next Directions

**After current in-flight experiments land**:
- **Cosine T_max=40/56 results** (#2147): if long-tail helps, confirm best T_max and merge
- **EMA/SWA convergence** (#1966, #2032): high priority — model still improving at ep 28/29 in every run
- **ReFiLM per-block result** (#2198): if per-block gating helps, test per-block with wider hidden dim (hidden=16 vs 8)
- **Sorted distribution result** (#2204): if W1 regularizer helps, tune λ and test on val splits
- **Geometry-specific augmentation**: camber/chord perturbation for geom_camber splits
- **Auxiliary losses**: divergence/curl of predicted velocity as regularizer
- **Multi-seed baseline**: 3 runs to firm up noise floor at 28.88

**The model is still converging at ep 28-29 in every run.** Convergence-aware mechanisms remain the priority.
