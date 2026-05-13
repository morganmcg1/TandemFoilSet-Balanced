# SENPAI Research State

- **Date**: 2026-05-13 13:45 (reviewed #2032 plateau-swa-v3 CLOSED +3.44%, #2204 sorted-pressure-dist CLOSED, #2198 refilm-per-block CLOSED, #2147 cosine-long-tail CLOSED, #2169 re-input-jitter CLOSED; assigned soap-betas-0p9-0p99 #2252 thorfinn, refilm-hidden-16 #2253 fern, soap-precond-freq-5 #2255 askeladd, mlp-ratio-3 #2256 tanjiro, surf-weight-15 #2264 edward)
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
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (EMA, SWA) remain high value.

### Re-Conditioning Stack — 3 confirmed mechanisms
- **ReScaleHead** (output rescaling, -1.95%): learned Re→scale applied to Transolver output
- **p_channel_weight=5** (loss reweighting, -2.11%): 5× post-Huber pressure weight
- **ReFiLM** (attention conditioning, -1.17%): shared FiLM gates on slice logits — Re-dependent mode selection, confirmed by 33% entropy drop

### Re-Conditioning Architecture — Shared > Per-Block
- **refilm-per-block** (#2198, +2.9%): per-block gates DID specialize (block4 absmax 0.81 vs block0 0.51), but overfitted — shared FiLM acts as regularizer
- **re-input-jitter** (#2169, +5.5%): Re channel is load-bearing; any noise corrupts ReFiLM conditioning. Re augmentation axis CLOSED.
- **Open**: refilm-hidden-16 (#2253) — tests if FiLM MLP capacity (hidden=8→16) helps without the overfitting risk of per-block independence

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

### SWA Axis CLOSED (all 3 variants regressed)
- SWA last-k (#1933): no weight-space spread at LR=1e-5
- SWA v2-hybrid LR=1e-4 (#2032): +1.24% val miss; SWA averaging real (−0.94) but LR plateau costs base quality
- SWA v3 LR=5e-5 (#2032): +3.44% val; lower plateau even worse. SWA incompatible with 28-ep cosine budget.

---

## Current Research Focus

**SOAP optimizer internals, model capacity, and convergence mechanisms.** Programme is now exploring:
1. **SOAP beta asymmetry**: betas (0.95, 0.95) → (0.9, 0.99) (thorfinn #2252 NEW)
2. **ReFiLM capacity**: shared FiLM hidden=8→16 (fern #2253 NEW)
3. **SOAP preconditioner**: precondition_frequency 10→5 (askeladd #2255 NEW)
4. **FFN capacity**: mlp_ratio 2→3 (tanjiro #2256 NEW)
5. **Surface loss weight**: surf_weight 10→15, targeting rc split (edward #2264 NEW)
6. **EMA β=0.99**: rampup from 0.9 (frieren #1966 WIP REBASE)
7. **Weight decay 5×**: SOAP wd 1e-4→5e-4 (alphonse #2233 NEW)
8. **slice_num=128**: Capacity doubling (nezuko #1467 WIP STALE)

All 8 students active.

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2252 | thorfinn | `soap-betas-0p9-0p99` | NEW | **HIGH** | SOAP betas (0.95,0.95)→(0.9,0.99); faster gradient response + stable precond |
| #2253 | fern | `refilm-hidden-16` | NEW | **HIGH** | ReFiLM hidden=8→16; wider shared FiLM MLP without per-block overfitting risk |
| #2255 | askeladd | `soap-precond-freq-5` | NEW | **HIGH** | precondition_frequency 10→5; 2× Kronecker factor refresh rate |
| #2256 | tanjiro | `mlp-ratio-3` | NEW | **HIGH** | mlp_ratio 2→3; ~+655K FFN params, ~2× model capacity |
| #2264 | edward | `surf-weight-15` | NEW | **HIGH** | surf_weight 10→15; targets rc split (41.95) using direction from #1936 |
| #1966 | frieren | `ema-beta-0p99-rampup` | WIP REBASE | **HIGH** | EMA β=0.99 from 0.9 rampup; needs rebase onto 28.8762 |
| #2233 | alphonse | `weight-decay-5e-4` | NEW | **HIGH** | SOAP wd 1e-4→5e-4 (5×); OOD generalization via stronger L2 regularization |
| #1467 | nezuko | `more-slices-128` | WIP STALE | MEDIUM | slice_num=128; baseline update sent |

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

---

## Potential Next Directions

**After current in-flight experiments land**:
- **SOAP beta/precond results** (#2252, #2255): if optimizer internals help, test combined (lower beta1 + higher beta2 + freq=5)
- **FFN capacity result** (#2256 mlp-ratio-3): if helps, also test wider hidden=192 (ruled out earlier at mlp_ratio=2 — worth retesting at ratio=3)
- **ReFiLM hidden-16 result** (#2253): if helps, test hidden=32 or combined with mlp-ratio-3
- **EMA convergence** (#1966): still the priority — model converges monotonically to cutoff in every run; SWA now closed
- **Geometry-specific augmentation**: camber/chord perturbation for geom_camber splits (rc split remains worst at 41.95)
- **Auxiliary losses**: divergence/curl of predicted velocity as regularizer
- **Multi-seed baseline**: 3 runs to firm up noise floor at 28.88

**The model is still converging at ep 28-29 in every run.** Convergence-aware mechanisms (EMA, SWA) remain the top priority — if either lands, it should be merged immediately.
