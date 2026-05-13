# SENPAI Research State

- **Date**: 2026-05-13 09:25 (reviewed #2077 soap-linear-warmup closed; assigned tanjiro coord-translation-aug #2092, askeladd sgdr-warm-restarts #2095)
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
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (EMA, SWA, SGDR warm restarts, OneCycleLR) are high value.

### Re-Conditioning Stack — 3 confirmed mechanisms
- **ReScaleHead** (output rescaling, -1.95%): learned Re→scale applied to Transolver output
- **p_channel_weight=5** (loss reweighting, -2.11%): 5× post-Huber pressure weight  
- **ReFiLM** (attention conditioning, -1.17%): FiLM gates on slice logits — Re-dependent mode selection, confirmed by 33% entropy drop

### Loss-Weighting Axis CLOSED
p_weight=5 is the optimum. p_weight=15 regressed +4.20% — cross-channel backbone coupling prevents monotonic improvement.

### Input Augmentation Axis — per-node jitter CLOSED
Coord-jitter (per-node, +1.93% regression on rebased stack) does not compound with p_weight+ReFiLM stack. **Translation augmentation** (whole-mesh rigid shift, NSE-invariant) is being tested as a physically valid alternative.

### LR Warmup for SOAP CLOSED
3-epoch LinearLR warmup regressed +1.38% val. SOAP already trains stably from lr=1e-3 — no instability to fix. Warmup wastes budget epochs. Direction closed.

---

## Current Research Focus — Convergence + Budget-Aware Mechanisms

**Three confirmed Re-conditioning mechanisms in baseline**: SOAP + ReScaleHead + p_weight + ReFiLM. Programme is now exploring:
1. **Budget-aware training**: EMA β=0.99 (frieren, rebasing), SWA plateau with η_min=1e-4 (edward, rebasing), OneCycleLR peak 2e-3 (alphonse), SGDR warm restarts T_0=14 (askeladd NEW)
2. **Architecture depth/capacity**: n_layers=6 (fern), slice_num=128 (nezuko)
3. **Loss reformulation**: Per-channel Huber δ (thorfinn: velocity δ=0.5, pressure δ=0.1)
4. **Augmentation**: Translation augmentation (tanjiro: rigid mesh shift, NSE-invariant NEW)

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2095 | askeladd | `sgdr-warm-restarts` | NEW | **HIGH** | CosineAnnealingWarmRestarts T_0=14, T_mult=1; 2 cosine cycles in 28 epochs |
| #2092 | tanjiro | `coord-translation-aug` | NEW | **HIGH** | Rigid mesh translation NSE-invariant aug; distinct from per-node jitter |
| #2081 | thorfinn | `per-channel-huber-delta` | WIP | **HIGH** | velocity δ=0.5, pressure δ=0.1 in Huber loss |
| #2079 | fern | `n-layers-6` | WIP | **HIGH** | Deeper Transolver stack n_layers 5→6 (+20% depth) |
| #2032 | edward | `plateau-swa` | WIP REBASE | **HIGH** | SWA over 1e-4 plateau; fixes zero-spread failure; needs rebase onto 28.8762 |
| #1966 | frieren | `ema-beta-0p99-rampup` | WIP REBASE | **HIGH** | EMA β=0.99; needs rebase onto 28.8762 |
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3); higher peak LR than warmup-only tests |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; baseline update sent |
| #1457 | askeladd | `surf-weight-50` | CLOSED/REPLACED | — | Replaced by sgdr-warm-restarts |

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
- **soap-linear-warmup** (#2077): +1.38% val; no instability to fix + wastes budget epochs

---

## Potential Next Directions

**After current in-flight experiments land**:
- **SGDR cycle length sweep**: If T_0=14 T_mult=1 works, test T_0=10 T_mult=2 (shorter first cycle, longer second)
- **Higher peak LR**: OneCycleLR (#1884) tests peak 2e-3; if that works, try peak 3e-3 with SGDR
- **ReFiLM per-block (not shared)**: Current ReFiLM is shared across all 5 blocks; per-block FiLM adds 4×more params but allows layer-specific Re gating
- **Geometry-specific augmentation**: camber/chord perturbation to directly improve geom_camber splits
- **Mixup on input features**: blend 2 samples for OOD generalization
- **Auxiliary losses**: predict vorticity/divergence as auxiliary targets
- **Multi-seed baseline validation**: 3 runs to firm up noise floor now that baseline is tighter
- **Karras post-hoc EMA**: average across multiple β values offline (zero training cost) — needs EMA mechanism first

**The model is still converging at ep 28-29 in every run.** Convergence-aware mechanisms (EMA, SWA, SGDR, OneCycleLR) remain the priority.
