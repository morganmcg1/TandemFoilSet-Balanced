# SENPAI Research State

- **Date**: 2026-05-13 07:10 (3 PRs sent back for rebase; fern assigned FiLM-re-attention; 2ch ReScaleHead closed)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 29.2179** — PR #1614 (edward/per-channel-loss-weights p=5), merged 2026-05-13.

**-75.1% cumulative from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head, out_channels=3) **+ p_channel_weight=5** (post-Huber linear weight on pressure channel), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 29 epochs / 30 min. Peak GPU 23.91/96 GB.

Per-split val: single_in_dist=28.56, rc=42.69, cruise=13.77, re_rand=31.85. Test avg 25.6024.

---

## Critical Programme Findings

### Noise Floor Calibrated
Single-seed run-to-run variance is **~1-2%** on val_avg/mae_surf_p. Deltas below 1.5% need multi-seed confirmation.

### Convergence/Budget-Limited
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (OneCycleLR, EMA, SWA with plateau) are high value.

### Re-Conditioning Stack
Two confirmed mechanisms: ReScaleHead (output rescaling, -1.95%) + p_channel_weight=5 (loss reweighting, -2.11%). Next step: FiLM inside attention (slice-selection conditioning).

### Ux Channel Is Load-Bearing Even Near-Identity
PR #1952 (drop Ux from ReScaleHead) REGRESSED +4.63% on rebased stack with p_weight=5. The Ux channel provides gradient stability for velocity channels when pressure is upweighted in the loss. 3-channel ReScaleHead stays in baseline.

### Gradient Mass Targeting
p dominates error by ~70× even after 5× weighting (gradient mass ratio now ~7×). There is headroom for further p_weight increase (edward running p_weight=15).

---

## Current Research Focus — Convergence + Multi-Mechanism Compounding

**Three confirmed mechanisms compound**: SOAP, ReScaleHead, p_weight=5. The programme is now exploring:
1. Whether additional loss-axis mechanisms (surf_weight, coord-jitter) compound with p_weight
2. Whether Re-conditioning can go deeper than output rescaling (FiLM in attention)
3. Whether budget-aware training (EMA, OneCycleLR) stacks on top
4. Whether loss-axis p_weight can go higher (p_weight=15)

**In-flight experiments awaiting rebase results** (ran on old 29.8463 baseline):
- #1963 tanjiro/coord-jitter-aug: -0.78% val / -1.51% test (borderline, rebase needed)
- #1964 thorfinn/surf-weight-15: -0.74% val / -1.72% test (borderline, rebase needed)
- #1966 frieren/ema-beta-0p99-rampup: -2.6% val / -3.8% test (strong, rebase needed)

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3); baseline update sent (29.2179) |
| #1966 | frieren | `ema-beta-0p99-rampup` | WIP REBASE | **HIGH** | EMA β=0.99; strong -2.6% on old baseline; needs rebase |
| #1963 | tanjiro | `coord-jitter-aug` | WIP REBASE | **HIGH** | Coord jitter std=0.005; needs rebase onto 29.2179 |
| #1964 | thorfinn | `surf-weight-15` | WIP REBASE | **HIGH** | surf=15 mild +; needs rebase onto 29.2179 |
| #1985 | edward | `p-channel-weight-15` | WIP | **HIGH** | p_weight=5→15 sweep |
| #2011 | fern | `film-re-attention` | NEW | **HIGH** | FiLM Re-cond inside PhysicsAttention slice logits |
| #1457 | askeladd | `surf-weight-50` | WIP | MEDIUM | surf_weight=50; baseline update sent (29.2179) |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; baseline update sent (29.2179) |

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

---

## Potential Next Directions

**After current in-flight experiments land**:
- **p_weight sweep continuation**: if p_weight=15 wins, try p_weight=25 (approaching 70× error ratio)
- **Higher-LR plateau SWA (correct version)**: depends on OneCycleLR result first
- **Multi-seed baseline validation**: 3 runs to firm up noise floor
- **Karras post-hoc EMA**: average across multiple β values offline (zero training cost)
- **rc-specific augmentation**: coord jitter may not be enough; try geometric augmentations specific to OOD-rc shapes
- **Mixup on input features**: blend 2 samples for OOD generalization
- **Auxiliary losses**: predict vorticity/divergence as auxiliary targets
- **Single-run FiLM baseline**: verify FiLM works before combining with deeper architecture

**The model is still converging at ep 29-30 in every run.** Convergence-aware + Re-conditioning + loss-targeted mechanisms remain the priority.
