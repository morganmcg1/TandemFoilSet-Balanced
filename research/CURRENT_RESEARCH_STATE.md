# SENPAI Research State

- **Date**: 2026-05-13 06:20 (PR #1614 merged; #1952 rebasing; #1985 assigned to edward)
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
From thorfinn's #1933 SWA analysis: **single-seed run-to-run variance is ~1-2%** on val_avg/mae_surf_p.
- Deltas below 1.5% are within noise → need multi-seed validation
- Both recent wins (#1599 at -1.95%, #1614 at -2.11%) are above the noise floor with consistent per-split signatures

### Convergence/Budget-Limited Diagnosis
Model is convergence/budget-limited at 30-min floor. Loss monotonically descends to cutoff in every run. Best epoch = last epoch in all recent experiments.

**Implication**: mechanisms that enable more/better gradient steps (OneCycleLR, better loss targeting) are higher value than regularization or model shrinkage.

### Gradient Mass Targeting (p_channel_weight finding)
p still dominates error by ~70× after 5× weighting. With p_weight=5, gradient mass ratio is 7× (p vs each velocity) vs 70× actual error ratio. There is room to increase p_weight further.

---

## Current Research Focus — Convergence + Pressure Targeting Programme

**Two orthogonal winning mechanisms confirmed**:
1. **Re-conditioning (ReScaleHead)**: Separates shape learning from Re-scale calibration. -1.95% (#1599)
2. **Pressure channel upweighting (p_weight=5)**: Directs gradient mass to dominant error channel. -2.11% (#1614)

**What's been ruled out** (this round):
- Stochastic depth, attention dropout (regularization-limited refuted)
- EMA β=0.999 with 30-ep budget (window too long)
- SWA-at-cosine-floor (no weight-space spread at LR=1e-5)
- surf_weight=7 (surface gradients load-bearing; rc went WRONG direction)

**What's in flight**:
- **OneCycleLR** (#1884 alphonse): warmup + peak 2e-3 → cosine; needs rebase onto 29.2179 baseline
- **ReScaleHead 2-channel** (#1952 fern): rebasing onto 29.2179 to verify Ux-drop compounds with p_weight=5
- **p_channel_weight=15** (#1985 edward): sweep p_weight upward — 5 was first anchor; 15 still below 70× error ratio
- **surf_weight=50** (#1457 askeladd): opposite-direction probe (rebasing)
- **slice_num=128** (#1467 nezuko): capacity expansion (rebasing)
- **coord-jitter-aug** (#1963 tanjiro): ±std=0.005 input spatial coord noise
- **surf-weight-15** (#1964 thorfinn): test if MORE surf weight helps rc
- **ema-beta-0.99-rampup** (#1966 frieren): β=0.99 with Karras warmup

All 8 students active.

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3); baseline update sent (new target: 29.2179) |
| #1952 | fern | `rescale-head-2ch` | WIP | **HIGH** | Rebasing onto 29.2179; Ux-drop showed -1.55% vs old baseline |
| #1985 | edward | `p-channel-weight-15` | NEW | **HIGH** | Sweep p_weight 5→15; error ratio gap still ~70× |
| #1963 | tanjiro | `coord-jitter-aug` | WIP | **HIGH** | std=0.005 spatial coord noise |
| #1964 | thorfinn | `surf-weight-15` | WIP | **HIGH** | surf_weight=10→15 probe |
| #1966 | frieren | `ema-beta-0p99-rampup` | WIP | **HIGH** | EMA β=0.99 + Karras rampup |
| #1457 | askeladd | `surf-weight-50` | WIP | MEDIUM | surf_weight=50; rebasing onto 29.2179 |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; rebasing onto 29.2179 |

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
- **stochastic-depth** (#1897): refutes regularization-limited +8.48%
- **attention-dropout** (#1900): smoking-gun "still descending at ep 29" +0.47%
- **surf-weight-7** (#1936): surface gradients load-bearing, rc went WRONG direction +2.94%
- **swa-last-k** (#1933): SWA-at-floor flatlines because weights don't oscillate at LR=1e-5
- **ema-weights v1** (#1704): dual-val overhead +5.9%
- **ema-weights v2** (#1917): β=0.999 too high for 30-ep budget +2.9%

---

## Potential Next Directions

**After current in-flight experiments land**:
- **p_channel_weight sweep continuation**: if p_weight=15 wins, try p_weight=25 (approaching 70× error ratio); if it regresses, the optimum is between 5 and 15
- **Single-channel (p-only) ReScaleHead**: fern suggested; Uy std=0.266 real but modest vs p std=0.518. Test whether p-only head further reduces noise
- **FiLM-style Re conditioning**: inject log(Re) into PhysicsAttention slice weighting — more aggressive than output rescaling; attack OOD-rc directly (still at 42.69 val)
- **rc-specific intervention**: rc dominates error; rc-only loss weight or targeted augmentation — or per-split loss weighting (rc_weight separate from overall surf_weight)
- **Higher-LR plateau SWA (correct version)**: cosine to lr=1e-4 for 25 ep + constant for last 5 + SWA over those 5 (requires alphonse's OneCycleLR result first)
- **Multi-seed baseline validation**: run baseline 3× to establish noise floor (currently ~1-2%)
- **Karras post-hoc EMA**: average across multiple β values offline (zero training cost)
- **Mixup on input features**: blend 2 samples for OOD generalization on rc
- **Auxiliary losses**: predict vorticity, divergence as auxiliary targets (may help rc generalization)

**The model is still converging at ep 29-30 in every recent run.** Convergence-aware experiments (OneCycleLR, higher-LR-plateau SWA) plus OOD-rc targeted approaches are the priority.
