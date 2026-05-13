# SENPAI Research State

- **Date**: 2026-05-13 05:55 (surf-weight-7, swa-last-k, ema-v2 all closed; noise floor calibrated at ~1-2%; new directions assigned)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 29.8463** — PR #1599 (fern/re-conditioned-scaling), merged 2026-05-13.

**-74.5% cumulative from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head), **SOAP** (`lr=1e-3, wd=1e-4`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 29 epochs / 30 min. Peak GPU 24/96 GB.

Per-split val: single_in_dist=30.20, rc=43.11, cruise=14.54, re_rand=31.54. Test avg 26.1005.

---

## Critical Programme Finding — Noise Floor Calibrated

From thorfinn's #1933 SWA analysis: **single-seed run-to-run variance is ~1-2%** on val_avg/mae_surf_p. This is a key calibration:
- Deltas below 1.5% are within noise → need multi-seed validation
- PR #1599 won by 1.95% (right at the edge) but had a CONSISTENT per-split signature across 3 runs and a confirmed mechanism (Re-correlation 0.86-0.94)
- All recent regressions of +2-3% are real but small

**Implication**: future "borderline wins" must include a mechanism signature OR be confirmed by multiple seeds.

---

## Current Research Focus — Convergence/Budget-Limited Programme

**Diagnosis** (5 experiments confirming): Model is convergence/budget-limited at 30-min floor. Loss monotonically descends to cutoff in every run.

**What's been ruled out** (this round):
- Stochastic depth, attention dropout (regularization-limited refuted)
- EMA β=0.999 with 30-ep budget (window too long — averages worse old weights)
- SWA-at-cosine-floor (no weight-space spread at LR=1e-5)
- surf_weight=7 (surface gradients load-bearing; LESS surf weight worse, especially on rc)

**What's still in flight**:
- **OneCycleLR** (#1884 alphonse): warmup + peak 2e-3 → cosine
- **ReScaleHead 2-channel** (#1952 fern): drop unused Ux channel
- **surf_weight=30** (#1457 askeladd): opposite-direction probe (rebasing)
- **slice_num=128** (#1467 nezuko): capacity expansion (rebasing)
- **per-channel loss weights** (#1614 edward): p_weight=5 (rebasing)

**Newly assigned** (this round):
- **coord-jitter-aug** (tanjiro): ±std=0.005 input spatial coord noise — input-domain regularization
- **surf-weight-15** (thorfinn): opposite direction of failed surf-weight-7 — test if MORE weight helps rc
- **ema-beta-0.99-rampup** (frieren): β=0.99 with Karras warmup, fixes the v2 window-too-long failure

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1457 | askeladd | `surf-weight-50` | WIP | MEDIUM | surf_weight=30; rebasing onto 29.8463 |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128; rebasing onto 29.8463 |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5; rebasing onto 29.8463 |
| #1884 | alphonse | `onecycle-lr` | WIP | **HIGH** | OneCycleLR(max_lr=2e-3); convergence pivot |
| #1952 | fern | `rescale-head-2ch` | WIP | **HIGH** | Drop Ux channel from ReScaleHead |
| TBD | tanjiro | `coord-jitter-aug` | NEW | **HIGH** | std=0.005 spatial coord noise on inputs |
| TBD | thorfinn | `surf-weight-15` | NEW | **HIGH** | Opposite-direction surf_weight probe |
| TBD | frieren | `ema-beta-0.99-rampup` | NEW | **HIGH** | EMA β=0.99 + Karras rampup |

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

## Potential Next Directions

**After current in-flight experiments land**:
- **Multi-seed baseline validation**: run baseline 3× to establish noise floor (currently estimated ~1-2%)
- **rc-specific intervention**: rc is dominant error source (43.11 val); rc-only loss weight or targeted augmentation
- **Test-time augmentation**: predict on K perturbed inputs, average
- **Coord jitter compound** (if tanjiro wins): mirror-symmetry augmentation for symmetric foils
- **Higher-LR plateau SWA**: cosine to lr=1e-4 for 25 ep + constant for last 5 + SWA over those 5 (thorfinn's correct version)
- **Mixup on input features**: blend 2 samples for OOD generalization
- **Karras post-hoc EMA**: average across multiple β values offline (zero training cost)
- **FiLM-style Re conditioning**: inject log(Re) into PhysicsAttention slice weighting (more aggressive Re injection than output rescaling)
- **Curriculum learning**: train on easier in-dist samples first, ramp to OOD-augmented
- **Auxiliary losses**: predict vorticity, divergence as auxiliary targets

**The model is still converging at ep 29-30 in every recent run.** Convergence-aware experiments (OneCycleLR, higher-LR-plateau SWA, ema-beta-0.99) plus OOD-rc targeted approaches are the priority.
