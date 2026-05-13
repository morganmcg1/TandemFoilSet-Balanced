# SENPAI Research State

- **Date**: 2026-05-13 00:10 (PCGrad closed, EMA assigned to frieren)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 42.4015** — PR #1613 (thorfinn/soap-optimizer), merged 2026-05-12.

**LARGEST SINGLE IMPROVEMENT: -52.6% vs previous 89.3940.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4, precondition_frequency=10, max_precond_dim=256`), `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+relative-L2 loss. ~13 epochs / 30 min.

Test avg 36.40 (all 4 splits).

---

## Current Research Focus

**SOAP paradigm shift validated. Now compounding on top of SOAP across orthogonal axes.**

Key properties of the SOAP baseline to exploit:
- Val still falling at epoch 13 — model NOT converged → bf16-amp and EMA directly exploit this
- clip_frac=0.984 → SOAP clipped ~9×/step (grad_norm=9.16 vs clip=1.0) → soap-relax-clip targets this
- Only 13 epochs in 30 min → bf16-amp should give ~17 epochs
- Gradient conflict is sparse (4% of tensors per PCGrad) — SOAP + loss normalization already tames magnitude noise

**Three highest-priority running experiments**:
1. **soap-relax-clip** (thorfinn #1668): grad_clip 1.0→5.0 to unlock SOAP step magnitude
2. **bf16-amp** (alphonse #1456 rebasing): +29% throughput confirmed; ~17 epochs on SOAP
3. **ema-weights** (frieren #1704, new): EMA of SOAP weights = free smoothed checkpoint, zero wall-clock cost

**Per-split profile at new baseline**:
- cruise (val 24.32 / test 19.79) — dramatically improved, near-saturating
- re_rand (val 43.22 / test 35.97) — much improved
- single_in_dist (val 46.09 / test 41.76) — still harder but major gains
- rc (val 55.98 / test 48.10) — hardest split, most room to improve

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (rebasing) | **HIGHEST** | +29% throughput → ~17 epochs on SOAP; compound |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (rebasing) | HIGH | ReScaleHead (scale corr +0.92 with log_Re); SOAP compound test |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5 on SOAP base; orthogonal |
| #1630 | tanjiro | `cosine-eta-min` | WIP (rebasing, pivoted) | MEDIUM | Monotone cosine + eta_min=1e-5 floor on SOAP |
| #1668 | thorfinn | `soap-relax-clip` | WIP | **HIGH** | grad_clip 1.0→5.0; unlocks SOAP step magnitude |
| #1704 | frieren | `ema-weights` | WIP (new) | **HIGH** | EMA β=0.999 of SOAP weights; zero wall-clock cost |

All 8 student pods healthy. All on SOAP-base or rebasing onto SOAP.

---

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling; SOAP may change this
- **wider-deeper-3M** (PR #1458): epoch-limited
- **SGDR T_0=7** (PR #1630, pivoted): restart cost ~4 epochs of re-convergence at 14-epoch budget — replaced with monotone cosine + eta_min floor
- **PCGrad gradient surgery** (PR #1579, closed): mechanism confirmed (9× grad-norm reduction) but 1.63× wall-clock penalty can't be earned back at 30-min budget; gradient conflict is sparse (4% of tensors) and already mostly tamed by SOAP + loss normalization

---

## Potential Next Directions

**Currently in-flight on SOAP base** (see active experiments above).

**Architecture-level** (not yet assigned):
- **FNO spectral layer**: Not yet tried; may outperform attention on turbulent flows
- **GNOT multi-query attention**: Not yet tried; designed for CFD
- **Larger model under SOAP**: SOAP may unlock capacity that was epoch-limited under AdamW

**Optimizer/schedule**:
- **Higher lr (2e-3) under SOAP**: LR ceiling may have shifted; next priority after soap-relax-clip results
- **SOAP precondition_frequency=5** (more frequent updates): trades compute for better conditioning

**Loss**:
- **Per-split loss weighting by OOD difficulty** (rc hardest, cruise easiest): explicit curriculum
- **Charbonnier loss instead of Huber**: smoother near zero, differentiable everywhere

**PCGrad status**: Confirmed mechanism but structural wall-clock loss. No further follow-up unless budget increases.

**Plateau protocol**: Baseline moved from 89.39 to 42.40 — NOT a plateau. All 8 slots active.
