# SENPAI Research State

- **Date**: 2026-05-13 00:30 (cosine-eta-min merged, soap-higher-lr assigned to tanjiro)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 39.8693** — PR #1630 (tanjiro/cosine-eta-min), merged 2026-05-13.

**-5.97% vs previous 42.4015 (SOAP baseline).**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), **SOAP** (`lr=1e-3, wd=1e-4, precondition_frequency=10, max_precond_dim=256`), **`CosineAnnealingLR(T_max=14, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+relative-L2 loss. ~13 epochs / 30 min.

Test avg 35.22 (all 4 splits).

**Convergence trace**: 167.84 → 134.09 → 107.90 → 97.98 → 84.20 → 81.79 → 76.84 → 62.82 → 52.34 → 50.44 → 45.42 → 42.63 → 39.87 (still falling at ep 13).

---

## Current Research Focus

**Compounding orthogonal wins on the SOAP + cosine-eta-min stack.** Model still converging at ep 13 — every improvement compounds.

**Key constraints remaining**:
- Val still falling at epoch 13 — model NOT converged → bf16-amp and higher LR target this
- clip_frac=0.984 → SOAP clipped ~9×/step → soap-relax-clip and higher LR both target this
- LR ceiling under SOAP unknown — previously 1e-3 under AdamW, now being probed at 2e-3
- Only 13 epochs in 30 min → bf16-amp should give ~17 epochs

**Three highest-priority running experiments**:
1. **soap-relax-clip** (thorfinn #1668): grad_clip 1.0→5.0; direct step-size unlock
2. **bf16-amp** (alphonse #1456 rebasing): ~17 epochs → compounds with everything
3. **soap-higher-lr** (tanjiro #1740, new): lr=2e-3; LR ceiling probe under SOAP

**Per-split profile at new baseline**:
- cruise (val 20.89 / test 17.24) — near-saturating
- re_rand (val 38.49 / test 31.37) — strong improvement
- single_in_dist (val 47.81 / test 45.95) — hardest to improve consistently
- rc (val 52.28 / test 46.33) — hardest OOD split, most room

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (rebasing) | **HIGHEST** | +29% throughput → ~17 epochs; compound |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1599 | fern | `re-conditioned-scaling` | WIP (rebasing) | HIGH | ReScaleHead; SOAP compound test; training now |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5 on SOAP base; orthogonal |
| #1668 | thorfinn | `soap-relax-clip` | WIP | **HIGH** | grad_clip 1.0→5.0; unlocks SOAP step magnitude |
| #1704 | frieren | `ema-weights` | WIP | **HIGH** | EMA β=0.999 of SOAP weights; zero wall-clock cost |
| #1740 | tanjiro | `soap-higher-lr` | WIP (new) | **HIGH** | lr=1e-3→2e-3; LR ceiling probe under SOAP |

All 8 students active, all running on SOAP + cosine-eta-min base (target = 39.8693).

---

## Merged Winners (chronological)

| PR | Student | Slug | val_avg | Delta |
|----|---------|------|---------|-------|
| #1479 | thorfinn | grad-clip-1 | 117.17 | — |
| #1518 | thorfinn | higher-lr-cosine-14 | 96.5587 | −17.6% |
| #1460 | fern | relative-l2-loss | 89.6121 | −7.2% |
| #1473 | tanjiro | huber-loss | 89.3940 | −0.24% |
| #1613 | thorfinn | soap-optimizer | 42.4015 | **−52.6%** |
| #1630 | tanjiro | cosine-eta-min | 39.8693 | −5.97% |

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling; SOAP may shift this
- **wider-deeper-3M** (PR #1458): epoch-limited
- **SGDR T_0=7** (PR #1630 original): restart cost ~4 epochs; pivoted to cosine-eta-min
- **PCGrad gradient surgery** (PR #1579): mechanism confirmed but 1.63× wall-clock loses at 30-min budget

## Potential Next Directions

**After current in-flight results land**:
- **Higher LR further** (3e-3?): if 2e-3 succeeds, test next step up
- **FNO spectral layer**: Not yet tried; may outperform attention on turbulent flows
- **Larger model under SOAP**: 1M-3M params; SOAP may unlock capacity
- **SOAP precondition_frequency=5**: more frequent updates, better conditioning
- **OneCycleLR**: warmup → peak → steep decay, popular for competitions
- **Per-split weighted loss**: rc is hardest (val 52.28) → give it higher loss weight explicitly
