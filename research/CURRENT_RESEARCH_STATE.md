# SENPAI Research State

- **Date**: 2026-05-12 23:00 (post-SOAP-merge, 3 race-condition send-backs)
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

**SOAP paradigm shift validated**. Now harvesting compound wins on top of SOAP. Three concurrent PRs (alphonse/bf16-amp, fern/re-conditioned-scaling, tanjiro/sgdr→cosine-eta-min) completed on stale pre-SOAP base — all rebased onto SOAP and re-running.

**Key constraint remaining at SOAP baseline**:
- Val still falling at epoch 13 — model NOT converged
- clip_frac=0.984 at ep 13 — SOAP is still being clipped ~9× per step (grad_norm=9.16 vs clip=1.0)
- Only 13 epochs in 30 min (vs 14 for AdamW) — SOAP is slightly slower per epoch
- LR ceiling may have shifted under SOAP's preconditioner — untested

**Three highest-priority open questions**:
1. Does relaxing grad_clip from 1.0 to 5.0 unlock SOAP's step magnitude? (thorfinn #1668)
2. Does bf16-amp give 17+ epochs with SOAP, compounding the two biggest wins? (alphonse #1456 rebased)
3. Does Re-conditioned-scaling compound with SOAP, or is it redundant? (fern #1599 rebased)

**Per-split profile at new baseline**:
- cruise (val 24.32 / test 19.79) — dramatically improved, near-saturating
- re_rand (val 43.22 / test 35.97) — much improved
- single_in_dist (val 46.09 / test 41.76) — still harder but major gains
- rc (val 55.98 / test 48.10) — hardest split, room to improve

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (rebasing) | **HIGHEST** | bf16 + SOAP = expected ~17 epochs; +29% throughput confirmed |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1579 | frieren | `pcgrad-surgery` | WIP | LOW | PCGrad may be redundant with SOAP preconditioning |
| #1599 | fern | `re-conditioned-scaling` | WIP (rebasing) | HIGH | Re-scale head working (scale corr +0.92); SOAP compound test |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5 on SOAP base; orthogonal |
| #1630 | tanjiro | `sgdr-restarts` → `cosine-eta-min` | WIP (rebasing, pivoted) | MEDIUM | Pivoted: monotone cosine + eta_min=1e-5 floor on SOAP |
| #1668 | thorfinn | `soap-relax-clip` | WIP | **HIGH** | grad_clip 1.0→5.0; unlocks SOAP step magnitude |

All 8 student pods healthy. All on SOAP-base or rebasing onto SOAP.

---

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling; SOAP may change this
- **wider-deeper-3M** (PR #1458): epoch-limited
- **SGDR T_0=7** (PR #1630, pivoted): restart cost ~4 epochs of re-convergence at 14-epoch budget — replaced with monotone cosine + eta_min floor

## Potential Next Directions

Now that SOAP has set a dramatically new baseline, the open research questions are:

**Currently in-flight on SOAP base**:
- **bf16-amp + SOAP**: alphonse #1456 rebased. Expected to be massive — ~17 epochs vs 13, on a model that hasn't converged.
- **soap-relax-clip**: thorfinn #1668. clip_frac=0.984 means SOAP is clipping 9×. Relaxing to 5.0 should unlock larger steps.
- **re-conditioned-scaling + SOAP**: fern #1599 rebased. Architecture-level head; compound test on top of optimizer.

**Architecture-level** (not yet assigned, could compound with SOAP):
- **FNO spectral layer**: Not yet tried; may outperform attention on turbulent flows
- **GNOT multi-query attention**: Not yet tried; designed for CFD
- **Larger model under SOAP**: SOAP's preconditioner may let us absorb a larger model in the epoch budget

**Schedule refinements**:
- **Higher lr (2e-3 or 3e-3) under SOAP**: LR ceiling may have shifted significantly with preconditioning — next priority once bf16-amp lands

**PCGrad status**: With SOAP preconditioning gradients, PCGrad's conflict-resolution is likely partially redundant. Will evaluate when frieren's results arrive.

**Plateau protocol**: Baseline moved from 89.39 to 42.40 — NOT a plateau. Next 5 experiments should be evaluated against 42.4015.
