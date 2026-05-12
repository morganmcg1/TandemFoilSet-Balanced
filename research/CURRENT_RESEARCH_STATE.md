# SENPAI Research State

- **Date**: 2026-05-12 22:45 (SOAP paradigm shift; new baseline 42.4015)
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

**SOAP paradigm shift**: The optimizer was the dominant bottleneck. SOAP's Kronecker-factored quasi-Newton preconditioner gives 4.2× grad norm reduction — each step is far better conditioned than AdamW. This is now the default optimizer on the advisor branch.

**Key constraint remaining**:
- Val still falling at epoch 13 — model NOT converged
- clip_frac=0.984 at ep 13 — SOAP is still being clipped ~9× per step (grad_norm=9.16 vs clip=1.0)
- Only 13 epochs in 30 min (vs 14 for AdamW) — SOAP is slightly slower per epoch

**Three highest-priority open questions**:
1. Does relaxing grad_clip from 1.0 to 5.0 unlock SOAP's step magnitude? (thorfinn #1668)
2. Does bf16-amp give 15-17 epochs with SOAP, compounding the two biggest wins? (alphonse #1456)
3. Does re-conditioned-scaling (architecture-level scale head) compound with SOAP? (fern #1599)

**Per-split profile at new baseline**:
- cruise (val 24.32 / test 19.79) — dramatically improved, near-saturating
- re_rand (val 43.22 / test 35.97) — much improved
- single_in_dist (val 46.09 / test 41.76) — still harder but major gains
- rc (val 55.98 / test 48.10) — hardest split, room to improve

---

## Active Experiments

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (v2) | **HIGHEST** | bf16 + SOAP = expected major compound; needs rebase |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | MEDIUM | surf_weight=30 on SOAP base; needs rebase |
| #1467 | nezuko | `more-slices-128` | WIP | MEDIUM | slice_num=128 on SOAP base; needs rebase |
| #1579 | frieren | `pcgrad-surgery` | WIP | LOW | PCGrad may be redundant with SOAP preconditioning |
| #1599 | fern | `re-conditioned-scaling` | WIP | HIGH | Re-scale head; orthogonal to optimizer |
| #1614 | edward | `per-channel-loss-weights` | WIP | MEDIUM | p_weight=5 on SOAP base; orthogonal |
| #1630 | tanjiro | `sgdr-restarts` | WIP | MEDIUM | SGDR T_0=7; needs rebase to get SOAP |
| #1668 | thorfinn | `soap-relax-clip` | WIP (new) | **HIGH** | grad_clip 1.0→5.0; unlocks SOAP step magnitude |

All 8 student pods healthy. SOAP update comment posted to all 7 WIP PRs.

---

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3 (AdamW)** (PR #1539): above AdamW LR ceiling; SOAP may change this
- **wider-deeper-3M** (PR #1458): epoch-limited

## Potential Next Directions

Now that SOAP has set a dramatically new baseline, the open research questions are:

**Highest priority (not yet assigned)**:
- **bf16-amp + SOAP**: alphonse is running this. Expected to be massive — 15-17 epochs vs 13, on a model that hasn't converged.
- **soap-relax-clip**: thorfinn running. clip_frac=0.984 means SOAP is clipping 9×. Relaxing to 5.0 should unlock larger steps.

**Architecture-level** (could compound with SOAP):
- **re-conditioned-scaling** (fern running): Re-scale head orthogonal to optimizer
- **FNO spectral layer**: Not yet tried; may outperform attention on turbulent flows
- **GNOT multi-query attention**: Not yet tried; designed for CFD

**Schedule refinements** (lower priority now that SOAP dominates):
- **T_max=13** (match SOAP's actual epoch count): Small alignment fix
- **Higher lr (2e-3 or 3e-3) under SOAP**: LR ceiling may have shifted significantly

**PCGrad status**: With SOAP preconditioning gradients, PCGrad's conflict-resolution is likely partially redundant. Will evaluate when frieren's results arrive.

**Plateau protocol**: Baseline moved from 89.39 to 42.40 — NOT a plateau. Next 5 experiments should be evaluated against 42.4015.
