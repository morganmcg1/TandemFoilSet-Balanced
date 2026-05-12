# SENPAI Research State

- **Date**: 2026-05-12 21:30 (round 2 underway; relative-l2-loss is new baseline)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 89.6121** — PR #1460 (fern/relative-l2-loss), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, **per-sample relative L2 loss** (`||pred-y||²/||y||²`). ~14 epochs / 30 min.

Test avg 78.14 (all 4 splits).

---

## Current Research Focus

**The key insight from round 2a**: Loss normalization matters. Relative-L2 loss normalizes by sample energy, creating a flatter cross-split loss landscape. Gave -7.20% vs MSE baseline. Gradient clip fraction dropped from 1.0 to 0.984 — the loss surface is genuinely smoother.

**LR ceiling confirmed**: lr=1.5e-3 is above the ceiling (PR #1539, val 100.24). lr=1e-3 is optimal for AdamW at grad_clip=1.0. To push further, need a better optimizer — hence SOAP (H1).

**Architecture scaling ruled out**: wider-deeper (PR #1458) gets only 6-7 epochs in 30 min with 3M params. Not competitive with the 662K baseline that gets 14 epochs.

**Val still falling at epoch 14** — core constraint. Four parallel levers now active:
1. More epochs (bf16-amp — alphonse v2; T_max=18 to match extended budget)
2. Better optimizer (SOAP — thorfinn; quasi-Newton replaces AdamW)
3. Architecture-level scale handling (re-conditioned-scaling — fern; orthogonal to loss-level fix)
4. Per-channel loss focus (per-channel-loss-weights — edward; up-weight pressure channel)
5. Loss composition (Huber on relative-L2 — tanjiro v3; intra-sample node outliers)
6. Gradient surgery (PCGrad — frieren; orthogonal to loss and LR)

**Per-split profile** at new baseline:
- Easiest: `val_geom_camber_cruise` (67.09 val, 56.35 test) — well-solved
- Hardest: `val_single_in_dist` (109.07) and `val_geom_camber_rc` (97.99) — high-Re, OOD
- `val_re_rand` (84.29) intermediate

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (v2) | Rebase + T_max=18; 30% more epochs, needs schedule alignment |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | surf_weight=30 + new baseline; pod alive, awaiting results |
| #1467 | nezuko | `more-slices-128` | WIP | Pre-new-baseline; needs to beat 89.61 |
| #1473 | tanjiro | `huber-loss` | WIP (v3) | Relative-Huber: Huber(δ≈0.05) on normalized residuals |
| #1579 | frieren | `pcgrad-surgery` | WIP | PCGrad vol/surf gradient surgery |
| #1599 | fern | `re-conditioned-scaling` | WIP (new) | Learned Re-scale head; HIGH priority |
| #1613 | thorfinn | `soap-optimizer` | WIP (new) | SOAP quasi-Newton optimizer; HIGH priority |
| #1614 | edward | `per-channel-loss-weights` | WIP (new) | p_weight=5 in relative-L2 loss |

All 8 student pods healthy (1/1 each).

---

## Ruled Out

- **warmup-cosine** (PR #1462, closed): within-noise tie (+0.5% worse). Warmup is redundant with grad_clip=1.0 at this budget.
- **lr=1.5e-3** (PR #1539, closed): confirmed above LR ceiling. 100% clip rate with lr=1.5e-3. AdamW ceiling = lr=1e-3.
- **wider-deeper-3M** (PR #1458, closed): only 6-7 epochs in 30 min, not competitive. Would need bf16-amp to be viable.

## Potential Next Research Directions

Round 2 remaining HIGH priority:
- (All HIGH priority items now assigned: pcgrad-surgery #1579, re-conditioned-scaling #1599, soap-optimizer #1613)

Round 2 MEDIUM priority (unassigned):
- **`sgdr-restarts`**: CosineAnnealingWarmRestarts(T_0=7) — two cycles in 14 epochs.
- **`lion-optimizer`**: Sign-based updates, lr ~3-10× lower than AdamW. Somewhat redundant with SOAP.
- **`attention-temperature-anneal`**: Soft→sharp physics-slice annealing.

Round 3 compound candidates:
- **relative-L2 + bf16-amp + T_max=18**: 18 epochs with aligned schedule
- **re-conditioned-scaling + relative-L2**: Already set up — fern's PR #1599 runs on rel-L2 base
- **per-channel-loss-weights + relative-L2**: Edward's #1614 runs on rel-L2 base
- **PCGrad + re-conditioned-scaling**: If pcgrad improves gradient quality, scaling should compound
- **SOAP + re-conditioned-scaling**: SOAP + learned scale head — orthogonal mechanisms

**Plateau protocol**: If 5 consecutive experiments fail to beat 89.6121, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
