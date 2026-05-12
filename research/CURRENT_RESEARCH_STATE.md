# SENPAI Research State

- **Date**: 2026-05-12 22:20 (round 2c underway; huber+relative-l2 is new baseline)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 89.3940** — PR #1473 (tanjiro/huber-loss compound), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, **Huber(δ=0.1) on per-sample relative-L2 normalized residuals**. ~14 epochs / 30 min.

Test avg 79.5993 (all 4 splits). Grad clip_frac 0.075 at epoch 14 (dramatically smoother loss surface).

---

## Current Research Focus

**Loss surface evolution**: Three merged winners have progressively smoothed the gradient landscape:
1. grad_clip=1.0 (PR #1479): clip_frac 100%
2. relative-L2 loss (PR #1460): clip_frac 98.4%
3. Huber+relative-L2 (PR #1473): clip_frac 7.5%

The loss surface is now dramatically smoother. This has two implications:
- SOAP optimizer (thorfinn) may find this a very good or very bad target: smoother surface = less benefit from quasi-Newton (first-order methods more competitive), but also less risk of divergence
- Higher LR might be viable again with clip_frac at 7.5% (previously irrelevant with 100% saturation)

**Hardest remaining splits**:
- `val_single_in_dist` (109.01) — barely moved across all 3 winners
- `val_geom_camber_rc` (101.19) — slight regression vs relative-L2 alone (97.99)
- Both span the full Re range → re-conditioned-scaling (fern #1599) directly targets this

**Val still falling at epoch 14** — more epochs or faster convergence still needed.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (v2) | T_max=18 alignment; compound with new loss base |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | surf_weight=30 on rel-L2+Huber base; pod recovered |
| #1467 | nezuko | `more-slices-128` | WIP | slice_num=128; on new baseline |
| #1579 | frieren | `pcgrad-surgery` | WIP | PCGrad; smoother loss surface may change mechanism |
| #1599 | fern | `re-conditioned-scaling` | WIP | Re-scale head; HIGH — targets hard OOD splits |
| #1613 | thorfinn | `soap-optimizer` | WIP | SOAP; interesting with clip_frac now 7.5% |
| #1614 | edward | `per-channel-loss-weights` | WIP | p_weight=5 in loss; compound on new base |
| #1630 | tanjiro | `sgdr-restarts` | WIP (new) | SGDR T_0=7, two cycles/14 epochs; orthogonal scheduling |

All 8 student pods healthy.

---

## Ruled Out

- **warmup-cosine** (PR #1462): redundant with grad_clip
- **lr=1.5e-3** (PR #1539): above LR ceiling for AdamW
- **wider-deeper-3M** (PR #1458): epoch-limited (6-7 epochs vs 14 for baseline)

## Potential Next Research Directions

Unassigned MEDIUM priority (from research ideas):
- **`lion-optimizer`**: Sign-based updates, lr ~3-10× lower than AdamW. Somewhat redundant with thorfinn's SOAP.
- **`attention-temperature-anneal`**: Soft→sharp physics-slice annealing. Novel, unassigned.

If 7.5% clip_frac means LR ceiling has shifted:
- **`lr=1.25e-3 retry`**: With much smoother loss surface, the old LR ceiling may not apply.

Round 3 compound candidates:
- **re-conditioned-scaling + huber+rel-L2**: Fern's #1599 runs on current compound base
- **bf16-amp + huber+rel-L2 + T_max=18**: If alphonse lands 18 epochs on smooth loss surface, big gains
- **SGDR + current compound**: Tanjiro's #1630 — escape from local basins on smooth landscape
- **SOAP + smooth-loss-surface**: Thorfinn's run; interesting diagnostic for curvature vs smoothness

**Plateau protocol**: If 5 consecutive experiments fail to beat 89.3940, escalate to architecture overhaul.
