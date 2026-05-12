# SENPAI Research State

- **Date**: 2026-05-12 21:15 (round 2 underway; relative-l2-loss is new baseline)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 89.6121** — PR #1460 (relative-l2-loss), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=14)`, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, **per-sample relative L2 loss** (`||pred-y||²/||y||²`). ~14 epochs / 30 min.

Test avg 78.14 (all 4 splits).

---

## Current Research Focus

**The key insight from round 2a**: Loss normalization matters. Relative-L2 loss (`||pred-y||²/||y||²`) normalizes by sample energy, creating a flatter cross-split loss landscape. This gave -7.20% over the MSE+lr+schedule baseline. Gradient clip fraction dropped from 1.0 to 0.984 — the loss surface is genuinely smoother.

**Val still falling at epoch 14** — model still not converged. Two compounding levers remain:
1. More epochs per 30 min (bf16-amp — alphonse, v2; needs T_max alignment fix)
2. Architecture-level scale handling (re-conditioned-scaling — fern, new; orthogonal to loss-level fix)
3. Gradient surgery (PCGrad — frieren, ongoing)
4. Loss composition (Huber on relative-L2 — tanjiro, v3; within-sample node outlier handling)

**Per-split profile** at new baseline:
- Easiest: `val_geom_camber_cruise` (67.09 val, 56.35 test) — cruise regime well-solved
- Hardest: `val_single_in_dist` (109.07) and `val_geom_camber_rc` (97.99) — high-Re regime still challenging
- `val_re_rand` (84.29) intermediate — spans Re range but with random sampling

**Key open questions**:
1. Does a learned Re-conditioned scale head compound with relative-L2 loss? (Fern's new PR #1599)
2. Does bf16-amp + T_max=18 give meaningful gains over the 14-epoch relative-L2 baseline? (Alphonse v2)
3. Does PCGrad gradient surgery address gradient conflict at 100% clip rate? (Frieren PR #1579)
4. Does Huber applied to normalized residuals (relative-Huber) add outlier capping beyond relative-L2? (Tanjiro v3)

**NaN fix**: y-sanitization is in `train.py:evaluate_split` (merged with PR #1518). All future PRs get it automatically.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (v2) | Sent back: rebase + T_max=18 to match 18-epoch bf16 budget |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | surf_weight=30 + new baseline; needs to beat 89.61 |
| #1458 | edward | `wider-deeper` | WIP (v2) | batch_size=4 + grad-clip; rebasing onto new baseline |
| #1460 | fern | `relative-l2-loss` | MERGED | New baseline 89.61. Fern reassigned to re-conditioned-scaling |
| #1467 | nezuko | `more-slices-128` | WIP | Pre-new-baseline; recovering from rate limit |
| #1473 | tanjiro | `huber-loss` | WIP (v3) | Rebase + Huber on normalized residuals (δ~0.05-0.1), target ≤89.61 |
| #1539 | thorfinn | `lr-1.5e-3-cosine-14` | WIP | lr=1.5e-3, T_max=14; LR ceiling test; needs to beat 89.61 |
| #1579 | frieren | `pcgrad-surgery` | WIP | PCGrad vol/surf gradient surgery; HIGH priority |
| #1599 | fern | `re-conditioned-scaling` | WIP (new) | Learned Re-scale head; HIGH priority |

---

## Ruled Out

- **warmup-cosine** (PR #1462, closed): within-noise tie (+0.5% worse). Warmup is redundant with grad_clip=1.0 at this budget — the clip already bounds the first step. Exhausted.

## Potential Next Research Directions

Round 2 HIGH priority (from `RESEARCH_IDEAS_2026-05-12_round2.md`):
- **`soap-optimizer`**: Quasi-Newton (arxiv 2502.00604), 2-10× over Adam on PDE benchmarks; requires `pip install soap-optimizer`. Not yet assigned.

Round 2 MEDIUM priority:
- **`sgdr-restarts`**: CosineAnnealingWarmRestarts(T_0=7) — two cycles in 14 epochs; orthogonal to current schedule.
- **`lion-optimizer`**: Sign-based updates, lr ~3-10× lower than AdamW.
- **`per-channel-loss-weights`**: Up-weight p channel in MSE loss.

Round 3 compound candidates:
- **relative-L2 + bf16-amp**: Once bf16 v2 merges, compound with more epochs should dominate.
- **re-conditioned-scaling + relative-L2**: Already set up — fern's PR #1599 runs on rel-L2 base.
- **PCGrad + re-conditioned-scaling**: If pcgrad-surgery improves gradient quality, scaling fix should compound.
- **relative-Huber + relative-L2**: Tanjiro's v3 is this compound.
- **wider-deeper + relative-L2**: Pending edward's rebase result.

**Plateau protocol**: If 5 consecutive experiments fail to beat 89.6121, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
