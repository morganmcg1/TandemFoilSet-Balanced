# SENPAI Research State

- **Date**: 2026-05-12 20:05 (round 2 underway; higher-lr-cosine-14 is new baseline)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 96.5587** — PR #1518 (higher-lr-cosine-14), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=1e-3, wd=1e-4)`, **`CosineAnnealingLR(T_max=14)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, MSE loss. NaN scoring fix in `train.py:evaluate_split`. ~14 epochs / 30 min.

Test avg 85.87 (all 4 splits, NaN-free for first time).

---

## Current Research Focus

**The key insight from round 2**: Schedule alignment matters enormously. By setting T_max=14 to match the real epoch budget, the cosine schedule actually reaches its low-LR fine-tuning phase. Combined with lr=1e-3 (safe under grad_clip=1.0), this gave -17.6% over the previous best. The old T_max=50 was wasting 64% of the schedule.

**Val still falling at epoch 14** — the model has not converged within the 30-min cap. This is the key constraint on further progress. Options:
1. More epochs per 30 min (bf16-amp — alphonse)
2. Faster convergence (higher lr — thorfinn; Huber loss — tanjiro; warmup — frieren)
3. Architecture that converges faster (wider-deeper — edward)

**Key open questions**:
1. Does Huber loss compound on top of the new LR/schedule baseline? (Tanjiro's rebase will answer this — now needs to beat 96.56 instead of 117.17)
2. Does lr=1.5e-3 push past 96.56 or find the LR ceiling? (Thorfinn's new PR #1539)
3. Does bf16-amp enable 18-20 epochs in 30 min, and what would that yield with the new LR?
4. Can architecture scaling (wider-deeper) converge faster to a better solution?

**Per-split profile** at new baseline:
- Easiest: `val_geom_camber_cruise` (74.35 val, 61.86 test) — big improvement, cruise regime
- Hardest: `val_geom_camber_rc` (110.59) and `val_single_in_dist` (108.58) — raceCar regime still challenging

**NaN fix**: y-sanitization is now in `train.py:evaluate_split` (merged with PR #1518). All future PRs get it automatically via rebase.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP | Just picked up assignment (rate limit cleared); running now |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | surf_weight=30 + grad-clip baseline |
| #1458 | edward | `wider-deeper` | WIP (v2) | batch_size=4 + grad-clip; just picked up, rebasing |
| #1460 | fern | `relative-l2-loss` | WIP | Pre-new-baseline; needs to beat 96.56 |
| #1467 | nezuko | `more-slices-128` | WIP | Pre-new-baseline; recovering from rate limit |
| #1473 | tanjiro | `huber-loss` | WIP (v2, pending rebase) | Rebase onto new LR baseline, target ≤96.56 |
| #1539 | thorfinn | `lr-1.5e-3-cosine-14` | WIP (new) | lr=1.5e-3, T_max=14; LR ceiling test |
| #1579 | frieren | `pcgrad-surgery` | WIP (new) | PCGrad vol/surf gradient surgery; HIGH priority |

---

## Ruled Out

- **warmup-cosine** (PR #1462, closed): within-noise tie (+0.5% worse). Warmup is redundant with grad_clip=1.0 at this budget — the clip already bounds the first step. Exhausted. Would only help if we drop grad_clip or push LR well above 1e-3.

## Potential Next Research Directions

Round 2 HIGH priority (from `RESEARCH_IDEAS_2026-05-12_round2.md`):
- **`soap-optimizer`**: Quasi-Newton (arxiv 2502.00604), 2-10× over Adam on PDE benchmarks; requires `pip install soap-optimizer`. Not yet assigned.
- **`re-conditioned-scaling`**: Re-conditioned output scale head reading `log_Re` from input dim 13; targets output scale heterogeneity across Re=100K–5M. Not yet assigned.

Round 2 MEDIUM priority:
- **`sgdr-restarts`**: CosineAnnealingWarmRestarts(T_0=7) — two cycles in 14 epochs; orthogonal to current schedule.
- **`lion-optimizer`**: Sign-based updates, lr ~3-10× lower than AdamW.
- **`per-channel-loss-weights`**: Up-weight p channel in MSE loss.

Round 3 compound candidates:
- **Huber + bf16-amp**: Once both merge independently, compound should dominate.
- **PCGrad + re-conditioned-scaling**: If pcgrad-surgery improves gradient quality, scaling fix should compound.
- **wider-deeper + new LR**: Pending edward's rebase result.

**Plateau protocol**: If 5 consecutive experiments fail to beat 96.5587, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
