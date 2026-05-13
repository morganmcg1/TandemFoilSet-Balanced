# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 05:30 (round-16: closed #1789 stale tanjiro surf_weight, reassigned #1883 n_head=8. No new results to process. All 8 students active.)
- **round-15:**  MERGED #1619 sampler 2× single on L1 — **6th winner, new baseline 56.6217**. Closed #1826 cosine eta_min (LR floor backfired on L1). Assigned #1870 nezuko sampler both-racecar + #1871 thorfinn surf-p-weight-2x.)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None on this branch; scoped to launch constraints (no W&B, SENPAI_TIMEOUT_MINUTES=30 hard cap).

## Current research focus

**6 merged winners → NEW baseline 56.6217 (sampler+L1 stack, -49% from 110.76 at round-1 start).**

- All PRs must beat `val_avg/mae_surf_p < 56.6217`.
- Model still improving at terminal epoch in every recent run — systematically budget-bound. More gradient steps = more gain; but the 30-min cap is hard.
- **Primary axes still live:**
  1. **Sampler tuning** (new open question: boost both RaceCar vs boost single only)
  2. **Surface pressure gradient weighting** (direct metric optimization, orthogonal to loss shape and sampler)
  3. **Optimizer/preconditioner** (#1845 edward AdamW betas; #1774 alphonse lr retest)
  4. **Regularization** (#1775 fern WD, #1653 askeladd grad-clip — both L1 rebases in flight)
  5. **Architecture** (#1846 frieren slice_num=32)

## Merged winners

| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|---|
| #1619 ✓ | nezuko | Sampler 2× racecar_single on L1 | **56.62** | **50.43** |
| #1700 ✓ | thorfinn | Pure L1 loss | 59.54 | 51.47 |
| #1633 ✓ | thorfinn | Huber β=0.5 | 64.07 | 55.50 |
| #1568 ✓ | thorfinn | torch.compile + bf16 | 69.83 | 61.87 |
| #1532 ✓ | thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 |
| #1444 ✓ | thorfinn | MSE → Huber β=1.0 | 110.76 | NaN |

**Current baseline: val_avg/mae_surf_p = 56.6217, test_avg/mae_surf_p = 50.4310 (PR #1619)**

> Advisor config: Pure L1 + bf16 AMP + torch.compile(dynamic=True) + scoring-NaN workaround + sampler 2× racecar_single.
> ~39 epochs in 30 min at baseline. Peak GPU: ~24 GB. 96 GB available.
> Best epoch = terminal in all recent runs — model is systematically undertrained.

## In-flight (WIP)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1653 | askeladd | Grad clip max_norm=1.0 — L1 rebase | Rebasing; β=0.5 result was −6.94% vs 64.07 |
| #1775 | fern | WD=5e-5 — L1 rebase | Rebasing; β=0.5 result was −4.43% vs 64.07 |
| #1883 | tanjiro | n_head 4 → 8 (last architecture axis) | **New round-16 assignment** (replaced stale #1789) |
| #1774 | alphonse | lr 5e-4 → 7.5e-4 — L1 rebase | Rebasing to L1; result vs new 56.62 baseline TBD |
| #1845 | edward | AdamW betas=(0.9, 0.95) | New L1 assignment (pre-sampler merge baseline) |
| #1846 | frieren | slice_num 64 → 32 | New L1 assignment (pre-sampler merge baseline) |
| #1870 | nezuko | Sampler both-racecar 2× (single=2, tandem=2, cruise=1) | **New round-15 assignment** |
| #1871 | thorfinn | surf_loss p-channel weight [1,1,2] | **New round-15 assignment** |

**Note on stale baselines:** PRs #1653, #1775, #1774, #1845, #1846 were assigned on L1 baseline (59.54) but the current baseline is now 56.62 (sampler merged). If these come in between 56.62 and 59.54, I'll need to send them back for sampler rebase. If they beat 56.62, merge.

## Portfolio assessment

**High-confidence remaining levers:**
- Grad-clip (#1653) + WD=5e-5 (#1775) — proven on β=0.5, rebasing to L1. Should land ~55-57 on L1. May need sampler rebase too.
- surf_loss p-weight (#1871 thorfinn) — direct metric alignment, 2-line change. Predicted ~54-56.
- Sampler both-racecar (#1870 nezuko) — should recover geom_camber_rc while keeping single win. Predicted ~54-56.

**Medium-confidence:**
- AdamW betas=(0.9, 0.95) (#1845 edward) — L1 sign gradients benefit from shorter preconditioner memory. Predicted ~57-59 on pure L1 base, or ~54-57 with sampler.
- slice_num=32 (#1846 frieren) — CFD spatial argument; small budget gain. Predicted ~54-56.
- lr=7.5e-4 (#1774 alphonse) — noise-bound on β=0.5; L1 may respond differently.

## Closed axes (round 1-15)

- **Capacity (any form):** width, depth, mlp_ratio — all lose under 30-min cap.
- **Attention dropout:** per-weight, 30-min budget makes convergence the bottleneck.
- **LR floor (cosine eta_min):** counterproductive on L1 — removes the only step-damping on sign gradients.
- **Batch=8:** step-count starvation (-54% updates).
- **WD up (5e-4):** under-regularized on 36 epochs.
- **Warmup:** substituted by β sharpening; closed.
- **LR=1e-3 + warmup:** overshots at high peak.
- **Higher surf_weight (10→30):** biases away from volume.
- **Per-channel loss global reweight [1,1,3]:** distorts velocity in vol_loss.

## Open questions / next experiments if more slots open

1. **Sampler factor sweep** (1.5×, 3× on single only) — once both-racecar result is in.
2. **WD=1e-5 or 1e-6** — fern's original suggestion; continue decreasing if 5e-5 wins.
3. **n_head 4 → 8** — last untested architecture axis; dim_head halves per head.
4. **AoA reflection augmentation** — flip AoA sign, Uy, surface_area_fraction simultaneously.
5. **Block-level DropPath** — frieren's own suggestion; architectural regularization.
6. **Lookahead optimizer wrapper** — slow/fast weight averaging; helps with high-variance L1 gradients.
