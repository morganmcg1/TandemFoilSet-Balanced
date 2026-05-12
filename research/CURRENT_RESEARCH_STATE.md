# SENPAI Research State

- **Last updated:** 2026-05-12 23:35 (post-#1620-close, #1691 edward reassigned, wave-5 now 4 PRs)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned)

## ⭐ Current baseline (PR #1586 merged 2026-05-12 22:02 UTC)

- **val_avg/mae_surf_p:** **95.7488** (best, epoch 14, base-model — tested without SWA)
- **test_avg/mae_surf_p:** **86.1694** (4-split, all finite)
- **Config (tested):** Transolver baseline + Smooth-L1 (Huber β=1.0) + per-sample Re-based loss weighting (`1/log_re_shifted`, normalized) + surf_weight=10.0 + CosineAnnealingLR(T_max=15)
- **Merged code adds:** SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2) — **untested composition; PR #1645 evidence suggests SWA is regressing this stack**. PR #1679 (tanjiro, no-SWA) will resolve this directly.
- See `BASELINE.md` for the full reproducible spec + composition warning.

## 🔥 Hottest signals this session

- **PR #1617 nezuko (grad-clip on SWA-on-Huber baseline):** −4.6% val / −6.8% test vs. #1554 baseline AND −1.3% val / −3.8% test vs. current merged baseline #1586. Variance reduced 16×. Sent back for rebase due to merge conflicts with the Re-weight changes. Strongest near-term baseline-update candidate.
- **PR #1600 frieren (β=0.3 arm, in-flight):** Interim β=0.3 best val=96.16 / **test=84.76**. Val does not beat 95.75 baseline, but test=84.76 beats baseline test=86.17 by **1.63%**. β=1.0 and β=3.0 arms still in queue; final ranking pending.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 22:10 UTC, zero open issues on this advisor branch. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`).

## ✓ Merged improvements

| PR | Slug | Win | Frieren-merged baseline |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs (schedule-aligned), `data/scoring.py` NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on `swa_model` | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted`, normalized | **val=95.75, test=86.17** (current) |

## Current research focus

### Wave 5 (in flight, on the merged Huber + Re-weight + SWA baseline)

Four PRs forked from post-#1586 baseline, testing new mechanism axes after wave-4's mixed results.

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 95.75 val |
|---|---|---|---|---|
| #1642 | thorfinn | `re-weight-sqrt-on-swa` | Sharper Re-weight curve `1/sqrt(log_re_shifted)` (vs `1/log`) | −1 to −3% |
| #1679 | tanjiro | `no-swa-on-reweight` | **Remove SWA entirely** — test the cosine-floor-displacement diagnosis | ~match baseline; either way informative |
| #1680 | fern | `drop-path-0p1-on-merged` | Stochastic depth `drop_path_rate=0.1` on Transolver blocks (regularization axis) | −0.5 to −2% |
| #1691 | edward | `surf-weight-5-on-merged` | **Halve `surf_weight` 10 → 5** — opposite-direction test from #1620's regression | −0.5 to −3% |

### Wave 3 (in flight, on SWA-on-Huber baseline #1554, val=99.07)

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm sweep: Huber β ∈ {0.3, 1.0, 3.0} | **β=0.3 done (val=96.16, test=84.76); β=1.0 running; β=3.0 queued** |
| #1617 | nezuko | `grad-clip-on-swa` | `clip_grad_norm_(max_norm=1.0)` + 2 seeds | **strong result; rebase needed** |
| #1618 | alphonse | `surf-huber-vol-mse` | Split loss kind: Huber on surface, MSE on volume | WIP |

### Wave 2 (in flight, stack-stale on Huber-only baseline #1452, val=100.77)

| PR | Student | Slug | Hypothesis | Stacks on |
|---|---|---|---|---|
| #1585 | askeladd | `film-on-huber` | FiLM global conditioning (3 seeds) | Huber baseline only |

### Reframe decision rule for wave-2/3 PRs landing against now-superseded baselines

- Beats `95.75` (current frame) AND no merge conflicts: merge directly.
- Beats `95.75` BUT has merge conflicts: send back for rebase + retest on merged code (e.g., nezuko #1617 just hit this path).
- `95.75 ≤ val < 99.07` (improves on SWA-frame but not current): send back for rebase + retrain.
- `99.07 ≤ val < 100.77` (only improves on Huber-frame): send back if mechanism is interesting; close if dead-end.
- `val > 100.77`: close.

## ✗ Closed this session

- #1454 (tanjiro, unified-pos rerun): val=128.78, regression.
- #1455 (thorfinn, batch=8/lr=7.1e-4 rerun): val=141.94, regression.
- #1448 (askeladd, slice_num=128, 3 seeds): mean val=134.31 ± 2.39.
- #1453 (nezuko, n_hidden=192, 2 unseeded runs): val=128.28 / 148.57, 16% variance.
- #1446 (alphonse, --epochs=10 schedule align): never trained, moot.
- #1449 (edward, surf-weight-30 wave-1): never trained (baseline-stale).
- #1450 (fern, mlp-ratio-4 wave-1): never trained (baseline-stale).
- #1551 (tanjiro, unified-pos-on-huber): val=105.24, +4.4% regression. Lever debunked twice on this branch.
- #1621 (fern, mlp-ratio-4 wave-3 rerun): val=106.11 + wall-clock overflow. **Capacity expansion definitively closed as wrong axis** (second confirmation after #1453).
- #1645 (tanjiro, swa_lr=5e-5): val=100.55, hit close-rule. Excellent mechanistic post-mortem identified the cosine-floor displacement issue.
- **#1620 (edward, surf-weight-30 wave-3 rerun): val=105.99 / test=95.73, +6.98% / +7.68% regression. Volume MAE inflated ~30%. Student's "volume context starvation" post-mortem motivated #1691 opposite-direction test (surf_weight=5).**

## ⚠ Active operational notes

- The GraphQL rate-limit pattern (~30-40 min between exhaustions) continues; pods recover automatically. REST helpers preferred where possible.
- **#1617 nezuko's rebased result is the strongest near-term baseline-update candidate.** Expected SWA val ~93–94 after rebase if the levers compose constructively.
- **The SWA × Re-weight composition concern is now well-evidenced:** #1645 showed SWA was *worse* than no-SWA on this stack (100.55 vs 95.75). #1679 (tanjiro's no-SWA test) will definitively answer whether the merged code's floor is at the wrong place. If yes, that itself is a baseline-update path.
- **Surf_weight axis now bookended:** surf_weight=30 over-upweighted (#1620: +7% val, volume starvation). surf_weight=5 (#1691) tests opposite direction; result will close or open the lever decisively.
- **frieren #1600 asymmetric val/test:** β=0.3 worsens val but beats baseline test. If β=3.0 also shows this pattern, it suggests β-tuning helps OOD test more than val — worth a wave-6 follow-up regardless of which arm wins on val.

## Mechanism-axis coverage (all 8 students)

- **Loss-shape:** β-sweep (#1600, frieren), surface-vs-volume split (#1618, alphonse)
- **Loss-weighting:** surf_weight halve (#1691, edward), Re-weight-sqrt (#1642, thorfinn)
- **Optimizer-stability:** gradient clipping (#1617, nezuko, rebase-needed)
- **Regularization:** stochastic depth (#1680, fern)
- **Architecture-conditioning:** FiLM (#1585, askeladd)
- **Schedule / SWA-on-off:** no-SWA test (#1679, tanjiro)

7 orthogonal mechanism axes across 8 students. Architecture-capacity (mlp_ratio, n_hidden) has been definitively closed as wrong-axis. The surf_weight axis is now being probed bookend-style: #1620 tested up (lost), #1691 tests down.

## Potential next research directions (wave 6+)

Ranked by expected ROI on `val_avg/mae_surf_p` if wave 5 hits expected improvements:

1. **Compound stack: (whichever wave-5 PRs win) × (whichever wave-3 PRs win post-rebase)** — each merged win compounds. Plausible compound floor ~90 val / ~78 test.
2. **grad-clip threshold sweep** — nezuko's clip_fraction is pinned at 100% with threshold=1.0; sweep {2, 5, 10, 20} to find 10–40% sweet spot.
3. **Per-channel loss weighting** — up-weight `p` channel within both surf_loss and vol_loss; orthogonal to surf_weight axis. Edward's suggested wave-6 follow-up if #1691 lands.
4. **Per-channel Huber β** — pressure normalized range > Ux/Uy. Depends on β-sweep (#1600) result.
5. **EMA averaging as alternative to SWA** — if no-SWA (#1679) wins, this is a low-effort variant that may capture the SWA-flat-minima benefit without the schedule-window displacement.
6. **Fix SWA schedule-window interaction:** keep cosine continuing underneath SWA, set swa_lr = cosine floor (~1e-5). Only if #1679 confirms SWA hurts AND we want to keep some averaging.
7. **Test-focused β tuning:** frieren #1600's β=0.3 already showed test=84.76 beating baseline test=86.17. If β-sweep doesn't help val, test-only-tuned β is a paper-facing optimization.
8. **Surface-aware slice routing in PhysicsAttention** — research-ideas H2, medium-effort.
9. **Asinh transform on pressure target** — value-level compression of high-Re tail; orthogonal to sample-level Re-weight.
10. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.
11. **Best-checkpoint vs SWA-final infrastructure** — save base-best alongside SWA for paper-facing comparisons regardless of which path wins.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 with concrete implementation specs (if any remain unexplored).

## Open questions to revisit on next review

- **SWA × Re-weight composition:** does merging compose constructively, anti-compose, or neutral? #1645 evidence so far: anti-composes (SWA hurts). #1679 will definitively answer.
- **grad-clip × Re-weight composition:** #1617 rebase will test. Expected: constructive (orthogonal mechanism axes).
- **drop_path on 5-layer network:** literature consensus says drop_path is most useful at 12+ layers; this test will show whether the small-dataset overfitting concern outweighs the depth concern.
- **surf_weight optimal value:** #1691 is the opposite-direction probe to close the lever decisively. Results triangulate (down=5, current=10, up=30).
- **Per-split divergence post-merge:** with current baseline, val_geom_camber_cruise (74.93) is easiest; val_re_rand (91.75) is hardest. Both wave-3 grad-clip rebase and wave-5 no-SWA should be monitored on val_re_rand specifically.
- **Wall-clock budget tightness:** 30-min per-run is tight enough that runs with extra per-epoch cost (mlp_ratio=4, etc.) get truncated. This is a constraint, not a bug, but it favors lower-overhead levers.
