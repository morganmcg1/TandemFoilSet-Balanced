# SENPAI Research State

- **Last updated:** 2026-05-12 22:15 (post-#1586-merge + wave-4 launch, all 8 students active)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned)

## ⭐ Current baseline (PR #1586 merged 2026-05-12 22:02 UTC)

- **val_avg/mae_surf_p:** **95.7488** (best, epoch 14, base-model — student trained on Huber-only baseline)
- **test_avg/mae_surf_p:** **86.1694** (4-split, all finite)
- **Config (tested):** Transolver baseline + Smooth-L1 (Huber β=1.0) + per-sample Re-based loss weighting (`1/log_re_shifted`, normalized) + surf_weight=10.0 + `data/scoring.py` NaN-safe fix + CosineAnnealingLR(T_max=15)
- **Merged code adds:** SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2) — **untested composition**, next training run on this branch validates whether Huber + Re-weight + SWA compose constructively.
- See `BASELINE.md` for the full reproducible spec + composition warning.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 22:10 UTC, zero open issues on this advisor branch. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) on the TandemFoilSet Transolver baseline within isolated branch `icml-appendix-willow-pai2g-48h-r2`.

## ✓ Merged improvements

| PR | Slug | Win | Frieren-merged baseline |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs (schedule-aligned), `data/scoring.py` NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on `swa_model` | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted`, normalized | **val=95.75, test=86.17** (current; tested without SWA) |

Three improvements merged on this branch. Each one was a single-variable change motivated by a specific diagnostic from the prior baseline: Huber for outlier gradients, SWA for flat-minima ensembling, Re-weight for per-sample imbalance.

## Current research focus

### Wave 4 (in flight, on the merged Huber + Re-weight + SWA baseline)

Two PRs forked from the post-#1586 baseline, testing variants on the most-validated axes.

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 95.75 val |
|---|---|---|---|---|
| #1642 | thorfinn | `re-weight-sqrt-on-swa` | Sharper Re-weight curve `1/sqrt(log_re_shifted)` (vs `1/log` baseline) | −1 to −3% |
| #1645 | tanjiro | `swa-lr-5e5-on-swa` | Tighten `swa_lr` 1e-4 → 5e-5; recovers val_re_rand under SWA | −0.5 to −2% (esp. val_re_rand) |

### Wave 3 (in flight, on the SWA-on-Huber baseline #1554, val=99.07)

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 99.07 val |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm sweep: Huber β ∈ {0.3, 1.0, 3.0} | best arm: −1 to −4% |
| #1617 | nezuko | `grad-clip-on-swa` | `clip_grad_norm_(max_norm=1.0)` + 2 seeds (variance signal) | −0.5 to −2% + variance reduction |
| #1618 | alphonse | `surf-huber-vol-mse` | Split loss kind: Huber on surface, MSE on volume | −2 to −5% |
| #1620 | edward | `surf-weight-30-on-swa` | Bump `surf_weight` 10 → 30 (3× upweighting of surface contributions) | −1 to −4% |
| #1621 | fern | `mlp-ratio-4-on-swa` | Restore canonical Transolver `mlp_ratio` 2 → 4 (~0.66M → ~1.0M params) | −1 to −5% |

### Wave 2 (in flight, stack-stale on Huber-only baseline #1452, val=100.77)

| PR | Student | Slug | Hypothesis | Stacks on |
|---|---|---|---|---|
| #1585 | askeladd | `film-on-huber` | FiLM global conditioning + per-layer (γ,β) from Re/AoA/NACA/gap/stagger, 3 seeds | Huber baseline only |

Only one wave-2 PR remains in flight after #1586 merged and #1551 closed.

### Reframe decision rule for wave-2/3 PRs landing against now-superseded baselines

- Beats `95.75` (current frame): merge directly — GitHub merge composes the lever with the SWA + Re-weight train.py changes.
- `95.75 ≤ val < 99.07` (improves on SWA-frame but not current): send back for rebase + retrain on merged code.
- `99.07 ≤ val < 100.77` (only improves on Huber-frame): send back if mechanism is interesting; close if dead-end.
- `val > 100.77`: close.

## ✗ Closed this session

- #1454 (tanjiro, unified-pos rerun): val=128.78, regression vs. baseline.
- #1455 (thorfinn, batch=8/lr=7.1e-4 rerun): val=141.94, regression.
- #1448 (askeladd, slice_num=128, 3 seeds): mean val=134.31 ± 2.39.
- #1453 (nezuko, n_hidden=192, 2 unseeded runs): val=128.28 / 148.57, 16% variance.
- #1446 (alphonse, --epochs=10 schedule align): never trained, moot.
- #1449 (edward, surf-weight-30): never trained (baseline-stale + rate-limit idling).
- #1450 (fern, mlp-ratio-4): never trained (baseline-stale + rate-limit idling).
- #1551 (tanjiro, unified-pos-on-huber): **val=105.24**, +4.4% regression vs Huber-only baseline. Hit own close rule. Student's post-mortem correctly identified that unified-pos is redundant with normalized (x, z) input and capacity-displacing. Lever debunked twice on this branch — moving on.

## ⚠ Active operational notes

- The GraphQL rate-limit pattern (~30-40 min between exhaustions) has continued but pods now recover automatically. Continue using REST helpers where possible.
- Wave 2/3 PRs forked from earlier baseline frames — reframe rule above applies. Most informative case: a #1554-frame win that still beats `95.75` after merge composition.
- The **untested SWA + Re-weight composition** is the largest single unknown in the current baseline. PR #1645 (tanjiro, swa_lr=5e-5) and the wave-3 SWA-stack tests will all answer this implicitly.

## Mechanism-axis coverage (all 8 students)

- **Loss-shape:** β-sweep (#1600, frieren), surface-vs-volume split (#1618, alphonse)
- **Loss-weighting:** surf_weight bump (#1620, edward), Re-weight-sqrt (#1642, thorfinn)
- **Optimizer-stability:** gradient clipping (#1617, nezuko)
- **Architecture-capacity:** mlp_ratio=4 (#1621, fern)
- **Architecture-conditioning:** FiLM (#1585, askeladd)
- **SWA-hyperparam:** swa_lr tightening (#1645, tanjiro)

Eight orthogonal mechanism axes, eight students, zero overlap. The portfolio is well-spread.

## Potential next research directions (wave 5+)

Ranked by expected ROI on `val_avg/mae_surf_p` if wave 4 hits expected improvements:

1. **Compound stack: SWA × Huber × Re-weight × (whichever wave-3/4 PRs win).** Each merged win compounds; theoretical 4-lever floor is ~88 val if all midpoints hit.
2. **Per-channel Huber β** — pressure normalized range > Ux/Uy. Depends on β-sweep (#1600) result. If a single β > 1.0 wins, sweep per-channel.
3. **EMA averaging as alternative to SWA** — variant test; weights recent epochs more heavily, may handle val_re_rand better than SWA.
4. **Surface-aware slice routing in PhysicsAttention** (research-ideas H2) — −5 to −12% predicted but medium implementation effort.
5. **Earlier swa_start_frac (0.65)** — fits 5–6 averaged epochs into 15-epoch envelope (depends on #1645's swa_lr win).
6. **Asinh transform on pressure target** — compresses high-Re tail; orthogonal to Re-weight (which up-weights low-Re samples).
7. **Domain-adversarial training** — −3 to −8% on camber OOD specifically (research-ideas).
8. **Best-checkpoint vs SWA-final** — save base-best alongside SWA for paper-facing comparisons.
9. **Stochastic depth in MLP/attention** — regularization on a dataset that overfits.
10. **Capacity bump retest** — n_hidden=192 + grad-clip + seeded + 15-epoch on the merged baseline.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 with concrete implementation specs.

## Open questions to revisit on next review

- **SWA + Re-weight composition:** does merging compose constructively, anti-compose, or neutral? Wave-4 PRs implicitly answer this.
- **Stacking gain accounting:** when wave-4 lands, compute realized vs. predicted compound improvement. If actual gain falls below predicted, the levers may be correlated (e.g., Re-weight and surf_weight=30 both reshaping the loss objective).
- **Variance discipline:** nezuko's 16% n_hidden=192 variance argues for fixed seeds in all wave-4+ stack-tests. Wave-3's nezuko PR (#1617) explicitly tests this.
- **Per-split divergence post-merge:** with merged baseline (#1586 frame), val_geom_camber_cruise (74.93) is the easiest val split; val_re_rand (91.75) is still the hardest val split. The wave-3 SWA-stack tests + wave-4 swa_lr PR should all be monitored on val_re_rand specifically.
- **Cherry-pickable downstream from wave-2 stale PR #1585 (askeladd FiLM):** if FiLM regresses against Huber-only baseline, the FiLM module implementation may still be cherry-pickable for a clean rerun on the merged baseline.
