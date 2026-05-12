# SENPAI Research State

- **Last updated:** 2026-05-12 21:55 (post-wave-2-merge + wave-3 portfolio complete)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned)

## ⭐ Current baseline (PR #1554 merged 2026-05-12 21:06 UTC)

- **val_avg/mae_surf_p:** **99.0704** (SWA model, end of training)
- **test_avg/mae_surf_p:** **88.8955** (4-split, all finite)
- **Config:** Transolver baseline + Smooth-L1 (Huber β=1.0) + `data/scoring.py` NaN-safe fix + CosineAnnealingLR(T_max=15) + **SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2, terminal eval on swa_model.module)**
- **W&B run:** `cnu8v9i2`
- See `BASELINE.md` for the full reproducible spec.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-12 21:30 UTC, zero open issues. Workflow assumed: drive primary ranking metric `val_avg/mae_surf_p` (and `test_avg/mae_surf_p`) on the TandemFoilSet Transolver baseline within isolated branch `icml-appendix-willow-pai2g-48h-r2`.

## ✓ Merged improvements

| PR | Slug | Win | Frieren-merged baseline |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs (schedule-aligned), `data/scoring.py` NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on `swa_model` | **val=99.07, test=88.90** (current) |

frieren has merged 2/2 of their PRs on this branch — they own both the Huber and SWA implementations.

## Current research focus

**Wave 3 (in flight):** Five high-ROI levers stacked directly on the merged SWA-on-Huber baseline, spanning orthogonal mechanism axes (loss-shape, weighting, stability, capacity).

| PR | Student | Slug | Hypothesis | Predicted Δ vs. 99.07 val |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm sweep: Huber β ∈ {0.3, 1.0, 3.0} | best arm: −1 to −4% |
| #1617 | nezuko | `grad-clip-on-swa` | `clip_grad_norm_(max_norm=1.0)` + 2 seeds (variance signal) | −0.5 to −2% + variance reduction |
| #1618 | alphonse | `surf-huber-vol-mse` | Split loss kind: Huber on surface, MSE on volume | −2 to −5% |
| #1620 | edward | `surf-weight-30-on-swa` | Bump `surf_weight` 10 → 30 (3× upweighting of surface contributions) | −1 to −4% |
| #1621 | fern | `mlp-ratio-4-on-swa` | Restore canonical Transolver `mlp_ratio` 2 → 4 (~0.66M → ~1.0M params) | −1 to −5% |

**Wave 2 (in flight, stack-stale on Huber baseline):** Three levers based on the pre-SWA Huber baseline. They'll need to be re-evaluated when results land — a win on Huber-baseline doesn't directly compare to the SWA-baseline number.

| PR | Student | Slug | Hypothesis | Stacks on |
|---|---|---|---|---|
| #1551 | tanjiro | `unified-pos-on-huber` | unified_pos=True ref=8 (2D Transolver ref²=64 grid) | Huber baseline only |
| #1585 | askeladd | `film-on-huber` | FiLM global conditioning + per-layer (γ,β) from Re/AoA/NACA/gap/stagger, 3 seeds | Huber baseline only |
| #1586 | thorfinn | `re-weight-on-huber` | Per-sample loss reweighting by 1/(shifted log Re), normalized | Huber baseline only |

**Wave 1 (closed/reassigned this session):**

- **#1449 edward** and **#1450 fern** were closed as baseline-stale (never produced results; pods idled after rate-limit episodes while their branches went 2 merges out of date). Both reassigned as fresh wave-3 stack-tests forked from the current SWA-on-Huber baseline:
  - **#1620 edward** (`surf-weight-30-on-swa`) — same lever as #1449, fresh branch on new baseline
  - **#1621 fern** (`mlp-ratio-4-on-swa`) — same lever as #1450, fresh branch on new baseline

## ✗ Closed this session

- #1454 (tanjiro, unified-pos rerun): val=128.78, regression vs. new baseline — reassigned as wave-2 stack-test PR #1551.
- #1455 (thorfinn, batch=8/lr=7.1e-4 rerun): val=141.94, regression — reassigned as wave-2 PR #1586 (Re-weight).
- #1448 (askeladd, slice_num=128, 3 seeds): mean val=134.31 ± 2.39 — reassigned as wave-2 PR #1585 (FiLM).
- #1453 (nezuko, n_hidden=192, 2 unseeded runs): val=128.28 / 148.57, 16% variance — reassigned as wave-3 PR #1617 (gradient clipping; lever motivated directly by their variance observation).
- #1446 (alphonse, --epochs=10 schedule align): never trained, **moot** — schedule alignment is implicit in the merged baseline. Reassigned as wave-3 PR #1618 (split-loss-by-node-type).
- #1449 (edward, surf-weight-30): never trained (baseline-stale + rate-limit idling) — reassigned as wave-3 PR #1620 (`surf-weight-30-on-swa`, same lever, fresh branch).
- #1450 (fern, mlp-ratio-4): never trained (baseline-stale + rate-limit idling) — reassigned as wave-3 PR #1621 (`mlp-ratio-4-on-swa`, same lever, fresh branch).

## ⚠ Active operational note

- The GraphQL rate-limit pattern (~30-40 min between exhaustions) has continued but pods now recover automatically. Continue using REST helpers where possible.
- Three wave-2 PRs (#1551, #1585, #1586) are stack-stale: they fork from the Huber-only baseline, not the merged SWA-on-Huber baseline. Their training results will need to be reframed against the appropriate baseline. **Decision rule:** if their result beats val=99.07 on the Huber baseline they tested on, it's a clear stack win (merge directly — GitHub merge will compose the lever with the SWA train.py changes that are now in advisor). If they only beat the old Huber baseline (val=100.77) but not 99.07, send back for rebase + retrain. If they regress vs. 100.77, close.

## Potential next research directions (wave 4+)

Ranked by expected ROI on `val_avg/mae_surf_p` if wave 3 hits expected improvements:

1. **Compound stack: SWA × Huber × (whichever wave-3 PR wins) × (whichever wave-2 PR wins after rebase).** Each merged win compounds; theoretical 4-lever floor is ~85 val if all midpoints hit.
2. **Per-channel Huber β** — pressure normalized range > Ux/Uy. Depends on β-sweep (#1600) result. If a single β > 1.0 wins, sweep per-channel.
3. **EMA averaging instead of SWA** — variant test; cheap to swap.
4. **Surface-aware slice routing in PhysicsAttention** (research-ideas H2) — −5 to −12% predicted but medium implementation effort. Wave-4 if a simple stack-test doesn't land.
5. **Lower swa_lr (0.05× base lr)** — addresses the val_re_rand regression observed in PR #1554's SWA eval.
6. **Earlier swa_start_frac (0.65)** — fits 5 averaged epochs into 14-epoch envelope.
7. **Capacity bump retest** — n_hidden=192 + grad-clip + seeded + 15-epoch on SWA-on-Huber baseline.
8. **Domain-adversarial training** — −3 to −8% on camber OOD specifically (research-ideas).
9. **Best-checkpoint vs SWA-final** — save base-best alongside SWA for paper-facing comparisons.
10. **Re-rebase wave-2 stack-stales** — give the FiLM/unified-pos/Re-weight PRs a clean re-run on SWA-on-Huber.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` doc has H1–H10 with concrete implementation specs.

## Open questions to revisit on review

- **Stacking gain accounting:** when wave-3 lands, compute realized vs. predicted compound improvement. If actual gain falls below predicted, the levers may be correlated (e.g., gradient clipping and SWA both stabilizing the same noise source).
- **Variance discipline:** nezuko's 16% n_hidden=192 variance argues for ALL future stack-tests to either use a fixed seed (preferably) or report 2-seed results. Wave-3's nezuko PR (#1617) explicitly tests this.
- **Per-split divergence post-SWA-merge:** with SWA-on-Huber baseline, val_geom_camber_cruise (79.18) is the easiest val split; val_re_rand (95.12) is the hardest val split AND the one that slightly regressed vs. Huber-only (93.04 → 95.12). The val_re_rand regression is the most diagnostic per-split signal in the merged baseline.
- **PR #1554's val_re_rand regression** (+2.2%) is most likely from only 3 SWA-active epochs (not 4) + `swa_lr` above cosine floor. Tighter SWA tuning is on the wave-4 list.
- **Cherry-pickable downstream from wave-2 stale PRs:** if any of #1551, #1585, #1586 implements a fix that's not in the merged baseline (e.g., FiLM module, unified-pos constructor fix), and they regress, the implementation might still be cherry-pickable for a clean rerun.
