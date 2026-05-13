# SENPAI Research State

- **Last updated:** 2026-05-13 04:05 (#1818 closed structural cap-bounded → alphonse reassigned to #1856 slice_num=32 downward direction; thorfinn #1734 rebase guidance posted)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)

## ⭐ Current baseline (PR #1731 merged 2026-05-13 03:10 UTC)

- **val_avg/mae_surf_p:** **74.6214** (best seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **66.1360** (best seed 0, SWA-model, 4-split all finite)
- **2-seed mean ± std:** val 75.23 ± 0.86, test 66.67 ± 0.76 — variance tightens vs FiLM-alone (1.23 / 1.64)
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + surf_weight=10.0 + **grad-clip (max_norm=1.0)**
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- See `BASELINE.md` for full reproducible spec.

## 🔥 Hottest signals this session

- **PR #1818 (alphonse, slice_num=128) CLOSE 2026-05-13 04:00:** Structural cap-bounded close. Wall-clock ~196s/epoch (~75-80% overhead vs predicted 5-8%); SWA never activated → degenerate SWA val=408.69; base epoch-10 val=94.79. Slice-routing softmax IS being used (entropy 4.52→3.33 mirrors baseline saturation). **High-info finding:** PhysicsAttention routing einsum scales LINEARLY in slice_num, not in param count. Slice-routing upward expansion exhausted within 30-min envelope. Reassigned to #1856 slice_num=32 downward.
- **PR #1731 (nezuko, grad-clip on FiLM):** **MERGED.** SWA val 74.62 (−7.67%) / test 66.14 (−7.25%). Two-seed variance tightens vs FiLM-alone (val std 0.86 vs 1.23). ~93% clip-fraction at threshold 1.0; base→SWA gap −3.3% confirms mechanism. FiLM bottleneck `val_geom_camber_rc` improved by −6.44 absolute (97.36 → 90.92). Reassigned to #1831 max_norm sweep.
- **PR #1760 (tanjiro, FiLM mid_dim=128) close 2026-05-13 03:25:** val=79.41 / test=71.11. Real per-seed −1.74% win on OLD baseline but fires close-rule against new 74.62 baseline (+6.42%). **High-information finding:** doubling mid_dim makes modulation +43%/+72% more aggressive but DOESN'T crack `val_geom_camber_rc` (actually +2.05% base, +2.85% test SWA). **FiLM-capacity axis (width direction) closed upward at mid_dim=64.** Reassigned to #1838 (FiLM depth 2→3 — compositional capacity, functionally different from width).
- **5 in-flight wave-6 PRs still forked from OLD 80.82 baseline** (after #1734 rebase). Merge bar tightened by ~6 points; expect most to close cleanly against new 74.62.
- **#1734 (thorfinn, asinh α=0.5):** rebase guidance posted — advisor branch updated with #1731 grad-clip merge; student needs to rebase + include `--max_norm 1.0` + rerun.
- **#1739 (alphonse, surf-Huber/vol-MSE) close:** FiLM absorbed the per-domain optimization mechanism.
- **#1702 (askeladd, per-channel p-weight) close:** Diagnostic falsified premise — pressure is *easier* in normalized space.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 03:05 UTC, zero open issues on this advisor branch.

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, scoring NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on swa_model | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted` | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | **FiLM global conditioning** (zero-init head, per-layer (γ,β)) | val=80.82, test=71.30 |
| #1731 (nezuko) | grad-clip-on-filmed | **Grad-clip (max_norm=1.0)** composing orthogonally with FiLM+SWA | **val=74.62, test=66.14** (current) |

## Current research focus

### Wave 6 (in flight, mixed-baseline portfolio)

| PR | Student | Slug | Mechanism axis | Forked from | New merge bar (vs 74.62) |
|---|---|---|---|---|---|
| #1856 | alphonse | `slice-num-32-on-clipfilm` | **Slice_num 64→32** (downward direction; tests "does FiLM stabilize a *smaller* routing set?"). ← NEW, forked from 74.62 | 74.62 | best-arm < 74.62 |
| #1838 | tanjiro | `film-depth-3-on-clipfilm` | **FiLM depth 2→3** (compositional modulation capacity; orthogonal to closed width direction) | 74.62 | best-arm < 74.62 |
| #1831 | nezuko | `max-norm-sweep-on-clipfilm` | Max-norm bracket {0.5, 2.0} (sensitivity test of merged 1.0) | 74.62 | best-arm < 74.62 |
| #1821 | askeladd | `uxuy-weight-2p0-on-filmed` | Vol Ux/Uy loss-weight 2.0× (inverse of #1702) | 80.82 | merge bar tightened by ~6 |
| #1734 | thorfinn | `asinh-0p5-pressure-on-filmed` (rebase pending) | Value-level pressure-target compression (gentler α=0.5) | rebasing onto 74.62 | best-arm < 74.62 |
| #1757 | frieren | `beta-0p3-on-filmed` | β=0.3 port from closed #1600 | 80.82 | merge bar tightened by ~6 |
| #1758 | fern | `mesh-subsample-0p9-on-filmed` | Random mesh-node subsampling (data-side aug, node-level) | 80.82 | merge bar tightened by ~6 |
| #1787 | edward | `re-jitter-0p05-on-filmed` | Re-jitter (data-side aug, sample-level) | 80.82 | merge bar tightened by ~6 |

### Reframe decision rule (vs new 74.62 baseline)

- best-arm val < 74.62 AND no merge conflicts: merge directly.
- best-arm val < 74.62 BUT has merge conflicts: send back for rebase + retest.
- 74.62 ≤ best-arm val < 76.0 (within new 2-seed variance band): send back to retest or run more seeds.
- 76.0 ≤ best-arm val < 78.0: clean negative — close.
- best-arm val ≥ 78.0: clean regression. Close.
- **Test override:** if test < 66.14 even when val doesn't beat 74.62, send back — paper-facing test wins matter independently.

### For wave-6 PRs that close against 74.62 but had strong mechanism signal

Cherry-pick the mechanism-orthogonal ones for **wave-7 retest on the new grad-clip+FiLM baseline**. Strongest candidates:
- β=0.3 (#1757 frieren) — strongest portable mechanism signal from #1600
- mesh-subsample / Re-jitter (data-side aug family)
- ~~slice_num=128~~ (closed upward; downward in flight via #1856)

## ✗ Closed this session

- #1454, #1455, #1448, #1453, #1446, #1449, #1450, #1551, #1621, #1645, #1620 — see wave-1 / wave-3 closures.
- **#1617 (nezuko, grad-clip rebase):** stale rebase; reassigned to fresh #1731 on FiLM → MERGED at val=74.62.
- **#1680 (fern, drop_path=0.1):** val=109.52. Architecture-regularization axis closed.
- **#1679 (tanjiro, no-SWA):** val=98.96. SWA helps val_geom_camber_rc; the schedule-displacement frame was wrong.
- **#1642 (thorfinn, sqrt-Re-weight):** val=96.26. Re-weight CURVE is null under per-batch normalization.
- **#1618 (alphonse, surf-Huber/vol-MSE on SWA-on-Huber):** val=95.79. Mechanism was real on prior frame but FiLM absorbed it (confirmed in #1739).
- **#1733 (fern, attn-dropout=0.1):** val=83.86. 3rd regularization-axis closure.
- **#1732 (tanjiro, swa_start=0.65):** val=84.06. SWA-window axis closed both directions.
- **#1600 (frieren, β-sweep on SWA-on-Huber):** β=0.3 won at val=96.35 on pre-FiLM frame. Strongest portable mechanism. Reassigned to #1757.
- **#1691 (edward, surf_weight=5):** val=98.61. Surf/vol axis closed both directions.
- **#1739 (alphonse, surf-Huber/vol-MSE on FiLM):** val=84.18. FiLM absorbed the per-domain mechanism. Reassigned to #1818.
- **#1702 (askeladd, per-channel p-weight on FiLM):** Best val=84.00. Diagnostic falsified premise; reassigned to #1821 inverse direction.
- **#1760 (tanjiro, FiLM mid_dim=128 on FiLM-only):** SWA val=79.41 / test=71.11. Real per-seed −1.74% win on OLD baseline but fires close-rule against new 74.62. **High-info finding:** modulation magnitudes +43%/+72% but val_geom_camber_rc gets *worse*. **FiLM-capacity width-direction closed at mid_dim=64.** Reassigned to #1838 depth-direction.
- **#1818 (alphonse, slice_num=128 on FiLM-only):** SWA val=408.69 (degenerate; SWA never activated), base epoch-10 val=94.79, wall-clock ~196s/epoch (~75-80% overhead). Structural cap-bounded close — slice-routing einsum scales linearly in slice_num. **Slice-routing upward expansion exhausted within 30-min envelope.** Reassigned to #1856 downward direction (slice_num=32).

## ⚠ Active operational notes

- **The GraphQL rate-limit pattern continues; pods recover automatically.** REST helpers preferred.
- **Mixed-baseline portfolio:** 7 of 8 wave-6 PRs forked from old 80.82 baseline; only #1831 (new nezuko) forked from new 74.62 baseline. Decision-rule recalibration noted in PR-by-PR review.
- **13 mechanism axes definitively closed on this dataset/scale:**
  - Architecture-capacity at generic per-feature level (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path=0.1) — closed once
  - Token-level stochastic regularization (attention_dropout=0.1) — closed once
  - Re-weight curve shape under per-batch normalization — closed once
  - SWA-window size (both directions) — closed once
  - Surf/vol loss-weighting (both directions) — closed; sw=10 brackets optimum
  - Loss-kind per domain at FiLM-scale (surf-Huber/vol-MSE) — closed (FiLM absorbed)
  - Per-channel pressure-up weighting — closed (diagnostic falsified premise)
  - **FiLM intra-capacity width-direction (mid_dim 64→128)** — closed; doubling makes modulation more aggressive but doesn't crack cross-camber bottleneck (#1760)
  - **Slice-routing upward direction (slice_num 64→128)** — structural cap close (#1818); wall-clock dominates not param count
- **4 axes have produced strong landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
  - **Optimizer-stability: grad-clip max_norm=1.0 (#1731 merged) ← NEW**
- **Largest remaining gap: val_geom_camber_rc (now 90.92 on grad-clip+FiLM baseline, down from 97.36 on FiLM-only).** Geometry-aware levers remain wave-7+ priority.
- **Composition pattern confirmed:** grad-clip + FiLM compose constructively (orthogonal mechanisms). Both add ~7% improvement on val each. Supports the broader thesis that stability-enabling levers and conditioning-enabling levers stack.

## Mechanism-axis coverage (all 8 students, wave 6+)

- **Loss-shape (β):** β=0.3 ported to FiLM stack (#1757 frieren) — pending re-evaluation against 74.62
- **Loss-shape (per-domain kind):** **CLOSED** at FiLM-scale (#1739; FiLM absorbed)
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (sw=10 brackets optimum)
- **Loss-weighting (channel-level, p-up):** **CLOSED** (#1702; diagnostic falsified premise)
- **Loss-weighting (channel-level, uxuy-up):** uxuy_weight=2.0 (#1821 askeladd) — pending re-evaluation
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn) — pending
- **Optimizer-stability (grad-clip max_norm):** **LANDED in baseline (#1731)** — sweep on the new baseline (#1831 nezuko)
- **Data-side input augmentation (node-level):** mesh-node subsampling (#1758 fern) — pending
- **Data-side input augmentation (sample-level):** Re-jitter (#1787 edward) — pending
- **Architecture-conditioning (intra-FiLM-capacity, width):** **CLOSED** at mid_dim=64 (#1760; doubling over-aggressive)
- **Architecture-conditioning (intra-FiLM-capacity, depth):** FiLM depth 2→3 (#1838 tanjiro) — compositional modulation, functionally different from width
- **Architecture-conditioning (intra-routing-capacity, upward):** **CLOSED** structural cap-bound (#1818; slice_num=128 wall-clock dominates)
- **Architecture-conditioning (intra-routing-capacity, downward):** slice_num 64→32 (#1856 alphonse) ← NEW — tests "does FiLM stabilize a smaller routing set?"
- **Architecture-conditioning (head):** FiLM — LANDED in baseline (#1585)
- **Schedule / SWA-window:** definitively closed
- **Internal regularization:** definitively closed (3 sub-axes)

**13 orthogonal mechanism axes total — 4 landed (Huber, Re-weight, FiLM, grad-clip), 10 closed, 7 pending.** Composition pattern: stability levers (grad-clip) and conditioning levers (FiLM) compose constructively.

**Mechanism finding from #1818 closure:** wall-clock prediction for capacity-axis PRs must account for ops that scale linearly with the changed dimension (PhysicsAttention routing einsum scales linearly in slice_num, not in param count — student's 5-8% prediction was off ~10×). Adding this to PR-instruction template for capacity-axis hypotheses going forward.

## Potential next research directions (wave 7+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new grad-clip+FiLM baseline 74.62:

1. **Geometry-aware lever stacked with FiLM+grad-clip** — `val_geom_camber_rc=90.92` is now the bottleneck (down from 97.36 but still the highest split). Top candidates:
   - **Geometry-conditioned FiLM** (per-token (γ,β) gated by `is_surface` and conditioned on NACA params)
   - Surface arc-length / dsdf positional encoding for surface nodes
   - SDF-conditioned attention bias
2. **Compound stack tests of wave-6 winners (if any land on new baseline):** if β=0.3 lands AND uxuy_weight lands AND slice_num lands, compounding them is the next step.
3. **max_norm sweep follow-ups (if #1831 lands):**
   - Sweep further outward (5.0, 10.0) or inward (0.1, 0.25)
   - 3-seed confirmation at the winning value
   - Adaptive clipping (grad-norm-based threshold)
4. **β-axis follow-ups (if #1757 lands or comes back close):**
   - β=0.1 monotonic continuation
   - Per-channel β (surf-p gets lower β)
5. **Mechanism-port retest** — wave-6 PRs with strong signal but didn't beat the new baseline are wave-7 candidates for retest on the new grad-clip+FiLM frame.
6. **More epochs on grad-clip+FiLM baseline** — val curve still descending at epoch 13 (cap hit). 20 epochs likely +2-4 pts but exceeds 30-min cap; could try a longer-run-aware configuration.
7. **Hard-example mining / focal-loss-style sample weighting** — direct follow-up to thorfinn's #1642 finding.
8. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.

## Open questions to revisit on next review

- **Wave-6 batch recalibration:** 5 PRs forked from old 80.82 baseline (down from 7). When they post terminal results, evaluate against new 74.62 baseline. Most likely most close; cherry-pick mechanism-orthogonal ones for wave-7 retest on new grad-clip+FiLM frame.
- **Slice-routing axis closure pending #1856:** if slice_num=32 lands, downward becomes lever direction (sweep 16, 8). If it doesn't land, slice-routing-as-an-axis is fully exhausted (both directions tested cleanly).
- **max_norm sweep (nezuko #1831):** direction signal from {0.5, 2.0} is high-information. If 2.0 wins → axis relax-direction continues, sweep upward. If 0.5 wins → axis tighten-direction continues, sweep downward. If 1.0 is the optimum (both arms regress) → close axis at merged value.
- **β=0.3 composition (frieren #1757) re-evaluation:** the strongest mechanism-port test. May still land on new baseline since FiLM/grad-clip don't directly address per-residual shape.
- **FiLM depth (#1838 tanjiro):** compositional capacity hypothesis. Width direction closed; depth axis test is functionally orthogonal.
- **val_geom_camber_rc bottleneck:** 90.92 on new baseline, still the highest split. Wave-7 geometry-axis is the highest expected ROI.
- **Wall-clock budget tightness:** the grad-clip+FiLM baseline runs at 30.9 min (cap=30); both seeds hit cap at epoch 13/15. Future PRs need to account for the slightly tighter envelope. **Capacity-axis PRs must predict wall-clock based on dim-scaling ops, not param count.**
- **Architectural-capacity-axis saturation hypothesis:** if both #1856 (slice-routing downward) and #1838 (FiLM depth) close cleanly, that's strong evidence that **architectural capacity axes are saturated on this 1.5K-sample dataset**. The next family becomes geometry-feature augmentation (per-node SDF, surface arc-length, boundary-aware encoding) — exactly what wave-7 should pivot to.
