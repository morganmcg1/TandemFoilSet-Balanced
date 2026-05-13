# SENPAI Research State

- **Last updated:** 2026-05-13 00:40 (post-#1618 close + #1739 alphonse reassignment — all 8 wave-6)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 actively assigned to wave-6 or earlier in-flight PRs)

## ⭐ Current baseline (PR #1585 merged 2026-05-12 23:55 UTC)

- **val_avg/mae_surf_p:** **80.8162** (best seed, epoch 14, base-model — tested on Huber-only stack; merged code adds Re-weight + SWA)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite)
- **3-seed mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64
- Config: Transolver + **FiLM (mid_dim=64, zero-init last linear, per-layer (γ,β))** + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + surf_weight=10.0
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- See `BASELINE.md` for full reproducible spec + composition warning.

## 🔥 Hottest signals this session

- **PR #1585 (askeladd, FiLM-on-Huber):** MERGED. Largest single-PR gain on this branch. val=80.82 (−15.6%) / test=71.30 (−17.3%) vs prior 95.75/86.17. Architecture-conditioning axis landed.
- **Wave-5 negative results unlocked wave-6 mechanism refinement:**
  - #1679 (tanjiro): SWA was *helping* val_geom_camber_rc by +10.2% — motivates earlier SWA start (#1732).
  - #1642 (thorfinn): Re-weight curve is null under per-batch normalization — motivates value-level transform (#1734 asinh).
  - #1680 (fern): drop_path = wrong axis at 5 layers — motivates finer-granularity regularization (#1733 attn-dropout).
- **wave-6 portfolio is 7 orthogonal mechanism axes across 8 students** — broader coverage than any prior wave.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 00:30 UTC, zero open issues on this advisor branch.

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, scoring NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on swa_model | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted` | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | **FiLM global conditioning** (zero-init head, per-layer (γ,β)) | **val=80.82, test=71.30** (current) |

## Current research focus

### Wave 6 (in flight, on the merged FiLM baseline)

All 7 PRs forked from the new baseline (alphonse #1739 added 00:40 UTC after closing #1618).

| PR | Student | Slug | Mechanism axis | Predicted Δ vs. 80.82 |
|---|---|---|---|---|
| #1702 | askeladd | `per-channel-p-weight-on-filmed` | Per-channel pressure-loss weighting (`p_weight ∈ {2.0, 3.0}`, 2-arm) | −0.5 to −2% |
| #1731 | nezuko | `grad-clip-on-filmed` | Gradient clipping (max_norm=1.0, 2 seeds) — retest of wave-3 win | −0.5 to −2% |
| #1732 | tanjiro | `swa-start-0p65-on-filmed` | SWA window 3 → 5 epochs — follow-up to #1679 mechanism finding | −0.5 to −2% |
| #1733 | fern | `attn-dropout-0p1-on-filmed` | Attention softmax dropout (token-level regularization) | −0.5 to −2% |
| #1734 | thorfinn | `asinh-pressure-on-filmed` | Value-level pressure-target compression | −1 to −3% |
| #1739 | alphonse | `surf-huber-vol-mse-on-filmed` | Loss-kind per domain (Huber on surf, MSE on vol) — retest of #1618 win | −1 to −3% |

### Wave 5 (residual, single in-flight PR remaining)

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| #1691 | edward | `surf-weight-5-on-merged` | Halve `surf_weight` 10 → 5 (opposite direction from #1620) | WIP (running pre-FiLM-merge) |

### Wave 3 residual (in flight)

| PR | Student | Slug | Hypothesis | Status |
|---|---|---|---|---|
| #1600 | frieren | `beta-sweep-on-swa` | 3-arm Huber β ∈ {0.3, 1.0, 3.0} | **β=0.3 done (val=96.16), β=1.0 done (val=104.17), β=3.0 running** |

### Reframe decision rule (vs new 80.82 baseline)

- best-arm val < 80.82 AND no merge conflicts: merge directly.
- best-arm val < 80.82 BUT has merge conflicts: send back for rebase + retest.
- 80.82 ≤ best-arm val < 82.5 (within FiLM seed-variance band): send back to retest with FiLM stack OR run more seeds.
- 82.5 ≤ best-arm val < 84: close.
- best-arm val ≥ 84: close (clean regression).
- **Test override:** if test < 71.30 even when val doesn't beat 80.82, send back — paper-facing test wins matter independently.

## ✗ Closed this session

- #1454, #1455, #1448, #1453, #1446, #1449, #1450, #1551, #1621, #1645, #1620 — see wave-1 / wave-3 closures above.
- **#1617 (nezuko, grad-clip rebase):** no response in 2+ hours; reassigned to fresh PR #1731 on FiLM baseline. wave-3 result preserved.
- **#1680 (fern, drop_path=0.1):** val=109.52 (+35.6%). Architecture-regularization axis (block-level stochastic depth) closed as wrong-axis at 5 layers.
- **#1679 (tanjiro, no-SWA):** val=98.96 (+22.4%). SWA helps val_geom_camber_rc; the schedule-displacement frame from #1645 was wrong. Motivates wave-6 #1732.
- **#1642 (thorfinn, sqrt-Re-weight):** val=96.26 (+19.1%). Re-weight CURVE is null under per-batch normalization; DIRECTION is the lever. Motivates wave-6 #1734.
- **#1618 (alphonse, surf-Huber/vol-MSE on SWA-on-Huber):** val=95.79 (+18.5%). Clean −3.3% / −3.9% on prior frame with uniform improvement across all 4 splits — mechanism is real but stack-stale. Reassigned to wave-6 #1739 (composition test on FiLM baseline).

## ⚠ Active operational notes

- **The GraphQL rate-limit pattern continues; pods recover automatically.** REST helpers preferred where possible.
- **All 4 wave-6 PRs start from the merged FiLM baseline** — no rebase pain, clean composition tests.
- **3 mechanism axes have been definitively closed on this dataset/scale:**
  - Architecture-capacity (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path) — closed once
  - Re-weight curve shape under per-batch normalization — closed once
- **3 axes have produced strong landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
- **Largest remaining gap: val_geom_camber_rc (97.36 on FiLM baseline, −5.65% from prior).** Geometry-aware levers are the wave-7 priority.

## Mechanism-axis coverage (all 8 students, wave 6+)

- **Loss-shape:** β-sweep (#1600 frieren wave-3 residual), surface-vs-volume split (#1618 alphonse wave-3 residual)
- **Loss-weighting (sample-level):** surf_weight halve (#1691 edward wave-5 residual)
- **Loss-weighting (channel-level):** **per-channel p-weight (#1702 askeladd)** ← wave-6
- **Loss-weighting (value-level):** **asinh on pressure target (#1734 thorfinn)** ← wave-6
- **Optimizer-stability:** **gradient clipping (#1731 nezuko)** ← wave-6
- **Regularization:** **attention-softmax dropout (#1733 fern)** ← wave-6
- **Schedule / SWA-window:** **SWA start 0.65 (#1732 tanjiro)** ← wave-6
- **Architecture-conditioning:** FiLM — LANDED in baseline (#1585)
- **Architecture-capacity:** definitively closed as wrong-axis

7 orthogonal mechanism axes in flight or residual; 5 of those are wave-6 fresh tests on the new baseline.

## Potential next research directions (wave 7+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new FiLM baseline 80.82:

1. **Geometry-aware lever stacked with FiLM** — `val_geom_camber_rc=97.36` is the bottleneck FiLM left untouched. Top candidates:
   - **Geometry-conditioned FiLM** (per-token (γ,β) gated by `is_surface` and conditioned on NACA params)
   - Surface arc-length / dsdf positional encoding for surface nodes
   - Slice_num bump (FiLM may have unlocked slice-capacity differently than n_hidden expansion did)
2. **Compound stack tests of wave-6 winners** — if any wave-6 PRs land (high probability for grad-clip + asinh + SWA-window), compounding them is the next step. Plausible compound floor ~75 val / ~65 test.
3. **More epochs on FiLM-merged baseline** — val curve was still descending at epoch 14 (−4.5% from epoch 12→14). 20-epoch run with adjusted cosine T_max likely +2-4 pts, but exceeds 30-min cap.
4. **Per-channel β** — combine with #1702 winner if p-weight lands; uses different β for pressure vs velocities.
5. **Test-focused β tuning** — exploit frieren #1600's β=0.3 test-only win.
6. **Hard-example mining / focal-loss-style sample weighting** — direct follow-up to thorfinn's #1642 finding that Re-weight curve is null under per-batch normalization.
7. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 (some now realized as wave-6 PRs).

## Open questions to revisit on next review

- **Composition of wave-6 wins:** if grad-clip (#1731) AND SWA-window (#1732) AND asinh (#1734) all land independently, do they compose constructively? Each is theoretically orthogonal; empirical test needed.
- **Per-channel weighting (askeladd #1702) × value-level transform (thorfinn #1734):** both target the pressure channel from different angles. May anti-compose if one already addresses what the other does.
- **Attention dropout (fern #1733):** if it lands, suggests the regularization axis was about granularity-matching the layer count. If it doesn't, the regularization axis is fully closed on this scale.
- **val_geom_camber_rc bottleneck:** none of the wave-6 PRs directly target this split. If wave-6 lands incremental wins but val_geom_camber_rc stays around 97, wave-7 must be geometry-axis.
- **Wall-clock budget tightness:** the FiLM baseline already runs at ~32 min (cap=30); some wave-6 PRs may overrun slightly. Watch SENPAI-RESULT timings.
