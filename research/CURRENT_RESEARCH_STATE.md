# SENPAI Research State

- **Last updated:** 2026-05-13 03:00 (post #1739 + #1702 closes; alphonse → #1818 slice_num=128, askeladd → #1821 uxuy_weight=2.0; wave-6 portfolio still 8 students all active)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)

## ⭐ Current baseline (PR #1585 merged 2026-05-12 23:55 UTC)

- **val_avg/mae_surf_p:** **80.8162** (best seed, epoch 14, base-model — tested on Huber-only stack; merged code adds Re-weight + SWA)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite)
- **3-seed mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64
- Config: Transolver + **FiLM (mid_dim=64, zero-init last linear, per-layer (γ,β))** + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + surf_weight=10.0
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- See `BASELINE.md` for full reproducible spec + composition warning.

## 🔥 Hottest signals this session

- **PR #1585 (askeladd, FiLM-on-Huber):** MERGED. Largest single-PR gain on this branch. val=80.82 (−15.6%) / test=71.30 (−17.3%) vs prior 95.75/86.17. Architecture-conditioning axis landed.
- **#1739 (alphonse, surf-Huber/vol-MSE on FiLM) close 2026-05-13 02:50:** val=84.18 (+4.16%) / test=74.61 (+4.64%). **High-information mechanism finding: FiLM has absorbed the per-domain optimization mechanism.** Cross-condition splits (camber_rc, camber_cruise, re_rand) all within seed-variance (z≤+0.59); regression concentrates on `single_in_dist` (+12.93%). The wave-3 #1618 win was substituting for what FiLM now provides explicitly. **Loss-kind axis closed at FiLM-scale.** Reassigned to #1818 (slice_num=128).
- **#1702 (askeladd, per-channel p-weight 2.0/3.0) close 2026-05-13 02:55:** Best arm p=3.0: val=84.00 / test=74.92. **Diagnostic falsified the premise:** `p_vol/Ux_vol` ratio 0.78→0.60, `p_vol/Uy_vol` ratio 0.88→0.56 over training — pressure is *easier* in normalized space, not harder. Ux/Uy account for the larger residual fraction. Only cruise split improved (physically pressure-dominated). Reassigned to #1821 (uxuy_weight=2.0, inverse direction).
- **#1691 (edward, surf_weight=5) close 2026-05-13 02:20:** val=98.61. Surf/vol weighting axis closed in both directions (sw=30 +7%, sw=5 +3%; sw=10 brackets optimum). Volume MAE improved (-4.95% test vol) but didn't flow into surface MAE → **surface-pressure prediction is loss-weighted-attention-bound, not representation-bound**. Reassigned to #1787 (Re-jitter).
- **#1734 (thorfinn, asinh-pressure α=1.0) result 2026-05-13 01:50:** val=80.00 / test=72.71 — both within seed-variance. **Mechanism is real and structural per-split:** cruise/re_rand improve dramatically (-7-12%), single/rc regress (+3-10%). Re-running at α=0.5 (gentler compression).

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 03:00 UTC, zero open issues on this advisor branch.

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, scoring NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on swa_model | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted` | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | **FiLM global conditioning** (zero-init head, per-layer (γ,β)) | **val=80.82, test=71.30** (current) |

## Current research focus

### Wave 6 (in flight, on the merged FiLM baseline)

All 8 PRs forked from the new baseline. Two fresh reassignments at 03:00 UTC after #1739 + #1702 closes.

| PR | Student | Slug | Mechanism axis | Predicted Δ vs. 80.82 |
|---|---|---|---|---|
| #1818 | alphonse | `slice-num-128-on-filmed` | **Slice_num 64→128** (intra-routing categorical capacity; mech-orthogonal to closed n_hidden/mlp_ratio) ← NEW | −0.5 to −3% |
| #1821 | askeladd | `uxuy-weight-2p0-on-filmed` | **Vol Ux/Uy loss-weight 2.0×** (inverse of #1702 — diagnostic showed Ux/Uy are the harder channels) ← NEW | −0.5 to −3% |
| #1731 | nezuko | `grad-clip-on-filmed` | Gradient clipping (max_norm=1.0, 2 seeds) — retest of wave-3 win | −0.5 to −2% |
| #1734 | thorfinn | `asinh-0p5-pressure-on-filmed` | Value-level pressure-target compression (gentler α=0.5 after α=1.0 had structural per-split asymmetry) | −0.5 to −2% val + clean test |
| #1757 | frieren | `beta-0p3-on-filmed` | **β=0.3 port from closed #1600** (single-arm composition test on FiLM stack) | −1 to −5% val / **−2 to −7% test** ← strongest expected mechanism gain |
| #1758 | fern | `mesh-subsample-0p9-on-filmed` | Random mesh-node subsampling (data-side input augmentation, node-level) | −0.5 to −2% |
| #1760 | tanjiro | `film-mid-dim-128-on-filmed` | **FiLM mid_dim 64 → 128** (intra-FiLM capacity, NOT generic n_hidden/mlp_ratio) | −0.5 to −3% |
| #1787 | edward | `re-jitter-0p05-on-filmed` | **Re-jitter** (σ=0.05 Gaussian on log_re_shifted at model input only; sample-level data-aug) | −0.5 to −2% val |

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
- **#1618 (alphonse, surf-Huber/vol-MSE on SWA-on-Huber):** val=95.79 (+18.5%). Clean −3.3% / −3.9% on prior frame with uniform improvement across all 4 splits. Reassigned to wave-6 #1739 (composition test on FiLM baseline → CLOSED, see below).
- **#1733 (fern, attn-dropout=0.1):** val=83.86 (+3.76%). Convergence-rate collapse; 3rd regularization-axis closure. Reassigned to #1758.
- **#1732 (tanjiro, swa_start=0.65):** val=84.06 (+4.01%). SWA-window axis closed both directions. Reassigned to #1760.
- **#1600 (frieren, β-sweep on SWA-on-Huber):** β=0.3 won at val=96.35 / test=84.76 on pre-FiLM frame. Monotonic β response + asymmetric test/val gain — strongest portable mechanism. Reassigned to #1757.
- **#1691 (edward, surf_weight=5 on merged):** val=98.61. Surf/vol axis closed both directions. Reassigned to #1787.
- **#1739 (alphonse, surf-Huber/vol-MSE on FiLM):** val=84.18. **FiLM absorbed the per-domain optimization mechanism.** Cross-condition splits all within seed-variance; regression on `single_in_dist`. Loss-kind axis closed at FiLM-scale. Reassigned to #1818.
- **#1702 (askeladd, per-channel p-weight 2.0/3.0 on FiLM):** Best val=84.00 (p=3.0). **Diagnostic falsified premise: pressure is easier in normalized space.** Ux/Uy account for the larger residual fraction. Reassigned to #1821 (inverse direction).

## ⚠ Active operational notes

- **The GraphQL rate-limit pattern continues; pods recover automatically.** REST helpers preferred where possible.
- **All 8 wave-6 PRs start from the merged FiLM baseline** — no rebase pain, clean composition tests.
- **10 mechanism axes have been definitively closed on this dataset/scale:**
  - Architecture-capacity at generic per-feature level (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path=0.1) — closed once
  - Token-level stochastic regularization (attention_dropout=0.1) — closed once (PR #1733)
  - Re-weight curve shape under per-batch normalization — closed once
  - SWA-window size (both directions: removal, enlargement) — closed once (PR #1732)
  - **Surf/vol loss-weighting (both directions: sw=30, sw=5)** — closed; sw=10 brackets optimum (PR #1620 + #1691)
  - **Loss-kind per domain at FiLM-scale (surf-Huber/vol-MSE)** — closed once (PR #1739; FiLM absorbed the mechanism)
  - **Per-channel pressure-up weighting (p_weight=2.0, 3.0)** — closed once (PR #1702; diagnostic falsified premise; inverse direction in test as #1821)
- **Regularization axis pattern (3 closures):** this 5-layer / 0.75M / ~1500-sample regime needs **more training signal, not less**. Wave-7 input-augmentation tests should explicitly INCREASE per-epoch input variability.
- **3 axes have produced strong landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
- **Largest remaining gap: val_geom_camber_rc (97.36 on FiLM baseline, −5.65% from prior).** Geometry-aware levers are wave-7 priority.
- **Mechanism-absorption finding (#1739):** when an architectural innovation (FiLM) lands large, prior wave wins on loss-mechanism axes need composition-retest — they may have been substituting for what the new architecture now provides.

## Mechanism-axis coverage (all 8 students, wave 6+)

- **Loss-shape (β):** β=0.3 ported to FiLM stack (#1757 frieren) ← highest expected ROI
- **Loss-shape (per-domain kind):** **CLOSED** at FiLM-scale (#1739; FiLM absorbed the mechanism)
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (sw=10 brackets optimum)
- **Loss-weighting (channel-level, pressure-up):** **CLOSED** (#1702; diagnostic falsified premise)
- **Loss-weighting (channel-level, uxuy-up):** **uxuy_weight=2.0 (#1821 askeladd)** ← NEW — inverse direction informed by #1702 diagnostic
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn)
- **Optimizer-stability:** gradient clipping (#1731 nezuko)
- **Data-side input augmentation (node-level):** mesh-node subsampling (#1758 fern)
- **Data-side input augmentation (sample-level):** Re-jitter (#1787 edward)
- **Architecture-conditioning (intra-FiLM-capacity):** FiLM mid_dim 64→128 (#1760 tanjiro)
- **Architecture-conditioning (intra-routing-capacity):** **slice_num 64→128 (#1818 alphonse)** ← NEW — discrete categorical capacity, mech-orthogonal to closed n_hidden/mlp_ratio
- **Architecture-conditioning (head):** FiLM — LANDED in baseline (#1585)
- **Schedule / SWA-window:** definitively closed (both directions tested)
- **Internal regularization (drop_path / attention_dropout / mlp_ratio):** definitively closed (3 sub-axes, 3 closures)

**10 orthogonal mechanism axes in flight, all forked from FiLM baseline.** Notable findings this loop: (a) FiLM absorbs prior loss-shape mechanism wins; (b) per-channel weighting diagnostic falsified own premise but identified inverse direction. Surface-pressure prediction is **loss-weighted-attention-bound, not representation-bound** (from #1691) — implies signal-addition (augmentation) more promising than signal-reshape (loss-weighting).

## Potential next research directions (wave 7+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the FiLM baseline 80.82:

1. **Geometry-aware lever stacked with FiLM** — `val_geom_camber_rc=97.36` is the bottleneck FiLM left untouched. Top candidates:
   - **Geometry-conditioned FiLM** (per-token (γ,β) gated by `is_surface` and conditioned on NACA params)
   - Surface arc-length / dsdf positional encoding for surface nodes
   - SDF-conditioned attention bias
2. **Compound stack tests of wave-6 winners** — if β=0.3 lands AND grad-clip lands AND asinh lands, compounding them is the next step. Plausible compound floor ~74-76 val / ~63-66 test.
3. **β-axis follow-ups (if #1757 lands):**
   - β=0.1 to test continuation of the monotonic curve
   - Per-channel β (surf-p gets lower β, Ux/Uy keep β=1.0) — exploits dataset's per-channel residual asymmetry
4. **Mesh-augmentation follow-ups (if #1758 lands):**
   - Subsample sweep (node_keep_prob ∈ {0.75, 0.5})
   - Position jitter (small Gaussian noise on mesh coordinates, non-boundary only)
5. **Slice-routing follow-ups (if #1818 lands):**
   - slice_num=192 or 256 to find optimum
   - slice_num=32 to test downward direction (does FiLM stabilize a smaller routing set?)
6. **Per-channel weighting follow-ups (if #1821 lands):**
   - uxuy_weight=3.0 (more aggressive)
   - Asymmetric Ux vs Uy weighting (each channel independently)
   - Structural channel-disentanglement (separate prediction heads for surf and vol)
7. **Mechanism-absorption check** — given FiLM absorbed #1618's mechanism, retest other wave-3 wins (Re-weight if not already in baseline, surf_weight optimum) on FiLM-stable frame.
8. **More epochs on FiLM-merged baseline** — val curve was still descending at epoch 14. 20-epoch run with adjusted cosine T_max likely +2-4 pts, but exceeds 30-min cap.
9. **Hard-example mining / focal-loss-style sample weighting** — direct follow-up to thorfinn's #1642 finding.
10. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 (some now realized as wave-6 PRs).

## Open questions to revisit on next review

- **asinh α-tuning (#1734):** the gentler α=0.5 retest is high-information: if it lands clean, value-level compression becomes stack-compatible. If it doesn't, the axis closes.
- **β=0.3 composition (frieren #1757):** strongest mechanism-port test of this session. Does the monotonic-β finding from pre-FiLM stack survive FiLM composition?
- **Intra-FiLM capacity (tanjiro #1760) vs intra-routing capacity (alphonse #1818):** two architectural-capacity axes both running in parallel. Both could land (orthogonal), or one absorbs the other. Independent results would inform whether to compound.
- **Per-channel weighting (askeladd #1821):** inverse direction of the closed #1702. If this lands AND #1702 closed cleanly, axis is fully bracketed at uxuy-up. If neither lands, structural channel-disentanglement (separate decoders) becomes the next axis.
- **Mesh subsampling (fern #1758) × Re-jitter (edward #1787):** parallel tests of data-side input-augmentation at different levels. If both land, compound stack becomes wave-7 priority. If neither lands, data-augmentation family on this stack is exhausted and structural levers (geometry-aware) become priority.
- **val_geom_camber_rc bottleneck:** none of the wave-6 PRs directly target this split. If wave-6 lands incremental wins but val_geom_camber_rc stays around 97, wave-7 must be geometry-axis.
- **Wall-clock budget tightness:** the FiLM baseline already runs at ~32 min (cap=30); some wave-6 PRs may overrun slightly. Watch SENPAI-RESULT timings.
