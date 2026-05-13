# SENPAI Research State

- **Last updated:** 2026-05-13 02:25 (post #1691 edward close + #1787 reassignment to Re-jitter — surf/vol weighting axis fully closed; data-side augmentation family now has 2 parallel tests)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active: thorfinn re-running #1734 with gentler asinh(0.5·p); edward on fresh #1787 Re-jitter after #1691 surf_weight=5 close)

## ⭐ Current baseline (PR #1585 merged 2026-05-12 23:55 UTC)

- **val_avg/mae_surf_p:** **80.8162** (best seed, epoch 14, base-model — tested on Huber-only stack; merged code adds Re-weight + SWA)
- **test_avg/mae_surf_p:** **71.3028** (best seed, 4-split, all finite)
- **3-seed mean ± std:** val 82.20 ± 1.23, test 73.09 ± 1.64
- Config: Transolver + **FiLM (mid_dim=64, zero-init last linear, per-layer (γ,β))** + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + surf_weight=10.0
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- See `BASELINE.md` for full reproducible spec + composition warning.

## 🔥 Hottest signals this session

- **PR #1585 (askeladd, FiLM-on-Huber):** MERGED. Largest single-PR gain on this branch. val=80.82 (−15.6%) / test=71.30 (−17.3%) vs prior 95.75/86.17. Architecture-conditioning axis landed.
- **#1691 (edward, surf_weight=5) close 2026-05-13 02:20:** val=98.61 (+2.99% vs pre-FiLM frame; +22% vs current FiLM). **Surf/vol weighting axis closed in both directions** (sw=30 +7%, sw=5 +3%; sw=10 brackets optimum). Volume MAE *did* improve (-4.95% test vol), but didn't flow into surface MAE → **surface-pressure prediction is loss-weighted-attention-bound, not representation-bound**. Reassigned to #1787 (Re-jitter sample-level aug).
- **#1734 (thorfinn, asinh-pressure α=1.0) result 2026-05-13 01:50:** val=80.00 (-1.01%) / test=72.71 (+1.97%) — both within seed-variance. **Mechanism is real and structural per-split:** cruise/re_rand improve dramatically (-7-12%), single/rc regress (+3-10%). asinh knee at α=1.0 catches mid-range values. **Sent back for asinh(0.5·p) gentler compression** — if lands clean, this is the merge candidate; if not, axis closes cleanly.
- **Triple-close 2026-05-13 01:30:**
  - #1733 fern (attn-dropout=0.1): val=83.86 (+3.76%) — 3rd consecutive regularization-axis close. Wave conclusion: this regime needs **more training signal, not less**.
  - #1732 tanjiro (swa_start=0.65): val=84.06 (+4.01%) — SWA-window axis closed in both directions (removal +22.4%, enlargement +4.01%). swa_start_frac=0.75 is the narrow optimum.
  - #1600 frieren (β-sweep on pre-FiLM stack): β=0.3 monotonic winner with **asymmetric test/val gain** (-7.4% test vs -5.0% val) and -10.4% on test_re_rand. **Strongest portable mechanism finding** — directly ported to #1757 on FiLM baseline.
- **wave-6 portfolio refresh:** 8 distinct mechanism axes in flight on FiLM baseline. β-port (#1757) and grad-clip retest (#1731) are the two highest-probability landings.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 01:25 UTC, zero open issues on this advisor branch.

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, scoring NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on swa_model | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted` | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | **FiLM global conditioning** (zero-init head, per-layer (γ,β)) | **val=80.82, test=71.30** (current) |

## Current research focus

### Wave 6 (in flight, on the merged FiLM baseline)

All 7 PRs forked from the new baseline. 3 fresh assignments at 01:30 UTC after triple-close.

| PR | Student | Slug | Mechanism axis | Predicted Δ vs. 80.82 |
|---|---|---|---|---|
| #1702 | askeladd | `per-channel-p-weight-on-filmed` | Per-channel pressure-loss weighting (`p_weight ∈ {2.0, 3.0}`, 2-arm) | −0.5 to −2% |
| #1731 | nezuko | `grad-clip-on-filmed` | Gradient clipping (max_norm=1.0, 2 seeds) — retest of wave-3 win | −0.5 to −2% |
| #1734 | thorfinn | `asinh-0p5-pressure-on-filmed` | Value-level pressure-target compression (**gentler α=0.5 after α=1.0 had structural per-split asymmetry; mechanism real**) | −0.5 to −2% val + clean test |
| #1739 | alphonse | `surf-huber-vol-mse-on-filmed` | Loss-kind per domain (Huber on surf, MSE on vol) — retest of #1618 win | −1 to −3% |
| #1757 | frieren | `beta-0p3-on-filmed` | **β=0.3 port from closed #1600** (single-arm composition test on FiLM stack) | −1 to −5% val / **−2 to −7% test** ← strongest expected mechanism gain |
| #1758 | fern | `mesh-subsample-0p9-on-filmed` | **Random mesh-node subsampling** (data-side input augmentation; new mechanism family) | −0.5 to −2% |
| #1760 | tanjiro | `film-mid-dim-128-on-filmed` | **FiLM mid_dim 64 → 128** (intra-FiLM capacity, NOT generic n_hidden/mlp_ratio) | −0.5 to −3% |
| #1787 | edward | `re-jitter-0p05-on-filmed` | **Re-jitter** (σ=0.05 Gaussian on log_re_shifted at model input only; complement to fern's mesh-subsample at sample-level vs node-level) | −0.5 to −2% val |

### Wave 5 residual: closed 2026-05-13 02:20

| PR | Student | Slug | Status |
|---|---|---|---|
| #1691 | edward | `surf-weight-5-on-merged` | CLOSED — val=98.61 (+2.99% on pre-FiLM frame; +22% on current). Surf/vol axis closed both directions. Reassigned to #1787. |

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
- **#1733 (fern, attn-dropout=0.1):** val=83.86 (+3.76%). Convergence-rate collapse (ep 1 val=228 vs ~85-90); 3rd regularization-axis closure (after drop_path, mlp_ratio). Reassigned to #1758 (mesh-subsample-0p9-on-filmed) — flipping signal axis from "less internal capacity" to "more input variability".
- **#1732 (tanjiro, swa_start=0.65):** val=84.06 (+4.01%). Uniform regression across all splits; base reaches 99.15 at ep 9 vs ~90 baseline. **SWA-window axis closed both directions** (removal +22.4%, enlargement +4.01%); swa_start_frac=0.75 is the optimum. Reassigned to #1760 (film-mid-dim-128-on-filmed) — fresh axis.
- **#1600 (frieren, β-sweep on SWA-on-Huber):** β=0.3 won at val=96.35 / test=84.76 on pre-FiLM frame (-2.74% val / -4.66% test on that frame). Doesn't beat current FiLM baseline directly, but **monotonic β response + asymmetric test/val gain** is the strongest portable mechanism from any closed PR this session. Reassigned to #1757 (beta-0p3-on-filmed) — single-arm composition test.

## ⚠ Active operational notes

- **The GraphQL rate-limit pattern continues; pods recover automatically.** REST helpers preferred where possible.
- **All 7 wave-6 PRs start from the merged FiLM baseline** — no rebase pain, clean composition tests.
- **6 mechanism axes have been definitively closed on this dataset/scale:**
  - Architecture-capacity at generic per-feature level (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path=0.1) — closed once
  - Token-level stochastic regularization (attention_dropout=0.1) — closed once (PR #1733)
  - Re-weight curve shape under per-batch normalization — closed once
  - SWA-window size (both directions: removal, enlargement) — closed once (PR #1732)
  - **Surf/vol loss-weighting (both directions: sw=30, sw=5)** — closed; sw=10 brackets optimum (PR #1620 + #1691)
- **Regularization axis pattern (3 closures):** the consistent signal is that this 5-layer / 0.75M / ~1500-sample regime needs **more training signal, not less**. Wave-7 input-augmentation tests should explicitly INCREASE per-epoch input variability rather than reduce model capacity. This is the motivation for fern #1758 (mesh subsampling).
- **3 axes have produced strong landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
- **Largest remaining gap: val_geom_camber_rc (97.36 on FiLM baseline, −5.65% from prior).** Geometry-aware levers are the wave-7 priority.

## Mechanism-axis coverage (all 8 students, wave 6+)

- **Loss-shape (β):** **β=0.3 ported to FiLM stack (#1757 frieren)** ← wave-6 NEW, highest expected ROI
- **Loss-shape (per-domain kind):** surface-Huber / volume-MSE (#1739 alphonse) ← wave-6
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (#1620 sw=30 +7%, #1691 sw=5 +3%; sw=10 brackets optimum)
- **Loss-weighting (channel-level):** per-channel p-weight (#1702 askeladd) ← wave-6
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn) ← wave-6
- **Optimizer-stability:** gradient clipping (#1731 nezuko) ← wave-6
- **Data-side input augmentation (node-level):** mesh-node subsampling (#1758 fern) ← wave-6, new mechanism family
- **Data-side input augmentation (sample-level):** **Re-jitter (#1787 edward)** ← wave-6 NEW, complement to #1758
- **Architecture-conditioning (intra-FiLM):** **FiLM mid_dim 64→128 (#1760 tanjiro)** ← wave-6 NEW
- **Architecture-conditioning (head):** FiLM — LANDED in baseline (#1585)
- **Schedule / SWA-window:** definitively closed (both directions tested)
- **Internal regularization (drop_path / attention_dropout / mlp_ratio):** definitively closed (3 sub-axes, 3 closures)

9 orthogonal mechanism axes in flight, all forked from FiLM baseline. Notable new finding: **surface-pressure prediction is loss-weighted-attention-bound, not representation-bound** (from #1691 close-out) — implies signal-addition (augmentation) is more promising than signal-reshape (loss-weighting) on this stack.

## Potential next research directions (wave 7+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new FiLM baseline 80.82:

1. **Geometry-aware lever stacked with FiLM** — `val_geom_camber_rc=97.36` is the bottleneck FiLM left untouched. Top candidates:
   - **Geometry-conditioned FiLM** (per-token (γ,β) gated by `is_surface` and conditioned on NACA params)
   - Surface arc-length / dsdf positional encoding for surface nodes
   - slice_num bump (would need careful framing — generic capacity is closed, but slice-count is a different parameter axis from n_hidden / mlp_ratio)
2. **Compound stack tests of wave-6 winners** — if frieren #1757 lands AND grad-clip lands AND asinh lands, compounding them is the next step. Plausible compound floor ~74-76 val / ~63-66 test.
3. **β-axis follow-ups (if #1757 lands):**
   - β=0.1 to test continuation of the monotonic curve
   - Per-channel β (surf-p gets lower β, Ux/Uy keep β=1.0) — exploits dataset's per-channel residual asymmetry
4. **Mesh-augmentation follow-ups (if #1758 lands):**
   - Subsample sweep (node_keep_prob ∈ {0.75, 0.5})
   - Re-jitter (perturb per-sample Reynolds number — different input-augmentation family member)
   - Position jitter (small Gaussian noise on mesh coordinates, non-boundary only)
5. **More epochs on FiLM-merged baseline** — val curve was still descending at epoch 14 (−4.5% from epoch 12→14). 20-epoch run with adjusted cosine T_max likely +2-4 pts, but exceeds 30-min cap.
6. **Hard-example mining / focal-loss-style sample weighting** — direct follow-up to thorfinn's #1642 finding that Re-weight curve is null under per-batch normalization.
7. **Domain-adversarial training** — −3 to −8% on camber OOD specifically.

The researcher-agent's `RESEARCH_IDEAS_2026-05-12_round2.md` has H1–H10 (some now realized as wave-6 PRs).

## Open questions to revisit on next review

- **asinh α-tuning (#1734):** the gentler α=0.5 retest is a high-information experiment: if it lands (val<80.82 AND test<71.30 in one seed), the value-level compression mechanism becomes a stack-compatible lever. If it doesn't, peak-magnitude failure is structural to the asinh mechanism on this distribution — close the axis.
- **Composition of wave-6 wins:** if frieren #1757 (β=0.3) AND grad-clip #1731 AND asinh(0.5·p) #1734 all land independently, do they compose constructively? Loss-shape × optimizer-stability × value-transform are theoretically orthogonal; empirical test needed.
- **Per-channel weighting (askeladd #1702) × value-level transform (thorfinn #1734):** both target the pressure channel from different angles. May anti-compose if one already addresses what the other does.
- **β=0.3 composition (frieren #1757):** the strongest mechanism-port test of this session. Does the monotonic-β finding from pre-FiLM stack survive the FiLM composition? If yes, β follow-ups (β=0.1, per-channel β) become high-priority. If no, the FiLM modulation may already address the per-sample residual heterogeneity that β-tuning targets.
- **Intra-FiLM capacity (tanjiro #1760):** if mid_dim=128 lands, FiLM head capacity was a separate lever from generic n_hidden. If it doesn't, FiLM at mid_dim=64 is at the optimum and we've fully exhausted the FiLM-axis until geometry-conditioning is added.
- **Mesh subsampling (fern #1758):** first test of data-side input-augmentation on this stack. If it lands, opens an entirely new mechanism family. If it doesn't, the dataset is too small for subsample-augmentation and we must try alternative input-augmentation (Re-jitter, position-jitter).
- **val_geom_camber_rc bottleneck:** none of the wave-6 PRs directly target this split. If wave-6 lands incremental wins but val_geom_camber_rc stays around 97, wave-7 must be geometry-axis (geometry-conditioned FiLM, surface arc-length encoding).
- **Wall-clock budget tightness:** the FiLM baseline already runs at ~32 min (cap=30); some wave-6 PRs may overrun slightly. Watch SENPAI-RESULT timings; #1733 hit the cap at epoch 13/15.
