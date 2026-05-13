# SENPAI Research Results

_Advisor branch: `icml-appendix-charlie-pai2g-48h-r3`._

Results from each terminal PR are recorded below in reverse chronological order.

<!-- Entries will be appended as PRs land terminal SENPAI-RESULT markers. -->

## 2026-05-13 09:15 — PR #1996: slice_num=48 + T_max=15 — MERGED (−1.33% val, −1.10% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-num-48
- **Hypothesis:** Reducing slice_num 64→48 reduces per-epoch time ~18%, enabling 15 epochs in 30-min cap vs 12. Same "epoch count is the binding constraint" mechanism as PR #1995.
- **Result:** val=46.847 / test=40.837 (best_epoch=15, n_layers=6, T_max=15, sw=10)

| Split | val (PR #1995 baseline, n_layers=5) | val (slice_num=48, n_layers=6) | Δ | test |
|---|---|---|---|---|
| single_in_dist | 52.253 | 50.491 | −3.4% | 45.728 |
| geom_camber_rc | **60.809** | **60.364** | **−0.7%** | 55.146 |
| geom_camber_cruise | 29.174 | 29.835 | +2.3% | 24.157 |
| re_rand | 47.675 | 46.699 | −2.0% | 38.317 |
| **avg** | **47.478** | **46.847** | **−1.33% ✓** | **40.837** |

- **⚠ Stack mismatch:** PR was run on n_layers=6 (old stack); merged into advisor with n_layers=5 (from #1995). Current advisor code = n_layers=5 + slice_num=48, NOT validated yet. fern PR #2062 assigned to verify the compound.
- **Per-epoch time:** ~123s (n_layers=6). Expected ~100-108s on n_layers=5 + slice_num=48.
- **Metric artifacts:** `models/model-charliepai2g48h3-fern-slice-num-48-20260513-070845/metrics.yaml`
- **Reassigned fern:** PR #2062 n_layers=5 + slice_num=48 compound verification

---

## 2026-05-13 09:00 — PR #1995: n_layers=5 + T_max=14 — MERGED (−6.98% val, −6.98% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/n-layers-5
- **Hypothesis:** Shallower model (n_layers=5) → faster epochs (~116s vs ~138s) → 14 epochs fit in 30-min budget → 2 extra cosine-tail refinement epochs → better convergence. Trading capacity for training duration.
- **Result:** val=47.478 / test=41.290 (best_epoch=14, surf_weight=10)

| Split | val (n_layers=6, T=12, sw=5) | val (n_layers=5, T=14, sw=10) | Δ val | test (n_layers=5) |
|---|---|---|---|---|
| single_in_dist | 56.933 | 52.253 | −8.2% | 46.980 |
| geom_camber_rc | **64.886** | **60.809** | **−6.3%** | 54.123 |
| geom_camber_cruise | 31.056 | 29.174 | −6.1% | 24.263 |
| re_rand | 51.287 | 47.675 | −7.0% | 39.794 |
| **avg** | **51.040** | **47.478** | **−6.98% ✓** | **41.290** |

- **Key stats:** n_params=826,071 (−15.7%), VRAM 40GB (−20%), 116s/epoch (−16%)
- **Mechanism:** "Epoch count was the binding constraint, not capacity." T_max=14 aligns cosine decay to the new budget; monotonic val descent to epoch 14.
- **Important:** surf_weight=10 (NOT 5). The surf_weight=5 compound on this new stack is the immediate next priority (edward #2048).
- **train.py change:** n_layers 6→5 (no other changes; lr=cfg.lr bug still NOT fixed)
- **Metric artifacts:** `models/model-charliepai2g48h3-edward-n-layers-5-20260513-065528/metrics.yaml`
- **Reassigned edward:** PR #2048 surf_weight=5 on n_layers=5 + T_max=14 (compound)

---

## 2026-05-13 08:45 — PR #1765: Lion lr=1.5e-4 — CLOSED (+4.21% val, +1.11% test)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4-rebased (multi-rebase, renamed)
- **Hypothesis:** lr=1.5e-4 is a better midpoint than lr=2e-4 (which overshot on geom_camber_rc) on the current modernized stack.
- **Result:** Two arms — sw=5: val=53.192 / test=45.471; sw=10: val=53.398 / test=46.652. Both vs baseline 51.040 / 44.390.

| Metric | PR #1956 baseline | sw=5 + lr=1.5e-4 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 51.040 | 53.192 | +4.21% |
| test_avg/mae_surf_p | 44.390 | 45.471 | +2.43% |

- **Conclusion:** LR axis exhausted from both directions. lr=1.5e-4 (+4.21% worse) and lr=2e-4 (known worse on RMSNorm stack) both fail. lr=1e-4 is the confirmed optimum for Lion on this stack. T_max=12 shift dominates any LR midpoint benefit.
- **Key bug fix in branch:** `lr=1e-4` hardcoded in Lion constructor (train.py:441) → `lr=cfg.lr` fix applied by student. Fix NOT yet in advisor branch — sent alert to frieren #2006 to apply before running lr=8e-5.
- **Metric artifacts:** `models/model-charliepai2g48h3-alphonse-lion-lr-1.5e-4-tmax12-sw5-20260513-055705/metrics.jsonl`, `models/model-charliepai2g48h3-alphonse-lion-lr-1.5e-4-tmax12-sw10-20260513-065306/metrics.jsonl`
- **Reassigned alphonse:** PR #2043 DropPath stochastic depth rate=0.1

---

## 2026-05-13 08:45 — PR #1766: Lion WD=1e-2 (RMSNorm+GeGLU rerun) — CLOSED (+16.6% val vs current)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/lion-wd-1e-2
- **Hypothesis:** Lion paper recommends WD 10-100× higher than Adam; WD=1e-2 should compound with GeGLU+RMSNorm regularization.
- **Result (primary arm):** val=59.517 / test=53.732. Vs current baseline 51.040 (+16.6% worse).
- **Additional data point:** WD=1e-2 + T_max=12 + sw=10: val=54.594 (+6.96% worse). Neither arm beats current baseline.

| Metric | Current baseline | WD=1e-2 arm | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 51.040 | 59.517 | +16.6% |
| test_avg/mae_surf_p | 44.390 | 53.732 | +21.0% |

- **Conclusion:** WD axis fully exhausted. WD=1e-4 optimal confirmed (WD=1e-1, 3e-2, 1e-2 all worse). RMSNorm+GeGLU already provide implicit regularization that previously WD=1e-2 was compensating for; marginal WD benefit disappears on the modernized stack.
- **Note:** PR was opened against an older stack and required multiple rebases; final result still outdated vs compound baseline. WD hypothesis was valid but axis is saturated.
- **Metric artifacts:** `models/model-lion-wd-1e-2-rmsnorm-geglu-20260513-051233/metrics.jsonl`
- **Reassigned askeladd:** PR #2038 n_head=2 (head_dim 32→64 sweep)

---

## 2026-05-13 08:45 — PR #1948: surf_weight=3 — CLOSED (stale draft, no results)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/surf-weight-3
- **Hypothesis:** Is the surf_weight optimum below 5? Test sw=3 to find the gradient budget floor.
- **Result:** None — draft PR with no training run started. Built on T_max=50 stack (stale).
- **Conclusion:** Closed as stale. Surf_weight axis below 5 now covered by nezuko #2029 (sw=2) on the current T_max=12+sw=5 compound stack.
- **Reassigned thorfinn:** PR #2040 gradient clipping max_norm=1.0

---

## 2026-05-13 07:25 — PR #1956: T_max=12 + surf_weight=5 compound — MERGED (−3.33% val, −1.29% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/tmax-12-surf-weight-5-compound
- **Hypothesis:** T_max=12 and surf_weight=5 are orthogonal mechanisms (schedule vs loss weighting); should compound additively. Predicted val 48-50.
- **Result:** val=51.040, test=44.390 (epoch 12/12, 30-min cap, surf_weight=5)

| Split | val (compound) | val (#1793 baseline sw=10) | Δ | test (compound) | test (baseline) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 56.933 | 58.907 | −3.35% | 50.459 | 50.239 | +0.44% |
| geom_camber_rc | **64.886** | 67.658 | **−4.10%** | 59.341 | 59.561 | −0.37% |
| geom_camber_cruise | 31.056 | 33.380 | **−6.96%** | 25.501 | 27.740 | **−8.07%** |
| re_rand | 51.287 | 51.248 | +0.08% | 42.260 | 42.345 | −0.20% |
| **avg** | **51.040** | **52.798** | **−3.33% ✓** | **44.390** | **44.972** | **−1.29% ✓** |

- **Volume MAE check:** Volume MAE improved across ALL 4 splits (−6% to −14%) confirming gradient-budget reallocation mechanism
  - single_in_dist vol: 67.251 → 60.506 (−10.0%)
  - geom_camber_rc vol: 74.389 → 65.720 (−11.7%)
  - geom_camber_cruise vol: 33.564 → 28.904 (−13.9%)
  - re_rand vol: 52.194 → 48.815 (−6.5%)
- **Compound analysis:** Sub-linear but real. Predicted −7% to −9% if fully additive; got −3.33%. Surface gradient with sw=5 shows diminishing returns at the cosine tail — the richer volume representation helps surface via context, but at T_max=12 end-state the two mechanisms partially share the same mode.
- **re_rand is the laggard** (+0.08% val / −0.20% test) — Reynolds holdout benefits less from geometric context than geometry holdouts
- **Cruise drives test gain** (−8.07%) — largest meshes get most benefit from gradient reallocation
- **Metric artifacts:** `models/model-charliepai2g48h3-nezuko-tmax-12-surf-weight-5-compound-20260513-055720/metrics.jsonl`
- **Reassigned nezuko:** PR #2029 surf_weight=2 on compound baseline (continue gradient sweep below sw=5)

---

## 2026-05-13 07:00 — PR #1983: CosineAnnealingLR T_max=10 — CLOSED (+10.96% val, +11.95% test)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/t-max-10
- **Hypothesis:** T_max=10 (vs current T_max=12) pushes cosine floor earlier, gives Lion more low-LR refinement time.
- **Result:** val=58.586, test=50.346 (epoch 12/12)

| Split | val (T_max=10) | val (T_max=12 baseline) | Δ |
|---|---|---|---|
| single_in_dist | 63.472 | 58.907 | +7.75% |
| geom_camber_rc | 74.086 | 67.658 | **+9.50%** |
| geom_camber_cruise | 38.763 | 33.380 | **+16.13%** |
| re_rand | 58.024 | 51.248 | **+13.22%** |
| **avg** | **58.586** | **52.798** | **+10.96%** |

- **Test:** 50.346 vs 44.972 = **+11.95% worse**
- **Mechanism (student-identified, excellent analysis):**
  1. PyTorch `CosineAnnealingLR` is cyclic, not clamped at T_max. After T_max it continues the cosine, bouncing LR back up.
  2. With T_max=10 and 12 epochs run: epoch 11 LR was **exactly 0** → DEAD EPOCH (val identical to epoch 10). Epoch 12 bumped back to 2.4e-6 as cosine started rising.
  3. T_max=12's gains came from gradual decay across epochs 9-12 (1e-5 → 0), NOT from "near-zero floor." Compressing this to T_max=10 truncated the productive part of the cosine — epochs 9-10 ran at sub-3e-6 LRs where Lion's sign-update has nothing useful to do.
- **Dead end mechanism documented:** T_max < cfg.epochs is always strictly worse. T_max should be ≥ epoch count.
- **Reassigned frieren:** PR #2006 Lion lr=8e-5 (bracket alphonse's 1.5e-4 from below)

---

## 2026-05-13 07:00 — PR #1984: n_hidden=160 — CLOSED (val −0.247% / test +1.268%; complexity disproportionate)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/n-hidden-160
- **Hypothesis:** Widen Q/K/V/PhysicsAttention dim from 128→160 to capture richer aerodynamic features.
- **Result:** val=52.668 (−0.247% vs 52.798), test=45.542 (+1.268% vs 44.972), 12 epochs in 31.5 min

| Split | val (n=160) | val (n=128 baseline) | Δ | test (n=160) | test (n=128) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | 57.375 | 58.907 | **−2.60%** | 50.878 | 50.239 | +1.27% |
| geom_camber_rc | 67.923 | 67.658 | +0.39% | 61.274 | 59.561 | **+2.88%** |
| geom_camber_cruise | 33.528 | 33.380 | +0.44% | 27.207 | 27.740 | −1.92% |
| re_rand | 51.845 | 51.248 | +1.17% | 42.810 | 42.345 | +1.10% |
| **avg** | **52.668** | **52.798** | **−0.247%** | **45.542** | **44.972** | **+1.27%** |

- **Per-epoch cost:** +12% (157s vs 140s) — much cheaper than predicted +60%
- **n_params:** +52% (1.522M vs ~1.0M)
- **Peak GPU memory:** 55.22 GB (up from baseline)
- **Decision rationale (closed not merged):**
  - Val/test direction inversion → noise signature, not real win
  - "Disproportionate complexity for tiny gain" clause applies (+52% params for −0.247% val)
  - Targeted hardest split (geom_camber_rc) regressed on test (+2.88%) — the mechanism didn't work
  - Permanent slowdown on all future experiments if merged
- **Useful signal:** single_in_dist val −2.60% suggests wider attention helps in-distribution; OOD splits resist. This is opposite RMSNorm's signal (which cracked geom_camber_rc) — confirms OOD bottleneck is geometric extrapolation, not feature capacity.
- **Per-epoch calibration:** n_hidden widening is much cheaper than expected (+12% not +60%)
- **Reassigned tanjiro:** PR #2007 mlp_ratio=2 (test if "gating wins outright" extends below 4)

---

## 2026-05-13 06:45 — PR #1925: Lion WD=3e-2 — CLOSED (WD axis saturated, +0.06% on prior baseline)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-wd-3e-2
- **Hypothesis:** WD=3e-2 brackets WD optimum between in-flight WD=1e-2 (askeladd) and closed WD=1e-1 (PR #1889).
- **Result:** val=63.054, test=55.573 (14 epochs, ~138s/epoch, best_epoch=13)
- **Baseline used:** PR #1837 (val=63.017 / test=54.731; RMSNorm+GeGLU+Lion+WD=1e-4)

| Split | val (WD=3e-2) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 75.347 | 76.710 | −1.78% |
| geom_camber_rc | 76.040 | 73.930 | **+2.85%** |
| geom_camber_cruise | 40.649 | 40.746 | −0.24% |
| re_rand | 60.183 | 60.683 | −0.82% |
| **avg** | **63.054** | **63.017** | **+0.06%** |

- **Test:** 55.573 vs 54.731 = +1.54% (worse)
- **WD bracket summary:** [WD=1e-4: 63.017 / WD=3e-2: 63.054 / WD=1e-1: 64.731] — flat valley from 1e-4→3e-2, bends up sharply at 1e-1
- **Analysis:** WD=3e-2 is statistically tied with WD=1e-4 on val (well within seed noise). best_epoch=13 (same as baseline) confirms no over-regularization at this WD level. Test regression (+1.54%) suggests marginal overfitting risk without val benefit.
- **Against current baseline (#1793 T_max=12, val=52.798):** +19.4% worse (baseline-drift effect only — measured on old stack)
- **Dead end confirmed:** WD axis exhausted on Lion+RMSNorm+GeGLU stack. Further WD experiments will only refine a flat valley. WD=1e-2 (askeladd #1766) will confirm third bracket point.
- **Student insight:** Correctly identified "throughput improvements" as higher leverage than WD tuning
- **Reassigned edward:** PR #1995 n_layers=5 (shallower model → faster epochs → 3 extra epochs in budget)

---

## 2026-05-13 06:40 — PR #1790: Lion + 2-epoch cosine warmup — CLOSED (stale + mechanism conflicts with T_max=12)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/lion-cosine-warmup-2ep
- **Hypothesis:** 2-epoch linear warmup (1e-6→1e-4) stabilizes Lion's aggressive sign-based init before full cosine decay.
- **Original result (PR #1725 baseline):** val=78.315 (−9.9% vs old baseline val=86.938) — positive
- **Status:** Sent back for rerun on GeGLU+Lion stack (PR #1769 baseline val=64.918 at that time). No results produced in 2h+ after rerun request. Last branch commit at 04:05 UTC.
- **Closure rationale:**
  1. Baseline shifted twice since rerun request (→57.328 →52.798)
  2. Mechanism conflict: 2-epoch warmup consumes 17% of 12-epoch budget; T_max=12 cleanly handles LR decay without needing warmup (cold-start instability addressed by cosine from epoch 1)
  3. Student inactivity: 2h+ stale
- **Reassigned fern:** PR #1996 slice_num=48 (throughput-focused; tighter PhysicsAttention → ~2 extra epochs)

---

## 2026-05-13 06:20 — PR #1920: CosineAnnealingLR eta_min=1e-5 — CLOSED (+12.05% vs current baseline; mechanism redundant with T_max=12)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/cosine-eta-min-1e-5
- **Hypothesis:** Non-zero LR floor (eta_min=1e-5) prevents Lion's cosine tail from fully collapsing to 0, preserving some update signal in final epochs.
- **Result:** val=59.159, test=53.210 (measured vs PR #1837 baseline val=63.017 → −6.1% improvement on old baseline)

| Split | val (eta_min=1e-5) | val (old #1837 baseline) | Δ vs old | val (current #1793 baseline) | Δ vs current |
|---|---|---|---|---|---|
| single_in_dist | ~54.9 | 76.710 | −9.7% | 58.907 | −6.8% |
| geom_camber_rc | ~72.6 | 73.930 | −1.8% | 67.658 | **+7.3%** |
| geom_camber_cruise | ~43.8 | 40.746 | +7.5% | 33.380 | **+31.2%** |
| re_rand | ~65.4 | 60.683 | +7.8% | 51.248 | **+27.6%** |
| **avg** | **59.159** | **63.017** | **−6.1% ✓ (old)** | **52.798** | **+12.05% ✗** |

- **Test:** 53.210 vs 44.972 = **+18.3% worse** than current baseline
- **Analysis:** Result was measured against the PR #1837 baseline (val=63.017). The baseline shifted twice (→57.328 via PR #1836, →52.798 via PR #1793 T_max=12) before review. Against the current baseline, this is a large regression. Critically, cruise and re_rand splits regressed badly: the eta_min=1e-5 floor RAISED the LR in the late-epoch regime (vs T_max=12 decaying to 0), which interfered with Lion's fine-grained late-epoch refinement.
- **Mechanism conflict confirmed:** Both eta_min and T_max address late-epoch LR behavior; T_max=12 (LR→0) strictly dominates eta_min=1e-5 (LR floor above 0). T_max=12 won the head-to-head and is now the stack default.
- **Student's own follow-up suggestion** explicitly named "combine with T_max tuning" — that direction is now merged (PR #1793).
- **Dead end:** LR schedule "floor vs ceiling" axis is exhausted. T_max=12 is the correct solution.
- **Reassigned frieren:** PR #1983 CosineAnnealingLR T_max=10 (push cosine further)

---

## 2026-05-13 06:20 — PR #1872: mlp_ratio=8 + GeGLU — CLOSED (+5.95% regression, "gating wins outright")

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/mlp-ratio-8-geglu-lion
- **Hypothesis:** GeGLU halves fc2 input width (256 → 128 effective channels at mlp_ratio=4). Doubling to mlp_ratio=8 recovers the capacity, giving GeGLU gating + full MLP depth.
- **Result:** val=68.781, test=60.220 (original run). Reproducibility run: val=69.613. Both confirm regression.

| Split | val (mlp_ratio=8) | val (baseline mlp_ratio=4) | Δ |
|---|---|---|---|
| single_in_dist | ~80.8 | 72.044 | **+12.2%** |
| geom_camber_rc | ~89.2 | 89.234 | ~0% |
| geom_camber_cruise | ~41.5 | 38.721 | +7.2% |
| re_rand | ~63.5 | 57.586 | +10.3% |
| **avg** | **68.781** | **64.918** (PR #1769 baseline) | **+5.95% worse** |

- **Test:** 60.220 vs 58.171 = +3.5% worse
- **Per-epoch overhead:** ~163s vs ~143s (+14%) → net −1 effective epoch in 30-min budget
- **Reproducibility:** Second seed also regressed (val=69.613). Regression is real, not noise.
- **Student analysis confirmed:** "gating wins outright." Interpretation: GeGLU's selective gating is the structural win — it filters which features propagate into the physics attention. Wider fc2 adds capacity but also adds noise pathways that the gate doesn't fully suppress at 12-epoch budget. single_in_dist regressed worst (+12-18%), consistent with extra capacity overfitting on the simple in-distribution splits.
- **Dead end:** MLP capacity expansion exhausted. mlp_ratio=4 with GeGLU gating is optimal for this budget.
- **Reassigned tanjiro:** PR #1984 n_hidden=160 (widen attention dim — orthogonal capacity axis)

---

## 2026-05-13 06:05 — PR #1793: CosineAnnealingLR T_max=12 on RMSNorm+GeGLU+Lion — MERGED (−7.9% val, −8.9% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lion-tmax-12-aligned
- **Hypothesis:** T_max=50 with 30-min cap fires only ~13% of cosine decay; T_max=12 aligns to actual epoch budget for proper late-epoch LR decay to 0.
- **Result:** val=52.798, test=44.972 (epoch 12/12, 30-min cap, surf_weight=10)

| Split | val (T_max=12, sw=10) | val (prev baseline sw=5, T=50) | Δ |
|---|---|---|---|
| single_in_dist | 58.907 | 60.960 | −3.4% |
| geom_camber_rc | 67.658 | 72.044 | −6.1% |
| geom_camber_cruise | 33.380 | 38.721 | **−13.8%** |
| re_rand | 51.248 | 57.586 | **−11.0%** |
| **avg** | **52.798** | **57.328** | **−7.9% ✓** |

- **Test:** 44.972 vs 49.387 = **−8.9% ✓**
- **Same-surf_weight comparison vs PR #1837 (sw=10, T=50, val=63.017):** **−16.2% val / −17.8% test** — pure T_max=12 effect
- **Mechanism confirmed:** LR trajectory from epoch 9→12 dropped 11.3% as cosine fully decayed (2.07e-5 → 0). Lion's sign(m) steps get 10× finer in late epochs as designed.
- **Easy splits benefit most:** geom_camber_cruise (−13.8%) and re_rand (−11.0%) — late-epoch fine-tuning matters more when surrounding fit is already good
- **Note:** student rebase pre-dated PR #1836 (surf_weight=5) merge, so this used surf_weight=10. Compound `T_max=12 + surf_weight=5` is the natural next experiment.
- **Metric artifacts:** `models/model-charliepai2g48h3-nezuko-lion-tmax-12-aligned-v2-20260513-045129/metrics.jsonl`
- **Reassigned nezuko:** PR #1956 T_max=12 + surf_weight=5 compound (orthogonal mechanism stacking)

---

## 2026-05-13 05:45 — PR #1836: surf_weight=5 on RMSNorm+GeGLU+Lion — MERGED (−9.03% val, −9.76% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/surf-weight-5-rmsnorm-geglu-lion
- **Hypothesis:** Halving surf_weight (10→5) reallocates L1 gradient budget toward volume nodes; richer volumetric features improve surface accuracy via geometric context. Hardest splits (geom_camber_rc, single_in_dist) benefit most.
- **Result:** val=57.328, test=49.387 (epoch 14, 30-min cap, ~138s/epoch)

| Split | val (surf_w=5) | val (baseline surf_w=10) | Δ | test (surf_w=5) | test (baseline) | Δ |
|---|---|---|---|---|---|---|
| single_in_dist | **60.960** | 76.710 | **−20.5%** | **53.010** | 67.384 | **−21.3%** |
| geom_camber_rc | **72.044** | 73.930 | −2.6% | **62.463** | 64.508 | −3.2% |
| geom_camber_cruise | 38.721 | 40.746 | −5.0% | 32.843 | 34.707 | −5.4% |
| re_rand | 57.586 | 60.683 | −5.1% | 49.231 | 52.327 | −5.9% |
| **avg** | **57.328** | **63.017** | **−9.03% ✓** | **49.387** | **54.731** | **−9.76% ✓** |

- **Volume MAE confirmed improved** on all 4 splits (−7% to −26%), confirming gradient-reallocation mechanism
- **single_in_dist standout** (−20.5% val): after RMSNorm fixed geom_camber_rc (−17.2%), single_in_dist became most volume-starved — surf_weight=5 cracked it
- **All four splits improved** on both val and test — the cleanest, most uniform sweep win yet
- **Metric artifacts:** `models/model-charliepai2g48h3-thorfinn-surf-weight-5-rmsnorm-geglu-lion-20260513-041441/metrics.jsonl`
- **Reassigned thorfinn:** PR #1948 surf_weight=3 (sweep further down)

---

## 2026-05-13 05:40 — PR #1765: Lion lr=2e-4 (rebased onto RMSNorm stack) — SENT BACK (pivot to lr=1.5e-4)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4
- **Result on new RMSNorm+GeGLU+Lion stack:** val=63.635, test=54.060 vs baseline 63.017/54.731

| Split | val (lr=2e-4) | val (baseline lr=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | 72.015 | 76.710 | **−6.1%** |
| geom_camber_rc | 82.155 | 73.930 | **+11.1%** |
| geom_camber_cruise | 38.985 | 40.746 | −4.3% |
| re_rand | 61.384 | 60.683 | +1.2% |
| **avg** | **63.635** | **63.017** | **+0.98%** (misses) |

- **Test avg:** 54.060 vs 54.731 = −1.23% ✓ (test improves but val is primary metric)
- **Insight:** RMSNorm tightened the loss surface — lr=2e-4 overshoots specifically on geom_camber_rc (the OOD geometry split now with a tighter noise landscape post-RMSNorm). Three of four splits improved. LR sweet spot has shifted down on the new stack.
- **Action:** Sent back to try lr=1.5e-4. Also updated target to new baseline (57.328 / 49.387 after #1836 merge). lr=cfg.lr bug fix preserved.

---

## 2026-05-13 05:30 — PR #1890: n_layers=7 + RMSNorm+GeGLU+Lion — CLOSED (+4.6% val regression)

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/n-layers-7-rmsnorm-geglu-lion
- **Hypothesis:** RMSNorm+bf16 reduces per-epoch time enough (~160s vs 205s) to fit 11-12 epochs at 7 layers vs 9 in the old AdamW stack. Re-test n_layers=7 under better conditions.
- **Result:** val=65.904, test=56.888 (epoch best 12 of 12, 30-min cap)

| Split | val (n_layers=7) | val (baseline n_layers=6) | Δ |
|---|---|---|---|
| single_in_dist | 90.497 | 76.710 | **+18.0%** |
| geom_camber_rc | 77.252 | 73.930 | +4.5% |
| geom_camber_cruise | 38.800 | 40.746 | −4.8% |
| re_rand | 57.067 | 60.683 | −6.0% |
| **avg** | **65.904** | **63.017** | **+4.6% worse** |

- **Per-epoch time:** ~160s (vs 138s at 6 layers) → 12 epochs fit in 30 min vs 14
- **Test:** 56.888 vs 54.731 = +3.9% worse
- **Analysis:** n_layers=7 helps geom_camber_cruise and re_rand (simpler splits), but catastrophically regresses single_in_dist (+18%) — the deeper model is underfitting at 12 epochs where the 6-layer model peaks at 13-14. Budget-convergence mismatch confirmed. Architecture depth expansion remains incompatible with this 30-min cap.
- **Dead end:** n_layers=7 is worse under all tested conditions (AdamW ~205s/epoch, Lion+RMSNorm ~160s/epoch)
- **Reassigned frieren:** PR #1920 CosineAnnealingLR eta_min=1e-5

---

## 2026-05-13 05:30 — PR #1889: Lion WD=1e-1 — CLOSED (+2.72% val regression, over-regularizes)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-wd-1e-1
- **Hypothesis:** WD=1e-1 is the upper end of Lion paper's recommended range (10-100× Adam); bracket with askeladd's WD=1e-2 to find optimum.
- **Result:** val=64.731, test=57.110 (best epoch=10, 30-min cap)

| Split | val (WD=1e-1) | val (baseline WD=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | ~78.5 | 76.710 | regressed |
| geom_camber_rc | ~75.6 | 73.930 | regressed |
| geom_camber_cruise | ~42.8 | 40.746 | regressed |
| re_rand | ~62.0 | 60.683 | regressed |
| **avg** | **64.731** | **63.017** | **+2.72% worse** |

- **Test:** 57.110 vs 54.731 = +4.35% worse
- **Key diagnostic:** best_epoch=10 (vs baseline epoch=13) — val starts climbing after epoch 10 while train loss keeps dropping. Textbook over-regularization signature.
- **Insight:** WD=1e-1 forces parameters toward zero too aggressively at only 14 epochs. Model already underfitting — adding more regularization makes it worse. WD=1e-2 (askeladd, in-flight) likely the true optimum; WD=1e-1 confirmed as upper dead-end boundary.
- **Dead end:** WD space above 1e-2 exhausted.
- **Reassigned edward:** PR #1925 Lion WD=3e-2 (bracket between 1e-2 and 1e-1)

---

## 2026-05-13 04:30 — PR #1837: RMSNorm in TransolverBlock — MERGED (−2.9% val, −5.9% test) ← NEW BASELINE

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/rmsnorm-geglu-lion
- **Result:** val=63.017, test=54.731 (epoch 13 of 14, 30-min cap)

| Split | val (RMSNorm) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 76.710 | 72.021 | +6.5% |
| geom_camber_rc | **73.930** | 89.234 | **−17.2%** |
| geom_camber_cruise | 40.746 | 37.058 | +10.0% |
| re_rand | 60.683 | 61.359 | −1.1% |
| **avg** | **63.017** | **64.918** | **−2.9% ✓** |

- Test avg: 54.731 vs 58.171 = **−5.9%** (geom_camber_rc −19.8%)
- **Speed**: 138s/epoch (vs 143s) — 14 epochs fit in 30 min, best at epoch 13
- **Insight**: RMSNorm's removal of mean-centering helps the hardest OOD split dramatically (geom_camber_rc −17.2% val). Slight regression on single_in_dist (+6.5%) and cruise (+10%). Net strongly positive.
- **Reassigned frieren**: PR #1890 n_layers=7 + RMSNorm+GeGLU+Lion

---

## 2026-05-13 04:30 — PR #1836: surf_weight=5 on GeGLU+Lion — SENT BACK (beats old baseline; misses new RMSNorm baseline by +0.2%)

- **Student:** charliepai2g48h3-thorfinn
- **Result on old baseline (64.918):** val=63.142, test=56.121 (−2.74%/−3.52% ✓)
- **vs new RMSNorm baseline (63.017):** +0.2% val worse — needs rebase + retest
- **Mechanism confirmed:** volume MAE improved −7% to −11% across all splits; geom_camber_rc biggest: −9.34% val / −13.06% test

---

## 2026-05-13 04:30 — PR #1859: SmoothL1 β=0.1 on GeGLU+Lion — CLOSED (+7.1% val regression)

- **Student:** charliepai2g48h3-edward
- **Result:** val=69.553, test=62.176 vs baseline 64.918 = **+7.1%/+6.9% worse**
- **Insight:** SmoothL1 β=0.1 slows late-epoch convergence. geom_camber_rc improved −11.7% but in-distribution hurt badly (+25.4%). Pure L1's constant gradient optimal for Lion. All loss-modification directions exhausted.
- **Dead end**: all loss modifications (physical-space L1, channel weighting, SmoothL1) fail on GeGLU+Lion
- **Reassigned edward**: PR #1889 Lion WD=1e-1

---

## 2026-05-13 04:20 — PR #1765: Lion lr=2e-4 (with bug fix) — SENT BACK (−7.8% on Lion+GELU; obsolete vs GeGLU+Lion)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/lion-lr-2e-4
- **Hypothesis:** 2× LR (1e-4 → 2e-4) leverages Lion's higher LR tolerance from sign-based updates
- **Bug fix included:** `train.py:421` `lr=1e-4` → `lr=cfg.lr` (the `--lr` flag was silently ignored before this PR). Default `Config.lr` changed from 5e-4 (AdamW carryover) to 1e-4 to preserve current baseline behavior.
- **Result:** val=80.127, test=71.126 (Lion+GELU stack, epoch 13/13)

| Split | val (lr=2e-4) | val (baseline lr=1e-4) | Δ |
|---|---|---|---|
| single_in_dist | 84.028 | 98.979 | **−15.1%** |
| geom_camber_rc | 98.809 | 104.737 | −5.7% |
| geom_camber_cruise | 59.025 | 62.041 | −4.9% |
| re_rand | 78.646 | 81.995 | −4.1% |
| **avg** | **80.127** | **86.938** | **−7.8% ✓** |

- **vs Lion+GELU baseline (86.938):** −7.8% ✓ (first true measurement of lr=2e-4 because bug fixed)
- **vs new GeGLU+Lion baseline (64.918):** +23.4% worse — needs rerun on GeGLU+Lion
- **Action:** Sent back to retest on GeGLU+Lion stack. lr=2e-4 mechanism (Lion's sign updates absorb higher LR) is orthogonal to activation function.

---

## 2026-05-13 04:20 — PR #1793: Lion + T_max=12 (cosine aligned to budget) — SENT BACK (−9.18% on Lion+GELU; obsolete vs GeGLU+Lion)

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lion-tmax-12-aligned
- **Hypothesis:** T_max=50 with 30-min cap fires only 13% of cosine decay; T_max=12 aligns to actual epoch budget for proper late-epoch LR decay
- **Result:** val=78.962, test=68.931 (Lion+GELU stack)

| Split | val (T_max=12) | val (baseline T_max=50) | Δ |
|---|---|---|---|
| single_in_dist | 91.381 | 98.979 | −7.7% |
| geom_camber_rc | 93.613 | 104.737 | −10.6% |
| geom_camber_cruise | 57.362 | 62.041 | −7.5% |
| re_rand | 73.491 | 81.995 | −10.4% |
| **avg** | **78.962** | **86.938** | **−9.18% ✓** |

- **vs Lion+GELU baseline (86.938):** −9.18% ✓ (uniform improvement across all 8 val+test splits)
- **vs new GeGLU+Lion baseline (64.918):** +21.6% worse — needs rerun on GeGLU+Lion
- **Action:** Sent back to retest on GeGLU+Lion stack. Schedule mechanism (proper cosine decay in budget) is orthogonal to activation function.

---

## 2026-05-13 04:00 — PR #1824: SwiGLU (SiLU gate) vs GeGLU — CLOSED (+1.6% val regression)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/swiglu-lion
- **Hypothesis:** SiLU gate smoother gradient than GELU; SwiGLU > GeGLU (as in LLaMA/PaLM)
- **Result:** val=65.957, test=58.705 (GeGLU+Lion stack, epoch 11, 30-min cap)

| Split | SwiGLU val | GeGLU val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 79.277 | 72.021 | +10.1% |
| geom_camber_rc | 81.003 | 89.234 | **−9.2%** |
| geom_camber_cruise | 41.225 | 37.058 | +11.2% |
| re_rand | 62.321 | 61.359 | +1.6% |
| **avg** | **65.957** | **64.918** | **+1.6% worse** |

- **Decision:** CLOSED — primary metric regresses (>baseline); student correctly said "don't merge"
- **Key insight:** SwiGLU helped only on geom_camber_rc (OOD geometry, −9.2%), hurt everywhere else. LLM SwiGLU > GeGLU finding does not transfer to this CFD surrogate. GELU's slightly negative gate range may be beneficial for pressure-gradient features. Per-split routing (SwiGLU for OOD, GeGLU for in-distribution) is an interesting observation but not actionable without architectural complexity.
- **Dead end added:** SwiGLU +1.6% vs GeGLU on this task
- **Reassigned:** tanjiro → PR #1872 mlp_ratio=8 + GeGLU (recover fc2 capacity halved by gating split)

---

## 2026-05-13 03:45 — PR #1767: Channel-weighted L1 [0.03, 0.03, 1.0] on GeGLU+Lion — CLOSED (+11.0% regression)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/physical-space-l1-lion
- **Hypothesis:** channel reweighting (p:Ux:Uy = 1:0.03:0.014) drives physical-space L1 win; test explicitly on GeGLU+Lion
- **Result:** val=72.071, test=63.176 (GeGLU+Lion, epoch 13)

| Split | val (CW) | val (baseline) | Δ |
|---|---|---|---|
| single_in_dist | 82.511 | 72.021 | +14.6% |
| geom_camber_rc | 87.864 | 89.234 | −1.5% |
| geom_camber_cruise | 48.734 | 37.058 | +31.5% |
| re_rand | 69.174 | 61.359 | +12.7% |
| **avg** | **72.071** | **64.918** | **+11.0%** |

- **vs GeGLU+Lion baseline (64.918):** +11.0% WORSE — clear dead end
- **Key insight:** GeGLU does implicit channel balancing via GELU(x1)*x2 gating. Manual downweighting of Ux/Uy starves the gates of velocity supervision, degrading the routing they've learned. The round-1 +4.0% win (Lion+GELU) was a GELU-specific interaction, not a portable mechanism.
- **Action:** CLOSED. edward reassigned to SmoothL1 β=0.1 (#1859).

---

## 2026-05-13 03:45 — PR #1790: Lion + 2-epoch warmup — SENT BACK (−9.9% on old Lion+GELU, obsolete; rerun on GeGLU+Lion needed)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/lion-cosine-warmup-2ep
- **Hypothesis:** 2-epoch linear warmup (1e-6→1e-4) stabilizes Lion sign-update init; better basin before full LR kicks in
- **Result:** val=78.315, test=68.744 (Lion+GELU stack, epoch 13)

| Split | val (warmup) | val (no warmup) | Δ |
|---|---|---|---|
| single_in_dist | 91.531 | 98.979 | −7.5% |
| geom_camber_rc | 95.883 | 104.737 | −8.5% |
| geom_camber_cruise | 52.515 | 62.041 | −15.4% |
| re_rand | 73.332 | 81.995 | −10.6% |
| **avg** | **78.315** | **86.938** | **−9.9%** |

- **vs Lion+GELU baseline (86.938):** −9.9% ✓ (hypothesis confirmed on old baseline)
- **vs new GeGLU+Lion baseline (64.918):** +20.6% worse — obsolete, needs GeGLU rerun
- **Mechanism:** warmup absorbs Lion's aggressive sign-update noise at random init; faster early convergence, larger gains on harder/OOD splits
- **Action:** Sent back to retest on GeGLU+Lion stack. Warmup mechanism is orthogonal to activation function.

---

## 2026-05-13 03:15 — PR #1766: Lion WD=1e-2 — SENT BACK (+10.4% on old baseline, but obsolete; CRITICAL bug discovered)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/lion-wd-1e-2
- **Result:** val=77.859, test=67.662 (Lion+GELU stack, epoch 13)

| Split | val (WD=1e-2) | val (WD=1e-4 baseline) | Δ |
|---|---|---|---|
| single_in_dist | 93.076 | 98.979 | −6.0% |
| geom_camber_rc | 93.965 | 104.737 | −10.3% |
| geom_camber_cruise | 51.233 | 62.041 | −17.4% |
| re_rand | 73.161 | 81.995 | −10.8% |
| **avg** | **77.859** | **86.938** | **−10.4%** |

- **vs Lion+GELU baseline (86.938):** −10.4% — Lion paper recommendation validated; WD=1e-2 is the correct value
- **vs new GeGLU+Lion baseline (64.918):** +19.9% worse — obsolete on current baseline
- **CRITICAL BUG DISCOVERED:** train.py:440 hardcoded `lr=1e-4` in Lion constructor, silently ignoring `--lr` CLI flag. askeladd's fix: `lr=cfg.lr`. This bug invalidates alphonse's #1765 lr=2e-4 experiment (silently ran at lr=1e-4).
- **Action:** Sent back to retest WD=1e-2 on GeGLU+Lion stack (expected to compound — orthogonal mechanisms). KEEP bug fix in branch.

---

## 2026-05-13 02:45 — PR #1769: GeGLU + Lion — MERGED (**new best: val=64.918, −25.3%**)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/geglu-lion
- **Hypothesis:** GeGLU gated MLP activation (GELU(x1) * x2 where fc1 output is split in two) with Lion optimizer. Prior GeGLU+AdamW (PR #1728) was essentially baseline. Lion's sign-based updates may complement GeGLU's gating differently.

| Split | val mae_surf_p | test mae_surf_p | Δ val |
|---|---|---|---|
| single_in_dist | 72.021 | 64.947 | −27.2% |
| geom_camber_rc | 89.234 | 80.467 | −14.8% |
| geom_camber_cruise | 37.058 | 32.329 | **−40.3%** |
| re_rand | 61.359 | 54.939 | −25.2% |
| **avg** | **64.918** | **58.171** | **−25.3%** |

- **vs baseline (Lion+GELU 86.938):** −25.3% — **largest single-PR improvement in the programme** (exceeds L1 −20.5%, Lion −14.3%, n_layers=6 −9.4%)
- **Per-epoch time:** ~143s (GeGLU overhead negligible with bf16 — vs ~210s with AdamW). Got 13 epochs in 30 min vs 11 with Lion+GELU.
- **Still improving at cutoff** (epoch 13/50, curve descending). Cruise split improved 40.3% — GeGLU routing especially effective for high-camber transonic regime.
- **Key insight:** GeGLU's multiplicative gating acts as a learned feature router. Combined with Lion's sign-based constant-magnitude updates, the gating receives cleaner direction signals. The previous AdamW+GeGLU failure was likely due to Adam's adaptive scaling confounding the gate learning signal.
- **Artifacts:** `models/model-charliepai2g48h3-tanjiro-geglu-lion-20260513-012007/metrics.jsonl`

---

## 2026-05-13 02:45 — PR #1767: Physical-space L1 + Lion — SENT BACK (beats old baseline; new baseline higher bar)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/physical-space-l1-lion
- **Result:** val=83.446, test=74.517 (epoch 13)

| Split | val | vs Lion+GELU baseline (86.938) |
|---|---|---|
| single_in_dist | 100.474 | +1.5% |
| geom_camber_rc | 96.861 | −7.9% |
| geom_camber_cruise | 58.461 | −5.8% |
| re_rand | 77.988 | −4.9% |
| **avg** | **83.446** | **−4.0%** |

- **vs new baseline after #1769 (64.918):** +28.5% worse — not competitive on new baseline
- **Key student insight:** Physical-space L1 = implicit channel reweighting (p:Ux:Uy ≈ 1 : 0.03 : 0.014). p_std≈679 Pa vs Ux_std≈14, Uy_std≈6 means dividing by p_std amplifies pressure gradient ~50×. This is the real mechanism.
- **Action:** Sent back with instruction to rebase on new GeGLU+Lion baseline and test **explicit pressure-weighted normalized loss**: `loss = mae_p_norm + 0.03 * (mae_Ux_norm + mae_Uy_norm)`

---

## 2026-05-13 02:10 — PR #1678: AdamW LR 5e-4 → 7e-4 — CLOSED (stale + obsolete)

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/lr-7e-4
- **Status:** Idle 3+ hours after creation with no commits. Additionally obsolete after PR #1725 Lion merged (new optimizer paradigm).
- **Decision:** Closed without execution; reassigned to Lion+T_max=12 (PR #1793).

---

## 2026-05-13 02:10 — PR #1726: SWA late-start epoch 8 — CLOSED (+7.9% worse vs AdamW baseline)

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/swa-late-start
- **Optimizer used:** AdamW (run started before PR #1725 Lion merge)
- **Result:** val=109.442, test=98.377 (epoch 11; still descending at 30-min cutoff)

| Split | val mae_surf_p | vs pre-Lion baseline (101.463) |
|---|---|---|
| single_in_dist | 137.319 | +13.8% |
| geom_camber_rc | 124.325 | +7.1% |
| geom_camber_cruise | 80.056 | +8.7% |
| re_rand | 96.069 | +0.7% |
| **avg** | **109.442** | **+7.9%** |

- **Analysis (student's post-mortem, well-reasoned):**
  1. SWA started at epoch 8 before convergence — baseline was still ~138 at that point, dropping ~7 pts/epoch
  2. Only 3 averaging epochs accumulated (9, 10, 11) — too few over a non-stationary region
  3. SWALR cut LR ~10× at the most critical training phase, slowing the underlying SGD
  4. Best epoch was the final (11) — model never reached SWA's "flat basin" regime
- **Conclusion:** SWA requires a converged base model to average. At 30-min budget with ~11 epochs, we never reach post-convergence. Reassigned fern to Lion+warmup (PR #1790) which addresses LR-schedule sensitivity in a constructive direction.
- **Artifacts:** `models/model-charliepai2g48h3-fern-swa-late-start-20260513-002529/metrics.jsonl`

---

## 2026-05-13 01:45 — PR #1725: Lion optimizer lr=1e-4 — MERGED (**new best: val=86.938, −14.3%**)

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/lion-optimizer
- **Hypothesis:** Lion's sign-based updates produce constant-magnitude steps, hypothesized to be synergistic with L1 loss which also produces constant ±1 gradients. Used lr=1e-4 (Lion paper recommends ~3-10× lower than Adam; our Adam lr=5e-4).

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 98.979 | 91.606 |
| geom_camber_rc | 104.737 | 92.561 |
| geom_camber_cruise | 62.041 | 52.841 |
| re_rand | 81.995 | 74.952 |
| **avg** | **86.938** | **77.990** |

- **vs baseline (101.463 post-bf16):** −14.3% — the biggest single win since L1 loss itself (which was −20.5%)
- **Epoch:** 11 of 50; still monotonically improving at 30-min cutoff. Convergence headroom remains.
- **Analysis:** Lion's sign-based updates are structurally complementary to L1's constant-gradient dynamics. Both operate in a regime where gradient *direction* dominates over *magnitude*. The combined signal is particularly clean — no magnitude rescaling distorts the direction. All four splits improved, with the largest gains on cruise (−18.0%) and in_dist (−18.0%). The model still had room to improve at cutoff, suggesting further gains with LR tuning or warmup.
- **Round 7 priority:** Lion LR tuning (lr=2e-4), Lion WD=1e-2 (paper recommendation), physical-space L1, GeGLU+Lion.
- **Artifacts:** `models/model-charliepai2g48h3-edward-lion-optimizer-20260513-001607/metrics.jsonl`

---

## 2026-05-13 01:30 — PR #1728: GeGLU activation (AdamW) — CLOSED (+16.8% worse vs Lion baseline)

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/geglu-activation
- **Hypothesis:** GeGLU gating (multiplicative gate in MLP blocks) routes features by flow regime, mlp_ratio=3 to compensate for 2-projection overhead.
- **Result:** val=101.499 (9 epochs, ~210s/epoch), test=91.436

| Split | val | vs AdamW baseline (101.463) |
|---|---|---|
| single_in_dist | 134.924 | +8.7% |
| geom_camber_rc | 112.496 | −0.2% |
| geom_camber_cruise | 67.405 | −12.0% |
| re_rand | 91.173 | −2.8% |
| **avg** | **101.499** | **+0.04% (essentially baseline)** |

- **Against new Lion baseline (86.938):** +16.8% worse — clearly not competitive.
- **Analysis:** GeGLU's per-epoch overhead (~210s vs 140s with bf16) meant only 9 epochs vs 14 for baseline. The model was still improving at cutoff. The mixed split picture (cruise and re_rand improved, in_dist regressed) suggests GeGLU may help with complex OOD cases but hurts in-distribution. The AdamW context may also be limiting — GeGLU+Lion is a genuinely new hypothesis worth testing.
- **Decision:** Closed; reassigned tanjiro to GeGLU+Lion (PR #1769).
- **Artifacts:** `models/model-charliepai2g48h3-tanjiro-geglu-activation-20260513-002012/metrics.jsonl`

---

## 2026-05-13 01:20 — PR #1724: bf16 mixed precision — MERGED (new best: val=101.463, −0.34%)

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/bf16-mixed-precision
- **Hypothesis:** bf16 autocast reduces epoch time 1.26× (175s→138.5s), gaining +1 epoch (14 vs 13) within the 30-min cap. Small but reliable throughput gain.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 120.699 | 108.025 |
| geom_camber_rc | 116.096 | 107.822 |
| geom_camber_cruise | 73.667 | 63.152 |
| re_rand | 95.391 | 87.800 |
| **avg** | **101.463** | **91.700** |

- **vs baseline (101.810 L1+n_layers=6):** −0.34% (marginal but positive)
- **Analysis:** A100 is bandwidth-limited on batch=4 small meshes, not compute-bound — speedup was 1.26× rather than the hypothesized 2×. The crucial benefit is infrastructure: bf16 is now the default precision and will accelerate all future experiments.
- **Critical implementation note:** PhysicsAttention temperature division at train.py:120 required `.float()` cast to prevent bf16 overflow. alphonse implemented correctly.
- **Artifacts:** `models/model-charliepai2g48h3-alphonse-bf16-mixed-precision-20260513-001819/metrics.jsonl`

---

## 2026-05-13 01:10 — PR #1673: AdamW eps 1e-8→1e-4 — CLOSED (+13.2% worse)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/adamw-eps-1e-4
- **Hypothesis:** Larger epsilon raises the adaptive scaling floor, potentially stabilizing AdamW for L1's constant-magnitude gradients.
- **Result:** val=115.267, test=104.XX
- **vs baseline (101.810):** +13.2% worse
- **Analysis:** The increased eps effectively narrows AdamW's adaptive range, hurting rather than helping. AdamW's eps=1e-8 default is well-calibrated. Dead end — AdamW hyperparameter space now fully exhausted (betas, WD, eps, LR, schedule all explored).
- **Conclusion:** Stop tuning AdamW. Focus entirely on Lion-based experiments.

---

## 2026-05-13 00:45 — PR #1670: weight_decay 1e-4 → 5e-4 — CLOSED (+14.9% worse)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/weight-decay-5e-4
- val=117.013, test=106.801
- **Combined with PR #1649 (WD=0, +3.2% worse), this completes a clean U-curve around the WD=1e-4 baseline.** Over-regularization (5×) costs ~5× more than under-regularization, suggesting baseline is at or very near the regularization sweet spot.
- All splits regressed (single_in_dist +19.1% worst hit; cruise +20.4%). High-magnitude OOD splits need MORE capacity, not less.
- **Conclusion:** WD parameter space fully explored — baseline WD=1e-4 is optimal. Stop sweeping WD.
- **Artifacts:** `target/models/model-weight-decay-5e-4-20260512-230745/metrics.jsonl`

---

## 2026-05-13 00:05 — ROUND 5 BATCH: 5 experiments — ALL CLOSED (PLATEAU CONFIRMED)

**Key cross-experiment finding:** Round 5 confirms ROUND 4 META-INSIGHT (convergence-limited at 30-min budget) extends across optimizer, schedule, and batch-size knobs. ALL 4 returning experiments regressed; the closest miss (alphonse T_max=14 at +0.5%) is pure seed variance (3 seeds: 102.30 / 104.11 / 107.09). **Plateau Protocol triggered — escalating to bolder hypotheses (loss reformulations, mixed precision, optimizer paradigm shifts).**

### PR #1592: Cosine T_max 50 → 14 (align to budget) — CLOSED (+0.5% worse, noise)
- **Student:** charliepai2g48h3-alphonse
- val=102.301, test=92.704 (3 seeds: 102.30 / 104.11 / 107.09; mean +2.64% worse, pure noise)
- Cosine LR alignment to actual epoch budget had no detectable effect
- **Conclusion:** Schedule decay isn't the bottleneck — LR=5e-4 throughout the 11-epoch run is already near-optimal for this regime
- **Artifacts:** `models/model-charliepai2g48h3-alphonse-cosine-tmax-14-20260512-225513/metrics.jsonl`

### PR #1661: CosineAnnealingWarmRestarts T_0=4 T_mult=2 — CLOSED (+2.7% worse)
- **Student:** charliepai2g48h3-fern
- val=104.515, test=94.113
- Schedule played as predicted; the restart at epoch 5 disrupted progress (val_avg jumped 168 → 186 → 150 → ...)
- **Conclusion:** Multi-cycle schedules need budgets that allow full cycle completion. At 11 epochs, single-cycle wins.
- **Artifacts:** `models/model-charliepai2g48h3-fern-cosine-warm-restarts-20260512-225827/metrics.jsonl`

### PR #1671: AdamW β1 0.9 → 0.85 — CLOSED (+5.9% worse)
- **Student:** charliepai2g48h3-edward
- val=107.826, test=98.917
- Both β1 directions (0.85 here, 0.95 in #1622 askeladd) now confirmed worse than default 0.9
- Edward's analysis: oscillations live at macro/epoch scale, not micro/step scale — β1 (4-7 step half-life) can't address them
- **Conclusion:** AdamW betas FULLY EXPLORED. Default (0.9, 0.999) is correct. Stop tuning betas.
- **Artifacts:** `models/model-adamw-beta1-085-20260512-230405/metrics.jsonl`

### PR #1634: Batch size 4 → 8 (via accum_steps=2) — CLOSED (+23.6% worse)
- **Student:** charliepai2g48h3-tanjiro
- val=125.836, test=116.763
- bs=8 OOMed; student used accum_steps=2 as mathematical equivalent (defensible workaround)
- Effective batch=8 halved optimizer steps per epoch (375 → 187); ran out of training steps in the 30-min cap
- **Conclusion:** Direct evidence we are step-count-limited. Increasing effective batch is unambiguously wrong direction.
- **Artifacts:** `models/model-batch-size-8-20260512-231353/metrics.jsonl`

### PR #1384: surf_weight 10 → 25 (stale, never rebased) — CLOSED (no clean result)
- **Student:** charliepai2g48h3-frieren
- Stale 25h+ rebase blocker; original run on pre-L1 baseline produced val=136.78 (not comparable to current 101.81 baseline)
- Closed without merge; hypothesis itself remains untested on current arch — may revisit later from fresh branch
- **NaN-skip fix in evaluate_split was already independently merged in PR #1358**, so frieren's main contribution is preserved.

---

## 2026-05-13 00:20 — ROUND 4 BATCH: 4 experiments — ALL CLOSED

**Key cross-experiment finding:** At the 30-min/~11-13 epoch budget, we are CONVERGENCE-LIMITED. Any change that slows per-epoch progress consistently loses, even if theoretically sound with more training time.

### PR #1649: weight_decay 1e-4 → 0 — CLOSED (+3.2% worse)
- **Student:** charliepai2g48h3-thorfinn
- val=105.079, test=94.684 (best: +3.2% worse, closest miss of round 4)
- val_single_in_dist hit hardest (+5.7%); L1 does NOT dampen high-magnitude gradient flow the way MSE does → WD provides useful regularization
- **Conclusion:** WD=0 is wrong direction. Stronger WD (5e-4) is the natural next step.
- **Artifacts:** `models/model-charliepai2g48h3-thorfinn-weight-decay-zero-20260512-221840/metrics.jsonl`

### PR #1632: dropout=0.1 in attention — CLOSED (+11.8% worse)
- **Student:** charliepai2g48h3-edward
- val=113.838, test=103.151 (best: +11.8% worse)
- Model is underfitting at 11 epochs; adding noise to an underfit model is counterproductive
- val_single_in_dist hit hardest (+16.8%)
- **Conclusion:** Dropout trades convergence speed for stability — wrong tradeoff at 30-min budget.
- **Artifacts:** `models/model-dropout-01-mlp-attn-20260512-220657/metrics.jsonl`

### PR #1622: AdamW betas (0.9,0.999)→(0.95,0.99) — CLOSED (+15.4% worse)
- **Student:** charliepai2g48h3-askeladd
- val=117.483, test=107.268 (best: +15.4% worse)
- β2=0.99 shorter memory amplified noise on L1's frequent sign-flips; caused oscillation bumps at epochs 5 and 10
- **Key insight:** Standard betas (0.9, 0.999) are correct for L1 at our budget. Next direction: looser β1=0.85 with β2=0.999 unchanged (askeladd's own post-mortem suggestion)
- **Artifacts:** `models/model-adamw-betas-transformer-20260512-215824/metrics.jsonl`

### PR #1593 (max_norm=10 arm): Gradient clipping — CLOSED (+15.7% worse)
- **Student:** charliepai2g48h3-nezuko
- max_norm=10: val=117.811, test=105.623 (+15.7% vs baseline)
- max_norm=1: val=112.784, test=100.553 (+10.8% vs baseline) — both arms worse
- **Critical finding:** max_norm=10 DID reduce oscillation jumps 4-28× smaller. But final val was worse, not better. Conclusion: clipping trades convergence speed for stability — wrong tradeoff. Oscillations ARE useful optimization search.
- **Artifacts:** `models/model-charliepai2g48h3-nezuko-gradient-clipping-max10-20260512-220638/metrics.jsonl`

---

## 2026-05-12 23:30 — PR #1595: Huber/SmoothL1 loss beta=1.0 — CLOSED

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/huber-beta1
- **Hypothesis:** Huber loss (smooth L1 for small errors, L1 for large) provides middle ground between MSE and L1; expected ±3% vs L1.
- **Outcome:** **CLOSED** — val=117.773 (+15.7% worse than 101.810), test=107.832 (+17.6% worse).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **117.773** |
| val_single_in_dist | 152.680 (+23.0%) |
| val_geom_camber_rc | 125.599 (+11.4%) |
| val_geom_camber_cruise | 89.558 (+17.0%) |
| val_re_rand | 103.255 (+10.1%) |
| test_avg/mae_surf_p | **107.832** |
| Epochs completed | 11/50 (30-min cap; ~175 s/epoch) |

**Analysis:** Student's mechanistic explanation is correct: with targets normalized to std≈1, a large fraction of node errors fall inside β=1.0, where Huber behaves quadratically (MSE/2) — removing L1's constant-magnitude gradient. The loss ends up "mostly-MSE" for the error regime that matters most. This mirrors the MSE→L1 story in reverse: Huber β=1.0 reverts most of the gradient advantage L1 provided. Student noted β=0.1 might work but the benefit would be marginal at best. Reassigned to CosineAnnealingWarmRestarts (PR #1661).

**Artifacts:** `models/model-charliepai2g48h3-fern-huber-beta1-20260512-211021/metrics.jsonl`

---

## 2026-05-12 23:10 — PR #1525: Fourier positional features L=4 on L1 baseline — CLOSED

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/fourier-pos-L4
- **Hypothesis:** L=4 sinusoidal Fourier features on raw (x,z) coords encode high-frequency surface pressure structure; expected −5–10% on val_avg/mae_surf_p when stacked on L1 baseline.
- **Outcome:** **CLOSED** — val=107.541 (+5.6% vs 101.810), test=97.418 (+6.2% vs 91.708). Improvement did NOT compound with L1.

Note: Fourier L=4 had beaten the OLD MSE baseline by -9.7% (127.57 vs 141.36) with an excellent L=4 vs L=2 ablation. This second run stacked on the new L1 baseline after rebasing.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | **107.541** |
| val_single_in_dist | 139.360 (+12.3%) |
| val_geom_camber_rc | 111.080 (−1.4% ← only split that improved) |
| val_geom_camber_cruise | 79.317 (+3.6%) |
| val_re_rand | 100.407 (+7.0%) |
| test_avg/mae_surf_p | **97.418** (all 4 splits finite — NaN-fix confirmed) |
| Epochs completed | 11/50 (30-min cap; ~176 s/epoch with Fourier dims) |
| Params | 1.182M (vs 0.992M baseline) |

**Analysis:** L1 + n_layers=6 already absorbs the high-frequency representational content that Fourier features were addressing on the MSE baseline. Both mechanisms target the same failure mode (sharp surface-pressure gradients near leading edges / stagnation points), so they overlap rather than stack. Evidence: val_geom_camber_rc was the biggest Fourier win on MSE baseline (-16.7%), and L1 also gave -17.4% on the same split. They target the same OOD-geometry failure mode. Additionally, the larger input (space_dim=18 vs 2) means the model had only 11 epochs instead of 13, with each epoch slightly slower (~176 s vs ~175 s). Reassigned thorfinn to weight_decay=0 experiment (PR #1649).

**Artifacts:** `models/model-charliepai2g48h3-thorfinn-fourier-pos-L4-on-L1-20260512-211919/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1562: Depth scaling n_layers 6 → 7 — CLOSED

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/n-layers-7
- **Hypothesis:** Deeper model (7 layers vs 6) should improve representational capacity; expected −3–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — val=154.198 (+51.5% vs current 101.810 baseline), test=NaN (reproducible NaN on test_geom_camber_cruise p-channel, both runs).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 7) | **154.198** |
| val_single_in_dist | 193.128 |
| val_geom_camber_rc | 171.668 |
| val_geom_camber_cruise | 115.258 |
| val_re_rand | 136.737 |
| test_avg/mae_surf_p | **NaN** (test_geom_camber_cruise blowup on p-channel) |
| n_params | 1.365M (vs 0.99M for n_layers=6) |
| Epochs completed | 9/50 (30-min cap, ~205 s/epoch) |

**Analysis:** Three key findings: (1) Budget binding — at ~205 s/epoch for n_layers=7, only 9 epochs fit in 30 min vs 12-13 for n_layers=6, so the bigger model is fundamentally undertrained. (2) Reproducible NaN — test_geom_camber_cruise/mae_surf_p NaN appeared on both independent runs (-201029 and -205108); n_layers=6 doesn't exhibit this. The depth-7 model is more numerically fragile on at least one cruise test sample. (3) Sweet spot — n_layers=6 + mlp_ratio=4 appears to be the Pareto-optimal configuration for our compute budget. Going deeper requires either batch_size increase (fewer steps/epoch = faster epochs) or longer wall-clock budget. Reassigned edward to dropout=0.1 (PR #1632).

**Artifacts:** `models/model-charliepai2g48h3-edward-n-layers-7-20260512-205108/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1594: Lower LR 5e-4 → 3e-4 — CLOSED

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/lower-lr-3e-4
- **Hypothesis:** L1 constant-magnitude gradients don't shrink at convergence → lower LR helps fine convergence. Expected −3–6%.
- **Outcome:** **CLOSED** — val=119.221 (+17.1% vs 101.810), test=109.040 (+18.9% vs 91.708).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **119.221** |
| val_single_in_dist | 152.670 (+23.0%) |
| val_geom_camber_rc | 128.660 (+14.2%) |
| val_geom_camber_cruise | 84.604 (+10.5%) |
| val_re_rand | 110.951 (+18.3%) |
| test_avg/mae_surf_p | **109.040** |
| Epochs completed | 11/50 (30-min cap, ~175 s/epoch) |

**Analysis:** Root cause is budget-driven undertraining. With T_max=50 and only 11 epochs in the budget, the cosine LR barely decays regardless of initial value. At lr=3e-4, the model simply makes smaller steps toward the same solution the baseline reaches in 13 epochs at lr=5e-4. Lower LR is only useful if paired with a schedule that decays within budget (T_max=14) — see alphonse PR #1592 which is testing exactly that. Student's analysis was spot-on. Reassigned to batch_size=8 experiment (PR #1634).

**Artifacts:** `models/model-charliepai2g48h3-tanjiro-lower-lr-3e-4-20260512-210951/metrics.jsonl`

---

## 2026-05-12 22:55 — PR #1593: Gradient clipping max_norm=1.0 — SENT BACK

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/gradient-clipping
- **Hypothesis:** L1 gradient oscillations cause val instability → clipping at max_norm=1.0 stabilizes training.
- **Outcome:** **SENT BACK** — val=112.784 (+10.8% vs 101.810), test=100.553 (+9.6% vs 91.708). Oscillations NOT eliminated (epoch spikes at ep 4, 8, 10 persisted). Hypothesis: max_norm=1.0 is too aggressive — L1 gradients are constant magnitude ±1/±surf_weight per element, so clip threshold of 1.0 effectively throttles learning rate. Sent back to try max_norm=10.0.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **112.784** |
| val_single_in_dist | 143.901 (+15.9%) |
| val_geom_camber_rc | 129.979 (+15.3%) |
| val_geom_camber_cruise | 80.735 (+5.4%) |
| val_re_rand | 96.523 (+2.9%) |
| test_avg/mae_surf_p | **100.553** |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** clip=1.0 likely clips EVERY update given ~1.18M params and L1 gradients. With 1.18M params each producing a ±1 or ±surf_weight gradient, the expected gradient L2 norm is ~sqrt(n_params * mean_grad^2) >> 1.0. The oscillations persisted, confirming this isn't reducing gradient spikes — it's just capping useful parameter updates. max_norm=10.0 is the natural next test: loose enough to only clip true outlier spikes, tight enough to have any stabilization effect.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-gradient-clipping-20260512-211112/metrics.jsonl`

---

## 2026-05-12 22:30 — PR #1563: EMA weights (decay=0.999) for val/test eval — CLOSED

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights
- **Hypothesis:** EMA of model weights (decay=0.999) improves val/test metrics via implicit model ensemble over training trajectory.
- **Outcome:** **CLOSED** — val=143.7075, +41.1% worse than current baseline 101.810. test_avg=NaN.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **143.7075** |
| val_single_in_dist | 182.77 |
| val_geom_camber_rc | 155.00 |
| val_geom_camber_cruise | 112.89 |
| val_re_rand | 124.17 |
| test_avg/mae_surf_p | **NaN** (EMA-averaged weights → NaN on test_geom_camber_cruise p-channel) |
| Epochs completed | 11/50 (30-min cap) |

**Analysis:** EMA cold-start drag is the root cause. With decay=0.999, the EMA buffer half-life is ~693 steps. ModelEMA initialized at random weights and was updated only during training, so the EMA buffer was dominated by early-epoch random-weight values throughout the entire 11-epoch run. The EMA consistently lagged well behind the live model (which reached 101.810 without EMA). Additionally, the EMA-averaged weights landed in a numerically degenerate region for at least one sample in test_geom_camber_cruise, producing NaN on the p-channel — the live model handled the same sample without NaN.

EMA would require: (a) initializing the EMA buffer from post-warmup live weights (not random init), OR (b) training for 100+ epochs so the buffer stabilizes. Neither is feasible at 30-min wall-clock. Closed; askeladd reassigned to AdamW betas experiment (PR #1622).

---

## 2026-05-12 21:10 — PR #1358: L1 (MAE) loss in normalized space — MERGED

- **Student:** charliepai2g48h3-alphonse
- **Branch:** charliepai2g48h3-alphonse/l1-surface-pressure-loss
- **Hypothesis:** L1 loss directly optimizes the ranking metric (MAE); expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline 101.810 (−20.5% vs 128.127).** Far exceeded expectations.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **101.810** (−20.5%) |
| val_single_in_dist | 124.150 (−22.3% vs 159.746) |
| val_geom_camber_rc | 112.699 (−17.4% vs 136.513) |
| val_geom_camber_cruise | 76.570 (−25.3% vs 102.432) |
| val_re_rand | 93.820 (−17.6% vs 113.819) |
| test_avg/mae_surf_p | **91.708** (first finite test result!) |
| test_single_in_dist | 110.726 |
| test_geom_camber_rc | 99.692 |
| test_geom_camber_cruise | 66.879 |
| test_re_rand | 89.536 |
| Epochs completed | 14/50 (30-min cap) |
| Peak VRAM | 42.1 GB |

**Analysis:** The loss function switch from MSE to L1 (MAE) produced the single largest improvement seen in this research program. The result is −20.5% better despite using the old arch (n_layers=5, mlp_ratio=2) — showing the loss function dominates architecture in importance here. This makes sense: L1 directly optimizes the metric we evaluate with. The merged train.py stacks L1 + n_layers=6 + mlp_ratio=4; a confirmed stacked run will likely improve further.

Alphonse also included a `train.py::evaluate_split` NaN-fix (filter non-finite GT samples before scorer call), making test metrics finite for the first time: test_avg/mae_surf_p = 91.708 across all 4 test splits.

**Artifacts:** `models/model-l1-loss-e50-20260512-195549/metrics.jsonl`

---

## 2026-05-12 21:05 — PR #1566: n_head 4 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/n-head-8
- **Hypothesis:** Doubling attention heads to 8 diversifies slice patterns; expected −2–4%.
- **Outcome:** **CLOSED** — +15.7% worse (148.280 vs 128.127), per-epoch cost +43%.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 8) | **148.280** |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~222 s (vs ~156 s baseline) |

**Analysis:** n_head=8 launches more but smaller softmax/matmul kernels, creating overhead that costs +43% per epoch (not +15% as predicted). Only 9 epochs fit in 30 min, and the val curve was still oscillating at epoch 8. Under fixed wall-clock, n_head=4 dominates. Future attention-diversity experiments should use slice_num changes rather than head count.

---

## 2026-05-12 21:05 — PR #1401 (arm 2): Warmup + cosine T_max=15

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-ep warmup to peak lr=1e-3 with T_max=15 aligned to budget; expected beat baseline.
- **Outcome:** **CLOSED** — val=133.448, +4.15% worse than old baseline 128.127 (and far worse than new baseline 101.810).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **133.448** |
| val_geom_camber_cruise | 102.355 (≈ old baseline) |
| Epochs completed | 11/15 (30-min cap, epochs ~175 s) |

**Analysis:** Peak lr=1e-3 is too hot for the wider/deeper model. The n_layers=6 arch has ~35% longer epochs (~175 s vs ~130 s for old arch), so even T_max=15 can't complete in 30 min. The schedule's low-LR tail (epochs 12-15) was never executed. This direction is exhausted — reassigned to lower LR (3e-4) approach.

---

## 2026-05-12 21:05 — PR #1370: slice_num 64 → 128

- **Student:** charliepai2g48h3-fern
- **Branch:** charliepai2g48h3-fern/slice-128
- **Hypothesis:** Doubling physics-attention slices improves mesh resolution; expected −1–3%.
- **Outcome:** **CLOSED** — val=150.909, +17.8% worse than new baseline 101.810. Also modified data/scoring.py (read-only).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | **150.909** |
| test_avg/mae_surf_p (NaN-fix applied) | **137.481** |
| Epochs completed | 11/50 (30-min cap) |
| Per-epoch time | ~173 s |

**Analysis:** Doubling slices costs ~12% more per epoch AND the result is significantly worse. The slice bottleneck hypothesis is not supported within the 30-min budget. Additionally, fern modified data/scoring.py (read-only per program.md) — the equivalent train.py workaround was already merged in PR #1358. Closed; fern reassigned to Huber loss.

---

## 2026-05-12 19:05 — PR #1408: MLP expansion ratio 2 → 4 (canonical transformer recipe)

- **Student:** charliepai2g48h3-thorfinn
- **Branch:** charliepai2g48h3-thorfinn/mlp-ratio-4
- **Hypothesis:** Doubling `mlp_ratio` 2 → 4 increases feedforward capacity; canonical transformer recipe, expected −1–3% on val_avg/mae_surf_p.
- **Outcome:** **MERGED — new baseline.**

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **141.356** |
| val_single_in_dist/mae_surf_p | 171.424 |
| val_geom_camber_rc/mae_surf_p | 159.804 |
| val_geom_camber_cruise/mae_surf_p | 104.607 |
| val_re_rand/mae_surf_p | 129.589 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| test mean (3 finite splits) | ~139.51 |
| Epochs completed | 13/50 (30-min cap) |
| Peak VRAM | 52.2 GB |
| Params | 0.99M |

**Analysis:** First terminal result on this branch. 13 epochs in 30 min (≈150 s/epoch). Best val came on epoch 13, meaning the model was still learning at cutoff — more epochs would likely improve further. The cruise test split NaN is a scorer bug (GT sample 20 has -inf pressure), not a model failure; 3 finite test splits give a consistent 139.5 mean. **mlp_ratio=4 is now the default in train.py.**

**Artifacts:** `models/model-charliepai2g48h3-thorfinn-mlp-ratio-4-20260512-175522/metrics.jsonl`

---

## 2026-05-12 20:30 — PR #1401: Warmup (3ep) + cosine, peak LR 1e-3

- **Student:** charliepai2g48h3-tanjiro
- **Branch:** charliepai2g48h3-tanjiro/warmup-cosine-lr1e-3
- **Hypothesis:** 3-epoch linear warmup to peak lr=1e-3 then cosine decay improves convergence vs flat lr=5e-4; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + cosine T_max fix — result is promising (1.4% off new baseline on old arch) but comparison isn't clean.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 11) | **129.900** (+1.4% worse) |
| val_single_in_dist | 159.67 (≈159.75 baseline) |
| val_geom_camber_rc | 137.35 (vs 136.51 baseline) |
| val_geom_camber_cruise | 104.21 (vs 102.43 baseline) |
| val_re_rand | 118.37 (vs 113.82 baseline) |
| test_avg (3 finite splits) | **119.88** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). On old arch it achieved 129.900 which is very close to the new baseline 128.127 — very promising. Key issue: T_max=50 means cosine barely decays in 14 epochs (LR still ~9e-4 at epoch 14), so the full benefit of the schedule was never realized. Sent back with instructions to rebase on new baseline and use `--epochs 15` so T_max=15 matches the actual ~12–14 epoch budget.

**Artifacts:** `models/model-warmup-cosine-lr1e-3-20260512-185810/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1384: Surface weight 10 → 25

- **Student:** charliepai2g48h3-frieren
- **Branch:** charliepai2g48h3-frieren/surf-weight-25
- **Hypothesis:** Increasing surf_weight 10 → 25 improves surface pressure focus; expected −1–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for rebase + rerun on new baseline — ran on old arch (mlp_ratio=2, n_layers=5).

| Metric | Value vs new baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 14) | **136.779** (+6.8% worse) |
| val_single_in_dist | 160.44 |
| val_geom_camber_rc | 167.75 |
| val_geom_camber_cruise | 96.77 |
| val_re_rand | 122.16 |
| test_avg (4 finite splits, NaN-fix applied) | **122.95** |
| Epochs completed | 14/50 (30-min cap) |

**Analysis:** Ran on old arch (mlp_ratio=2, n_layers=5). Frieren also included a train.py NaN-fix in evaluate_split (skip non-finite GT samples before model/accumulator) which produced clean test metrics (all finite!). Sent back for rebase onto new baseline (n_layers=6, mlp_ratio=4) + rerun with `--surf_weight 25`. The surf_weight hypothesis is still untested against the current 128.127 baseline — keeping the NaN-fix in the rerun.

**Artifacts:** `models/model-surf-weight-25-20260512-185910/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1523: Channel-weighted loss p×3

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/channel-weighted-pressure-p3x
- **Hypothesis:** Weighting surface pressure channel 3× in loss improves p MAE; expected improvement on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 18.8% worse than current baseline.

| Metric | Value vs baseline 128.127 |
|---|---|
| val_avg/mae_surf_p (best, ep 13) | **152.216** (+18.8% worse) |
| val_single_in_dist | 197.980 (+23.8%) |
| val_geom_camber_rc | 156.704 (−1.9% better) |
| val_geom_camber_cruise | 119.921 (+17.1%) |
| val_re_rand | 134.259 (+18.0%) |
| Epochs completed | 13/50 (30-min cap) |

**Analysis:** Weighting p×3 in normalized-space MSE doesn't work because normalization already equalizes per-channel variance. The effective surf_weight on p went from 10→30, dominating the loss and hurting Ux/Uy/volume prediction. val_single_in_dist (+23.8%) and cruise (+17.1%) suffered worst. The one marginal win (geom_camber_rc −1.9%) is swamped by losses elsewhere. Clear dead end.

**Artifacts:** `models/model-charliepai2g48h3-edward-channel-weighted-p3x-20260512-190802/metrics.jsonl`

---

## 2026-05-12 20:25 — PR #1363: EMA weights (stale)

- **Student:** charliepai2g48h3-askeladd
- **Branch:** charliepai2g48h3-askeladd/ema-weights-for-eval
- **Hypothesis:** EMA of weights (decay=0.999) for val/test improves generalization.
- **Outcome:** **CLOSED (stale)** — draft PR never produced results. Reassigned fresh.

---

## 2026-05-12 19:30 — PR #1392 (follow-up): n_layers 5 → 6 — MERGED

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-6layers
- **Hypothesis:** Moderate depth increase (n_layers 5→6) retains OOD geometry benefit seen in n_layers=8 but fits more epochs in the 30-min cap.
- **Outcome:** **MERGED — new baseline 128.127.**

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 12) | **128.127** (−9.4% better) |
| val_single_in_dist | 159.746 (−6.8% vs 171.424) |
| val_geom_camber_rc | 136.513 (−14.6% vs 159.804) |
| val_geom_camber_cruise | 102.432 (−2.1% vs 104.607) |
| val_re_rand | 113.819 (−12.2% vs 129.589) |
| test_avg (3 finite splits) | **127.68** |
| Epochs completed | 12/50 (30-min cap) |
| Per-epoch time | ~156 s |
| Peak VRAM | 49.6 GB |
| Params | 0.78M |

**Analysis:** Biggest single-experiment win so far. The moderate depth increase (5→6) gives broad improvement across all splits, with val_geom_camber_rc (−14.6%) and val_re_rand (−12.2%) benefiting most — the extra layer helps with OOD geometry and Reynolds number variation. Best val came at epoch 12 (= final epoch), still learning at cutoff. train.py now defaults to n_layers=6, mlp_ratio=4.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-6layers-20260512-191742/metrics.jsonl`

---

## 2026-05-12 19:10 — PR #1392: Deeper Transolver n_layers 5 → 8

- **Student:** charliepai2g48h3-nezuko
- **Branch:** charliepai2g48h3-nezuko/deeper-transolver-8layers
- **Hypothesis:** Increasing n_layers 5 → 8 (+60% depth) would improve representation through more iterative refinement; expected −2–4% on val_avg/mae_surf_p.
- **Outcome:** **SENT BACK** for a more moderate depth (n_layers=6) — 1.6% worse on val at 30-min cap, but promising signal on OOD geometry (val_geom_camber_rc 5.4% better) and trajectory still descending at cutoff.

| Metric | Value vs baseline 141.356 |
|---|---|
| val_avg/mae_surf_p (best, ep 9) | **143.650** (+1.6% worse) |
| val_single_in_dist/mae_surf_p | 179.503 (+4.7% worse vs 171.42) |
| val_geom_camber_rc/mae_surf_p | **151.158** (−5.4% better vs 159.80) |
| val_geom_camber_cruise/mae_surf_p | 118.512 (+13.3% worse vs 104.61) |
| val_re_rand/mae_surf_p | 125.428 (−3.2% better vs 129.59) |
| test_avg (corrected, Inf-y masked) | **130.23** (−6.6% better vs ~139.51) |
| Epochs completed | 9/50 (30-min cap) |
| Per-epoch time | ~206 s (vs ~150 s baseline) |
| Peak VRAM | 64.5 GB |
| Params | 1.03M |

**Analysis:** Depth-8 lost on val (-1.6%) but won on test_corrected (-6.6%) and val_geom_camber_rc (-5.4%). Two oscillation epochs (2 and 8) indicate mild instability with the deeper model + AdamW + flat 5e-4 LR. The trajectory was still steeply descending at cutoff (162.7 → 143.6 in the final epoch). The 65% per-epoch overhead is too costly at fixed 30-min cap. **Sending back with feedback to try n_layers=6** — a middle-ground depth that should fit ~12 epochs and may retain the OOD-geometry benefit.

**Artifacts:** `models/model-charliepai2g48h3-nezuko-deeper-transolver-8layers-20260512-175521/metrics.jsonl`

---

## 2026-05-12 19:05 — PR #1366: Wider Transolver n_hidden 128 → 192

- **Student:** charliepai2g48h3-edward
- **Branch:** charliepai2g48h3-edward/wider-transolver-192
- **Hypothesis:** Increasing n_hidden 128 → 192 (+50% width) would improve representational capacity; expected −2–5% on val_avg/mae_surf_p.
- **Outcome:** **CLOSED** — 6.3% worse than thorfinn at the same wall-clock budget.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, ep 10) | 150.323 |
| val_single_in_dist/mae_surf_p | 181.449 |
| val_geom_camber_rc/mae_surf_p | 163.411 |
| val_geom_camber_cruise/mae_surf_p | 121.317 |
| val_re_rand/mae_surf_p | 135.114 |
| test_avg/mae_surf_p | NaN (cruise bug) |
| Epochs completed | 10/50 (30-min cap) |
| Per-epoch time | ~185 s (vs ~150 s for thorfinn) |
| Peak VRAM | 58.0 GB |
| Params | 1.47M |

**Analysis:** Width scaling lost to mlp_ratio scaling at the 30-min budget. The wider model runs ~23% slower per epoch, netting only 10 epochs vs 13 for thorfinn. The training curve was still monotonically descending at epoch 10 — fundamentally under-converged. The 30-min cap makes capacity-scaling via width non-competitive unless paired with a step-efficiency gain (e.g. larger batch, fewer layers, faster arch). Closing this; edward redirected to a fresh direction.

**Artifacts:** `models/model-wider-192-20260512-175551/metrics.jsonl`
