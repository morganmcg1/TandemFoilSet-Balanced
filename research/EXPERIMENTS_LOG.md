# SENPAI Research Results

_Branch: icml-appendix-charlie-pai2i-48h-r3_
_Primary metric: val_avg/mae_surf_p (lower is better)_

---

## Round 1 — PRs assigned 2026-05-15

Eight first-round experiments assigned simultaneously to probe the primary bottlenecks in the Transolver baseline.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3154 | alphonse | H5: n_hidden 128→256 | WIP |
| #3156 | askeladd | H1: p-channel surf loss x3, x5 | WIP |
| #3158 | edward | H2: EMA decay=0.999 | WIP |
| #3160 | fern | H4: Huber loss delta=1.0, 0.5 | WIP |
| #3163 | frieren | H3: Grad clip max_norm=1.0 + 5-ep warmup | WIP |
| #3166 | nezuko | H7: FiLM Re/AoA conditioning | WIP |
| #3168 | tanjiro | H10: slice_num 64→128, 96 | WIP |
| #3170 | thorfinn | H11: n_layers 5→7, 5→8 | WIP |

Results will be appended as PRs complete.

---

## 2026-05-15 14:10 — PR #3166: H7: FiLM Re/AoA conditioning on Transolver blocks

- Branch: `charliepai2i48h3-nezuko/h7-film-re-aoa-conditioning`
- Hypothesis: FiLM conditioning extracts global flow params (log(Re), AoA, NACA, gap, stagger = 11 dims) from input and applies per-layer (γ, β) affine modulation on each Transolver block's post-attention output. Identity-initialized ConditionMLP. Implementation: `cond = x[:, 0, 13:]` (node 0, always real — avoids padding dilution of mean).

| Split | val_avg/mae_surf_p |
|---|---|
| val_single_in_dist | 129.7991 |
| val_geom_camber_rc | 129.0683 |
| val_geom_camber_cruise | 94.8233 |
| val_re_rand | 104.8163 |
| **val_avg** | **114.6268** |
| test_single_in_dist | 111.0206 |
| test_geom_camber_rc | 118.4236 |
| test_geom_camber_cruise | NaN (⚠ scoring bug) |
| test_re_rand | 104.0226 |
| **test_avg (3-split)** | **111.155** |

- Artifacts: `models/model-charliepai2i48h3-nezuko-h7-film-re-aoa-cond-20260515-131301/metrics.jsonl`
- Params: 835K (n_hidden=128, 5 layers, cond_dim=11)
- Peak memory: 13.0 GB / 96 GB H100
- Best epoch: 14 / 50 (cut by 30-min timeout; curve still improving at cutoff)

**Analysis:** Training cut at epoch 14/50 — model not converged. Val was still dropping rapidly (138.7 → 117.3 → 114.6 over last 3 epochs). Result is underestimate of converged FiLM performance. val_re_rand (104.82) is the lowest split, consistent with Re-conditioning hypothesis. val_geom_camber_rc (129.07) hardest split — Re conditioning may not help geometry generalization as much.

**Key findings:**
1. FiLM identity-init works well — no early instability despite the added conditioning.
2. Cosine T_max=50 with only 14 epochs means LR stayed near 4e-4 throughout — no annealing benefit. This affects ALL Round 1 experiments equally.
3. `data/scoring.py` has NaN propagation bug: sample index 20 of test_geom_camber_cruise has non-finite GT, and `nan * 0 = nan` in masked sum. All val metrics unaffected.

**Status: MERGED** — establishes first val baseline at 114.63.

**Follow-up for R2:** Run clean cond_dim=0 baseline to isolate FiLM's true contribution. Fix T_max to match actual training duration for all students.

---

## 2026-05-15 15:30 — PR #3168: H10: More Transolver slices (64→128) for finer flow representation

- Branch: `charliepai2i48h3-tanjiro/h10-more-slices-128`
- Hypothesis: Increasing slice_num from 64 to 128 (or 96) would give the model more representational bandwidth to capture finer pressure gradients along foil surfaces.
- Artifacts: `models/model-charliepai2i48h3-tanjiro-h10-slices-128-20260515-130242/metrics.jsonl`, `models/model-h10-slices-96-20260515-140030/metrics.jsonl`

| Arm | slice_num | val_avg/mae_surf_p | test_avg (NaN-safe) | Best epoch | Peak VRAM |
|---|---|---|---|---|---|
| Arm 1 | 128 | 151.62 | 142.60 | 9 | 54.5 GB |
| Arm 2 (best) | 96 | 149.27 | 137.35 | 12 | 47.6 GB |

Per-split (best arm, slice_num=96):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 189.22 |
| val_geom_camber_rc | 153.69 |
| val_geom_camber_cruise | 114.65 |
| val_re_rand | 139.51 |
| **val_avg** | **149.27** |

**Analysis:** Both arms significantly worse than FiLM baseline (114.63). However, this is the first **clean unmodified Transolver** result on this branch — the current best includes FiLM conditioning, and 149.27 at slice_num=96 is now our true no-conditioning reference. FiLM conditioning accounts for roughly -35 points (149 → 114) — a very large effect. The "more slices = better" hypothesis did not hold: slice_num=128 is actually slightly worse than 96, and the sweet spot may be at or below 96. Per-split split: 128 is better on val_geom_camber_cruise (large meshes, 108 vs 115) but much worse on val_geom_camber_rc (185 vs 154), showing a mesh-size-dependent slice tradeoff.

Key student observation: scoring.py NaN bug confirmed and well-documented; student used `recompute_test.py` workaround successfully. `torch.where` fix proposed but not authorized (read-only file).

**Status: CLOSED** — not merge-eligible (30% regression vs baseline). Useful clean-baseline data point.

**Follow-up:** Assign tanjiro H13 (surface dual-head).

---

## 2026-05-15 15:26 — PR #3154: H5: Wider Transolver (n_hidden 128→256, n_head 4→8) — Sent back

- Branch: `charliepai2i48h3-alphonse/h5-wider-hidden-256`
- Status: Sent back to student. Insufficient epoch budget — only 7/50 epochs completed (4.4 min/epoch × 7 = 30 min). val_avg/mae_surf_p = 157.25 at epoch 7.
- Analysis: The comparison is invalid because the baseline ran 14 epochs vs 7 for wider model. Hypothesis not disproven — undertested. Actual param count 2.54M (not ~5.5M as estimated), peak VRAM 83.89 GB (not 18 GB as estimated).
- Action: Requested matched-budget paired comparison — Arm A (standard, T_max=14, 14 epochs) vs Arm B (wider, T_max=7, 7 epochs), both without FiLM.

---

## 2026-05-15 15:24 — PR #3158: H2: EMA weight averaging (decay=0.999) — Sent back

- Branch: `charliepai2i48h3-edward/h2-ema-weight-averaging`
- Status: Sent back to student. val_avg/mae_surf_p = 122.52 at epoch 14 — worse than baseline 114.63.
- Analysis: No paired control (no-EMA run) so EMA contribution cannot be isolated. EMA benefit requires longer training to accumulate; at 14 epochs the shadow model lag may actually hurt. T_max=50 mismatch present.
- Action: Requested Arm A (plain baseline, T_max=14) + Arm B (EMA-only no FiLM, T_max=14) paired comparison.

---

## 2026-05-15 15:35 — PR #3160: H4: Huber loss (δ=1.0, 0.5) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-fern/h4-huber-loss` (predates FiLM merge — Huber alone, no FiLM)
- Hypothesis: Huber loss reduces extreme-Re gradient dominance by linearizing the right tail of normalized errors.

| Arm | δ | val_avg/mae_surf_p | test_avg (3-split) |
|---|---|---|---|
| 1 | 1.0 | 115.99 | 114.02 |
| 2 (best) | **0.5** | **112.84** | **113.44** |

Per-split (δ=0.5, best):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 144.92 |
| val_geom_camber_rc | 125.53 |
| val_geom_camber_cruise | 81.82 |
| val_re_rand | 99.10 |
| **val_avg** | **112.84** |

- Artifacts: `models/model-h4-huber-delta-0.5-20260515-135951/metrics.jsonl`, `models/model-h4-huber-delta-1.0-20260515-130252/metrics.jsonl`
- Best epoch: 14/50 (timeout-capped, not converged)
- Peak VRAM: 42.12 GB

**Analysis:** δ=0.5 beats FiLM baseline (114.63) by 1.79 points despite NO FiLM. δ=0.5 wins on 3/4 splits, losing only on val_geom_camber_rc (raceCar M=6-8 OOD). Mechanism plausible: tighter Huber threshold linearizes more of the right tail, damping extreme-Re gradient dominance. The improvement is mild (-1.6%) relative to the -3% to -8% hypothesis target, but real. Student suggested δ=0.25 to test if monotone trend continues.

**Status: MERGED — new best at 112.84.** Sets new baseline. Merged train.py now contains BOTH FiLM and Huber but their compound has not been tested.

**Follow-up:** Assign fern to test FiLM + Huber compound (the merged code's effective config).

---

## 2026-05-15 16:20 — PR #3284: H12: Clean baseline + corrected cosine T_max=15 vs T_max=50 (no FiLM) — CLOSED

- Branch: `charliepai2i48h3-nezuko/h12-baseline-corrected-tmax`
- Hypothesis: The CosineAnnealingLR(T_max=50) used in all Round 1 runs never anneals because only ~14 epochs fit in the 30-min wall-clock cap. Setting T_max=15 should let the schedule actually decay within budget. Both arms use cond_dim=0 (FiLM off) to establish a clean baseline.

| Arm | Schedule | val_avg/mae_surf_p | test_avg_3split | Best epoch |
|-----|----------|-------------------|------------------|------------|
| A | T_max=50, no FiLM | 125.90 | 128.43 | 14 |
| B (best) | **T_max=15, no FiLM** | **114.19** | **111.97** | **14** |

- Artifacts on student branch metrics JSONL files.
- Current baseline (PR #3160 Huber δ=0.5): 112.84.

**Analysis:** Arm B does NOT beat baseline (114.19 vs 112.84). But the T_max=15 fix is large and important: 11.7 points improvement over T_max=50 at identical capacity / wall-clock. The previously claimed FiLM benefit (149.27 → 114.63, ~35 points) is reframed: only ~0.4 points of that gap survives once the schedule is fixed. **FiLM's apparent benefit in #3166 was largely masking a broken LR schedule.**

Key findings to propagate:
1. T_max=15 (matching actual ~14 achievable epochs in 30-min budget) is materially better than T_max=50.
2. FiLM-vs-bare gap collapses to ~0.4 once schedule is fixed.
3. The 4th cell (FiLM + T_max=15) is now low priority. The more useful compound is **Huber δ=0.5 + T_max=15**.

**Status: CLOSED — negative on primary metric, but high-value diagnostic.** T_max=15 fix becomes standard recommendation for all future assignments.

---

## 2026-05-15 16:20 — PR #3170: H11: Deeper Transolver (n_layers 5→7, 5→8) — CLOSED

- Branch: `charliepai2i48h3-thorfinn/h11-deeper-model-7-layers`
- Hypothesis: Adding Transolver layers (5→7, 5→8) should increase modeling capacity for complex tandem-foil interactions.

| Arm | n_layers | val_avg/mae_surf_p | sec/epoch | Epochs in 30 min |
|-----|----------|-------------------|-----------|-------------------|
| A | 7 | 153.73 | ~181 | ~9-10 |
| B | 8 | 162.48 | ~203 | ~9 |

Best 3-split test (n_layers=7): 153.28.

**Analysis:** Both arms ~+34-44% worse than baseline (112.84). Root cause is the compute-budget interaction: deeper model → slower sec/epoch → fewer epochs in 30-min cap → less training time. Extra capacity cannot offset lost training time at this budget. Depth scaling is gated by the wall-clock cap.

Student also applied a local NaN-safe eval patch (`torch.where(torch.isfinite(err), err, 0)`) to work around the read-only `data/scoring.py` cruise-sample-20 NaN bug — committed only to `target/logs/test_eval_fixed.json`, not to the read-only source. Correctly handled.

**Status: CLOSED — clear negative on primary metric.** n_layers>5 should not be revisited until per-epoch cost drops (mixed precision, lower slice_num) or wall-clock cap is lifted.

---

## 2026-05-15 16:20 — PR #3156: H1: Per-channel surface pressure loss upweighting (p_surf_weight=3x, 5x) — CLOSED

- Branch: `charliepai2i48h3-askeladd/h1-p-channel-surf-upweight`
- Hypothesis: The p (pressure) channel is the only one entering the primary metric; up-weighting it within the surface loss should focus optimization on the scored quantity.

| Arm | p_surf_weight | val_avg/mae_surf_p | test 3-split |
|-----|---------------|--------------------|---------------|
| A | 3x | 151.76 | 146.62 |
| B | 5x | 153.48 | — |

No `p_surf_weight=1.0` control arm was run, so the delta vs default is not isolable.

**Analysis:** Both arms ~+34-36% worse than baseline (112.84). Magnitude is far beyond what a simple per-channel rescaling should produce — something in the loss reformulation destabilized training (likely interacting with the already-aggressive surf_weight=10 multiplier). Peak VRAM was only 5-6 GB (vs 57-64 GB for the deeper-model PR), suggesting a different effective batch behavior worth noting.

Useful negative signal: simple per-channel surface upweighting on top of surf_weight=10 does not help. Future surface-pressure-focused work should likely operate at a different abstraction (robust losses, magnitude normalization, output-head scaling) rather than per-channel weighting.

**Status: CLOSED — negative; no isolable delta due to missing control.**

---

## 2026-05-15 17:15 — PR #3297: H13: Surface dual-head (dedicated surface MLP) — CLOSED

- Branch: `charliepai2i48h3-tanjiro/h13-surface-dual-head`
- Hypothesis: A dedicated lightweight MLP applied to surface nodes only would specialize the decoder for the primary metric (surface pressure).
- Implementation: SurfaceHead(in_dim=128, hidden_dim=128, out_dim=4, depth=2) applied without FiLM (pre-FiLM-merge base).

| Config | val_avg/mae_surf_p | Notes |
|--------|-------------------|-------|
| Surface head (no FiLM) | 130.54 | +15.7% worse than baseline |

**Analysis:** Surface head ALONE (without FiLM) makes things significantly worse. The model lacks the per-condition context to specialize; without knowing Re and AoA, the surface head cannot distinguish high-Re from low-Re surface distributions. The student's own analysis identified FiLM as the missing ingredient. This is a critical direction-setter: surface head alone is a dead end; surface head + FiLM compound is the logical follow-up.

**Status: CLOSED.** Follow-up assigned as H16 (FiLM + surface head compound) to askeladd, PR #3338.

---

## 2026-05-15 17:15 — PR #3338: H16: FiLM + Surface Head compound — ASSIGNED to askeladd

- Branch: `askeladd/h16-film-surface-head-compound`
- Hypothesis: FiLM conditioning provides the per-condition context that makes a dedicated surface head effective. With FiLM on (cond_dim=11, merged default), adding a SurfaceHead(in_dim=128, hidden_dim=128, out_dim=1, depth=2) as an additive residual on pressure channel of surface nodes should reduce val_avg/mae_surf_p below 112.84.
- Arms: Arm A (depth=2, hidden=128), Arm B (depth=3, hidden=256) if time permits.
- Status: WIP

---

## 2026-05-15 17:15 — PR #3339: H8: Per-sample adaptive loss normalization — ASSIGNED to tanjiro

- Branch: `tanjiro/h8-per-sample-normalization`
- Hypothesis: GT surface pressure std varies 50–2077 Pa across the Re range (40x). MSE/Huber without normalization gives ~1700x more gradient to high-Re samples. Normalizing each sample's loss by its per-sample GT std equalizes gradient flow, improving OOD generalization especially on val_geom_camber_rc.
- Arms: Arm A (Huber δ=0.5 + FiLM + per-sample norm), Arm B (MSE + FiLM + per-sample norm).
- Status: WIP

---

## 2026-05-15 17:15 — PR #3340: H9: WSD schedule + AdamW beta2=0.98 — ASSIGNED to thorfinn

- Branch: `thorfinn/h9-wsd-schedule-beta2`
- Hypothesis: CosineAnnealingLR(T_max=50) never anneals in the ~14-epoch wall budget. WSD (Warmup-Stable-Decay) schedule with warmup=3, stable=8, decay=rest ensures a proper decay phase regardless of timeout. AdamW beta2=0.98 stabilizes second-moment estimates faster (~50 steps vs ~1000) for the short run.
- Arms: Arm A (WSD + beta2=0.98), Arm B (WSD + default beta2=0.999) to isolate effects.
- Status: WIP

---

## 2026-05-15 18:30 — PR #3335: H15: Huber δ=0.5 + T_max=15 compound (nezuko) — MERGED NEW BEST

- Branch: `nezuko/h15-huber-tmax15-compound`
- Hypothesis: The two independently confirmed improvements (Huber δ=0.5 and T_max=15 schedule fix) are mechanistically orthogonal and should stack. Prediction: ~105-110 val_avg (additive estimate).
- Artifacts: `models/model-h15-huber-tmax15-compound-20260515-172641/metrics.jsonl`, `metrics.yaml`
- Note: FiLM was **OFF** (`cond_dim=0`) in this run — student disabled via `--cond_dim 0`. Only Huber δ=0.5 + T_max=15 were active.

| Config | val_avg/mae_surf_p | Δ vs prev best |
|--------|---------------------|----------------|
| H15: Huber δ=0.5 + T_max=15 (no FiLM) | **94.6764** | **−18.16 (−16.1%)** |
| Prev best: PR #3160 (Huber δ=0.5, no FiLM) | 112.8406 | — |
| Naive additive prediction from 125.90 | ~101.1 | — |

Per-split breakdown:

| Split | H15 val_avg | Prev (PR #3160) | Δ |
|-------|-------------|-----------------|---|
| val_single_in_dist | 112.48 | 144.92 | −32.44 |
| val_geom_camber_rc | 102.48 | 125.53 | −23.05 |
| val_geom_camber_cruise | 72.96 | 81.82 | −8.86 |
| val_re_rand | 90.79 | 99.10 | −8.31 |
| **val_avg** | **94.68** | **112.84** | **−18.16** |

Test splits (from best-val-epoch checkpoint):

| Split | test mae_surf_p |
|-------|----------------|
| test_single_in_dist | 100.67 |
| test_geom_camber_rc | 93.19 |
| test_re_rand | 83.42 |
| test_geom_camber_cruise | NaN (scoring bug) |
| test_avg (3-split, excl. cruise) | **92.42** |

**Analysis:** Super-linear stacking: observed 94.68 vs naive additive prediction ~101.1 (6.5 pts better than additive). Mechanism: broken T_max=50 schedule kept LR high throughout, washing out Huber's tail-damping benefit via persistent gradient noise. With T_max=15 annealing to ~0 at epoch 14, the model enters a stable low-LR refinement phase where Huber's balanced gradient bias (resisting over-correction on extreme-Re samples) pays off most. Every split improved by 8–32 pts — broad-spectrum gain, not artefact. Largest absolute gain on val_single_in_dist (−32.4 pts), previously highest-error split.

**Key open question from this result:** FiLM was off. Adding FiLM on top of (Huber δ=0.5 + T_max=15) is the highest-priority next experiment. FiLM alone gave ~35 pts over raw baseline; with the properly annealed schedule, the conditioning benefit may be much larger.

**Status: MERGED — NEW BEST (94.6764)**


---

## 2026-05-15 19:35 — PR #3340: H9: WSD schedule + AdamW beta2=0.98 (thorfinn) — PENDING VERIFICATION (sent back for rebase)

- Branch: `thorfinn/h9-wsd-schedule-beta2`
- Hypothesis: WSD (Warmup-Stable-Decay) schedule with warmup=3, stable=8, decay over epochs 11–14 outperforms cosine T_max=15 in the fixed ~14-epoch wall-budget regime. AdamW β2=0.98 should stabilize faster for short runs.
- Two arms (paired comparison):

| Arm | Scheduler | β2 | best epoch | val_avg/mae_surf_p | test 3-split avg |
|-----|-----------|----|-----------|--------------------|-----------------|
| Arm A: WSD + β2=0.98 | WSD (3/8/4) | 0.98 | 12/13 | 102.20 (+7.52 vs 94.68) | 102.55 |
| **Arm B: WSD + β2=0.999** | **WSD (3/8/4)** | **0.999** | **14/14** | **89.04 (−5.64 vs 94.68)** | **85.90** |

**Per-split breakdown (Arm B winner):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| val_single_in_dist | 109.74 | 95.50 |
| val_geom_camber_rc | 98.22 | 87.39 |
| val_geom_camber_cruise | 66.30 | NaN |
| val_re_rand | 81.89 | 74.80 |
| **val_avg** | **89.04** | **85.90** (3-split) |

**Key observation (LR trajectory in Arm B):** per-epoch lr after `scheduler.step()`:
- ep 1–2 warmup (1.67e-4 → 5e-4)
- ep 3–11 stable plateau (5e-4)
- ep 12–14 decay (5e-4 → 4.27e-4 → 2.5e-4 → 7.32e-5)
- val descent during decay: ep 11 → 113.30, ep 12 → 117.22, ep 13 → 96.18, **ep 14 → 89.04**

The sharp final-decay improvement (96 → 89 in the last epoch) shows the model was still converging at the wall cap. **WSD's sharp-decay shape vs cosine's smooth-decay shape appears to drive the win.**

**Why β2=0.98 underperformed:** thorfinn's analysis is correct — at ~5250 steps/run, β2=0.98 (~50-step half-life) has too high preconditioner variance for the long-tailed surf_weight=10 loss. β2=0.999 (~1000-step half-life) tracks better and is the right scale for this loss landscape.

**Confound:** Arm B used the merged default (FiLM on cond_dim=11, Huber δ=1.0) while the 94.68 baseline used `--cond_dim 0 --huber_delta 0.5`. So 5.64 pt gain includes WSD effect + potential FiLM/δ effect. Need a controlled re-run.

**Status: SENT BACK FOR REBASE + VERIFY.** thorfinn's branch was 4 commits behind the current advisor (pre-H15 merge), causing conflicts that reverted advisor-owned files (BASELINE.md, research/*.md) and removed merged H15 artifacts. Asked thorfinn to: (1) rebase carefully keeping their WSD code; (2) re-run Arm B to verify 89.04 holds on rebased branch; (3) optionally also run a controlled arm (WSD + `--huber_delta 0.5 --cond_dim 0`) to isolate WSD vs cosine T_max=15 effect cleanly. Merge once rebase + re-run confirms.


---

## 2026-05-15 21:30 — PR #3449: H24: EMA weight averaging (d=0.999) on H19 stack (edward) — CLOSED, dead end

- Branch: `charliepai2i48h3-edward/h24-ema0999`
- Hypothesis: EMA shadow weights at decay=0.999 should improve OOD generalization on the H19 triple-compound base.
- Single arm, FiLM+Huber δ=0.5+T_max=15 + EMA shadow weights for eval.

| Metric | H19 baseline | H24 EMA d=0.999 | Δ |
|--------|--------------|-----------------|---|
| val_avg/mae_surf_p | 83.8136 | 91.4897 | **+7.68 (+9.2% regression)** |
| val_single_in_dist | 96.4406 | 106.3657 | +9.93 |
| val_geom_camber_rc | 93.7378 | 103.7417 | +10.00 |
| val_geom_camber_cruise | 62.8339 | 69.6562 | +6.82 |
| val_re_rand | 82.2422 | 86.1954 | +3.95 |
| test_avg (3-split) | 80.2415 | 88.9924 | +8.75 |

**Why EMA failed:** Per edward's own analysis: 13 epochs × 375 steps = 4,875 optimizer steps total; EMA decay=0.999 has ~1,000-step half-life. Validation MAE is dropping 30%+ between epochs 5 and 13 — the model is still rapidly improving, so shadow weights lag a regime where the live model is materially better. The lagging shadow weights become a strictly-worse estimator. EMA literature (diffusion, ViT) uses 100k–1M+ steps where the late-training plateau dominates; that regime does not apply at 4,875 steps.

**Pivot:** SWA (Stochastic Weight Averaging) captures the same idea (averaging late-training weights) but with equal-weight averaging of the FINAL K epochs only, skipping the rapidly-improving early phase. Reassigned to edward as H28.

**Status: CLOSED — clear regression, dead end at this decay/budget combination.**


---

## 2026-05-15 22:30 — PR #3450: H25: Per-channel Huber loss (askeladd) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-askeladd/h25-per-channel-huber`
- Hypothesis: The pressure channel (mae_surf_p) is the primary metric and has heavier tails than velocities at high-Re. Decoupling Huber's δ per channel — aggressive clip on p, near-MSE on velocities — should target the metric directly.
- Two arms on H19 triple-compound base (FiLM cond_dim=11 + T_max=15):

| Arm | δ_vel | δ_p | val_avg/mae_surf_p | test 3-split avg |
|-----|-------|-----|--------------------|-----------------|
| Arm A | 0.5 | 0.25 | 78.2286 (−5.58 vs H19) | 75.35 |
| **Arm B** | **1.0** | **0.25** | **75.7713 (−8.04 vs H19, −9.6%)** | **73.07** |

**Per-split breakdown (Arm B winner, NEW BEST):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| val_single_in_dist | 86.5482 | 74.5330 |
| val_geom_camber_rc | 87.4861 | 78.8537 |
| val_geom_camber_cruise | 55.2883 | NaN (scoring bug) |
| val_re_rand | 73.7625 | 65.8246 |
| **val_avg** | **75.7713** | **73.0704** (3-split) |

**Why per-channel Huber works:** Pressure has heavier tails at high-Re — δ_p=0.25 clamps extremes aggressively, preventing pressure outliers from dominating optimization. Velocities follow a more Gaussian distribution; δ_vel=1.0 keeps near-MSE behavior in the well-behaved bulk. Arm B's δ_vel=1.0 beating Arm A's δ_vel=0.5 confirms velocity gradients should not be over-clipped — the asymmetry is what wins.

- Artifacts: `models/model-charliepai2i48h3-askeladd-h25-perchan-huber-vel10-p025-h19-20260515-222431/metrics.yaml`
- Best epoch: 14 / 50 (30-min wall cap; LR fully annealed via T_max=15)
- Params: 662K + FiLM heads (cond_dim=11)

**Status: MERGED — new best.** All R3 follow-ups should rebase onto this per-channel Huber base.


---

## 2026-05-15 22:35 — PR #3447: H22: Uniform Huber δ sweep on H19 (fern) — MERGED, beats H19 but not H25

- Branch: `charliepai2i48h3-fern/h22-huber-delta-sweep`
- Hypothesis: Continue the monotone δ trend (1.0 → 0.5 → ?) — does even smaller uniform δ keep improving on the H19 triple-compound base?
- Two arms on H19 base:

| Arm | δ | val_avg/mae_surf_p | test 3-split avg | Δ vs H19 (83.81) |
|-----|---|--------------------|-----------------|------------------|
| Arm A | 0.25 | 79.1517 | 76.19 | −4.66 |
| **Arm B** | **0.1** | **78.8321 (won arm)** | **76.50** | **−4.98** |

**Per-split breakdown (Arm B winner):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| val_single_in_dist | 89.2334 | 78.9430 |
| val_geom_camber_rc | 91.7254 | 81.1670 |
| val_geom_camber_cruise | 58.2720 | NaN |
| val_re_rand | 76.0975 | 69.3794 |
| **val_avg** | **78.8321** | **76.50** (3-split) |

**Analysis:** Both arms beat H19, confirming the monotone δ trend (1.0 → 0.5 → 0.25 → 0.1 all improve). But the uniform approach is now superseded by H25's per-channel decoupling (δ_vel=1.0, δ_p=0.25 → 75.77). The reason: pressure benefits from aggressive clipping but velocities lose information when clipped uniformly that hard. H25 captures both effects.

**Diminishing returns visible:** H19 (83.81) → H22 δ=0.25 (79.15, −4.66) → H22 δ=0.1 (78.83, −0.32). Uniform-δ knob nearly exhausted; the next gain came from changing the *shape* (per-channel) not the magnitude.

- Artifacts: `models/model-h22-huber-delta01-h19-20260515-222734/`, `models/model-h22-huber-delta025-h19-20260515-212555/`

**Status: MERGED.** Provides confirmation point on δ trend; per-channel Huber (H25) is the better path forward.



---

## 2026-05-15 21:23 — PR #3445: H20: Gradient clip=1.0 on H19 triple compound (nezuko) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-nezuko/H20-grad-clip-confirm`
- Hypothesis: Grad clip max_norm=1.0 (from H18B signal ~74.23 on pre-H19 base) on the merged H19 triple-compound stack.
- Two arms: clip=1.0 (Arm A) and clip=0.5 (Arm B).
- **Effective base config:** FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 (defaults on merged train.py) + T_max=15 + clip.

| Arm | clip | val_avg/mae_surf_p | test 3-split avg | Δ vs H19 (83.81) |
|-----|------|--------------------|-----------------|------------------|
| **Arm A** | **1.0** | **75.4955 (NEW BEST)** | **73.1556** | **−8.32** |
| Arm B | 0.5 | 77.0687 | 73.0051 | −6.74 |

**Per-split breakdown (Arm A, NEW BEST):**

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| val_single_in_dist | 85.7272 | 77.4314 |
| val_geom_camber_rc | 85.4700 | 77.5658 |
| val_geom_camber_cruise | 55.7886 | NaN |
| val_re_rand | 74.9964 | 64.4696 |
| **val_avg** | **75.4955** | **73.1556** (3-split) |

**Why it works:** Pre-clip grad norm was 5–17× throughout (clipping active at every step). With Huber already suppressing pressure tail loss terms, clip=1.0 adds a second layer: bounding the per-step update magnitude so that no single high-Re sample's gradient dominates a step. clip=0.5 is too aggressive (17–34× reduction), starving useful gradient signal.

**Subtle finding:** H20 used `--huber_delta 0.5` which is NOT used in the loss code (train.py always uses `huber_delta_vel`/`huber_delta_p`). H20 actually ran with δ_vel=0.5, δ_p=0.25 (the merged defaults from H25). So H20's effective config differs from H25 only in: δ_vel=0.5 vs 1.0 AND clip=1.0 vs None.

- Artifacts: `models/model-h20-clip1-h19-20260515-212335/`, `models/model-h20-clip05-h19-20260515-222355/`
- Best epoch: 14 (30-min wall cap; T_max=15 fully annealed)

**Status: MERGED — new best.** All R3+ experiments should now include clip_grad_norm=1.0 by default.


---

## 2026-05-15 21:35 — PR #3446: H21: WSD schedule on H19 triple compound (thorfinn) — CLOSED, dead end

- Branch: `charliepai2i48h3-thorfinn/H21-wsd-schedule-h19`
- Hypothesis: WSD (Warmup-Stable-Decay) schedule should outperform cosine on H19 stack.
- Two arms: warmup=2/stable=9 (Arm A) and warmup=1/stable=10 (Arm B).

| Arm | val_avg/mae_surf_p | vs H19 (83.81) |
|-----|---------------------|----------------|
| Arm A (warmup=2,stable=9) | 96.71 | +15.4% |
| Arm B (warmup=1,stable=10) | 101.09 | +20.6% |

**Why WSD failed:** The 30-min cap cuts at epoch 14. WSD's decay phase barely fires — LR is ~5e-4 through most of training and only begins decaying at epoch 12-13. CosineAnnealingLR(T_max=15) anneals smoothly from epoch 1, giving stable convergence across all 14 epochs. WSD requires ~25+ epochs for the decay phase to be meaningful. Second confirmed WSD failure (H9 Arm B at 89.04 was on a less-tuned base).

**Status: CLOSED — dead end. Do not revisit WSD unless SENPAI_TIMEOUT_MINUTES is raised.**


---

## 2026-05-15 23:25 — PR #3448: H23: surf_weight sweep (5, 20) on H19 triple compound (tanjiro) — SENT BACK for rebase

- Branch: `charliepai2i48h3-tanjiro/H23-surf-weight-sweep`
- Hypothesis: Default surf_weight=10 is above optimum; reducing to 5 improves primary metric.
- Two arms: surf_weight=5 (Arm A) and surf_weight=20 (Arm B).

| Arm | surf_weight | val_avg/mae_surf_p | vs H19 (83.81) |
|-----|-------------|---------------------|----------------|
| **Arm A** | **5** | **81.91** | **−1.90 (wins vs H19)** |
| Arm B | 20 | 84.77 | +0.96 |

Arm A beats H19 (83.81) but not H20 current best (75.50). Also improved vol_p metrics simultaneously — no surface/volume tradeoff, suggests 10 was too high.

**Status: SENT BACK.** Request rebase onto H20 base (clip=1.0) + test surf_weight=5 and surf_weight=2 with clip.


---

## 2026-05-15 23:26 — PR #3452: H27: LR sweep (3e-4, 7e-4) on H19 triple compound (frieren) — SENT BACK for rebase

- Branch: `charliepai2i48h3-frieren/H27-lr-sweep-h19`
- Hypothesis: lr=5e-4 is suboptimal on H19 stack; higher peak LR covers more loss landscape.
- Two arms: lr=3e-4 (Arm A) and lr=7e-4 (Arm B).

| Arm | lr | val_avg/mae_surf_p | vs H19 (83.81) |
|-----|----|--------------------|----------------|
| Arm A | 3e-4 | 86.78 | +2.97 |
| **Arm B** | **7e-4** | **79.79** | **−4.02 (wins vs H19)** |

Arm B beats H19 but not H20 (75.50). frieren's prediction of stability at lr=7e-4 confirmed — no spikes. Monotone LR direction confirmed: 3e-4 < 5e-4 < 7e-4.

**Status: SENT BACK.** Rebase onto H20 base (clip=1.0) + test lr=7e-4 and lr=1e-3 with clip.


---

## 2026-05-15 23:30 — PR #3451: H26: FiLM cond_dim ablation (3 vs 1) on H19 (alphonse) — SENT BACK for rebase

- Branch: `charliepai2i48h3-alphonse/H26-film-cond-dim-ablation`
- Hypothesis: FiLM with cond_dim=11 includes noisy geometry dimensions that zero out for single-foil samples.
- Two arms: cond_dim=3 (Re, AoA1, NACA1_camber) and cond_dim=1 (Re only).

| Arm | cond_dim | val_avg/mae_surf_p | vs H19 (83.81) |
|-----|----------|--------------------|----------------|
| **Arm A** | **3** | **82.51** | **−1.30 (wins vs H19)** |
| Arm B | 1 | 86.42 | +2.61 |

cond_dim=3 beats cond_dim=11! Geometry tail dims (AoA2, NACA2, gap, stagger) zero out for single-foil samples (~half of training data), creating distribution mismatch that hurts FiLM. cond_dim=3 (Re, AoA1, NACA1_camber) is leaner and better. But doesn't beat H20 (75.50).

**Status: SENT BACK.** Rebase onto H20 base (clip=1.0) + test cond_dim=3 and cond_dim=2 with clip.


---

## 2026-05-16 00:23 — PR #3491: H28: SWA on H19 triple compound (edward) — CLOSED, dead end

- Branch: `charliepai2i48h3-edward/h28-swa-on-h19-stack`
- Hypothesis: SWA (equal-weight avg of last 7 epochs from start_epoch=7) should find a flatter minimum than the terminal checkpoint.
- Single arm: swa_start_epoch=7 on H19 base.

| Source | val_avg/mae_surf_p | vs H19 (83.81) |
|--------|---------------------|----------------|
| SWA (start_epoch=7) | 89.68 | +5.86 regression |
| Live checkpoint (diagnostic) | 85.21 | +1.40 regression |

**Why SWA failed:** Model improves by ~44 pts between epoch 7 and 14 (129→85 val_avg). SWA averages weights from the steep-descent phase, yielding mid-trajectory weights far worse than the endpoint. CosineAnnealingLR(T_max=15) never plateaus within 14 epochs, so there's no converged basin to average. Same failure mode as H24 EMA.

Principle confirmed: Both EMA and SWA fail at this budget. Averaging requires a post-convergence regime.

**Status: CLOSED — dead end. Revisit SWA only if budget exceeds ~20 epochs (swa_start_epoch=12+).**
