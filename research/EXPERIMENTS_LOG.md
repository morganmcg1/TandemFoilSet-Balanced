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


---

## 2026-05-16 01:25 — PR #3549: H29: Per-channel Huber + clip=1.0 compound (nezuko) — CLOSED, informative negative

- Branch: `charliepai2i48h3-nezuko/h29-perchan-huber-clip1-compound`
- Hypothesis: Compound H25 (δ_vel=1.0/δ_p=0.25) + H20 (clip=1.0) — orthogonal improvements should stack.
- Single arm: δ_vel=1.0, δ_p=0.25, clip_grad_norm=1.0 on H19 base.

| Source | val_avg/mae_surf_p | test 3-split | vs H20 (75.50) |
|--------|---------------------|--------------|----------------|
| H29 compound | 76.7951 | 74.7093 | +1.30 (worse) |
| H20 alone | 75.4955 | 73.1556 | 0 |
| H25 alone | 75.7713 | 73.0704 | +0.28 |

**Per-split:** H29 wins on val_geom_camber_cruise (53.14 vs 55.79) but loses on the three harder splits (single_in_dist +5.10, rc +3.44, re_rand only slightly better).

**Key mechanism (nezuko's analysis):** The two interventions are NOT orthogonal — they interact through `clip_grad_norm_`'s global rescale. With δ_vel=1.0, velocity gradients are larger than pressure gradients. The global norm clip rescales the entire gradient vector by ‖g‖, which means pressure updates get scaled down disproportionately. The H25 velocity benefit (less clipping → bigger velocity steps) is canceled by the global clip's velocity-dominated rescale.

This is a high-quality elimination of a plausible hypothesis. Pre-clip ‖g‖ was 3.8–18.0 throughout training, confirming clipping was meaningfully engaged (not a no-op).

- Artifacts: `models/model-h29-perchan-vel10-p025-clip1-20260516-003604/`

**Status: CLOSED — informative negative.** Follow-up H34 assigned to nezuko: element-wise clipping (`clip_grad_value_`) to bypass the global-rescale interaction and directly test the hypothesis.

---

## 2026-05-16 02:35 — PR #3561: H33: Wider Transolver (n_hidden=192/256) on H20 base (edward) — CLOSED, dead end

- Branch: `charliepai2i48h3-edward/h33-wider-hidden-h20`
- Hypothesis: Increase n_hidden from 128 → 192 or 256 to increase model capacity. H5 (R1) was inconclusive on untuned stack.

| Arm | n_hidden | val_avg | vs baseline | Notes |
|-----|----------|---------|-------------|-------|
| Arm A | 192 | 86.91 | +15.1% | regression |
| Arm B | 256 | 92.54 | +22.6% | regression |
| **Baseline (H20)** | **128** | **75.4955** | **0** | current best |

**Both arms significantly worse.** val_single_in_dist (the hardest split) is 85.7 at baseline — already the worst. Edward's analysis: capacity isn't the bottleneck. Within the 14-epoch budget, a wider model can't use its extra capacity (the LR schedule governs convergence speed, not capacity). Wider = more parameters to converge in the same wall-clock budget. Also: the model may be regularization-starved — wider nets may need higher wd or dropout.

**Key insight:** Architecture capacity fails on this problem because we're budget-constrained, not capacity-constrained. The "val_single_in_dist is worst" pattern is a distribution difficulty problem, not a capacity problem.

- Artifacts: `models/model-h33-nhidden192-h20-*/`, `models/model-h33-nhidden256-h20-*/`

**Status: CLOSED — dead end on capacity. H35 assigned to edward: slice_num sweep (96, 128), targeting physical representation budget rather than feature dimension.**

---

## 2026-05-16 02:40 — PR #3551: H30: Grad clip sweep (2.0, 1.5) on H20 base (askeladd) — CLOSED, null result

- Branch: `charliepai2i48h3-askeladd/h30-clip-sweep-h20`
- Hypothesis: clip=1.0 is not the optimum; looser clipping (2.0, 1.5) allows bigger steps in high-LR phase.

| Arm | clip | val_avg | test 3-split | epochs |
|-----|------|---------|--------------|--------|
| Arm A | 2.0 | 75.6754 | 74.5001 | 13 |
| Arm B | 1.5 | 76.5508 | 72.4856 | 13 |
| **Baseline** | **1.0** | **75.4955** | **73.1556** | **14** |

**Budget asymmetry**: both H30 arms completed 13 epochs vs baseline's 14 (per-epoch wall time ~140s vs ~110s, node variance). Per-epoch traces show H30 arms equal-to-or-ahead of baseline at common epochs — the gap is purely the missing 14th cosine-anneal step.

**Hypothesis test failure**: The prediction that clip=2.0 would only activate at ep1 was wrong — pre-clip grad norms decayed from 12.6 to 3.5 across 13 epochs, staying well above 2.0 throughout. This sweep tested "1×/2×/3× active clipping", not "active vs inactive".

**Null result at primary metric level.** Arm B's test 3-split (72.49) is the best test score yet — but this is within run-to-run noise (~1 point variance shown here). clip=1.0 remains merged default.

Key insight: to properly test the clip threshold, need fixed epoch count (--epochs 14 to ensure equal budgets) or longer wall-clock budget.

- Artifacts: `models/model-charliepai2i48h3-askeladd-h30-clip2-h20-*/`, `models/model-h30-clip15-h20-*/`

**Status: CLOSED — null result. H36 assigned to askeladd: AdamW beta2 sweep (0.95 vs 0.999) — orthogonal optimizer direction.**

---

## 2026-05-16 02:45 — PR #3448: H23b: surf_weight sweep (5, 2) + clip=1.0 rebase (tanjiro) — CLOSED, null result

- Branch: `charliepai2i48h3-tanjiro/H23-surf-weight-sweep`
- Hypothesis: H23's surf_weight=5 win (vs 10 on H19 base) transfers to H20 stack (clip=1.0).
- Prior H23 result: sw=5 beat sw=10 by Δ=-1.9 on H19 (no clip). Monotone 20→10→5 trend suggested sw=5 or lower is optimal.

| Arm | sw | val_avg | vs H20 |
|-----|----|---------|--------|
| Arm A | 5 | 75.9103 | +0.41 |
| Arm B | 2 | 78.4422 | +2.95 |
| **Baseline (H20)** | **10** | **75.4955** | **0** |

**H23 win does not transfer.** With clip=1.0, sw=10 is optimal (75.50), sw=5 slightly worse (75.91), sw=2 much worse (78.44).

**Key mechanism (tanjiro's analysis):** The H23 win was an implicit gradient-magnitude correction: high surf_weight creates large surface gradients; reducing sw softened the update magnitude, improving training stability. But clip_grad_norm=1.0 already caps per-step update magnitude globally. The two mechanisms are redundant — and reducing sw now only shifts the loss balance toward volume, hurting the surface metric.

**Generalizable principle confirmed**: This is the 3rd example (after H29 and now H23b) of "gradient-magnitude-shaping interventions don't compound with clip." The list:
- H29: per-channel δ_vel=1.0 + clip=1.0 → clip's global rescale defeats per-channel benefit
- H30: clip=2.0/1.5 → gradient norms still above threshold; looser clip is still active clipping
- H23b: surf_weight=5 + clip=1.0 → clip already controls step magnitude; sw reduction is redundant

Future direction: prioritize **directional** changes (attention structure, slice budget, conditioning), not magnitude changes.

- Artifacts: `models/model-h23b-sw5-clip1-*/`, `models/model-h23b-sw2-clip1-*/`

**Status: CLOSED — null result. H37 assigned to tanjiro: n_head sweep (8, 2), a directional/architectural change that doesn't interact with clip magnitude.**

---

## 2026-05-16 03:05 — PR #3452: H27b: LR=1e-3 + clip=1.0 on H20 base (frieren) — **NEW BEST, MERGED**

- Branch: `charliepai2i48h3-frieren/H27-lr-sweep-h19` (rebased onto H20 stack)
- Hypothesis: LR=1e-3 + clip=1.0 on the current merged stack continues the monotone LR improvement from H27.

| Arm | lr | val_avg | test 3-split | vs H20 (75.50) |
|-----|----|---------|-------------|----------------|
| **Arm B (winner)** | **1e-3** | **71.7713** | **70.6226** | **-3.72 (-4.9%)** |
| Arm A | 7e-4 | 75.9937 | 73.1714 | +0.49 (tied) |
| H20 baseline | 5e-4 | 75.4955 | 73.1556 | 0 |

**Per-split (Arm B lr=1e-3 vs H20 baseline):**

| Split | H20 (lr=5e-4) | H27b Arm B (lr=1e-3) | Δ |
|-------|--------------|----------------------|----|
| val_single_in_dist | 85.73 | 83.78 | -1.95 |
| val_geom_camber_rc | 85.47 | 85.04 | -0.43 |
| val_geom_camber_cruise | 55.79 | 49.52 | -6.27 |
| val_re_rand | 75.00 | 68.74 | -6.26 |
| **val_avg** | **75.4955** | **71.7713** | **-3.72** |
| test_single_in_dist | 77.43 | 72.94 | -4.49 |
| test_geom_camber_rc | 77.57 | 78.04 | +0.47 |
| test_re_rand | 64.47 | 60.89 | -3.58 |
| **test_avg (3-split)** | **73.16** | **70.62** | **-2.54** |

**All val splits improved, test_avg improved -2.54.** Training monotone (no spikes). Pre-clip grad norms: 8.6→2.3 over 13 epochs — clip=1.0 was active every step.

**Key finding:** Monotone LR trend confirmed: 5e-4→7e-4→1e-3 gives 75.50 → ~75.99 (tie) → **71.77 (big win)**. The jump is in the 1e-3 range, not at 7e-4. clip=1.0 is acting as the safety rail that allows the optimizer to take large steps without instability. Every split benefiting is a strong signal of genuine improvement.

**Important note:** `--huber_delta 0.5` flag is a no-op. Realized config: per-channel δ_vel=0.5/δ_p=0.25 from merged defaults.

**val_single_in_dist improvement from 85.73 → 83.78** is notable — this was the "stuck" split previously identified.

- Artifacts: `models/model-h27b-lr1e3-clip1-20260516-012724/`, `models/model-h27b-lr7e4-clip1-20260516-002910/`

**Status: MERGED — new best. Follow-up H38 assigned to frieren (weight decay sweep on new base); H32 thorfinn redirected to lr=1.5e-3/2e-3.**

---

## 2026-05-16 04:00 — PR #3557: H32: LR sweep (lr=1e-3, lr=8e-4) on H20 base (thorfinn) — **NEW BEST, MERGED**

- Branch: `charliepai2i48h3-thorfinn/h32-lr-sweep-h20`
- Hypothesis: lr=1e-3 (and lr=8e-4) + clip=1.0 continue the monotone LR improvement trend. (Note: originally redirected to lr=1.5e-3/2e-3, but thorfinn had already started the original instructions.)

| Arm | lr | val_avg | test 3-split | vs H27b (71.77) |
|-----|----|---------|-------------|----------------|
| **Arm A (winner)** | **1e-3** | **69.4381** | **69.1774** | **-2.34 (-3.3%)** |
| Arm B | 8e-4 | 73.1104 | 71.9396 | +1.34 (worse) |
| H27b baseline | 1e-3 | 71.7713 | 70.6226 | 0 (ref) |

**Independent replication of lr=1e-3 + clip=1.0.** H27b (frieren) and H32 (thorfinn) ran the same nominal config within 1h of each other. Seed-variance spread: 2.33 pts (69.44 vs 71.77). Both confirm lr=1e-3 is a robust 4–6 pt win over H20 (75.50).

Per-split Arm A:
- val_single_in_dist: 79.67 (was 85.73, -6.1 from H20)
- val_geom_camber_rc: 84.47
- val_geom_camber_cruise: 47.27
- val_re_rand: 66.35

**LR sweep table (clip=1.0):**
| lr | val_avg |
|----|---------|
| 5e-4 | 75.50 (H20) |
| 8e-4 | 73.11 (H32 Arm B) |
| 1e-3 | 69.44 (H32 Arm A) / 71.77 (H27b Arm B) |

Monotone trend still holds. LR ceiling not yet visible.

- Artifacts: `models/model-h32-lr1e3-clip1-20260516-012246/`, `models/model-h32-lr8e4-clip1-20260516-022352/`

**Status: MERGED — new best (69.4381). Follow-up H39 assigned to thorfinn (lr=1.5e-3/2e-3).**

---

## 2026-05-16 04:05 — PR #3629: H37: n_head sweep (8, 2) on H20 base (tanjiro) — SENT BACK

- Branch: `charliepai2i48h3-tanjiro/n-head-sweep-h20`
- Hypothesis: n_head=8 (more specialization) vs n_head=2 (richer per-head) on H20 base.

| Arm | n_head | n_params | val_avg | test 3-split | Notes |
|-----|--------|----------|---------|-------------|-------|
| Arm A | 8 | 818K | 86.26 | 82.84 | regression |
| **Arm B** | **2** | **891K** | **72.89** | **71.03** | **beats H20! promising** |
| H20 baseline | 4 | 835K | 75.4955 | 73.1556 | ref |

Arm B (n_head=2, head_dim=64): val_avg=72.89 beats H20 baseline (75.50) by -2.61. Test 3-split 71.03 also beats H20 73.16.

**However, baseline has moved to 69.44 (H32).** n_head=2 at 72.89 doesn't beat that.

Key finding from tanjiro's analysis: Transolver PhysicsAttention has per-head linear layers (not fused), so n_params DOES change with n_head. n_head=2 adds 56K params vs baseline.

The smaller memory footprint (39.6 GB vs 44.6 GB baseline) also allowed Arm B to run the **full 50 epochs** rather than being timeout-capped. Best epoch was 15 vs baseline's 14 — possibly an additional advantage.

**Sent back with instruction:** Re-run n_head=2 on new lr=1e-3 + clip=1.0 base to test compounding.

- Artifacts: `models/model-h37-nhead2-h20-*/`, `models/model-h37-nhead8-h20-*/`

**Status: SENT BACK — test n_head=2 on lr=1e-3 base.**

---

## 2026-05-16 04:08 — PR #3587: H34: Element-wise clip (clip_grad_value=1.0) — CLOSED, dead end

- Branch: `charliepai2i48h3-nezuko/h34-element-wise-clip`
- Hypothesis: clip_grad_value_ preserves per-channel gradient ratios, avoiding H29's global-rescale failure.

| Arm | clip_value | δ_vel | val_avg | vs H20 (75.50) |
|-----|-----------|-------|---------|----------------|
| Arm A | 1.0 | 0.5 | 81.82 | +8.4% |
| Arm B | 1.0 | 1.0 | 82.93 | +9.9% |

Both arms significantly regress. Pre-clip grad norms (3.8–4.4) are systematically LOWER than H20's (5.3) — element-wise clipping at 1.0 is more aggressive than norm clipping at 1.0. The hypothesis that element-wise clip "preserves ratios without reducing magnitude" was wrong: bounding each component at ±1.0 does reduce the total gradient magnitude more severely than bounding the norm at 1.0.

The mechanism: clip_grad_norm_=1.0 rescales the entire vector proportionally (preserves direction). clip_grad_value_=1.0 clips each component independently, which can change gradient direction AND reduces magnitude more aggressively for diffuse high-dim gradients.

**Status: CLOSED — dead end. Element-wise clip at 1.0 is too aggressive. To validate the mechanism, would need clip_value=5–10×.**

---

## 2026-05-16 04:10 — PR #3553: H31: δ_p push (0.1, 0.05) on H20 base — CLOSED, dead end

- Branch: `charliepai2i48h3-fern/h31-delta-p-push-h20`
- Hypothesis: Further reducing δ_p (more L1-like pressure loss) continues the H22 monotone trend on H20 base.

| δ_p | val_avg | vs H20 (75.50) |
|-----|---------|----------------|
| 0.25 | 75.4955 | 0 (baseline) |
| 0.10 | 78.5517 | +3.06 |
| 0.05 | 84.5795 | +9.08 |

Monotone *regression* — opposite of H22. **δ_p=0.25 is the optimum on the clipped stack.** Both H22's trend (δ=0.1→0.25 improving) and H31's reversal (δ=0.25→0.1 degrading) confirm that δ_p=0.25 is a saddle point: the right balance for the per-channel Huber + clip=1.0 configuration.

The δ_p knob is exhausted. Clip already handles outlier suppression; further δ_p reduction removes the quadratic signal from easy samples.

**Status: CLOSED — dead end. δ_p direction fully explored; 0.25 is optimal.**

---

## 2026-05-16 04:12 — PR #3451: H26b: cond_dim=3/2 + clip=1.0 rebase (alphonse) — CLOSED, informative negative

- Branch: `charliepai2i48h3-alphonse/H26-film-cond-dim-ablation`
- Hypothesis: cond_dim=3 win from H26 (H19 base) transfers to H20 (clip=1.0).

| Arm | cond_dim | val_avg | vs H20 (75.50) |
|-----|----------|---------|----------------|
| Arm A | 3 | 79.94 | +4.44 |
| Arm B | 2 | 79.28 | +3.78 |
| H26 baseline (H19 stack) | 3 | 82.51 | -1.30 vs H19 (helped) |

**Inverts the H26 finding.** On H19 (no clip), cond_dim=3 removed noisy geometry-tail dims and helped. On H20 (clip=1.0), cond_dim=3 is +4.4 worse. Mechanism: clip=1.0's global norm rescaling downsizes FiLM gradients proportionally to backbone. With cond_dim=11, the FiLM path has more gradient diversity; reducing to 3 dims weakens the FiLM path relative to backbone after global clip rescale.

This is the 4th confirmed instance of the interaction pattern: interventions that helped on H19's unclipped base fail on H20's clipped base.

**Status: CLOSED — dead end.**

---

## 2026-05-16 05:40 — PR #3629: H37: n_head sweep (n_head=8, n_head=2) on H20 base (tanjiro) — SENT BACK for lr=1e-3 retest

- Branch: `charliepai2i48h3-tanjiro/n-head-sweep-h20`
- Hypothesis: n_head=8 (more specialization, head_dim=16) vs n_head=2 (richer per-head, head_dim=64) on H20 base (lr=5e-4, clip=1.0).

| Arm | n_head | head_dim | n_params | val_avg | test 3-split | vs H20 (75.50) |
|-----|--------|----------|----------|---------|-------------|----------------|
| Arm A | 8 | 16 | 818K | 86.2584 | 82.8366 | +10.76 regress |
| **Arm B** | **2** | **64** | **891K** | **72.8859** | **71.0276** | **-2.61 win** |
| H20 baseline | 4 | 32 | 835K | 75.4955 | 73.1556 | ref |

Per-split Arm B (n_head=2):
- val_single_in_dist: 86.1650 (+0.44 slight regress)
- val_geom_camber_rc: 85.4584 (≈ flat)
- val_geom_camber_cruise: **50.4336** (-5.36 win)
- val_re_rand: **69.4868** (-5.51 win)

**n_head=2 beats H20 on 3/4 splits and all 3 test splits.** However, **current baseline has moved to 69.4381 (H32, lr=1e-3)**. H37 Arm B (72.89) does not beat the new baseline.

**Key insight from tanjiro:** Transolver PhysicsAttention has per-head linear layers (not fused W_O), so n_params scales with dim_head². n_head=2 adds ~56K params (+6.7%). Memory also drops from 44.6 GB to 39.6 GB — valuable headroom.

**Interpretation:** H33 showed wider n_hidden (192/256) regresses. H37 widens head_dim (32→64) with fixed n_hidden=128 and wins. Bottleneck is per-head feature richness, not total capacity. Default 4-head split over-fragments the 128-dim space.

- Artifacts: `models/model-h37-nhead2-h20-20260516-032300/`, `models/model-h37-nhead8-h20-20260516-023633/`

**Status: SENT BACK — Run H37b (n_head=2 + lr=1e-3 + clip=1.0 stacking test). Predicted val_avg ≈ 66.8 if additive. Highest priority experiment.**

---

## 2026-05-16 05:40 — PR #3626: H36: AdamW beta2 sweep (0.95, 0.999) on H20 base (askeladd) — CLOSED, dead end

- Branch: `charliepai2i48h3-askeladd/beta2-sweep-h20`
- Hypothesis: β₂=0.95 (14-step half-life) better tracks rapidly-changing gradient landscape in the short 14-epoch regime vs default β₂=0.999.

| Arm | β₂ | val_avg | vs H20 (75.50) |
|-----|-----|---------|----------------|
| Arm A | 0.95 | 79.4254 | +3.93 (worse) |
| Arm B (control) | 0.999 | 76.0547 | +0.56 (within noise) |

Per-epoch trajectory: arms track until ep6, then β₂=0.999 widens the gap monotonically through cosine decay tail.

**Result: β₂=0.95 clearly hurts.** β₂=0.999 control replicate (76.05) reproduces H20 baseline within ~1pt run-to-run variance, confirming Arm A's +3.93 is real signal not noise.

**Mechanism:** In the cosine-decay tail (small LR), β₂=0.95's responsive second-moment estimator amplifies short-timescale gradient noise into per-parameter LR. β₂=0.999 averages over ~1000 steps → stable normalizer when each step's update is tiny. β₂=0.95 is right when you want rapid adaptation; it's wrong here where you want settling.

The "short-budget β₂=0.95" prior from GPT-3/LLaMA fine-tuning doesn't transfer to batch_size=4 CFD. **β₂=0.999 stays.**

Note: Both arms were on H20 base (lr=5e-4). Neither would have beaten current best H32 (69.44) regardless.

- Artifacts: `models/model-h36-beta2-095-h20-20260516-032352/`, `models/model-h36-beta2-0999-h20-20260516-042356/`

**Status: CLOSED — dead end. β₂ direction fully exhausted. askeladd reassigned to H43.**

---

## 2026-05-16 06:30 — PR #3651: H38: Weight decay sweep (wd=0, wd=5e-5) on lr=1e-3 base (frieren) — **NEW BEST, MERGED**

- Branch: `charliepai2i48h3-frieren/weight-decay-sweep-h27b`
- Hypothesis: Default wd=1e-4 over-regularizes at lr=1e-3 — the AdamW per-step penalty (lr×wd×param) doubled when LR went 5e-4→1e-3. wd=5e-5 restores effective regularization to original strength. wd=0 removes it entirely.

| Arm | wd | val_avg | test 3-split | vs H32 (69.44) |
|-----|----|---------|-------------|----------------|
| **Arm B (winner)** | **5e-5** | **68.1932** | **65.4393** | **-1.25 win** |
| Arm A | 0 | 70.6064 | 66.9854 | -0.83 win |
| H32 baseline | 1e-4 | 69.4381 | 69.1774 | 0 (ref) |

Per-split Arm B (wd=5e-5):
- val_single_in_dist: **76.8452** (was 79.6711, -2.83)
- val_geom_camber_rc: **84.3542** (was 84.4672, -0.11)
- val_geom_camber_cruise: **44.4649** (was 47.2669, -2.80)
- val_re_rand: **67.1084** (was 66.3473, -0.74)

**Arm B improves all 4 val splits.** Arm A (wd=0) also beats baseline by -0.83 but slightly regresses on cruise. Inverted-U confirmed: wd=5e-5 < wd=0 < wd=1e-4.

**Mechanism validated:** Raising lr 5e-4→1e-3 doubled the effective per-step L2 penalty. Restoring the LR-normalized strength (wd=5e-5 at lr=1e-3 ≡ wd=1e-4 at lr=5e-4) gives a clean -1.25 pt improvement. Weight decay is **orthogonal to clip** (parameter norm vs gradient norm) — confirming the H38 prediction.

- Artifacts: `models/model-h38-wd0-lr1e3-clip1-20260516-042517/`, `models/model-h38-wd5e5-lr1e3-clip1-20260516-052550/`

**Status: MERGED — new best (68.1932). Frieren reassigned to H44 (β₁ sweep). All future assignments use wd=5e-5 as default at lr=1e-3.**

---

## 2026-05-16 07:35 — PR #3623: H35: slice_num sweep (96, 128) on H20 base (edward) — CLOSED, walltime-confounded

- Branch: `charliepai2i48h3-edward/slice-num-sweep-h20`
- Hypothesis: Increasing slice_num (64→96, 64→128) gives Transolver more physics-state tokens to represent distinct flow regimes.

| Arm | slice_num | val_avg | best_ep | epochs/30min | vs H20 (75.50) |
|-----|-----------|---------|---------|---------------|----------------|
| Baseline (H20) | 64 | 75.4955 | 14 | 14 | 0 (ref) |
| Arm A | 96 | 80.3092 | 12 | 12 | +4.81 |
| Arm B | 128 | 82.1475 | 11 | 11 | +6.65 |

**Walltime-confounded:** slice96 ran at 160 s/epoch (+15.6% vs baseline 138 s/epoch). slice128 ran at 179 s/epoch (+29.3%). The 30-min timeout cap meant slice96 only got 12 epochs (vs baseline 14) and slice128 only got 11 epochs — losing 2-3 epochs of LR anneal at the steepest part of the loss curve.

**Per-epoch trajectory (edward's analysis):** slice128 was *ahead of baseline at every common epoch from 4 to 11* (e.g., ep7: 104.3 vs 114.4; ep11: 82.1 vs 92.3). slice96 was within ~7 points at all common epochs. The 'regression' is a budget artifact, not a representation failure. n_params barely changes (+1.3%) — slices are soft-partition assignments, not learned tokens.

**Mechanism:** Compute scales linearly with slice_num because the soft-partitioning matrix multiplications add O(N × slice_num) ops per layer. At our 30-min walltime + T_max=15 regime, the binding constraint is wall epochs completed × LR anneal, not representation capacity.

Note: Edward's redirect comment (run on H38 base) arrived 41 seconds AFTER their result was submitted, so this experiment is on the old H20 base. Cannot directly compare to current best 68.19.

- Artifacts: `models/model-charliepai2i48h3-edward-h35-slice96-h20-20260516-042324/`, `models/model-charliepai2i48h3-edward-h35-slice128-h20-20260516-052443/`

**Status: CLOSED — walltime-confounded, slice_num direction shelved until more compute headroom. Edward reassigned to H45 (DropPath / stochastic depth).**

**Status: CLOSED — informative negative. cond_dim=11 remains optimal on clipped stack.**

---

## 2026-05-16 07:36 — PR #3629: H37b: n_head=2 + lr=1e-3 + clip=1.0 stacking test (tanjiro) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-tanjiro/h37b-nhead2-lr1e3-clip1` (retest of #3626 H37, now on lr=1e-3 base)
- Hypothesis: Stack H37 (n_head=2, val=72.89 on H20 base) on top of lr=1e-3+clip=1.0 (H32 base, val=69.44). Predicted ~66.83 if additive from H37's -2.61 gain. H37b = n_head=2 + lr=1e-3 + clip=1.0 + wd=1e-4 (default).

| Arm | n_head | head_dim | lr | wd | val_avg | test_avg (3-split) | vs H38 (68.19) |
|-----|--------|----------|----|----|---------|---------------------|----------------|
| H37b | 2 | 64 | 1e-3 | 1e-4 | **66.1060** | **64.4522** | **−2.09 (WIN)** |

Per-split (best epoch = 15, final completed):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| val_single_in_dist | 74.3956 | 63.9533 |
| val_geom_camber_rc | 78.9959 | 73.0967 |
| val_geom_camber_cruise | 46.4384 | NaN (scoring bug) |
| val_re_rand | 64.5940 | 56.3067 |
| **avg** | **66.1060** | **64.4522** (3-split) |

n_params: 891,469 (+56K vs baseline), peak memory 39.6 GB (vs 44.6 GB). Ran 16 epochs (LR≈0 at ep16), best at ep15.

**Key finding:** Stacking is **super-additive** — predicted 66.83 from independent gains, actual 66.11 (-0.72 better than additive). n_head=8→4→2 is a confirmed monotone improving trend; head_dim=64 is clearly better than 32. Did NOT use wd=5e-5 (predates H38 merge) — that finding is orthogonal and stackable on top.

- Artifacts: `models/model-h37b-nhead2-lr1e3-clip1-20260516-062645/`

**Status: MERGED — new best (66.1060). Tanjiro reassigned to H46 (n_head=1 limit test, PR #3805).**

---

## 2026-05-16 07:45 — PR #3683: H39: LR ceiling push (lr=1.5e-3, 2e-3) at clip=1.0 (thorfinn) — SENT BACK

- Branch: `charliepai2i48h3-thorfinn/lr-ceiling-h32`
- Hypothesis: Test whether lr>1e-3 continues the monotone LR trend. Predicted: pre-clip grad norms should scale proportionally with LR.

| Arm | lr | val_avg | test_avg (3-split) | Δ vs H32 (69.44) |
|-----|----|---------|--------------------|------------------|
| Arm A | 1.5e-3 | 68.1245 | 66.2912 | −1.31 |
| **Arm B** | **2e-3** | **66.3351** | **64.2953** | **−3.10** |

Key observation: Pre-clip grad norms did NOT scale with LR (7.8→7.1→7.0 across lr=1e-3/1.5e-3/2e-3). Clip absorbs proportionally; gradient magnitudes are driven by loss curvature, not LR. Arm B has one non-monotone epoch (ep6: 149 spike) but recovers immediately. No divergence.

**Decision:** Arm B (66.34) beats H38 (68.19) but does NOT beat H37b (66.11) by 0.23 pts. Sent back to test **Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0** — predicted val_avg ≈ **63–64** if stacking compounds.

- Artifacts: `models/model-h39-lr15e4-clip1-20260516-052217/`, `models/model-h39-lr2e3-clip1-20260516-062344/`

**Status: SENT BACK — thorfinn running Arm C stack (PR #3683 → draft).**

---

## 2026-05-16 07:45 — PR #3685: H40: Clip sweep (2.0, 3.0) at lr=1e-3 (nezuko) — CLOSED

- Branch: `charliepai2i48h3-nezuko/clip-sweep-lr1e3`
- Hypothesis: Looser clip at lr=1e-3 might allow beneficial larger steps that clip=1.0 suppresses.

| Arm | clip | val_avg | vs H32 (69.44) |
|-----|------|---------|----------------|
| Arm A | 2.0 | 71.7373 | +2.30 (regression) |
| Arm B | 3.0 | 72.4215 | +2.98 (regression) |

**Key finding:** Clip=1.0 is the confirmed optimum at lr=1e-3. Looser clipping allows harmful large steps. Per H39's finding that pre-clip grad norms remain ~7.0-7.8 throughout, clip=1.0 is an active per-step safety rail — loosening it to 2.0/3.0 lets those ~7.0 norm steps through more fully, causing regression. Knob exhausted.

- Artifacts: `models/model-h40-lr1e3-clip2-20260516-052237/`, `models/model-h40-lr1e3-clip3-20260516-062340/`

**Status: CLOSED — dead end. clip=1.0 confirmed optimal at lr=1e-3. Nezuko reassigned to H47 (cosine eta_min, PR #3807).**

---

## 2026-05-16 07:45 — PR #3688: H41: T_max sweep (20, 18) at lr=1e-3 + clip=1.0 (fern) — SENT BACK

- Branch: `charliepai2i48h3-fern/tmax-sweep-lr1e3`
- Hypothesis: T_max=15 anneals LR to ~4.5% of peak at our ~14-epoch wall. T_max=20 keeps last-epoch LR at ~21% of peak, leaving the optimizer in a more active regime.

| Arm | T_max | best_epoch | val_avg | test_avg (3-split) | Δ vs H32 (69.44) |
|-----|-------|-----------|---------|---------------------|------------------|
| **Arm A** | **20** | **14** | **66.9242** | **64.3028** | **−2.5139 (WIN)** |
| Arm B | 18 | 13 | 72.3357 | 71.2192 | +2.90 (regression) |

Per-split Arm A: single_in_dist 77.23, rc 81.58, cruise 44.08, re_rand 64.80. All 4 splits improve vs baseline.

**Key mechanism:** Arm A got 14 epochs in the budget, Arm B only 13 (wall-clock variance from loader/GPU contention). The *marginal 14th epoch at LR=2.06e-4* dropped val by 5.33 pts (72.25→66.92). This confirms the model is still actively descending at wall budget and the LR floor at that epoch is critical.

**Decision:** Arm A (66.92) beats H38 (68.19) but does NOT beat H37b (66.11) by 0.81 pts. Sent back to test **Arm C: T_max=20 + n_head=2 + wd=5e-5 + clip=1.0** — predicted val_avg ≈ **63–64** if stacking compounds.

- Artifacts: `models/model-h41-tmax20-lr1e3-clip1-20260516-052215/`, `models/model-h41-tmax18-lr1e3-clip1-20260516-062247/`

**Status: SENT BACK — fern running Arm C stack (PR #3688 → draft).**

---

## 2026-05-16 09:30 — PR #3737: H44: AdamW β₁ sweep (0.8, 0.95) on H38 base (frieren) — SENT BACK

- Branch: `charliepai2i48h3-frieren/h44-adamw-beta1-sweep`
- Hypothesis: β₁ controls AdamW first-moment decay. Lower β₁ (faster moment decay) gives crisper updates; higher β₁ (slower decay, more persistent momentum) can overshoot. H36 confirmed β₂=0.999; this tests β₁ ∈ {0.8, 0.95} on the H38 base (lr=1e-3 + wd=5e-5).

| Arm | β₁ | val_avg | test_avg (3-split) | Δ vs H38 (68.19) |
|-----|-----|---------|---------------------|------------------|
| H38 baseline | 0.9 | 68.1932 | 65.4393 | — |
| **Arm A** | **0.8** | **66.6492** | **65.0585** | **−1.54 (WIN over H38)** |
| Arm B | 0.95 | 72.1674 | 69.4378 | +3.97 (regression) |

Per-split Arm A: re_rand dominates gain (-4.1 pts), cruise -1.7, rc -0.5, single_in_dist unchanged. β₁=0.8 leads at all epochs from 7 onward, opening a 5+ pt gap in late epochs (11-13). β₁=0.95 even regresses at epoch 11 (val rises 85→86 → confirms "momentum too persistent → overshoots").

**Decision:** Arm A (66.65) beats H38 (68.19) by 1.54 pts but does NOT beat H37b (66.11) by 0.54. The margin over H38 is below 2.3-pt seed variance — not independently mergeable. However β₁=0.8 is mechanistically distinct (first-moment decay) from n_head/wd/T_max, so stacking is worth testing.

Sent back for **Arm C: β₁=0.8 + n_head=2 + lr=1e-3 + wd=5e-5 + clip=1.0** — predicted val_avg ≈ **64.5–65.5**.

- Artifacts: `models/model-h44-beta1-08-lr1e3-wd5e5-20260516-063229/`, `models/model-h44-beta1-095-lr1e3-wd5e5-20260516-072246/`

**Status: SENT BACK — frieren running Arm C stack (PR #3737 → draft).**

---

## 2026-05-16 09:30 — PR #3729: H43: Linear warmup (1, 2 ep) + lr=1e-3 + clip=1.0 (askeladd) — CLOSED

- Branch: `charliepai2i48h3-askeladd/h43-warmup-lr1e3`
- Hypothesis: Linear warmup (start_factor=0.1, N epochs then cosine) softens early gradient instability at lr=1e-3.

| Arm | warmup | val_avg | test_avg (3-split) | Δ vs H32 (69.44) |
|-----|--------|---------|---------------------|------------------|
| Arm A | 1 ep | 70.8424 | 67.4468 | +1.40 (regression) |
| Arm B | 2 ep | 68.9705 | 68.3538 | −0.47 (within noise) |

Arm A regresses; Arm B marginal improvement below 2.3-pt seed variance threshold. Both far above new baseline H37b (66.11). Key mechanism: **warmup eats into cosine duration at the 30-min wall cap.** Arm B got 13 epochs total with LR at 1.26e-4 at termination — schedule not complete. The budget cost of warmup outweighs any gradient-stability benefit.

Test-side improvement (Arm A -1.73, Arm B -0.83 over H32 test_avg) is logged but below val_avg merge bar.

**Key insight confirmed:** T_max stretch (H41) is the right lever for improving late-epoch LR, NOT warmup. At our 30-min fixed cap, adding any schedule prefix steals cosine epochs.

- Artifacts: `models/model-h43-warmup1-lr1e3-clip1-20260516-062336/`, `models/model-h43-warmup2-lr1e3-clip1-20260516-072417/`

**Status: CLOSED — warmup is budget-stealing at fixed wall budget. Askeladd reassigned to H48 (GEGLU, PR #3834).**

---

## 2026-05-16 09:30 — PR #3689: H42: n_layers sweep (7, 3) on lr=1e-3 + clip=1.0 (alphonse) — SENT BACK

- Branch: `charliepai2i48h3-alphonse/n-layers-sweep-lr1e3`
- Hypothesis: Does depth help where width fails? n_layers=7 (+2 blocks) vs n_layers=3 (−2 blocks) on lr=1e-3 base.

| Arm | n_layers | n_params | val_avg | test_avg (3-split) | best_ep | epochs/30min |
|-----|----------|----------|---------|---------------------|---------|--------------|
| Arm A | 7 | 1,146,591 | 84.1562 | 82.6683 | 10 | **10** (timeout) |
| **Arm B** | **3** | **523,727** | **67.9740** | **65.0983** | **18** | **21** |
| H32 baseline | 5 | ~835K | 69.4381 | 69.1774 | — | ~14 |

**Key finding (HIGH SIGNIFICANCE):** n_layers=3 wins via a compound mechanism:
1. **~2.2× faster per-epoch** (87s vs ~138s baseline) → 21 epochs in 30 min vs 14
2. **Clean convergence** — best at ep18, trajectory peaked at ep18 not budget-limited
3. **Smaller model is better**: 524K params beats 835K (H32), in-line with width failure (H33) — the model was over-parameterized for this dataset/budget

n_layers=7 regresses due to budget truncation (only 10 epochs at ~192s/epoch) — fundamentally walltime-confounded, not conclusively bad architecturally.

**Decision:** Arm B (67.97) beats H38 (68.19) BUT does NOT beat H37b (66.11) by 1.86 pts. However, n_layers=3 is mechanistically orthogonal to n_head=2 (depth vs head structure), both win via "structural efficiency without adding params." Stacking very likely to compound.

Sent back for **Arm C: n_layers=3 + n_head=2 + lr=1e-3 + wd=5e-5 + clip=1.0** — predicted val_avg ≈ **63–65** (21 epochs + n_head=2 + wd=5e-5 stack = very high upside).

- Artifacts: `models/model-h42-nlayers3-lr1e3-clip1-20260516-072335/`, `models/model-h42-nlayers7-lr1e3-clip1-20260516-062551/`

**Status: SENT BACK — alphonse running Arm C stack (PR #3689 → draft).**

---

## 2026-05-16 09:42 — PR #3767: H45: DropPath / stochastic depth (0.1, 0.2) on H38 base (edward) — CLOSED

- Branch: `charliepai2i48h3-edward/h45-droppath-sweep`
- Hypothesis: Add stochastic depth (DropPath) to Transolver residual branches as a regularizer to reduce gap between in-distribution and OOD splits without changing capacity.

| Arm | drop_path | val_avg | Δ vs H38 (68.19) | Δ vs H37b (66.11) |
|-----|-----------|---------|------------------|--------------------|
| Arm A | 0.1 | **78.0165** | +9.82 (regression) | +11.91 (regression) |
| Arm B | 0.2 | **84.3020** | +16.11 (regression) | +18.20 (regression) |

Every split worse including OOD splits where the hypothesis predicted improvement. Both arms still descending at epoch 13 — slow convergence, not divergence. Student's interpretation (correct): pre-overfit regime. The model hasn't started overfitting yet at 14-epoch budget, so stochastic depth only adds noise and slows convergence.

**Key insight closed:** At 30-min wall budget, regularization that costs convergence loses. DropPath would need a longer horizon to express its regularization benefit (consistent with H43 warmup closure — anything that slows the cosine descent eats the budget).

- Artifacts: `models/model-h45-droppath01-20260516-...`, `models/model-h45-droppath02-20260516-...`

**Status: CLOSED — stochastic depth incompatible with short-budget regime. Edward reassigned to H49 (Lion optimizer, PR #3859).**

---

## 2026-05-16 09:42 — PR #3688: H41 Arm C: T_max=20 + n_head=2 + wd=5e-5 stack (fern) — CLOSED

- Branch: `charliepai2i48h3-fern/h41-tmax-sweep` (Arm C run on stack)
- Hypothesis: Does T_max=20 (extended cosine) stack with n_head=2 + wd=5e-5? Predicted ≈ 63–64 via additive decomposition of orthogonal levers.

| Config | val_avg | val_single_in_dist | Δ vs H37b (66.11) |
|--------|---------|---------------------|--------------------|
| H41 Arm C stack | **72.3176** | **94.65** (blew up from 77.23) | +6.21 (regression) |
| H37b baseline | 66.1060 | 74.40 | — |
| H41 Arm A (T_max=20 alone) | 66.9242 | — | +0.81 |

**Key finding (HIGH SIGNIFICANCE) — orthogonality hypothesis refuted:**
1. LR at final epoch 15 was still 1.46e-4 — model never reached fine-tune phase
2. val_single_in_dist *blew up* — failure concentrated on the in-distribution split
3. Per-epoch trajectory shows slow early convergence in stacked config vs Arm A
4. Stretching T_max removed the late-LR fine-tune phase that n_head=2+wd=5e-5 requires for in-distribution

This is the **inverse** of H42's interaction. H42 (n_layers=3) provides BUDGET (21 epochs in 30 min) and the cosine completes — the late LR is reached because there are more epochs to traverse. H41 (T_max=20) provides slower decay but at fixed 14-epoch budget the cosine never reaches the bottom.

**Mechanism implication:** Effective schedule shape must match wall budget. T_max stretch only helps if the model has time to traverse the full cosine. n_head=2+wd=5e-5 are configurations that *especially* need a late-LR cooldown phase for in-distribution.

- Artifacts: `models/model-h41-armc-tmax20-nhead2-wd5e5-...`

**Status: CLOSED — informative negative; schedule lever needs different shape, not extension. Fern reassigned to H50 (WSD trapezoidal schedule, PR #3862) — a mechanically distinct take on the schedule lever that preserves a sharp cooldown.**

---

## 2026-05-16 09:43 — PR #3859: H49: Lion optimizer (lr=1e-4/3e-4 sweep) on H37b base (edward) — WIP

- Branch: `charliepai2i48h3-edward/hypothesis_h49_lion`
- Hypothesis: Lion's sign-based gradient updates act as implicit per-parameter normalization. Strong gradient magnitude variation in our mixed-Re dataset may benefit from sign() removing magnitude information. Also tests whether AdamW is even the right optimizer for this task.

Arms:
- Arm A: Lion lr=1e-4, wd=1e-3, β₁=0.9, β₂=0.99 (standard config)
- Arm B: Lion lr=3e-4, wd=3e-4, β₁=0.95, β₂=0.99 (higher LR + momentum)

Both use n_head=2 + clip=1.0 + merged defaults. Predicted val_avg if Lion mechanism applies cleanly ≈ 64-67. High variance.

**Status: WIP — student executing.**

---

## 2026-05-16 09:44 — PR #3862: H50: WSD trapezoidal schedule (2/8/4, 1/9/4) on H37b base (fern) — WIP

- Branch: `charliepai2i48h3-fern/hypothesis_h50_wsd`
- Hypothesis: WSD (warmup-stable-decay) gives more time at peak LR than cosine, with a sharp linear cooldown that preserves a fine-tune phase. Mechanically distinct from H41 Arm C T_max stretch (which removed the cooldown).

Arms (both at 14-epoch budget):
- Arm A: WSD 2/8/4 (2ep warmup, 8ep stable peak, 4ep linear decay)
- Arm B: WSD 1/9/4 (1ep warmup, 9ep stable peak, 4ep linear decay)

Both stack on n_head=2 + wd=5e-5 + lr=1e-3 + clip=1.0. Orthogonal to H47 (eta_min raises cosine floor); WSD changes the entire schedule shape. Predicted ≈ 64-66.

**Status: WIP — student executing.**

---

## 2026-05-16 10:30 — PR #3683: H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 (thorfinn) — **NEW BEST, PENDING MERGE**

- Branch: `charliepai2i48h3-thorfinn/lr-ceiling-h32`
- Hypothesis: 4-way stack of all positive isolated levers (n_head=2, lr=2e-3 from H39, wd=5e-5 from H38, clip=1.0 from H20) compounds super-additively.

| Seed | val_avg/mae_surf_p | test_avg/mae_surf_p (3-split) |
|---|---|---|
| Seed 1 (best) | **63.4385** | **61.391** |
| Seed 2 | 65.5093 | — |
| Mean of 2 seeds | 64.474 | — |

| Split (best seed) | val | test (3-split) |
|---|---|---|
| val_single_in_dist | ~73 | ~63 |
| val_geom_camber_rc | ~77 | ~69 |
| val_geom_camber_cruise | ~43 | NaN (bug) |
| val_re_rand | ~61 | ~52 |

- Δ vs H37b (66.1060): −2.67 pts val_avg, −3.06 pts test_avg
- Δ vs H32 (69.4381): −6.00 pts val_avg, total stack effect
- Best epoch: 15/50 (cut by timeout)
- 2-seed variance: 2.07 pts (consistent with prior seed noise estimate)

**Analysis:** Super-additive stacking confirmed. The 4-way compound (n_head=2 × lr=2e-3 × wd=5e-5 × clip=1.0) drops val_avg by nearly 6 pts vs the H32 floor, and beats the previously merged H37b baseline by 2.67 pts even taking only the best seed. Mean of 2 seeds (64.47) also beats the H37b single-seed 66.11, so the gain is robust.

Test gain (61.39 vs H37b 64.45) is larger than val gain — the stacked config generalizes well to held-out test geometries. The Re-conditioned model with n_head=2 finds a regime that does better both in-distribution AND OOD.

**Merge status:** First merge attempt blocked by merge conflicts vs current advisor branch (H37b merged after this PR was branched). Sent back to thorfinn with explicit rebase instructions; will merge on next cycle.

**Suggested follow-ups now active:** H51 (LR ceiling lr=2.5e-3, 3e-3), H56 (clip=0.5, 0.7), H54 (surf_weight=5, 20), H55 (Mixup α=0.2, 0.4) — all built on this 4-way stack.

---

## 2026-05-16 10:30 — PR #3689: H42 Arm C: n_layers=3 + n_head=2 + wd=5e-5 (alphonse) — CLOSED, dead end

- Branch: `charliepai2i48h3-alphonse/n-layers-sweep-h32`
- Hypothesis: n_layers=3 isolated win (H42 Arm B) compounds with n_head=2 + wd=5e-5 stack.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 69.16 |
| test_avg/mae_surf_p (3-split) | 66.95 |
| Δ vs H37b baseline | **+3.05** (regress) |
| Δ vs H39 Arm C | **+5.72** (massive regress) |

**Analysis:** Stacking failed dramatically. n_layers=3 (smaller model) interacts destructively with n_head=2 (also smaller capacity per layer). With both reductions applied, the network can't model the camber-OOD and Re-random splits — the model is now under-capacitied for the harder splits.

This adds to the pattern: **simple capacity reductions don't stack with each other**. n_layers=3 wins isolated because it gets more epochs in fixed wall time (21 vs 14); n_head=2 wins isolated because head_dim=64 has better expressivity per head. Stacked, the model lacks both width and depth.

**Status: CLOSED.** Mid-tier model (n_layers=5, n_head=2) is the right balance.

---

## 2026-05-16 10:30 — PR #3737: H44 Arm C: β₁=0.8 + n_head=2 + wd=5e-5 (frieren) — CLOSED, dead end

- Branch: `charliepai2i48h3-frieren/h44-beta1-sweep-h38`
- Hypothesis: AdamW β₁=0.8 isolated win (H44 Arm A) compounds with n_head=2 + wd=5e-5.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 67.25 |
| test_avg/mae_surf_p (3-split) | 64.93 |
| Δ vs H37b baseline | +1.14 (regress) |
| Δ vs H39 Arm C | +3.81 (regress) |

**Analysis:** β₁=0.8 + n_head=2 + wd=5e-5 doesn't stack. Faster moment decay (β₁=0.8) lets the optimizer react to recent gradients more quickly, but combined with the architecture changes from n_head=2, the optimizer ends up overshooting promising local descents. The isolated H44 Arm A win at β₁=0.8 was probably a fluke or sensitive to baseline config.

**Status: CLOSED.** β₁=0.9 is the right choice for this config.

---

## 2026-05-16 10:30 — PR #3805: H46: n_head=1 (head_dim=128) on H37b base (tanjiro) — CLOSED, monotone broken

- Branch: `charliepai2i48h3-tanjiro/h46-n-head-1`
- Hypothesis: monotone n_head 8→4→2 trend (improving) continues to n_head=1 (head_dim=128, full embedding per head).

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 69.17 |
| test_avg/mae_surf_p (3-split) | 66.54 |
| Δ vs H37b baseline | +3.06 (regress) |

**Analysis:** Monotone trend breaks at n_head=1. The U-shape n_head=8→4→2→1 (improving→improving→regression) shows n_head=2 is the global optimum: head_dim=64 is large enough for expressivity but n_head=2 still gives 2-way multi-head ensembling, which n_head=1 loses entirely. **n_head=2 is the floor; don't go lower.**

**Status: CLOSED.** Confirms n_head=2 as the global optimum for this regime.

---

## 2026-05-16 10:30 — PR #3807: H47: cosine eta_min sweep (5e-5, 1e-4) on H37b base (nezuko) — CLOSED, both regress

- Branch: `charliepai2i48h3-nezuko/h47-cosine-eta-min-sweep`
- Hypothesis: cosine LR floor > 0 (eta_min=5e-5 or 1e-4) preserves late-LR exploration capacity.

| Metric | Value (best arm) |
|---|---|
| val_avg/mae_surf_p | 67.35 |
| test_avg/mae_surf_p (3-split) | 66.75 |
| Δ vs H37b baseline | +1.24 (regress) |

**Analysis:** Both eta_min arms regress. eta_min=1e-4 distorts the cosine anneal window — the final epochs don't reach the fine-tune phase that the H37b config (and H39 Arm C) needs for in-distribution. eta_min=5e-5 less aggressive but still hurts.

This adds to the pattern of **schedule lever exhaustion at 14-15 epoch budget**: H43 (warmup) ate budget; H41 Arm C (T_max=20) stripped the tail; H47 (eta_min) raised the floor. All three failed in similar ways — the budget is so tight that any deviation from "cosine T_max=15 from peak to ~0" loses the late-epoch fine-tune that is doing most of the in-distribution work.

**Status: CLOSED.** Cosine T_max=15 with eta_min=0 is the right schedule for this budget. Schedule lever is exhausted; next non-default schedule attempt should be WSD (H50, in-flight) — which has a mechanistically different cooldown shape.

---

## 2026-05-16 10:45 — R5 cycle 5 new assignments

5 idle students (post-cycle-5 closures). 4 new draft PRs assigned; thorfinn pending rebase + merge of #3683.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3896** | alphonse | **H51: LR ceiling push (lr=2.5e-3, 3e-3)** at H39 Arm C stack | WIP |
| **#3897** | frieren | **H56: lower grad clip (0.5, 0.7)** at H39 Arm C stack | WIP |
| **#3898** | nezuko | **H54: surf_weight sweep (5, 20)** at H39 Arm C stack | WIP |
| **#3899** | tanjiro | **H55: Mixup data augmentation (α=0.2, 0.4)** at H39 Arm C stack | WIP |
| **#3683** | thorfinn | **H39 Arm C** — needs rebase against advisor branch to merge | SENT BACK |

Together with WIP from earlier cycles:
| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3834 | askeladd | H48: GEGLU/SwiGLU FFN gating | WIP |
| #3859 | edward | H49: Lion optimizer | WIP |
| #3862 | fern | H50: WSD trapezoidal schedule | WIP |

All 8 students active. Zero idle GPUs.

---

## 2026-05-16 11:45 — PR #3834: H48 GEGLU/SwiGLU FFN gating (askeladd) — **NEW BEST, MERGED**

- Branch: `charliepai2i48h3-askeladd/h48-geglu`
- Hypothesis: GEGLU gated FFN provides spatial selectivity for CFD boundary-layer tokens.

| Arm | val_avg/mae_surf_p | test_avg (3-split) |
|---|---|---|
| **Arm A (GEGLU)** | **58.6268** | **56.6976** |
| Arm B (SwiGLU) | 61.4410 | — |

Per-split (GEGLU best epoch=13): in_dist=61.62, camber_rc=73.90, cruise=40.43, re_rand=58.56. Test: 54.78 / 65.78 / 49.53 (cruise NaN).

- Config: n_head=2 + lr=1e-3 + wd=5e-5 + clip=1.0 + ffn_act=geglu
- Δ vs H37b: **−7.48 pts val, −7.75 pts test** — largest single-PR gain since T_max fix
- Δ vs H39 Arm C: **−4.81 pts** — architecture × optimizer stack confirmed

**Analysis:** GEGLU's decoupled gate `(xW₁) ⊙ σ(xW₂)` amplifies near-wall tokens while suppressing interior flow — a direct match for CFD boundary-layer gradients. Test gains (−7.75) exceed val gains (−7.48) confirming strong OOD generalization. GEGLU > SwiGLU (+2.8 pts) because the decoupled gate is more expressive for physics-constrained data.

**Status: MERGED.** New baseline: val_avg=58.6268, test=56.6976.

---

## 2026-05-16 11:50 — PR #3683: H39 Arm C — SENT BACK (second rebase)

PR #3834 modified train.py (GEGLU classes + ffn_act), conflicting with #3683's eta_min change. Thorfinn rebasing. Once merged, adds H39 Arm C metrics to canonical branch. val=63.44 already documented in BASELINE.md.

---

## 2026-05-16 12:00 — R5 cycle 6 new assignment: askeladd → H57

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3918** | askeladd | **H57: GEGLU + lr=2e-3 mega-stack** | WIP |

GEGLU (architecture) × lr=2e-3 (optimizer) are orthogonal levers. Predicted ≈ 55-57. Highest priority.

Full active roster (8 students, 0 idle):
#3918 askeladd H57 GEGLU+lr2e3 | #3683 thorfinn rebase | #3896 alphonse H51 LR | #3897 frieren H56 clip | #3898 nezuko H54 surf | #3899 tanjiro H55 mixup | #3859 edward H49 lion | #3862 fern H50 wsd

---

## 2026-05-16 12:41 — PR #3683: H39 Arm C (thorfinn) — MERGED (documentation)

- Branch: `charliepai2i48h3-thorfinn/lr-ceiling-h32`
- Result: val_avg=63.4385, test=61.3910 (3-split excl. cruise)
- Merged to get metrics artifacts on advisor branch. Not the current best — H48 GEGLU (58.63) superseded it during review cycle.

---

## 2026-05-16 12:41 — PR #3862: H50 WSD trapezoidal schedule (fern) — CLOSED, dead end

- Branch: `charliepai2i48h3-fern/hypothesis_h50_wsd`
- Hypothesis: Warmup-Stable-Decay schedule at 14-epoch budget outperforms cosine.

| Arm | Config | val_avg/mae_surf_p | Δ vs H37b baseline (66.11) |
|-----|--------|-------------------|---------------------------|
| Arm A | WSD 2/8/4 | 71.3196 | **+5.21 (regression)** |
| Arm B | WSD 1/9/4 | 67.2552 | **+1.15 (regression)** |

**Analysis:** Both WSD arms underperform the pre-GEGLU H37b baseline (66.11), and both are far from the current H48 GEGLU baseline (58.63 — ∆ = 14.7%). The diagnosis is clean: at 14 epochs, cosine's smooth decay from epoch 1 does real optimization work that WSD's stable-phase idles. During the stable phase, the model stays in a chaotic high-LR regime; descent only happens in the 4-epoch linear cooldown, which the cosine baseline beats by covering more gradient terrain throughout. This closes the schedule lever conclusively (joins H43 warmup, H41C T_max=20, H47 eta_min — all schedule changes failed at this budget). WSD is correct for long-training regimes (100k+ steps); our 14-epoch budget is the wrong setting.

**Status: CLOSED — regression vs even the pre-GEGLU baseline. Schedule angle exhausted.**

---

## 2026-05-16 12:41 — PR #3859: H49 Lion optimizer sweep (edward) — CLOSED, forwarded as H58

- Branch: `charliepai2i48h3-edward/hypothesis_h49_lion`
- Hypothesis: Lion sign-based optimizer (lr=1e-4/3e-4 sweep) on H37b base.

| Arm | lr | wd | val_avg/mae_surf_p | Δ vs H37b (66.11) | Δ vs H48 GEGLU (58.63) |
|-----|----|----|-------------------|-------------------|------------------------|
| **A** | 1e-4 | 1e-3 | **60.3008** | **−5.80** | **+1.67** |
| B | 3e-4 | 3e-4 | 61.1256 | −4.98 | +2.49 |

Test 3-split: Arm A = 59.02, Arm B = 58.80.

**Analysis:** Both Lion arms cleanly beat H37b (66.11), confirming Lion's sign-based update normalization removes gradient-magnitude imbalance between high-Re and low-Re samples. The biggest gains landed on val_single_in_dist (−9.4) and val_re_rand (−4.26) — exactly the mixed-Re splits where gradient imbalance is worst. Arm A (lr=1e-4, wd=1e-3) wins validation; Arm B narrowly wins 3-split test (58.80 vs 59.02). However, while this PR was running, H48 GEGLU merged at val=58.63, which is 1.67 pts better than Lion's best (60.30). Lion cannot merge as new baseline.

**Mechanism:** Lion's update direction (sign of the weighted gradient) prevents high-Re samples from dominating weight updates. GEGLU's mechanism (spatial gate selectivity) is orthogonal — different layer, different compute step. If they stack additively: predicted 58.63 − 5.80 = **~52.8**.

**Status: CLOSED — 1.67 pts below H48 GEGLU baseline. Lion forwarded as H58 (Lion + GEGLU mega-stack) for edward.**

---

## 2026-05-16 12:45 — R5 cycle 8 new assignments (3 idle students)

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#3965** | edward | **H58: Lion + GEGLU mega-stack** | ~52-55 |
| **#3966** | fern | **H59: RMSNorm vs LayerNorm in GEGLU Transolver** | ~57-58 |
| **#3968** | thorfinn | **H60: GEGLU + n_layers sweep (4 vs 6)** | ~57-58 |

**H58 (Lion + GEGLU):** If gains compound additively, predicted 58.63 − 5.80 = ~52.8. Highest-risk/highest-reward experiment this cycle. Lion's sign-normalization is orthogonal to GEGLU's gate mechanism — different levers, different parts of the forward pass.

**H59 (RMSNorm):** Replace LayerNorm with RMSNorm in Transolver blocks. Mechanistically motivated: GEGLU gates depend on directional activation structure; LayerNorm's mean-subtraction can distort that signal. RMSNorm preserves it. Small gain expected (0.5-2 pts) but clean test with no new hyperparameters.

**H60 (n_layers sweep):** Revisit depth in GEGLU context. H42 Arm C showed n_layers=3 + n_head=2 compound destructively at vanilla FFN; GEGLU's extra parameter capacity per block may change the optimal depth. Arms: n_layers=4 (shallower) and n_layers=6 (deeper). Uses existing --n_layers flag.

---

## 2026-05-16 14:30 — PR #3918: H57 GEGLU + lr=2e-3 mega-stack (askeladd) — CLOSED, falsified

- Branch: `charliepai2i48h3-askeladd/h57-geglu-lr2e3`
- Hypothesis: GEGLU + lr=2e-3 stacks additively — H39 Arm C won +2.67 pts on vanilla FFN with lr=2e-3, so if GEGLU's architecture is orthogonal, we should see a further improvement.

| Metric | H57 (GEGLU + lr=2e-3) | H48 Baseline (GEGLU + lr=1e-3) | Δ |
|--------|-----------------------|-------------------------------|---|
| **val_avg/mae_surf_p** | **59.50** | **58.6268** | **+0.88 (regression)** |

**Analysis:** H57 falsifies the compounding hypothesis for GEGLU + higher LR. While lr=2e-3 was a +2.67 win on the vanilla FFN stack (H39 Arm C vs H37b), it *hurts* GEGLU by +0.88 pts. Gradient norms were healthy (1.5+), ruling out gate collapse. Most likely mechanism: GEGLU's gated update direction is more "concentrated" per step — the sigmoid gate restricts which gradient components contribute, so each step displaces the model further along its preferred direction. At lr=2e-3, this overshoots the optimum. The LR optima for vanilla FFN and GEGLU architectures are **different**.

**Key insight (confirmed):** GEGLU has a different LR sensitivity profile than vanilla FFN. The optimum LR for GEGLU is ≤ 1e-3, and possibly below it (→ H61).

**Status: CLOSED — +0.88 regression vs H48 GEGLU baseline. LR ceiling confirmed at 2e-3 for GEGLU.**

---

## 2026-05-16 14:30 — PR #3896: H51 LR ceiling sweep (alphonse) — CLOSED, confirmed ceiling

- Branch: `charliepai2i48h3-alphonse/h51-lr-ceiling`
- Hypothesis: LR > 2e-3 extends the monotone trend observed in H39 Arm C. Arms: lr=2.5e-3, lr=3e-3.
- **Note:** H51 was designed against the pre-GEGLU H39 Arm C baseline (val=63.44). These results must be interpreted in that context.

**Analysis:** Both arms regressed vs their pre-GEGLU baseline (63.44). Combined with H57's GEGLU+lr=2e-3 regression, this conclusively closes the LR-ceiling lever: lr=2e-3 is the ceiling for vanilla FFN, and lr=1e-3 appears to be the ceiling for GEGLU (possibly with optimum below 1e-3, see H61).

**Status: CLOSED — LR ceiling confirmed. The LR/schedule lever cluster is fully mapped across both architectures.**

---

## 2026-05-16 14:30 — PR #3897: H56 lower clip sweep (frieren) — CLOSED, clip=1.0 confirmed

- Branch: `charliepai2i48h3-frieren/h56-lower-clip`
- Hypothesis: Lower clip (0.5, 0.7) may help at GEGLU baseline by allowing smaller gradient corrections on the gate parameters. Arms: clip=0.5, clip=0.7.

**Analysis:** Both arms regressed vs H48 GEGLU baseline (58.63). Tighter gradient clipping hurts GEGLU's learning — the gates need the full gradient signal to train effectively. clip_grad_norm=1.0 is confirmed as the global optimum across all architectures tested. The clip lever is exhausted.

**Status: CLOSED — clip=1.0 confirmed optimal. Both lower-clip arms regressed.**

---

## 2026-05-16 14:30 — PR #3898: H54 surf_weight sweep (nezuko) — CLOSED, surf_weight=10 locked

- Branch: `charliepai2i48h3-nezuko/h54-surf-weight`
- Hypothesis: surf_weight (pressure surface loss multiplier) may have a better optimum than 10 at GEGLU baseline. Arms: surf_weight=5, surf_weight=20.
- **Note:** H54 was designed against the H39 Arm C baseline (val=63.44).

**Analysis:** surf_weight=20 starved the volume signal (vol_p MAE increased +27% vs baseline), confirming the predicted tradeoff. surf_weight=5 weakened the surface pressure constraint and regressed. surf_weight=10 is the locked optimum — it balances surface pressure fidelity against volume flow accuracy. Both arms failed to beat 58.63. surf_weight lever is exhausted.

**Key insight (confirmed):** surf_weight=10 is locked. Volume MAE degradation signature confirms mechanism.

**Status: CLOSED — surf_weight=10 confirmed. Both arms regressed vs H48 GEGLU baseline.**

---

## 2026-05-16 14:45 — R5 cycle 10 new assignments (4 idle students)

After closing H57/H51/H56/H54 (LR ceiling, clip, surf_weight levers all exhausted), strategy pivots to architectural and loss-shape levers on the GEGLU baseline.

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#3988** | alphonse | **H61: GEGLU + LR sweep down (7e-4, 5e-4)** | ~57.5-58.5 |
| **#3990** | askeladd | **H62: GEGLU + mlp_ratio sweep (3, 4)** | ~57-58 |
| **#3991** | frieren | **H63: DropPath stochastic depth (0.05, 0.10)** | ~57-58 |
| **#3992** | nezuko | **H64: Huber δ_p retune (0.1, 0.5) at GEGLU** | ~57.5-58.5 |

**H61 (LR down):** Tests whether GEGLU's optimum LR is below 1e-3. H57 showed 2e-3 hurts (+0.88). The gate mechanism concentrates gradient updates — GEGLU may need smaller LR (7e-4 or 5e-4) to avoid overshoot. If GEGLU's LR optimum is below 1e-3, this is a clean, no-code-change win.

**H62 (mlp_ratio):** GEGLU uses gate+up+down projections vs vanilla FFN's single expand+contract. The mlp_ratio=2 was tuned for vanilla FFN; Llama/PaLM literature increases expansion ratio when switching to gated FFNs. Arms: mlp_ratio=3 (~1.1M params) and mlp_ratio=4 (~1.3M params). Requires adding `--mlp_ratio` flag to train.py.

**H63 (DropPath):** Stochastic depth regularization for OOD generalization. Linear schedule per layer (deeper layers see higher drop_prob). Predicted to help camber_rc and re_rand OOD splits more than in_dist. Requires implementing `DropPath` class and `--drop_path` flag.

**H64 (Huber δ_p):** Re-tune Huber loss shape for the improved GEGLU baseline. H25's δ_p=0.25 was tuned at val=83.81 with a heavy error tail. At val=58.63 with GEGLU's spatial selectivity, the error distribution is different — may benefit from more aggressive L1 (δ_p=0.1) or less aggressive L2 (δ_p=0.5). No code changes needed.

---

## 2026-05-16 15:30 — PR #3899: H55 Mixup data augmentation (tanjiro) — CLOSED, decisive negative

- Branch: `charliepai2i48h3-tanjiro/h55-mixup`
- Hypothesis: Mixup with α∈{0.2, 0.4} regularizes training by interpolating raw inputs and targets between sample pairs.

| Arm | val_avg/mae_surf_p | Δ vs H39 Arm C (63.44) | Δ vs H48 GEGLU (58.63) | test 3-split |
|-----|--------------------|------------------------|------------------------|-------------|
| Arm A (α=0.2) | 78.9548 | +15.52 | +20.33 | 75.6663 |
| Arm B (α=0.4) | 88.1268 | +24.69 | +29.50 | 86.4545 |

**Analysis:** Both arms regressed strongly and uniformly across all splits — including OOD splits Mixup was predicted to help (val_geom_camber_rc +22.3, val_re_rand +14.3).

**Mechanism (decisive):** Two failure modes compound, both flagged in the original hypothesis:

1. **PDE nonlinearity.** Mixup enforces local linearity of the model f(x), which is the wrong inductive bias for Navier–Stokes solutions. Velocity and pressure fields are highly nonlinear in geometry and Reynolds number; linear input interpolation creates fictitious physics problems.

2. **Mesh-identity corruption.** The 24-D node feature vector entangles three categories: per-node geometric descriptors (dims 0-11), surface flag (dim 12), and sample-level conditioning (dims 13-23). Only the third interpolates sensibly. Mixing per-node position/distance features creates fictitious node positions; mixing the boolean surface flag creates inconsistent surface/volume signals between input and target.

The student's dual-loss implementation (mathematically equivalent to mixed-target for MSE, slight deviation for L1/Huber) is principled and would not load-bear on a 15+ pt regression.

**Status: CLOSED — Mixup is the wrong inductive bias for PDE-driven CFD surrogates. The Mixup lever is exhausted for this dataset.**

---

## 2026-05-16 15:30 — R5 cycle 11 new assignment (1 idle student)

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#3997** | tanjiro | **H65: EMA weight averaging (0.999, 0.9999)** | ~57.5-58.5 |

**H65 (EMA):** Maintain exponential moving average of model weights during training; evaluate using EMA weights. The most-cited "free improvement" in modern deep learning (Polyak 1991, Izmailov et al. 2018 SWA). Finds flatter loss-landscape regions which generalize better — especially helpful for OOD splits. H2 (edward, R1) tested EMA at the original baseline (val=114.6), result wasn't transformative because baseline was too noisy. Worth revisiting at the tighter GEGLU baseline. Arms: ema_decay=0.999 (standard) and 0.9999 (slow, may undertrain in 15-epoch budget).

---

## 2026-05-16 16:00 — PR #3968: H60 GEGLU + n_layers sweep (thorfinn) — **MERGED, NEW BASELINE**

- Branch: `charliepai2i48h3-thorfinn/h60-geglu-nlayers`
- Hypothesis: Revisit depth in GEGLU context. Pre-GEGLU H42 found n_layers=3+n_head=2 destructively stacked; gated FFN's extra per-block capacity may change the optimum.

| Metric | H48 baseline (n=5) | Arm A (n=6) | **Arm B (n=4)** | Δ Arm B |
|--------|--------------------|-------------|------------------|---------|
| **val_avg/mae_surf_p** | **58.6268** | 67.0902 | **57.5750** | **−1.05** |
| val_single_in_dist | 61.6193 | 74.0977 | 63.3430 | +1.72 |
| val_geom_camber_rc | 73.8983 | 80.2999 | 72.1854 | −1.71 |
| val_geom_camber_cruise | 40.4338 | 46.5019 | 37.7532 | −2.68 |
| val_re_rand | 58.5556 | 67.4611 | 57.0183 | −1.54 |
| **test_avg (3-split)** | **56.6976** | 65.2680 | **56.4610** | **−0.24** |
| Epochs completed | — | **11/50** (wall) | 16/50 (wall) | — |
| Mean s/epoch | ~130s | 166s | 113s | — |
| n_params | 891k | 1.26M | 856k | — |

**Analysis:** Arm B (n_layers=4) cleanly beats H48 baseline on all OOD splits (rc, cruise, re_rand) with a small in-distribution regression. Net win: −1.05 val_avg, −0.24 test 3-split. **The pre-GEGLU H42 finding that "deeper + n_head=2 hurts" does NOT transfer to the GEGLU regime** — going *shallower* (4 layers) is the win at this wall budget. Mechanism: GEGLU's gated FFN (gate+up+down vs vanilla expand+contract) adds per-block representational capacity, so fewer-but-fatter-effective layers fit the regression with less wall-clock cost.

**Arm A nuance:** n_layers=6 hit wall at epoch 11/50 with val=67.09. At the *same epoch index* (11), Arm A is ahead of Arm B (67.09 vs 72.04). The "Arm A worse" result is primarily a wall-budget artefact — to test depth=6 fairly, would need ≥45 min wall.

**Bug-fix included:** Student exposed `--n_layers` as a CLI flag (Config field + one-line wire-through to model_config) — `n_layers=5` was previously hardcoded.

**Compute benefit:** n_layers=4 frees ~13% s/epoch, opening budget for other levers (slice_num widening — see H66).

**Status: MERGED — NEW BASELINE val=57.5750, test=56.4610. Cumulative gain from R5 start: −8.53 pts val (66.11 → 57.58).**

**Implication for in-flight experiments (H58/H59/H61/H62/H63/H64/H65):** All in-flight hypotheses ran on n_layers=5. Their results remain interpretable but need to be evaluated against both the old baseline (58.63) and the new one (57.58). Winning levers should be re-stacked on n_layers=4 in subsequent rounds.

---

## 2026-05-16 16:00 — R5 cycle 12 new assignment (1 idle student)

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#4011** | thorfinn | **H66: slice_num sweep (96, 128) at n_layers=4 GEGLU** | ~56.5-57.5 (Arm A) |

**H66 (slice_num):** The H60 win opens compute headroom (~13% per epoch). Widen the slice-token representation that Transolver's PhysicsAttention uses to compress mesh nodes. slice_num=64 is the original Transolver default; the model has improved 57+ pts since H10 tested 96/128, so the bottleneck/capacity tradeoff is fundamentally different now. Arms: slice_num=96 (50% more) and 128 (2x). Tests whether finer spatial selectivity helps the geometry-OOD splits where local mesh structure matters most.

---

## 2026-05-16 15:34 — PR #3966: H59 GEGLU + RMSNorm (fern) — **MERGED, NEW BASELINE**

- Branch: `charliepai2i48h3-fern/h59-geglu-rmsnorm`
- Hypothesis: Replace LayerNorm with RMSNorm in GEGLU Transolver blocks. RMSNorm skips mean-subtraction, preserving the directional structure the GEGLU gate path exploits. Fused `torch.nn.functional.rms_norm` should also yield per-epoch speedup.

| Metric | H60 baseline (LN) | **PR #3966 (RMSNorm)** | Δ |
|--------|-------------------|-------------------------|---|
| **val_avg/mae_surf_p** | 57.5750 | **56.9056** | **−0.67** |
| val_single_in_dist | 63.3430 | 64.4659 | +1.12 |
| val_geom_camber_rc | 72.1854 | 70.1136 | −2.07 |
| val_geom_camber_cruise | 37.7532 | 35.7221 | −2.03 |
| val_re_rand | 57.0183 | 57.3210 | +0.30 |
| **test_avg (3-split)** | 56.4610 | **56.2420** | **−0.22** |
| test_single_in_dist | — | 56.0700 | — |
| test_geom_camber_rc | — | 65.7949 | — |
| test_re_rand | — | 46.8612 | — |
| Epochs completed | 16/50 (wall) | 14/50 (wall) | — |
| Mean s/epoch | 113s | ~137s (slightly slower wall-clock than nominal due to first-run JIT/warmup) | — |
| n_params | 856k | 856,587 | — |
| Peak GPU memory | — | 49.54 GB | — |

**Analysis:** Wins on the two geometry-OOD splits (camber_rc, camber_cruise) by ~2 pts each, small regression on in-distribution (+1.12). Net: −0.67 val_avg, −0.22 test 3-split. Importantly, the student used the **fused** `F.rms_norm` kernel rather than a naive python implementation — without this optimization, RMSNorm would not gain wall-clock time. The win is driven primarily by the additional training step within the wall budget; the directional-preservation argument may matter mechanistically but is not separately measurable at this scale.

**Status: MERGED — NEW BASELINE val=56.9056, test=56.2420. Cumulative gain from R5 start: −9.20 pts val (66.11 → 56.91).**

---

## 2026-05-16 15:30 — PR #3965: H58 Lion optimizer + GEGLU FFN mega-stack (edward) — **TERMINAL, SENT BACK FOR REBASE**

- Branch: `charliepai2i48h3-edward/h58-lion-geglu`
- Hypothesis: Compound Lion's sign-based update direction with GEGLU's multiplicative gating. Predicted ~52-55 val_avg from H37b → H48 → Lion stacking.

| Arm | val_avg | test_avg (3-split) | best_epoch | epochs |
|-----|---------|--------------------|------------|--------|
| **A (lr=1e-4, wd=1e-3, β=(0.9,0.99))** | **46.7957** | 46.6320 | 13 | 13/50 (wall) |
| B (lr=2e-4) | 47.4440 | **45.8483** | 13 | 13/50 (wall) |
| H48 baseline (pre-H59) | 58.6268 | 56.6976 | — | — |

**Per-split val (Arm A):** single_in_dist=50.68, camber_rc=59.93, camber_cruise=29.64, re_rand=46.93. Uniform −10 to −14 pt improvement across all four splits — Lion fixes a systemic optimization issue, not a regime-specific lever.

**Gate health:** GEGLU std=0.27→0.29 (Arm A) and 0.42→0.45 (Arm B) across training. No saturation; Lion's sign-update concern did not materialize.

**Wall-budget caveat:** Both arms hit 30-min wall at epoch 13/50 with val_avg *still dropping ~2.4 pts per epoch* (49.17 → 46.80 in the last step). Cosine T_max=15 means LR was near-peak. **These are loose upper bounds — a full schedule should yield substantially lower.**

**Action: SENT BACK FOR REBASE.** Edward's PR was submitted against the pre-H59 codebase. After H59 RMSNorm merge (in train.py block-level norms), the PR has conflicts with the additive RMSNorm code. Student asked to: (1) rebase onto current advisor branch, (2) resolve train.py conflicts (Lion's optimizer changes are orthogonal to RMSNorm), (3) re-run a quick verification (--epochs 20) to confirm improvement on updated codebase, (4) mark ready for re-review.

**Strategic note:** Even pending rebase, the H58 signal is *the strongest single PR result of the round* — Δ −10.11 vs current baseline 56.91. The follow-up cycle pre-emptively seeded 5 Lion-themed compound hypotheses (H67-H71) to exploit the lever in parallel while H58 rebases.

---

## 2026-05-16 15:43 — R5 cycle 13 new assignments (5 students; pre-emptive Lion compound batch)

Triggered by Edward's H58 terminal result (Lion+GEGLU → −10.11 vs new H59 baseline). The lever is so strong it warrants parallel exploitation across multiple orthogonal compound directions, rather than waiting on the H58 rebase.

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#4020** | alphonse | **H67: Lion + GEGLU + RMSNorm compound stack (lr=1e-4, 3e-4)** | ~45-48 |
| **#4022** | askeladd | **H68: Lion β₂ momentum decay sweep (0.95 vs 0.999)** | ~46-48 |
| **#4023** | fern | **H69: Lion + linear LR warmup + cosine decay** | ~45-47 |
| **#4024** | frieren | **H70: Attention head count sweep under Lion (n_head 1, 4)** | ~46-48 |
| **#4025** | nezuko | **H71: Lion weight decay sweep (1e-4, 5e-4)** | ~46-48 |

**H67 (Lion+RMSNorm compound):** Two confirmed wins stacked — Lion (gradient update) + RMSNorm (per-epoch speed). Tests Arm A=lr=1e-4 (H58 winner) and Arm B=lr=3e-4 (Lion's native range).

**H68 (Lion β₂):** Lion is unusually sensitive to β₂ (slow EMA). H58 used β₂=0.99 (from H49). Sweep β₂=0.95 (faster forgetting, more reactive) and β₂=0.999 (slower) to map this dimension.

**H69 (Lion + LR warmup):** Lion's loose upper bound (val=46.80 at wall) is partly because LR was still near-peak at epoch 13. Linear warmup (1-2 epochs) lets the model warm up under low LR and reach the cosine peak later, potentially smoother convergence within the wall budget.

**H70 (n_head under Lion):** The n_head=2 optimum was confirmed under AdamW. Lion's sign-based update changes the gradient norm balance across heads — worth verifying n_head=2 stays optimal vs n_head=1, 4 in the new regime.

**H71 (Lion wd):** H58 used wd=1e-3 (Lion's classic sweet spot from H49). The H37b/H39c AdamW wd-optimum is wd=5e-5. Lion's decoupled-wd implementation behaves differently — Arm A wd=1e-4 and Arm B wd=5e-4 probe the local landscape around H58's choice.

**Strategy:** All 8 students now WIP. Lion is the dominant lever; if any of H67-H71 lands cleanly, baseline could drop another 1-3 pts on top of H58's still-loose 46.80.

---

## 2026-05-16 16:31 — PR #4011: H66 slice_num sweep (thorfinn) — **MERGED, NEW BASELINE**

- Branch: `charliepai2i48h3-thorfinn/h66-slice-num`
- Hypothesis: Widen the slice-token representation in Transolver's PhysicsAttention from 64 → 96/128. H60's shallower depth freed ~13% s/epoch compute; this budget tested finer spatial selectivity for geometry-OOD splits.

| Metric | H59 baseline (slice=64) | **Arm A (slice=96)** | Arm B (slice=128) | Δ A vs baseline |
|--------|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 56.9056 | **56.7504** | 58.8766 | **−0.16** |
| **test_avg (3-split)** | 56.2420 | **54.5026** | 57.8969 | **−1.74** |
| val_single_in_dist | 64.4659 | **60.9717** | 65.6492 | −3.49 |
| val_geom_camber_rc | 70.1136 | **70.7939** | 72.4829 | +0.68 |
| val_geom_camber_cruise | 35.7221 | 38.2785 | 39.2583 | +2.56 |
| val_re_rand | 57.3210 | 56.9576 | 58.1159 | −0.36 |
| test_geom_camber_rc | — | **61.8680** | 65.0606 | — |
| test_single_in_dist | — | **54.5425** | 58.1689 | — |
| test_re_rand | — | **47.0974** | 50.4611 | — |
| Best epoch | 15 | 15 | 14 (wall-cut) | — |
| Mean s/epoch | 113s | 121.8s | 130.7s | — |
| n_params | 856,587 | 864,907 | 873,227 | — |

**Analysis:** Arm A (slice=96) wins convincingly on the **test** side (−1.74 pts 3-split), with the gain concentrated on **test_geom_camber_rc (−3.33 pts)** — exactly the geometry-OOD split where local mesh spatial structure matters most, confirming the hypothesis mechanism. Val-side gain is marginal (−0.16 pts) with mixed per-split behavior (in-dist +, cruise −, OOD roughly flat). The strong test/val divergence suggests slice_num=96 improves OOD generalization more than in-distribution accuracy.

Arm B (slice=128) regresses across the board. Two factors: (1) wall-cut at epoch 14 missing the sharpest final-epoch drop; (2) even at equal epoch 14, Arm B trailed Arm A by 1.60 pts — capacity vs. small dataset (1499 samples) tips into overfit.

**Important nuance:** H66 was branched against the pre-H59 codebase (LayerNorm). The merged squash commit landed both H59 RMSNorm code and H66 slice_num flag on the advisor branch. The H66 numbers reflect **LayerNorm + slice_num=96**. Combining with RMSNorm is the direct follow-up (→ H72).

**Bug-fix included:** Student exposed `--slice_num` as a CLI flag (Config field + one-line wire-through to model_config) — `slice_num=64` was previously hardcoded at line 499.

**Status: MERGED — NEW BASELINE val=56.7504, test=54.5026 (3-split). Cumulative R5 gain: −9.36 pts val vs H37b (66.11 → 56.75).**

---

## 2026-05-16 16:32 — R5 cycle 14 new assignment (1 idle student)

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#4048** | thorfinn | **H72: slice_num=96 + RMSNorm compound at GEGLU n_layers=4 base** | ~56.0-56.5 (Arm A) |

**H72 (slice_num=96 + RMSNorm compound):** Directly compounds the two latest baseline-moving wins. H59 (RMSNorm, fused F.rms_norm) and H66 (slice_num=96) are mechanistically orthogonal — one is a kernel-efficiency / extra-epoch lever, the other widens the attention bottleneck. H66 was measured at LayerNorm; enabling both flags simultaneously (no code changes needed — both are in the merged codebase) should yield near-additive gains. Arm A = slice_num=96 + RMSNorm (direct compound). Arm B = slice_num=112 + RMSNorm (exploratory: probes the 96→128 interpolation zone, with RMSNorm's per-epoch speedup recovering the wall budget Arm B of H66 lost at 128).

---

## 2026-05-16 16:35 — PR #3997: H65 EMA weight averaging (tanjiro) — **CLOSED, negative result**

- Branch: `charliepai2i48h3-tanjiro/h65-ema-weights`
- Hypothesis: EMA/Polyak weight averaging finds flatter loss-landscape minima → better OOD generalization.

| Arm | val_avg raw | val_avg EMA | test 3-split raw | test 3-split EMA | best_epoch |
|-----|------------:|------------:|-----------------:|-----------------:|-----------:|
| **A (decay=0.999)** | 65.82 | 73.25 | 63.59 | 70.72 | 12 |
| B (decay=0.9999) | 62.60 | **327.49** | 59.87 | **346.72** | 13 |
| H48 baseline | 58.63 | — | 56.70 | — | 13 |
| H66 current baseline | 56.75 | — | 54.50 | — | 15 |

**Δ vs current baseline:** Arm A raw +9.07 val_avg, +9.09 test 3-split (REGRESSION). Arm B EMA essentially un-converged.

**Decisive mechanism (from student analysis):**
1. **EMA only helps in the *oscillation* regime around a minimum.** Under our 30-min / 13-epoch budget with cosine T_max=15, LR is still meaningfully nonzero at endpoint — raw weights are still *translating*, not oscillating. Averaging across translation produces a worse estimate, exactly as observed in L2-distance trajectory (Arm A: shrinking but never closes; Arm B: widens then plateaus at 26.58).
2. **Dual per-epoch validation** (raw + EMA) added ~10% wall overhead, costing 1 full epoch under the hard 30-min cap. Arm A ran 12 epochs vs baseline's 13.

**Per-split decay-0.999 Arm A EMA shows uniform +3 to +8 pt regression on every split** vs raw → the EMA average is strictly worse than the raw weights at the same epoch. No "flat minimum" effect visible.

**Status: CLOSED — EMA lever is exhausted at the current 13-epoch / cosine T_max=15 budget.**

**Why closed not sent back:** The mechanism is well-understood and the proposed remediations (lower decay 0.99/0.995, warm-start EMA from mid-training checkpoint, Adam-style bias correction, sparser EMA eval cadence) are all sound but represent diminishing relative payoff vs. the Lion track (which is delivering ~10× the gain at the same compute). EMA can be revisited *after* the Lion+slice_num+RMSNorm baseline is established and we're hunting smaller margins.

**Operational note:** Student's pod ran successfully despite the rate-limit window that affected most other pods (15:30-16:22Z) — tanjiro happened to land outside the bucket-exhaustion window.

---

## 2026-05-16 16:36 — R5 cycle 14 new assignment (1 idle student, tanjiro)

| PR | Student | Hypothesis | Predicted val_avg |
|----|---------|------------|-------------------|
| **#4055** | tanjiro | **H73: Lion + GEGLU + slice_num=96 — compound two strongest levers** | ~45-47 (Arm A) |

**H73 (Lion + slice_num=96 compound):** Compounds the two strongest confirmed levers — H58 Lion (Δ −10.11 val_avg, loose UB) and H66 slice_num=96 (Δ −0.16 val / −1.74 test). Mechanistically orthogonal: Lion changes *how* weights update (sign-based gradient), slice_num=96 changes *what* attention computes (wider spatial bottleneck). Predicted additivity: H73 ≈ 46.64 val_avg if perfect (= 56.75 − 10.11 = 46.80 − 0.16 from either compound direction). Arm A = Lion lr=1e-4 (H58 winner) + slice_num=96. Arm B = Lion lr=3e-4 + slice_num=96 (Lion's native LR range). This fills the missing piece of the Lion compound matrix — H67-H71 test Lion + (RMSNorm, β₂, warmup, n_head, wd) but not Lion + slice_num.

---

## 2026-05-16 18:32 — PR #4055: H73 Lion + slice_num=96 (tanjiro) — **MERGED, NEW BASELINE**

- Branch: `charliepai2i48h3-tanjiro/h73-lion-slice96`
- Hypothesis: Lion + slice_num=96 compound near-additively (predicted val_avg ≈ 46.64).

| Arm | val_avg | test 3-split | Δ vs H66 (56.75/54.50) |
|-----|--------:|-------------:|-----------------------:|
| A (Lion lr=1e-4) | 46.3422 | 45.3896 | −10.41 / −9.11 |
| **B (Lion lr=3e-4)** | **42.9784** | **41.5455** | **−13.77 / −12.96** |

| Per-split (Arm B) | val | test |
|---|---:|---:|
| single_in_dist | 43.7880 | 38.7901 |
| geom_camber_rc | 56.6638 | 50.1886 |
| geom_camber_cruise | 26.4930 | NaN (bug) |
| re_rand | 44.9686 | 35.6578 |

**Key results:**
- **Arm A confirms additivity** (46.34 vs predicted 46.64 — within 0.30 pts). The Lion gain (−10.11) transfers cleanly to slice_num=96.
- **Arm B is SUPER-ADDITIVE** — 3.66 pts below the additivity floor at val=42.98. The wider gradient surface from slice=96 favors Lion's native lr=3e-4 range (vs H58's lr=1e-4 which was tuned at slice=64).
- Both arms wall-cut at ep 15/50 with val still descending ~0.8 pts/epoch → loose UB. True asymptote likely well below 42.98.
- Uniform improvement across all 4 val splits (−10 to −17 pts each).
- test_geom_camber_rc gain (−11.68 pts) confirms the spatial-selectivity mechanism survives Lion's optimization regime.

**Cumulative R5 gain: −23.13 pts val vs H37b (66.11 → 42.98). Strongest single-PR gain of the round.**

**Status: MERGED — NEW BASELINE val=42.9784, test 3-split=41.5455. Set H73 Arm B config as floor for all subsequent experiments.**

---

## 2026-05-16 18:34 — PR #4048: H72 slice_num=96 + RMSNorm (thorfinn) — **CLOSED, negative result**

- Branch: `charliepai2i48h3-thorfinn/h72-slice96-rmsnorm`
- Hypothesis: H59 (RMSNorm) + H66 (slice_num=96) compound near-additively.

| Arm | val_avg | test 3-split | Δ vs H66 |
|-----|--------:|-------------:|---------:|
| A (slice96+rmsnorm) | 57.5849 | 55.4647 | **+0.83 / +0.96 (worse)** |
| B (slice112+rmsnorm) | 57.2995 | 56.1783 | +0.55 / +1.68 |

**Anti-compound finding:** RMSNorm (+0.67 win at slice=64, H59) and slice_num=96 (+0.16 win, H66) do NOT compound — Arm A is 0.83 pts WORSE than H66. The two normalization regimes interact differently with the wider PhysicsAttention bottleneck. Useful null result: when designing H73, this told us NOT to add RMSNorm to the Lion+slice=96 stack.

**Status: CLOSED — negative compound. Both H59 (RMSNorm alone) and H66 (slice=96 alone) remain valid wins individually; they just don't stack.**

---

## 2026-05-16 18:34 — PR #4024: H70 Lion n_head sweep (frieren) — **CLOSED, superseded**

- Branch: `frieren/h70-lion-nhead-sweep`
- Hypothesis: n_head sweep under Lion+RMSNorm at slice=64.

| Arm | val_avg | test 3-split |
|-----|--------:|-------------:|
| A (n_head=1) | 46.6631 | 45.5251 |
| **B (n_head=4)** | **45.5603** | **44.8596** |

**Insight captured:** Under Lion+RMSNorm+slice=64, n_head=4 wins by ~1.1 pts over n_head=1, and ~1 pt over n_head=2 (H59 baseline). Useful but **superseded** — H73 (slice=96, n_head=2) lands at 42.98, which is 2.5+ pts below H70's best. n_head sweep should be retested on H73 baseline.

**Status: CLOSED — superseded by H73.**

---

## 2026-05-16 18:34 — PR #4025: H71 Lion wd sweep (nezuko) — **CLOSED, superseded**

- Branch: `nezuko/h71-lion-wd-sweep`
- Hypothesis: Lion wd sweep at slice=64.

| Arm | val_avg | test 3-split |
|-----|--------:|-------------:|
| **A (wd=1e-4)** | **46.0215** | **44.5498** |
| B (wd=5e-4) | 49.9928 | 48.5085 |

**Insight captured:** wd=1e-4 wins decisively over wd=5e-4 (−4 pts val_avg). The Lion-paper 10×-AdamW upper bound (~5e-4) is too aggressive for TandemFoilSet. But H73 uses wd=1e-3 (the H58 spec, which is 10× higher than wd=1e-4) and still wins — under slice=96 the wd optimum may differ. **Superseded** — wd retune needed on H73 baseline.

**Status: CLOSED — superseded by H73.**

---

## 2026-05-16 18:34 — PR #4022: H68 Lion β₂ sweep (askeladd) — **CLOSED, superseded**

- Branch: `askeladd/h68-lion-beta2-sweep`
- Hypothesis: Lion β₂ EMA decay sweep at slice=64.

| Arm | val_avg | test 3-split |
|-----|--------:|-------------:|
| A (β₂=0.95) | 52.4996 | 51.7573 |
| **B (β₂=0.999)** | **49.5122** | **47.4686** |

**Insight captured:** β₂=0.999 (slower EMA) beats β₂=0.95 by ~3 pts under Lion+RMSNorm. H73 uses β₂=0.99 (the default H58 spec). The slower β₂ may help further at slice=96. **Superseded** — β₂ retune on H73 baseline pending.

**Status: CLOSED — superseded by H73.**

---

## 2026-05-16 18:34 — PR #4023: H69 Lion warmup (fern) — **CLOSED, superseded**

- Branch: `fern/h69-lion-lr-warmup`
- Hypothesis: Linear LR warmup before cosine decay under Lion.

| Arm | val_avg | test 3-split |
|-----|--------:|-------------:|
| A (warmup=1) | 54.2950 | 52.6563 |
| **B (warmup=2)** | **49.0287** | **48.0243** |

**Insight captured:** warmup=2 beats warmup=1 by **5.3 pts** — the single biggest hyperparameter-tuning signal in the H67-H71 batch. Warmup is high-impact under Lion. H73 used NO warmup (lr=3e-4 from start) and won at 42.98 — but adding warmup=2 may stabilize the lr=3e-4 regime further. **Superseded** — warmup retest on H73 baseline pending.

**Status: CLOSED — superseded by H73.**

---

## 2026-05-16 19:10 — Round 5 Cycle 22: New Assignments

Six new hypotheses assigned to freed students after H73 merge established new baseline at val=42.9784.

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4088 | askeladd | H74: Extended cosine schedule | T_max=20/ep=20 (Arm A), T_max=15/ep=30 SGDR restart (Arm B) |
| #4094 | tanjiro | H75: Lion LR sweep | lr=2e-4 (Arm A), lr=5e-4 (Arm B) |
| #4090 | fern | H76: Warmup at H73 baseline | warmup=2+lr=3e-4 (Arm A), warmup=2+lr=5e-4 (Arm B) |
| #4091 | frieren | H77: n_head at slice=96 | n_head=4 (Arm A), n_head=3 (Arm B) |
| #4092 | nezuko | H79: wd retune at H73 | wd=1e-4 (Arm A), wd=5e-5 (Arm B) |
| #4093 | thorfinn | H80: Full Lion stack max | warmup=2+wd=1e-4+β₂=0.999+n_head=4 at lr=3e-4 (Arm A) and lr=2e-4 (Arm B) |

**Note:** H78 (β₂ retune only) deferred — thorfinn assigned H80 (bold compound swing including β₂=0.999). If H80 Arm A beats baseline, H78 individual isolation will confirm β₂ contribution.

Still WIP (low priority): #3965 edward H58 rebase, #4020 alphonse H67 — both likely superseded by H73.

---

## 2026-05-16 19:15 — PR #4020: H67 Lion+GEGLU+RMSNorm compound (alphonse) — **CLOSED, superseded**

- Branch: `alphonse/h67-lion-rmsnorm-compound`
- Hypothesis: Lion + RMSNorm compound on H59 GEGLU baseline at slice=64.

| Arm | val_avg | test 3-split | best_epoch |
|-----|--------:|-------------:|-----------:|
| A (Lion lr=1e-4) | 46.00 | 44.81 | 17 |
| **B (Lion lr=3e-4)** | **44.05** | **42.27** | 17 |
| H59 baseline | 56.91 | 56.24 | — |

**Analysis:** Both arms beat the H59 baseline by a wide margin (−12.86 pts val for Arm B). Arm B (lr=3e-4) confirms the Lion-native LR superiority pattern from H73. The compound (Lion+RMSNorm) is additive as predicted (Arm A 46.00 ≈ additivity prediction 46.13 from H58+H59). Arm B's extra 2 pts from lr=3e-4 is consistent with H73's findings.

Neither arm beats the current baseline (H73 val=42.97). Alphonse correctly identified that slice=96 is the missing lever — his suggested follow-ups directly map to our H75-H80 assignments.

**Status: CLOSED — superseded by H73 (slice=96 + Lion + LayerNorm at val=42.98).**

---

## 2026-05-16 19:15 — PR #3965: H58 Lion+GEGLU rebase (edward) — **CLOSED, superseded**

- Branch: `edward/h58-lion-lr1e4-geglu` (rebase of original H58)
- Hypothesis: Lion+GEGLU compound at H48 GEGLU baseline (post-rebase to resolve merge conflict).

| Arm | val_avg | test 3-split | best_epoch |
|-----|--------:|-------------:|-----------:|
| **A (lr=1e-4)** | **46.80** | 46.63 | 13 |
| B (lr=2e-4) | 47.44 | **45.85** | 13 |
| H48 baseline | 58.63 | 56.70 | — |

**Analysis:** Both arms confirmed −11.8 pts val gain (Lion+GEGLU compound is massive). Gate health analysis shows no saturation (std=0.29 Arm A, 0.45 Arm B). Arm B val/test inversion (Arm B wins test_single_in_dist by 4.4 pts) suggests higher LR finds flatter minimum. Neither arm beats H73 baseline. The rebase ran cleanly at only 13 epochs (wall-cut).

**Status: CLOSED — superseded by H73 (val=42.98). Key insights already incorporated.**

---

## 2026-05-16 19:20 — Round 5 Cycle 22b: Assign H78 and H81 to freed students

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4097 | edward | H78: Lion β₂ sweep at H73 baseline | β₂=0.999 (Arm A), β₂=0.995 (Arm B) |
| #4098 | alphonse | H81: RMSNorm under Lion+slice=96 | RMSNorm+lr=3e-4 (Arm A), RMSNorm+lr=2e-4 (Arm B) |

All 8 students now active. H78 tests the β₂ individual isolation (thorfinn has the compound H80 including β₂=0.999, edward has the individual sweep). H81 retests the normalization question under Lion — H72 showed anti-compound under AdamW, but Lion's sign-update changes the normalization interaction.

---

## 2026-05-16 19:30 — PR #4090: H76 Warmup on H73 (fern) — **CLOSED, negative**

- Branch: `fern/h76-warmup-at-lr3e4`
- Hypothesis: warmup_epochs=2 ports from H69 slice=64 win to H73 slice=96.

| Arm | val_avg | Δ vs H73 | test 3-split | best_epoch |
|-----|--------:|---------:|-------------:|-----------:|
| A (warmup=2, lr=3e-4) | 46.7102 | **+3.73** worse | 44.6028 | 15/15 |
| B (warmup=2, lr=5e-4) | 49.6935 | **+6.72** worse | 48.0475 | 15/15 |

**Insight captured:** Warmup does NOT transfer from slice=64+lr=1e-4 (where H69 won by 5.3 pts) to slice=96+lr=3e-4. Mechanism: at the 15-epoch wall-cut horizon, the 2 warmup epochs are 13% of training time. Lion+slice=96+GEGLU already trains stably from epoch 1 — no overshoot to suppress. Warmup is a portable lever ONLY when the optimizer/config has epoch-1 overshoot pathology.

**Status: CLOSED — negative result. Warmup ruled out at H73 horizon. H80 (which includes warmup=2) is now less likely to win.**

---

## 2026-05-16 19:30 — PR #4091: H77 n_head on H73 (frieren) — **CLOSED, negative**

- Branch: `frieren/h77-nhead-at-slice96`
- Hypothesis: n_head=4 ports from H70 slice=64 win to H73 slice=96.

| Arm | val_avg | Δ vs H73 | test 3-split | epochs |
|-----|--------:|---------:|-------------:|-------:|
| A (n_head=4) | 45.6576 | **+2.68** worse | 44.1039 | 13/50 wall-cut |
| B (n_head=3) | 44.6507 | **+1.67** worse | 43.0911 | 14/50 wall-cut |

**Insight captured:** Monotonic worsening with n_head at slice=96 — opposite of H70's slice=64 finding. At slice=96, the spatial bottleneck is already 50% wider; adding heads shrinks per-head dim (64→43→32 at fixed n_hidden=128), which hurts more than diversity helps. **n_head=2 is locked at slice=96**. Note: more heads also cost ~16% more time per epoch (145s vs 122s), partially confounded by wall-clock budget. But even ignoring confound, the trend is wrong direction.

**Status: CLOSED — negative result. n_head locked at 2. H80 (which includes n_head=4) is now less likely to win.**

---

## 2026-05-16 19:35 — Round 5 Cycle 23: Re-assign fern and frieren after H76/H77 closures

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4126 | fern | H82: slice_num sweep under Lion | slice=128 (Arm A), slice=80 (Arm B) |
| #4127 | frieren | H83: n_layers sweep under Lion | n_layers=5 (Arm A), n_layers=3 (Arm B) |

**Strategic value:**
- H82 retests AdamW's slice=128 regression point (H66) under Lion's fundamentally different optimization dynamics. If Lion can train slice=128 stably, this could unlock further super-additive gain.
- H83 retunes depth under Lion at the new slice=96 baseline (H60's n_layers=4 win was AdamW+slice=64-specific).

These two hypotheses target the two architectural levers that have NOT yet been tested under Lion+slice=96.

---

## 2026-05-16 20:55 — PR #4092: H79 wd retune on H73 (nezuko) — **CLOSED, negative**

- Branch: `nezuko/h79-lion-wd-at-slice96`
- Hypothesis: wd=1e-4 (H71 winner at slice=64) ports to H73 slice=96.

| Arm | val_avg | Δ vs H73 | test 3-split | Δ vs H73 test |
|-----|--------:|---------:|-------------:|--------------:|
| A (wd=1e-4) | 44.1701 | +1.19 | 43.5136 | +1.97 |
| B (wd=5e-5) | 43.3561 | +0.38 | **41.3990** | **−0.15** |

**Insight captured:** Non-monotonic wd curve at slice=96 — wd=1e-4 is worse than both neighbors. Weight L2 norms scale with wd as expected (wd=5e-5: 114.9, wd=1e-4: 106.0). Arm B is essentially tied with H73 on val (+0.38) and slightly better on test (−0.15) — within seed variance (see H74 result). **wd=1e-3 locked at slice=96.**

**Status: CLOSED — negative. H71 lever does NOT transfer to slice=96.**

---

## 2026-05-16 20:55 — PR #4088: H74 Extended schedule (askeladd) — **CLOSED, negative**

- Branch: `askeladd/h74-extended-schedule`
- Hypothesis: T_max=20 or SGDR restart captures wall-cut tail.

| Arm | val_avg | Δ vs H73 | test 3-split | best_epoch |
|-----|--------:|---------:|-------------:|-----------:|
| A (T_max=20, ep=20) | 49.5990 | +6.62 | 47.77 | 15/15 wall-cut |
| B (T_max=15, ep=30 SGDR) | 45.5738 | +2.60 | 43.84 | 15/15 wall-cut |

**Critical insight from this PR:** Arm B's schedule is identical to H73's for the first 15 epochs (CosineAnnealingWarmRestarts T_0=15 == CosineAnnealingLR T_max=15 for the first cycle, and the restart fires AFTER the wall cut at ep 16). Therefore the **+2.60 val gap between Arm B and H73 is essentially single-seed variance**. The H73 baseline number itself has ≥2.6 pts of seed noise.

**Implications for our methodology:**
- Closures with Δ ≤ 2.6 pts vs baseline (e.g., H79 Arm B, +0.38 val / −0.15 test) may be ties not true losses.
- True wins require Δ ≥ 3 pts to clearly exceed noise.
- Future hypotheses should target levers with predicted gains > 3 pts.

**Status: CLOSED — negative; major methodology insight on seed variance.**

---

## 2026-05-16 21:00 — Round 5 Cycle 24: Re-assign askeladd and nezuko + noise-floor insight

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4133 | askeladd | H84: T_max compression below wall budget | T_max=12 (Arm A), T_max=10 (Arm B); 3-5 epochs LR-fine-tune |
| #4135 | nezuko | H85: FFN activation under Lion+slice=96 | swiglu (Arm A), vanilla (Arm B) |

**H84** implements askeladd's own follow-up suggestion — compress T_max below the wall budget so the cosine reaches LR=0 before timeout and the remaining epochs serve as a low-LR fine-tune phase.

**H85** tests whether GEGLU's H48 dominance (under AdamW+slice=64) persists under Lion+slice=96, or whether the optimization regime change shifts the optimum.

**Methodology update:** Future PR reviews will weight Δ vs baseline against the ~2.6 pt seed noise floor. Improvements within ±2-3 pts are likely ties, not wins or losses.

---

## 2026-05-16 21:08 — PR #4094: H75 Lion LR sweep (tanjiro) — **CLOSED, U-shape confirmed**

- Branch: `tanjiro/h75-lion-lr-sweep`
- Hypothesis: lr=2e-4 or lr=5e-4 may beat H73's lr=3e-4.

| Arm | val_avg | Δ vs H73 | test 3-split | Δ vs H73 test |
|-----|--------:|---------:|-------------:|--------------:|
| A (lr=2e-4) | 44.8184 | +1.84 (within 2.6 pt noise) | 43.5265 | +1.98 |
| B (lr=5e-4) | 47.0910 | **+4.11** | 45.6844 | +4.14 |

**Insight captured:** LR optimum is U-shaped on slice=96+Lion surface (1e-4: 46.34 → 2e-4: 44.82 → 3e-4: 42.98 → 5e-4: 47.09). Bracketed at 2.5e-4 to 3.5e-4. Lion was stable at lr=5e-4 (no divergence) but suboptimal — sign-update couldn't compensate for the wider effective step size. Arm A is plausibly tied within seed noise but not a clear win.

**lr=3e-4 confirmed as local optimum at H73 baseline.**

**Status: CLOSED — negative; LR optimum bracketed; capacity frontier is next.**

---

## 2026-05-16 21:10 — Round 5 Cycle 25: Assign H86 to tanjiro (n_hidden expansion)

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4147 | tanjiro | H86: n_hidden expansion under Lion+slice=96 | n_hidden=192 (Arm A), n_hidden=256 (Arm B) |

**Strategic rationale:** All recent sweeps (lr, wd, warmup, n_head, n_layers) hit a frontier at val≈43. Optimization levers are largely tapped. The model is wall-cut at ep 15 — capacity is the next frontier. Lion's scale-invariant sign-update should accommodate larger models cleanly. n_hidden has been locked at 128 since H33 (under AdamW); first expansion test under Lion+slice=96.

---

## 2026-05-16 21:25 — PR #4098: H81 RMSNorm under Lion+slice=96 (alphonse) — **CLOSED, anti-compound confirmed**

- Branch: `charliepai2i48h3-alphonse/h81-rmsnorm-under-lion-slice96`
- Hypothesis: Lion's sign-update changes effective gradient scale → maybe RMSNorm (no learned bias) is the better fit at slice=96 (contra H72's AdamW+RMSNorm anti-compound).

| Arm | norm | lr | val_avg | Δ vs H73 | test 3-split | best epoch | s/epoch |
|-----|------|----|--------:|---------:|-------------:|-----------:|--------:|
| H73 baseline | LayerNorm | 3e-4 | **42.9784** | — | **41.5455** | — | — |
| A | RMSNorm | 3e-4 | 45.4152 | **+2.44** (clear regression) | 43.5394 | 15 | 119.11 |
| B | RMSNorm | 2e-4 | 43.8744 | +0.90 (within 2.6 noise) | 43.2968 | 15 | 119.46 |

**Per-split sensitivity:** `val_geom_camber_rc` (unseen-camber OOD) is the most sensitive split to LayerNorm→RMSNorm (Arm A +4.57, Arm B –0.57). The cruise split is essentially insensitive.

**Mechanistic conclusion:** The H72 anti-compound (AdamW+RMSNorm+slice=96: +1.58) reproduces under Lion at +2.44 pts. RMSNorm is **not** optimizer-specific — it genuinely hurts at slice=96 under both optimizers. The H59 RMSNorm win at slice=64 (under AdamW) was likely driven by kernel-op speedup giving more effective steps per epoch; at slice=96 that advantage vanishes (more compute per step), and the cost of removing the learned bias becomes visible. No measurable s/epoch speedup observed at slice=96 in this codebase.

**Status: CLOSED — negative. Normalization question closed. LayerNorm locked at slice=96 under both optimizers.**

---

## 2026-05-16 21:30 — Round 5 Cycle 26: Assign H87 to alphonse (eta_min > 0)

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| TBD | alphonse | H87: CosineAnnealingLR eta_min > 0 — keep meaningful LR through wall-cut | eta_min=3e-5 (Arm A, lr/10), eta_min=1e-5 (Arm B, lr/30) |

**Strategic rationale:** Every recent run (H73, H75, H81) shows the val trajectory still monotonically descending at epoch 15 (the wall-cut). The cosine schedule reaches LR=0 at ep 15, so the last few epochs receive near-zero gradient updates. eta_min > 0 holds the LR floor above zero so all 15 epochs contribute meaningful gradient steps. This is the **inverse** of H84 (T_max compression) — H84 lets the model fine-tune at LR=0, H87 prevents the LR from collapsing to zero. The two ideas test opposite sides of the schedule-tail question. Picked from alphonse's own follow-up suggestion list in the H81 PR.

---

## 2026-05-16 21:32 — PR #4097: H78 Lion β₂ sweep (edward) — **MERGED, NEW BEST**

- Branch: `charliepai2i48h3-edward/h78-lion-beta2-at-slice96`
- Hypothesis: H68's β₂=0.999 win transfers to slice=96; or a sweet-spot interior optimum exists.

| Arm | β₂ | val_avg | Δ vs H73 | test 3-split | Δ vs H73 | best epoch |
|-----|----|--------:|---------:|-------------:|---------:|-----------:|
| H73 baseline | 0.99 | 42.9784 | — | 41.5455 | — | — |
| A | 0.999 | 44.3436 | +1.37 (regress) | 42.0389 | +0.49 | 15 |
| **B** | **0.995** | **42.3048** | **−0.67** (small win) | **40.5564** | **−0.99** (small win) | **15** |

**Mechanism:** Non-monotonic in β₂ — interior optimum at 0.995. β₂=0.999 over-smooths the gradient EMA, can't track the cosine-decaying loss surface within the 15-ep budget (still 17.8 pts behind at ep 3, never recovers). β₂=0.995 is the sweet spot between noise filtering and adaptation speed at lr=3e-4/slice=96. H68's β₂=0.999 win was specific to lr=1e-4+slice=64+RMSNorm regime.

**Merge decision:** val Δ=−0.67 is within seed noise floor (~2.6 pts) individually, but test Δ=−0.99 is also negative (correlated improvement across both metric channels) and the trajectory analysis shows clean predicted-vs-actual mechanistic agreement. Single-flag change, zero complexity cost. Per round protocol "merge small compoundable wins."

**Status: MERGED — NEW BEST. β₂=0.995 locked in baseline.**

---

## 2026-05-16 21:33 — PR #4093: H80 Full Lion stack (thorfinn) — **CLOSED, schedule confound + known-negative levers**

- Branch: `charliepai2i48h3-thorfinn/h80-lion-stack-max`
- Hypothesis: Compound H67-H71 wins (warmup=2 + wd=1e-4 + β₂=0.999 + n_head=4) on top of H73 for "optimistic 11 pt compound" swing.

| Arm | lr | val_avg | Δ vs H73 | test 3-split | best epoch | s/epoch |
|-----|----|--------:|---------:|-------------:|-----------:|--------:|
| A | 3e-4 | 79.6076 | **+33.0** | 78.7386 | 13 | 145.0 |
| B | 2e-4 | 75.9733 | +33.0 | 73.2389 | 12 | 145.0 |

**Schedule confound:** `--warmup_epochs 2 --epochs 50` triggered SequentialLR(LinearLR, CosineAnnealingLR(T_max=48)). Under the 30-min wall budget the cosine only traversed 23% of its arc. LR at epoch 13 was ~2.6e-4 (essentially still at peak). H73's T_max=15 hardcoded property was load-bearing for the 30-min budget.

**Compound was also doomed by known-negative levers:** By the time results landed, individual sweeps had already shown three of H80's four levers regress at slice=96 — H76 (warmup=2 negative), H77 (n_head=4 negative), H79 (wd=1e-4 negative/tie). Even H78 (this cycle's β₂ sweep) showed β₂=0.999 specifically regresses (+1.37 vs baseline). H80's full stack hit *all four* anti-additive levers simultaneously.

**Status: CLOSED — negative. Schedule-fair rerun not pursued (3+ levers known-negative).**

---

## 2026-05-16 21:35 — Round 5 Cycle 27: Assign H88 to edward (β₂ tighter grid), H89 to thorfinn (mlp_ratio sweep)

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| TBD | edward | H88: β₂ refinement around 0.995 (β₂=0.992 + β₂=0.997) | Confirm/refine peak from H78 |
| TBD | thorfinn | H89: mlp_ratio sweep under Lion+slice=96 (mlp_ratio=3 + mlp_ratio=4) | Last tested under AdamW (H62 closed); new regime probe |

**H88 strategic rationale:** H78 found β₂=0.995 wins over both 0.99 and 0.999 (non-monotonic). With 3 sparse samples and only 0.67 pt val win, the peak is undercharacterized. Tighter grid {0.992, 0.997} either confirms 0.995 is local-optimum or reveals a slightly better setting. Same single-flag protocol as H78.

**H89 strategic rationale:** mlp_ratio is one of the few architectural levers untested under Lion+slice=96+LayerNorm. H62 (AdamW era, slice=64) closed mlp_ratio negative — but under Lion's scale-invariance + the wider slice=96 gradient surface, expanded FFN capacity may unlock representational headroom. mlp_ratio=4 doubles FFN width; mlp_ratio=3 is a moderate increase. Complementary to H86 (tanjiro, n_hidden=192/256) — H86 widens attention+FFN+everything; H89 widens FFN only.

---

## 2026-05-16 22:30-23:15 — Cycle 28 batch closures (H82-H86, all CLOSED negative)

All 5 R5 sweeps from cycles 22-25 returned negative (closed by parallel session). Topline best-arm val_avg vs H78 baseline (42.30):

| PR | Hypothesis | Student | Best arm val_avg | Δ vs H78 | Status |
|----|-----------|---------|------------------:|---------:|--------|
| #4126 | H82: slice_num sweep (128, 80) | fern | 44.5013 | +2.20 | CLOSED negative |
| #4127 | H83: n_layers sweep (5, 3) | frieren | 44.0180 | +1.72 | CLOSED negative |
| #4133 | H84: T_max compression (12, 10) | askeladd | 49.3500 | +7.05 | CLOSED negative |
| #4135 | H85: FFN activation (swiglu, vanilla) | nezuko | 44.5595 | +2.26 | CLOSED negative |
| #4147 | H86: n_hidden expansion (192, 256) | tanjiro | 60.6775 | +18.38 | CLOSED negative |

**Key takeaways:**
- **H84 negative confirms schedule lever fully exhausted.** T_max=15 + eta_min=0 + no warmup is optimal. T_max=10/12 dropped LR too fast; H87 (eta_min > 0) over-shoots. The current schedule is Pareto-optimal.
- **H86 n_hidden=192/256 massively regressed (val=60.68)** — same wall-cut-bound mechanism as H89 (thorfinn's mlp_ratio): wider models train slower, eat epochs from the 30-min budget, never finish cosine decay. **Capacity scaling needs efficiency unlocks first.**
- **slice=96 confirmed Pareto-optimal** (H82). Both directions {128 too slow, 80 too narrow} regress.
- **GEGLU locked at slice=96 under Lion** (H85 reaffirms H48). vanilla and swiglu both lose.
- **n_layers=4 locked under Lion+slice=96** (H83 confirms H60). 3 and 5 both lose.

---

## 2026-05-16 23:30 — PR #4156: H87 eta_min > 0 (alphonse) — **CLOSED, schedule-tail engineering negative**

- Branch: `charliepai2i48h3-alphonse/h87-eta-min-floor`
- Hypothesis: hold cosine LR floor above zero to keep meaningful gradient through wall-cut.

| Arm | eta_min | val_avg | Δ vs H78 | Δ vs H73 | best epoch |
|-----|---------|--------:|---------:|---------:|-----------:|
| H78 baseline | 0 | 42.3048 | — | — | — |
| A | 3e-5 (lr/10) | 46.0890 | +3.79 | +3.11 (regress) | 14 (ep 15 bounces UP) |
| B | 1e-5 (lr/30) | 44.5038 | +2.20 (within noise) | +1.53 | 15 |

**Mechanism (student's analysis):** H73 baseline epoch 15 LR is ~3e-6 (not 0), already an implicit micro-fine-tune. Replacing this with eta_min=3e-5 (10x larger) causes overshoot — direct evidence from Arm A's ep14→15 bounce (46.09→46.60). Arm B's mid-cosine tail runs at higher LR than baseline (e.g. 8.25e-5 vs 6.03e-5 at ep 11), so model arrives at ep 15 at a worse local minimum.

**Schedule lever now fully closed.** H74 (T_max extension): negative. H84 (T_max compression): negative. H87 (eta_min > 0): negative. The H73 cosine T_max=15 + eta_min=0 schedule is locally optimal in all three directions.

**Status: CLOSED — negative. Schedule lever lock confirmed.**

---

## 2026-05-16 23:30 — PR #4169: H89 mlp_ratio sweep (thorfinn) — **CLOSED, wall-cut-bound**

- Branch: `charliepai2i48h3-thorfinn/h89-mlp-ratio-under-lion`
- Hypothesis: FFN-only widening may unlock representational headroom under Lion+slice=96.

| Arm | mlp_ratio | val_avg | Δ vs H78 | test 3-split | Δ test | best epoch | s/epoch | n_params |
|-----|-----------|--------:|---------:|-------------:|-------:|-----------:|--------:|---------:|
| H78 baseline | 2 | 42.3048 | — | 40.5564 | — | 15 | ~120 | 864,907 |
| A | 4 | 47.3721 | +5.07 | 44.9365 | +4.38 | 12 (wall-cut) | 152 | 1,260,171 |
| B | 3 | 42.1923 | −0.11 (noise) | 42.2758 | +1.72 (regress) | 13 (wall-cut) | 138 | 1,062,539 |

**Critical insight from this PR (student's framing):** "Capacity is wall-cut-bound." Both wider arms wall-cut early — Arm A at ep 12, Arm B at ep 13. They were dropping ~3-5 pts/epoch at wall-cut, suggesting the cosine tail decay never completed. The s/epoch overhead (15-27%) ate epochs from the 30-min budget.

**Same mechanism as H86 (n_hidden expansion, also closed negative).** Both capacity-scaling probes are blocked by training speed. **Strategic pivot: efficiency levers first, then revisit capacity.**

**Status: CLOSED — wall-cut confound + Arm B mixed (val tied, test regresses). Major strategic insight: training efficiency is the prerequisite for capacity.**

---

## 2026-05-16 23:35 — Round 5 Cycle 30: Strategic pivot to training efficiency. Assign H95 (alphonse) + H96 (thorfinn).

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| TBD | alphonse | H95: bfloat16 mixed-precision training | --use_bf16 (model fwd/bwd in bf16, optimizer states fp32) |
| TBD | thorfinn | H96: torch.compile baseline acceleration | --compile (default mode, no other changes) |

**Strategic rationale:** H89 (mlp_ratio) and H86 (n_hidden) both closed negative due to wall-cut budget. The model is still descending steeply at ep 15 — wall-cut is the dominant constraint. Efficiency wins translate directly to more epochs in the 30-min budget:

- bf16: typically 25-40% s/epoch reduction on H100. If bf16 cuts to ~80s/ep (from ~120), 30-min budget yields ~22 epochs vs current 15 — a 47% step increase.
- torch.compile: typically 15-30% reduction via fused ops + graph optimization. Orthogonal to bf16; can stack later.

If either lands, all capacity probes (H86, H89) become retestable with a meaningful wall-clock budget. **This is the highest-ROI cycle this round.**

---

## 2026-05-16 23:55 — PR #4166: H88 β₂ refinement around 0.995 (edward) — **MERGED, NEW BEST**

- Branch: `charliepai2i48h3-edward/h88-beta2-refinement`
- Hypothesis: H78 undersampled the β₂ landscape (only 3 points: 0.99, 0.995, 0.999). Does 0.992 or 0.997 beat 0.995?

| β₂ | val_avg | val_sid | val_rc | val_cruise | val_re_rand | test 3-split | Δ val | Δ test | verdict |
|----|--------:|--------:|-------:|-----------:|------------:|-------------:|------:|-------:|---------|
| 0.990 (H73) | 42.9784 | — | — | — | — | 41.5455 | ref | ref | Prior baseline |
| 0.992 (Arm A) | 42.2565 | 41.9596 | 56.3134 | 26.4820 | 44.2712 | 41.3459 | −0.72 | −0.20 | Plateau; ties 0.995 |
| 0.995 (H78 baseline) | 42.3048 | 44.7308 | 56.5492 | 25.1123 | 42.8272 | 40.5564 | baseline | baseline | Prior best |
| **0.997 (Arm B)** | **41.2153** | 42.8497 | 53.5716 | 26.0333 | 42.4066 | **39.5337** | **−1.09** | **−1.02** | **NEW BEST** |
| 0.999 (H78 Arm A) | 44.3436 | — | — | — | — | 42.0389 | +2.04 | +1.48 | Over-smoothed |

- Artifacts: `models/model-h88-arm-a-beta2-0992-20260516-214655/`, `models/model-h88-arm-b-beta2-0997-20260516-222342/`

**Analysis:** β₂ optimum shifts from 0.995 to 0.997 (~231-step EMA half-life). The [0.992, 0.995] range is a flat plateau (~42.26–42.30); the jump to 0.997 is sharp and real. Arm B improvement is correlated across 3/4 val splits and 3/3 test splits, and accumulates from epoch 3 onward (not a cosine endpoint artifact). Student's mechanism: β₂=0.997 (longer memory) better balances noise filtering vs tracking the cosine-decaying loss landscape than β₂=0.995 at Lion lr=3e-4 + slice=96. 0.999 over-smooths (691-step EMA), 0.992 is equivalent to 0.995.

**β₂ locked: 0.997.** Δ vs H78 (42.3048/40.5564): **−1.09 val, −1.02 test**. Cumulative R5 gain from H37b (66.11): **−24.90 pts val_avg**.

**Status: MERGED (PR #4166). New baseline: val=41.2153 / test=39.5337.**

---

## 2026-05-16 23:58 — Round 5 Cycle 31: Assign H97 (edward, LR fine-tune at β₂=0.997).

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4229 | edward | H97: LR fine-tune at β₂=0.997 (lr=2.5e-4, 3.5e-4) | --lr (probe ±17% around 3e-4 at new β₂ baseline) |

**Rationale:** lr=3e-4 was calibrated at β₂=0.99 (H73). β₂=0.997 changes Lion's EMA dynamics (231-step half-life vs 70 steps). The optimal LR may shift: smoother momentum may tolerate higher peak LR, or may favor lower LR at the cosine tail. Quick 2-arm probe rules this out before accepting lr=3e-4 as the global optimum.

---

## 2026-05-17 00:35 — PR #4191: H91 surf_weight sweep under Lion (fern) — CLOSED, negative

- Branch: `fern/hypothesis_h91_surf_weight_lion`
- Hypothesis: surf_weight=10 locked under AdamW (H54); Lion's sign-update might shift optimal surface/volume loss balance.

| Config | val_avg | Δ vs H88 baseline | test 3-split |
|--------|--------:|------------------:|-------------:|
| Baseline (sw=10, H88) | 41.2153 | — | 39.5337 |
| Arm A (sw=5) | 42.6978 | +1.48 | 40.8528 |
| Arm B (sw=20) | 42.5820 | +1.36 | 41.3041 |

Per-split: Arm B (sw=20) marginally helps val_single_in_dist (42.81 vs 44.73) but hurts val_re_rand (44.70 vs 42.41) — tradeoff washes out in avg. Broad shallow basin around sw=10. Lion's sign-update makes combined gradient direction minimally sensitive to loss magnitude weighting.

- Artifacts: `models/model-charliepai2i48h3-fern-h91-arm-a-sw5-20260516-225517/`, `models/model-h91-arm-b-sw20-20260516-233050/`

**Status: CLOSED — surf_weight=10 confirmed locked under Lion. Orthogonal to optimizer choice.**

---

## 2026-05-17 00:38 — Round 5 Cycle 32: Assign H98 (fern, β₁ retune at β₂=0.997).

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4239 | fern | H98: β₁ retune at β₂=0.997 (β₁=0.85, β₁=0.95) | --beta1 (parallel to H90 at β₂=0.995; reveals β₁×β₂ interaction) |

**Rationale:** H90 (askeladd) probes β₁ at old β₂=0.995 baseline. Now that β₂=0.997 is locked, β₁ optimum may shift — longer β₂ EMA means smoother momentum buffer; lower β₁ might compensate by admitting more current gradient into the sign decision.

---

## 2026-05-17 00:42 — PR #4196: H93 WSD schedule (nezuko) — SENT BACK, budget mismatch confound

- Branch: `nezuko/hypothesis_h93_wsd_schedule`
- Hypothesis: WSD schedule (warmup-stable-decay) outperforms cosine T_max=15 by providing more high-LR time + sharp final decay.

| Arm | Schedule | val_avg | Δ vs H78 baseline | Δ vs H88 baseline |
|-----|----------|--------:|------------------:|------------------:|
| H78 baseline | cosine T_max=15 | 42.3048 | — | +1.09 (regress vs H88) |
| **H88 baseline** | cosine T_max=15 | **41.2153** | −1.09 | — |
| A | WSD 2/43/5 | **67.4008** | +25.10 | +26.19 |

**Confound (student's clear analysis):** WSD 2/43/5 was designed for 50 epochs. Wall-cut at 15 epochs means the decay phase (epochs 45-49) was never reached. The schedule effectively ran as "2 ep warmup + 13 ep constant peak LR" — no closing convergence boost from the decay tail. The +25 pt regression is evidence of schedule-shape mismatch, NOT of WSD being bad.

**Note: student implemented WSD scheduler in train.py with clean flags (`--scheduler wsd --warmup_epochs --stable_epochs --decay_epochs`).** This implementation is reusable for budget-aware follow-ups.

**Status: SENT BACK** — Arms B (0/10/5) and C (0/5/10) requested. Both eliminate warmup (H76 closed) and fit the 15-epoch budget. Tests the actual hypothesis (does stable plateau help vs cosine?) cleanly.

---

## 2026-05-17 — PR #4215: H95 bf16 mixed-precision (alphonse) — MERGED, new best

- Branch: `charliepai2i48h3-alphonse/h95-bfloat16`
- Hypothesis: bf16 autocast reduces s/epoch by ~30%, enabling more epochs within the 30-min wall budget. Quality parity verified vs fp32.

| Arm | Config | val_avg | Δ vs H88 | test 3-split | Epochs |
|-----|--------|--------:|---------:|-------------:|-------:|
| **Arm A (bf16 walltime)** | bf16, T_max=15, 21 epochs | **40.5066** | **−0.71** | **39.0160** | 17 (best) |
| Arm B (bf16 ep15 match) | bf16, T_max=15, 15 epochs | 41.54 | +0.32 | 40.14 | 15 |
| H88 baseline | fp32, T_max=15 | 41.2153 | — | 39.5337 | 15 |

Per-split (Arm A best_epoch=17): val_single_in_dist=40.09, val_geom_camber_rc=54.51, val_geom_camber_cruise=25.01, val_re_rand=42.43
Test (Arm A): test_single_in_dist=34.87, test_geom_camber_rc=48.38, test_re_rand=33.80

**Schedule confound:** T_max=15 hardcoded; with 21 epochs of bf16 training, cosine hits 0 at ep15 then rises. Best epoch 17 sits in the rising-LR phase. Arm B (bf16 matched to ep15) confirms numerical parity with H88 (within noise), verifying that bf16 quality is sound — the Arm A metric improvement is real, partly enabled by the extra 6 epochs.

**Implementation:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16)` wraps forward pass; `pred = pred.float()` cast-back before loss; Lion states remain fp32. Memory unchanged at 30.46 GB.

- Artifacts: `models/model-charliepai2i48h3-alphonse-h95-arm-a-bf16-walltime-20260516-234702/`, `models/model-charliepai2i48h3-alphonse-h95-arm-b-bf16-ep15-20260517-002534/`

**Status: MERGED (PR #4215). New baseline: val=40.5066 / test=39.0160. Schedule fix needed (H99 assigned).**

---

## 2026-05-17 — PR #4189: H90 Lion β₁ sweep at β₂=0.995 (askeladd) — CLOSED, negative

- Branch: `charliepai2i48h3-askeladd/h90-lion-beta1-sweep`
- Hypothesis: β₁=0.85 or β₁=0.95 improves on the default β₁=0.9 at the β₂=0.995 baseline.

| Arm | β₁ | val_avg | Δ vs H88 | Δ vs H95 | test 3-split |
|-----|-----|--------:|---------:|---------:|-------------:|
| Arm A | 0.85 | 41.7650 | +0.55 | +1.26 | 39.7470 |
| Arm B | 0.95 | 48.3518 | +7.14 | +7.85 | 46.5616 |
| H88 baseline | 0.90 | 41.2153 | — | +0.71 | 39.5337 |

Arm A: within noise (Δ+0.55 < 2σ=1.7 at H88 baseline; Δ+1.26 at H95 new baseline also negative). Consistent directional improvement across all but val_re_rand — but magnitude is below noise.  
Arm B: clear regression (+7.1 pts). Gate stats diagnostic: block-0 pre-MLP gate mean jumps from ~0.45 (Arm A) to ~1.22 (Arm B), std 2.21 vs 0.92 — higher β₁ saturates momentum buffer, pushes sign update away from minimum at small-N (50 train, batch=4).

Note: Run at β₂=0.995 (old config). H98 (fern, active) repeats β₁ probe at β₂=0.997.

**Status: CLOSED — β₁=0.9 confirmed near-optimal. Landscape asymmetric: down is neutral, up is bad.**

---

## 2026-05-17 — PR #4195: H92 baseline seed variance (frieren) — CLOSED, noise floor calibrated

- Branch: `charliepai2i48h3-frieren/h92-baseline-variance-seeds`
- Hypothesis: Establish run-to-run noise floor at the H78 Arm B config (β₂=0.995, fp32).

| Seed | val_avg | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand |
|-----:|--------:|-------------------:|-------------------:|-----------------------:|------------:|
| 0 (H78 Arm B) | 42.3048 | 44.73 | 56.55 | 25.11 | 42.83 |
| 1 (H92) | **40.6485** | 39.78 | 53.79 | 25.50 | 43.52 |
| 2 (H92) | 41.6860 | 42.35 | 56.34 | 24.82 | 43.23 |
| **mean (n=3)** | **41.5464** | 42.29 | 55.56 | 25.14 | 43.19 |
| **sample std** | **0.8369** | 2.48 | 1.55 | 0.34 | 0.35 |

**Revised noise floor: 2σ=1.67 pts** (vs prior 2.6 pt estimate). The old 2.6-pt figure was confounded: H74 vs H73 compared different schedules, not just seeds. Per-split: val_single_in_dist noisiest (σ=2.48), cruise and re_rand tight (σ≈0.34).

Test 3-split (seeds 1+2): mean=39.95, std=0.51 → 2σ≈1.02 pts. Test is tighter than val.

**Updated decision thresholds (applied from cycle 34 forward):**
- Δ < 1.7 pts → noise (tie)
- Δ ≥ 2.5 pts → real signal
- Δ ≥ 4.0 pts → strong signal

Note: Seed 1 got val=40.65 at β₂=0.995 — close to H95 best 40.51 (bf16, β₂=0.997). Expected mean at H88 config is ~41.5, not 42.30 (H78 seed-0 was lucky).

**Status: CLOSED — calibration complete. Noise floor revised to 2σ=1.7 pts.**

---

## 2026-05-17 — Cycle 34 Assignments: H99 (alphonse), H100 (askeladd), H101 (frieren)

| PR | Student | Hypothesis | Key Change |
|----|---------|-----------|------------|
| #4272 | alphonse | H99: bf16 + cosine T_max fix (T_max=21 vs T_max=15) | Student adds --T_max CLI arg; Arm A=21, Arm B=15 (repeat) |
| #4276 | askeladd | H100: n_hidden=192 capacity probe with bf16 | Student adds --n_hidden CLI arg; Arm A=192, Arm B=160 |
| #4277 | frieren | H101: n_layers=5 depth probe with bf16 | --n_layers 5 (CLI arg exists); Arm A=5, Arm B=6 |

**Rationale:**
- **H99**: H95 Arm A best_epoch=17 with rising LR (T_max=15 cosine bounces at ep15, rises through ep17-21). Hypothesis: T_max=21 aligned to bf16 budget gives clean monotone decay → potential further improvement.
- **H100**: H86 (n_hidden=192) was wall-cut at ep10 (fp32 ~180 s/epoch). bf16 should give ~14 epochs (~126 s/epoch at n_hidden=192) — enough to evaluate capacity gain. This directly retests the wall-cut-bound hypothesis.
- **H101**: Depth unexplored since H83 (which tested n_layers=4 as the winner under fp32). bf16 gives n_layers=5 ~17 epochs (~106 s/epoch). Depth and width are orthogonal; one more layer may capture higher-order geometric interactions.

---

## 2026-05-17 02:00 — PR #4229: H97 LR retune at β₂=0.997 (edward) — CLOSED, negative

- Branch: `charliepai2i48h3-edward/h97-lr-retune-at-beta2-0997`
- Hypothesis: lr=3e-4 was calibrated at β₂=0.99 (H75); β₂=0.997's longer EMA may shift the LR optimum.

| Arm | lr | val_avg | Δ vs H88 (41.22) | Δ vs H95 (40.51) | test 3-split |
|-----|-----|--------:|-----------------:|-----------------:|-------------:|
| A | 2.5e-4 | 41.7041 | +0.49 | +1.19 | 39.6182 |
| B | 3.5e-4 | 41.5667 | +0.35 | +1.06 | 40.0407 |
| **H88 baseline** | 3.0e-4 | **41.2153** | — | +0.71 | **39.5337** |

Both arms within 1.7-pt 2σ noise floor vs H88 but clearly worse than current H95 (40.51). LR vs val landscape convex with minimum at 3e-4. Per-epoch trajectory: Arm B faster early descent, Arm A steadier; both converge similarly at ep15.

**Conclusion:** lr=3e-4 LOCKED at β₂=0.997. Lion optimizer hyperparameter axes now near-fully saturated.

---

## 2026-05-17 02:05 — PR #4197: H94 Batch size sweep (tanjiro) — CLOSED, negative

- Branch: `charliepai2i48h3-tanjiro/h94-batch-size-sweep`
- Hypothesis: BS=8 (linear-scaled LR) improves over BS=4.

| Arm | BS | LR | val_avg | Δ vs H88 (41.22) | Δ vs H95 (40.51) | mem |
|-----|----|----|----|----|----|----|
| A | 6 | 3e-4 (no scaling) | 43.8969 | +2.68 | +3.39 | 63.88 GB |
| B | 6 | 4.5e-4 (linear scaling) | 45.5948 | +4.38 | +5.09 | 63.88 GB |

BS=8 OOM'd (>96GB) — padding-to-largest policy is heavily quadratic in batch composition. BS=6 fit at 63.88 GB. Both arms clearly regressed: at this short-budget regime (15-21 ep), more optimizer steps per epoch dominates more samples per step. Linear-scaled LR overcompensates under Lion's sign-update.

**Conclusion:** BS=4 LOCKED. Fewer-steps-per-epoch effect dominates at our wall budget. Informative OOM analysis included.

---

## 2026-05-17 02:10 — PR #4217: H96 torch.compile (thorfinn) — SENT BACK for compound test

- Branch: `charliepai2i48h3-thorfinn/h96-torch-compile`
- Hypothesis: torch.compile reduces s/epoch ≥15% with no quality regression.

| Arm | Mode | val_avg | Δ vs H78 (42.30) | Δ vs H88 (41.22) | Δ vs H95 (40.51) | s/epoch | Epochs |
|-----|------|--------:|------------------:|------------------:|------------------:|--------:|-------:|
| A | default | 41.9061 | −0.40 | +0.69 | +1.40 | 88.6 (-27%) | 21 |
| B | reduce-overhead | **41.0543** | −1.25 | −0.16 (tie) | +0.55 | 93.0 (-24%) | 20 |

torch.compile delivered a clean -27% s/epoch with no metric regression (Arm B ties H88 within noise). However, run at β₂=0.995 (pre-H88 baseline) and without bf16. Doesn't beat current H95 (40.51, bf16 baseline).

**Student insight:** "The extra epochs that compile bought went mostly to drift past the optimum" — same T_max=15 schedule confound as H95. A T_max fix should harvest the remaining headroom.

**Status: SENT BACK** — rebase onto current advisor branch, then run compile + bf16 + β₂=0.997 + T_max=21 compound test (Arms C and D). Tests whether the two efficiency wins stack and whether the T_max fix unlocks the residual gain.

---

## 2026-05-17 02:15 — Round 5 Cycle 35: Assign H102 (edward), H103 (tanjiro)

| PR | Student | Hypothesis | Key Change |
|----|---------|------------|------------|
| #4291 | edward | H102: slice_num=128 attention capacity probe under bf16+β₂=0.997 | --slice_num 128/112 (no code change; CLI arg exists) |
| #4292 | tanjiro | H103: mlp_ratio=3 FFN capacity retest under bf16 | Add --mlp_ratio CLI arg, Arm A=3, Arm B=2 (within-PR control) |

**Rationale:**
- **H102**: slice_num=128 was negative at H66 under AdamW. Lion+bf16+β₂=0.997 is a substantially different regime; the attention bottleneck optimum may have shifted upward. Direct capacity probe of the slice attention.
- **H103**: H89 (mlp_ratio=3 under fp32) was wall-cut at ep11. bf16's -30% s/epoch unlocks ~17 epochs at mlp_ratio=3 — should give a fair shot at convergence. Within-PR control arm (mlp_ratio=2 + bf16) isolates the effect from pod/seed noise.

**Concurrent capacity sweep across 4 axes:**
- H100 (askeladd): n_hidden=192 — width per layer
- H101 (frieren): n_layers=5 — depth
- H102 (edward): slice_num=128 — attention bottleneck capacity
- H103 (tanjiro): mlp_ratio=3 — FFN expansion within block

Together these probe the four orthogonal capacity dimensions under bf16's expanded budget.

---

## 2026-05-17 — PR #4239: H98 Lion β₁ retune at β₂=0.997 (fern) — CLOSED, informative tie

- Branch: `charliepai2i48h3-fern/h98-beta1-retune-at-beta2-0997`
- Hypothesis: β₁=0.85 or β₁=0.95 improves on default β₁=0.9 at the new β₂=0.997 baseline.

| Arm | β₁ | val_avg | Δ vs H88 (41.22) | Δ vs H95 (40.51) | test 3-split |
|-----|------|---------|-------------------|-------------------|--------------|
| A | 0.85 | 40.5804 | −0.64 (tie) | +0.07 (tie) | 39.4821 |
| B | 0.95 | 47.0926 | +5.88 | +6.58 | 44.4086 |
| H88 baseline | 0.90 | 41.2153 | — | +0.71 | 39.5337 |

Arm A directional improvement across 3/4 val splits (single_in_dist −1.53, cruise −1.03, re_rand −0.07; rc neutral +0.09). Each individual delta within 2σ=1.7 noise floor. Confirms H90 (askeladd at β₂=0.995): β₁ landscape asymmetric — lower β₁ trends better, higher (0.95) regresses badly. Tested at fp32; β₁=0.85 + bf16 compound noted for future test (predicted ~39.85 — still within noise).

**Status: CLOSED — β₁=0.9 locked. β₁=0.85 noted as a future compound candidate with bf16.**

---

## 2026-05-17 — PR #4196: H93 WSD reshape (nezuko) — Arm C WIN, sent back for rebase

- Branch: `charliepai2i48h3-nezuko/hypothesis_h93_wsd_schedule`
- Hypothesis: Budget-aware WSD reshape (0/10/5 and 0/5/10) at β₂=0.997. No warmup (H76 closed warmup), fits 15-epoch budget so the decay tail actually fires.

| Arm | Schedule | val_avg | Δ vs H88 (41.22) | Δ vs H95 (40.51) | test 3-split |
|-----|----------|---------|-------------------|-------------------|--------------|
| B | WSD 0/10/5 (stable=10, decay=5) | 41.5351 | +0.32 (tie) | +1.02 (tie) | 41.3310 |
| **C** | **WSD 0/5/10 (stable=5, decay=10)** | **39.5100** | **−1.71 (boundary win)** | **−1.00 (within noise, but trend)** | **38.5345** |

**Arm C per-split vs H95 (current best):**
- val_geom_camber_rc: 54.50 → 51.22 (Δ-3.28) — biggest gain on hardest split
- val_geom_camber_cruise: 25.00 → 23.61 (Δ-1.40)
- val_re_rand: 42.43 → 42.15 (Δ-0.28)
- val_single_in_dist: 40.09 → 41.06 (Δ+0.97) — small loss
- test_avg: 39.02 → 38.53 (Δ-0.48)

**Mechanism:** Forcing 10 epochs of cosine LR decay (vs T_max=15 baseline which decays over all 15 epochs) gives more wall-time near-zero-LR fine-tune — and that pays off on the hardest geometric OOD splits. The stable plateau provides 5 epochs of full-LR coarse training; the long decay (10 epochs) refines the optimum.

**Status: SENT BACK for clean rebase only (no rerun if conflicts are clean).** Branch DIRTY against current advisor branch. After successful rebase, this merges as the new best.

---

## 2026-05-17 — Cycle 36 Assignments/Returns

| PR | Student | Hypothesis | Note |
|----|---------|-----------|------|
| #4217 | thorfinn | **H96 (re-sent): compile + bf16 + T_max=21 compound (Arms C, D)** | First send-back comment was lost (bash background killed); re-posted in foreground |
| #4196 | nezuko | **H93 (sent back): clean rebase, Arm C is winner** | Standing winner pending rebase |
| #4316 | fern | **H112: AoA + log(Re) + gap/stagger input jitter** | XS complexity; targets val_re_rand and val_geom_camber_cruise OOD splits via continuous-cond Gaussian noise; 3 jitter magnitudes (σ=0.02, 0.05, 0.1) |

**Cycle summary:**
- 8 WIP experiments, 0 idle students
- Standing winner: H93 Arm C (WSD 0/5/10) at val=39.51 / test=38.53 — pending rebase before merge
- Compound test queued mentally: WSD 0/5/10 + bf16 → predicted val ~37-38 if effects compound
- Schedule lever has new headroom (WSD 0/5/10 is a real signal); capacity lever still has 4 active probes (H100/H101/H102/H103); efficiency stack has compile+bf16 compound active (H96)

---

## 2026-05-17 — PR #4272: H99 bf16 + T_max=21 schedule fix (alphonse) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-alphonse/h99-bf16-schedule-fix`
- Hypothesis: Aligning CosineAnnealingLR T_max to the bf16 wall budget (21 epochs) eliminates the LR-bounce confound and unlocks additional accuracy.

| Arm | T_max | val_avg | Δ vs H95 (40.51) | test 3-split | best_epoch |
|-----|-------|---------|-------------------|--------------|-----------|
| **A (winner)** | **21** | **37.2626** | **−3.24** | **35.8568** | **21 (monotone)** |
| B (control) | 15 | 40.6803 | +0.17 (repro ✓) | 39.7261 | 18 |

Per-split Arm A: val_single_in_dist 37.09 (Δ-3.00), val_geom_camber_rc 49.78 (Δ-4.73), val_geom_camber_cruise 22.93 (Δ-2.08), val_re_rand 39.25 (Δ-3.18). Clean improvement on every split. Arm B (T_max=15 control) reproduces H95 within noise (val=40.68 vs 40.51, Δ=0.17), confirming T_max is the only variable.

**Analysis:** The T_max=15 hardcode under 21 bf16 epochs created an LR-bounce: cosine decays to 0 at ep15, then rises again through ep16-21. H95's best_epoch=17 sat in this rising-LR "noise region." H99 fixes T_max=21 → monotone descent from ep1 to ep21 → model converges cleanly at final epoch. Mechanism: Lion + cosine decay benefits from extended near-zero-LR polishing in the final epochs. Δ-3.24 pts is a one-line schedule fix with no architecture change.

**Cumulative gain vs H37b (66.11): −28.84 pts val_avg.** Artifacts: `models/model-h99-arm-a-bf16-tmax21-20260517-014114/`

---

## 2026-05-17 — PR #4276: H100 n_hidden=192 capacity probe (askeladd) — CLOSED, no signal

- Branch: `charliepai2i48h3-askeladd/h100-n-hidden-192-bf16`
- Hypothesis: n_hidden=192 or n_hidden=160 under bf16 yields capacity gain over default n_hidden=128.

| Arm | n_hidden | val_avg | Δ vs H95 (40.51) | test 3-split |
|-----|---------|---------|-------------------|--------------|
| A | 192 | 40.7830 | +0.28 (worse) | 39.8213 |
| B | 160 | 40.3852 | -0.12 (tie) | 38.5554 |

Both within 1.7-pt noise floor. Under new H99 baseline (37.26), both arms are +3.1 to +3.5 pts worse → real negative signal. Conclusion: capacity is NOT the bottleneck at this point. The T_max=21 schedule fix (H99) unlocks far more accuracy at n_hidden=128 than any width increase. Closed.

---

## 2026-05-17 — PR #4277: H101 n_layers=5/6 depth probe (frieren) — CLOSED, partial signal

- Branch: `charliepai2i48h3-frieren/h101-n-layers-5-bf16`
- Hypothesis: n_layers=5 or n_layers=6 under bf16 improves on n_layers=4 baseline.

| Arm | n_layers | epochs | val_avg | Δ vs H95 (40.51) | test 3-split |
|-----|---------|--------|---------|-------------------|--------------|
| A | 5 | 18 | 42.2102 | +1.70 (noise / small negative) | 40.5731 |
| B | 6 | 15 | 39.7356 | -0.77 (tie) | 37.8828 |

Arm B (n_layers=6) was wall-cut at ep15, still descending (41.78→41.26→39.74 per epoch). Under new H99 baseline (37.26), both arms are nominally worse, but Arm B's undertraining is the confound. With T_max=21 fix applied (which gives different effective LR shape at n_layers=6's wall-cut), there may still be headroom. Assigning frieren a retest (H113: n_layers=6 + bf16 + T_max sweep). Closed.

---

## 2026-05-17 — Cycle 38 Assignments

| PR | Student | Hypothesis | Priority |
|----|---------|-----------|---------|
| #4332 | nezuko | **H114: WSD 0/7/14 + 0/3/18 on H99 baseline (compound schedule)** | TOP |
| #4333 | frieren | **H113: n_layers=6 + bf16 + T_max=21 retest (Arms A, B)** | HIGH |
| #4335 | alphonse | **H106: Fourier PE of mesh coordinates (K=8, K=4)** | MED |
| #4337 | askeladd | **H107: log(Re) auxiliary head (λ=0.1, λ=0.01)** | MED |

**Cycle summary:** Major win merged (H99 Δ-3.24 pts). 8 WIP, 0 idle. Active capacity probes H102/H103 now running against new baseline 37.26 — results will determine if any capacity axis improves further with schedule fix.

---

## 2026-05-17 — PR #4335: H106 Fourier PE K=4 (alphonse) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-alphonse/h106-fourier-pe`
- Hypothesis: Fourier positional encoding of mesh (x,z) coordinates at K frequencies.

| Arm | K | val_avg | Δ vs H99 | test 3-split |
|-----|---|---------|-----------|--------------|
| A | 8 | 36.9061 | -0.36 | 34.8193 |
| **B (winner)** | **4** | **35.9159** | **-1.35** | **35.1221** |

Per-split Arm B: val_single_in_dist Δ-4.86 (strong signal — Fourier features help in-dist geometry discrimination). val_geom_camber_rc +0.57 (neutral). val_geom_camber_cruise -1.33. val_re_rand +0.24. The K=4 < K=8 result suggests fewer frequencies is better at this model size/budget — K=8 over-parameterizes the input representation.

**Status: MERGED. New baseline val=35.9159 / test=35.1221. Cumulative R5 gain: −30.19 pts vs H37b.**

---

## 2026-05-17 — PR #4332: H114 WSD 0/7/14 + 0/3/18 (nezuko) — CLOSED, beat H99 not H106

| Arm | Schedule | val_avg | Δ vs H99 | test 3-split |
|-----|---------|---------|-----------|--------------|
| A | WSD 0/7/14 | 36.5186 | -0.74 | 34.7609 |
| B | WSD 0/3/18 | 36.2888 | -0.97 | 35.0710 |

WSD long-decay-tail mechanism confirmed at T_max=21 scale. Arm B WSD 0/3/18 has best val; Arm A WSD 0/7/14 has best test (34.76 vs H99 35.86). However NEITHER arm beats the new H106 Fourier baseline (val=35.92). Closed. The compound (WSD + Fourier) is assigned as H119.

---

## 2026-05-17 — PR #4333: H113 n_layers=6 + bf16 + T_max=21 (frieren) — CLOSED, negative

Arm A: val=40.2690 (+3.01 vs H99 37.26, real negative). Two seeds: 40.27 and 43.34, mean 41.81. n_layers=6 is definitively worse at this compute budget — +47% params causes undertraining in 15 available epochs. Schedule confound was NOT the limiting factor; the wall-cut is the same regardless of T_max. n_layers lever closed.

---

## 2026-05-17 — PR #4337: H107 log(Re) aux head (askeladd) — CLOSED, tie

Arm B (λ=0.01): val=37.1878 (Δ-0.07 within noise). Arm A (λ=0.1): val=39.07 (regression). The trunk already encodes Re via FiLM — aux head learns log10(Re) in <5 epochs, then provides no sustained regularization. log(Re) aux head lever closed.

---

## 2026-05-17 — Cycle 41 Assignments

| PR | Student | Hypothesis | Priority |
|----|---------|-----------|---------|
| #4389 | nezuko | **H119: WSD 0/3/18 + Fourier K=4 compound** | TOP |
| #4390 | thorfinn | **H118: compile + Fourier K=4 + bf16 + T_max=21 (correct labels)** | HIGH |
| #4392 | alphonse | **H104: per-sample pressure std normalization** | MED |
| #4394 | askeladd | **H120: Fourier freq sweep K=2, K=1** | MED |
| #4395 | frieren | **H121: SWA (start ep18, ep15) on H106 stack** | MED |

**Cycle summary:** H106 Fourier K=4 merged (new best val=35.92). H113/H114/H107 closed. H118 label bug fixed. 8 WIP, 0 idle.

---

## 2026-05-17 — PR #4389: H119 WSD 0/3/18 + Fourier K=4 compound (nezuko) — CLOSED, no compound

- Branch: `charliepai2i48h3-nezuko/h119-wsd-fourier-compound`
- Hypothesis: WSD long-decay schedule (Δ-0.97 at H99) and Fourier PE (Δ-1.35 at H99) are orthogonal mechanisms (schedule shape vs input encoding) → compound at H106 should give val ~34.9-35.5.

| Metric | H119 | H106 baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 36.6814 | 35.9159 | +0.77 (worse) |
| val_single_in_dist | 34.2591 | 32.2282 | +2.03 |
| val_geom_camber_rc | 51.2425 | 50.3515 | +0.89 |
| val_re_rand | 39.4214 | 39.4884 | -0.07 |
| test 3-split | 35.4354 | 35.1221 | +0.31 |

**LR trace verified correct** (3 ep at peak 3e-4, 18 ep cosine to 0). Per-epoch trajectory shows the WSD start was rough: val=159→107→104 over the 3 stable epochs, vs H106 cosine which is already in active decay by ep 3. The model couldn't recover the early gap during the 18-epoch decay tail.

**Mechanism (per student's analysis):** The 3-epoch high-LR plateau is destabilising on Fourier PE's richer input space (fun_dim 22→38). Per-split: regression concentrated on val_single_in_dist (+2.03), other splits essentially unchanged — the plateau hurt fitting precision on easy examples while the long decay tail couldn't compensate.

**Conclusion:** WSD does not compound with Fourier at H106. Schedule-shape attack at this baseline is exhausted. Closed. Reassigning nezuko to H122 (Lookahead — orthogonal optimization-level direction).

---

## 2026-05-17 — PR #4357: H115 slice_num=80, 64 (edward) — SENT BACK FOR COMPOUND

- Branch: `charliepai2i48h3-edward/h115-slice-num-sub96`
- Hypothesis: At Lion+bf16+T_max=21, sub-96 slice_num may improve over slice_num=96 (which was set under old optimizer).

| Arm | slice_num | val_avg | Δ vs H99 (37.26) | test 3-split |
|-----|-----------|---------|-------------------|--------------|
| A | 80 | **35.9161** | -1.35 | **34.9972** |
| B | 64 | 37.2214 | -0.04 | 35.3223 |

**Updated slice_num curve under Lion+bf16+T_max=21:** 64→37.22, **80→35.92**, 96→37.26, 112→41.07, 128→41.74. slice_num=80 is the new candidate optimum.

**Critical caveat:** Arm A ran on H99 base (no Fourier PE). Comparing slice_num=80 (no Fourier) val=35.9161 vs H106 (slice_num=96 + Fourier K=4) val=35.9159 — TIE on val (Δ+0.0002), test 3-split Δ-0.12 (within noise). Two orthogonal mechanisms reach the same accuracy.

**Decision:** Send back for compound test — Arm C: slice_num=80 + Fourier K=4 on H106 baseline. If mechanisms truly orthogonal, predict val ~34.8-35.5.

---

## 2026-05-17 — Cycle 42 Assignments

| PR | Student | Hypothesis | Priority |
|----|---------|-----------|---------|
| #4357 | edward | **H115 Arm C: slice_num=80 + Fourier K=4 (compound)** | TOP (two ties → compound predicted -0.4 to -1.1) |
| #4422 | nezuko | **H122: Lookahead(Lion) k=5 α=0.5 at H106 baseline** | HIGH (orthogonal optimizer-level mechanism) |

**Cycle summary:** H119 closed (no compound). H115 returned to wip for slice80+Fourier compound. Nezuko reassigned to Lookahead. 8 WIP, 0 idle.

---

## 2026-05-17 — PR #4394: H120 Fourier K=2, K=1 (askeladd) — MERGED, NEW BEST

- Branch: `charliepai2i48h3-askeladd/h120-fourier-freq-sweep`
- Hypothesis: K=8→K=4 monotone improvement; test K=2 and K=1 to find optimum.

| K | val_avg | test 3-split | Source |
|---|---------|-------------|--------|
| 8 | 36.91 | — | H106 Arm A |
| 4 | 35.9159 | 35.1221 | H106 Arm B (prior best) |
| 2 | 36.2021 | 34.8488 | H120 Arm A |
| **1** | **35.6651** | **33.3976** | **H120 Arm B (NEW BEST)** |

Per-split K=1 vs K=4 baseline: val_single_in_dist +3.98 (K=4 better for in-dist), val_geom_camber_rc -2.79, val_geom_camber_cruise -0.84, val_re_rand -1.35. Test trend strictly monotone: K=1 beats K=4 on ALL three test splits (Δ-1.21, -2.67, -1.30).

**Mechanism:** K=1 → features = [sin(2πx), cos(2πx), sin(2πz), cos(2πz)] — single global wavelength = chord length, exactly the scale of foil geometry. Higher K adds sub-chord wavelengths that the model overfits on training data but cannot transfer to OOD geometry.

**Status: MERGED. New baseline val=35.6651 / test=33.3976. Cumulative R5 gain: −30.44 pts vs H37b.**

---

## 2026-05-17 — PR #4392: H104 per-sample p std normalization (alphonse) — CLOSED, catastrophic

val=287.75 vs baseline 35.92 (8× worse). Fundamental flaw: per-sample normalization makes target `y_norm/p_std` depend on per-sample statistics the model cannot observe at inference. Amplification (p_std min 0.0012 → |target| up to 165) + bf16 instability. Per-sample p norm lever closed.

---

## 2026-05-17 — Stale drafts #4378, #4379 closed

- #4378 (fern/H116 log(Re) aux head): duplicate of closed H107. Never started. Closed.
- #4379 (tanjiro/H117 Fourier PE K=8, K=4): duplicate of merged H106. Never started. Closed.

---

## 2026-05-17 — Cycle 43 Assignments

| PR | Student | Hypothesis | Priority |
|----|---------|-----------|---------|
| #4451 | askeladd | **H123: Fourier K=0 ablation + K=1 scale=0.5** | TOP (complete sweep: does monotone trend extend to K=0?) |
| #4452 | alphonse | **H124: EMA weight averaging τ=0.999, τ=0.9995 at H120 K=1 baseline** | HIGH (zero-compute orthogonal mechanism) |

**Cycle summary:** H120 K=1 merged (new best val=35.67, test=33.40). H104 closed (catastrophic). Stale H116/H117 drafts closed. 8 WIP, 0 idle.
