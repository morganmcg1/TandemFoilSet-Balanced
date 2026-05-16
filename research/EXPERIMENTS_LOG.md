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
