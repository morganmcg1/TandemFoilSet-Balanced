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
