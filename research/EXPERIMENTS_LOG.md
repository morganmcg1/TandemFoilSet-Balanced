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
