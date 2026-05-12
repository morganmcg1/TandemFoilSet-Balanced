# SENPAI Research Results

_Advisor branch: `icml-appendix-charlie-pai2g-48h-r3`._

Results from each terminal PR are recorded below in reverse chronological order.

<!-- Entries will be appended as PRs land terminal SENPAI-RESULT markers. -->

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
