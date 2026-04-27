# SENPAI Research Results — icml-appendix-charlie-pai2d-r4

## 2026-04-27 23:20 — PR #300: Wider Transolver: n_hidden 128->192, slice_num 64->96
- Branch: `charliepai2d4-edward/wider-model` (deleted on close)
- Student: charliepai2d4-edward
- Hypothesis: Increase Transolver capacity (n_hidden 128→192, slice_num 64→96, ~1.48 M params) on the under-utilized 96 GB GPU. Predicted Δ on `val_avg/mae_surf_p`: -5% to -10% vs. published baseline.
- **Outcome: CLOSED** (under-trained, not directly comparable to baseline).

### Best validation metrics (epoch 9 of 50; training stopped at 30-min cap)
| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 193.90 | 2.510 | 0.982 | 174.91 |
| val_geom_camber_rc     | 157.52 | 3.349 | 1.300 | 147.12 |
| val_geom_camber_cruise | 104.49 | 2.052 | 0.669 |  99.36 |
| val_re_rand            | 125.49 | 2.583 | 0.937 | 118.76 |
| **val_avg**            | **145.35** | 2.624 | 0.972 | 135.04 |

### Test metrics (best ckpt)
| split | mae_surf_p |
|---|---|
| test_single_in_dist     | 169.31 |
| test_geom_camber_rc     | 139.23 |
| test_geom_camber_cruise | NaN ← *scoring bug, see below* |
| test_re_rand            | 125.83 |
| **test_avg (3 valid)**  | **144.79** |

JSONL summary: `research/EXPERIMENT_METRICS.jsonl` (PR=300 records).

### Analysis
- **Run hit `SENPAI_TIMEOUT_MINUTES=30` cap at end of epoch 9** (~205 s/epoch). Validation was still improving sharply (epoch 8 → 9 dropped 16 pts on `val_avg/mae_surf_p`); the model had not converged. We can't compare a 9-epoch result to a hypothetical 50-epoch baseline.
- **No overfitting yet** — train loss still well above val, consistent with under-training. The held-out camber gap (rc=157.5, cruise=104.5) wasn't catastrophic.
- **Peak GPU memory: 63 GB / 96 GB**, no OOM.
- **Critical bug found in `data/scoring.py`**: sample 20 of `test_geom_camber_cruise` has 761 inf values in the p channel of ground-truth y. The masking logic uses `error * mask` (float-mask multiply), and `inf * 0 = NaN` in IEEE-754, poisoning the float64 accumulator for that channel. Every wide- or deep-model experiment on this branch will hit the same NaN.
- Decision rationale: closing rather than re-running; assigning edward to fix the scoring bug next as the higher-leverage action. A more conservative widening (n_hidden=144-160) can be revisited once we have a fully-trained baseline number on this branch from one of the cheap in-flight experiments (alphonse, askeladd).
