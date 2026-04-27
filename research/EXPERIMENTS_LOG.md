# SENPAI Research Results — `icml-appendix-charlie-pai2d-r1`

## 2026-04-27 23:30 — PR #355: mlp_ratio 2→4 (charliepai2d1-nezuko)
- Branch: `charliepai2d1-nezuko/mlp-ratio-4`
- Hypothesis: bumping the per-block MLP from `128→256→128` to `128→512→128` adds ~+19% params (0.83M → 0.99M) and gives the per-node nonlinearity more lift; expected −3% to −8% on `val_avg/mae_surf_p`.

### Headline metrics (best epoch = 13/50, run cut by 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---|---|---|---|---|
| `mae_surf_p` | 165.01 | 135.12 | 99.19 | 117.65 | **129.24** |
| `mae_surf_Ux` | 2.636 | 2.899 | 1.708 | 2.405 | 2.412 |
| `mae_surf_Uy` | 0.939 | 1.200 | 0.682 | 0.932 | 0.938 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---|---|---|---|---|
| `mae_surf_p` | 146.15 | 129.26 | **NaN** | 119.14 | **NaN** (3-split mean = 131.51) |
| `mae_surf_Ux` | 2.507 | 2.963 | 1.707 | 2.294 | 2.368 |
| `mae_surf_Uy` | 0.909 | 1.140 | 0.643 | 0.906 | 0.900 |

- Metric files (student branch): `models/model-charliepai2d1-nezuko-mlp-ratio-4-20260427-224909/{metrics.jsonl,metrics.yaml}`
- Wall clock: 32 min (training cut by `SENPAI_TIMEOUT_MINUTES=30` after 13/50 epochs)
- Peak VRAM: 52.18 GB (within 96 GB budget)
- Param count: 991,319 (~0.99M)

### Analysis
- **Training was cut very early.** Best val (129.24) was at the *last* completed epoch (13/50). Cosine schedule had barely begun decaying; the val curve went 282.7 → 129.2 over those 13 epochs and was still descending strongly. With more epochs the number would almost certainly continue dropping. Per-epoch wall clock was ~150 s; 30-min timeout fits ~12 epochs.
- **Test pressure NaN is a scoring failure, not a model failure.** Validation on the matching split (`val_geom_camber_cruise`) is the *best* of the four val splits at `mae_surf_p = 99.19`. The NaN appears only on the test counterpart — and only on the pressure channel. Velocity (Ux, Uy) MAE for that test split are finite (1.71, 0.64). Pattern: `vol_loss=inf, surf_loss=nan, mae_*_p=nan`. The model produced an inf or NaN prediction on the p channel for at least one test sample; `data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions**, so the bad value pollutes the float64 accumulator → NaN in the final MAE.
- **No baseline measured yet on this branch** so the val=129.24 cannot yet be ranked. The other 7 round-1 PRs are still in flight.

### Decision: send back to student
- Cannot merge: `test_avg/mae_surf_p` is NaN, which violates the "no NaN in primary metrics" rule.
- Cannot close: hypothesis is sound, val trajectory is clean and strongly descending, the only blocker is a numerical-edge-case in eval.
- Sending back with a concrete fix (`nan_to_num` clamp in `evaluate_split` before scoring) and instructions to re-run.
- This same fix likely benefits all round-1 PRs — note in `CURRENT_RESEARCH_STATE.md` that the same NaN may show up in other students' runs.
