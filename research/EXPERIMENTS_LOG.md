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
- 23:30: sent back with `nan_to_num`-on-pred fix instructions. **Fix instructions were wrong — see correction below.**
- 23:42: corrective follow-up posted. After PRs #356/#351 landed with independent diagnoses, the actual root cause is one `test_geom_camber_cruise` sample with non-finite **ground truth** in the `p` channel. `data/scoring.py` masks bad samples but `(finite − inf).abs() * 0 = NaN` defeats it. Fix lives in `train.py:evaluate_split` (sanitize `y`, drop bad samples from `mask`) and is now in baseline post-#356. Nezuko instructed to rebase onto baseline + retain `mlp_ratio=4`.

## 2026-04-27 23:42 — PR #356: EMA(0.999) shadow for val + checkpoint (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/ema-eval` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `208f1cc`).
- Hypothesis: maintain EMA shadow weights with decay 0.999, evaluate val on the shadow, save the shadow's state_dict, and run final test eval from it. Predicted −2% to −7% on `val_avg/mae_surf_p` from variance reduction alone.

### Headline metrics (best epoch = 13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 170.491 | 144.104 | 100.492 | 114.015 | **132.276** |
| `mae_surf_p` (raw, same epoch) | 231.699 | 180.205 | 144.919 | 156.747 | 178.392 |
| `mae_surf_p` (best raw, ep11) | — | — | — | — | 136.526 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 147.132 | 127.917 | 84.026 | 113.089 | **118.041** |
| `mae_surf_Ux` | 1.948 | 2.988 | 1.332 | 1.949 | 2.054 |
| `mae_surf_Uy` | 0.827 | 1.076 | 0.553 | 0.833 | 0.822 |

- Metric files (now in baseline): `models/model-charliepai2d1-tanjiro-ema-eval-20260427-225450/{metrics.jsonl,metrics.yaml,config.yaml}`
- Wall clock: 30.5 min (timeout-cut at ep13/50, +companion raw eval ~doubles val cost vs EMA-only)
- Peak VRAM: 42.11 GB

### Analysis
- **EMA delivered the predicted free lunch.** 132.28 (EMA) vs 136.53 (best raw) = −3.1% on `val_avg/mae_surf_p`, in band. EMA val curve is monotonic (324 → 132 over 13 epochs); raw is noisy (191 → 164 → 249 mid-training). Smoothing buys most at noisy epochs; both are still descending hard at the timeout.
- **Real bug-find in the scoring path.** Tanjiro identified that one test sample (`test_geom_camber_cruise` index 20) has `y[p]` non-finite. `data/scoring.py:accumulate_batch` builds the right per-sample mask but computes `err = |pred − y|` *before* the masked sum, and IEEE-754 `NaN*0 = NaN` poisons the float64 accumulator. Fix: pre-pass in `evaluate_split` that drops non-finite-y samples from `mask` and zeros their `y`. Same root cause flagged independently by askeladd on PR #351.
- **Bonus instrumentation.** Each epoch now logs both EMA and raw `val_avg/mae_surf_p` in `metrics.jsonl`, plus `best_raw_val_avg/mae_surf_p` and `best_raw_epoch` in `metrics.yaml`. Future EMA experiments can be audited for raw vs EMA gap directly.

### Decision: merge as new round-1 baseline
- Predicted delta achieved (−3.1%, in the −2% to −7% band).
- Test number clean (no NaN) thanks to the workaround.
- The workaround benefits all in-flight round-1 PRs once they rebase.
- BASELINE.md updated; #355 and #351 routed to rebase onto this baseline retaining their respective levers.

## 2026-04-27 23:42 — PR #351: surf_weight 10→50 (charliepai2d1-askeladd)
- Branch: `charliepai2d1-askeladd/surf-weight-50`
- Hypothesis: raising surface-loss weight from 10 to 50 should reduce volume gradient dominance and align training signal more directly with the metric.

### Headline metrics (best epoch = 10/50, run cut by 30-min timeout; concurrent GPU contention slowed eps 7–8)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 166.86 | 143.62 | 108.19 | 122.08 | **135.19** |

- `test_avg/mae_surf_p`: NaN (same scoring bug; 3-split mean over the clean splits = 134.00; cruise NaN as expected).
- Metric files (student branch): `models/model-surf-weight-50-20260427-225133/{metrics.jsonl,metrics.yaml}`
- Wall clock: 30.2 min, peak VRAM: 42.12 GB

### Analysis
- **Did not beat the new baseline (132.28).** 135.19 vs 132.28 = +2.2%. Not a clear regression (>5%), and the run was timeout-cut at ep10 plus had GPU contention costing ~2 epochs.
- **Trade-off direction sane.** Surface MAE is decent across splits; cruise (108.19) easiest, single (166.86) hardest. Volume MAE not blowing up — `vol_p` ranges 153–199 across splits.
- **Independently rediscovered the scoring NaN bug.** Same root cause as tanjiro's; clean diagnosis with the IEEE-754 NaN×0 explanation. Bug-fix suggestion logged.

### Decision: send back for rebase + retain surf_weight=50
- Surf-weight is orthogonal to EMA — right move is to test compounding rather than close.
- Rebase onto post-#356 baseline (gets EMA + NaN-safe path), keep `surf_weight=50.0`, re-run, report Δ vs new baseline.
- If "EMA + surf_weight=50" beats 132.28 by any margin, merge as next baseline.
