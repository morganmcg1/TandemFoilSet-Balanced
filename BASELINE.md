# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best

### 2026-05-12 20:10 — PR #1512: [scoring-nan-fix] Default config + NaN-fix patch (fern)

- **`val_avg/mae_surf_p`:** **123.99** (best epoch 14, 30-min cap)
- **`test_avg/mae_surf_p`:** **110.97** (from best-val checkpoint; NOW FINITE — scoring bug fixed in this PR)
- **Per-split surface-p MAE (val):** single_in_dist=N/A (not in SENPAI-RESULT), cruise=N/A, rc=N/A, re_rand=N/A _(see metrics.jsonl for full per-split)_
- **Per-split surface-p MAE (test):** single_in_dist=134.23, geom_camber_rc=121.93, geom_camber_cruise=76.78, re_rand=110.93
- **Config:** default — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, CosineAnnealingLR(T_max=50), AdamW`
- **Change vs. original default:** one-line fix in `data/scoring.py::accumulate_batch` (`torch.nan_to_num(err, …)`) — does not affect training, only MAE reporting. Val/test numbers are the same as the unpatched default config would produce.
- **Metric artifacts:** `models/model-charliepai2g48h4-fern-scoring-nan-fix-20260512-185620/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-fern --experiment_name "charliepai2g48h4-fern/scoring-nan-fix"`

**Two-point default-config baseline estimate (2026-05-12 20:43):**
- fern scoring-fix run: val_avg = **123.99** (this PR)
- alphonse baseline-ref (#1368): val_avg = **137.57**
- Two-run mean: ~**130.8**; σ ≈ **6.8**; peak-to-peak range: **13.6 pts (~10%)**
- Practical comparison threshold: differences < ~10 pts are inconclusive on a single unseeded run.

**Note on test metric:** `test_avg/mae_surf_p = 110.97` is the first clean test number on this branch — the scoring-nan-fix PR is what made it possible. All prior test numbers (NaN) were the result of the `Inf * 0 = NaN` propagation bug in `data/scoring.py::accumulate_batch` now fixed in this merge.

## Merged improvements (within noise floor of val_avg baseline range 123.99–137.57)

These PRs were merged; all within the noise floor (run-to-run σ ≈ 6.8 pts). All beat alphonse's canonical baseline of 137.57:

| PR | Description | val_avg | test_avg | Status |
|---|---|---|---|---|
| #1513 (tanjiro) | bf16 autocast | 125.40 | 126.57 (3-split) | **MERGED** → 24% per-epoch speedup; 18 effective epochs/30 min |
| #1416 (thorfinn) | unified_pos=True, ref=8 | 125.78 | 117.12 | **MERGED** → best cruise: val=91.85, test=80.27 |
| #1369 (askeladd) | surf_weight=10→20 | 127.94 | 117.35 | **MERGED** → best raceCar: single=150.41, rc=135.82 |

**Current advisor-branch recipe** (after 4 merges): `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=20.0, batch_size=4, CosineAnnealingLR(T_max=50), AdamW`. Future experiments inherit this recipe unless otherwise specified. Alphonse's `seed42-baseline` (#1577, in flight) will establish the definitive reproducible baseline for this merged recipe.

---

## Previous best
- **Status (before 2026-05-12 20:10):** No baseline — fresh research track.
- **Default Transolver config (in `train.py`):** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`; `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50`; CosineAnnealingLR; AdamW optimizer.
- Each training execution capped at `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock bound).

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
