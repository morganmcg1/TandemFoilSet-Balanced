# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best

### 2026-05-13 00:08 — PR #1374: [huber-loss] Smooth L1 (Huber, beta=1.0) instead of MSE (edward)

- **`val_avg/mae_surf_p`:** **110.59** (best epoch 15/18)
- **`test_avg/mae_surf_p`:** **102.28** (from best-val checkpoint, all 4 splits clean)
- **Per-split surface-p MAE (val):** single_in_dist=127.85, geom_camber_rc=111.05, geom_camber_cruise=95.72, re_rand=107.73
- **Per-split surface-p MAE (test):** single_in_dist=113.36, geom_camber_rc=105.68, geom_camber_cruise=85.87, re_rand=104.20
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, unified_pos=True, ref=8, bf16 autocast, loss=Huber(beta=1.0)`
- **Key change:** MSE → Smooth L1 / Huber loss (`F.smooth_l1_loss(pred, target, beta=1.0, reduction='sum') / (pred.shape[-1] * count)`) applied to both training loop and evaluate_split. Huber caps outlier gradients from high-Re samples while preserving quadratic behavior near zero.
- **Improvement vs previous best (#1542 T_max=15):** val −4.22 (−3.7%), test −2.40 (−2.3%)
- **Improvement vs directly-comparable seeded baseline (#1577):** val −5.84 (−5.0%), test −6.59 (−6.1%) — exceeds 2× cross-seed σ (~3.5 pts, calibrated by alphonse #1685)
- **Caveat:** cruise split slightly regressed (+7.97 val, +10.46 test vs #1542). Huber pulls down hard high-Re splits at a small cost to the easy cruise split. Net clearly positive.
- **Metric artifacts:** `models/model-charliepai2g48h4-edward-huber-loss-20260512-231342/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-edward --experiment_name "charliepai2g48h4-edward/huber-loss"`

**Open questions after this merge:**
- Cross-seed σ on Huber baseline now needed — alphonse reassigned seed=7 on Huber recipe.
- Askeladd EMA (#1540) still in conflict; rebase + seeded rerun on Huber HEAD is the next stacking test.
- Nezuko #1695 (T_max=18) and frieren #1696 (grad-clip) running against the old T_max=15/MSE base; results remain valid for their respective levers.

---

## Previous bests (chronological)

### 2026-05-12 23:25 — PR #1542: [cosine-trunc-t15] Truncate cosine T_max 50→15 (nezuko)
- **val_avg/mae_surf_p:** 114.81 / **test:** 104.68
- Config: merged recipe + T_max=15 + seed=42. Per-split val: single_in_dist=139.82, geom_camber_rc=120.59, geom_camber_cruise=87.75, re_rand=111.06
- Artifact: `models/model-charliepai2g48h4-nezuko-cosine-trunc-t15-merged-20260512-215533/metrics.jsonl`

### 2026-05-12 23:05 — PR #1577: [seed42-baseline] Seeding + surf_weight=10 rollback (alphonse)
- **val_avg/mae_surf_p:** 116.43 / **test:** 108.87
- Config: merged recipe (unified_pos + bf16 + scoring-fix), surf_weight=10, seed=42, T_max=50
- Adds determinism infrastructure; byte-identical across 2 runs.

### 2026-05-12 20:10 — PR #1512: [scoring-nan-fix] Default config + NaN-fix patch (fern)

- **`val_avg/mae_surf_p`:** **123.99** (best epoch 14, 30-min cap)
- **`test_avg/mae_surf_p`:** **110.97** (from best-val checkpoint)
- **Per-split surface-p MAE (test):** single_in_dist=134.23, geom_camber_rc=121.93, geom_camber_cruise=76.78, re_rand=110.93
- **Config:** default — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, CosineAnnealingLR(T_max=50), AdamW`
- **Metric artifacts:** `models/model-charliepai2g48h4-fern-scoring-nan-fix-20260512-185620/metrics.jsonl`

## Merged improvements (within noise floor)

| PR | Description | val_avg | test_avg | Status |
|---|---|---|---|---|
| #1513 (tanjiro) | bf16 autocast | 125.40 | 126.57 (3-split) | **MERGED** → 24% per-epoch speedup |
| #1416 (thorfinn) | unified_pos=True, ref=8 | 125.78 | 117.12 | **MERGED** → best cruise OOD |
| #1369 (askeladd) | surf_weight=10→20 | 127.94 | 117.35 | **MERGED but effectively reverted** → regression confirmed (#1570: val=127.86), rolled back via #1577 |
| #1577 (alphonse) | seed=42 + surf_weight=10 rollback | 116.43 | 108.87 | MERGED |
| #1542 (nezuko) | T_max=15 cosine truncation | 114.81 | 104.68 | MERGED → superseded by #1374 |
| **#1374 (edward)** | **Huber loss (beta=1.0)** | **110.59** | **102.28** | **MERGED — NEW BEST** |

**Current advisor-branch recipe** (after 7 effective merges):
`unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, loss=Huber(beta=1.0)`

**Comparison threshold:** cross-seed σ ≈ 3.5 val / 0.5 test (calibrated from alphonse #1685 seed=42 vs seed=7 on the pre-Huber recipe). Use 5+ pt val difference as practical significance threshold.

---

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.
- **All future experiments must use `seed=42` to be comparable against this seeded baseline.** Include `--experiment_name` and don't override seed.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
