# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best

### 2026-05-12 23:05 — PR #1577: [seed42-baseline] Seeding + surf_weight=10 rollback (alphonse)

- **`val_avg/mae_surf_p`:** **116.43** (best epoch 18/18 completed; still descending at timeout)
- **`test_avg/mae_surf_p`:** **108.87** (from best-val checkpoint)
- **Per-split surface-p MAE (val):** single_in_dist=131.15, geom_camber_rc=121.66, geom_camber_cruise=100.47, re_rand=112.46
- **Per-split surface-p MAE (test):** single_in_dist=117.35, geom_camber_rc=113.75, geom_camber_cruise=86.85, re_rand=117.52
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=50), AdamW, unified_pos=True, ref=8, bf16 autocast`
- **Key change:** Adds deterministic seeding (`seed=42`, `cudnn.deterministic=True`, seeded DataLoader/sampler). Also effectively rolls back `surf_weight` 20→10 (alphonse's branch predated #1369 merge; squash-merge 3-way resolution confirmed surf_weight=10 is correct).
- **Determinism:** Two independent runs produce byte-identical metrics. σ across seeds is unknown — see "open questions" below.
- **Metric artifacts:** `models/model-charliepai2g48h4-alphonse-seed42-baseline-20260512-215112/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-alphonse --experiment_name "charliepai2g48h4-alphonse/seed42-baseline"`

**Note:** val=116.43 was achieved with surf_weight=10 on the 3-merge recipe (unified_pos + bf16 + scoring-fix). The surf_weight=20 (PR #1369) is now rolled back — fern's #1570 confirmed it was a regression (+7.5% test on merged recipe) and alphonse's seeded result (116.43 at surf_weight=10) is 9.6% better than fern's unseeded merged recipe (127.86 at surf_weight=20).

**Open questions after this merge:**
- Across-seed σ is unknown for the new recipe. A seed=7 rerun would establish it.
- The val curve was still descending at epoch 18/50 — the model is undertrained in 30 min.
- All 3 rebase-in-flight PRs (#1374 Huber, #1540 EMA, #1542 T15) will now inherit seed=42 + surf_weight=10 on rebase.

---

## Previous best

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
| **#1577 (alphonse)** | **seed=42 + surf_weight=10 rollback** | **116.43** | **108.87** | **MERGED — NEW BEST** |

**Current advisor-branch recipe** (after 5 effective merges, surf_weight=20 rolled back):
`unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=50), AdamW`

**Comparison threshold:** with seeding, σ is unknown. Need a seed=7 cross-check. Use 5+ pt val difference as practical significance threshold until across-seed σ is measured.

---

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.
- **All future experiments must use `seed=42` to be comparable against this seeded baseline.** Include `--experiment_name` and don't override seed.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
