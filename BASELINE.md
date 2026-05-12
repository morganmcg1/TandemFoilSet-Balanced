# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best

### 2026-05-12 23:25 — PR #1542: [cosine-trunc-t15] Truncate cosine T_max 50→15 to anneal inside cap (nezuko)

- **`val_avg/mae_surf_p`:** **114.81** (best epoch 17/18)
- **`test_avg/mae_surf_p`:** **104.68** (from best-val checkpoint)
- **Per-split surface-p MAE (val):** single_in_dist=139.82, geom_camber_rc=120.59, geom_camber_cruise=87.75, re_rand=111.06
- **Per-split surface-p MAE (test):** single_in_dist=120.31, geom_camber_rc=113.90, geom_camber_cruise=75.41, re_rand=109.09
- **Config:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, epochs=50, seed=42, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW, unified_pos=True, ref=8, bf16 autocast`
- **Key change:** `CosineAnnealingLR(T_max=50)` → `CosineAnnealingLR(T_max=15)`. T_max=15 matches the achievable epoch count under the 30-min cap, so the schedule actually anneals to lr≈0 around epoch 16; best-val is in the second cycle's near-zero-lr regime (epoch 17, lr=5.46e-6).
- **Caveat:** Nezuko's run was on the pre-rollback advisor base (surf_weight=20.0, no seed). The 3-way squash merge correctly resolved surf_weight=10 + seed=42 + T_max=15 in the final state. A seeded confirmation run on this exact recipe is expected from the other rebase students (edward Huber, askeladd EMA).
- **Metric artifacts:** `models/model-charliepai2g48h4-nezuko-cosine-trunc-t15-merged-20260512-215533/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --agent charliepai2g48h4-nezuko --experiment_name "charliepai2g48h4-nezuko/cosine-trunc-t15-merged"`

**Note on schedule:** PyTorch's `CosineAnnealingLR` continues cycling past `T_max`. Best epoch (17) lands in the second cycle's lr≈0 climb-back regime, suggesting `T_max=18` (matching achievable epoch count exactly) may give another 1-2 pts. Follow-up assigned to nezuko.

**Open questions after this merge:**
- Across-seed σ is still unknown. Alphonse #1685 (seed=7) will calibrate.
- T_max=15 vs T_max=18 (matching epoch count exactly) — nezuko follow-up
- Edward Huber rebase (#1374) is highest-priority next merge candidate (was val=112.06 unseeded on default config; stacks with this schedule should land sub-112).

---

## Previous bests (chronological)

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
| **#1542 (nezuko)** | **T_max=15 cosine truncation** | **114.81** | **104.68** | **MERGED — NEW BEST** |

**Current advisor-branch recipe** (after 6 effective merges):
`unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, seed=42, batch_size=4, CosineAnnealingLR(T_max=15, eta_min=0.0), AdamW`

**Comparison threshold:** with seeding, σ is unknown. Need a seed=7 cross-check (alphonse #1685 in flight). Use 5+ pt val difference as practical significance threshold until across-seed σ is measured.

---

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.
- **All future experiments must use `seed=42` to be comparable against this seeded baseline.** Include `--experiment_name` and don't override seed.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
