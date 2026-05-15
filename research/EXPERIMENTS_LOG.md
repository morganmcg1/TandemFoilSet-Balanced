# SENPAI Research Results — charlie-pai2i-24h-r4

Per-PR results log. Earliest at the bottom; latest at the top.

## 2026-05-15 14:35 — PR #3226: H10 Re-stratified sampler (thorfinn) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-thorfinn/re-strat-high2x`
- Hypothesis: Upweight Re>1e6 samples by 2x in the `WeightedRandomSampler`, on top of the existing domain-balanced weights. Targets the higher-error high-Re regime which dominates `val_avg/mae_surf_p` via the per-sample y std (164→2077 across the dataset).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 14) | **127.84** |
| `val_single_in_dist/mae_surf_p` | 160.10 |
| `val_geom_camber_rc/mae_surf_p` | 148.67 |
| `val_geom_camber_cruise/mae_surf_p` | 91.50 |
| `val_re_rand/mae_surf_p` | 111.08 |
| `test_avg/mae_surf_p` | NaN (data quirk; 3 finite splits) |
| Re>1e6 samples upweighted | 1303 / 1499 (≈87%) |
| Wall clock | 30 min cap; epoch 14 of 50 |

- Metric artifact: `models/<experiment>/metrics.jsonl` on the student branch.
- Diagnostic: `val_re_rand` (111.08) and `val_geom_camber_cruise` (91.50) — the OOD-Re-stratified splits — were the two lowest, consistent with the high-Re upweight paying off where the test boundary stresses Re generalization.
- Implementation note: student verified that stored `x[..., 13]` is **already** `log(Re)` (not normalized), so used `log_re = float(x_i[0, 13].item())` with `threshold = log(1e6)` instead of denormalizing via stats. Mathematically equivalent to the PR-body recipe.
- Decision: **Merge** — clear winner, simple sampler change, +Re-strat now part of the baseline.

## 2026-05-15 14:30 — PR #3197: H8 EMA model weights (askeladd) — sent back

- Branch: `charliepai2i24h4-askeladd/ema-weights-decay-0p999`
- Hypothesis: Maintain shadow EMA (decay=0.999) of model weights and evaluate val/test against the EMA model, not the live model. Reduces step-to-step variance and tends to improve OOD generalization.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best EMA, epoch 13) | 132.17 |
| `test_avg/mae_surf_p` | NaN (data quirk) |
| Live val_avg (best) | similar/slightly worse than EMA |
| Peak memory | 42.11 GB |

- Diagnostic: EMA worked as designed — clean dual live/EMA tracking, EMA consistently beat live across epochs. But the absolute number is now above the new baseline (127.84 from H10).
- Decision: **Send back** — Re-run EMA on top of the merged Re-strat baseline. Mechanism is orthogonal and should stack.

## 2026-05-15 14:30 — PR #3224: H13 Persistent geometry conditioning (tanjiro) — sent back

- Branch: `charliepai2i24h4-tanjiro/geom-cond-additive`
- Hypothesis: Inject the per-sample global geometry context (dims 13-23: Re, AoA, NACA params, gap, stagger) at every Transolver block via a gated additive projection, GALE-style.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 10) | 134.31 |
| `val_geom_camber_cruise/mae_surf_p` | 103.45 (lowest of 4 splits) |
| `val_geom_camber_rc/mae_surf_p` | 141.89 |
| `test_avg/mae_surf_p` | NaN (data quirk) |
| Param count | 698K (+36K vs baseline) |
| Final geom_gates | [0.05, 0.13, 0.16, 0.17, 0.17] |
| Wall clock | 30 min cap; epoch 14 of 50 |

- Diagnostic: Gates started at 0 (identity init) and learned to non-zero values monotonically — the model used the conditioning. But training was cap-bound, cosine LR barely decayed (ratio 0.83 at e14), val_avg still oscillating.
- Decision: **Send back** — Re-run on merged baseline, fix cap issue (reduce slice_num=32 or T_max=15) so cosine actually anneals within the time budget.

## 2026-05-15 14:25 — PR #3210: H2 Scale Transolver to ~4M params (fern) — sent back

- Branch: `charliepai2i24h4-fern/scale-256x6x8-lr3e4`
- Hypothesis: Scale n_hidden=128→256, n_layers=5→6, n_head=4→8 (≈4M params).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 6) | 158.40 |
| Param count | 3.01M |
| Peak memory | 94 GB at bs=4 → fell back to bs=2 |
| Wall clock | 30 min cap; epoch 6 of 7 |

- Diagnostic: Training very noisy (val_avg 158 → 239 → 158 over 7 epochs). Cap-bound, bs=2 after OOM fallback. Capacity vs. epochs tradeoff is paying too much for epochs.
- Decision: **Send back** — Add `clip_grad_norm_(1.0)`, drop lr to 2e-4, try mid-size variant (n_hidden=192, n_layers=6, n_head=6) to fit ~12-15 epochs.

## Known branch-wide quirk

`data/scoring.py` (read-only per program.md) accumulates MAE via `(pred - y).abs() * surf_mask`. Sample 20 in `test_geom_camber_cruise` has 761 `inf` values in `y[..., 2]` (p channel). `NaN * 0 = NaN` (IEEE 754) propagates through the multiplication, contaminating `test_avg/mae_surf_p` for every experiment. Both askeladd and tanjiro independently identified this. **Workaround:** `val_avg/mae_surf_p` is the canonical ranking metric. `test_avg/mae_surf_p` should be reported as NaN-aware (the 3 finite test splits averaged, or skip).
