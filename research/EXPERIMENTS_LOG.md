# SENPAI Research Results — charlie-pai2i-24h-r4

Per-PR results log. Earliest at the bottom; latest at the top.

## 2026-05-15 18:20 — PR #3326: H12 MLP dropout=0.1 (fern) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-fern/mlp-dropout`
- Hypothesis: Add `dropout=0.1` to `MLP` class and apply in `TransolverBlock.mlp` FFN sub-layers. `PhysicsAttention`, preprocess MLP, and final head remain at `dropout=0.0`. With only 1499 training samples, the FFN path was memorizing training-distribution feature correlations.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 13) | **112.49** |
| `val_single_in_dist/mae_surf_p` | 136.83 |
| `val_geom_camber_rc/mae_surf_p` | 118.25 |
| `val_geom_camber_cruise/mae_surf_p` | 87.31 |
| `val_re_rand/mae_surf_p` | 107.55 |
| `test_avg/mae_surf_p` | **104.83** |
| `test_single_in_dist/mae_surf_p` | 126.77 |
| `test_geom_camber_rc/mae_surf_p` | 112.01 |
| `test_geom_camber_cruise/mae_surf_p` | 75.35 |
| `test_re_rand/mae_surf_p` | 105.20 |
| Wall clock | 30 min cap; epoch 14 of 50; best at epoch 13 |

- Metric artifact: `models/model-fern-mlp-dropout-0p1-20260515-163433/metrics.jsonl`
- Diagnostic: Clean OOD-dominant signature: geom_camber_cruise -14.1%, re_rand -9.6%, geom_camber_rc -6.1%, single_in_dist -5.4% on val. Test single_in_dist slightly regressed (+2.3%) — classic regularizer tradeoff near the sweet spot. The regularizer reduces memorization of training-set feature correlations, which pays dividends most where the test distribution differs from train.
- Follow-up: fern assigned H12b dropout rate sweep {0.05, 0.15, 0.20} to find optimal beyond 0.1.
- Decision: **Merge** — -8.4% val_avg, strongest single-PR improvement so far. Clean implementation, OOD signature exactly as hypothesized.

## 2026-05-15 18:20 — PR #3222: H9 Cautious AdamW (nezuko) — **sent back, v2 rerun needed**

- Branch: `charliepai2i24h4-nezuko/cautious-adamw`
- Hypothesis: Replace AdamW with from-scratch CautiousAdamW that masks update components where sign(m_t) ≠ sign(g_t).

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 13) | **118.91** |
| `val_single_in_dist/mae_surf_p` | 143.16 |
| `val_geom_camber_rc/mae_surf_p` | 128.82 |
| `val_geom_camber_cruise/mae_surf_p` | 94.85 |
| `val_re_rand/mae_surf_p` | 108.80 |
| `test_avg/mae_surf_p` | 109.23 |
| Wall clock | 30 min cap; epoch 14 of 50; best at epoch 13 |

- Metric artifact: `models/model-charliepai2i24h4-nezuko-cautious-adamw-20260515-163837/metrics.jsonl`
- Diagnostic: v1 beat old baseline 122.81 → 118.91 (-3.2%). But fern's dropout merged during review, raising the bar to 112.49. v1 is above the new target. CONFLICTING (needs rebase, had duplicate NaN workaround). Sent back for v2 rerun on top of new baseline (including H12 dropout) to test composition.
- Decision: **Sent back** — mechanism is proven; v2 tests composition with dropout.

## 2026-05-15 18:20 — PR #3201: H3 channel-weighted surface loss p=3 (edward) — **sent back, p=1.5 rerun**

- Branch: `charliepai2i24h4-edward/channel-weighted-surf-loss-p3x`
- Hypothesis: Weight pressure channel 3× in surface loss (ch_weights=[1,1,3] / mean), keeping surf_weight=10.0.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best run, epoch 14) | 138.39 |
| `val_single_in_dist/mae_surf_p` | 178.75 (178.75 vs 144.70 baseline = +23.5%) |
| `val_geom_camber_cruise/mae_surf_p` | 99.36 (-2.2% slight improvement) |
| `test_avg/mae_surf_p` (3-split partial) | 138.95 |
| Seeds tested | 4 runs (137.64 to 149.19 range, ~8% spread) |

- Diagnostic: Over-emphasis on pressure at the cost of velocity coupling. Single_in_dist badly regressed. Only cruise improved marginally. v2 (p=1.5, milder) sent back to test: if still regresses, channel-reweighting direction closed.
- Decision: **Sent back** — milder ratio one more try; if p=1.5 also regresses, hypothesis closed.

## 2026-05-15 17:00 — PR #3291: H7 two-branch output head (thorfinn) — **CLOSED, regressed**

- Branch: `charliepai2i24h4-thorfinn/two-branch-head`
- Hypothesis: Replace the shared output MLP with two separate decoders — a wider `surf_head` (n_layers=2) and a narrower `vol_head` (n_layers=1) — to let surface and volume predictions specialize their feature pathways.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 15) | **135.74** |
| `val_single_in_dist/mae_surf_p` | 166.35 |
| `val_geom_camber_rc/mae_surf_p` | 139.93 |
| `val_geom_camber_cruise/mae_surf_p` | 111.16 |
| `val_re_rand/mae_surf_p` | 125.53 |
| `test_avg/mae_surf_p` | NaN (run before frieren NaN fix) |
| Param count | 670,810 (~0.67M) |
| Wall clock | 30 min cap; epoch 15 of 50 |

- Metric artifact: `models/model-charliepai2i24h4-thorfinn-two-branch-head-*/metrics.jsonl`
- Diagnostic: 3 of 4 splits regressed vs current baseline 122.81 (+10.5% worse val_avg). Only geom_camber_rc nominally improved (139.93 vs 125.95 — but still well above baseline). Student correctly self-assessed: "hypothesis did NOT pan out."
- Root cause analysis: The shared-decoder's cross-channel feature sharing likely provides implicit regularization that benefits generalization. Separating surface/volume decoders loses this shared representation and adds little value with only ~0.67M params. The in-dist split (166.35) regressed the most, suggesting the two-branch design does not help the bottleneck split.
- Decision: **Closed** — unambiguous regression on all splits including the target split (single_in_dist). Thorfinn reassigned to H11 (log1p target normalization, PR #3345).

## 2026-05-15 16:25 — PR #3217: H5 RFF coord encoding + NaN fix (frieren) — **MERGED, new baseline**

- Branch: `charliepai2i24h4-frieren/rff-coord-nfreq32-sigma1`
- Hypothesis: Replace raw (x,z) positional dims 0-1 with a fixed 64-dim Random Fourier Feature expansion (n_freq=32, sigma=1.0) to lift spectral bandwidth for boundary-layer gradients. Bonus: added a `y_finite_sample` mask + `nan_to_num` in `evaluate_split` to resolve the branch-wide `test_avg` NaN.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, epoch 12) | **122.81** |
| `val_single_in_dist/mae_surf_p` | 144.70 |
| `val_geom_camber_rc/mae_surf_p` | 125.95 |
| `val_geom_camber_cruise/mae_surf_p` | 101.61 |
| `val_re_rand/mae_surf_p` | 119.00 |
| `test_avg/mae_surf_p` (now finite!) | **111.16** |
| `test_single_in_dist/mae_surf_p` | 123.91 |
| `test_geom_camber_rc/mae_surf_p` | 114.82 |
| `test_geom_camber_cruise/mae_surf_p` | 88.14 |
| `test_re_rand/mae_surf_p` | 117.78 |
| Wall clock | 30 min cap; epoch 12 of 50 |

- Metric artifact: `models/model-frieren-rff-nfreq32-sigma1-20260515-140556/metrics.jsonl`
- Diagnostic: Training curve showed pronounced noise until cosine annealing began to bite at epoch 11, with a sharp ~30-point drop then plateau (156→125→122→122). The RFF expansion gave the preprocess MLP a higher-frequency vocabulary for boundary-layer features; the gain shows up most in `val_geom_camber_rc` (-22.7 vs Re-strat baseline) and `val_single_in_dist` (-15.4).
- NaN fix: `evaluate_split` now masks and zero-fills samples where `isfinite(y).all(dim=-1)` is False before computing `err = (pred - y).abs()`. This resolves the IEEE 754 `NaN * 0 = NaN` propagation through `surf_mask` for test_geom_camber_cruise sample 20.
- Decision: **Merge** — -4.3% val_avg improvement, clean RFF implementation with buffer-registered B matrix (non-trainable), plus a valuable branch-wide bug fix now baked in.

## 2026-05-15 14:35 — PR #3226: H10 Re-stratified sampler (thorfinn) — **MERGED**

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
