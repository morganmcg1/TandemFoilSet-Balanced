# SENPAI Research Results

<!-- This log is maintained on the icml-appendix-charlie-pai2f-r5 advisor branch. -->
<!-- Each entry records a reviewed PR with metrics and analysis. -->

## Round 5 Experiments — Launched 2026-04-29

All 8 students assigned WIP experiments on 2026-04-29. No results yet — experiments in progress.

| PR | Student | Experiment | Status |
|----|---------|------------|--------|
| #1118 | edward | epochs=50 extended training | WIP |
| #1119 | thorfinn | cosine eta_min=5e-5 | WIP |
| #1120 | nezuko | n_layers=2 shallower model | WIP |
| #1121 | fern | huber_delta=0.1 tighter loss | WIP |
| #1122 | alphonse | lr=1e-3 higher LR | WIP |
| #1123 | tanjiro | n_hidden=320 wider model | WIP |
| #1124 | askeladd | weight_decay=0 no L2 | WIP |
| #1125 | frieren | surf_weight=5 reduced surface emphasis | WIP |

## Prior Round Winners (Full History, most recent first)

### PR #1050 — PSN + epochs=30 on compound stack (2026-04-29) — CURRENT BEST
**Student:** charliepai2e1-edward | **Branch:** charliepai2e1-edward/psn-plus-epochs-30

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **61.5855** (epoch 22/30 — terminated by 30-min timeout) |
| `val_single_in_dist/mae_surf_p` | 68.3069 |
| `val_geom_camber_rc/mae_surf_p` | 72.6498 |
| `val_geom_camber_cruise/mae_surf_p` | 44.8940 |
| `val_re_rand/mae_surf_p` | 60.4914 |
| `val_avg/mae_surf_Ux` | 0.9179 |
| `val_avg/mae_surf_Uy` | 0.4509 |
| `val_avg/mae_vol_p` | 67.5720 |
| `test_avg/mae_surf_p` | **54.3573** |
| `test_single_in_dist/mae_surf_p` | 61.7523 |
| `test_geom_camber_rc/mae_surf_p` | 64.2811 |
| `test_geom_camber_cruise/mae_surf_p` | 37.8047 |
| `test_re_rand/mae_surf_p` | 53.5912 |
| `test_avg/mae_surf_Ux` | 0.8541 |
| `test_avg/mae_surf_Uy` | 0.4187 |
| `test_avg/mae_vol_p` | 60.2983 |

- **vs prior baseline (PR #1015):** 61.5855 vs 66.8085 → -7.8% val improvement
- **Test improvement:** 54.3573 vs 58.7266 → -7.4% test improvement
- **Model parameters:** 1,606,219 | **Peak VRAM:** ~30.44 GB | **Train time:** ~30 min (hit timeout)
- **Note:** Val still falling ~2.8%/epoch at epoch 22 when 30-min timeout hit (LR=8.27e-5). More epochs likely to yield further gains.

### PR #1015 — Longer training: epochs=24 on compound stack (2026-04-28)
**Student:** charliepai2e1-edward | **Branch:** charliepai2e1-edward/longer-training-epochs-24

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **66.8085** (epoch 22/24 — timeout at 30 min) |
| `val_single_in_dist/mae_surf_p` | 73.9641 |
| `val_geom_camber_rc/mae_surf_p` | 79.1014 |
| `val_geom_camber_cruise/mae_surf_p` | 48.9877 |
| `val_re_rand/mae_surf_p` | 65.1809 |
| `test_avg/mae_surf_p` | **58.7266** |
| `test_single_in_dist/mae_surf_p` | 67.5104 |
| `test_geom_camber_rc/mae_surf_p` | 70.2042 |
| `test_geom_camber_cruise/mae_surf_p` | 40.5897 |
| `test_re_rand/mae_surf_p` | 56.6022 |

- **vs prior baseline (PR #795):** 66.8085 vs 90.4014 → -26.1% val improvement
- **Test improvement:** 58.7266 vs 80.3748 → -27.0% test improvement
- **Peak VRAM:** 30.45 GB | **Train time:** 30.42 min (hit 30-min timeout)

### PR #795 — Per-sample loss normalization (PSN) on compound stack (2026-04-28)
**Student:** charliepai2e1-thorfinn | **Branch:** charliepai2e1-thorfinn/per-sample-loss-norm

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **90.4014** (epoch 12/12) |
| `val_single_in_dist/mae_surf_p` | 108.5561 |
| `val_geom_camber_rc/mae_surf_p` | 101.4393 |
| `val_geom_camber_cruise/mae_surf_p` | 66.9027 |
| `val_re_rand/mae_surf_p` | 84.7074 |
| `test_avg/mae_surf_p` | **80.3748** |

- **vs prior baseline (PR #1005):** 90.4014 vs 94.6541 → -4.50% improvement

### PR #1005 — n_layers=3, slice_num=16 reference architecture (2026-04-29)
**Student:** charliepai2e1-edward

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **94.6541** (epoch 12/12) |
| `test_avg/mae_surf_p` | **83.7608** |

- **vs prior baseline (PR #882):** 94.6541 vs 103.2182 → -8.31% improvement

### PR #882 — EMA model weights (decay=0.999) on compound baseline (2026-04-29)
**Student:** charliepai2e1-nezuko

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **103.2182** (epoch 10/12) |
| `test_avg/mae_surf_p` | **92.4867** |

- **vs prior baseline (PR #808):** 103.22 vs 104.11 → -0.86% improvement

### PR #808 — bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12 (2026-04-28)
**Student:** charliepai2e1-fern

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **104.1120** (epoch 10/12) |
| `test_avg/mae_surf_p` | **94.7010** |

- **vs prior baseline (PR #827):** 104.11 vs 109.57 → -4.97% improvement

### PR #827 — Huber loss + surf_weight=30 (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.5716** (epoch 13/14) |

- **vs Huber baseline (PR #788):** 109.57 vs 115.65 → -5.26% improvement

### PR #788 — Huber loss instead of MSE (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (epoch 10/14) |

- **vs MSE baseline:** 115.65 vs 126.88 → -8.85% improvement

## Infrastructure Fixes

### NaN guard (PR #792)
- `--grad_clip 1.0` + upstream pred/GT sanitization in `evaluate_split` resolves NaN propagation.
- Root cause: IEEE 754 `Inf * False (==0.0) = NaN` — `(pred - y).abs()` computed before masking.
- `test_geom_camber_cruise/000020.pt` has 761 Inf values in p channel — correctly skipped (n_skipped_nonfinite=1).

### accumulate_batch NaN bug fix (PR #791)
- `0 * NaN = NaN` in `evaluate_split` — fixed in PR #791. All subsequent experiments include this fix.
