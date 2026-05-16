# Baseline — icml-appendix-charlie-pai2i-48h-r5

## Current Best

### 2026-05-16 21:15 — PR #4083: bs=2 + n_freqs=8 compound — charliepai2i48h5-alphonse

- **val_avg/mae_surf_p**: **58.27** (best_epoch=18/18, timeout-bound, still descending)
- **test_avg/mae_surf_p**: **51.12** (from best-val checkpoint)
- **Improvement over prior best**: -3.96% val / -3.75% test vs PR #4026 (60.67/53.11)
- **Cumulative improvement**: -54.7% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior (60.67/53.11) |
  |---|---|---|
  | single_in_dist | 57.42 | -0.99% ✓ |
  | geom_camber_rc | 64.11 | -3.45% ✓ |
  | geom_camber_cruise | 33.68 | -5.11% ✓ |
  | re_rand | 49.27 | -6.23% ✓ |
- **Metric artifacts**: `models/model-bf16-layerscale-bs2-n8-20260516-184225/metrics.jsonl`
- **Stack**: BF16 + LayerScale γ-init=0.01 + n_freqs=**8** + **batch_size=2** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **Key findings**:
  - **All 4 test splits improve** — compound is ~89% additive (expected 57.56 from linear decomp, observed 58.27)
  - arm-2 (lr_t_max=18) val=60.75 — regresses; cosine reaches lr≈0 by epoch 18, premature freeze
  - clip_frac drops from 1.000 → 0.987 at epoch 18 — first sign of late-epoch gradient escape
  - Memory unchanged: 18.43 GB (vs 18.5 GB with n=10); throughput 102.4 s/epoch
  - best_epoch=18/18 still timeout-bound — more epochs would help
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --bf16 --batch_size 2 \
      --layer_scale_init 0.01 \
      --n_freqs 8 --huber_delta 0.3 --lr_t_max 20 --grad_clip_max_norm 0.25 \
      --experiment_name bf16-layerscale-bs2-n8 \
      --agent charliepai2i48h5-alphonse
  ```

---

### 2026-05-16 19:10 — PR #4026: batch_size=2 on BF16+LS+n10 — charliepai2i48h5-alphonse

- **val_avg/mae_surf_p**: **60.67** (best_epoch=18/18, timeout-bound, still descending)
- **test_avg/mae_surf_p**: **53.11** (from best-val checkpoint)
- **Improvement over prior best**: -5.32% val / -3.52% test vs PR #4006 (64.08/55.05)
- **Cumulative improvement**: -52.8% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior (64.08/55.05) |
  |---|---|---|
  | single_in_dist | 57.99 | -6.62% ✓ |
  | geom_camber_rc | 66.40 | -2.54% ✓ |
  | geom_camber_cruise | 35.50 | -3.09% ✓ |
  | re_rand | 52.54 | -1.52% ✓ |
- **Metric artifacts**: `models/model-bf16-layerscale-bs2-20260516-162303/metrics.jsonl`
- **Stack**: BF16 + LayerScale γ-init=0.01 + n_freqs=10 + **batch_size=2** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **Key findings**:
  - **All 4 test splits improve** — uniform generalization effect
  - bs=2 gets 4.5× more updates (13,500) than bs=8 (3,008) in same 30-min budget
  - clip_frac=1.0 throughout — per-step magnitude fixed at `0.25 × dir(grad)`, so total steps dominates
  - Mechanism: clip-saturation makes batch_size purely a "steps in budget" lever, not a "gradient quality" lever
  - bs=2 runs at 102.6 s/epoch (vs 111s baseline) — 8% faster
  - Peak memory **18.5 GB** (vs 33 GB baseline) — 44% reduction
  - arm-2 bs=8: val=77.24 / test=67.03 — regresses badly (only 3,008 updates, 73.8 GB memory)
  - **n_freqs=10 (not n=8) used in this experiment** — was assigned before PR #4006 merged. Compounding with n=8 not yet tested.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --bf16 --batch_size 2 \
      --layer_scale_init 0.01 \
      --n_freqs 10 --huber_delta 0.3 --lr_t_max 20 --grad_clip_max_norm 0.25 \
      --experiment_name bf16-layerscale-bs2 \
      --agent charliepai2i48h5-alphonse
  ```

---

### 2026-05-16 17:15 — PR #4006: n_freqs=8 on BF16+LS — charliepai2i48h5-fern

- **val_avg/mae_surf_p**: **64.08** (best_epoch=17/50, timeout-bound, still descending)
- **test_avg/mae_surf_p**: **55.05** (from best-val checkpoint)
- **Improvement over prior best**: -2.47% val / -4.76% test vs PR #4009 (65.70/57.80)
- **Cumulative improvement**: -50.2% val vs round-5 start (~128.69) — **breaks the -50% threshold**
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior (65.70/57.80) |
  |---|---|---|
  | single_in_dist | 62.10 | -4.81% ✓ |
  | geom_camber_rc | 68.13 | -4.62% ✓ |
  | geom_camber_cruise | 36.63 | -4.39% ✓ |
  | re_rand | 53.35 | -5.09% ✓ |
- **Metric artifacts**: `models/model-bf16-layerscale-n8-20260516-144643/metrics.jsonl`
- **Stack**: BF16 + LayerScale γ-init=0.01 + **n_freqs=8** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **Key findings**:
  - **All four test splits improve** — uniform generalization effect, not split-specific
  - n_freqs=8 has space_dim=34 (vs n=10 space_dim=42, n=12 space_dim=50, n=14 space_dim=58)
  - n_freqs ordering on this stack: val n=8 (64.08) < n=12 (65.18) < n=14 (66.99) < n=10 (67.19)
  - test ordering: n=8 (55.05) < n=12 (56.71) < n=10 (58.05) < n=14 (59.31)
  - **Lower aliasing wins decisively** at 1499 train samples — fewer Fourier components reduce noise
  - clip_frac=1.0 throughout (still ran at clip=0.25); ~111 s/epoch; peak memory 36.83 GB
  - **Not yet compounded with clip=1.0** — that combination is the obvious next test
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --bf16 \
      --layer_scale_init 0.01 \
      --n_freqs 8 --huber_delta 0.3 --lr_t_max 20 --grad_clip_max_norm 0.25 \
      --experiment_name bf16-layerscale-n8 \
      --agent charliepai2i48h5-fern
  ```

---

### 2026-05-16 16:55 — PR #4009: Gradient clip relaxation clip=1.0 on BF16+LS+n10 — charliepai2i48h5-nezuko

- **val_avg/mae_surf_p**: **65.70** (best_epoch=17/50, timeout-bound, still descending)
- **test_avg/mae_surf_p**: **57.80** (from best-val checkpoint)
- **Improvement over prior best**: -2.22% val / -0.44% test vs BF16+LS+n10+clip=0.25 (67.19/58.05)
- **Cumulative improvement**: -49.0% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior (67.19/58.05) |
  |---|---|---|
  | single_in_dist | 65.24 | -3.24% ✓ |
  | geom_camber_rc | 71.43 | +2.35% |
  | geom_camber_cruise | 38.31 | -0.91% ✓ |
  | re_rand | 56.21 | -0.25% ✓ |
- **Metric artifacts**: `models/model-bf16-layerscale-clip10-20260516-152629/metrics.jsonl`
- **Stack**: BF16 + LayerScale γ-init=0.01 + n_freqs=10 + Huber-0.3 + T_max=20 + **clip=1.0** (no EMA)
- **Key findings**:
  - clip=0.25 was double-regularizing with LayerScale's gating — clip_frac=1.000 throughout (every step clipped)
  - clip=1.0: clip_frac drops from 1.0 → 0.95 by epoch 17 — first run where ~5% of steps escape clipping
  - The mechanism is mostly "larger effective step" (clip acts as lr-scale in the clip-bound regime)
  - arm-1 (clip=0.5): val=66.07/test=57.69 — also beats baseline but weaker on val
  - LayerScale γ dynamics healthy: no instability, γ_attn 0.009-0.018, γ_mlp 0.032-0.045
  - Peak memory 36.91 GB (same as no-clip-relaxation baseline), ~111 s/epoch
  - **3 out of 4 test splits improve; rc regresses +2.35%**
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --bf16 \
      --layer_scale_init 0.01 \
      --n_freqs 10 --huber_delta 0.3 --lr_t_max 20 --grad_clip_max_norm 1.0 \
      --experiment_name bf16-layerscale-clip10 \
      --agent charliepai2i48h5-nezuko
  ```

---

### 2026-05-16 14:35 — PR #3527: BF16 mixed precision + LayerScale γ=0.01 + n_freqs=10 — charliepai2i48h5-tanjiro

- **val_avg/mae_surf_p**: **67.19** (best_epoch=17/50, timeout-bound, still descending)
- **test_avg/mae_surf_p**: **58.05** (NaN-safe eval)
- **Improvement over prior best**: -5.6% val / -7.4% test vs triple compound (71.20/62.71)
- **Cumulative improvement**: -47.8% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 67.42 | -5.3% ✓ |
  | geom_camber_rc | 69.79 | -3.4% ✓ |
  | geom_camber_cruise | 38.66 | -14.5% ✓ |
  | re_rand | 56.35 | -9.4% ✓ |
- **Metric artifacts**: `models/model-bf16-layerscale-fullstack-20260516-082748/metrics.jsonl`
- **Stack**: BF16 autocast forward + LayerScale γ-init=0.01 + n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 (no EMA, no n_freqs=14)
- **Key findings**:
  - BF16 delivers 1.30× speedup + −21% peak memory (42→33 GB) → 17 epochs vs 12 epochs in 30-min budget
  - With the extended budget, n_freqs=10 BEATS n_freqs=14: lower aliasing risk at 1499 train samples pays off at 17 epochs
  - EMA+n14 (quad-compound) also beats baseline at val=68.50/test=60.15 but is weaker: EMA costs ~9% per epoch (loses ~2 epochs vs n10), and n14 over-fits at this horizon
  - All four test splits improve — uniform convergence effect, not one-split shift
  - BF16 does NOT need GradScaler (same exponent range as FP32); only forward pass is BF16; Huber loss + optimizer in FP32
  - LayerScale γ dynamics unaffected: γ_attn ~0.01 mean, γ_mlp grows 2–3× — same as FP32 baseline
- **Reproduce** (arm-1: BF16 + LayerScale + n_freqs=10, the winner):
  ```bash
  cd target && python train.py --epochs 50 \
      --bf16 \
      --layer_scale_init 0.01 \
      --n_freqs 10 --huber_delta 0.3 --lr_t_max 20 --grad_clip_max_norm 0.25 \
      --experiment_name bf16-layerscale-fullstack \
      --agent charliepai2i48h5-tanjiro
  ```
- **Secondary result** (quad-compound, beats old baseline but weaker than arm-1):
  - val=68.50/test=60.15 — BF16+LS+n14+EMA 0.998 — `models/model-charliepai2i48h5-tanjiro-bf16-quad-compound-20260516-133104/metrics.jsonl`

---

### 2026-05-16 10:45 — PR #3192: EMA 0.998 + LayerScale γ=0.01 + n_freqs=14 on full stack — charliepai2i48h5-edward

- **val_avg/mae_surf_p**: **71.20** (best_epoch=12/50, timeout-bound)
- **test_avg/mae_surf_p**: **62.71** (NaN-safe eval)
- **Improvement over prior best**: -2.16% val / -3.71% test vs LayerScale baseline (72.77/65.12)
- **Cumulative improvement**: -44.7% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 71.22 | -9.65% ✓ |
  | geom_camber_rc | 72.24 | -4.72% ✓ |
  | geom_camber_cruise | 45.19 | +3.04% |
  | re_rand | 62.19 | +0.36% |
- **Metric artifacts**: `models/model-layerscale-001-ema-0998-n14-fullstack-20260516-073558/metrics.jsonl`
- **Stack**: Fourier n_freqs=14 + Huber-0.3 + T_max=20 + clip=0.25 + LayerScale γ-init=0.01 + **EMA decay=0.998**
- **Key finding**: Triple compound (LayerScale + n_freqs=14 + EMA 0.998) yields a further -2.16% val win. EMA enables the LayerScale+n_freqs=14 pairing that alphonse's #3730 confirmed fails without EMA (val=75.76 vs 72.77). EMA checkpoint averaging via `torch.optim.swa_utils.AveragedModel` (decay=0.998) smooths the effective model at best_epoch=12. Biggest OOD gains on single (-9.65%) and rc (-4.72%); cruise marginally regresses (+3.04%) but averages out to a net win. LayerScale γ dynamics healthy: γ_attn mean ~0.01 throughout, γ_mlp growing 2-3× — EMA does not disrupt the selective gating behaviour. Peak memory 48.1 GB, ~152 s/epoch.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 14 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 0.25 \
      --layer_scale_init 0.01 \
      --ema_decay 0.998 \
      --experiment_name layerscale-001-ema-0998-n14-fullstack \
      --agent charliepai2i48h5-edward
  ```

---

### 2026-05-16 06:22 — PR #3593: LayerScale γ-init=0.01 on full stack — charliepai2i48h5-alphonse

- **val_avg/mae_surf_p**: **72.77** (best_epoch=13, timeout-bound)
- **test_avg/mae_surf_p**: **65.12** (NaN-safe eval)
- **Improvement over prior best**: -10.2% val / -8.9% test vs n_freqs=14 baseline (81.08/71.52)
- **Cumulative improvement**: -43.5% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 78.83 | -3.0% ✓ |
  | geom_camber_rc | 75.82 | -10.7% ✓ |
  | geom_camber_cruise | 43.86 | -12.1% ✓ |
  | re_rand | 61.97 | -11.4% ✓ |
- **Metric artifacts**: `models/model-layerscale-0.01-fullstack-20260516-042447/metrics.jsonl`
- **Stack**: Fourier n_freqs=10 + Huber-0.3 + T_max=20 + clip=0.25 + **LayerScale γ-init=0.01**
- **Key finding**: Per-channel learnable residual gain (CaiT/LayerScale) gives massive -10.2% val improvement. Mechanism: with init=0.01, `gamma_attn` stays near zero (mean ~0.01, max ~0.085 per block — sparse channel selectivity) while `gamma_mlp` grows 3× (mean ~0.03-0.04, max ~0.12 — MLP residual stream unlocked). EVERY test split improves (single -3%, rc -10.7%, cruise -12.1%, re_rand -11.4%) — best OOD generalization seen this round. arm-1 (γ-init=0.1) also beat baseline (val=74.74) but with high run-to-run variance (~7-point spread across two seeds); γ=0.01 had cleaner dynamics. clip_frac=1.000 throughout — clip=0.25 still rate-limiting; LayerScale wins purely through model-side selective residuals, not gradient regularization. Stack still uses n_freqs=10 not 14 (alphonse rebased before #3438 merged); combining LayerScale + n_freqs=14 is the next high-priority compound test.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 0.25 \
      --layer_scale_init 0.01 \
      --experiment_name layerscale-0.01-fullstack \
      --agent charliepai2i48h5-alphonse
  ```

---

### 2026-05-16 03:25 — PR #3438: Fourier n_freqs=14 on full stack (deconfounded) — charliepai2i48h5-nezuko

- **val_avg/mae_surf_p**: **81.08** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **71.52** (NaN-safe eval)
- **Improvement over prior best**: -3.5% val / -2.0% test vs clip=1.0 baseline (84.01/72.95)
- **Cumulative improvement**: -37.1% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 81.31 | -1.9% ✓ |
  | geom_camber_rc | 84.95 | +0.7% |
  | geom_camber_cruise | 49.89 | -3.5% ✓ |
  | re_rand | 69.92 | -1.5% ✓ |
- **Metric artifacts**: `models/model-fourier-n14-fullstack-20260516-002646/metrics.jsonl`
- **Key finding**: Clean isolated test of n_freqs: 10 → 14 on full stack (Huber+T_max=20+clip=0.25). Improved every single test split, largest gain on single_in_dist (-6.4% vs n=10 baseline). Spectrum still not saturated at ep14 (both arms timeout-bound, best epoch = last). Prior confounded run (n=14 + clip added simultaneously) had regression due to clip addition; this run adds only n_freqs. n=12 is roughly a wash on test (+0.6%); n=14 is the sweet spot at this budget. clip_frac=1.000 throughout — clip=0.25 still too tight with n=14 (same as n=10). Note: this stack uses clip=0.25 not clip=1.0; combining clip=1.0 with n=14 is the next high-priority test.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 14 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 0.25 \
      --experiment_name fourier-n14-fullstack \
      --agent charliepai2i48h5-nezuko
  ```

---

### 2026-05-16 03:21 — PR #3529: Grad-clip relaxed to 1.0 on full stack — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **84.01** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **72.95** (NaN-safe eval)
- **Improvement over prior best**: -0.69% val / -1.27% test vs clip=0.25 baseline (84.59/73.89)
- **Cumulative improvement**: -34.7% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p | Δ vs prior |
  |---|---|---|
  | single_in_dist | 82.86 | -4.6% ✓ |
  | geom_camber_rc | 84.34 | -2.2% ✓ |
  | geom_camber_cruise | 53.59 | +4.1% |
  | re_rand | 71.01 | 0.0% |
- **Metric artifacts**: `models/model-fourier-tmax20-clip10-20260516-013254/metrics.jsonl`
- **Key finding**: clip=1.0 outperforms clip=0.25. clip_frac drops below 1.0 starting at epoch 10 (0.997 at ep10, 0.984 at ep14) — the only threshold where the clip stops saturating within the 14-epoch budget. Pre-clip grad_norm_mean at ep14 is ~5.4; clip=0.25 is 21× below this, eliminating almost all gradient magnitude information. clip=1.0 is the first threshold where the clip is actually adaptive. cruise split slightly regresses (+4.1%) but single and rc show clear improvement.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 1.0 \
      --experiment_name fourier-tmax20-clip10 \
      --agent charliepai2i48h5-frieren
  ```

---

### 2026-05-15 23:28 — PR #3333: Fourier n=10 + Huber-0.3 + T_max=20 + clip=0.25 — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **84.59** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **73.89** (NaN-safe eval)
- **Improvement over prior best**: -5.2% val / -7.0% test vs Fourier-only baseline (89.27/79.43)
- **Cumulative improvement**: -34% val vs round-5 start (~128.69)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 86.87 |
  | geom_camber_rc | 86.21 |
  | geom_camber_cruise | 51.47 |
  | re_rand | 71.01 |
- **Metric artifacts**: `models/model-fourier-n10-tmax20-clip025-20260515-222425/metrics.jsonl`
- **Key finding**: All four orthogonal improvements compose cleanly: Fourier n=10 + Huber delta=0.3 + LR cosine T_max=20 + grad_clip_max_norm=0.25. Monotone val improvement across all 14 epochs (still learning at timeout). clip_frac=1.0 at 0.25 with Fourier — clip is still doing real work. Cruise and re_rand splits show largest absolute gains (-9.6% / -9.2%).
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --lr_t_max 20 \
      --grad_clip_max_norm 0.25 \
      --experiment_name fourier-n10-tmax20-clip025 \
      --agent charliepai2i48h5-frieren
  ```

---

### 2026-05-15 19:52 — PR #3221: Fourier positional features (n_freqs=10) — charliepai2i48h5-nezuko

- **val_avg/mae_surf_p**: **89.27** (best_epoch=14/14, timeout-bound)
- **test_avg/mae_surf_p**: **79.43** (NaN-safe eval)
- **Improvement over prior best**: -9.5% val / -9.9% test vs Huber-0.3+clip-0.25 (98.62/88.14)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 93.65 |
  | geom_camber_rc | 88.94 |
  | geom_camber_cruise | 56.92 |
  | re_rand | 78.20 |
- **Metric artifacts**: `models/model-charliepai2i48h5-nezuko-fourier-n10-20260515-191358/metrics.jsonl`
- **Key finding**: Replacing raw (x,z) coordinates with multi-frequency Fourier positional embeddings (sin/cos at log-spaced frequencies) gives a 9.5% val improvement with near-zero parameter overhead (~4k extra params). `space_dim = 2 + 4*n_freqs = 42`. Best epoch was the last wall-clock-capped epoch — improvement is NOT from running longer, the budget cutoff fired before epoch 50.
- **Stack**: Huber delta=0.3 (no grad_clip in this run); Fourier features alone beat the Huber+clip baseline.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --n_freqs 10 \
      --huber_delta 0.3 \
      --experiment_name fourier-n10 \
      --agent charliepai2i48h5-nezuko
  ```

---

### 2026-05-15 19:26 — PR #3182: Huber loss + gradient clipping (clip=0.25) — charliepai2i48h5-askeladd

- **val_avg/mae_surf_p**: **98.62** (best_epoch=14/50)
- **test_avg/mae_surf_p**: **88.14** (NaN-safe eval)
- **Improvement over prior best**: -4.4% val / -4.2% test vs Huber-only (103.18/92.02)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 104.75 |
  | geom_camber_rc | 104.65 |
  | geom_camber_cruise | 59.24 |
  | re_rand | 83.90 |
- **Metric artifacts**: `models/model-charliepai2i48h5-askeladd-huber-0.3-clip-0.25-20260515-182526/metrics.jsonl`
- **Key finding**: Huber-0.3 + grad_clip=0.25 are additive — both attack heavy-tail gradients at different scales (per-sample residual vs batch-level update). clip_frac=1.0 at both 0.5 and 0.25; residual tail pressure signal still present after Huber, so clipping contributes genuine variance reduction.
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --experiment_name huber-0.3-clip-0.25 \
      --huber_delta 0.3 \
      --grad_clip_max_norm 0.25 \
      --agent charliepai2i48h5-askeladd
  ```

---

### 2026-05-15 16:28 — PR #3213: Huber loss (delta=0.3) — charliepai2i48h5-frieren

- **val_avg/mae_surf_p**: **103.18** (best_epoch=13/50)
- **test_avg/mae_surf_p**: **92.02** (NaN-safe re-eval; baseline eval had NaN from data bug)
- **Per-split test surface p MAE**:
  | Split | test surf_p |
  |---|---|
  | single_in_dist | 111.93 |
  | geom_camber_rc | 102.85 |
  | geom_camber_cruise | 62.84 |
  | re_rand | 90.45 |
- **Metric artifacts**: `models/model-huber-0.3-20260515-140457/metrics.jsonl`
- **Also included**: NaN-safe `evaluate_split` fix (sample-level skip for non-finite GT, works around data bug in `test_geom_camber_cruise/000020.pt`)
- **Reproduce**:
  ```bash
  cd target && python train.py --epochs 50 \
      --experiment_name huber-0.3 \
      --huber_delta 0.3 \
      --agent charliepai2i48h5-frieren
  ```

## Reference Configuration (baseline `train.py`)
- Model: Transolver
  - n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  - space_dim=2, fun_dim=22, out_dim=3
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Cosine annealing LR schedule, T_max=epochs
- batch_size=4
- surf_weight=10.0 (loss = vol_loss + 10 * surf_loss)
- Default epochs=50 cap, SENPAI_TIMEOUT_MINUTES wall-clock cap
- Balanced WeightedRandomSampler across domains (single/RC-tandem/cruise-tandem)
- Loss in normalized target space; metrics in denormalized physical units

## Splits (lower is better — surface MAE on pressure)
| Split | Test source | Notes |
|---|---|---|
| val_single_in_dist | random holdout from single-foil | sanity |
| val_geom_camber_rc | raceCar M=6-8 front foil | geometry extrapolation |
| val_geom_camber_cruise | cruise M=2-4 front foil | geometry extrapolation |
| val_re_rand | stratified Re across all tandem domains | Re generalization |

Primary metric: equal-weight average across all 4 splits.

## Notes
- Round 5, charlie arm, 48h budget, local JSONL metrics only.
- 8 students, 1 GPU (96GB) each.
- First batch of hypotheses includes a clean baseline run to anchor the metric.
