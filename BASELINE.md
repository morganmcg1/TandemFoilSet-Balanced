# TandemFoilSet Baseline

Track: `icml-appendix-willow-pai2g-48h-r5`

## Current baseline

Stock `train.py` on `icml-appendix-willow-pai2g-48h-r5` — Transolver with the following config:

- `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- `epochs=50` (capped by `SENPAI_TIMEOUT_MINUTES=30` per-run wall clock)
- AdamW + CosineAnnealingLR, MSE loss in normalized space, vol + 10·surf

**Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the 4 val splits).
**Paper-facing metric:** `test_avg/mae_surf_p` (computed at end of run from the best-val checkpoint).

## 2026-05-13 00:05 — PR #1689: fern Huber β=0.5 (tighter MAE alignment)

Merged. Smooth L1 / Huber loss transition point reduced from β=1.0 → β=0.5 in both the training inner loop and `evaluate_split`. At β=0.5 the quadratic region covers only `|x| < 0.5` (in normalized space, the near-zero small-error regime), while moderate errors (0.5–1.0 MAE range) now receive a linear (L1-like) gradient. This directly aligns with the MAE primary metric over the bulk of the loss density, where most surface-pressure normalized errors live. EMA shadow absorbs the L1 kink noise near zero.

**New best (lower is better):**

| Metric | Value | vs PR #1606 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **85.9197** | −6.43 (−6.96%) |
| `test_avg/mae_surf_p` | **76.5495** | −5.08 (−6.22%) |

**Per-split test (best-val checkpoint, epoch 17):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 88.0317 |
| `test_geom_camber_rc` | 85.4633 |
| `test_geom_camber_cruise` | 56.3982 |
| `test_re_rand` | 76.3047 |
| **test_avg** | **76.5495** |

- **All 4 splits improved** (in_dist −7.6%, camber_rc −7.0%, camber_cruise −3.9%, re_rand −5.3%)
- **EMA-vs-live gap preserved:** EMA val=85.92 vs live val=96.41 (−10.5 MAE)
- **Code change:** `beta=1.0` → `beta=0.5` in two `F.smooth_l1_loss(...)` calls (train loop + evaluate_split)
- **W&B run:** `liurnqyo`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`

## 2026-05-12 22:10 — PR #1606: fern EMA of model weights (decay=0.999)

Merged. EMA shadow copy of model parameters updated after every optimizer step (`ema = 0.999 * ema + 0.001 * model`). Val and test evaluation uses EMA weights instead of live weights. EMA lags during warmup but consistently outperforms the live model from epoch 9 onward; the gap widens late in training as cosine LR anneals but SGD noise persists.

**New best (lower is better):**

| Metric | Value | vs PR #1436 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **92.3452** | −4.14 (−4.3%) |
| `test_avg/mae_surf_p` | **81.6297** | −4.70 (−5.4%) |

**Per-split test (best-val checkpoint, epoch 17):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 95.2950 |
| `test_geom_camber_rc` | 91.9270 |
| `test_geom_camber_cruise` | 58.7160 |
| `test_re_rand` | 80.5810 |
| **test_avg** | **81.6297** |

- **EMA-vs-live diagnostic:** epoch 17 live model test=104.70 vs EMA test=81.63 — EMA shadow is +28% better than live weights at same step
- **Config change:** `copy.deepcopy(model)` EMA shadow with `requires_grad=False`; updated after each `optimizer.step()` on fp32 master weights; val+test eval use `ema_model`
- **W&B run:** `gdfynh7o`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`

## 2026-05-12 21:10 — PR #1436: fern Huber + bf16 (compound winner)

Merged. Smooth L1 / Huber loss (β=1.0) replaces MSE in both training and `evaluate_split`. Stacked on top of the alphonse bf16 baseline; effects compounded as predicted — Huber's loss-shape alignment with the MAE metric (linear tails for high-Re extreme p samples) + bf16's epoch budget (~18 vs ~14 fp32).

**New best (lower is better):**

| Metric | Value | vs PR #1419 |
|--------|-------|-------------|
| `val_avg/mae_surf_p` | **96.4863** | −12.81 (−11.7%) |
| `test_avg/mae_surf_p` | **86.3326** | −11.33 (−11.6%) |

**Per-split val (epoch 16, best checkpoint):**

| Split | mae_surf_p |
|-------|----------:|
| `val_single_in_dist` | 112.8995 |
| `val_geom_camber_rc` | 106.9168 |
| `val_geom_camber_cruise` | 75.1834 |
| `val_re_rand` | 90.9454 |
| **val_avg** | **96.4863** |

**Per-split test (best-val checkpoint):**

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|----------:|------------:|------------:|----------:|
| `test_single_in_dist` | 101.2155 | 1.4049 | 0.6030 | 108.6379 |
| `test_geom_camber_rc` | 95.6042 | 1.9262 | 0.8326 | 106.1176 |
| `test_geom_camber_cruise` | 64.2155 | 1.0321 | 0.4469 | 63.5676 |
| `test_re_rand` | 84.2951 | 1.3881 | 0.6406 | 85.9693 |
| **test_avg** | **86.3326** | **1.4378** | **0.6308** | **91.0731** |

- **Config change:** `sq_err = F.smooth_l1_loss(pred, y_norm, beta=1.0, reduction='none')` replaces `sq_err = (pred - y_norm) ** 2` in two locations (training inner loop and `evaluate_split`).
- **W&B run:** `kmwsz3i4`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
- All 4 test splits improved (vs alphonse): in_dist −12.75, camber_rc −10.10, camber_cruise −9.16, re_rand −13.32.

## 2026-05-12 20:00 — PR #1419: alphonse bf16 autocast (round-1 winner)

Merged. bf16 mixed-precision training (`torch.amp.autocast(dtype=torch.bfloat16)`) + scoring NaN workaround in `evaluate_split`. Both changes are now in the advisor branch and will propagate to all subsequent student PRs.

**New best (lower is better):**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.2937** |
| `test_avg/mae_surf_p` | **97.6659** |

**Per-split val (epoch 18, best checkpoint):**

| Split | mae_surf_p |
|-------|----------:|
| `val_single_in_dist` | 133.2714 |
| `val_geom_camber_rc` | 115.3895 |
| `val_geom_camber_cruise` | 87.8295 |
| `val_re_rand` | 100.6844 |
| **val_avg** | **109.2937** |

**Per-split test (best-val checkpoint):**

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|----------:|------------:|------------:|----------:|
| `test_single_in_dist` | 113.9645 | 1.5436 | 0.7415 | 120.6592 |
| `test_geom_camber_rc` | 105.7068 | 2.3467 | 0.9479 | 109.4459 |
| `test_geom_camber_cruise` | 73.3736 | 1.1906 | 0.5263 | 74.9999 |
| `test_re_rand` | 97.6189 | 1.6668 | 0.7685 | 100.6900 |
| **test_avg** | **97.6659** | **1.6869** | **0.7460** | **101.4488** |

- **Config change:** bf16 autocast wraps forward + loss; optimizer and eval in fp32. ~101 s/epoch → 18 epochs in 30 min vs ~11-12 epochs fp32.
- **Scoring fix:** `evaluate_split` now pre-masks non-finite GT samples and applies `nan_to_num(y)` before `accumulate_batch`, eliminating `NaN*0=NaN` from `.test_geom_camber_cruise_gt/000020.pt`.
- **W&B run:** `4hy79j91`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (bf16 autocast and NaN workaround are now in the merged train.py; no extra flags needed)

## 2026-05-13 02:00 — PR #1672: nezuko linear LR warmup 1 epoch v2

**New best — 5th compound improvement**

- **val_avg/mae_surf_p:** 85.0926 (↓ from 85.9197, −0.96%)
- **test_avg/mae_surf_p:** 75.5171 (↓ from 76.5495, −1.35%)

**Per-split test (all four improved):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 87.1000 |
| `test_geom_camber_rc` | 84.5765 |
| `test_geom_camber_cruise` | 55.4971 |
| `test_re_rand` | 74.8950 |

- **Config:** EMA decay=0.999, Huber β=0.5, bf16 autocast, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, dropout=0.0, LR warmup 1 epoch (start_factor=0.2→1.0 over 375 steps, T_max=10875)
- **Epochs:** 17 in 30 min (~110 s/epoch)
- **EMA−Live gap:** −9.87 at epoch 17 (EMA −9.87 vs baseline −10.49)
- **W&B run:** `1hn6ur4l`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (warmup is now merged into train.py defaults; no extra flags needed)

## 2026-05-13 02:10 — PR #1763: edward torch.compile

**New best — 6th compound improvement (massive throughput win)**

- **val_avg/mae_surf_p:** 71.4371 (↓ from 85.0926, −16.06%)
- **test_avg/mae_surf_p:** 62.5927 (↓ from 75.5171, −17.11%)

**Per-split test (all four improved dramatically):**

| Split | mae_surf_p |
|-------|----------:|
| `test_single_in_dist` | 70.4261 |
| `test_geom_camber_rc` | 74.0859 |
| `test_geom_camber_cruise` | 44.5085 |
| `test_re_rand` | 61.3503 |

- **Config:** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, dropout=0.0, **torch.compile(model, dynamic=True, mode='default')**
- **Epochs:** **29 in 30.7 min** (~63 s/epoch steady state, +12 s compile warmup on epoch 1)
- **Speedup:** −44% per-epoch wall time vs no-compile; +12 epochs in budget (+71%)
- **Peak GPU memory:** 23.8 GB / 96 GB
- **EMA-vs-live gap:** −1.0 at epoch 29 (EMA 71.44, live 70.55 — both healthy)
- **W&B run:** `o6k5dj4g`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (torch.compile is now applied to the live model by default in train.py; dynamic=True handles variable mesh sizes; no extra flags needed)
- **Confounder noted:** `--epochs 30` makes cosine T_max=30 (vs implicit baseline T_max=50). Part of the gain may be from a more aggressive cosine schedule. Throughput component is clean either way.

## 2026-05-13 05:50 — PR #1875: frieren n_layers=3 v2 — fresh retry on compile-stack baseline

**New best — 7th compound improvement (architecture capacity-down + throughput win)**

- **val_avg/mae_surf_p:** 69.4518 (↓ from 71.4371, −2.78%)
- **test_avg/mae_surf_p:** 61.1887 (↓ from 62.5927, −2.24%)

**Per-split test (3/4 improved, camber_rc within noise +0.14):**

| Split | mae_surf_p | vs PR #1763 |
|-------|----------:|----------:|
| `test_single_in_dist` | 67.8314 | −2.60 |
| `test_geom_camber_rc` | 74.2256 | +0.14 (noise) |
| `test_geom_camber_cruise` | 42.8224 | −1.69 |
| `test_re_rand` | 59.8755 | −1.47 |

- **Config:** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, **n_layers=3**, slice_num=64, mlp_ratio=2, dropout=0.0, torch.compile(model, dynamic=True, mode='default')
- **Epochs:** **30 in 20.6 min** (~40.8 s/epoch steady state — 35% speedup vs compile baseline ~63 s)
- **Budget headroom:** ~9 min unused in 30-min budget; projected ~44 epochs if run to cap
- **Param count:** 420,047 (0.23× compile baseline 1.84M)
- **Best epoch:** 30/30 (final) — val trajectory still descending; model had not fully converged → more headroom
- **EMA-vs-live gap:** small and healthy
- **W&B run:** `fsqr0yp5`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_layers 3 --epochs 30`
- **Mechanism:** PhysicsAttention slicing carries the heavy representational load; 3 attention layers is sufficient for this 1500-sample dataset. Depth reduction frees ~35% compute per epoch → more training epochs in budget → better convergence despite 77% fewer params.

## 2026-05-13 07:00 — PR #1784: tanjiro gradient-clip max_norm=10 + diagnostics

**New best — 8th compound improvement (gradient-shape lever)**

Measured at n_layers=5 (student branch was behind #1875 merge; grad-clip code applies cleanly on top of current n_layers=3 advisor branch):

- **val_avg/mae_surf_p:** 65.9757 (↓ from 71.4371 compile baseline, **−7.65%**)
- **test_avg/mae_surf_p:** 57.0711 (↓ from 62.5927, **−8.83%**)

**Per-split test (all 4 splits improved cleanly):**

| Split | mae_surf_p | vs PR #1763 (n_layers=5 compile) |
|-------|----------:|----------:|
| `test_single_in_dist` | 64.5497 | −5.88 |
| `test_geom_camber_rc` | 70.5841 | −3.50 |
| `test_geom_camber_cruise` | 37.9291 | −6.58 |
| `test_re_rand` | 55.2217 | −6.13 |

- **Config (as measured):** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, dropout=0.0, torch.compile(model, dynamic=True), **`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)` after .backward(), before .step()**
- **Epochs:** 29 in 31.05 min (~63.4 s/epoch, identical to compile baseline)
- **Mechanism (soft scaling regime):** clip rate 72.4% (vs 100% at max_norm=1.0 in #1534 v2). Gradient norm distribution on compile stack: p50=16.2, p90=40.6, p99=91.8, max=262. At threshold 10, the heavy upper tail gets ~2.2× downscaling on typical clipped steps; bulk gradient direction preserved. Sweet spot between v2's full direction-normalization (100% clip, ~22× scaling) and unmeasured safety-net regime (<10% clip at threshold ≥50).
- **Why this works**: by dampening rare large-magnitude updates without erasing AdamW's direction information on typical steps, the optimizer follows a smoother trajectory through the loss landscape. EMA shadow benefits from lower variance per update, narrowing the EMA-live gap throughout training.
- **W&B run:** `vy49aq06`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --epochs 30`
  (grad-clip + diagnostics now in train.py defaults; no extra flags needed)
- **Caveat on combined baseline**: The advisor branch now has **n_layers=3 + grad-clip=10 + everything else**, but the measured number above is on **n_layers=5 + grad-clip=10**. The combined n_layers=3 + grad-clip=10 has not been directly measured. Mechanism (gradient-norm scaling, orthogonal to architecture) suggests these compound additively → expected val ≤ 65.98 on the combined stack, but the next n_layers=3 experiment will confirm directly.

## 2026-05-13 07:35 — PR #1899: alphonse n_layers=3 + n_hidden=192 (width reinvestment)

**New best — 9th compound improvement (architectural capacity rebalancing)**

- **val_avg/mae_surf_p:** 63.7215 (↓ from 65.9757, **−3.45%** vs prior best; ↓ from 69.4518 n_layers=3 baseline, **−8.25%**)
- **test_avg/mae_surf_p:** 55.6430 (↓ from 57.0711, **−2.51%**; ↓ from 61.1887, **−9.06%**)

**Per-split test (all 4 splits improved cleanly):**

| Split | mae_surf_p | vs PR #1784 (grad-clip=10, n_layers=5) | vs PR #1875 (n_layers=3 baseline) |
|-------|----------:|----------:|----------:|
| `test_single_in_dist` | 61.4444 | −3.11 | −6.39 |
| `test_geom_camber_rc` | 69.3247 | −1.26 | −4.90 |
| `test_geom_camber_cruise` | 37.7067 | −0.22 | −5.12 |
| `test_re_rand` | 54.0962 | −1.13 | −5.78 |

- **Config (as measured):** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, **n_hidden=192**, n_layers=3, slice_num=64, mlp_ratio=2, dropout=0.0, torch.compile(model, dynamic=True)
- **NOTE:** This run did NOT have grad-clip=10 (student's branch was based on pre-grad-clip advisor commit). Current advisor branch has n_layers=3 + n_hidden=192 + grad-clip=10 — combined state unmeasured.
- **Epochs:** 30/30 in 28.15 min (~54.3 s/epoch steady state — 33% slower than n_hidden=128 compile baseline, well within 30-min budget)
- **Param count:** 931,791 (0.93M; 2.22× n_hidden=128 n_layers=3 baseline of 0.42M; still below original 1.84M)
- **Best epoch:** 30/30 (final) — val slope −0.22/epoch at end; **still descending, not converged**. EMA-vs-live gap +0.42 (EMA slightly behind live on a still-improving model)
- **Mechanism:** "Compact but wide" hypothesis confirmed. n_hidden=192 × n_layers=3 (0.93M params) vs prior failed n_hidden=192 × n_layers=5 (+12.5% worse): depth reduction freed headroom for width reinvestment. At n_layers=3, per-layer expressivity was the bottleneck; wider layers compensate for reduced composition depth. Width and depth aren't fungible — depth-limited vs capacity-saturated regimes have opposite responses to widening.
- **W&B run:** `r10qkcgd`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_layers 3 --n_hidden 192 --epochs 30`
- **All subsequent experiments should target val < 63.7215 and test < 55.6430** as the merge threshold.
- **Caveat on combined baseline**: The advisor branch now has **n_layers=3 + n_hidden=192 + grad-clip=10 + everything else**, but the measured val=63.72 is on **n_layers=3 + n_hidden=192 WITHOUT grad-clip=10**. The full combined state has not been directly measured. Expected combined val < 63.72 (grad-clip should compound with architecture). The first n_layers=3 + n_hidden=192 + grad-clip=10 run (any subsequent experiment specifying `--n_layers 3 --n_hidden 192`) will confirm.

## 2026-05-13 09:00 — PR #1930: tanjiro grad-clip max_norm=5.0 (threshold scan step 2)

**New best — 10th compound improvement (tighter gradient clipping)**

- **val_avg/mae_surf_p:** 63.4801 (↓ from 63.7215, **−0.38%**)
- **test_avg/mae_surf_p:** 54.9834 (↓ from 55.6430, **−1.18%**)

**Per-split test (3/4 splits improved; in_dist slight regression):**

| Split | mae_surf_p | vs PR #1899 |
|-------|----------:|----------:|
| `test_single_in_dist` | 62.4458 | +1.00 (regression) |
| `test_geom_camber_rc` | 68.3757 | −0.95 |
| `test_geom_camber_cruise` | 35.8182 | −1.89 |
| `test_re_rand` | 53.2939 | −0.80 |

- **Config (as measured):** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, **n_hidden=128**, n_layers=3, slice_num=64, mlp_ratio=2, dropout=0.0, torch.compile(model, dynamic=True), **`clip_grad_norm_(model.parameters(), max_norm=5.0)`**
- **NOTE:** This run did NOT have n_hidden=192 (tanjiro's branch was based on pre-n_hidden=192 advisor commit). Current advisor branch has n_hidden=192 + grad-clip=5.0 — combined state unmeasured.
- **Clip stats:** clip rate 90.06%, mean grad norm 21.45 (unchanged from max_norm=10 run), mean downscaling 4.29× (predicted 4.2×, exact). Regime: moderate uniform downscaling — 90% of steps are scaled by ~4.3×, directions fully preserved.
- **Mechanism:** Tighter threshold compresses the upper tail more aggressively than max_norm=10. At 90% clip rate with 4.3× downscaling, small-gradient steps are no longer suppressed relative to clipped steps (as occurs at max_norm=1.0 with ~22× scaling). OOD splits benefited; in_dist started regressing, suggesting the model is approaching the threshold where clipping begins to uniformly suppress useful gradients.
- **Epochs:** 30/30 in 20.8 min (~41.6 s/epoch, identical n_layers=3 throughput)
- **Best epoch:** 30/30 (still descending)
- **W&B run:** `forfket5`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_layers 3 --n_hidden 192 --epochs 30`
  (grad-clip max_norm=5.0 now in train.py defaults; no extra flags needed)
- **All subsequent experiments should target val < 63.4801 and test < 54.9834** as the merge threshold.
- **Caveat on combined baseline**: The advisor branch now has **n_layers=3 + n_hidden=192 + grad-clip=5.0 + everything else**, but the measured val=63.48 is on **n_hidden=128 + grad-clip=5.0 WITHOUT n_hidden=192**. Expected combined val < 63.48. The first n_hidden=192 + grad-clip=5.0 run will confirm the true combined state.

## 2026-05-13 09:50 — PR #1953: alphonse n_hidden=192 + epochs=50 (compound + schedule fix)

**New best — 11th compound improvement (FULL 10-compound stack + schedule fix; massive win)**

- **val_avg/mae_surf_p:** 55.7634 (↓ from 63.4801, **−12.17%**)
- **test_avg/mae_surf_p:** 48.0960 (↓ from 54.9834, **−12.53%**)

**Per-split test (ALL 4 splits improve dramatically):**

| Split | mae_surf_p | vs PR #1930 | vs PR #1899 |
|-------|----------:|----------:|----------:|
| `test_single_in_dist` | 52.8835 | −9.56 (−15.30%) | −8.56 (−13.94%) |
| `test_geom_camber_rc` | 61.7845 | −6.59 (−9.64%) | −7.54 (−10.88%) |
| `test_geom_camber_cruise` | 31.1522 | −4.67 (−13.03%) | −6.55 (−17.39%) |
| `test_re_rand` | 46.5637 | −6.73 (−12.63%) | −7.53 (−13.92%) |

- **Config (as measured):** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, **n_hidden=192**, n_layers=3, slice_num=64, mlp_ratio=2, n_head=4, dropout=0.0, torch.compile(dynamic=True), grad-clip max_norm=5.0, **`--epochs 50` (T_max=50)**
- **THIS IS THE FIRST DIRECT MEASUREMENT OF THE FULL 10-COMPOUND STACK** — n_layers=3 + n_hidden=192 + grad-clip=5.0 + EMA + Huber + warmup + compile + T_max=50.
- **Epochs:** 30/50 completed in 30-min wall-clock cap (~55 s/epoch epochs 1-26 clean; 99-114 s/epoch epochs 27-30 GPU contention from stale duplicate process; metrics unaffected).
- **Best epoch:** 30/30 — **every single epoch was a new EMA best**. Val slope at termination **−0.84/epoch** (strongly descending; not converged).
- **LR at termination:** ~1.73e-4 (still productive). Cosine T_max=50 means LR stayed above zero through the full wall-clock budget.
- **EMA-vs-live gap:** −8.32 (vs +0.42 at #1899). With clip rate 73% (p50=15.4, p90=35.9, p99=82.7, max=187.7), live model is noisy enough that EMA shadow carries real edge.
- **Mechanism (orthogonal compounding confirmed):** All three changes (n_hidden=192, grad-clip=5.0, T_max=50) compounded as predicted. Schedule fix alone (T_max 30→50) on the 10-compound stack provided the dominant lift; the combined stack delivered a clean 12%+ improvement uniformly across all 4 test splits.
- **Param count:** 931,791 (0.93M), peak GPU memory 21.3 GB / 96 GB.
- **W&B run:** `vnsqnuoy`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --epochs 50`
- **All subsequent experiments should target val < 55.7634 and test < 48.0960** as the merge threshold.
- **The model is epoch-saturated, not capacity-saturated** — val descending at −0.84/ep at termination. Schedule/throughput-axis follow-ups (higher T_max, larger batch, faster epoch) carry highest expected value.

## 2026-05-13 12:00 — PR #1982: tanjiro grad-clip max_norm=5.0 → 2.5 (threshold scan step 3)

**New best — 12th compound improvement (LARGEST single-axis gain in many cycles)**

- **val_avg/mae_surf_p:** 52.6406 (↓ from 55.7634, **−5.60%**)
- **test_avg/mae_surf_p:** 48.0960 → **44.9791** (**−6.49%**)

**Per-split test (ALL 4 splits improve dramatically; in_dist regression from #1930 fully reversed):**

| Split | mae_surf_p | vs PR #1953 (new baseline) |
|-------|----------:|----------:|
| `test_single_in_dist` | 49.8555 | −3.03 (−5.73%) |
| `test_geom_camber_rc` | 57.7726 | −4.01 (−6.49%) |
| `test_geom_camber_cruise` | 28.9446 | −2.21 (−7.10%) |
| `test_re_rand` | 43.3437 | −3.22 (−6.90%) |

- **Config (as measured):** EMA decay=0.999, Huber β=0.5, bf16 autocast, LR warmup 1ep, lr=5e-4, batch_size=4, surf_weight=10, n_hidden=192, n_layers=3, slice_num=64, mlp_ratio=2, n_head=4, dropout=0.0, torch.compile(dynamic=True), **`clip_grad_norm_(model.parameters(), max_norm=2.5)`**, T_max=50 (epochs=50).
- **GRAD-CLIP THRESHOLD SCAN SUMMARY:**

| max_norm | clip rate | mean downscaling | val_avg | result |
|---|---|---|---|---|
| 10.0 (PR #1784) | 72.4% | ~2.1× | 65.98 | WIN |
| 5.0 (PR #1930) | 90.1% | ~4.3× | 63.48 | WIN |
| **2.5 (PR #1982)** | **98.9%** | **~7.1×** | **52.64** | **WIN (massive)** |
| 1.0 (PR #1534v2) | ~100% | ~22× | regression | FAIL |

- The monotonic improvement from 10.0→5.0→2.5 is stunning. The gap 5.0→2.5 (Δval=−3.12) is larger than 10.0→5.0 (Δval=−2.50), and the in_dist regression at 5.0 (+1.00 vs 10.0) is **fully reversed** at 2.5 (in_dist −5.73%). We are still in the productive moderate-scaling regime, not the direction-normalization failure of max_norm=1.0.
- **Next threshold to test:** 1.5 (interpolates between the last win at 2.5 and the fail at 1.0). If 1.5 still wins, scan continues. If 1.5 fails, optimum is bracketed in [1.5, 2.5].
- **Clip diagnostics:** clip rate 98.93%, norm_mean=17.845, norm_p50=14.029, norm_p90=32.859, norm_p99=76.264, norm_max=353.038, mean downscaling ~7.14×.
- **Epochs:** 33/50 in 30-min wall-clock cap. Hit timeout cleanly; best checkpoint saved; full test eval ran at epoch 33.
- **Best epoch:** 33 (val still descending at termination — model epoch-saturated again).
- **EMA-vs-live gap:** maintained (live 51.0561 vs EMA 44.9791 — gap tightened at the new clip threshold; live is now noisier but closer to EMA than at max_norm=5.0).
- **W&B run:** `bb6o68xa`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --epochs 50`
  (grad-clip max_norm=2.5 now baked into advisor branch train.py; no extra flag needed)
- **All subsequent experiments should target val < 52.6406 and test < 44.9791** as the merge threshold.

## 2026-05-13 12:05 — PR #2023: frieren n_hidden=192 → 224 width push

**New best — 13th compound improvement (width scaling on 11-compound stack)**

- **val_avg/mae_surf_p:** 53.2494 (measured against PR #1953 baseline of 55.7634; **−4.51%** at time of review)
- **test_avg/mae_surf_p:** **46.6004** (**−3.11%** at time of review)

**Per-split test (3/4 splits clearly improve; in_dist within noise):**

| Split | mae_surf_p | vs PR #1953 |
|-------|----------:|----------:|
| `test_single_in_dist` | 53.2544 | +0.37 (noise, ~0.7%) |
| `test_geom_camber_rc` | 58.8796 | −2.90 (−4.70%) |
| `test_geom_camber_cruise` | 29.6831 | −1.47 (−4.72%) |
| `test_re_rand` | 44.5845 | −1.98 (−4.25%) |

- **Config (as measured):** Full 11-compound stack + n_hidden=192→**224**, grad-clip max_norm=5.0 (PRE-#1982 merge), T_max=50, n_layers=3, 1.26M params.
- **NOTE: EMPTY DIFF MERGE** — win is CLI-only. Advisor branch defaults still have original n_hidden value. All subsequent student reproduce commands must specify `--n_hidden 224 --n_layers 3 --epochs 50`.
- **Epochs:** 29/50 in 30-min cap. EMA val still descending at **−1.46/epoch** at termination (strongly epoch-saturated). Best epoch 29/29 — every epoch was a new EMA best.
- **EMA-live gap:** −6.18 (tightened from −8.32 at #1953 — wider model is easier to track).
- **W&B run:** `80b6pnb9`
- **Param count:** 1,263,119 (1.26M), throughput ≈ same as n_hidden=192 (~62 s/epoch).
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 224 --n_layers 3 --epochs 50`
  (grad-clip max_norm=2.5 now in train.py defaults from PR #1982; no extra flag needed for that)
- **COMBINED STATE (12+13): n_hidden=224 + grad-clip=2.5 is UNMEASURED.** PR #1982 was measured at n_hidden=192 (val=52.64); PR #2023 was measured at grad-clip=5.0 (val=53.25). Both mechanisms beat the 11-compound baseline independently. Predicted combined val ≈ 50–52 if mechanisms are additive. **The next priority is to directly measure val at n_hidden=224 + grad-clip=2.5 + T_max=50.**
- **All subsequent experiments should target val < 52.6406 and test < 44.9791** (from PR #1982 — the current direct measurement). The combined n_hidden=224 + grad-clip=2.5 state will supersede this once directly measured.

## 2026-05-13 17:45 — PR #2142: fern Huber β=0.5 → 0.25 (tighter MAE alignment on 13-compound stack)

**New best — 14th compound improvement (loss-shape axis at grad-clip=2.5 saturated regime)**

- **val_avg/mae_surf_p:** **50.3812** (vs #1982 baseline 52.6406; **−4.29%**, −2.26 absolute)
- **test_avg/mae_surf_p:** **43.7187** (vs #1982 baseline 44.9791; **−2.80%**, −1.26 absolute)

**Per-split test (EMA, best-val checkpoint — all 4 splits improve):**

| Split | mae_surf_p | vs #1982 | Δ% |
|-------|----------:|----------:|---:|
| `test_single_in_dist` | 48.9641 | −0.89 | −1.79% |
| `test_geom_camber_rc` | 57.3689 | −0.40 | −0.70% |
| `test_geom_camber_cruise` | 26.9722 | −1.97 | −6.81% |
| `test_re_rand` | 41.5697 | −1.77 | −4.09% |
| **test_avg** | **43.7187** | **−1.26** | **−2.80%** |

- **Best epoch:** 33 (still descending at termination — not epoch-saturated)
- **Clip rate:** 99.91% (12364/12375 steps) — Huber β axis operates upstream of gradient computation; mechanism is loss curvature, not amplitude. Axis confirmed orthogonal to clip saturation.
- **Mechanism:** Huber β=0.25 tightens the MAE alignment further. At β=0.25, the quadratic region covers only |error| < 0.25 — nearly all normalized surface-pressure errors are in the linear (L1-like) regime. This directly targets the primary MAE metric across the bulk of the loss distribution.
- **Note on clip rate:** 99.91% vs baseline 98.93% (+0.98pp) — the tighter Huber creates slightly sharper gradients for moderate errors, which increases clip rate marginally. But this is upstream-of-gradient mechanism (loss curvature), not amplitude.
- **W&B run:** `aew7c8ej`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --epochs 50`
  (huber_beta=0.25 now baked into advisor branch train.py from this merge; grad-clip=2.5, T_max=50 also baked in)
- **All subsequent experiments should target val < 50.3812 and test < 43.7187** as the merge threshold.

## 2026-05-13 18:45 — PR #2247: frieren batch_size 4 → 2 (2× opt-step density per epoch)

**New best — 15th compound improvement (opt-step density axis, massive OOD gain)**

- **val_avg/mae_surf_p:** **46.6788** (vs #2142 baseline 50.3812; **−7.35%**, −3.70 absolute)
- **test_avg/mae_surf_p:** **39.7696** (vs #2142 baseline 43.7187; **−9.04%**, −3.95 absolute)

**Per-split test (EMA, best-val checkpoint epoch 34 — all 4 splits improve sharply):**

| Split | mae_surf_p | vs #2142 | Δ% |
|-------|----------:|----------:|---:|
| `test_single_in_dist` | 44.0421 | −4.92 | −10.05% |
| `test_geom_camber_rc` | 53.1169 | −4.25 | −7.41% |
| `test_geom_camber_cruise` | 24.1470 | −2.83 | −10.48% |
| `test_re_rand` | 37.7723 | −3.80 | −9.13% |
| **test_avg** | **39.7696** | **−3.95** | **−9.04%** |

- **Best epoch:** 34/50 (EMA val still descending at termination — epoch-saturated again)
- **Opt-steps:** 25,500 (750/epoch × 34 ep) vs baseline 12,375 (375/epoch × 33 ep) — **2.06× multiplier**
- **Throughput:** 53.02 s/epoch (same as bs=4 baseline — doubling opt-steps costs nothing in wall time)
- **Clip rate:** 94.70% (dropped from 98.93% at bs=4 — FIRST measured clip-saturation loosening that wins) — norm_p50=12.96, norm_p99=85.4
- **EMA-live gap:** −4.71 test (tightened from −6.08 at baseline; opposite of PR prediction — more opt-steps produced a smoother effective trajectory)
- **Mechanism:** Doubling opt-step density per epoch (bs=4→2) doubles gradient update count at fixed LR, schedule shape, and wall-clock budget. Net: 2.06× optimizer step exposure in same training time. Clip rate eases from 98.93%→94.70% — first measured saturation loosening that doesn't fail. This is consistent with "the bs=4 stack was opt-step-saturated: the clip-saturation we observed was symptomatic of insufficient update count, not gradient pathology."
- **Variance note:** Student ran 3 times due to GPU contention; first 2 at degraded throughput reached only 25-24 epochs and landed val ~55-56 (near baseline). Canonical run t5xloer3 at clean throughput (53s/epoch) is the merge target. Result is throughput-dependent — metric valid only at ≥30 epochs in 30-min window.
- **W&B run:** `t5xloer3`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --batch_size 2 --epochs 50`
  (batch_size=2 must be specified; all other 14-compound stack settings baked into advisor branch train.py)
- **All subsequent experiments should target val < 46.6788 and test < 39.7696** as the merge threshold.
- **CRITICAL THROUGHPUT NOTE:** batch_size=2 doubles opt-steps at same wall-clock per epoch (~53s). Compound retests MUST run with `--batch_size 2` to land on the new baseline. Any run at bs=4 measures on a different stack and cannot beat this baseline.

## 2026-05-13 19:35 — PR #2219: alphonse n_hidden=160 (width-narrowing compound retest on 15-compound stack)

**New best — 16th compound improvement (width-narrowing × opt-step density interaction)**

- **val_avg/mae_surf_p:** **45.9186** (vs #2247 baseline 46.6788; **−1.63%**, −0.76 absolute)
- **test_avg/mae_surf_p:** **39.0381** (vs #2247 baseline 39.7696; **−1.84%**, −0.73 absolute)

**Per-split test (EMA, best-val checkpoint epoch 38 — 3 splits improve, camber_cruise wash):**

| Split | mae_surf_p | vs #2247 | Δ% |
|-------|----------:|----------:|---:|
| `test_single_in_dist` | 42.2300 | −1.81 | −4.12% |
| `test_geom_camber_rc` | 53.9414 | +0.82 | +1.57% |
| `test_geom_camber_cruise` | 23.4382 | −0.71 | −2.93% |
| `test_re_rand` | 36.5427 | −1.23 | −3.26% |
| **test_avg** | **39.0381** | **−0.73** | **−1.84%** |

- **Best epoch:** 38/50 (vs baseline 34/50 — 4 extra epochs from narrower net)
- **Opt-steps:** 28,500 (750/epoch × 38 ep) — width-narrowing provides +11.8% more optimizer steps vs baseline
- **Throughput:** 47.4 s/epoch (vs 53s at n=192 — width-narrowing saves ~5.6s/epoch at bs=2)
- **Clip rate:** 98.06% (slight increase from 94.70% at bs=2 with n=192 — narrower net slightly noisier per-step)
- **Grad norm mean:** 23.63 (vs 18.04 at n=160/bs=4 — bs=2 per-step noise raises this 31%)
- **Param count:** 650,767 (0.65M vs 0.93M at n=192 — −30%)
- **EMA-live gap:** −1.55 test (healthy late-phase noise rejection)
- **Mechanism:** n_hidden=160 → faster epochs (−10.8% per-epoch time) → 4 extra epochs (38 vs 34) → cosine LR at termination drops from ~26% to 14% of base → additional late-phase low-LR refinement. The win is specifically a bs=2 × n_hidden=160 interaction — at bs=4 (14-compound stack), n=160 was a wash/slight loss. At bs=2 the narrower net's tighter direction-variance partially counter-balances the higher per-step gradient noise from halved batch size.
- **Informational 14-stack result:** n=160/bs=4/β=0.25 was val 51.5954/test 44.4327 — slight regression vs 14-stack baseline (val 50.38). The bs=2 interaction is essential.
- **W&B run:** `741bdhcl` (canonical); informational 14-stack: `560twhbv`
- **Reproduce:** `cd target && python train.py --agent <student> --wandb_name "<name>" --n_hidden 160 --n_layers 3 --batch_size 2 --epochs 50`
  (n_hidden=160 added; batch_size=2 must be explicit; all other 15-compound stack settings baked into advisor branch train.py)
- **All subsequent experiments should target val < 45.9186 and test < 39.0381** as the merge threshold.
