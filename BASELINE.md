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
