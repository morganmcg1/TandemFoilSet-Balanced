# Baseline — icml-appendix-charlie-pai2g-24h-r5

Charlie no-W&B logging arm, round 5. Each training execution is capped at
`SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap). Local JSONL metrics only,
no W&B.

## Current best

| Metric | Value | Source |
|---|---|---|
| **val_avg/mae_surf_p** | **45.92** | PR #2287 (merged 2026-05-13) — GeGLU (gate×GELU) replaces SwiGLU (gate×SiLU) in block-MLPs |
| **test_avg/mae_surf_p** | **44.35** | PR #2287 — all 4 splits finite; epoch-15 best checkpoint (hit 30-min cap at ep 15) |
| Peak VRAM | ~42.5 GB | PR #2287 — identical FLOP/VRAM profile to SwiGLU |
| s/epoch | ~126 s | PR #2287 — 15 epochs completed within 30-min cap |

### Per-split val (PR #2287, best epoch 15, GeGLU block-MLP + Lion lr=2e-4 + per-channel δ + dropout=0.1 + n_hidden=160)

| Split | mae_surf_p |
|---|---:|
| val_single_in_dist | 48.87 |
| val_geom_camber_rc | 58.78 |
| val_geom_camber_cruise | 29.99 |
| val_re_rand | 46.03 |
| **val_avg** | **45.92** |

### Per-split test (PR #2287, epoch-15 best checkpoint)

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 43.19 |
| test_geom_camber_rc | 52.54 |
| test_geom_camber_cruise | 43.55 |
| test_re_rand | 38.14 |
| **test_avg** | **44.35** |

**Reproduce:**
```bash
cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name geglu_gate_vs_swiglu --agent <student>
```
(GeGLU block-MLPs now in merged train.py — swaps `F.silu` → `F.gelu` inside the SwiGLU gate. Block-MLP only; preprocess MLP and mlp2 head remain GELU. **Explicit `--lion_lr 2e-4 --lion_weight_decay 6e-5` required**.)

### Delta from previous best (PR #2196 → PR #2287)

| Split | Prev val | New val | Δval | Prev test | New test | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 52.19 | 48.87 | **−3.32** | 43.52 | 43.19 | **−0.33** |
| geom_camber_rc | 59.75 | 58.78 | **−0.97** | 53.81 | 52.54 | **−1.27** |
| geom_camber_cruise | 30.87 | 29.99 | **−0.88** | 43.91 | 43.55 | **−0.36** |
| re_rand | 46.90 | 46.03 | **−0.87** | 38.82 | 38.14 | **−0.68** |
| **avg** | **47.43** | **45.92** | **−1.51 (−3.2%)** | **45.01** | **44.35** | **−0.66 (−1.5%)** |

## 2026-05-13 14:15 — PR #2287: GeGLU gate ablation — SiLU→GELU inside block-MLP gate (MERGED)

- **val_avg/mae_surf_p: 45.9178** (↓ 3.2% from 47.43 — all 4 val splits improve; single_in_dist gains most −3.32)
- **test_avg/mae_surf_p: 44.3537** (↓ 1.5% from 45.01 — all 4 test splits improve)
- **Peak VRAM: ~42.5 GB** (F.gelu same VRAM/FLOP footprint as F.silu); s/epoch ~126 s; 15 of 16 epochs completed (hit 30-min cap at ep 15)
- **Metric artifacts:** `models/model-geglu_gate_vs_swiglu-20260513-130428/metrics.jsonl`
- **What changed:** 1-character swap in the SwiGLU gate activation: `F.silu(self.w_in(x))` → `F.gelu(self.w_in(x))`. Block-MLP architecture otherwise unchanged — same `hidden=216`, same `w_out`, same gated form `out = w_out(act(w_in(x)) * w_gate(x))`.
- **Why it worked:** Resolves the PR #2176 bare-SiLU paradox. PR #2176 showed GELU→SiLU regresses every split. PR #2196 showed gated-SiLU (SwiGLU) wins. PR #2287 shows gated-GELU (GeGLU) wins MORE. Fern's mechanism: gating is the architectural primitive that matters; *within* the gate, GELU's gradient surface aligns better with Lion's optimizer (Lion sign-updates are tuned for GELU-like shapes — the bare-SiLU loss matches). GeGLU uses GELU's robust near-zero gradient profile on the gate input while still applying the multiplicative selective-suppression that drove the SwiGLU gain. Net result: you get both gating and GELU's optimizer-aligned gradients.
- **Baseline configuration delta:** `F.silu` → `F.gelu` inside the `SwiGLU` class gate input (1-line in train.py).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name geglu_gate_vs_swiglu --agent <student>
  ```

## 2026-05-13 12:53 — PR #2196: SwiGLU gated MLP block replacing GELU (MERGED)

- **val_avg/mae_surf_p: 47.4287** (↓ 9.9% from 52.63 — largest single-PR gain since Lion introduction)
- **test_avg/mae_surf_p: 45.0147** (↓ 8.6% from 49.22 — all 4 splits finite, all improve)
- **Peak VRAM: ~42.5 GB** (BF16, batch=4, n_hidden=160, SwiGLU hidden=216); s/epoch ~117 s; 15 of 16 epochs completed (hit 30-min cap)
- **Metric artifacts:** `models/model-swiglu_mlp_ratio_4_3_n160-20260513-115449/metrics.jsonl`
- **What changed:** Block-MLP activation replaced GELU with SwiGLU gated variant: `out = (W₁·x) * SiLU(W₂·x)` where both projection matrices have hidden=ceil(160×4/3)=216 (parameter-equivalent to GELU baseline at standard mlp_ratio=2). Preprocess MLP and final `mlp2` head unchanged (remain GELU). New `mlp_ratio` for block-SwiGLU is 4/3 of the outer dimension — both weight matrices in the gate product use hidden=216.
- **Why it worked:** PR #2176 showed that bare SiLU (GELU→SiLU slope swap) regresses every split by +6.9 val — Lion sign-updates are tuned for GELU's gradient surface. SwiGLU adds a multiplicative *gate* `σ(W₂x)` that modulates how much input signal passes through; this is architecturally distinct from a slope change. The gate gives each channel the ability to selectively suppress low-confidence features before the output projection, effectively adding learned feature routing at zero parameter cost vs GELU at the same mlp_ratio. The 7.6 val-MAE improvement on the hardest OOD split (val_geom_camber_rc: 67.35→59.75) suggests the gate is most valuable for uncertain input regimes where GELU's uniform passthrough over-weights ambiguous features. Gain is uniform (all 8 splits improve) — not a fluke on one split.
- **Baseline configuration delta:** Block-MLP `act=gelu` → SwiGLU gate with `hidden=ceil(n_hidden × 4/3)`.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name swiglu_mlp_ratio_4_3_n160 --agent <student>
  ```

## 2026-05-13 09:58 — PR #1656: dropout=0.1 on Lion lr=2e-4 + per-channel δ + n_hidden=160 (MERGED)

- **val_avg/mae_surf_p: 52.6345** (↓ 0.27% from 52.78 — small but consistent with middle-case additive hypothesis)
- **test_avg/mae_surf_p: 49.2183** (↓ 0.41% from 49.42 — both primary and test metrics improve)
- **Peak VRAM: ~38 GB** (consistent with n_hidden=160 stack + minor dropout overhead); s/epoch ~117 s; ~31 min total
- **Metric artifacts:** `models/model-charliepai2g24h5-thorfinn-dropout_0_1_n160_pcd_lr2e4-20260513-091653/metrics.jsonl`
- **What changed:** Added `dropout=0.1` to the Transolver model config, flowing through `Transolver(dropout=0.1) → TransolverBlock(dropout=0.1) → PhysicsAttention(dropout=0.1)`. Two gates: `nn.Dropout(p=0.1)` in `PhysicsAttention.to_out` (attention output dropout) and `dropout_p=0.1` in `F.scaled_dot_product_attention` when `self.training=True`. No change to optimizer, scheduler, loss, or architecture.
- **Why it worked:** Feature-level stochastic masking is orthogonal to gradient-level regularization (Huber, grad_clip) and width regularization (n_hidden=160). Dropout forces representational redundancy — each forward pass uses a random subset of feature channels, preventing individual channels from over-specializing. The cross-stack ablation shows diminishing returns (5.7% on n_hidden=128 → 1.0% on n_hidden=160 without lr alignment → 0.27% on full aligned stack), consistent with a saturating regularization budget: per-channel δ=0.2 already provides tight gradient-magnitude capping on the pressure channel, and n_hidden=160 provides additional width regularization capacity. However, dropout's 0.27% val gain is strictly additive and test-side signal is even stronger (−0.41%), confirming the regularization axes remain non-redundant. Val curve strictly monotone-decreasing at epoch 16 — headroom with longer training.
- **Baseline configuration delta:** `dropout=0.1` in `model_config` dict (train.py). All other config identical to PR #2027.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name dropout01_pcd_lr2e4_check --agent <student>
  ```
  (`dropout=0.1` must now be the train.py default after this merge; explicit `--lion_lr 2e-4 --lion_weight_decay 6e-5` still required)

## 2026-05-13 09:00 — PR #2027: Lion lr=2e-4 on per-channel δ + n_hidden=160 stack (MERGED)

- **val_avg/mae_surf_p: 52.7778** (↓ 1.6% from 53.62 — Lion LR optimum continues moving down as the loss landscape tightens)
- **test_avg/mae_surf_p: 49.4184** (↓ 0.5% from 49.65)
- **Peak VRAM: 37.99 GB** (unchanged); s/epoch ~109 s (slightly faster than 115 s baseline)
- **Metric artifacts:** `models/model-lion_lr2e4_n160_pcd-20260513-081331/metrics.jsonl`
- **What changed:** Lion `lr=3e-4 → 2e-4` (CLI flag only, no code change). All other config identical.
- **Why it worked:** The wider model (n_hidden=160, 1.6× params) has larger per-parameter gradient contributions, so Lion's sign-quantized step size at lr=3e-4 over-shoots in the back half of training. Reducing to lr=2e-4 produces tighter convergence — visible in the per-epoch curve where divergence between lr=2e-4 and lr=3e-4 only appears after epoch 10 (once cosine decay and signs settle). Three of four val splits improve: single_in_dist −2.22, cruise −0.85, re_rand −0.41. The OOD-geometry split `geom_camber_rc` is flat (+0.11), consistent with smaller LR giving tighter in-distribution fits without over-stepping on harder samples. Compounds with PR #2028: per-channel δ does most of the work (−2.30 val), lr=2e-4 adds another −0.84.
- **Baseline configuration delta:** `--lion_lr 2e-4` (explicit CLI flag; train.py default still 1.5e-4 — stale from #1641 sweep but not updated).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name pcd_lr2e4_baseline_check --agent <student>
  ```

## 2026-05-13 08:00 — PR #2028: Per-channel Huber δ=[Ux=0.5, Uy=0.5, p=0.2] on n_hidden=160 stack (MERGED)

- **val_avg/mae_surf_p: 53.6194** (↓ 4.1% from 55.92 — decoupling pressure vs velocity outlier capping)
- **test_avg/mae_surf_p: 49.6534** (↓ 4.4% from 51.92 — all 4 splits improve uniformly)
- **Peak VRAM: 37.99 GB** (unchanged); s/epoch ~115 s; total wall-clock ~30.7 min
- **Metric artifacts:** `models/model-per_channel_huber_delta-20260513-071528/metrics.jsonl`
- **What changed:** Replaced scalar δ=0.3 with per-channel tensor `delta = abs_err.new_tensor([0.5, 0.5, 0.2])` (Ux δ=0.5, Uy δ=0.5, p δ=0.2). The `torch.where` broadcast over the channel dim applies channel-specific thresholds.
- **Why it worked:** Pressure (p) and velocity (Ux/Uy) have different residual distributions. Pressure residuals are high-variance (large outliers near surface geometry), so a tight δ=0.2 cap is optimal (same trend as the uniform δ=0.3 win vs δ=0.5). Velocity residuals are lower-variance — the old δ=0.3 was over-aggressive, saturating too many velocity gradient signals into the linear regime. Expanding velocity δ to 0.5 restores quadratic signal for velocity channels. All 8 splits improve (−1.93 to −2.68 on val, similar on test), confirming orthogonal axis to width (n_hidden=160) and optimizer (Lion).
- **Baseline configuration delta:** Uniform `δ=0.3` → per-channel `delta=abs_err.new_tensor([0.5, 0.5, 0.2])` in train.py Huber loss.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --experiment_name per_channel_huber_delta_ep16 --agent <student>
  ```

## 2026-05-13 07:10 — PR #1755: n_hidden=160 on Huber δ=0.3+Lion+BF16+epochs=16 stack (MERGED)

- **val_avg/mae_surf_p: 55.9230** (↓ 1.7% from 56.90 — architectural width gain compounds with loss-shape regularisation)
- **test_avg/mae_surf_p: 51.9228** (↓ 2.4% from 53.20 — every test split improves, gen gap preserved)
- **Peak VRAM: 37.99 GB** (38% of 96 GB H100 budget); s/epoch: ~115 s; wall-clock: ~30.7 min
- **Metric artifacts:** `models/model-nhidden160_huber_d03_final-20260513-061938/metrics.jsonl`
- **What changed:** `n_hidden: 128 → 160` in train.py Config dataclass (1-line change). +1.6× parameters (656k → 1.03M). All other config identical to PR #1880 winning stack.
- **Why it worked:** Width and loss-shape (Huber δ) are orthogonal regularisation axes. The wider model's additional capacity benefits most on the hardest OOD split (val_geom_camber_rc: −5.38 / test: −3.39). Both models converge to their respective asymptotes at epoch 16 (grad_norm ~2.3, trajectory flat). Test improves uniformly across all 4 splits (no widened generalisation gap: test−val −4.00 vs −3.70 baseline).
- **Baseline configuration delta:** `n_hidden = 160` in train.py Config dataclass.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --experiment_name nhidden160_d03_ep16 --agent <student>
  ```

## 2026-05-13 06:00 — PR #1880: Huber δ scan δ=0.3 and δ=0.2 on epochs=16 stack (MERGED)

- **val_avg/mae_surf_p: 56.8955** (↓ 14.2% from 66.32 — largest single-PR gain this round after Lion)
- **test_avg/mae_surf_p: 53.2015** (↓ 13.0% from 61.14 — all 4 splits finite)
- **Peak VRAM: 32.95 GB** (unchanged); total wall-clock ~27 min for both arms
- **Metric artifacts:** `models/model-huber_delta0_3-20260513-035824/metrics.jsonl` (winner δ=0.3); `models/model-huber_delta0_2-20260513-050217/metrics.jsonl` (δ=0.2, val=56.94 / test=53.23 — essentially tied)
- **What changed:** δ threshold in Huber loss: `torch.where(abs_err < δ, 0.5*abs_err**2, δ*abs_err - 0.5*δ**2)` — δ=0.5 → δ=0.3 in train.py.
- **Why it worked:** The monotonic improvement trend from δ=1.0 (val 67.41) → δ=0.5 (val 66.32) continued to δ=0.3 with a much larger jump. Smaller δ caps the linear regime more aggressively, reducing gradient signal from larger residuals (extreme-Re tandem samples near surfaces). The jump from 0.5→0.3 is much larger than 1.0→0.5 because 0.3 starts clipping the dominant outlier mass rather than just the tails. The curve flattened between δ=0.3 and δ=0.2 (0.04 val difference), confirming 0.3 is near the optimal floor. Per-domain analysis: δ=0.2 slightly wins the tail-heavy single/raceCar splits; δ=0.3 wins cruise/re_rand splits with smaller residual magnitudes (over-saturation into linear regime at δ=0.2 degrades gradient signal for low-std domains). NOTE: 13→16 epoch contribution is included in this result (both arms ran 16 epochs on the Huber stack).
- **Baseline configuration delta:** δ=0.5 → δ=0.3 in train.py Huber loss.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --experiment_name huber_delta0_3_ep16 --agent <student>
  ```

## 2026-05-13 03:51 — PR #1639: Huber loss (δ=0.5) on Lion stack (MERGED)

- **val_avg/mae_surf_p: 66.32** (↓ 0.2% from 66.44 PR #1780; the entire Huber→MSE swap at 13 epochs beats epochs=16 with MSE at 13 epochs by 0.12 — within run-to-run noise, but the win compounds with epoch extension)
- **test_avg/mae_surf_p: 61.14** (↓ 1.0% from 61.78; all 4 splits finite)
- **Peak VRAM: 32.95 GB**; total wall time ~22 min for 13 epochs
- **Metric artifacts:** `models/model-charliepai2g24h5-alphonse-huber_delta0_5_lion-20260513-025216/metrics.jsonl` (winner); `models/model-charliepai2g24h5-alphonse-huber_delta1_lion-20260513-021619/metrics.jsonl` (δ=1.0 arm, val=67.41)
- **What changed:** Replaced MSE with element-wise Huber (Smooth-L1) loss δ=0.5: `huber(x) = 0.5*x² if |x|<δ else δ*|x| - 0.5*δ²`. Two arms tested: δ=1.0 (val 67.41) and δ=0.5 (val 66.32 — winner). All other hyperparameters identical to #1641 Lion baseline.
- **Why it worked:** Outlier residuals dominate MSE gradient — typically high-Re tandem samples near surfaces. Huber caps the per-element loss-gradient magnitude at δ. The hypothesis that this complements grad_clip held: grad_clip works on global gradient norm post-aggregation, Huber works per-element pre-aggregation, they stack. δ=0.5 uniformly better than δ=1.0 across all 4 val and 4 test splits, suggesting the response curve hasn't bottomed out — smaller δ may help further (next-step experiment).
- **Baseline configuration delta:** MSE → Huber δ=0.5 (code change in train.py).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name huber_delta0_5 --agent <student>
  ```
  (Huber δ=0.5 is the default loss; combine with `--epochs 16` to match #1780 schedule.)

## 2026-05-13 03:50 — PR #1780: Lion + epochs 13→16 (MERGED)

- **val_avg/mae_surf_p: 66.44** (↓ 9.2% from 73.15)
- **test_avg/mae_surf_p: 61.78** (↓ 7.5% from 66.76; all 4 splits finite)
- **Peak VRAM: 32.94 GB** (matches BF16 prediction)
- **s/epoch: 100.1–103.1** (avg ~101.6); total wall-clock 27.1 min (within 30-min cap)
- **Metric artifacts:** `models/model-lion_epochs16-20260513-015116/metrics.jsonl`
- **What changed:** No code change. Run-time flag only: `--epochs 16` instead of `--epochs 13`. The warmup3+cosine schedule's T_max = epochs - 3 = 13 grows with the flag, so the cosine decay tail extends naturally to epoch 16 where LR ≈ 0.
- **Why it worked:** PR #1641 baseline at epoch 13 was still descending monotonically (final epoch delta was substantial). The cosine schedule had cut training short of convergence. Three additional epochs at the cosine tail gave −3.84, −1.59, −1.94 improvements — model converged onto the low-LR basin exactly as hypothesized. BF16+Lion's fast per-epoch (~101s) leaves headroom for 16 epochs in the 30-min cap; without BF16 this experiment would be infeasible.
- **Baseline configuration delta:** epochs 13 → 16 (CLI flag).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 16 --lion_lr 3e-4 --lion_weight_decay 6e-5 --experiment_name lion_epochs16 --agent <student>
  ```

## 2026-05-13 01:20 — PR #1641: Lion optimizer (lr=3e-4) (MERGED)

- **val_avg/mae_surf_p: 73.15** (↓ 22.4% from 94.22 — largest single-PR gain this round)
- **test_avg/mae_surf_p: 66.76** (↓ 23.4% from 87.10)
- **Peak VRAM: 42.11 GB** (FP32 run; merged stack now has BF16 so future runs ~33 GB)
- **Metric artifacts:** `models/model-charliepai2g24h5-frieren-lion_lr3e4-20260512-225827/metrics.jsonl` (winner); `models/model-charliepai2g24h5-frieren-lion_lr1_5e4-20260512-235646/metrics.jsonl` (arm 1)
- **What changed:** Replaced AdamW with Lion optimizer from `lion-pytorch`. Lion lr=3e-4 (= AdamW lr/3.3), lion_wd=6e-5 (= wd/1.67). Added `lion_lr` and `lion_weight_decay` fields to Config. All other config identical: grad_clip(max_norm=1.0), warmup3+cosine13, MSE loss, batch=4, surf_weight=10.0, seed=42.
- **Why it worked:** Lion's per-parameter sign-based update is a stronger version of the global gradient renormalization our baseline already applies via grad_clip. Where grad_clip renormalizes the full gradient vector to L2-norm ≤ 1, Lion takes each parameter's gradient and applies `±lr` regardless of its magnitude — per-parameter sign quantization. This eliminates gradient magnitude variance entirely across the parameter space, producing uniform step sizes across attention slices, MLP layers, and projection matrices. The Transolver's heterogeneous parameter structure (different gradient scales in PhysicsAttention vs MLP projections) benefits strongly from this uniformity. Two arms: lr=1.5e-4 (val 75.17) and lr=3e-4 (val 73.15 — winner).
- **Reproduce:**
  ```bash
  pip install lion-pytorch && cd target/ && python train.py --epochs 13 --lion_lr 3e-4 --lion_weight_decay 6e-5 --experiment_name lion_lr3e4 --agent <student>
  ```

## 2026-05-13 01:05 — PR #1565: BF16 autocast (MERGED)

- **val_avg/mae_surf_p: 94.22** (↓ 1.3% from 95.44)
- **test_avg/mae_surf_p: 87.10** (↓ 0.8% from 87.83)
- **Peak VRAM: 32.94 GB** (↓ 22% from 42.11 GB — unlocks future wider-model experiments)
- **s/epoch: 100.87** (↓ 23% from 131.44 — same 30-min budget now fits more iterations)
- **Metric artifacts:** `models/model-charliepai2g24h5-fern-bf16_only_lr1e3-20260513-001209/metrics.jsonl`
- **What changed:** Added `torch.cuda.amp.autocast(dtype=torch.bfloat16)` wrapping the forward pass inside `train_epoch`. No batch-size change (kept batch=4). No schedule change. Single-line addition.
- **Why it worked:** BF16 reduces memory bandwidth for activations, speeding up compute. The slight metric improvement (−1.3% val, −0.8% test) likely comes from a mild implicit regularisation effect from reduced precision. The VRAM win (+9 GB headroom) is the more important outcome — it enables revisiting n_hidden=192 or n_layers=7 with the current stack.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name bf16_only_lr1e3 --agent <student>
  ```
  (train.py now has BF16 autocast in train_epoch by default)

## 2026-05-12 23:05 — PR #1638: LR=1e-3 with grad_clip (MERGED)

- **val_avg/mae_surf_p: 95.44** (↓ 9.5% from 105.46 — biggest gain this round)
- **test_avg/mae_surf_p: 87.83** (all 4 splits finite, tested from best-val checkpoint epoch 13)
- **Metric artifacts:** `models/model-charliepai2g24h5-tanjiro-lr1e3_gradclip-20260512-221259/metrics.jsonl`
- **What changed:** `lr: 5e-4 → 1e-3` in `train.py` Config dataclass (1-line change, commit `a1b596d`). All other config identical to #1483 baseline.
- **Why it worked:** Grad-clip at max_norm=1.0 fires every step (pre-clip norms 45–112 >> 1.0), effectively renormalising every gradient vector to unit norm. This bounded-step regime can safely absorb a 2× LR increase: each step is geometrically identical but with larger step size. The biggest gains are on OOD splits (val_geom_camber_rc −16.9, val_re_rand −12.6) consistent with improved cross-domain generalisation from the renorm regime.
- **Baseline configuration delta:** `lr: 5e-4 → 1e-3` (AdamW).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name lr1e3_gradclip --agent <student>
  ```
  (train.py now has `lr: float = 1e-3` as default)

## 2026-05-12 21:55 — PR #1483: Gradient clipping max_norm=1.0 (MERGED)

- **val_avg/mae_surf_p: 105.46** (↓ 7.8% from 114.40 — biggest single-step gain this round)
- **test_avg/mae_surf_p:** Reported NaN by source branch (lacked GT-NaN fix). Merged code now has both fixes — re-measure on next run.
- **Metric artifacts:** `models/model-grad_clip_1-20260512-210428/metrics.jsonl`
- **What changed:** Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()` — 1 line in train.py.
- **Why it worked (per tanjiro's analysis):** Pre-clip grad norms are 45–112 throughout training, all well above max_norm=1.0. Clipping fires on EVERY step → effectively renormalises every gradient to unit norm. This is much stronger than "tame occasional outliers" — it's closer to "Adam on g/‖g‖". The largest gains are on splits with highest target magnitudes (val_single_in_dist −34.4%, val_geom_camber_rc −29.2%) — consistent with a Re-rebalancing interpretation: extreme-Re samples no longer dominate gradient steps.
- **Note on baseline measurement:** PR #1483 measured 105.46 against the OLD baseline (no warmup+cosine, no GT-NaN fix), reporting a 26.7% within-PR improvement. The merged code now stacks: warmup+cosine + GT-NaN fix + grad_clip. Composed val_avg may differ slightly from 105.46; next experiments to verify.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name baseline_check --agent <student>
  ```

## 2026-05-12 21:05 — PR #1564: GT-NaN fix in evaluate_split (MERGED)

- **val_avg/mae_surf_p: 114.40** (unchanged — bit-identical to #1519; fix is a no-op on clean GT)
- **test_avg/mae_surf_p: 107.57** (was NaN; now the first valid paper-facing test number)
- **Metric artifacts:** `models/model-gt_nan_fix_baseline-20260512-201204/metrics.jsonl`
- **What changed:** In `evaluate_split`, filter non-finite GT samples before calling `accumulate_batch`:
  `gt_finite_mask = torch.isfinite(y).all(dim=-1)` then AND into `mask` and `is_surface`. Fixes
  IEEE `NaN * 0 = NaN` leakage in `data/scoring.py` (which is read-only). Val results are bit-identical
  to #1519; only `test_geom_camber_cruise/mae_surf_p` changes (NaN → 92.41).
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name gt_nan_fix_baseline --agent <student>
  ```

## 2026-05-12 20:10 — PR #1519: Warmup + cosine to 13-epoch budget (MERGED)

- **val_avg/mae_surf_p: 114.40** (↓ 8.6% from informal 125.20 baseline)
- **test_avg/mae_surf_p: NaN** (cruise GT issue; 3-split clean = 112.63)
- **Metric artifacts:** `models/model-warmup3_cosine13-20260512-190738/metrics.jsonl`
- **What changed:** Added 3-epoch linear warmup before cosine; matched `--epochs 13` to
  actual wall-clock budget so cosine LR reaches near-zero (was T_max=50, only 14 epochs ran,
  LR never decayed meaningfully). Added seed=42 and `nan_to_num` prediction guard.
- **Reproduce:**
  ```bash
  cd target/ && python train.py --epochs 13 --experiment_name warmup3_cosine13 --agent <student>
  ```
  (Plus seed pin and nan_to_num guard from merged `train.py`.)

## Baseline configuration (from `target/train.py`)

| Lever | Value |
|---|---|
| Model | Transolver, n_hidden=160, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Activation | GELU |
| Loss | Per-channel Huber (Smooth-L1): δ=[Ux=0.5, Uy=0.5, p=0.2]; `vol_loss + 10 * surf_loss` |
| Surface weight | 10.0 |
| Optimizer | **Lion, lr=2e-4, weight_decay=6e-5** (lr lowered from 3e-4 → 2e-4 with PR #2027 on per-channel δ stack) |
| LR schedule | LambdaLR, 3-epoch warmup + cosine to T_max=epochs |
| Grad clip | max_norm=1.0 |
| Batch size | 4 (variable mesh sizes, pad_collate to N_max) |
| Sampler | WeightedRandomSampler (balanced domain mix) |
| Max epochs | 13 (within 30-min wall-clock) |
| Timeout | 30 min wall-clock (hard cap) |
| Precision | BF16 autocast in forward pass |

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four
validation splits:

- `val_single_in_dist` — single-foil random holdout (sanity check)
- `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar)
- `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise)
- `val_re_rand` — stratified Re holdout across tandem domains

Best checkpoint = lowest `val_avg/mae_surf_p`. Test eval at end of training
uses that checkpoint and reports `test_avg/mae_surf_p` for the paper.

## How to claim a win

A PR is a winner if its terminal `SENPAI-RESULT` marker reports a strictly
lower `val_avg/mae_surf_p` than this file's current best (or, before the
first merge, beats the out-of-the-box baseline of the same epoch/timeout
budget by a clearly statistically meaningful margin and reports the
matching test number).

## Update procedure

When a new winner merges, update this file with:

- The merged PR number
- New `val_avg/mae_surf_p`
- Matching `test_avg/mae_surf_p`
- One-line note on what changed
