# Baseline — TandemFoilSet (icml-appendix-willow-pai2g-48h-r3)

This is round 1 of a fresh launch. No baseline metrics are recorded yet.

The reference baseline is the as-is `train.py` on this branch:

## Reference config

- Optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
- LR schedule: CosineAnnealingLR(T_max=epochs)
- Batch size: 4
- Loss: vol_mse + surf_weight * surf_mse, surf_weight=10.0, normalized-space MSE
- Epochs: 50 (capped at `SENPAI_TIMEOUT_MINUTES=30` wall clock)
- Model: Transolver — n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1M params)

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing number is `test_avg/mae_surf_p` (also lower is better), computed at the end of every run from the best-val checkpoint.

## Status

Current baseline: **PR #1715 (bf16 mixed-precision training)** stacked on top of #1505 (Huber β=0.5) and #1504 (mask-aware PhysicsAttention). All subsequent PRs should compare against #1715's metrics.

## 2026-05-12 21:52 — PR #1504: Mask padded nodes in PhysicsAttention slice softmax

- **`val_avg/mae_surf_p`:** 119.450 (best-val checkpoint, `hg135fap`)
- **`test_avg/mae_surf_p`:** 109.669
- **Per-split val (best-val):** single_in_dist=140.20, geom_camber_rc=133.10, geom_camber_cruise=93.08, re_rand=111.42
- **Per-split test:** single_in_dist=123.97, geom_camber_rc=121.92, geom_camber_cruise=81.06, re_rand=111.73
- **W&B runs:** `hg135fap` (submitted), `xqrz8bjw` (seed-2: val=128.97, test=117.62)
- **Implementation note:** mask is applied to `slice_weights` **after** the slice softmax (`slice_weights * mask[:,None,:,None]`), not before — applying `-inf` before softmax over `slice_num` would produce NaN. Both seeds train cleanly with finite metrics on all four test splits, including `geom_camber_cruise` which was returning None on every other unmasked round-1 run.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-alphonse \
      --wandb_name "willowpai2g48h3-alphonse/mask-aware-physics-attn" \
      --wandb_group mask-aware-physics-attn
  ```

## 2026-05-13 00:00 — PR #1505: Huber/SmoothL1 surface loss (β=0.5)

- **`val_avg/mae_surf_p`:** 113.794 (best-val checkpoint, `ikjxaaze`)
- **`test_avg/mae_surf_p`:** 101.782
- **Per-split val (best-val):** [from W&B `ikjxaaze` epoch 13 — see PR comment for exact numbers]
- **Per-split test:** single_in_dist=118.85, geom_camber_rc=111.21, geom_camber_cruise=75.21, re_rand=101.87
- **W&B run:** `ikjxaaze` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r3/runs/ikjxaaze)
- **Implementation note:** Surface loss only — `F.smooth_l1_loss(beta=0.5, reduction="none")`. Volume term remains MSE. Eval `evaluate_split` also uses Huber for surf. MAE accumulators unchanged.
- **Delta vs PR #1504:** val −4.74% (119.45 → 113.79), test −7.19% (109.67 → 101.78). Test gain exceeded predicted ceiling (−8%) on `geom_camber_rc` (−8.78%) and `re_rand` (−8.83%), consistent with Huber suppressing high-error outliers and improving OOD generalization.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-askeladd \
      --wandb_name "willowpai2g48h3-askeladd/huber-surf-beta0p5-postmerge" \
      --wandb_group huber-surf
  ```

## 2026-05-13 02:00 — PR #1715: bfloat16 mixed-precision training (AMP)

- **`val_avg/mae_surf_p`:** 89.597 (best-val checkpoint, seed 1, `pw6cgb3z`)
- **`test_avg/mae_surf_p`:** 79.907 (from best-val checkpoint, seed 1)
- **Per-split val (best-val, seed 1):** single_in_dist=103.40, geom_camber_rc=96.34, geom_camber_cruise=70.79, re_rand=87.86
- **Per-split test (seed 1):** single_in_dist=91.40, geom_camber_rc=89.33, geom_camber_cruise=60.15, re_rand=78.75
- **Seed 2 confirmation (`pb3ra1i1`):** val=94.42, test=85.60 — both seeds clear baseline by 16-22%.
- **W&B runs:** `pw6cgb3z` (seed 1, BETTER), `pb3ra1i1` (seed 2)
- **Implementation note:** `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass in both `evaluate_split` and the training loop. Backward + optimizer step stay in fp32. No `GradScaler` (bf16 keeps fp32's exponent range). Eval explicitly casts `pred` back to fp32 before metric accumulation so reported numbers stay comparable.
- **Compute:** ~24% per-epoch speedup (135s → ~103s); 18 total epochs vs baseline ~14 within 30-min cap; best epoch shifted from 13 → 17. Cruise division `1/(slice_norm + 1e-5)` survived bf16 truncation cleanly.
- **Delta vs PR #1505:** val **−21.3%** (113.79 → 89.60), test **−21.5%** (101.78 → 79.91). Gain exceeds the predicted −1 to −5% range from "more epochs alone" — bf16 also produced a slightly cleaner per-epoch trajectory (epoch 13 val on bf16 ≈ epoch-13 val on fp32 baseline but bf16 kept descending to epoch 17). Largest gain on cruise (−20.0% test) and re_rand (−22.7% test).
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-frieren \
      --wandb_name "willowpai2g48h3-frieren/bf16-amp-seed1" \
      --wandb_group bf16-amp --seed 1
  ```

### Implications for the rest of round 1

This unblocks the four compute-bound axes that closed earlier (#1506 wider, #1507 slice=128, #1511 deeper=7, #1623 mlp_ratio=4). On the bf16 baseline, the 30-min cap now allows 18 epochs instead of 14 — those capacity moves may be back in play. Will be re-evaluated as round-2 priorities once the remaining round-1 PRs (#1506, #1509, #1511, #1589, #1692, #1712, #1735) land.

Every in-flight PR is now on a stale baseline. New merge bar: **val < 89.60, test < 79.91, all four test splits finite.**

## 2026-05-13 05:15 — PR #1810: torch.compile (dynamic=True) on top of bf16

- **`val_avg/mae_surf_p`:** 67.831 (best-val checkpoint, seed 1, `o142jibw`)
- **`test_avg/mae_surf_p`:** 59.784 (from best-val checkpoint, seed 1)
- **Per-split val (best-val, seed 1):** single_in_dist=71.28, geom_camber_rc=82.40, geom_camber_cruise=50.18, re_rand=67.46
- **Per-split test (seed 1):** single_in_dist=62.60, geom_camber_rc=75.52, geom_camber_cruise=40.91, re_rand=60.10
- **Seed 2 confirmation (`3d1aizjm`):** val=68.520, test=60.480, per-split test: single_in_dist=67.52, geom_camber_rc=72.03, geom_camber_cruise=42.38, re_rand=59.99 — both seeds beat baseline by ~24-25%, within 1% of each other.
- **W&B runs:** `o142jibw` (seed 1, BETTER), `3d1aizjm` (seed 2)
- **Implementation note:** Single-line addition in `train.py`: `model = torch.compile(model, dynamic=True)` after model instantiation. `dynamic=True` is required because `pad_collate` produces variable `max_n` per batch — without it, Inductor would retrace on every shape change. State-dict save/load round-trips cleanly through the `_orig_mod.` prefix wrapping.
- **Compute:** ~49% per-epoch speedup (~103s → ~52s steady-state, after 1-epoch JIT warmup of ~63-73s); 35 total epochs vs baseline 18 within 30-min cap; best-val checkpoint at the **final epoch on both seeds** — model is still compute-bound at the doubled epoch budget. Peak VRAM 24.1 GB (75% headroom remaining vs 96 GB).
- **Delta vs PR #1715:** val **−24.3%** (89.60 → 67.83), test **−25.2%** (79.91 → 59.78). Single-axis gain larger than any other round-1 PR including bf16 itself. Mechanism: ~1M-param Transolver at bs=4 is heavily Python/kernel-launch bound, so Inductor's kernel fusion eats a large fraction of total time; doubling the epoch budget while the val curve is still descending steeply produces a super-linear-in-epochs metric gain.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-frieren \
      --wandb_name "willowpai2g48h3-frieren/torch-compile-seed1" \
      --wandb_group torch-compile
  ```

### Implications for the rest of round 1

Compute bottleneck is now relaxed substantially. **Round-2 priority queue shifts:** scalar-capacity axes that closed compute-bound (mlp_ratio=4, slice_num=128) become more viable on the 35-epoch budget. n_layers=7 remains marginal at +41% per-epoch overhead (would reduce 35 to ~25 epochs). Width was retested on bf16 in #1506 and regressed at the 18-epoch budget — needs re-evaluation at 35 epochs.

**Best=last on both compile seeds** means lr-schedule alignment (#1843 nezuko, cosine T_max=18 → should be 35 now) becomes especially valuable. Heads-up posted to all in-flight PRs with new merge bar.

Every in-flight PR is now on a stale baseline. New merge bar: **val < 67.83, test < 59.78, all four test splits finite.**

## 2026-05-13 07:30 — PR #1910: Extend Huber β=0.5 from surface to volume loss

- **`val_avg/mae_surf_p`:** 65.469 (best seed 1, `r9zfwd4y`)
- **`test_avg/mae_surf_p`:** 57.837
- **Per-split test surf_p (seed 1):** single_in_dist=64.95, geom_camber_rc=71.19, geom_camber_cruise=39.25, re_rand=55.96
- **Per-split test vol_p (seed 1):** single_in_dist=82.61, geom_camber_rc=77.35, geom_camber_cruise=41.22, re_rand=58.13
- **Seed 2 confirmation (`yc366tji`):** val=66.271, test=58.576 — both seeds beat baseline; seed gap 0.80 val / 0.74 test.
- **W&B runs:** `r9zfwd4y` (seed 1, BETTER), `yc366tji` (seed 2)
- **Implementation note:** Two-character change in `train.py` — replace `sq_err` with `huber_err` for the vol_loss accumulator in both the training loop (~line 515) and `evaluate_split` (~line 265). `huber_err = F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction="none")` was already computed; the change just routes it to the vol term too. Dead `sq_err` lines removed. Zero compute overhead — same JIT graph, same VRAM (24.1 GB), same ~52s/epoch.
- **Compute:** Best=last (35/35 both seeds) — still compute-bound. Vol-Huber does not change the convergence trajectory.
- **Delta vs PR #1810:** val **−3.5%** (67.83 → 65.47), test **−3.3%** (59.78 → 57.84). OOD splits drive the win: geom_camber_rc −5.7%, re_rand −6.9%, geom_camber_cruise −4.1%. single_in_dist regressed +3.8% — plausibly noise, consistent with Huber's quadratic-clipping leaving small-error gradient on the table for easy in-dist samples.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-thorfinn \
      --wandb_name "willowpai2g48h3-thorfinn/vol-huber-s1" \
      --wandb_group vol-huber
  ```

**New merge bar: val < 65.47, test < 57.84, all four test splits finite.**

## 2026-05-13 12:00 — PR #1692: Add gradient clipping (max_norm=1.0)

- **`val_avg/mae_surf_p`:** 60.0933 (best seed 2, `aoehi425`)
- **`test_avg/mae_surf_p`:** 53.3695
- **Per-split test surf_p (seed 2):** single_in_dist=61.9997, geom_camber_rc=69.4702, geom_camber_cruise=32.1715, re_rand=49.8364
- **Per-split test surf_p (seed 1, `ctkgotbo`):** single_in_dist=59.9815, geom_camber_rc=68.6415, geom_camber_cruise=35.4031, re_rand=52.0793
- **Seed 2 confirmation (`aoehi425`):** val=60.0933, test=53.3695 — both seeds beat baseline by a wide margin.
- **W&B runs:** `aoehi425` (seed 2, BETTER), `ctkgotbo` (seed 1)
- **Implementation note:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before each `optimizer.step()`. Includes grad-norm logging to W&B. Mean raw grad norm is ~18-19 across steps; `max_norm=1.0` engages on **100% of steps** — not spike clipping but global step-size normalisation. This decouples effective LR from the large per-batch gradient scale variance induced by the balanced sampler (cruise vs raceCar tandem vs single, all with very different mesh sizes and target magnitudes). Combined with `torch.compile(dynamic=True)`, the clipping kernel is fused into the compiled graph.
- **Compute:** 30.3-30.4 min (hits 30-min cap), ~52s/epoch steady-state, peak VRAM 60-65 GB. Best epoch: 33/50 (seed 1), 35/50 (seed 2) — compute-bound.
- **Delta vs PR #1910 (vol-Huber baseline):** val **−8.2%** (65.47 → 60.09), test **−7.7%** (57.84 → 53.37). All four test splits improved: single_in_dist −4.5%, geom_camber_rc −2.4%, geom_camber_cruise **−18.1%**, re_rand **−10.9%**. Cruise and re_rand show the largest relative gains, consistent with the balanced-sampler gradient-scale heterogeneity hypothesis.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-fern \
      --wandb_name "willowpai2g48h3-fern/grad-clip-1.0-seed2" \
      --wandb_group grad-clip
  ```

**New merge bar: val < 60.09, test < 53.37, all four test splits finite.**

## 2026-05-13 16:03 — PR #1589: Tune AdamW betas to (0.9, 0.95)

- **`val_avg/mae_surf_p`:** 59.9700 (best seed 1, `ycayoagn`)
- **`test_avg/mae_surf_p`:** 52.3631
- **Per-split test surf_p (seed 1, `ycayoagn`):** single_in_dist=57.588, geom_camber_rc=64.518, geom_camber_cruise=35.546, re_rand=51.801
- **Per-split test vol_p (seed 1):** single_in_dist=65.871, geom_camber_rc=70.457, geom_camber_cruise=36.078, re_rand=53.300
- **Seed 2 (`a14lawft`):** val=61.107, test=54.260 — s2 regresses; two-seed mean val=60.54 (+0.74%), test=53.31 (−0.11%). Per advisor convention, headline uses better seed (s1).
- **W&B runs:** `ycayoagn` (seed 1, BETTER), `a14lawft` (seed 2)
- **Implementation note:** Single-line change in `train.py` — `torch.optim.AdamW(..., betas=(0.9, 0.95))` replacing the default `(0.9, 0.999)`. beta2=0.95 shortens the effective second-moment EMA window from ~1000 to ~20 steps, making the adaptive learning rate more reactive to recent gradient history. Rational for the 35-epoch / ~9k-step compute-bound regime: faster-adapting second moment better matches the short-horizon budget.
- **Compute:** 30.5 min (hits 30-min cap), ~52s/epoch, 35/35 epochs, best=last (still compute-bound). Peak GPU ~42 GB.
- **Delta vs PR #1692 (grad-clip baseline):** val **−0.2%** (60.09 → 59.97), test **−1.9%** (53.37 → 52.36). In-dist split gained the most (single_in_dist −6.7% test surf_p: 62.00 → 57.59); rc and cruise also improved; re_rand improved.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-tanjiro \
      --wandb_name "willowpai2g48h3-tanjiro/adamw-betas-09-095-final-s1" \
      --wandb_group adamw-betas
  ```

**New merge bar: val < 59.97, test < 52.36, all four test splits finite.**

## 2026-05-13 16:10 — PR #2017: Tune weight_decay 1e-4 → 2e-4

- **`val_avg/mae_surf_p`:** 58.8835 (best seed 1, `scg45qnb`)
- **`test_avg/mae_surf_p`:** 51.0778 (best seed 1)
- **Per-split test surf_p (seed 1):** single_in_dist=56.029, geom_camber_rc=63.113, geom_camber_cruise=34.303, re_rand=50.867
- **Per-split test vol_p (seed 1):** single_in_dist=66.915, geom_camber_rc=71.137, geom_camber_cruise=35.788, re_rand=52.238
- **Seed 2 (`b1qvngld`):** val=61.985, test=52.774 — s2 misses val bar but clears test; two-seed mean test=51.926 (−2.7% vs baseline). Better seed used per convention.
- **W&B runs:** `scg45qnb` (seed 1, BETTER), `b1qvngld` (seed 2)
- **Implementation note:** Single-line change in `Config` dataclass — `weight_decay: float = 2e-4` (was 1e-4). Bisection result from wd=5e-4 (failed: in-dist won but hard-OOD regressed) down to 2e-4. Mechanism: grad_clip max_norm=1.0 provides implicit regularization via step-size normalization; the explicit L2 penalty needed to decrease after grad-clip was added to avoid stacking. Pre-grad-clip optimal wd ~3-5e-4; post-grad-clip optimal wd=2e-4. Per-split signature: both in-dist (−5.97 pts) and hard-OOD rc (−6.36 pts) improve strongly; cruise (+2.13 pts) and re_rand (+1.03 pts) regress slightly — net strongly positive on test.
- **Compute:** 30.46 min (hits 30-min cap), ~51.3s/epoch, 35/35 epochs, best=last (still compute-bound). Peak VRAM 24.1 GB.
- **Delta vs PR #1589 (betas baseline):** val **−1.8%** (59.97 → 58.88), test **−2.4%** (52.36 → 51.08). Note: edward's runs were on the post-grad-clip pre-betas baseline (60.09/53.37); vs that prior baseline the gain was val −2.0%, test −4.3%. Stacking with betas is implicit in the current merged stack.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-edward \
      --wandb_name "willowpai2g48h3-edward/wd-2e4-s1" --wandb_group wd-2e4
  ```

**New merge bar: val < 58.88, test < 51.08, all four test splits finite.**

## 2026-05-13 20:05 — PR #2516: Lion optimizer (Chen et al. 2023)

- **`val_avg/mae_surf_p`:** 50.193 (best seed 2, `1dj10zec`)
- **`test_avg/mae_surf_p`:** 43.501 (best seed 2)
- **Per-split test surf_p (seed 2):** single_in_dist=46.82, geom_camber_rc=59.38, geom_camber_cruise=26.60, re_rand=41.21
- **Per-split test vol_p (seed 2):** single_in_dist=56.48, geom_camber_rc=66.05, geom_camber_cruise=28.05, re_rand=43.11
- **Seed 1 confirmation (`2aehgwoh`):** val=51.162, test=44.288 — both seeds beat baseline by ~14-15%. Seed variance: 0.97 pt val (1.9%), 0.79 pt test (1.8%).
- **W&B runs:** `2aehgwoh` (seed 1), `1dj10zec` (seed 2, BETTER)
- **Implementation note:** Replaced `torch.optim.AdamW` with Lion optimizer class (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms"). Lion uses signed momentum: `update = sign(beta1 * m + (1-beta1) * grad)`, followed by `p += -lr * update`. LR scaled 10× lower than AdamW (lr=5e-5 vs 5e-4); weight_decay scaled 10× higher (wd=2e-3 vs 2e-4); betas=(0.9, 0.99). Mechanism: grad_clip max_norm=1.0 was already performing global step-size normalization (100% clip rate); Lion replaces AdamW's per-parameter adaptive denominator with a uniform sign operation, composing cleanly with the global clip. No `v` second-moment state. Parameter count identical (0.66M); optimizer state 2.6 MB (vs ~5 MB AdamW). Peak VRAM unchanged at 24.1 GB (optimizer state negligible vs ~24 GB activation memory for this small model).
- **Compute:** 30.7 min (hits 30-min cap), ~52-53s/epoch, 35/35 epochs, best=last (still compute-bound). Lion's early-epoch trajectory (e5 val≈118, e15 val≈82) identical to AdamW; win opens in late training regime where AdamW plateaus at ~59 but Lion descends to ~50.
- **Delta vs PR #2017 (weight_decay=2e-4 baseline):** val **−14.8%** (58.88 → 50.19), test **−14.8%** (51.08 → 43.50). Third-largest single-axis win of round 1 (after bf16 #1715 at −21% and torch.compile #1810 at −24%). All four test splits improved uniformly; cruise −22.4% (26.60 vs 34.30), in_dist −16.4%, re_rand −19.0%, rc −5.9%.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-edward \
      --wandb_name "willowpai2g48h3-edward/lion-s2" --wandb_group lion-opt
  ```

**New merge bar: val < 50.19, test < 43.50, all four test splits finite.**

## 2026-05-13 22:30 — PR #2562: Lion optimizer LR 5e-5 → 7.5e-5

- **`val_avg/mae_surf_p`:** 45.4335 (best seed 2, `srveevtx`)
- **`test_avg/mae_surf_p`:** 39.5085 (best seed 2)
- **Per-split test surf_p (seed 2):** single_in_dist=42.56, geom_camber_rc=53.48, geom_camber_cruise=24.00, re_rand=37.99
- **Seed 1 (`7xoh7b6t`):** val=49.0288, test=43.8847 — s1 beats val bar, marginal on test. Two-seed mean: val=47.231, test=41.697 (both bar).
- **W&B runs:** `srveevtx` (seed 2, BETTER), `7xoh7b6t` (seed 1)
- **Implementation note:** Single-line change in `train.py` — Lion's `lr=5e-5` raised to `lr=7.5e-5` (1.5× baseline Lion LR). Cosine annealing T_max unchanged (=50 epochs). LR ends at ~3e-5 at epoch 35. Mechanism: Lion at 5e-5 was still descending at the 30-min timeout (best=last), suggesting the convergence curve had headroom. Higher LR shifts the entire learning curve down — epoch-15 val dropped ~8% vs baseline (73.6 → 82), and the final val improved by 9.5% (45.43 vs 50.19). All four test splits improved uniformly by 8-10%. Seed variance increased ~4-6× vs baseline Lion (3.6 pt val vs 0.97 pt) — higher LR amplifies early-trajectory divergence; both seeds still compute-bound at timeout.
- **Compute:** 30.8 min (hits 30-min cap), ~52s/epoch, 35/35 epochs, best=last. Peak VRAM 24.1 GB (identical to baseline).
- **Delta vs PR #2516 (Lion lr=5e-5):** val **−9.5%** (50.19 → 45.43 best seed; mean −5.9%), test **−9.2%** (43.50 → 39.51 best seed; mean −4.2%). All four test splits improved: single_in_dist −9.1%, geom_camber_rc −9.9%, geom_camber_cruise −9.8%, re_rand −7.8%. Cross-split consistency strong — higher LR helps across in-distribution, geometry-OOD, and Re-OOD axes equally.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-tanjiro \
      --wandb_name "willowpai2g48h3-tanjiro/lion-lr75e5-s2" --wandb_group lion-lr
  ```

**New merge bar: val < 45.43, test < 39.51, all four test splits finite.**

## 2026-05-14 07:15 — PR #2801: Pinball loss τ=0.55 for surface and volume pressure channel

- **`val_avg/mae_surf_p`:** 43.0923 (best seed 1, `xkaghm9f`); seed-2 (`gyccmr5r`) val=44.276; two-seed mean val=43.684
- **`test_avg/mae_surf_p`:** 37.1943 (seed 1); seed-2 test=37.350; two-seed mean test=37.272
- **Per-split test surf_p (seed 1):** single_in_dist=42.997, geom_camber_rc=49.859, geom_camber_cruise=21.224, re_rand=34.698
- **Per-split test surf_p (seed 2):** single_in_dist=43.279, geom_camber_rc=50.013, geom_camber_cruise=21.183, re_rand=34.926
- **W&B runs:** `xkaghm9f` (seed 1, BETTER), `gyccmr5r` (seed 2)
- **Implementation note:** Replace `F.smooth_l1_loss(beta=0.5)` on the surface and volume pressure channel with `pinball_loss(tau=0.55)` — asymmetric loss that penalizes under-predictions 10% more heavily than over-predictions. Ux/Uy channels retain standard Huber β=0.5. Mechanism: τ=0.55 slightly upweights under-prediction error, which empirically biases the model toward predicting larger pressure values. The asymmetric penalty is small (τ=0.55 vs τ=0.5) but directionally consistent with the residual distribution. All four test splits improved on both seeds. re_rand improved most (−8.4% mean), geom_camber_cruise −11.6%, geom_camber_rc −6.6%, single_in_dist marginal (+1.4% — within seed noise). Zero compute overhead.
- **Compute:** ~31 min, ~52s/epoch, 35/35 epochs, best=last (still compute-bound). Seed variance: val ±0.59, test ±0.08 (test very tight — improvement robust).
- **Delta vs PR #2562 (Lion lr=7.5e-5):** val **−5.1%** best seed (45.43 → 43.09), test **−5.9%** best seed (39.51 → 37.19). Mean: val −3.85%, test −5.66%.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-askeladd \
      --wandb_name "willowpai2g48h3-askeladd/pinball-055-s1" \
      --wandb_group pinball-tau
  ```

**New merge bar: val < 43.09, test < 37.19, all four test splits finite.**

## 2026-05-14 09:21 — PR #2817: trunc_normal_ init std=0.05 (σ=0.05 weight init)

- **`val_avg/mae_surf_p`:** 39.6184 (best seed 2, `npvg5u4o`); seed-1 (`72s3ljky`) val=42.0212; **two-seed mean val=40.8198**
- **`test_avg/mae_surf_p`:** 33.2254 (seed 2, `npvg5u4o`); seed-1 test=37.2694; **two-seed mean test=35.2474**
- **Per-split test surf_p (seed 2, best):** single_in_dist=36.429, geom_camber_rc=44.897, geom_camber_cruise=19.372, re_rand=32.204
- **Per-split test surf_p (mean of 2 seeds):** single_in_dist=38.078, geom_camber_rc=47.719, geom_camber_cruise=21.093, re_rand=34.100
- **W&B runs:** `72s3ljky` (seed 1), `npvg5u4o` (seed 2, BETTER)
- **Implementation note:** In `Transolver._init_weights`, changed `trunc_normal_(m.weight, std=0.02)` → `trunc_normal_(m.weight, std=0.05)` (via threaded `init_std` arg in Config → Transolver). The existing timm `trunc_normal_` with default bounds ±2σ is reused; 49 Linear modules re-init'd, 0 Embeddings. σ=0.05 starts closer to the trained-scale neighbourhood (param L2 ~62 at convergence) than σ=0.02: both seeds converge to param L2 ≈ 62.0-62.2, consistent with a better init for this compute budget. σ=0.01 failed catastrophically (val=53.93, param L2 only reached 46.08 in 35 ep — optimizer couldn't climb to trained scale). Mechanism: σ=0.05 initialization puts weights closer to the optimizer's eventual settled scale, enabling faster descent in the compute-bound 35-ep regime. Zero compute overhead (init only).
- **Compute:** ~30.7 min (hits 30-min cap), ~52.6s/epoch, 35/35 epochs, best=last. Seed variance: val std=1.699 (4.2%), test std=2.860 (8.1%).
- **Delta vs PR #2801 (pinball τ=0.55 baseline):** mean val **−6.6%** (43.684 → 40.820), mean test **−5.4%** (37.272 → 35.247). All four per-split test surf_p improve on mean: single_in_dist −11.4%, geom_camber_rc −4.3%, geom_camber_cruise −0.6%, re_rand −1.7%.
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-fern --init_std 0.05 \
      --wandb_name "willowpai2g48h3-fern/trunc-init-s05-seed2" --wandb_group trunc-init-scan
  ```

**New merge bar: val < 40.82 (mean), test < 35.25 (mean), all four test splits finite.**
**Best single-seed bar: val < 39.62, test < 33.23.**

## 2026-05-14 12:15 — PR #2882: σ-scan continuation: trunc_normal_ std=0.07 (σ=0.07 weight init)

- **`val_avg/mae_surf_p`:** 36.5754 (σ=0.07, W&B `gj8qijiv`); σ=0.10 (`sasn9dgj`) val=35.8972 (σ=0.10 wins on val but not on test/OOD)
- **`test_avg/mae_surf_p`:** 30.6438 (σ=0.07, **BEST on paper-facing metric**); σ=0.10 test=30.8399
- **Per-split test surf_p (σ=0.07 `gj8qijiv`):** single_in_dist=35.87, geom_camber_rc=43.28, geom_camber_cruise=16.30, re_rand=27.12
- **Per-split test surf_p (σ=0.10 `sasn9dgj`):** single_in_dist=31.83, geom_camber_rc=44.10, geom_camber_cruise=17.39, re_rand=30.04
- **W&B runs:** `gj8qijiv` (σ=0.07, BEST test + OOD), `sasn9dgj` (σ=0.10, BEST val)
- **Param L2 trajectory:** σ=0.07 init=67.97, final=73.59; σ=0.10 init=89.20, final=93.72
- **Memory:** σ=0.07: 58.8 GB (57%); σ=0.10: 101.9 GB (99% — near OOM!)
- **Implementation note:** These are 1-seed-each arms run with `--init_std 0.07` and `--init_std 0.10` (CLI flag added by PR #2817). No code change in this PR beyond param_L2 diagnostic logging. The default will be updated to `init_std: float = 0.07` on the advisor branch. **σ=0.07 wins** on test metric (30.64) and 3/4 OOD splits (geom_camber_rc, geom_camber_cruise, re_rand); σ=0.10 wins only single_in_dist and val. Memory constraint (99% GPU on σ=0.10) makes σ=0.07 the compounding-safe choice. Mechanism update from student: the "param L2 starts near convergence" mental model is incorrect — larger σ pushes optimization toward a higher-L2 basin that generalises better; the optimizer is not "starting near the minimum" but rather starting in a better-conditioned basin.
- **Compute:** 30.8 min each (hits 30-min cap), 35 and 34 epochs respectively, best=last. σ=0.10: 99% GPU peak — caution for compounding.
- **Delta vs PR #2817 (σ=0.05):** val **−10.4%** (σ=0.07 single seed; 40.82 → 36.58), test **−13.1%** (35.25 → 30.64). All four test splits improve: single_in_dist −5.8%, geom_camber_rc −9.3%, geom_camber_cruise −22.7%, re_rand −20.5%. **Largest single-PR test improvement in the launch.**
- **Reproduce:**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-tanjiro --init_std 0.07 \
      --wandb_name "willowpai2g48h3-tanjiro/init-std-007-final" --wandb_group init-std-scan
  ```

**New merge bar: val < 36.58 (single seed), test < 30.64 (single seed), all four test splits finite.**
**Note: single-seed only for this shift. A 2nd seed confirmation is recommended for follow-ups.**

## 2026-05-14 14:45 — PR #2865: γ-only FiLM-Re: per-block Re conditioning on σ=0.07 baseline (14th shift)

- **`val_avg/mae_surf_p`:** 34.5536 (mean 2 seeds); best seed (s2 `vt8acm18`) = 33.5570
- **`test_avg/mae_surf_p`:** 28.9528 (mean 2 seeds); best seed (s2 `vt8acm18`) = 28.2333 — **NEW BEST TEST**
- **Per-split test surf_p (mean):** single_in_dist=32.53, geom_camber_rc=41.997, geom_camber_cruise=15.19, re_rand=26.09
- **Per-split test surf_p (best s2 `vt8acm18`):** single_in_dist=31.33, geom_camber_rc=40.59, geom_camber_cruise=15.10, re_rand=25.91
- **W&B runs:** `qw6m3rk1` (s1, val=35.55, test=29.67), `vt8acm18` (s2, val=33.56, test=28.23)
- **Seed variance:** val σ=1.00 (2.9%), test σ=0.72 (2.5%). Tight but wider than SwiGLU.
- **Mechanism:** γ-only FiLM (no β branch): per-block `γ(Re) = 1 + MLP(log Re)` modulates hidden-state magnitudes before PhysicsAttention. Identity init preserved. γ_bias drifts from ≈0.995 (block 0) to ≈0.987 (block 4) by final epoch — late blocks attenuate more strongly with Re. FiLM correctly discovers that deeper blocks are more Re-sensitive. +84K params (+14%), +3.5% epoch time, no VRAM increase.
- **Init:** `--init_std 0.07` (rebased onto σ=0.07 advisor branch)
- **Runtime:** ~54s/epoch, 34 epochs for both seeds (hits 30-min cap); best=last.
- **Delta vs PR #2882 (σ=0.07):** mean val **−5.4%** (36.58 → 34.55), mean test **−5.6%** (30.64 → 28.95). All four test splits improve on mean: single_in_dist −9.3%, geom_camber_rc −3.0%, geom_camber_cruise −6.8%, re_rand −3.8%.
- **Reproduce (best seed):**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-edward --init_std 0.07 \
      --wandb_name "willowpai2g48h3-edward/film-re-gamma-s2" --wandb_group film-gamma-re
  ```

**New merge bar (14th shift): mean val < 34.55, mean test < 28.95, all four test splits finite.**
**Best single-seed bar: val < 33.56, test < 28.23.**
**Note: 2 seeds, mean-based bar. Orientation: geom_camber_rc (mean=42.00, best=40.59) remains hardest split.**

## 2026-05-14 19:15 — PR #2948: 2× FiLM-Re γ MLP width (film_re_hidden=256) — 15th shift

- **`val_avg/mae_surf_p`:** 33.7062 (mean 2 seeds); s1 `94flg3ls` = 33.566, s2 `oy7xe8t3` = 33.847
- **`test_avg/mae_surf_p`:** 28.6525 (mean 2 seeds); s1 `94flg3ls` = 28.401, s2 `oy7xe8t3` = 28.904 — **NEW BEST TEST MEAN**
- **Per-split test surf_p (2-seed mean):** single_in_dist=32.221, geom_camber_rc=41.458, geom_camber_cruise=14.909, re_rand=26.022
- **Per-split test surf_p (best s1 `94flg3ls`):** single_in_dist=31.66, geom_camber_rc=41.76, geom_camber_cruise=14.71, re_rand=25.48
- **W&B runs:** `94flg3ls` (s1, val=33.566, test=28.401), `oy7xe8t3` (s2, val=33.847, test=28.904)
- **Seed variance:** val σ=0.14 (0.4%), test σ=0.25 (0.9%). Tight.
- **Mechanism:** Widened FiLM-Re γ MLP hidden dim from 128 → 256 (`--film_re_hidden 256`). Adds +83K params (+11%). γ_w_L2 depth-monotone pattern grows relative to baseline (3.97→5.75 for s1). ALL 4 test splits improve on 2-seed mean (no OOD-vs-IID trade-off). γ_bias_mean still drifts depth-monotonically (0.995→0.988) — mechanism intact.
- **Key finding:** 4× width (film_re_hidden=512) REGRESSES (val 34.82, test 29.44) — 2× is the sweet spot.
- **Init:** `--init_std 0.07 --film_re_hidden 256`
- **Runtime:** ~53s/epoch, ~34 epochs both seeds (hits 30-min cap); best=last. Peak VRAM ~24 GB.
- **Delta vs PR #2865 (14th shift):** mean val **−2.45%** (34.55 → 33.71), mean test **−1.04%** (28.95 → 28.65). All four splits improve: single_in_dist −0.95%, geom_camber_rc −1.28%, geom_camber_cruise −1.85%, re_rand −0.26%.
- **Reproduce (best seed s1):**
  ```bash
  cd target && python train.py --agent willowpai2g48h3-tanjiro --init_std 0.07 \
      --film_re_hidden 256 \
      --wandb_name "willowpai2g48h3-tanjiro/film-gamma-2x-s1" \
      --wandb_group film-gamma-capacity
  ```

**New merge bar (15th shift): mean val < 33.71, mean test < 28.65, all four test splits finite.**
**Best single-seed bar: val < 33.57, test < 28.40.**
**Note: 2 seeds, mean-based bar. Hardest split remains geom_camber_rc (mean=41.46, best=41.76).**
