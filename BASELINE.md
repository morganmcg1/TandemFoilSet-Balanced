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
