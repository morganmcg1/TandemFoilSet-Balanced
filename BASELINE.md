# Baseline — icml-appendix-willow-pai2g-24h-r5

This is the per-launch baseline tracker. Branch `icml-appendix-willow-pai2g-24h-r5` was cut from `icml-appendix-willow` with no prior advisor work, so the starting point is `train.py` at HEAD.

## Starting configuration (train.py HEAD)

- Model: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (≈ 0.8M params)
- Optimizer: AdamW (`lr=5e-4, weight_decay=1e-4`)
- Schedule: CosineAnnealingLR(T_max=epochs)
- Loss: weighted MSE in normalized space, `surf_weight=10.0` (volume+surface losses summed)
- Batch size 4, default `epochs=50`
- Per-training cap: `SENPAI_TIMEOUT_MINUTES=30` wall-clock

## Primary ranking metrics

- val: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- test: `test_avg/mae_surf_p` (equal-weight mean across 4 test splits, evaluated from the best-val checkpoint)

## Best result so far

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| **#2338 n_head=1 on n_head=2+slice_num=32+Lion+MAE+lr=1e-4** | **46.67** | **40.69** | −3.9% val / −1.9% test vs #2335; wins all 4 test splits; 26 epochs in 31 min (71.1s/ep) |
| #2335 slice_num=32 + surf_weight=5 on n_head=2+Lion+MAE+lr=1e-4 | 48.57 | 41.48 | −2.59% val / −1.68% test vs #2218; synergistic: observed −2.54 val vs additive −1.45; 3/4 test splits improve; 22 epochs |
| #2218 slice_num=32 on n_head=2+Lion+MAE+lr=1e-4 | 49.86 | 42.19 | −2.06% val / −3.40% test vs #2210 baseline; wins all 4 test splits; 23 epochs in budget |
| #2210 sw=5 on n_head=2+Lion+MAE+lr=1e-4 | 50.91 | 43.68 | −0.39% val / −1.13% test vs n_head=2 baseline; wins 2/4 test splits (single_in_dist −2.81, re_rand −0.91) |
| #2069 n_head=2 on Lion+MAE+lr=1e-4 | 51.11 | 44.18 | −7.76% val / −7.78% test vs lr=2e-4 baseline; wins all 4 test splits; 20 epochs in budget |
| #1932 Lion lr=2e-4 (wd=1e-4) on Lion+MAE | 55.41 | 47.90 | −2.06% val / −1.88% test vs MAE baseline; wins 3/4 test splits |
| #1825 MAE (L1) loss on Lion+EMA | 56.58 | 48.82 | −7.71% val / −7.34% test vs Lion baseline; wins all 4 test splits |
| #1781 Lion optimizer lr=1e-4 | 61.30 | 52.68 | −20.4% val / −22.8% test vs EMA-0.99 baseline; uniform 20–28% across all 4 test splits |
| #1607 EMA decay=0.99 | 77.05 | 68.27 | −22.1% val / −23.1% test vs prior best; uniform gains all 4 splits |
| #1367 Dropout=0.2 + clip_grad=1.0 | 98.96 | 88.74 | dropout standalone on BF16; merged into Fourier+Huber base |
| #1357 Huber δ=1.0 (on Fourier base) | 98.79 | 88.90 | -4.31% val, -2.13% test vs Fourier baseline |
| #1386 Fourier pos encoding L=6 mf32 BF16 | 103.24 | 90.83 | All 4 test splits improve |
| #1541 Scoring fix + BF16 rerun | 120.40 | 106.67 | All 4 test splits now finite |

### Note on test_avg/mae_surf_p — now FIXED
`data/scoring.py` now has a `torch.where(isfinite(...))` guard preventing `0×inf=NaN` from poisoning the cruise split. Merged in PR #1541. All `test_avg/mae_surf_p` values from here forward are full 4-split averages.

Whenever a PR improves on the current best, update this table in the same commit that runs `senpai:merge-winner`.

---

## 2026-05-13 16:00 — PR #2338: n_head=1 on slice_num=32+n_head=2 compound (edward)

- **val_avg/mae_surf_p (best epoch 26):** 46.672 — **−3.90% vs #2335 baseline (48.573)**
- **test_avg/mae_surf_p:** 40.687 — **−1.92% vs #2335 baseline (41.483)**
- **Per-test-split:** single_in_dist=43.50 (−8.2% vs #2218), geom_camber_rc=55.79 (−0.4%), geom_camber_cruise=24.58 (−4.3%), re_rand=38.88 (−6.5%) — **all 4 splits improve vs #2218**
- **Epochs completed:** 26 in ~31 min (71.1s/ep; 8.7% faster than n_head=2 at 78s/ep)
- **W&B run:** `g71iu8ae`
- **Compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=1** + slice_num=32 + surf_weight=10
- **Reproduce:** `cd "target/" && python train.py --n_head 1 --slice_num 32 --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-edward --wandb_name "willowpai2g24h5-edward/n-head-1-slice32-lion-mae" --wandb_group "willow-pai2g-24h-r5-n-head-1"`

**Key change:** n_head 2 → 1. Per-head dim doubles from 64 → 128. Monotonic trend confirmed all the way to n_head=1: 1 < 2 < 4 < 8. Single global attention head with full 128-dim capacity dominates multi-head diversity for physics-aware slice attention on this task. Speed dividend: 71.1s/ep vs 82s/ep (n_head=2) → 26 epochs vs 22 in 30 min. val still descending at cap (26th epoch was best). Note: runs with sw=10 (default); sw=5 interaction untested on n_head=1 (assigned to alphonse in #2416).

---

## 2026-05-13 15:40 — PR #2335: slice_num=32 + surf_weight=5 interaction on n_head=2 compound (alphonse)

- **val_avg/mae_surf_p (best epoch 22):** 48.573 — **−2.59% vs #2218 baseline (49.864)**
- **test_avg/mae_surf_p:** 41.483 — **−1.68% vs #2218 baseline (42.187)**
- **Per-test-split:** single_in_dist=47.41 (+4.30% ⚠ regress), geom_camber_rc=54.56 (−2.64% ✓), geom_camber_cruise=24.63 (−4.10% ✓), re_rand=39.33 (−5.39% ✓) — **3/4 splits improve**
- **Epochs completed:** 22 in ~30 min (82.07s/ep); val still descending at cap — NOT converged
- **VRAM:** ~81.1 GB peak
- **W&B run:** `k5262fzu`
- **Compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **slice_num=32** + **surf_weight=5**
- **Reproduce:** `cd "target/" && python train.py --n_head 2 --slice_num 32 --surf_weight 5 --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-alphonse --wandb_name "willowpai2g24h5-alphonse/slice32-sw5-n2-lion-mae" --wandb_group "willow-pai2g-24h-r5-slice32-sw5"`

**Key change:** Stack slice_num=32 (#2218) and surf_weight=5 (#2210) together. Interaction is **synergistic on val** (observed −2.54 vs additive −1.45; 1.75× predicted) and roughly additive on test_avg (−2.70 vs predicted −2.49). OOD camber + re_rand splits all gain strongly; single_in_dist regresses ~2 pts vs #2218-alone (coarser slices + softer surface emphasis removes guidance the high-magnitude in-dist single-foil split needed). Net test_avg still improves clearly.

---

## 2026-05-13 13:50 — PR #2218: slice_num=32 on n_head=2 compound (alphonse)

- **val_avg/mae_surf_p (best epoch 23):** 49.864 — **−2.06% vs #2210 baseline (50.91)**
- **test_avg/mae_surf_p:** 42.187 — **−3.40% vs #2210 baseline (43.68)**
- **Per-test-split:** single_in_dist=45.46 (−0.96 vs baseline 46.42), geom_camber_rc=56.04 (−2.56 vs baseline 58.60), geom_camber_cruise=25.68 (−1.65 vs baseline 27.33), re_rand=41.57 (−0.82 vs baseline 42.39) — **all 4 splits improve**
- **Epochs completed:** 23 in ~30 min (vs 20 at slice_num=64 — 13% more epochs, 81.4s/ep vs 93.5s/ep); val still descending at cap — NOT converged
- **W&B run:** `8qjqtb70`
- **Compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **slice_num=32** + surf_weight=10
- **Note:** surf_weight=5 from #2210 is NOT included here (run used default sw=10). Interaction slice_num=32 × sw=5 is unexplored.
- **Reproduce:** `cd "target/" && python train.py --n_head 2 --slice_num 32 --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-alphonse --wandb_name "willowpai2g24h5-alphonse/slice32-n2-lion-mae" --wandb_group "willow-pai2g-24h-r5-slice-num"`

**Key change:** slice_num 64 → 32. Coarser spatial abstraction at n_head=2 (per-head dim=64) concentrates slice capacity over fewer but richer tokens. Crucially, slice_num=32 is 13% faster per epoch, yielding 23 epochs in the same 30-min wall-clock (vs 20 at slice_num=64). Wins all 4 test splits including OOD camber splits (−1.40 rc, −1.06 cruise). slice_num=128 regresses +5.06/+4.60 val/test and is slower (114.8s/ep, 16 epochs). Monotonic signal: 32 < 64 < 128.

---

## 2026-05-13 12:27 — PR #2210: surf_weight=5 on n_head=2 compound (nezuko)

- **val_avg/mae_surf_p (best epoch 20):** 50.9119 — **−0.39% vs n_head=2 baseline (51.11)**
- **test_avg/mae_surf_p:** 43.6823 — **−1.13% vs n_head=2 baseline (44.18)**
- **Per-test-split:** single_in_dist=46.42 (−2.81 vs baseline), geom_camber_rc=58.60 (+1.16), geom_camber_cruise=27.33 (+0.59), re_rand=42.39 (−0.91)
- **Epochs completed:** 20 in ~30.9 min; val still descending at cap — NOT converged
- **W&B run:** `qkyx47iv`
- **Compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **surf_weight=5**
- **Reproduce:** `cd "target/" && python train.py --n_head 2 --surf_weight 5 --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-nezuko --wandb_name "willowpai2g24h5-nezuko/sw5-n-head-2-lion-mae" --wandb_group "willow-pai2g-24h-r5-surf-weight-n2"`

**Key change:** surf_weight 10 → 5. MAE loss's uniform per-node weighting reduces the relative importance of an explicit surface emphasis; sw=5 better matches the MAE regime. Non-monotonic response in [5,10]: sw=5 (50.91) < sw=10 (51.11) < sw=7 (52.02) — sw=7 is a local maximum, not a linear interpolation. Majority of gain on in-distribution splits; marginal losses on camber-OOD splits. Both arms reached 20 epochs; val still descending at cap.

---

## 2026-05-13 11:09 — PR #2069: n_head=2 on Lion+MAE+EMA compound (alphonse)

- **val_avg/mae_surf_p (best epoch 20):** 51.1069 — **−7.76% vs lr=2e-4 baseline (55.41)**
- **test_avg/mae_surf_p:** 44.1776 — **−7.78% vs lr=2e-4 baseline (47.90)**
- **Per-val-split:** single_in_dist=N/A (val traj: ep17=55.30, ep18=54.02, ep19=52.37, ep20=51.11 — still descending at cap)
- **Per-test-split:** single_in_dist=49.23, geom_camber_rc=57.44, geom_camber_cruise=26.74, re_rand=43.30
- **Epochs completed:** 20 in ~30 min; val still descending (−1.3/epoch) at cap — NOT converged
- **Per-epoch time:** ~93.5 s/epoch (vs ~110 s/epoch at n_head=4 — architectural change unlocks faster epochs)
- **W&B run:** `2lo9mn88`
- **Compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=2**
- **Reproduce:** `cd "target/" && python train.py --n_head 2 --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-alphonse --wandb_name "willowpai2g24h5-alphonse/n-head-2-lion-mae" --wandb_group "willow-pai2g-24h-r5-n-head"`

**Key change:** n_head 4 → 2. At n_hidden=128 with slice_num=64, per-head dimension doubles from 32 → 64, giving each head sufficient capacity to model slice-vs-slice relationships. n_head=8 (per-head dim=16) regressed monotonically, confirming head-undersizing was the bottleneck. Note: ran at lr=1e-4 (the MAE-baseline lr), not lr=2e-4 — the lr × n_head interaction remains to be explored. Val still descending steeply at cap.

---

## 2026-05-13 08:30 — PR #1932: Lion lr=2e-4 scaling on Lion+MAE compound (thorfinn)

- **val_avg/mae_surf_p (best epoch 16):** 55.4117 — **−2.06% vs MAE baseline (56.577)**
- **test_avg/mae_surf_p:** 47.8993 — **−1.88% vs MAE baseline (48.817)**
- **Per-val-split:** single_in_dist=61.41, geom_camber_rc=67.37, geom_camber_cruise=37.14, re_rand=55.73
- **Per-test-split:** single_in_dist=51.084 (−4.85%), geom_camber_rc=62.288 (−1.49%), geom_camber_cruise=31.211 (+1.30%), re_rand=47.014 (−1.10%)
- **Epochs completed:** 16 in ~30.5 min; val still descending steeply at cap (last 4 epochs: 61.27→59.07→57.82→55.41, ≈−2 pts/epoch)
- **W&B run:** `y8oh2gxf`
- **Reproduce:** `cd "target/" && python train.py --optimizer lion --lr 2e-4 --weight_decay 1e-4 --loss_type mae --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-thorfinn --wandb_name "willowpai2g24h5-thorfinn/lion-lr2e-4-wd1e-4-mae" --wandb_group "willow-pai2g-24h-r5-lion-lr"`

**Key change:** Lion lr 1e-4 → 2e-4, wd unchanged at 1e-4. Arm 2 (canonical wd=5e-4 scaling) regressed slightly (+0.4% val) — EMA+dropout already provide enough regularization; additional wd over-constrains. Winning arm confirms the lr-doubling trend holds for the third octave (5e-5→1e-4→2e-4) without saturation. Val curve still steep at cap — model is NOT converged.

**Compound:** Fourier(L=6, mf32) + MAE loss + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=2e-4, wd=1e-4)

---

## 2026-05-13 06:35 — PR #1825: MAE (L1) loss on Lion+EMA base (askeladd)

- **val_avg/mae_surf_p (best epoch 16):** 56.577 — **−7.71% vs Lion baseline (61.302)**
- **test_avg/mae_surf_p:** 48.817 — **−7.34% vs Lion baseline (52.682)**
- **Per-val-split:** single_in_dist=60.859, geom_camber_rc=71.849, geom_camber_cruise=36.842, re_rand=56.757
- **Per-test-split:** single_in_dist=53.687 (−10.24%), geom_camber_rc=63.234 (−2.09%), geom_camber_cruise=30.812 (−12.32%), re_rand=47.535 (−7.14%)
- **Epochs completed:** 16 in ~30.7 min; val still descending at cap
- **W&B run:** `03w5fnvm`
- **Reproduce:** `cd "target/" && python train.py --loss_type mae --optimizer lion --lr 1e-4 --weight_decay 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-askeladd --wandb_name "willowpai2g24h5-askeladd/mae-loss-lion-ema" --wandb_group "willow-pai2g-24h-r5-loss-type"`

**Key change:** Replace Huber(δ=1.0) loss with MAE/L1 loss (`F.l1_loss`). MAE weights every node's residual linearly and uniformly, directly matching the evaluation metric (`mae_surf_p`). On Lion base the gain is *larger* than on AdamW base (−7.71% vs −3.15%), because Lion's sign-magnitude update already removes per-parameter gradient scale, so L1's per-node uniform aggregation compounds cleanly rather than being partially absorbed by AdamW's second-moment normalization.

**Why bigger on Lion:** Huber's quadratic well around zero down-weights small-residual nodes in the *scalar loss* before backprop starts. L1 gives all nodes equal weight. This property is independent of optimizer choice — it operates at the loss aggregation level. Lion + L1 together produce the largest uniform gains: single_in_dist (−10.24%) and cruise (−12.32%) are the biggest movers.

---

## 2026-05-13 05:10 — PR #1781: Lion optimizer (thorfinn)

- **val_avg/mae_surf_p (best epoch 16):** 61.3017 — **−20.44% vs prior best (77.054)**
- **test_avg/mae_surf_p:** 52.6824 — **−22.83% vs prior best (68.265)**
- **Per-val-split:** single_in_dist=71.675 (test: 59.813), geom_camber_rc(test: 64.584), geom_camber_cruise(test: 35.140), re_rand(test: 51.193)
- **Per-test-split:** single_in_dist=59.813, geom_camber_rc=64.584, geom_camber_cruise=35.140, re_rand=51.193
- **Epochs completed:** 16 in ~31 min; val still descending steeply at cap (not converged)
- **W&B runs:** `e2l23xny` (winning arm lr=1e-4), `9fjjfgjt` (lr=5e-5, val=64.01)
- **Reproduce (best arm):** `cd "target/" && python train.py --optimizer lion --lr 1e-4 --dropout 0.2 --ema_decay 0.99 --agent willowpai2g24h5-thorfinn --wandb_name "willowpai2g24h5-thorfinn/lion-lr1e-4-ema" --wandb_group "willow-pai2g-24h-r5-lion"`

**Key change:** Replace AdamW with Lion optimizer (sign-based momentum update). Canonical Lion: `c_t = β1·m_{t-1} + (1-β1)·g_t; update = sign(c_t); m_t = β2·m_{t-1} + (1-β2)·g_t` with β1=0.9, β2=0.99, lr=1e-4. Student identified and fixed β1/β2 swap in original PR diff before running. Lion+EMA decouples exploration (Lion's noisy sign-magnitude steps) from smoothing (EMA averaging), yielding a substantially larger gain than AdamW+EMA — uniform 20–28% across all 4 test splits.

**Note:** val/test both still descending at epoch-16 cap. Longer budget (100 epochs, higher timeout) is the highest-EV immediate follow-up.

---

## 2026-05-13 01:15 — PR #1607: EMA weight averaging decay=0.99 (edward)

- **val_avg/mae_surf_p (best epoch 16):** 77.0540 — **−22.1% vs prior best (98.96)**
- **test_avg/mae_surf_p:** 68.2650 — **−23.1% vs prior best (88.74)**
- **Per-val-split (EMA model, epoch 16):** single_in_dist=85.45, geom_camber_rc=88.60, geom_camber_cruise=57.80, re_rand=76.36
- **Per-test-split (EMA model):** single_in_dist=75.31, geom_camber_rc=80.81, geom_camber_cruise=48.52, re_rand=68.41
- **Epochs completed:** 16 in ~30 min; val still descending at cap
- **W&B run:** `nl3llszv`
- **Reproduce:** `cd "target/" && python train.py --ema_decay 0.99 --dropout 0.2 --huber_delta 1.0 --agent willowpai2g24h5-edward --wandb_name "willowpai2g24h5-edward/ema-0.99-fourier" --wandb_group "willow-pai2g-24h-r5-ema"`

**Key change:** Exponential moving average over model weights (`ema_decay=0.99`). After each optimizer step, `ema_p ← 0.99×ema_p + 0.01×p`. Val/test scoring uses the EMA model exclusively; the main model val tracked under `main_*` prefix. Checkpoint saves EMA weights. EMA does all the work: main val at epoch 16 is ~100 (matching old baseline), EMA val is 77.05 — the 23-point gap is pure averaging benefit.

**Note:** Student ran on default `dropout=0.1` (current default). To reproduce with the full Fourier+Huber+Dropout+EMA compound, add `--dropout 0.2`.

---

## 2026-05-12 23:56 — PR #1367: Dropout=0.2 + grad-clip=1.0 (fern)

- **val_avg/mae_surf_p (best epoch 18):** 98.9622
- **test_avg/mae_surf_p:** 88.7390
- **Per-test-split:** single_in_dist=110.77, geom_camber_rc=97.23, geom_camber_cruise=58.81, re_rand=88.14
- **Per-val-split:** single_in_dist=121.99, geom_camber_rc=107.52, geom_camber_cruise=70.70, re_rand=95.63
- **Epochs completed:** 18 in ~31 min (cap-bound); val still descending
- **W&B run:** `otwlgvo7`
- **Reproduce:** `cd "target/" && python train.py --dropout 0.2 --agent willowpai2g24h5-fern --wandb_name "willowpai2g24h5-fern/dropout-0.2-bf16-clean" --wandb_group "willow-pai2g-24h-r5-regularization"`

**Key change:** Add `dropout=0.2` to Transolver attention (`nn.Dropout(dropout)` after attention output and at FFN), and add `clip_grad_norm_(model.parameters(), 1.0)` in training step. PR default is `dropout: float = 0.1` — **use `--dropout 0.2` to reproduce the winning arm**.

**Note:** Student's run `otwlgvo7` had `fun_dim=22` in W&B config (pre-Fourier, pre-Huber base). Reported val=98.96 was achieved with dropout-only. The squash-merge applied dropout to the current Fourier+Huber base, producing a Fourier+Huber+Dropout codebase. Actual compound performance to be verified by next baseline run.

---

## 2026-05-12 23:55 — PR #1357: Huber loss δ=1.0 (askeladd)

- **val_avg/mae_surf_p (best epoch 18):** 98.7905
- **test_avg/mae_surf_p:** 88.8965
- **Per-test-split:** single_in_dist=103.88, geom_camber_rc=96.54, geom_camber_cruise=66.61, re_rand=88.55
- **Per-val-split:** single_in_dist=117.40, geom_camber_rc=107.57, geom_camber_cruise=76.87, re_rand=93.32
- **Epochs completed:** 18 in ~30.75 min; peak VRAM 90.7 GB / 96 GB (high — possible allocator artifact, worth investigating later)
- **W&B run:** `m733u17z`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-askeladd --wandb_name "willowpai2g24h5-askeladd/huber-delta-1-bf16" --wandb_group "willow-pai2g-24h-r5-huber-loss"`

**Key change:** Replace MSE with `F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)` in both training loop and `evaluate_split`. `huber_delta` default = 1.0 (1σ in normalized space). Per-split improvements concentrate on high-Re tail (re_rand, single_in_dist) as the hypothesis predicted.

**Note:** Student's reported run `m733u17z` shows `fun_dim=22` in W&B config (pre-Fourier dim), so the val=98.79 number itself was from a Huber-alone run on the pre-Fourier base. The squash-merge applied Huber on top of the current Fourier base — the merged code is the Fourier+Huber compound. Compound performance to be verified by next clean baseline run.

---

## 2026-05-12 23:05 — PR #1386: Fourier positional encoding L=6 mf32 BF16 (nezuko)

- **val_avg/mae_surf_p (best epoch 18):** 103.2393
- **test_avg/mae_surf_p:** 90.828
- **Per-test-split:** single_in_dist=105.79, geom_camber_rc=102.99, geom_camber_cruise=64.21, re_rand=90.31
- **Epochs completed:** 18 in 30 min; peak VRAM ~33 GB / 96 GB
- **W&B run:** `bpbykd9z` (L=6 primary), `qwmh06uh` (L=4 secondary)
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-nezuko --wandb_name "willowpai2g24h5-nezuko/fourier-L6-mf32-bf16" --wandb_group "willow-pai2g-24h-r5-fourier-pos"`

**Key change:** Replace raw (x,z) coords with Fourier features — L=6 log-spaced frequencies, min_freq=1.0, max_freq=32.0, positions standardized before encoding. `fun_dim` updated from `X_DIM-2` to `X_DIM-2+4*L`. L=6 beats L=4 by ~3.9 points on test (-14.8% vs -11.2% vs baseline).

---

## 2026-05-12 21:00 — PR #1541: Scoring fix + BF16 rerun (frieren)

- **val_avg/mae_surf_p (best epoch 17):** 120.40
- **test_avg/mae_surf_p:** 106.67
- **Per-val-split:** (not reported per-split by student; best epoch is 17)
- **Per-test-split:** single_in_dist=125.29, geom_camber_rc=113.23, geom_camber_cruise=81.16, re_rand=106.99
- **Epochs completed:** 18 in 30 min (~101 s/epoch); peak VRAM ~33 GB / 96 GB
- **W&B run:** `x7snuii5`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-frieren --wandb_name "willowpai2g24h5-frieren/baseline-bf16-scoring-fix" --wandb_group "willow-pai2g-24h-r5-baseline"`

**Key change:** One-line guard in `data/scoring.py::accumulate_batch`:
```python
err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))  # guard 0×inf=NaN
```

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **val_avg/mae_surf_p (best epoch 13):** 123.72
- **Per-val-split:** single_in_dist=153.36, geom_camber_rc=129.40, geom_camber_cruise=99.23, re_rand=112.87
- **test_avg/mae_surf_p:** NaN (cruise data bug); 3-split partial=121.90
- **Epochs completed:** 18 in 30 min (~101 s/epoch); peak VRAM 32.9 GB / 96 GB
- **W&B run:** `6zx5vuja`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-frieren --wandb_name "run_name" --wandb_group "willow-pai2g-24h-r5-amp"`
