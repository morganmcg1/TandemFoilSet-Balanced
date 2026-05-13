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
| #1825 MAE (L1) loss on Lion+EMA | **56.58** | **48.82** | −7.71% val / −7.34% test vs Lion baseline; wins all 4 test splits |
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
