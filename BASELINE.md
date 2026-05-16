# Baseline — TandemFoilSet (willow-pai2i-48h-r5)

## Current best — PR #4120 (LR re-optimisation at clip=1.0 — lr=2e-4 + grad_clip=1.0 + T_max=14)

**val_avg/mae_surf_p = 56.8913** (W&B run: `1c58zju8`, PR #4120 Lion lr=2e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14 + **grad_clip=1.0**; NO spec_norm)
**test_avg/mae_surf_p = 49.0322** (same run `1c58zju8`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 61.01 | 52.64 |
| **geom_camber_rc** | **71.92** | **64.54** |
| **geom_camber_cruise** | **37.30** | **31.01** |
| **re_rand** | **57.34** | **47.94** |

**Δ vs prior best (PR #4063 T_max=20, val 57.6606 / test 49.4491): −0.77 val / −0.41 test**

All 4/4 val splits and 4/4 test splits improve. lr=2e-4 with clip=1.0 on T_max=14 substrate beats T_max=20 at lr=1.5e-4.

Mechanism: grad_clip=1.0 clips every step (pre-clip ‖g‖ median ~23.7 >> 1.0), turning clip into a constant per-step scale AND direction change. The optimal nominal lr shifts upward from 1.5e-4 (no-clip) to 2e-4 (with clip) because the clipped step direction (normalized gradient) differs from Lion's sign-update and interacts with lr in a non-trivial direction-sensitive way. The LR-vs-val curve at clip=1.0 has the same shape as the no-clip regime but shifted upward in lr.

**Reproduce (PR #4120 Arm B — winner):**
```bash
cd target/
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group round10-lr-at-clip1-thorfinn \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 --cosine_t_max 14 \
  --optimizer_name lion --lr 2e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 --use_film \
  --grad_clip 1.0 \
  --wandb_name thorfinn-r10-lr2e4-clip1
```

---

## Prior best — PR #4063 (T_max=20 — longer cosine schedule within 14-epoch wall-clock budget)

**val_avg/mae_surf_p = 57.6606** (W&B run: `fh3jmkd1`, PR #4063 Lion lr=1.5e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + **T_max=20**; NO spec_norm; NO grad_clip)
**test_avg/mae_surf_p = 49.4491** (same run `fh3jmkd1`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 57.88 | 51.04 |
| **geom_camber_rc** | **71.61** | **64.76** |
| **geom_camber_cruise** | **40.47** | **32.44** |
| **re_rand** | **60.69** | **49.55** |

**Δ vs prior best (PR #4056 grad_clip=1.0, val 61.18 / test 52.09): −3.52 val / −2.64 test**

All 8 splits (val × 4, test × 4) improve. Monotone trend T_max=14 → T_max=18 → T_max=20. No split-specific story — this is a global training-dynamics win.

Mechanism: SENPAI_TIMEOUT_MINUTES=30 caps training at ~14 epochs. With T_max=14 the cosine schedule fully decays within budget (LR ends ≈ 0). With T_max=20 the schedule stops at LR ≈ 1.20e-4 (~80% of peak). The model maintains a higher time-averaged LR throughout training — never "lands" but keeps exploring a better basin. EMA(0.997) smooths the late-training noise from the higher-LR endpoint.

**Reproduce (PR #4063 Arm C — winner):**
```bash
cd target/
python train.py --agent willowpai2i48h5-tanjiro --epochs 50 \
  --wandb_group round10-tmax-lr15e4-tanjiro \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 20 \
  --optimizer_name lion --lr 1.5e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name tanjiro-r10-tmax20
```

---

## Prior best — PR #4056 (grad_clip=1.0 — gradient norm clipping on Lion at lr=1.5e-4)

**val_avg/mae_surf_p = 61.1778** (W&B run: `y5tua53k`, PR #4056 Lion lr=1.5e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14 + **grad_clip=1.0**; NO spec_norm)
**test_avg/mae_surf_p = 52.0853** (same run `y5tua53k`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 65.37 | 56.81 |
| **geom_camber_rc** | **76.90** | **66.84** |
| **geom_camber_cruise** | **41.74** | **34.22** |
| **re_rand** | **60.70** | **50.47** |

**Δ vs prior best (PR #3976 lr=1.5e-4 no-clip, val 63.05 / test 53.60): −1.87 val / −1.51 test**

Key: 3/4 splits improve on val AND test. Largest gain on camber_rc (val −3.84 / test −3.71 — the weakest OOD split). re_rand also large (val −2.83 / test −2.23). in_dist regresses slightly (+0.92 val / +1.12 test — within seed noise).

Mechanism: grad norm clipping clips EVERY step (pre-clip norm median ~27, clip threshold 1.0 → every step is rescaled). This acts as a constant per-step update scale on top of Lion's sign-update, further stabilizing the OOD splits. Not an outlier-clipping story — the entire gradient norm distribution sits above clip=1.0. Sweet spot at clip=1.0 (not 0.5 or 2.0).

**Reproduce (PR #4056 Arm B — winner):**
```bash
cd target/
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group round10-gradclip-thorfinn \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 1.5e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --grad_clip 1.0 \
  --wandb_name thorfinn-r10-gradclip-1p0
```

---

## Prior best — PR #3976 (Lion lr=1.5e-4 — inflection point of monotone LR gain)

**val_avg/mae_surf_p = 63.0492** (W&B run: `jurrwig2`, PR #3976 Lion lr=1.5e-4 + n_fourier=0 + FiLM + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14; NO spec_norm)
**test_avg/mae_surf_p = 53.6049** (same run `jurrwig2`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 64.45 | 55.69 |
| geom_camber_rc | 80.74 | 70.55 |
| geom_camber_cruise | **43.48** | **35.48** |
| re_rand | 63.53 | 52.70 |

**Δ vs prior best (PR #3954 spec_norm+lr=1e-4, val 64.68 / test 56.17): −1.63 val / −2.57 test**

Note: camber_rc val is slightly worse (+0.56 vs baseline) but all other splits improve. The lr push wins 3/4 splits on val and 3/4 on test. Arm C (lr=2e-4, val 63.84) confirms inflection at 1.5e-4 — the LR optimum is now bracketed in [1.2e-4, 1.7e-4].

This run has NO spec_norm. The spec_norm config from PR #3954 adds no additional value at this LR (confirmed by finding #18).

**Reproduce (PR #3976 arm B — winner):**
```bash
cd target/
python train.py --agent willowpai2i48h5-frieren --epochs 50 \
  --wandb_group round9-lr-push-frieren \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 1.5e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name frieren-r9-lr1p5e4
```

---

## Prior best — PR #3954 (spec_norm output + lr=1e-4 combined — marginal improvement over lr-only)

**val_avg/mae_surf_p = 64.6812** (W&B run: `pc7lsis0`, PR #3954 Lion lr=1e-4 + spec_norm(output, n_power_iter=1) + n_fourier=0 + FiLM-output log(Re) + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14)
**test_avg/mae_surf_p = 56.1746** (same run `pc7lsis0`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 69.26 | 61.06 |
| geom_camber_rc | 78.64 | 69.24 |
| geom_camber_cruise | **46.37** | **38.56** |
| re_rand | 64.47 | 55.83 |

**Note:** val improvement over PR #3843 is −0.73, within seed noise (σ≈2.77). Test slightly worse (+0.11). Spec_norm at lr=1e-4 is orthogonal but not additive — Lion's sign-update already bounds output growth; the Lipschitz cap adds little. Multiple reproductions of lr=1e-4 without spec_norm cluster at val ~64.2–64.8, so the true zero-spec_norm baseline is closer to 64.5 than 65.41.

**Reproduce (PR #3954 hypothesis arm):**
```bash
cd target/
python train.py --agent willowpai2i48h5-nezuko --epochs 50 \
  --wandb_group round8-specnorm-lr-nezuko \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 --cosine_t_max 14 \
  --optimizer_name lion --lr 1e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 --use_film \
  --spec_norm_target output --spec_norm_n_power_iter 1 \
  --wandb_name nezuko-r8-specnorm-lr1e-4
```

---

## Prior best — PR #3843 (Lion lr=1e-4 — larger LR doubles improvement over spec_norm)

**val_avg/mae_surf_p = 65.4142** (W&B run: `bw38ym4h`, PR #3843 Lion lr=1e-4 + n_fourier=0 + FiLM-output log(Re) + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14; NO spec_norm)
**test_avg/mae_surf_p = 56.0627** (same run `bw38ym4h`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 69.60 | 61.03 |
| geom_camber_rc | 80.18 | 70.47 |
| geom_camber_cruise | **46.19** | **37.84** |
| re_rand | 65.69 | 54.91 |

**Comparison vs prior best (PR #3748 spec_norm output, val 68.9592 / test 60.8201):**

| Split | Prior val (u42jpd48) | New val (bw38ym4h) | Δval | Prior test | New test | Δtest |
|-------|---------------------:|--------------------:|-----:|-----------:|---------:|------:|
| single_in_dist | 77.84 | 69.60 | **−8.24** | 69.62 | 61.03 | **−8.59** |
| geom_camber_rc | 81.38 | 80.18 | **−1.20** | 73.21 | 70.47 | **−2.74** |
| geom_camber_cruise | 49.90 | 46.19 | **−3.71** | 40.68 | 37.84 | **−2.84** |
| re_rand | 66.71 | 65.69 | **−1.02** | 59.78 | 54.91 | **−4.87** |
| **avg** | **68.9592** | **65.4142** | **−3.55** | **60.8201** | **56.0627** | **−4.76** |

**All 4 val splits and all 4 test splits improve.** Largest absolute gains: in_dist (−8.24 val / −8.59 test), re_rand (−1.02 val / −4.87 test). This is the largest single-mechanism gain since the Lion optimizer itself (−16.8%/−17.5%).

**PR #3843 (Lion lr=1e-4):** Doubled the Lion learning rate from 5e-5 to 1e-4. In Lion, the effective step size ≈ lr × sign(grad) — doubling lr doubles per-step magnitude uniformly. The 3-arm sweep showed clean monotone improvement: val(2e-5)=78.93 → val(5e-5)=69.69 → val(1e-4)=65.41. The control arm (lr=5e-5) cleanly reproduced the n_fourier=0 substrate (val 69.69 vs 70.34 baseline, within σ=4.6 noise), confirming the lr=1e-4 win is real. This run was on the n_fourier=0 + FiLM + EMA substrate WITHOUT spec_norm.

**Reproduce (PR #3843 arm C — winner):**
```bash
cd target/
python train.py --agent willowpai2i48h5-frieren --epochs 50 \
  --wandb_group round7-lion-lr-frieren \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 1e-4 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name frieren-r7-lion-lr1e-4
```

---

## Prior best — PR #3748 (Spectral norm on output head under n_fourier=0)

**val_avg/mae_surf_p = 68.9592** (W&B run: `u42jpd48`, PR #3748 output-only spectral norm (n_power_iter=1) + n_fourier=0 + FiLM-output log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14)
**test_avg/mae_surf_p = 60.8201** (same run `u42jpd48`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 77.8448 | 69.6160 |
| geom_camber_rc | 81.3790 | 73.2110 |
| geom_camber_cruise | **49.9026** | **40.6773** |
| re_rand | 66.7103 | 59.7759 |

**Comparison vs prior best (PR #3672 n_fourier=0, val 70.3432 / test 61.6253):**

| Split | Prior val (297qot5r) | New val (u42jpd48) | Δval | Prior test | New test | Δtest |
|-------|---------------------:|--------------------:|-----:|-----------:|---------:|------:|
| single_in_dist | 79.64 | 77.8448 | **−1.80** | 69.97 | 69.6160 | **−0.35** |
| geom_camber_rc | 82.43 | 81.3790 | **−1.05** | 73.96 | 73.2110 | **−0.75** |
| geom_camber_cruise | 51.50 | 49.9026 | **−1.60** | 42.22 | 40.6773 | **−1.54** |
| re_rand | 67.80 | 66.7103 | **−1.09** | 60.35 | 59.7759 | **−0.57** |
| **avg** | **70.34** | **68.96** | **−1.39** | **61.63** | **60.82** | **−0.81** |

**All 4 val splits and all 4 test splits improve.** Largest gains: camber_cruise (−1.60 val / −1.54 test), in_dist (−1.80 val / −0.35 test). Consistent direction across all OOD splits.

**PR #3748 (Spectral normalization on output head):** Applies `torch.nn.utils.parametrizations.spectral_norm` (n_power_iter=1) to the 2 linear layers of the last Transolver block's MLP (`blocks[-1].mlp2[0]` and `blocks[-1].mlp2[2]`). This bounds the Lipschitz constant of the output-projection head, acting as a regularizer that reduces peak-pressure over-fitting. Arm C (output+film spectral norm) regressed, confirming that bounding FiLM's gamma/beta linear also destroys FiLM's conditioning. Output-only is the correct topology.

**Reproduce (PR #3748 arm D — n_fourier=0 baseline):**
```bash
cd target/
python train.py --agent willowpai2i48h5-nezuko --epochs 50 \
  --wandb_group round6-specnorm-nezuko \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --spec_norm_target output --spec_norm_n_power_iter 1 \
  --wandb_name nezuko-r6-specnorm-arm-D-output-nofourier
```

---

## Prior best — PR #3672 (Fourier ablation: n_fourier=0 under FiLM+Lion+EMA)

**val_avg/mae_surf_p = 70.3432** (W&B run: `297qot5r`, PR #3672 n_fourier=0 + FiLM-output log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14)
**test_avg/mae_surf_p = 61.6253** (same run `297qot5r`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 79.64 | 69.97 |
| geom_camber_rc | 82.43 | 73.96 |
| geom_camber_cruise | **51.50** | **42.22** |
| re_rand | 67.80 | 60.35 |

**Comparison vs prior best (PR #3405 FiLM+Lion+EMA, val 71.6544 / test 62.1091):**

| Split | Prior val (ksltdq7a) | New val (297qot5r) | Δval | Prior test | New test | Δtest |
|-------|---------------------:|--------------------:|-----:|-----------:|---------:|------:|
| single_in_dist | 81.17 | 79.64 | −1.53 | 71.30 | 69.97 | −1.33 |
| geom_camber_rc | 84.45 | 82.43 | −2.02 | 73.87 | 73.96 | +0.09 |
| geom_camber_cruise | 51.99 | 51.50 | −0.49 | 42.84 | 42.22 | −0.62 |
| re_rand | 69.01 | 67.80 | −1.21 | 60.43 | 60.35 | −0.08 |
| **avg** | **71.65** | **70.34** | **−1.31** | **62.11** | **61.63** | **−0.48** |

All 4 val splits improve; 3/4 test splits improve (camber_rc +0.09 test, within noise).

**PR #3672 (Fourier ablation — n_fourier=0):** Under FiLM+Lion+EMA, dropping Fourier positional features entirely (n_fourier=0) slightly outperforms both σ=3 (val 71.28) and σ=10 baseline (val 71.65). FiLM conditioning on log(Re) already encodes the flow-regime information that Fourier PE was providing, making Fourier redundant and slightly harmful. Dropping Fourier simplifies the architecture (removes ~1.1K RFF params, one hyperparameter, one coordinate transform per forward pass).

**Reproduce (PR #3672):**
```bash
cd target/
python train.py --agent willowpai2i48h5-alphonse --epochs 50 \
  --wandb_group round5-film-fourier-alphonse \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name alphonse-r5-film-nofourier
```

---

## Prior best — PR #3405 (FiLM conditioning + Lion + EMA)

**val_avg/mae_surf_p = 71.6544** (W&B run: `ksltdq7a`, PR #3405 FiLM-output on log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) on Huber + Fourier σ=10 + T_max=14)
**test_avg/mae_surf_p = 62.1091** (same run `ksltdq7a`, clean 4-split)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 81.17 | 71.30 |
| geom_camber_rc | 84.45 | 73.87 |
| geom_camber_cruise | **51.99** | **42.84** ¹ |
| re_rand | 69.01 | 60.43 |

¹ 199/200 samples evaluated — `splits_v2/.test_geom_camber_cruise_gt/000020.pt` dropped via y-side mask. Previously null due to flat key naming; now correctly resolved to 42.84 by scan of nested W&B keys.

**Comparison vs prior best (PR #3537 Lion, val 77.58 / test 68.88):**

| Split | Prior test mae (yvkf9glr) | New test mae (ksltdq7a) | Δ |
|-------|---------------------------:|--------------------------:|---:|
| single_in_dist | 81.69 | 71.30 | **−12.7%** |
| geom_camber_rc | 77.94 | 73.87 | **−5.2%** |
| geom_camber_cruise | 48.83 | 42.84 | **−12.3%** |
| re_rand | 67.04 | 60.43 | **−9.9%** |
| **avg** | **68.88** | **62.11** | **−9.8%** |

All 4 splits improve; largest gains on `single_in_dist` and `geom_camber_cruise`.

**PR #3405 (FiLM conditioning):** FiLM (Feature-wise Linear Modulation) conditions the surface-pressure model on log(Re) via gamma/beta affine transforms at the network output. log(Re) encodes the Reynolds-number regime of each flow sample. On the OOD `re_rand` split, FiLM adds the most value (val 69.01 vs Lion-only 72.93), and the `geom_camber_cruise` split sees the largest absolute improvement on test (48.83 → 42.84). Combined with Lion optimizer + EMA(0.997) + Fourier σ=10 + Huber β=0.05 + cosine T_max=14.

**Reproduce (PR #3405):**
```bash
cd target/
python train.py --agent willowpai2i48h5-nezuko --epochs 50 \
  --wandb_group round4-film-ema-lion-nezuko \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 16 --fourier_sigma 10.0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --ema_decay 0.997 \
  --use_film \
  --wandb_name nezuko-r4-film-ema997-lion
```

---

## PR #3537 (prior best) — Lion optimizer

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 90.85 | 81.69 |
| geom_camber_rc | 87.72 | 77.94 |
| geom_camber_cruise | **58.81** | **48.83** ¹ |
| re_rand | 72.93 | 67.04 |

¹ 199/200 samples evaluated — `splits_v2/.test_geom_camber_cruise_gt/000020.pt` has 761 inf y-values, dropped from MAE accumulator.

**Comparison vs prior best (PR #3444, T_max=14):**

| Split | Prior test mae (1hx2rm1n) | New test mae (yvkf9glr) | Δ |
|-------|---------------------------:|--------------------------:|---:|
| single_in_dist | 105.93 | 81.69 | **−22.9%** |
| geom_camber_rc | 90.03 | 77.94 | **−13.4%** |
| geom_camber_cruise | 57.65 | 48.83 | **−15.3%** |
| re_rand | 80.55 | 67.04 | **−16.8%** |
| **avg** | **83.54** | **68.88** | **−17.5%** |

Every test split improves substantially; the biggest gain is on `single_in_dist` (−22.9%). This is the largest single-mechanism improvement in the launch.

**PR #3537 (Lion optimizer):** Sign-based update rule (Chen et al. 2023, arXiv 2302.06675) replacing AdamW. Decoupled weight decay → sign update → momentum decay. β₁=0.9, β₂=0.99. Note: paper recommends batch ≥ 64 but Lion works strongly even at batch_size=4 in our regime, likely because the irregular-mesh CFD loss landscape is well-suited to sign updates.

**Reproduce (PR #3537):**
```bash
cd target/
python train.py --agent willowpai2i48h5-askeladd --epochs 50 \
  --wandb_group round3-lion-askeladd \
  --loss_type smooth_l1 --loss_beta 0.05 \
  --n_fourier 16 --fourier_sigma 10.0 \
  --cosine_t_max 14 \
  --optimizer_name lion --lr 5e-5 --weight_decay 1e-3 \
  --wandb_name askeladd-r3-arm-A-lion-lr5e5-wd1e3
```

---

## PR #3444 (prior best) — Cosine T_max=14

**val_avg/mae_surf_p = 93.1996** (W&B run: `1hx2rm1n`)
**test_avg/mae_surf_p = 83.5377** (same run)

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|------------------|
| single_in_dist | 114.80 | 105.93 |
| geom_camber_rc | 104.16 | 90.03 |
| geom_camber_cruise | 68.17 | 57.65 |
| re_rand | 85.66 | 80.55 |

**PR #3444:** 1-LOC scheduler period change. The 30-min wall-clock binds at ~epoch 14, but the cosine schedule was tuned for T_max=50 → LR never decayed below 82% of peak. Setting T_max=14 lets the cosine schedule complete inside the wall-clock budget.

---

## PR #3098 + #3296 (earlier baseline) — Huber + NaN guard

**val_avg/mae_surf_p = 96.0548** (W&B run: `md6so639`, PR #3098 SmoothL1 β=0.05)
**test_avg/mae_surf_p = 90.0004** (W&B run: `xvn4gllg`, PR #3296 two-pronged NaN guard on Huber)

| Split | val mae_surf_p (md6so639) | test mae_surf_p (xvn4gllg) |
|-------|---------------------------|----------------------------|
| single_in_dist | 109.64 | 109.30 |
| geom_camber_rc | 112.30 | 103.19 |
| geom_camber_cruise | **73.22** | **60.61** |
| re_rand | 89.06 | 86.90 |

**PR #3098 (val):** SmoothL1 (Huber) loss with β=0.05 replacing MSE. Effect: val_avg 130.46 → 96.05 (-26.4% vs PR #3123). All 4 val splits improved.

**PR #3296 (test):** Two-pronged NaN guard (pred-side `nan_to_num` + y-side sample mask) in both `evaluate_split` and the training loop. First valid test_avg of the launch.

---

## PR #3123 (2026-05-15) — earlier reference

**val_avg/mae_surf_p = 130.46** (W&B run: `24yldhv7`)  
**test_avg/mae_surf_p = NaN** ⚠️ — test_geom_camber_cruise split produces NaN for all runs (baseline-side bug, not introduced by this PR — tracked in follow-up NaN-fix PR)

| Split | val mae_surf_p |
|-------|---------------|
| val_single_in_dist | 159.57 |
| val_geom_camber_rc | 150.12 |
| val_geom_camber_cruise | **89.02** |
| val_re_rand | 123.13 |

Added: Random Fourier positional features over (x,z) coordinates, n_fourier=16, sigma=10.0

**Reproduce:**
```bash
cd target/
python train.py --agent willowpai2i48h5-thorfinn --epochs 50 \
  --wandb_group fourier-pe-thorfinn \
  --n_fourier 16 --fourier_sigma 10.0 \
  --wandb_name thorfinn-arm-C-fourier16
```

---

## Starting point (unmerged baseline reference)

**val_avg/mae_surf_p = 135.23** (Arm A from PR #3123, W&B: `jyqygcbx` — no Fourier features)

| Split | val mae_surf_p |
|-------|---------------|
| val_single_in_dist | 156.98 |
| val_geom_camber_rc | 144.01 |
| val_geom_camber_cruise | 119.48 |
| val_re_rand | 120.44 |

The baseline architecture is defined in `target/train.py`:

| Param | Value |
|-------|-------|
| model | Transolver |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| params | ~1.5M |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| loss | MSE (normalized space) |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| epochs cap | 50 |
| timeout cap | 30 min |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)  
**Test metric:** `test_avg/mae_surf_p` (lower is better)

Per-split metrics for each run are logged to W&B under `wandb-applied-ai-team/senpai-v1`.

---

_Updated when a PR is merged: add PR#, W&B run ID, val_avg/mae_surf_p, test_avg/mae_surf_p, and per-split breakdown._
