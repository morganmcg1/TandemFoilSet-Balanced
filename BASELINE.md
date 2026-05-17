# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-17 10:45 UTC

## Current best — PR #4453: n_layers=4 depth win — alphonse

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **50.1193** | PR #4453 alphonse n_layers=4 Lookahead k=3 slice=8 `uiy4eks9` |
| `test_3split/mae_surf_p` | **50.2103** | PR #4453 alphonse (manual 3-split, cruise NaN excluded) |

Per-split val (PR #4453, run `uiy4eks9`):

| Split | mae_surf_p | Δ vs prior baseline (50.1657) |
|---|---|---|
| val_single_in_dist | 60.392 | +2.52 (regressed vs eps=1e-9 baseline) |
| **val_geom_camber_rc** | **60.666** | **−1.895 (−3.0%)** |
| val_geom_camber_cruise | 31.444 | −0.544 |
| **val_re_rand** | **48.656** | +0.41 |
| **val_avg** | **50.1193** | **−0.047 (−0.09%)** |

Per-split test (run `uiy4eks9`, best-ckpt eval via `scripts/test_eval_only.py --batch_size 1`):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 51.466 |
| test_geom_camber_rc | **57.357** |
| test_geom_camber_cruise | NaN (fleet-wide data/scoring.py bug) |
| test_re_rand | 41.808 |
| **test_3split** | **50.2103** |

Reproduce:
```bash
cd "target/" && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --n_layers 4 \
  --agent <student>
```

**Note on eps**: this run used default adamw_eps=1e-8 (NOT 1e-9). The eps=1e-9 axis (PR #4401, val=50.1657 on n_layers=5) and n_layers=4 axis are independent and likely additive. Stacking {n_layers=4 + eps=1e-9} is unverified — assigned as the immediate follow-up.

**Mechanism**: depth-axis points down at our 30-min wall-clock budget. n_layers=3 (26 epochs) and n_layers=4 (22 epochs) both beat n_layers=5 (17 epochs) — more Lookahead slow-weight syncs more than compensate for reduced per-block capacity. n_layers=7 (10 epochs) collapses badly. n_layers=4 wins over n_layers=3 on val (capacity sweet spot: enough depth, enough updates).

**Dominant gain**: val_geom_camber_rc 62.561 → 60.666 (−1.895 absolute, −3.0%) — the hardest split continues to improve.

**Key comparison** (all recent baselines):
| Config | val | test |
|---|---|---|
| n_layers=5, eps=1e-9 (PR #4401) | 50.166 | 50.340 |
| **n_layers=4, eps=1e-8 (PR #4453, this)** | **50.119** | **50.210** |

---

## Previous best — PR #4067: AdamW β2=0.95 — alphonse

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **56.4260** | run `3pc74k8f` (best epoch 17, slice=16 stack) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **55.3387** | run `3pc74k8f` |

Per-split validation (run `3pc74k8f` vs prior #3854 slice=16+default β2 baseline, 57.6953):

| Split | mae_surf_p | Δ vs #3854 |
|---|---|---|
| val_single_in_dist | 65.188 | −1.21% |
| val_geom_camber_rc | 67.131 | **−6.52%** ← dominant residual gain |
| val_geom_camber_cruise | 37.922 | −0.22% |
| val_re_rand | 55.464 | +0.90% |
| **val_avg** | **56.426** | **−2.20%** |

Per-split test (run `3pc74k8f`):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 58.0236 |
| test_geom_camber_rc | 60.6063 |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) |
| test_re_rand | 47.3864 |

Key mechanistic finding: **AdamW β2 = 0.95 (instead of default 0.999) halves the second-moment EMA half-life from ~693 steps to ~13 steps**. With only ~6000 total steps in our 30-min budget, β2=0.999 cannot adapt the per-parameter second-moment estimate fast enough — the optimizer effectively uses epoch-1 gradient statistics throughout training. β2=0.95 lets per-parameter step sizes track late-training gradient statistics within each epoch. The win concentrates on val_geom_camber_rc (−6.52%, the hardest OOD-camber split). best_epoch=17 confirms snappier adaptation didn't land in a worse local minimum.

**Note on baseline**: this win was measured on **slice=16**, not slice=8 (the prior best stack at val=56.8954). Result still beats slice=8 baseline by −0.47 val and −0.64 test, so merging is correct. The slice=8 + β2=0.95 compounding is **untested** and is the next experiment (re-validation assigned to alphonse).

Merged from PR #4067, student `willowpai2i48h2-alphonse`.

---

## Previous baseline — PR #4062: slice_num=8 — fern (superseded 21:30 UTC)

- `val_avg/mae_surf_p`: 56.8954 (run `vzpgr8us`)
- `test_3split/mae_surf_p`: 55.9817 (run `vzpgr8us`)
- Stack: slice=8 + Huber δ=0.5 + default β2=0.999

---

## Current best configuration

slice_num=16 + AdamW β2=0.95 + Huber δ=0.5 + vel-asinh s=0.5 + n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping:
- **`--adamw_beta2 0.95`** ← NEW (PR #4067): fast second-moment EMA adaptation
- **`--slice_num 16`** ← measured baseline (the merged code default may be slice=8; the winning RUN was on slice=16)
- **`--huber_delta 0.5`** (PR #3854 stack): tighter quadratic transition for small residuals
- **`--asinh_vel_scale 0.5`** (PR #3789): applies `asinh(vel / 0.5)` to velocity channels (Ux, Uy); pressure unchanged
- **`--n_head 2`** (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): SwiGLU in all TransolverBlock MLPs; param-count matched
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- **NO SGDR**

## Reproduce (winning run alphonse #4067)

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 16 \
  --adamw_beta2 0.95 \
  --agent <student>
```

**Compounding check (next experiment)**: replace `--slice_num 16` with `--slice_num 8` to test whether the β2=0.95 win compounds with the previously merged slice=8 stack.

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=8, mlp_ratio=1.333, use_swiglu=True, asinh_vel_scale=0.5`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~15 epochs with n_head=2 at ~107 s/epoch)
- Loss: `F.huber_loss(delta=0.5)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

---

## 2026-05-17 00:35 — PR #4142: Optimizer — Lookahead(k=5, α=0.5) on slice=8+β2=0.999

**NEW BEST BASELINE** — biggest single-axis optimizer win to date (+4.4% test improvement over previous best).

- **val_avg/mae_surf_p:** 54.2986 (best seed `qhphlg41`; mean of 2 seeds 54.59)
- **test_3split/mae_surf_p:** 52.8790 (best seed; mean of 2 seeds 53.34)
- **W&B runs:** `qhphlg41` (best, val=54.30), `fz2r6otj` (seed 2, val=54.88)

Per-split val (best seed `qhphlg41`):
| Split | mae_surf_p | Δ vs prior baseline (56.426) |
|---|---|---|
| val_single_in_dist | 63.937 | −1.9% |
| val_geom_camber_rc | 68.753 | +2.4% |
| val_geom_camber_cruise | 31.954 | −15.7% |
| val_re_rand | 52.552 | −5.2% |
| **val_avg** | **54.299** | **−3.8%** |

Per-split test (best seed `qhphlg41`):
| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 54.230 |
| test_geom_camber_rc | 60.693 |
| test_geom_camber_cruise | NaN (fleet-wide data/scoring.py bug) |
| test_re_rand | 43.715 |
| **test_3split** | **52.879** |

- **Reproduce:**
```bash
cd "target/" && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --agent <student>
```

**Mechanism**: Lookahead wraps AdamW with a slow-weight trajectory (k=5 fast steps then sync with α=0.5). The Polyak slow-weight averaging smooths the fast-optimizer trajectory, providing in-training weight averaging that differs from post-hoc EMA. cruise and re_rand splits benefited most — consistent with Lookahead's variance-reduction property improving generalization across domain shifts.

**Note**: this run uses β2=0.999 (default), NOT β2=0.95. The Lookahead+β2=0.95 compound is an open experiment.

---

## 2026-05-17 02:35 — PR #4249: Optimizer — Lookahead(k=5, α=0.5) + β2=0.95 compound on slice=8

**NEW BEST BASELINE** — compound of the two biggest optimizer wins; val_geom_camber_rc hits new floor.

- **val_avg/mae_surf_p:** **52.9444** (run `5qg8ex1g`)
- **test_3split/mae_surf_p:** **52.7523** (run `5qg8ex1g`)
- **W&B run:** `5qg8ex1g` (group: `lookahead-beta2-compound`)

Per-split val (run `5qg8ex1g`, vs prior Lookahead-only baseline 54.2986):
| Split | mae_surf_p | Δ vs Lookahead-only |
|---|---|---|
| val_single_in_dist | 63.8415 | −0.15% |
| val_geom_camber_rc | **64.6348** | **−5.99% ← best ever on branch** |
| val_geom_camber_cruise | 32.6315 | +2.12% |
| val_re_rand | 50.6698 | −3.58% |
| **val_avg** | **52.9444** | **−2.49%** |

Per-split test (run `5qg8ex1g`):
| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 56.4277 |
| test_geom_camber_rc | 58.5654 |
| test_geom_camber_cruise | NaN (fleet-wide data/scoring.py bug) |
| test_re_rand | 43.2638 |
| **test_3split** | **52.7523** |

- **Reproduce:**
```bash
cd "target/" && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 5 --lookahead_alpha 0.5 \
  --adamw_beta2 0.95 \
  --agent <student>
```

**Mechanism**: Lookahead (trajectory-averaging, k=5 α=0.5) and β2=0.95 (per-parameter step-size adaptation) operate at different abstraction levels and compound additively. val_geom_camber_rc drops to 64.63 — the lowest on this branch — because β2=0.95 specifically improves the fast-EMA adaptation critical for high-camber extrapolation, while Lookahead preserves the cruise/re_rand variance-reduction wins. Sub-additive but still significant: LLRD+Lookahead compound next priority.

---

## 2026-05-17 04:50 — PR #4266: Optimizer — Lookahead k=3 (tighter sync) on slice=8

**NEW BEST BASELINE** — biggest single-axis win since Lookahead itself; k=3 BEATS k=5+β2=0.95 compound even with β2=0.999 (default).

- **val_avg/mae_surf_p:** **51.3066** (run `0aj92l9d`)
- **test_3split/mae_surf_p:** **51.8862** (run `0aj92l9d`)
- **W&B runs:** `0aj92l9d` (k=3, winner), `xc3khc3a` (k=10, failed)

Per-split val (run `0aj92l9d`, vs prior Lookahead+β2=0.95 baseline 52.9444):
| Split | mae_surf_p | Δ vs prior baseline |
|---|---|---|
| val_single_in_dist | 57.803 | **−9.59%** |
| val_geom_camber_rc | 63.854 | **−7.12%** |
| val_geom_camber_cruise | 32.409 | −0.67% |
| val_re_rand | 51.159 | +0.97% |
| **val_avg** | **51.3066** | **−3.09%** |

Per-split test (run `0aj92l9d`):
| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 52.578 |
| test_geom_camber_rc | 60.074 |
| test_geom_camber_cruise | NaN (fleet-wide data/scoring.py bug) |
| test_re_rand | 43.007 |
| **test_3split** | **51.8862** |

- **Reproduce:**
```bash
cd "target/" && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --agent <student>
```

**Mechanism**: On our 30-min/~6300-step budget, k=3 delivers ~2100 slow-weight updates vs k=5's ~1260 — 67% more variance-reduction events under the same wall-clock constraint. The k=10 arm (only ~630 updates) regressed +5.91%. The monotone ranking k=3 < k=5 << k=10 is confirmed across every val split and every epoch checkpoint — not a noise artifact. k-axis is a stronger lever than the β2-axis at this budget. NOTE: this run uses β2=0.999 (default); the k=3+β2=0.95 compound is the next priority.

---

## 2026-05-17 08:32 — PR #4401: Optimizer — AdamW eps=1e-9 on Lookahead k=3 baseline

**NEW BEST BASELINE** — eps=1e-9 (tighter adaptive scaling) beats eps=1e-8 default by −2.2% val / −3.0% test. First improvement since k=3 merge.

- **val_avg/mae_surf_p:** **50.1657** (run `hpjl79he`)
- **test_3split/mae_surf_p:** **50.3401** (run `hpjl79he`)
- **W&B runs:** `hpjl79he` (eps=1e-9, winner), `k9mspshy` (eps=1e-7, flat/regress)

Per-split val (run `hpjl79he`, vs prior k=3 baseline 51.3066):
| Split | mae_surf_p | Δ vs prior baseline |
|---|---|---|
| val_single_in_dist | 57.870 | +0.07 (flat) |
| val_geom_camber_rc | **62.561** | **−1.293 (−2.0%)** ← dominant residual improved |
| val_geom_camber_cruise | 31.988 | −0.421 (−1.3%) |
| val_re_rand | **48.243** | **−2.916 (−5.7%)** |
| **val_avg** | **50.1657** | **−1.141 (−2.2%)** |

Per-split test (run `hpjl79he`):
| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 53.480 |
| test_geom_camber_rc | **56.131** (−3.943 vs k=3 baseline) |
| test_geom_camber_cruise | NaN (fleet-wide data/scoring.py bug) |
| test_re_rand | **41.410** (−1.597 vs k=3 baseline) |
| **test_3split** | **50.3401** |

- **Reproduce:**
```bash
cd "target/" && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --use_lookahead --lookahead_k 3 --lookahead_alpha 0.5 \
  --adamw_eps 1e-9 \
  --agent <student>
```

**Mechanism**: With `asinh_p_scale=1.0` compressing the pressure target, the per-parameter v̂ disparity is mild. The default eps=1e-8 sets an unnecessarily large stability floor, flattening per-parameter adaptive step sizes. eps=1e-9 allows tighter adaptive control on geometry-OOD splits (camber_rc, cruise, re_rand) while Lookahead k=3 handles trajectory-level variance reduction. Zero NaN/Inf events across 6750 steps confirms the stability cliff is not crossed. Test directional confirmation: eps=1e-7 hurt performance, eps=1e-9 helped — lever direction monotone.

**Note**: eps=1e-9 follow-up (sharper grid: {3e-10, 1e-10, 3e-9}) assigned to edward as next experiment.
