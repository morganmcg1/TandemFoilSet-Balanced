## Hypothesis

**H46: n_head=1 (head_dim=128) — extrapolate the monotone n_head trend to its limit.**

The n_head sweep has been monotone improving:

| n_head | head_dim | val_avg/mae_surf_p | source |
|--------|----------|---------------------|--------|
| 8 | 16 | regression | (closed earlier) |
| 4 | 32 | 69.4381 | H32 baseline (lr=1e-3+clip=1.0) |
| 2 | 64 | **66.1060** | **H37b current best** |
| **1** | **128** | **???** | **this experiment** |

The Transolver block uses `head_dim = n_hidden / n_head = 128 / n_head`. As `n_head` shrinks, each head gets a wider projection and the attention operates over a richer per-head channel space. Single-head attention with head_dim=128 is the limit of this trajectory.

**Mechanism for why fewer heads might help in this regime:**
- The Transolver attention here operates on **slice tokens** (n_slice=64 tokens after slice projection), not on a long sequence. With only 64 tokens, multi-head splitting is mostly redundant — each head has too few tokens to learn distinct attention patterns.
- More channels per head (head_dim=128) gives the per-head query/key projection more representational room to discriminate between slice tokens.
- The slice projection itself is `Linear(64, 64)` per head — at head_dim=128 it becomes `Linear(128, 128)`, a 4× wider projection space.

**Risk:** Single-head attention loses ensembling benefit, may regress if the slice tokens are diverse enough to benefit from parallel attention specialization. If it regresses, n_head=2 is confirmed as the optimum and the monotone trend has hit its floor.

**Two-arm test (n_head=1 with and without wd=5e-5):**

The current best (H37b) used wd=1e-4 default. H38 showed wd=5e-5 is better at lr=1e-3. So we test both:

- **Arm A — n_head=1 + lr=1e-3 + wd=1e-4 + clip=1.0**: direct apples-to-apples extrapolation from H37b's config.
- **Arm B — n_head=1 + lr=1e-3 + wd=5e-5 + clip=1.0**: full stack with the H38 finding.

This gives us two data points to assess (1) whether n_head=1 still beats n_head=2 on the same wd, and (2) whether the wd=5e-5 stack works at the new attention config.

## Instructions

No code changes required — `--n_head` is already a CLI flag and the model handles arbitrary `n_head` dividing `n_hidden=128`.

**Arm A — n_head=1 + lr=1e-3 + wd=1e-4 + clip=1.0:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h46-nhead1-lr1e3-clip1 \
  --agent charliepai2i48h3-tanjiro \
  --n_head 1 --lr 1e-3 --clip_grad_norm 1.0
```

**Arm B — n_head=1 + lr=1e-3 + wd=5e-5 + clip=1.0:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h46-nhead1-lr1e3-wd5e5-clip1 \
  --agent charliepai2i48h3-tanjiro \
  --n_head 1 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown (val_single_in_dist, val_geom_camber_rc, val_geom_camber_cruise, val_re_rand)
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test breakdown
- Number of epochs completed before wall, best epoch
- Peak GPU memory, total parameter count (model size grows because slice projection scales with head_dim)
- Per-epoch val_avg trajectory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

## Baseline

Current best — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **66.1060** |
| val_single_in_dist/mae_surf_p | 74.3956 |
| val_geom_camber_rc/mae_surf_p | 78.9959 |
| val_geom_camber_cruise/mae_surf_p | 46.4384 |
| val_re_rand/mae_surf_p | 64.5940 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **64.4522** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 cosine + clip_grad_norm=1.0 + lr=1e-3 + **n_head=2 (head_dim=64)** + wd=1e-4 (default).

**Beat this: val_avg/mae_surf_p < 66.1060**

Reference points:
| n_head | head_dim | wd | val_avg | source |
|--------|----------|----|---------|--------|
| 4 | 32 | 1e-4 | 69.4381 | H32 |
| 4 | 32 | 5e-5 | 68.1932 | H38 (best wd, n_head=4) |
| 2 | 64 | 1e-4 | **66.1060** | H37b ← baseline |
| 1 | 128 | 1e-4 | ??? | Arm A |
| 1 | 128 | 5e-5 | ??? | Arm B |

Seed variance is ~2.3 pts (H27b 71.77 vs H32 69.44, same config). If either arm beats H37b by >2.3 pts, that's strong signal.

⚠ `test_geom_camber_cruise` NaN scoring bug — report 3-split excl. cruise.

⚠ `test_avg/mae_surf_p` will appear NaN (pre-existing scoring bug). Report 3-split excl. cruise.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h37b-nhead2-lr1e3-clip1 \
  --agent charliepai2i48h3-tanjiro \
  --n_head 2 --lr 1e-3 --clip_grad_norm 1.0
```
