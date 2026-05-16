## Hypothesis

**H51: LR ceiling push — extend the monotone LR trend with lr=2.5e-3 and lr=3e-3 at the new H39 Arm C stack.**

The LR monotone trend so far: 5e-4 → 8e-4 → 1e-3 → 1.5e-3 → 2e-3 gives val_avg 75.50 → 73.11 → 69.44 → 68.12 → 66.34 (Arm B prior at n_head=4) → 63.44 (Arm C at n_head=2 + wd=5e-5). At lr=2e-3 the trend has NOT broken yet — improvements continue monotonically.

**Mechanism:** With grad clip=1.0 binding throughout training (pre-clip norms 7→1.5 across epochs), higher peak LR means the model covers more loss-landscape area during the high-LR phase of the cosine schedule. The clip bounds per-step magnitude regardless of LR — so the gain from higher LR is purely in *how the optimizer explores* during the initial high-LR epochs.

**Risk:** At lr=3e-3 or above the optimizer may overshoot — large early-epoch updates may blow up before the cosine has decayed enough. Watch for divergence at epoch 1-2.

**Two arms to find the ceiling:**

- **Arm A — lr=2.5e-3 + n_head=2 + wd=5e-5 + clip=1.0**: One step beyond the current best at 2e-3.
- **Arm B — lr=3e-3 + n_head=2 + wd=5e-5 + clip=1.0**: 1.5× the current best LR — a bigger step.

Both predict val_avg ≈ 62-65 if the monotone trend continues. If both regress vs the current 63.44, the ceiling is between 2e-3 and 2.5e-3.

## Instructions

No code changes needed. Both arms use existing CLI flags.

**Arm A — lr=2.5e-3:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h51-lr25e3-nhead2-wd5e5 \
  --agent charliepai2i48h3-alphonse \
  --n_head 2 --lr 2.5e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

**Arm B — lr=3e-3:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h51-lr3e3-nhead2-wd5e5 \
  --agent charliepai2i48h3-alphonse \
  --n_head 2 --lr 3e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- **Pre-clip gradient norms** at epochs 1, 7, 13, 15 (critical — confirm clip is still binding)
- Per-epoch val_avg trajectory (watch epochs 1-3 for divergence)
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if Arm B diverges:** if val_avg at epoch 3 exceeds 250 or grows between epoch 1 and 2, kill Arm B and report. The LR may be past the stability boundary.

## Baseline (pending merge)

**Current best — PR #3683 — H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 (thorfinn)** (in rebase, will merge shortly)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p (best of 2 seeds)** | **63.4385** |
| val_avg/mae_surf_p (2nd seed) | 65.5093 |
| test_avg/mae_surf_p (3-split, excl. cruise, best seed) | **61.3910** |
| Best epoch | 15/50 (cut by timeout) |

**Beat this: val_avg/mae_surf_p < 63.44 (best) or < 64.47 (mean of 2 seeds)**

Prior baseline (merged) — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0**:
| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 66.1060 |
| test_avg/mae_surf_p (3-split) | 64.4522 |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), AdamW (β₁=0.9, β₂=0.999).

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce H39 Arm C baseline (lr=2e-3):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h39c-nhead2-lr2e3-wd5e5-clip1 \
  --agent charliepai2i48h3-alphonse \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```
