## Hypothesis

**H43: Linear warmup + lr=1e-3 + clip=1.0 — probe early-training instability at the new optimal LR.**

H32 established lr=1e-3 as the new optimum (val_avg=69.4381, -2.34 vs H27b at the same nominal config, -6.06 vs H20). The LR sweep monotone trend (5e-4 → 8e-4 → 1e-3 gives 75.50 → 73.11 → 69.44) shows the LR ceiling is not yet visible.

But at lr=1e-3, the **first few training steps see large gradients** (pre-clip norms 5-17 from prior H20 logs). Clip=1.0 truncates these per-step, but the optimizer state (AdamW first/second moments) still gets initialised from those clipped-but-large early gradients. If the early gradient distribution is non-representative (e.g., dominated by high-Re samples), the optimizer state becomes biased before the cosine LR decay kicks in.

**Linear warmup** (start_factor=0.1, ramp to 1.0 linearly over N epochs) softens this:
- The effective LR during warmup is 0.1 × 1e-3 = 1e-4 at step 0, ramping linearly to 1e-3 at end of warmup
- AdamW second moment estimates accumulate over warmup with smaller updates → more stable bias correction
- Clip=1.0 still active throughout
- After warmup, switch to cosine annealing with reduced T_max to keep total schedule length constant

**Two arms test warmup length:**

- **Arm A — warmup=1 ep**: Single epoch of linear warmup, then 13 epochs of cosine (T_max=13). Minimal disruption to current schedule; tests whether even a brief warmup helps.
- **Arm B — warmup=2 ep**: Two epochs of linear warmup, then 12 epochs of cosine (T_max=12). More aggressive warmup; tests whether sustained gentle ramp helps more.

**Why this is orthogonal to clip:** Clip bounds the *gradient norm* before the optimizer step. Warmup bounds the *effective learning rate*. They act on different terms in the update equation:
```
param ← param - lr_t × clip_grad(g_t, max_norm=1.0)
```
Where `lr_t` is the scheduled LR (controlled by warmup) and `clip_grad` bounds `g_t`. Independent levers.

**Why this is orthogonal to LR-magnitude:** H39 (thorfinn) tests lr=1.5e-3/2e-3 — *higher* LR with no warmup. If H39 succeeds, warmup may not be needed. If H39 fails (instability), warmup may rescue it. Either way, this experiment gives an independent answer about early-training instability at the current optimum.

## Instructions

Two code changes required in `train.py`:

**1. Add to Config dataclass (after `clip_grad_norm` line ~404):**
```python
warmup_epochs: int = 0   # Linear warmup epochs; 0 = no warmup (default cosine only)
```

**2. Replace scheduler construction (line 457):**

Replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

With:
```python
if cfg.warmup_epochs > 0:
    # Linear warmup then cosine annealing
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15 - cfg.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.warmup_epochs]
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

This keeps the total schedule length (15 epochs) constant — warmup eats into cosine duration rather than extending training. Important so the comparison vs H32 is on equal total budget.

Then run **two arms** in sequence:

**Arm A — warmup=1 ep:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h43-warmup1-lr1e3-clip1 \
  --agent charliepai2i48h3-askeladd \
  --warmup_epochs 1 \
  --lr 1e-3 \
  --clip_grad_norm 1.0
```

**Arm B — warmup=2 ep:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h43-warmup2-lr1e3-clip1 \
  --agent charliepai2i48h3-askeladd \
  --warmup_epochs 2 \
  --lr 1e-3 \
  --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, n_head=4, slice_num=64. Merged defaults.

**Report per-arm:** val_avg/mae_surf_p, per-split breakdown, and the **per-epoch val_avg trajectory** (especially epochs 1-3 to see if warmup smooths early dynamics). Also report the per-epoch effective LR (you can compute this from the scheduler — e.g. epoch 1 should be ~1e-4 for warmup=1, epoch 1 should be ~3.3e-4 / epoch 2 should be ~6.7e-4 for warmup=2).

Compare against H32 (lr=1e-3, no warmup) per-split values below.

## Baseline

Current best — **PR #3557 — H32: lr=1e-3 + clip=1.0 (thorfinn)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **69.4381** |
| val_single_in_dist/mae_surf_p | 79.6711 |
| val_geom_camber_rc/mae_surf_p | 84.4672 |
| val_geom_camber_cruise/mae_surf_p | 47.2669 |
| val_re_rand/mae_surf_p | 66.3473 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **69.1774** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 (no warmup) + clip_grad_norm=1.0 + lr=1e-3

**Beat this: val_avg/mae_surf_p < 69.4381**

LR + schedule comparison table:
| schedule | warmup | lr | val_avg |
|----------|--------|----|---|
| cosine T_max=15 | 0 | 5e-4 | 75.4955 (H20) |
| cosine T_max=15 | 0 | 1e-3 | 69.4381 (H32) ← baseline |
| warmup 1 + cosine T_max=14 | 1 | 1e-3 | ??? (Arm A) |
| warmup 2 + cosine T_max=13 | 2 | 1e-3 | ??? (Arm B) |

Note: seed variance is ~2.3 pts (H27b 71.77 vs H32 69.44, same config). If either arm beats H32 by >2.3 pts, that's strong signal.

⚠ test_avg/mae_surf_p will appear NaN (pre-existing scoring bug). Report 3-split excl. cruise.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h32-lr1e3-clip1 \
  --agent charliepai2i48h3-askeladd \
  --lr 1e-3 --clip_grad_norm 1.0
```
