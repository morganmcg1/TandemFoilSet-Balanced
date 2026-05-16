## Hypothesis

**H47: Cosine eta_min > 0 — raise the LR floor so the last epoch is still doing real work.**

H41 just confirmed that **stretching T_max** (T_max=15 → 20) helps because at our 30-min wall budget (~14 epochs completed), the cosine schedule's late-epoch LR with T_max=15 anneals to ~4.5% of peak — the model is coasting. T_max=20 keeps last-epoch LR at ~21% of peak and improves val_avg by -2.51.

**Eta_min** is the **orthogonal lever**: instead of stretching the schedule, raise the *floor* of the cosine. `CosineAnnealingLR(T_max, eta_min)` interpolates from `lr_init` down to `eta_min` rather than down to 0. Setting `eta_min=5e-5` means the schedule never drops below 5e-5 (5% of peak), regardless of T_max.

**Why this might compound with T_max:** T_max=20 controls the *shape* of the curve (how slow it falls). Eta_min controls the *floor* (how far it falls). They act on different axes of the schedule:
- T_max=20 + eta_min=0 = current best (PR #3688 Arm A waiting on retest)
- T_max=15 + eta_min=5e-5 = this experiment, Arm A
- T_max=15 + eta_min=1e-4 = this experiment, Arm B (higher floor)

**Mechanism:** Pre-clip grad norms remain ~1.7-2.1 at the final epoch (per H39 data). The optimizer has gradient signal but a too-small effective LR to act on it. Eta_min puts a hard floor under the per-step update size, ensuring late-epoch budget is spent on real loss descent.

**Why this is different from T_max:** With T_max=15+eta_min=0, the LR at epoch 13 is ~4.5e-5. With T_max=15+eta_min=5e-5, the LR at epoch 13 is at least 5e-5. The shape of the curve is similar, but the late tail is bounded. This is closer in spirit to a "trapezoidal" or "warmup-stable-decay" schedule than a full cosine.

**Two arms test eta_min magnitude:**

- **Arm A — eta_min=5e-5** (5% of peak lr=1e-3): mild floor, mostly preserves cosine shape, only raises the last 2-3 epochs.
- **Arm B — eta_min=1e-4** (10% of peak lr=1e-3): stronger floor, model still actively descending at end of schedule.

Both stack on top of current best config (n_head=2, wd=5e-5, lr=1e-3, clip=1.0). Note this combines H37b (n_head=2) + H38 (wd=5e-5) + the new eta_min lever. Predicted val_avg if eta_min effect is real and stacks ≈ 64-65.

## Instructions

One small code change in `train.py`:

**1. Add to Config dataclass (after `clip_grad_norm` line ~404):**
```python
eta_min: float = 0.0   # CosineAnnealingLR floor; 0 = anneal to zero (default)
```

**2. Update scheduler construction (line 457):**

Replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

With:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15, eta_min=cfg.eta_min
)
```

Default `eta_min=0.0` reproduces the current schedule exactly. CLI flag should be `--eta_min`.

**Arm A — eta_min=5e-5:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h47-etamin5e5-nhead2-wd5e5 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0 \
  --eta_min 5e-5
```

**Arm B — eta_min=1e-4:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h47-etamin1e4-nhead2-wd5e5 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0 \
  --eta_min 1e-4
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test breakdown
- Number of epochs completed before wall, best epoch
- Per-epoch effective LR (should reach the eta_min floor by epoch ~12-13)
- Per-epoch val_avg trajectory (especially the last 3 epochs — is the model still descending?)
- Peak GPU memory

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

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default).

**Beat this: val_avg/mae_surf_p < 66.1060**

Note this experiment ALSO uses wd=5e-5 (H38 orthogonal finding) so the bar to beat reflects both ingredients. If the per-experiment compounding seen earlier holds (H37b at 66.11 vs predicted 66.83 from purely additive decomposition), even a small eta_min effect should push below 65.

⚠ `test_geom_camber_cruise` has NaN scoring bug — report 3-split excl. cruise.
⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h37b-nhead2-lr1e3-clip1 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 1e-3 --clip_grad_norm 1.0
```
