## Hypothesis

**H50: WSD (Warmup-Stable-Decay) trapezoidal schedule — hold LR at peak longer, then sharp linear cooldown.**

H41 confirmed that stretching T_max from 15→20 helps because at our 30-min wall budget (~14 epochs completed), cosine decay anneals too aggressively (final LR ~4.5% of peak with T_max=15). H43 showed that warmup alone eats budget. H41 Arm C (T_max=20 + n_head=2 + wd=5e-5 stack) **failed** because extending T_max stripped the late-epoch fine-tune phase that in-distribution data needs.

The clean diagnosis: at 14-epoch budget, the model spends too few iterations at peak LR, but it ALSO needs a real cooldown for the in-distribution split. **WSD addresses both** with a trapezoidal LR profile:

```
LR
peak |    ____________
     |   /            \
     |  /              \
     | /                \
     |/                  \
   0 +____________________ epoch
     0  warmup  stable  end
```

**Mechanism (Hu et al. 2024 "MiniCPM"; Hägele et al. 2024 "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"):**
- **Warmup** (epochs 0..W): linear ramp from 0 to peak_lr. Stabilizes early training.
- **Stable** (epochs W..W+S): hold at peak_lr. Main optimization work happens at full LR.
- **Decay** (epochs W+S..W+S+D): linear (NOT cosine) drop from peak_lr to 0. Sharp cooldown induces final consolidation.

**Why this might win vs cosine in our regime:**
1. Cosine wastes LR in early epochs (LR is already dropping at epoch 1). WSD provides MORE time at peak.
2. The sharp linear cooldown forces the model into a sharper local minimum than cosine's smooth tail. For regression tasks with high-curvature loss valleys near boundary layers (pressure-gradient cliffs), this can help.
3. Validated by Hägele et al. 2024: WSD matches cosine at less compute when the schedule shape matches the budget. Our budget is fixed at 14 epochs, so WSD's three phases can be tuned to fit.
4. **Orthogonal to H47 eta_min**: H47 raises the *floor* of cosine; WSD changes the *entire shape*.

**Distinct from H43 (closed) warmup-on-cosine:** H43 added warmup *and kept cosine* — the warmup ate budget that cosine would have used for descent. WSD uses warmup PLUS a stable peak phase, so the model gets MORE peak-LR descent, not less. The trade is replacing the cosine tail with a sharper linear tail.

**Distinct from H41 Arm C T_max stretch:** Your closed H41 Arm C showed that stretching the schedule strips the fine-tune phase. WSD keeps a sharp cooldown (the linear decay reaches 0 within budget), so the model still gets a fine-tune phase. Arm A's 4-epoch linear decay reaches LR=0 at epoch 14 (within budget), preserving the late-LR-low-finetune that n_head=2+wd=5e-5 requires for in-distribution.

**Two arms compare phase splits:**

- **Arm A — WSD 2/8/4** (2ep warmup, 8ep stable peak, 4ep linear decay, total 14 epochs): Balanced. ~57% time at peak.
- **Arm B — WSD 1/9/4** (1ep warmup, 9ep stable peak, 4ep linear decay, total 14 epochs): Shorter warmup, even more peak time (~64% at peak). Tests whether 1ep warmup is sufficient for our model.

Both stack on current best config (n_head=2, wd=5e-5, lr=1e-3, clip=1.0). Predicted val_avg if WSD effect is real and the stable phase compounds with the stacked optimizer config ≈ 64-66.

## Instructions

One code change in `train.py`: replace `CosineAnnealingLR` with `LambdaLR` for the WSD branch.

**1. Add Config fields** (after `clip_grad_norm` line ~404):
```python
schedule: str = "cosine"        # 'cosine' (default) or 'wsd'
wsd_warmup_epochs: int = 2      # warmup phase length (WSD only)
wsd_stable_epochs: int = 8      # stable phase length (WSD only)
wsd_decay_epochs: int = 4       # decay phase length (WSD only)
```

**2. Wire WSD scheduler** (replace scheduler construction at line ~457):
```python
if cfg.schedule == "wsd":
    W = cfg.wsd_warmup_epochs
    S = cfg.wsd_stable_epochs
    D = cfg.wsd_decay_epochs
    def wsd_lambda(epoch):
        if epoch < W:
            return (epoch + 1) / W                          # linear warmup
        elif epoch < W + S:
            return 1.0                                       # stable peak
        elif epoch < W + S + D:
            return max(0.0, 1.0 - (epoch - W - S) / D)       # linear decay
        else:
            return 0.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_lambda)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

CLI flags: `--schedule`, `--wsd_warmup_epochs`, `--wsd_stable_epochs`, `--wsd_decay_epochs`.

**Important verification**: confirm `scheduler.step()` is called once per **epoch** (not per batch) in train.py. If it's per-batch, you'll need to scale the phase lengths by `steps_per_epoch` and pass the epoch index correctly. Inspect this before launching to avoid a schedule mismatch.

**Arm A — WSD 2/8/4 (lr=1e-3, wd=5e-5, n_head=2):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h50-wsd-2-8-4 \
  --agent charliepai2i48h3-fern \
  --schedule wsd --wsd_warmup_epochs 2 --wsd_stable_epochs 8 --wsd_decay_epochs 4 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

**Arm B — WSD 1/9/4 (lr=1e-3, wd=5e-5, n_head=2):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h50-wsd-1-9-4 \
  --agent charliepai2i48h3-fern \
  --schedule wsd --wsd_warmup_epochs 1 --wsd_stable_epochs 9 --wsd_decay_epochs 4 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test breakdown
- Number of epochs completed before wall, best epoch
- **Per-epoch effective LR trajectory** (must show: linear ramp up → flat at peak → linear ramp down. If LR doesn't look trapezoidal, scheduler wiring is wrong)
- **Per-epoch val_avg trajectory** (especially: does the model still descend during the stable phase, or plateau? If plateau, peak LR is too low; if descending, the stable phase is doing work)
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

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + CosineAnnealingLR T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), AdamW.

**Beat this: val_avg/mae_surf_p < 66.1060**

Note this experiment ALSO uses wd=5e-5 (H38 orthogonal finding) so the bar to beat reflects both ingredients.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h37b-nhead2-lr1e3-clip1 \
  --agent charliepai2i48h3-fern \
  --n_head 2 --lr 1e-3 --clip_grad_norm 1.0
```

## Note on closing H41 Arm C

You ran H41 Arm C right before this. The lesson: schedule stretching can backfire when it removes the late-epoch fine-tune phase. WSD is mechanically different — it keeps a sharp cooldown (the linear decay), so the model still gets a fine-tune phase. Arm A's 4-epoch linear decay reaches LR=0 at epoch 14 (within budget), preserving the late-LR-low-finetune that n_head=2+wd=5e-5 requires for in-distribution. This is a clean re-test of the schedule lever from a different angle.
