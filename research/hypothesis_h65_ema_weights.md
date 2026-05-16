## Hypothesis

**H65: Exponential Moving Average (EMA) of model weights gives a free generalization improvement at GEGLU baseline.**

EMA weight averaging is the most-cited "free improvement" technique in modern deep learning (Izmailov et al. 2018 SWA, Polyak 1991 averaging, Tarvainen & Valpola 2017 Mean Teacher). The core idea: maintain a slow-moving exponential moving average of model weights during training, then evaluate using the EMA weights. EMA finds *flatter* regions of the loss landscape, which empirically generalize better.

**Mechanism:** SGD/AdamW trajectories oscillate around minima at the end of training (especially with cosine annealing where LR → 0 but not exactly 0 in the last epochs). The instantaneous weights are noisy. Averaging across the oscillations finds a point closer to the basin's center — a wider, flatter minimum. Loss-surface geometry connects flatness ↔ generalization (Keskar et al. 2017, Foret et al. 2020).

For CFD surrogates with strong OOD evaluation (camber_rc, re_rand), flatter minima should help disproportionately because the OOD splits sample loss landscape points where the model's confidence is lower.

**Note:** H2 (edward, R1) tested EMA decay=0.999 at the original baseline (val=114.6). Result wasn't transformative because the baseline was so bad that any small improvement was noise. Worth revisiting at the much-tighter GEGLU baseline (58.63) where small gains are visible.

**Two arms (EMA decay sweep):**
- **Arm A: ema_decay=0.999** — standard SWA/EMA decay. Effective averaging window ≈ 1000 steps ≈ 2-3 epochs at our batch size.
- **Arm B: ema_decay=0.9999** — slow-update EMA. Effective averaging window ≈ 10000 steps ≈ 25 epochs. May undertrain within our 13-15 epoch budget.

**Predicted:** Arm A ≈ 57.5-58.5 (small but reliable improvement). Arm B may regress slightly if EMA lags too far behind (decay too aggressive for our short budget).

**Risk:** Negligible. EMA adds one extra forward pass at eval time (uses ema_model instead of model). Training compute is unchanged. The main risk is no-op if loss landscape is already smooth at convergence.

## Instructions

Add `--ema_decay` CLI flag (default 0.0 = off). Implementation in `train.py`:

```python
import copy

# After model construction
if args.ema_decay > 0:
    ema_model = copy.deepcopy(model)
    ema_model.eval()  # freeze BN-style state; we have LayerNorm so this is harmless
    for p in ema_model.parameters():
        p.requires_grad = False

# Inside the training step, after optimizer.step():
if args.ema_decay > 0:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

# At validation/test time, swap to ema_model for evaluation:
if args.ema_decay > 0:
    eval_model = ema_model
else:
    eval_model = model
```

The validation loop should call `eval_model(...)` for predictions. **Both raw model and EMA model should be evaluated and logged separately** so we can verify EMA is helping (or not).

Run both arms:

```bash
# Arm A — ema_decay=0.999 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h65-ema999-geglu \
  --agent charliepai2i48h3-tanjiro \
  --ffn_act geglu \
  --ema_decay 0.999 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — ema_decay=0.9999 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h65-ema9999-geglu \
  --agent charliepai2i48h3-tanjiro \
  --ffn_act geglu \
  --ema_decay 0.9999 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags use current merged defaults: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15, ffn_act=geglu.

**Report:**
- val_avg/mae_surf_p for **both raw model AND EMA model** per epoch — this is critical to verify EMA is helping
- test_avg/mae_surf_p (3-split, excl. cruise) for both raw and EMA — primary comparison
- Per-split breakdown for EMA at best epoch
- Best epoch and epochs completed
- Per-epoch trajectory of EMA val_avg vs raw val_avg — EMA should track raw closely early, then "smooth" at the end
- **L2 distance between EMA and raw weights** at epochs 1, 7, 13 — confirms EMA is updating but lagging as expected
- Peak GPU memory (should increase by ~param size for storing EMA model — for our 891k params that's ~3.6 MB, negligible)
- Mean s/epoch (EMA update is ~1% overhead, should be unchanged)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report. (Unlikely — EMA cannot make training worse, only the eval-side choice changes.)

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (no EMA)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_p=0.25, δ_vel=0.5 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + **ema_decay=0.0**.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted: Arm A (ema_decay=0.999) ≈ 57.5-58.5. Arm B (ema_decay=0.9999) ≈ 58-60 (may undertrain).

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
