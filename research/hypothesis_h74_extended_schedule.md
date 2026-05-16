## Hypothesis

**H74: Extend the cosine schedule on H73 to capture the wall-cut tail.**

H73 Arm B set the new baseline at val_avg=42.9784 / test 3-split=41.5455, but was wall-cut at epoch 15/50 with val still descending ~0.8 pts/epoch. The cosine schedule had T_max=15, so by epoch 15 the LR was already at the minimum — the loss curve cleanly stopped at full anneal, NOT at a plateau. There is substantial untapped descent.

The 30-min wall budget fits ~15 epochs at H73's mean s/epoch=~120s. With a longer schedule (more time spent at meaningful LR), the model can keep descending. Two arms:

- **Arm A: T_max=20, epochs=20** — full 20 epochs of cosine, modest budget extension. Expected wall ~ 40 min (may need to extend SENPAI_TIMEOUT_MINUTES; if 30 min is hard cap, the run will stop at ~ep 15 still at non-zero LR — already an improvement over T_max=15).
- **Arm B: T_max=15, epochs=30 with restart** — same T_max but allows cosine restart at ep 16, exploring a "two cosine cycles" SGDR-style schedule.

**Risk:** If SENPAI_TIMEOUT_MINUTES is a hard 30-min cap, both arms may not get to run their full schedules. Arm A is the safer bet: a T_max=20 cosine within 30 min still spends more total time at meaningful LR than the original T_max=15.

**Predicted:**
- Arm A: ~38-41 val_avg (1-4 pts below H73's 42.98 if the descent rate continues)
- Arm B: ~40-43 val_avg (warm restart may give a small extra gain via re-exploration)

If neither beats baseline, that tells us H73 is near its own LR-trajectory floor; the next lever should be hyperparameters (warmup, β₂, n_head, wd) which are being tested in H76-H79 in parallel.

## Instructions

The merged codebase already has `--epochs` and `--t_max` (or equivalent) as CLI flags. No code changes needed.

Run both arms:

```bash
# Arm A — T_max=20, epochs=20
cd target/ && python train.py --epochs 20 --t_max 20 \
  --experiment_name h74-arm-a-tmax20-ep20 \
  --agent charliepai2i48h3-askeladd \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — T_max=15, epochs=30 with restart
cd target/ && python train.py --epochs 30 --t_max 15 \
  --experiment_name h74-arm-b-tmax15-ep30-restart \
  --agent charliepai2i48h3-askeladd \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

If `--t_max` is not a CLI flag in train.py, check the Config dataclass or argparse setup and use whatever name maps to the cosine T_max parameter. If the schedule does NOT support warm restart natively, fall back to Arm A only and report.

All other flags match the H73 winning config exactly.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test for both arms
- Per-epoch val_avg trajectory — does it continue descending through ep 15, 16, 17, ...?
- Mean s/epoch, peak GPU memory
- Best epoch (the model from this is what generalizes; report metrics at best_epoch, not last)
- Total epochs run before wall cut (SENPAI_TIMEOUT_MINUTES enforced)
- Per-epoch LR — verify the cosine annealing follows the expected curve

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 + GEGLU + n_layers=4 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| val_single_in_dist/mae_surf_p | 43.7880 |
| val_geom_camber_rc/mae_surf_p | 56.6638 |
| val_geom_camber_cruise/mae_surf_p | 26.4930 |
| val_re_rand/mae_surf_p | 44.9686 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **41.5455** |
| test_single_in_dist/mae_surf_p | 38.7901 |
| test_geom_camber_rc/mae_surf_p | 50.1886 |
| test_re_rand/mae_surf_p | 35.6578 |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15 + epochs=50 (wall-cut at ep 15).

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
