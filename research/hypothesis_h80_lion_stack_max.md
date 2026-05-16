## Hypothesis

**H80: Lion full-stack max — combine ALL winning Lion hyperparameters from the H67-H71 batch on top of H73.**

The H67-H71 batch (now all closed) identified four orthogonal Lion hyperparameter wins at slice=64+RMSNorm+lr=1e-4:
- **warmup=2** (H69): +5.3 pts
- **wd=1e-4** (H71): +4.0 pts (vs wd=5e-4)
- **β₂=0.999** (H68): +3.0 pts
- **n_head=4** (H70): +1.1 pts

H73 is testing each individually (H76-H79). H80 takes the bold-swing approach: **stack all 4 wins simultaneously on top of H73 baseline** and see if compound super-additivity continues (as it did for H73 itself).

Two arms:

- **Arm A: Full Lion stack** — `warmup_epochs=2 + wd=1e-4 + β₂=0.999 + n_head=4` on top of H73 (Lion lr=3e-4 + slice=96 + GEGLU + n_layers=4 + LayerNorm).
- **Arm B: Same as Arm A but lr=2e-4** — slightly lower LR with warmup and slower β₂, hedging against any interaction-induced instability.

**Predicted:**
- **Optimistic compound (5.3 + 1.1 + 3.0 + ~2 from wd) ~ 11 pts additive gain.** Applied to H73's 42.98 → ~32.
- **Realistic discount factor 0.5-0.7** (because each H67-H71 lever was measured against a different baseline). Predicted Arm A range: **35-40 val_avg**.
- **Worst case (interactions cancel)**: ~42-44, no improvement over H73.

**Risk:** Multiple simultaneous changes make attribution impossible if Arm A regresses. That's the trade-off — we're prioritizing the BIG SWING for the headline result. The H76-H79 individual tests will isolate which lever helped most.

**Strategic value:** If Arm A lands at ~35, this is the new floor for the entire round. If it regresses, we have clean H76-H79 individual results to fall back on.

## Instructions

**Step 1: Verify warmup flag is in train.py.** Run `grep -n "warmup" train.py`. If `--warmup_epochs` is supported, proceed. If not, this hypothesis requires a small code change to add SequentialLR(LinearLR → CosineAnnealingLR).

**Step 2: Run both arms:**

```bash
# Arm A — Full Lion stack at lr=3e-4
cd target/ && python train.py --epochs 50 \
  --experiment_name h80-arm-a-lion-fullstack-lr3e4 \
  --agent charliepai2i48h3-thorfinn \
  --optimizer lion --lr 3e-4 --weight_decay 1e-4 \
  --beta1 0.9 --beta2 0.999 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 4 --clip_grad_norm 1.0 \
  --warmup_epochs 2

# Arm B — Same stack at lr=2e-4 (safer)
cd target/ && python train.py --epochs 50 \
  --experiment_name h80-arm-b-lion-fullstack-lr2e4 \
  --agent charliepai2i48h3-thorfinn \
  --optimizer lion --lr 2e-4 --weight_decay 1e-4 \
  --beta1 0.9 --beta2 0.999 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 4 --clip_grad_norm 1.0 \
  --warmup_epochs 2
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- Best epoch, mean s/epoch, peak GPU memory
- Per-epoch val_avg trajectory (vs H73's trajectory)
- **Compound attribution:** if Arm A is significantly better than H73 (say, > 2 pts), this is real super-additive evidence. If <1 pt diff, the levers are not orthogonal at slice=96. Discuss which lever is likely the dominant contributor by comparing to H76-H79 individual results (when they land).

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15. **NO warmup**.

**Source signals from closed PRs (informational):**
- H69 (slice=64+RMSNorm+lr=1e-4): warmup=2 vs warmup=1 → +5.3 pts (49.03 vs 54.30)
- H71 (slice=64+RMSNorm+lr=1e-4): wd=1e-4 vs wd=5e-4 → +4.0 pts (46.02 vs 49.99)
- H68 (slice=64+RMSNorm+lr=1e-4): β₂=0.999 vs β₂=0.95 → +3.0 pts (49.51 vs 52.50)
- H70 (slice=64+RMSNorm+lr=1e-4): n_head=4 vs n_head=2 → +1.1 pts (45.56 vs 46.66 H59-base ref)

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
