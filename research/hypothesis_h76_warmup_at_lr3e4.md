## Hypothesis

**H76: Add linear LR warmup on top of H73 (Lion lr=3e-4 + slice=96).**

H69 (just closed, slice=64 + Lion + RMSNorm) found warmup=2 epochs beats warmup=1 by **5.3 pts** — the largest single hyperparameter signal in the entire H67-H71 batch. Warmup is known to be critical for stable training under Lion at higher LR ranges (because Lion's sign-update produces large effective step sizes early when momentum is unfilled).

H73 used NO warmup and won at val=42.98 with lr=3e-4. Adding warmup should:
1. Stabilize the early epochs (epoch 1 val_avg = 404 → ~50 over the first 5 epochs in H69 — much more controlled start).
2. Reduce the "wasted" first 1-2 epochs spent overshooting.
3. Potentially enable a deeper LR (or richer late-epoch refinement).

Two arms:

- **Arm A: warmup_epochs=2, lr=3e-4** — direct port of H69's winning warmup setting to H73 baseline.
- **Arm B: warmup_epochs=2, lr=5e-4** — combines warmup with higher LR. With warmup stabilizing the early phase, lr=5e-4 may train cleanly where it might diverge without warmup.

**Predicted:**
- Arm A: ~40-42 val_avg (H73's 42.98 minus ~1-2 pts from warmup's "wasted-epoch recovery")
- Arm B: ~38-42 val_avg (best case: warmup unlocks lr=5e-4; worst case: just matches Arm A)

**Risk:** If train.py doesn't yet support `--warmup_epochs` natively, this requires a small code change to add a SequentialLR(warmup → cosine) wrapper. H69's PR likely added this — check if it merged into train.py. If not in main code yet, this hypothesis requires a code change.

## Instructions

**Step 1: Verify warmup flag is in current train.py.** Run `grep -n "warmup" train.py` from the advisor branch. If `--warmup_epochs` is supported, proceed to Step 2. If not, you may need to re-add the SequentialLR warmup logic (linear warmup from `start_factor=1e-6` to `1.0` over `warmup_epochs`, followed by CosineAnnealingLR with `T_max = epochs - warmup_epochs`).

**Step 2: Run both arms:**

```bash
# Arm A — warmup=2 at lr=3e-4 (H73 baseline + warmup)
cd target/ && python train.py --epochs 50 \
  --experiment_name h76-arm-a-warmup2-lr3e4 \
  --agent charliepai2i48h3-fern \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --warmup_epochs 2

# Arm B — warmup=2 at lr=5e-4 (warmup + higher LR)
cd target/ && python train.py --epochs 50 \
  --experiment_name h76-arm-b-warmup2-lr5e4 \
  --agent charliepai2i48h3-fern \
  --optimizer lion --lr 5e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0 \
  --warmup_epochs 2
```

All other flags match H73's exact winning config.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg (3-split, excl. cruise) and per-split test
- **Per-epoch val_avg trajectory** — most important; compare to H73's trajectory (which was 404, 131, 102, 112, 83, 78, 69, 64, 60, 56, 54, 49, 47 for ep 1-13).
- **Per-epoch LR** — verify warmup ramp from ~lr*1e-6 to lr over the first 2 epochs.
- Best epoch, mean s/epoch, peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4055 — H73 Arm B: Lion lr=3e-4 + slice_num=96 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **42.9784** |
| **test_avg/mae_surf_p (3-split, excl. cruise)** | **41.5455** |

Config: optimizer=lion + lr=3e-4 + wd=1e-3 + β=(0.9, 0.99) + slice_num=96 + GEGLU + n_layers=4 + n_head=2 + clip_grad_norm=1.0 + LayerNorm + T_max=15 + **NO warmup**.

H69 reference (just closed, slice=64+RMSNorm+Lion lr=1e-4): warmup=2 wins by 5.3 pts over warmup=1.

**Beat this: val_avg/mae_surf_p < 42.9784**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
