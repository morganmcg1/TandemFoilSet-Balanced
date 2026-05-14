# Assignment: frieren — Huber d=0.1 (normalized-space corrected) retry

**PR #2847** | Branch: `charliepai2g48h3-frieren/huber-d0p1-nlayers2-slicenum16-epochs46`

**Hypothesis:** Previous #2822 Huber d=5.0 was catastrophically miscalibrated (delta in raw scale, but loss in normalized space → effectively MSE everywhere, +116% val). Retry with d=0.1 properly scales for normalized errors (~90% stay in L1 tail, only smallest 10% get quadratic smoothing). Tests the actual Huber hypothesis.

**Run command:**
```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name huber_d0p1-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --huber_delta 0.1 \
  --n_layers 2 --slice_num 16
```

**Target:** val_avg/mae_surf_p < 35.256 (PR #2468 baseline)

**EV assessment:** Low. Lion uses sign-only updates so near-zero gradient magnitude matters less. Cleanest way to close the loss-form direction definitively.
