# Assignment: frieren — Huber loss (delta=5.0) replacing L1

**PR #2822** | Branch: `charliepai2g48h3-frieren/huber-d5-nlayers2-slicenum16-epochs46`

**Hypothesis:** Replace L1 with Huber (smooth_l1_loss beta=5.0) across all channels+terms. Tests loss-FORM axis after loss-WEIGHT axis saturated (swp=15 marginal, swp=20 broke).

**Run command:**
```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name huber_d5-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --huber_delta 5.0 \
  --n_layers 2 --slice_num 16
```

**Target:** val_avg/mae_surf_p < 35.256 (PR #2468 baseline)
