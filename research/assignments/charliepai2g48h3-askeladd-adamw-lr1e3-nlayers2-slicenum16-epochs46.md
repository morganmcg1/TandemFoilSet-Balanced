# Assignment: askeladd — AdamW lr=1e-3 retry (10× Lion lr, step-size matched)

**PR #2850** | Branch: `charliepai2g48h3-askeladd/adamw-lr1e3-nlayers2-slicenum16-epochs46`

**Hypothesis:** Previous #2824 AdamW lr=3e-4 UNDER-CONVERGED (val 45.68 vs 35.256 baseline, +29.6%, best_ep=46 still descending). Lion→AdamW lr×3 rule is calibrated for large-batch/image-classification. On small-data (1499) + small-batch (4) regression, AdamW needs ~10× Lion lr to match the sign-update step size. Retrying with lr=1e-3.

**Run command:**
```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name adamw-lr1e3-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-3 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --optimizer adamw \
  --n_layers 2 --slice_num 16
```

**Target:** val_avg/mae_surf_p < 35.256 (PR #2468 baseline)

**Watch for:** divergence at lr=1e-3 (epoch 1 val > 200); under-convergence still (epoch 10 val > 60 → axis is dead).
