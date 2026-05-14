# Assignment: askeladd — AdamW optimizer (lr=3e-4) replacing Lion

**PR #2824** | Branch: `charliepai2g48h3-askeladd/adamw-nlayers2-slicenum16-epochs46`

**Hypothesis:** Replace Lion with AdamW (betas=0.9/0.95, lr=3e-4). Tests the OPTIMIZER AXIS — fully orthogonal to all HP/arch/schedule/loss axes tested in Rounds 38-40. Schedule+capacity+loss axes all exhausted; optimizer is the next clean lever.

**Run command:**
```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name adamw-lr3e4-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 3e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --optimizer adamw \
  --n_layers 2 --slice_num 16
```

**Target:** val_avg/mae_surf_p < 35.256 (PR #2468 baseline)
