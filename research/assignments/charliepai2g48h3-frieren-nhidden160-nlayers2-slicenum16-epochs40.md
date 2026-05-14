# n_hidden=160 + epochs=40 on n_layers=2+slice_num=16: CAPACITY BUMP targeting geom_camber OOD

## Hypothesis

**Architectural pivot per Round 37 insight (your own PR #2638 diagnostic):** geom_camber OOD splits are **capacity-limited**, not regularization-limited. We've now closed the LR axis, WD axis, surf_weight axis, mlp_ratio axis (n_layers=2 in-flight), and slice_num axis at this stack. The remaining high-EV axis is the **model width (`n_hidden`) axis**, which has been fixed at 128 throughout the entire project.

**Why n_hidden=160 might help at n_layers=2:**

1. **Direct capacity increase**: n_hidden 128→160 (+25%) increases parameters in attention/FFN/embeddings ~1.56x. More capacity globally → more capacity available for geom_camber representations specifically.
2. **Per #2638 split-dependent OOD finding**: re_rand improved with regularization (WD=3e-4); geom_camber regressed. The geom_camber direction needs capacity, not constraint. n_hidden bump is the cleanest capacity-targeted intervention.
3. **Wall-clock fit**: n_hidden=128→160 (1.25x linear) → per-step time ~1.4-1.6x → estimated 50s/epoch (vs 35s at 128). 40 epochs × 50s = 33 min — uses bs=4 default but **drops epochs from 46→40** to fit in the 30-min cap (40×45s = 30 min target).
4. **Variance context (PR #2523)**: Run-to-run variance ~±1.0 val units. n_hidden=128→160 is a +25% capacity change — expect signal above noise floor.
5. **Memory headroom**: Currently 13.5 GB peak at n_hidden=128 (bs=4). n_hidden=160 → ~21 GB. Well under our 96GB budget.

## CODE CHANGE REQUIRED

`--n_hidden` is **not currently a CLI flag**. You'll need to add it.

**Step 1: Add `--n_hidden` CLI flag to train.py**
- Find the Config class (likely around `class Config:` near n_head, n_layers, slice_num, mlp_ratio fields)
- Add `n_hidden: int = 128` field if missing
- Find where the model_config is built (around line 435 per prior research notes) — ensure `n_hidden=cfg.n_hidden` is passed
- Verify by running `python train.py --help` and checking `--n_hidden` appears

**Step 2: Run experiment with --n_hidden 160 --epochs 40**

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name nhidden160-nlayers2-slicenum16-epochs40 \
  --epochs 40 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16 \
  --n_hidden 160
```

**Wall-clock gate:** If epoch 1 wall-clock > 48s, reduce epochs to 35. If > 52s, reduce to 33. If > 60s, abort and report — capacity too expensive at this depth/partition combo.

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits, especially geom_camber_rc and geom_camber_cruise** — KEY: does capacity bump rescue the camber OOD bottleneck identified in your #2638 analysis?
4. **single_in_dist** — does extra capacity help or hurt?
5. Best epoch — is best_epoch=final? If yes, more epochs might compound the capacity gain.
6. Per-epoch wall-clock at n_hidden=160
7. Param count vs baseline (expect ~1.56x from 361K → ~560K)
8. Peak memory, total wall-clock
9. Confirm the `--n_hidden` flag was actually parsed and applied (sanity check via param count or train config dump)

## Baseline (PR #2468)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.476 | 33.035 |
| geom_camber_rc | 48.297 | 44.333 |
| geom_camber_cruise | 18.326 | 15.496 |
| re_rand | 37.923 | 28.116 |
| **avg** | **35.256** | **30.245** |

**Reproduce baseline:**
```bash
cd target/ && python train.py \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --n_layers 2 --slice_num 16
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
