# weight_decay=3e-4 (alone) on n_layers=2+slice_num=16+epochs=46: WD axis isolation

## Hypothesis

**Isolate WD=3e-4 effect from LR.** PR #2601 (askeladd) tested lr=1.5e-4 + wd=3e-4 compound and showed WD partially rescued the OOD damage from high LR. Their explicit suggestion #2: test wd=3e-4 alone at baseline LR to determine whether the OOD bottleneck is **regularization** (cheap fix via WD) or **capacity/inductive bias** (need architecture change).

**Why isolated WD=3e-4 might help at n_layers=2:**

1. **OOD bottleneck targeting (PR #2525, #2601)**: At n_layers=2, OOD splits dominate val_avg. Stronger weight decay favors flatter minima → known association with OOD generalization. Without the high-LR confounder, we get a clean read on WD's regularization effect.
2. **Per #2601 OOD partial rescue**: Even at lr=1.5e-4, all 3 OOD test splits improved when WD went 1e-4→3e-4 (rc -2.36%, cruise -3.27%, re_rand -0.64%). At baseline LR, this OOD effect may compound *additively* with the already-fine in-dist behavior, potentially netting positive.
3. **Risk: over-regularization of in-dist**: PR #2601 showed in-dist val regressed (36.48 → 36.88) under WD=3e-4 even with the LR=1.5e-4 boost. We expect some in-dist regression — the question is whether it's smaller than the OOD gain.
4. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. WD 1e-4 → 3e-4 is a 3x change — expect signal above noise floor.
5. **Clean diagnostic**: If wd=3e-4 alone wins → OOD is regularization-limited → try wd=5e-4 next. If wd=3e-4 alone loses → OOD is capacity/inductive-bias limited → pivot to architectural levers (aux surface head, physics-informed loss).

## Instructions

Single flag change from PR #2468 winner: `--weight_decay 3e-4`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name wd3e-4-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 3e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: does WD=3e-4 alone help OOD without the LR=1.5e-4 confounder?
4. **single_in_dist** — magnitude of in-dist regression from stronger WD alone
5. Best epoch — does stronger WD push best_epoch later (slower convergence)?
6. Train loss curve epochs 1-5: slower descent vs WD=1e-4?
7. Total wall-clock, peak memory

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
