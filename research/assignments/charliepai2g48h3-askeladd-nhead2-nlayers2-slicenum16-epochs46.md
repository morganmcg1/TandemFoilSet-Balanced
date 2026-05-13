# n_head=2 on n_layers=2+slice_num=16+epochs=46: Attention head tuning at new depth+partition

## Hypothesis

n_head was tested at n_layers=3+slice_num=24 (closed: n_head=4 won, n_head=2 lost by ~0.7%). But the new optimum is n_layers=2+slice_num=16+epochs=46 (val=35.256). The n_head optimum may shift at this stack because:

1. **Per-head expressiveness matters more at shallow depth**: With only 2 transformer blocks, each attention layer's representational capacity is more critical. n_head=2 doubles head_dim from 32 → 64, giving each head a wider per-head feature space.

2. **Slice partition shift may favor wider heads**: n_head=2 lost at slice_num=24. At slice_num=16 the routing problem is different — fewer slices to attend over. Wider heads (head_dim=64) may capture richer per-slice attention patterns at this partition.

3. **OOD generalization may benefit from wider heads**: The lr=1.5e-4 result (PR #2525) showed n_layers=2 OOD splits are the bottleneck (in-dist regression resolves easily). Wider heads may build more robust attention features that generalize better to OOD geometries.

4. **The bottleneck-conclusion is stack-dependent**: The original n_head conclusion ("bottleneck is slice routing, not Q/K/V capacity") was at slice_num=24. At slice_num=16 with shallower depth, the slice mechanism is doing more per-token work and may interact differently with head count.

## Instructions

Single flag change from PR #2468 winner: `--n_head 2`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name nhead2-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16 \
  --n_head 2
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — does n_head=2 help with the in-dist regression at n_layers=2?
4. **OOD splits (geom_camber_rc, geom_camber_cruise, re_rand)** — KEY: do wider heads help OOD generalization?
5. Best epoch, total wall-clock, peak memory

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
