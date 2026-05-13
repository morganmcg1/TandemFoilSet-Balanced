# slice_num=32: continue the slice sweep (epoch-count mechanism, step 2)

## Hypothesis

The slice_num reduction axis has already won once:
- **PR #1996 (slice_num=64→48)**: ~18% faster per-epoch on n_layers=6 → 15 epochs in budget → T_max=15 alignment → val −1.33%

Current stack has n_layers=4 + slice_num=48. The slice_num sweep is incomplete. Reducing slice_num 48→32 gives a further ~15-20% per-epoch speedup on top of the n_layers=4 compound — which already yields 17 epochs in 30 min.

Expected timing:
- n_layers=4 + slice_num=48: ~94s/epoch → 17 epochs in 30 min
- n_layers=4 + slice_num=32: ~80s/epoch (est.) → ~22 epochs in 30 min → T_max=21 aligned

**Mechanism:** slice_num is the number of learned slice partitions in Transolver's PhysicsAttention. Fewer slices = smaller `in_project_slice` matrix = less computation per attention head = faster forward+backward. The question is whether 32 partitions is sufficient for this 2D CFD problem (boundary layer, wake, freestream) or whether we're below the useful granularity floor.

**Two predictions:**
1. **If slice_num=32 wins:** partition granularity is over-specified at 48; the epoch-budget gain continues to dominate. Would then test slice_num=24 or 16.
2. **If slice_num=32 loses:** 48 is near the optimal granularity for this dataset — the tradeoff between per-epoch speed and PhysicsAttention quality floors around 48. Close this axis.

## Instructions

Change ONLY `--slice_num` from 48 to 32. Set `--epochs 21` to leverage the expected speedup. Keep all other current baseline defaults (n_layers=4, surf_weight=10, GeGLU+RMSNorm, Lion lr=1e-4 WD=1e-4, bf16, batch=4, n_head=4).

Note: n_layers defaults to 5 in train.py — you MUST pass `--n_layers 4` to stay on the current stack.

**Budget guardrail:** If per-epoch time exceeds ~86s at epoch 1, the 30-min cap may cut you before epoch 21. Report however many epochs completed.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-thorfinn \
  --experiment_name slice-num-32-nlayers4 \
  --epochs 21 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 4 \
  --slice_num 32
```

Verify `--slice_num` is plumbed through: the model is initialized with `slice_num=cfg.slice_num` (or hardcoded 48 may need to be changed to `cfg.slice_num` — check train.py before running).

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=46.344 / test=39.950)
2. Per-split `mae_vol_p`
3. Per-epoch wall-clock (epoch 1 vs last completed)
4. Total wall-clock, epochs completed
5. Best epoch, parameter count (should be ~670K — slice_num doesn't change param count significantly)
6. Peak memory

## Baseline (PR #2080, current advisor HEAD: n_layers=4 + slice_num=48)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---|---|
| single_in_dist | 49.979 | 44.746 |
| geom_camber_rc | 61.558 | 54.155 |
| geom_camber_cruise | 27.318 | 22.876 |
| re_rand | 46.518 | 38.025 |
| **avg** | **46.344** | **39.950** |

**Target to beat:** `val_avg/mae_surf_p < 46.344`

Baseline reproduce:
```bash
cd target/ && python train.py --epochs 17 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
