# surf_weight=2 on new compound stack: compound the gradient sweep with epoch-budget wins

## Hypothesis

Two orthogonal mechanisms have independently driven improvements this session:

1. **Epoch-budget mechanism** (n_layers depth + slice_num + T_max alignment): n_layers 6→5→4 and slice_num 64→48 both win via faster epochs → more epochs in 30-min budget. Latest win: n_layers=4 + T_max=17, val −1.07% (PR #2080).

2. **Volume gradient reallocation mechanism** (surf_weight sweep): lowering surf_weight reallocates L1 gradient to volume nodes → richer volumetric features → better surface via geometric context. Sweep so far on OLD stack:
   - sw=10: 52.798 val (PR #1793 baseline)
   - sw=5: 51.040 (−3.33%, PR #1956)
   - sw=2: 49.267 (−3.48% from sw=5, PR #2029) — run on OLD n_layers=6+T_max=12 stack

Your PR #2029 tested sw=2 on the old n_layers=6+T_max=12 stack. It beat the old baseline by −3.48% but is still above the current compound baseline (46.344). Now test sw=2 on the NEW compound stack (n_layers=4 + slice_num=48 + T_max=17). Since both mechanisms are orthogonal — epoch scheduling vs loss weighting — they should compound.

**Expected outcome:** If the −3.48% gain transfers, starting from 46.344 would give ~44.7 val. Even partial transfer would be a new best.

**Note from your PR #2029 writeup:** re_rand was the biggest winner on sw=2 (val −5.52%, test −6.17%), contradicting the prior concern about Reynolds holdout getting starved. geom_camber_cruise had a mild regression (+1.45% val). Watch both as diagnostic canaries.

## Instructions

Set `--surf_weight 2`. Use `--epochs 17` to match the n_layers=4 epoch budget. Keep all other defaults at current baseline (n_layers=4, slice_num=48, GeGLU+RMSNorm, Lion lr=1e-4 WD=1e-4, bf16, batch=4, n_head=4).

Note: n_layers defaults to 5 in train.py — you MUST pass `--n_layers 4`.

### Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name surf-weight-2-nlayers4 \
  --epochs 17 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 2 \
  --n_layers 4
```

### Reporting requirements

1. Per-split val and test `mae_surf_p` vs current baseline (val=46.344 / test=39.950)
2. Per-split `mae_vol_p` — should show improvement vs baseline if the volume-gradient mechanism is still active
3. Epoch 1 vs last epoch wall-clock (confirm ~94s/epoch as expected for n_layers=4)
4. Best epoch and total wall-clock
5. Peak memory

## Baseline (PR #2080: n_layers=4 + slice_num=48 + T_max=17)

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
