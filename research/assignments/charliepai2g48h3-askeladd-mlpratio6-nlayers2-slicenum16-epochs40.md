# mlp_ratio=6 + epochs=40 on n_layers=2+slice_num=16: FFN CAPACITY BUMP

## Hypothesis

**Follow-up to your own PR #2684 closure (your suggestion #2: explore non-depth capacity axes at n_layers=2).** Depth-down trajectory has bottomed out at n_layers=2 (n_layers=1 catastrophic +12.7% loss). FFN width is the next capacity axis to probe at this depth.

**Why mlp_ratio=6 might help at n_layers=2:**

1. **#2638 split-dependent OOD finding**: geom_camber OOD is capacity-limited. mlp_ratio=6 is a +50% increase in FFN capacity — directly tests the capacity hypothesis along the FFN axis (vs frieren #?? testing along the n_hidden width axis in parallel).

2. **mlp_ratio=6 at n_layers=3 lost (-5.4%, PR #2278)** — BUT that was at higher depth where each block already had more capacity. At n_layers=2, each transformer block must do more work, so wider FFN may now be necessary rather than excess.

3. **Param count**: At n_layers=2, mlp_ratio=4→6 increases FFN params ~50%. Total: ~360K → ~490K (1.36x). Adds capacity to where #2638 indicated it's needed (camber OOD representation).

4. **Parallels frieren #?? (n_hidden=160 + slice_num=12 ISO-EPOCH)**: Together these two experiments probe the capacity-rescue hypothesis along orthogonal axes. If both win, broad capacity story confirmed. If only one wins, specific direction matters. If both fail, capacity isn't the right lever at this stack.

5. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. mlp_ratio=4→6 is a +50% FFN change — well above the noise floor.

**Wall-clock budget reasoning:**
- mlp_ratio=4→6 adds 50% to FFN, ~25-30% to total step time
- Predicted ~44-48 s/epoch (vs 35s at baseline)
- 40 epochs × 45 s ≈ 30 min — at cap with thin margin

**Wall-clock gate:** If epoch 1 wall-clock > 46s, reduce epochs to 37 (37×46=28.4 min, safe). If > 50s, reduce to 35. If > 55s, abort.

## Instructions

Two flag changes from PR #2468 winner: `--mlp_ratio 6 --epochs 40`.

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name mlpratio6-nlayers2-slicenum16-epochs40 \
  --epochs 40 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16 \
  --mlp_ratio 6
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **OOD splits, especially geom_camber_rc and geom_camber_cruise** — KEY: does FFN-capacity bump rescue the camber OOD bottleneck?
4. **single_in_dist** — does extra FFN capacity help or hurt in-dist?
5. **Best epoch** — is best_epoch=40 (final)? Compare epoch-starvation signature to frieren n_hidden=160 result.
6. Per-epoch wall-clock at mlp_ratio=6
7. Param count and peak memory (expect ~490K params)
8. Total wall-clock
9. Compare iso-epoch vs frieren n_hidden=160 + slice_num=12 if available

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
