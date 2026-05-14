# Per-channel surface weighting: surf_weight_p=15, surf_weight_uv=10 — direct attack on val_avg/mae_surf_p

## Hypothesis

**Pivot away from capacity-bump axis (refuted twice: #2685 +2.53%, #2737 +7.55%).** Under 30-min compute budget, baseline (n_hidden=128) Pareto-dominates n_hidden=160. The capacity-along-n_hidden axis is dead at this stack.

**This experiment attacks `val_avg/mae_surf_p` directly via the loss function, not architecture.** Currently `surf_weight=10` applies uniformly across surface channels {Ux, Uy, p}. Since our primary validation metric is **surface pressure MAE**, we hypothesize that emphasizing pressure relative to velocity in the loss will improve val_avg/mae_surf_p without adding compute cost.

**Why this might help:**

1. **Direct optimization signal**: `val_avg/mae_surf_p` is the metric we're scored on. Current loss gives pressure 1/3 of surface signal (uniform over {Ux, Uy, p}). Boosting pressure-specific signal should improve the optimized metric.

2. **No compute cost**: Pure loss reformulation — no architecture change, no extra params, no per-step overhead. Stays well under 30-min cap (same ~35s/epoch × 46 epochs = ~27 min).

3. **Conservative ratio**: surf_weight_p=15, surf_weight_uv=10 is a 1.5× pressure emphasis, holds velocity at baseline. Total surface weight 20+10=30 → ~17% reduction in absolute surface signal but with pressure-biased focus. Volume loss unchanged.

4. **Building on observed channel ordering**: train.py line 436 confirms `output_fields=["Ux", "Uy", "p"]` so channel index 2 is pressure. Per-channel slicing is straightforward.

5. **Variance context (PR #2523)**: Seed variance ~±1.0 val units. A 1.5× pressure emphasis is a substantial signal change — well above noise floor.

## Code change required

Edit `train.py` to:

1. **Add CLI args** (in the @dataclass for cfg, around line 389):
   ```python
   surf_weight_p: float = -1.0   # If <0, fallback to surf_weight (backwards compat)
   surf_weight_uv: float = -1.0  # If <0, fallback to surf_weight
   ```

2. **Modify the training-step loss** (lines 486–492) to per-channel weighting:
   ```python
   abs_err = (pred - y_norm).abs()
   vol_mask = mask & ~is_surface
   surf_mask = mask & is_surface
   vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1) / abs_err.shape[-1]

   # Per-channel surface losses
   sw_p = cfg.surf_weight_p if cfg.surf_weight_p >= 0 else cfg.surf_weight
   sw_uv = cfg.surf_weight_uv if cfg.surf_weight_uv >= 0 else cfg.surf_weight
   abs_err_uv = abs_err[..., 0:2]  # Ux, Uy
   abs_err_p = abs_err[..., 2:3]    # pressure
   surf_loss_uv = (abs_err_uv * surf_mask.unsqueeze(-1)).sum() / (surf_mask.sum() * 2).clamp(min=1)
   surf_loss_p  = (abs_err_p  * surf_mask.unsqueeze(-1)).sum() /  surf_mask.sum().clamp(min=1)

   surf_loss = surf_loss_uv + (sw_p / sw_uv) * surf_loss_p  # normalized for logging
   loss = vol_loss + sw_uv * surf_loss_uv + sw_p * surf_loss_p
   ```
   (Note: the logged `surf_loss` is for tensorboard/jsonl consistency; the **true loss** is the last line.)

3. **Also update evaluate_split** (lines 243–299) so val/test logging uses the same per-channel weighting. For safety, just pass `sw_p`, `sw_uv` and compute the same way.

## Instructions

Single-arm experiment, three flag changes from PR #2468 winner: `--surf_weight_p 15 --surf_weight_uv 10` (and the code modifications above).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name swp15-swuv10-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --surf_weight_p 15 \
  --surf_weight_uv 10 \
  --n_layers 2 \
  --slice_num 16
```

**Verify backwards compatibility BEFORE running the experiment**: with `--surf_weight 10` (and surf_weight_p/uv at default -1), the loss should be EXACTLY equivalent to baseline. Run 1-2 epochs to confirm baseline reproduction before launching full run.

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_surf_Ux and mae_surf_Uy — does velocity REGRESS?
3. Per-split mae_vol_p — does volume pressure improve too (transfer)?
4. **OOD splits** — geom_camber_rc and geom_camber_cruise specifically — does pressure-emphasized loss help OOD?
5. Best epoch — same as baseline (46)?
6. Per-epoch wall-clock — should match baseline ~35s
7. Total wall-clock

## Baseline (PR #2468)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p |
|---|---|---|---|---|
| single_in_dist | 36.476 | — | — | 33.035 |
| geom_camber_rc | 48.297 | — | — | 44.333 |
| geom_camber_cruise | 18.326 | — | — | 15.496 |
| re_rand | 37.923 | — | — | 28.116 |
| **avg** | **35.256** | — | — | **30.245** |

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
