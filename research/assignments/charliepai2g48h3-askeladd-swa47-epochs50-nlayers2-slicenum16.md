# Assignment: askeladd — SWA from epoch 47 on epochs=50 stack (SWA compound on new baseline)

**Branch (use exactly):** `charliepai2g48h3-askeladd/swa47-epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

Your epochs=50 win (#2872) established the precise plateau location: **val_avg bottoms at epoch 47** and stays flat in [34.54, 34.79] across e47-50. Per-epoch trajectory in the plateau:
- e47: 34.544 (BEST)
- e48: 34.794 (+0.250)
- e49: 34.638 (−0.156)
- e50: 34.646 (+0.008)

This is a **confirmed flat loss landscape** — exactly the regime where Stochastic Weight Averaging (SWA) should help:
- SWA computes a running average of model weights across recent epochs.
- In a FLAT region, adjacent weight iterates represent different points in a wide loss basin. Averaging them finds a more central point with better generalization than any single iterate.
- In a DESCENDING region (epochs <47), SWA averages over non-stationary weights — this is why frieren's SWA-at-e46 (#2857) failed: the window included earlier, worse iterates.

**Hypothesis:** SWA over the e47-50 plateau (4 epochs of flat iterates) will find a better-generalizing basin center than the single best checkpoint at e47.

## Why this is different from the closed SWA axis

**#2857 SWA failure (epochs=46):**
- SWA window: epochs 31-46 (16 epochs)
- Val in window: descended from 42.22 → 36.28 (14% relative descent)
- Weight averaging pulled backward toward earlier, worse iterates → SWA worse than best checkpoint on every split

**This experiment (epochs=50, SWA from e47):**
- SWA window: epochs 47-50 (4 epochs, start of plateau)
- Val in window: [34.544, 34.794, 34.638, 34.646] — flat, NON-descending
- Averaging over a genuinely flat region finds a more central basin point

The previous SWA closure was explicitly scoped: "POST-HOC AVERAGING CLOSED **for 46-epoch schedule** (still-descending trajectory invalidates flat-region premise)." With epochs=50, the flat region EXISTS at e47-50. The axis is re-opened under the new conditions.

## Implementation

This requires the same SWA plumbing as frieren's #2857. Add to Config in `train.py`:

```python
swa_start: int = -1   # If >= 0, start SWA averaging at this epoch (0-indexed or 1-indexed — match the loop convention)
```

Initialize after model creation:

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
swa_model = None
if cfg.swa_start >= 0:
    swa_model = AveragedModel(model)
```

In the training loop, after each epoch:

```python
if swa_model is not None and epoch >= cfg.swa_start:
    swa_model.update_parameters(model)
```

After training completes, update batch norm statistics (required by PyTorch SWA):

```python
if swa_model is not None:
    update_bn(train_loader, swa_model)  # One pass over training data to update BN running stats
```

**Eval:** Use `swa_model` instead of `model` for the final evaluation. Report BOTH the SWA checkpoint metrics AND the best single-epoch metrics (so we can compare).

**Note on epoch indexing:** Set `--swa_start` to match epoch 47 in the loop's indexing (likely 47 if 1-indexed, or 46 if 0-indexed). Double-check which convention train.py uses.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name swa47-epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --swa_start 47
```

## Baseline to beat

PR #2872 (n_layers=2 + slice_num=16 + **epochs=50**, best_epoch=47, **no SWA**):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **34.544** |
| val_single_in_dist | 35.113 |
| val_geom_camber_rc | 48.106 |
| val_geom_camber_cruise | 18.895 |
| val_re_rand | 36.060 |
| **test_avg/mae_surf_p** | **29.916** |

## Per-run constraints

- Hard timeout: 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`). 50×35.2s ≈ 29.3 min (same as #2872) + 1 BN update pass (~30s). Should fit.
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.
- SWA BN update requires one forward pass over training data. If this pushes total wall-clock >30 min, report partial results and note the timeout.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_of_swa_or_best_epoch>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Report BOTH:
1. **SWA checkpoint metrics** (the averaged model after BN update)
2. **Best single-epoch metrics** (same as #2872 best_epoch=47 — should be identical since we're running the same config)

This lets us directly measure the SWA improvement over single-best.

## Suggested follow-ups

- **If SWA wins (val < 34.544):** try SWA from e45 (wider window, more averaging) and SWA from e40 (much wider window, test if early averaging helps).
- **If SWA neutral (|Δ| < 0.5 val):** SWA cannot improve on this flat plateau. The individual checkpoints ARE the basin center. Axis closed.
- **If SWA loses (val > 34.544):** BN update may have introduced noise, or the plateau is too shallow for SWA to find a better center. Close axis.

## EV assessment

**Medium-high.** SWA has solid theoretical and empirical backing for flat loss landscapes. The plateau at e47-50 provides exactly the right conditions (unlike the previous failed test at e46 which lacked a plateau). Implementation is lightweight (~15 lines, same torch.optim.swa_utils used in frieren's #2857). Wall-clock overhead is minimal (4 plateau epochs already run + 1 BN pass). Worst case: neutral/slight loss confirming the single best checkpoint is already well-centered. Best case: 0.5-1% additional improvement from ensemble of plateau iterates.
