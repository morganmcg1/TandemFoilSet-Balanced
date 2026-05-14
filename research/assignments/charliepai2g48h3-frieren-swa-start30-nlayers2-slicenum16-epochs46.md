# Assignment: frieren — Stochastic Weight Averaging (SWA) on baseline config

**Branch (use exactly):** `charliepai2g48h3-frieren/swa-start30-nlayers2-slicenum16-epochs46`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

SWA (Stochastic Weight Averaging) averages model weights along the late-training trajectory to find a **flatter minimum** with better generalization than any single checkpoint.

The current baseline (PR #2468) ends with **best_epoch=46 (the final epoch), still descending** with slope ~−0.2/epoch on val. This is the textbook setting for SWA: Lion is in late-stage refinement, taking small consistent steps along a nearly-flat valley. Averaging weights from this trajectory should:

1. Reduce variance from the noisy late-stage updates
2. Capture a flatter, wider minimum (better OOD generalization)
3. Be **orthogonal to all closed axes** — doesn't change loss, optimizer, schedule, or capacity

Typical reported gains: 0.5–2% on regression benchmarks. Even a 0.5% improvement (35.08) would beat baseline; 1% (34.90) would be a meaningful Round 41 win.

## Why this is the right next bet

All closed axes from this launch:
- **Capacity** (n_hidden, mlp_ratio, depth) — REFUTED
- **Schedule** (warmup HEAD, truncated cosine TAIL) — REFUTED
- **Loss-weight** (swp=15 marginal, swp=20 saturated) — SATURATED
- **Loss-form** (Huber d=5.0 catastrophic, d=0.1 +5.54%) — CLOSED (your work)

SWA is in a **fresh axis** (post-hoc weight averaging) untouched by any prior experiment. It's also low-risk: if SWA doesn't help, it's a clean negative; if it does, it stacks with future winners (independent of optimizer choice, so will compose with askeladd's AdamW work in #2850).

## Implementation

Add to Config in `train.py`:
```python
swa_start: int = -1   # If >= 0, start SWA averaging at this epoch (0-indexed). -1 = disabled.
swa_lr: float = -1.0  # If > 0, override LR after swa_start (constant). -1 = keep cosine schedule.
```

In the training loop (after creating `model` and `optimizer`):
```python
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = None
if cfg.swa_start >= 0:
    swa_model = AveragedModel(model)
    # Optional: switch to constant SWA LR (recommended off for first try; keep cosine)
    # swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr) if cfg.swa_lr > 0 else None
```

After each training epoch (before val eval):
```python
if swa_model is not None and epoch >= cfg.swa_start:
    swa_model.update_parameters(model)
```

After training loop ends (in addition to existing best-checkpoint eval):
```python
if swa_model is not None:
    # No BN in Transolver (LayerNorm-based), so update_bn() is a no-op but safe to skip.
    swa_model.eval()
    swa_val_metrics = evaluate(swa_model.module, val_loaders, ...)
    swa_test_metrics = evaluate(swa_model.module, test_loaders, ...)
    # Log both with `swa_` prefix in metrics.jsonl
    # Also log a comparison: "swa_vs_best_val: <swa_val_avg> vs <best_val_avg>"
```

**Important:** When you call `evaluate(swa_model.module, ...)`, you may need to pass the underlying model since AveragedModel wraps it. Use `swa_model.module` to access the wrapped model.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-frieren \
  --experiment_name swa-start30-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --swa_start 30
```

`swa_start=30` means averaging begins at epoch 30 (0-indexed) — captures the last **17 epochs** of training (epoch 30 through epoch 46, ~37% of training). This is a sweet-spot window: long enough to average meaningfully, short enough to weight the late-stage flat minimum heavily.

## Baseline to beat

PR #2468 (n_layers=2 + slice_num=16 + epochs=46, **Lion + L1 + cosine, surf_weight=10**):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **35.256** |
| val_single_in_dist | 36.476 |
| val_geom_camber_rc | 48.297 |
| val_geom_camber_cruise | 18.326 |
| val_re_rand | 37.923 |
| **test_avg/mae_surf_p** | **30.245** |

Both `swa_val_avg` and `best_val_avg` should be reported so we can see if SWA strictly improves over the single best-epoch checkpoint.

## Per-run constraints

- Hard timeout: 30 minutes per training execution (`SENPAI_TIMEOUT_MINUTES=30`).
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT add/import/configure/query/log to W&B. If any stale prompt or code comment references `--wandb_name` or `wandb`, treat it as stale guidance.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`. Do not rebase onto unrelated branches.

## Terminal result format

Post a comment with a single-line `SENPAI-RESULT` marker:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_of_swa_and_best_epoch>},"test_metric":{"name":"test_avg/mae_surf_p","value":<matching_test>}}
```

Use **the better of SWA-model and best-epoch-model** as the primary metric (i.e. report whichever generalizes better as val_avg/mae_surf_p), but include BOTH numbers in the prose tables so we can attribute the improvement.

## Suggested follow-ups (in case of any signal)

- **swa_start=20** — wider averaging window (27 epochs), more late-trajectory smoothing
- **swa_start=37 with constant swa_lr=5e-5** — late-start with frozen LR, classic SWA recipe
- **EMA with warmup decay** — `decay = min(0.99, (1+n)/(10+n))`, applied every step instead of every epoch (finer-grained averaging)
- **Stacking SWA with the AdamW winner** (if #2850 wins) — orthogonal levers should compose

## EV assessment

**Medium-high.** SWA is empirically reliable on regression tasks with still-descending end-of-training behavior. Low implementation cost (~20 lines), zero training-cost overhead, no interference with existing pipeline. Worst case is a clean ~0% delta confirming Lion already finds a flat minimum (still informative). Best case is +1% val, a meaningful Round 41 win.
