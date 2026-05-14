# Assignment: askeladd — Cosine eta_min=5e-6 (schedule-FLOOR axis)

**Branch (use exactly):** `charliepai2g48h3-askeladd/etamin5e6-nlayers2-slicenum16-epochs46`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

The current cosine schedule decays LR all the way to zero at epoch 46:
```
lr(t) = 1e-4 * 0.5 * (1 + cos(π * t / 46))
lr(46) = 0
```

But our baseline (#2468) shows **best_epoch=46 STILL DESCENDING** — the model is making real progress at the very end. With LR=0 at the final epoch, the final update step is effectively zero, so the very-last-epoch weights are nearly identical to epoch-45's. We're throwing away the final epoch.

**Hypothesis:** Setting `eta_min=5e-6` (5% of initial LR) maintains a productive non-zero LR floor through the polish tail. At epoch 46, the model would still take ~5% of its initial step magnitude, allowing additional convergence at the still-descending trajectory.

For Lion's sign-only updates, eta_min=5e-6 means the final updates still have full directional information at 5% step magnitude — clean, additional refinement without the late-stage oscillation risk of a higher floor (e.g., 10% / 1e-5).

## Why this is a fresh axis

The schedule axis was previously declared DEAD, but only the schedule **SHAPE** was tested:
- **#2760 truncated cosine (T_max=60, epochs=46):** SHAPE change — removes the polish tail. +1.63% LOSS. Refuted: polish tail matters.
- **#2797 warmup_epochs=3 + standard cosine:** SHAPE change — adds linear warmup ramp. +2.04% LOSS. Refuted: head shape doesn't help.

**Schedule FLOOR is mechanistically distinct.** Both refuted experiments preserved eta_min=0 (decay-to-zero). This experiment tests whether a non-zero floor at the END (orthogonal to head/tail shape) helps.

This is supported by the empirical observation that the still-descending-at-final-epoch behavior has held for 8+ consecutive baseline experiments. The schedule-shape refutations don't tell us about the schedule-floor effect.

## Why eta_min=5e-6 (not 1e-5 or 1e-6)

- **1e-5 (10%):** Aggressive floor. For Lion which uses sign updates, this might cause late-stage oscillation around the minimum (similar to the 49.83→49.05→49.25 jitter observed in AdamW lr=1e-3 #2850).
- **5e-6 (5%):** **Sweet spot.** Productive floor, low oscillation risk. Sufficient signal to nudge convergence in the final 5-10 epochs.
- **1e-6 (1%):** Likely too small to matter — effectively still decaying to ~zero.

If 5e-6 wins, we have a follow-up to test 1e-5 (more aggressive). If 5e-6 loses, we test 1e-6 (smaller) before declaring the axis dead.

## Implementation

Add to Config in `train.py`:
```python
eta_min: float = 0.0   # Cosine annealing minimum LR. 0.0 = decay to zero (default).
```

In the scheduler creation:
```python
# Existing line (likely):
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
# Modified:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
)
```

That's the entire change.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name etamin5e6-nlayers2-slicenum16-epochs46 \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --eta_min 5e-6
```

## Baseline to beat

PR #2468 (n_layers=2 + slice_num=16 + epochs=46, **Lion + L1 + cosine, surf_weight=10, eta_min=0**):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **35.256** |
| val_single_in_dist | 36.476 |
| val_geom_camber_rc | 48.297 |
| val_geom_camber_cruise | 18.326 |
| val_re_rand | 37.923 |
| **test_avg/mae_surf_p** | **30.245** |

## Per-run constraints

- Hard timeout: 30 minutes per training execution (`SENPAI_TIMEOUT_MINUTES=30`).
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT add/import/configure/query/log to W&B. If any stale prompt or code comment references `--wandb_name` or `wandb`, treat it as stale guidance.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`. Do not rebase onto unrelated branches.

## Terminal result format

Post a comment with a single-line `SENPAI-RESULT` marker:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val_checkpoint>}}
```

Include the full per-split table and the **per-epoch LR trace** (or at least final-5-epoch LRs) so we can verify the floor took effect.

## Suggested follow-ups

- **eta_min=1e-5 (10%)** — if 5e-6 wins, more aggressive floor
- **eta_min=1e-6 (1%)** — if 5e-6 is neutral/marginal, smaller floor before declaring axis dead
- **eta_min=5e-6 + SWA stacking** — if both this and frieren's SWA win, compose them

## EV assessment

**Medium-high.** The still-descending-at-final-epoch signal is robust and unexplained. eta_min>0 is the textbook intervention for it. Low implementation cost (one CLI flag + one scheduler arg), zero training overhead, no interference with existing pipeline. Worst case is a clean ~0% delta confirming the LR=0 endpoint isn't load-bearing (still informative). Best case is +0.5-1% val (a clean Round 41 win).
