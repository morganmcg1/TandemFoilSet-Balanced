# Assignment: askeladd — epochs=50 retest at n_layers=2 stack (epoch-budget axis)

**Branch (use exactly):** `charliepai2g48h3-askeladd/epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

The baseline (PR #2468) ends at `best_epoch=46 (=final) STILL DESCENDING` with slope ~−0.2/epoch. This signal has been observed for 8+ consecutive baseline experiments across the n_layers=2 stack.

**Hypothesis:** Extending epochs from 46 to 50 (4 more cosine epochs) captures the still-descending tail and beats baseline. At ~35s/epoch, 50 epochs × 35s = 29.2 min — fits inside the 30-min cap with a small margin.

## Why this retest is justified

Previous epoch-budget test #2523 (frieren epochs=50): +2.30% loss. But:

1. **#2523 was at the OLDER n_layers=3+slice_num=24 stack.** The current optimal stack (n_layers=2 + slice_num=16) has different per-epoch wall-clock (~35s vs ~54s) and different convergence dynamics.
2. **Seed variance is ~±1.0 val unit** (~+2.85% relative at the 35.256 scale). #2523's +2.30% loss is **within seed-variance noise**, not a definitive negative.
3. The CURRENT stack's `best_epoch=46 still-descending` signal is the empirical motivation. Until this is properly tested at the current stack, we don't know if more epochs help.

Recent failed schedule-FLOOR experiment (#2861 eta_min=5e-6): diagnosed Lion's sign-update oscillation at any non-zero late-epoch LR floor as the structural issue. This further motivates testing the simpler alternative — just give the cosine more time to decay properly at eta_min=0.

## Implementation

**Zero code changes required.** Just an extended training run with `--epochs 50` instead of 46.

The cosine annealing `T_max=cfg.epochs=50` auto-adjusts. The cosine schedule will now decay over 50 epochs instead of 46, meaning:
- At epoch 46 in the new schedule: lr ≈ 1e-5 (10% of initial, still meaningful)
- At epoch 50 (new final): lr = 0 (as before, zero step)

This is functionally a **slower cosine decay** + **4 extra epochs at the tail**.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16
```

## Baseline to beat

PR #2468 (n_layers=2 + slice_num=16 + **epochs=46**, Lion + L1 + cosine, surf_weight=10):

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
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override). 50 epochs at ~35s/epoch should fit (~29.2 min).
- **Local JSONL metrics only.** Do NOT add/import/configure/query/log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Stop-rule

If per-epoch wall-clock balloons (e.g., > 40s/epoch in early epochs), the run will not fit 50 epochs in 30 min. In that case, report the partial run honestly — do NOT push the epoch count down or claim convergence early.

## Terminal result format

Post a comment with a single-line `SENPAI-RESULT` marker:

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Include the **per-epoch val trajectory for the final 5-10 epochs** so we can see whether the model continues descending past epoch 46 (where the baseline cuts off) or whether the slope flattens.

## Expected outcomes

- **Win scenario:** val < 35.256 at best epoch (likely best_epoch ∈ {48, 49, 50}). The still-descending signal converted into a real improvement.
- **Neutral scenario:** val between 34.5 and 35.5 (within seed variance). Inconclusive — epoch-budget axis effectively dead at this stack but couldn't prove it.
- **Loss scenario:** val > 35.5 across all checkpoints. Either the slower cosine hurt convergence (less aggressive late-stage descent) OR seed variance is the dominant effect. Closes the axis.

## Suggested follow-ups

- If 50 wins → try 48 (intermediate) and 52 (just above cap, may require slight stack adjustment)
- If 50 is neutral → multi-seed confirmation of baseline (run 3 seeds of #2468 config to establish noise floor rigorously)
- If 50 loses by within-noise margin → axis is dead at this stack; pivot to architectural

## EV assessment

**Medium.** Tests the longest-running empirical signal (still-descending) at the current optimal stack. Simple zero-code-change experiment. The previous epochs=50 test was at a different stack and within seed-variance noise, so this is genuinely under-tested at the current configuration. Worst case: ~+1% loss confirming the epoch-budget axis is saturated. Best case: ~−1% gain confirming the empirical signal was real.
