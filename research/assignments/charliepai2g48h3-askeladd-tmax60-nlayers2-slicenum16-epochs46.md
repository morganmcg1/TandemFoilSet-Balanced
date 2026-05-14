# Truncated cosine T_max=60 + epochs=46 on n_layers=2+slice_num=16: keep LR alive past end of training

## Hypothesis

**Pivot from your own PR #2738 closure (your suggestion #3).** All three capacity-via-param-count axes have now been refuted at this stack (n_hidden ×2 via #2685, #2737; mlp_ratio ×1 via #2738). The bottleneck is NOT model capacity.

**Your own diagnostic from #2738**: best_epoch=40 (final, still descending) — and the baseline #2468 also shows best_epoch=46 still descending. The model is being **prematurely terminated by the scheduler**, not by lack of capacity. At end of cosine annealing T_max=epochs, lr→0, no further updates possible — but the val curve says the model still has room to improve.

**This experiment decouples scheduler T_max from training duration.** With `T_max=60` and `epochs=46`, the cosine schedule only completes ~77% of its arc, leaving residual LR ≈ 13% of peak (≈1.3e-5) at the final epoch. The model gets continued descent room without extending wall-clock.

**Why this might help:**

1. **Direct evidence of premature termination**: PR #2468 best_epoch=46 STILL DESCENDING, your #2738 best_epoch=40 STILL DESCENDING, multiple prior runs same pattern. The model is training-time-limited, NOT capacity-limited — yet LR drops to zero.

2. **Residual LR math**: cos(π × 46/60) ≈ -0.74, so lr_final = lr_min + 0.5*(lr_max-lr_min)*(1 + cos(π × 46/60)) ≈ 0.13 × lr_max ≈ 1.3e-5. This is still in the "learning" regime, not the "polish" regime.

3. **No compute cost**: Same 46 epochs, same per-step cost. Should match baseline ~35s/epoch × 46 = ~27 min total.

4. **Cheap to refute**: If this fails (val unchanged or worse), we know the descent at epoch 46 was minor or training-loss-only (overfit). If it wins, scheduler-bound hypothesis confirmed.

5. **Conservative ratio**: T_max=60 is +30% over current. Not so extreme that early epochs have wildly different LR profiles.

## Code change required

Edit `train.py`:

1. **Add CLI arg** (in @dataclass around line 389):
   ```python
   t_max: int = -1  # If <=0, fallback to MAX_EPOCHS (cfg.epochs). Otherwise overrides scheduler T_max.
   ```

2. **Modify scheduler creation** (line 445):
   ```python
   _t_max = cfg.t_max if cfg.t_max > 0 else MAX_EPOCHS
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_t_max)
   ```

**Verify backwards compatibility BEFORE running the experiment**: with default `--t_max -1`, scheduler should be exactly `T_max=MAX_EPOCHS=cfg.epochs` as before. Run a single-epoch sanity check (--epochs 1) to confirm baseline reproduction.

## Instructions

Two flag changes from PR #2468 winner: `--t_max 60` (plus the code modification above).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name tmax60-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --t_max 60 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **Best_epoch** — does best_epoch=46 (final) still hold, or does best_epoch move later/earlier? KEY DIAGNOSTIC.
4. **Per-epoch LR** — print or log lr at epoch 1, 10, 20, 30, 40, 46 (confirm cosine is truncated correctly)
5. **OOD splits** — geom_camber_rc and geom_camber_cruise specifically: does truncated cosine help OOD generalization?
6. **single_in_dist** — does it regress (would suggest the extra LR causes overfit)?
7. Total wall-clock and per-epoch s

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
