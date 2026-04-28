# Baseline — TandemFoilSet (willow-pai2d-r5)

**Status:** Round 1 in flight. PR #441 (bf16 standalone) merged on commit `b605b44` as the round-1 winner so far. 2-seed mean: **117.37 ± 0.85** at 19 epochs (vs ~131 implied pre-bf16 cluster, -10.4%). Multi-seed distribution being further calibrated by thorfinn's PR #428. Several round-2 stack candidates in flight (Huber #413, budget-aware cosine #427, grad-clip #434, lr=3e-4 #505).

## Reference configuration (current `train.py` HEAD)

The baseline is the default Transolver in `train.py` at HEAD of `icml-appendix-willow-pai2d-r5`:

- **Model:** Transolver, `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2` (~0.67M params)
- **Optimizer:** AdamW `lr=5e-4`, `weight_decay=1e-4`
- **Schedule:** CosineAnnealingLR with `T_max=epochs`
- **Batch size:** 4
- **Loss:** MSE in normalized space, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10`
- **Training:** `epochs=50`, capped by `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Sampling:** `WeightedRandomSampler` over balanced domain weights

## Reproduce command

```bash
cd /workspace/senpai/target
python train.py --epochs 50
```

## Primary metric

**`val_avg/mae_surf_p`** — equal-weight mean of surface pressure MAE across the four validation splits:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

Lower is better. The matching test metric `test_avg/mae_surf_p` is computed at the end of every run from the best validation checkpoint.

## Best results

_(round 1 in flight; baseline distribution being established by thorfinn's PR #428)_

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| **#441** | **117.37 ± 0.85** (n=2) | 115.59 (3-finite-split) * | bf16 mixed precision standalone; 19 epochs reached vs ~14 fp32; CV ~0.7% — variance-tight |

\* `test_avg/mae_surf_p` 4-split mean is still NaN on cruise pending PR #375 (data/scoring.py fix). Per-channel test surf MAEs: single 122.76, geom_camber_rc 118.27, re_rand 105.73 (3-finite mean). Once #375 lands, can re-evaluate the saved bf16 artifacts (`model-bf16_seed0-cgitj1dc`, `model-bf16_seed1-i45ys5ih`) for canonical 4-split numbers.

### Reverted

- **PR #336** (slice_num 64→128, val_avg=139.83 single seed) was reverted on commit `605b439` after direct apples-to-apples evidence (PRs #329 and #338) showed slice_num=128 loses by 10-20 MAE inside the 30-min wall-clock cap. slice_num=128 may convert better with longer wall-clock; revisit in round 2 if `SENPAI_TIMEOUT_MINUTES` increases.
