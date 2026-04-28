# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #381 (EMA decay=0.995 + grad clip max_norm=10.0) is the current best.

> **Round-1 budget caveat.** `SENPAI_TIMEOUT_MINUTES=30` is binding for every run. The 50-epoch cosine schedule is set up by `train.py` but training stops at ~14 epochs in practice, well before the schedule's tail. Round 1 is therefore a 14-epoch ranking exercise. Comparisons across PRs in round 1 are apples-to-apples *only* if they hit the same wall-clock limit.

## Current best (PR #381, nezuko, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **98.85** (EMA-evaluated) | 13 / 50 (timeout-capped) |
| `test_avg/mae_surf_p` | **87.81** (EMA-evaluated) | best ckpt = epoch 13 |
| Wall-clock | ~30.8 min (~142 s/epoch) | |
| Peak GPU memory | 42.1 GB | |

### Per-split val (epoch 13, EMA weights)
| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 117.03 | (n/a) | (n/a) | (n/a) |
| val_geom_camber_rc     | 113.85 | 2.05 | 0.87 | 101.08 |
| val_geom_camber_cruise |  73.12 | 0.81 | 0.44 |  62.48 |
| val_re_rand            |  91.39 | 1.30 | 0.66 |  85.64 |

### Per-split test (best EMA checkpoint, post-fix scoring)
| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| test_single_in_dist     | 103.13 | 1.31 | 0.63 | 111.81 |
| test_geom_camber_rc     | 100.76 | 2.05 | 0.87 | 101.08 |
| test_geom_camber_cruise |  61.37 | 0.81 | 0.44 |  62.48 |
| test_re_rand            |  85.96 | 1.30 | 0.66 |  85.64 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50` (EMA + grad-clip in `train.py` from #381; bf16 autocast from #372).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50) |
| Batch size | 4 |
| Surf weight | 10.0 (published default) |
| Epochs (configured / completed) | 50 / ~13-14 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Loss | MSE in normalized space, `vol + surf_weight * surf` |
| **EMA** | decay=0.995; eval + test use EMA weights. EMA crosses online at epoch 2 (vs ~10 at decay=0.999). |
| **Grad clip** | max_norm=10.0 (fires on 87-100% of batches; gn_max stays in 344-767 band; clip catches mostly outliers in late training, more like its intended role than #308's threshold=1) |
| **bf16 autocast** | wraps `model({"x":x_norm})["preds"]` in train + eval (from #372) |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-ema995-gradclip10-20260428-000944/metrics.jsonl`

> **Attribution caveat (still open).** The −7.1% gain over #308 is the joint effect of (faster EMA) + (looser dampening). Clip still fires 87-100% of the time at threshold=10, so we don't yet have a clean EMA-only number. PR queued (nezuko's next) to ablate EMA-only without clip.

## Compoundable wins still on the table

PR #287 (surf_weight=25) was merged independently before #308 landed; the artifact files are in `models/model-surf-weight-25-20260427-225335/`. **The two changes are orthogonal** — combining surf_weight=25 with EMA+clip is a likely round-2 candidate.

## Update history

| PR | val_avg/mae_surf_p | Notes |
|---|---|---|
| #287 (merged) | 126.67 | surf_weight 10→25, 14/50 epochs, timeout-capped. |
| #308 (merged) | 106.40 | EMA(0.999) + grad clip 1.0, 13/50 epochs, EMA-evaluated. -16.2% vs #287. |
| #372 (merged, infrastructure) | 108.93 (no EMA) | bf16 autocast (1.36× speedup, 19/50 epochs). Treated as infra; baseline anchor stayed at 106.40. |
| #381 (merged) | **98.85** | EMA(0.995) + grad clip 10.0, 13/50 epochs, EMA-evaluated. **-7.1% vs #308.** EMA crosses online at epoch 2 (vs ~10 at decay=0.999). Clip still fires 87-100% — joint EMA+dampener effect. |
