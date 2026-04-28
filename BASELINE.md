# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #401 (torch.compile + bf16 + EMA + grad clip) is the current best — by far.

> **Round-1 budget caveat (revised after #401).** `SENPAI_TIMEOUT_MINUTES=30` is still binding, but with `torch.compile(mode=reduce-overhead, dynamic=True)` on top of bf16, per-epoch wall-clock dropped from 141 s → 55 s. **Round 1 is now a ~33-epoch ranking exercise** — the cosine schedule actually enters its decay tail and EMA has time to do its job. The bottleneck has shifted from "compute-bound" to "architecture and effective EMA horizon". Future architectural-scale PRs (wider, deeper) that previously couldn't fit the budget should be revisited.

## Current best (PR #401, alphonse, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **66.89** (EMA-evaluated) | 33 / 50 (timeout-capped, still descending) |
| `test_avg/mae_surf_p` | **57.86** (EMA-evaluated) | best ckpt = epoch 33 |
| Per-epoch wall-clock | **54.6 s** (median) | 1.76× faster than #372 (bf16-only); 2.58× faster than #308 |
| Total epochs in budget | **33** | vs 13 at #308, 19 at #372 |
| Peak GPU memory | 23.8 GB | (vs 42.1 GB at bf16-only — large headroom for capacity) |

### Per-split val (epoch 33, EMA weights)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     |  75.99 |
| val_geom_camber_rc     |  77.53 |
| val_geom_camber_cruise |  48.50 |
| val_re_rand            |  65.51 |

### Per-split test (best EMA checkpoint, post-fix scoring)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     |  63.90 |
| test_geom_camber_rc     |  70.64 |
| test_geom_camber_cruise |  40.68 |
| test_re_rand            |  56.21 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50` (compile + bf16 + EMA + clip all in `train.py` from #401, #372, #381 stack).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50). At 33 epochs we reach 66% of decay — first time we're in the cosine tail. |
| Batch size | 4 |
| Surf weight | 10.0 (published default) |
| Epochs (configured / completed) | 50 / ~33 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Loss | MSE in normalized space, `vol + surf_weight * surf` |
| **EMA** | decay=0.995; eval + test use EMA weights. EMA crosses online at epoch 2. |
| **Grad clip** | max_norm=10.0 |
| **bf16 autocast** | wraps `model({"x":x_norm})["preds"]` in train + eval (from #372) |
| **torch.compile** | `mode="reduce-overhead", dynamic=True` on both live and EMA model (from #401). 3 unique dynamo graphs cover all mesh sizes; first compile + forward = 8.8 s; steady-state 54.6 s/epoch. |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-compile-bf16-emaclip-20260428-005532/metrics.jsonl`

> **Round-2 implication.** The compile-driven epoch-budget recovery is the dominant mechanism behind the −37% jump. Future PRs that previously closed for blowing per-epoch budget (wider/deeper architectures) should be revisited. Memory headroom is also large (23.8 GB / 96) so capacity-side experiments have plenty of room.

## Compoundable wins still on the table

PR #287 (surf_weight=25) was merged independently before #308 landed; the artifact files are in `models/model-surf-weight-25-20260427-225335/`. **The two changes are orthogonal** — combining surf_weight=25 with EMA+clip is a likely round-2 candidate.

## Update history

| PR | val_avg/mae_surf_p | Notes |
|---|---|---|
| #287 (merged) | 126.67 | surf_weight 10→25, 14/50 epochs, timeout-capped. |
| #308 (merged) | 106.40 | EMA(0.999) + grad clip 1.0, 13/50 epochs, EMA-evaluated. -16.2% vs #287. |
| #372 (merged, infrastructure) | 108.93 (no EMA) | bf16 autocast (1.36× speedup, 19/50 epochs). Treated as infra; baseline anchor stayed at 106.40. |
| #381 (merged) | 98.85 | EMA(0.995) + grad clip 10.0, 13/50 epochs, EMA-evaluated. -7.1% vs #308. EMA crosses online at epoch 2. |
| #401 (merged) | **66.89** | torch.compile(reduce-overhead, dynamic) + bf16 + EMA + clip. **33/50 epochs in budget**, val curve still descending at cap. **-37.1% vs #308**, **-32.3% vs #381**. The throughput-budget recovery is the dominant mechanism. |
