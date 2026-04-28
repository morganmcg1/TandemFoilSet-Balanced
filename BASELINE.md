# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-28 (after PR #343 merged)

## Current best — Round 0, PR #343 (askeladd H6 bf16+compile × FiLM × EMA, Run G)

| Metric | Value | Δ vs prior baseline (PR #442) |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` (active=raw) | **80.91** | **−25.7%** |
| `val_avg_ema/mae_surf_p` (best across epochs) | 81.68 | −25.2% |
| **`test_avg/mae_surf_p`** | **72.73** | **−26.1%** |
| `test/test_single_in_dist/mae_surf_p` | 78.68 | −29.5% |
| `test/test_geom_camber_rc/mae_surf_p` | 84.98 | −24.4% |
| `test/test_geom_camber_cruise/mae_surf_p` | 53.78 | −22.7% |
| `test/test_re_rand/mae_surf_p` | 73.47 | −26.7% |
| Throughput | 53.9 s/epoch (vs ~129 baseline) | **2.4× faster** |
| Epochs in 30-min budget | 34 (vs 14 baseline) | **+143%** |
| Peak GPU memory | 23.6 GB | −44% |

- **W&B run:** [`bi8m16pa`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r4/runs/bi8m16pa) (`willowpai2d4-askeladd/h6-G-on-film-ema`)
- **Best epoch:** 34 of 34 actually trained (run hit 30-min wall clock; cosine reached lr=0.0)
- **Active source at best epoch:** raw (EMA was 81.68 at epoch 33; raw was 80.91 at epoch 34, slightly better)

## Cumulative round-0 progress vs original baseline

| | val_avg/mae_surf_p | test_avg/mae_surf_p |
|--|--|--|
| Original (vanilla Transolver, pre-#344 baseline) | ~125-130 | ~113-119 |
| PR #344 (warmup+cosine+NaN fix) | 120.97 | 109.92 |
| PR #404 (FiLM-on-Re + wd=5e-4) | 119.36 | 107.54 |
| PR #442 (EMA decay=0.99) | 109.19 | 98.47 |
| **PR #343 (bf16+compile)** | **80.91** | **72.73** |

**Cumulative improvement: −33% / −33% over four merges in round 0.** The bf16+compile mechanism (PR #343) is the dominant compounding lever, delivering the largest single-PR gain by enabling 2.4× throughput → 34 epochs vs 14.

## What changed from prior baseline (PR #442)

The merged code on `icml-appendix-willow-pai2d-r4` now includes:

1. **bf16 autocast** wrapping the model forward in both training and evaluation. Loss accumulation stays in fp32 (cast back via `pred.float()`). Defensive fp32 fallback in `evaluate_split` for batches where bf16 produces non-finite preds at masked-in positions (never triggered in practice, kept as safety net).
2. **`torch.compile(mode="default", dynamic=True)`** on the model after instantiation. First-epoch overhead ~13s (compile warmup), then steady ~50s/epoch. `dynamic=True` handles variable mesh sizes without recompilation thrash. `_raw_module()` helper unwraps `OptimizedModule` for state_dict save/load.
3. **`--batch_size`, `--amp_dtype`, `--compile`, `--grad_accum_steps`, `--compile_mode` CLI flags** for full configurability.
4. **Per-optimizer-step scheduler**: scheduler steps once every `grad_accum_steps` batches (correct for any future grad-accum use; identical to merged behavior at grad_accum_steps=1).
5. **Cumulative from PR #442 + #404 + #344:** EMA decay=0.99 + every-other-epoch eval, Re-conditional FiLM modulation, linear warmup + per-step cosine-to-zero schedule, defensive `nan_to_num` in `evaluate_split`, `--seed` CLI flag.

## Recommended training command (reproduces current best)

```bash
cd target/ && python train.py \
    --agent <student-name> \
    --batch_size 4 \
    --amp_dtype bf16 \
    --compile True \
    --film_re True \
    --use_ema True --ema_decay 0.99 --ema_eval_every 2 \
    --epochs 37 \
    --lr 7e-4 \
    --weight_decay 5e-4 \
    --seed 123 \
    --wandb_name "<student-name>/<experiment-tag>"
```

**Important note on `--epochs`:** at the new ~50 s/epoch throughput, ~34-37 epochs fit in 30 min. Setting `--epochs 37` lets cosine reach near zero at the end (lr=0.0 verified at epoch 36 of Run G). Future hypothesis comparisons should use `--epochs 37` as the default; if a hypothesis requires shorter or longer training, motivate the change explicitly.

**EMA's marginal value at this convergence regime is now ~0.** Run G's per-epoch EMA-vs-raw gap narrowed from ~16 pts at epoch 1 to 0.13 pts at epoch 33; raw caught up by epoch 34 and was selected as the active checkpoint. EMA still acts as a defensive measure for any future PR that introduces optimization noise, so we keep it on by default — but don't expect it to add much when the underlying training is already converged.

## Setup recap

| Setting | Value |
|---------|-------|
| Model | Transolver + Re-conditional FiLM (~0.75M params) |
| Optimizer | AdamW, weight_decay=5e-4 |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10 |
| Schedule | Linear warmup (5%) + cosine-to-zero, per-step (`LambdaLR`) |
| Epochs (recommended) | 37 (cosine reaches lr=0 at our 30-min wall clock with bf16+compile) |
| Epochs (default in train.py) | 50, capped by `SENPAI_TIMEOUT_MINUTES=30` |
| Forward dtype | bf16 (`--amp_dtype bf16`) |
| torch.compile | `mode="default", dynamic=True` (`--compile True`) |
| EMA decay | 0.99 (half-life ~0.2 epoch) |
| EMA eval frequency | every 2 epochs (`--ema_eval_every 2`) |
| Recommended `--lr` | 7e-4 |
| Recommended `--seed` | 123 |
| Primary metric | `val_avg/mae_surf_p` — selects best of raw vs EMA |
| Paper metric | `test_avg/mae_surf_p` — uses the model that produced the best val |

## Validation/test splits

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains

## Round-0 history

- **PR #344 (edward H2):** linear warmup + per-step cosine + NaN fix → val=120.97, test=109.92.
- **PR #404 (edward H11):** Re-conditional FiLM + wd=5e-4 → val=119.36, test=107.54 (Run E, seed=123).
- **PR #442 (thorfinn H12):** EMA decay=0.99 + every-other-epoch eval → val_ema=109.19, test=98.47 (Run F, seed=123 on FiLM-merged baseline).
- **PR #343 (askeladd H6):** bf16 + torch.compile (mode=default, dynamic=True) → val=80.91, test=72.73 (Run G, seed=123 on EMA+FiLM baseline).

## Key methodological tooling shipped this round

- **`--seed` CLI flag** (PR #404) — enables seed-controlled comparisons. Reproducibility now demonstrated four times: PR #442 Run F, PR #523 Run A, PR #576 Run A, PR #343 Run G all reproduce their respective baselines to 4 decimals.
- **`--ema_decay`, `--ema_eval_every`, `--ema_warmup_steps` flags** (PR #442) — EMA is default-on with active-checkpoint selection between raw and EMA.
- **`--amp_dtype`, `--compile`, `--compile_mode`, `--grad_accum_steps` flags** (PR #343) — bf16 forward, torch.compile JIT, gradient accumulation all configurable.
- **Defensive `nan_to_num` in `evaluate_split`** (PR #344) — robust against `test_geom_camber_cruise` sample 20's `-inf` GT.
- **bf16 fp32-fallback in `evaluate_split`** (PR #343) — defensive against bf16 producing non-finite preds at masked-in positions.
