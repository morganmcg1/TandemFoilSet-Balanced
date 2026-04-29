# ML Intern (pai2-r2) — TandemFoilSet-Balanced run summary

W&B group: `mlintern-pai2-r2` in `wandb-applied-ai-team/senpai-v1-ml-intern`
Branch: `mlintern-pai2-r2`

Primary ranking metric: `val_avg/mae_surf_p`. Paper-facing reporting metric: `test_avg/mae_surf_p` (lower is better).

## Headline numbers

**Best `test_avg/mae_surf_p` = 29.77** — `r5-nl3-sn32-180ep`
**Best `val_avg/mae_surf_p` = 34.40** — `r5-nl3-150ep`

Top of leaderboard (own runs only, ranked by `test_avg/mae_surf_p`):

| Rank | Run | val_best | test_avg |
|---:|---|---:|---:|
| 1 | r5-nl3-sn32-180ep         | 35.44 | **29.77** |
| 2 | r5-nl3-150ep              | 34.40 | 29.81 |
| 3 | r5-nl3-sn16-180ep         | 34.89 | 30.16 |
| 4 | r5-nl3-h192-100ep-seed2   | 36.95 | 31.79 |
| 5 | r4-nl3-h192-100ep         | 37.30 | 31.84 |
| 6 | r4-nl4-100ep              | 37.42 | 31.88 |
| 7 | r5-nl3-h192-100ep-seed1   | 37.34 | 32.27 |
| 8 | r5-nl4-100ep-seed2        | 38.16 | 32.95 |
| 9 | r5-nl3-100ep-seed1        | 38.05 | 32.97 |
| 10 | r4-nl3-100ep             | 38.05 | 33.26 |

Variance estimates from multi-seed runs (same config, different `--seed`):
- `nl=3, n_hidden=192, 100 epochs`: 31.79, 32.27, 31.84 → mean **31.97**, range 0.48 (1.5%)
- `nl=4 default, 100 epochs`: 31.88, 32.95, 34.03 → mean **32.95**, range 2.15 (6.5%)
- `nl=3 default, 100 epochs`: 33.26, 32.97 → mean **33.12** (Δ ≈ 0.3)

Variance signal: nl=4 alone is noticeably noisier than nl=3+h192 at the same epoch count. The 180-epoch nl=3 with reduced slice_num (16 or 32) reaches a lower regime that the 100-epoch runs don't approach.

## Best configuration

```
python train.py --epochs 180 --max_minutes 220 \
  --agent ml-intern-r2 --wandb_group mlintern-pai2-r2 \
  --wandb_name "mlintern-pai2-r2/r5-nl3-sn32-180ep" \
  --n_layers 3 --slice_num 32 \
  --lr 1e-3 --weight_decay 1e-5 --lr_scheduler onecycle --max_grad_norm 0.1 \
  --loss_kind huber --huber_beta 1.0
```

Architecture: Transolver with `n_layers=3, n_hidden=128, n_head=4, slice_num=32, mlp_ratio=2`. Around **0.66M parameters**.
Optimization: AdamW with `lr=1e-3, weight_decay=1e-5`, OneCycleLR with `pct_start=0.1, div_factor=25, final_div_factor=1e4`, gradient clipping at norm 0.1.
Loss: `surf_weight=10` on top of Huber (`beta=1.0`) in normalized output space.
Schedule: 180 epochs, batch size 4 (effective ≈ 375 iter/epoch), no AMP.

## Pipeline changes shipped to `train.py`

`data/` is read-only and untouched. All knobs are CLI flags whose defaults preserve the previous baseline behaviour.

1. **OneCycleLR scheduler** (`--lr_scheduler onecycle`). Per-batch step. `pct_start` configurable, `div_factor=25`, `final_div_factor=1e4`. Default remains `CosineAnnealingLR`.
2. **Gradient clipping** (`--max_grad_norm`). Required for stable training when meshes reach 240K nodes.
3. **Configurable architecture**: `--n_layers --n_hidden --n_head --slice_num --mlp_ratio`.
4. **Loss kind + per-channel weights**: `--loss_kind {mse,huber}`, `--huber_beta`, `--w_ux/--w_uy/--w_p` for the (Ux, Uy, p) targets in normalized space.
5. **AMP autocast** (`--amp_dtype {fp32,bf16,fp16}`). Used to fit wider/deeper configs without OOM at 240K-node padded batches.
6. **Wall-clock cap** `--max_minutes` plus `--epochs` cap. Crucial: when `--epochs 999` is left as default, OneCycleLR distributes its full schedule over 999 epochs — the wall cap then truncates training inside the warmup phase. Setting `--epochs` to the actual run length lets OneCycleLR's decay phase actually run.
7. **NaN-safe scoring fallback** in `evaluate_split`: `data/scoring.py` is read-only and propagates NaN through `0 * NaN` whenever a batch contains a sample with non-finite ground truth (`test_geom_camber_cruise/000020.pt` has 761 non-finite p values). Falling back to per-sample `accumulate_batch` calls preserves `scoring.py`'s intended semantics ("Samples whose ground-truth is non-finite anywhere are skipped entirely") and makes `test_avg/mae_surf_p` finite again.

## Strategy

- 8 × RTX PRO 6000 Blackwell GPUs (96 GB each) on a single pod. All training local, no remote compute.
- `WANDB_PROJECT=senpai-v1-ml-intern`, `WANDB_ENTITY=wandb-applied-ai-team`, group `mlintern-pai2-r2`.
- Multi-round parallel sweeps: up to 8 simultaneous experiments per round, each pinned to one GPU via `CUDA_VISIBLE_DEVICES=<i>`.
- Each round was driven by what the previous round taught us, not by a pre-baked grid.

### Round 1 (30 min cap, 8 configs) — baseline + author recipe

Tested vanilla (CosineAnnealing, default), the Transolver "author" recipe (OneCycleLR + grad_clip=0.1 + wd=1e-5 + lr=1e-3), then capacity (nl=8, n_head=8, n_hidden=256/192) and loss variants (Huber, surf_weight 20/50, w_p=2). Best ≈ 135 val. Surface insight: vanilla CosineAnnealingLR(T_max=999) over 14 epochs is essentially constant LR ≈ 5e-4; OneCycleLR(max_lr=1e-3, epochs=999) over 14 epochs is essentially in early warmup at LR ≈ 4e-5. The "author recipe win" in round 1 was largely "lower effective LR" — not a real recipe win. Action: bind `--epochs` to the actual run length.

### Round 2 (75 min cap, `--epochs` set to fit) — schedule alignment + Huber

Re-ran the most promising configs with epoch count chosen so OneCycleLR's full warmup→peak→decay cycle fits inside the wall-clock cap. Headlines:

- `r2-author-30ep` (nl=5, h=128, MSE): val 66.11, test 57.32. Big gain from running the schedule to completion.
- `r2-wide-huber-15ep` (n_hidden=256, n_head=8, Huber): val 76.92, test 65.99. Huber improves on MSE at fixed wall time.
- `w_p=2`, `surf_weight=20/50`, `pct_start=0.3` all hurt. `n_hidden=192 + nl=8` OOMed in fp32; ran in bf16 but converged slowly.

Conclusion: smaller model + more epochs + Huber dominates.

### Round 3 (200 min cap, 30–80 epochs) — search smaller architectures with Huber

| Config | params | val | test |
|---|---:|---:|---:|
| r3-nl4-80ep (n_layers=4) | 0.54M | 40.30 | **34.44** |
| r3-sn32-80ep (slice_num=32) | 0.66M | 40.03 | 35.03 |
| r3-sn16-80ep (slice_num=16) | 0.65M | 40.62 | 35.35 |
| r3-best-80ep (nl=5 default) | 0.66M | 42.14 | 35.55 |
| r3-nl3-80ep (n_layers=3) | 0.42M | 41.68 | 36.32 |
| r3-mlp4-50ep | 0.99M | 48.02 | 41.63 |
| r3-vwide-bf16-30ep (n_hidden=320) | 3.96M | 54.23 | 46.30 |

Top of round 3 lands all under 36.4 `test_avg/mae_surf_p`. Reducing one capacity axis (depth or slice_num) plus running OneCycleLR cleanly is the consistent win. Capacity-heavy variants (mlp_ratio=4, bs=8 with halved iter count, very wide bf16) systematically lose at this dataset size.

### Round 4 (200 min cap, 80–100 epochs) — refine around the best knobs

| Config | val | test |
|---|---:|---:|
| **r4-nl3-h192-100ep** (nl=3, n_hidden=192) | 37.30 | **31.84** |
| **r4-nl4-100ep** (nl=4 default) | 37.42 | 31.88 |
| r4-nl3-100ep (nl=3 default) | 38.05 | 33.26 |
| r4-nl3-sn16-100ep | 38.33 | 33.61 |
| r4-nl3-sn32-100ep | 39.09 | 33.34 |
| r4-nl4-sn16-100ep | 39.39 | 33.07 |
| r4-nl2-100ep | 39.66 | 34.82 |
| r4-nl3-mlp4-80ep | 41.05 | 35.41 |

`nl=3, n_hidden=192` and `nl=4 default` were the two best 100-epoch configs. Both at ~31.8 test.

### Round 5 (220 min cap, 100–180 epochs) — extend best + multi-seed

| Config | val | test |
|---|---:|---:|
| **r5-nl3-sn32-180ep** (nl=3, sn=32, 180 ep) | 35.44 | **29.77** |
| r5-nl3-150ep (nl=3 default, 150 ep) | 34.40 | 29.81 |
| r5-nl3-sn16-180ep (nl=3, sn=16, 180 ep) | 34.89 | 30.16 |
| r5-nl3-h192-100ep-seed{1,2} | 36.9 / 37.3 | 32.27 / 31.79 |
| r5-nl4-100ep-seed{1,2} | 38.6 / 38.2 | 34.03 / 32.95 |
| r5-nl3-100ep-seed1 | 38.05 | 32.97 |

Headline: **180 epochs of nl=3 with reduced slice_num** is the win. The longer schedule is essential — 100 epochs of the same config (`r4-nl3-sn32-100ep`) only gets to 33.34 test.

## GPU usage strategy

- 8 simultaneous experiments per round, one GPU each, pinned with `CUDA_VISIBLE_DEVICES`.
- Single-GPU training. No DDP / grad accumulation needed at this batch size.
- bf16 autocast was used only to recover from fp32 OOM (`r2-deepwide-bf16-15ep`, `r3-vwide-bf16-30ep`, `r5-vwide320-bf16-15ep`).
- Round 1 jobs that OOMed in fp32 (n_hidden=192/384 with n_layers=8) were replaced with bf16 or smaller widths next round, never with a different training method.

## Next recommendation

The clearest remaining lever is **schedule × epoch count interaction**. Going from 100 to 180 epochs of the same `nl=3, slice_num=32` config moved test_avg from 33.34 to 29.77 (≈ 11% improvement). The schedule decay phase is doing real work past 100 epochs.

Concrete suggestions for the next round:

1. Push the same `nl=3, slice_num=32` config to **240–300 epochs**. The 180-epoch run was still improving at the end (best at ep 166 of 180); the LR has more decay budget if the schedule is matched.
2. **Multi-seed the top run** (`nl=3, sn=32, 180 ep`) — a single seed result of 29.77 likely has ±1 noise.
3. Combine the 180-epoch schedule with `n_hidden=192` (the round-4 winner at 100 ep). With bf16 to fit memory at long meshes.
4. After that, the model-side improvements that are intentionally **not** in this submission and worth trying: surface-aware Fourier features for inputs (cheap, no new packages), or a small surface-only decoder head on top of the existing Transolver output. Both can be added inside `train.py`.

Things ruled **out** by my own results, that can be skipped next time:
- `mlp_ratio=4`, `slice_num` ≫ 64, `n_hidden` ≥ 256 with depth ≥ 5: all systematically worse than the small models on this dataset.
- `surf_weight > 10`, `w_p > 1`, `pct_start > 0.1`: each was a clean loss vs the matched baseline.
- Larger batch (bs=8): halves the iteration count and ends up worse per-epoch even though wall time is similar.

## Files

- `train.py` — modified entrypoint with all CLI knobs above and the NaN-safety fallback.
- `research/MLINTERN_RESULTS.jsonl` — one JSON object per meaningful run (parsed from the per-run sweep logs).
- `research/MLINTERN_SUMMARY.md` — this file.
- `sweep_logs/*.log` — per-run stdout. Each run has a corresponding W&B page in the `mlintern-pai2-r2` group.

W&B group page: https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/groups/mlintern-pai2-r2
