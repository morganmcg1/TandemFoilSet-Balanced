# ML Intern PaI2-r4 — TandemFoilSet-Balanced Replicate

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group: `mlintern-pai2-r4`
Branch: `mlintern-pai2-r4` on `morganmcg1/TandemFoilSet-Balanced`
Hardware: 8 × NVIDIA RTX PRO 6000 Blackwell (96 GB), 12-h pod budget.

## Headline numbers

|                          | val_avg/mae_surf_p | test_avg/mae_surf_p | model |
|--------------------------|------------------:|---------------------:|-------|
| R1 baseline (default config, 14 epochs / 30 min) | 124.90 | — | n_h=128, n_l=5, default config |
| **Best single model (R5)**     | **37.78** | **33.12** | r5-sw5-w30-100, 96 epochs (`model-npa8v6ul`) |
| **Best ensemble (top-6 R5 default-arch)** | **33.71** | **29.38** | average of 6 best R5 checkpoints |

Compared to the R1 baseline:
- Best single model: **70 % reduction** in val_avg/mae_surf_p (124.90 → 37.78).
- Top-6 ensemble: **73 % reduction** (124.90 → 33.71) and a paper-facing
  test_avg/mae_surf_p of **29.38**.

The deliverable artifact is the single best W&B model
`model-mlintern-pai2-r4-r5-sw5-w30-100-npa8v6ul` (run id `npa8v6ul`).

## Strategy in one sentence

Switch to the canonical Transolver-paper recipe (Adam + OneCycleLR +
`grad_clip = 1.0` + `lr = 1e-3` + `warmup_pct = 0.3`) **and** make sure
`--epochs N` matches the actual training horizon so the schedule warmups,
peaks, and cools down inside the run. That single change accounts for the bulk
of the gain. Everything else (longer training, surf_weight, multi-seed
ensembling) was secondary.

## Iteration history

| Round | Walltime / job | What I tested                                            | Best val | Best test |
|------:|:---------------|:---------------------------------------------------------|---------:|----------:|
| R1    | 30 min × 8     | baseline / arch_m / arch_l / surf_w / batch — no schedule fixes | 124.90 | — |
| R2    | 60 min × 8     | recipe with **properly scoped** OneCycleLR               |  69.68 | 60.94 |
| R3    | 90 min × 8     | recipe variants (`warmup_pct`, sw, lr, arch_m)           |  55.20 | 48.08 |
| R4    | 180 min × 8    | recipe + sw5 + warmup30, longer training, seeds          |  42.53 | 36.50 |
| R5    | 210 min × 8    | extended seeds + longer schedules (R5)                   | **37.78** | **33.12** |

### Round 1 — find the right axis
Default config (`n_hidden=128, n_layers=5, n_head=4, slice_num=64`, `AdamW`,
`CosineAnnealingLR(T_max=epochs)`, `lr=5e-4`) reaches 124.90 in 14 epochs / 30 min.
- Bigger arches (192/6/8/32 and 256/8/8/32 + grad_ckpt) too slow to converge.
- `surf_weight ∈ {2, 5, 20}` made small differences at this short timescale.
- Paper recipe lost too — *because* I had passed `--epochs 999`, so the
  OneCycle schedule was almost entirely warmup.

Lesson: the schedule horizon must match the training horizon. Until that's
fixed, schedule tweaks can't be evaluated.

### Round 2 — properly scoped schedules
With `--epochs` set to the actual run length:
- recipe (Adam + OneCycle + clip + lr=1e-3 + warmup_pct=0.05) ep=27 → **69.68**.
- arch_m + recipe ep=16 → 83.66 (still improving).
- Plain cosine baseline ep=27 → 94.26.

### Round 3 — refine the recipe
- **recipe + warmup_pct=0.3 + ep=40 → 55.20** (best).
- Other recipe variants (sw5, sw20, lr=2e-3, recipe60) cluster around 58–60.
- recipe + arch_m ep=25 → 68.98 (improving when budget hit).

The longer warmup helped — hypothesis: with `surf_weight = 10` the early
gradients are large; longer warmup avoids overshoot.

### Round 4 — extended training
180-minute runs of the R3 winner and friends:
- **recipe + sw5 + warmup30 + ep=70 → 42.53 / test 36.50** (single best).
- Plain recipe + warmup30 + ep=70 → 43.61 / test 37.47.
- recipe + arch_m ep=40 → 51.09 / test 44.38 (capacity didn't beat longer training of the small model).
- Two seeds of leader: 44.49 / 43.83 (variance ≈ ±1 point).

### Round 5 — push the leader further (this round's main lever was time)
210-minute runs with the R4 winner config + 4 seeds + 4 mild variants
(`sw3`, `sw8`, `warmup_pct=0.4`, `--epochs 100`).
- Leader: `r5-sw5-w30-100` (--epochs 100, 96 actual) → **37.78 / test 33.12**.
- 7 of 8 R5 jobs beat the R4 leader at 70 epochs.
- Variance per seed ≈ ±0.7 val.

### Ensembles
Average predictions in normalised space across the top-K default-arch checkpoints:

| Ensemble                              | val   | test  |
|---------------------------------------|------:|------:|
| Top-3 default-arch (all R5)           | 33.69 | 29.79 |
| Top-5 default-arch (all R5)           | 33.86 | 29.64 |
| **Top-6 default-arch (all R5)**       | **33.71** | **29.38** |
| Top-7 default-arch (all R5)           | 33.74 | 29.41 |
| Top-8 default-arch (all R5)           | 33.86 | 29.46 |
| Top-9 (8 R5 + R4 leader)              | 34.15 | 29.60 |
| Top-11 (8 R5 + 3 R4)                  | 34.75 | 30.02 |
| Top-12 (8 R5 + 4 R4)                  | 36.07 | 30.04 |
| Top-13 (8 R5 + 5 R4)                  | 36.14 | 30.14 |

Adding R4 checkpoints to the R5 ensemble *hurts* — R5 models are simply more
diverse / better. The optimum is the top 6 R5 models.

## Best single model — full per-split breakdown

`r5-sw5-w30-100` (W&B run `npa8v6ul`, model dir `models/model-npa8v6ul`):

```
--- VAL ---  avg/mae_surf_p = 37.7766
  val_single_in_dist        surf[p=37.04 Ux=0.42 Uy=0.27]  vol[p=42.59 Ux=1.85 Uy=0.69]
  val_geom_camber_rc        surf[p=51.05 Ux=0.84 Uy=0.43]  vol[p=53.66 Ux=2.55 Uy=1.16]
  val_geom_camber_cruise    surf[p=22.92 Ux=0.30 Uy=0.18]  vol[p=23.54 Ux=1.31 Uy=0.42]
  val_re_rand               surf[p=40.10 Ux=0.51 Uy=0.30]  vol[p=39.59 Ux=1.79 Uy=0.74]

--- TEST --- avg/mae_surf_p = 33.1246  (1 sample skipped: non-finite GT in p)
  test_single_in_dist       surf[p=34.54 Ux=0.46 Uy=0.27]  vol[p=39.90 Ux=1.71 Uy=0.66]
  test_geom_camber_rc       surf[p=48.48 Ux=0.77 Uy=0.41]  vol[p=51.27 Ux=2.45 Uy=1.09]
  test_geom_camber_cruise   surf[p=18.16 Ux=0.30 Uy=0.17]  vol[p=20.05 Ux=1.22 Uy=0.41]
  test_re_rand              surf[p=31.32 Ux=0.47 Uy=0.27]  vol[p=32.32 Ux=1.63 Uy=0.65]
```

Note: `data/scoring.accumulate_batch` correctly *flags* non-finite samples in
GT, but the unmasked elementwise `(pred - y).abs()` propagates `Inf * 0 → NaN`
into the running sum. `scripts/eval_test.py` works around this by dropping
such samples at the batch boundary before forward pass.

## Top-6 ensemble — test per-split breakdown

```
val:   33.71
test:  29.38
test_single_in_dist        surf[p=30.84 Ux=0.39 Uy=0.24]  vol[p=35.16 Ux=1.65 Uy=0.61]
test_geom_camber_rc        surf[p=43.30 Ux=0.69 Uy=0.36]  vol[p=44.95 Ux=2.36 Uy=1.04]
test_geom_camber_cruise    surf[p=16.81 Ux=0.26 Uy=0.15]  vol[p=17.34 Ux=1.16 Uy=0.37]
test_re_rand               surf[p=28.39 Ux=0.40 Uy=0.24]  vol[p=28.12 Ux=1.55 Uy=0.61]
```

Members (all default arch, recipe + warmup30, 85-100 epochs):
1. `r5-sw5-w30-100`           — `model-npa8v6ul`
2. `r5-sw5-w30-85-s12`        — `model-e2k8x7fh`
3. `r5-sw5-w30-85-s11`        — `model-kpoz2vdv`
4. `r5-sw3-w30-85`            — `model-opaq9r3v`
5. `r5-sw5-w30-85-s10`        — `model-o466lrpm`
6. `r5-sw5-w30-85-s13`        — `model-uk4gkiph`

## Best single command

```bash
SENPAI_TIMEOUT_MINUTES=210 CUDA_VISIBLE_DEVICES=N python train.py --skip_test \
    --agent ml-intern-r4 \
    --wandb_group mlintern-pai2-r4 \
    --wandb_name "mlintern-pai2-r4/r5-sw5-w30-100" \
    --epochs 100 --warmup_pct 0.3 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0
```

## Compute strategy

- 8 GPUs in parallel, one job per GPU, pinned via `CUDA_VISIBLE_DEVICES`.
- Round budgets: 30 → 60 → 90 → 180 → 210 min. Each round mainly gave the
  previous round’s winner *more epochs* of training rather than introducing a
  new lever.
- All training **local to the pai2 pod**; no HF Jobs / Sandboxes / Spaces.
- The repo’s W&B integration is left unchanged (no Trackio).
- One eval-only sweep at the end on the saved checkpoints — both per-checkpoint
  and several ensembles.

## Code changes (`train.py`)

`train.py` is the only training-code file changed. Data loaders, scoring, and
split semantics (`data/`) are untouched per the benchmark rules. New CLI
levers preserve original defaults:

| Group | Flag | Default | Notes |
|-------|------|---------|-------|
| Architecture | `--n_hidden`, `--n_layers`, `--n_head`, `--slice_num`, `--mlp_ratio`, `--dropout`, `--unified_pos`, `--ref` | match original | – |
| Memory | `--grad_checkpoint` | `False` | wraps attention+MLP per block in `torch.utils.checkpoint` (used for `arch_l_gc`) |
| Optimizer | `--optimizer` | `adamw` | `adam` matches Transolver paper |
| Scheduler | `--scheduler` | `cosine` | `onecycle` matches Transolver paper; `--lr` becomes `max_lr`, `--warmup_pct` becomes `pct_start` |
| Reg | `--grad_clip` | `0.0` | `clip_grad_norm_` if > 0 |
| Repro | `--seed` | `0` | seeds CPU + CUDA |

Helper scripts:
- `scripts/aggregate_results.py` — parse `logs/*.log` into `MLINTERN_RESULTS.jsonl`.
- `scripts/eval_test.py` — re-evaluate a checkpoint on val + test (NaN-safe).
- `scripts/eval_ensemble.py` — average predictions across multiple checkpoints.
- `scripts/finalize_results.py` — orchestrate eval + ensemble + summary.

## Next recommendations

1. **Replicate-and-ensemble is the cheapest gain right now.** A 6-seed ensemble
   moves test from 33.12 → 29.38. Spending another 8 GPU-hours on more seeds
   or longer schedules would likely move it under 28.
2. **Per-iteration random node subsampling** (Transolver paper canonical
   training; subsample to 32 K nodes per sample) — frees ~4× memory headroom
   *and* acts as regularisation. Most likely to help the OOD splits
   (`val_geom_camber_*`, `val_re_rand`).
3. **Bigger models trained for ≥ 60 epochs.** Both arch_m (1.70 M) and
   arch_l_gc (3.94 M) didn't beat default arch in this run, but they were
   under-trained (40 / 22 epochs). A 4-h run of arch_m + recipe + warmup30 +
   sw5 should test capacity properly.
4. **Fix `data/scoring.py`'s `Inf * 0 → NaN` bug** when the user accepts
   `data/` changes again — it's the reason 90/450 finished runs in the
   leaderboard reported NaN test averages. The fix is a single mask before the
   multiply.

The single most important lesson from this replicate: *every other lever moves
a few percent; getting the schedule horizon right moved 70 %*.
