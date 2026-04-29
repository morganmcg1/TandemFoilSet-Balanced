# ML Intern PaI2-r4 — TandemFoilSet-Balanced Replicate

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group: `mlintern-pai2-r4`
Branch: `mlintern-pai2-r4` on `morganmcg1/TandemFoilSet-Balanced`
Hardware: 8 × NVIDIA RTX PRO 6000 Blackwell (96 GB), 12-h pod budget.

## Headline numbers

| | val_avg/mae_surf_p | test_avg/mae_surf_p | model |
|---|---:|---:|---|
| R1 baseline (default config, 14 ep) | 124.90 | — | n_h=128, n_l=5, default |
| **Best single model (R4 winner)** | **42.53** | **36.50** | r4-w30-sw5-70 (model-kmzi8u63) |
| **5-model ensemble (R4 default-arch)** | **38.64** | **32.91** | average of 5 R4 default-arch checkpoints |

Compared to the R1 baseline (default config trained for 30 min): a **66% reduction
in val_avg/mae_surf_p** for the best single model, and a **69% reduction** for
the 5-model ensemble.

## TL;DR strategy

The first thing I checked was that the baseline arch is fine — but the **default
LR schedule was wildly mis-scoped** for short runs. With `--epochs 100` and
`CosineAnnealingLR(T_max=epochs)` over a 14-epoch wall-clock cap, the LR is
nearly constant and the schedule contributes nothing. Same problem with
`OneCycleLR` if I forgot to align `total_steps`.

The single biggest lever was switching to the canonical Transolver-paper
recipe **and matching `--epochs` to the wall-clock budget so the schedule
actually warmups, peaks, and cools down inside the run**:

- optimizer: `Adam` (not AdamW)
- scheduler: `OneCycleLR` (`anneal_strategy='cos'`, `final_div_factor=1000`)
- `max_lr = 1e-3`, `pct_start = 0.3`, `grad_clip = 1.0`
- `--epochs` set to roughly the number of epochs that will fit in budget

That single change (no architectural modification) takes
`val_avg/mae_surf_p` from **124.90** (R1, default, 14 epochs/30 min) down to
**55.20** (R3, recipe + warmup_pct=0.3, 40 epochs/60 min) and finally to
**42.53** (R4, recipe + sw5 + warmup_pct=0.3, 70 epochs/180 min).

## Iteration history

| Round | Walltime / job | Goal                                        | Best val / test |
|------:|:---------------|:--------------------------------------------|----------------:|
| R1    | 30 min × 8     | Compare baseline / arch / surf_w / batch    | 124.90 / —      |
| R2    | 60 min × 8     | Re-test recipe with **properly scoped** LR  | 69.68 / 60.94   |
| R3    | 90 min × 8     | Sweep recipe variants (warmup, sw, arch_m)  | 55.20 / 48.08   |
| R4    | 180 min × 8    | Long runs of best recipe + variants + seeds | **42.53 / 36.50** |

### Round 1 — find the right axis
Default config (`n_hidden=128, n_layers=5, n_head=4, slice_num=64`,
`AdamW`, `CosineAnnealingLR(T_max=epochs)`, `lr=5e-4`) reached **124.90** in
14 epochs / 30 min.
- Bigger arches (`192/6/8/32` and `256/8/8/32 + grad_ckpt`) were too slow to
  converge in 30 min and lost.
- `surf_weight ∈ {2, 5, 20}` made small differences at this short timescale.
- The “paper recipe” lost too — *but only because I had passed `--epochs 999`*,
  so the OneCycle schedule was effectively in pure warmup throughout.

Lesson: the schedule horizon must match the training horizon, otherwise
schedule changes can’t be evaluated.

### Round 2 — properly-scoped schedules
All cosine / OneCycle schedules built with `total_steps` matched to the run
length (`--epochs 27` or `16`).
- **recipe-cos27** (Adam + OneCycle + clip + lr=1e-3 + warmup_pct=0.05) drops
  to **69.68** — clearly the new winner.
- `arch-m-recipe-cos16` reaches 83.66 (1.70 M params, only 16 epochs, still
  improving) — bigger arch is promising if extended.
- Plain cosine baseline at 27 epochs only reaches 94.26.

### Round 3 — refine the recipe
- **Best:** recipe + `warmup_pct=0.3` + `--epochs 40` → **55.20**.
- Other recipe variants (sw5, sw20, lr=2e-3, longer schedule recipe60) all
  cluster in the 58–60 band.
- recipe + arch_m, ep=25 → 68.98 — still improving when budget hit.
- recipe + arch_l_gc, ep=12 → 90.88 — bottlenecked by epoch count.

The longer warmup (30%) was a surprise. Hypothesis: the heavy `surf_weight=10`
in the loss makes the early steps very sharp; a longer warmup avoids
overshooting in those steps.

### Round 4 — extended training of the leaders
180-minute jobs centred on the R3 winner:
- **recipe + sw5 + warmup30 + 70 epochs → 42.53 val / 36.50 test**.
- Plain recipe + warmup30 + 70 epochs (no sw): 43.61 val / 37.47 test.
- Longer warmup_pct=0.5 + 70 epochs: 43.59 val / 37.88 test.
- Two seeds of the leader: 44.49 / 43.83 val (variance ≈ ±1 point).
- recipe + arch_m + 40 epochs: 51.09 val / 44.38 test — capacity didn't beat
  the longer-trained default arch in this budget.

### Ensemble (5 model average)
Averaging the predictions of the five R4 default-arch winners (kmzi8u63,
2uwed6m8, qurgdnbf, m5vmvmce, 1857hy2h) cuts test from **36.50 → 32.91** (≈ 10 %
relative). Adding R3 models or larger-arch models hurt the ensemble.

| Ensemble                    | val   | test  |
|-----------------------------|------:|------:|
| Top-3 R4 default-arch       | 39.69 | 34.02 |
| Top-4 R4 default-arch       | 38.97 | 33.17 |
| **Top-5 R4 default-arch**   | **38.64** | **32.91** |
| Top-6 (5 R4 + arch_m+sw5)   | 39.25 | 33.23 |
| Top-8 (5 R4 + 3 R3 default) | 41.54 | 35.31 |

## Best single model — full per-split breakdown

`r4-w30-sw5-70` (W&B run `kmzi8u63`, model dir `models/model-kmzi8u63`):

```
--- VAL ---  avg/mae_surf_p = 42.5268
  val_single_in_dist        surf[p=42.69 Ux=0.51 Uy=0.30]  vol[p=47.17 Ux=2.08 Uy=0.76]
  val_geom_camber_rc        surf[p=57.25 Ux=0.93 Uy=0.47]  vol[p=60.41 Ux=2.83 Uy=1.30]
  val_geom_camber_cruise    surf[p=25.71 Ux=0.37 Uy=0.22]  vol[p=27.41 Ux=1.48 Uy=0.51]
  val_re_rand               surf[p=44.45 Ux=0.67 Uy=0.34]  vol[p=44.36 Ux=2.03 Uy=0.86]

--- TEST --- avg/mae_surf_p = 36.4991  (1 sample skipped: non-finite GT in p)
  test_single_in_dist       surf[p=37.64 Ux=0.53 Uy=0.31]  vol[p=44.19 Ux=1.95 Uy=0.74]
  test_geom_camber_rc       surf[p=51.14 Ux=0.86 Uy=0.44]  vol[p=54.79 Ux=2.67 Uy=1.19]
  test_geom_camber_cruise   surf[p=21.56 Ux=0.34 Uy=0.19]  vol[p=23.24 Ux=1.37 Uy=0.46]
  test_re_rand              surf[p=35.65 Ux=0.55 Uy=0.30]  vol[p=36.97 Ux=1.83 Uy=0.73]
```

Note: `data/scoring.accumulate_batch` correctly *flags* non-finite samples in
GT, but the unmasked elementwise `(pred - y).abs()` propagates `Inf * 0 → NaN`
into the running sum. `scripts/eval_test.py` works around this by dropping
such samples at the batch boundary before forward pass.

## Best single command

```bash
SENPAI_TIMEOUT_MINUTES=180 CUDA_VISIBLE_DEVICES=N python train.py --skip_test \
    --agent ml-intern-r4 \
    --wandb_group mlintern-pai2-r4 \
    --wandb_name "mlintern-pai2-r4/r4-w30-sw5-70" \
    --epochs 70 --warmup_pct 0.3 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0
```

## Compute strategy

- 8 GPUs in parallel, one job per GPU, pinned via `CUDA_VISIBLE_DEVICES`.
- Round budgets: 30 → 60 → 90 → 180 min. Each round chosen to give the
  previous round’s winner *more epochs of training*, not to introduce a new
  lever.
- All training **local to the pai2 pod**; no HF Jobs / Sandboxes / Spaces.
- The repo’s W&B integration is left unchanged — no Trackio.
- One eval-only sweep at the end on the saved checkpoints to compute test
  metrics + run ensemble.

## Code changes (`train.py`)

`train.py` is the only training-code file changed (data loaders + scoring are
read-only per the benchmark rules). New CLI levers preserve original defaults:

| Group | Flag | Default | Notes |
|-------|------|---------|-------|
| Architecture | `--n_hidden`, `--n_layers`, `--n_head`, `--slice_num`, `--mlp_ratio`, `--dropout`, `--unified_pos`, `--ref` | match original | – |
| Memory | `--grad_checkpoint` | `False` | wraps attention+MLP per block in `torch.utils.checkpoint` |
| Optimizer | `--optimizer` | `adamw` | `adam` matches Transolver paper |
| Scheduler | `--scheduler` | `cosine` | `onecycle` matches Transolver paper; uses `--lr` as `max_lr` and `--warmup_pct` as `pct_start` |
| Reg | `--grad_clip` | `0.0` | uses `clip_grad_norm_` if > 0 |
| Repro | `--seed` | `0` | seeds CPU + CUDA |

Helper scripts:
- `scripts/aggregate_results.py` — parse `logs/*.log` into `MLINTERN_RESULTS.jsonl`.
- `scripts/eval_test.py` — re-evaluate a checkpoint on val + test (NaN-safe).
- `scripts/eval_ensemble.py` — average predictions across multiple checkpoints.

## Next recommendations

1. **Train the leader for longer.** All five R4 default-arch winners are still
   improving at epoch 70. Pushing to 100–120 epochs (≈ 4 h) should knock more
   off both val and test. If it doesn’t plateau, also try `--epochs 200` with
   the same OneCycle schedule.
2. **Replicate-and-ensemble.** With 8 GPUs free, training 8 seeds of
   `r4-w30-sw5-70` and ensembling at the end is the highest-EV thing to do
   next. Empirically the 5-model ensemble already moves test from 36.5 → 32.9.
3. **Per-iteration random node subsampling** (Transolver paper canonical
   training; subsample to 32 K nodes per sample). Frees ~4× memory headroom
   *and* acts as regularisation. Likely best for the OOD splits
   (`val_geom_camber_*`).
4. **Fix `data/scoring.py`'s `Inf * 0 → NaN` bug.** When the user accepts
   `data/` changes again, masking err *before* the multiply removes the need
   for the workaround in `eval_test.py` and stops 20 % of legitimate runs from
   reporting NaN test averages (per the README leaderboard's note that 90/450
   finished runs lost their test score this way).

The `--epochs N` ↔ schedule-horizon coupling is the single most important
lesson from this replicate: every other lever moves a few percent; getting the
schedule horizon right moved 56 %.
