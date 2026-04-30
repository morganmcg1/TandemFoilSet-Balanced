# ML Intern — TandemFoilSet-Balanced Benchmark Summary

**Branch:** `mlintern-pai2-24h-v2-r1` · **W&B project:** `wandb-applied-ai-team/senpai-v1-ml-intern`
**Date:** 2026-04-29 → 2026-04-30 · **GPU budget:** 8 × NVIDIA RTX PRO 6000 Blackwell (96 GB) · **Wall budget:** 24h

> _This document is updated incrementally during the run. Final version pushed at the end._

## TL;DR

The leading recipe is the **baseline Transolver** (n_hidden=128, n_layers=5,
n_head=4, slice_num=64, mlp_ratio=2) trained with **5 % linear warmup + cosine
decay + grad-clip 1.0 at lr=1e-3** for **100 epochs**.

The single best run is `wc-lr1e3-100ep-seed0` (run id `necjs23h`):

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-----|-------------------:|--------------------:|
| **5-model ensemble** (best 5 wave 3 checkpoints, see below) | **34.33** | **29.33** |
| 2-seed ensemble (`necjs23h` + `pcatusyd`)                   | 36.75 | 31.74 |
| `wc-lr1e3-100ep-seed0` (single best)                        | 39.61 | 34.21 |
| `wc-lr1e3-100ep-seed1`                                      | 39.94 | 34.98 |
| `wc-lr1e3-sw50-100ep-seed0`                                 | 40.50 | 35.34 |
| `wc-lr1e3-sub32k-h256-l8-mlp4-100ep` (big-model variant)    | 40.59 | 35.64 |
| `wc-lr1e3-sub32k-h256-l8-mlp4-fourier-100ep`                | 42.79 | 36.37 |
| `default-lr1e3-warmup-clip` (50 ep, wave 1 baseline)        | 51.05 | 44.19 |
| `baseline-default` (50 ep, vanilla control)                 | 80.68 | 72.93 |

Bottom line: at the same data, same architecture, same optimizer, the difference
between the unmodified baseline and our best **single** run is `72.93 → 34.21`
(53 % reduction in surface-pressure MAE) just from `lr=1e-3 + 5 % warmup +
grad-clip 1.0 + 100 epochs`. The 5-model ensemble pushes this to `29.33`
(60 % reduction).

The ensemble averages predictions from 5 diverse 100-epoch checkpoints:
`necjs23h, pcatusyd, nhj06b8t, rnb24j1e, 8s5o27g0` (small seeds 0/1, small +
sw=50, big sub32k, big sub32k + fourier).

**Surprise findings**:
- Surface-loss-weighted (sw=50, AirfRANS-recommended) helped *less* than just
  doubling the schedule from 50 → 100 epochs.
- Bigger model with mesh subsampling matches but does not beat the small model
  trained for 100 epochs.
- Fourier features + sw=50 *together* hurt — the two regularizers compose
  poorly with this dataset's already-rich shape descriptors.
- Transolver++ (Ada-Temp + Rep-Slice) NaN'd unless temperature is clamped;
  even after the clamp, it did not improve over vanilla within 50 epochs.

**`scoring.py` inf bug**: `data/scoring.py` is read-only per the contract. I
hit a corner-case: `test_geom_camber_cruise/000020.pt` has 761 inf values in
its ground-truth y, and the `inf*0 = nan` propagation poisons the test sum so
the in-training test eval prints `test_avg=nan`. Worked around by filtering
non-finite samples out of the batch *before* calling `accumulate_batch`
inside `evaluate_split` in `train.py` — semantics unchanged (same per-sample
skip rule that `scoring.py` already documents). All test numbers in this
summary come from this fix; `scripts/eval_test.py` re-runs evaluation from
saved checkpoints with the fix applied.

## Strategy

1. **Read the program/SPLITS docs** first; treat `data/` as read-only and only edit `train.py`.
2. **Pull the recipe from the literature** (`Transolver` arXiv 2402.02366,
   `Transolver++` 2502.02414, `MARIO` 2505.14704, `AB-UPT` 2502.09692,
   `DoMINO` 2501.13350, `Transolver-3` 2602.04940) — see
   `research/MLINTERN_RESULTS.jsonl` and the in-line comments in `train.py`.
3. **Expose the relevant hyperparameters** in `train.py` (model size, optimizer,
   schedule, surface weights, Fourier features, Transolver++ ada-temp / rep-slice,
   subsampling, AMP, optional Lion).
4. **Wave-based ablation** on 8 parallel GPUs:
   - **Wave 1**: scan major axes — capacity, Transolver++ flags, pressure weight,
     Fourier PE, warmup+clip, lr=1e-3.
   - **Wave 2**: bigger model + train-time mesh subsampling (32 K nodes/sample)
     to fit larger Transolvers in 96 GB while still using the full mesh at eval.
   - **Wave 3**: confirm the warmup+clip+lr=1e-3 win and combine it with the best
     extras (Fourier, surf_weight=50, subsampling+bigger model). 100-epoch budget.
5. **Hardware pinning**: every job pinned to a specific GPU via `CUDA_VISIBLE_DEVICES`.

## Key changes to `train.py`

These are framework changes; the *default* CLI invocation (no flags) reproduces the
original baseline up to one harmless correctness fix (mask zero-out of padded nodes
in the slice attention).

- **Model knobs as CLI flags**: `n_hidden`, `n_layers`, `n_head`, `slice_num`,
  `mlp_ratio`, `dropout`.
- **Transolver++ ablations** (`use_ada_temp`, `use_rep_slice`) with NaN-safe
  clamps on the per-point temperature and the Gumbel noise tail.
- **Fourier positional encoding** of normalized (x, z) coordinates, gated by
  `fourier_freq` (and `fourier_scale`).
- **Per-channel surface weight** (`surf_p_weight`) plus the global `surf_weight`.
- **Optimizer choice** (`adamw` / `adam` / `lion`) and **LR schedule** (`cosine`
  with optional `warmup_frac`) and **grad clip** (`grad_clip`).
- **AMP** (`bf16` / `fp16`).
- **Train-time mesh subsampling** (`subsample`, `surf_oversample`) with
  surface-biased sampling so big models fit while eval still runs on full meshes.
- **y-axis flip augmentation** function (`yflip_batch`); kept off by default
  because the dataset's z is positive-only and the dsdf features are not
  guaranteed to be sign-invariant — so a free reflection is not a safe symmetry.
- **Mask-aware physics attention** (zeros out padded nodes when forming slice
  tokens) to match the masked downstream loss.
- **Plain stdout epoch lines** with a `disable=True` tqdm so logs are greppable.
- Saved hyperparameters into the W&B model artifact metadata.

`data/`, `data/scoring.py`, the scoring contract, normalization, and the
`{x_norm, mask}` model interface are unchanged.

## Best runs (live, will be finalized)

See `research/MLINTERN_RESULTS.jsonl` for one JSON object per run. Use
`python scripts/collect_results.py` to refresh it from `session_logs/*.log`.

| Rank | Run                        | val_avg/mae_surf_p | Test avg | Status   |
|----:|----------------------------|-------------------:|---------:|----------|
| 1   | default-lr1e3-warmup-clip  | (in progress)      | (pending)| running  |
| 2   | default-warmup-clip        | (in progress)      | (pending)| running  |
| 3   | default-fourier16          | (in progress)      | (pending)| running  |
| 4   | baseline-default           | (in progress)      | (pending)| running  |

## GPU usage

- 8 single-GPU jobs in parallel, each pinned to one GPU via
  `CUDA_VISIBLE_DEVICES`.
- One wave was killed by OOM (h192/l8/mlp4 batch_size=4 fp32 didn't fit in 96 GB
  with full meshes); the wave 1b restart with `batch_size=2` worked. Wave 2's
  subsampling unblocks 8x bigger model dimensions in 23 GB instead of 47 GB peak.

## Next experiments (wave 4 backlog)

- Multi-seed (s=1, s=2) confirmation of the leading config.
- Lion optimizer at the AB-UPT recipe (lr=5e-5, wd=0.05, grad_clip=0.25, 100 ep).
- 200-epoch run on the leading config to see if the cosine tail keeps gaining.
- Smallest meaningful regularizer test: dropout=0.1 to see if there's any
  generalization gap on the unseen-camber tracks.
- Bigger model (n_hidden=256, n_layers=8, mlp_ratio=4) with the warmup+clip
  recipe and subsampling, longer than 100 epochs.
- Final candidate: best of wave 3 retrained from scratch for the test eval.

## Final recommendation (placeholder, finalized at end)

TBD.
