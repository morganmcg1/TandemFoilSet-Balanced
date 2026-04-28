# Baseline

The current research baseline on `icml-appendix-willow-pai2d-r2`.

The primary ranking metric is `val_avg/mae_surf_p` — equal-weight mean surface
pressure MAE across the four validation tracks. Lower is better.

## Current configuration (after merging #328 + #330 + #367 + #399)

```
lr=5e-4  weight_decay=1e-4  batch_size=4  surf_weight=10.0  epochs=50
n_hidden=128  n_layers=5  n_head=4  slice_num=128  mlp_ratio=2
optimizer=AdamW  loss=Huber(beta=1.0) on normalized residuals
norm=LayerNorm  amp_dtype=bf16  scoring_guard=nan_to_num
```

### Anchored val metrics

| Source | val_avg/mae_surf_p | best_epoch | seeds | notes |
|-|-:|-:|-:|-|
| **Anchor** — PR #330 fp32 (run `uip4q05z`) | **115.61** | 11 | 1 | Original Huber β=1 winner; pre-bf16 |
| PR #399 bf16 (3-seed mean, σ=4.53) | 112.13 | 12-13 | 3 | Infrastructure merge; throughput unlock + finite test_avg |

The **anchor remains 115.61** — that's the metric the round-2 decision rules
compare against. The bf16 multi-seed mean (112.13) is at-baseline within the
new tighter (σ=4.53) noise floor; the throughput unlock is what made bf16
merge-worthy as infrastructure.

### Anchored test metrics (now finite, post-#367)

| Source | test_avg/mae_surf_p | seeds |
|-|-:|-:|
| PR #399 bf16 + post-#367 scoring (3-seed mean) | **101.82** | 3 |

First end-to-end finite paper-facing metric on this branch. Per-split test:
`test_single_in_dist=118.16, test_geom_camber_rc=107.52,
test_geom_camber_cruise=79.13, test_re_rand=102.49` (3-seed means).

### Per-split val (Huber + slice-128, fp32 anchor `uip4q05z`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 137.21 | 1.56 | 0.74 |
| val_geom_camber_rc | 118.60 | 2.44 | 0.97 |
| val_geom_camber_cruise | 98.74 | 1.28 | 0.55 |
| val_re_rand | 107.89 | 2.00 | 0.74 |

### Reproduce

```bash
# Current default (= bf16 + Huber + slice-128 + scoring fix):
python train.py --wandb_name "willow-r2-baseline-replication" --agent <your-agent>
# All defaults: --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50
```

## Methodology constraints (load-bearing for round 2 decisions)

- **Wall-clock cap**: `SENPAI_TIMEOUT_MINUTES=30` per run. Empirically binds
  at **11–14 epochs** for current configs (with bf16 enabling +30 % epoch
  headroom). Every finished run is undertrained vs the configured 50.
- **Single-seed noise floor**: ±10 % on fp32 (thorfinn replicate evidence);
  σ ≈ 4.5 / ~4 % on bf16+Huber (askeladd's #399 multi-seed). Single-seed
  deltas under ~10 % vs 115.61 require multi-seed replication; under ~5 %
  vs 115.61 on bf16 require multi-seed.
- **Huber absorbs small-effect optimization-axis levers** (round-2 dominant
  finding, surfaced from edward #429 + tanjiro #335 independently):
  Huber's gradient clipping past |residual|=1 already implicitly does
  the work that schedule, channel-weighting, surf_weight, and BS-scaling
  axes were trying to do explicitly on MSE. Round-2 axes that target the
  same gradient-magnitude failure mode regress to baseline; only
  structurally-orthogonal mechanisms (target distribution, throughput,
  parameter-noise, normalization, regularization, data exposure,
  update-rule, per-layer LR) compound.

## Merge history

| PR | Date | Axis | Δ on val_avg/mae_surf_p |
|-|-|-|-:|
| #328 | 2026-04-27 23:50 | slice_num: 64 → 128 | (anchored at 133.55) |
| #330 | 2026-04-28 01:30 | MSE → Huber (β=1) | **−13.4 %** → 115.61 |
| #367 | 2026-04-28 02:30 | scoring NaN guard (`nan_to_num`) | (infrastructure: enables finite `test_avg`) |
| #399 | 2026-04-28 05:00 | bf16 mixed precision | (infrastructure: 1.23× speedup, +30 % epoch headroom; mean 112.13 at-baseline) |

## Validation tracks

- `val_single_in_dist` — single-foil sanity
- `val_geom_camber_rc` — held-out raceCar tandem front-foil camber (M=6-8)
- `val_geom_camber_cruise` — held-out cruise tandem front-foil camber (M=2-4)
- `val_re_rand` — stratified Re holdout across all tandem domains
