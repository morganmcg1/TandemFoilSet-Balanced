# Baseline

The current research baseline on `icml-appendix-willow-pai2d-r2`.

The primary ranking metric is `val_avg/mae_surf_p` — equal-weight mean surface
pressure MAE across the four validation tracks. Lower is better.

## Current configuration (after merging #328 + #330 + #367 + #399 + #553)

```
lr=5e-4  weight_decay=1e-4  batch_size=4  surf_weight=10.0  epochs=50
n_hidden=128  n_layers=5  n_head=4  slice_num=128  mlp_ratio=2
optimizer=AdamW  loss=Huber(beta=1.0) on normalized residuals
norm=LayerNorm  amp_dtype=bf16  compile_mode=default
scoring_guard=nan_to_num
```

### Anchored val metrics

| Source | val_avg/mae_surf_p | best_epoch | total_epochs | seeds | notes |
|-|-:|-:|-:|-:|-|
| Anchor (fp32 #330, run `uip4q05z`) | 115.61 | 11 | 11 | 1 | Original Huber β=1 winner; pre-bf16, pre-compile |
| bf16 (PR #399, 3-seed mean) | 112.13 (σ=4.53) | 11–13 | 13 | 3 | Infrastructure unlock; +30 % epoch headroom |
| **Current — compile+bf16 (PR #553, 3-seed mean)** | **80.70 (σ=2.20)** | **28–29** | **29** | **3** | **Round-2 winner: −28 % vs bf16 baseline; 2.23× speedup; 18 GB memory freed** |

The **anchor remains 115.61** for historical comparability, but the **operative
baseline for round-2 PRs is now compile+bf16+Huber 3-seed mean = 80.70 (σ=2.20)**.
All subsequent merge decisions compare against this.

### Anchored test metrics (post-#367 + bf16-finite + compile)

| Source | test_avg/mae_surf_p | seeds |
|-|-:|-:|
| bf16 (PR #399, 3-seed mean) | 101.82 | 3 |
| **Current — compile+bf16 (PR #553, 3-seed mean)** | **71.45 (σ=2.88)** | **3** |

Per-split test (3-seed compile+bf16 mean): `test_single_in_dist=84.02`,
`test_geom_camber_rc=84.15`, `test_geom_camber_cruise=48.91`, `test_re_rand=68.71`.

### Per-split val (3-seed compile+bf16 mean)

| Split | mae_surf_p (mean) | σ |
|-|-:|-:|
| val_single_in_dist | 96.96 | 8.66 |
| val_geom_camber_rc | 91.90 | 5.03 |
| val_geom_camber_cruise | 58.23 | 1.08 |
| val_re_rand | 75.72 | 1.03 |
| **val_avg** | **80.70** | **2.20** |

### Reproduce

```bash
# Current default (= compile+bf16+Huber+slice-128+scoring-fix):
python train.py --wandb_name "willow-r2-baseline-replication" --agent <your-agent>
# All defaults: --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50
```

## Methodology constraints (load-bearing for round-2/round-3 decisions)

- **Wall-clock cap**: `SENPAI_TIMEOUT_MINUTES=30` per run. With compile+bf16,
  this allows **~29 epochs** of training (vs 13 at bf16-only, vs ~11 at
  fp32-eager). The val curve at the cap is now meaningfully closer to
  convergence than at any prior baseline.
- **Single-seed noise floor**: σ ≈ 2.20 on compile+bf16+Huber (down from
  σ=4.53 at bf16-only and ±10 % at fp32-eager). Single-seed deltas under
  ~3 % vs 80.70 require multi-seed replication; this is the new tighter
  floor for round-2 onward decisions.
- **Memory headroom**: peak 30 GB at compile+bf16 vs 48 GB at bf16-only
  vs ~12 GB at fp32-eager. **+18 GB of memory freed by compile** opens
  substantial headroom for round-3 architecture stacks (depth-8, width-160,
  larger slice_num, larger batch).

## Round-2 dominant lessons (load-bearing for round-3 axis selection)

1. **Throughput axis was the highest-leverage round-2 lever**. bf16 + compile
   compounded multiplicatively (1.23× × 1.81× = 2.23×) and the extra epoch
   budget converted directly to metric improvement. The cap was the
   binding constraint; lifting it (via efficiency, not extending wall-clock)
   was the right move.
2. **Huber absorbs small-effect optimization-axis levers** (round-2 finding
   from edward #429 + tanjiro #335). Huber's gradient clipping past
   |residual|=1 already implicitly does the work that schedule, channel-
   weighting, surf_weight, and BS-scaling axes were trying to do explicitly
   on MSE.
3. **Per-distribution-shift pattern is real but its causal mechanism is
   NOT Huber-clipping** (round-2 finding from fern #559 closed). Up-weighting
   raceCar samples (the hypothesized fix for Huber-induced cruise favoring)
   degraded raceCar's own held-out splits monotonically. The per-distribution
   shift has a different cause — probably per-region (spatial) or per-channel,
   not per-sample.
4. **Architecture axes that were budget-confounded at fp32 may now work**.
   depth-8 (closed at val=162 with 9/50 epochs), width-160 (multi-seed at
   126.18), slice_num=256 (closed at 133.30) — all were closed because the
   30-min cap couldn't fit enough training. With compile's 2.23× speedup and
   29 epochs in budget, these axes deserve re-validation.

## Merge history

| PR | Date | Axis | Δ on val_avg/mae_surf_p (vs prior) |
|-|-|-|-:|
| #328 | 2026-04-27 23:50 | slice_num: 64 → 128 | (anchored at 133.55) |
| #330 | 2026-04-28 01:30 | MSE → Huber (β=1) | **−13.4 %** → 115.61 |
| #367 | 2026-04-28 02:30 | scoring NaN guard (`nan_to_num`) | (infrastructure: enables finite `test_avg`) |
| #399 | 2026-04-28 05:00 | bf16 mixed precision | (infrastructure: 1.23× speedup, +30 % epoch headroom; bf16+Huber multi-seed mean 112.13) |
| #553 | 2026-04-28 09:00 | torch.compile (mode=default) | **−28 % vs bf16 baseline; 2.23× speedup; 18 GB memory freed** → 80.70 |

## Validation tracks

- `val_single_in_dist` — single-foil sanity
- `val_geom_camber_rc` — held-out raceCar tandem front-foil camber (M=6-8)
- `val_geom_camber_cruise` — held-out cruise tandem front-foil camber (M=2-4)
- `val_re_rand` — stratified Re holdout across all tandem domains
