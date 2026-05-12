# SENPAI Research State — willow-pai2g-24h-r1

- 2026-05-12 20:35 (round 2: data-bug diagnosed, fleet unblocked)
- Most recent research direction from human researcher team: controlled
  24h/48h Charlie-vs-Willow logging ablation. Each training run is hard-capped
  at `SENPAI_TIMEOUT_MINUTES=30`. Scope is `willow-pai2g-24h-r1` only —
  isolated from all other PRs/branches/experiments.

## Critical finding this round: fleet-wide test_avg/mae_surf_p NaN is a DATA bug

Nezuko (#1377) localised the cause of the persistent `test_avg/mae_surf_p = NaN`
that affected every round-1 PR: `test_geom_camber_cruise/000020.pt` has `y`
containing `+inf` in the hidden pressure GT. `data/scoring.py::accumulate_batch`
intends to skip non-finite-GT samples via `sample_mask`, but does
`err = (pred - y).abs()` (→ `inf` for that sample) *before* applying the mask,
and `err * surf_mask = inf * 0 = NaN` then propagates through the accumulator.
This is the **only** non-finite-GT sample across all 8 val/test splits, but it
makes `test_avg/mae_surf_p = NaN` for every PR on this fleet regardless of
model behaviour. Filed as issue #1567 with the human research team.

Implication: **all five round-1/round-2 completed runs (#1372, #1378, #1382,
#1377, #1515) were misdiagnosed as model-stability failures.** They are not.
The pressure outputs are mostly fine; the test-side aggregator just can't
ignore a sample with inf GT. `val_geom_camber_cruise` is unaffected because
that split has no corrupted samples.

`data/` and `data/scoring.py` are read-only per `program.md`, so the workaround
is **a train.py-side filter that drops samples with non-finite `y` BEFORE
`accumulate_batch`**. Bundled into the #1515 rework. Once that lands, every
subsequent PR on this fleet will report finite `test_avg/mae_surf_p`.

## Current research focus and themes

Two round-1 PRs (#1377, #1515) just completed and round 2 is anchored on the
data-bug workaround + stability/throughput levers.

Best fleet reading so far (val_avg, not yet a merged baseline because test
side is blocked by the data bug until #1515 rework lands):

| PR | Change | val_avg/mae_surf_p | partial test_avg (3 of 4) | Status |
|---|---|---:|---:|---|
| **#1515** | grad-clip max_norm=1.0 | **115.78** | 114.96 | sent back: bundle train.py filter for data bug |
| #1377 | mlp_ratio=4 | 146.34 | 146.32 | closed (superseded on val; bug diagnosis credited) |
| #1382 | wd=3e-4 | 149.40 | 153.20 | closed (round 1) |
| #1372 | n_head=8 | 153.84 | 141.53 | closed (round 1) |
| #1378 | n_hidden=192 | 155.16 | 159.62 | closed (round 1) |

**The grad-clip-1.0 result is the strongest signal on the fleet by far.**
Frieren's grad-norm analysis showed median pre-clip grad norm = 44.68 and
**100% of steps clipped** — `max_norm=1.0` is acting as a global ~45× LR-shrink,
not "occasional outlier damping." Effective LR ≈ 1.1e-5. The val win is
plausibly LR-scale-driven (low effective LR with adaptive damping on the
long tail).

## In flight

- **frieren #1515 (rework)** — `grad-clip-1.0 + bad-sample filter`: rerun
  grad-clip-1.0 with a ~10-line train.py-side filter that drops samples with
  non-finite `y` before `accumulate_batch` (in both val and test eval loops).
  Bundled because the hypothesis can't be evaluated on the paper-facing
  metric without the filter. Becomes the round-2 merge anchor on resubmit.
- **tanjiro #1516** — `bf16-autocast`: wrap forward in
  `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` with `.float()`
  cast on outputs. Targets ~1.3–1.8× throughput → ~15–22 epochs in 30 min vs
  10–14 today. Compounds with every future hypothesis.
- **thorfinn #1538** — `huber-loss-vol`: replace MSE on the volume term with
  `F.huber_loss(pred, y_norm, reduction="none", delta=1.0)`; keep surface MSE.
  Robustness against high-Re outliers; may indirectly stabilise pressure.

## Round-1 in-flight (4 PRs still WIP — pods running, no SENPAI-RESULT yet)

All four pods are healthy and burning GPU (40–95 GB usage). They have not
posted terminal results since 17:48–17:49, which is ~2.5 h. Worth a status
ping on the next loop if they remain silent.

- alphonse #1353 — `surf-weight-25`
- askeladd #1354 — `lr-1e-3`
- edward #1356 — `n-layers-7`
- fern #1360 — `slice-num-128`

## Next assignment (this turn)

- **nezuko (idle after #1377 close)** — `lr-1e-5`: direct test of whether the
  grad-clip-1.0 val win is fundamentally an LR-scale phenomenon. With LR =
  1e-5 (≈ grad-clip-1.0's median effective LR) and no grad clipping, do we
  reach val_avg ≈ 115? If yes, the win is LR-scale-driven and frieren's
  adaptive damping is mostly the LR-shrink mechanism. If no (val ≫ 115),
  grad-clip's per-step adaptive damping is doing real work beyond LR scale.

## Potential next research directions (queued, after round 2 lands)

- **Schedule mismatch fix**: pass `--epochs <fits-in-30min>` so
  `CosineAnnealingLR(T_max=epochs)` actually anneals. With #1516 bf16 lifting
  epoch count from ~14 to ~22, the right `--epochs` value will change.
- **Compound winners**: best LR × stability/throughput fixes into a multi-knob
  frontier run, once each piece is individually validated.
- **Larger `max_norm`** (10 or 20): top 1% of grads still clipped, median
  steps untouched. Tests whether grad-clip's win is the LR shrink or the tail
  damping.
- **LR schedule variants**: OneCycleLR (built-in warmup+anneal for short
  budgets), linear warmup + cosine, lower min-LR.
- **Architecture**: RoPE / Fourier positional encoding on (x, z), gated FFN
  (SwiGLU), pre-norm vs post-norm placement — only after the simpler levers
  have settled.
- **Data**: coordinate normalisation (centering on foil 1), per-domain stats
  vs global stats.
