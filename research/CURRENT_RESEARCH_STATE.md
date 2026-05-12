# SENPAI Research State

- **Date:** 2026-05-12 19:00
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 143.15** (PR #1486, merged)  
Config: bs=8/lr=7e-4/baseline-model, 14 epochs in 30 min (timeout-cut, still improving)  
Note: floor was set by a fallback run (bs=16 OOMed) — true bs=4 baseline value unknown.

**Known test NaN bug:** `test_geom_camber_cruise/000020.pt` has NaN p-channel GT. The `0 * NaN = NaN` mask propagation in `data/scoring.py` makes `test_avg/mae_surf_p` NaN for all experiments. Val metrics are clean (4 splits all finite). Ranking on `val_avg/mae_surf_p` is unaffected.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever |
|---|---|---|---|
| #1464 | alphonse | Per-channel loss weighting (pressure ×5) | Loss alignment |
| #1468 | askeladd | surf_weight 10 → 30 | Loss alignment |
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency |
| #1482 | frieren | 3-epoch warmup + peak lr=1e-3 + cosine | Optimization |
| #1485 | nezuko | slice_num 64 → 128 | Physics-token resolution |
| #1489 | thorfinn | AoA-sign flip augmentation (50% prob) | OOD geometry generalization |
| #1524 | tanjiro | Gradient accumulation (accum=4, eff_bs=16) | Gradient quality |
| #1526 | edward | Model scaling: n_hidden=224, n_layers=7 (~3.4M) | Capacity |

## Key findings from round 1 (first completions)

1. **pad_collate is the memory bottleneck for batch scaling** — bs=8 peaks at 84 GB; bs=16 OOMs. Gradient accumulation is the correct way to test effective batch scaling.
2. **256-8-8 doesn't fit at bs=4** — activation memory fills 94 GB. Must use bs=2 or AMP to try the full-size model.
3. **Test NaN bug** — `test_geom_camber_cruise/000020.pt` has corrupt p GT. Needs fix in scoring or data.

## Potential next research directions (round 3 candidates)

- **Stack round-1 winners**: AMP + warmup + channel weighting should compound if orthogonal.
- If AMP wins: re-attempt 256-8-8 at bs=4 with bf16 (halves activation memory).
- **NaN guard fix**: add `nan_to_num` or sample-skip in `evaluate_split` for clean test metrics.
- **Sort-by-size sampler**: batch similar mesh sizes to reduce padding waste (frees VRAM headroom).
- Dual surface/volume output heads (AB-UPT style) — if loss alignment levers are exhausted.
- Fourier positional encoding of (x, z).
- surf_weight > 30 if askeladd's result wins cleanly.
- Larger slice_num (192/256) if nezuko (128) wins cleanly.
</content>
