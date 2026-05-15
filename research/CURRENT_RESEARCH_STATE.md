# SENPAI Research State

- **Date**: 2026-05-15 14:35
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 1 results coming in; Round 2 assigned to nezuko
- **Most recent human research directive**: None received yet

## Current Best

**PR #3166 (FiLM) — val_avg/mae_surf_p = 114.6268** (merged 2026-05-15)
⚠ Note: this includes FiLM conditioning — not a pure baseline. Clean baseline A/B running in PR #3284.

## Current Research Focus

**Phase**: Round 1 experiments still in flight (7 WIP), nezuko running R2.

Key insight from PR #3166: `CosineAnnealingLR(T_max=50)` with 30-min timeout means LR stays near 5e-4 throughout training — all Round 1 experiments are running without LR decay. This is a systematic bias that affects comparison but does not invalidate relative ranking.

**Round 1 WIP** (7 in flight):

| PR | Student | Hypothesis | Category |
|----|---------|------------|----------|
| #3154 | alphonse | H5: Wider model (n_hidden 128→256, n_head 4→8) | Architecture |
| #3156 | askeladd | H1: p-channel surface loss upweight (3x, 5x) | Loss |
| #3158 | edward | H2: EMA weight averaging (decay=0.999) | Training |
| #3160 | fern | H4: Huber loss (delta=1.0, 0.5) | Loss |
| #3163 | frieren | H3: Gradient clip + 5-epoch LR warmup | Optimization |
| #3168 | tanjiro | H10: More slices (slice_num 64→128, 96) | Architecture |
| #3170 | thorfinn | H11: Deeper model (5→7, 5→8 layers) | Architecture |

**Round 2 (nezuko):**

| PR | Student | Hypothesis | Category |
|----|---------|------------|----------|
| #3284 | nezuko | H12: Clean baseline + cosine T_max=15 vs T_max=50 | Ablation/Optimization |

## Key Open Questions

1. **Is FiLM actually helping?** PR #3284 Arm A will tell us — if baseline (no FiLM) ≈ 114.63, FiLM adds nothing. If baseline >> 114.63, FiLM is a genuine improvement.
2. **Does LR annealing matter?** PR #3284 Arm B (T_max=15) will show if cosine decay helps within the 30-min window.
3. **Capacity bottleneck?** H5 (alphonse, wider) is the most likely winner — watch for it.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT; affects `test_avg/mae_surf_p` for all PRs. File is read-only. Students should report 3-split test avg as workaround.

## Potential Next Research Directions (Round 2, pending R1 results)

- **If H5 (wider) wins big**: compound n_hidden=256 + FiLM + T_max fix
- **If H1 (p-upweight) wins big**: try higher weights (7x, 10x); combine with Huber
- **If T_max fix helps**: apply corrected schedule to ALL subsequent experiments as new default
- **If FiLM confirmed helpful**: stack FiLM + wider model, try domain-specific conditioning
- **Novel ideas queued**: per-sample loss normalization (H8), WSD schedule (H9), separate p decoder head (H6), FiLM + wider (H7+H5 compound)
