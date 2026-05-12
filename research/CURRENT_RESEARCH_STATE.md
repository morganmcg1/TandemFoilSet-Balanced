# SENPAI Research State

- **Date:** 2026-05-12
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2)
- **Idle students:** 0/8 after round-1 assignments
- **Most recent human researcher direction:** none on this branch

## Current research focus

This is a fresh-track round-1 launch. No prior experiments on this advisor branch — the first round establishes which lever (loss, optimization, capacity, training-efficiency, augmentation) has the largest impact on `val_avg/mae_surf_p` for the Transolver baseline on TandemFoilSet.

Hard constraints: each training execution capped at 30 min wall clock. Local JSONL metrics only — no W&B/wandb experiment logging. 1 GPU per student, 96 GB VRAM (heavily underutilized at the bs=4 baseline).

## Round-1 portfolio (assigned 2026-05-12)

Eight complementary hypotheses spanning five levers. Each is intentionally isolated so a winning result attributes cleanly to a single mechanism.

| Student | Hypothesis | Lever |
|---|---|---|
| alphonse | per-channel loss weighting (p × 5, Ux/Uy × 1) | loss alignment with metric |
| askeladd | surf_weight 10 → 30 | loss alignment with metric |
| edward | larger Transolver (n_hidden=256, n_layers=8, n_head=8) | capacity scaling |
| fern | AMP bf16 + gradient clipping (max_norm=1.0) | training efficiency |
| frieren | 3-epoch linear warmup + peak lr=1e-3 | optimization trajectory |
| nezuko | slice_num 64 → 128 | physics-token resolution |
| tanjiro | batch_size 4 → 16 + scaled lr=1e-3 | training efficiency / gradient noise |
| thorfinn | AoA-sign flip augmentation (50% prob, raceCar + cruise) | OOD geometry generalization |

## Potential next research directions (round 2 candidates)

- Stack the round-1 winners (orthogonal levers should compound: e.g. AMP + warmup + channel weighting).
- Dual surface/volume output heads (AB-UPT style, Alkin et al. 2025) — only if metric alignment hasn't already saturated.
- Fourier positional encoding for (x, z) before preprocess MLP (FNO / NeRF-style).
- Surface-only loss as a discriminating diagnostic if surf_weight monotonically improves results.
- Larger slice_num (192 / 256) if H6 (slice 128) is the best round-1 lever.
- Per-domain loss reweighting if val_geom_camber_cruise and val_geom_camber_rc errors diverge.
- Huber loss for the high-Re outliers (per-sample y std spans 10x within each split).
- Random horizontal-flip augmentation if AoA flip pays off.

## Notes

- Baseline (`train.py` defaults) records `val_avg/mae_surf_p` but no measured floor exists on this branch yet — first terminal `SENPAI-RESULT` defines the floor.
- Researcher-agent ideas log: `research/RESEARCH_IDEAS_2026-05-12_initial.md`.
</content>
</invoke>