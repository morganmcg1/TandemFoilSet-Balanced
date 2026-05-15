# SENPAI Research State

- 2026-05-15 — round 1 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Current focus

Establish strong first-round improvements on the Transolver baseline by hitting orthogonal axes simultaneously. Primary metric: `val_avg/mae_surf_p` (and `test_avg/mae_surf_p` at the end of every run). No committed baseline metrics yet on this branch — first-round results will define our internal reference.

Baseline configuration (from `train.py` at HEAD):

| Field | Value |
|------|------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Activation / init | GELU, trunc_normal_(std=0.02) |
| Optimizer | AdamW lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR(T_max=epochs) |
| Loss | MSE in normalized space, vol_loss + 10·surf_loss |
| Batch / epochs | batch_size=4, epochs=50 (capped by SENPAI_TIMEOUT_MINUTES) |

## Round 1 hypothesis themes (8 idle GPUs)

1. **alphonse — Capacity scale-up:** n_hidden 128→192, n_layers 5→6, n_head 4→6.
2. **askeladd — Surface loss weight:** surf_weight 10→30 to align loss with primary metric.
3. **edward — OneCycleLR + warmup:** swap CosineAnnealingLR for OneCycleLR(max_lr=1e-3, pct_start=0.1).
4. **fern — Per-channel surface weighting:** weight surface-p 4× over Ux, Uy in the loss.
5. **frieren — Slice/head scale-up:** slice_num 64→128, n_head 4→8, mlp_ratio 2→3.
6. **nezuko — Batch + LR scaling:** batch_size 4→8, lr 5e-4→8e-4 (sqrt scaling).
7. **tanjiro — Re-conditional FiLM:** FiLM modulation on `fx` after preprocess using log(Re).
8. **thorfinn — Longer training:** epochs 50→80 with cosine schedule (uses full budget).

## Potential next research directions

- Best-checkpoint physical-space target transforms (asinh / log-p) if val gradients are dominated by extreme-Re samples.
- Multi-scale / hierarchical Transolver variants if capacity scale-up shows positive but limited returns.
- Geometry-aware feature engineering (LE distance, signed surface-normal distance) for the cruise/raceCar camber-OOD splits.
- Lion or AdamW-β2-tuning if optimization plateau emerges.
- SwiGLU / RMSNorm activation-and-norm swap if architecture is the bottleneck.
- Geometry-conditional FiLM (camber, AoA) and joint Re+camber gating for the geom-OOD splits.
- Channel-decoupled head architectures (separate decoders for Ux/Uy vs p).
