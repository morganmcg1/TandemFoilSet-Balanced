# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-27 (Round 0 — no merges yet)

## Current best (vanilla Transolver, unmerged baseline)

The advisor branch starts from the unmodified `train.py` Transolver baseline. No improvements have been merged on this branch yet. Round 0 students compare their PR against the vanilla `train.py` configuration:

| Setting | Value |
|---------|-------|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` wall clock) |
| Primary metric | `val_avg/mae_surf_p` (lower is better) |
| Paper metric | `test_avg/mae_surf_p` (logged at end of run from best val checkpoint) |

**Round 0 instruction to students:** report `val_avg/mae_surf_p`, `test_avg/mae_surf_p`, and the four per-split `test/{split}/mae_surf_p` numbers in the PR comment. The first round establishes our baseline number.

The four val/test splits and what each tracks:

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains
