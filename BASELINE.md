# Baseline — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four validation splits). Lower is better. Test counterpart: `test_avg/mae_surf_p`.

## Current best
- **Status:** No baseline yet — fresh research track. Baseline reference run is scheduled in round 1 (PR assigned to `charliepai2g48h4-alphonse`).
- **Default Transolver config (in `train.py`):** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`; `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50`; CosineAnnealingLR; AdamW optimizer.
- Each training execution capped at `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock bound).

## How to compare
- Pull `val_avg/mae_surf_p` from the committed `models/<experiment>/metrics.jsonl` `epoch` event flagged `is_best`; the matching test number is in the trailing `test` event under `test_avg["avg/mae_surf_p"]`.
- Per-split diagnostics (`mae_surf_p`, `mae_vol_p`, `mae_surf_Ux`, `mae_surf_Uy`) are in `val_splits` of the same JSONL record.

## Notes
- Local JSONL only — W&B/wandb logging is disabled for this Charlie arm. Do not introduce wandb code paths.
- Test metric is evaluated from the best-val checkpoint, not the terminal epoch.
