# Baseline — `icml-appendix-charlie-pai2g-48h-r5`

This branch is the **Charlie no-W&B logging ablation, round 5 (charlie-pai2g-48h-r5)**.

Experiment metrics are written to local JSONL only (`models/<experiment>/metrics.jsonl`).
**Do not** add or query W&B / wandb experiment logging for this arm.

## Primary ranking metric

- **Validation:** `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE
  across the four val tracks (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` from the best-val checkpoint.

## Reference configuration

The current advisor branch was created clean — no prior round-5 winners have
been measured yet. Use the `train.py` defaults as the reference setup students
must beat:

```
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
model_config = dict(
    space_dim=2, fun_dim=22, out_dim=3,
    n_hidden=128, n_layers=5, n_head=4,
    slice_num=64, mlp_ratio=2,
)
optimizer = AdamW; scheduler = CosineAnnealingLR(T_max=epochs)
```

Each training execution is hard-capped by `SENPAI_TIMEOUT_MINUTES=30` (wall clock).
`--epochs 50` is an upper bound; many runs will be wall-clock-bound and stop
earlier.

## Current best (val)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | _unset_ | — | First terminal result establishes the floor. |
| `test_avg/mae_surf_p` | _unset_ | — | From best-val checkpoint. |

The first cleanly-terminal experiment that completes with finite metrics on all
four val splits + four test splits becomes the round-5 baseline. Subsequent PRs
are merged only if they improve `val_avg/mae_surf_p`.

## Reproduce command

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name "<student>/<short-description>" \
    --epochs 50
```

Commit `models/<experiment>/metrics.jsonl` and `metrics.yaml` with the PR and
quote the key values in the PR results comment plus the
`SENPAI-RESULT` terminal marker.
