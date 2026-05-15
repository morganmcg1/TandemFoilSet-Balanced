# Baseline — `icml-appendix-charlie-pai2i-48h-r1`

This is the fresh-track baseline for the Charlie local-metrics arm (research tag
`charlie-pai2i-48h-r1`, advisor branch `icml-appendix-charlie-pai2i-48h-r1`,
target base `icml-appendix-charlie`).

## Configuration (target `train.py` defaults)

| Group | Value |
|-------|-------|
| Model | Transolver, `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `unified_pos=False` |
| Optim | AdamW, `lr=5e-4`, `weight_decay=1e-4`, batch 4, cosine `T_max=epochs` |
| Loss  | MSE in normalized space, `surf_weight=10.0` (vol_loss + 10 · surf_loss) |
| Sampler | `WeightedRandomSampler` over 3 domain groups |
| Caps  | `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MIN=30.0` (hard per-run wall clock) |
| Test  | Best-val checkpoint evaluated on 4 test splits at end of run |

## Reference numbers

**Not yet measured.** This is round 1 of a fresh track; the round 1 baseline
reproduction PR will establish `val_avg/mae_surf_p` and `test_avg/mae_surf_p`
ground truth. Until then, beat the lowest `val_avg/mae_surf_p` from any
round 1 PR.

Reproduce:

```bash
cd target/
python train.py --experiment_name baseline
# metrics committed to models/model-baseline-<stamp>/metrics.jsonl
```

## Primary ranking metric

`val_avg/mae_surf_p` for checkpoint selection; `test_avg/mae_surf_p` for
paper-facing ranking. Lower is better. Equal-weight mean of surface-pressure
MAE across the four val/test splits in physical (denormalized) units.

## How this file is updated

After every merged winner, the advisor:
1. Replaces the "Reference numbers" block with the new PR's `val_avg/mae_surf_p`
   and `test_avg/mae_surf_p` (and the per-split surface-p MAE table).
2. Appends a one-line entry under "History" with PR #, hypothesis tag, and the
   new score.

## History

_(empty — round 1 will populate this)_
