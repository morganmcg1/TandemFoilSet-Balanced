# Baseline — icml-appendix-willow-pai2g-24h-r5

This is the per-launch baseline tracker. Branch `icml-appendix-willow-pai2g-24h-r5` was cut from `icml-appendix-willow` with no prior advisor work, so the starting point is `train.py` at HEAD.

## Starting configuration (train.py HEAD)

- Model: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (≈ 0.8M params)
- Optimizer: AdamW (`lr=5e-4, weight_decay=1e-4`)
- Schedule: CosineAnnealingLR(T_max=epochs)
- Loss: weighted MSE in normalized space, `surf_weight=10.0` (volume+surface losses summed)
- Batch size 4, default `epochs=50`
- Per-training cap: `SENPAI_TIMEOUT_MINUTES=30` wall-clock

## Primary ranking metrics

- val: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- test: `test_avg/mae_surf_p` (equal-weight mean across 4 test splits, evaluated from the best-val checkpoint)

## Best result so far

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| #1371 BF16 autocast | **123.72** | NaN (data bug cruise split) | 3-split test avg 121.90 — see notes |

### Note on test_avg/mae_surf_p NaN
`test_geom_camber_cruise` has a single bad GT sample (`000020.pt`) with 761 nodes having `y[:,2]=-inf`. This causes `0 × inf = NaN` in `data/scoring.py:accumulate_batch`. The fix is a guard in scoring.py (`data/` read-only per program.md, requires advisor authorization). 3-split test avg (single_in_dist, geom_camber_rc, re_rand) = **121.90** is the usable paper-facing signal until the data bug is patched.

Whenever a PR improves on the current best, update this table in the same commit that runs `senpai:merge-winner`.

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **val_avg/mae_surf_p (best epoch 13):** 123.72
- **Per-val-split:** single_in_dist=153.36, geom_camber_rc=129.40, geom_camber_cruise=99.23, re_rand=112.87
- **test_avg/mae_surf_p:** NaN (cruise data bug); 3-split partial=121.90
- **Epochs completed:** 18 in 30 min (~101 s/epoch); peak VRAM 32.9 GB / 96 GB
- **W&B run:** `6zx5vuja`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-frieren --wandb_name "run_name" --wandb_group "willow-pai2g-24h-r5-amp"`
