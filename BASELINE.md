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
| #1541 Scoring fix + BF16 rerun | **120.40** | **106.67** | All 4 test splits now finite |

### Note on test_avg/mae_surf_p — now FIXED
`data/scoring.py` now has a `torch.where(isfinite(...))` guard preventing `0×inf=NaN` from poisoning the cruise split. Merged in PR #1541. All `test_avg/mae_surf_p` values from here forward are full 4-split averages.

Whenever a PR improves on the current best, update this table in the same commit that runs `senpai:merge-winner`.

---

## 2026-05-12 21:00 — PR #1541: Scoring fix + BF16 rerun (frieren)

- **val_avg/mae_surf_p (best epoch 17):** 120.40
- **test_avg/mae_surf_p:** 106.67
- **Per-val-split:** (not reported per-split by student; best epoch is 17)
- **Per-test-split:** single_in_dist=125.29, geom_camber_rc=113.23, geom_camber_cruise=81.16, re_rand=106.99
- **Epochs completed:** 18 in 30 min (~101 s/epoch); peak VRAM ~33 GB / 96 GB
- **W&B run:** `x7snuii5`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-frieren --wandb_name "willowpai2g24h5-frieren/baseline-bf16-scoring-fix" --wandb_group "willow-pai2g-24h-r5-baseline"`

**Key change:** One-line guard in `data/scoring.py::accumulate_batch`:
```python
err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))  # guard 0×inf=NaN
```

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **val_avg/mae_surf_p (best epoch 13):** 123.72
- **Per-val-split:** single_in_dist=153.36, geom_camber_rc=129.40, geom_camber_cruise=99.23, re_rand=112.87
- **test_avg/mae_surf_p:** NaN (cruise data bug); 3-split partial=121.90
- **Epochs completed:** 18 in 30 min (~101 s/epoch); peak VRAM 32.9 GB / 96 GB
- **W&B run:** `6zx5vuja`
- **Reproduce:** `cd "target/" && python train.py --agent willowpai2g24h5-frieren --wandb_name "run_name" --wandb_group "willow-pai2g-24h-r5-amp"`
