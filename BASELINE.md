# Baseline — icml-appendix-willow-pai2e-r2

Active branch: `icml-appendix-willow-pai2e-r2`.

## Current best (this branch)

- **PR**: #971 — "LR warmup (5ep linear) + flip loss_type default to relative_mae" (merged 2026-04-29)
- **Config**: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2` + `loss_type=relative_mae` (default) + `lr=2e-3` + `batch_size=16` + `compile=True` + `warmup_epochs=5`
- **val_avg/mae_surf_p** (best checkpoint, epoch 49): **54.70** (W&B `1xfcb5h5`, default seed — no `PYTHONHASHSEED` env var)
- **test_avg/mae_surf_p** (best checkpoint): **48.15** (finite across all 4 splits)
- **Wall clock**: 22.4 min / 50 epochs

> ⚠️ **Seed-swap caveat (read carefully)**: Under the new warmup schedule, the seed-to-best-basin mapping has flipped. The default seed (no `PYTHONHASHSEED`) is now the better seed (val=54.70); seed42 lands at val=67.28 — opposite of the round-3 PR #821 baseline. **Do NOT pin `PYTHONHASHSEED=42` when reproducing the new baseline.** Future PRs should run ≥ 2 seeds (default + seed42) and report both — a single seed of warmup vs. no-warmup is no longer a reliable A/B because warmup re-randomizes basin assignment per seed.
>
> Variance reduction is real but incomplete: 2-seed spread narrowed from 27.07 → 12.58, mean improved 8.4 pts. A 3rd seed is the next priority before letting other PRs benchmark against this number.

### Per-split test metrics (default seed `1xfcb5h5`, best checkpoint)

| Split | test mae_surf_p |
|-------|----------------|
| `test_single_in_dist`       | 67.22  |
| `test_geom_camber_rc`       | 60.38  |
| `test_geom_camber_cruise`   | 23.79  |
| `test_re_rand`              | 41.20  |
| **test_avg/mae_surf_p**     | **48.15** |

### Reproduce (new defaults after PR #971)

```bash
cd target && python train.py \
    --epochs 50 \
    --wandb_group lr-warmup-5ep \
    --wandb_name lr-warmup-5ep-default \
    --agent willowpai2e2-askeladd
```

`model_config` in `train.py` is now compound base by default: `n_layers=3, n_head=1, slice_num=16, n_hidden=128, mlp_ratio=2`. CLI defaults after PR #971: `batch_size=16`, `lr=2e-3`, `compile=True`, `loss_type="relative_mae"`, `warmup_epochs=5`. Schedule = `LinearLR(0.05→1.0)` for the first 5 epochs, then `CosineAnnealingLR(T_max=45, eta_min=1e-6)` for the remaining 45.

### Paired seed (seed42, W&B `9a9di1dz`)

val=67.28 / test=57.80 — same config, worse local minimum under warmup. The warmup ramp moved seed42 from the round-3 "good basin" (55.90) into a different basin. Spread vs default seed = 12.58.

---

## 2026-04-29 — PR #971: LR warmup (5-epoch linear) + flip loss_type default to relative_mae (merged)

- **val_avg/mae_surf_p:** 54.70 (best epoch 49, default seed `1xfcb5h5`)
- **test_avg/mae_surf_p:** 48.15 (finite across all 4 splits)
- **Per-split test:** single=67.22, rc=60.38, cruise=23.79, re_rand=41.20
- **Paired seed (seed42, `9a9di1dz`):** val=67.28, test=57.80 — seed-swap vs round-3
- **Variance:** 2-seed spread 12.58 (was 27.07), mean 60.99 (was 69.43) → narrowed and improved
- **Delta vs previous best (PR #821 seed42):** −1.20 (−2.1%) val_avg / −1.49 (−3.0%) test_avg
- **Wall clock:** 22.4 min / 50 epochs
- **Status:** Merged. New CLI defaults: `loss_type="relative_mae"`, `warmup_epochs=5`. Schedule = `SequentialLR(LinearLR(start=0.05) for 5ep, then CosineAnnealingLR(T_max=45, eta_min=1e-6))`.

---

## 2026-04-29 — PR #821: AMP/bf16 + bs=16 + lr=2e-3 + torch.compile + NaN-safe eval (merged)

- **val_avg/mae_surf_p:** 55.90 (epoch 50, seed42 `66c4gac6` — still descending)
- **test_avg/mae_surf_p:** 49.64 (finite across all 4 splits)
- **Per-split test:** single=63.94, rc=62.62, cruise=26.87, re_rand=45.11
- **Alternate seed (default, `1d8nkjir`):** val=82.97, test=72.01 — 27-pt spread; LR warmup needed
- **Delta vs previous best (PR #840 seed42):** −13.5% val_avg / −10.9% test_avg
- **Wall clock:** 22.5 min / 50 epochs (AMP + bs=16 + compile ≈ 1.5–1.8× speedup over fp32/bs=4)
- **Status:** Merged. New CLI defaults: `lr=2e-3`, `batch_size=16`, `compile=True`

---

## 2026-04-28 18:00 — PR #840: per-sample relative MAE (merged)

- **val_avg/mae_surf_p:** 64.16 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 55.73 (finite across all 4 splits)
- **Per-split val:** single=77.07, rc=84.10, cruise=36.86, re_rand=58.58
- **Per-split test:** single=71.33, rc=70.62, cruise=30.92, re_rand=50.04
- **Delta vs previous best (PR #783):** −11.77 (−15.5%) on val_avg/mae_surf_p
- **W&B run:** t5p9xzxx (rebased re-run)
- **Status:** Merged into advisor branch

---

## 2026-04-28 16:00 — PR #840: per-sample relative MAE (new best — pending merge)

- **val_avg/mae_surf_p:** 64.73 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 56.92 (finite across all 4 splits)
- **Per-split val:** single=80.41, rc=78.51, cruise=40.13, re_rand=60.73
- **Per-split test:** single=77.25, rc=67.74, cruise=32.35, re_rand=50.35
- **Delta vs previous best (PR #783):** −11.20 (−14.7%) on val_avg/mae_surf_p
- **W&B run:** nz8eev8e
- **Status:** Winner declared; sent back for rebase (merge conflict on advisor branch)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (prev best)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (new best)

- **Surface MAE (val_avg):** 75.93 (epoch 32, timed out at 32/50 — still improving)
- **Per-split val:** single=85.84, rc=91.20, cruise=54.68, re_rand=71.99
- **Per-split test (finite):** single=79.35, rc=82.61, re_rand=64.29; cruise=NaN (scoring bug)
- **Delta vs previous best:** −20.87 (−21.6%) on val_avg/mae_surf_p
- **W&B run:** 2y1lj209
- **Reproduce:** see above — add `--huber_delta 1.0` to the compound anchor command

---

## 2026-04-28 12:00 — PR #779: Round 1 anchor

- **Surface MAE (val_avg):** 96.80
- **W&B run:** ez3f10h3
- **Reproduce:** see above

---

## Reference context (from `target/README.md` leaderboard)

A previous senpai-vs-kagent investigation against this same dataset/Transolver
baseline found that a compounded reduction of model size dominated the
leaderboard. Use these as targets, not as merged baselines on this branch:

- Reference baseline (default config, similar to our `train.py`): `test_avg/mae_surf_p ≈ 80–82`
- Reference compound winner (PR #32 in that older repo): `test_avg/mae_surf_p = 40.927`
  - Configuration: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`
  - Compound was the combination of three independent reductions (depth, slice
    count, single-head attention) on top of the default optimizer/loss.

## Default training command

```bash
cd target && python train.py --epochs 50 --wandb_name <descriptive-name>
```

Architecture parameters (`n_hidden`, `n_layers`, `n_head`, `slice_num`,
`mlp_ratio`) are not CLI flags — students must edit `model_config` in
`target/train.py` to change them. Optimizer and loss parameters (`lr`,
`weight_decay`, `batch_size`, `surf_weight`, `epochs`) are CLI flags via
`Config`.

## Per-split structure (4 val + 4 test tracks)

The primary metric `val_avg/mae_surf_p` is the **equal-weight mean of surface
pressure MAE across the four validation splits**. The same average across the
four held-out test splits is `test_avg/mae_surf_p`. Lower is better. Best
checkpoint is selected on `val_avg/mae_surf_p` and that checkpoint is used for
the end-of-run test eval. See `target/program.md` for the full split design.

| Track | Tests |
|-------|-------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity (single-foil random holdout) |
| `val_geom_camber_rc` / `test_geom_camber_rc` | RaceCar tandem, unseen front-foil camber M=6–8 |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Cruise tandem, unseen front-foil camber M=2–4 |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem domains |
