# SENPAI Research Results — willow-pai2g-24h-r1

Track: `willow-pai2g-24h-r1` | Advisor branch: `icml-appendix-willow-pai2g-24h-r1`

## 2026-05-12 18:55 — PR #1378 (CLOSED): Widen Transolver hidden dim 128->192
- Branch: `willowpai2g24h1-tanjiro/n-hidden-192`
- Hypothesis: Increase model width from 128 → 192 to expand all projections, expected to improve fitting capacity on all splits with biggest gain on geometric OOD.
- Single isolated change: `model_config["n_hidden"]: 128 → 192` in `train.py:421`.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 8) | 155.1646 | best of 10 epochs run |
| test_avg/mae_surf_p | **NaN** | corrupted by inf on `test_geom_camber_cruise/mae_surf_p` |
| test_avg/mae_surf_p (partial 3 of 4) | 159.6191 | excluding cruise split |
| Params | 1.47 M (vs ~0.65 M baseline) | +2.25× |
| Peak VRAM | 58.0 GB / 96 GB | comfortable |
| Epochs completed | 10 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~183 s | vs ~120 s baseline-implied |
| W&B run id | `m66sbam1` | group `width-sweep` |

- Per-split test (best-val checkpoint @ epoch 8): single_in_dist=194.51 · geom_camber_rc=151.73 · geom_camber_cruise=**inf** · re_rand=132.62.
- Per-split val (epoch 8): single_in_dist=231.42 · geom_camber_rc=149.30 · geom_camber_cruise=123.25 · re_rand=131.50.

**Conclusion — INCONCLUSIVE → CLOSED.** Two blockers compound under the 30-min cap: (1) wider model + same wall clock = only 10/50 epochs reached, and the cosine T_max=50 schedule means LR is effectively unchanged from peak across the entire run; (2) test_geom_camber_cruise produces non-finite pressure on at least one OOD sample, so the paper-facing `test_avg/mae_surf_p` is NaN. Can't isolate width effect without first fixing throughput (more epochs in the budget) and stability (no inf pressure outputs). Both fixes are good single-knob hypotheses on their own — re-assigning tanjiro to bf16-autocast which directly attacks throughput.

## 2026-05-12 18:54 — PR #1372 (CLOSED): Increase attention heads 4->8
- Branch: `willowpai2g24h1-frieren/n-head-8`
- Hypothesis: Doubling heads (with `dim_head=64` unchanged, raising inner_dim 256→512) gives more parallel physics features and should help OOD splits most.
- Single isolated change: `model_config["n_head"]: 4 → 8` in `train.py:423`.

| Metric | Value | Notes |
|---|---|---|
| val_avg/mae_surf_p (best, epoch 10) | **153.8400** | best of 11 epochs run |
| test_avg/mae_surf_p | **NaN** | corrupted by inf on `test_geom_camber_cruise/mae_surf_p` |
| test_avg/mae_surf_p (partial 3 of 4) | 141.5282 | excluding cruise split |
| Params | 0.65 M | same order as baseline |
| Peak VRAM | 54.5 GB / 96 GB | comfortable |
| Epochs completed | 11 / 50 | hit `SENPAI_TIMEOUT_MINUTES=30` cap |
| Per-epoch | ~170 s | vs ~120 s baseline-implied |
| W&B run id | `pfxtxxms` | group `head-sweep` |

- Per-split test (best-val checkpoint @ epoch 10): single_in_dist=152.99 · geom_camber_rc=133.32 · geom_camber_cruise=**NaN** · re_rand=138.28.
- Per-split val (epoch 10): single_in_dist=231.42 · geom_camber_rc=149.30 · geom_camber_cruise=123.25 · re_rand=131.50.

**Conclusion — INCONCLUSIVE → CLOSED.** Same wall-clock + cosine-schedule mismatch as #1378; only 11/50 epochs and LR essentially flat at peak. Same `test_geom_camber_cruise/mae_surf_p` inf as #1378 — strongly suggests the issue is a model-stability blow-up on extreme OOD samples under undertraining, not a knob-specific failure. Mildly better val than width arm (153.84 vs 155.16) and better partial test (141.53 vs 159.62), but cannot anchor a baseline with NaN test_avg. Re-assigning frieren to grad-clip-1.0 to attack the cruise-pressure inf directly so future capacity experiments produce clean test_avg numbers.

## Cross-result note (round 1 first results)

Two completely independent capacity changes (width vs heads) produced the same pathology: model output goes non-finite on `test_geom_camber_cruise/mae_surf_p` while remaining finite on the other 3 test splits and on Ux/Uy of the cruise split. The aggregate `test_avg/mae_surf_p` is NaN for both. Hypothesis: under-trained Transolver (LR near peak for the entire 10-epoch run) produces extreme pressure predictions on the unseen camber range (M=2–4) in the cruise tandem domain. Round-2 priorities:
1. **Stability**: `clip_grad_norm_(parameters, 1.0)` — assigned to frieren.
2. **Throughput**: `torch.autocast(bf16)` mixed precision — assigned to tanjiro.
3. **Schedule mismatch**: separate follow-up — once we know roughly how many epochs fit, future experiments should pass `--epochs` close to that count so `CosineAnnealingLR(T_max=epochs)` actually anneals.
