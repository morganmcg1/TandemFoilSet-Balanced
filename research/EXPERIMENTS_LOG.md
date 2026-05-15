# SENPAI Research Results — TandemFoilSet — `icml-appendix-willow-pai2i-24h-r2`

This file logs each reviewed PR. Newest entries at the top.

## Format

```
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>
- student: <name>
- branch: <branch-name>
- hypothesis: <one-line statement>
- results table (val_avg/mae_surf_p, test_avg/mae_surf_p, per-split, wandb run id)
- analysis & conclusions
- next steps
```

## Entries

## 2026-05-15 14:45 — PR #3194: 5-epoch LR warmup + cosine annealing
- student: willowpai2i24h2-askeladd
- branch: `willowpai2i24h2-askeladd/lr-warmup-cosine`
- hypothesis: linear LR warmup over the first 5 epochs prevents cold-start damage to the PhysicsAttention slice projection and improves `val_avg/mae_surf_p`
- runs: `5jtgoadb` (warmup=3), `gyin7q96` (warmup=5)

| Arm | val_avg/mae_surf_p (best, ep 13) | test_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| warmup=3 | **136.55** | NaN ⚠ | 159.58 | 152.82 | 109.78 | 124.01 |
| warmup=5 | 153.72 | NaN ⚠ | 207.68 | 155.53 | 116.98 | 134.70 |

- analysis: Two arms compared; warmup=3 beat warmup=5 across every val split. Both hit the 30-min wall clock at epoch 14 (cosine barely decayed). The student rightly flagged that this is a warmup-3-vs-warmup-5 comparison only — there is no no-warmup arm to confirm warmup itself beats the existing schedule. `test_geom_camber_cruise` returned `Infinity` in the pressure channel for at least one cruise sample on both arms, which propagates through `data/scoring.py`'s global accumulator and poisons `test_avg/mae_surf_p` to NaN. NaN on the paper-facing metric is a merge blocker per the program contract.
- decision: **sent back** to the student with two requirements: (1) defensively zero out predictions in padded positions and apply `nan_to_num(...).clamp_(-50, 50)` inside `evaluate_split` to localize any overflow without touching the read-only `data/scoring.py`; (2) re-run with two arms in the same wandb_group `willow-pai2i-24h-r2/warmup-cosine-v2` — `warmup=0` (proper baseline) and `warmup=3` (winner). The warmup=5 arm is dropped.
- next steps: once the re-run clears NaN and shows warmup=3 ≥ warmup=0 by any margin, merge. The `nan_to_num` fix will become the baseline for all subsequent PRs.

