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

## 2026-05-15 15:25 — PR #3207: PGOT-style geometry-conditioned slice assignment
- student: willowpai2i24h2-nezuko
- branch: `willowpai2i24h2-nezuko/geom-slice-injection`
- hypothesis: injecting per-node geometry features (NACA M/P/T, AoA, Re, gap, stagger) into PhysicsAttention's slice projection improves generalization across the camber-holdout splits without hurting in-distribution
- runs: `pjmkgg22` (wandb_group `willow-pai2i-24h-r2/geom-slice-injection`)

| Arm | val_avg/mae_surf_p (best, ep 12) | test_avg/mae_surf_p (W&B) | val_single | val_camber_rc | val_camber_cruise | val_re_rand |
|---|---|---|---|---|---|---|
| geom-slice | **128.34** | NaN ⚠ | 145.96 | 142.21 | 107.66 | 117.51 |

- analysis: The hypothesis is supported on validation — geom-slice beats the warmup=3 val_avg (136.55, PR #3194) by ~6% with no regression on any of the four val tracks, and `val_geom_camber_rc` drops from 152.82 → 142.21 (–7%), exactly the split the hypothesis targeted. The run completed all 50 epochs in 31.5 min (just over wall clock cap, last-epoch eval was the bottleneck) and converged cleanly with `val_avg` still falling slowly after epoch 12, suggesting more headroom with a slightly larger model or schedule adjustment. **However, `test_avg/mae_surf_p` is NaN in W&B** — same global bug as PR #3194 (data/scoring.py:48 computes `(pred - y).abs() * mask` BEFORE the per-sample skip, so `inf*0 = NaN` poisons the accumulator when GT has non-finite values; reproduced to `test_geom_camber_cruise/000020.pt` having `y[..., 2] = -inf` at 761 volume nodes). The student computed an offline-corrected `test_avg = 115.71` by re-running scoring with NaN-zeroed samples, but the program contract requires the W&B-logged metric to be the source of truth.
- decision: **sent back** to draft with the exact `evaluate_split` patch (pre-zero non-finite y samples and exclude them from the metric via the mask/is_surface, before calling `accumulate_batch`). Asked the student to re-run the same single arm and confirm the W&B `test_avg/mae_surf_p` reads ~115.71. The hypothesis is the strongest candidate so far; if the re-run lands a finite test number, this becomes the first merge-eligible Round-1 result.
- next steps: on a clean re-run, merge this as the new baseline (val_avg/mae_surf_p=128.34, test=115.71). Then Round 2 priorities: (a) stack geom-slice + warmup=3 (small additive risk, both target different bottlenecks), (b) per-block geometry conditioning (FiLM-style modulation), (c) sweep `slice_num` since slice-token capacity is the load-bearing component.


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

