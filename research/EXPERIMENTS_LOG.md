# SENPAI Research Results — willow-pai2g-24h-r5

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **Branch:** `willowpai2g24h5-frieren/bf16-mixed-precision`
- **Hypothesis:** BF16 mixed precision halves per-step time, buying more epochs in the 30-min wall-clock budget.
- **W&B run:** `6zx5vuja`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 13) | **123.72** |
| val_single_in_dist/mae_surf_p | 153.36 |
| val_geom_camber_rc/mae_surf_p | 129.40 |
| val_geom_camber_cruise/mae_surf_p | 99.23 |
| val_re_rand/mae_surf_p | 112.87 |
| test_avg/mae_surf_p | NaN (cruise data bug) |
| 3-split test avg (no cruise) | **121.90** |
| Epochs completed | 18 in 30 min |
| Peak VRAM | 32.9 GB / 96 GB |

**Result:** MERGED as new baseline. BF16 completed 18 epochs vs ~14 at FP32 (estimated), establishing val_avg=123.72.

**Key observation:** Pre-existing data corruption in `test_geom_camber_cruise/000020.pt` (761 nodes with y[:,2]=-inf) poisons 4-split test_avg via `0×inf=NaN` in scoring.py. Affects every run on this branch. 3-split test avg is the usable paper-facing signal until fixed.

---

## 2026-05-12 18:56–19:51 — PR #1412: Warmup 3ep then cosine / Warmup 5ep then cosine (thorfinn)

- **Branch:** `willowpai2g24h5-thorfinn/warmup-3ep-then-cosine`
- **Hypothesis:** Linear LR warmup before cosine annealing stabilizes early training steps.
- **W&B runs:** `3chdcivo` (warmup-3ep), `jcd79mzi` (warmup-5ep)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| warmup-3ep | 144.50 | 12 | 144.54 |
| warmup-5ep | **135.37** | 14 | **131.12** |

Per-split (warmup-5ep):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 164.28 | 142.88 |
| geom_camber_rc | 143.91 | 130.88 |
| geom_camber_cruise | 110.76 | NaN |
| re_rand | 122.52 | 119.61 |

**Result:** SENT BACK for rebase. warmup-5 (135.37) did not beat the BF16 baseline (123.72) as a standalone, but warmup and BF16 are orthogonal. Student rebasing to test the combo (warmup-5 + BF16 already in base).

**Key observation:** Warmup=5 strictly dominates warmup=3 across all splits except geom_camber_cruise (+5%). Single_in_dist improved 15.7%, re_rand 1.7%, rc 6.2%. Per-epoch time ~131s; 14 epochs in 30 min without BF16.

---

## 2026-05-12 19:56 — PR #1367: Dropout=0.1/0.2 + grad-clip=1.0 (fern) — **PENDING REBASE**

- **Branch:** `willowpai2g24h5-fern/dropout-0.1-grad-clip`
- **Hypothesis:** Light dropout + grad clipping improves OOD generalization.
- **W&B runs:** `7brl22oo` (dropout=0.1), `3wz81r3d` (dropout=0.2)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| dropout=0.1, clip=1.0 | 146.31 | 8 | 146.51 |
| **dropout=0.2, clip=1.0** | **113.86** | **12** | **114.77** |

Per-split (dropout=0.2):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 145.19 | 132.80 |
| geom_camber_rc | 120.81 | 112.25 |
| geom_camber_cruise | 83.27 | NaN |
| re_rand | 106.18 | 99.28 |

**Result:** SENT BACK for rebase. val_avg=113.86 BEATS current BF16 baseline (123.72) by 7.7%. Merge conflict with PR #1371 — student rebasing to test dropout=0.2 + BF16 combination. Expected to combine to an even lower metric.

**Key observation:** dropout=0.2 beats dropout=0.1 across EVERY split (−22% overall), not just OOD. Probably acts as a smoother loss landscape rather than just generalization: 5 Transolver layers with slice_num=64 attention have many co-adaptation opportunities that dropout disrupts usefully. Validation still descending at 30-min cap — suggests this configuration has more headroom.

---

*Log format: one block per PR review. Stale WIP PRs (1400 tanjiro, 1386 nezuko, 1365 edward, 1357 askeladd, 1352 alphonse) have no results yet after 2+ hours — advisor check-in comments posted 2026-05-12 ~20:00.*
