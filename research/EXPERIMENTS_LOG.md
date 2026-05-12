# SENPAI Research Results — Charlie pai2g 24h r3

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3`.
Records of reviewed and merged experiment PRs. Add a new section under the
appropriate heading whenever an experiment terminal-completes.

---

## Round 1 — broad coverage (assigned 2026-05-12)

Hypotheses sourced from `/research/RESEARCH_IDEAS_2026-05-12_18:00.md`.

**Cross-round findings (apply to all round 1 results):**
- All 4 reviewed runs hit the 30-min timeout. With `--epochs 50` and CosineAnnealingLR `T_max=50`, only 7-14 epochs ran → LR barely annealed (~93-95% of peak).
- `test_geom_camber_cruise/mae_surf_p` is NaN for all runs due to a NaN propagation bug in `data/scoring.py` (IEEE 754: `NaN * 0 = NaN` in `err * surf_mask`). File is read-only; use 3-split proxy (single + rc + re_rand).
- grad_clip=1.0 fires on 100% of training batches (real norms 41-115). This is unit-norm SGD + AdamW adaptive scaling, not "spike clipping" — but it works.

---

### 2026-05-12 18:56 — PR #1491: Gradient clipping + weight_decay tuned (fern)
**Branch:** `charliepai2g24h3-fern/grad-clip-adamw-tuned` | **Status: MERGED** ⭐

- **Hypothesis:** grad_clip=1.0 + weight_decay 1e-4→1e-3 would stabilize training on high-Re outliers.
- **val_avg/mae_surf_p: 115.403** (epoch 12/14)
- **Per-split:** single=133.09, rc=129.76, **cruise=88.99** (best), re_rand=109.76
- **Test (3-split proxy):** single=116.98, rc=119.26, re_rand=109.15 → proxy avg ~115.1
- **Analysis:** Clipping fired on 100% of batches (norms 41-115). Produces the smoothest val trajectory of round 1 (249 → 115 over 12 epochs, nearly monotone). The wd=1e-3 + clip combination outperforms all other round-1 variants. This is the new baseline.
- **Artifacts:** `models/model-grad-clip-wd1e-3-20260512-181000/metrics.jsonl`

---

### 2026-05-12 18:56 — PR #1495: AoA + NACA camber jitter augmentation (thorfinn)
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: SENT BACK**

- **Hypothesis:** Online jitter of AoA (±0.5°) and NACA camber (±0.002) improves OOD camber splits.
- **val_avg/mae_surf_p: 129.694** (epoch 12/14)
- **Per-split:** single=155.30, rc=141.33, **cruise=102.93** (OOD best), re_rand=119.22
- **Test (3-split proxy):** single=139.75, rc=129.04, re_rand=122.90 → ~130.56
- **Analysis:** 12% worse than fern's run, but same epoch budget. Camber OOD splits are NOT the worst — single-foil in-dist (155.3) is worst, possibly because extreme high-Re raceCar pressures dominate. Cannot isolate augmentation effect without equal-budget no-aug control. Cosine T_max mismatch (same as all round 1). Sent back: rebase on #1491 baseline + fix T_max.
- **Artifacts:** `models/model-geom-aoa-augment-20260512-181104/metrics.jsonl`

---

### 2026-05-12 18:53 — PR #1492: mlp_ratio 2→4 wider FFN (frieren)
**Branch:** `charliepai2g24h3-frieren/mlp-ratio-4-wider-ffn` | **Status: SENT BACK**

- **Hypothesis:** Restoring mlp_ratio to the paper's default (4) improves FFN capacity.
- **val_avg/mae_surf_p: 144.334** (epoch 11/13)
- **Per-split:** single=183.46, rc=153.62, cruise=105.23, re_rand=135.03
- **Test (3-split proxy):** single=155.33, rc=139.70, re_rand=125.49 → ~140.18
- **Analysis:** 25% worse than fern's run. Same 30-min timeout issue. mlp_ratio=4 is ~21% slower per epoch, so fewer epochs completed. Without proper cosine annealing the comparison is unfair. Sent back: rebase on #1491 + set --epochs 12 to match actual budget.
- **Artifacts:** `models/model-mlp-ratio-4-20260512-180817/metrics.jsonl`

---

### 2026-05-12 18:53 — PR #1490: Scale model n_hidden=256, n_head=8 (edward)
**Branch:** `charliepai2g24h3-edward/scale-model-256` | **Status: SENT BACK**

- **Hypothesis:** n_hidden 128→256, n_head 4→8 (~2.54M params) improves capacity.
- **val_avg/mae_surf_p: 172.262** (epoch 6/7; 30-min cap after 7 epochs)
- **Per-split:** single=199.15, rc=194.97, cruise=131.46, re_rand=163.47
- **Test:** NaN overall; 3-split proxy: single=191.41, rc=186.78, re_rand=159.10 → ~179.09
- **Analysis:** Severely under-budgeted — ~260 s/epoch means only 7 epochs in 30 min. Model trending down (172 → 176 at epoch 7) but far from converged. Also: model is 2.54M not the predicted ~6M (mlp_ratio=2 was not changed). OOM risk (83.9GB peak). Sent back: scale down to n_hidden=192 + set --epochs 10 + rebase on #1491.
- **Artifacts:** `models/model-scale-model-256-20260512-180850/metrics.jsonl`
