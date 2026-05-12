# SENPAI Research Results

## 2026-05-12 19:30 — PR #1468: surf_weight 10 → 30 (surface loss emphasis)

- Branch: `charliepai2g24h2-askeladd/surf-weight-30`
- Hypothesis: Increasing surf_weight from 10 to 30 corrects the imbalance where vol_loss dominates despite surface nodes being a small fraction of total nodes
- Artifacts: `models/model-charliepai2g24h2-askeladd-surf-weight-30-20260512-180309/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 181.99 | 2.01 | 0.99 | 190.74 |
| val_geom_camber_rc     | 173.61 | 3.18 | 1.26 | 184.65 |
| val_geom_camber_cruise | 133.91 | 1.76 | 0.76 | 159.82 |
| val_re_rand            | 142.96 | 2.38 | 0.99 | 158.10 |
| **val_avg**            | **158.12** |     |     |       |
| test_single_in_dist    | 168.46 | 1.99 | 0.94 |       |
| test_geom_camber_rc    | 155.70 | 3.17 | 1.18 |       |
| test_geom_camber_cruise| NaN   | 1.67 | 0.72 |       |
| test_re_rand           | 140.25 | 2.25 | 0.98 |       |
| 3-split test avg       | 154.80 |      |      |       |

**Config:** surf_weight=30, bs=4, lr=5e-4, all other defaults. 14 epochs (30 min timeout-cut, best at epoch 11). Peak VRAM 42.1 GB.

**Decision: CLOSED — val_avg=158.12 is 18% worse than floor=133.94 and 10% worse than previous floor=143.15.**

**Analysis:** Increasing surf_weight uniformly for all surface channels harms optimization at 14 epochs. The channel weighting approach (chan_w=[1,1,5]) is a more surgical lever — it targets specifically the pressure channel rather than uniformly upweighting the surface. Both levers act on the same axis (loss alignment with primary metric) but channel weighting is strictly better. Student produced an excellent bug report on the 0×NaN propagation in test eval — assigned follow-up PR #1536 to apply the train.py guard and give us the first ever clean test_avg.

---

## 2026-05-12 19:15 — PR #1464: Per-channel loss weighting (pressure ×5)

- Branch: `charliepai2g24h2-alphonse/channel-weight-p5`
- Hypothesis: chan_w=[1,1,5] applied to sq_err aligns gradient with primary metric (val_avg/mae_surf_p)
- Artifacts: `models/model-charliepai2g24h2-alphonse-channel-weight-p5-20260512-181154/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 155.84 | 2.73 | 1.14 | 182.03 |
| val_geom_camber_rc     | 146.50 | 3.80 | 1.44 | 170.88 |
| val_geom_camber_cruise | 103.54 | 2.06 | 0.88 | 115.59 |
| val_re_rand            | 129.86 | 2.90 | 1.18 | 141.11 |
| **val_avg**            | **133.94** | 2.87 | 1.16 | 152.40 |
| test_single_in_dist    | 141.26 | 2.46 | 1.12 | 163.72 |
| test_geom_camber_rc    | 145.90 | 3.91 | 1.38 | 167.33 |
| test_geom_camber_cruise| NaN   | 1.98 | 0.83 | NaN   |
| test_re_rand           | 127.03 | 2.79 | 1.17 | 135.38 |
| 3-split test avg       | 125.48 |      |      |       |

**Config:** chan_w=[1,1,5], bs=4, lr=5e-4, surf_weight=10, n_hidden=128, n_layers=5, n_head=4, slice_num=64. 14 epochs (30 min timeout-cut), still improving. Peak VRAM 42.1 GB.

**Decision: MERGED — new floor at val_avg/mae_surf_p=133.9353 (beats previous 143.15 by 6.4%).**

**Analysis:** Channel weighting directly aligned training gradient with the primary metric. Improvement spans all 4 splits (val_re_rand marginal but positive). Val curve still descending at epoch 14 — this result is timeout-limited. Next: try chan_w=[1,1,10] to map the response curve. Also flag: two students independently found and documented the test NaN bug (data/scoring.py `0*NaN` propagation from one bad GT sample in test_geom_camber_cruise).

---

## 2026-05-12 19:15 — PR #1489: AoA-sign flip augmentation (50% per-batch)

- Branch: `charliepai2g24h2-thorfinn/aoa-flip-aug`
- Hypothesis: 50% per-batch AoA flip augmentation increases AoA coverage for OOD generalization
- Artifacts: `models/model-charliepai2g24h2-thorfinn-aoa-flip-aug-20260512-180844/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist     | 185.20 | 2.53 | **2.40** |
| val_geom_camber_rc     | 150.35 | 3.21 | **2.88** |
| val_geom_camber_cruise | 113.38 | 1.70 | **1.30** |
| val_re_rand            | 136.74 | 2.51 | **1.87** |
| **val_avg**            | **146.42** |     |      |

**Config:** per-batch 50% AoA flip, bs=4, lr=5e-4, all baseline defaults. 14 epochs (timeout-cut at epoch 11 best). Peak VRAM 42.1 GB.

**Decision: SENT BACK for changes (146.42 > 133.94 floor, plus Uy degradation 2.11 vs 0.98).**

**Analysis:** mae_surf_Uy doubled vs unaugmented (2.11 vs 0.98 for tanjiro), suggesting per-batch flipping harms Uy precision — the model sees Uy flipped for all samples in the batch 50% of the time, which may cause hedging. The interesting OOD cruise result (113.38) warrants follow-up. Sent back to try per-sample flip at p=0.25.

---

## 2026-05-12 19:00 — PR #1486: Scale batch size 4 → 8 (fallback, bs=16 OOMed)

- Branch: `charliepai2g24h2-tanjiro/batch-size-16`
- Hypothesis: bs=4 is underutilizing 96GB VRAM; bs=16 + scaled lr=1e-3 should reduce gradient noise
- Artifacts: `models/model-charliepai2g24h2-tanjiro-batch-size-8-fallback-20260512-180842/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 179.61 | 2.21 | 1.04 | 203.56 |
| val_geom_camber_rc     | 151.83 | 2.82 | 1.24 | 156.45 |
| val_geom_camber_cruise | 114.07 | 1.43 | 0.68 | 112.80 |
| val_re_rand            | 127.06 | 2.20 | 0.96 | 127.11 |
| **val_avg**            | **143.15** | 2.17 | 0.98 | 149.98 |
| test_single_in_dist    | 156.25 | 2.10 | 0.95 | 178.28 |
| test_geom_camber_rc    | 148.55 | 2.85 | 1.17 | 149.25 |
| test_geom_camber_cruise| NaN   | 1.32 | 0.64 | NaN   |
| test_re_rand           | 137.14 | 2.01 | 0.92 | 129.92 |

**Config run:** bs=8, lr=7e-4 (fallback: bs=16 OOMed at 93 GB due to pad_collate), wd=1e-4, surf_weight=10, ~1.4M baseline model. 14 epochs in 30 min (timeout-cut, still improving). Best epoch=14.

**Decision: MERGED — establishes floor at val_avg/mae_surf_p=143.15.**

**Key findings:**
- `pad_collate` makes batch scaling memory-expensive: bs=4→bs=8 pushed peak VRAM to 84 GB (far above the naive ~10 GB estimate). bs=16 needs ~160 GB — not feasible without AMP.
- With bs=8, per-epoch time ~130 s vs ~30-35 s for bs=4 → only 14 epochs in 30 min instead of ~50. Hypothesis about gradient-noise reduction can't be evaluated cleanly here.
- **Critical data bug:** `test_geom_camber_cruise/000020.pt` has 761 NaN values in p channel. scoring.py's NaN-skip logic fails because `0 * NaN = NaN` in masked reductions → test_avg/mae_surf_p is NaN for all experiments. This affects only test metrics, not val (val data is clean).
- Suggested fix: gradient accumulation (accum_steps=4 at bs=4 → effective_bs=16 without extra memory).

---

## 2026-05-12 19:00 — PR #1472: Bigger Transolver fallback 192-6-8 (~1.7M params)

- Branch: `charliepai2g24h2-edward/bigger-model-256-8-8`
- Hypothesis: 1.4M params is underparameterized; 256-8-8 (~8.8M) should improve capacity
- Artifacts: `models/model-charliepai2g24h2-edward-bigger-model-256-8-8-20260512-181250/metrics.jsonl`

| Split | mae_surf_p | epoch |
|---|---:|---:|
| val_single_in_dist     | 194.04 | 6 |
| val_geom_camber_rc     | 182.21 | 6 |
| val_geom_camber_cruise | 136.44 | 6 |
| val_re_rand            | 148.52 | 6 |
| **val_avg**            | **165.30** | 6 |

**Config run:** n_hidden=192, n_layers=6, n_head=8 (fallback: 256-8-8 OOMed at 94 GB), bs=4, lr=5e-4. 7 epochs in 30 min (per-epoch ~265 s → very few epochs). test_avg=NaN (data bug).

**Decision: CLOSED — 165.30 > 143.15 (floor). Fallback config is only 1.22x bigger than baseline and had only 7 epochs.**

**Key findings:**
- 256-8-8 with mlp_ratio=2 OOMs at bs=4 (MLP hidden=512, 8 layers × 242K-node activations). Requires bs=2 or AMP to fit.
- 192-6-8 at 7 epochs is not a valid test of the scaling hypothesis (not enough epochs to converge).
- Intermediate-size n_hidden=224, n_layers=7, n_head=8 (~3.4M, dim_head=28) should fit at bs=4 with headroom and get ~15-20 epochs.
- Model NaN in p-channel at test_geom_camber_cruise may indicate early-training numerical instability in the slice attention temperature.
</content>
