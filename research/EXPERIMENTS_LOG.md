# SENPAI Research Results — `icml-appendix-charlie-pai2i-48h-r1`

Chronological log of advisor reviews for the Charlie local-metrics arm.
Results live in committed `models/<experiment>/metrics.jsonl` and `metrics.yaml`.

## 2026-05-15 13:35 — PR #3107 — baseline reproduction (CLOSED, no merge)

- **Branch:** `charliepai2i48h1-alphonse/baseline-r1`
- **Hypothesis:** Calibration run — default Transolver config, no code changes
- **Results:**

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | 143.52 (best epoch 11, 14 epochs / 30 min) |
| `val_single_in_dist/mae_surf_p` | 181.35 |
| `val_geom_camber_rc/mae_surf_p` | 163.47 |
| `val_geom_camber_cruise/mae_surf_p` | 105.77 |
| `val_re_rand/mae_surf_p` | 123.49 |
| `test_avg/mae_surf_p` (NaN-safe) | 130.34 (3-split partial) |

- **Metrics path:** `models/model-charliepai2i48h1-alphonse-baseline-r1-20260515-124500/metrics.jsonl`
- **Action:** Closed (calibration only). Numbers recorded in BASELINE.md.
- **Key finding:** `data/scoring.py` NaN propagation bug on `test_geom_camber_cruise/000020.pt` (inf GT pressure on 761 volume nodes). `NaN * 0 = NaN` in IEEE 754 poisons the accumulator. Val splits clean; test_avg NaN. Bug fix needed (see round 2 assignment).

---

## 2026-05-15 13:35 — PR #3111 — SmoothL1 loss replaces MSE (MERGED → new baseline)

- **Branch:** `charliepai2i48h1-askeladd/smooth-l1-loss`
- **Hypothesis:** Replace `(pred - y_norm)**2` with `F.smooth_l1_loss(pred, y_norm, reduction='none', beta=1.0)` in training loop and `evaluate_split`. Direct MAE-metric alignment; linear gradient on large residuals.
- **Results vs baseline (143.52):**

| Split | Baseline val mae_surf_p | SmoothL1 val mae_surf_p | Δ |
|-------|----:|----:|----:|
| `single_in_dist`     | 181.35 | 144.61 | **-20.3%** |
| `geom_camber_rc`     | 163.47 | 124.04 | **-24.1%** |
| `geom_camber_cruise` | 105.77 |  89.33 | **-15.5%** |
| `re_rand`            | 123.49 | 102.70 | **-16.8%** |
| **avg**              | **143.52** | **115.17** | **-19.7%** |

Test (3 valid splits, NaN-safe):

| Split | Baseline | SmoothL1 | Δ |
|-------|----:|----:|----:|
| `test_single_in_dist` | 160.17 | 125.70 | -21.5% |
| `test_geom_camber_rc` | 146.86 | 111.81 | -23.9% |
| `test_re_rand` | 122.78 | 101.43 | -17.4% |

- **Metrics path:** `models/model-smooth-l1-loss-20260515-124521/metrics.jsonl`
- **Action:** MERGED → new baseline val_avg/mae_surf_p = 115.17
- **Commentary:** Decisive win on every val split. The loss/metric alignment story is confirmed. 14 epochs total in 30 min; training was still improving at epoch 13. Students suggested: try beta=0.5 (more L1 character), fix scoring.py NaN bug, longer training if budget allows.

---

## 2026-05-15 12:35 — Round 1 assigned (8 PRs)

| PR | Student | Hypothesis | Knob |
|----|---------|------------|------|
| #3107 | alphonse | baseline reproduction | (none — control) |
| #3111 | askeladd | SmoothL1 loss replaces MSE | loss formulation |
| #3116 | edward   | surf_weight 10 → 25 | loss formulation |
| #3120 | fern     | slice_num 64 → 128 | capacity / resolution |
| #3124 | frieren  | mlp_ratio 2 → 4 | capacity |
| #3129 | nezuko   | bf16 autocast | throughput |
| #3132 | tanjiro  | linear LR warmup over 10% epochs | optim stability |
| #3135 | thorfinn | surf-loss (Ux,Uy,p)=(1,1,3) per-channel weights | loss formulation |

All PRs target `icml-appendix-charlie-pai2i-48h-r1`; each is a single-knob
change from the `target/train.py` defaults so effects are attributable.
