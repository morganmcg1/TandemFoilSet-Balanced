# Baseline — icml-appendix-charlie-pai2g-48h-r3

## 2026-05-13 09:00 — PR #1995: n_layers=5 + T_max=14 (edward)

**New best: `val_avg/mae_surf_p` = 47.478** (epoch 14, 30-min wall-clock cap, surf_weight=10)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | **5** ← updated |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Normalization | RMSNorm |
| MLP activation | GeGLU (gated) |
| Loss | L1 (MAE) in normalized space, `surf_weight=10` ← note: sw=5 not yet tested on n_layers=5 |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR **T_max=14** ← updated |
| `epochs` | **14** ← updated |
| `batch_size` | 4 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |
| `n_params` | 826,071 (−15.7% vs n_layers=6 ≈ 979,995) |

> **Mechanism:** n_layers=5 reduces per-epoch time from ~138s to ~116s, enabling 14 epochs in the 30-min budget (vs 12 epochs with n_layers=6). T_max=14 aligns cosine decay to the new epoch count — the same mechanism that won PR #1793 (T_max=12 for 12 epochs), now extended by 2 extra low-LR refinement epochs. This is NOT a capacity trade-off: the model trains for longer and converges better. Peak VRAM also drops from ~50 GB to ~40 GB (−20%). surf_weight=5 compound not yet tested on this stack — immediate priority.

### Val metrics (best checkpoint, epoch 14/14)

| Split | `mae_surf_p` | `mae_vol_p` |
|---|---|---|
| val_single_in_dist | 52.253 | 62.425 |
| val_geom_camber_rc | **60.809** | 67.690 |
| val_geom_camber_cruise | 29.174 | 30.132 |
| val_re_rand | 47.675 | 49.484 |
| **val_avg/mae_surf_p** | **47.478** | — |

### Test metrics (best-val checkpoint, epoch 14)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 46.980 |
| test_geom_camber_rc | 54.123 |
| test_geom_camber_cruise | 24.263 |
| test_re_rand | 39.794 |
| **test_avg/mae_surf_p** | **41.290** |

### Improvement vs PR #1956 baseline (51.040 val / 44.390 test)

| Split | Old val (n_layers=6, T=12, sw=5) | New val (n_layers=5, T=14, sw=10) | Δ val |
|---|---|---|---|
| single_in_dist | 56.933 | 52.253 | −8.2% |
| geom_camber_rc | 64.886 | 60.809 | −6.3% |
| geom_camber_cruise | 31.056 | 29.174 | −6.1% |
| re_rand | 51.287 | 47.675 | −7.0% |
| **avg** | **51.040** | **47.478** | **−6.98% ✓** |

**Reproduce:**
```bash
cd target/ && python train.py --epochs 14 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10
```

**Metric artifacts:** `models/model-charliepai2g48h3-edward-n-layers-5-20260513-065528/metrics.yaml`

---

## 2026-05-13 07:25 — PR #1956: T_max=12 + surf_weight=5 compound (nezuko)

**New best: `val_avg/mae_surf_p` = 51.040** (epoch 12, 30-min wall-clock cap, surf_weight=5)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Normalization | RMSNorm |
| MLP activation | GeGLU (gated) |
| Loss | L1 (MAE) in normalized space, **`surf_weight=5`** ← updated |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR T_max=12 |
| `epochs` | 12 |
| `batch_size` | 4 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

> **Compound mechanism:** T_max=12 aligns cosine decay to the actual epoch budget (LR→0 at epoch 12). surf_weight=5 reallocates L1 gradient from surface to volume nodes, yielding −10% to −14% volume MAE improvement across all splits. Both mechanisms are orthogonal and compound sub-linearly (−3.33% additional val improvement on top of T_max=12 baseline, vs the −9.03% surf_weight=5 achieved on the older T_max=50 stack).

### Val metrics (best checkpoint, epoch 12/12)

| Split | `mae_surf_p` | `mae_vol_p` |
|---|---|---|
| val_single_in_dist | 56.933 | 60.506 |
| val_geom_camber_rc | **64.886** | 65.720 |
| val_geom_camber_cruise | 31.056 | 28.904 |
| val_re_rand | 51.287 | 48.815 |
| **val_avg/mae_surf_p** | **51.040** | — |

### Test metrics (best-val checkpoint, epoch 12)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 50.459 |
| test_geom_camber_rc | 59.341 |
| test_geom_camber_cruise | 25.501 |
| test_re_rand | 42.260 |
| **test_avg/mae_surf_p** | **44.390** |

### Improvement vs PR #1793 baseline (52.798 val / 44.972 test)

| Split | Old val (T=12, sw=10) | New val (T=12, sw=5) | Δ val |
|---|---|---|---|
| single_in_dist | 58.907 | 56.933 | −3.35% |
| geom_camber_rc | 67.658 | 64.886 | −4.10% |
| geom_camber_cruise | 33.380 | 31.056 | −6.96% |
| re_rand | 51.248 | 51.287 | +0.08% |
| **avg** | **52.798** | **51.040** | **−3.33%** |

**Test: 44.390 vs 44.972 = −1.29%**

### Metric artifacts
`models/model-charliepai2g48h3-nezuko-tmax-12-surf-weight-5-compound-20260513-055720/metrics.jsonl`

### Reproduce command
```bash
cd target/ && python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 5
```

---

## 2026-05-13 06:05 — PR #1793: CosineAnnealingLR T_max=12 on RMSNorm+GeGLU+Lion (nezuko)

**New best: `val_avg/mae_surf_p` = 52.798** (epoch 12, 30-min wall-clock cap, surf_weight=10)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Normalization | RMSNorm |
| MLP activation | GeGLU (gated) |
| Loss | L1 (MAE) in normalized space, `surf_weight=10` |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| **Scheduler** | **CosineAnnealingLR T_max=12** ← aligned to actual epoch budget |
| **`epochs`** | **12** ← matches T_max for proper cosine decay |
| `batch_size` | 4 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

> **T_max=12 mechanism:** Previous schedule used T_max=50 (full configured epochs), but only ~14 epochs run in the 30-min cap → only ~13% of the cosine decay actually fires. Aligning T_max to actual epochs lets the cosine schedule fully complete: LR drops from 1e-4 → 0 over 12 epochs. Lion's sign(m) updates get the proper 10× finer steps in late epochs (visible in trajectory: epoch 9→12 dropped 11.3% as LR went 2.07e-5 → 0).
>
> **Note on surf_weight:** This run used surf_weight=10 because the student's rebase pre-dated PR #1836 (surf_weight=5 merge). The result still beats the surf_weight=5 baseline by −7.9% val / −8.9% test, but the predicted compound `T_max=12 + surf_weight=5` (assigned next) should yield further improvement.

### Val metrics (best checkpoint, epoch 12/12)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 58.907 |
| val_geom_camber_rc | 67.658 |
| val_geom_camber_cruise | 33.380 |
| val_re_rand | 51.248 |
| **val_avg/mae_surf_p** | **52.798** |

### Test metrics (best-val checkpoint, epoch 12)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | ~50.24 |
| test_geom_camber_rc | ~59.56 |
| test_geom_camber_cruise | ~27.74 |
| test_re_rand | ~42.35 |
| **test_avg/mae_surf_p** | **44.972** |

### Improvement vs PR #1836 baseline (57.328 val / 49.387 test)

| Split | Old val (sw=5,T=50) | New val (sw=10,T=12) | Δ val |
|---|---|---|---|
| single_in_dist | 60.960 | 58.907 | −3.4% |
| geom_camber_rc | 72.044 | 67.658 | −6.1% |
| geom_camber_cruise | 38.721 | 33.380 | −13.8% |
| re_rand | 57.586 | 51.248 | −11.0% |
| **avg** | **57.328** | **52.798** | **−7.9%** |

### Improvement vs PR #1837 same-surf_weight baseline (63.017 val with sw=10)

| Split | Old val (sw=10,T=50) | New val (sw=10,T=12) | Δ |
|---|---|---|---|
| single_in_dist | 76.710 | 58.907 | **−23.2%** |
| geom_camber_rc | 73.930 | 67.658 | −8.5% |
| geom_camber_cruise | 40.746 | 33.380 | −18.1% |
| re_rand | 60.683 | 51.248 | −15.5% |
| **avg** | **63.017** | **52.798** | **−16.2%** |

- **Metric artifacts:** `models/model-charliepai2g48h3-nezuko-lion-tmax-12-aligned-v2-20260513-045129/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10`

---

## 2026-05-13 05:45 — PR #1836: surf_weight=5 on RMSNorm+GeGLU+Lion (thorfinn)

**New best: `val_avg/mae_surf_p` = 57.328** (epoch 14, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Normalization | RMSNorm |
| MLP activation | GeGLU (gated) |
| Loss | L1 (MAE) in normalized space, **surf_weight=5** ← changed |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

> **surf_weight=5 mechanism:** Halving the surface/volume loss weighting (10→5) reallocates gradient budget toward volume nodes, producing richer volumetric features. Surface accuracy then improves via better geometric context (confirmed: vol MAE improved −7% to −26% across all splits). Hardest splits benefit most: single_in_dist −20.5% val, geom_camber_rc −2.6% val (already improved by RMSNorm in previous round).

### Val metrics (best checkpoint, epoch 14)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 60.960 |
| val_geom_camber_rc | **72.044** |
| val_geom_camber_cruise | 38.721 |
| val_re_rand | 57.586 |
| **val_avg/mae_surf_p** | **57.328** |

### Test metrics (best-val checkpoint, epoch 14)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 53.010 |
| test_geom_camber_rc | **62.463** |
| test_geom_camber_cruise | 32.843 |
| test_re_rand | 49.231 |
| **test_avg/mae_surf_p** | **49.387** |

### Improvement vs PR #1837 baseline (63.017 val / 54.731 test)

| Split | Old val | New val | Δ val | Old test | New test | Δ test |
|---|---|---|---|---|---|---|
| single_in_dist | 76.710 | 60.960 | −20.5% | 67.384 | 53.010 | −21.3% |
| geom_camber_rc | 73.930 | 72.044 | −2.6% | 64.508 | 62.463 | −3.2% |
| geom_camber_cruise | 40.746 | 38.721 | −5.0% | 34.707 | 32.843 | −5.4% |
| re_rand | 60.683 | 57.586 | −5.1% | 52.327 | 49.231 | −5.9% |
| **avg** | **63.017** | **57.328** | **−9.0%** | **54.731** | **49.387** | **−9.8%** |

- **Metric artifacts:** `models/model-charliepai2g48h3-thorfinn-surf-weight-5-rmsnorm-geglu-lion-20260513-041441/metrics.jsonl`
- **Reproduce:** `cd "target/" && python train.py --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 5`

---

## 2026-05-13 04:30 — PR #1837: RMSNorm in TransolverBlock (frieren)

**New best: `val_avg/mae_surf_p` = 63.017** (epoch 13, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| **Normalization** | **RMSNorm** (replaces LayerNorm) ← changed |
| MLP activation | GeGLU (gated) |
| Loss | L1 (MAE) in normalized space, surf_weight=10 |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

> **RMSNorm implementation:** replaces all `nn.LayerNorm` instances in `TransolverBlock` with `nn.RMSNorm`. RMSNorm drops mean-centering (computes scale only via RMS), ~3.4% faster than LayerNorm with bf16 → 14 epochs fit in 30 min vs 13. Best checkpoint was epoch 13 (not 14).

### Val metrics (best checkpoint, epoch 13)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 76.710 |
| val_geom_camber_rc | **73.930** |
| val_geom_camber_cruise | 40.746 |
| val_re_rand | 60.683 |
| **val_avg/mae_surf_p** | **63.017** |

### Test metrics (best-val checkpoint, epoch 13)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 67.384 |
| test_geom_camber_rc | **64.508** |
| test_geom_camber_cruise | 34.707 |
| test_re_rand | 52.327 |
| **test_avg/mae_surf_p** | **54.731** |

### Improvement vs PR #1769 baseline (64.918 val / 58.171 test)

| Split | Old val | New val | Δ val | Old test | New test | Δ test |
|---|---|---|---|---|---|---|
| single_in_dist | 72.021 | 76.710 | +6.5% | 64.947 | 67.384 | +3.7% |
| geom_camber_rc | 89.234 | 73.930 | **−17.2%** | 80.467 | 64.508 | **−19.8%** |
| geom_camber_cruise | 37.058 | 40.746 | +10.0% | 32.329 | 34.707 | +7.4% |
| re_rand | 61.359 | 60.683 | −1.1% | 54.939 | 52.327 | −4.7% |
| **avg** | **64.918** | **63.017** | **−2.9%** | **58.171** | **54.731** | **−5.9%** |

> RMSNorm's removal of mean-centering allowed the model to attend to relative pressure magnitudes more cleanly. This had the most pronounced effect on `geom_camber_rc` (the hardest OOD geometry split), cutting it by −17.2% val / −19.8% test. `single_in_dist` and `geom_camber_cruise` regressed slightly — consistent with RMSNorm routing gradient toward harder splits via its changed scale normalization.

### Metric artifacts

`models/model-charliepai2g48h3-frieren-rmsnorm-geglu-lion-20260513-031035/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: RMSNorm, GeGLU activation, Lion optimizer, and bf16 are now all defaults in `train.py` (merged from PRs #1769, #1837).

---

## 2026-05-13 02:45 — PR #1769: GeGLU activation + Lion optimizer (tanjiro)

**New best: `val_avg/mae_surf_p` = 64.918** (epoch 13, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| **MLP activation** | **GeGLU (gated)** ← changed |
| Loss | L1 (MAE) in normalized space, surf_weight=10 |
| Optimizer | Lion, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

> **GeGLU implementation:** fc1 output split into two halves; `GeGLU(x) = GELU(x1) * x2`. fc2 input dim = n_hidden * mlp_ratio / 2. With mlp_ratio=4: fc1 projects n_hidden → 4*n_hidden, splits into 2×n_hidden chunks, gate reduces to 2*n_hidden, fc2 projects 2*n_hidden → n_hidden. Per-epoch time ~143s (essentially identical to GELU+Lion thanks to bf16 absorbing FLOP overhead).

### Val metrics (best checkpoint, epoch 13)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 72.021 |
| val_geom_camber_rc | 89.234 |
| val_geom_camber_cruise | 37.058 |
| val_re_rand | 61.359 |
| **val_avg/mae_surf_p** | **64.918** |

### Test metrics (best-val checkpoint, epoch 13)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 64.947 |
| test_geom_camber_rc | 80.467 |
| test_geom_camber_cruise | 32.329 |
| test_re_rand | 54.939 |
| **test_avg/mae_surf_p** | **58.171** |

### Improvement vs PR #1725 baseline (86.938 val / 77.990 test)

| Split | Old val | New val | Δ val | Old test | New test | Δ test |
|---|---|---|---|---|---|---|
| single_in_dist | 98.979 | 72.021 | −27.2% | 91.606 | 64.947 | −29.1% |
| geom_camber_rc | 104.737 | 89.234 | −14.8% | 92.561 | 80.467 | −13.1% |
| geom_camber_cruise | 62.041 | 37.058 | **−40.3%** | 52.841 | 32.329 | **−38.8%** |
| re_rand | 81.995 | 61.359 | −25.2% | 74.952 | 54.939 | −26.7% |
| **avg** | **86.938** | **64.918** | **−25.3%** | **77.990** | **58.171** | **−25.4%** |

> Largest single-PR improvement in the research programme (+Lion was -14.3%, +L1 was -20.5%; GeGLU+Lion exceeds both). Training was still descending at epoch 13 cutoff — significant headroom remains. The cruise split saw the most dramatic gain (-40.3%), suggesting GeGLU routing is especially effective for high-camber transonic regime features.

### Metric artifacts

`models/model-charliepai2g48h3-tanjiro-geglu-lion-20260513-012007/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: GeGLU activation is now the default in `train.py` (merged from PR #1769). Lion optimizer and bf16 also already defaults.

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 64.918** — lower is better.

Test metric benchmark: **`test_avg/mae_surf_p` < 58.171**.

Hardest splits: geom_camber_rc (89.2 val) and single_in_dist (72.0 val). Cruise improved most dramatically (-40.3%) and is now the easiest split at 37.1 val.

---

## 2026-05-13 01:45 — PR #1725: Lion optimizer lr=1e-4 (edward)

**New best: `val_avg/mae_surf_p` = 86.938** (epoch 11, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Loss | L1 (MAE) in normalized space, surf_weight=10 |
| Optimizer | **Lion, lr=1e-4, weight_decay=1e-4** ← changed |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Mixed precision | bf16 autocast |
| Run cap | 30 min wall clock per training execution |

### Val metrics (best checkpoint, epoch 11)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 98.979 |
| val_geom_camber_rc | 104.737 |
| val_geom_camber_cruise | 62.041 |
| val_re_rand | 81.995 |
| **val_avg/mae_surf_p** | **86.938** |

### Test metrics (best-val checkpoint, epoch 11)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 91.606 |
| test_geom_camber_rc | 92.561 |
| test_geom_camber_cruise | 52.841 |
| test_re_rand | 74.952 |
| **test_avg/mae_surf_p** | **77.990** |

### Improvement vs PR #1724 baseline (101.463)

| Split | Old | New | Δ |
|---|---|---|---|
| val_single_in_dist | 120.699 | 98.979 | −18.0% |
| val_geom_camber_rc | 116.096 | 104.737 | −9.8% |
| val_geom_camber_cruise | 73.667 | 62.041 | −15.8% |
| val_re_rand | 95.391 | 81.995 | −14.0% |
| **val_avg** | **101.463** | **86.938** | **−14.3%** |

> Run hit the 30-min wall-clock cap at epoch 11/50, still improving monotonically. Significant convergence headroom remains — Lion+lr=1e-4 was still descending at cutoff. With proper LR tuning and longer budget, further gains likely.

### Metric artifacts

`models/model-charliepai2g48h3-edward-lion-optimizer-20260513-001607/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: Lion optimizer is now the default in `train.py` (merged from PR #1725). Pass `--lr 1e-4` explicitly (Lion's optimal LR is 3-10× lower than Adam's 5e-4).

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 86.938** — lower is better.

Test metric benchmark: **`test_avg/mae_surf_p` < 77.990**.

Per-split breakdown: single_in_dist=98.979, geom_camber_rc=104.737, geom_camber_cruise=62.041, re_rand=81.995.
geom_camber_rc (104.7) and single_in_dist (99.0) are now the hardest splits.

---

## 2026-05-13 01:20 — PR #1724: bf16 mixed precision (alphonse)

**New best: `val_avg/mae_surf_p` = 101.463** (epoch 14, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 6 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 4 |
| Loss | L1 (MAE) in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Mixed precision | **bf16 autocast** ← changed |
| Run cap | 30 min wall clock per training execution |

### Val metrics (best checkpoint, epoch 14)

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 120.699 |
| val_geom_camber_rc | 116.096 |
| val_geom_camber_cruise | 73.667 |
| val_re_rand | 95.391 |
| **val_avg/mae_surf_p** | **101.463** |

### Test metrics (best-val checkpoint, epoch 14)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 108.025 |
| test_geom_camber_rc | 107.822 |
| test_geom_camber_cruise | 63.152 |
| test_re_rand | 87.800 |
| **test_avg/mae_surf_p** | **91.700** |

### Throughput
- 138.5s/epoch (1.26× faster vs fp32 baseline of ~175s/epoch)
- 14 epochs in 30 min (vs 13 with fp32)

### Improvement vs PR #1358 baseline (101.810)

| Split | Old | New | Δ |
|---|---|---|---|
| val_single_in_dist | 124.150 | 120.699 | −2.8% |
| val_geom_camber_rc | 112.699 | 116.096 | +3.0% |
| val_geom_camber_cruise | 76.570 | 73.667 | −3.8% |
| val_re_rand | 93.820 | 95.391 | +1.7% |
| **val_avg** | **101.810** | **101.463** | **−0.34%** |

### Metric artifacts

`models/model-charliepai2g48h3-alphonse-bf16-mixed-precision-20260513-001819/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: bf16 autocast is now the default in `train.py` (merged from PR #1724). No extra flags needed.

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 101.463** — lower is better.

Test metric benchmark: **`test_avg/mae_surf_p` < 91.700**.

The hardest splits are `val_geom_camber_rc` (116.1) and `val_single_in_dist` (120.7). Note: bf16 improved cruise and in_dist but slightly regressed rc and re_rand — noise at this scale.

---

## 2026-05-12 21:10 — PR #1358: L1 (MAE) loss in normalized space

**New best: `val_avg/mae_surf_p` = 101.810** (epoch 13, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 (run config) / **6** (merged default) |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 (run config) / **4** (merged default) |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | **L1 (MAE) in normalized space** ← changed |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution |

> **Note on arch:** Alphonse's run used n_layers=5, mlp_ratio=2 (branched before PR #1408 and #1392).
> The merged train.py now defaults to n_layers=6, mlp_ratio=4 + L1 loss (stacked). The measured
> improvement below is L1 loss alone; the stacked result should be even better.

> **Note on NaN-fix:** Alphonse also added a `train.py::evaluate_split` guard that skips non-finite
> GT samples before calling the scorer. This makes test metrics finite for the first time —
> `test_avg/mae_surf_p = 91.708` is the first reliable test number on this branch.

### Val metrics (best checkpoint, epoch 13)

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| val_single_in_dist | 124.150 | — | — |
| val_geom_camber_rc | 112.699 | — | — |
| val_geom_camber_cruise | 76.570 | — | — |
| val_re_rand | 93.820 | — | — |
| **val_avg/mae_surf_p** | **101.810** | — | — |

### Improvement vs PR #1392 baseline (128.127)

| Split | Old | New | Δ |
|---|---|---|---|
| val_single_in_dist | 159.746 | 124.150 | −22.3% |
| val_geom_camber_rc | 136.513 | 112.699 | −17.4% |
| val_geom_camber_cruise | 102.432 | 76.570 | −25.3% |
| val_re_rand | 113.819 | 93.820 | −17.6% |
| **val_avg** | **128.127** | **101.810** | **−20.5%** |

### Test metrics (best-val checkpoint, epoch 13) — all finite

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 110.726 |
| test_geom_camber_rc | 99.692 |
| test_geom_camber_cruise | 66.879 (first finite cruise test result!) |
| test_re_rand | 89.536 |
| **test_avg/mae_surf_p** | **91.708** |

### Metric artifacts

`models/model-l1-loss-e50-20260512-195549/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: L1 loss is now the default in `train.py` (merged from PR #1358). `n_layers=6, mlp_ratio=4` also
baked in. No extra flags needed.

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 101.810** — lower is better.

Test metric benchmark: **`test_avg/mae_surf_p` < 91.708**.

The hardest splits are `val_single_in_dist` (124.2) and `val_geom_camber_rc` (112.7).

---

## 2026-05-12 19:30 — PR #1392: n_layers 5 → 6 (moderate depth increase)

**New best: `val_avg/mae_surf_p` = 128.127** (epoch 12, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | **6** ← changed |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | **4** (default from PR #1408) |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | MSE in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution |

> **Note:** The empirical run used `mlp_ratio=2` (branched before PR #1408). The merged train.py
> now defaults to `mlp_ratio=4, n_layers=6`. Future runs stack both improvements.

### Val metrics (best checkpoint, epoch 12)

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| val_single_in_dist | 159.746 | 1.890 | 0.915 |
| val_geom_camber_rc | 136.513 | 3.068 | 1.235 |
| val_geom_camber_cruise | 102.432 | 1.675 | 0.656 |
| val_re_rand | 113.819 | 2.338 | 0.914 |
| **val_avg/mae_surf_p** | **128.127** | — | — |

### Improvement vs PR #1408 baseline (141.356)

| Split | Old | New | Δ |
|---|---|---|---|
| val_single_in_dist | 171.424 | 159.746 | −6.8% |
| val_geom_camber_rc | 159.804 | 136.513 | −14.6% |
| val_geom_camber_cruise | 104.607 | 102.432 | −2.1% |
| val_re_rand | 129.589 | 113.819 | −12.2% |
| **val_avg** | **141.356** | **128.127** | **−9.4%** |

### Test metrics (best-val checkpoint, epoch 12)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 145.477 |
| test_geom_camber_rc | 122.697 |
| test_geom_camber_cruise | **NaN** (scoring bug — GT sample 20 has -inf pressure) |
| test_re_rand | 114.851 |
| test_avg (3 finite splits) | **~127.68** |
| **test_avg/mae_surf_p** | NaN (blocked by cruise bug) |

> **Note on test NaN:** Same scorer bug as PR #1408. Use `val_avg/mae_surf_p` as primary ranking
> metric for this branch.

### Metric artifacts

`models/model-charliepai2g48h3-nezuko-deeper-transolver-6layers-20260512-191742/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: `mlp_ratio=4, n_layers=6` are now the defaults in `train.py` (merged from PRs #1408, #1392).
No extra flags needed.

---

## 2026-05-12 18:56 — PR #1408: MLP expansion ratio 2 → 4 (canonical transformer recipe)

**Previous best: `val_avg/mae_surf_p` = 141.356** (epoch 13, 30-min wall-clock cap)

| Hyperparameter | Value |
|---|---|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | **4** ← changed |
| `space_dim` | 2 |
| `unified_pos` | False |
| Loss | MSE in normalized space, surf_weight=10 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Run cap | 30 min wall clock per training execution |

### Val metrics (best checkpoint, epoch 13)

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| val_single_in_dist | 171.424 | 2.560 | 1.119 |
| val_geom_camber_rc | 159.804 | 3.611 | 1.420 |
| val_geom_camber_cruise | 104.607 | 1.759 | 0.718 |
| val_re_rand | 129.589 | 2.258 | 0.940 |
| **val_avg/mae_surf_p** | **141.356** | — | — |

### Test metrics (best-val checkpoint)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 149.585 |
| test_geom_camber_rc | 142.249 |
| test_geom_camber_cruise | **NaN** (scoring bug — GT sample 20 has -inf pressure; `inf * 0 = NaN` in float64 accumulator) |
| test_re_rand | 126.704 |
| test_avg (3 finite splits) | **~139.51** |
| **test_avg/mae_surf_p** | NaN (blocked by cruise bug) |

> **Note on test NaN:** `data/scoring.py` (read-only) computes `err = (pred - y).abs()` before
> applying the per-sample finite-GT mask, so a single `-inf` in GT propagates to NaN even on
> masked (zero-weight) nodes. This affects all models on `test_geom_camber_cruise` until the
> scorer is patched. Use `val_avg/mae_surf_p` as the primary ranking metric for this branch.

### Metric artifacts

`models/model-charliepai2g48h3-thorfinn-mlp-ratio-4-20260512-175522/metrics.jsonl`

### Reproduce

```bash
cd target/ && python train.py \
  --agent <student> \
  --experiment_name <name> \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10
```

Note: `mlp_ratio=4` is now the default in `train.py` (merged from PR #1408). No extra flag needed.

---

## Benchmark to beat

**`val_avg/mae_surf_p` < 128.127** — lower is better.

All new student experiments should compare against this number. The per-split breakdown above
shows `val_single_in_dist` (159.7) and `val_geom_camber_rc` (136.5) are the hardest splits to
improve; `val_geom_camber_cruise` (102.4) and `val_re_rand` (113.8) are relatively stronger.
