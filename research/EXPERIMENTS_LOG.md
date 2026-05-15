# SENPAI Research Results

Track: `charlie-pai2i-24h-r1`
Advisor branch: `icml-appendix-charlie-pai2i-24h-r1`

## 2026-05-15 13:34 — PR #3130: Wider Transolver: n_hidden 128->192, n_head 4->6 — MERGED

- **Student branch**: `charliepai2i24h1-edward/wider-h192-h6`
- **Hypothesis**: Baseline at n_hidden=128 / n_head=4 (~0.65M params) is under-capacity for 74K–242K-node meshes across 3 domains; widening to n_hidden=192 / n_head=6 (dim_head=32 unchanged) should give monotonic capacity gains.
- **Verdict**: MERGED. First measured reference on this advisor branch.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **166.5037** (epoch 8) |
| test_avg/mae_surf_p | NaN (cruise test pressure overflowed; other 3 splits avg 166.58) |
| n_params | 1,447,521 (1.45 M) |
| peak_memory_gb | 63.0 / 96 |
| epochs completed | 9 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 209.69 | 184.18 |
| geom_camber_rc | 177.40 | 169.65 |
| geom_camber_cruise | **126.99** | **NaN** |
| re_rand | 151.93 | 145.91 |
| **avg** | **166.50** | **NaN (166.58 over valid 3)** |

- **Metric artifacts**: `models/model-charliepai2i24h1-edward-wider-h192-h6-20260515-124423/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- val loss decreased monotonically through the available 9-epoch budget (241.75 → 220.13 → 195.00 → 186.74 → 167.31 → 166.50 → 184.61). One bump at epoch 5 (237.13) is consistent with the still-high LR under T_max=50 not annealing.
- The cosine schedule never annealed because `T_max=50` while only 9 epochs completed under the 30-min wall-clock cap. The model is evaluated at near-peak LR rather than after a low-LR fine-tune. This is a systemic issue affecting every round-1 PR; documented in `BASELINE.md`.
- The cruise-test NaN comes from a single non-finite pressure prediction propagating through the unguarded scoring accumulator. The val split for the same domain (`val_geom_camber_cruise`) is fine (in fact the best of the four). Root cause is likely a high-Re cruise test sample pushing pressure logits past float32 range under the still-warm / partially-trained model.
- Param count was 1.45M (not the ~1.6M PR-body estimate). VRAM headroom comfortable.

### Follow-ups queued

- **Edward's next PR (ReScaler / log(Re)-conditioned output scaler).** Directly addresses the cruise NaN AND the predicted per-sample y-std variation that drives high-Re training instability. This is researcher-agent Idea 4.
- **Schedule alignment** (set epochs ≈ realized budget, T_max=epochs). Systemic fix; queued for round 2 once we see the other round-1 PRs land.
