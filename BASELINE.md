# TandemFoilSet Baseline — charlie-r3

**Advisor branch:** `icml-appendix-charlie-r3`
**Research tag:** `charlie-r3`
**Track:** ICML appendix experiments
**W&B:** disabled for charlie-r3 (local metric logging only — `models/<exp>/metrics.jsonl` + `metrics.yaml`)

---

## Current best (charlie-r3)

**No baseline measured yet on this branch.** The branch is fresh from `main` with vanilla Transolver in `train.py`:

- MSE loss, surf_weight=10, no AMP, no Fourier PE, GELU FFN
- n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- AdamW lr=5e-4, cosine schedule, 50 epochs, batch_size=4
- The first merged PR will set the first numerical baseline here.

---

## Target to beat (proven SOTA from kagent_v_students round)

The previous research round (`kagent_v_students`, completed 2026-04-24) compounded 8 components and reached:

- **val_avg/mae_surf_p: 49.077** (best single seed, run `n0nwqkoz`); **49.443** (2-seed mean)
- **test_avg/mae_surf_p: 42.473** (best single seed); **42.450** (2-seed mean)

### Per-split (best single seed `n0nwqkoz`, ep=38)

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| single_in_dist | ~57 | 49.18 |
| geom_camber_rc | ~61 | 54.56 |
| geom_camber_cruise | ~30 | 25.22 |
| re_rand | ~48 | 40.93 |
| **avg** | **49.08** | **42.47** |

### Proven kagent recipe (target to reproduce)

| Component | Value | Source |
|-----------|-------|--------|
| Loss | L1 | PR #3 |
| surf_weight | 1.0 | PR #11 |
| AMP | bf16 autocast | PR #12 |
| grad_accum | 4 (effective bs=16) | PR #12 |
| Fourier PE | fixed Gaussian, m=160, σ=0.7 | PR #7 → #20 → #24 |
| FFN | SwiGLU | PR #20 |
| n_layers | 3 | PR #35 |
| slice_num | 8 | PR #39 |
| n_hidden | 128 | unchanged |
| n_head | 4 | unchanged |
| mlp_ratio | 2 | unchanged |
| lr | 5e-4 | unchanged |
| weight_decay | 1e-4 | unchanged |
| epochs | 50 | unchanged |
| optimizer | AdamW | unchanged |
| scheduler | CosineAnnealingLR(T_max=optimizer_steps) | unchanged |

### Reproduce command (kagent SOTA — needs flag wiring on charlie-r3)

```bash
cd target && python train.py \
    --agent <student> \
    --loss_type l1 \
    --surf_weight 1 \
    --amp true \
    --grad_accum 4 \
    --batch_size 4 \
    --fourier_features fixed \
    --fourier_m 160 \
    --fourier_sigma 0.7 \
    --swiglu \
    --slice_num 8 \
    --n_layers 3 \
    --epochs 50 \
    --seed 1
```

**Note**: charlie-r3's vanilla `train.py` does NOT yet expose `--loss_type`, `--amp`, `--grad_accum`, `--fourier_features`, `--fourier_m`, `--fourier_sigma`, `--swiglu`, `--slice_num`, `--n_layers`, `--n_head`, `--mlp_ratio`, or `--seed`. The first compound recipe PR (alphonse, round 1) wires these.

### Noise calibration (from kagent)

| Recipe | Seeds | val_std |
|--------|-------|---------|
| nl=3/sn=8 | 2 | 0.517 |
| nl=3/sn=16 | 3 | 0.300 |
| nl=3/sn=32 | 2 | 0.376 |

Merge threshold (charlie-r3): **winner 2-seed mean ≤ 48.9 val_avg/mae_surf_p** (using nl=3/sn=8 anchor std 0.517).

---

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `icml-appendix-charlie-r3` (use `senpai:merge-winner`).
2. Update this file with new metric, PR number, brief recipe diff.
3. Commit on advisor branch.

**Multi-seed requirement:** merge claims < 5% require 2-seed anchors. Winner 2-seed mean must beat current 2-seed anchor mean by > 1σ.

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four val splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint on four test splits. Reported in metrics.yaml.

## Provenance

Round 1 of charlie-r3 begins **2026-04-27** with 8 idle students:
charlie3-{alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn}.
