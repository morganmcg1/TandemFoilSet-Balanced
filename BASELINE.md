# TandemFoilSet Baseline — willow-pai2-r5

**Advisor track:** `icml-appendix-willow-pai2-r5`
**Research tag:** `willow-pai2-r5`
**W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-pai2-r5`

---

## Current best (TARGET — to be reproduced this round)

The codebase on this branch is the **bare baseline** Transolver (MSE, sw=10, no AMP, no Fourier, no SwiGLU, slice_num=64, n_layers=5). No PRs have merged on this branch yet.

The prior best result (from `kagent_v_students` round, PR #39 merged):

- **val_avg/mae_surf_p: 49.077** (best single-seed) / **49.443** (2-seed mean)
- **test_avg/mae_surf_p: 42.473** (best single-seed) / **42.450** (2-seed mean)
- W&B runs: `3fyx76kw` (s=0), `n0nwqkoz` (s=1) — group `nezuko/nl3-sn16-compound`
- Best epoch: 38 (both seeds at terminal — model NOT converged, headroom remains)

### Per-split val surface-p MAE (best single-seed `n0nwqkoz`, s=1, ep=38)

| Split | val mae_surf_p |
|-------|----------------|
| val_single_in_dist | ~57 |
| val_geom_camber_rc | ~61 |
| val_geom_camber_cruise | ~30 |
| val_re_rand | ~48 |
| **val_avg** | **49.077** |

### Per-split test surface-p MAE (best single-seed `n0nwqkoz`, best-val checkpoint)

| Split | test mae_surf_p |
|-------|-----------------|
| test_single_in_dist | 49.18 |
| test_geom_camber_rc | 54.56 |
| test_geom_camber_cruise | 25.22 |
| test_re_rand | 40.93 |
| **test_avg** | **42.473** |

### Best UNMERGED frontier (PR #32 from kagent_v_students)

Single-head triple compound (nh=1 × nl=3 × sn=16) — **never merged or 2-seed confirmed**:
- `test_avg/mae_surf_p ≈ 40.927` (best single-seed `ip8hn4tx`)
- Per-split test: SID=46.57 / RC=52.86 / Cruise=24.72 / Re_rand=39.56

This represents real headroom (~−3.5% test vs PR #39) and is the round's frontier-closure target.

### Proven recipe (kagent_v_students PR #39)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| grad_accum | 4 (effective bs=16) |
| amp | True (bf16 autocast) |
| surf_weight | 1.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 3 |
| n_head | 4 |
| **slice_num** | **8** |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 0.7 |
| ffn | SwiGLU |

### Reproduce command (target — once flags are wired by anchor PR)

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
    --n_head 4 \
    --mlp_ratio 2 \
    --seed 0 \
    --wandb_name "<student>/<experiment>"
```

**Note on bare-baseline defaults:** `train.py` Config defaults are `loss=mse` (implicit via squared-error in evaluate_split), `amp=False`, `grad_accum=1`, `fourier_features=none`, `swiglu=False`, `slice_num=64`, `n_layers=5`, `n_head=4`, `mlp_ratio=1`, `surf_weight=10`. The full flag list above MUST be passed explicitly until the anchor PR merges and updates the defaults.

---

## Baseline history (this branch)

_(none yet — round 1)_

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four val splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint on four test splits.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `icml-appendix-willow-pai2-r5`.
2. Update this file with new metric, PR number, W&B run link.
3. Commit on advisor branch.

**Multi-seed requirement:** merge claims < 5% require 2-seed anchors. Winner 2-seed mean must beat current 2-seed anchor mean by > 1σ of anchor spread.

**Noise calibration (from prior kagent_v_students round):**
- nl=3/sn=8 2-seed std: 0.517 val
- nl=3/sn=16 3-seed std: 0.300 val
- nl=3/sn=32 anchor 2-seed std: 0.376 val
