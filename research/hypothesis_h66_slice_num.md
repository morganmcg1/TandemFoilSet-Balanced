## Hypothesis

**H66: Wider slice token representation (slice_num 96, 128) at the new n_layers=4 GEGLU baseline.**

H60 (just merged) demonstrated that n_layers=4 at GEGLU base wins (val=57.5750 vs n_layers=5 baseline 58.6268). The shallower depth freed ~25% wall-clock per epoch (113s vs ~130s/ep at n_layers=5). This budget surplus opens a new lever: **widening the slice token representation**.

**Mechanism — what slice_num controls:** Transolver's `PhysicsAttention` block (line 153 in train.py) projects mesh nodes into a fixed-size "slice token" space via `in_project_slice = nn.Linear(dim_head, slice_num)`. These slice tokens are a learned compression of the mesh — each token attends over a soft-assigned subset of nodes that share fluid-mechanical features. The number of slice tokens controls the **bottleneck capacity** of the spatial aggregation step.

- slice_num=64 (current): each token aggregates ~30 nodes on a typical mesh. May not be enough resolution to capture both the boundary-layer dynamics near the airfoil surface AND the wake region simultaneously.
- slice_num=96: 50% more tokens. Each token aggregates ~20 nodes. Finer spatial selectivity, potentially captures more localized flow features.
- slice_num=128: 2x tokens. Each token aggregates ~15 nodes. Approaches the granularity needed for resolving thin boundary-layer pressure cliffs near trailing edges.

**Note on H10 history:** H10 (R1, tanjiro) tested slice_num=96/128 at the original baseline (val=114.6, FiLM-only). Those results are stale — the model has improved ~57 pts since, and the bottleneck/capacity tradeoff is fundamentally different at the current scale. Worth re-testing at the new GEGLU n_layers=4 baseline.

**Two arms:**
- **Arm A: slice_num=96** — moderate widening, should complete full ~15 epochs within wall
- **Arm B: slice_num=128** — full doubling, may complete fewer epochs (predicted ~12-13 epochs)

**Predicted:** Arm A ≈ 56.5-57.5 (small but reliable improvement from finer spatial selectivity). Arm B uncertain — extra capacity may overfit the small training set (1499 samples) or undertrain due to wall budget.

**Risk:** slice_num=128 might double the `in_project_slice` parameter count *per attention block* (currently 64×64=4096 per block, becomes 64×128=8192 per block). Total model params: from 856k → ~890k for Arm A, ~920k for Arm B. Negligible. Real risk is wall-budget bottleneck.

## Instructions

The `slice_num` config field is **hardcoded at 64** in `train.py` line 499. You'll need to expose it.

```python
# In Config dataclass (around line 455):
slice_num: int = 64   # Transolver attention slice token count

# In model_config (around line 499), change:
slice_num=64,
# to:
slice_num=cfg.slice_num,
```

This is the same pattern as the H60 n_layers fix. Two lines: one new Config field, one wire-through.

Run both arms:

```bash
# Arm A — slice_num=96 at new GEGLU n_layers=4 baseline
cd target/ && python train.py --epochs 50 \
  --experiment_name h66-slice96-geglu-n4 \
  --agent charliepai2i48h3-thorfinn \
  --slice_num 96 \
  --n_layers 4 --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — slice_num=128 at new GEGLU n_layers=4 baseline
cd target/ && python train.py --epochs 50 \
  --experiment_name h66-slice128-geglu-n4 \
  --agent charliepai2i48h3-thorfinn \
  --slice_num 128 \
  --n_layers 4 --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags use current merged defaults: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, T_max=15.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- **Epochs completed before wall** (CRITICAL — wider slices may complete fewer epochs)
- Best epoch and per-epoch val_avg trajectory
- **Parameter count** for each arm: `sum(p.numel() for p in model.parameters())`
- Peak GPU memory and **mean s/epoch** (slice_num=128 will be slower)
- Final-epoch comparison vs H60 n_layers=4 baseline at the same epoch (account for fewer completed epochs)
- **Per-split OOD gain breakdown** — slice_num gain should land most on geometry-OOD (camber_rc) where local spatial structure matters most

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #3968 — H60 Arm B: n_layers=4 + GEGLU (thorfinn, slice_num=64)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **57.5750** |
| val_single_in_dist/mae_surf_p | 63.3430 |
| val_geom_camber_rc/mae_surf_p | 72.1854 |
| val_geom_camber_cruise/mae_surf_p | 37.7532 |
| val_re_rand/mae_surf_p | 57.0183 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.4610** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + **n_layers=4** + **slice_num=64** (current). n_params=856,587. Mean s/epoch=113s. Best epoch=15.

**Beat this: val_avg/mae_surf_p < 57.5750**

Predicted: Arm A (slice_num=96) ≈ 56.5-57.5. Arm B (slice_num=128) uncertain — better signal-to-noise per token vs wall-budget undertraining.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
