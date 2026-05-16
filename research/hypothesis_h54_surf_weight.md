## Hypothesis

**H54: Surface weight sweep — surf_weight=5 and surf_weight=20 at the H39 Arm C stack.**

The current loss aggregates volume terms and surface terms with `surf_weight=10` as the scalar multiplier on surface MAE. The primary validation metric is `val_avg/mae_surf_p` — *surface pressure error* across 4 splits. Logically, increasing the loss weight on surface terms should redirect optimization toward what we actually measure. Yet `surf_weight=10` has been the default since the start, never swept against the new stack (n_head=2 + lr=2e-3 + wd=5e-5).

**Why this could win now (and not earlier):**
1. **The improved stack reveals new gradient geometry.** With n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0, the loss has new minima. The relative balance between volume gradient signal and surface gradient signal at this configuration is fundamentally different than at the original H37b config. The optimal `surf_weight` is config-dependent.
2. **Per-channel Huber (H25) re-balanced channels but not boundary vs. interior.** δ_p=0.25/δ_vel=0.5 re-weights pressure-vs-velocity but says nothing about surface-vs-volume — those are spatial regions, not field channels. The boundary samples are the hardest (high curvature, pressure-gradient cliffs), so weighting them more might pay off independently of channel balance.
3. **Direct alignment with metric.** This is the most direct possible loss-metric alignment: increase weight on surface → reduce surface error.

**Risk:** If `surf_weight=20` is *too* surface-heavy, volume signal underfits, which can degrade surface error indirectly — the volume field's smoothness constrains plausible boundary values via PDE consistency. Going too low (e.g., surf_weight=2) under-emphasizes the metric we care about.

**Two arms bracket the current default:**

- **Arm A — surf_weight=5 + H39 Arm C stack**: Half the current default. Tests whether the model is currently *over-emphasizing* surface (e.g., if it's overfitting boundary noise).
- **Arm B — surf_weight=20 + H39 Arm C stack**: Double the current default. Tests whether more surface focus on the new stack yields direct val_avg gains.

Predicted val_avg if mechanism holds ≈ 62-64. If both regress, surf_weight=10 is optimal at this config too. If Arm B wins, the trend continues — a follow-up at surf_weight=40 is a natural next step.

## Instructions

No code changes needed. `--surf_weight` flag already exists.

**Arm A — surf_weight=5:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h54-surf5-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0 --surf_weight 5
```

**Arm B — surf_weight=20:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h54-surf20-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0 --surf_weight 20
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- **Volume MAE per-channel** (mae_p, mae_Ux, mae_Uy in the *interior*) — track volume degradation if any. Note: report this even though our primary metric is surface, because volume blow-up indicates an unstable trade-off.
- Number of epochs completed before wall, best epoch
- Per-epoch val_avg trajectory
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

## Baseline (pending merge)

**Current best — PR #3683 — H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 (thorfinn)** (in rebase, will merge shortly)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p (best of 2 seeds)** | **63.4385** |
| val_avg/mae_surf_p (2nd seed) | 65.5093 |
| test_avg/mae_surf_p (3-split, excl. cruise, best seed) | **61.3910** |
| Best epoch | 15/50 (cut by timeout) |

**Beat this: val_avg/mae_surf_p < 63.44 (best) or < 64.47 (mean of 2 seeds)**

Prior baseline (merged) — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0**:
| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 66.1060 |
| test_avg/mae_surf_p (3-split) | 64.4522 |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), AdamW (β₁=0.9, β₂=0.999), surf_weight=10.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce H39 Arm C baseline (surf_weight=10 default):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h39c-nhead2-lr2e3-wd5e5-clip1 \
  --agent charliepai2i48h3-nezuko \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```
