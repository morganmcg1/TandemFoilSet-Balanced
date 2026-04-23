# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #3 — frieren: L1 loss with surf_weight=10**
- **val_avg/mae_surf_p: 103.036** (lower is better)
- W&B run: `w2jsabii` (`frieren/loss-l1-sw10`)
- Best epoch: 14 (run timeout-bounded at 30 min; still improving)
- test_avg/mae_surf_p: NaN (pre-existing +Inf bug in `test_geom_camber_cruise/000020.pt` — see [scoring bug issue](https://github.com/morganmcg1/tandemfoil2/issues))

### Per-split val surface-p MAE (best checkpoint)

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 133.194 |
| val_geom_camber_rc | 117.209 |
| val_geom_camber_cruise | 70.109 |
| val_re_rand | 91.634 |
| **val_avg** | **103.036** |

### Current default config (post-merge)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| **loss** | **L1 (abs, vol + surf_weight × surf) in normalized space** |

Reproduce:
```bash
cd target && python train.py \
    --agent <student> \
    --loss_type l1 \
    --surf_weight 10 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-23 21:40 — PR #3: frieren Huber/L1 loss reformulation

- **val_avg/mae_surf_p: 103.036** (previous: no baseline on this track)
- W&B run: `w2jsabii` (group: `frieren/loss-reformulation-v2`)
- Change: L1 loss in normalized space instead of MSE. surf_weight=10 unchanged.
- Delta: −21.9% vs MSE baseline at same run budget (131.985).
- Wins uniformly: −27.7% in_dist, −29.8% camber_rc, −39.5% camber_cruise, −32.5% re_rand.

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four validation splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same quantity, computed from the best-val checkpoint on the four held-out test splits. Currently blocked by +Inf bug in `test_geom_camber_cruise/000020.pt`.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here, the advisor:

1. Squash-merges the winning PR into `kagent_v_students`.
2. Updates this file with the new metric, PR number, and W&B run link.
3. Commits the update on the advisor branch.
