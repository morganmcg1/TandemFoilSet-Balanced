## Hypothesis

**H60: Revisit n_layers depth sweep in the new GEGLU architectural context — does depth become a fresh lever?**

In the pre-GEGLU era, n_layers=3 won in isolation (H42) but **destroyed** the n_head=2 win when stacked (H42 Arm C: n_layers=3 + n_head=2 + lr=2e-3 + wd=5e-5 regressed). The interpretation was "capacity reductions destroy each other" — n_head=2 and n_layers=3 both reduce model capacity and the combination undershoots.

GEGLU adds parameter capacity *per FFN block* (one extra weight matrix per gate path: +n_hidden² params per block in our config, +0.49 M extra params total at 5 layers). This changes the model's effective capacity per layer — fewer layers may now have *more* effective capacity than they did at n_layers=5 with vanilla FFN, because each GEGLU layer is doing more representational work via the gate.

**Two hypotheses to test:**
- **Arm A (deeper):** n_layers=6 — adds one extra GEGLU block. If GEGLU's per-block expressivity is the bottleneck, scaling depth gives more spatial-selectivity stages, helping the boundary-layer gradient structure compose hierarchically.
- **Arm B (shallower):** n_layers=4 — removes one GEGLU block. If GEGLU's per-block capacity is now sufficient and we were over-parameterized at n_layers=5, shallower is faster + still expressive.

Both are 1-block deltas from the current default (n_layers=5). These are conservative — H33 (n_hidden=192/256) showed that width changes hurt at n_head=4; depth changes haven't been tested in the new GEGLU context.

**Mechanism:** Transolver's slice-token attention aggregates spatial information across the geometry. Each layer = one round of slice-token communication + one round of GEGLU spatial selectivity. With GEGLU, each layer is now more selective, so the question becomes: does more or fewer rounds work better?

**Predicted:**
- Arm A (deeper, n_layers=6) more likely to help: extra capacity in GEGLU regime, +20% step cost but only 2 epochs of total runtime added. Predicted val_avg = 57-58 if depth helps.
- Arm B (shallower, n_layers=4) faster but lower capacity. Predicted val_avg = 58-60. Likely worse than Arm A but provides a clean efficiency comparison.

**Risk:** n_layers=6 may exceed the 30-min wall budget at the same epoch count → fewer epochs trained → undertrained → worse final val. If Arm A hits timeout at epoch <12, the apples-to-apples comparison breaks. Mitigation: report epoch_completed alongside val_avg so we can normalize.

## Instructions

The `--n_layers` flag already exists on advisor branch (added in H42). No new code needed.

Run two arms:

```bash
# Arm A — n_layers=6 (deeper GEGLU stack)
cd target/ && python train.py --epochs 50 \
  --experiment_name h60-geglu-nlayers6 \
  --agent charliepai2i48h3-thorfinn \
  --n_layers 6 \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — n_layers=4 (shallower GEGLU stack)
cd target/ && python train.py --epochs 50 \
  --experiment_name h60-geglu-nlayers4 \
  --agent charliepai2i48h3-thorfinn \
  --n_layers 4 \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15.

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Epochs completed before wall (CRITICAL — this affects interpretation)
- Best epoch
- Per-epoch val_avg trajectory for both arms — overlay against H48 GEGLU (n_layers=5) baseline trajectory
- Parameter count for each arm (vs H48's 891k baseline)
- Peak GPU memory and mean s/epoch (the deeper arm will be slower)
- **Final-epoch comparison vs H48 GEGLU at the same epoch index** — if Arm A only completed 12 epochs vs H48's 13, compare both at epoch 12

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** if val_avg at epoch 3 exceeds 250, kill and report. Both arms should track H48 GEGLU's trajectory closely; large early deviations indicate a depth-induced training pathology.

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd, n_layers=5)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + **n_layers=5**.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted (Arm A deeper) ≈ 57-58 if GEGLU's per-layer capacity benefits from extra stages.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
