## Hypothesis

**H61: GEGLU has its own optimal LR ≤ 1e-3 — sweep down to find it.**

H57 (askeladd, closed) demonstrated that GEGLU + lr=2e-3 *regresses* by +0.88 vs the H48 GEGLU baseline at lr=1e-3. This is striking because lr=2e-3 was a +2.67 win on the *pre-GEGLU* stack (H39 Arm C → H37b). The two architectures have different LR optima.

**Question:** is lr=1e-3 the GEGLU optimum, or is it actually below 1e-3?

**Mechanism:** GEGLU's gated update direction `∇σ(xW_gate) * ∇(xW_value)` is more concentrated per step than vanilla FFN's `∇(xW₁)·∇(xW₂)`. The gate sigmoid restricts which gradient components contribute, so each step takes a more "directed" update. At the same LR, GEGLU effectively moves further along its preferred direction than vanilla FFN. Therefore:
- Vanilla FFN optimum LR: ~2e-3 (H39 Arm C win)
- GEGLU optimum LR: probably **lower** — maybe 7e-4 or 5e-4 — to match the per-step displacement magnitude

If true, GEGLU + lr=5e-4 or 7e-4 could give us a small additional win (~0.5-1.5 pts) just by re-tuning the LR for the new architecture.

**Two arms:**
- **Arm A: lr=7e-4** — modest reduction. If GEGLU's LR optimum is just below 1e-3, this should be near-optimal.
- **Arm B: lr=5e-4** — larger reduction. Tests whether GEGLU needs substantially less LR than vanilla FFN.

Both at the H48 GEGLU config (n_head=2, wd=5e-5, clip=1.0, ffn_act=geglu).

**Risk:** Lower LR + 15-epoch budget may undertrain. If the per-epoch trajectory at epoch 8 is more than 5 pts behind H48's epoch 8 trajectory, the budget is the bottleneck not the LR. Report epoch-by-epoch comparison.

## Instructions

No code changes needed — all flags already exist.

Run both arms:

```bash
# Arm A — lr=7e-4 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h61-geglu-lr7e4-nhead2-wd5e5 \
  --agent charliepai2i48h3-alphonse \
  --ffn_act geglu \
  --n_head 2 --lr 7e-4 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — lr=5e-4 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h61-geglu-lr5e4-nhead2-wd5e5 \
  --agent charliepai2i48h3-alphonse \
  --ffn_act geglu \
  --n_head 2 --lr 5e-4 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Best epoch and epochs completed before wall
- Per-epoch val_avg trajectory — overlay against H48 GEGLU (lr=1e-3) trajectory
- **Final-epoch comparison vs H48 at same epoch** — if your arm at epoch 13 is more than 5 pts behind H48 at epoch 13, undertraining is the issue
- Pre-clip gradient norms at epochs 1, 7, 13 — confirm clip behavior is unchanged at lower LR
- Peak GPU memory and mean s/epoch

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report. (Unlikely at lower LR.)

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu.

**Reference — H57 (lr=2e-3, closed):** val=59.50, **+0.88 vs baseline**. Higher LR hurts GEGLU.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted: Arm A (lr=7e-4) ≈ 57.5-58.5 if GEGLU's LR optimum is just below 1e-3. Arm B (lr=5e-4) ≈ 58-60 (likely worse than 1e-3 unless GEGLU needs much smaller LR).

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
