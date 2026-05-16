## Hypothesis

**H64: Re-tune per-channel Huber δ on the new GEGLU baseline — H25's optimum may have shifted.**

H25 (askeladd, merged earlier) found per-channel Huber δ_p=0.25 and δ_vel=0.5 optimal at the pre-GEGLU base (val=83.81). Since then, the model has compounded multiple wins to val=58.63 (H48 GEGLU). The pressure prediction error distribution has changed dramatically:
- At val_avg=83.81, the error tail was heavy and Huber's δ_p=0.25 was an aggressive clamp.
- At val_avg=58.63, the model is much more accurate; the *current* error distribution near the wall is different.

**Mechanism:** Huber's δ controls where the loss transitions from L2 (quadratic, gradient ∝ residual) to L1 (linear, gradient ∝ sign). For a well-trained model, most pressure residuals are small. The δ controls how "aggressive" the optimizer is on the remaining hard examples:
- Lower δ (e.g., 0.1) → almost everything is in L1 regime → optimizer focuses on directional improvement, gives equal weight to all residuals regardless of magnitude
- Higher δ (e.g., 0.5) → more of the loss is in L2 regime → optimizer focuses on the biggest residuals, neglects small ones

With GEGLU's improved spatial selectivity, the model may benefit from a *different* tradeoff. The hardest pressure points (boundary-layer cliffs near trailing edges) may already be well-handled by the gate; the remaining error is likely more uniformly distributed and may benefit from lower δ.

**Two arms (δ_p sweep, δ_vel fixed at 0.5):**
- **Arm A: δ_p=0.1** — more aggressive (more L1) — for uniformly-distributed remaining error
- **Arm B: δ_p=0.5** — less aggressive (more L2) — for heavy-tailed remaining error

**Predicted:** If GEGLU has substantially flattened the error tail, Arm A wins (~57.5-58.5). If the tail is still heavy, Arm B wins.

**Risk:** Loss reshape can interact with cosine schedule. If gradients become small at epoch 13+ (per H48 data), δ effectively controls only the late-training phase. Both arms may show similar val curves until ~epoch 10.

## Instructions

The `--huber_delta_p` and `--huber_delta_vel` flags already exist (merged in H25). No code changes needed.

Run both arms:

```bash
# Arm A — δ_p=0.1 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h64-geglu-deltap01 \
  --agent charliepai2i48h3-nezuko \
  --ffn_act geglu \
  --huber_delta_p 0.1 --huber_delta_vel 0.5 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — δ_p=0.5 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h64-geglu-deltap05 \
  --agent charliepai2i48h3-nezuko \
  --ffn_act geglu \
  --huber_delta_p 0.5 --huber_delta_vel 0.5 \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, surf_weight=10, n_hidden=128, slice_num=64, T_max=15, ffn_act=geglu (current merged defaults).

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Best epoch and epochs completed
- Per-epoch val_avg trajectory — overlay against H48 GEGLU (δ_p=0.25) trajectory
- **L1 fraction analysis:** for one batch at epoch 13, log the fraction of pressure residuals that exceed δ (i.e., in the L1 regime). If this fraction is <10%, almost all residuals are in L2 regime and δ barely matters; if >50%, δ is actively shaping the loss.
- Peak GPU memory (should be unchanged from H48 baseline)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report. (Unlikely with Huber.)

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd, δ_p=0.25, δ_vel=0.5)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber **δ_p=0.25, δ_vel=0.5** + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu.

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted: Arm A (δ_p=0.1) ≈ 57.5-58.5 if GEGLU has flattened the error tail. Arm B (δ_p=0.5) ≈ 58-59 if tail is still heavy.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
