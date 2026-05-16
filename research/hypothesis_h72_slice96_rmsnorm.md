## Hypothesis

**H72: Compound the two latest baseline-moving wins — slice_num=96 (H66) + RMSNorm (H59) — at the n_layers=4 GEGLU base.**

The two most recent merged improvements are mechanistically orthogonal:

- **H59 RMSNorm (fused F.rms_norm kernel):** Per-epoch wall-clock speedup (~3s/ep saved) yields one extra training step within the 30-min wall budget. The gain comes from compute efficiency, not from the normalization's directional-preservation property per se.
- **H66 slice_num=96:** Architectural change in `PhysicsAttention.in_project_slice`. Widens the slice-token bottleneck from 64 → 96 (+50%), giving the model finer spatial selectivity. The gain concentrates on geometry-OOD splits (test_geom_camber_rc: −3.33 pts), exactly where local mesh structure matters.

These two changes do not interact — one is a kernel-efficiency / extra-epoch lever, the other is a representational-capacity lever. We expect their gains to compound near-additively.

**Implementation note:** When H66 was branched, H59 had not yet merged, so H66's measured numbers were with `norm_type=layernorm`. After the H66 merge, the advisor branch contains **both** the RMSNorm code (from H59) and slice_num as a CLI flag (from H66). H72 simply enables both flags at runtime: `--norm_type rmsnorm --slice_num 96`.

**Two arms:**

- **Arm A: slice_num=96 + RMSNorm** — direct compound of the two latest merged wins. Expected: ~56.3-56.7 val_avg (small additive gain on top of H66's 56.7504); test 3-split ~53.5-54.5.
- **Arm B: slice_num=112 + RMSNorm** — explores the 96→128 interpolation zone. H66 Arm B (128) regressed; the optimum may sit at 96-112. With RMSNorm's per-epoch speedup, 112 gets back the extra epoch that 128 lost, potentially closing the gap. Expected: ~56-57 val_avg.

**Predicted:** Arm A should be the safer winner (compound of two confirmed wins). Arm B is the exploratory probe — if slice_num optimum is monotone up to ~112 then Arm B could win bigger, but if 96 is already optimal then Arm B will trail Arm A.

**Risk:** Negligible. Both flags are merged in the codebase. Mean s/epoch should be ~117-120s (RMSNorm speedup partially offset by slice widening).

## Instructions

The required CLI flags are already merged into `train.py` (no code changes needed). Just pass them at runtime.

Run both arms:

```bash
# Arm A — slice_num=96 + RMSNorm (compound of H66 + H59 wins)
cd target/ && python train.py --epochs 50 \
  --experiment_name h72-slice96-rmsnorm-geglu-n4 \
  --agent charliepai2i48h3-thorfinn \
  --slice_num 96 --norm_type rmsnorm \
  --n_layers 4 --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — slice_num=112 + RMSNorm (exploratory)
cd target/ && python train.py --epochs 50 \
  --experiment_name h72-slice112-rmsnorm-geglu-n4 \
  --agent charliepai2i48h3-thorfinn \
  --slice_num 112 --norm_type rmsnorm \
  --n_layers 4 --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags use current merged defaults: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, T_max=15.

**Report:**
- val_avg/mae_surf_p and per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test for both arms
- Best epoch and per-epoch val_avg trajectory for both arms
- **Compound additivity check:** Δ(Arm A vs H66 baseline) should ≈ Δ(H59 vs LayerNorm baseline) = −0.67 if perfectly additive. Report the actual Δ and discuss whether the wins compound, sub-add, or super-add.
- Mean s/epoch and peak GPU memory per arm
- **Per-split OOD gain breakdown** — does Arm A retain H66's test_geom_camber_rc gain (−3.33) when RMSNorm is also active? If not, that signals interaction.
- Epochs completed before wall

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4011 — H66 Arm A: slice_num=96 at GEGLU n_layers=4 (thorfinn, norm_type=layernorm)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **56.7504** |
| val_single_in_dist/mae_surf_p | 60.9717 |
| val_geom_camber_rc/mae_surf_p | 70.7939 |
| val_geom_camber_cruise/mae_surf_p | 38.2785 |
| val_re_rand/mae_surf_p | 56.9576 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **54.5026** |
| test_single_in_dist/mae_surf_p | 54.5425 |
| test_geom_camber_rc/mae_surf_p | 61.8680 |
| test_re_rand/mae_surf_p | 47.0974 |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + n_layers=4 + slice_num=96 + **norm_type=layernorm**. n_params=864,907. Mean s/epoch=121.8.

**Beat this: val_avg/mae_surf_p < 56.7504**

H59 reference (norm_type=rmsnorm at slice_num=64): val=56.9056, test=56.2420. **The H59 norm change improved val by −0.67 over its baseline. If that gain compounds with H66's −0.16, Arm A should land ~56.1 val_avg.**

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
