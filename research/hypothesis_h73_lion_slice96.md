## Hypothesis

**H73: Lion + GEGLU + slice_num=96 — compound the two strongest confirmed levers.**

Two independent wins from this round:
- **H58 Lion + GEGLU**: val_avg=46.7957 (loose UB, wall-cut at ep13/15) — Δ −10.11 vs baseline. Sign-based gradient update fixes a systemic optimization issue (uniform −10 to −14 pts across all 4 val splits).
- **H66 slice_num=96**: val=56.7504, test 3-split=54.5026 (current baseline). Architectural change widening the slice-token bottleneck. Gain concentrates on geometry-OOD (test_geom_camber_rc −3.33 pts).

These two levers are **mechanistically orthogonal**: Lion changes *how* gradients update weights; slice_num=96 changes *what* the attention block computes. There is no theoretical reason they should interact destructively, and substantial reason to expect them to compound additively.

**Predicted floor**: If Lion's −10 pt gain on H48 baseline (58.63 → 46.80) transfers to the slice_num=96 base (56.75 → ~46.7), Arm A should land around val_avg = **45-47**. Given Lion's runs are loose upper bounds (wall-cut mid-cosine), the true asymptote may be **lower than 45**.

**Strategic value:** If H73 lands as predicted, this becomes the new floor for all subsequent Lion-compound experiments. H67-H71 (the in-flight Lion variants) test Lion + RMSNorm, Lion β₂, Lion warmup, Lion n_head, Lion wd — but none test Lion + slice_num. H73 is the missing piece of the Lion compound matrix.

**Two arms:**

- **Arm A: Lion lr=1e-4 (H58 winner) + slice_num=96 + GEGLU + n_layers=4** — direct compound of the two confirmed wins at their proven settings.
- **Arm B: Lion lr=3e-4 (Lion's native range) + slice_num=96 + GEGLU + n_layers=4** — tests whether the slightly higher LR (still in Lion's regime) gives better epoch utilization within the wall budget. H58 Arm B (lr=2e-4) won on test but lost on val by 0.65 pts; lr=3e-4 sits in between, may be the sweet spot for the wider-bottleneck model.

**Predicted:**
- Arm A: ~45-47 val_avg / ~44-46 test 3-split (compound of confirmed wins at known-good Lion settings)
- Arm B: ~45-48 val_avg (slightly higher LR may favor wider slice's larger gradient surface, or may slightly over-shoot)

**Risk:** Lion+slice_num could interact non-additively if Lion's sign-update changes how the slice-token bottleneck is utilized. But Lion's uniform per-split gain in H58 suggests the optimizer doesn't differentially help any one architectural component, so the prior on additivity is strong.

## Instructions

**No code changes required.** Both `--optimizer lion` and `--slice_num 96` are available as CLI flags in the merged codebase. Lion-specific flags (`--beta1`, `--beta2`, `--weight_decay`) match the H58 winner.

```bash
# Arm A — Lion lr=1e-4 (H58 winner) + slice_num=96
cd target/ && python train.py --epochs 50 \
  --experiment_name h73-lion-lr1e4-slice96-geglu \
  --agent charliepai2i48h3-tanjiro \
  --optimizer lion --lr 1e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0

# Arm B — Lion lr=3e-4 (Lion native range) + slice_num=96
cd target/ && python train.py --epochs 50 \
  --experiment_name h73-lion-lr3e4-slice96-geglu \
  --agent charliepai2i48h3-tanjiro \
  --optimizer lion --lr 3e-4 --weight_decay 1e-3 \
  --beta1 0.9 --beta2 0.99 \
  --slice_num 96 --n_layers 4 --ffn_act geglu \
  --n_head 2 --clip_grad_norm 1.0
```

All other flags use current merged defaults: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, T_max=15, norm_type=layernorm (matching H66's measured config).

**Report:**
- val_avg/mae_surf_p and per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test for both arms
- Best epoch, epochs completed, mean s/epoch, peak GPU memory
- **Compound additivity check:**
  - H58 added −10.11 val_avg on top of H48 baseline (58.63 → 46.80 at slice_num=64+LayerNorm).
  - H66 added −0.83 val_avg on top of H59 baseline (57.58 → 56.75 at slice_num=64→96+LayerNorm).
  - If perfectly additive: H73 Arm A ≈ 56.75 − 10.11 = **46.64 val_avg**, or H73 ≈ 46.80 − 0.16 = **46.64** from the other direction.
  - Report whether the actual gain is additive, sub-additive, or super-additive.
- **Per-split OOD analysis:** Does Arm A retain H66's test_geom_camber_rc gain (−3.33 pts) when Lion is also active? If so, the slice_num spatial-selectivity mechanism survives Lion's optimization regime.
- Per-epoch val_avg trajectory (Lion runs are wall-cut; trajectory shape matters)
- GEGLU gate health (mean/std at epochs 7, 13) — Lion+slice_num=96 may shift gate utilization vs H58's slice_num=64

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 250 → kill and report.

## Baseline

**Current best — PR #4011 — H66 Arm A: slice_num=96 at GEGLU n_layers=4 (thorfinn, AdamW + LayerNorm)**

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

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + n_layers=4 + slice_num=96 + norm_type=layernorm + **optimizer=adamw**. n_params=864,907.

**Beat this: val_avg/mae_surf_p < 56.7504**

**Stretch goal: < 47** (if Lion + slice_num=96 compound additively, this should be reachable).

H58 reference (Lion + slice_num=64): val=46.7957 / test=46.6320. H73 explores whether adding slice_num=96 to the Lion stack pushes further down.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
