# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-28 (after PR #404 merged)

## Current best — Round 0, PR #404 (edward H11 FiLM-on-Re, Run E)

| Metric | Value | Δ vs prior baseline (PR #344) |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` | **119.36** | −1.3% |
| `test_avg/mae_surf_p` | **107.54** | −2.2% |
| `test/test_single_in_dist/mae_surf_p` | 120.69 | −5.0% |
| `test/test_geom_camber_rc/mae_surf_p` | 120.45 | −2.5% |
| `test/test_geom_camber_cruise/mae_surf_p` | 80.70 | −0.6% |
| `test/test_re_rand/mae_surf_p` | 108.32 | +0.5% (within noise) |
| Params | 0.75M | +12.6% (FiLM head adds ~83K) |

- **W&B run:** [`p0a1daar`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r4/runs/p0a1daar) (`willowpai2d4-edward/h11-E-on-wd5e-4-seed123`)
- **Best epoch:** 13 of 14 actually trained (run hit 30-min wall clock)

## What changed from prior baseline (PR #344)

The merged code on `icml-appendix-willow-pai2d-r4` now includes:

1. **Re-conditional FiLM modulation between Transolver blocks.** A small MLP produces `(γ, β)` per layer from the per-sample `log(Re)` (input dim 13). Each `TransolverBlock` applies `γ ⊙ ln_1(fx) + β` after the first LayerNorm. Identity-init (γ=1, β=0 with zero-weight last layer) ensures `--film_re False` reproduces the prior architecture exactly.
2. **`--seed` CLI flag** for RNG-state determinism (model init + GPU ops). Note: dataloader workers and sampler are NOT seeded; this gives partial reproducibility (sufficient for variance checks like Run E vs Run C).
3. **Cumulative from PR #344:** linear warmup + per-step cosine-to-zero schedule, defensive `nan_to_num` in `evaluate_split`.

## Recommended training command (reproduces current best)

```bash
cd target/ && python train.py \
    --agent <student-name> \
    --film_re True \
    --epochs 25 \
    --lr 7e-4 \
    --weight_decay 5e-4 \
    --seed 123 \
    --wandb_name "<student-name>/<experiment-tag>"
```

**Important note on the FiLM × wd interaction.** Run D (FiLM off + wd=5e-4) regressed to val=123.95 vs the prior baseline (120.97), and Run B (FiLM on + wd=1e-4) only landed at val=126.63. Neither lever helps alone — together they unlock a stable higher-regularization regime. Future hypothesis comparisons should use **wd=5e-4 + FiLM on** as the merged-baseline configuration, not just one of the two.

## Setup recap

| Setting | Value |
|---------|-------|
| Model | Transolver + Re-conditional FiLM (~0.75M params total: 0.66M Transolver + ~83K FiLM head) |
| Optimizer | AdamW, weight_decay=**5e-4** (from PR #404; previously 1e-4) |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10 |
| Schedule | Linear warmup (5%) + cosine-to-zero, per-step (`LambdaLR`) |
| Epochs (default) | 50, capped by `SENPAI_TIMEOUT_MINUTES=30` (~13–14 actually fit) |
| Recommended `--lr` | 7e-4 (from PR #344; default still 5e-4) |
| Recommended `--seed` | 123 (for variance-check reproducibility; default None) |
| Primary metric | `val_avg/mae_surf_p` (lower is better) |
| Paper metric | `test_avg/mae_surf_p` |

## Validation/test splits

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains

## Round-0 history

- **PR #344 (edward H2):** linear warmup + per-step cosine + NaN fix → val=120.97, test=109.92.
- **PR #404 (edward H11):** Re-conditional FiLM + wd=5e-4 → val=119.36, test=107.54 (Run E, seed=123).
