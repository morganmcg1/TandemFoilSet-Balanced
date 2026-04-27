# Willow-R5 Baseline — TandemFoilSet

- **Track:** `willow-r5` (advisor branch `icml-appendix-willow-r5`)
- **Dataset:** TandemFoilSet (CFD on tandem airfoils)
- **Model:** Transolver (physics-aware attention)
- **Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four validation splits (lower is better)
- **Paper-facing metric:** `test_avg/mae_surf_p` — same aggregation on the four held-out test splits

## Current best (round 1 in flight)

**No measured willow-r5 baseline yet.** Round 1 establishes solo baselines for each hypothesis family. The reference configuration is the bare Transolver default committed in `target/train.py`.

### Reference configuration (bare Transolver)

```
n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
loss=MSE (vol + 10*surf), surf_weight=10.0
optimizer=AdamW, lr=5e-4, weight_decay=1e-4, batch_size=4
schedule=CosineAnnealingLR(T_max=epochs)
SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30
balanced domain sampler (WeightedRandomSampler)
```

**Reproduce command:**
```bash
cd target && python train.py --agent <student> --wandb_name "<student>/baseline-ref"
```

## Round 1 hypothesis slate (8 PRs in flight)

| Hypothesis | Family | Predicted Δ on `val_avg/mae_surf_p` |
|------------|--------|--------------------------------------|
| H1 huber-loss-delta-sweep | loss | −10 to −20% |
| H3 fourier-pe-film-re | data repr/conditioning | −15 to −25% |
| H4 slice-num-down-sweep | capacity/arch | −5 to −10% |
| H5 n-layers-down-sweep | capacity/arch | −3 to −8% |
| H6 swiglu-feedforward | architecture | −3 to −7% |
| H7 amp-bf16-throughput | optimizer/systems | −2 to −6% (more epochs) |
| H8 surf-weight-sweep | loss | −3 to −8% |
| H9 domain-aware-conditioning-tokens | conditioning (NEW) | −5 to −12% |

## External reference (parallel kagent_v_students track)

A separate, parallel research track on the same dataset converged to a stacked recipe (Huber/L1 + Fourier-PE + FiLM + SwiGLU + nl=3 + sn=16 + AMP + grad_accum) reaching **val_avg/mae_surf_p ≈ 54** (2-seed mean). Willow-r5 starts fresh from the bare baseline and rebuilds the recipe with single-variable purity for clean ICML-appendix attribution. We expect round 1's bare baseline to land considerably above 54 — likely in the 80–150 range — and the round-1 winners to begin closing the gap.
