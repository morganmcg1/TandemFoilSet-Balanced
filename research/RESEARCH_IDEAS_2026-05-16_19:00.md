# Next-Round Hypothesis Bank — 2026-05-16 19:00

## Context

- **Current baseline (PR #3994, merged):** val=62.10, test=59.55 — T_max=17 SwiGLU h=128 seed=0
- **Imminent new baseline (PR #3995 fern, pending SENPAI-RESULT):** val=60.43, test=57.44 — triple-stack = T_max=17 + AdamW β2=0.95 + GeGLU FFN
- **SWA mechanism validated** (PR #3644 closed) — being re-tested in #4089 nezuko (in-cosine SWA, no kick-out)
- **Expected idle students after this round closes:** alphonse, askeladd, edward, fern, frieren, tanjiro, thorfinn (7 students)

## Confirmed dead-end levers (DO NOT re-suggest)

- Dropout / DropPath — regression
- weight_decay ≥1e-2 — null/regression (cumulative shrinkage math: only ~3% pull-back)
- LR=1e-3 under T_max=15 — divergence
- head+embed LR boost (1.5×, 2×, 2.5×) — all null/worse
- T_max < 17 — confirmed suboptimal (PyTorch Gotcha #3)
- RMSNorm replacement of LayerNorm — 5pt regression (#3973)
- slice_num=128 (2× attention) — big regression (#3998)
- gradient clipping clip_norm=1.0 — regression (#3999)
- Warmup before cosine — worsens early dynamics (#3993)

## Top 12 hypotheses (ranked by expected impact, after fern triple-stack merges as baseline)

| Rank | Slug | Expected Δval | Rationale |
|---|---|---|---|
| 1 | triple-stack-seed1 | seed-variance ±0.9 | Critical 3-seed canonical for new best |
| 2 | triple-stack-seed2 | seed-variance ±0.9 | Same — confirm μ̂ of triple-stack |
| 3 | triple-stack-swa-tail4 | −0.5 to −2.0 | Stack validated SWA on validated triple-stack (after #4089) |
| 4 | b1-095-tripleStack | −0.3 to −1.0 | β2=0.95 worked; β1=0.95 may compound (heavier momentum smoothing) |
| 5 | mlp-ratio-3-tripleStack | −0.2 to −0.8 | More FFN capacity at modest param cost |
| 6 | lion-optimizer-geglu-tmax17 | −0.5 to +1.5 | Lion (Chen 2023) often beats AdamW on smaller batches; risky |
| 7 | sgdr-warm-restarts-tripleStack | 0 to −1.0 | T_0=6 T_mult=2 schedule; cyclic basin exploration |
| 8 | ema-weights-tripleStack | −0.3 to −1.0 | EMA of model weights at decay=0.999 — pseudo-SWA without snapshots |
| 9 | beta2-fine-scan-tripleStack | ±0.5 | β2 ∈ {0.93, 0.97, 0.98}; check if 0.95 is optimum |
| 10 | sam-rho005-tripleStack | −0.5 to −2.0 | Sharpness-Aware Minimization (Foret 2020); 2× compute, big when works |
| 11 | layerwise-lr-decay-085 | ±0.5 | LLRD across 5 transformer layers; risky for small networks |
| 12 | label-smoothing-huber | −0.2 to −0.7 | σ-temperature smoothing of normalized targets; risky |

## Concrete prescriptions for top-5

### 1. triple-stack-seed=1 (~3-seed canonical)

```bash
cd target/ && python train.py --agent <student> \
  --wandb_name "<student>/triple_stack_tmax17_b095_geglu_seed1" \
  --wandb_group triple_stack_seed_scan \
  --use_geglu --beta2 0.95 --seed 1
```

Identical to fern's winning recipe but seed=1. Expected: val ∈ [59.5, 61.3] given typical σ̂≈0.9.

### 2. triple-stack-seed=2

Same as above with `--seed 2`. The two seed-confirmation PRs establish whether fern's seed=0 was a lucky draw or representative.

### 3. triple-stack-swa-tail4 (compound SWA + triple-stack)

After #4089 nezuko confirms in-cosine SWA works on SwiGLU baseline, port the SWA wrapper onto the triple-stack recipe:

```bash
cd target/ && python train.py --agent <student> \
  --wandb_name "<student>/triple_stack_swa_tail4_seed0" \
  --wandb_group triple_stack_swa \
  --use_geglu --beta2 0.95 --seed 0 \
  # plus SWA wrapper averaging epochs 14-17 (same as #4089)
```

Hypothesis: if both wins are orthogonal, val=60.43 + SWA Δ ≈ −0.5 to −1.5 → val ∈ [58.9, 59.9].

### 4. β1=0.95 on triple-stack

```bash
cd target/ && python train.py --agent <student> \
  --wandb_name "<student>/triple_stack_b1_095_seed0" \
  --wandb_group triple_stack_beta1_scan \
  --use_geglu --beta1 0.95 --beta2 0.95 --seed 0
```

Default β1=0.9; β1=0.95 = 2× momentum half-life. Combined with β2=0.95, this is "heavier momentum smoothing both for gradient and squared gradient" — should help stability near minimum. May also help OOD splits (smoother updates).

### 5. mlp_ratio=3 with triple-stack

```bash
cd target/ && python train.py --agent <student> \
  --wandb_name "<student>/triple_stack_mlp3_seed0" \
  --wandb_group triple_stack_mlp_scan \
  --use_geglu --beta2 0.95 --mlp_ratio 3 --seed 0
```

Currently mlp_ratio=2 (FFN hidden=256). mlp_ratio=3 gives hidden=384. GeGLU inner dim becomes `round(384*2/3)=256`. Param count: ~750K (12.5% increase). May help capacity in OOD splits.

## Speculative / ambitious ideas (if conservative plateaus)

- **Physics-informed loss:** Add soft continuity penalty `λ * |∇·u|^2` on predicted (Ux, Uy) at internal mesh nodes. λ=1e-3 initial. This is a CFD-native regularizer — should help OOD where pressure/velocity are most decoupled.
- **TTA (test-time augmentation):** If the mesh supports horizontal flip symmetry, predict on flipped + unflipped meshes and average. Free improvement at inference.
- **FNO (Fourier Neural Operator):** Pure FNO baseline — completely different model class. May give competitive numbers and inform whether Transolver attention is actually needed.
- **Cross-slice attention:** Transolver uses within-slice attention. Add a between-slice attention head (or alternating layer) — captures global mesh structure.
- **SAM (sharpness-aware):** ρ=0.05, 2× forward/backward. Has reproducibly given 0.5–2pt gains in vision; CFD surrogate analog is untested.

## Assignment plan (when 7 students go idle)

| Student | Hypothesis | Slug |
|---|---|---|
| thorfinn | triple-stack-seed=1 | triple-stack-seed1 |
| alphonse | triple-stack-seed=2 | triple-stack-seed2 |
| askeladd | β1=0.95 on triple-stack | b1-095-tripleStack |
| fern | mlp_ratio=3 on triple-stack | mlp-ratio-3-tripleStack |
| frieren | β2 fine scan {0.93, 0.97} on triple-stack | beta2-fine-scan-tripleStack |
| tanjiro | EMA weights decay=0.999 on triple-stack | ema-weights-tripleStack |
| edward | Lion optimizer with cosine T_max=17 + GeGLU | lion-optimizer-geglu-tmax17 |

(nezuko already running #4089 SWA-tail4-cosine-tmax17 on SwiGLU baseline — if that confirms, the next round can have a `triple-stack-swa-tail4` student.)

## Notes

- All triple-stack-derived hypotheses depend on PR #3995 (fern) merging cleanly with SENPAI-RESULT. If fern's second training round (currently running) gives a different val than the first run's W&B val=60.43, re-evaluate priorities.
- The seed=1/seed=2 confirmations are highest priority because the new baseline is single-seed; we cannot safely build on it until σ̂ is characterized.
- If any of #4028 thorfinn or #4050 alphonse already give val<62.10, evaluate against new baseline once fern merges; otherwise close.
