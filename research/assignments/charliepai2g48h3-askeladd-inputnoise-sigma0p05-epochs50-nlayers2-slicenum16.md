# Assignment: askeladd — Input-embedding Gaussian noise σ=0.05 on epochs=50 stack

**Branch (use exactly):** `charliepai2g48h3-askeladd/inputnoise-sigma0p05-epochs50-nlayers2-slicenum16`

**Base branch:** `icml-appendix-charlie-pai2g-48h-r3`

## Hypothesis

Four same-config runs at the current baseline (n_layers=2 + slice_num=16 + epochs=50 + surf_weight=10 + Lion + L1 + cosine) gave val_avg = [34.544 (#2872), 35.414, 35.697 (#2888 r1/r2), 34.887 (#2907)] — sample std 0.531. The bimodality (2 runs at 34.5-34.9, 2 runs at 35.4-35.7) suggests **two convergence basins selected by init seed**. This is the same underlying mechanism driving the seed-variance signal: init-path determines basin geometry.

Two parallel regularization probes are now in flight:
- **frieren #2917**: dropout p=0.1 on attention + FFN — activation-side stochastic regularization
- **askeladd (this PR)**: input-embedding noise σ=0.05 — data-side stochastic regularization

**Hypothesis:** Adding Gaussian noise (σ=0.05) to the input embeddings DURING TRAINING (not eval) acts as a smoothness/robustness regularizer at the entry of the network. Mechanism is distinct from dropout: dropout removes activations inside the transformer, input-noise perturbs the representation pre-transformer. Both should flatten the loss basin around the optimum, but they affect different layers' gradients differently.

Specific expectations:
1. **Reduces seed variance**: forces the model to be robust to small embedding-space perturbations, which makes the optimization landscape less sensitive to init differences
2. **Improves OOD splits**: input-noise during training is analogous to test-time data perturbation; the model learns to handle small input variations, plausibly transferring to geometric OOD splits (geom_camber_rc=48 is the dominant val_avg contributor)
3. **Compounds with dropout (if both work)**: data-side + activation-side regularization stack naturally — they regularize different parts of the network

## Why input-embedding (not raw input features)?

The Transolver model embeds raw input node features (coordinates, surface flags, etc.) into a `n_hidden=128` representation before the transformer blocks. Adding noise to RAW features could break input semantics (e.g., coordinate noise = geometry perturbation). Adding noise to the EMBEDDED features is:
- Semantically safe (no constraint on what each dim "means")
- Standard transformer technique (used in language models for input embedding noise / SwitchOut)
- Per-dimension uniform — no need to think about which raw features can tolerate noise

## Implementation

Find where the embedded features enter the transformer in the Transolver model class. This is typically:
- After the input projection layer (e.g., `self.embed = nn.Linear(in_dim, n_hidden)`)
- Before the first transformer block

**Add a Config flag:**
```python
input_noise_sigma: float = 0.0   # Gaussian noise std added to input embeddings during training (0.0 = disabled)
```

**Apply noise** in the model's `forward` method, between embedding and first transformer block:
```python
# Standard input projection
embed = self.input_proj(features)   # (B, N, n_hidden)

# Add noise during training only
if self.training and self.cfg.input_noise_sigma > 0:
    embed = embed + torch.randn_like(embed) * self.cfg.input_noise_sigma

# Continue into transformer blocks
out = self.transformer(embed, mask)
```

`self.training` is automatically False during `.eval()` (val/test) — no noise at inference.

**That's it.** No other code changes. The mechanism is intentionally minimal: one line of noise injection at the embedding entry.

**At training end, please report:**
1. Best single-epoch val + test (full per-split table)
2. Per-epoch val_avg trajectory for last 8 epochs (so we can compare convergence shape to the 4 prior same-config runs)
3. Total wall-clock + per-epoch timing

## Choosing σ=0.05

The Transolver model uses RMSNorm normalization, so embedding magnitudes are O(1) after normalization. σ=0.05 corresponds to ~5% perturbation of the embedding magnitude — a meaningful but not destructive perturbation. Standard transformer noise injection values are σ ∈ [0.01, 0.1]; σ=0.05 is the centroid.

If this experiment is neutral, suggested follow-ups are σ ∈ {0.02, 0.10, 0.15} sweeps.

## Run command

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name inputnoise-sigma0p05-epochs50-nlayers2-slicenum16 \
  --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 --slice_num 16 \
  --input_noise_sigma 0.05
```

## Baseline to beat

PR #2872 (n_layers=2 + slice_num=16 + epochs=50, **no input noise, no dropout**) — current best (single seed):

| Metric | Value |
|---|---:|
| **val_avg/mae_surf_p** | **34.544** |
| val_single_in_dist | 35.113 |
| val_geom_camber_rc | 48.106 |
| val_geom_camber_cruise | 18.895 |
| val_re_rand | 36.060 |
| **test_avg/mae_surf_p** | **29.916** |

**Updated seed-variance context (n=4):** val_avg = [34.544, 35.414, 35.697, 34.887], mean ≈ 35.135, sample std ≈ 0.531. The baseline (34.544) is ~1.1σ above the mean — a moderately favorable seed.

**Decision thresholds (against true mean ~35.1):**
- **Clear win:** val < 34.0 — input noise significantly reduces error or seed variance
- **Ambiguous:** val ∈ [34.5, 35.6] — within seed noise, indeterminate
- **Loss:** val > 36.0 — input noise hurts

## Per-run constraints

- Hard timeout: 30 min (`SENPAI_TIMEOUT_MINUTES=30`). Noise injection overhead is negligible (~0.05% per step). Expected ~29.3 min, same as baseline #2872.
- Hard epoch cap: `SENPAI_MAX_EPOCHS` (do not override).
- **Local JSONL metrics only.** Do NOT log to W&B.
- Branch only from `icml-appendix-charlie-pai2g-48h-r3`.

## Terminal result format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/.../metrics.jsonl"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best_val_avg>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test_at_best_val>}}
```

Include:
1. Full per-split val + test table at best_epoch
2. Per-epoch val_avg trajectory for last 8 epochs
3. The actual best_epoch reached + wall-clock

## Decision criteria

- **Win (val < 34.0):** Input noise works. Sweep σ to find optimum, then consider compound with frieren's dropout if both win.
- **Ambiguous within seed noise:** input noise neither clearly helps nor hurts at σ=0.05. Try σ=0.10 (stronger) before closing.
- **Loss (val > 36.0):** Input noise hurts. Likely σ too high; try σ=0.02 or close axis.

## Suggested follow-ups

- **If wins:** sweep σ ∈ {0.02, 0.1, 0.15}; try input-noise + dropout compound.
- **If neutral:** try DropPath (residual-path stochastic dropout) — another mechanism-distinct regularizer.
- **If loses:** input perturbation isn't tolerated at this stack; rules out one canonical regularization family.

## EV assessment

**Medium.** Input-embedding noise is a standard transformer technique with mixed-but-positive empirical track record (SwitchOut, embedding dropout, R-Drop family). At seed variance ~0.5 val-std, even a 1-2% improvement in basin width could show through. Worst case: neutral, axis partially closed. Best case: 1-3% val improvement, especially on the dominant OOD split (geom_camber_rc), with possible reduction in seed variance.

This is mechanism-distinct from frieren's dropout (#2917) — both can be run in parallel without redundancy. If frieren's dropout wins and this is neutral, dropout is the effective lever. If this wins and dropout is neutral, input noise is the effective lever. If both win independently, compound is the next experiment.
