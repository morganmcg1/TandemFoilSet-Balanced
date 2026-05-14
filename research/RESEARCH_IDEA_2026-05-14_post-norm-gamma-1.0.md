# Round 138 — Post-norm topology + LayerScale γ=1.0 (no LayerScale)

## Hypothesis

Re-test post-norm topology, but with LayerScale γ_attn = γ_mlp = 1.0 instead of baseline γ=1e-4. Tests whether LayerScale-was-tuned-for-pre-norm was the confound in #2951's clear LOSS, isolating topology from LayerScale init scheme.

## Motivation (#2951 close)

#2951 nezuko post-norm-topology was a decisive LOSS (val +14.73%, test +13.18%), with cruise worst-hit at +25.03% val / +30.74% test — and crucially META-SIGNAL BROKE (cruise no longer wins under post-norm).

**Mechanism diagnosed:** Pre-norm + γ=1e-4 deliberately keeps residual stream "wide" early — `x = x + γ·attn(LN(x))` with tiny γ means attn contributes a tiny perturbation to a large residual, allowing the model to slowly grow useful signal on top of a near-identity stream. Post-norm with γ=1e-4 cancels this dynamic: `x = LN(x + γ·attn(x))` renormalizes after every block, washing out the LayerScale-controlled slow build-up.

**Student suggestion #2 verbatim:** *"Post-norm with γ=1e-4 might be unfair to post-norm: LayerScale was tuned for pre-norm. A post-norm + γ=1.0 test would isolate topology from LayerScale init scheme. This is the clean post-norm test under the original Transformer recipe."*

This is the CLEAN post-norm test under the original Transformer recipe (Vaswani 2017 post-norm + standard residual without LayerScale). It separates two conflated mechanisms:
- TOPOLOGY: where centering happens in residual stream
- LAYERSCALE: how residual contributions are gated

## Architecture

Two changes from baseline (build on #2951 post-norm code):

1. **Topology**: pre-norm → post-norm at all 9 LN sites (identical to #2951)
2. **LayerScale removal**: replace `self.gamma_attn = nn.Parameter(torch.full(..., 1e-4))` and `self.gamma_mlp` with constant scalar 1.0 (or set init to 1.0 with `requires_grad=False`)

```python
# Block.__init__:
# OLD: self.gamma_attn = nn.Parameter(torch.full((n_hidden,), 1e-4))
# OLD: self.gamma_mlp = nn.Parameter(torch.full((n_hidden,), 1e-4))
self.gamma_attn = 1.0  # constant, no LayerScale
self.gamma_mlp = 1.0   # constant, no LayerScale

# Block.forward (POST-NORM, from #2951):
# x = self.ln_1(x + self.gamma_attn * self.attn(x, fx, T))
# x = self.ln_2(x + self.gamma_mlp * self.mlp(x))
```

Param count: 407,940 - 9*96*2 (two LayerScale γ vectors per block × 4 blocks + final?) ≈ 407,200 (~ -740 params). Exact count depends on where LayerScale is applied.

## Falsifiable predictions

- **WIN** (val < 30.5605): LayerScale init scheme was the confound. Clean post-norm + γ=1.0 works. Suggests revisiting other γ-vs-topology combinations.
- **PARTIAL** (val ≈ 30.5605 ± 1%): Post-norm is recoverable with the right γ init. Closes topology-+-γ jointly at γ=1.0.
- **LOSS** (val > 31.0): Post-norm itself is the load-bearing problem regardless of γ. Decisively closes post-norm topology axis. Pre-norm is structurally load-bearing for this Transolver recipe.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/post-norm-gamma-1.0" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline AND Δ vs #2951 (post-norm + γ=1e-4)
3. Param count (~407,200)
4. **Residual magnitude diagnostic at ep1 and ep60:** mean ||x|| before/after each LN site
5. **Meta-signal check:** does cruise WIN return under post-norm + γ=1.0, or stay broken like #2951?
6. **TOPOLOGY-vs-LAYERSCALE attribution:** combining #2951 (post-norm + γ=1e-4 catastrophic LOSS) and this PR:
   - If γ=1.0 recovers most of #2951's loss → LayerScale init scheme was the confound
   - If γ=1.0 still LOSS → topology itself is load-bearing
7. Train→val gap at convergence
8. **Plain-language verdict:** WIN / PARTIAL / LOSS
