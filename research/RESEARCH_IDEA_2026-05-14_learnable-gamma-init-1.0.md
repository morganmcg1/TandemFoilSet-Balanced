# Round 138 — Learnable LayerScale γ init=1.0 at post-norm topology

## Hypothesis

Restore learnable LayerScale γ params but initialize at 1.0 instead of 1e-4 — pairing the NEW post-norm topology (from #2964 WIN) with adaptive per-channel residual scaling. Tests whether adaptivity on top of γ=1.0 provides additional gains beyond fixed γ=1.0 constant.

Per student suggestion #2 verbatim from #2964: "Learnable γ init=1.0 at post-norm: turn gamma_attn/mlp back into nn.Parameter(torch.ones(hidden_dim)) so it can adapt per channel during training. +768 params, same init as this PR."

## Motivation

PR #2964 showed two critical facts:
1. Post-norm + γ=1.0 WINS (val 30.0382, -1.71% vs baseline 30.5605; -4.93% test)
2. Pre-norm + γ=1e-4 (old baseline) was a DIFFERENT PAIR — the mechanism is PAIR-SPECIFIC

Key question: Was the benefit of #2964 due to:
(a) The γ=1.0 VALUE (correct scale for post-norm), OR
(b) The FIXED γ (no adaptivity overhead), OR
(c) BOTH equally?

If (a): learnable γ init=1.0 should WIN further — model can adapt away from 1.0 where helpful
If (b): learnable γ should LOSE — adaptivity adds noise, fixed is better
If (c): learnable may be wash or mild improvement

The prior baseline used learnable γ init=1e-4 with pre-norm — this is the FIRST test of learnable γ at post-norm with init=1.0.

## Architecture

```python
# OLD (baseline #2879, pre-norm):
self.gamma_attn = nn.Parameter(torch.full((n_hidden,), 1e-4))
self.gamma_mlp = nn.Parameter(torch.full((n_hidden,), 1e-4))
x = x + self.gamma_attn * self.attn(self.ln_1(x), fx=fx, T=T)
x = x + self.gamma_mlp * self.mlp(self.ln_2(x))

# Current (post-norm, #2964 WIN):
self.gamma_attn = 1.0  # constant scalar
self.gamma_mlp = 1.0
x = self.ln_1(x + self.attn(x, fx=fx, T=T))  # post-norm
x = self.ln_2(x + self.mlp(x))

# THIS PR (post-norm + learnable γ init=1.0):
self.gamma_attn = nn.Parameter(torch.ones(n_hidden))  # +96 params per branch
self.gamma_mlp = nn.Parameter(torch.ones(n_hidden))
x = self.ln_1(x + self.gamma_attn * self.attn(x, fx=fx, T=T))
x = self.ln_2(x + self.gamma_mlp * self.mlp(x))
```

+768 total params (8 blocks × 2 branches × 96 dims), same as the old pre-norm baseline (407,172 → 407,940, back to old param count).

## Falsifiable predictions

- **WIN** (val < 30.0382): Learnable γ at post-norm with correct init helps further. Adaptivity adds value beyond fixed γ=1.0.
- **PARTIAL** (val ≈ 30.0382 ± 0.3%): Comparable — init is the KEY, not adaptivity.
- **LOSS** (val > 30.3): Fixed γ=1.0 outperforms learnable γ at post-norm. Learnable adds optimization noise.

## Key diagnostics to report

1. **γ convergence trajectory** at ep60 — where does learnable γ land from 1.0 init?
   - If γ drifts DOWN toward ~0.85-0.95 (similar to pre-norm baseline convergence): adaptivity is real
   - If γ stays near 1.0 (frozen by training): init is optimal, adaptivity doesn't engage
2. **Per-block γ asymmetry** (γ_attn vs γ_mlp): does post-norm topology alter the MLP>>attn preference seen in pre-norm baseline?
3. **Param count: 407,940** (back to pre-#2964 count)

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/learnable-gamma-init-1.0" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```
