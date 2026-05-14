# Round 138 — Routing weight dropout p=0.1 (stochastic routing regularization)

## Hypothesis

Apply `nn.Dropout(p=0.1)` to slice_weights AFTER softmax in PhysicsAttention, re-normalized to maintain sum-to-1. Tests stochastic regularization on routing — different mechanism from temperature softening (#2944 LOSS) or entropy regularization (#2884 LOSS).

## Motivation

- #2944 decisively showed sharp routing at block-3 is LOAD-BEARING (softening LOSS)
- #2884 showed entropy regularization (soft induction) LOSS
- These attacked routing via SOFTNESS. Dropout attacks via STOCHASTICITY: preserve sharp routing on average, but force robustness by randomly dropping the dominant slice.
- Standard Transformer recipe (attention dropout in BERT/GPT). Lion + dropout is well-behaved.

## Architecture

```python
# In __init__
self.routing_dropout = nn.Dropout(p=0.1)

# In forward (AFTER softmax)
slice_weights = F.softmax(routing_logits / temperature, dim=-1)
slice_weights = self.routing_dropout(slice_weights)
slice_weights = slice_weights / (slice_weights.sum(dim=-1, keepdim=True) + 1e-8)  # re-normalize
```

Zero new params. Eval-time deterministic (nn.Dropout disabled).

## Why this might WIN

- Robustifies against single-slice over-reliance without breaking specialization
- Preserves sharp eval-time routing (which #2944 confirmed is desirable)
- Lion sign-step + dropout: well-paired in literature
- In_dist over-specialization (recurring meta-signal) is exactly what dropout regularizes

## Why this might LOSS

- Dropout may interrupt block-3's specialization gradient
- p=0.1 may be too aggressive at slice_num=24
- Re-normalization could create gradient pathologies

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-edward \
    --experiment_name "charliepai2g48h5-edward/routing-weight-dropout-0.1" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Falsifiable predictions

- WIN: val < 30.5605, eval-time routing retains sharp pattern
- PARTIAL: routing dropout neutral
- LOSS: routing dropout breaks specialization, val > 31.0
