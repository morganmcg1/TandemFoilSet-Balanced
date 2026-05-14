# Round 138 — Per-block hardcoded T schedule [1.0, 0.5, 0.5, 0.25]

## Hypothesis

Replace baseline's 8 learnable per-head temperatures (4 blocks × 2 heads, init 0.5) with hardcoded per-block schedule `T_blocks = [1.0, 0.5, 0.5, 0.25]` — matching the implicit learned per-depth schedule inferred from #2955's entropy diagnostic. Tests whether the per-depth SCHEDULE is the load-bearing piece, or whether per-head ADAPTIVE LEARNING is itself load-bearing.

## Motivation (#2955 close)

#2955 tanjiro T=0.25 hardcoded LOSS (val +3.40%). Combined with #2944 (T=2.0 hardcoded LOSS, val +4.43%), the temperature axis is U-shaped, both endpoints LOSS.

**MECHANISTIC INSIGHT:** Baseline learnable per-head temperatures encode a per-depth schedule:
- Block 0: entropy 1.32–1.71 nats (soft routing, T~1.0)
- Block 3: entropy 0.01–0.92 nats (sharp routing, T~0.25)
- **8 learnable params save >4%** — 0.002% of param budget for 4% gain

Student suggestion #2 verbatim: *"Per-block hardcoded T schedule: hand-pick T = [1.0, 0.5, 0.5, 0.25] (matching the implicit learned schedule). Tests whether learnability can be replaced by a static schedule. If WINs → confirms it's the per-block adaptation, not the optimizer learning fine-grained adjustments. If LOSSes → the per-head granularity matters too."*

## Architecture

```python
# In PhysicsAttention.__init__, replace:
# self.temperature = nn.Parameter(torch.full((n_head,), 0.5))

# With block-aware hardcoded T:
# Pass block_idx through Block construction
T_PER_BLOCK = [1.0, 0.5, 0.5, 0.25]
self.temperature = T_PER_BLOCK[block_idx]  # scalar constant, not Parameter

# In forward:
slice_weights = F.softmax(routing_logits / self.temperature, dim=-1)
```

Param count: -8 (removed 8 learnable temperatures); same as #2944 / #2955.

## Falsifiable predictions

- **WIN** (val < 30.5605): Static per-depth schedule suffices — learnability adds nothing beyond the right schedule. Suggests per-depth schedules can be hand-tuned.
- **PARTIAL** (val ≈ 30.5605 ± 1%): Schedule does most of the work; per-head adaptation is fine-tuning. Modest gap from baseline.
- **LOSS** (val > 31.0): Per-head ADAPTIVE LEARNING is load-bearing, not just the schedule. Closes static-schedule axis. Suggests follow-up with learnable per-head AND per-block schedule init.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-tanjiro \
    --experiment_name "charliepai2g48h5-tanjiro/per-block-temperature-schedule" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Reporting

1. val_avg/test_avg vs baseline + per-split breakdown
2. **Routing entropy at ep57 per block** — verify the schedule produces the predicted entropies
3. **Compare to #2944 (T=2.0) and #2955 (T=0.25) entropy values** — confirm the schedule recovers baseline's per-depth pattern
4. Param count (407,932 = baseline-8, same as #2944/#2955)
5. Meta-signal check
6. Plain-language verdict: WIN / PARTIAL / LOSS
