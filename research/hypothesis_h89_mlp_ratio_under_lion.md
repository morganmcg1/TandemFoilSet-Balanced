## Hypothesis

**H89: mlp_ratio sweep under Lion+slice=96 — never tested in this regime.**

H62 closed mlp_ratio negative under AdamW+slice=64. Under Lion's scale-invariant sign-update and the wider slice=96 gradient surface, expanded FFN capacity may unlock representational headroom the optimization regime change couldn't access before.

mlp_ratio scales the inner FFN dimension: `FFN: x → x·W_up (n_hidden → mlp_ratio·n_hidden) → activation → ·W_down`. Doubling FFN width adds parameters where most of the model's computation already happens (FFN is typically 2/3 of transformer FLOPs).

Two arms:
- **Arm A: mlp_ratio=4** — doubles FFN inner dim (4× n_hidden vs current 2×).
- **Arm B: mlp_ratio=3** — moderate increase (3× n_hidden).

**Predicted:**
- Arm A: ~40-44 val_avg (1-2 pt improvement if FFN width unlocks headroom).
- Arm B: ~41-44 val_avg (smaller change; risk of being noise-level).
- **Need Δ ≥ 3 pts to clearly exceed noise floor.**

**Complementary to H86** (n_hidden expansion, tanjiro): H86 widens everything (attention + FFN + projections); H89 widens FFN only. If both win, the FFN-specific contribution can be isolated.

**Risk:** Wider FFN may slow per-epoch training enough that the 15-ep budget undershoots, masking a true improvement. Watch s/epoch carefully.

## Baseline

H78 Arm B val=42.3048 / test=40.5564 (PR #4097, MERGED).

⚠ Noise floor ~2.6 pts (from H74). Aim for ≥3 pt improvement.
