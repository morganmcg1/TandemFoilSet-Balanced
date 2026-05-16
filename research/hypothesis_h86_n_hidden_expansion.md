## Hypothesis

**H86: Expand n_hidden under Lion+slice=96 to unlock capacity scaling.**

All recent sweeps (lr, wd, warmup, n_head, n_layers, ffn_act) hit a frontier at val≈43. Optimization levers are largely tapped. The model is wall-cut at ep 15 — capacity is the next frontier.

Lion is famously good at scaling. With slice_num=96 already widening the gradient surface, expanding n_hidden may unlock further super-additive gain.

Two arms:
- **Arm A: n_hidden=192** — 50% capacity increase.
- **Arm B: n_hidden=256** — 100% capacity increase.

**Predicted:** Arm A: ~38-42 val_avg; Arm B: ~36-44 val_avg. Need ≥3 pt improvement to exceed seed noise.

**Risks:** memory pressure, wall-cut shift (slower per epoch), LR-mismatch.

## Baseline
H73 Arm B val=42.9784 / test=41.5455 (PR #4055).

⚠ Noise floor ~2.6 pts (from H74). Aim for ≥3 pt improvement.
