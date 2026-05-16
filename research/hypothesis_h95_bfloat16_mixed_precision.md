## Hypothesis

**H95: Enable bf16 mixed-precision training — efficiency unlock to break the wall-cut bottleneck.**

H86 (n_hidden expansion) and H89 (mlp_ratio sweep) both closed negative because wider models eat epochs from the 30-min budget. The baseline is still descending steeply at ep 15 — wall-cut, not capacity, is the binding constraint.

bf16 mixed precision is a standard recipe: model forward/backward in bf16, optimizer states in fp32. On H100, this typically yields 25-40% s/epoch reduction with no model quality impact (bf16 has the same exponent range as fp32, just fewer mantissa bits).

If s/epoch drops from ~120 to ~80, the 30-min budget yields ~22 epochs instead of ~15. That's a 47% step increase — large enough that the cosine tail finally completes, and capacity probes (H86, H89) become retestable.

Two arms (sanity check + same-epoch comparison):
- **Arm A: bf16 enabled** — full recipe, see how many epochs fit in 30-min budget
- **Arm B: bf16 + ep15 stop** — match baseline at 15 epochs to isolate precision effect from epoch-count effect

**Predicted:**
- Arm A: val ~40-42 (if bf16 unlocks more epochs)
- Arm B: val ~42-43 (similar to baseline at matched horizon; minor precision drift)

**Risk:** Lion sign-update should be robust to bf16 (sign(g) is precision-insensitive). LN/GEGLU are typically fine in bf16. Attention may need careful gradient scaling — Pytorch's `torch.autocast(dtype=bfloat16)` handles this.

## Baseline

H78 Arm B val=42.3048 / test=40.5564 (PR #4097, MERGED). ~120 s/epoch, ~15 epochs/30-min budget.
