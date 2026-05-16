## Hypothesis

**H97: LR fine-tune at β₂=0.997 baseline — test whether optimal lr shifts with β₂.**

H75 characterized the LR U-shape at β₂=0.99 (H73 base), establishing lr=3e-4 as the minimum. H88 shifted β₂ from 0.99 → 0.995 → 0.997, but lr=3e-4 has not been re-evaluated since β₂ was tuned.

β₂ and lr couple in Lion: β₂=0.997 gives a longer EMA half-life (~231 steps), meaning the momentum term is smoother and less reactive to the current gradient. A longer EMA potentially tolerates a slightly larger lr before destabilizing (since the update direction is less noisy), or conversely favors a slightly lower lr at the cosine tail (since the smoother momentum may overshoot fine-grained loss valleys).

Two arms probing ±20% around lr=3e-4:
- **Arm A: lr=2.5e-4** — 17% lower; tests if the tail LR (cosine endpoint) is overshooting at the new β₂
- **Arm B: lr=3.5e-4** — 17% higher; tests if the smoother EMA tolerates a higher peak LR

**Predicted:**
- Arm A: ~41.0-42.5 (if optimal lr shifts left with longer EMA)
- Arm B: ~40.8-42.5 (if smoother momentum tolerates higher LR stably)
- Null: ~41.2 (if lr=3e-4 is already robust to ±20% change)

**Mechanistic rationale:** With β₂=0.997, Lion's momentum evolves more slowly. The gradient's 10% contribution (with β₁=0.9) is dampened further by the longer EMA window. At cosine tail (LR≈0), the momentum magnitude determines final convergence quality. A slightly higher peak LR with a 231-step half-life may better exploit the smooth cosine decay without the instability that β₂=0.99 would show at the same LR.

## Baseline

H88 Arm B val=41.2153 / test=39.5337 (PR #4166, MERGED). ~122 s/epoch, ~15 epochs/30-min budget.

Config: optimizer=lion + lr=**3e-4** + β=(0.9, **0.997**) + wd=1e-3 + slice_num=96 + n_layers=4 + n_head=2 + ffn_act=geglu + clip_grad_norm=1.0 + T_max=15.
