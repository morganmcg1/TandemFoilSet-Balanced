## Hypothesis

**H98: Lion β₁ retune at β₂=0.997 baseline — test if β₁ optimum shifts with the new β₂.**

H90 (askeladd, PR #4189, WIP) is sweeping Lion β₁ ∈ {0.85, 0.95} at the OLD β₂=0.995 baseline. With β₂ now locked at 0.997 (H88 merged), there's a real question whether the optimal β₁ has shifted.

Lion's update direction is: `u_t = sign(β₁ · m_{t-1} + (1−β₁) · g_t)`.

- β₁ controls how much current gradient enters the sign computation
- β₂ controls the EMA half-life of the momentum buffer itself

With β₂=0.997 (~231-step half-life — longer memory than 0.995's ~138 steps), the momentum m_{t-1} is already smoother. The optimal β₁ may shift:
- **Higher β₁** (less current gradient influence) compounds with longer β₂ EMA — may over-smooth and crash like β₂=0.999 did
- **Lower β₁** (more current gradient influence) compensates for the smoother momentum — could keep update reactive while preserving the β₂=0.997 noise filtering

Two arms (same range as H90 to enable direct comparison):
- **Arm A: β₁=0.85, β₂=0.997** — more reactive; compensates for smoother β₂ EMA
- **Arm B: β₁=0.95, β₂=0.997** — more smoothing; potential over-smoothing risk

**Predicted:**
- Arm A: ~40.5-42.5 val_avg (lower β₁ likely better-matches β₂=0.997's longer memory)
- Arm B: ~41.0-43.0 val_avg (higher β₁ may over-smooth, similar mechanism to β₂=0.999 crash)
- Null: ~41.2 (β₁ optimum is robust)

**Information value (independent of win):**
- Cross-reference with H90's β₁ results at β₂=0.995
- If H90 wins at β₁=0.85 AND H98 wins at β₁=0.85 → β₁=0.85 robustly better, lock it
- If H90 wins but H98 doesn't (or vice versa) → β₁ × β₂ interaction worth probing further

## Baseline

H88 Arm B val=41.2153 / test=39.5337 (PR #4166, MERGED). ~122 s/epoch, ~15 epochs/30-min budget.

Config: optimizer=lion + lr=3e-4 + **β=(0.9, 0.997)** + wd=1e-3 + slice_num=96 + n_layers=4 + n_head=2 + ffn_act=geglu + clip_grad_norm=1.0 + T_max=15.
