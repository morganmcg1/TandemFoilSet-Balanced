# SENPAI Research State

- 2026-05-15 21:35 — round 9 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| #3293 (nezuko, Lion) | 117.5014 | −7.8% | AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4 |
| #3357 (tanjiro, asinh-loss) | 84.9819 | −27.7% | asinh(z) on pressure channel z-scores in training loss |
| **#3382 (askeladd, EMA+asinh)** | **83.1874** | **−2.1%** | **EMA shadow decay=0.999 at val/test passes** |

**Current HEAD:** Lion + surf_weight=30 + NaN-safe eval + asinh pressure-loss + EMA(0.999). val=83.19 at epoch 14 (timeout-bound, curve still descending, EMA shadow still converging).

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3442 | tanjiro | signed log1p on pressure (stronger tail compression than asinh) | WIP | On asinh baseline 84.98; new target 83.19 |
| #3354 | nezuko | Lion + cap-matched cosine (T_max=12) | WIP | Notified of 84.98 target; new target 83.19 |
| #3384 | fern | Gradient clipping (max_norm=1.0) — RERUN with asinh | SENT BACK | First arm: 87.98 (on pre-asinh code) |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP | Notified of 84.98 target; new target 83.19 |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP | Notified of 84.98 target; new target 83.19 |
| #3106 | frieren | Slice128/head8/mlp3 + Lion lr=3.4e-4 rerun | WIP | Notified of 84.98 target; new target 83.19 |
| #3099 | alphonse | Capacity 192h/6L/6H + Lion lr=3.4e-4 rerun | WIP | Notified of 84.98 target; new target 83.19 |

## Students awaiting new assignments

| Student | Status |
|---------|--------|
| askeladd | IDLE — PR #3382 just merged; needs new experiment |

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression; FiLM helps in-dist but hurts OOD |
| #3328 (askeladd surf_weight=50) | +25% regression; instability above sw=30 |
| #3329 (fern AdamW β2=0.95) | +21% regression; wrong smoothing direction for B=4 |
| #3102 (edward OneCycleLR) | +20% regression; schedule shape wrong for 14-epoch budget |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores are light-tailed, asinh compresses meaningful velocity gradient signal |

## Critical insight: stacking wins

**Two compounding wins so far:**
1. asinh pressure-loss: −27.7% (the dominant mechanism — heavy-tail gradient compression)
2. EMA weights: −2.1% on top of asinh (parameter trajectory smoothing on an already smoother landscape)

The val curve continues to descend at epoch 14 (timeout-bound) for both mechanisms. More headroom exists with longer training budgets.

**Key observation:** EMA gain was reduced from −10% (pre-asinh) to −2.1% (post-asinh), confirming the advisor's prediction that asinh already reduces the gradient variance EMA corrects. Both mechanisms are orthogonal and compose cleanly.

## Pending verification (round 7 sent-backs)

**#3384 fern grad-clip (max_norm=1.0):** val=87.98 on pre-asinh code. 100% clip rate confirmed heavy-tail mechanism. Must rerun on new HEAD (Lion + asinh + EMA). If EMA already smooths Lion's update variance, grad-clip may add less; but the mechanisms differ (per-step norm cap vs parameter trajectory averaging) — likely to compose.

## Key open questions (round 9)

1. **Does signed log1p beat asinh?** (tanjiro #3442) — more aggressive tail compression; predicted ±2% from 84.98. Now also needs to beat new 83.19 EMA baseline.
2. **Does grad-clip compose with asinh+EMA?** (fern #3384 rerun) — mechanism orthogonal to EMA but may overlap with asinh's gradient-scale reduction.
3. **Do architecture changes (SwiGLU, capacity, Slice128) beat 83.19?** — architectural capacity may matter less now that loss landscape is smoother.
4. **Does schedule shape (cosine T_max, warmup) matter on the asinh+EMA landscape?** — with faster convergence, schedule may need re-calibration.

## Potential next research directions

- **EMA decay ablation:** 0.995 vs 0.999 — at ~5k steps, 0.995 EMA horizon (~200 steps) may fully converge within budget while 0.999 (~1000 steps) is still catching up at timeout. Faster decay could give same benefit earlier.
- **EMA warmup-start:** Skip EMA tracking for first 2-3 epochs (weights changing fastest) so shadow doesn't drag in early-LR noise.
- **Signed log1p tau sweep:** asinh(z/tau)*tau with tau>1 weakens compression, tau<1 strengthens — sweep for optimal pressure-channel compression knee.
- **Grad-clip (max_norm=1.0) on asinh+EMA HEAD:** Third mechanism in the stack — if orthogonal, could push below 80.
- **surf_weight ablation on asinh+EMA stack:** surf_weight=30 was tuned on vanilla MSE. With asinh, heavy-tail pressure samples are already de-emphasized — may want lower weight (e.g. 20) to avoid double-downweighting.
- **WeightedRandomSampler:** Inverse-error reweighting after epoch 1 — different mechanism from all above, could stack.
- **Channel-decoupled output heads:** Separate mlp2 for Ux/Uy vs p — stronger inductive bias for pressure-specific modeling.
- **asinh tau parameter:** Learnable or annealed per-channel compression knee.
