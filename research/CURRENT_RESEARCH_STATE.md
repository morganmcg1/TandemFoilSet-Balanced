# SENPAI Research State

- 2026-05-15 22:35 — round 10 of `icml-appendix-charlie-pai2i-48h-r2`
- No active research directives from the human research team

## Baseline progression

| PR | val_avg/mae_surf_p | Change | Config delta |
|----|-------------------|--------|-------------|
| #3119 (thorfinn, epochs=80) | 135.0153 | initial | AdamW lr=5e-4, surf_weight=10 |
| #3101 (askeladd, surf_weight=30) | 127.4122 | −5.6% | surf_weight: 10→30 |
| #3293 (nezuko, Lion) | 117.5014 | −7.8% | AdamW→Lion, lr: 5e-4→1.7e-4, wd: 1e-4→3e-4 |
| #3357 (tanjiro, asinh-loss) | 84.9819 | −27.7% | asinh(z) on pressure channel z-scores in training loss |
| **#3382 (askeladd, EMA+asinh)** | **83.1874** | **−2.1%** | **EMA shadow decay=0.999 at val/test passes** |

**Current HEAD:** Lion + surf_weight=30 + NaN-safe eval + asinh pressure-loss + EMA(0.999). val=83.19 at epoch 14 (timeout-bound, shadow still converging).

## Active experiments

| PR | Student | Theme | Status | Baseline context |
|----|---------|-------|--------|-----------------|
| #3442 | tanjiro | signed log1p on pressure (stronger tail compression than asinh) | WIP | On EMA+asinh baseline 83.19 |
| #3470 | askeladd | EMA decay ablation: 0.997/0.995/0.990 vs 0.999 | WIP | On EMA+asinh baseline 83.19 |
| #3485 | alphonse | bf16 autocast: faster forward → more epochs per 30-min | WIP — NEW | On EMA+asinh baseline 83.19 |
| #3354 | nezuko | Lion + cap-matched cosine (T_max=12) | NEEDS REBASE | Sent back to rebase onto EMA+asinh HEAD |
| #3384 | fern | Gradient clipping (max_norm=1.0) — RERUN with EMA+asinh | WIP | Mechanism verified; rerun on full stack |
| #3383 | edward | Lion + 2-epoch linear warmup then cosine | WIP | On EMA+asinh baseline 83.19 |
| #3275 | thorfinn | SwiGLU gated activation in TransolverBlock MLPs | WIP | On EMA+asinh baseline 83.19 |
| #3106 | frieren | Slice128/head8/mlp3 + Lion lr=3.4e-4 rerun | WIP | On EMA+asinh baseline 83.19 |

## Closed / falsified experiments this round

| PR | Reason |
|----|--------|
| #3115 (tanjiro Re-FiLM) | +6.3% regression; FiLM helps in-dist but hurts OOD |
| #3328 (askeladd surf_weight=50) | +25% regression; instability above sw=30 |
| #3329 (fern AdamW β2=0.95) | +21% regression; wrong smoothing direction for B=4 |
| #3102 (edward OneCycleLR) | +20% regression; schedule shape wrong for 14-epoch budget |
| #3411 (tanjiro asinh-all-channels) | +5.8% regression; velocity z-scores are light-tailed |
| #3099 (alphonse capacity 192h/6L/6H) | +60.5% regression; wall-clock budget dominates — 8 epochs vs baseline's 14 |

## Key research understanding

**Compounding wins so far:**
1. surf_weight 10→30: −5.6% (align training loss with eval metric)
2. Lion optimizer: −7.8% (sign-based update, better for this batch size)
3. asinh pressure-loss: −27.7% (heavy-tail gradient compression, dominant mechanism)
4. EMA(0.999): −2.1% on top of asinh (parameter trajectory smoothing)

**Wall-clock budget is a hard constraint:** The 30-min cap limits baseline to ~14 epochs. Anything that increases per-epoch cost (capacity, higher batch) directly reduces epoch count. Throughput improvements (bf16) could unlock the inverse — more epochs per budget.

**Mechanism map:**
- asinh: operates at the loss function level (gradient scale compression)
- EMA: operates at the parameter level (trajectory smoothing)
- grad-clip (pending fern rerun): operates at the gradient level (norm cap)
- bf16 (alphonse new): operates at the throughput level (more steps per budget)
- signed log1p (tanjiro): alternative at the loss function level (more aggressive compression)
- EMA decay sweep (askeladd): EMA horizon tuning (shadow convergence within budget)

## Pending verification

**#3384 fern grad-clip (max_norm=1.0):** Mechanism verified (100% clip rate, mean ~137). Rerun on full EMA+asinh+Lion stack pending. Expected: orthogonal to EMA (different layers of intervention), should compound. If it does, the triple stack (asinh+EMA+clip) could push into the 79-81 range.

## Key open questions (round 10)

1. **Does signed log1p beat asinh?** (#3442 tanjiro) — more aggressive compression; now needs to beat 83.19
2. **What's the optimal EMA decay for ~5k steps?** (#3470 askeladd) — decay=0.999 shadow still converging at timeout; 0.995 may give full benefit within budget
3. **Can bf16 unlock meaningful extra epochs?** (#3485 alphonse) — if 131s→90s, +4-6 epochs could push into 75-79 range
4. **Does cosine T_max=12 improve schedule alignment?** (#3354 nezuko) — after rebase onto full stack
5. **Does grad-clip stack with EMA+asinh?** (#3384 fern) — three orthogonal mechanisms could triple-compound
6. **Do SwiGLU and Slice128 beat baseline after inheriting the full stack?** (#3275 thorfinn, #3106 frieren) — architectural ideas now running on EMA+asinh baseline

## Potential next research directions

- **Batch size 4→8 with bf16** (if alphonse's bf16 runs clean): bf16 saves ~10GB memory, opening headroom for 2× batch size. Lion+asinh+EMA may benefit from larger batches.
- **Grad-clip max_norm tuning** (if fern's rerun wins): once confirmed, sweep 0.5 / 1.0 / 2.0 to find optimal clipping aggressiveness.
- **EMA decay=0.995 composition with bf16**: if both win individually, stack them — more steps (bf16) + better-converged shadow (0.995) should compound.
- **surf_weight ablation on EMA+asinh+bf16 stack**: surf_weight=30 set at AdamW+MSE era; asinh de-emphasizes tails, EMA smooths — may need weight reduction (20-25) to avoid double-downweighting.
- **Channel-decoupled output heads**: separate MLP for Ux/Uy vs p — pressure-specific inductive bias.
- **WeightedRandomSampler**: inverse-error reweighting after epoch 1 — mechanism orthogonal to all current wins.
