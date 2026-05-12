# SENPAI Research State

- **Date**: 2026-05-12 20:05 (round 2 underway; higher-lr-cosine-14 is new baseline)
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`

---

## Current Baseline

**val_avg/mae_surf_p = 96.5587** — PR #1518 (higher-lr-cosine-14), merged 2026-05-12.

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params), `AdamW(lr=1e-3, wd=1e-4)`, **`CosineAnnealingLR(T_max=14)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, MSE loss. NaN scoring fix in `train.py:evaluate_split`. ~14 epochs / 30 min.

Test avg 85.87 (all 4 splits, NaN-free for first time).

---

## Current Research Focus

**The key insight from round 2**: Schedule alignment matters enormously. By setting T_max=14 to match the real epoch budget, the cosine schedule actually reaches its low-LR fine-tuning phase. Combined with lr=1e-3 (safe under grad_clip=1.0), this gave -17.6% over the previous best. The old T_max=50 was wasting 64% of the schedule.

**Val still falling at epoch 14** — the model has not converged within the 30-min cap. This is the key constraint on further progress. Options:
1. More epochs per 30 min (bf16-amp — alphonse)
2. Faster convergence (higher lr — thorfinn; Huber loss — tanjiro; warmup — frieren)
3. Architecture that converges faster (wider-deeper — edward)

**Key open questions**:
1. Does Huber loss compound on top of the new LR/schedule baseline? (Tanjiro's rebase will answer this — now needs to beat 96.56 instead of 117.17)
2. Does lr=1.5e-3 push past 96.56 or find the LR ceiling? (Thorfinn's new PR #1539)
3. Does bf16-amp enable 18-20 epochs in 30 min, and what would that yield with the new LR?
4. Can architecture scaling (wider-deeper) converge faster to a better solution?

**Per-split profile** at new baseline:
- Easiest: `val_geom_camber_cruise` (74.35 val, 61.86 test) — big improvement, cruise regime
- Hardest: `val_geom_camber_rc` (110.59) and `val_single_in_dist` (108.58) — raceCar regime still challenging

**NaN fix**: y-sanitization is now in `train.py:evaluate_split` (merged with PR #1518). All future PRs get it automatically via rebase.

---

## Active Experiments

| PR | Student | Slug | Status | Notes |
|----|---------|------|--------|-------|
| #1456 | alphonse | `bf16-amp` | WIP (round 1) | Pre-new-baseline; needs to beat 96.56 |
| #1457 | askeladd | `surf-weight-50` | WIP (v2) | surf_weight=30 + old-grad-clip baseline; may need round 3 |
| #1458 | edward | `wider-deeper` | WIP (v2, dirty) | batch_size=4 + grad-clip; sent back for rebase |
| #1460 | fern | `relative-l2-loss` | WIP (round 1) | Pre-new-baseline; needs to beat 96.56 |
| #1462 | frieren | `warmup-cosine` | WIP (v2, dirty) | 1-epoch warmup; sent back for rebase |
| #1467 | nezuko | `more-slices-128` | WIP (round 1) | Pre-new-baseline; needs to beat 96.56 |
| #1473 | tanjiro | `huber-loss` | WIP (v2, pending rebase) | Rebase onto new LR baseline, target ≤96.56 |
| #1539 | thorfinn | `lr-1.5e-3-cosine-14` | WIP (new) | lr=1.5e-3, T_max=14; LR ceiling test |

---

## Potential Next Research Directions

Round 2 refinements + round 3 candidates:

- **Per-channel Huber delta**: `delta_p < delta_Ux ≈ delta_Uy` — pressure has widest residuals. Try `{p: 0.3, Ux: 0.7, Uy: 0.7}`.
- **EMA of model weights**: Shadow param set for val only (~5 lines, no hyperparams). Smooths the noisy 14-epoch trajectory, often +1-3%.
- **CosineAnnealingWarmRestarts(T_0=7)**: Two complete cosine cycles in 14 epochs; each restart re-energizes the optimizer. Orthogonal to LR scaling.
- **SiLU activation**: Replace GELU in Transolver MLP. Cheap, orthogonal, often helps in physics-informed settings.
- **Lion optimizer**: Lower memory, sign-based updates. LR should be ~3-10× lower than AdamW equivalent. Try lr=1e-4 with Lion.
- **surf_weight sweep**: Test {15, 20, 25} on the new LR baseline to see if the surface/volume tradeoff optimum shifts.
- **Compound: Huber + bf16-amp**: If both work independently, compound should dominate. Best round-3 assignment.
- **Wider-deeper + new LR**: If edward's wider-deeper run completes, merge and then test lr=1e-3 T_max=14 on top.
- **Fourier/positional encoding**: Improve geometric OOD encoding for raceCar regime generalization.

**Plateau protocol**: If 5 consecutive experiments fail to beat 96.5587, escalate to architecture overhaul (FNO spectral layer, GNOT multi-query attention, graph neural operators).
