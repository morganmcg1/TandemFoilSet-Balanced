# SENPAI Research State

- 2026-05-15 21:40 — **Round-8 review complete**. Sent back #3370 (tanjiro GLU,
  geglu winner with OOD-concentrated signal on stale base) and #3348 (fern
  Fourier, L=8 winner with −18.3 absolute on val_single_in_dist on stale base),
  both for Charbonnier-base rebase + 2-arm re-run (dropping confirmed-null SwiGLU
  and fourier_rich variants). Fixed routing label on #3457. All 8 students now
  have active WIP assignments.
- No directives from the human researcher team on this launch.

## Current research focus

Round 8+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`. **Three items
merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143), NaN evaluate_split
bug fix (#3138). Primary validation target: **val_avg/mae_surf_p < 98** (best
single-seed 98.60 from PR #3143). Paper-facing test metric: **test_avg/mae_surf_p
= 92.71** (compose sanity u2k87wan; PR #3138). 8 hypotheses in flight, **5 of
them in active rebase mode** waiting for Charbonnier-base composition results.

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`, `warmup_epochs=3`, `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143) — **⚠️ default still "mse" pending PR #3440**
- NaN filter in `evaluate_split` (PR #3138)

## Key constraints / insights confirmed

- **Charbonnier robust loss is the dominant single lever.** −18.6% on val_avg/mae_surf_p
  vs MSE. Linearizes gradient for large residuals; surface pressure has order-of-magnitude
  dynamic range. Improves every channel, every split.
- **EMA is a likely major winner on rebase.** PR #3151 showed −17.8% test within-PR
  on stale base. Sent back.
- **AMP (bf16) is a proven lever, rebasing.** PR #3330 showed 1.34× epoch speedup
  → −13% within-PR.
- **GeGLU shows OOD-favorable signal.** PR #3370 (−2.9% within-PR overall, but
  −13.6% on val_geom_camber_cruise, −13.0% on test_re_rand). SwiGLU is null.
  GELU > SiLU as gate activation for Transolver. Sent back for rebase.
- **Fourier position encoding L=8 is a real architectural lever.** PR #3348
  (−18.3 absolute on val_single_in_dist surf_p). L>8 hurts because standardized
  position σ≈1 makes bands above 2^7 sub-mesh-spacing noise. Sent back for rebase.
- **Separate output heads is dead** (#3331 closed). Shared trunk is NS cross-field
  inductive bias.
- **Surf_weight tune is weak in current regime** (#3142 closed). May revisit after
  AMP/EMA compose.
- **30-min wall-clock cap binds at ~14/50 epochs** for baseline width. AMP +
  peak LR tune may help reach deeper cosine decay.
- **NaN bug fixed.** `test_avg/mae_surf_p` now finite for all future runs.
- **Config default loss_fn="mse" is a live issue.** PR #3440 (alphonse) will fix.
- **Run-to-run variance ~3-4 mae_surf_p units.** ≥5 unit improvement needed for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn  | EMA model weights (rebase) | sent back 20:55; rebasing on Charbonnier base |
| #3330 | frieren   | bf16 AMP (rebase) | sent back 19:26; back online 21:21 |
| #3348 | fern      | Fourier position encoding (rebase) | **sent back 21:35**; rebase + 2 arms (raw_charb + fourier_L8_charb) |
| #3370 | tanjiro   | Gated MLPs - GeGLU (rebase) | **sent back 21:30**; rebase + 2 arms (vanilla_charb + geglu_charb) |
| #3398 | edward    | Charbonnier ε sweep ({3e-4, 1e-3, 3e-3}) | wip; needs --loss_fn flag (notified) |
| #3418 | nezuko    | Gradient clipping sweep (max_norm ∈ {0, 0.5, 1.0}) | wip; needs --loss_fn flag (notified) |
| #3440 | alphonse  | Fix Config default: loss_fn="charbonnier" | wip (critical fix) |
| #3457 | askeladd  | Peak LR sweep on Charbonnier base (5e-4 vs 1e-3 vs 2e-3) | wip (new) |

## Closed / merged this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; closed |
| #3149 | nezuko  | Per-channel surf-p loss weighting — capacity steal; closed |
| #3145 | fern    | Deeper Transolver — wall-clock confound; closed |
| #3150 | tanjiro | **Warmup + cosine LR — merged ⭐ (val 125.83)** |
| #3143 | edward  | **Charbonnier robust loss — merged ⭐⭐ (val 98.60, −18.6%)** |
| #3138 | alphonse | **NaN bug fix — merged ✅ (first valid test_avg=92.71)** |
| #3331 | nezuko  | Separate per-channel heads — closed |
| #3142 | askeladd | Surf weight sweep — closed (0.69%, non-monotonic) |
| #3151 | thorfinn | EMA (R1) — sent back (−17.8% test, stale base) |
| #3330 | frieren | bf16 AMP (R1) — sent back (−13%, stale base) |
| #3348 | fern    | Fourier L=8 (R1) — sent back (−3.6%, −18.3 abs on single_in_dist, stale base) |
| #3370 | tanjiro | GLU MLPs (R1) — sent back (−2.9%, OOD-favorable, stale base) |

## Strategic outlook

**The rebase queue is unusually full** — 4 sent-back PRs (#3151, #3330, #3348, #3370)
all have meaningful within-PR signals but on the pre-Charbonnier base. The next
2-3 hours of student work will be heavily rebase-driven.

**Expected high-value composition paths** (in priority order, all on new merged
Charbonnier+warmup+NaN-fix base):

1. **EMA rebase (#3151)** — strongest within-PR signal (−17.8% test). Expected ~80-90 val.
2. **AMP rebase (#3330)** — 1.34× more epochs, −13% within-PR. Expected ~85-92 val.
3. **Fourier L=8 rebase (#3348)** — −18.3 absolute on hardest split within-PR. Expected ~90-95 val.
4. **GeGLU rebase (#3370)** — noise-bordered overall but OOD-favorable. May win on test_avg.
5. **Charbonnier ε tune (#3398)** — fine-tuning of merged lever.
6. **Peak LR (#3457)** — Charbonnier may allow higher LR for faster cosine decay.
7. **Grad clip (#3418)** — regularization with proven track record in transformers.
8. **Loss default fix (#3440)** — operational; eliminates flag-mistake risk.

**Compose math (if all proportional gains hold):** If EMA gives −15% on Charbonnier
base, AMP gives −10%, Fourier gives −5%, GeGLU gives −2% (mostly OOD), the
multiplicative compose lands in the high 50s val. Even partial composition (any
2-3 of these) plausibly puts us in 70-80 range.

**Risk:** if rebased runs simply track ~98 baseline with within-PR signal not
proportionally preserved, the round-9 close rate will be high. The −17.8% EMA
signal is the most insurance against this.

## Potential next-round directions (when current 8 PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive outlier suppression than Charbonnier)
- Per-channel ε (surf_p may want different ε from Ux/Uy)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- RMSNorm vs LayerNorm
- torch.compile (1.3-2× speedup, fully orthogonal)
- L sweep on fourier_basic (4, 6, 8 — geometric argument suggests optimum may be < 8)
- GeGLU at param-matched mlp_ratio=8/3 (clean ablation)

**Training:**
- Per-step warmup (vs per-epoch)
- Larger batch with AMP + scaled LR
- Gradient accumulation

**Data:**
- AoA + Re jitter augmentation
- Curriculum learning (single-foil first)
- Hard example mining
