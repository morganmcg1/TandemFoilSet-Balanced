# SENPAI Research State

- 2026-05-15 22:30 — **Round-9 review complete**. Merged 2 PRs: **#3418
  nezuko grad-clip lever (clip_0p5 winner; new best val_avg/mae_surf_p =
  **97.47**, test_avg = 95.96 on 3 splits)** and **#3440 alphonse loss-fn
  default flip (Charbonnier auto-applies on bare `train.py`)**. Assigned
  2 new follow-ups: **#3494 nezuko grad-clip-default-fix** (flip Config
  default 0.0 → 0.5) and **#3499 alphonse RMSNorm** (replace LayerNorm with
  RMSNorm in TransolverBlocks). Notified all 4 rebasing PRs (#3151, #3330,
  #3348, #3370) to incorporate `--grad_clip_max_norm 0.5` and drop the now-
  unnecessary `--loss_fn charbonnier` flag in their rerun commands.
- No directives from the human researcher team on this launch.

## Current research focus

Round 9+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Five items merged**: warmup+cosine (#3150), Charbonnier robust loss
(#3143), NaN evaluate_split bug fix (#3138), Charbonnier-default-flip
(#3440), grad-clip lever (#3418, default still 0.0 pending #3494). Primary
validation target: **val_avg/mae_surf_p < 97.47** (best single-seed,
clip_0p5 win in #3418, run 221dquoy). Paper-facing test metric:
**test_avg/mae_surf_p = 92.71** (compose sanity u2k87wan; PR #3138).
8 hypotheses in flight, **4 of them in active rebase mode** waiting for
combined Charbonnier+clip0p5-base composition results.

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`, `warmup_epochs=3`,
  `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143)
- Charbonnier is the default loss now (PR #3440)
- NaN filter in `evaluate_split` (PR #3138)
- `--grad_clip_max_norm` CLI lever, default 0.0 pending #3494 flip to 0.5
  (PR #3418)

## Key constraints / insights confirmed

- **Charbonnier robust loss is the dominant single lever.** −18.6% on
  val_avg/mae_surf_p vs MSE. Linearizes gradient for large residuals;
  surface pressure has order-of-magnitude dynamic range. Improves every
  channel, every split.
- **Grad clipping at max_norm=0.5 is the 2nd merged lever.** −1.13 absolute
  on top of Charbonnier (98.60 → 97.47), with a 9.4-unit within-PR signal
  on the same seed/base. clip=1.0 is too loose; clip=0.5 catches the spiky
  surface-pressure batches.
- **EMA is a likely major winner on rebase.** PR #3151 showed −17.8% test
  within-PR on stale base. Sent back.
- **AMP (bf16) is a proven lever, rebasing.** PR #3330 showed 1.34× epoch
  speedup → −13% within-PR.
- **GeGLU shows OOD-favorable signal.** PR #3370 (−2.9% within-PR overall,
  but −13.6% on val_geom_camber_cruise, −13.0% on test_re_rand). SwiGLU
  is null. GELU > SiLU as gate activation for Transolver. Sent back for
  rebase.
- **Fourier position encoding L=8 is a real architectural lever.** PR #3348
  (−18.3 absolute on val_single_in_dist surf_p). L>8 hurts because
  standardized position σ≈1 makes bands above 2^7 sub-mesh-spacing noise.
  Sent back for rebase.
- **Separate output heads is dead** (#3331 closed). Shared trunk is NS
  cross-field inductive bias.
- **Surf_weight tune is weak in current regime** (#3142 closed). May revisit
  after AMP/EMA compose.
- **30-min wall-clock cap binds at ~14/50 epochs** for baseline width. AMP +
  peak LR tune may help reach deeper cosine decay.
- **NaN bug fixed.** `test_avg/mae_surf_p` now finite for all future runs.
- **Default-flip pattern recognized.** When a PR adds a CLI flag with a
  "safe" default (MSE in #3143, no-clip in #3418), bare `train.py` doesn't
  benefit. Need a separate default-flip PR. #3440 fixed loss_fn; #3494
  will fix grad_clip.
- **Run-to-run variance ~3-4 mae_surf_p units.** ≥5 unit improvement needed
  for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn  | EMA model weights (rebase) | sent back r7; rebasing |
| #3330 | frieren   | bf16 AMP (rebase) | sent back r6; rebasing on new base |
| #3348 | fern      | Fourier position encoding (rebase) | sent back r8; rebasing |
| #3370 | tanjiro   | Gated MLPs - GeGLU (rebase) | sent back r8; rebasing |
| #3398 | edward    | Charbonnier ε sweep ({3e-4, 1e-3, 3e-3}) | wip; default flip means simpler CLI |
| #3457 | askeladd  | Peak LR sweep on Charbonnier base (5e-4/1e-3/2e-3) | wip |
| #3494 | nezuko    | Flip Config default grad_clip 0.0 → 0.5 (+ optional clip_0p25 bonus) | wip (R9) |
| #3499 | alphonse  | RMSNorm replacement for LayerNorm in TransolverBlocks | wip (R9) |

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
| #3440 | alphonse | **Loss-fn default flip — merged ✅ (operational, eliminates flag risk)** |
| #3418 | nezuko  | **Grad-clip lever — merged ⭐ (clip_0p5 new best val=97.47)** |

## Strategic outlook

**The compose stack is getting deeper.** With clip_0p5 now merged baseline,
all 4 rebasing PRs (#3151 EMA, #3330 AMP, #3348 Fourier L=8, #3370 GeGLU)
must re-run with `--grad_clip_max_norm 0.5`. The notification went out to
all 4. Expected within ~3-6 hours given training cycle.

**Expected high-value composition paths** (all on new
Charbonnier+warmup+NaN-fix+clip0p5 base, target val_avg < 97.47):

1. **EMA rebase (#3151)** — strongest within-PR signal (−17.8% test).
   Expected ~80-90 val on new base.
2. **AMP rebase (#3330)** — 1.34× more epochs, −13% within-PR. Expected
   ~85-92 val.
3. **Fourier L=8 rebase (#3348)** — −18.3 absolute on hardest split
   within-PR. Expected ~90-95 val.
4. **GeGLU rebase (#3370)** — noise-bordered overall but OOD-favorable.
   May win on test_avg.
5. **Peak LR (#3457)** — Charbonnier+clip may allow higher LR for faster
   cosine decay.
6. **Charbonnier ε tune (#3398)** — fine-tuning of merged lever.
7. **RMSNorm (#3499)** — forward-pass speedup + slight conditioning
   improvement.
8. **Grad-clip default flip (#3494)** — operational, eliminates flag risk.

**Compose math (if proportional gains hold):** If EMA gives −15% on
Charbonnier+clip base, AMP gives −10%, Fourier gives −5%, GeGLU gives −2%
(mostly OOD), the multiplicative compose lands in the high 50s val. Even
partial composition (any 2-3 of these) plausibly puts us in 70-80 range.

**Risk:** if rebased runs simply track ~97 baseline with within-PR signal
not proportionally preserved, the round-10 close rate will be high. The
−17.8% EMA signal is the most insurance against this.

## Potential next-round directions (when current 8 PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive outlier suppression than
  Charbonnier)
- Per-channel ε (surf_p may want different ε from Ux/Uy)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- torch.compile (1.3-2× speedup, fully orthogonal)
- L sweep on fourier_basic (4, 6, 8 — geometric argument suggests optimum
  may be < 8)
- GeGLU at param-matched mlp_ratio=8/3 (clean ablation)
- Attention-output head bias (allow per-token output shift)

**Training:**
- Per-step warmup (vs per-epoch)
- Larger batch with AMP + scaled LR
- Gradient accumulation
- SAM (Sharpness-Aware Minimization) — orthogonal to grad clip

**Data:**
- AoA + Re jitter augmentation
- Curriculum learning (single-foil first)
- Hard example mining
- Mixup at the mesh-node level
