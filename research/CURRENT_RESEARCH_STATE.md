# SENPAI Research State

- 2026-05-16 01:00 — **Round-13 review complete**. Closed #3398 (edward
  Charbonnier ε sweep — null result, ε=1e-3 default confirmed optimal by
  clean 3-point curve; best arm 101.40 does not beat merged baseline 97.47).
  Assigned edward to #3570 (torch.compile orthogonal speedup). Sent back
  #3457 (askeladd stuck on lr5e-4 re-runs; directed to move to lr1e-3/lr2e-3).
  All 8 students now have active WIP PRs.
- No directives from the human researcher team on this launch.

## Current research focus

Round 13+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Five items merged**: warmup+cosine (#3150), Charbonnier robust loss
(#3143), NaN evaluate_split bug fix (#3138), Charbonnier-default-flip
(#3440), grad-clip lever (#3418, clip_0p5 new best val=97.47, default
still 0.0 pending #3494). Primary validation target:
**val_avg/mae_surf_p < 97.47**. Paper-facing test metric:
**test_avg/mae_surf_p = 92.71** (compose sanity u2k87wan; PR #3138).

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`, `warmup_epochs=3`,
  `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143)
- Charbonnier is the default loss (PR #3440)
- NaN filter in `evaluate_split` (PR #3138)
- `--grad_clip_max_norm` CLI lever, default 0.0 (flip to 0.5 pending #3494)

## Key constraints / insights confirmed

- **Charbonnier robust loss is the dominant single lever.** −18.6% on
  val_avg/mae_surf_p vs MSE. All residuals in L1 regime at ε=1e-3.
- **Grad clipping max_norm=0.5 is the 2nd merged lever.** −1.13 absolute on
  top of Charbonnier (98.60 → 97.47 best single-seed).
- **ε=1e-3 is the confirmed Charbonnier optimum** (#3398 null sweep).
  3-point curve (3e-4, 1e-3, 3e-3) shows L2-helps-IID / L1-helps-OOD
  split pattern — ε=3e-3 wins val_single_in_dist but loses all OOD splits.
- **EMA is likely a major winner.** #3151 thorfinn re-rebased onto 0148797,
  running 3 arms with --grad_clip_max_norm 0.5. Expected ~80-92 val.
- **AMP (bf16) proven lever, rebasing.** #3330 frieren running.
- **GeGLU shows OOD-favorable signal.** #3370 tanjiro MERGEABLE, rebasing.
- **Fourier L=8 architectural lever.** #3348 fern running fourier_L8_charb.
- **Separate output heads dead** (#3331 closed). Surf_weight weak in current
  regime (#3142 closed).
- **30-min wall-clock cap binds at ~14/50 epochs.** torch.compile (#3570)
  targets this directly: 1.3-2× per-step speedup → 18-22 epochs in same
  budget.
- **Default-flip pattern.** grad_clip default 0.5 pending #3494; loss_fn
  default already flipped (#3440).
- **Run-to-run variance ~3-4 mae_surf_p units.** ≥5 unit improvement needed
  for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (rebase R2) | student re-rebased 01:28, running 3 arms with clip_0p5 |
| #3330 | frieren  | bf16 AMP (rebase) | running arms |
| #3348 | fern     | Fourier position encoding L=8 (rebase) | fourier_L8_charb arm running |
| #3370 | tanjiro  | Gated MLPs - GeGLU (rebase) | MERGEABLE, running geglu_charb arm |
| #3457 | askeladd | Peak LR sweep (5e-4 vs 1e-3 vs 2e-3) | redirected to lr1e-3/lr2e-3 arms |
| #3494 | nezuko   | Flip Config default grad_clip 0.0 → 0.5 | single-arm done (val=101.19), results pending |
| #3499 | alphonse | RMSNorm replacement for LayerNorm | layernorm_ref done (102.38), rmsnorm arm TBD |
| #3570 | edward   | torch.compile orthogonal speedup | newly assigned (R13) |

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
| #3348 | fern    | Fourier L=8 (R1) — sent back (−3.6%, −18.3 abs on single_in_dist) |
| #3370 | tanjiro | GLU MLPs (R1) — sent back (−2.9%, OOD-favorable, stale base) |
| #3440 | alphonse | **Loss-fn default flip — merged ✅** |
| #3418 | nezuko  | **Grad-clip lever — merged ⭐ (clip_0p5 new best val=97.47)** |
| #3398 | edward  | Charbonnier ε sweep — closed (null: ε=1e-3 confirmed optimal, 3-point curve) |

## Strategic outlook

The rebase queue is resolving. Two PRs (#3348 fern, #3370 tanjiro) are
MERGEABLE. #3151 thorfinn and #3330 frieren are actively training arms.

**Expected high-value composition paths** (all targeting val < 97.47):

1. **EMA rebase (#3151)** — strongest within-PR signal (−17.8% test on stale
   base). Expected ~80-92 val on Charbonnier+clip base.
2. **AMP rebase (#3330)** — 1.34× more epochs, −13% within-PR. Expected
   ~85-92 val.
3. **Fourier L=8 rebase (#3348)** — −18.3 absolute on hardest split, L=8
   confirmed by spectral bandwidth argument. Expected ~90-95 val.
4. **GeGLU rebase (#3370)** — noise-bordered overall but OOD-favorable on
   cruise+re splits. May win on test_avg.
5. **torch.compile (#3570)** — pure wall-clock lever, may give 1.3-2× more
   training per budget, no numerical trade-off.
6. **Peak LR (#3457)** — Charbonnier+clip may tolerate higher LR; lr1e-3
   and lr2e-3 arms pending.
7. **RMSNorm (#3499)** — rmsnorm arm pending.
8. **Grad-clip default flip (#3494)** — operational, single-arm done.

**Compose math (if proportional gains hold):** EMA −15%, AMP −10%, Fourier
−5%, GeGLU −2% → multiplicative compose lands in high 50s val. Even 2-3
of these composes plausibly puts us in 70-80 range.

## Potential next-round directions (when current 8 PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive suppression, but edward's ε sweep
  suggests we may already be near the L1 boundary benefit)
- Prediction-NaN guard in scoring.py (defensive fix found by edward #3398)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- L sweep on fourier_basic (4, 6, 8) — if L=8 merges clean
- GeGLU at param-matched mlp_ratio=8/3 (clean ablation)
- Attention output projection bias

**Training:**
- SAM (Sharpness-Aware Minimization) — orthogonal to grad clip
- Per-step warmup (currently per-epoch)
- OneCycle LR schedule (different shape)
- Gradient accumulation for larger effective batch

**Systems:**
- torch.compile + AMP composition confirmation run
- AoA + Re jitter augmentation
- Curriculum learning (single-foil first)
