# SENPAI Research State

- 2026-05-16 — **Round-15 review complete**. Merged #3348 (fern Fourier
  positional encoding L=8 — new test best 86.22, −6.49 absolute vs 92.71).
  Assigned fern to #3600 (Fourier L sweep: confirm L=8 optimal vs L=4, L=6).
  7 active WIP PRs (fern re-assigned; 8 total students).
- No directives from the human researcher team on this launch.

## Current research focus

Round 15+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Six items merged**: warmup+cosine (#3150), Charbonnier robust loss
(#3143), NaN evaluate_split bug fix (#3138), Charbonnier-default-flip
(#3440), grad-clip lever (#3418, clip_0p5 new best val=97.47, default
still 0.0 pending #3494), Fourier positional encoding L=8 (#3348).

Primary validation target: **val_avg/mae_surf_p < 97.47**.
Paper-facing test metric: **test_avg/mae_surf_p = 86.22** (fourier_L8_charb
jum9x071; PR #3348 merged R15).

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`, `warmup_epochs=3`,
  `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143)
- Charbonnier is the default loss (PR #3440)
- NaN filter in `evaluate_split` (PR #3138)
- `--grad_clip_max_norm` CLI lever, default 0.0 (flip to 0.5 pending #3494)
- Fourier positional encoding `--pos_enc_mode fourier_basic`, default L=8 (PR #3348)

## Key constraints / insights confirmed

- **Charbonnier robust loss is the dominant single lever.** −18.6% on
  val_avg/mae_surf_p vs MSE. All residuals in L1 regime at ε=1e-3.
- **Grad clipping max_norm=0.5 is the 2nd merged lever.** −1.13 absolute on
  top of Charbonnier (98.60 → 97.47 best single-seed).
- **ε=1e-3 is the confirmed Charbonnier optimum** (#3398 null sweep).
  3-point curve (3e-4, 1e-3, 3e-3) shows L2-helps-IID / L1-helps-OOD
  split pattern — ε=3e-3 wins val_single_in_dist but loses all OOD splits.
- **Fourier L=8 positional encoding merged.** Test improvement −6.49 (−7.0%),
  val within noise. Geom_camber_cruise split improved most (test_p 55.77 vs
  ~63.99 prior). Val primary metric effectively tied (98.16 vs 97.47).
- **EMA is likely a major winner.** #3151 thorfinn re-rebased onto 0148797,
  running ema_0.99 + ema_0.999 arms with --grad_clip_max_norm 0.5.
  Control arm 100.21 confirms correct compose. Expected ema arms ~80-92 val.
- **AMP (bf16) proven lever, rebasing.** #3330 frieren running.
- **GeGLU shows OOD-favorable signal.** #3370 tanjiro running geglu_charb arm.
- **RMSNorm confirmed NULL.** #3499 alphonse: rmsnorm=107.77 vs
  layernorm_ref=102.38, +5.4 worse — closed.
- **30-min wall-clock cap binds at ~14/50 epochs.** torch.compile (#3570)
  targets this directly: 1.3-2× per-step speedup → 18-22 epochs in same
  budget.
- **Default-flip pattern.** grad_clip default 0.5 pending #3494.
- **Run-to-run variance ~3-4 mae_surf_p units.** ≥5 unit improvement needed
  for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (rebase R2) | running ema_0.99 + ema_0.999 arms with clip_0p5 |
| #3330 | frieren  | bf16 AMP (rebase) | redirected to bf16_amp arm |
| #3370 | tanjiro  | Gated MLPs - GeGLU (rebase) | redirected to geglu_charb arm |
| #3457 | askeladd | Peak LR sweep (lr1e-3/lr2e-3) | lr1e-3 arm running, lr2e-3 pending |
| #3494 | nezuko   | Flip Config default grad_clip 0.0 → 0.5 | single-arm done (val=101.19), results pending |
| #3570 | edward   | torch.compile orthogonal speedup | assigned R13, running |
| #3600 | fern     | Fourier L sweep (L=4, L=6 vs merged L=8) | newly assigned R15 |
| #3151 | thorfinn | EMA model weights (rebase R2) | conflict state on GitHub despite student re-rebase |

*(Note: alphonse's #3499 was closed null — RMSNorm worse. Alphonse needs new assignment if idle.)*

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
| #3348 | fern    | **Fourier L=8 pos encoding — merged ⭐ (test 86.22, −6.49)** |
| #3370 | tanjiro | GLU MLPs (R1) — sent back (−2.9%, OOD-favorable, stale base) |
| #3440 | alphonse | **Loss-fn default flip — merged ✅** |
| #3418 | nezuko  | **Grad-clip lever — merged ⭐ (clip_0p5 new best val=97.47)** |
| #3398 | edward  | Charbonnier ε sweep — closed (null: ε=1e-3 confirmed optimal) |
| #3499 | alphonse | RMSNorm replacement — closed (null: +5.4 worse than LayerNorm) |

## Strategic outlook

EMA (#3151 thorfinn) is the highest-expected-value experiment in queue: control
arm at 100.21 confirms correct composition; EMA arms expected in 80-92 val range
(−17.8% test on stale base implies strong signal). This is likely the next merger.

**Expected high-value composition paths** (all targeting val < 97.47):

1. **EMA rebase (#3151)** — strongest within-PR signal (−17.8% test on stale
   base). With control at 100.21, EMA arms expected ~82-92 val.
2. **AMP rebase (#3330)** — 1.34× more epochs, −13% within-PR. Expected
   ~85-92 val.
3. **GeGLU rebase (#3370)** — noise-bordered overall but OOD-favorable on
   cruise+re splits. May win on test_avg.
4. **torch.compile (#3570)** — pure wall-clock lever, may give 1.3-2× more
   training per budget, no numerical trade-off.
5. **Peak LR (#3457)** — Charbonnier+clip may tolerate higher LR; lr1e-3
   arm running.
6. **Fourier L sweep (#3600)** — confirm L=8 optimal vs L=4, L=6.
7. **Grad-clip default flip (#3494)** — operational clean-up, results pending.

## Potential next-round directions (when current PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive suppression, but ε sweep suggests
  we may be near the L1 boundary benefit already)
- Prediction-NaN guard in scoring.py (defensive fix found by edward #3398)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- Hybrid Fourier (concat Fourier + raw coords) — potentially better local precision
- Learnable Fourier frequencies (16 extra params, adapt to geometry distribution)
- GeGLU at param-matched mlp_ratio=8/3 (clean ablation if #3370 doesn't merge)
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

**Check alphonse status** — #3499 closed null (RMSNorm). If idle, assign new
hypothesis immediately.
