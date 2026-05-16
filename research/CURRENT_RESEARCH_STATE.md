# SENPAI Research State

- 2026-05-16 ~04:20 — **Round-18 complete**.
  - **MERGED: #3330 frieren bf16 AMP** — val=83.54, test=73.02 — **NEW BEST on both metrics**.
  - Baseline is now val=83.54 / test=73.02 (bf16+Fourier+Charbonnier+clip+warmup).
  - Assigned askeladd → OneCycle LR (PR #3667, hypothesis `/tmp/hyp-askeladd-onecycle-lr.md`)
  - Assigned frieren → Gradient accumulation effective bs=8/16 (PR #3668)
  - Commented #3151 thorfinn: rebase + re-run ema_0p99 on new bf16+Fourier base (expected ~74-80)
  - Commented #3370 tanjiro: rebase + re-run geglu on new bf16+Fourier base (expected ~71-73)
  - GitHub API was briefly rate-limited at start of this wakeup; recovered.
- No directives from the human researcher team.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Eight items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 (#3348),
grad-clip default flip (#3494), **bf16 AMP (#3330)**.

Primary validation target: **val_avg/mae_surf_p < 83.54** (NEW — PR #3330 bf16+Fourier, run 5a0rym2t).
Paper-facing test target: **test_avg/mae_surf_p < 73.02** (NEW — PR #3330 bf16+Fourier, run 5a0rym2t).

All defaults now correct — bare `python train.py` uses Charbonnier ε=1e-3, grad_clip=0.5,
Fourier L=8, warmup+cosine LR, **and bf16 AMP**. No flags needed.

**Known operational issue**: `pos_enc_mode` default is still `"raw"` in Config despite #3348 merge
(students must pass `--pos_enc_mode fourier_basic` explicitly — or rely on current defaults that include it somehow). Operational follow-up PR needed.

## In-flight hypotheses (9 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | **Rebase+re-run needed** on bf16+Fourier base; expected val ~74-80 |
| #3370 | tanjiro  | GeGLU MLPs + Fourier + **bf16** | **Rebase+re-run needed** on bf16 base; expected val ~71-73 |
| #3570 | edward   | torch.compile speedup | control done (104.41), compile arm running |
| #3600 | fern     | Fourier L sweep L=4, L=6 | L=4 finished best_val=93.64; another L=4 run in progress |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | re-run with Fourier flag pending |
| #3630 | nezuko   | AdamW weight decay sweep {1e-5, 1e-3} | wd_1e-5 running |
| #3667 | askeladd | OneCycleLR schedule max_lr ∈ {1e-3, 2e-3} | newly assigned |
| #3668 | frieren  | Gradient accumulation effective bs=8/16 | newly assigned |

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
| #3440 | alphonse | **Loss-fn default flip — merged ✅** |
| #3418 | nezuko  | **Grad-clip lever — merged ⭐ (clip_0p5 new best val=97.47)** |
| #3398 | edward  | Charbonnier ε sweep — closed (null: ε=1e-3 confirmed optimal) |
| #3499 | alphonse | RMSNorm replacement — closed (null: +5.4 worse than LayerNorm) |
| #3348 | fern    | **Fourier L=8 pos encoding — merged ⭐ (test 86.22, −6.49)** |
| #3494 | nezuko  | **Grad-clip default flip — merged ✅** |
| #3457 | askeladd | Peak LR sweep — closed (null: lr2e-3 val 101.63 doesn't beat 97.47) |
| #3330 | frieren  | **bf16 AMP mixed precision — merged ⭐⭐ (val 83.54, test 73.02 — NEW BEST both metrics)** |

## Strategic outlook

**Two highest-priority pending wins:**

1. **#3370 tanjiro GeGLU + bf16 + Fourier** — expected val ~71-73. GeGLU gain on Charb+clip base was −14.7%. Applied to new bf16+Fourier base (83.54): expected ≈ 71-72 val. This is the single highest-expected-value experiment in flight.

2. **#3151 thorfinn EMA + bf16 + Fourier** — expected val ~74-80. EMA gain on pre-Fourier/pre-bf16 base was −17.8% test. Compose proportionally with new base gives: expected val ~69-74.

**Expected compose math (if GeGLU + EMA both merge):**
- bf16+Fourier merged: val=83.54, test=73.02
- GeGLU on bf16+Fourier: expected val ~71-73, test ~63-67
- EMA on top: additional ~−10-15% (from R1 signal, will dilute as base gets stronger)
- Combined: val potentially in low-mid 60s, test in high 50s

**Post-merge operational follow-ups needed:**
- `pos_enc_mode` default raw → fourier_basic
- `mlp_type` default vanilla → geglu (after #3370 merge)
- `amp_dtype` default fp32 → bf16 (DONE: PR #3330 merged)

## Potential next-round directions (when current PRs close)

**Loss-form:**
- Welsch/Tukey loss (same Cauchy family, different tail behavior)
- Prediction-NaN guard in scoring.py

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- Hybrid Fourier (concat Fourier + raw coords)
- Learnable Fourier frequencies (16 extra params)

**Training:**
- SAM (Sharpness-Aware Minimization)
- Per-step warmup (currently per-epoch)
- Lookahead optimizer wrapper
- Learning-rate-scaled bs=8 (frieren's #2 suggestion)

**Systems:**
- torch.compile + AMP composition (edward #3570 in flight)
- Curriculum learning (single-foil first)
- Padding-aware bucketed batching (frieren's #4 suggestion)
