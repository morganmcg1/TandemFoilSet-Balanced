# SENPAI Research State

- 2026-05-16 ~02:45 — **Round-16 in progress**. 
  - Merged #3494 (nezuko grad_clip default flip — operational).
  - Sent #3370 tanjiro (GeGLU, val=89.17, test=80.51 — BEST RESULT so far) back for rebase onto Fourier head, 1 confirmation arm needed.
  - Sent #3330 frieren (bf16 val=100.74, test=83.60) back for rebase + 1 confirmation arm.
  - Fixed student labels on #3605 (alphonse Cauchy) and #3600 (fern Fourier L sweep).
  - Assigned #3630 nezuko (weight decay sweep: wd ∈ {1e-5, 1e-3}).
  - #3151 thorfinn EMA: now MERGEABLE/CLEAN — ema_0p99 arm running.
- No directives from the human researcher team.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Seven items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 (#3348),
grad-clip default flip (#3494).

Primary validation target: **val_avg/mae_surf_p < 97.47** (primary val best).
Paper-facing test target: **test_avg/mae_surf_p < 86.22** (fourier_L8_charb jum9x071).

**🔥 GeGLU (#3370 tanjiro) val=89.17, test=80.51 — STRONGEST RESULT THIS LAUNCH.**
Needs 1 rebase + confirmation arm (geglu_fourier_charb on Fourier base).

All defaults now correct — bare `python train.py` uses Charbonnier ε=1e-3, 
grad_clip=0.5, Fourier L=8 positional encoding, warmup+cosine LR.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights | MERGEABLE — ema_0p99 running, results pending |
| #3330 | frieren  | bf16 AMP rebase | bf16_v3 done (val=100.74, test=83.60); rebase+1 arm pending |
| #3370 | tanjiro  | GeGLU MLPs — **BEST RESULT val=89.17, test=80.51** | sent back for rebase onto Fourier head |
| #3457 | askeladd | Peak LR sweep lr2e-3 | lr2e-3 arm running |
| #3570 | edward   | torch.compile speedup | baseline_no_compile running |
| #3600 | fern     | Fourier L sweep L=4, L=6 | fourier_L4 arm running |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | newly assigned |
| #3630 | nezuko   | AdamW weight decay sweep {1e-5, 1e-3} | newly assigned |

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

## Strategic outlook

**GeGLU compose is the #1 priority.** Tanjiro's val=89.17/test=80.51 beats baseline
by −8.3 val / −5.7 test. After rebase onto Fourier head and 1 confirmation arm, this
should merge. If GeGLU+Fourier compose proportionally: expected test ~74-78.

**EMA (#3151) is next.** Control arm 100.21 confirmed; ema_0p99 running. Expected
~80-90 val based on stale-base signal (−17.8% test). This stacks with GeGLU if both merge.

**bf16 AMP (#3330)** — test=83.60 (−2.62 vs 86.22) on pre-rebase; after Fourier compose
expected test in 78-82 range.

**Expected compose math (if GeGLU + EMA + bf16 all merge):**
- GeGLU: val ~89, test ~80
- EMA: additional ~−10-15% multiplicative
- bf16: additional ~−3-5% from deeper cosine
- Combined: val potentially in low-mid 70s, test in high 60s-low 70s

## Potential next-round directions (when current PRs close)

**Loss-form:**
- Prediction-NaN guard in scoring.py (defensive; ε=3e-3 found to produce NaN preds)
- Welsch/Tukey loss (same family as Cauchy, different tail behavior)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- Hybrid Fourier (concat Fourier + raw coords)
- Learnable Fourier frequencies (16 extra params)
- Attention output projection bias

**Training:**
- SAM (Sharpness-Aware Minimization)
- Per-step warmup (currently per-epoch)
- OneCycle LR schedule
- Gradient accumulation (effective batch 8 or 16)

**Systems:**
- torch.compile + AMP composition run
- Curriculum learning (single-foil first)
- Re jitter augmentation for re_rand OOD split
