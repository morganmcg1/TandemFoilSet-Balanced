# SENPAI Research State

- 2026-05-16 ~03:45 — **Round-17 in progress**.
  - Closed #3457 askeladd peak LR sweep (null: best lr2e-3 val=101.63 doesn't beat 97.47).
  - Wrote OneCycle LR hypothesis (`/tmp/hyp-askeladd-onecycle-lr.md`) for askeladd reassignment.
  - **THREE major winners pending student SENPAI-RESULT to merge:**
    - #3370 tanjiro GeGLU+Fourier: run 8ile1q1j val=81.48, test=72.68 (BEST RESULT THIS LAUNCH; needs branch push to clear DIRTY state)
    - #3330 frieren bf16+Fourier: run 5a0rym2t val=83.54, test=73.02, best_epoch=19 (PR MERGEABLE/CLEAN)
    - #3151 thorfinn EMA: run 8ck1dtrb val=87.89, test=78.23 (PR MERGEABLE/CLEAN)
  - All three pinged for terminal SENPAI-RESULT markers.
  - Sent alphonse (#3605 Cauchy) clarification re: pos_enc_mode default bug — charb_control hit val=113.97 on raw coords (expected ~98), needs re-run with `--pos_enc_mode fourier_basic`.
  - Edward (#3570 compile): control done at val=104.41, asked to run compile arm.
  - GitHub API rate limit hit at end of survey; assignment deferred to next wakeup.
- No directives from the human researcher team.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Seven items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 (#3348),
grad-clip default flip (#3494).

Primary validation target: **val_avg/mae_surf_p < 97.47** (primary val best).
Paper-facing test target: **test_avg/mae_surf_p < 86.22** (fourier_L8_charb jum9x071).

**🔥 GeGLU+Fourier (#3370 tanjiro) val=81.48, test=72.68 — STRONGEST RESULT THIS LAUNCH.**
Needs branch push + terminal SENPAI-RESULT to merge.

**🔥 bf16+Fourier (#3330 frieren) val=83.54, test=73.02 — STRONG SECOND WINNER.**
PR MERGEABLE/CLEAN. Needs terminal SENPAI-RESULT to merge.

**🔥 EMA (#3151 thorfinn) val=87.89, test=78.23 — THIRD WINNER.**
PR MERGEABLE/CLEAN. Needs terminal SENPAI-RESULT to merge.

All defaults now correct — bare `python train.py` uses Charbonnier ε=1e-3, 
grad_clip=0.5, Fourier L=8 positional encoding, warmup+cosine LR.

**Known operational issue**: `pos_enc_mode` default is still `"raw"` despite #3348 merge. Students must pass `--pos_enc_mode fourier_basic` explicitly. Operational follow-up PR needed (flip default raw→fourier_basic).

## In-flight hypotheses (7 active PRs + 1 about to assign)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights | **WINNER pending SENPAI-RESULT** — val=87.89/test=78.23 (ema_0p99) |
| #3330 | frieren  | bf16 AMP rebase | **WINNER pending SENPAI-RESULT** — val=83.54/test=73.02 (best_epoch=19) |
| #3370 | tanjiro  | GeGLU MLPs + Fourier | **WINNER pending SENPAI-RESULT + branch push** — val=81.48/test=72.68 |
| #3570 | edward   | torch.compile speedup | control done (104.41), compile arm running |
| #3600 | fern     | Fourier L sweep L=4, L=6 | L=4 finished best_val=93.64; another L=4 run in progress |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | charb_control hit val=113.97 on raw coords; re-run pending |
| #3630 | nezuko   | AdamW weight decay sweep {1e-5, 1e-3} | wd_1e-5 running (step 515+) |
| (pending) | askeladd | OneCycle LR schedule | hypothesis written, assignment pending rate limit recovery |

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

## Strategic outlook

**Three winners ready to compound.** Once students post terminal SENPAI-RESULTs:
- Merge GeGLU+Fourier first (largest gain): val 97.47 → 81.48 (−16.0), test 86.22 → 72.68 (−13.5)
- Merge bf16 second: orthogonal precision lever, val=83.54/test=73.02
- Merge EMA third: weight averaging, val=87.89/test=78.23

These three are roughly orthogonal (architecture × precision × weight averaging), so they should stack. Expected post-stack: val potentially in low-mid 70s, test in mid-60s.

**Post-merge operational follow-ups needed:**
- `pos_enc_mode` default raw → fourier_basic (silent foot-gun like grad_clip was)
- `mlp_type` default vanilla → geglu (after #3370 merge)
- `use_amp` default False → True (after #3330 merge)

## Potential next-round directions (when current PRs close)

**Loss-form:**
- Welsch/Tukey loss (same Cauchy family, different tail behavior)
- Prediction-NaN guard in scoring.py (defensive; ε=3e-3 found to produce NaN preds)

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)`
- Hybrid Fourier (concat Fourier + raw coords)
- Learnable Fourier frequencies (16 extra params)
- Attention output projection bias

**Training:**
- SAM (Sharpness-Aware Minimization)
- Per-step warmup (currently per-epoch)
- Gradient accumulation (effective batch 8 or 16)
- Lookahead optimizer wrapper

**Systems:**
- torch.compile + AMP composition (if both merge separately)
- Curriculum learning (single-foil first)
- Re jitter augmentation for re_rand OOD split
