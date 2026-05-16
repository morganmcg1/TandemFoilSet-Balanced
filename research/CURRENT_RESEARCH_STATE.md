# SENPAI Research State

- 2026-05-16 ~06:40 — **CRITICAL: defaults bug discovered and corrected**.
  - Both **fern (#3600)** and **alphonse (#3605)** independently flagged that BASELINE.md falsely claimed bf16 (#3330) and Fourier (#3348) are active defaults.
  - **Reality verified**: `train.py` Config defaults are still `pos_enc_mode="raw"`, `amp_dtype="fp32"`, `mlp_type="vanilla"`. All three PRs added LEVERS, not default flips.
  - Per fern's W&B verification, tanjiro's val=81.48 run 8ile1q1j had `amp_dtype=None` (fp32), meaning the published best was actually **fp32 + raw + GeGLU + Charbonnier + clip 0.5** — NOT the stacked GeGLU+bf16+Fourier as previously documented.
  - **BASELINE.md fixed in commit 254940b** — three levers now correctly marked as ⚠️ levers requiring explicit flags.
  - **All in-flight PRs updated** with corrected commands explicitly passing `--mlp_type geglu --pos_enc_mode fourier_basic --amp_dtype bf16`.
  - **Implication**: We have never measured the truly-stacked GeGLU+bf16+Fourier baseline. The tanjiro #3704 control arm will be the first ever measurement of that config.

- 2026-05-16 ~05:54 — **Round-20 in progress**.
  - **#3630 nezuko wd sweep:** wd_1e-3 gave val=93.78/test=85.34 on OLD base. Sent back for re-run on truly-stacked base.
  - **#3151 thorfinn EMA, #3570 edward compile, #3605 alphonse Cauchy:** Sent back with rebase + re-run instructions targeting the truly-stacked base.

- 2026-05-16 ~05:22 — **Round-19 complete**.
  - **MERGED: #3370 tanjiro GeGLU** — val=81.48, test=72.68 — **NEW BEST on both metrics**.
  - Baseline now val=81.48 / test=72.68 (per the actual 8ile1q1j config: fp32 + raw + GeGLU + Charbonnier + clip 0.5).
  - Assigned tanjiro → GeGLU readout PR #3704 (extend gating to mlp2 + first measurement of truly-stacked baseline)

- No directives from the human researcher team.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Nine items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 lever (#3348),
grad-clip default flip (#3494), bf16 AMP lever (#3330), **GeGLU MLPs lever (#3370)**.

Primary validation target: **val_avg/mae_surf_p < 81.48** (PR #3370 run 8ile1q1j on fp32+raw+GeGLU base).
Paper-facing test target: **test_avg/mae_surf_p < 72.68** (same run).

Bare `python train.py` uses: Charbonnier ε=1e-3 ✅, grad_clip=0.5 ✅, but **fp32** (not bf16), **raw_pos_enc** (not Fourier), **vanilla_mlp** (not GeGLU) — these three are levers requiring explicit flags.

**Known operational issues — defaults that need flipping** (no idle student available):
- `pos_enc_mode` default `"raw"` → `"fourier_basic"` (PR #3348 verified +6.49 test gain)
- `amp_dtype` default `"fp32"` → `"bf16"` (PR #3330 verified +1.33× speedup → −10.5 val gain)
- `mlp_type` default `"vanilla"` → `"geglu"` (PR #3370 verified −14.7% val on Charb+clip base)

## In-flight hypotheses (8 active PRs — all sent updated stacked-flag commands)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | Updated cmd posted — rebase + re-run on truly-stacked base |
| #3570 | edward   | torch.compile speedup | Updated cmd posted — rebase + re-run on truly-stacked base |
| #3600 | fern     | Fourier L sweep L=4, L=6, L=8 | Updated cmd posted — first measurement of truly-stacked baseline (L8 arm) |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} + Charb stacked ctrl | Updated cmd posted — rebase + re-run |
| #3630 | nezuko   | AdamW weight decay sweep {1e-5, 1e-3, 1e-4 ctrl} | Updated cmd posted — rebase + re-run on truly-stacked base |
| #3667 | askeladd | OneCycleLR schedule max_lr ∈ {1e-3, 2e-3} + cosine ctrl | Pinged (stale_wip 0 comments) — updated stacked cmd |
| #3668 | frieren  | Gradient accumulation effective bs=8/16 | Updated cmd posted — stacked-base commands |
| #3704 | tanjiro  | GeGLU readout (mlp2) + baseline sanity arm | Updated cmd posted — control arm = first truly-stacked baseline measurement |

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
| #3142 | askeladd | Surf weight sweep — closed |
| #3440 | alphonse | **Loss-fn default flip — merged ✅** |
| #3418 | nezuko  | **Grad-clip lever — merged ⭐ (clip_0p5 val=97.47)** |
| #3398 | edward  | Charbonnier ε sweep — closed |
| #3499 | alphonse | RMSNorm replacement — closed |
| #3348 | fern    | **Fourier L=8 lever — merged ⭐ (test 86.22 with explicit flag)** |
| #3494 | nezuko  | **Grad-clip default flip — merged ✅** |
| #3457 | askeladd | Peak LR sweep — closed |
| #3330 | frieren  | **bf16 AMP lever — merged ⭐⭐ (val 83.54, test 73.02 with explicit flag)** |
| #3370 | tanjiro  | **GeGLU MLPs lever — merged ⭐⭐ (val 81.48, test 72.68 — NEW BEST both metrics)** |

## Strategic outlook

**Highest pending experiments by expected impact:**

1. **#3704 tanjiro GeGLU readout sanity arm** — first ever measurement of truly-stacked GeGLU+bf16+Fourier baseline. Critical data point.
2. **#3151 thorfinn EMA on truly-stacked base** — expected val ~67-72 if EMA's R1 −17.8% gain composes on the (predicted) stacked baseline of ~70-73.
3. **#3667 askeladd OneCycleLR** — alternative LR schedule shape, potential super-convergence at 30-min cap.
4. **#3668 frieren gradient accumulation** — cleaner effective-batch gradient signal on truly-stacked base.

**Expected compose ceiling**: If GeGLU + bf16 + Fourier stack proportionally (each lever's effect verified separately), val potentially in low 70s, test in mid 60s. Then EMA adds another −15-20%. Most aggressive optimistic scenario: val ~58-65, test ~52-58.

**Post-merge operational follow-ups needed** (NO idle student to assign these to right now):
- `pos_enc_mode` default raw → fourier_basic
- `mlp_type` default vanilla → geglu
- `amp_dtype` default fp32 → bf16

These three default flips are pure operational hygiene — they should be batched into a single small PR by whichever student becomes idle next (winning their current arm or having a clear close).

## Potential next-round directions

**Architecture:**
- Per-channel readout heads — `shared_head(z) + α·per_channel_correction(z)`
- Hybrid Fourier (concat Fourier + raw coords)
- Learnable Fourier frequencies (16 extra params)
- Deeper Transolver now that bf16 frees VRAM (~52 GB headroom)

**Training:**
- SAM (Sharpness-Aware Minimization)
- Lookahead optimizer wrapper
- Per-step warmup (currently per-epoch)
- SGDR cosine restarts

**Systems:**
- Padding-aware bucketed batching (orthogonal throughput lever)
- torch.compile + AMP composition (edward #3570)
- Curriculum learning (single-foil first)

**Operational hygiene (HIGH priority — to assign on next idle):**
- Triple default flip PR (pos_enc_mode, amp_dtype, mlp_type)
