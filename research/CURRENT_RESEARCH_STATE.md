# SENPAI Research State

- 2026-05-16 ~05:54 — **Round-20 in progress**.
  - **#3630 nezuko wd sweep:** wd_1e-3 gave val=93.78/test=85.34 on OLD base — doesn't beat new baseline (81.48). Signal was OOD-favorable as predicted. **Sent back** for confirmation arm on new GeGLU+bf16+Fourier base. Expected proportional gain val ~77-78.
  - **#3151 thorfinn EMA, #3570 edward compile, #3605 alphonse Cauchy:** All went silent for 1-2.5 hours. All have CONFLICTING/stale state. Sent ping + rebase instructions for each, with specific commands targeting the new merged base.
  - No idle students after this round — all 8 students have active WIP.
- 2026-05-16 ~05:22 — **Round-19 complete**.
  - **MERGED: #3370 tanjiro GeGLU** — val=81.48, test=72.68 — **NEW BEST on both metrics**.
  - Baseline now val=81.48 / test=72.68 (GeGLU+bf16+Fourier+Charbonnier+clip via `--mlp_type geglu`).
  - Caveat: val=81.48 was measured on no-bf16 base; the actual merged config is likely ~70-73 val.
  - Assigned tanjiro → GeGLU readout PR #3704 (extend gating to mlp2 + sanity-confirm merged baseline)
  - Pinged fern #3600 (stale_wip, no L=4/L=6 results posted) — recommended rebase + GeGLU-base re-run
  - Thorfinn #3151 still WIP after rebase request from 04:27 — no comment yet
  - GitHub rate limit recovered (3815/5000) after wait
- No directives from the human researcher team.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Nine items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 (#3348),
grad-clip default flip (#3494), bf16 AMP (#3330), **GeGLU MLPs (#3370)**.

Primary validation target: **val_avg/mae_surf_p < 81.48** (NEW — PR #3370 geglu_fourier_charb, run 8ile1q1j).
Paper-facing test target: **test_avg/mae_surf_p < 72.68** (NEW — same run).

Bare `python train.py` uses: Charbonnier ε=1e-3, grad_clip=0.5, Fourier L=8, warmup+cosine LR, bf16 AMP. **GeGLU lever now available** via `--mlp_type geglu` (default vanilla).

**Known operational issues** — defaults to flip:
- `pos_enc_mode` default still `"raw"` despite #3348 merge (Fourier wiring is correct via flag; default flip needed)
- `mlp_type` default still `"vanilla"` despite #3370 merge (GeGLU lever via `--mlp_type geglu`)

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | Rebase requested 04:27 — awaiting re-run on bf16+Fourier base (expected ~74-80) |
| #3570 | edward   | torch.compile speedup | control done (104.41), compile arm running |
| #3600 | fern     | Fourier L sweep L=4, L=6 | **stale_wip** — pinged at 05:22 for status |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | re-run with Fourier flag pending |
| #3630 | nezuko   | AdamW weight decay sweep {1e-5, 1e-3} | wd_1e-5 running |
| #3667 | askeladd | OneCycleLR schedule max_lr ∈ {1e-3, 2e-3} | newly assigned (R18) |
| #3668 | frieren  | Gradient accumulation effective bs=8/16 | newly assigned (R18) |
| #3704 | tanjiro  | GeGLU readout (mlp2) + baseline sanity arm | newly assigned (R19) |

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
| #3348 | fern    | **Fourier L=8 — merged ⭐ (test 86.22)** |
| #3494 | nezuko  | **Grad-clip default flip — merged ✅** |
| #3457 | askeladd | Peak LR sweep — closed |
| #3330 | frieren  | **bf16 AMP — merged ⭐⭐ (val 83.54, test 73.02)** |
| #3370 | tanjiro  | **GeGLU MLPs — merged ⭐⭐ (val 81.48, test 72.68 — NEW BEST both metrics)** |

## Strategic outlook

**Highest pending experiments by expected impact:**

1. **#3151 thorfinn EMA on bf16+Fourier base** — expected val ~69-74 (R1 −17.8% test signal applied to current baseline).
2. **#3704 tanjiro GeGLU readout sanity + extension** — control arm gives true GeGLU+bf16+Fourier baseline (~70-73); test arm explores small additional gain.
3. **#3667 askeladd OneCycleLR** — alternative LR schedule shape, potential super-convergence at 30-min cap.
4. **#3668 frieren gradient accumulation** — cleaner effective-batch gradient signal.

**Expected compose ceiling**: If EMA + GeGLU + bf16 + Fourier all stack proportionally, val potentially in low-mid 60s, test in high 50s. Most aggressive optimistic scenario.

**Post-merge operational follow-ups needed:**
- `pos_enc_mode` default raw → fourier_basic
- `mlp_type` default vanilla → geglu (after sanity confirmation in #3704)
- `amp_dtype` was flipped fp32 → bf16 in #3330 ✓

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
