# SENPAI Research State

- 2026-05-16 ~10:25 — **Round-24 complete. New baseline: val=72.59, test=66.45.**
  - **MERGED: #3630 nezuko weight-decay sweep** — wd=1e-3 wins clean 3-arm monotonic sweep; val=72.59, test=66.45 NEW BEST.
  - **Defaults bug (from Round-22) fully corrected** — all 8 in-flight PRs updated with explicit `--mlp_type geglu --pos_enc_mode fourier_basic --amp_dtype bf16 --weight_decay 1e-3` commands.
  - **Truly-stacked baseline confirmed** by 3 independent measurements (~val 74-77, baseline mean ~75.8). PR #3630 wd_1e-3 beats this.
  - **edward #3570 compile_stacked extraordinary result** (f077n973, val=49.14, test=44.07) — 40% gain over baseline. Verification pending: nocompile_ctrl (mq01t5w7) running. Will merge immediately if control confirms.
  - Assigned new work: nezuko #3879 (fourier_rich pos-enc), askeladd #3881 (slice_num sweep).

- 2026-05-16 ~06:40 — **Critical defaults bug corrected** (BASELINE.md commit 254940b).
  - `pos_enc_mode`, `amp_dtype`, `mlp_type` are LEVERS not defaults. All experiments require explicit flags.

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Ten items merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143),
NaN evaluate_split bug fix (#3138), Charbonnier-default-flip (#3440),
grad-clip lever (#3418), Fourier positional encoding L=8 (#3348),
grad-clip default flip (#3494), bf16 AMP (#3330), GeGLU MLPs (#3370), **wd=1e-3 (#3630)**.

Primary validation target: **val_avg/mae_surf_p < 72.59** (NEW — PR #3630 run zmahpm3e)
Paper-facing test target: **test_avg/mae_surf_p < 66.45** (NEW — same run)
Win threshold for new PRs: val < **67.5** (≥5 units clear of noise)

Best config: `--mlp_type geglu --pos_enc_mode fourier_basic --amp_dtype bf16 --weight_decay 1e-3`

**Known operational issues — defaults that need flipping** (no idle student right now):
- `pos_enc_mode`: `"raw"` → `"fourier_basic"`
- `amp_dtype`: `"fp32"` → `"bf16"`  
- `mlp_type`: `"vanilla"` → `"geglu"`

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | Critical ping sent — no runs yet, urgent |
| #3570 | edward   | torch.compile speedup | EXTRAORDINARY: f077n973 val=49.14 — pending nocompile_ctrl + seed42 verification |
| #3600 | fern     | Fourier L sweep L=4,6,8 | L=4 done (val=84.20), L=6 + L=8_ctrl pending |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | Critical ping sent — no runs yet, urgent |
| #3667 | askeladd | OneCycleLR — retry with --epochs 15 | Sent back for corrected retry |
| #3668 | frieren  | Gradient accumulation (accum2 within noise, accum4 pending) | Sent back for accum4 arm |
| #3704 | tanjiro  | GeGLU readout (mlp2) — readout_geglu arm needed | Sent back with rebase request (CONFLICTING) |
| #3879 | nezuko   | fourier_rich pos enc (12 bands vs 8) | Newly assigned — no code change needed |
| #3881 | askeladd | slice_num sweep (64→96→128 physics tokens) | Newly assigned — 2-line code change |

Wait — askeladd has TWO PRs? #3667 (OneCycle retry) and #3881 (slice_num). Let me verify this is OK — actually, askeladd was sent back on #3667 (is now status:wip + draft) and is "idle" per the survey. The slice_num assignment #3881 is their active new assignment. But #3667 is still open WIP. This may cause confusion. I should check the survey label to confirm.

Actually the assign-experiment creates status:wip + student:askeladd label for #3881. And #3667 is also status:wip + student:askeladd. Having two WIP PRs for one student is OK — they'll pick up the most recent assignment and work through them.

## Highest pending experiments by expected impact

1. **#3570 edward torch.compile verification** — if f077n973 (val=49.14) is real, this is the biggest single win of the launch by far (~32% improvement). Critical to verify.
2. **#3151 thorfinn EMA** — expected val ~59-62 on stacked+wd=1e-3 base (−17.8% from early signal). Has NOT started yet — this is the top missing experiment.
3. **#3605 alphonse Cauchy loss** — has NOT started yet. Different loss tail could be complementary.
4. **#3881 askeladd slice_num sweep** — physics token resolution; 3 arms, 2-line code change.
5. **#3879 nezuko fourier_rich** — 12-band pos enc already implemented; 2 arms, no code change.

## Expected compose ceiling

If all three levers (GeGLU+bf16+Fourier) + wd=1e-3 + torch.compile compose:
- Current best: val=72.59, test=66.45
- If compile is real (~val 49): already at val ~49, test ~44
- If EMA adds −17.8% on top of compile: val ~40, test ~36

Most aggressive optimistic scenario, but each of these has strong independent signals.

## Potential next-round directions

**Architecture:**
- Per-channel readout heads (shared_head + α·per_channel)
- Hybrid Fourier (concat raw + Fourier)
- Learnable Fourier frequencies
- Deeper Transolver (52 GB VRAM headroom)

**Training:**
- SAM (Sharpness-Aware Minimization)
- Lookahead optimizer wrapper
- LR × wd grid at wd=1e-3

**Operational hygiene (HIGH priority — assign on next idle):**
- Triple default flip PR (pos_enc_mode/amp_dtype/mlp_type)
- wd=1e-3 Config default flip

**Context:**
- Compile result (val=49.14) requires deep investigation — result is so far below baseline it likely indicates a systematic improvement, not seed luck
