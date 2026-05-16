# SENPAI Research State

- 2026-05-16 ~11:05 — **Round-25 complete. New baseline: val=69.98, test=62.47.**
  - **MERGED: #3600 fern Fourier L sweep** — L=4 beats L=6 and L=8 on ALL 4 test splits. val=69.98, test=62.47. Counter-intuitive but robust: lower L generalizes better OOD at this model scale/mesh density.
  - **Labels fixed**: #3879 (nezuko) and #3881 (fern) had wrong student routing; corrected.
  - **#3879 redirected**: fourier_rich (12-band) is now directionally wrong given L=4 wins. Nezuko redirected to test L=4+wd=1e-3 composition (the single most valuable run) + L=2 probe.
  - **#3881 reassigned to fern** (slice_num sweep; askeladd already has #3667 in-flight).
  - All 6 in-flight PRs updated with new baseline and `--pos_enc_num_freqs 4 --weight_decay 1e-3` config.
  - **Thorfinn (#3151) and alphonse (#3605) still have NO training runs** despite 5+ hours of pings. Critical operational concern.

- 2026-05-16 ~10:25 — **Round-24: wd=1e-3 merged, askeladd/nezuko assigned.**
- 2026-05-16 ~06:40 — **Critical defaults bug corrected** (all three levers require explicit flags).

## Current research focus

Advisor branch `icml-appendix-willow-pai2i-24h-r1`.
**Eleven items merged**: warmup+cosine (#3150), Charbonnier (#3143), NaN fix (#3138), Charbonnier-default (#3440), grad-clip lever (#3418), Fourier pos-enc L=8 (#3348), grad-clip default (#3494), bf16 AMP (#3330), GeGLU MLPs (#3370), wd=1e-3 (#3630), **Fourier L=4 (#3600)**.

Primary validation target: **val_avg/mae_surf_p < 69.98** (PR #3600 L=4, run 9nliedqj)
Paper-facing test target: **test_avg/mae_surf_p < 62.47** (same run)
Win threshold: val < **65.0** (≥5 units to clear noise)

Best config: `--pos_enc_num_freqs 4 --weight_decay 1e-3 --mlp_type geglu --pos_enc_mode fourier_basic --amp_dtype bf16`

**Known operational issues — defaults needing flip** (no idle student):
- `pos_enc_mode`: `"raw"` → `"fourier_basic"`
- `pos_enc_num_freqs`: `8` → `4`
- `amp_dtype`: `"fp32"` → `"bf16"`
- `mlp_type`: `"vanilla"` → `"geglu"`
- `weight_decay`: `1e-4` → `1e-3`

## In-flight hypotheses (9 active PRs)

| PR | Student | Lever | Status |
|----|---------|-------|--------|
| #3151 | thorfinn | EMA model weights (decay=0.99) | CRITICAL — 5+ hours no runs. Updated with L=4+wd=1e-3 commands. |
| #3570 | edward   | torch.compile speedup | f077n973 val=49.14 extraordinary — verification arms pending |
| #3605 | alphonse | Cauchy/Lorentzian loss γ ∈ {0.1, 1.0} | CRITICAL — 5+ hours no runs. Updated with L=4+wd=1e-3 commands. |
| #3667 | askeladd | OneCycleLR — epochs=15 retry | Updated with L=4+wd=1e-3 commands |
| #3668 | frieren  | Gradient accumulation accum4 | Updated with L=4+wd=1e-3 commands |
| #3704 | tanjiro  | GeGLU readout mlp2 | Sent back for rebase + updated with L=4+wd=1e-3 |
| #3879 | nezuko   | ~~fourier_rich~~ → L=4+wd=1e-3 compose + L=2 probe | Redirected — no code change needed |
| #3881 | fern     | PhysicsAttention slice_num sweep (64→96→128) | Reassigned to fern, updated with L=4+wd=1e-3 config |

## Highest-priority pending experiments

1. **#3570 edward torch.compile** — if val=49.14 is confirmed, this is the biggest win. Verification arms running.
2. **#3879 nezuko L=4+wd=1e-3 composition** — most immediate priority: confirm these two wins compose.
3. **#3151 thorfinn EMA** — expected val ~57-60 on current best stack. Has not launched.
4. **#3605 alphonse Cauchy loss** — potentially different convergence profile. Has not launched.
5. **#3881 fern slice_num** — physics token resolution sweep.

## Key insight from L=4 win

Lower Fourier frequency count generalizes better OOD. Hypothesis: higher L overfits to training-set high-frequency geometry artifacts. The model at this scale (~570K params, 5 layers) may lack capacity to benefit from 8 bands — 4 bands provide enough geometric context without adding noise.

**Follow-up experiments needed:**
- L=4 + wd=1e-3 (nezuko #3879 — immediate) 
- L=2 probe (even lower?)
- L=4 + compile + wd=1e-3 (if compile is real, compose with L=4)
- EMA + L=4 + wd=1e-3

## Potential next-round directions

**Architecture:**
- Per-channel readout heads
- Hybrid Fourier (concat raw + L=4 Fourier)
- Deeper Transolver (52 GB VRAM headroom)
- slice_num > 64

**Training:**
- SAM optimizer
- Lookahead wrapper
- LR sweep at wd=1e-3 + L=4 base

**Operational hygiene (HIGH priority — batch into single PR on next idle):**
- Default flips: pos_enc_mode/fourier_L/amp_dtype/mlp_type/weight_decay
