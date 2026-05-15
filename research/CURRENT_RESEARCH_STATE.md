# SENPAI Research State

- 2026-05-15 20:50 — **Third winner merged**: PR #3138 (alphonse NaN bug fix in
  `evaluate_split`). Critical fix — unblocks `test_avg/mae_surf_p` for every
  future run. First finite test metric this launch: **test_avg/mae_surf_p = 92.71**.
  Assigned #3440 (alphonse, flip Config default loss_fn to charbonnier). Notified
  edward (#3398), nezuko (#3418), frieren (#3330) to pass `--loss_fn charbonnier`
  explicitly until #3440 lands.
- No directives from the human researcher team on this launch.

## Current research focus

Round 5+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`. **Three items
merged**: warmup+cosine (#3150), Charbonnier robust loss (#3143), NaN evaluate_split
bug fix (#3138). New primary validation target: **val_avg/mae_surf_p < 98** (best
single-seed 98.60 from PR #3143). First valid paper-facing test metric:
**test_avg/mae_surf_p = 92.71** (compose sanity, u2k87wan). 8 hypotheses in flight.

The merged config now includes:
- `SequentialLR(LinearLR warmup → CosineAnnealingLR)`, `warmup_epochs=3`, `eta_min=1e-6` (PR #3150)
- Charbonnier loss `sqrt(diff² + ε²)`, `ε=1e-3` (PR #3143) — **⚠️ default still "mse" pending PR #3440**
- NaN filter in `evaluate_split` (PR #3138)

## Key constraints / insights confirmed

- **Charbonnier robust loss is the dominant single lever.** −18.6% on val_avg/mae_surf_p
  vs MSE. Linearizes gradient for large residuals; surface pressure has order-of-magnitude
  dynamic range. Improves every channel, every split.
- **AMP (bf16) is a proven lever, rebasing needed.** PR #3330 showed 1.34× epoch
  speedup → −13% within-PR. Expected compose to land bf16_bs4 in ~85-95 range.
- **Separate output heads is dead** (#3331 closed). Shared trunk is NS cross-field
  inductive bias. Residual heads still worth trying in a later round.
- **30-min wall-clock cap binds at 50 epochs** for baseline width. All forks from
  pre-warmup, pre-Charbonnier base need rebase + re-run.
- **NaN bug fixed.** `test_avg/mae_surf_p` now finite for all future runs. Previous
  cruise test Inf `* 0.0 == NaN` mask-poisoning is resolved in-code.
- **Config default loss_fn="mse" is a live issue.** PR #3440 (alphonse) will fix.
  Until then, every reproduce command needs `--loss_fn charbonnier --charbonnier_eps 1e-3`.
- **Run-to-run variance ~3-4 mae_surf_p units.** Need ≥5 unit improvement for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Notes |
|----|---------|-------|-------|
| #3142 | askeladd  | Surface-loss weight (`surf_weight` ∈ {10,30,80}) | wip; training done per W&B (sw10 best); rate-limited from posting |
| #3151 | thorfinn  | EMA model weights (`ema_decay` ∈ {0, 0.999, 0.9999}) | wip; training done (ema999 best); rate-limited from posting |
| #3330 | frieren   | bf16 AMP (sent back for rebase on new baseline) | wip; needs rebase + re-run with --loss_fn charbonnier |
| #3348 | fern      | Fourier position encoding (multi-scale spatial frequency basis) | wip; training done per W&B (fourier_L12 best); rate-limited from posting |
| #3370 | tanjiro   | Gated MLPs (SwiGLU / GeGLU in TransolverBlocks) | wip; swiglu arm training ~20:24 UTC |
| #3398 | edward    | Charbonnier ε sweep ({3e-4, 1e-3, 3e-3}) | wip; needs --loss_fn charbonnier flag (notified) |
| #3418 | nezuko    | Gradient clipping sweep (`max_norm` ∈ {0, 0.5, 1.0}) | wip; needs --loss_fn charbonnier flag (notified) |
| #3440 | alphonse  | Fix Config default: loss_fn="charbonnier" (was "mse") | wip (new critical fix) |

### Stale-WIP status (as of 20:50 UTC)
- **#3142 askeladd**: All arms done. W&B shows sw10 (mae~131-134) beats sw30/sw80. Likely close when results post.
- **#3151 thorfinn**: All arms done. W&B shows ema999 (mae~119) beating baseline (126). Worth inspecting — within-PR win on pre-Charb base. If strong signal, send back for rebase.
- **#3348 fern**: All arms done. W&B shows fourier_L12 (mae~122) beating warmup-only baseline (126). Within-PR win on pre-Charb base. Send back for rebase.
- **#3370 tanjiro**: Still training swiglu arm as of 20:24. Will finish soon.

## Closed / merged this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; closed |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal; closed |
| #3145 | fern    | Deeper Transolver — same wall-clock confound as #3148; closed |
| #3150 | tanjiro | **Warmup + cosine LR — merged ⭐ (val_avg/mae_surf_p 125.83)** |
| #3143 | edward  | **Charbonnier robust loss — merged ⭐⭐ (val_avg/mae_surf_p 98.60, −18.6%)** |
| #3138 | alphonse | **NaN bug fix in evaluate_split — merged ✅ (test_avg/mae_surf_p = 92.71, first finite)** |
| #3331 | nezuko  | Separate per-channel heads — closed ✗ (shared trunk is NS inductive bias) |

## Strategic outlook

The three merged PRs represent two orthogonal dimensions:
1. **Optimization/loss quality** — warmup + Charbonnier (fast stable convergence + robust loss)
2. **Evaluation correctness** — NaN fix (paper-facing metric now valid)

With the test metric now real, we have a new optimization target: drive **test_avg/mae_surf_p** below 90 (current: 92.71 from compose sanity). The val floor from existing in-flight PRs composing should land this in the 80-90 range if AMP + GLU MLPs both win.

Most expected high-value paths:
1. **AMP rebase (#3330)** — proven 13% within-PR; compose should put bf16_bs4 at ~85-92 val, test likely in the low 80s
2. **Charbonnier ε (#3398)** — tuning ε could give 2-5% further
3. **GLU MLPs (#3370)** — strongest remaining architecture lever
4. **Fourier pos enc (#3348)** — if within-PR signal survives rebase

## Potential next-round directions (when current 8 PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive outlier suppression than Charbonnier)
- Per-channel ε (surf_p may want different ε from Ux/Uy)
- Re-tune surf_weight after Charbonnier gradient rebalancing

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)` (nezuko's follow-up)
- RMSNorm vs LayerNorm — slightly faster, often comparable
- Per-layer slice_num schedule (coarse-to-fine)

**Training:**
- Per-step warmup (tanjiro's #3 follow-up) — may unlock higher peak LRs
- Larger batch with AMP — retest batch=8 with scaled LR once AMP wins
- Padding-aware batching (frieren's #4 follow-up)

**Data:**
- Data augmentation — chord rotation + NACA-symmetric flip for raceCar
- Curriculum learning, test-time augmentation
