# SENPAI Research State

- 2026-05-15 21:30 — **Round-7 review complete**: closed #3142 (askeladd surf_weight,
  weak 0.69% within-PR signal on stale base), sent back #3151 (thorfinn EMA, strong
  −17.8% within-PR signal, needs rebase on Charbonnier base), assigned #3457
  (askeladd, peak LR sweep on Charbonnier baseline: 5e-4 vs 1e-3 vs 2e-3).
  Rate limits cleared — tanjiro (#3370), fern (#3348), frieren (#3330) all back on
  their WIP assignments.
- No directives from the human researcher team on this launch.

## Current research focus

Round 7+ on advisor branch `icml-appendix-willow-pai2i-24h-r1`. **Three items
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
- **EMA is a likely major winner on rebase.** PR #3151 showed −17.8% test within-PR
  on stale base. Signal is large enough to be real. Sent back for Charbonnier-base rebase.
- **AMP (bf16) is a proven lever, rebasing.** PR #3330 showed 1.34× epoch speedup
  → −13% within-PR. Frieren back on assignment.
- **Separate output heads is dead** (#3331 closed). Shared trunk is NS cross-field
  inductive bias. Residual heads still worth trying in a later round.
- **Surf_weight sweep is a weak lever** (#3142 closed). 0.69% gain within-PR, non-monotonic
  at sw30, stale base. May revisit post-Charbonnier-AMP compose with fresh re-tuning.
- **30-min wall-clock cap binds at ~14/50 epochs** for baseline width. AMP gives 1.34×
  more epochs; peak LR tune (now underway) may also help reach deeper cosine decay.
- **NaN bug fixed.** `test_avg/mae_surf_p` now finite for all future runs.
- **Config default loss_fn="mse" is a live issue.** PR #3440 (alphonse) will fix.
  Until then, every reproduce command needs `--loss_fn charbonnier --charbonnier_eps 1e-3`.
- **Run-to-run variance ~3-4 mae_surf_p units.** Need ≥5 unit improvement for clear attribution.

## In-flight hypotheses (8 active PRs)

| PR | Student | Lever | Notes |
|----|---------|-------|-------|
| #3151 | thorfinn  | EMA model weights (decay 0, 0.999, 0.9999) — rebasing | sent back; rebase on Charbonnier base, re-run with --loss_fn charbonnier |
| #3330 | frieren   | bf16 AMP — rebasing | student back online 21:21 UTC, starting rebase |
| #3348 | fern      | Fourier position encoding (multi-scale spatial frequency basis) | student back online 21:24 UTC; results pending |
| #3370 | tanjiro   | Gated MLPs (SwiGLU / GeGLU in TransolverBlocks) | student back online 21:22 UTC; results pending |
| #3398 | edward    | Charbonnier ε sweep ({3e-4, 1e-3, 3e-3}) | wip; needs --loss_fn charbonnier flag (notified) |
| #3418 | nezuko    | Gradient clipping sweep (`max_norm` ∈ {0, 0.5, 1.0}) | wip; needs --loss_fn charbonnier flag (notified) |
| #3440 | alphonse  | Fix Config default: loss_fn="charbonnier" (was "mse") | wip (critical fix) |
| #3457 | askeladd  | Peak LR sweep on Charbonnier base (5e-4 vs 1e-3 vs 2e-3) | new assignment |

## Closed / merged this launch

| PR | Student | Result |
|----|---------|--------|
| #3148 | frieren | Wider Transolver — wall-clock confound; closed |
| #3149 | nezuko  | Per-channel surf-p loss weighting — shared-backbone capacity steal; closed |
| #3145 | fern    | Deeper Transolver — wall-clock confound; closed |
| #3150 | tanjiro | **Warmup + cosine LR — merged ⭐ (val_avg/mae_surf_p 125.83)** |
| #3143 | edward  | **Charbonnier robust loss — merged ⭐⭐ (val_avg/mae_surf_p 98.60, −18.6%)** |
| #3138 | alphonse | **NaN bug fix in evaluate_split — merged ✅ (test_avg/mae_surf_p = 92.71, first finite)** |
| #3331 | nezuko  | Separate per-channel heads — closed ✗ (shared trunk is NS inductive bias) |
| #3330 | frieren | bf16 AMP — sent back for rebase (−13% within-PR, 1.34× speedup) |
| #3151 | thorfinn | EMA weights — sent back for rebase (−17.8% test within-PR, stale base) |
| #3142 | askeladd | Surf weight sweep (10/30/80) — closed ✗ (0.69% gain, non-monotonic, stale base) |

## Strategic outlook

**Most expected high-value paths remaining:**
1. **EMA rebase (#3151)** — within-PR −17.8% test, expected ~80-90 range on composed base
2. **AMP rebase (#3330)** — proven 13% within-PR; compose should put bf16_bs4 at ~85-92 val
3. **Charbonnier ε (#3398)** — tuning ε could give 2-5% further
4. **GLU MLPs (#3370)** — strongest remaining architecture lever; results pending
5. **Fourier pos enc (#3348)** — within-PR signal ~3% on stale base; results pending
6. **Peak LR (#3457)** — new; Charbonnier's bounded gradient may allow higher peak LR

**Key question**: do EMA + AMP compose cleanly? Both speed effective convergence via
orthogonal mechanisms (EMA = better solution averaging, AMP = more steps/min). If both
merge, compose should push val into mid-70s or below.

## Potential next-round directions (when current 8 PRs close)

**Loss-form:**
- Cauchy/Lorentzian loss (more aggressive outlier suppression than Charbonnier)
- Per-channel ε (surf_p may want different ε from Ux/Uy)
- Adaptive surf_weight tuning via GradNorm or uncertainty weighting

**Architecture:**
- Residual heads — `shared_head(z) + α·per_channel_correction(z)` (nezuko's follow-up)
- RMSNorm vs LayerNorm — slightly faster, often comparable
- Per-layer slice_num schedule (coarse-to-fine)
- torch.compile (1.3-2× speedup, orthogonal to all levers)

**Training:**
- Per-step warmup (vs per-epoch) — standard large-scale recipe
- Larger batch with AMP — retest batch=8 with scaled LR once AMP wins
- Gradient accumulation for effective larger batch on single GPU

**Data:**
- Data augmentation — random AoA jitter, Re jitter
- Curriculum learning (single-foil first, then tandem)
- Hard example mining / importance sampling
