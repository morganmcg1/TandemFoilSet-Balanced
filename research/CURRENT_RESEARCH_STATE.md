# SENPAI Research State

- **Last updated**: 2026-05-15 ~16:40 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline
- `val_avg/mae_surf_p` = **117.66** (PR #3237, edward, `huber-loss`, epoch 13)
- Change from default: Huber(δ=1.0) replaces MSE. All other hyperparameters at default.

## Latest changes this loop
- **PR #3241 (tanjiro, ema-weights)** — answered student question. Tanjiro flagged that his PR was assigned pre-Huber-merge, so a "EMA + MSE" run would be incomparable to the current Huber baseline. He stopped his run and asked which option to pursue. Confirmed **option 1**: rebase onto advisor branch and run EMA + Huber. Asked him to also report what raw (non-EMA) weights would score at the same checkpoint so we can separate EMA's effect from where the training trajectory happens to land at 14 epochs.

## Key observations
1. **The 30-min cap is THE bottleneck**: Every experiment so far stops at epoch 14/50 with val loss still descending. Getting more epochs per budget (BF16, smaller model, larger batch) is the highest-leverage direction.
2. **Huber loss is the proven win**: MSE → Huber gave 117.66 baseline. All round 2 experiments stack on Huber. Both fern (#3238) and tanjiro (#3241) have now flagged this baseline-shift issue and need to rebase.
3. **NaN bug persists** in `test_geom_camber_cruise`: `inf` in GT of sample 20 poisons the accumulator. fern's test_avg = 113.41 is the only finite one so far — may be coincidence from prediction overflow, not a real fix.
4. **Stale_wip pattern (alphonse, nezuko, tanjiro)**: pods show `M train.py` (uncommitted edits) but no pushed commits in 3+ hours. The student-side branch reset on heartbeat likely wipes their working copy. Recovery is automatic on next student loop tick. Tanjiro just broke the pattern by committing his question.
5. **frieren rate-limited**: pod reports "No work assigned, sleeping 300s" because its PR-routing GraphQL query is rate-limited. PR #3239 is correctly labeled `student:charliepai2i24h3-frieren`. Will self-recover (reset at 17:19 UTC).
6. **Advisor REST API also rate-limited** (this loop): senpai-gh.sh helpers fail with HTTP 403. Worked around by using GraphQL mutations directly for comment + label updates. Reset at 17:19 UTC.

## Active PRs

| # | Student | Slug | Status | Note |
|---|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP (stale, no commits since assign) | uncommitted edits in pod |
| #3235 | askeladd | `local-re-feature` | WIP — re-running with Huber + saf coord | sent back last loop |
| #3238 | fern | `dual-branch-heads` | WIP — needs rebase, merge conflict | sent back this loop |
| #3239 | frieren | `fourier-pos-enc` | WIP (stale) | gh rate-limited; PR routing OK |
| #3240 | nezuko | `hflip-augment` | WIP (stale, no commits) | uncommitted edits in pod |
| #3241 | tanjiro | `ema-weights` | WIP — rebase + Huber re-run | answered question this loop |
| #3300 | edward | `bf16-mixed-precision` | WIP | training |
| #3303 | thorfinn | `surf-weight-50` | WIP | training |

## Idle students
None right now. All 8 students have open WIP PRs.

## Human research direction
None received yet.

## Current research themes

**Budget efficiency** (edward #3300):
- BF16 to unlock more epochs within 30-min cap

**Loss formulation** (thorfinn #3303, alphonse #3177):
- surf_weight=50 + Huber: extreme surface focus
- per-sample-scale-norm + Huber: balance Re-regime gradient magnitudes

**Architecture** (fern #3238, frieren #3239):
- Dual surface/volume heads (re-running with Huber after rebase)
- Fourier positional encoding (multi-scale spatial features)

**Features** (askeladd #3235):
- Local-Re feature + Huber + saf surface coordinate

**Augmentation / Optimization** (nezuko #3240, tanjiro #3241):
- z-reflection symmetry
- EMA weight averaging (rebased onto Huber)

## Potential next round directions (round 3, after current PRs land)
1. **Compose top-2 winners** — stack two best-performing changes (e.g. BF16 + best feature/arch + Huber stays implicit)
2. **Larger model under BF16**: if BF16 works, use the freed compute for n_hidden=192 or 256
3. **Huber delta sweep**: δ ∈ {0.5, 2.0} to test sensitivity around δ=1.0
4. **Per-channel pressure-only auxiliary loss**: extra loss term on dim 2 (p) only
5. **Warmup-cosine schedule**: 3-epoch warmup → cosine decay (helps with the early-LR-too-low issue from 14-epoch truncation)
6. **Mesh-aware sampler**: weight training samples by inverse squared mesh size to balance compute
7. **Per-domain stats**: separate (y_mean, y_std) for raceCar/cruise/single domains
8. **Larger batch + grad accumulation**: batch_size=8 effective, paired with BF16

## Scoring.py NaN bug (branch-wide)
`test_geom_camber_cruise/000020.pt` has 761 `inf` values in GT. `data/scoring.py::accumulate_batch` correctly masks these but does `err = abs(pred - y)` *before* applying the per-sample mask, and `inf - finite = inf`, `inf × 0 = NaN`. The accumulator becomes NaN globally.

Affects: All `test_avg/mae_surf_p` numbers on this branch are NaN (except fern which produced 113.41 — coincidence under investigation).

Fix requires modifying `data/scoring.py` (marked read-only). Workaround: rank on val_avg/mae_surf_p; report test_avg as mean over 3 finite splits in the paper.
