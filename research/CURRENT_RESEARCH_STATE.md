# SENPAI Research State

- **Last updated**: 2026-05-15 ~20:30 UTC
- **Branch**: `icml-appendix-charlie-pai2i-24h-r3`
- **Target**: TandemFoilSet 2D CFD surrogate; Transolver
- **Primary metric**: `val_avg/mae_surf_p` — lower is better
- **Per-run budget**: SENPAI_MAX_EPOCHS=50, SENPAI_TIMEOUT_MINUTES=30 (hard caps)

## Current best baseline
- `val_avg/mae_surf_p` = **117.66** (PR #3237, edward, `huber-loss`, epoch 13)
- Change from default: Huber(δ=1.0) replaces MSE. All other hyperparameters at default.

## Operational issue: training completes, results not pushed

**Pattern observed in this loop (~17:30–18:25 UTC):**
- **fern (#3238)**: pod GPU showed 92–95 GB / 99–100% utilization for 44 minutes (iterations 47–53, ~17:32–18:16 UTC) — clear evidence of a full retraining run after the rebase-on-Huber sendback. But the branch remote HEAD is still `270024d` (the original pre-Huber MSE commit). No new commits pushed, no SENPAI-RESULT comment posted. Results from the 44-minute run are likely lost when the iteration-54 heartbeat reset the branch to origin.
- **thorfinn (#3303)**: pod showed full GPU utilization in the iteration around 17:33 UTC. GPU has been at 0% since ~17:57. No commits pushed; branch HEAD still at assignment commit `690b6ce`.
- **edward (#3300)**: same pattern — long Claude session (243s) completed at 17:32 UTC, then GPU idle. No commits pushed; branch HEAD still at assignment commit `c162502`.
- **tanjiro (#3241)**: rebased locally to `d30e353` after my 16:39 confirmation, but the pod restarted at iteration 1 (18:23 UTC, fresh Hivemind setup), wiping that commit. Remote HEAD is now back at assignment commit `aea79b9`.

**Diagnosis**: the student loop completes training successfully but doesn't appear to be committing+pushing results to origin before the next heartbeat reset. The harness `git reset --hard origin/<branch>` between iterations wipes local-only commits and any uncommitted JSONL artifacts. Cannot intervene from advisor side — this is a student/harness-side flow issue.

**Advisor stance**: No code intervention. The reps these students have run are largely lost; future iterations need to commit before the heartbeat resets. Keep monitoring; do NOT close PRs as dead-end based on stale_wip alone — the underlying experiments may still produce a result on the next successful iteration.

**Active training observed in prior loop (~19:30 UTC)**: thorfinn (#3393, 47 GB / 92%), fern (#3238, 45 GB), alphonse (#3177, 96 GB / 100%), frieren (#3239, 96 GB / 100%). Four students burning GPU on training runs. Branch HEADs remote remain at assignment commits — same stale-push pattern. Posted informational rebase heads-up comments on three round-1 PRs that pre-date the Huber merge: #3177 alphonse, #3239 frieren, #3240 nezuko — so they don't burn another run on MSE if/when their push flow works.

**Push-flow has recovered for thorfinn**: PR #3393 successfully posted SENPAI-RESULT at 20:24 UTC with metrics committed and pushed. This proves the system *can* work end-to-end; the other students just haven't completed a full commit+push cycle yet.

## Key observations
1. **The 30-min cap is THE bottleneck**: Every experiment so far stops at epoch 14/50 with val loss still descending. Getting more epochs per budget (BF16, smaller model, larger batch) is the highest-leverage direction.
2. **Huber loss is the proven win**: MSE → Huber gave 117.66 baseline. All round 2 experiments stack on Huber.
3. **NaN bug persists** in `test_geom_camber_cruise`: `inf` in GT of sample 20 poisons the accumulator. fern's test_avg = 113.41 is the only finite one so far — may be coincidence from prediction overflow, not a real fix.
4. **Stale_wip is now pandemic** (7 of 8 PRs): caused by the operational issue above. Even students who *complete* training (fern ran 44 min, thorfinn/edward each had complete training cycles) end up with no commits pushed. Branch remote HEADs remain at the assignment commit. Only the bookkeeping/baseline commits on the advisor branch have landed since boot.

## Active PRs

| # | Student | Slug | Status | Note |
|---|---|---|---|---|
| #3177 | alphonse | `per-sample-scale-norm` | WIP (stale) | no commits since assign |
| #3235 | askeladd | `local-re-feature` | WIP (stale) | sendback received, no rerun pushed yet |
| #3238 | fern | `dual-branch-heads` | WIP — conflict, no rebase pushed | ran 44-min training in-pod, results not pushed |
| #3239 | frieren | `fourier-pos-enc` | WIP (stale) | no commits since assign |
| #3240 | nezuko | `hflip-augment` | WIP (stale) | no commits since assign |
| #3241 | tanjiro | `ema-weights` | WIP — pod restarted, prior rebase wiped | needs to redo rebase |
| #3300 | edward | `bf16-mixed-precision` | WIP (stale) | trained but no commits pushed |
| #3303 | thorfinn | `surf-weight-50` | **CLOSED** — 3.5% regression | surf_weight=50 hurts 3/4 splits |
| #3393 | thorfinn | `surf-p-channel-weight` | WIP — sent back this loop | extra=4 was neutral (+0.28); trying extra=2 next — mechanism works (-15 on single_in_dist) but redistributes |

## Idle students
None right now. All 8 students have open WIP PRs.

## Human research direction
None received yet.

## Current research themes

**Budget efficiency** (edward #3300):
- BF16 to unlock more epochs within 30-min cap

**Loss formulation** (thorfinn #3393, alphonse #3177):
- per-channel surface pressure weighting (surf_p_weight_extra=4, +5× on dim 2 only)
- per-sample-scale-norm + Huber: balance Re-regime gradient magnitudes
- (surf_weight=50 closed — uniform scaling hurts 3/4 splits)

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
