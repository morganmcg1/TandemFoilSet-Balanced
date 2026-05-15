# SENPAI Research State

- **Last updated:** 2026-05-15 ~15:30 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 109.42** — from PR #3091 (edward, warmup + clip + lr=1e-3), merged 2026-05-15. See `BASELINE.md` for full details.

**Pending merge: PR #3089 (alphonse, L1 loss) — val_avg/mae_surf_p = 102.37 (W&B-verified, −6.4%).** Re-sent back at 16:30 for **rebase + flip default + verify**:
- Alphonse's branch was forked pre-#3091, so the run that produced 102.37 used the OLD lr=5e-4 + no warmup + no clip code. Git's squash auto-merge actually composes both diffs cleanly (verified by local sim) so no revert risk — but the composed config (L1 + warmup + clip + lr=1e-3) hasn't been measured. Asked alphonse to rebase, flip `Config.loss_type` default to `"l1"`, then run a single `--epochs 10` confirmation arm for the composed-config benchmark with a W&B-logged `test_avg/mae_surf_p`.

Note: `test_avg/mae_surf_p` is NaN on all runs due to scoring bug. Alphonse's PR includes a robust fix (handles NaN+Inf via `torch.isfinite`); edward's #3288 has a simpler `nan_to_num` fix. We'll keep alphonse's fix and have #3288 only bump the lr default.

## Most recent research direction from human researcher team

No GitHub Issues open for this track. Proceeding from the program contract only.

## Cross-cutting findings (apply to ALL in-flight PRs)

1. **Cosine schedule mis-tuned:** `SENPAI_TIMEOUT_MINUTES=30` → ~14-15 epochs realized; `T_max=50` means LR barely anneals. Future PRs should pass `--epochs 10` to get proper cosine decay within budget. Confirmed by both fern (#3092) and edward (#3091).
2. **Scoring NaN:** `test_geom_camber_cruise/mae_surf_p` = NaN on all runs due to `0 * NaN = NaN` in `accumulate_batch`. Fix in PR #3288 (edward) — `y_safe = torch.nan_to_num(y, nan=0.0)` before accumulate_batch. Until fixed, compare on val_avg/mae_surf_p and 3-split test workaround.
3. **Grad norm:** Pre-clip gradient norm was 160 at lr=5e-4 in edward's Arm A. Now clipped at max_norm=1.0 (merged). Future PRs benefit automatically.
4. **Model is not converged** at the 30-min timeout — edward's Arm B best epoch was the last completed epoch (14/15). There is significant headroom at longer training or larger epoch budgets.

## In-flight experiments

| # | Student | Hypothesis | Status |
|---|---------|-----------|--------|
| #3089 | alphonse | L1 loss (val=102.37 ✓ verified) | **Sent back for 2 trivial fixes — merge candidate** |
| #3288 | edward | Scoring-bug fix + bump lr default to 1e-3 | WIP — consolidation (training observed) |
| #3092 | fern | slice_num 64 vs 128 at --epochs 10 (proper schedule) | WIP — sent back |
| #3090 | askeladd | Width: n_hidden 128→192 (+256) | WIP — recovering from rate-limit storm |
| #3093 | frieren | bf16 + batch_size 4→8 | WIP — recovering from rate-limit storm |
| #3095 | nezuko | surf_weight 10→30 + per-channel p weighting | WIP — recovering from rate-limit storm |
| #3096 | tanjiro | x-axis symmetry augmentation | WIP — recovering from rate-limit storm |
| #3097 | thorfinn | Depth: n_layers 5→8 + DropPath 0.1 | WIP — bug-fix posted, training |

## Merged wins

| PR | Description | val_avg/mae_surf_p |
|---|---|---|
| #3091 | LR warmup + clip + lr=1e-3 (edward) | **109.42** ← current baseline |

## Operational note: GitHub API rate-limit storm (resolved)

Between ~14:55 and ~15:20 UTC the GitHub API hit secondary rate limits, causing student poll cycles to fail with HTTP 403 → JSONDecodeError → "No assigned PRs" → 300s sleep without launching Claude. Then a second storm at ~16:05–16:20 UTC affected askeladd, frieren, nezuko similarly. All recovered between 16:19–16:24 UTC and have fresh Claude sessions in progress.

## Cross-cutting infrastructure issue: stale student branches

ALL 7 non-edward student branches were forked from advisor branch BEFORE #3091 (warmup + clip + lr=1e-3) merged. Their `train.py` does not have edward's changes. Git's auto-merge correctly composes both diffs (no revert), but the resulting code paths haven't been benchmarked. Before merging any of these PRs, the student should rebase onto current advisor tip and run a single confirmation arm to get clean numbers for the composed config.

Edward's #3288 has `f3a71a2` (the #3091 merge commit) in its history, so it's already on top of the new baseline.

## Next decisions (when in-flight PRs complete)

1. **Merge alphonse #3089 ASAP once student returns with the two trivial fixes** — this is the next baseline (102.37). After merge, restack: lr=1e-3 + warmup + clip + L1.
2. **Merge any experiment that beats the new baseline.** All in-flight PRs are running with mis-tuned cosine schedules (50-epoch T_max but ~15 epochs realized). If they still beat baseline at 15 epochs, strong signal. If close, request `--epochs 10` re-run before declaring improvement.
3. **Edward #3288 trim:** Once alphonse's PR merges (with the more robust scoring fix), have edward drop the duplicate scoring fix and only keep the lr default bump.
4. **Priority follow-ups (round 2):** L1 + LR sweep (alphonse suggested 3e-4, 1e-3 at different curvatures), L1 + longer schedule (training still improving at timeout), L1 + surf_weight sweep.

## Potential next research directions (round 2+)

1. **Longer training runs** — model was still improving at timeout. With proper schedule (--epochs 10 + T_max=10), the cosine tail likely holds significant gains.
2. **L1/Huber loss composition** — if alphonse wins, stack with edward's merged changes
3. **Wider model (askeladd)** — if 192 hidden wins, stack with winning loss + optimizer
4. **Symmetry aug (tanjiro)** — OOD generalization boost on geom_camber tracks
5. **EMA** — stabilize best-val checkpoint selection (especially important with mis-tuned cosine)
6. **Separate per-channel output heads** (p, Ux, Uy get distinct decoder networks)
7. **Position encoding** — Fourier features on (x, z) or unified_pos=True
8. **Physics-aware loss** — divergence-free penalty, near-surface gradient consistency
