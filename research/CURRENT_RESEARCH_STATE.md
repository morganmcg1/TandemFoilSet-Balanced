# SENPAI Research State

- **Last updated:** 2026-05-15 ~20:32 UTC
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

## In-flight experiments — all 7 actively training as of 20:31 UTC

| # | Student | Hypothesis | Status / W&B run |
|---|---------|-----------|--------|
| #3089 | alphonse | L1 + warmup + clip + lr=1e-3 confirmation arm (target: val < 109.42) | **TRAINING** — `alphonse-l1-rebased` (ydztkz9s) started 20:22 |
| #3414 | tanjiro | SWA: stochastic weight averaging over last K checkpoints | **TRAINING** — `…/swa-la…` (udfmekyw) started 20:26 |
| #3092 | fern | slice_num 64 vs 128 at --epochs 10 (proper schedule) | **TRAINING** — `fern-slices-64-baseline-e10` (d62uhu5g) started 20:28 (restart after API storm) |
| #3288 | edward | Scoring-bug fix + bump lr default to 1e-3 | Correctly idle — waiting for #3089 to merge; scope reduced to lr-default only |
| #3093 | frieren | bf16 + bs=8 rebased confirmation arm | **TRAINING** — `frieren-bf16-bs8-rebased` (ytpl95nk) started 20:29 |
| #3095 | nezuko | surf_weight 10→20 rebased confirmation arm | **TRAINING** — `nezuko-surf20-rebased` (6amjj7jr) started 20:23 |
| #3371 | thorfinn | EMA of weights (Polyak averaging decay=0.9999) | **TRAINING** — `thorfinn-ema-999` (yr4bbbg8) started 20:28 |
| #3372 | askeladd | Fourier positional encoding on (x,z) coords | **TRAINING** — `askeladd-fourier-pe-4freq-lr1e` (xmcndd46) started 20:28 |

## Merged wins

| PR | Description | val_avg/mae_surf_p |
|---|---|---|
| #3091 | LR warmup + clip + lr=1e-3 (edward) | **109.42** ← current baseline |

## Operational note: GitHub API rate-limit storms

Between ~14:55 and ~15:20 UTC the GitHub API hit secondary rate limits; second storm at ~16:05–16:20 UTC; third at ~17:49–17:57 UTC (during advisor PR-creation calls; PRs #3371/#3372 still landed via retries). **Fourth storm at ~20:10–20:25 UTC** — all 8 student pods saw "No assigned PRs" for 2–3 cycles because their assignment-polling helper failed with `JSONDecodeError` on the 403 response. Recovered by 20:23; all 7 active students kicked off training in the post-storm cycle (20:22–20:29).

## Operational note: label routing on assignment skill

**Persistent bug:** The `senpai:assign-experiment` skill's `create_assignment_pr_from_file` helper creates `student:<name>` (plain) instead of `student:willowpai2i48h4-<name>` (namespaced). Affected: #3371 (thorfinn), #3372 (askeladd), #3414 (tanjiro). All fixed manually with `gh pr edit --remove-label / --add-label`. Student pods on this track route on the namespaced label; the plain labels are visible-but-not-routed.

**Workaround rule for this track:** After every new PR assignment, run: `gh pr edit <pr#> --repo morganmcg1/TandemFoilSet-Balanced --remove-label "student:<name>" --add-label "student:willowpai2i48h4-<name>"`

## Cross-cutting infrastructure issue: stale student branches

ALL 7 non-edward student branches were forked from advisor branch BEFORE #3091 (warmup + clip + lr=1e-3) merged. Their `train.py` does not have edward's changes. Git's auto-merge correctly composes both diffs (no revert), but the resulting code paths haven't been benchmarked. Before merging any of these PRs, the student should rebase onto current advisor tip and run a single confirmation arm to get clean numbers for the composed config.

Edward's #3288 has `f3a71a2` (the #3091 merge commit) in its history, so it's already on top of the new baseline.

## Next decisions (when in-flight PRs complete)

1. **Merge alphonse #3089 ASAP once student returns** — next baseline (102.37). After merge: lr=1e-3 + warmup + clip + L1. Edward #3288 trims to lr-default-only after alphonse merges.
2. **Frieren #3093 rebased confirmation arm** — bf16+bs=8 speed unlock; if val < 109.42 on rebased code → merge immediately. This enables more epoch-budget for architecture experiments.
3. **Tanjiro #3096 rebased arm** — if val < 109.42 → merge; val 109-115 → conditional; val > 115 → close.
4. **Fern #3092 slice_num result** — waiting for rebased --epochs 10 run.
5. If frieren's bf16 lands: revisit depth (n_layers=7, not 8) and width (n_hidden=160, not 192) with the speed budget.

## Cross-cutting observations from round 2 (depth + width + bf16)

- **Every architecture change tried so far (depth, width) is runtime-budget-bound**, not capacity-bound. The model's biggest val drops happen at epochs 10–14; slower models can't reach that regime in 30 min.
- **bf16+bs=8 is the highest-leverage change in-flight** because it unlocks more epochs for all future experiments.
- **Depth at 8L and width at 192/256 are NOT dead ends** — they failed because they couldn't complete enough epochs. If frieren's bf16 win materializes, retry n_layers=6 and n_hidden=160 (both within ~20% slower than baseline) as the next architectural steps.

## Potential next research directions (round 3+)

1. **L1 + LR sweep** — alphonse suggested 3e-4 and 1e-3 at different curvatures. After L1 merges.
2. **Fourier PE (askeladd #3372)** — high-freq geometry; zero per-step cost.
3. **EMA (thorfinn #3371)** — stabilize checkpoint selection; zero per-step cost.
4. **Symmetry aug (tanjiro) stacked with L1** — the 47% regression was likely code+schedule; L1 may compose better with aug.
5. **Wider model at smaller step (n_hidden=160, n_head=5)** — after bf16 unlock.
6. **Depth at n_layers=6** — adds only ~20% per-step cost vs baseline. Within budget.
7. **Separate per-channel output heads** — p, Ux, Uy decoupled decoders.
8. **Physics-aware loss** — divergence-free penalty, near-surface gradient consistency.
