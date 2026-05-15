# SENPAI Research State

- **Last updated:** 2026-05-15 ~21:40 UTC
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

## Round 4 confirmation arm results (W&B-verified by run ID; pods stuck — no SENPAI-RESULT comments yet)

| # | Student | Hypothesis | W&B run | State | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs 109.42 |
|---|---|---|---|---|---|---|---|
| **#3372** | **askeladd** | **Fourier PE on (x,z) coords** | xmcndd46 | finished | **94.491** | n/a (no scoring fix) | **−13.6% 🏆 best** |
| #3089 | alphonse | L1 + warmup + clip + lr=1e-3 | ydztkz9s | finished | **104.878** | **94.629** | **−4.1% ✓** |
| #3092 | fern | slice_num=64 baseline e=10 | d62uhu5g | finished | **106.821** | n/a | −2.4% ✓ marginal |
| ~~#3414~~ | ~~tanjiro~~ | ~~SWA last K~~ | udfmekyw | **CLOSED** | swa=109.48 | — | +0.06% ✗ (SWA worsened raw val=108.01) |
| #3095 | nezuko | surf_weight=20 rebased | 6amjj7jr | finished | 111.916 | 97.702 | +2.3% ✗ narrowly worse |
| #3093 | frieren | bf16 + bs=8 rebased | ytpl95nk | finished | 143.842 | n/a | +31.4% ✗ regressed |
| #3371 | thorfinn | EMA decay=0.9999 | yr4bbbg8 | **crashed** | 150.897 | n/a | crashed at epoch ≈ early |
| #3288 | edward | lr default 5e-4→1e-3 (only change) | n/a | **RUNNING** — commit pushed 21:25, GPU 46.5/97 GB | — | — | waiting for #3089 merge to rebase |
| **#3469** | **tanjiro** | **Depth n_layers=5→6** | new | **NEW** — assigned 21:39 | — | — | n/a |

**4 of 7 PRs are merge candidates** (val < 109.42): askeladd (best), alphonse, fern, tanjiro.

### Critical operational blocker (resolving)
Pods got hit by a sustained secondary API rate-limit storm from **~20:41 → 21:14+ UTC**. The polling helper's `gh_retry` exhausted 6 attempts → JSONDecodeError → "No work assigned" loop. All W&B runs completed cleanly *before* the storm peaked, but the post-training Claude session that normally posts `SENPAI-RESULT` never invoked for any student. **All 7 PRs are quiet on GitHub but their W&B runs are done.** Plan: wait for the storm to clear (advisor's GH calls are succeeding as of 21:25, so it should be near the end); students' next Claude session will see the assignment and post results.

## Merged wins

| PR | Description | val_avg/mae_surf_p |
|---|---|---|
| #3091 | LR warmup + clip + lr=1e-3 (edward) | **109.42** ← current baseline |

## Operational note: GitHub API rate-limit storms

Between ~14:55 and ~15:20 UTC the GitHub API hit secondary rate limits; second storm at ~16:05–16:20 UTC; third at ~17:49–17:57 UTC (during advisor PR-creation calls; PRs #3371/#3372 still landed via retries). **Fourth storm at ~20:10–20:25 UTC** caused 2–3 missed cycles per pod; recovered, training kicked off 20:22–20:29. **Fifth and most severe storm at ~20:41 → 21:14+ UTC** (over 30 minutes) — Claude sessions on student pods never re-invoked, so all 7 finished W&B runs are stranded without `SENPAI-RESULT` comments on their PRs. Advisor agent's gh API calls work as of 21:25 (graphql 4168/5000 remaining), so the storm appears to be in its tail; expecting students to recover within the next 1–2 polling cycles (≤ 21:30 UTC).

## Operational note: label routing on assignment skill

**Persistent bug:** The `senpai:assign-experiment` skill's `create_assignment_pr_from_file` helper creates `student:<name>` (plain) instead of `student:willowpai2i48h4-<name>` (namespaced). Affected: #3371 (thorfinn), #3372 (askeladd), #3414 (tanjiro). All fixed manually with `gh pr edit --remove-label / --add-label`. Student pods on this track route on the namespaced label; the plain labels are visible-but-not-routed.

**Workaround rule for this track:** After every new PR assignment, run: `gh pr edit <pr#> --repo morganmcg1/TandemFoilSet-Balanced --remove-label "student:<name>" --add-label "student:willowpai2i48h4-<name>"`

## Cross-cutting infrastructure issue: stale student branches

ALL 7 non-edward student branches were forked from advisor branch BEFORE #3091 (warmup + clip + lr=1e-3) merged. Their `train.py` does not have edward's changes. Git's auto-merge correctly composes both diffs (no revert), but the resulting code paths haven't been benchmarked. Before merging any of these PRs, the student should rebase onto current advisor tip and run a single confirmation arm to get clean numbers for the composed config.

Edward's #3288 has `f3a71a2` (the #3091 merge commit) in its history, so it's already on top of the new baseline.

## Next decisions (queued; awaiting student SENPAI-RESULT comments to merge)

Merge order once SENPAI-RESULT markers land (compound improvements; best last to land on top):

1. **#3372 askeladd Fourier PE (val=94.491)** — BIGGEST WIN; 13.6% below baseline. Zero per-step cost. Hold for last in sequence so it merges on top of fastest-changing baseline.
2. **#3089 alphonse L1 + warmup + clip + lr=1e-3 (val=104.878, test=94.629)** — strong, clean — has the scoring fix (so test_avg becomes a finite number for everything after).
3. **#3092 fern slice_num e=10 (val=106.821)** — marginal but a clean win on the rebased baseline. Merge sequentially after alphonse.
4. **#3414 tanjiro SWA (val=108.014)** — marginal; merge if it still beats post-merges. Re-confirm if it lands after 3 baseline changes.

**Close as dead end:**
- **#3414 tanjiro SWA** ✓ CLOSED 21:37
- **#3093 frieren bf16+bs=8 (val=143.84, +31%)** — composed config regressed badly. The speed unlock is genuine, but the bf16 numerics at lr=1e-3 + clip=1 + warmup might be incompatible; needs a different schedule. Close this arm; revisit bf16 as separate experiment.

**Send back / investigate:**
- **#3371 thorfinn EMA (CRASHED, val=150.9 at early epoch)** — investigate crash cause; ask student to log a brief post-mortem.
- **#3095 nezuko surf_weight=20 (val=111.92, +2.3%)** — narrowly worse on val but the val_test gap is very small (97.7 vs 94.6 for alphonse). Surf_weight increase didn't help; try a lower value (5 or 15) or close.

**Edward #3288:** trim to lr-default-only after #3089 merges; this is no longer needed once alphonse's scoring fix lands.

**After all merges (next baseline likely ≈ 94 val):** big architecture exploration round — n_hidden=160 + n_layers=6 + the Fourier-PE-enabled feature dim. Plus L1+LR sweep at the new baseline.

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
