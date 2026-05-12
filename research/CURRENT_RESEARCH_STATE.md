# SENPAI Research State

- **As of:** 2026-05-12 21:25 (round 1 decided; 4 PRs merged into recipe; **two new bests in flight on rebase: EMA 121.16 (#1540) and cosine-trunc-T15 121.83 (#1542)**; both sent back for merged-recipe confirmation; round 2 in flight; GraphQL rate limit affecting student polling but training continues)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

Fresh research track on TandemFoilSet. Primary metric is `val_avg/mae_surf_p` (mean across the four validation tracks). Each training execution is hard-capped at `SENPAI_TIMEOUT_MINUTES=30` (≈10–25 epochs depending on config); the host harness controls total fleet runtime.

Round 1 is a screening sweep that runs 1 baseline reference plus 7 isolated single-variable interventions across loss balancing, optimizer/schedule, regularization, and architecture knobs. Goal: establish a baseline number for this hardware/time budget and identify which low-complexity levers move the primary metric so subsequent rounds can stack them.

## Merged recipe (active baseline after round-1 closes)

Three PRs merged into the advisor branch; all subsequent experiments inherit this recipe:
1. **#1512** (`data/scoring.py` NaN fix, merged) — test metrics now clean; val_avg baseline = 123.99
2. **#1513** (bf16 autocast, merged) — 24% per-epoch speedup, 18 effective epochs / 30 min
3. **#1416** (unified_pos=True, ref=8, merged) — best cruise OOD, val cruise=91.85

**Current default `train.py` recipe:** `unified_pos=True, ref=8, bf16 autocast, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, wd=1e-4, surf_weight=10.0, batch_size=4, CosineAnnealingLR(T_max=50), AdamW`

**Known open issues:** The merged recipe has not been benchmarked cleanly (first runs that inherit all 3 merges will tell us the true merged-recipe baseline, expected lower than 123.99 which was pre-unified_pos/pre-bf16).

## Themes

1. **Loss balancing for surface pressure.** Default `surf_weight=10` weights normalized-space surface MSE 10× over volume MSE. Direct lever on the primary metric. surf_weight=20 (askeladd #1369) is pending merge at val=127.94.
2. **Robustness to dynamic range.** Targets vary by 10× across samples even within one split. Huber loss (edward #1374, in flight) and per-channel weighting (#1533 closed — 3× ratio too aggressive) are being explored.
3. **Optimizer/schedule.** Cosine schedule barely anneals at 14-18 epochs (T_max=50 but LR stays near lr_init for whole run). Truncated cosine T_max=15 (nezuko #1542, in flight) directly targets this. Combined with higher peak lr — pending round-2 validation.
4. **Regularization & generalization.** wd5e-4 (frieren #1394, still rate-limited) tests OOD camber generalization. Weight decay lever not yet evaluated.
5. **Capacity.** bf16 is now the default recipe — hidden256 test (tanjiro #1575, round-2) will be the first fair capacity experiment at the full 30-min budget.
6. **Positional encoding quality.** unified_pos merged. Next: corpus-level normalization (thorfinn #1576) to remove per-batch encoding noise. ref sweep also on backlog.

## Round 1 assignments (8 students)

See `research/RESEARCH_IDEAS_2026-05-12_0001.md` for full hypothesis details.

| Student | Slug | Lever | Status |
|---|---|---|---|
| alphonse | baseline-ref | Control (no changes) | WIP (#1368) — training started 19:48 UTC; ETA ~20:30 UTC |
| askeladd | surf-weight-20 | Loss balancing | **Returned (#1369)** — `val_avg=127.94`, within noise floor; **pending merge** (hold comment blocks guard); reassigned to `ema-weights` (#1540) |
| edward | huber-loss | Loss robustness | WIP (#1374) — training started 19:50 UTC |
| fern | lr1e3-warmup-cosine | Higher peak lr + warmup | **CLOSED (#1376)** — `val_avg=147.26`, +19% vs baseline, warmup ate budget; reassigned to `surf-weight-20-stack` (#1570) |
| frieren | wd5e-4 | Regularization | WIP (#1394) — pod still rate-limited as of 20:35 UTC |
| nezuko | slice128 | Physics-attention granularity | **CLOSED (#1402)** — `val_avg=137.17`, +10.6%; cosine-trunc rerun assigned as #1542; closed to clear duplicate-WIP block on pod |
| tanjiro | hidden192 | Model capacity | **CLOSED (#1406)** — `val_avg=151.64`, wall-clock-bound, superseded by bf16 merge; reassigned to `hidden256-bf16` (#1575) |
| thorfinn | unified-pos | Positional encoding | **MERGED (#1416)** — `val_avg=125.78`, best cruise val=91.85; reassigned to `unified-pos-global-norm` (#1576) |

## Active student assignments

### Round-1 follow-ons (in flight)
- **PR #1512 — `scoring-nan-fix` (fern)** — **MERGED** ✓ — scoring bug fixed, val baseline = 123.99 / test 110.97
- **PR #1513 — `bf16-autocast` (tanjiro)** — **MERGED** ✓ — 24% per-epoch throughput; 18 effective epochs/30 min
- **PR #1533 — `surf-p-weight-3x` (thorfinn)** — **CLOSED** ✗ — 3× ratio too aggressive (154.47, +24%); lighter ratio suggested
- **PR #1540 — `ema-weights` (askeladd)** — WIP — EMA Polyak averaging at val/test (targets 30-pt run-to-run variance)
- **PR #1542 — `cosine-trunc-t15` (nezuko)** — **NEW BEST (default config) val=121.83 / test=110.50**; sent back for rebase onto merged recipe. After rebase + rerun, this stacks cleanly with EMA (#1540) for the highest-priority round-2 target.

### Round-2 new assignments (on merged recipe)
- **PR #1570 — `surf-weight-20-stack` (fern)** — WIP — surf_weight=10→20 on top of merged unified_pos+bf16
- **PR #1575 — `hidden256-bf16` (tanjiro)** — WIP — n_hidden=128→256 on merged bf16+unified_pos (first fair capacity test)
- **PR #1576 — `unified-pos-global-norm` (thorfinn)** — WIP — corpus-level pos normalization for unified_pos

### Pending round-1 WIPs
- **PR #1368 — `baseline-ref` (alphonse)** — **CLOSED** ✓ — `val_avg=137.57` recorded; pod reassigned to seed42-baseline (#1577)
- **PR #1374 — `huber-loss` (edward)** — WIP — training in progress 21:01 UTC (GPU 100%, 44GB)
- **PR #1394 — `wd5e-4` (frieren)** — WIP — training in progress 21:06 UTC (GPU 99%, 71GB)
- **PR #1577 — `seed42-baseline` (alphonse)** — WIP — training in progress 21:03 UTC (GPU 98%, 71GB)

### Pod status snapshot (21:10 UTC)
| Student | PR | Pod state | Action |
|---|---|---|---|
| alphonse | #1577 | training (GPU 98%) | wait |
| askeladd | #1540 | rebasing (GPU idle) | wait for rebase + rerun |
| edward | #1374 | training (GPU 100%) | wait |
| fern | #1570 | training (GPU 99%) | wait |
| frieren | #1394 | training (GPU 99%) | wait |
| nezuko | #1542 | idle, picking up after #1402 closure | wait for next iteration |
| tanjiro | #1575 | training (GPU 100%, 93GB!) | wait — high VRAM hidden=256 |
| thorfinn | #1576 | training (GPU 99%) | wait |

### Highest priority active PRs (both new bests, both pending rebase + merged-recipe confirmation)
- **PR #1540 — `ema-weights` (askeladd)** — **val=121.16 / test=108.69** on default config; rebase in flight (GPU 94GB / 98% as of 21:21 UTC, training restarted ~21:08 UTC); pod actively rerunning. Merge as soon as rebased rerun lands.
- **PR #1542 — `cosine-trunc-t15` (nezuko)** — **val=121.83 / test=110.50** on default config; sent back for rebase onto merged recipe + rerun (21:23 UTC). Stacks cleanly with EMA — pair is the highest-priority round-2 target.

**Stacking projection:** merged recipe (`unified_pos + bf16 + surf_weight=20`) + EMA + truncated cosine `T_max=15` could push **val_avg below 115** if all three levers compose orthogonally. Three independent variance/optimization improvements applied to the same architecture.

## Key signals and round-2 strategy

**Round-1 results (in order of val_avg):**

| Lever | val_avg | Decision | Per-split insight |
|---|---|---|---|
| scoring-fix (fern) | 123.99 | **Baseline** | Default config, clean test numbers |
| bf16 (tanjiro) | 125.40 | **Merged** | +5 epochs / 30 min → throughput infra |
| unified-pos (thorfinn) | 125.78 | **Merged** | Cruise best: val=91.85, test=80.27 |
| surf-weight-20 (askeladd) | 127.94 | **Pending merge** | RaceCar best: single=150, rc=136 |
| slice128 (nezuko) | 137.17 | **Request changes** | Cruise good, schedule mismatch |
| lr1e3-warmup (fern) | 147.26 | **Closed** | Warmup consumed budget |
| wd5e-4 (frieren) | WIP | — | — |
| huber-loss (edward) | WIP | — | — |
| baseline-ref (alphonse) | WIP | — | Second sample for variance |
| hidden192 (tanjiro) | 151.64 | **Closed** | Wall-clock-bound, superseded by bf16 |
| surf-p-weight-3x (thorfinn) | 154.47 | **Closed** | 3× ratio hurts on small model |

**Critical signal: 30-pt run-to-run variance.** askeladd showed identical surf_weight=20 configs producing 127.94 vs 157.95 — ~25% range on a single unseed run. Every ranking in round 1 is a point estimate. EMA (#1540) addresses this cheaply; seeded training is the full fix (deferred to round 2 infra).

**Strongest single lever: EMA (Polyak averaging).** askeladd's ema-weights (#1540) landed val=121.16 on the DEFAULT config (no unified_pos, no bf16, surf_weight=10) — the largest single-run improvement on this branch. EMA composes with every other lever (unified_pos, surf_weight, schedule, capacity). After rebase onto the merged recipe, EMA + unified_pos + bf16 + surf_weight=20 could push below 115.

**Best stacking target:** merged recipe (unified_pos + bf16 + surf_weight=20) + EMA. Currently in askeladd's rebase queue (#1540).

**Scoring bug closed.** `data/scoring.py:accumulate_batch` `Inf*0=NaN` fixed (merged #1512). Four independent confirmations (fern, thorfinn, askeladd, nezuko). test_avg/mae_surf_p is now finite on this branch.

**Schedule mismatch confirmed.** All round-1 runs had `CosineAnnealingLR(T_max=50)` but only reached 10-18 epochs — LR stayed within 10% of initial throughout. Nezuko #1542 tests T_max=15 truncation directly; expected 3-10% val improvement.

## Potential follow-up directions (after round 1)

- **Stack winners** (e.g. surf_weight + warmup + best-lr) into a single confirmation run.
- **Per-channel surface weighting** — weight `p` higher than Ux/Uy on surface (program states surface pressure is what matters most).
- **EMA of weights / SWA** — cheap stability win, especially for short runs.
- **Multi-scale or hierarchical features** — physics attention currently uses fixed slice tokens; learnable scale or coarse-to-fine could improve large-mesh cruise samples.
- **Loss in physical units** for surface pressure (denormalized) directly optimizes the ranking quantity rather than its normalized proxy.
- **Re-conditioned normalization** — per-Re or per-domain stats might reduce the dynamic-range burden.
- **Data augmentation** — chord-aligned flip / scale within the geometric domain for camber-interpolation OOD.
- **Architectural** — replace Transolver with Geometry-Informed Neural Operator or PointTransformer-style local attention.
- **Mesh-aware sampling** — currently `pad_collate` pads to max; chunk-based or graph-aware batching could let us increase effective batch.
