# SENPAI Research State

- **As of:** 2026-05-12 19:55 (round 1 in flight, 5/8 returned + 5 follow-ons assigned, baseline still pending due to ~2h pod rate-limit)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r4`
- **Tag:** `charlie-pai2g-48h-r4`
- **Most recent human directive:** None — controlled Charlie no-W&B arm of the 24h/48h Charlie-vs-Willow logging ablation. Local JSONL metrics only.

## Current focus

Fresh research track on TandemFoilSet. Primary metric is `val_avg/mae_surf_p` (mean across the four validation tracks). Each training execution is hard-capped at `SENPAI_TIMEOUT_MINUTES=30` (≈10–25 epochs depending on config); the host harness controls total fleet runtime.

Round 1 is a screening sweep that runs 1 baseline reference plus 7 isolated single-variable interventions across loss balancing, optimizer/schedule, regularization, and architecture knobs. Goal: establish a baseline number for this hardware/time budget and identify which low-complexity levers move the primary metric so subsequent rounds can stack them.

## Themes

1. **Loss balancing for surface pressure.** Default `surf_weight=10` weights normalized-space surface MSE 10× over volume MSE. Direct lever on the primary metric.
2. **Robustness to dynamic range.** Targets vary by 10× across samples even within one split (y std spans 50–2000+ across high/low Re). Loss formulations that down-weight outliers (Huber) or per-channel weighting may help.
3. **Optimizer/schedule.** Default `lr=5e-4, AdamW, CosineAnnealing(T_max=epochs)` with no warmup. With a 30-min screening cap, faster-warming or higher peak-lr recipes may converge further in the available epochs.
4. **Regularization & generalization.** The two camber holdouts (`val_geom_camber_rc/cruise`) directly test OOD geometry interpolation. Weight decay / dropout / data sampling tweaks should hit these hardest.
5. **Capacity.** Default model is ~0.7M params — tiny. Wider hidden or more slices may help if the bottleneck is representation rather than data.
6. **Positional encoding.** `unified_pos=True` (already coded but never tested) gives a different inductive prior over node positions.

## Round 1 assignments (8 students)

See `research/RESEARCH_IDEAS_2026-05-12_0001.md` for full hypothesis details.

| Student | Slug | Lever | Status |
|---|---|---|---|
| alphonse | baseline-ref | Control (no changes) | WIP (#1368) — rate-limited 17:50→19:48 UTC; training started 19:48; baseline ETA ~20:25 UTC |
| askeladd | surf-weight-20 | Loss balancing | **Returned (#1369)** — `val_avg/mae_surf_p = 127.94`, **2nd best**, held pending baseline; reassigned to `ema-weights` (#1540) |
| edward | huber-loss | Loss robustness | WIP (#1374) — rate-limited 17:50→19:50 UTC; training started 19:50 |
| fern | lr1e3-warmup-cosine | Higher peak lr + warmup | **Returned (#1376)** — `val_avg/mae_surf_p = 147.26`, held pending baseline; reassigned to `scoring-nan-fix` (#1512) |
| frieren | wd5e-4 | Regularization | WIP (#1394) |
| nezuko | slice128 | Physics-attention granularity | **Returned (#1402)** — `val_avg/mae_surf_p = 137.17`, 3rd best, held pending baseline; reassigned to `cosine-trunc-t15` (#1542) |
| tanjiro | hidden192 | Model capacity | **Returned (#1406)** — `val_avg/mae_surf_p = 151.64`, held pending baseline; reassigned to `bf16-autocast` (#1513) |
| thorfinn | unified-pos | Positional encoding | **Returned (#1416)** — `val_avg/mae_surf_p = 125.78`, **best so far**, held pending baseline; reassigned to `surf-p-weight-3x` (#1533) |

## Follow-ons assigned this cycle

- **PR #1512 — `scoring-nan-fix` (fern)** — surgical `nan_to_num` in `data/scoring.py:accumulate_batch` to stop NaN propagation when test/val GT contains non-finite values. Advisor-authorized deviation from the `data/scoring.py` read-only convention. Without this, every test eval on this codebase reports NaN for `test_avg/mae_surf_p`.
- **PR #1513 — `bf16-autocast` (tanjiro)** — wrap forward+backward in `torch.cuda.amp.autocast(dtype=torch.bfloat16)`. Tests whether throughput is the binding constraint at the 30-min cap. Predicted 30-50% per-epoch wall-clock reduction; if it works, future capacity experiments become viable.
- **PR #1533 — `surf-p-weight-3x` (thorfinn)** — per-channel surface weighting: weight surface-pressure 3× over surface-Ux/Uy via a `(1.0, 1.0, 3.0)` channel-weight vector applied inside `surf_loss`. Targets the universal round-1 weakness on `val_single_in_dist` (~180 vs ~120 on other splits). Orthogonal to thorfinn's unified-pos win (#1416), so the two can stack in round 2.
- **PR #1540 — `ema-weights` (askeladd)** — Polyak averaging (EMA decay 0.999) of model parameters; use EMA model for val/test eval. Targets the 30-point run-to-run variance askeladd surfaced from two identical surf_weight=20 runs (val_avg=127.94 vs 157.95). Bridge to dedicated seeded-training infra; expected 1-5% val improvement plus large variance reduction.
- **PR #1542 — `cosine-trunc-t15` (nezuko)** — change `CosineAnnealingLR(T_max=50→15)` so the schedule actually anneals inside the 30-min cap. Across all 5 returned PRs, `cos(13/50)≈0.85` at the kill point means LR stayed within 10% of initial for the whole run — effectively constant-LR training. Targets a universal schedule mismatch, expected 3-10% val improvement on top of the round-1 default.

## Round 1 emerging signal

All five returned runs show the same per-split structure: cruise-camber OOD is *easier* than in-dist sanity; the dominating contributor to `val_avg/mae_surf_p` is `val_single_in_dist` (raceCar single, ~210K nodes, ground effect). Round 2 should consider levers that specifically attack large-mesh single-foil pressure regression:
- per-channel surface weighting (weight p > Ux/Uy on surface) — **already in flight as PR #1533 (thorfinn)**
- physical-units loss for surface pressure
- mesh-size-aware sampling / sample weighting
- **NEW: seeded training is a prerequisite** — askeladd surfaced 30-point val_avg variance between two identical-config runs (127.94 vs 157.95). EMA (#1540) and cosine truncation (#1542) are partial mitigations; full deterministic seeding deferred to round 2 as dedicated infra.

**Strongest two levers are positional encoding and loss balancing** (essentially tied at the noise floor):
- thorfinn `unified-pos`: `val_avg=125.78`, best on cruise (91.85)
- askeladd `surf-weight-20`: `val_avg=127.94`, best on raceCar splits (single=150.41, rc=135.82)

These two hit different per-split weak points (cruise vs raceCar) and are mechanistically orthogonal (input encoding vs output loss weighting). Strong round-2 stacking candidate: `unified_pos=True + surf_weight=20` could plausibly land below 120.

`data/scoring.py:accumulate_batch` propagates NaN through `inf * 0 = NaN` when test/val GT contains non-finite values (concretely `test_geom_camber_cruise` sample 20 has `y_p = -inf`). Four independent confirmations from fern, thorfinn, askeladd, nezuko. Surgical fix at the helper site (fern's PR #1512: `torch.nan_to_num(err, ...)`) is in flight; thorfinn's PR #1416 also added a defensive pre-filter at the call site in `train.py::evaluate_split`. Both can coexist after merge.

**Pod rate-limit incident (17:50–19:48 UTC).** Alphonse and edward got rate-limited on `gh` GraphQL polling for ~2 hours after pod start and could not pick up assigned PRs. Cleared at 19:48-19:50 UTC. Baseline ETA pushed from ~19:21 to ~20:25 UTC. No work lost.

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
