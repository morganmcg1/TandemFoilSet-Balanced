# SENPAI Research State

- **As of:** 2026-05-12 18:35 (round 1 in flight, 1/8 returned)
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
| alphonse | baseline-ref | Control (no changes) | WIP (#1368) |
| askeladd | surf-weight-20 | Loss balancing | WIP (#1369) |
| edward | huber-loss | Loss robustness | WIP (#1374) |
| fern | lr1e3-warmup-cosine | Higher peak lr + warmup | **Returned (#1376)** — `val_avg/mae_surf_p = 147.26`, held pending baseline |
| frieren | wd5e-4 | Regularization | WIP (#1394) |
| nezuko | slice128 | Physics-attention granularity | WIP (#1402) |
| tanjiro | hidden192 | Model capacity | WIP (#1406) |
| thorfinn | unified-pos | Positional encoding | WIP (#1416) |

## Follow-on: scoring bug fix — assigned to fern as PR #1512

`data/scoring.py:accumulate_batch` propagates NaN through `inf * 0 = NaN` when test/val GT contains non-finite values (concretely `test_geom_camber_cruise` sample 20 has `y_p = -inf`). Every test eval on this codebase reports NaN for `test_avg/mae_surf_p`. Surgical one-line fix: `torch.nan_to_num(err, ...)` after computing `err`. Advisor-authorized deviation from the `data/scoring.py` read-only convention.

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
