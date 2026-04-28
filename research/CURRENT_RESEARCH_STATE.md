# SENPAI Research State

- **Date:** 2026-04-28 01:22
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet
- **Empirical baseline (round 1):** `val_avg/mae_surf_p = 139.83` from PR #336 (slice_num=128). All future runs compound on top of this.
- **Cross-cutting bug being fixed:** `data/scoring.py:accumulate_batch` propagates `NaN` through the per-sample-skip mask (`NaN * 0.0 = NaN`, plus `0 * inf = NaN` per alphonse's independent diagnosis). Root cause is 761 non-finite values in the `p` channel of `test_geom_camber_cruise/000020.pt`'s ground truth `y`. Fix in flight as PR #375 (edward) — advisor-authorized exception to the read-only contract on `data/`.
- **Slice_num re-evaluation — strong evidence accumulating:** Frieren's PR #338 close gave us a direct apples-to-apples (slice=64 vs slice=128, same warmup, same lr): **slice_num=64 wins by 13.5 MAE / 9.7%**. Combined with the cluster of slice_num=64 single-seed results at 130-132 vs the only slice_num=128 result at 139.83, this is now the leading hypothesis: PR #336 was a partial-credit merge inside the 30-min cap. Two pending data points to resolve: (a) alphonse's #329 rebased re-run of `surf_weight=50` on slice_num=128, (b) thorfinn's #428 multi-seed baseline calibration. If both confirm slice_num=64 dominates, we open a revert PR for #336 in the next iteration.
- **Seed variance (NEW from #331 close):** measured at **±10-15% on `val_avg/mae_surf_p` at 12 epochs** (askeladd's v1=141.998 vs v2=163.280 same config). Many round-1 apparent wins on single seeds are inside this noise band. Going forward, ask winning candidates for a 2-seed confirmation before merge.
- **bf16 calibration (NEW from #331 close):** bf16 buys ~26% per-epoch wall-time with zero clamp events (no model-output overflow at our dynamic range). Capacity-axis hypotheses should default to bf16. bs=8 still OOMs at `n_hidden=192` even with bf16; bs=6 is the practical ceiling.

## Current research focus

Round 1 in progress. Strategy:

1. Independent axes tested first (one hypothesis per PR) to attribute gains cleanly.
2. Winners merge sequentially, best-first, each becoming the new baseline.
3. Round 2 compounds the orthogonal winners.

## In-flight PRs (status as of 2026-04-27 23:55)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #329 | alphonse  | Surface-loss weight sweep (`surf_weight ∈ {20, 30, 50}`)        | wip (sent back; sweep all beats baseline at slice_num=64 but apples-to-apples needs rebase + re-run on slice_num=128) |
| #413 | askeladd  | Huber loss for surface pressure (delta=1.0)                     | wip (new; replaces closed #331 — Huber attacks the heavy-tailed-pressure mechanism behind round-1 seed variance) |
| #427 | frieren   | Budget-aware cosine (T_max=11 matched to realized epochs)       | wip (new; replaces closed #338) |
| #339 | nezuko    | Larger batch (`batch_size 4→8`) with √2 LR scale                | wip |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | wip |
| #428 | thorfinn  | Multi-seed baseline calibration (3 seeds of default config)     | wip (new; replaces closed #341) |
| #375 | edward    | Bugfix: nan_to_num in `data/scoring.py`                         | wip (sent back; fix is bit-exact correct, but branch needs rebase before squash-merge to drop reverts to BASELINE.md / research/*.md) |
| #405 | fern      | Spatial Fourier features (NeRF-style, L=8 octaves)              | wip (new; replaces #376) |

## Closed / merged

| PR | Student | Outcome |
|----|---------|---------|
| #334 | edward | Deeper (n_layers 5→8) — **closed**, clear regression vs slice_num=128 |
| #336 | fern   | More slices (slice_num 64→128) — **merged**, val_avg=139.83 (round 1 baseline) |
| #376 | fern   | Wider MLP (mlp_ratio 2→4) — **closed**, +4.9% regression and OOD splits all worse |
| #331 | askeladd | Wider (n_hidden 192, n_head 6) — **closed** after bf16+bs6 retry; v1=141.998 vs v2=163.280 reveals ±10-15% seed variance, no clean win |
| #338 | frieren | LR warmup post-rebase (slice_num=128) — **closed**, +2.9% regression; slice_num=64+warmup vs slice_num=128+warmup direct comparison shows slice_num=64 wins by 9.7% |
| #341 | thorfinn | EMA(0.999) on slice_num=64 — **closed**, apparent win is single-oscillation absorption + slice_num confound; not statistically separated from baseline |

## Potential next research directions (round 2+)

After round 1 fully resolves, the strongest candidates are:

- **Compound orthogonal round-1 winners** — e.g. slice_num=128 (already merged) + winning loss tweak (alphonse/tanjiro) + winning schedule (frieren/nezuko) + EMA (thorfinn).
- **Wider MLP × wider hidden** — if both #376 and the askeladd retry win, stack them.
- **Modern transformer ergonomics** — SwiGLU, stochastic depth, RMSNorm.
- **Spatial inductive bias** — Fourier features on `(x, z)` fed into the preprocess MLP.
- **Mixed precision** — bf16 to free VRAM/time for wider/deeper models inside the 30-min cap (askeladd is testing this).
- **Multi-scale slice tokenization** — different `slice_num` per layer (global + local physics).
- **Per-domain calibration heads** — output routing on geometry features.
- **Boundary-layer-aware sampling** — over-sample high-Re extremes (which dominate the metric).
- **Longer wall-clock budget** — once a 30-min run shows we're cutting cosine schedules in half, the question of whether `n_layers=8` was truly worse vs just under-trained becomes worth re-asking.
