# SENPAI Research State

- **Last updated:** 2026-05-15 ~14:40 UTC
- **Track / Research tag:** willow-pai2i-48h-r4
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r4` (forked from `icml-appendix-willow`)
- **Target metric:** `val_avg/mae_surf_p` (validation), `test_avg/mae_surf_p` (paper-facing). Lower is better.

## Current baseline

**val_avg/mae_surf_p = 109.42** — from PR #3091 (edward, warmup + clip + lr=1e-3), merged 2026-05-15. See `BASELINE.md` for full details.

Note: `test_avg/mae_surf_p` is NaN on all runs due to scoring bug (fixed in PR #3288, in progress).

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
| #3288 | edward | Scoring-bug fix (nan_to_num) + bump lr default to 1e-3 | WIP — consolidation |
| #3092 | fern | slice_num 64 vs 128 at --epochs 10 (proper schedule) | WIP — sent back |
| #3089 | alphonse | L1 / smooth-L1 loss | WIP |
| #3090 | askeladd | Width: n_hidden 128→192 (+256) | WIP |
| #3093 | frieren | bf16 + batch_size 4→8 | WIP |
| #3095 | nezuko | surf_weight 10→30 + per-channel p weighting | WIP |
| #3096 | tanjiro | x-axis symmetry augmentation | WIP |
| #3097 | thorfinn | Depth: n_layers 5→8 + DropPath 0.1 | WIP |

## Merged wins

| PR | Description | val_avg/mae_surf_p |
|---|---|---|
| #3091 | LR warmup + clip + lr=1e-3 (edward) | **109.42** ← current baseline |

## Next decisions (when in-flight PRs complete)

1. **Merge any experiment that beats val_avg/mae_surf_p=109.42.** All in-flight PRs are running with mis-tuned cosine schedules — if they still beat 109.42 at 15 epochs, that's a strong signal. If they're close, request a re-run at `--epochs 10` with proper annealing before declaring improvement.
2. **Compose winners.** Once consolidation PR #3288 lands (lr default=1e-3, scoring fix), all subsequent runs will start from a clean baseline with proper defaults.
3. **Priority for round 2:** L1/Huber loss (alphonse #3089) is still the highest-conviction untested hypothesis. If it wins, stack L1 + warmup + clip + higher LR as the new platform.

## Potential next research directions (round 2+)

1. **Longer training runs** — model was still improving at timeout. With proper schedule (--epochs 10 + T_max=10), the cosine tail likely holds significant gains.
2. **L1/Huber loss composition** — if alphonse wins, stack with edward's merged changes
3. **Wider model (askeladd)** — if 192 hidden wins, stack with winning loss + optimizer
4. **Symmetry aug (tanjiro)** — OOD generalization boost on geom_camber tracks
5. **EMA** — stabilize best-val checkpoint selection (especially important with mis-tuned cosine)
6. **Separate per-channel output heads** (p, Ux, Uy get distinct decoder networks)
7. **Position encoding** — Fourier features on (x, z) or unified_pos=True
8. **Physics-aware loss** — divergence-free penalty, near-surface gradient consistency
