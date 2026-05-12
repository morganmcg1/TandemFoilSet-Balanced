# SENPAI Research Results — `icml-appendix-charlie-pai2g-48h-r4`

Primary metric: `val_avg/mae_surf_p` (lower is better). Test counterpart: `test_avg/mae_surf_p`.

## 2026-05-12 18:31 — PR #1376: [lr1e3-warmup-cosine] lr=1e-3 with 3-epoch linear warmup + cosine
- Student branch: `charliepai2g48h4-fern/lr1e3-warmup-cosine`
- Hypothesis: replace plain `CosineAnnealingLR(T_max=50)` + `lr=5e-4` with `LinearLR(3 ep warmup) → CosineAnnealingLR(T_max=47)` and `lr=1e-3` to converge further inside the 30-min wall-clock cap.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **147.2556** |
| `test_avg/mae_surf_p` | NaN (scoring bug — see below) |
| `test_avg/mae_surf_p` (3-split mean, excludes broken `test_geom_camber_cruise`) | 142.36 |
| `val_geom_camber_cruise/mae_surf_p` | 128.93 |
| `val_geom_camber_rc/mae_surf_p`     | 138.84 |
| `val_re_rand/mae_surf_p`            | 127.21 |
| `val_single_in_dist/mae_surf_p`     | 194.05 |
| Wall clock | 30.8 min (stopped mid-ep 15) |
| Peak VRAM | 42.1 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-fern-lr1e3-warmup-cosine-20260512-175601/metrics.jsonl` |

**Analysis.** Schedule worked as predicted. Val curve descended monotonically 335 → 150 over ep 1–9, mild plateau/bounce 10–13 (155–192), recovered to best 147 at ep 14. No divergence at `lr=1e-3` peak — the 3-epoch warmup absorbed the initial-step risk. `val_single_in_dist` is the worst-performing split (194 vs ~128 on the others), driven by raceCar single-foil mesh size and ground-effect physics. Burned 3/14 (~21%) of the budget on warmup; shorter warmup or higher peak lr is a clean next iteration.

**Held pending baseline (PR #1368).** Merge decision deferred to when alphonse's 5e-4/no-warmup run lands at the same 30-min cap.

**Scoring bug discovered.** `data/scoring.py:accumulate_batch` propagates `NaN` into `mae_surf`/`mae_vol` when any sample's GT contains non-finite values, because `(inf * 0.0).sum() = NaN`. The per-sample `y_finite` skip filters the counts but `err` is still poisoned. Concretely `test_geom_camber_cruise` sample 20 has `y_p = -inf`. Fix assigned in PR #1512 (`scoring-nan-fix` — surgical `torch.nan_to_num(err, ...)` after the abs).

**Suggested follow-ups (kept on backlog):**
1. Higher peak lr (2e-3 / 3e-3) with the same warmup scaffold.
2. Truncated cosine `T_max ≈ effective_epochs (~15)` so the schedule actually anneals at this wall-clock budget.
3. Per-channel surface-pressure-loss treatment — `single_in_dist` has 1.5× the surf_p MAE of the other splits.

_(Round 1 still partially in flight — more results landing as PRs come back for review.)_
