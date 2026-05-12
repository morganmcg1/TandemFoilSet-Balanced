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

## 2026-05-12 18:55 — PR #1406: [hidden192] Widen Transolver hidden 128→192
- Student branch: `charliepai2g48h4-tanjiro/hidden192`
- Hypothesis: widen `n_hidden=128→192` (0.7M→1.47M params) to test whether capacity is a bottleneck.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 9/10) | **151.6438** |
| `test_avg/mae_surf_p` | NaN (same scoring bug as #1376) |
| `test_avg/mae_surf_p` (3-split mean, excl. broken cruise) | 146.09 |
| `val_geom_camber_cruise/mae_surf_p` | 127.69 |
| `val_geom_camber_rc/mae_surf_p`     | 164.61 |
| `val_re_rand/mae_surf_p`            | 134.68 |
| `val_single_in_dist/mae_surf_p`     | 179.60 |
| Wall clock | 30 min cap; 10 epochs landed |
| Per-epoch wall clock | ~184 s (vs ~133 s at n_hidden=128, +38%) |
| Peak VRAM | 58.0 GB (bs=4, n_hidden=192) |
| Params | 1.47 M |
| Metrics path | `models/model-charliepai2g48h4-tanjiro-hidden192-20260512-175550/metrics.jsonl` |

**Analysis.** Widening worked as a code change but did not deliver an obvious win at this wall-clock budget. The val curve was still actively improving at ep 9 (151.90→151.64 best, with 160.28 at ep 10 — not converged), and the 38% per-epoch cost shrank the effective epoch budget from 14 to 10. So `hidden192` is a wall-clock-bound result, not a capacity-saturated one. Per-split pattern matches fern: cruise-camber OOD is the *easiest* (127.69), in-dist sanity is the *hardest* (179.60) — the recipe's bottleneck is single-foil pressure regardless of width.

**Held pending baseline (PR #1368).**

**Implication for round 2:** at this 30-min cap, throughput is the actually binding constraint. Reassigning tanjiro to `bf16-autocast` (PR #1513) to test whether mixed-precision training can buy back the per-epoch cost of wider models (or simply land more epochs at any model size). If bf16 delivers ~30-50% throughput, future capacity experiments become much more attractive.

**Suggested follow-ups:** none merged in immediately — the hypothesis "capacity helps" is unproven but not falsified.

## 2026-05-12 19:15 — PR #1416: [unified-pos] Unified positional encoding with `ref=8`
- Student branch: `charliepai2g48h4-thorfinn/unified-pos`
- Hypothesis: replace raw `(x,z)` positions through the preprocess MLP with `Transolver`'s `unified_pos=True` soft-grid encoding (`ref=8`, 2D so `ref^space_dim=64` features); regularized fixed-grid encoding should give a stronger spatial inductive bias on irregular meshes.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 13/14) | **125.7759** |
| `test_avg/mae_surf_p` (best-val checkpoint) | **117.1233** |
| `val_geom_camber_cruise/mae_surf_p` | **91.85** |
| `val_geom_camber_rc/mae_surf_p`     | 145.70 |
| `val_re_rand/mae_surf_p`            | 114.24 |
| `val_single_in_dist/mae_surf_p`     | 151.32 |
| `test_geom_camber_cruise/mae_surf_p` | **80.27** |
| `test_geom_camber_rc/mae_surf_p`     | 138.57 |
| `test_re_rand/mae_surf_p`            | 114.75 |
| `test_single_in_dist/mae_surf_p`     | 134.89 |
| Wall clock | 30 min cap; ep 13 best of 14 landed |
| Peak VRAM | 42.5 GB |
| Params | 0.68 M (+20 K vs default — input MLP grows from `(2+22)→256` to `(64+22)→256`) |
| Metrics path | `models/model-charliepai2g48h4-thorfinn-unified-pos-20260512-175707/metrics.jsonl` |

**Analysis.** Strongest round-1 result by a wide margin: `val_avg/mae_surf_p=125.78` beats fern (147.26) by ~14% and tanjiro (151.64) by ~17% at the same 30-min cap, and ships with a valid test number. Cruise-camber OOD is *especially* strong here (`val=91.85, test=80.27`) — soft-grid encoding plausibly resolves the larger, more uniformly distributed cruise meshes better than raw-coordinate-through-MLP. The raceCar single-foil + ground-effect samples (`val_single_in_dist=151.32`, `val_geom_camber_rc=145.70`) remain the dominant residual error, same per-split pattern as fern and tanjiro. Schedule was wall-clock-limited (`cosine(13/50)≈0.84`, so LR barely annealed) — a `T_max≈effective_epochs` follow-up could plausibly buy further headroom.

**Held pending baseline (PR #1368).** Given fern and tanjiro both land ~150 at the same cap, the baseline is almost certainly above 125 — this PR is on track to merge.

**Independent scoring-bug fix in this PR.** Thorfinn independently identified the same `inf * 0 = NaN` propagation that fern reported in #1512, and added a defensive workaround in `train.py::evaluate_split` that pre-filters non-finite-y samples before they reach `accumulate_batch`. Without this guard, `test_avg/mae_surf_p` is NaN on this branch for every experiment that hits a `geom_camber_cruise` test sample. Workaround at the call site coexists with fern's surgical fix at the helper site (#1512).

**Suggested follow-ups (kept on backlog):**
1. **Corpus-level position normalization.** Current `unified_pos` uses per-batch `pos.amin/amax`, so a sample's encoding depends on its batch-mates and on padding zeros. Replacing with fixed corpus-level bounds (from `stats`) would make the encoding deterministic per-sample and remove a noise source.
2. **`ref` sweep ∈ {12, 16}.** Wider soft grid = finer spatial membership; +small param cost on input MLP.
3. **Truncated cosine `T_max ≈ effective_epochs`** — applies branch-wide.

## 2026-05-12 19:18 — Round 1 status snapshot

| Student | Slug | val_avg/mae_surf_p | Status |
|---|---|---|---|
| thorfinn | `unified-pos` (#1416) | **125.78** | Held pending baseline — best so far |
| fern | `lr1e3-warmup-cosine` (#1376) | 147.26 | Held pending baseline |
| tanjiro | `hidden192` (#1406) | 151.64 | Held pending baseline |
| alphonse | `baseline-ref` (#1368) | — | WIP (started 18:51 UTC, ETA ~19:21 UTC) |
| askeladd | `surf-weight-20` (#1369) | — | WIP |
| edward | `huber-loss` (#1374) | — | WIP |
| frieren | `wd5e-4` (#1394) | — | WIP |
| nezuko | `slice128` (#1402) | — | WIP |

Follow-on assignments active this cycle: PR #1512 (fern, `scoring-nan-fix`), PR #1513 (tanjiro, `bf16-autocast`), PR #1533 (thorfinn, `surf-p-weight-3x` — per-channel surface weighting, 3× pressure vs Ux/Uy).

_(Round 1 still partially in flight — more results landing as PRs come back for review.)_
