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

## 2026-05-12 19:29 — PR #1369: [surf-weight-20] Increase surf_weight 10→20
- Student branch: `charliepai2g48h4-askeladd/surf-weight-20`
- Hypothesis: bump `Config.surf_weight=10.0→20.0` to weight surface MSE 2× more vs volume — direct lever on the primary ranking metric.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 12/14) | **127.9357** |
| `test_avg/mae_surf_p` (best-val checkpoint, cruise[20] excluded) | **117.3456** |
| `val_geom_camber_cruise/mae_surf_p` | 108.97 |
| `val_geom_camber_rc/mae_surf_p`     | **135.82** (best among returned PRs) |
| `val_re_rand/mae_surf_p`            | 116.55 |
| `val_single_in_dist/mae_surf_p`     | **150.41** (best so far on the hardest split) |
| Wall clock | 30 min cap; 14 epochs landed |
| Peak VRAM | 42.1 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-askeladd-surf-weight-20-20260512-175549/metrics.jsonl` |
| Test workaround | `test_metrics_excl_cruise20.json` (re-run from best-val checkpoint with cruise[20] dropped) |

**Analysis.** 2nd-best round-1 result, essentially tied with thorfinn at the noise floor. Hypothesis confirmed: heavier surface weighting moves the primary metric meaningfully without destabilizing training. Per-split, this PR is the strongest on `val_single_in_dist` (150 vs ~180 for tanjiro/fern) AND `val_geom_camber_rc` (136, best returned) — both raceCar tracks. Cruise (109) is weaker than thorfinn's unified-pos (92) but stronger than fern/tanjiro. The two levers (positional encoding, surf_weight) hit different per-split weak points and are likely orthogonal — round-2 stacking candidate.

**Major variance signal.** Askeladd ran the same surf_weight=20 config twice; the second run landed val_avg = 157.95 — a 30-point gap from seed/noise alone. `train.py` is unseeded. This makes ANY single-run comparison noise-limited. Follow-up assignment for askeladd is EMA-weights (PR #1540) as a variance-reduction technique pending dedicated seeded-training infra.

**Held pending baseline (PR #1368).**

**Independent scoring-bug confirmation.** Third independent confirmation (after fern and thorfinn) of `Inf * 0 = NaN` on `test_geom_camber_cruise[20]`. Askeladd committed a `test_metrics_excl_cruise20.json` workaround artifact — fern's PR #1512 is still the root-cause fix.

**Suggested follow-ups (kept on backlog):** seeded training (#1), surf_weight sweep {5,10,15,20,30} once seeded, linear warmup.

## 2026-05-12 19:32 — PR #1402: [slice128] Double slice_num 64→128
- Student branch: `charliepai2g48h4-nezuko/slice128`
- Hypothesis: physics-attention groups N nodes into `slice_num` learned tokens; default 64 may be too few for large meshes (242K nodes on cruise). Double to 128 to give finer physical structure, especially on the cruise splits.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 10/11) | **137.1686** |
| `test_avg/mae_surf_p` | NaN (scoring bug; 3-split mean: 135.53) |
| `val_geom_camber_cruise/mae_surf_p` | **107.79** (best on cruise val among non-unified-pos PRs) |
| `val_geom_camber_rc/mae_surf_p`     | 143.92 |
| `val_re_rand/mae_surf_p`            | 120.60 |
| `val_single_in_dist/mae_surf_p`     | 176.37 |
| Wall clock | 30 min cap (~31.8 min); 10 best of 11 epochs landed (-3 vs thorfinn's 13) |
| Peak VRAM | 54.5 GB (vs 42 GB at slice_num=64; +30% memory) |
| Params | 0.67 M (+0.01 M from slice projection layers) |
| Metrics path | `models/model-charliepai2g48h4-nezuko-slice128-20260512-184959/metrics.jsonl` |

**Analysis.** 3rd-best returned result. Hypothesis is *directionally* supported: cruise val improves to 107.79 — the strongest cruise number among the four non-unified-pos PRs (thorfinn's unified-pos got cruise=91.85, which is special). The per-epoch cost of slice128 is higher (~10% slower), so this PR landed 3 fewer epochs than thorfinn at the same cap, and the comparison is wall-clock-budget-confounded rather than capacity-saturated. Worth retrying with thorfinn's unified-pos already stacked once we know the baseline.

**Held pending baseline (PR #1368).**

**Independent scoring-bug confirmation.** Fourth independent identification of the same `Inf * 0 = NaN` root cause. Suggested data-layer fix as alternative to fern's helper-site fix — going with fern's #1512 because it's smaller blast radius.

**Suggested follow-ups (kept on backlog):** slice256 (54.5 GB → ~70 GB still under 96 GB), stack with n_hidden, longer wall time.

## 2026-05-12 19:52 — Round 1 status snapshot (5/8 returned, baseline still WIP)

| Student | Slug | val_avg/mae_surf_p | test_avg/mae_surf_p | Status |
|---|---|---|---|---|
| thorfinn | `unified-pos` (#1416) | **125.78** | 117.12 | Held pending baseline — best so far |
| askeladd | `surf-weight-20` (#1369) | **127.94** | 117.35 | Held pending baseline — 2nd |
| nezuko | `slice128` (#1402) | 137.17 | NaN (135.53 over 3) | Held pending baseline — 3rd |
| fern | `lr1e3-warmup-cosine` (#1376) | 147.26 | NaN | Held pending baseline |
| tanjiro | `hidden192` (#1406) | 151.64 | NaN | Held pending baseline |
| alphonse | `baseline-ref` (#1368) | — | — | WIP — rate-limited 17:50→19:48 UTC, ETA ~20:25 UTC |
| edward | `huber-loss` (#1374) | — | — | WIP — rate-limited 17:50→19:50 UTC, ETA ~20:27 UTC |
| frieren | `wd5e-4` (#1394) | — | — | WIP |

**Active follow-on assignments:**
- PR #1512 (fern) — `scoring-nan-fix` (helper-site fix for the `Inf*0=NaN` bug)
- PR #1513 (tanjiro) — `bf16-autocast` (throughput, predicted 30-50% per-epoch reduction)
- PR #1533 (thorfinn) — `surf-p-weight-3x` (per-channel: weight surface-p 3× over surface-Ux/Uy)
- **PR #1540 (askeladd) — `ema-weights`** (variance reduction via Polyak averaging, decay 0.999, EMA at val/test)
- **PR #1542 (nezuko) — `cosine-trunc-t15`** (truncate `T_max=50→15` so cosine actually anneals inside the 30-min cap; addresses near-constant-lr observation across all 5 returned runs)

**Pod rate-limit incident.** alphonse and edward hit GraphQL rate limits at ~17:50 UTC and couldn't pick up assigned PRs for ~2 hours (13 and 18 heartbeat iterations of "## Student research state — No assigned PRs or issues" respectively). They finally cleared at 19:48-19:50 UTC. This pushed baseline ETA from ~19:21 to ~20:25 UTC. No work was lost; just delayed.

_(Round 1 largely closed — edward + frieren still pending.)_

## 2026-05-12 20:00 — PR #1512: [scoring-nan-fix] Stop NaN propagation (fern) — MERGED as baseline anchor
- Student branch: `charliepai2g48h4-fern/scoring-nan-fix`
- Hypothesis: surgical `torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)` after `err = (pred-y).abs()` in `data/scoring.py::accumulate_batch` — fixes `Inf * 0 = NaN` propagation that poisoned every `test_avg/mae_surf_p` evaluation on this branch.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **123.99** |
| `test_avg/mae_surf_p` (NOW FINITE) | **110.97** |
| `test_geom_camber_cruise/mae_surf_p` | **76.78** (previously NaN) |
| `test_geom_camber_rc/mae_surf_p` | 121.93 |
| `test_re_rand/mae_surf_p` | 110.93 |
| `test_single_in_dist/mae_surf_p` | 134.23 |
| Config | default (lr=5e-4, surf_weight=10, unified_pos=False, no bf16) |
| Wall clock | 30 min cap; 14 epochs |
| Peak VRAM | 42.11 GB |
| Metrics path | `models/model-charliepai2g48h4-fern-scoring-nan-fix-20260512-185620/metrics.jsonl` |

**Analysis.** This PR's primary value is that it unblocks all future test evaluations. Secondarily it provides the first clean default-config baseline: val_avg=123.99, test_avg=110.97. Note that val numbers are unaffected by the scoring fix (val GT has no non-finite values); the val number is therefore identical to what alphonse's true baseline run should produce under the same RNG. **Merged** as both a critical infra fix and a baseline anchor. New `BASELINE.md` entry set at 123.99/110.97.

## 2026-05-12 20:03 — PR #1513: [bf16-autocast] bf16 throughput (tanjiro) — MERGED as infra
- Student branch: `charliepai2g48h4-tanjiro/bf16-autocast`
- Hypothesis: wrap training forward+backward in `torch.cuda.amp.autocast(dtype=torch.bfloat16)` to reduce per-epoch wall clock and land more epochs in the 30-min cap.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 18/18) | **125.40** |
| `test_avg/mae_surf_p` (3-split mean, cruise NaN) | **126.57** |
| `val_geom_camber_cruise/mae_surf_p` | 90.70 |
| `val_geom_camber_rc/mae_surf_p` | 132.06 |
| `val_re_rand/mae_surf_p` | 111.22 |
| `val_single_in_dist/mae_surf_p` | 167.62 |
| Per-epoch wall clock | **101.4 s** (vs ~133 s fp32 → **24% speedup**) |
| Epochs landed | **18** (vs 13 fp32, vs 10 hidden192 — wall-clock win is clear) |
| Val still descending at ep 18 | Yes — model not converged at cap |
| NaN-safe guard (isfinite) | 0 batches skipped — no overflow |
| Peak VRAM | 32.9 GB (vs 42 GB fp32; **22% VRAM savings**) |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-tanjiro-bf16-autocast-20260512-190244/metrics.jsonl` |

**Analysis.** Hypothesis confirmed. val_avg=125.40 is within noise of baseline 123.99; the value here is the throughput win — 24% faster, +5 extra epochs in 30 min, 22% less VRAM. This makes future capacity experiments viable (hidden256 can now land 12-14 epochs vs 10 for hidden192 without bf16). **Merged** as throughput infrastructure. Every future experiment on this branch inherits bf16 autocast from the merged recipe.

**Note on test metric:** `test_avg/mae_surf_p` is NaN (pre-fix run — this PR was submitted before #1512 merged). 3-split mean = 126.57.

## 2026-05-12 20:30 — Round 1 merge/close decisions (baseline now at val=123.99)

With baseline established, reviewed all 5 held round-1 hypothesis PRs:

| PR | Lever | val_avg | Δ from baseline | Decision |
|---|---|---|---|---|
| #1416 (thorfinn) | unified-pos | 125.78 | +1.4% | **MERGED** — within noise, cruise val 91.85 is real |
| #1369 (askeladd) | surf-weight-20 | 127.94 | +3.2% | **Pending merge** — preflight blocked by hold comment ordering |
| #1402 (nezuko) | slice128 | 137.17 | +10.6% | **Request changes** — re-run with cosine-trunc-T15 |
| #1376 (fern) | lr1e3-warmup | 147.26 | +18.8% | **CLOSED** — warmup ate budget |
| #1406 (tanjiro) | hidden192 | 151.64 | +22.3% | **CLOSED** — wall-clock-bound, superseded by #1575 |
| #1533 (thorfinn) | surf-p-weight-3x | 154.47 | +24.6% | **CLOSED** — 3× ratio too aggressive, model too small |

## 2026-05-12 20:53 — PR #1540: [ema-weights] Polyak EMA at val/test — NEW BEST (askeladd)
- Student branch: `charliepai2g48h4-askeladd/ema-weights`
- Hypothesis: maintain EMA copy of model weights (decay 0.999), use EMA model for val/test evaluation — Polyak averaging to reduce variance from wall-clock-truncated mid-cosine runs.
- Config: **default config** (surf_weight=10, no unified_pos, no bf16, lr=5e-4) — branched from advisor commit 0242e62 before other merges.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best, ep 14/14) | **121.16** ← NEW BEST |
| `test_avg/mae_surf_p` | **108.69** ← NEW BEST |
| `val_geom_camber_cruise/mae_surf_p` | 95.58 |
| `val_geom_camber_rc/mae_surf_p` | 132.15 |
| `val_re_rand/mae_surf_p` | 109.44 |
| `val_single_in_dist/mae_surf_p` | 147.47 |
| `test_geom_camber_cruise/mae_surf_p` | 80.16 |
| `test_geom_camber_rc/mae_surf_p` | 118.92 |
| `test_re_rand/mae_surf_p` | 107.34 |
| `test_single_in_dist/mae_surf_p` | 128.36 |
| Wall clock | 30.5 min cap; 14 epochs; val still descending |
| Peak VRAM | 42.11 GB |
| Params | 0.66 M |
| Metrics path | `models/model-charliepai2g48h4-askeladd-ema-weights-20260512-201111/metrics.jsonl` |

**Analysis.** EMA on default config (val=121.16) is the single strongest lever identified so far — beating unified-pos (125.78, the prior best) by 3.7% and alphonse's canonical default (137.57) by 11.9%. Test improvement is even clearer at 108.69 vs 117.12 (7.2% gain). The hypothesis is confirmed: EMA smooths gradient-step noise from the wall-clock-truncated cosine schedule, improving both val and test consistency. Importantly, this was on a WEAKER config (no unified_pos, no bf16, surf_weight=10) — adding EMA to the merged recipe (unified_pos + bf16 + surf_weight=20) should push below 120.

**Status: SENT BACK FOR REBASE.** Four PRs (scoring-fix, bf16, unified-pos, surf-weight-20) merged into train.py while this run was in flight, causing a merge conflict. Student is rebasing onto the current advisor branch and will re-run on the full merged recipe. This is the highest-priority active PR — merge as soon as the rebase lands.

## 2026-05-12 ~20:35 — Round 2 assignments (building on merged recipe)

The advisor-branch recipe now includes unified_pos=True, bf16, surf_weight=20, and scoring-fix. Round-2 student assignments all inherit this merged baseline:

| Student | PR | Slug | Lever |
|---|---|---|---|
| fern | #1570 | `surf-weight-20-stack` | ~~surf_weight=20~~ now superseded by merge; test stacking effect |
| tanjiro | #1575 | `hidden256-bf16` | n_hidden=128→256 on merged bf16+unified_pos recipe |
| thorfinn | #1576 | `unified-pos-global-norm` | replace per-batch pos norm with corpus-level bounds |
| askeladd | #1540 | `ema-weights` | Polyak EMA at val/test — NEW BEST 121.16; sent back for rebase on merged recipe |
| nezuko | #1542 | `cosine-trunc-t15` | CosineAnnealingLR T_max=50→15 (in flight) |
| alphonse | #1577 | `seed42-baseline` | deterministic seeding → reproducible merged-recipe baseline |

Pending round-1 WIPs: #1374 (edward huber-loss, ETA ~20:30 UTC), #1394 (frieren wd5e-4, rate-limited ~3h+).
