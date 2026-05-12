# SENPAI Research State

- **Date:** 2026-05-12 (last update 21:25 UTC)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm,
  48h run)
- **Most recent human directive:** None — fresh launch, no human issues in
  the queue.

## Current best (as of 20:52 UTC) — BASELINE ESTABLISHED

**Merged:** PR #1355 alphonse — Pure L1 loss on normalized residuals
`val_avg/mae_surf_p = 94.291` | `test_avg_3of4 = 91.859` | best epoch 14/15

Per-split val: cruise=71.66, re_rand=87.50, single=110.41, rc=107.60

**All downstream experiments must use `--loss l1`.**

## Round-1 leaderboard (closed)

| Rank | PR | Student | Lever | val_avg/mae_surf_p | Status |
|------|----|---------|-------|---------------------|--------|
| 1 | **#1355** | alphonse | Pure L1 loss | **94.291** ✅ **MERGED** — baseline |
| 2 | #1355 | alphonse | Smooth L1 / Huber β=1.0 | 97.791 | (within-PR, not merged standalone) |
| 3 | #1393 | frieren | OneCycleLR peak=1e-3 (MSE era) | 111.30 | Closed — superseded by loss change |
| 4 | #1399 | nezuko | surf_w=10, CHANNEL_W=[1,1,2] (bugged) | 111.80 | Sent back, 3-arm replan in progress |
| 5 | #1393 | frieren | OneCycleLR peak=5e-4 (MSE era) | 113.84 | Closed — superseded |
| 6 | #1399 | nezuko | surf_w=20, CHANNEL_W=[1,1,2] (bugged) | 126.30 | Sent back |
| 7 | #1389 | fern | n_layers=8, lr=3e-4, 9ep realized | 147.40 | Closed — schedule-bottlenecked regression |
| 8 | #1389 | fern | n_layers=8, lr=5e-4, 9ep realized | 153.48 | (within-PR) |
| 9 | #1410 | thorfinn | Multi-scale Fourier features (xz) | 105.05 | Closed — 11% regression vs L1 baseline |
| 10 | #1385 | edward | slice_num=128, n_head=8 | 151.92 | Closed — 61% regression vs L1 baseline |

## Round-1 operational notes

- **GitHub API rate-limit storm 18:30–19:50 UTC.** Concurrent gh polling
  across all 24-h + 48-h arms briefly exceeded the API quota. Students hit
  "GraphQL: API rate limit already exceeded" on `gh pr list` for ~80 min and
  were unable to see their assignments during that window. Effect: 5/8
  round-1 students (askeladd, edward, fern, tanjiro, thorfinn) did not start
  training until 19:50+. Nezuko was unaffected (pushed first sweep at 19:09).
- **"Comment-posted but branch-empty" false alarm (#1393, #1389).** Sending
  these PRs back at 20:11–20:13 UTC for "missing diff" was wrong — the diffs
  were on the branches but my local refs were stale. Force-fetching with
  explicit `:refs/remotes/...` refspec resolved. Lesson: always cross-check
  with `gh api .../pulls/<n>/files` before declaring a branch empty.
- **Pre-existing data/scoring NaN.** Alphonse diagnosed (PR #1355 comment):
  sample `test_geom_camber_cruise/.test_geom_camber_cruise_gt/000020.pt`
  contains `+Inf` in the pressure channel of `y`. The scoring code masks
  the sample out, but `Inf * 0 = NaN` in pytorch corrupts the accumulator
  for the `p` channel of that one test split (val unaffected — no
  Inf-y samples in val). **Decision:** `data/scoring.py` is contractually
  read-only — we do not patch it. Convention for all PRs going forward:
  rank by `val_avg/mae_surf_p` (clean) and report
  `test_avg/mae_surf_p_3of4_finite_splits` paper-side.

## Research focus

Round 1 of the `charlie-pai2g-48h-r1` arm landed one clear winner: **pure L1
loss** (alphonse, -57% vs implied MSE). The four other completed round-1
experiments all regressed against the new L1 baseline, demonstrating that the
loss-formulation lever subsumed several of them:

- **Capacity expansion** (#1389 depth=8, #1385 slices/heads doubled) showed
  large regressions, likely because the 30-min wall-clock cap starves the
  cosine schedule of effective epochs on slower configurations. Deeper /
  finer-attention models lose more compute to the budget than the extra
  capacity buys back, **at the current epoch count**.
- **Multi-scale Fourier features** (#1410) regressed by 11%. The raw (x,z)
  coords seem to be already informative enough that synthetic frequency
  bands hurt more than help on a small dataset.
- **Bugged channel-weight + surf_weight sweep** (#1399) is being re-run by
  nezuko with the correct denominator and a baseline-control arm.

Round 2 pivots to **compounding the L1 winner with orthogonal levers** that
are cheap (no extra params, no compute hit) and that target known weak
spots of short-budget training in normalized space:

1. **Loss-weighting compounding** — surf_weight sweep on L1 (alphonse #1582)
   tests whether the existing 10:1 surf:vol ratio is still optimal under L1.
2. **Schedule compounding** — L1 + OneCycleLR with per-batch stepping
   (frieren #1581) compounds the loss winner with the schedule
   sub-winner.
3. **EMA of model weights** — exponential moving average (thorfinn #1601),
   a near-zero-cost regularizer that consistently helps in short-budget
   regimes. Two arms at decay=0.999 vs 0.9999.
4. **Gradient clipping** — `clip_grad_norm_` sweep at max_norm 0.5 / 1.0
   vs unclipped (fern #1602). Stabilizes against the pressure tail and
   typically improves OOD splits.
5. **Pressure-target reshaping** — asinh transform on pressure channel
   before normalization (edward #1605), two scale arms (100 aggressive,
   680 ≈ pressure_std gentle). Compresses the heavy pressure tail without
   distorting low-magnitude regions.

The three still-running round-1 PRs (#1381 wider, #1399 nezuko channel-weights
corrected, #1405 bf16) remain in flight; if they land any improvement vs the
94.29 L1 baseline they will be merged.

## Open questions / failure modes to watch

- Wall-clock cap is tight. The depth/slice arms in round 1 demonstrated that
  added compute per step under a hard cap costs more than the extra capacity
  buys back. Future capacity bets must explicitly budget realized epochs.
- 1499 training samples is small; capacity-increase arms (askeladd) might
  overfit on `val_single_in_dist` while failing OOD (`val_geom_camber_*`).
  Watch the four-split breakdown, not just the average.
- bfloat16 in tanjiro: low-magnitude `vol_loss` could underflow if the
  pressure normalization drives some channels near zero. Check NaNs in
  the JSONL.
- The two geometry-OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`)
  are the hardest. A common-recipe winner should improve **both**, not just
  one — that's our gen-gap signal.
- EMA (#1601) interaction with the cosine LR schedule: if LR is still high at
  the end of the run (sub-15 epoch realizations), the EMA may not have
  converged. Decay=0.999 ≈ effective window of 1000 steps ≈ 7 epochs at
  batch=4; decay=0.9999 ≈ 70 epochs — probably too slow at 15 epochs.

## Plateau / pivot plan (forward-looking, after round 2)

If round 2 lands within ~3% of each other (small effects) we have several
unused levers to escalate to:

- **Compound winners**: stack the best loss + best schedule + best
  capacity changes in round 3.
- **Data augmentation**: per-sample reflection (z → -z, AoA → -AoA,
  Uy → -Uy) for symmetric domains; per-sample Re-jitter for cross-regime
  generalization.
- **Domain re-weighting**: the current `WeightedRandomSampler` gives equal
  weight to single/raceCar-tandem/cruise. The val splits are 4×100 but
  three are tandem and one single; reweighting toward tandem may better
  match the metric.
- **Normalization choices**: per-domain normalization stats; robust
  (median/MAD) standardization for the pressure target tail; signed
  log-scaling for large Re pressure values (related to but distinct from
  edward's asinh transform).
- **Architecture switches**: graph transformer with kNN edges; FNO-style
  spectral mixing; per-channel decoder heads.
- **Test-time augmentation**: average prediction over geometric symmetries.

If round 3 also plateaus, escalate to deeper architectural changes
(Geo-FNO, GNO, mesh-graph-net) and explore self-supervised pretext
losses on the volume field as auxiliary heads.

## Active in-flight PRs (round 2 + round-1 stragglers)

Status as of 22:05 UTC. All 8 students have a WIP PR — zero idle GPUs.

| PR | Student | Hypothesis | State |
|----|---------|-----------|-------|
| **#1581** | **frieren** | **L1 + OneCycleLR compound (peak_lr 1e-3 vs 2e-3)** | Round-2 WIP |
| **#1582** | **alphonse** | **surf_weight sweep (5/10/20) on L1 baseline** | Round-2 WIP |
| **#1601** | **thorfinn** | **EMA of model weights (decay 0.999 vs 0.9999) on L1 baseline** | Round-2 WIP |
| **#1602** | **fern** | **Gradient clipping sweep (0 / 0.5 / 1.0) on L1 baseline** | Round-2 WIP |
| **#1605** | **edward** | **asinh transform on pressure target (scale 100 vs 680) with L1** | Round-2 WIP |
| **#1625** | **nezuko** | **Per-channel pressure surf weight [1,1,2/3/5] on L1 baseline** | Round-2 WIP (dispatched 22:05 after closing MSE-era #1399) |
| #1381 | askeladd | Wider Transolver: n_hidden 128→256, mlp_ratio 2→4 | Round-1 in flight (notified to add `--loss l1`) |
| #1405 | tanjiro | bfloat16 autocast + batch_size 8 + sqrt-scaled lr | Round-1 in flight (notified to add `--loss l1`) |

**Action items:** wait for in-flight PRs; prioritize merge of any that beat 94.29.

**Key insight from nezuko #1399 (closed):** Val signal for channel-weighting is monotone and real on MSE era (2% improvement on val_avg, 4% on hardest OOD split val_geom_camber_rc at k=3). Does it stack with L1? Answer comes from #1625.
