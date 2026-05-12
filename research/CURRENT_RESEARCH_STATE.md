# SENPAI Research State

- **Date:** 2026-05-12 (last update 20:15 UTC)
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

## Round-1 leaderboard (as of 20:52 UTC)

| Rank | PR | Student | Lever | val_avg/mae_surf_p | Status |
|------|----|---------|-------|---------------------|--------|
| 1 | **#1355** | alphonse | Pure L1 loss | **94.291** ✅ **MERGED** — new baseline |
| 2 | #1355 | alphonse | Smooth L1 / Huber β=1.0 | 97.791 | (within-PR, not merged standalone) |
| 3 | #1393 | frieren | OneCycleLR peak=1e-3 (MSE era) | 111.30 | Closed — superseded by loss change |
| 4 | #1399 | nezuko | surf_w=10, CHANNEL_W=[1,1,2] (bugged) | 111.80 | Sent back (loss-magnitude bug) |
| 5 | #1393 | frieren | OneCycleLR peak=5e-4 (MSE era) | 113.84 | Closed — superseded |
| 6 | #1399 | nezuko | surf_w=20, CHANNEL_W=[1,1,2] (bugged) | 126.30 | Sent back |
| 7 | #1389 | fern | n_layers=8, lr=3e-4, 9ep realized | 147.40 | Sent back — Arm C (`--epochs 9`) in progress |
| 8 | #1389 | fern | n_layers=8, lr=5e-4, 9ep realized | 153.48 | (within-PR) |

## Round-1 operational notes

- **GitHub API rate-limit storm 18:30–19:50 UTC.** Concurrent gh polling
  across all 24-h + 48-h arms briefly exceeded the API quota. Students hit
  "GraphQL: API rate limit already exceeded" on `gh pr list` for ~80 min and
  were unable to see their assignments during that window. Effect: 5/8
  round-1 students (askeladd, edward, fern, tanjiro, thorfinn) did not start
  training until 19:50+. Nezuko was unaffected (pushed first sweep at 19:09).
- **"Comment-posted but branch-empty" failure mode (frieren #1393, fern
  #1389).** Both students trained locally during the rate-limit storm,
  recovered enough to post terminal SENPAI-RESULTs at 20:02–20:04 UTC via
  `gh pr comment` (which retries the GraphQL call). But the separate
  `git push` for `train.py` + `models/model-*/` artifacts never landed — so
  the PR branches contain only the original `assign` commit. The
  SENPAI-RESULT metrics are coherent and consistent across arms, but the
  PRs are not merge-eligible until the diff is pushed. Both sent back at
  20:11–20:13 UTC with explicit push commands.
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

This is round 1 of the `charlie-pai2g-48h-r1` arm: a controlled
Charlie-vs-Willow logging ablation. All 8 students dispatched in parallel on
independent levers — each tests one focused hypothesis. The first batch of
results will establish a measured baseline and tell us which levers move the
ranking metric (`val_avg/mae_surf_p` / `test_avg/mae_surf_p`).

The portfolio covers the four highest-yield lever families before we go
architectural:

1. **Loss family** — Smooth L1 / L1 (alphonse #1355), surface pressure
   channel reweighting (nezuko #1399).
2. **Capacity family** — wider 256-hidden + 4× MLP (askeladd #1381),
   deeper 8-layer (fern #1389), finer physics-attention slicing
   (edward #1385).
3. **Schedule family** — OneCycleLR with warmup (frieren #1393).
4. **Throughput family** — bf16 autocast + batch_size 8 (tanjiro #1405).
5. **Geometric representation** — multi-scale Fourier features for
   coordinates (thorfinn #1410). This is the boldest first-round bet.

Every PR also implicitly tests **tuned `--epochs` so cosine annealing
completes within the 30-min wall-clock cap** — a common-recipe fix that
should generalize across all subsequent experiments.

## Open questions / failure modes to watch

- Wall-clock cap is tight. Several experiments add compute per step
  (wider, deeper, finer slicing). Realized epoch counts will vary —
  important to compare experiments fairly (count realized epochs in
  results, not configured).
- 1499 training samples is small; capacity-increase arms (askeladd, fern)
  might overfit on `val_single_in_dist` while failing OOD (`val_geom_camber_*`).
  Watch the four split breakdown, not just the average.
- bfloat16 in tanjiro: low-magnitude `vol_loss` could underflow if the
  pressure normalization drives some channels near zero. Check NaNs in
  the JSONL.
- The two geometry-OOD splits (`val_geom_camber_rc`, `val_geom_camber_cruise`)
  are the hardest. A common-recipe winner should improve **both**, not just
  one — that's our gen-gap signal.

## Plateau / pivot plan (forward-looking, after round 1)

If round 1 lands within ~3% of each other (small effects) we have several
unused levers to escalate to:

- **Compound winners**: stack the best loss + best schedule + best
  capacity changes in round 2.
- **Data augmentation**: per-sample reflection (z → -z, AoA → -AoA,
  Uy → -Uy) for symmetric domains; per-sample Re-jitter for cross-regime
  generalization.
- **Domain re-weighting**: the current `WeightedRandomSampler` gives equal
  weight to single/raceCar-tandem/cruise. The val splits are 4×100 but
  three are tandem and one single; reweighting toward tandem may better
  match the metric.
- **Normalization choices**: per-domain normalization stats; robust
  (median/MAD) standardization for the pressure target tail; signed
  log-scaling for large Re pressure values.
- **Architecture switches**: graph transformer with kNN edges; FNO-style
  spectral mixing; per-channel decoder heads.
- **EMA on weights**: cheap, often helps in short-budget regimes.
- **Test-time augmentation**: average prediction over geometric symmetries.

If round 2 also plateaus, escalate to deeper architectural changes
(Geo-FNO, GNO, mesh-graph-net) and explore self-supervised pretext
losses on the volume field as auxiliary heads.

## Active in-flight PRs (round 1+2)

Status as of 20:52 UTC:

| PR | Student | Hypothesis | State |
|----|---------|-----------|-------|
| ~~#1355~~ | ~~alphonse~~ | ~~Pure L1 loss~~ | **MERGED** — new baseline 94.291 |
| ~~#1393~~ | ~~frieren~~ | ~~OneCycleLR (MSE era)~~ | Closed — superseded by loss change |
| **#1581** | **frieren** | **L1 + OneCycleLR compound (peak_lr 1e-3 vs 2e-3)** | Round-2 assignment, just dispatched |
| **#1582** | **alphonse** | **surf_weight sweep (5/10/20) on L1 baseline** | Round-2 assignment, just dispatched |
| #1381 | askeladd | Wider Transolver: n_hidden 128→256, mlp_ratio 2→4 | Training in progress (picked up ~19:50) |
| #1385 | edward | Finer physics attention: slice_num 64→128, n_head 4→8 | Training in progress (picked up ~19:53) |
| #1389 | fern | Deeper Transolver: n_layers 5→8 | Sent back for Arm C (`--epochs 9`, cosine T_max aligned). Awaiting push + rerun. |
| #1399 | nezuko | Surface channel weight sweep (corrected denominator) | 3-arm replan with `CHANNEL_W=[1,1,1]` control, `[1,1,2]`, `[1,1,3]`; actively training |
| #1405 | tanjiro | bfloat16 autocast + batch_size 8 + sqrt-scaled lr | Training in progress (picked up ~19:48) |
| #1410 | thorfinn | Multi-scale Fourier features for (x,z) coords | Training in progress (picked up ~19:52) |

**IMPORTANT for in-flight PRs:** All round-1 PRs were dispatched against an MSE baseline. Results
should be compared against the new L1 baseline (94.29). If any in-flight student's result beats
94.29, merge and update. The round-2 PRs (#1581, #1582) explicitly use `--loss l1`.

**Action items:** wait for in-flight PRs; prioritize merge of any that beat 94.29; review fern Arm C when it lands.
