# SENPAI Research State

- **Date:** 2026-05-12 (last update 19:55 UTC)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r1`
- **Research tag:** `charlie-pai2g-48h-r1` (Charlie no-W&B logging-ablation arm,
  48h run)
- **Most recent human directive:** None — fresh launch, no human issues in
  the queue.

## Round-1 operational notes

- **GitHub API rate-limit storm 18:30–19:50 UTC.** Concurrent gh polling
  across all 24-h + 48-h arms briefly exceeded the API quota. Students hit
  "GraphQL: API rate limit already exceeded" on `gh pr list` for ~80 min and
  were unable to see their assignments during that window. Effect: 5/8
  round-1 students (askeladd, edward, fern, tanjiro, thorfinn) did not start
  training until 19:50+. Frieren and alphonse trained but couldn't push
  results before the rate-limit window ended — local metrics may have been
  lost on the next `git reset`. Nezuko completed and pushed their first
  2-arm sweep at 19:09 (before the worst of the rate limit hit).
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

## Active in-flight PRs (round 1)

Status as of 19:55 UTC:

| PR | Student | Hypothesis | State |
|----|---------|-----------|-------|
| #1355 | alphonse | Smooth L1 / pure L1 vs MSE on normalized residuals | Trained smooth-L1 17:59, posted NaN-bug diagnosis 19:06; advisor reply 19:55 with recovery instructions and 3-of-4 reporting convention |
| #1381 | askeladd | Wider Transolver: n_hidden 128→256, mlp_ratio 2→4 | Just picked up assignment 19:50; training pending |
| #1385 | edward | Finer physics attention: slice_num 64→128, n_head 4→8 | Just picked up assignment 19:53; training pending |
| #1389 | fern | Deeper Transolver: n_layers 5→8 | Just picked up assignment 19:50; training pending |
| #1393 | frieren | OneCycleLR with warmup replacing CosineAnnealingLR | Trained 18:02–18:31; Claude student turn stuck since 18:51 (watchdog reports poll-failed → leaving Claude running); local result may need re-run |
| #1399 | nezuko | Surface loss: pressure channel weight 2× + surf_weight sweep | First sweep finished 19:09 (val_avg 111.80 winner), sent back for fixed `.mean()`-denominator 3-arm replan; now actively training (GPU 99%) |
| #1405 | tanjiro | bfloat16 autocast + batch_size 8 + sqrt-scaled lr | Just picked up assignment ~19:48; training pending |
| #1410 | thorfinn | Multi-scale Fourier features for (x,z) coords | Just picked up assignment 19:52; training pending |

**Action items:** wait for round-1 training to complete; review and merge/send-back as terminal SENPAI-RESULTs land. No idle students at the moment.
