# SENPAI Research State

- 2026-04-29 11:30 (round 1 in flight, branch `icml-appendix-charlie-pai2f-r1`)
- No human researcher directives yet for this branch.
- Track: `charlie-pai2f-r1`, 8 students, 1 GPU each, 30 min/run, max 50 epochs effective.

## Round 1 status

| PR | Student | Hypothesis | Status | best val_avg/mae_surf_p |
|---|---|---|---|---|
| #1092 | alphonse | capacity-scale-up | sent back (n_h=160 only) | 168.749 |
| #1094 | askeladd | surf-weight-25 | sent back (bs↑, rebase) | 134.368 |
| #1095 | edward | pressure-channel-weight | sent back (formula) | 133.892 |
| #1096 | fern | huber-vol | wip | — |
| #1097 | frieren | slice-num-128 | sent back (bs↑, clamp) | 162.562 |
| #1099 | nezuko | lr1e-3-warmup5 | wip | — |
| #1100 | tanjiro | wider-bs8 (fallback bs=5) | sent back (mlp_ratio↓, clamp) | 165.304 |
| #1101 | thorfinn | warmup-cosine-floor | sent back (T_max=13, warmup=1) | 142.886 |

## Cross-experiment learnings so far

1. **30-min budget is the binding constraint.** All 5 finished runs hit timeout: edward 14/50, askeladd 14/50, frieren 11/50, tanjiro 8/50, alphonse 7/50. None reached the cosine LR low-LR phase. Per-epoch wall clock ranges from ~130s (baseline shape) to ~277s (n_hidden=192/layers=6/mlp_ratio=4, bs=3). **Lever: anything that buys more epochs in 30 min compounds with capacity changes.**
2. **Compute-cost asymmetry across capacity axes.** mlp_ratio dominates activation memory because it widens the MLP intermediate; layers and width compound multiplicatively in attention. From observed VRAM peaks: width-only is cheap, depth + width is moderate, depth × width × mlp_ratio is prohibitive. **Implication for round-2:** when stacking capacity, scale width first, layers second, mlp_ratio last.
3. **VRAM utilization varies wildly.** Edward and askeladd at default architecture used ~42 GB of 95 GB; frieren used 54 GB at slice_num=128. Tanjiro hit 92 GB at n_hidden=256+bs=5; alphonse hit 92 GB at 192/6/6/4 + bs=4. **Default architecture has roughly 2× bs headroom** that nobody is using — this is the highest-leverage round-1 fix for non-capacity hypotheses.
4. **Test pressure NaN is a multi-failure mode.**
   - **Mode A (data):** `test_geom_camber_cruise/000020.pt` has +Inf in p ground truth — exposed by `data/scoring.py` mask-multiply propagating NaN. **Branch-side fix applied** via `torch.where`-based masking. Confirmed independently by edward (#1095), frieren (#1097), askeladd (#1094), alphonse (#1092).
   - **Mode B (model):** wider tanjiro and undertrained alphonse produced fp32 overflow in pred_p on a cruise inference sample, blowing up vol_loss to +Inf. Output-side pressure clamping is the right fix; requested for tanjiro and frieren. Alphonse's narrower retry should fix it via more epochs alone.
5. **Surface-loss reweighting helps OOD splits.** Askeladd's surf_weight=25 run beat edward on val_re_rand and val_geom_camber_cruise, lost on val_single_in_dist. Even though aggregate val_avg is tied, this is exactly where surface boosting *should* help — the OOD splits whose paper-facing test_avg is what matters.
6. **Best so far: 133.89.** Provisional, edward only — and it's a confounded run (loss formula softened aggregate surface signal ~3×). Askeladd is at 134.4 (clean methodology, throughput-limited). True round-1 winner not yet decided.
7. **Schedule hyperparameters must be matched to the achievable horizon, not the nominal one.** Thorfinn's warmup-cosine-floor used `T_max = MAX_EPOCHS - warmup = 45`, but the 30-min cap halts training at ~14 epochs. Result: cosine traverses ~20% of its trajectory and `eta_min` floor is unreachable — the mechanism never fires. Round-2 schedule experiments must derive `T_max` from observed per-epoch wall-clock (~131 s/epoch at default arch) and the 30-min budget, not from the 50-epoch nominal cap.
8. **Schedule run-to-run variance is ~12%.** Two identical thorfinn runs hit 124.29 and 142.89. For low-effect-size hypothesis tests (predicted ±2-5%), single-run comparisons are below the noise floor. Round-2 implication: prioritize hypotheses with predicted larger effects, OR run multi-seed for borderline schedule/optim tweaks.

## Branch-side fixes

- **`data/scoring.py` NaN-propagation bug.** Multiply-mask let NaN ground-truth p
  values bleed past the sample-level filter, producing NaN
  `test_avg/mae_surf_p` whenever any test sample has non-finite y. Fixed via
  `torch.where`-based masking on the advisor branch (committed alongside this
  state update). In-flight student runs that finish before they can rebase will
  still report NaN test pressure on `test_geom_camber_cruise`; their val
  numbers are unaffected. Merge winners on val_avg, treat test_avg as paper
  number that will need rerun if NaN. **Confirmed independently by edward (PR
  #1095) and frieren (PR #1097).**

## Current research focus

Round 1 establishes a balanced sweep across the main optimization levers for the
default Transolver baseline on TandemFoilSet. The eight assignments cover three
families:

1. **Capacity scaling** — `alphonse` (n_hidden 192, layers 6, mlp_ratio 4),
   `tanjiro` (n_hidden 256 + bs 8), `frieren` (slice_num 128).
2. **Loss / metric alignment** — `askeladd` (`surf_weight 25`), `edward`
   (per-channel pressure-weighted surf loss), `fern` (Huber on volume).
3. **Optimization discipline** — `nezuko` (lr 1e-3 + warmup), `thorfinn`
   (warmup + non-zero cosine floor at default lr).

The intent is to pin down which lever moves `val_avg/mae_surf_p` most, then in
later rounds stack the winning levers and explore architecturally bolder
follow-ups (Fourier features, neural operator hybrids, attention variants,
physics-informed losses, EMA / SWA averaging, etc.).

## Next research directions (post-round-1 candidates)

- **Stack winners.** Whichever capacity, loss, and schedule changes win get
  combined into a single recipe and re-tested.
- **Per-domain specialization.** Inspect per-split metrics — if camber-rc /
  camber-cruise behave differently from re_rand, explore conditioning or domain
  embedding (e.g. learned domain token concatenated to features).
- **Geometry-aware augmentation.** Random AoA reflection (sign flip + y-axis
  flip), light positional jitter, or NACA cambered-thickness perturbation could
  expand effective sample count without changing the data contract.
- **Spectral / Fourier features.** Random Fourier features on x[:, 0:2] (node
  positions) often boost mesh-based surrogates in CFD.
- **Loss reformulation.** Sobolev-style loss (gradient matching), per-sample
  scale-aware losses (divide errors by sample y_std), pressure-only auxiliary
  head with a stronger weight.
- **Optimizer swap.** Lion or AdEMAMix as alternatives to AdamW, especially if
  capacity-scaling wins because larger models often respond better to
  alternative optimizers.
- **Sampler tweaks.** Sampling weighted by per-sample y_std (high-variance
  samples seen more often) to attack heavy-tailed errors.
- **Model averaging.** EMA of weights with decay 0.999, evaluating EMA at val
  time for noise-robust generalization.

## Constraints reminder

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary metric: `val_avg/mae_surf_p`; test metric: `test_avg/mae_surf_p`.
