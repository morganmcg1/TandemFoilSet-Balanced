# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-13
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** none received — this is a controlled 24/48h Charlie-vs-Willow logging ablation. Treated as research-isolated.

## Research focus

Round-1 baseline has shifted **six times in ~14 hours** through stacking compatible winners — cumulative **−49.7% val, −51.4% test**:

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 baseline |
| PR #1505 (Huber β=0.5 surf) | 2026-05-13 00:00 | 113.794 | 101.782 | −4.74% val, −7.19% test |
| PR #1715 (bf16 AMP) | 2026-05-13 02:00 | 89.597 | 79.907 | −21.3% val, −21.5% test |
| PR #1810 (torch.compile dynamic=True) | 2026-05-13 05:15 | 67.831 | 59.784 | −24.3% val, −25.2% test |
| PR #1910 (vol-Huber β=0.5) | 2026-05-13 07:30 | 65.469 | 57.837 | −3.5% val, −3.3% test |
| **PR #1692 (grad_clip max_norm=1.0)** | **2026-05-13 12:00** | **60.093** | **53.370** | **−8.2% val, −7.7% test** |

The current `train.py` now has **six** stacked changes: mask after slice softmax, Huber β=0.5 on surf and vol, bf16 autocast, `torch.compile(dynamic=True)`, and `clip_grad_norm_(max_norm=1.0)` before each optimizer step. **Still compute-bound (best=last on both PR #1692 seeds).**

**Pattern emerging from round 1:**
- **All six winners are orthogonal mechanisms** (correctness → loss(surf) → compute(bf16) → compute(compile) → loss(vol) → optimisation(grad_clip)). Cumulative val 119.45 → 60.09 (−49.7%).
- **Critical finding from #1692:** grad_clip max_norm=1.0 engages on **100% of training steps** (mean raw grad norm ~18-19). This is **global step-size normalisation**, not spike clipping. The balanced sampler's mesh-size heterogeneity (cruise vs raceCar) produces huge per-batch gradient variance; normalisation removes this variance from the effective LR.
- **All four test splits improved uniformly** — geom_camber_cruise −18%, re_rand −11% are the biggest movers.
- **New dominant research question:** what is the optimal max_norm? If 1.0 wins vs unclipped, and 1.0 is always clipping, is there a better threshold? fern → #2246 bisects with max_norm=5.0.
- **Loss-formulation wins stack:** surf-Huber (#1505, −4.7%) + vol-Huber (#1910, −3.5%) combined to −8% on val.
- **Scalar-capacity axis cluster firmly retired across all 3 baselines** (6 total failures).

Round 1 in-flight (8 PRs) — all must beat **val < 60.09, test < 53.37**:
- **#2246 fern (grad-clip max_norm=5.0 bisect)**: on grad-clip baseline (just assigned; bisect clip threshold between 1.0 winner and unclipped)
- **#2180 alphonse (dropout=0.1 in PhysicsAttention)**: on vol-Huber baseline, grad-clip heads-up posted
- **#2163 askeladd (per-channel β: β_p=0.25)**: on vol-Huber baseline, grad-clip heads-up posted
- **#2041 thorfinn (surf_weight 10 → 5)**: on vol-Huber baseline, grad-clip heads-up posted; code committed (11:12)
- **#2017 edward (weight_decay → 2e-4)**: sent back for bisection, grad-clip heads-up posted
- **#1940 frieren (batch_size=8 + sqrt-LR)**: on compile baseline, code committed (11:12), grad-clip heads-up posted
- **#1843 nezuko (Cosine T_max=35)**: on compile baseline, code committed (09:21), grad-clip heads-up posted
- **#1589 tanjiro (AdamW betas 0.9, 0.95)**: freshly rebased (11:19), grad-clip heads-up posted

**New merge bar: val < 60.09, test < 53.37, all four test splits finite.**

**Latest diagnostic finding (2026-05-13 03:00 from PR #1509 close):** The cosine schedule `T_max=MAX_EPOCHS=50` mis-tunes the LR decay to a never-reached horizon. At the bf16 baseline's 18 epochs, end-of-run LR is at ~81% of peak (4.07e-4 vs 5e-4) — the schedule never actually decays. PR #1843 isolates this as a single-axis test.

**Latest diagnostic finding (2026-05-13 03:45 from PR #1511 close on bf16):** depth=7 retry on bf16 still regresses compute-bound with best-val at the final epoch — the 18-epoch budget shrinks to 13 at +41% per-epoch overhead. Definitively closes the depth axis for round 1. Strengthens the portfolio rule that capacity moves should change *what* is computed (gating/attention reformulation), not scale existing scalar dimensions.

**Latest diagnostic finding (2026-05-13 05:20 from PR #1506 close on bf16):** width=192 retry on bf16 also regressed compute-bound (+19.4% val, +20.8% test). Still descending at 14/15 epochs when wall-clock timeout. Edward's analysis: +47% params × +28% per-epoch overhead ⇒ 15 epochs vs baseline 18. Cosine T_max=50 still mis-tuned. **6th scalar-capacity axis regression** (n_hidden=192 pre-mask AND on bf16; slice=128; n_layers=7 pre-bf16 AND on bf16; mlp_ratio=4 — see EXPERIMENTS_LOG.md). Scalar-capacity axis cluster is firmly retired.

**Critical implication post-#1810 merge (2026-05-13 05:15):** Both compile seeds finished best=last at epoch 35 — **compute is STILL binding even after −49% per-epoch speedup**. This makes lr-schedule alignment (nezuko #1843) and batch-size scaling (frieren #1940) the highest-value remaining compute-budget levers. The current `Config.epochs=50` with `T_max=50` cosine now leaves the model at ~lr_peak × (1 - sin²(0.5 × π × 35/50)) ≈ ~lr_peak × 0.7 at termination — still well above the cosine minimum that the model never reaches. Nezuko's T_max=18 target now shifts to T_max=35 on the compile baseline; heads-up posted.

**Portfolio re-evaluation 2026-05-13 05:15 (after #1810 merge):** the compute-bound axes that previously closed are re-evaluated yet again on the 35-epoch budget. Edward's #1939 tests mlp_ratio=4 (lowest per-epoch overhead, +18%) first. slice_num=128 remains a lower-priority round-2 revisit. n_layers=7 stays OUT (already failed twice). Width=192 fully retired (failed both pre-bf16 #1506 attempt and on-bf16 retry, both with edward's analysis matching). Portfolio rule "capacity should change *what* is computed" applies as the default; mlp_ratio=4 retry is the controlled exception.

## Round 1 portfolio (status)

| PR    | Student   | Hypothesis axis                  | Status |
|-------|-----------|----------------------------------|--------|
| #1504 | alphonse  | Mask-aware PhysicsAttention      | **MERGED** 21:52 (val=119.45, test=109.67) |
| #1505 | askeladd  | Huber surface loss (β=0.5)       | **MERGED** 00:00 (val=113.79, test=101.78) |
| #1506 | edward    | Wider hidden (128→192)           | CLOSED on bf16 (+19.4% val, +20.8% test — 6th compute-bound capacity-axis regression; width fully retired) |
| #1507 | fern      | More slices (64→128)             | CLOSED (compute-bound, +27%) — lower-priority bf16/compile-revisit candidate |
| #1508 | frieren   | surf_weight 10→25                | CLOSED (compute-bound, +16%) |
| #1509 | nezuko    | Warmup + lr=1e-3                 | CLOSED (+13.4% val on bf16; diagnostic surfaced T_max issue) |
| #1510 | tanjiro   | Fourier pos enc (L=6)            | CLOSED (cruise NaN, pre-mask) |
| #1511 | thorfinn  | Deeper (5→7 layers)              | CLOSED on bf16 (+19.5% val, +19.8% test — depth axis closed) |
| #1589 | tanjiro   | AdamW betas (0.9, 0.95)          | WIP, pre-mask, compile heads-up posted (4th baseline shift) |
| #1623 | alphonse  | mlp_ratio 2→4                    | CLOSED (compute-bound, +18% val) — retest in progress as #1939 |
| #1692 | fern      | Gradient clipping (max_norm=1.0) | **MERGED** 12:00 (val=60.09, test=53.37) — **6th baseline shift, −8.2% val, −7.7% test**; 100% clip rate reveals global step-size normalisation mechanism |
| #2246 | fern      | Grad-clip bisect: max_norm=5.0   | WIP, grad-clip baseline (just assigned; bisect the clip threshold) |
| #1712 | askeladd  | Huber β=0.25 (β-tune)            | CLOSED (+6.6% val on bf16; bounds β from below) |
| #1715 | frieren   | bf16 mixed-precision (AMP)       | **MERGED** 02:00 (val=89.60, test=79.91) |
| #1735 | alphonse  | SwiGLU FFN (matched params)      | CLOSED (stuck — 22 pod restarts, 0 commits in 10h; reset, not verdict on SwiGLU) |
| #2180 | alphonse  | Dropout p=0.1 in PhysicsAttention | WIP, vol-Huber baseline (just assigned; orthogonal to all in-flight) |
| #1810 | frieren   | torch.compile (dynamic=True)     | **MERGED** 05:15 (val=67.83, test=59.78) — largest single-axis win of round 1 |
| #1843 | nezuko    | Cosine T_max=18 (not 50)         | WIP, bf16 baseline — heads-up to retarget T_max=35 on compile |
| #1882 | askeladd  | Huber β=0.75 (β-tune from above) | CLOSED (+8.6% val, +10.0% test — β-axis fully bracketed: 0.25 fails, 0.5 optimum, 0.75 fails) |
| #1910 | thorfinn  | Volume Huber β=0.5               | **MERGED** 07:30 (val=65.47, test=57.84) — OOD splits drove win; in-dist regressed slightly; zero compute overhead |
| #1939 | edward    | mlp_ratio 2→4 retry on compile   | CLOSED on compile (+5.8% val, +6.6% test — 6th compute-bound capacity-axis regression; scalar-capacity cluster now firmly retired across all 3 baselines) |
| #1940 | frieren   | batch_size=8 + sqrt-LR (lr=7e-4) | WIP, current compile baseline — vol-Huber heads-up posted |
| #2017 | edward    | weight_decay 1e-4 → 5e-4         | SENT BACK for wd=2e-4 bisection (val +0.67% miss; in-dist improves, rc OOD regresses — over-regularization signature) |
| #2163 | askeladd  | Per-channel β: β_p=0.25, β_Ux=β_Uy=0.5 | WIP, vol-Huber baseline (just assigned) |
| #2041 | thorfinn  | surf_weight 10 → 5               | WIP, vol-Huber baseline (just assigned; re-calibrate after vol-Huber shifted gradient balance) |

**Merged:** 6 (mask-aware, Huber β=0.5 surf, bf16, compile, vol-Huber β=0.5, **grad_clip max_norm=1.0**). **Closed:** 11. **Open:** 8 (tanjiro #1589, frieren #1940, nezuko #1843, thorfinn #2041, edward #2017, askeladd #2163, alphonse #2180, **fern #2246**).

**Scalar-capacity axis cluster fully retired across THREE baselines.** All four scalar-capacity dimensions (n_hidden, n_layers, slice_num, mlp_ratio) have now been compute-bound at least once; both retries on the compile baseline (#1506 width, #1939 mlp_ratio) regressed. The portfolio rule "capacity should change *what* is computed, not scale existing components" has the strongest empirical support of any round-1 finding (7 total negative results across the cluster). Future capacity wins need to come from capacity-shape moves: alphonse's #1735 SwiGLU is the lone such axis in flight.

## Potential next research directions

Confirmed winners so far (all four stack): correctness (mask) + loss (Huber) + compute (bf16) + compute (compile). Likely follow-ups:

- **Compute is STILL binding at 35 epochs** (both #1910 vol-Huber seeds best=last, same as compile). Highest-EV remaining levers: lr-schedule alignment (#1843 nezuko T_max=35), batch-size scaling (#1940 frieren bs=8), possibly max-autotune compile mode.
- **Loss-formulation wins have stacked:** surf-Huber (#1505, −4.7%) + vol-Huber (#1910, −3.5%) = −8% combined. The OOD splits continue to be where Huber suppression helps most. β=0.75 (#1882 askeladd) is in-flight as a tune above the β=0.5 optimum.
- **surf_weight re-calibration is the natural next step** after vol-Huber (#2041 thorfinn): surf_weight=10 was calibrated with noisy-MSE vol; now that vol uses Huber, the 10× upweighting may over-emphasize surf. vol-Huber's in-dist surf_p +3.8% regression is the key motivating signal.
- **Scalar-capacity axis is CLOSED for round 1.** All 4 dimensions tried, all failed; no further retries.
- **If weight_decay=5e-4 wins (#2017 edward):** regularization was undertuned for the 35-epoch budget; follow-up with lr rescale.
- **β-axis CLOSED** (#1882 askeladd β=0.75 failed +8.6%/+10.0%, symmetric with β=0.25 failure). β=0.5 is the global optimum. Per-channel β (#2163 askeladd) is the active next test in this loss-shape family.
- **grad_clip MERGED (#1692, −8.2% val):** 100% clip rate = global step-size normalisation. fern #2246 bisects with max_norm=5.0 to find the sweet spot. After bisect: consider coupling with LR rescale (if max_norm=N always clips, the effective LR is lr × N/raw_norm — explicit LR tuning becomes redundant).
- **If dropout=0.1 wins (#2180 alphonse):** attention-layer regularization stacks with grad_clip; follow-up with dropout sweep (0.05, 0.15). SwiGLU stays as round-2 capacity-shape candidate.
- **If AdamW betas (#1589 tanjiro), Cosine T_max (#1843 nezuko), batch_size (#1940 frieren) land:** harvest and stack.

Round-2 priority queue (post-round-1-cleanup):

**Compute-bound revisits on compile baseline:** **CLOSED for round 1.** Three scalar-capacity retries on the compile baseline (mlp_ratio=4 via #1939, n_hidden=192 via #1506, and previously slice_num/depth across baselines) all regressed. The cluster is fully retired.
- **n_layers=7 OUT.** Failed twice; +41% per-epoch overhead too much.
- **n_hidden=192 OUT.** Failed twice (pre-mask + on-bf16); +28% per-epoch overhead, 1.47M params.
- **mlp_ratio=4 OUT** (#1939 closed, +5.8% val on compile baseline).
- **slice_num=128 OUT.** Lower priority than mlp_ratio originally; since mlp_ratio (lower overhead) failed, slice_num (higher overhead) inherits the closed verdict.

**Larger swings if round-1+round-2-revisits plateau:**
- **Surface-anchored cross-attention** (boundary nodes as queries against volume tokens) — directly addresses the "surface inherits from volume" structural relationship. Promising as the next round-2 architectural move.
- **Mirror/flow-symmetry data augmentation** — needs careful per-feature signed-column flipping; cruise samples (freestream BCs) admit y-flip + Uy-negation cleanly; raceCar samples (ground effect, asymmetric BCs) do not. Conditional augmentation by domain is feasible but adds complexity.
- **Per-sample Re normalization** (Reynolds-aware feature embedding) — re_rand and cruise are where compile gave the largest test gains (−23.7% / −32.0%), suggesting Re-dependent error structure remains.
- **Quantile / Pinball loss** — more aggressive median-targeting than Huber if β-tune still leaves gains on the table.
- **OneCycleLR / longer cosine warmup** — alternative schedule for the now-35-epoch budget; lower priority if nezuko's T_max=35 lands.
- **max-autotune-no-cudagraphs torch.compile mode** — frieren's #3 suggestion; potential further compile-time speedup at cost of variable JIT time.
- **Cruise data sanity check** (`scoring.py` cruise sample 20 inf-pressure) — mask currently hides the upstream data issue; worth a dedicated PR to either clean or document.

## Open questions and ruled-out paths

- **Strong signal (2026-05-12 21:15):** PR #1504 (mask-aware PhysicsAttention) is the only finished round-1 run with a populated `test_avg/mae_surf_p` — all other finished baselines hit `test_geom_camber_cruise=None`. This includes the Fourier PR #1510 (closed) and the finished runs of #1506/#1508/#1511. Strong evidence that the baseline's unmasked slice softmax produces inf/NaN on at least one cruise test sample, and mask-aware attention fixes it. **PR #1504 is now the highest-priority merge candidate** of round 1 — both a metric improvement and a correctness fix on the paper-facing metric.
- **Open:** Is the loss/metric mismatch (MSE training vs MAE evaluation) actually a big lever, or has the surface weight already absorbed it?
- **Open:** Is the dataset bottleneck on the geometry-camber holdouts (M=6-8 raceCar, M=2-4 cruise) inductive-bias-bound, or capacity-bound?
- **Resolved:** Padding in `pad_collate` does produce measurable noise — see PR #1504 evidence above.
- **Ruled-out:** Fourier positional encoding (PR #1510, both `pos_scale=1.0` and `pos_scale=0.1`) cannot be fairly evaluated until the cruise NaN is resolved. Not a Fourier-spectrum problem.

## Operating notes

- `SENPAI_TIMEOUT_MINUTES=30` per run; `SENPAI_MAX_EPOCHS` unset (defaults to `cfg.epochs=50`). Most runs will exit on wall clock first.
- All runs grouped in W&B project `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r3` with `--wandb_group` set per hypothesis family.
- This launch is isolated to `icml-appendix-willow-pai2g-48h-r3` and the 8 assigned student PR branches. No cross-launch comparison.
