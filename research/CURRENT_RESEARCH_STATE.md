# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-13
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** none received — this is a controlled 24/48h Charlie-vs-Willow logging ablation. Treated as research-isolated.

## Research focus

Round-1 baseline has shifted **four times in <11 hours** through stacking compatible winners — cumulative −43.2% val, −45.5% test:

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 baseline |
| PR #1505 (Huber β=0.5) | 2026-05-13 00:00 | 113.794 | 101.782 | −4.74% val, −7.19% test |
| PR #1715 (bf16 AMP) | 2026-05-13 02:00 | 89.597 | 79.907 | −21.3% val, −21.5% test |
| **PR #1810 (torch.compile dynamic=True)** | **2026-05-13 05:15** | **67.831** | **59.784** | **−24.3% val, −25.2% test** |

The current `train.py` has all four stacked: mask after slice softmax (~line 125), Huber β=0.5 at train+eval (lines 260, 508), bf16 autocast wrapping the forward in eval+train (lines 255, 506), and `torch.compile(model, dynamic=True)` after model instantiation (line 451). Best epoch shifted 13 → 17/18 (bf16) → 35/35 (compile). **Still compute-bound at 35 epochs on both compile seeds (best=last).**

**Pattern emerging from round 1:**
- **All four winners are orthogonal mechanisms** (correctness → loss formulation → compute → compute). They stack additively — cumulative val 119.45 → 67.83 (−43.2%).
- **Compile (#1810) is the biggest single-axis jump**, larger even than bf16. The mechanism is super-linear: small Transolver + bs=4 was heavily Python/kernel-launch bound, AND bf16 had already left val descending — so 2× per-epoch speedup + 17 extra epochs on a still-descending curve compounds.
- **Best=last at 35 epochs on both compile seeds** → compute remains the binding constraint even at the doubled epoch budget. More compute-side levers (bs=8 — frieren #1940; max-autotune — TBD) are still profitable. Schedule-alignment fixes (cosine T_max → 35) are now especially valuable.
- **Scalar-capacity axis cluster is firmly retired:** six failures across two baselines (n_hidden=192 pre-mask AND on bf16 in #1506, slice=128, n_layers=7 pre-mask AND on bf16 in #1511, mlp_ratio=4 pre-bf16). The portfolio rule "capacity moves should change *what* is computed, not scale existing components" has strong empirical support.
- The `test_geom_camber_cruise` NaN issue is fully resolved by the mask fix; bf16 + compile did not re-introduce it.

Round 1 in-flight (8 PRs):
- **#1735 alphonse (SwiGLU FFN)**: from Huber baseline, pre-bf16 — compile heads-up posted (third baseline shift on this branch)
- **#1589 tanjiro (AdamW betas)**: from pre-mask — compile heads-up posted (fourth baseline shift)
- **#1692 fern (grad_clip=1.0)**: from mask-aware baseline, pre-Huber — compile heads-up posted (third baseline shift on this branch)
- **#1843 nezuko (Cosine T_max=35)**: bf16 baseline, T_max confirmed at 35 on compile
- **#1882 askeladd (Huber β=0.75)**: bf16 baseline — compile heads-up posted (one baseline shift)
- **#1910 thorfinn (Volume Huber β=0.5)**: bf16 baseline — compile heads-up posted (one baseline shift)
- **#1940 frieren (batch_size=8 + sqrt-LR scaling)**: current compile baseline (follow-up #2 from frieren's #1810)
- **#2017 edward (weight_decay 1e-4 → 5e-4)**: current compile baseline (just assigned; compute-cheap regularization-axis test after #1939 retired the scalar-capacity axes)

All got compile heads-up. **New merge bar: val < 67.83, test < 59.78, all four test splits finite.**

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
| #1692 | fern      | Gradient clipping (max_norm=1.0) | WIP, post-mask pre-Huber, compile heads-up posted (3rd baseline shift); pod 13 restarts |
| #1712 | askeladd  | Huber β=0.25 (β-tune)            | CLOSED (+6.6% val on bf16; bounds β from below) |
| #1715 | frieren   | bf16 mixed-precision (AMP)       | **MERGED** 02:00 (val=89.60, test=79.91) |
| #1735 | alphonse  | SwiGLU FFN (matched params)      | WIP, pre-bf16, compile heads-up posted (3rd baseline shift) |
| #1810 | frieren   | torch.compile (dynamic=True)     | **MERGED** 05:15 (val=67.83, test=59.78) — largest single-axis win of round 1 |
| #1843 | nezuko    | Cosine T_max=18 (not 50)         | WIP, bf16 baseline — heads-up to retarget T_max=35 on compile |
| #1882 | askeladd  | Huber β=0.75 (β-tune from above) | WIP, bf16 baseline, compile heads-up posted |
| #1910 | thorfinn  | Volume Huber β=0.5               | WIP, bf16 baseline, compile heads-up posted |
| #1939 | edward    | mlp_ratio 2→4 retry on compile   | CLOSED on compile (+5.8% val, +6.6% test — 6th compute-bound capacity-axis regression; scalar-capacity cluster now firmly retired across all 3 baselines) |
| #1940 | frieren   | batch_size=8 + sqrt-LR (lr=7e-4) | WIP, current compile baseline |
| #2017 | edward    | weight_decay 1e-4 → 5e-4         | WIP, current compile baseline (just assigned) |

**Merged:** 4 (mask-aware, Huber β=0.5, bf16, compile). **Closed:** 9 (Fourier #1510, slice=128 #1507, surf_weight=25 #1508, mlp_ratio=4-pre-bf16 #1623, warmup+lr=1e-3 #1509, β=0.25 #1712, depth=7-on-bf16 #1511, width=192-on-bf16 #1506, **mlp_ratio=4-on-compile #1939**). **Open:** 8 (3 needing rebase from various baselines + 4 on bf16/compile baselines + 1 just-assigned: edward #2017 wd=5e-4).

**Scalar-capacity axis cluster fully retired across THREE baselines.** All four scalar-capacity dimensions (n_hidden, n_layers, slice_num, mlp_ratio) have now been compute-bound at least once; both retries on the compile baseline (#1506 width, #1939 mlp_ratio) regressed. The portfolio rule "capacity should change *what* is computed, not scale existing components" has the strongest empirical support of any round-1 finding (7 total negative results across the cluster). Future capacity wins need to come from capacity-shape moves: alphonse's #1735 SwiGLU is the lone such axis in flight.

## Potential next research directions

Confirmed winners so far (all four stack): correctness (mask) + loss (Huber) + compute (bf16) + compute (compile). Likely follow-ups:

- **Compute is STILL binding at 35 epochs** (both compile seeds best=last). Highest-EV remaining levers: lr-schedule alignment (#1843 with T_max=35), batch-size scaling (#1940), and possibly max-autotune-no-cudagraphs (frieren's #1810 follow-up #3).
- **Scalar-capacity axis is CLOSED for round 1.** #1939 retired mlp_ratio=4 on compile (+5.8% val); all 4 scalar-capacity dims have regressed at least once. No further scalar-capacity retries.
- **If weight_decay=5e-4 wins (#2017 edward):** indicates the optimizer/regularization axis was undertuned for the new 35-epoch budget. Follow-up: per-block weight decay (decouple FFN vs attention), lr decoupled from wd via AdamW.
- **If Huber β-tuning wins (#1882 askeladd):** sweep β around the optimum, consider per-channel β for surface p vs Ux vs Uy (different normalized scales). Pairs with Volume Huber (#1910 thorfinn) if both win.
- **If grad_clip wins (#1692 fern):** explore weight decay tuning and LR revisits, since clipping decouples optimizer stability from those.
- **If SwiGLU FFN wins (#1735 alphonse):** the gating mechanism's success would suggest other modern transformer-FFN moves are worth trying (e.g. GeGLU variant, larger gating dimension, per-block residual gating). This is now the LONE capacity-shape hypothesis in flight after the scalar-capacity cluster's retirement.
- **If AdamW betas (#1589 tanjiro), Cosine T_max (#1843 nezuko), batch_size (#1940 frieren) land:** harvest and stack; these are all schedule/optimizer levers that are mostly orthogonal to architecture.

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
