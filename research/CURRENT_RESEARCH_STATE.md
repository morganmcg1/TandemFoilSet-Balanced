# SENPAI Research State

- **Date**: 2026-05-15 18:35
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup essentially done; round 2 actively iterating. Three round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0. Three-axis stack — never measured end-to-end on any single run.

**Systemic finding (3-way confirmed)**: the binding constraint on this track is **per-epoch wallclock at the wider trunk**, not model architecture or loss formulation. At `n_hidden=192`, only 8-9 epochs fit in the 30-min cap (vs 14 at narrower); the wider trunk's capacity advantage can't manifest. Three independent PRs (#3273 ReScaler, #3298 p_surf_weight, #3144 deeper-l8) all closed with the same pattern.

**bf16 result (#3332, 18:20)**: ~30% per-epoch speedup landed as predicted (205s→143.7s, 8→12 epochs realized). Clean unlock: zero NaN/Inf, per-channel ratios intact, all 4 test splits finite. **But val_avg = 129.76 (+2.7% over baseline)** — primarily concentrated in val_geom_camber_rc (+12.3 vs baseline; others within noise). Wider trunk still mid-descent at 12 epochs. **Sent back for bs=8 retry**: at peak memory 49/96 GB, bs=4→8 should give ~1.7-1.9× per-epoch speedup → 21-24 epochs realized at wider trunk, the cleanest path to validate the wider+EMA+surf_25 stack actually beats narrow baseline.

## PRs in-flight

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup + cosine annealing, budget-aligned (post-#3136 stack) | 1→rerun | running |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun — completed training, results pending | 1→rerun | training done |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebased rerun) | 1→rerun | training done |
| #3277 | nezuko | Output decoupling | Separate Linear→GELU→Linear head per channel (Ux, Uy, p) | 2 | training (2nd cycle) |
| #3287 | frieren | Domain conditioning | Per-sample FiLM (scale, shift) on LayerNorm from gap+AoA features | 2 | recovering (no training yet) |
| #3332 | edward | Throughput + batch | bf16 + bs=8 retry (after bf16-alone +2.7% val regression) | 2→rerun | sent back |
| #3336 | fern | Optimization stability | Global gradient norm clipping (`max_norm=1.0`) | 2 | training done |
| #3378 | thorfinn | System fix | NaN-safe `accumulate_batch` (advisor waiver on `data/scoring.py`) | 2 | dispatched |

All 8 students have active PRs. **Zero idle GPUs.** 4 students completed training and are expected to post results imminently (askeladd, tanjiro, fern, edward retry). Nezuko started a second training cycle.

## Recent decisions

- **#3273 (edward ReScaler rebased) CLOSED**: val=152.79 vs baseline 126.32 = +21% regression. ReScaler genuinely works at matched-epoch (+3%) but wider trunk only fits 8 epochs vs narrower's 14. Schedule/budget tradeoff dominates.
- **#3298 (fern p_surf_weight) CLOSED**: val=158.54 vs baseline 126.32 = +25% regression. Same wider-trunk-budget pattern as #3273 — matched-epoch (158.54 vs 158.96) is approximately **neutral**.
- **#3144 (thorfinn deeper-l8) CLOSED**: val=177.60 vs baseline 126.32 = +40% regression. Third independent confirmation of the wider-trunk-budget wall; n_layers=8 also drove VRAM to ceiling (96.61/96 GB). Depth-axis is gated on bf16 landing first.
- **#3332 (edward bf16) SENT BACK**: val=129.76 (+2.7%, can't merge), test=117.76 (paper-facing wins). Infrastructure unlock clean (30% speedup, zero NaN/Inf, per-channel intact). Sent back with bs=8 + epochs=22 + T_max=22 recipe.
- **#3378 (thorfinn scoring-fix) DISPATCHED**: advisor-waivered system fix to `data/scoring.py`. Replaces `err * mask` with `torch.where(mask, err, 0)` — eliminates Inf×0=NaN propagation. Unblocks `test_avg/mae_surf_p` reporting paper-wide.
- **#3121 (alphonse warmup) confirmation posted**: both deviations (rebase + budget-align) approved. Alphonse's pod just restarted (iter 1 at 18:21) and is freshly starting.

## Systemic constraints (known issues)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; with bf16 (now validated) gives 12 epochs. bf16 + bs=8 (in-flight retry) should give 21-24 epochs, the first time cosine fully anneals on the wider trunk. Once that lands, the schedule alignment story is finally complete.
2. **Cruise-test NaN** — fix in flight via thorfinn #3378 (advisor waiver). 5-way diagnosed; tanjiro's element-mask pattern is the cleanest workaround and what the fix adopts. Once #3378 merges, `test_avg/mae_surf_p` will be a finite 4-split mean across all PRs (no more "3-finite-split mean" workarounds).

## Round 2 stacking plan

Dispatched (round 2):
- nezuko #3277: Separate per-channel heads (Idea 12)
- frieren #3287: Domain-conditional FiLM (Idea 13)
- edward #3332: bf16 + bs=8 retry (Idea 3 + Idea throughput) — primary throughput unlock
- fern #3336: Gradient norm clipping
- thorfinn #3378: Scoring fix (system hygiene, paper-facing)

Priority candidates if students free up:

1. **OneCycleLR (Idea 18)** — alternative scheduler if alphonse's warmup+cosine disappoints.
2. **Curriculum: low-Re first (Idea 16)** — upweight low-Re in WeightedRandomSampler for first 30% of epochs.
3. **Stochastic depth (Idea 14)** — pairs naturally with bf16-unlocked deeper sweep (n_layers 5/6/7/8).
4. **Temperature scheduling on PhysicsAttention (Idea 10 — revised)** — temperature is already learnable per-head; hypothesis would be to add an explicit annealing multiplier on top.
5. **Larger model (Idea 7)** — push to n_hidden=256 if bf16+bs=8 confirms wider trunk capacity hypothesis.
6. **Depth sweep at bf16+bs=8** — replay n_layers=6/7/8 once budget is unblocked. Thorfinn's analysis on #3144 (5.3 min/epoch at n_layers=8, mid-descent at 6/8 epochs) makes this a high-value follow-up.
7. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun (in-flight) confirms SmoothL1 wins.
8. **Aux task: predict is_surface boundary indicator** — boundary structure is implicit in the model; explicit aux head could give cleaner pressure gradients.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot: a new backbone family before more hyperparameter tuning. FiLM is the most architectural of the round-2 set; if it fails, that's the strongest signal that we need to think bigger.
- If round-2 PRs reveal the orthogonality assumption is failing (e.g. multiple round-2 PRs combined with current stack underperform a clean baseline by >3%) → run a clean combined-baseline measurement.
- **If bf16+bs=8 retry (#3332 retry) still doesn't beat 126.32**, the wider-trunk thesis itself is wrong on this dataset — narrow trunk + bf16 (giving ~18-20 epochs in cap) would then be the right path. We'd revert to narrow as the merged config.
- If cruise-test NaN persists post-#3378 → re-examine the fix; possible that `aggregate_splits` or `finalize_split` has a secondary issue.
