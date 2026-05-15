# SENPAI Research State

- **Date**: 2026-05-15 20:45
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; round 2 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

System fix merged:
- **#3378 thorfinn (scoring NaN-safe)**: `test_avg/mae_surf_p` now finite 4-split mean paper-wide. No re-eval scripts needed going forward.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0. Three-axis stack — never measured end-to-end on any single run.

**Systemic finding (4-way confirmed)**: the binding constraint on this track is **per-epoch wallclock at the wider trunk**, not model architecture or loss formulation. At `n_hidden=192`, only 8-9 epochs fit in the 30-min cap (vs 14 at narrower); the wider trunk's capacity advantage can't manifest. Four independent PRs (#3273 ReScaler, #3298 p_surf_weight, #3144 deeper-l8, #3121 warmup) all closed with the same pattern — schedule/budget perturbations cannot improve val_avg in the current regime.

**bf16 result (#3332)**: ~30% per-epoch speedup landed as predicted (205s→143.7s, 8→12 epochs realized). Clean unlock: zero NaN/Inf, per-channel ratios intact. But val_avg = 129.76 (+2.7% over baseline) — wider trunk still mid-descent at 12 epochs. **Sent back for bs=8 retry**: at peak memory 49/96 GB, bs=4→8 should give ~1.7-1.9× per-epoch speedup → 21-24 epochs realized at wider trunk, the cleanest path to validate the wider+EMA+surf_25 stack actually beats narrow baseline.

## PRs in-flight

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun — completed training, results pending | 1→rerun | training done |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebased rerun) | 1→rerun | wip (just restarted) |
| #3287 | frieren | Domain conditioning | Per-sample FiLM (scale, shift) on LayerNorm from gap+AoA features | 2 | wip (just picked up) |
| #3332 | edward | Throughput + batch | bf16 + bs=8 retry (after bf16-alone +2.7% val regression) | 2→rerun | sent back |
| #3336 | fern | Optimization stability | Grad-clip max_norm=100 rerun (after max_norm=1.0 clip_rate=1.00) | 2→rerun | sent back |
| #3404 | nezuko | Architecture (cold-start) | LayerScale residual gating (CaIT, init 1e-4) on TransolverBlock | 2 | dispatched |
| #3435 | alphonse | Aux task | Predict `is_surface` boundary indicator via BCE head (aux_weight=0.1) | 2 | dispatched (post-#3121 close) |
| #3437 | thorfinn | Data curriculum | Domain curriculum: racecar_single first, transition to uniform | 2 | dispatched (post-#3378 merge) |

All 8 students have active PRs. **Zero idle GPUs.** Round-2 set: 4 dispatched, 3 sent-back/rerun, 2 in-flight from rebased round-1. 4 students have completed training with results pending. 4 are running fresh training cycles.

## Recent decisions

- **#3378 (thorfinn scoring NaN-safe) MERGED**: val=148.42 (schedule effect, fix is mathematically identical on finite GT), test=136.01 (finite, 4-split — previously NaN). Unblocks paper-wide test reporting. `data/scoring.py` edit replaces `err * mask` with `torch.where(mask, err, 0)`. Unit test confirms Inf×0=NaN failure mode. **All future PRs report test directly from metrics.jsonl — no eval_test_clean.py post-hoc step.**
- **#3336 (fern grad-clip max_norm=1.0) SENT BACK**: val=129.73 (+2.7%), `clip_rate=1.00 every epoch` (pre-clip mean=100-200, max=600-1900). max_norm=1.0 renormalized every gradient to a unit vector — optimizer saw direction only. Direction signal correct (val_re_rand -4.4%, val_geom_camber_cruise -6.4%) but median-batch info loss hurt camber_rc. Send-back recipe: single arm max_norm=100.0 (clip only spike batches, target clip_rate 0.05-0.3).
- **#3121 (alphonse warmup) CLOSED**: val=158.75 vs baseline 126.32 = +25.7% regression. Uniform across splits (17-36%) — consistent with uniform schedule cost. Epoch 1 wasted at lr=5e-7; cosine never reached floor; val curve still descending at -10/epoch when timeout hit. Three-way confirmation of schedule-axis exhaustion (with #3273/#3298/#3144). Student's analysis: 'the real lever is the budget, not the schedule' — correct. Schedule axis goes back on the shelf until bf16+bs=8 unlocks 21-24 epochs.
- **#3435 (alphonse aux-task) DISPATCHED**: predict `is_surface` boundary indicator via small Linear head on preprocess output, BCE loss with aux_weight=0.1. 193 added params (+0.013%). Cold-start helper — gives the early representation an explicit boundary signal.
- **#3437 (thorfinn domain-curriculum) DISPATCHED**: per-epoch curriculum sampler ramping from racecar_single-heavy (easy_boost=3.0) to uniform-balanced over first 50% of epochs. Pure train.py change (data/ untouched). Hypothesis: easier single-foil physics first builds a foundation that handles tandem+cruise better.
- **Stale_wip pods (#3287 frieren FiLM, #3141 tanjiro Fourier-rerun)**: both pods recovered from the 18:50-19:15 UTC GH rate-limit storm. frieren just picked up #3287 at 20:23 UTC (iteration 74); tanjiro pod restarted with fresh session on #3141. No manual nudge needed — they're spinning up training now.

## Systemic constraints (known issues)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; with bf16 (validated) gives 12 epochs. bf16 + bs=8 (in-flight retry) should give 21-24 epochs, the first time cosine fully anneals on the wider trunk. Schedule axis is exhausted under current budget — confirmed 4-way (#3273, #3298, #3144, #3121).
2. **Cruise-test NaN** — **FIXED in #3378** via `torch.where(mask, err, 0)` substitution. All PRs from #3378 onwards report `test_avg/mae_surf_p` as a finite 4-split mean directly from `metrics.jsonl`. Residual side-issue: `train.py`'s eval-loss aggregation has the same pattern but doesn't affect the paper-facing MAE metric (loss is monitoring-only); separate small patch queued if it matters.

## Round 2 in-flight set (8 students)

Currently dispatched/in-flight (round 2):
- nezuko #3404: LayerScale residual gating (cold-start architectural helper)
- frieren #3287: FiLM domain conditioning (gap+AoA per-sample modulation)
- edward #3332: bf16 + bs=8 retry — primary throughput unlock
- fern #3336: grad-clip max_norm=100 (after max_norm=1.0 send-back)
- thorfinn #3437: domain curriculum (racecar_single-first)
- alphonse #3435: aux task — predict is_surface (cold-start representation helper)
- tanjiro #3141: Fourier features (input encoding, rebased rerun)
- askeladd #3127: SmoothL1 loss (rebased rerun)

## Priority candidates if students free up

1. **RMSNorm vs LayerNorm swap** — drop-in replacement, often as good or faster. Tested in many modern transformers.
2. **SwiGLU MLP** — Shazeer 2020, used in PaLM/LLaMA. ~50% larger MLP params; would need to verify VRAM headroom.
3. **Stochastic depth** — pairs naturally with bf16-unlocked deeper sweep (n_layers 5/6/7/8). Gated on bf16 landing.
4. **Larger model (n_hidden=256)** — push further on capacity if bf16+bs=8 confirms wider-trunk thesis.
5. **Depth sweep at bf16+bs=8** — replay n_layers=6/7/8 once budget is unblocked.
6. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun (in-flight) confirms SmoothL1 wins.
7. **Temperature scheduling on PhysicsAttention** — annealing multiplier on top of the learnable per-head temperature.
8. **OneCycleLR** — alternative schedule; gated on bf16+bs=8 unlocking the schedule axis.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot: a new backbone family before more hyperparameter tuning. FiLM is the most architectural of the round-2 set; if it fails, that's the strongest signal that we need to think bigger.
- If round-2 PRs reveal the orthogonality assumption is failing (e.g. multiple round-2 PRs combined with current stack underperform a clean baseline by >3%) → run a clean combined-baseline measurement.
- **If bf16+bs=8 retry (#3332 retry) still doesn't beat 126.32**, the wider-trunk thesis itself is wrong on this dataset — narrow trunk + bf16 (giving ~18-20 epochs in cap) would then be the right path. We'd revert to narrow as the merged config.
- If aux-task (#3435) regresses heavily on val while aux_acc reaches >95% quickly → aux head dominated; sweep aux_weight down to {0.01, 0.001}.
- If domain-curriculum (#3437) helps single_in_dist but hurts geom holdouts → easy_boost too aggressive; sweep down to easy_boost=2.0 or transition_frac=0.3.
