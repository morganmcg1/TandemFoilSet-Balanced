# SENPAI Research State

- **Date**: 2026-05-15 22:35
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; round 2 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

System fix merged:
- **#3378 thorfinn (scoring NaN-safe)**: `test_avg/mae_surf_p` now finite 4-split mean paper-wide.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0.

**Major event 22:30 UTC**: stop/pivot criterion fired. #3332 edward bf16+bs=8 retry CLOSED — bs=8 axis cleanly refuted (memory-bandwidth-bound model, +10% slower per-epoch, peak 98 GB hits HBM ceiling). bf16-bs4 prior arm gave val=129.76 (+2.7%) — close but not winning. Per my own documented stop/pivot in BASELINE.md, the next decisive experiment is **narrow trunk + bf16** (PR #3478 edward) to test whether the wider-trunk merge (#3130) should be reverted.

**Systemic finding (4-way confirmed + bf16 result)**: the binding constraint on this track is **per-epoch wallclock at the wider trunk**, not model architecture or loss formulation. At `n_hidden=192`, only 8-9 epochs fit in fp32 30-min cap (vs 14 at narrower); with bf16 we get 12. The wider trunk's capacity advantage cannot manifest under any throughput-on-the-cheap unlock attempted so far. Five independent PRs (#3273 ReScaler, #3298 p_surf_weight, #3144 deeper-l8, #3121 warmup, #3332 bf16+bs=8) all closed with the same pattern.

**FiLM disconfirmation (#3287 frieren)**: predicted-to-improve splits regressed worst. Rules out the family of low-dimensional per-sample conditioners on this dataset under the 30-min cap.

## PRs in-flight

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun — nudged for rebased commits | 1→rerun | wip (nudged 22:32) |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebased rerun) | 1→rerun | wip (pod active) |
| #3336 | fern | Optimization stability | Grad-clip max_norm=100 rerun (after max_norm=1.0 clip_rate=1.00) | 2→rerun | sent back |
| #3404 | nezuko | Architecture (cold-start) | LayerScale residual gating (CaIT, init 1e-4) on TransolverBlock | 2 | wip (nudged 22:33) |
| #3435 | alphonse | Aux task | Predict `is_surface` boundary indicator via BCE head (aux_weight=0.1) | 2 | dispatched |
| #3437 | thorfinn | Data curriculum | Domain curriculum: racecar_single first, transition to uniform | 2 | dispatched |
| #3455 | frieren | Regularization | DropPath stochastic depth (linear 0→0.1) on TransolverBlock residuals | 2 | dispatched |
| #3478 | edward | Width / throughput | Narrow trunk (n_h=128) + bf16: test stop/pivot — revert wider trunk? | 2 | dispatched |

All 8 students have active PRs. **Zero idle GPUs.** Round-2 set: 5 dispatched, 1 sent-back, 2 in-flight reruns from round-1. Two are stale-but-nudged (#3127, #3404).

## Recent decisions

- **#3332 (edward bf16+bs=8 retry) CLOSED**: bs=8 axis cleanly refuted. Two arms: bs=8 latest val=187.83 (+48.7%), bs=8 1st relaunch val=178.45. Per-batch time 382→846 ms (memory-bandwidth bound), peak 98 GB (HBM ceiling). bs=12/16 would OOM. The prior arm (bf16-bs4) gave val=129.76 = +2.7% but doesn't beat baseline. Student's mechanism analysis was excellent.
- **#3478 (edward narrow+bf16) DISPATCHED**: stop/pivot test. Revert n_hidden 192→128 and n_head 6→4 (pre-#3130) while keeping EMA+surf_weight=25+bf16. Schedule-aligned (epochs=18, T_max=18). Three decisive outcomes: (a) narrow+bf16 < 126.32 → revert width as merged config; (b) tie within 2% → width is a wash, keep merged for future bandwidth headroom; (c) narrow regresses → wider thesis holds, focus on throughput unlocks (torch.compile, smaller slices).
- **#3287 (frieren FiLM) CLOSED** (earlier this loop): val=145.99 (+15.6%). Predicted-to-improve splits regressed worst — strongest possible hypothesis disconfirmation. Rules out low-dimensional per-sample conditioners.
- **#3455 (frieren DropPath) DISPATCHED** (earlier this loop): stochastic depth on TransolverBlock residual branches, linear schedule [0.0, 0.025, 0.05, 0.075, 0.1]. CaIT-canonical, zero param/inference overhead. Targets OOD splits.
- **#3378 (thorfinn scoring NaN-safe) MERGED**: val=148.42 (schedule effect, fix mathematically identical), test=136.01 (finite, 4-split). Unblocks paper-wide test reporting.
- **#3336 (fern grad-clip max_norm=1.0) SENT BACK**: val=129.73 (+2.7%), clip_rate=1.00 every epoch. Send-back recipe: max_norm=100.0 (clip only spike batches).
- **#3121 (alphonse warmup) CLOSED**: +25.7% regression. Three-way confirmation of schedule-axis exhaustion.
- **#3435 (alphonse aux-task) DISPATCHED**: predict `is_surface` via BCE head, aux_weight=0.1. 193 added params.
- **#3437 (thorfinn domain-curriculum) DISPATCHED**: per-epoch curriculum sampler, easy_boost=3.0, transition_frac=0.5.
- **Nudges sent (22:32, 22:33 UTC)**: #3127 askeladd hasn't pushed rebased commits 7h after send-back; #3404 nezuko hasn't pushed any training output 3h after dispatch. Pods are heartbeating but Claude sessions are short. Posted explicit instruction recaps on both PRs.

## Systemic constraints (known issues)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; bf16 gives 12; bs=8 doesn't help (memory-bandwidth bound). Six-way confirmation (#3273, #3298, #3144, #3121, #3332). Per-epoch throughput unlock attempts on cheap axes exhausted on the wider trunk. Next move is to test whether the wider trunk is the right config at all (#3478).
2. **Cruise-test NaN** — **FIXED in #3378** via `torch.where(mask, err, 0)` substitution. All PRs from #3378 onwards report `test_avg/mae_surf_p` as a finite 4-split mean directly from `metrics.jsonl`.
3. **Low-dimensional per-sample conditioners ruled out** (via #3287 FiLM disconfirmation) — predicted-improve splits were most regressed. Future conditioning needs richer feature sets or different injection.
4. **Pod heartbeat / Claude-session short exits** — two stale_wip PRs (#3127, #3404) where pods heartbeat but Claude exits quickly without training. Explicit advisor nudges posted; monitor next iteration for actual training output. If still stale 1-2 hours from now, escalate to direct intervention.

## Round 2 in-flight set (8 students)

Currently dispatched/in-flight (round 2):
- nezuko #3404: LayerScale residual gating (nudged — start training)
- frieren #3455: DropPath stochastic depth (regularization for OOD)
- edward #3478: narrow trunk + bf16 (test stop/pivot — revert width?)
- fern #3336: grad-clip max_norm=100 (after max_norm=1.0 send-back)
- thorfinn #3437: domain curriculum (racecar_single-first)
- alphonse #3435: aux task — predict is_surface
- tanjiro #3141: Fourier features (input encoding, rebased rerun)
- askeladd #3127: SmoothL1 loss (rebased rerun — nudged)

## Priority candidates if students free up

1. **torch.compile()** on bf16+wider stack — graph fusion attacking memory-bandwidth bottleneck via different mechanism than bs=8. 10-20% per-epoch expected if it works at all on dynamic shapes.
2. **RMSNorm vs LayerNorm swap** — drop-in replacement, often as good or faster.
3. **SwiGLU MLP** — Shazeer 2020, used in PaLM/LLaMA. ~50% larger MLP params; would need to verify VRAM headroom.
4. **smaller slice_num** (e.g. 32 instead of 64) — direct memory-bandwidth attack; may compromise expressivity. Gated on the wider-trunk-vs-narrow decision from #3478.
5. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun (in-flight) confirms SmoothL1 wins.
6. **Temperature scheduling on PhysicsAttention** — annealing multiplier on top of the learnable per-head temperature.
7. **DropPath sweep** — if #3455 lands at rate=0.1, sweep to {0.2, 0.3}.
8. **LayerScale + DropPath jointly** — CaIT canonical pairing; conditional on both #3404 and #3455 landing.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- **#3478 result determines the next major direction**:
  - If narrow+bf16 < 126.32: revert wider trunk merge. Re-baseline all round-2 PRs against narrow+bf16.
  - If narrow+bf16 ≈ 126.32 ± 2%: width is a wash. Keep merged config; pursue architectural wins (LayerScale, DropPath, aux-task, curriculum) and torch.compile.
  - If narrow+bf16 > 130: wider thesis confirmed. Throughput unlocks (torch.compile, smaller slices) become priority.
- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot.
- If aux-task (#3435) regresses heavily on val while aux_acc reaches >95% quickly → aux head dominated; sweep aux_weight down.
- If domain-curriculum (#3437) helps single_in_dist but hurts geom holdouts → easy_boost too aggressive.
- If DropPath (#3455) regresses uniformly across splits → regularization axis not useful at this budget.
- If LayerScale (#3404) shows <1% improvement → CaIT recipe not applicable; close residual-gating axis.
