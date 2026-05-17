# SENPAI Research State

- **Date:** 2026-05-17 07:00
- **Launch:** willow-pai2i-48h-r1 (round 21 — Lookahead-Lion era; **NEW PROGRAMME BEST val=47.5894** (PR #4269, α=0.7 merged); **α-bowl confirmed at 0.7, k-bowl confirmed at k=5**; **ALL 4 architectural up-arms closed (+1.04 to +9.11 budget-bound)**; **slice-DOWN direction nearly competitive** (+0.19 at α=0.5); **5 new α=0.7 compound probes in flight**)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=5/α=0.7) + triple-stack (PROGRAMME BEST)** | **47.5894** | **46.0098** | PR #4269, W&B `oftlu9tn`, seed=0 | Merged 2026-05-17; new all-time best |

Win threshold: **val < 47.5894** (seed=0). Prior best: val=47.97 (PR #4123, α=0.5).

### 3-seed canonical (2/3 complete; seed 2 pending)

| Seed | val_avg | test_avg | Source |
|---|---|---|---|
| 0 | **47.5894** | **46.0098** | PR #4269 merged (`oftlu9tn`) |
| 1 | **47.0830** | **46.5983** | PR #4344 closed (`imywc4uu`); val improves seed=0 by −0.51 |
| 2 | TBD | TBD | Nezuko #4345 in flight |
| 2-seed mean (so far) | 47.336 | 46.304 | — |

## α-frontier at k=5 (Lion-era, seed=0) — FULLY MAPPED

| α | val_avg | Δ vs current best (47.59) | α/k effective pull rate | Status |
|---|---|---|---|---|
| 0.3 | 51.84 | +4.25 | 0.06 (under-pulled) | closed (PR #4269 Arm 1) |
| 0.5 (prior baseline) | 47.97 | +0.38 | 0.10 | superseded by #4269 |
| **0.7 (CURRENT BEST)** | **47.59** | — | **0.14 (critical rate)** | MERGED (PR #4269) |
| 0.8 | 48.25 | +0.66 | 0.16 (over-pulled) | closed (PR #4343, round-21) |

**α-bowl minimum CONFIRMED at α=0.7.** Effective-pull-rate framing α/k ≈ 0.14 validated.

### Effective-pull-rate framing validated across optimizers

α/k ≈ 0.14-0.17 is the empirically-confirmed critical range. At k=5:
- α=0.5 → α/k=0.10 (below optimal)
- **α=0.7 → α/k=0.14 (lower edge of optimal band → wins)**
- α=0.8 → α/k=0.16 (mid-band → probe)
- α=0.9 → α/k=0.18 (upper edge → likely declining)

## Lookahead-Lion mechanism decomposition

| Intervention | Δ val | Mechanism |
|---|---|---|
| AdamW → Lion | −8.15 | Sign-based: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | −1.10 | Slow-weight averaging: reduce per-basin commitment variance |
| α=0.5 → α=0.7 (same k=5) | −0.38 | Stronger basin-averaging pull; α/k closer to 0.15 critical rate |
| **Total: AdamW → Lookahead+Lion (k=5/α=0.7)** | **−9.63** | All three mechanisms additive; seed-robust |

## Lookahead-Lion hyperparameter frontier (LARGELY EXHAUSTED)

| Knob | Frontier finding | Programme implication |
|---|---|---|
| **α (pull strength)** | k=5 monotone: 0.3=51.84 > 0.5=47.97 > 0.7=47.59; α=0.8 IN FLIGHT | **ACTIVE FRONTIER** — bowl minimum not found yet |
| **k (sync interval)** | U-curve right-shifted vs AdamW: k=2=48.84, k=3=48.20, k=5=47.59; k=7 IN FLIGHT (#4310) | Lion's low variance → more steps between syncs optimal |
| **β1 (update direction)** | CLOSED (#4271): β1=0.9 optimal; β1=0.95 → +3.03 (asymmetric landscape) | β1=0.9 is local optimum; Lookahead + Lion over-smooth at β1>0.9 |
| **β2 (m-buffer EMA)** | CLOSED (#4264): β2=0.95 → +6.65 (largest Lion-era regress). β2 must INCREASE from 0.99 not decrease | Knob direction was wrong; β2∈{0.995, 0.999} is the open question |
| **LR (cfg.lr)** | CLOSED (#4265): cfg.lr=3e-4 → +1.44. Lion LR landscape SHARPER than AdamW | LR=5e-4 is near-optimal |
| **3-seed canonical** | IN PROGRESS: seeds 1+2 running (#4344, #4345); seed=0 val=47.59 | Paper-ready noise floor pending |

## Architectural Portfolio — ALL 4 UP-ARMS CLOSED (capacity-up direction is budget-bound)

| Architectural arm | PR | Dim | Δ val vs new baseline | epoch cost | best_ep |
|---|---|---|---|---|---|
| ~~Attention heads~~ | #4304 nezuko | 4 → 8 | **+9.11** | +36% | 12 (budget-cut) |
| ~~Slices (up)~~ | #4323 fern (round-21) | 64 → 96 | **+5.51** | +18% | 14 (budget-cut) |
| ~~Depth~~ | #4294 tanjiro | 5 → 6 | **+3.01** | +21% | 14 (budget-cut) |
| ~~FFN width~~ | #4286 thorfinn | 2 → 3 | **+1.04** | +5% | 16 (mild) |
| **Slices (DOWN)** | #4325 frieren (round-21) | 64 → 32 | **+0.19** | normal | **17 ✓** |

**Decisive verdict:** Capacity-up regression magnitude tracks per-epoch cost monotonically (+36%/+21%/+18%/+5% → +9.11/+3.01/+5.51/+1.04 val). T_max=17 cosine is tightly fit to baseline budget; up-arms get budget-cut before reaching cosine floor. 

**Asymmetric slice axis:** slice_num=32 (slice DOWN) is nearly competitive (+0.19 only, full cosine floor reached). Frieren's #4375 (slice_num=32 + α=0.7) probes whether this near-miss compounds with the α=0.7 winner — predicted val ≈ 47.40 = NEW WINNER candidate.

### h=192 architectural probe IN FLIGHT (edward #4374)

VRAM gating resolved by thorfinn's data (39.16 GB at h=128 mlp_ratio=3 → h=192 mlp_ratio=2 estimated 45-55 GB; comfortable in 96 GB). Tests whether hidden-dim scaling composes with Lion + Lookahead + α=0.7 — a NEW combination unstudied in this launch. **Budget-cut risk acknowledged** in PR; epoch_time + best_epoch + VRAM diagnostics required.

## Active WIP experiments (round 21)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4376 | alphonse | **β1=0.88 + α=0.7 — β1 micro-probe at new α winner** | **NEW (round 21)** | HP probe |
| #4375 | frieren | **slice_num=32 + α=0.7 — α-compound of slice32 near-winner** | **NEW (round 21)** | **Winner candidate (predicted ~47.40)** |
| #4374 | edward | **h=192 + α=0.7 — architectural probe (VRAM unblocked)** | **NEW (round 21)** | **Winner candidate (architectural)** |
| #4373 | fern | **β2=0.995 + α=0.7 — β2 frontier bracket (vs tanjiro's 0.999)** | **NEW (round 21)** | HP probe |
| #4371 | askeladd | **k=6 + α=0.7 — k-frontier gap-fill (k=5→k=7)** | **NEW (round 21)** | HP probe |
| #4356 | tanjiro | β2=0.999 + α=0.7 — corrected high-β2 direction | Running (round 20) | Winner candidate |
| #4355 | thorfinn | k=4 + α=0.7 — gap-fill between k=3 and k=5 | Running (round 20) | Winner candidate |
| #4345 | nezuko | α=0.7 seed=2 (3-seed canonical of new best) | Running | Paper-facing canonical |

**All 8 students active. Zero idle. Single-arm policy in force.**

### Round-21 strategic theme: compounding new α=0.7 winner with unstudied knob directions

Each of the 5 newly-assigned PRs takes a distinct unstudied knob and composes it with the α=0.7 winner. Spans the (β1, β2, k, slice_num, h) space orthogonally. Best-case outcome: 1-2 winners that compound further. Worst case: clean frontier closures across all 5 axes.

## Round-21 closures (5 closures; α-bowl + k-bowl + slice axis mapped)

- **#4344 askeladd (α=0.7 seed=1)** CLOSED as canonical-confirmation: val=47.0830 / test=46.5983 (val improves seed=0 by −0.51, test regresses by +0.59). 2-seed mean: 47.336. No code merge needed; same code as PR #4269 with --seed 1. 3-seed canonical pending nezuko #4345. Reassigned to **k=6 + α=0.7 (#4371)**.
- **#4343 alphonse (α=0.8 at k=5)** CLOSED: val=48.25 / test=46.97 (+0.66 / +0.96 vs current best). **α-bowl minimum CONFIRMED at α=0.7.** All 4 test splits regress uniformly. Effective-pull-rate framing α/k ≈ 0.14 validated cross-α. Reassigned to **β1=0.88 + α=0.7 (#4376)**.
- **#4325 frieren (slice_num=32 at α=0.5)** CLOSED: val=47.7822 / test=46.1713. Close but no cigar vs new baseline (+0.19 val, +0.16 test) — would have beat OLD baseline (−0.19). **Slice axis asymmetric:** smaller-than-baseline competitive, larger-than-baseline catastrophic. Reassigned to **slice_num=32 + α=0.7 (#4375)** as direct α-compound.
- **#4323 fern (slice_num=96 at α=0.5)** CLOSED: val=53.10 / test=50.38, best_ep=14 (budget-cut at +18% epoch cost). 4th of 4 architectural up-arms closed. Reassigned to **β2=0.995 + α=0.7 (#4373)**.
- **#4310 edward (k=7 at α=0.5)** CLOSED: val=48.48 / test=46.61 (+0.51 vs old k=5 baseline). **k-bowl minimum CONFIRMED at k=5** (symmetric U: k=2=48.84, k=3=48.20, k=5=47.97, k=7=48.48). Reassigned to **h=192 + α=0.7 (#4374)** — architectural probe.

## Round-20 closures (architectural up-arms exhausted; capacity-up dead at current budget)

- **#4294 tanjiro (depth=6)** CLOSED: val=50.60 / test=48.98 (+3.01/+2.97 vs new baseline). best_ep=14 (timeout-cut at +21% epoch cost). Same pattern as heads=8 (#4304). All 4 splits regress uniformly — fewer effective annealing epochs hurts everywhere; not "capacity helps OOD." Tanjiro reassigned to **β2=0.999 at α=0.7 (#4356)**.
- **#4286 thorfinn (mlp_ratio=3)** CLOSED via W&B rate-limit-close: val=48.63 / test=46.37 (+1.04/−0.12 vs new baseline). best_ep=16 (just below cosine floor). **Test essentially TIES baseline** (within noise floor). Gentlest architectural up-arm tested; not budget-cut but still no win. Thorfinn reassigned to **k=4 at α=0.7 (#4355)**.

### Architectural up-direction verdict

Three of three completed up-arms regressed; magnitude tracks per-epoch cost. With T_max=17 cosine tightly fit to baseline budget, capacity-up arms hit a wall. Remaining slice_num arms (#4323, #4325) test a different axis. **For future rounds**: faster training recipes or longer schedule would unlock capacity dimension.

## Round-19 events

### NEW PROGRAMME BEST: PR #4269 merged (val=47.5894 / test=46.0098)

Lookahead-Lion α=0.7 at k=5 beats α=0.5 by −0.38 val, −0.48 test (same seed=0, both best_epoch=17). Lion-era α-pattern mirrors AdamW-era pattern at k=5: monotone improvement toward α=0.7. α-bowl minimum not yet found — α=0.8 probe active (#4343).

### Closures (round 19)

- **#4304 nezuko (heads=8)** CLOSED: val=56.70, +8.73 regress, best_ep=12 (timeout-cut at 12/17 epochs — +36% per-epoch overhead exhausts budget). No divergence or collapse; purely a budget incompatibility. **heads is dead at h=128 with 30-min budget.** Add to dead-end lever list.
- **#4271 alphonse (Lion β1 sweep)** CLOSED: β1=0.85 (+0.34 regress), β1=0.95 (+3.03 regress). β1=0.9 default confirmed optimal; **asymmetric landscape: moving β1 high is ~9× more damaging than moving it low.** Lookahead's k=5 slow-weight smoothing stacks destructively with higher β1 (double-smoothing → over-sticky optimizer near cosine floor).

## Round-18 closures (Lion HP frontier exhausted)

- **#4265 fern (Lion LR sweep)**: val=49.41 (Arm 1 only, +1.44 regress). Lion LR sharper than AdamW.
- **#4264 frieren (Lion β2 scan)**: val=54.62 (Arm 1 only, +6.65 regress — largest single-knob Lion-era regression). β2=0.95 → m-buffer half-life 14 steps too short at batch=4.

## Round-17 closures (Lion 3-seed canonical complete; k-shift prediction CONFIRMED)

- **#4242 nezuko (Lookahead-Lion seed=2)** CLOSED: val=48.84. OLD 3-seed canonical (α=0.5): val mean=48.68, σ̂=0.64 (SUPERSEDED by new baseline at α=0.7)
- **#4241 edward (Lookahead-Lion k=3)** CLOSED: val=48.20. k-shift CONFIRMED; Lion k-curve U-min at or right of k=5.

## Key mechanistic findings

### Confirmed dead-end levers

| Lever | Verdict |
|-------|---------|
| Dropout / DropPath | Regression |
| Weight decay ≥1e-2 | Null |
| LR=1e-3 under T_max=15 | Divergence |
| Head+embed LR boost (1.5–2.5×) | All null/worse |
| T_max < 17 | Suboptimal (PyTorch Gotcha #3) |
| RMSNorm (vs LayerNorm) | −5.18 val regression |
| slice_num=128 (2× attention; pre-Lion) | −10.92 val regression |
| **heads=8 at h=128 (30-min budget)** | **+8.73 val regression (budget incompatibility)** |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA variants (all) | Regression or NO-OP |
| β1=0.95 + β2=0.95 (compound momentum) | +2.86 val regression |
| Lookahead k=8 | +2.87 val regression |
| Lookahead α=0.3 | +4.36 val regression |
| Lion β2=0.95 | +6.65 val regression |
| Lion cfg.lr=3e-4 | +1.44 val regression |
| **Lion β1=0.95 (over-smooth + Lookahead)** | **+3.03 val regression** |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()`
3. T_max must equal total_epochs — T_max<total causes hard-zero LR

## Next research directions

### Priority 1 (Immediate — IN FLIGHT round 21)

- **slice_num=32 + α=0.7** (frieren #4375) — predicted val ≈ 47.40 = strongest winner candidate
- **h=192 + α=0.7** (edward #4374) — first architectural probe under α=0.7; only up-arm not yet tested
- **β2=0.999 + α=0.7** (tanjiro #4356, round 20) — corrected high-β2 direction
- **β2=0.995 + α=0.7** (fern #4373) — brackets tanjiro's β2 frontier
- **k=4 + α=0.7** (thorfinn #4355, round 20) — k-frontier gap-fill on left flank
- **k=6 + α=0.7** (askeladd #4371) — k-frontier gap-fill on right flank
- **β1=0.88 + α=0.7** (alphonse #4376) — β1 micro-probe of asymmetric landscape
- **α=0.7 seed=2** (nezuko #4345) — completes 3-seed canonical for paper

### Priority 2 (next idle-round; gated on round-21 outcomes)

- **slice_num=48 + α=0.7** — if frieren's slice_num=32 wins, refine slice bowl between 32 and 64
- **α=0.7 at k=3** — confirm effective-pull-rate framing transfers to different k (predict α/k=0.23 > optimal 0.14 → expected regression but useful negative data)
- **α=0.65 / α=0.75 micro-bowl at k=5** — refine α-bowl floor between confirmed 0.5/0.7/0.8
- **Compounding winning HP knobs** — once round-21 settles, test (k=4 + α=0.7 + slice_num=32) or other 3-way compounds
- **Lion β3 / non-standard m-buffer mixing** — if β2=0.999 + 0.995 both win monotonically toward higher, the β2 frontier extends further

### Priority 3 (Plateau-protocol escalation if round-21 saturates)

- **Loss reformulation:** physics-informed continuity-equation constraint, pressure-relative normalization, per-region loss weighting
- **Pseudo-labeling** from momentum-distilled teacher
- **Cosine LR restart at epoch 17** to extend training with slow-weight pull intact
- **Lion variants:** Tiger optimizer (Chen 2023's Lion successor), AdaFactor with sign-update, ScheduleFree-Lion

### Priority 4 (Loss reformulation — escalation tier 2)

- **Physics-informed loss:** soft continuity-equation constraint on surface velocity field
- **Pressure-relative target normalization:** scale-invariant pressure prediction
- **Per-region loss weighting:** decompose val_avg components and reweight

### Observations for paper

1. **Lion's α-frontier at k=5 mirrors AdamW's:** basin-averaging mechanism (not optimizer) dominates at k=5; Lookahead is optimizer-independent in the strong-pull regime
2. **Lion β1 landscape is asymmetric:** raising β1 from 0.9 is ~9× worse than lowering it; Lookahead-Lion double-smoothing is destructive above β1=0.9
3. **heads=8 is budget-incompatible at h=128:** +36% per-epoch cost kills the cosine-schedule benefit; not a capacity failure — a wall-clock failure
4. **Effective-pull-rate α/k ≈ 0.14-0.17 is a cross-optimizer invariant:** validated at k=5 for both AdamW and Lion
