# SENPAI Research State

- **Date:** 2026-05-17 10:00
- **Launch:** willow-pai2i-48h-r1 (round 24 — Lookahead-Lion era; **CURRENT PROGRAMME BEST val=46.8383 / test=45.3196** (PR #4373); **OLD baseline 3-seed canonical resolved: 47.77 ± 0.80 / 46.51 ± 0.46** (PR #4345 closed today with corrected qohoymnk run); **k-bowl SHIFTED to k=6 under α=0.7** (PR #4371); **k=6 + β2=0.995 compound + α=0.65 micro-probe in flight at new baseline**)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=5/α=0.7) + Lion β2=0.995 + triple-stack (PROGRAMME BEST)** | **46.8383** | **45.3196** | PR #4373, W&B `3k6hob38`, seed=0 | Merged 2026-05-17 08:00; new all-time best |

Win threshold: **val < 46.8383** (seed=0). Prior best: val=47.5894 (PR #4269, β2=0.99).

### 3-seed canonical of NEW programme best (in flight; round-22)

| Seed | val_avg | test_avg | Source |
|---|---|---|---|
| 0 | **46.8383** | **45.3196** | PR #4373 MERGED (`3k6hob38`) |
| 1 | TBD | TBD | tanjiro #4386 in flight |
| 2 | TBD | TBD | thorfinn #4385 in flight |

### 3-seed canonical of OLD programme best (PR #4269 config, β2=0.99) — RESOLVED round-24

| Seed | val_avg | test_avg | best_ep | W&B run |
|---|---|---|---|---|
| 0 | 47.5894 | 46.0098 | 17 | `oftlu9tn` (PR #4269) |
| 1 | 47.0830 | 46.5983 | 17 | `imywc4uu` (PR #4344) |
| 2 | 48.6494 | 46.9130 | 17 | `qohoymnk` (PR #4345 closed today with corrected canonical) |
| **3-seed mean ± σ̂** | **47.7740 ± 0.7990** | **46.5070 ± 0.4570** | — | — |

Single-seed=0 baseline (47.59) is within 1σ of 3-seed mean (47.77). σ̂=0.80 sets the seed-noise floor for cross-seed comparisons. **Implication for round-22 canonical:** seed=1/seed=2 at β2=0.995 may exhibit similar ±0.80 envelope around the new mean.

## β2-frontier — FULLY MAPPED (narrow bowl at 0.995, NOT monotone toward 1.0)

| β2 | val_avg | m-buffer half-life | Status |
|---|---|---|---|
| 0.95 | 54.62 | 14 steps | catastrophic (PR #4264 round-18) |
| 0.99 | 47.59 | 69 steps | prior baseline (PR #4269) |
| **0.995 (CURRENT BEST)** | **46.84** | **138 steps** | **MERGED (PR #4373 round-22)** |
| 0.997 | TBD | 230 steps | **fern #4384 IN FLIGHT (round-22 right-edge probe)** |
| 0.999 | 52.04 | 692 steps | catastrophic — spans entire run (PR #4356 closed) |

**Mechanism:** β2 m-buffer half-life sweet spot = 138 steps; long enough for within-basin smoothing but short enough to refresh during cosine annealing's tail. β2=0.999's 692-step half-life spans the full 17-epoch run (~850 steps) — momentum from epoch 1 still bleeds into epoch 17 update directions and destroys the cosine-floor advantage.

## α-frontier at k=5 (Lion-era, seed=0) — FULLY MAPPED

| α | val_avg | Δ vs old best (47.59) | α/k effective pull rate | Status |
|---|---|---|---|---|
| 0.3 | 51.84 | +4.25 | 0.06 (under-pulled) | closed (PR #4269 Arm 1) |
| 0.5 (prior baseline) | 47.97 | +0.38 | 0.10 | superseded by #4269 |
| **0.7 (CURRENT BEST)** | **46.84 (β2=0.995)** | **−0.75** | **0.14 (critical rate)** | **MERGED (PR #4373)** |
| 0.8 | 48.25 | +0.66 | 0.16 (over-pulled) | closed (PR #4343, round-21) |

**α-bowl minimum CONFIRMED at α=0.7.** Effective-pull-rate framing α/k ≈ 0.14 validated.

## k-frontier under α=0.7 (Lion-era, seed=0) — k-BOWL SHIFTED to k=6 (round-23 finding)

Under β2=0.99 slice at α=0.7:

| k | val_avg | α/k | Status |
|---|---|---|---|
| 2 | 48.84 | 0.35 (over-pulled) | closed (round-17, AdamW era) |
| 3 | 48.20 | 0.23 (over-pulled) | closed (PR #4241) |
| 4 | ~48.30 | 0.175 (slight over-pull) | closed (PR #4355 round-22) |
| 5 | 47.59 | 0.14 | superseded by k=6 |
| **6 (NEW LOCAL MIN @ β2=0.99)** | **47.14** | **0.117** | closed (PR #4371) — beats k=5 by −0.45 val |
| 7 | 48.48 | 0.10 (under-pulled) | closed (PR #4310) |

**CRITICAL UPDATE:** k-bowl is at **k=6** under α=0.7, NOT k=5 as previously asserted. PR #4371 (just closed today round-23) showed k=6 beats k=5 by −0.45 val at β2=0.99. The bowl is asymmetric (k=4: +0.71, k=6: −0.45, k=7: +0.89 relative to k=5).

**Effective-pull-rate framing REFINED:** Optimal α/k is **~0.117 at α=0.7** (k=6), not 0.14 (k=5). The cross-optimizer invariant α/k≈0.14 from AdamW era does NOT cleanly transfer to Lion at α=0.7.

**Open question (NOW DISPATCHED #4402):** Does k=6 stack with β2=0.995 winner? Predicted val ≈ 46.84 + (47.14 − 47.59) = **46.39** if additive = potential NEW programme best.

## Compound winner decomposition

| Intervention | Δ val | Cumulative val | Mechanism |
|---|---|---|---|
| Baseline AdamW + triple-stack | — | 56.0 | T_max=17 + GeGLU + β2=0.95 |
| AdamW → Lion | −8.0 | ~48.0 | Sign-based: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | −0.0 to −0.4 | 47.97 | Slow-weight averaging (PR #4123) |
| α=0.5 → α=0.7 (same k=5) | −0.38 | **47.59** | Stronger basin-averaging pull; α/k closer to 0.14 critical rate (PR #4269 merged) |
| β2=0.99 → β2=0.995 (m-buffer half-life 69→138 steps) | **−0.75** | **46.84** | Within-basin gradient smoothing without spanning training run (PR #4373 merged) |
| **Total compound win** | **~−1.13 from old triple-stack** | **46.84** | **Three additive mechanisms; seed-robustness pending** |

## Architectural Portfolio — ALL 4 UP-ARMS CLOSED (capacity-up direction is budget-bound)

| Architectural arm | PR | Dim | Δ val vs old baseline | epoch cost | best_ep |
|---|---|---|---|---|---|
| ~~Attention heads~~ | #4304 nezuko | 4 → 8 | +9.11 | +36% | 12 (budget-cut) |
| ~~Slices (up)~~ | #4323 fern | 64 → 96 | +5.51 | +18% | 14 (budget-cut) |
| ~~Depth~~ | #4294 tanjiro | 5 → 6 | +3.01 | +21% | 14 (budget-cut) |
| ~~FFN width~~ | #4286 thorfinn | 2 → 3 | +1.04 | +5% | 16 (mild) |
| **Slices (DOWN)** | #4325 frieren | 64 → 32 | **+0.19** | normal | **17 ✓** |

**Decisive verdict:** Capacity-up regression magnitude tracks per-epoch cost monotonically. T_max=17 cosine is tightly fit to baseline budget; up-arms get budget-cut before reaching cosine floor.

**Asymmetric slice axis:** slice_num=32 (slice DOWN) is nearly competitive (+0.19 only); slice_num=96 catastrophic (+5.51). Frieren's #4375 (slice_num=32 + α=0.7) probes whether the near-miss compounds with the α=0.7 winner — **but now compared against new best (46.84)**, the probe needs to clear an additional 0.75 gap to win.

### h=192 architectural probe IN FLIGHT (edward #4374)

VRAM gating resolved by thorfinn's data (39.16 GB at h=128 mlp_ratio=3 → h=192 mlp_ratio=2 estimated 45-55 GB; comfortable in 96 GB). Tests whether hidden-dim scaling composes with Lion + Lookahead + α=0.7 + β2=0.99 — a NEW combination unstudied in this launch.

## Active WIP experiments (round 24)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4415 | nezuko | **α=0.65 + β2=0.995 + k=5 (α-bowl left-side micro-probe at new baseline)** | **NEW (round 24)** | α-floor probe at β2=0.995 |
| #4402 | askeladd | k=6 + β2=0.995 + α=0.7 compound (potential NEW best, predicted ~46.39) | Running (round 23) | **#1 Winner candidate** |
| #4386 | tanjiro | 3-seed canonical seed=1: α=0.7 + β2=0.995 | Running (round 22) | Paper-facing canonical |
| #4385 | thorfinn | 3-seed canonical seed=2: α=0.7 + β2=0.995 | Running (round 22) | Paper-facing canonical |
| #4384 | fern | β2=0.997 right-edge probe (refines 0.995→0.999 bowl) | Running (round 22) | β2-bowl micro-refinement |
| #4376 | alphonse | β1=0.88 + α=0.7 (β1 micro-probe at α winner) — at β2=0.99 | Running (round 21) | HP probe |
| #4375 | frieren | slice_num=32 + α=0.7 (α-compound of slice32 near-winner) — at β2=0.99 | Running (round 21) | Winner candidate (predicted ~47.40, now needs −0.75 more) |
| #4374 | edward | h=192 + α=0.7 (architectural probe, VRAM unblocked) — at β2=0.99 | Running (round 21) | Winner candidate (architectural) |

**All 8 students active. Zero idle. Single-arm policy in force.**

### Round-24 strategic theme: micro-bowl probes at new β2=0.995 baseline

Now that PR #4373 (β2=0.995 winner) is the merged baseline, the (α, k) frontiers need to be re-mapped at the new β2:
- #4402 (askeladd): k=6 compound at β2=0.995 (k-frontier under new β2)
- #4415 (nezuko, just dispatched): α=0.65 micro-probe at β2=0.995 (α-frontier under new β2)
- Round-22's 3-seed canonical (#4385, #4386) and β2=0.997 (#4384) refine the new baseline's variance and right-edge

Round-21's 4 still-running PRs (alphonse #4376, frieren #4375, edward #4374) are at β2=0.99 — their results inform direction discovery but the absolute val comparison is against the OLD 47.59 baseline; a winner there needs separate compound testing at β2=0.995.

### Heartbeat-rerun mitigation update (round-24 finding)

PR #4345's canonical reporting revealed a critical heartbeat-rerun failure mode: the student reported a LATER under-converged run instead of the EARLIER converged one (`qohoymnk`). Future canonical assignments now include explicit instructions to:
- Query W&B group for ALL finished runs (`wandb runs ls --group <group> --state finished`)
- Surface the EARLIEST converged (best_ep=17) run as canonical
- Kill duplicate processes but DO NOT report the truncated rerun

## Round-24 closures (1 closure; OLD 3-seed canonical resolved)

- **#4345 nezuko (α=0.7 seed=2 at β2=0.99)** CLOSED with corrected canonical: `qohoymnk` (val=48.6494, test=46.9130, best_ep=17) — student initially reported truncated `9sbzb9mq` (best_ep=14, val=51.94) but the W&B group contained an earlier converged run that completed all 17 epochs. The corrected canonical completes the 3-seed mean (47.77 ± 0.80) of the OLD baseline. Nezuko reassigned to **α=0.65 + β2=0.995 micro-probe (#4415)** — tests if α-bowl shifted leftward under stronger m-buffer smoothing.

## Round-23 closures (1 closure; k-bowl shift discovered)

- **#4371 askeladd (k=6 + α=0.7 at β2=0.99)** CLOSED: val=47.1361 / test=45.5200 (W&B `i8agcpvv`, finished, best_ep=17, 31.4 min). **BEATS old k=5 baseline by −0.45 val / −0.49 test**, but does NOT beat current best 46.84 (+0.30 val). **k-bowl SHIFTED to k=6 under α=0.7** (was k=5 under α=0.5). Effective-pull-rate framing REFINED to α/k=0.117 at α=0.7. Askeladd reassigned to **k=6 + β2=0.995 + α=0.7 compound (#4402)** — predicted val ≈ 46.39 = potential NEW programme best.

## Round-22 closures (3 closures + 1 merge; β2-bowl mapped; new programme best landed)

- **#4373 fern (β2=0.995 + α=0.7) MERGED — NEW PROGRAMME BEST** val=46.8383 / test=45.3196 (Δ −0.75 val / −0.69 test vs PR #4269). Compound winner: α=0.7 (PR #4269) + β2=0.995 stack additively, no destructive interaction. 3 of 4 test splits improve materially; mild +0.62 regression on test_re_rand only. Fern reassigned to **β2=0.997 right-edge probe (#4384)**.
- **#4356 tanjiro (β2=0.999 + α=0.7)** CLOSED: val≈52.04, +5.20 catastrophic. β2-bowl narrow at 0.995, NOT monotone toward 1.0. m-buffer half-life at β2=0.999 ≈ 692 steps spans entire training run. Tanjiro reassigned to **3-seed canonical seed=1 of new best (#4386)**.
- **#4355 thorfinn (k=4 + α=0.7)** CLOSED: val≈48.30, +0.71 vs new best. k-bowl sharp at k=5; α/k=0.175 over-pulls (slightly worse than prior k=3+α=0.5 at α/k=0.17). Thorfinn reassigned to **3-seed canonical seed=2 of new best (#4385)**.

## Round-21 closures (5 closures; α-bowl + k-bowl + slice axis mapped)

- **#4344 askeladd (α=0.7 seed=1)** CLOSED as canonical-confirmation: val=47.0830 / test=46.5983.
- **#4343 alphonse (α=0.8 at k=5)** CLOSED: val=48.25 / test=46.97. **α-bowl minimum CONFIRMED at α=0.7.**
- **#4325 frieren (slice_num=32 at α=0.5)** CLOSED: val=47.7822 / test=46.1713. Close but +0.19.
- **#4323 fern (slice_num=96 at α=0.5)** CLOSED: val=53.10 / test=50.38. 4th of 4 architectural up-arms.
- **#4310 edward (k=7 at α=0.5)** CLOSED: val=48.48 / test=46.61. **k-bowl minimum CONFIRMED at k=5.**

## Round-20 closures (architectural up-arms exhausted)

- **#4294 tanjiro (depth=6)** CLOSED: val=50.60 (+3.01).
- **#4286 thorfinn (mlp_ratio=3)** CLOSED: val=48.63 (+1.04 vs old / TIES on test).

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
| heads=8 at h=128 (30-min budget) | +8.73 val regression (budget incompatibility) |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA variants (all) | Regression or NO-OP |
| β1=0.95 + β2=0.95 (compound momentum) | +2.86 val regression |
| Lookahead k=8 | +2.87 val regression |
| Lookahead α=0.3 | +4.36 val regression |
| Lion β2=0.95 | +6.65 val regression |
| Lion β2=0.999 | +5.20 val regression (m-buffer too sticky) |
| Lion cfg.lr=3e-4 | +1.44 val regression |
| Lion β1=0.95 (over-smooth + Lookahead) | +3.03 val regression |
| Lookahead α=0.8 (at k=5) | +0.66 val regression |
| Lookahead k=4 (at α=0.7) | +0.71 val regression |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()`
3. T_max must equal total_epochs — T_max<total causes hard-zero LR

## Next research directions

### Priority 1 (Immediate — IN FLIGHT round 24)

- **k=6 + β2=0.995 + α=0.7 compound** (askeladd #4402) — TOP PRIORITY: composes the k-bowl shift finding with the β2=0.995 winner; predicted val ≈ 46.39 = potential NEW programme best
- **α=0.65 + β2=0.995 + k=5** (nezuko #4415) — α-bowl left-side micro-probe at new baseline; predicted 46.5-47.0 with new-winner tail risk
- **β2=0.997 right-edge probe** (fern #4384) — refines β2-bowl floor between 0.995 (winner) and 0.999 (catastrophic)
- **3-seed canonical seed=1** (tanjiro #4386) — α=0.7 + β2=0.995 with --seed 1 for paper-facing reporting
- **3-seed canonical seed=2** (thorfinn #4385) — α=0.7 + β2=0.995 with --seed 2 for paper-facing reporting

### Priority 1b (Still running round 21; results pending against OLD 47.59 baseline)

- **slice_num=32 + α=0.7** (frieren #4375) — predicted val ≈ 47.40; now needs to clear 46.84 to win outright
- **h=192 + α=0.7** (edward #4374) — first architectural probe under α=0.7
- **β1=0.88 + α=0.7** (alphonse #4376) — β1 micro-probe of asymmetric landscape

### Priority 2 (next idle-round; gated on round-24 outcomes)

- **α=0.75 + β2=0.995 + k=5** — α-bowl right-side companion to nezuko's #4415; complete the α-bowl map at new baseline
- **k=6 + slice_num=32 + α=0.7 + β2=0.995** — 4-way compound if both #4402 (k=6 stack) and #4375 (slice32 stack) show promise
- **k=7 + β2=0.995 + α=0.7** — gap-fill the new k-bowl right flank with the β2 winner (mechanism: confirm whether k-bowl asymmetry persists at β2=0.995)
- **k=5 / k=6 / k=7 at β2=0.997** — if fern's #4384 lands close to 0.995, map the (k, β2) interaction grid
- **α-frontier at k=6** — if #4402 wins, sweep α at the new k optimum (predict α-bowl narrower at k=6)

### Priority 3 (Plateau-protocol escalation if round-22+ saturates)

- **Loss reformulation:** physics-informed continuity-equation constraint, pressure-relative normalization, per-region loss weighting
- **Pseudo-labeling** from momentum-distilled teacher
- **Cosine LR restart at epoch 17** to extend training with slow-weight pull intact
- **Lion variants:** Tiger optimizer (Chen 2023's Lion successor), AdaFactor with sign-update, ScheduleFree-Lion

### Priority 4 (Loss reformulation — escalation tier 2)

- **Physics-informed loss:** soft continuity-equation constraint on surface velocity field
- **Pressure-relative target normalization:** scale-invariant pressure prediction
- **Per-region loss weighting:** decompose val_avg components and reweight

### Observations for paper

1. **k-bowl shifts under α regime:** at α=0.5, k=5 was optimal; at α=0.7, the k-bowl minimum shifted to k=6 (PR #4371). Effective-pull-rate framing α/k=0.14 (AdamW era) does NOT transfer cleanly to Lion at α=0.7 (where α/k=0.117 may be the new optimum). The basin-averaging mechanism is α-dependent, not optimizer-dependent.
2. **Lion β1 landscape is asymmetric:** raising β1 from 0.9 is ~9× worse than lowering it
3. **β2 bowl narrow at 0.995:** half-life ≈ 138 steps is the within-basin sweet spot; both lower (0.95→14 steps) and much higher (0.999→692 steps) catastrophic
4. **heads=8 is budget-incompatible at h=128:** wall-clock failure, not capacity failure
5. **Compound winners stack additively:** α=0.7 (PR #4269) + β2=0.995 (PR #4373) → combined Δ −1.13 val vs prior baseline; no destructive interaction. PR #4402 tests whether k=6 (−0.45 val at β2=0.99) also stacks with β2=0.995 (predicted: yes, → val ≈ 46.39).
6. **k-bowl asymmetric and α-dependent:** under α=0.7, gap k=4→5 is +0.71 vs gap k=7→5 is +0.89, but k=6 BEATS k=5 by −0.45 — the bowl is non-monotone and shifted.
