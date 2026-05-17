# SENPAI Research State

- **Date:** 2026-05-17 04:00
- **Launch:** willow-pai2i-48h-r1 (round 18 — Lookahead-Lion era; programme best val=47.97, **3-seed canonical complete: mean=48.68 σ̂=0.64**; **5 architectural arms in flight** (FFN width / depth / heads / slice_num↑ / slice_num↓); k=7 prediction test in flight; **Lion β2 sensitivity + Lion LR sharpness CONFIRMED — HP frontier exhausted, programme is now architecture-bound**)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=5/α=0.5) + triple-stack (PROGRAMME BEST)** | **47.9735** | **46.4900** | PR #4123, W&B `rx3negp7`, seed=0 | 3-seed canonical complete: val mean=48.68, σ̂=0.64; test mean=47.22, σ̂=0.64 |

Win threshold: **val < 47.97** (seed=0). Paper-facing 3-seed mean: val=48.68, test=47.22.

## Lookahead-Lion: seed-robustness CONFIRMED

| Recipe | Seed-0 val | Seed-1 val | Seed-2 val | best_ep pattern |
|---|---|---|---|---|
| Lookahead-AdamW k=5 | 57.22 | 78.50 ⚠️ | 57.05 | seed=1 = ep10 (bad basin) |
| **Lookahead-Lion k=5** | **47.97** | **49.21** | **48.84** | all = ep17 (cosine floor) ✓ |

3-seed mean: val=48.68, σ̂=0.64; test=47.22, σ̂=0.64. Cleanest 3-seed canonical in the programme.

## Lookahead-Lion mechanism decomposition

| Intervention | Δ val | Mechanism |
|---|---|---|
| AdamW → Lion | −8.15 | Sign-based: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | −1.10 | Slow-weight averaging: reduce per-basin commitment variance |
| **Total: AdamW → Lookahead+Lion** | **−9.25** | Orthogonal-additive composition; seed-robust |

## Lookahead-Lion hyperparameter frontier (EXHAUSTED)

| Knob | Frontier finding | Programme implication |
|---|---|---|
| **k (sync interval)** | U-curve right-shifted vs AdamW: k=2=48.84, k=3=48.20, **k=5=47.97**, k=7 IN FLIGHT (#4310) | Lion's low per-step variance → more steps between syncs |
| **α (slow-weight pull)** | 3 mechanism branches IN FLIGHT (#4269) | Round-17 trend: α=0.5 optimum near critical pull-rate 0.15 |
| **β1 (update direction)** | IN FLIGHT (#4271) — β1∈{0.85, 0.95} | Probe direction-weighting symmetry |
| **β2 (m-buffer EMA)** | CLOSED (#4264): β2=0.95 → +6.65 (largest Lion-era regress). β2 must INCREASE from 0.99 not decrease | Knob direction is wrong; β2∈{0.995, 0.999} is the open question |
| **LR (cfg.lr)** | CLOSED (#4265): cfg.lr=3e-4 → +1.44. Lion LR landscape SHARPER than AdamW; no adaptive scaling to forgive small-LR error | LR=5e-4 is near-optimal; tighter sweep would need >0.5x granularity |
| **3-seed canonical** | CLOSED (#4242): mean=48.68, σ̂=0.64 (val and test both) | Paper-ready noise floor |

**Conclusion:** The HP neighborhood around Lookahead-Lion (k=5, α=0.5) is now well-characterized. Open HP probes (k=7, α-sweep, β1) are running but unlikely to deliver >−1.0 val. **The path to val<47.97 is now architectural.**

## Plateau Protocol → Architectural Portfolio (round 18 expanded)

13+ closes since last winner (PR #4123, val=47.97 at 2026-05-16 23:45). The architectural portfolio has expanded to **5 orthogonal arms** covering FFN/depth/attention/latent capacity:

| Architectural arm | PR | Dim varied | Direction | Hypothesis |
|---|---|---|---|---|
| FFN width | #4286 thorfinn | mlp_ratio | 2 → 3 | Wider FFN; more channel mixing capacity |
| Depth | #4294 tanjiro | n_layers | 5 → 6 | Deeper net; more sequential refinement |
| Attention heads | #4304 nezuko | n_heads | 4 → 8 | More attention specialization per layer |
| Slices (up) | #4323 fern | slice_num | 64 → 96 | Richer latent attention basis |
| Slices (down) | #4325 frieren | slice_num | 64 → 32 | Capacity-light; defends against overfit at 17ep |

This is a **portfolio bet on the capacity axis** — if any one of these wins by >0 it changes the trajectory; if multiple win they may compose. **All 5 are SINGLE-ARM** to eliminate heartbeat-rerun risk.

## Active WIP experiments (round 18)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4325 | frieren | **slice_num=32 (architectural — slice count DOWN; pairs with #4323)** | **NEW (round 18)** | 5th architectural arm |
| #4323 | fern | **slice_num=96 (architectural — slice count UP; pairs with #4325)** | **NEW (round 18)** | 4th architectural arm |
| #4310 | edward | Lookahead-Lion k=7 (find Lion k-curve right edge — winner candidate) | Running | If val<47.97, NEW programme best |
| #4304 | nezuko | Lookahead-Lion + heads=8 (architectural — attention capacity) | Running | 3rd architectural arm |
| #4294 | tanjiro | Lookahead-Lion + depth=6 (architectural — transformer depth) | Running | 2nd architectural arm |
| #4286 | thorfinn | Lookahead-Lion + mlp_ratio=3 (architectural — FFN width) | Running | 1st architectural arm |
| #4271 | alphonse | Lion β1 sweep (β1∈{0.85, 0.95}) at k=5/α=0.5 | Running | Lion-era HP |
| #4269 | askeladd | Lookahead-Lion α sweep at k=5 (α∈{0.3, 0.7}) | Running | Lion-era HP — 3 mechanism branches |

**All 8 students active. Zero idle. Single-arm policy in force for all new assignments.**

### Refined α/k mechanism: critical effective-pull-rate ≈ 0.15

Across the full AdamW Lookahead α/k frontier closed in earlier rounds, optimal configs sit near α/k ≈ 0.14-0.17. Configs above this rate over-dampen fast exploration and accelerate loss-floor stagnation. This predicts Lookahead-Lion's α/k landscape will also center around 0.15, which means at k=3, α≈0.45-0.5 (matching the AdamW finding).

## Round-18 closures (Lion HP frontier exhausted)

- **#4265 fern (Lookahead-Lion LR sweep)** CLOSED: Arm 1 only, val=49.41 (cfg.lr=3e-4 → +1.44 regress). **Lion LR landscape SHARPER than AdamW.** No adaptive scaling forgives small-LR error in Lion's sign-update. Closing as canonical Lion-LR-sensitivity data. Reassigned to slice_num=96 (#4323).
- **#4264 frieren (Lookahead-Lion β2 scan)** CLOSED: Arm 1 only, val=54.62 (β2=0.95 → +6.65 regress, **largest single-knob regression in Lion era**). m-buffer half-life of 14 steps is too short at batch=4. Opposite of AdamW β2 finding (which was flat). Lion β2 should INCREASE from 0.99, not decrease. Reassigned to slice_num=32 (#4325).

## Round-17 closures (Lion 3-seed canonical complete; k-shift prediction CONFIRMED)

- **#4242 nezuko (Lookahead-Lion seed=2)** CLOSED: best run val=48.84 (best_ep=17). **3-seed canonical COMPLETE**: mean=48.68, σ̂=0.64 on val; mean=47.22, σ̂=0.64 on test. Cleanest 3-seed canonical in programme.
- **#4241 edward (Lookahead-Lion k=3)** CLOSED: best run val=48.20 (+0.23 vs k=5). **Round-16 prediction CONFIRMED.** Lion k-curve: k=2=48.84, k=3=48.20, k=5=47.97 — monotone descending; U-min at or right of k=5. Triggered Priority-1 follow-up: k=7 (edward #4310).

⚠️ **Process issue (now 4+ occurrences):** heartbeat-rerun pattern (alphonse #4202, nezuko #4242, edward #4241, fern #4265, frieren #4264) — pods retry the same arm instead of advancing, producing degraded later runs with best_ep<17. Round-17 mitigation: shifted to **SINGLE-ARM** for all new assignments. Round-18 carries this forward.

## Round-16 closure (Lion k-curve right-shift)

- **#4268 tanjiro (Lookahead-Lion k=2)** CLOSED: val=48.84 (+0.86 vs k=5=47.97). **Major mechanism finding:** the k-curve U-shape minimum **shifts right when switching AdamW → Lion**. AdamW min at k=3; Lion min at k≥5. Lion's low per-step variance reduces basin-exploration → more steps needed between syncs.

## Round-15 closure

- **#4213 thorfinn (Lookahead-AdamW k=3 α=0.8)** CLOSED: val=57.87 — **AdamW α-frontier fully exhausted**. Confirms accelerating regression beyond critical pull rate. Last AdamW PR for this launch.

## Round-14 closures (k-sweep U-curve confirmed; α-trend inversion; k=3 3-seed canonical clean)

- **#4211 askeladd (Lookahead-AdamW k=3 α sweep)** CLOSED: α=0.6 (56.31) / α=0.7 (56.24); both regress vs α=0.5 (55.97). **Key finding: α-trend INVERTS at k=3 (α=0.5 optimal) vs k=5 (α=0.7 optimal)** — α and k not independently additive
- **#4203 tanjiro (Lookahead-AdamW k=2)** CLOSED: val=56.49 > k=3=55.97. **k-sweep U-shaped, minimum at k=3**
- **#4202 alphonse (Lookahead-AdamW k=3 seed=1)** CLOSED: val=57.44 (canonical run, best_ep=17). **k=3 has NO outlier (vs k=5's 78.50)**; 3-seed mean=56.49, σ̂=0.77

## Round-13 closures (frontier closures: AdamW β2 + LR exhausted)

- **#4183 frieren (Lookahead-AdamW β2 fine scan)** CLOSED: val=57.50 (β2=0.93) / 57.28 (β2=0.97) — **β2 flat in [0.93, 0.97]**; AdamW β2 frontier closed
- **#4182 fern (Lookahead-AdamW LR sweep)** CLOSED: val=56.87 (lr=7e-4, marginal win) / 58.87 (lr=1e-3, regress) — **AdamW LR optimum in [5e-4, 7e-4]**; Lookahead does not unlock the lr=1e-3 ceiling

## Round-12 closures

- **#4224 edward (Lookahead-Lion seed=1)** CLOSED: val=49.21 / test=47.62 / best_ep=17 — **seed-robustness CONFIRMED**, seed-1 outlier eliminated by sign-based optimization
- **#4210 nezuko (Lookahead-AdamW k=3 seed=2)** CLOSED: val=56.05 / test=53.03 — k=3 era canonical (superseded by Lion era); stable-seed dispersion 0.08 MAE (tighter than k=5)

## Round-11b closures (carry-over context)

- **#4123 edward (Lookahead-Lion)** MERGED: val=47.97 — new all-time best (Δ=−9.25 vs AdamW triple-stack)
- **#4158 nezuko (Lookahead k=3)** MERGED: val=55.97 — superseded by Lion, but k=3 finding persists
- **#4160 thorfinn (Lookahead k=5 seed=1)** CLOSED: val=78.50 OUTLIER (best_ep=10)
- **#4174 alphonse (Lookahead k=5 seed=2)** CLOSED: val=57.05 canonical
- **#4175 askeladd (Lookahead α sweep at k=5)** CLOSED: superseded; reassigned to k=3 α sweep
- **#4176 tanjiro (Lookahead + SWA-of-slow)** CLOSED: NO-OP, val=57.22 bit-identical to Lookahead alone

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
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail | Regression + kick-out |
| SWA tail4 at T_max=17 cosine | Regression + non-stationary tail |
| EMA(0.999) on fast weights | Regression — same root cause |
| SWA on slow (Lookahead) weights | NO-OP — bit-identical to Lookahead alone |
| β1=0.95 + β2=0.95 (compound momentum) | +2.86 val regression |
| Lookahead k=8 | +2.87 val regression |
| Lookahead α=0.3 | +4.36 val regression |
| Lion β2=0.95 | +6.65 val regression (largest single knob, this round) |
| Lion cfg.lr=3e-4 | +1.44 val regression |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()`
3. T_max must equal total_epochs — T_max<total causes hard-zero LR

### Post-hoc weight averaging is exhausted at T_max=17

T_max=17 cosine has no stationary tail for either fast or slow trajectories. Online averaging (Lookahead's internal slow←fast sync) is the only averaging that works.

## Next research directions

### Priority 1 (Architectural portfolio — IN FLIGHT, 5 arms)

- **slice_num=96** (fern #4323) — richer latent attention basis (NEW round 18)
- **slice_num=32** (frieren #4325) — capacity-light; defends against overfit (NEW round 18)
- **heads=8** (nezuko #4304) — more attention specialization
- **depth=6** (tanjiro #4294) — deeper net
- **mlp_ratio=3** (thorfinn #4286) — wider FFN

### Priority 2 (Remaining HP probes — IN FLIGHT)

- **Lookahead-Lion k=7** (edward #4310) — direct prediction test for U-min location
- **Lookahead-Lion α sweep at k=5** (askeladd #4269) — 3 mechanism branches
- **Lookahead-Lion β1 scan** (alphonse #4271) — direction-weighting symmetry

### Priority 3 (Architectural follow-ups — pipeline for next idle round)

- **Combined-architectural compound** — if multiple of {mlp_ratio=3, depth=6, heads=8, slice_num∈{32,96}} win, compose them
- **Lookahead-Lion + h=192** — hidden-dim scale-up (gated on #4286 mlp_ratio=3 VRAM measurement; Lion's no-v_t saves headroom)
- **Lookahead-Lion + cross-slice attention** — break slice-independence assumption in Transolver
- **Lookahead-Lion + layer-wise LR decay** — finer-grained LR control to head vs backbone
- **Lion β2 INCREASE** (β2 ∈ {0.995, 0.999}) — Round-18 closure of #4264 suggests the direction is *up* not down, but probably <0.5 MAE headroom
- **Lookahead-Lion k=8** — direct extension of k=7 (#4310); if k=7 < 47.97 this is the natural next test

### Priority 4 (Loss reformulation — escalation tier 2 if architectural plateau)

- **Physics-informed loss:** soft continuity-equation constraint on surface velocity field
- **Pressure-relative target normalization:** scale-invariant pressure prediction
- **Per-region loss weighting:** decompose val_avg into surface/p/Ux/Uy components and reweight

### Priority 5 (Speculative; bigger swings)

- **SAM (sharpness-aware minimization)** with Lion as base (2× compute — budget-limited to ~8-9ep, gotcha #3 risk)
- **Pseudo-labeling** from a momentum-distilled teacher
- **LR cosine restart** at epoch 17 to extend training; cosine floor + slow-weight pull is unique
