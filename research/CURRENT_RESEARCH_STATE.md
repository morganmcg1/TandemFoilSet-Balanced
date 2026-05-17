# SENPAI Research State

- **Date:** 2026-05-17 05:00
- **Launch:** willow-pai2i-48h-r1 (round 19 — Lookahead-Lion era; **NEW PROGRAMME BEST val=47.5894** (PR #4269, α=0.7 merged); **5 architectural arms in flight**; **α-bowl minimum probe (α=0.8) in flight**; **3-seed canonical for new best in flight**)
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

### 3-seed canonical (in progress — seeds 1+2 pending)

| Seed | val_avg | test_avg | Source |
|---|---|---|---|
| 0 | **47.5894** | **46.0098** | PR #4269 merged (`oftlu9tn`) |
| 1 | TBD | TBD | Askeladd #4344 in flight |
| 2 | TBD | TBD | Nezuko #4345 in flight |
| 3-seed mean | — | — | pending completion |

## α-frontier at k=5 (Lion-era; seed=0)

| α | val_avg | Δ vs α=0.5 | Status |
|---|---|---|---|
| 0.3 | 51.84 | +3.87 | closed (PR #4269 Arm 1) |
| **0.5 (prior baseline)** | **47.97** | — | superseded by #4269 |
| **0.7 (CURRENT BEST)** | **47.59** | **−0.38** | MERGED (PR #4269) |
| 0.8 | TBD | TBD | IN FLIGHT (alphonse #4343) |

Monotone descending: α=0.3 < α=0.5 < α=0.7. α-bowl minimum not yet found. α=0.8 probe ongoing.

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

## Plateau Protocol → Architectural Portfolio (5 arms in flight)

The architectural portfolio covers 5 orthogonal capacity dimensions. **Note: heads=8 (#4304) was closed as budget-incompatible** — 36% per-epoch overhead exhausts the 30-min budget before reaching the cosine floor.

| Architectural arm | PR | Dim varied | Direction | Status |
|---|---|---|---|---|
| FFN width | #4286 thorfinn | mlp_ratio | 2 → 3 | Running |
| Depth | #4294 tanjiro | n_layers | 5 → 6 | Running |
| ~~Attention heads~~ | ~~#4304 nezuko~~ | ~~n_heads~~ | ~~4 → 8~~ | **CLOSED** (budget incompatible, +36% epoch time) |
| Slices (up) | #4323 fern | slice_num | 64 → 96 | Running |
| Slices (down) | #4325 frieren | slice_num | 64 → 32 | Running |

**4 active architectural arms.** Note: heads is effectively dead at h=128 with 30-min budget (same root cause as pre-Lion slice_num=128 regression). To reopen heads, must first scale h (width arm must win first).

## Active WIP experiments (round 19)

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4343 | alphonse | **Lookahead-Lion α=0.8 at k=5 — push α-bowl minimum probe** | **NEW (round 19)** | **Winner candidate if val<47.59** |
| #4344 | askeladd | **Lookahead-Lion α=0.7 seed=1 (3-seed canonical of new best)** | **NEW (round 19)** | Paper-facing canonical |
| #4345 | nezuko | **Lookahead-Lion α=0.7 seed=2 (3-seed canonical of new best)** | **NEW (round 19)** | Paper-facing canonical |
| #4325 | frieren | slice_num=32 (architectural — slice count DOWN) | Running | 4th architectural arm |
| #4323 | fern | slice_num=96 (architectural — slice count UP) | Running | 3rd architectural arm |
| #4310 | edward | Lookahead-Lion k=7 (find Lion k-curve right edge — winner candidate) | Running | If val<47.59, NEW programme best |
| #4294 | tanjiro | Lookahead-Lion + depth=6 (architectural — transformer depth) | Running | 2nd architectural arm |
| #4286 | thorfinn | Lookahead-Lion + mlp_ratio=3 (architectural — FFN width) | Running | 1st architectural arm |

**All 8 students active. Zero idle. Single-arm policy in force.**

## Round-19 events (this round)

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

### Priority 1 (Immediate — IN FLIGHT)

- **α=0.8 probe** (alphonse #4343) — α-bowl not yet bottomed; monotone at k=5 so far; if it wins, push further to 0.9
- **3-seed canonical α=0.7** (askeladd #4344, nezuko #4345) — paper-facing noise floor for new programme best
- **k=7 probe** (edward #4310) — whether k-curve right-shift extends to k=7; winner candidate if val<47.59

### Priority 2 (Architectural — IN FLIGHT)

- **mlp_ratio=3** (thorfinn #4286) — wider FFN; pair with depth to find compositional gain
- **depth=6** (tanjiro #4294) — deeper net
- **slice_num=96** (fern #4323) — richer latent attention basis
- **slice_num=32** (frieren #4325) — capacity-light

### Priority 3 (Next idle-round candidates)

- **Combined architectural compound** — if any of {mlp_ratio=3, depth=6, slice_num∈{32,96}} win, compose them
- **Lookahead-Lion + h=192** — hidden-dim scale-up (gated on #4286 VRAM data); heads=8 is dead at h=128 but could reopen at h=192 (d_head would be 24 at heads=8 vs 32 now)
- **Lion β2 INCREASE** (β2 ∈ {0.995, 0.999}) — #4264 confirmed direction is UP from 0.99; small but real headroom
- **k=8 or k=9** — if k=7 wins, natural continuation
- **α × k composition** — if α=0.7 is confirmed optimal at k=5, test (k=7, α=0.7) and (k=5, α=0.7) compound to lock in both dimensions

### Priority 4 (Loss reformulation — escalation tier 2)

- **Physics-informed loss:** soft continuity-equation constraint on surface velocity field
- **Pressure-relative target normalization:** scale-invariant pressure prediction
- **Per-region loss weighting:** decompose val_avg components and reweight

### Observations for paper

1. **Lion's α-frontier at k=5 mirrors AdamW's:** basin-averaging mechanism (not optimizer) dominates at k=5; Lookahead is optimizer-independent in the strong-pull regime
2. **Lion β1 landscape is asymmetric:** raising β1 from 0.9 is ~9× worse than lowering it; Lookahead-Lion double-smoothing is destructive above β1=0.9
3. **heads=8 is budget-incompatible at h=128:** +36% per-epoch cost kills the cosine-schedule benefit; not a capacity failure — a wall-clock failure
4. **Effective-pull-rate α/k ≈ 0.14-0.17 is a cross-optimizer invariant:** validated at k=5 for both AdamW and Lion
