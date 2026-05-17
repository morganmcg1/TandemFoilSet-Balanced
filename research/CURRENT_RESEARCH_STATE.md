# SENPAI Research State

- **Date:** 2026-05-17 02:35
- **Launch:** willow-pai2i-48h-r1 (round 16 — Lookahead-Lion era; programme best val=47.97, seed-robust ✓; architectural escalation expanding: 2 architectural arms in flight)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baseline

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **Lookahead-Lion (k=5/α=0.5) + triple-stack (PROGRAMME BEST)** | **47.9735** | **46.4900** | PR #4123, W&B `rx3negp7`, seed=0 | seed=1 verified val=49.21 (#4224 closed); seed=2 in flight (nezuko #4242) |

Win threshold: **val < 47.97**.

## Lookahead-Lion: seed-robustness CONFIRMED

| Recipe | Seed-0 val | Seed-1 val | Seed gap | best_ep pattern |
|---|---|---|---|---|
| Lookahead-AdamW k=5 | 57.22 | 78.50 ⚠️ | +21.28 | seed=1 = ep10 (bad basin) |
| **Lookahead-Lion k=5** | **47.97** | **49.21** | **+1.24** ✓ | both = ep17 (cosine floor) ✓ |

**Mechanism story validated empirically:** Lion's sign-update kills the seed=1 outlier (94% reduction in seed gap). Lookahead-Lion is the cleanest seed-robust baseline in the programme. The provisional 2-seed mean is 48.59, σ̂≈0.62 — exceptionally tight.

## Lookahead-Lion mechanism decomposition

| Intervention | Δ val | Mechanism |
|---|---|---|
| AdamW → Lion | −8.15 | Sign-based: eliminate per-step gradient-magnitude variance |
| Lion → Lookahead+Lion (k=5/α=0.5) | −1.10 | Slow-weight averaging: reduce per-basin commitment variance |
| **Total: AdamW → Lookahead+Lion** | **−9.25** | Orthogonal-additive composition; seed-robust |

## Lookahead-AdamW sweep findings (informing Lion-era space)

**k-sweep at α=0.5 (Lookahead-AdamW, monotone):**
| k | val (AdamW) | Δ vs k=5 |
|---|---|---|
| **k=3** | **55.97** | **−1.25** |
| k=5 | 57.22 | — |
| k=8 | 60.09 | +2.87 |
| k=2 | pending (tanjiro #4203) | — |

**α-sweep at k=5 (Lookahead-AdamW, monotone):**
| α | val (AdamW) | Δ vs α=0.5 |
|---|---|---|
| 0.3 | 61.58 | +4.36 |
| **0.5** | **57.22** | — |
| 0.7 | 56.92 | −0.30 |

**3-seed canonical at k=3 (stable seeds):**
| Seed | k=3 val |
|---|---|
| 0 | 55.97 |
| 1 | pending (alphonse #4202) |
| 2 | 56.05 |
| stable-seed Δ | 0.08 (tighter than k=5's 0.17) |

These findings suggest the optimal Lookahead-Lion hyperparameters are likely (k=3, α≥0.6) — **edward is now testing the k=3 transfer to Lion (#4241).**

## Active WIP experiments

| PR | Student | Hypothesis | Status | Priority |
|----|---------|-----------|--------|----------|
| #4294 | tanjiro | **Lookahead-Lion + depth=6 (architectural — transformer depth)** | **NEW — architectural** | Orthogonal capacity dim to #4286; pair-experiment |
| #4286 | thorfinn | **Lookahead-Lion + mlp_ratio=3 (architectural — FFN width)** | Running — architectural | First architectural escalation per Plateau Protocol |
| #4271 | alphonse | Lion β1 sweep (β1∈{0.85, 0.95}) at k=5/α=0.5 | Running | Complements frieren's β2 |
| #4269 | askeladd | Lookahead-Lion α sweep at k=5 (α∈{0.3, 0.7}) | Running | Lion-era α frontier (3 mechanism branches) |
| #4265 | fern | Lookahead-Lion LR sweep (cfg.lr∈{3e-4, 7.5e-4}) | Running | Probe Lion-era LR frontier vs paper /3 default |
| #4264 | frieren | Lookahead-Lion β2 scan (Lion m-buffer EMA ∈ {0.95, 0.98}) | Running | Lion-era β2 frontier vs default 0.99 |
| #4242 | nezuko | Lookahead-Lion seed=2 (complete 3-seed canonical for new best) | Running | Closes paper-facing seed-variance story |
| #4241 | edward | Lookahead-Lion k=3 (predicted val ~48.3-48.5 — k-shift hypothesis test) | Running | Will test k-shift hypothesis (see round-16 finding) |

**Note:** All currently-running AdamW sweeps were assigned before Lookahead-Lion's val=47.97 result landed. Let them run to completion — their k/α/LR/β2 findings still inform the Lookahead-Lion hyperparameter space. Once they return, those students will be reassigned to Lion-era experiments.

## Plateau Protocol triggered (round 15)

12 closes since last winner (PR #4123, val=47.97 at 2026-05-16 23:45). All hyperparameter tuning has been pulling diminishing returns within the Lookahead-Lion neighborhood. Per the Plateau Protocol, the lab is escalating to **architectural changes**:

- Round 15 begins architectural escalation (thorfinn #4286: mlp_ratio=3 — first architectural arm)
- When other students return, assign architectural variants (h=192, cross-slice attention, layer-wise LR decay, etc.) and/or loss reformulations (physics-informed continuity loss)
- Continue Lion-era HP sweeps for those already in flight; do not start new HP-only sweeps until architectural arms have landed

### Refined α/k mechanism: critical effective-pull-rate ≈ 0.15

Across the full AdamW Lookahead α/k frontier closed this round, optimal configs sit near α/k ≈ 0.14-0.17. Configs above this rate over-dampen fast exploration and accelerate loss-floor stagnation. This predicts Lookahead-Lion's α/k landscape will also center around 0.15, which means at k=3, α≈0.45-0.5 (matching the AdamW finding).

## Round-16 closure (Lion k-curve right-shift)

- **#4268 tanjiro (Lookahead-Lion k=2)** CLOSED: val=48.84 (+0.86 vs k=5=47.97). **Major mechanism finding:** the k-curve U-shape minimum **shifts right when switching AdamW → Lion**. AdamW min at k=3; Lion min at k≥5. Lion's low per-step variance reduces basin-exploration → more steps needed between syncs. Implication: edward's #4241 (Lookahead-Lion k=3) is predicted to also regress to val ~48.3-48.5.

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
| slice_num=128 (2× attention) | −10.92 val regression |
| clip_norm=1.0 | −3.68 val regression |
| Warmup before cosine | Worsens early dynamics |
| SWA + constant-LR tail | Regression + kick-out |
| SWA tail4 at T_max=17 cosine | Regression + non-stationary tail |
| EMA(0.999) on fast weights | Regression — same root cause |
| SWA on slow (Lookahead) weights | NO-OP — bit-identical to Lookahead alone |
| β1=0.95 + β2=0.95 (compound momentum) | +2.86 val regression |
| Lookahead k=8 | +2.87 val regression |
| Lookahead α=0.3 | +4.36 val regression |

### PyTorch scheduler gotchas

1. `CosineAnnealingLR(T_max=N)` un-clamped past T_max — LR rebounds
2. `group['lr']` overrides contaminate `CosineAnnealingLR.get_lr()`
3. T_max must equal total_epochs — T_max<total causes hard-zero LR

### Post-hoc weight averaging is exhausted at T_max=17

T_max=17 cosine has no stationary tail for either fast or slow trajectories. Online averaging (Lookahead's internal slow←fast sync) is the only averaging that works.

## Next research directions

### Priority 1 (Lion-era composition + frontier tests — IN FLIGHT)

- **Lookahead-Lion k=3** (edward #4241) — k=3 was −1.25 on AdamW; expect ~−0.5 to −1.5 on Lion if it transfers
- **Lookahead-Lion k=2** (tanjiro #4268) — extend Lion k-curve below k=3; pairs with #4241
- **Lookahead-Lion seed=2** (nezuko #4242) — close 3-seed canonical for paper
- **Lookahead-Lion α sweep at k=5** (askeladd #4269) — α∈{0.3, 0.7}; 3 mechanism branches possible
- **Lookahead-Lion β2 scan** (frieren #4264) — Lion m-buffer β2 ∈ {0.95, 0.98} vs default 0.99
- **Lookahead-Lion β1 scan** (alphonse #4271) — Lion update-direction weighting β1 ∈ {0.85, 0.95}
- **Lookahead-Lion LR sweep** (fern #4265) — cfg.lr ∈ {3e-4, 7.5e-4} → Lion lr ∈ {1e-4, 2.5e-4} vs default 1.667e-4

### Priority 2 (Architectural escalation — pipeline for next idle assignments)

- **Lookahead-Lion + h=192** — hidden-dim scale-up (gated on #4286 mlp_ratio=3 VRAM measurement; we have 35.86 GB headroom of 96 GB)
- **Lookahead-Lion + cross-slice attention** — break slice-independence assumption in Transolver
- **Lookahead-Lion + layer-wise LR decay** — finer-grained LR control to head vs. backbone
- **Lookahead-Lion + heads=8 (vs 4)** — more attention heads at same hidden dim
- **Lookahead-Lion k=7 or k=8** — if k-shift hypothesis holds, k=5 may not be the Lion k-optimum either
- **Combined-Lion-optima compound** — once HP sweeps return, compose multiple −0.3 to −1.0 MAE wins
- **Combined architectural compound** — if both depth=6 AND mlp_ratio=3 win, compose for 2-axis capacity

### Priority 3 (Loss reformulation — escalation tier 2 if architectural changes plateau)

- **Physics-informed loss:** soft continuity-equation constraint on surface velocity field
- **Pressure-relative target normalization:** scale-invariant pressure prediction

### Priority 3 (architectural / compositional)

- **Lookahead-Lion + h=192 or mlp_ratio=3** (test scale-up; Lion's no-v_t saves VRAM)
- **Lookahead-Lion + LR cosine restart** (one restart at epoch 17 to extend training; cosine floor + slow-weight pull is unique)
- **Cross-slice attention** (architectural — break the slice independence assumption)

### Priority 4 (speculative; bigger swings)

- **Physics-informed loss** (continuity equation soft constraint on surface velocity)
- **SAM (sharpness-aware minimization)** with Lion as base (2× compute — budget-limited to ~8-9ep, gotcha #3 risk)
- **Layer-wise LR decay** on top of Lookahead-Lion
- **Pseudo-labeling** from a momentum-distilled model
