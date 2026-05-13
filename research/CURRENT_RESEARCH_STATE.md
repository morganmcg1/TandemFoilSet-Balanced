# SENPAI Research State

- **Last updated:** 2026-05-13 14:01 (closed #2285 tanjiro EMA decay=0.999 regression; assigned #2342 tanjiro T_max ∈ {10,12} cosine sweep on Lion baseline)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (~13-15 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Parser gotcha:** Avoid inline `SENPAI-RESULT:` substring in advisor comments — parser treats any line with that substring as a terminal marker and tries `json.loads` on what follows. Use "terminal-result post" or "SENPAI_RESULT" (underscore) in prose.

## ⭐ Current baseline (PR #2063 merged 2026-05-13 13:10 — Lion lr=3e-4 wd=3e-4 on β=0.3+RFF+Kendall)

- **val_avg/mae_surf_p:** **47.6416** (seed 0, SWA-model)
- **test_avg/mae_surf_p:** **40.5651** (seed 0, SWA-model, 4-split finite)
- Improvement over prior #1757 (66.6617 / 58.3234): val **−28.54%**, test **−30.45%**
- Config: Transolver + FiLM (mid_dim=64) + Huber β=0.3 + Re-weight + Kendall σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=1.0) + **Lion lr=3e-4 wd=3e-4**
- W&B: `5hp3gid7`

### Per-split (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 48.447 | 42.396 |
| geom_camber_rc | 62.855 | 55.252 |
| geom_camber_cruise | 29.711 | 24.413 |
| re_rand | 49.553 | 40.197 |
| **avg** | **47.642** | **40.565** |

## ✓ Merged improvements (all-time)

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 | smooth-l1-loss | MSE→Huber | val=100.77, test=90.38 |
| #1554 | swa-on-huber | SWA | val=99.07, test=88.90 |
| #1586 | re-weight | Per-sample Re | val=95.75, test=86.17 |
| #1585 | film-on-huber | FiLM | val=80.82, test=71.30 |
| #1731 | grad-clip-1p0 | max_norm=1.0 | val=74.62, test=66.14 |
| #1831 | max-norm-0p5 | max_norm=0.5 | val=73.81, test=65.04 |
| #1906 | kendall-uncertainty | Kendall σ | val=71.43, test=62.99 |
| #2082 | fourier-coord-features | RFF σ=1.0 | val=70.63, test=62.09 |
| #1757 | beta-0p3-on-rff-kendall | Huber β=0.3 | val=66.66, test=58.32 |
| **#2063** | **lion-optimizer-on-beta0p3** | **Lion lr=3e-4** | **val=47.64, test=40.57 ← CURRENT** |

## Current active PRs — Wave 10 (post-Lion baseline)

**Decision rule (vs Lion baseline 47.64/40.57):**
- val < 47.64: **merge**
- 47.64–48.50 (within ~σ): too close — 2nd seed or close
- val ≥ 48.50: regression — close

| PR | Student | Status | Mechanism | Notes |
|---|---|---|---|---|
| **#2297** | **askeladd** | wip | Lion lr sweep {2e-4, 4e-4, 5e-4} | Fine-bracket the winning lr=3e-4 |
| #2168 | thorfinn | wip (rebase sent) | σ=0.5 on β=0.3 stack | Needs further rebase to Lion baseline now |
| #2170 | nezuko | wip (rebase sent) | nfeatures=32 on β=0.3 stack | Same — needs Lion rebase |
| #2240 | frieren | wip | GC on β=0.3 | Needs rebase to Lion after finishing |
| #2243 | edward | wip | β=0.2 bracket | Needs rebase to Lion after finishing |
| **#2311** | **fern** | wip | Hybrid Lion+AdamW-for-σ on Lion stack | Restores Kendall σ differentiation |
| #2270 | alphonse | wip | max_norm {0.75, 1.0} on β=0.3 | Needs rebase to Lion after finishing |
| **#2342** | **tanjiro** | wip (new) | T_max ∈ {10,12} cosine sweep on Lion stack | Faster cooling → bigger SWA flat-region window |

**⚠ Rebase wave incoming:** All 7 non-askeladd WIP PRs were launched against the β=0.3 baseline (val=66.66). When they complete and post SENPAI-RESULT, compare vs Lion baseline (47.64). Any that don't beat 47.64 go back for Lion-stack rerun. Expected outcome: most will be worth rerunning since their mechanisms are orthogonal to Lion.

## Key banked mechanisms

1. **Lion = the biggest single lever** — 28.5% win vs β=0.3 baseline; 37.7% vs RFF baseline; 52.8% vs Kendall baseline
2. **Lion collapses Kendall σ heads** — all 6 log_σ identical. Lion+Kendall ≡ Lion+uniform-weight. Hybrid AdamW-for-σ is an interesting follow-up.
3. **β=0.3 and Lion compound** — val improved 50.97→47.64 going from β=0.0 to β=0.3 on Lion stack. Mechanisms are independent.
4. **clip_fraction=100% under β** regime → max_norm relaxation (#2270) still worth testing on Lion stack.
5. **SWA frac bounded below** — only frac≥0.75 averages in flat-loss region. EMA decay=0.999 (#2285) did NOT fix it (val=70.34, regression) — its 5-epoch window dilutes late-epoch low-lr updates with stale high-lr snapshots. Right fix is schedule shape: faster cosine T_max → eta_min plateau covers more averaging window. → testing now in #2342.
6. **σ=0.5 direction real but marginal** — contributed −0.47 val on β=0.0 stack; may still compound
7. **LayerScale γ=1e-4 fails at 5 layers** — ReZero γ=1.0 is the fix (#2269 fern)
8. **β=0.3 = β optimum** — β=0.1 regresses. β=0.2 (#2243) closes the bracket.

## Key open bottlenecks

1. **geom_camber_rc** — still the largest absolute gap (val=62.86, test=55.25). All mechanisms help but it's still the hardest split.
2. **SWA only 2 averaging epochs** (timeout at 13/15 epochs) — EMA may help
3. **Lion lr = 3e-4 not yet confirmed as global optimum** — #2297 will map the curve

## Potential next directions (Wave 11)

1. **Lion + hybrid AdamW for Kendall σ heads** — restore per-channel σ differentiation. Separate param group with AdamW(lr=1e-3, wd=0) for log_σ, Lion for model. Could unlock another 2-5%.
2. **Lion + drop grad-clip** — clip fires 74% under Lion; max_norm=2.0 or off. Cross with max_norm sweep.
3. **Lion + larger model** — hidden_dim=192 (VRAM: ~49/96 GB used). More capacity may matter more under Lion.
4. **Lion + T_max sweep** — currently testing in #2342 (T_max ∈ {10,12}). If T_max=10 wins, the schedule-shape lever is bankable; if T_max=12 wins, search continues toward intermediate values; if neither wins, the current T_max=15 is already optimal under Lion's convergence profile.
5. **σ=0.5 on Lion stack** — if thorfinn's σ=0.5 shows promise on β=0.3, apply to Lion baseline too
6. ~~EMA + Lion~~ — EMA decay=0.999 closed (#2285). If T_max sweep (#2342) succeeds, that subsumes the averaging-window question.
7. **Coordinate system** — polar/arc-length around airfoil; geom_camber_rc primary target
8. **Second seed on Lion** — confirm win magnitude (currently single seed, >20σ effect makes flip unlikely)

## Mechanism-axis coverage

### ✓ Landed (9 axes)
1. Huber β=1.0 (#1452), 2. Per-sample Re-weight (#1586), 3. FiLM (#1585), 4. Grad-clip 1.0 (#1731), 5. Grad-clip 0.5 (#1831), 6. Kendall σ (#1906), 7. RFF σ=1.0 (#2082), 8. Huber β=0.3 (#1757), 9. **Lion lr=3e-4 wd=3e-4 (#2063)** ← CURRENT

### 🔬 In-flight (Wave 10)
- Lion lr fine-sweep (#2297 askeladd) — map lr optimum around 3e-4
- T_max cosine sweep on Lion (#2342 tanjiro) — direct Lion-baseline experiment
- Hybrid Lion+AdamW for Kendall σ (#2311 fern) — direct Lion-baseline experiment
- σ=0.5 on β=0.3 (#2168 thorfinn) — will need Lion rebase after completing
- nfeatures=32 on β=0.3 (#2170 nezuko) — same
- GC (#2240 frieren), β=0.2 (#2243 edward), max_norm {0.75,1.0} (#2270 alphonse) — all need Lion rebase when done

### ✗ Closed (25+ axes) — see prior entries
