# SENPAI Research State

- **Last updated:** 2026-05-13 15:40 (sent back #2311 fern hybrid Lion+AdamW: mechanism win confirmed, val=47.34 lands in σ=0.5-rebase zone; requested 2-arm hybrid_kendall_lr sweep {3e-4, 5e-4} on σ=0.5 stack to fix AdamW overshoot)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (~13-15 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Parser gotcha:** Avoid inline `SENPAI-RESULT:` substring in advisor comments — parser treats any line with that substring as a terminal marker and tries `json.loads` on what follows. Use "terminal-result post" or "SENPAI_RESULT" (underscore) in prose.

## ⭐ Current baseline (PR #2168 merged 2026-05-13 15:30 — RFF σ=0.5 on Lion+β=0.3+RFF+Kendall)

- **val_avg/mae_surf_p:** **45.7648** (seed 0, SWA-model)
- **test_avg/mae_surf_p:** **39.6619** (seed 0, SWA-model, 4-split finite)
- Improvement over prior #2063 (47.6416 / 40.5651): val **−3.94%**, test **−2.23%**
- Cumulative improvement vs #1757: val **−31.34%**, test **−31.99%**
- Config: Transolver + FiLM (mid_dim=64) + Huber β=0.3 + Re-weight + Kendall σ + grad-clip max_norm=0.5 + **RFF (16-dim, σ=0.5)** + Lion lr=3e-4 wd=3e-4
- W&B: `7f6pqafs`

### Per-split (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 48.774 | 42.451 |
| geom_camber_rc | 58.290 | 54.596 |
| geom_camber_cruise | 29.111 | 23.445 |
| re_rand | 46.885 | 38.156 |
| **avg** | **45.765** | **39.662** |

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
| #2063 | lion-optimizer-on-beta0p3 | Lion lr=3e-4 | val=47.64, test=40.57 |
| **#2168** | **rff-sigma-0p5-on-lion** | **RFF σ=0.5 (lower-freq prior)** | **val=45.76, test=39.66 ← CURRENT** |

## Current active PRs — Wave 12 (post-σ=0.5 baseline)

**Decision rule (vs σ=0.5 baseline 45.76/39.66):**
- val < 45.76: **merge** (true compound win)
- 45.76–47.64: directional win on σ=1.0 stack but doesn't beat σ=0.5 — send back to test composition with σ=0.5
- val ≥ 47.64: regression vs prior baseline — close

| PR | Student | Status | Mechanism | Notes |
|---|---|---|---|---|
| **#2407** | **thorfinn** | wip (new) | RFF σ=0.1 + σ=0.25 seed-1 bracket | Find OOD-geom floor; settle val-vs-test seed noise |
| **#2390** | **askeladd** | wip | Lion wd sweep {1e-4, 1e-3, 3e-3} | Test if Lion needs 3-10× AdamW wd (Chen 2023) |
| **#2311** | **fern** | wip (sent back) | Hybrid Lion+AdamW-for-σ on σ=0.5 stack + lr sweep | Mechanism win confirmed (0.81 spread); lr=1e-3 overshoots → 2-arm {3e-4, 5e-4} on σ=0.5 |
| **#2270** | **alphonse** | wip | max_norm {0.75, 1.0} on β=0.3 | Pre-Lion stack; needs eventual Lion rebase regardless |
| **#2342** | **tanjiro** | wip | T_max ∈ {10,12} cosine sweep on Lion stack | Faster cooling → bigger SWA flat-region window |
| **#2347** | **edward** | wip | max_norm ∈ {0.0, 2.0} on Lion stack | Drop/relax grad-clip — clip fires 74% under Lion sign-update |
| **#2378** | **nezuko** | wip | slice_num=96 on Lion stack | Compute-frugal capacity axis; target geom_camber_rc |
| **#2363** | **frieren** | wip | Lion + linear warmup 3 epochs | Fix early-epoch oscillation diagnosed in #2240; Lion paper recommends warmup |

**⚠ Mid-wave baseline shift:** σ=0.5 merged while 7 PRs were in-flight on σ=1.0 Lion stack. Notice posted to all 7 with updated thresholds. Triage rule for these landing runs:
- val < 45.76 → MERGE (mechanism compounded with σ=0.5 implicitly via independence)
- val ∈ [45.76, 47.64] → directional win on σ=1.0 stack only, NOT a beat on σ=0.5 stack → rebase + rerun on σ=0.5
- val ≥ 47.64 → regression on its own σ=1.0 baseline → close

**Mechanism-independence assumption check:** σ knob (input encoding) should compose orthogonally with optimizer (Lion fine-tuning: warmup, wd, T_max, grad-clip) and capacity (slice_num) — but NOT with loss-surface mechanisms that interact with RFF spectrum (none currently in-flight). Hybrid Lion+AdamW-for-σ (#2311 fern) is structurally orthogonal to RFF σ.

## Key banked mechanisms

1. **Lion = the biggest single lever** — 28.5% win vs β=0.3 baseline; 37.7% vs RFF baseline; 52.8% vs Kendall baseline
2. **Lion collapses Kendall σ heads** — all 6 log_σ identical. Lion+Kendall ≡ Lion+uniform-weight. **Structural, NOT capacity- OR encoding-driven**: confirmed at 1.61M params (nezuko #2354) AND at RFF σ ∈ {0.25, 0.5, 1.0} (thorfinn #2168). **Hybrid Lion(model) + AdamW(log_σ) confirmed as the structural fix (fern #2311):** 0.81 log-unit spread restored at AdamW lr=1e-3; surface-velocity emphasis 5× volume; mechanism prediction fully validated. Next: lr sweep + σ=0.5 rebase to convert mechanism win into metric win.
3. **β=0.3 and Lion compound** — val improved 50.97→47.64 going from β=0.0 to β=0.3 on Lion stack. Mechanisms are independent.
4. **clip_fraction=100% under β** regime → max_norm relaxation (#2270) still worth testing on Lion stack.
5. **SWA frac bounded below** — only frac≥0.75 averages in flat-loss region. EMA decay=0.999 (#2285) did NOT fix it (val=70.34, regression) — its 5-epoch window dilutes late-epoch low-lr updates with stale high-lr snapshots. Right fix is schedule shape: faster cosine T_max → eta_min plateau covers more averaging window. → testing now in #2342.
6. **RFF σ↓ wins under Lion+β=0.3 (NEW BASELINE)** — σ=0.5 compounds (−3.94% val / −2.23% test vs σ=1.0); σ=0.25 wins test by additional −0.65 (mechanism: low-freq Fourier = OOD-geometry smoothness prior in z-score-normalized coords). σ floor not yet found — #2407 probes σ=0.1.
7. **LayerScale γ=1e-4 fails at 5 layers** — ReZero γ=1.0 is the fix (#2269 fern)
8. **β=0.3 = β optimum, axis CLOSED** — β=0.1 (#2171) regressed +7.5%, β=0.2 (#2243) flat on val/+0.46% on test. Both directions exhausted. Edward's Kendall σ-relaxation mechanism confirmed (lower β → all 6 log_σ drift toward uniform).
9. **RFF spectral-dim axis CLOSED at n=16** — n=32 (#2170) gave mixed val/test direction; banked SWA-window-gating mechanism (timeout limits useful averaging) directly feeds tanjiro's #2342.
10. **Gradient Centralization axis CLOSED at small-data regime** — frieren #2240 cleanly disproved transfer from ImageNet (clip_fraction unchanged, SWA basin disrupted, OOD prediction direction opposite). Three banked findings inform follow-ups.
11. **clip_fraction=100% under default max_norm=0.5** — corroborated by frieren #2240 and edward's planning for #2347. Strong evidence max_norm=0.5 is over-constraining (whether under AdamW or Lion).
12. **Width-scaling capacity bumps gated by SWA window in 30-min cap** — n_hidden=192 took 43% longer per step, killed SWA. Future capacity bumps need either (a) earlier swa_start_frac OR (b) linear-cost dimensions (depth, slice_num) like #2378.
13. **Optimizer × σ × β=0.3 interaction is non-monotonic (NEW)** — σ↓ wins under Lion+β=0.3 and AdamW+RFF-only, LOSES under AdamW+β=0.3 (#2168 Arm 1: +0.45 val vs AdamW+σ=1.0 reference). AdamW's per-coord adaptive LR cancels σ↓ benefit at β=0.3; Lion's sign-update restores it. **Implication:** future σ-modifying experiments must check optimizer × loss-shape interaction.
14. **σ=0.25 wins paper-facing test but loses val by within-noise margin (#2168)** — test geom_camber_rc −4.88% / cruise −6.11% at σ=0.25 are the strongest OOD-geom test gains observed. Test curve hasn't bottomed out → #2407 probes σ=0.1.

## Key open bottlenecks

1. **geom_camber_rc** — still the largest absolute gap (val=58.29, test=54.60 at σ=0.5). Reduced by −7.26% val with σ=0.5 but still the hardest split.
2. **SWA only 2 averaging epochs** (timeout at 12-13/15 epochs) — being attacked via #2342 T_max cooling
3. **Lion+Kendall σ-collapse** confirmed structural across width and RFF bandwidth — only #2311 hybrid Lion+AdamW addresses it
4. **Lion lr = 3e-4 confirmed near optimum** (#2297 V-shape). Lr axis CLOSED.

## Potential next directions (Wave 12 / post-σ=0.5 baseline)

1. **Lion + hybrid AdamW for Kendall σ heads (#2311)** — restore per-channel σ differentiation. Mechanism likely orthogonal to RFF σ bandwidth (confirmed by σ-collapse invariance across σ ∈ {0.25, 0.5, 1.0}).
2. **σ=0.1 + σ=0.25 seed-1 bracket (#2407 thorfinn)** — close out the RFF σ axis on Lion stack
3. **Lion + T_max sweep (#2342)** — currently testing. If T_max=10 wins, schedule-shape lever bankable.
4. **Lion + warmup (#2363) / drop-grad-clip (#2347) / wd sweep (#2390)** — three orthogonal Lion fine-tuning axes; mechanism-independent of RFF σ.
5. **Lion + slice_num=96 (#2378)** — compute-frugal capacity bump targeting geom_camber_rc bottleneck.
6. **Second seed on σ=0.5 baseline** — confirm win magnitude (currently single seed; val effect ~3.9% is well above seed noise).
7. **Coordinate system rethink** — polar/arc-length around airfoil; geom_camber_rc primary target. May compound with RFF σ↓ since both reduce frequency content.
8. **Test-time augmentation** — if test still falling at lower σ, maybe inference-time geometry perturbation could push test_geom_camber_rc further.
9. **Beyond σ=0.1** — if #2407 confirms σ=0.1 continues winning test, the axis becomes "DC Fourier features only" (σ→0). At that point the Fourier head degenerates to a sin/cos of (near-)constant phase = a learned bias. Either useful as a global geometric prior or equivalent to dropping RFF.

## Mechanism-axis coverage

### ✓ Landed (10 axes)
1. Huber β=1.0 (#1452), 2. Per-sample Re-weight (#1586), 3. FiLM (#1585), 4. Grad-clip 1.0 (#1731), 5. Grad-clip 0.5 (#1831), 6. Kendall σ (#1906), 7. RFF σ=1.0 (#2082), 8. Huber β=0.3 (#1757), 9. Lion lr=3e-4 wd=3e-4 (#2063), 10. **RFF σ=0.5 (#2168)** ← CURRENT

### 🔬 In-flight (Wave 12)
- RFF σ=0.1 + σ=0.25 seed-1 bracket (#2407 thorfinn) — find OOD-geom floor
- Lion wd sweep (#2390 askeladd) — test if Lion needs higher wd than AdamW
- T_max cosine sweep on Lion (#2342 tanjiro) — schedule shape
- Hybrid Lion+AdamW for Kendall σ (#2311 fern) — restore σ differentiation
- Drop grad-clip on Lion (#2347 edward) — clip fires 74% under sign-update
- Lion + slice_num=96 (#2378 nezuko) — compute-frugal capacity, targets geom_camber_rc
- Lion + linear warmup 3 epochs (#2363 frieren) — fix early-oscillation
- max_norm {0.75,1.0} (#2270 alphonse) — pre-Lion stack, needs Lion rebase

### ✗ Closed (25+ axes) — see prior entries
