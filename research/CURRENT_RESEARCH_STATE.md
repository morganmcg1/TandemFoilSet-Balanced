# SENPAI Research State

- **Last updated:** 2026-05-13 13:00 (closed #2187 SWA frac=0.6 + sent #2170 nfeatures=32 back; nudged askeladd for SENPAI-RESULT on Lion+β=0.3 (val=47.64/test=40.57, biggest win ever); assigned #2285 tanjiro EMA)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (~13-15 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** Backticks in send_pr_back_to_student_with_comment args get evaluated by the shell — use Python subprocess with json.dumps for run IDs in feedback comments.

## ⭐ Current baseline (PR #1757 — β=0.3 on RFF+Kendall)

- **val_avg/mae_surf_p:** **66.6617** (seed 0, SWA-model)
- **test_avg/mae_surf_p:** **58.3234** (seed 0, SWA-model, 4-split finite)
- Config: Transolver + FiLM (mid_dim=64) + Huber **β=0.3** + Re-weight + Kendall σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=1.0)
- W&B: `sowno0vg`

**⚡ PENDING BASELINE RESET: askeladd Lion+β=0.3 run `5hp3gid7` finished at 12:26Z → val=47.64/test=40.57 (−28.5%/−30.4%). Awaiting student SENPAI-RESULT to run merge preflight. Once merged, baseline shifts to ~val=48.**

### Per-split (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 74.617 | 65.443 |
| geom_camber_rc | 79.810 | 72.473 |
| geom_camber_cruise | 44.650 | 38.187 |
| re_rand | 67.570 | 57.191 |
| **avg** | **66.662** | **58.323** |

## ✓ Merged improvements (all-time)

| PR | Slug | Baseline after merge |
|---|---|---|
| #1452 (frieren) | smooth-l1-loss | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | val=80.82, test=71.30 |
| #1731 (nezuko) | grad-clip-1p0 | val=74.62, test=66.14 |
| #1831 (nezuko) | max-norm-0p5 | val=73.81, test=65.04 |
| #1906 (askeladd) | kendall-uncertainty | val=71.43, test=62.99 |
| #2082 (alphonse) | fourier-coord-features | val=70.63, test=62.09 |
| **#1757 (frieren)** | **beta-0p3-on-rff-kendall** | **val=66.66, test=58.32 ← CURRENT** |

## 🔥 Priority action this loop

**#2063 askeladd Lion+β=0.3** — W&B `5hp3gid7` finished (val=47.64/test=40.57, −28.5%/−30.4%). Nudged student at 12:53Z. **As soon as student posts SENPAI-RESULT + marks review: run `senpai:merge-winner 2063 target/` immediately.** This is the largest single gain in the research programme.

## Current active PRs — Wave 9/10

| PR | Student | Status | Mechanism | Notes |
|---|---|---|---|---|
| **#2063** | **askeladd** | **wip (SENPAI-RESULT pending)** | Lion+β=0.3 — val=47.64 VERIFIED | Merge on next student post |
| #2168 | thorfinn | wip (rebase sent) | σ=0.5 on β=0.3 stack | Projection val ∈ [65,66.5] |
| #2170 | nezuko | wip (rebase sent) | nfeatures=32 on β=0.3 stack | val=67.73 on old stack, marginal |
| #2187 | tanjiro | CLOSED | SWA frac=0.6 +8.34% regression | EMA follow-up assigned #2285 |
| #2240 | frieren | wip | GC on β=0.3 | Zero-param optimizer hook |
| #2243 | edward | wip | β=0.2 bracket | β=0.3 appears optimal; may confirm |
| **#2269** | **fern** | wip (new) | ReZero γ=1.0 per-channel residual | Fixes LayerScale depth-starvation |
| **#2270** | **alphonse** | wip (new) | max_norm ∈ {0.75, 1.0} | clip_fraction=100% signal |
| **#2285** | **tanjiro** | wip (new) | EMA decay=0.999 on β=0.3 | Replaces SWA, no lr-flat requirement |

## Key banked mechanisms

1. **β optimum = 0.3** — β=0.1 regresses; β=0.3 wins; β=0.2 (#2243) confirms bracket
2. **SWA frac < 0.75 fails** — cosine lr only reaches flat region at epoch ~12-13. swa_start_frac bounded below by lr-schedule shape × timeout. EMA is the clean fix.
3. **Lion + β=0.3 compound** — Lion on β=0.0 stack: val=50.97; Lion on β=0.3 stack: val=47.64. ~6% additional improvement from compounding.
4. **Lion collapses Kendall σ heads** — all 6 log_σ channels = identical value. Mechanically equivalent to uniform-channel-weight. Doesn't block merge.
5. **LayerScale γ_init=1e-4 fails at 5 layers** — γ stays ~2e-5. ReZero γ=1.0 is the fix.
6. **σ-direction is monotonic** — σ=4.0 worst → σ=1.0 current → σ=0.5 better → σ=0.25 untested
7. **clip_fraction=100%** — under β=0.1 (and likely β=0.3), every gradient step hits max_norm=0.5
8. **nfeatures=32 marginal on β=0.0** — val=67.73, 0.21 above close threshold; rerun on β=0.3 may land it

## Key open bottlenecks

1. **geom_camber_rc** — still the largest absolute gap (val=79.81). Improved by RFF+β+Lion. FiLM Re-interaction axis unexplored.
2. **SWA averaging under timeout** — EMA (#2285) tests the fix. If EMA wins, consider applying EMA to Lion stack too.
3. **30-min compute bound** — only 13-15 epochs. Fastest improvements are pure config changes.

## Potential next directions (Wave 10, post-Lion-merge)

Once Lion lands as new baseline (~val=48), most in-flight PRs will need rebase+rerun against the ~48 bar:

1. **Lion-specific follow-ups** (highest priority once merged):
   - Hybrid AdamW/Lion: Lion for model params, AdamW for Kendall log_σ heads
   - Lion + drop grad-clip (max_norm=2.0 or off) — clip fires 74% under Lion
   - Lion lr fine-sweep {2e-4, 5e-4, 7e-4} around winning lr=3e-4
   - Second seed on Lion to put σ on the win magnitude

2. **σ=0.5 + β=0.3** (#2168 thorfinn in flight) — likely marginal given new Lion bar; will need rerun again

3. **Coordinate geometry** — polar/arc-length coordinates around airfoil; targets geom_camber_rc bottleneck

4. **Larger model** — hidden_dim=192 (1.7M params vs 746K); VRAM headroom ~49/96 GB

5. **AdamW β2 sweep** — β2=0.95 (Llama-style) for faster gradient adaptation on AdamW stack

## Mechanism-axis coverage

### ✓ Landed (8 axes)
1. Huber β=1.0 (#1452), 2. Per-sample Re-weight (#1586), 3. FiLM (#1585), 4. Grad-clip 1.0 (#1731), 5. Grad-clip 0.5 (#1831), 6. Kendall σ (#1906), 7. RFF σ=1.0 (#2082), 8. **Huber β=0.3 (#1757)** ← CURRENT

### 🔬 In-flight (Wave 9/10)
- Lion+β=0.3 (#2063 askeladd) — VERIFIED winner, pending SENPAI-RESULT
- σ=0.5 rerun (#2168 thorfinn), nfeatures=32 rerun (#2170 nezuko), GC (#2240 frieren), β=0.2 (#2243 edward), ReZero γ=1.0 (#2269 fern), max_norm sweep (#2270 alphonse), EMA (#2285 tanjiro)

### ✗ Closed (25+ axes)
DropPath, LayerScale γ=1e-4, β=0.1, SWA frac=0.6, OneCycle max_lr=1e-3, SDF, AdamW wd sweep, aux-Re prediction, position-jitter, mesh-subsample, FiLM variants, slice-routing, per-channel fixed weights, Re-weight sqrt, SWA-window both directions, grad-clip tight/loose extremes, asinh transform, routing temperature, positional encoding, unified pos-enc, HEM reweighting, per-token FiLM, capacity generics
