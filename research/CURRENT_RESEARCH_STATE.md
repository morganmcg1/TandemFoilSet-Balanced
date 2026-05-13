# SENPAI Research State

- **Last updated:** 2026-05-13 12:50 (closed #2220 LayerScale + #2171 β=0.1; sent #2168 σ=0.5 back for β=0.3 rerun; askeladd Lion β=0.3 rerun in progress `5hp3gid7`; assigned #2269 fern ReZero γ=1.0 + #2270 alphonse max_norm sweep)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (effective ~13 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** GraphQL API rate-limit storms (user ID 20516801) can block student entrypoints mid-loop. REST helpers (`pr_body`, `pr_all_comments`, `gh api repos/.../issues/N/comments`) are more reliable during storms.

## ⭐ Current baseline (PR #1757 merged 2026-05-13 — β=0.3 on RFF+Kendall)

- **val_avg/mae_surf_p:** **66.6617** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **58.3234** (seed 0, SWA-model, 4-split all finite)
- Improvement over prior #2082 RFF baseline: val **−5.62%**, test **−6.06%**
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber **β=0.3**) + per-sample Re-weight + Kendall uncertainty per-channel σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=1.0)
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- W&B baseline run: `sowno0vg`

### Per-split baseline (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 74.617 | 65.443 |
| geom_camber_rc | 79.810 | 72.473 |
| geom_camber_cruise | 44.650 | 38.187 |
| re_rand | 67.570 | 57.191 |
| **avg** | **66.662** | **58.323** |

## 🔥 Hottest pending signals

- **🚀 PR #2063 (askeladd, Lion+β=0.3+RFF+Kendall) — IN PROGRESS:** W&B `5hp3gid7` started 12:03Z, ~20 min remaining. Lion on β=0.0+RFF+Kendall stack landed val=50.97 / test=43.40 (−27.85%/−30.10%). β=0.3 rerun expected val ∈ [44, 52] — largest pending merge in the programme. When done: run preflight + `senpai:merge-winner 2063 target/`.

- **PR #2168 (thorfinn, σ=0.5 rerun) — SENT BACK:** σ=0.5 won old baseline by −0.66% but loses new β=0.3 baseline. σ-direction is monotonic and real. Sent back for β=0.3 rerun with σ=0.5 (and optionally σ=0.25). Projection: val ∈ [65.0, 66.5] if compounds.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 09:15 UTC, zero open issues.

## ✓ Merged improvements (all-time)

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss | MSE → Huber (β=1.0) + NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final epochs | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample Re-weight | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | FiLM global conditioning | val=80.82, test=71.30 |
| #1731 (nezuko) | grad-clip-1p0 | Grad-clip max_norm=1.0 | val=74.62, test=66.14 |
| #1831 (nezuko) | max-norm-0p5 | Tighter grad-clip max_norm=0.5 | val=73.81, test=65.04 |
| #1906 (askeladd) | kendall-uncertainty | Learned per-channel σ heads | val=71.43, test=62.99 |
| #2082 (alphonse) | fourier-coord-features | RFF σ=1.0, 16-dim | val=70.63, test=62.09 |
| **#1757 (frieren)** | **beta-0p3-on-rff-kendall** | **Huber β=0.3** | **val=66.66, test=58.32 ← CURRENT** |

## Current research focus — Wave 9 (post-β=0.3 baseline)

**Decision rule (vs β=0.3 baseline 66.66/58.32):**
- val < 66.66: **merge**
- 66.66 ≤ val < 67.52 (within σ=0.86): too close — 2nd seed or close depending on test
- val ≥ 67.52: clear regression — close

| PR | Student | Slug | Status | Mechanism | Target |
|---|---|---|---|---|---|
| **#2063** | **askeladd** | `lion-on-rff-kendall-beta0p3` | **IN PROGRESS** `5hp3gid7` | Lion lr=3e-4 wd=3e-4 + β=0.3 — highest-priority pending merge | val << 66.66 |
| #2168 | thorfinn | `fourier-sigma-refine` | SENT BACK (β=0.3 rerun) | σ=0.5 on β=0.3 stack | val ∈ [65,66.5] |
| #2170 | nezuko | `fourier-nfeatures-32` | wip (β=0.0 stack) | RFF num_features=32 — compare vs 66.66 when lands | val < 66.66 |
| #2187 | tanjiro | `swa-start-0p6` | wip (β=0.3 stack) | Earlier SWA start frac=0.6 | val < 66.66 |
| #2240 | frieren | `gradient-centralization-on-beta0p3` | wip | GC (Yong 2020) on β=0.3 stack | val < 66.66 |
| #2243 | edward | `beta-0p2-on-current-stack` | wip | β=0.2 bracket | val < 66.66 |
| **#2269** | **fern** | `rezero-gamma-1p0-on-rff-kendall-beta0p3` | **ASSIGNED (new)** | ReZero γ_init=1.0 per-channel residual gain — fixes LayerScale depth-starvation | val < 66.66 |
| **#2270** | **alphonse** | `max-norm-relax-sweep-on-beta0p3` | **ASSIGNED (new)** | max_norm ∈ {0.75, 1.0} — clip_fraction=100% signal | val < 66.66 |

## Closed this wave (Wave 9 closures)

- **#2220 fern CLOSED** — LayerScale γ_init=1e-4 regression (+11.2%). Depth-starvation: CaiT init needs 24+ layers, γ_attn mean stayed ~2e-5 after 13 epochs. Same mechanism as #1680 DropPath. **Axis remains open** via ReZero γ=1.0 (#2269).
- **#2171 alphonse CLOSED** — β=0.1 regression (+1.34% vs new β=0.3 baseline). Monotonic β trend breaks past β=0.3. clip_fraction=100% under β=0.1 (L1-like uniform gradient) prevents adequate convergence. β=0.3 is the β optimum; #2243 edward β=0.2 confirms bracket.
- **#2021 edward CLOSED** — OneCycleLR max_lr=1e-3 regression (+3.52%) on β=0.3 stack. Mechanism: β=0.3 smooth loss landscape doesn't support the high lr needed for "super-convergence."
- **#1873 fern CLOSED** — SDF on RFF+Kendall regression (+6.08%). Geometry-as-raw-input axis closed.
- **#2215 fern WITHDRAWN** — DropPath prior closure hit (#1680, +14.4%).

## Key banked mechanisms

1. **β optimum is β=0.3** — β=0.1 regresses (+1.34%), β=0.3 wins (new baseline). Edward's β=0.2 (#2243) closes the bracket.
2. **clip_fraction=100% under low-β** — β=0.1 (and likely β=0.3) has every gradient step hitting max_norm=0.5. max_norm relaxation may accelerate convergence (#2270 alphonse).
3. **LayerScale CaiT init (1e-4) fails at 5 layers** — γ_attn mean stays at ~2e-5. ReZero γ=1.0 is the correct fix (#2269 fern).
4. **σ-direction is monotonic and real** — σ=4.0 worst → σ=1.0 current → σ=0.5 better → σ=0.25 untested. σ and β=0.3 likely compose.
5. **Lion collapses Kendall σ heads** — all 6 log_σ channels converge to identical values under Lion's sign-update. Mechanically equivalent to Lion + uniform-channel-weight. Does not block merge (win is 27-30%).
6. **OneCycle lr=1e-3 is β-dependent** — calibrated for β=1.0 curvature, overshoots on β=0.3.

## Key open bottlenecks

1. **geom_camber_rc** val=79.81, test=72.47 (still largest absolute gap). RFF + β=0.3 both helped but it's still the primary target.
2. **re_rand OOD gap**: val=67.57, test=57.19. β=0.3 largest test gain here (−7.70%). FiLM Re-conditional interaction axis still unexplored.
3. **30-min timeout clips SWA**: ~13/15 epochs → only 2 SWA epochs. #2187 tanjiro (swa_start_frac=0.6) addresses this.

## Potential next research directions (Wave 10+)

1. **σ=0.25 bracket after thorfinn lands** — if σ=0.5 wins β=0.3 rerun, continue to σ=0.25
2. **Lion on β=0.3 expected ~val=48** — after merge, explore Lion-specific follow-ups: (a) Lion + hybrid AdamW for Kendall σ heads; (b) Lion + drop grad-clip (clip 74% under Lion, try max_norm=2.0 or off); (c) Lion lr fine-sweep {2e-4, 5e-4, 7e-4}
3. **num_features=32 at σ=0.5** — if both RFF wins compound, try σ=0.5 + 32 features
4. **Re-conditional attention bias** — learned per-Re-bin attention temperature in PhysicsAttention; targets the FiLM-interaction axis identified by aux-Re closure
5. **Coordinate system change** — polar/arc-length coords around airfoil; addresses geom_camber_rc's persistent gap
6. **EMA instead of SWA** — exponential moving average maintained every step; more robust to 13/15 timeout pattern than 2-epoch SWA window
7. **Larger model (hidden_dim=192)** — VRAM headroom ~49 GB used vs 96 GB available; 192 dim ~1.7M params vs current 746K
8. **Residual prediction** — predict Δ(p) from panel-method baseline rather than raw p; reduces leading-edge learning difficulty

## Mechanism-axis coverage

### ✓ Landed (8 axes, baseline = 66.66)

1. Loss-shape (Huber β=1.0) → #1452
2. Loss-weighting (per-sample Re-weight) → #1586
3. Architecture-conditioning (FiLM global) → #1585
4. Optimizer-stability (grad-clip max_norm=1.0) → #1731
5. Optimizer-stability (grad-clip max_norm=0.5) → #1831
6. Loss-weighting (channel-level Kendall σ) → #1906
7. Input-encoding (RFF coord features σ=1.0) → #2082
8. **Loss-shape (Huber β=0.3)** → **#1757** ← CURRENT BASELINE

### 🔬 In-flight (Wave 9)

- Lion optimizer (#2063 askeladd) — β=0.3 rerun in progress; 27-30% win confirmed on β=0.0 stack
- RFF σ=0.5 (#2168 thorfinn) — sent back for β=0.3 rerun
- RFF num_features=32 (#2170 nezuko) — running on β=0.0 stack
- SWA earlier-start (#2187 tanjiro) — on β=0.3 stack
- GC on β=0.3 (#2240 frieren) — optimizer-hook zero-param axis
- β=0.2 bracket (#2243 edward) — bracketing β optimum
- **ReZero γ=1.0 (#2269 fern)** — per-channel residual gain, fixes LayerScale depth-starvation ← NEW
- **max_norm relaxation {0.75, 1.0} (#2270 alphonse)** — targets clip_fraction=100% signal ← NEW

### ✗ Closed (27+ axes)

- Capacity/architecture generic: mlp_ratio, n_hidden — closed
- Stochastic regularization: drop_path, attn_dropout — closed
- Re-weight curve: sqrt variant — closed
- SWA-window: both directions — closed
- Surf/vol weighting: both directions — closed
- Loss per-domain: surf-Huber/vol-MSE — closed
- Per-channel fixed weighting: p-up, uxuy-up — both directions closed
- FiLM capacity: width=128, depth=3, tanh-bound — closed
- Slice-routing: up and down — closed
- Mesh-subsample Path B — closed
- Sample-level augmentation (Re-jitter) — closed
- Grad-clip tighten past max_norm=0.5 — closed (but relaxation still open via #2270)
- Unified positional encoding — closed
- Per-sample difficulty reweighting (HEM/EMA) — closed
- Asinh transform + Kendall — closed
- Learnable routing temperature — closed
- Position-jitter — closed
- DropPath sweep — closed (#1680, #2016, #2215)
- Aux-Re prediction — closed (FiLM already preserves Re)
- AdamW weight decay sweep — closed
- SDF as raw input channel — closed (#1873)
- OneCycleLR max_lr=1e-3 — closed on β=0.3 stack (#2021)
- **LayerScale γ_init=1e-4** — closed (#2220; depth-starvation at 5 layers) ← NEW
- **β=0.1** — closed (#2171; β=0.3 is the β optimum) ← NEW
