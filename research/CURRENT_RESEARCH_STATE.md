# SENPAI Research State

- **Last updated:** 2026-05-13 14:00 (BIG: #2063 askeladd Lion on RFF+Kendall verified via W&B `6tfv6y76` — val_avg=**50.9680**, test_avg=**43.4003** = **−27.85% / −30.10% vs current baseline 70.63/62.09**. Run completed 11:24Z but student hasn't posted SENPAI-RESULT — nudged with verified metrics. Once posted, will preflight + squash-merge. #2021 edward OneCycle RFF+Kendall rerun currently RUNNING at step 4382 (mid-training, val=75.65 pre-SWA). Previous: closed #1873 fern SDF clean negative, assigned fern #2220 LayerScale (replaces withdrawn #2215 DropPath after prior-closure registry hit).
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (effective ~13 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** GraphQL API rate-limit storms (user ID 20516801) can block student entrypoints mid-loop. REST helpers (`pr_body`, `pr_all_comments`, `gh api repos/.../issues/N/comments`) are more reliable during storms. `stale_wip` detection is unreliable during storms — verify activity via W&B before reassigning.

## ⭐ Current baseline (PR #2082 merged 2026-05-13 — RFF σ=1.0 on Kendall)

- **val_avg/mae_surf_p:** **70.6271** (seed 0, SWA-model eval) ← NEW
- **test_avg/mae_surf_p:** **62.0907** (seed 0, SWA-model, 4-split all finite) ← NEW
- Improvement over prior #1906 Kendall baseline: val **−1.13%**, test **−1.42%**
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + Kendall uncertainty per-channel σ + grad-clip max_norm=0.5 + **RFF (16-dim, σ=1.0)**
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- W&B baseline run: `2jqhk53m`
- See `BASELINE.md` for full reproducible spec.

### Per-split baseline (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 78.743 | 69.239 |
| geom_camber_rc | 84.063 | 75.741 |
| geom_camber_cruise | 50.114 | 41.418 |
| re_rand | 69.588 | 61.964 |
| **avg** | **70.627** | **62.091** |

## 🔥 Hottest signals this session

- **PR #2021 (edward, OneCycleLR + RFF+Kendall) RUNNING NOW:** Rerun on RFF+Kendall stack started W&B `kqmoul4a` at 10:59Z, currently mid-training (step 4382/4888, val_avg=75.65 pre-SWA). Should complete ~11:30Z with SWA averaging applied. The Kendall-only result was val=67.19/test=59.01 (−5.94%/−6.31% vs Kendall); expected with RFF: val ∈ [62, 67]. Will check after completion.

- **🚀 PR #2063 (askeladd, Lion + RFF+Kendall) AWAITING RESULTS POST — CONFIRMED MERGE:** Rebase rerun completed in W&B run `6tfv6y76` at 2026-05-13 11:24Z. SWA **val=50.9680 / test=43.4003** — **−27.85% / −30.10% vs current RFF+Kendall baseline (70.63/62.09)**. All 4 test splits finite (test_geom_camber_cruise=27.4556). Composition with RFF held (vs Kendall-only run 50.19, marginal RFF gain ≈0). Student hasn't posted SENPAI-RESULT marker yet — nudged at 11:28Z with verified metrics and template. **Next step: once SENPAI-RESULT lands, run preflight + squash-merge immediately.** This becomes the new baseline.

- **PR #2082 (alphonse, RFF σ=1.0) MERGED:** val=70.63 / test=62.09 — geom_camber_rc val **−4.57%** / test **−5.26%** (strongest single-split camber improvement since FiLM). RFF acts as low-frequency geometry prior (effective σ≈5 at normalized coord scale). Mechanism: distinguishes camber geometry patterns invisible to raw coords. σ=4.0 arm regressed uniformly. σ→gain is monotonically lower=better, follow-up brackets below σ=1.0 in flight (#2168 thorfinn).

- **PR #2049 (thorfinn, aux-Re prediction) CLOSED:** Both arms regress (arm 1: val=73.93, +3.5%; arm 2: val=80.96, +13.4%). **Key finding: FiLM already preserves Re information across all 5 blocks** (aux head achieves r≈0.94-0.97 correlation at every block). OOD test_re_rand gap is NOT from Re info loss — it comes from Re-conditional feature **interactions** (geometry×Re, attention behavior under shifted Re). Future test_re_rand attacks: target interactions, not preservation.

- **PR #1981 (nezuko, wd-sweep) CLOSED:** Best arm val=71.35, well within noise of Kendall baseline (71.43), now regresses vs new RFF baseline (70.63). Student's L2-norm diagnostics confirmed wd is not biting (total L2 delta of 0.043 over 13 epochs). Axis closes: wd is not a lever at lr=5e-4 and this run length.

- **PR #1873 (fern, SDF) CLOSED 2026-05-13 13:30:** Clean negative on RFF+Kendall — val=74.92 (+6.08%), test=65.69 (+5.79%). ALL 4 splits regress; even original target bottleneck geom_camber_rc gets worse (+5.22% val). **Banked mechanism findings:** (1) SDF + Kendall **compete** on test_single_in_dist headroom (pre-Kendall SDF val=74.89 ≈ Kendall+SDF val=74.92 — Kendall is no-op on top of SDF); (2) Kendall σ-head is robust to input-channel additions (σ drift ≤0.006 with +1 SDF channel); (3) **Geometry-as-raw-input axis closes on RFF+Kendall stack** — geometry needs to be injected through learned representations (RFF/attention biases), not raw scalar concat.

- **PR #2215 fern DropPath WITHDRAWN 2026-05-13 13:35:** Closed before student start. Process error: prior closure registry hit (PR #1680, fern, val=109.52 = +14.4% regression; **layer-count-dependent under-convergence pathology — at 5 layers, dropping any block removes 20% of forward path**). Same mechanism concern as #2016 withdrawal. Lesson: must search closure registry before assigning DropPath-family experiments.

- **PR #2220 fern LayerScale ASSIGNED 2026-05-13 13:50:** Replacement for #2215 — CaiT-style learnable per-channel residual gain γ initialized at 1e-4 (Touvron et al. ICCV 2021). Mechanism-distinct from DropPath: scales residuals continuously rather than dropping stochastically — no under-convergence risk. Single-arm on RFF+Kendall stack.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 09:15 UTC, zero open issues on this advisor branch.

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
| **#2082 (alphonse)** | **fourier-coord-features** | **RFF σ=1.0, 16-dim** | **val=70.63, test=62.09 ← CURRENT** |

## Current research focus — Wave 8

| PR | Student | Slug | Mechanism axis | Target |
|---|---|---|---|---|
| #2168 ← NEW | thorfinn | `fourier-sigma-refine` | RFF σ sweep {0.5, 2.0} — bracket below winning σ=1.0 | val < 70.63 |
| #2170 ← NEW | nezuko | `fourier-nfeatures-32` | RFF num_features=32 (σ=1.0) — double spectral dim | val < 70.63 |
| #2171 ← NEW | alphonse | `beta-0p1-rff-kendall` | Huber β=0.1 on RFF+Kendall stack | val < 70.63 |
| #1757 (rerun) | frieren | `beta-0p3-on-rff-kendall` | β=0.3 on full current stack (was on pre-Kendall) | val < 70.63 |
| #2063 ← REVISED | askeladd | `lion-optimizer-on-rff-kendall` (rebase pending) | Lion lr=3e-4 wd=3e-4 on full RFF+Kendall stack — rerun after verified 30% win on Kendall-only required rebase | val < 70.63 (likely val ∈ [48, 60] given Lion-on-Kendall = 50.19) |
| #2021 ← RERUN | edward | `onecycle-maxlr-1e-3-on-rff-kendall` (rebase pending) | OneCycleLR max_lr=1e-3 + warmup on full RFF+Kendall stack — verified win val=67.19/test=59.01 on Kendall-only (−5.94%) | val < 70.63 (likely val ∈ [62, 67]) |
| #2187 ← NEW | tanjiro | `swa-start-0p6` | Earlier SWA start (frac=0.6 → 4 SWA epochs vs 2) on RFF+Kendall | val < 70.63 |
| #2220 ← NEW | fern | `layerscale-on-rff-kendall` | CaiT LayerScale γ_init=1e-4 — per-channel residual rescaling, mechanism-orthogonal | val < 70.63 |

**#1938 tanjiro CLOSED:** per-token FiLM regressed +5.55% val. 4th FiLM-head modification to regress. Shared-γ IS the right inductive bias on 1499-sample dataset. FiLM-head axis is saturated.

**#1873 fern CLOSED 2026-05-13:** SDF on RFF+Kendall, val=74.92 = +6.08% regression. Banked: SDF/Kendall compete; geometry-as-raw-input axis closed on RFF+Kendall stack.

**#2215 fern DropPath WITHDRAWN:** prior closure #1680 (layer-count-dependent under-convergence at 5 blocks). Process error — registry not searched. Replaced with #2220 LayerScale (continuous residual rescaling, no under-convergence risk).

## Decision rule (vs new 70.63 baseline)

- best-arm val < 70.63: **merge**
- 70.63 ≤ val < 71.49 (within σ=0.86): send back for 2nd seed / rerun — too close to call
- val ≥ 71.49: close (regression)
- **Test override:** test < 62.09 even if val doesn't beat 70.63 → send back for investigation

## Mechanism-axis coverage

### ✓ Landed (7 axes, baseline = 70.63)

1. Loss-shape (Huber β=1.0) → #1452
2. Loss-weighting (per-sample Re-weight) → #1586
3. Architecture-conditioning (FiLM global) → #1585
4. Optimizer-stability (grad-clip max_norm=1.0) → #1731
5. Optimizer-stability (grad-clip max_norm=0.5) → #1831
6. Loss-weighting (channel-level learned σ — Kendall) → #1906
7. **Input-encoding (RFF coord features σ=1.0)** → **#2082** ← NEW

### 🔬 In-flight (wave 8)

- β=0.1 on RFF+Kendall (#2171 alphonse) — new outlier-suppression strength
- β=0.3 on RFF+Kendall (#1757 frieren) — known-working mechanism, needs RFF+Kendall rerun
- RFF σ refinement {0.5, 2.0} (#2168 thorfinn) — bracket the winning bandwidth
- RFF num_features=32 (#2170 nezuko) — more spectral coverage
- Lion optimizer sweep (#2063 askeladd) — fresh optimizer-family axis
- OneCycleLR sweep (#2021 edward) — schedule-axis
- SWA earlier-start (#2187 tanjiro) — `swa_start_frac=0.6` (4 averaging epochs vs current 2)
- LayerScale (#2220 fern) — CaiT-style per-channel residual gain, architecture-rescaling axis

### ✗ Closed (26 axes)

- Capacity/architecture generic: mlp_ratio, n_hidden — closed
- Stochastic regularization: drop_path=0.1, attn_dropout — closed
- Re-weight curve: sqrt variant — closed
- SWA-window: both directions — closed
- Surf/vol weighting: both directions — closed
- Loss per-domain: surf-Huber/vol-MSE at FiLM-scale — closed
- Per-channel fixed weighting: p-up (#1702), uxuy-up (#1821) — both directions closed
- FiLM capacity width: mid_dim=128 — closed
- FiLM capacity depth: depth=3 — closed
- FiLM output-bound (tanh): saturation 0% — closed
- Slice-routing upward: slice_num=128 cap-bound — closed
- Slice-routing downward: slice_num=32 routing collapse — closed
- Mesh-subsample Path B: bias contamination — closed
- Sample-level input augmentation (Re-jitter): closed
- Grad-clip tighten past max_norm=0.5: clip_fraction saturated — closed
- Unified positional encoding (grid-based): regression — closed
- Per-sample loss-difficulty reweighting (HEM/EMA): "difficulty ≠ OOD-distance" — closed
- Asinh transform + Kendall: σ-adaptation mechanism incompatibility — closed
- Learnable routing temperature: drift <10%, routing-sharpness axis closed
- Position-jitter (node coord noise): 2-arm × 2-baseline regression — closed
- DropPath sweep: closed (#1680 uniform 0.1 val=109.52 = +14.4% regression, layer-count-dependent under-convergence at 5 blocks; #2016 + #2215 withdrawn for same mechanism concern)
- **Aux-Re prediction (pooled hidden state)**: FiLM already preserves Re — closed
- **AdamW weight decay sweep {3e-4, 1e-3}**: not biting at this lr/budget — closed
- **Per-node SDF as input channel**: closed on RFF+Kendall (#1873) — SDF/Kendall compete on test_single_in_dist; geometry-as-raw-input axis closed ← NEW

## Key open bottlenecks

1. **geom_camber_rc** val=84.06, test=75.74 (largest remaining per-split gap). RFF helped (−5.26% test) but still the biggest target.
2. **test_re_rand OOD gap**: val=69.59, test=61.96 (now confirmed: NOT from Re info loss — FiLM preserves Re; gap from Re-conditional feature interactions).
3. **30-min timeout clips SWA**: both RFF arms hit epoch 13/15 → only 2 SWA epochs. Improving training efficiency or reducing per-epoch cost could unlock the full 4-epoch SWA window.

## Potential next research directions (wave 9+)

1. **β=0.3 / β=0.1 on RFF+Kendall** (in flight #2171, #1757) — if either lands, opens β sweep bracketing {0.05, 0.3, 0.5}
2. **SDF on current RFF+Kendall stack** (fern #1873 rerun) — geometry-aware × multi-task × input-encoding compound test
3. **OneCycleLR if edward lands** — opens: anneal schedule variants, warm-restart strategies
4. **Per-token FiLM if tanjiro lands** — opens: geometry-conditioned FiLM split heads, per-surface/volume modulation weights
5. **RFF follow-ups from wave 8:** if σ=0.5 wins → try σ=0.25; if num_features=32 wins → try 64
6. **Re-conditional feature interaction attacks** (motivated by aux-Re finding): geometry×Re cross terms in FiLM conditioning head, learned per-Re-bin attention temperature
7. **Coordinate systems**: try polar/cylindrical coordinates around airfoil instead of Cartesian; or per-airfoil normalized coordinates (arc-length parametrization)
8. **Learned SDF embedding**: replace log1p+standardize SDF scalar with MLP[1→8] — richer geometry representation
9. **Residual prediction with physics prior**: predict Δ(p) from baseline panel-method estimate rather than raw p — reduces learning difficulty near leading edge
10. **Data augmentation via Re-interpolation**: synthetic training samples from Re-interpolated flow fields — directly addresses the OOD-Re gap

## Open questions to revisit on next review

- **#2021 edward OneCycleLR:** arm 1 (max_lr=5e-4) finished at val=71.39 > new bar (70.63). Does arm 2 (max_lr=1e-3) clear 70.63?
- **#2063 askeladd Lion rebase:** rerun on full RFF+Kendall pending. Verified 30% win on Kendall-only — prediction val ∈ [48, 60].
- **#2021 edward OneCycle rebase:** rerun on full RFF+Kendall pending. Verified 6% win on Kendall-only — prediction val ∈ [62, 67].
- **#1757 frieren β=0.3:** Rerun on full current stack pending. High confidence this will improve (mechanism confirmed on 3 prior stacks).
- **#2168 thorfinn σ-refine {0.5, 2.0}:** brackets winning σ=1.0. If σ=0.5 lands → continue to σ=0.25.
- **#2170 nezuko nfeatures=32:** more spectral coverage at σ=1.0.
- **#2171 alphonse β=0.1:** outlier-suppression on RFF+Kendall stack.
- **#2187 tanjiro swa-start-0.6:** 4 averaging epochs vs 2 — directly attacks the 30-min timeout SWA clip.
- **#2220 fern LayerScale γ_init=1e-4:** fresh axis just assigned 2026-05-13 13:50.
- **Researcher-agent output:** `/research/RESEARCH_IDEAS_2026-05-13_12:00.md` available for wave-9+ ideas.
