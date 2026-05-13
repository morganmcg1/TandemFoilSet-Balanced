# SENPAI Research State

- **Last updated:** 2026-05-13 12:10 (**TWO WINS PROCESSED:** merged #1757 frieren β=0.3 on RFF+Kendall → new baseline val=**66.6617**/test=**58.3234** (−5.62%/−6.06%); closed #2021 edward OneCycle (69.02 > 66.66 on β=0.3 stack, OneCycle lr=1e-3 overshoots smoother loss landscape). Assigned frieren #2240 gradient centralization + edward #2243 β=0.2 bracket. #2063 Lion+RFF+Kendall confirmed W&B winner (val=50.97) but student hasn't posted SENPAI-RESULT — now needs rebase onto β=0.3 + SENPAI-RESULT post. Alphonse running β=0.1 rerun (W&B `ss0bu7jm`). Previous: closed #1873 SDF, assigned #2220 LayerScale to fern.)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock (effective ~13 epochs with SWA)
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** GraphQL API rate-limit storms (user ID 20516801) can block student entrypoints mid-loop. REST helpers (`pr_body`, `pr_all_comments`, `gh api repos/.../issues/N/comments`) are more reliable during storms. `stale_wip` detection is unreliable during storms — verify activity via W&B before reassigning.

## ⭐ Current baseline (PR #1757 merged 2026-05-13 — β=0.3 on RFF+Kendall)

- **val_avg/mae_surf_p:** **66.6617** (seed 0, SWA-model eval) ← NEW
- **test_avg/mae_surf_p:** **58.3234** (seed 0, SWA-model, 4-split all finite) ← NEW
- Improvement over prior #2082 RFF baseline: val **−5.62%**, test **−6.06%**
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber **β=0.3**) + per-sample Re-weight + Kendall uncertainty per-channel σ + grad-clip max_norm=0.5 + RFF (16-dim, σ=1.0)
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- W&B baseline run: `sowno0vg`
- See `BASELINE.md` for full reproducible spec.

### Per-split baseline (SWA)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 74.617 | 65.443 |
| geom_camber_rc | 79.810 | 72.473 |
| geom_camber_cruise | 44.650 | 38.187 |
| re_rand | 67.570 | 57.191 |
| **avg** | **66.662** | **58.323** |

## 🔥 Hottest signals this session

- **PR #2021 (edward, OneCycleLR) CLOSED 12:00:** W&B `kqmoul4a` (β=0.3+RFF+Kendall) finished val=69.02/test=61.25 = +3.52%/+5.00% regression vs new β=0.3 baseline. OneCycle max_lr=1e-3 overshoots on the smoother β=0.3 loss landscape. **Banked:** optimal OneCycle lr is β-dependent; max_lr=1e-3 calibrated for β=1.0 loss curvature. Axis is closed on this stack unless attempting smaller max_lr.

- **🚀 PR #2063 (askeladd, Lion+RFF+Kendall) NEEDS REBASE + SENPAI-RESULT:** W&B run `6tfv6y76` completed at 11:24Z — SWA **val=50.9680 / test=43.4003** — **still clears new baseline by ~23%**. BUT: (1) β=0.3 merge created conflict → student must rebase onto new advisor branch; (2) want to confirm Lion+β=0.3 composition (β and Lion are mechanistically distinct, likely compound); (3) student still hasn't posted SENPAI-RESULT for existing run. Sent back at 11:52Z with rebase instructions and full reproduce command with `--huber_beta 0.3 --optimizer lion --lr 3e-4 --weight_decay 3e-4`. **Expected val ∈ [44, 52] if compound; even worst-case same val=50.97 clears bar 66.66.**

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

## Current research focus — Wave 9 (post-β=0.3 baseline)

**New decision rule (vs β=0.3 baseline 66.66/58.32):**
- val < 66.66: **merge**
- 66.66 ≤ val < 67.52 (within σ=0.86): too close — 2nd seed or close
- val ≥ 67.52: clear regression — close

| PR | Student | Slug | Mechanism axis | Target |
|---|---|---|---|---|
| #2063 ← REBASE | askeladd | `lion-on-rff-kendall-beta0p3` | Lion lr=3e-4 wd=3e-4 on β=0.3+RFF+Kendall stack — confirmed val=50.97 on β=0.0 stack, rebase+rerun to verify compound | val < 66.66 (likely ∈ [44, 52]) |
| #2171 (running) | alphonse | `beta-0p1-rff-kendall` | β=0.1 on RFF+Kendall stack (W&B `ss0bu7jm` running now) — will need compare vs new 66.66 baseline | val < 66.66 |
| #2243 ← NEW | edward | `beta-0p2-on-current-stack` | β=0.2 bracket between β=0.3 (66.66 baseline) and β=0.1 (alphonse in flight) | val < 66.66 |
| #2240 ← NEW | frieren | `gradient-centralization-on-beta0p3` | Gradient Centralization (Yong 2020) — pre-update gradient projection, zero-parameter axis | val < 66.66 |
| #2168 | thorfinn | `fourier-sigma-refine` | RFF σ sweep {0.5, 2.0} on Kendall (needs rebase onto β=0.3 when done) | val < 66.66 |
| #2170 | nezuko | `fourier-nfeatures-32` | RFF num_features=32 (needs rebase onto β=0.3 when done) | val < 66.66 |
| #2187 | tanjiro | `swa-start-0p6` | Earlier SWA start (frac=0.6) — needs rebase onto β=0.3 when done | val < 66.66 |
| #2220 | fern | `layerscale-on-rff-kendall` | CaiT LayerScale γ_init=1e-4 (needs rebase onto β=0.3 when done) | val < 66.66 |

**Rebase notes:** PRs #2168/#2170/#2187/#2220/#2171 were launched before β=0.3 merged. Their results compare vs old baseline (70.63). When they post SENPAI-RESULT, I'll compare vs new 66.66 bar. If they beat 66.66: merge + request rebase if code conflicrs. If 66.66 ≤ val < 70.63: was a win vs old baseline, but now below bar — request rebase+rerun on β=0.3 stack. If val > 70.63: clear regression under any baseline — close.

**Closed this session:**
- **#1757 frieren MERGED** — β=0.3 on RFF+Kendall: val=66.66 / test=58.32 = −5.62%/−6.06%
- **#2021 edward CLOSED** — OneCycle lr=1e-3 regression on β=0.3 stack (+3.52%); mechanism doesn't survive loss landscape change
- **#1873 fern CLOSED** — SDF on RFF+Kendall regression (+6.08%)
- **#1938 tanjiro CLOSED** — per-token FiLM (+5.55%)
- **#2215 fern WITHDRAWN** — DropPath prior-closure hit (#1680)

## Decision rule (vs new 70.63 baseline)

- best-arm val < 70.63: **merge**
- 70.63 ≤ val < 71.49 (within σ=0.86): send back for 2nd seed / rerun — too close to call
- val ≥ 71.49: close (regression)
- **Test override:** test < 62.09 even if val doesn't beat 70.63 → send back for investigation

## Mechanism-axis coverage

### ✓ Landed (8 axes, baseline = 66.66)

1. Loss-shape (Huber β=1.0) → #1452
2. Loss-weighting (per-sample Re-weight) → #1586
3. Architecture-conditioning (FiLM global) → #1585
4. Optimizer-stability (grad-clip max_norm=1.0) → #1731
5. Optimizer-stability (grad-clip max_norm=0.5) → #1831
6. Loss-weighting (channel-level learned σ — Kendall) → #1906
7. Input-encoding (RFF coord features σ=1.0) → #2082
8. **Loss-shape (Huber β=0.3 — outlier suppression)** → **#1757** ← NEW BASELINE

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

1. **geom_camber_rc** val=79.81, test=72.47 (still largest per-split gap, improved from 84.06 by β=0.3). RFF + β=0.3 both helped but still the primary target.
2. **re_rand OOD gap**: val=67.57, test=57.19 (β=0.3 gave biggest test gain here −7.70%). RFF helps via coordinate encoding. FiLM interaction axis still unexplored.
3. **30-min timeout clips SWA**: hit epoch 13/15 → only 2 SWA epochs. #2187 tanjiro (swa_start_frac=0.6) directly targets this.

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
- **#2063 askeladd Lion:** needs SENPAI-RESULT post for old run + rebase onto β=0.3 + new rerun. Expected val ∈ [44, 52] if compound with β=0.3. Highest-priority pending merge.
- **#2171 alphonse β=0.1:** W&B run `ss0bu7jm` running now. Will compare vs new 66.66 baseline. If wins → β is better at 0.1; if loses → β=0.3 is optimal.
- **#2243 edward β=0.2:** fresh assignment 2026-05-13 12:05. Closes the {0.1, 0.2, 0.3} bracket.
- **#2240 frieren GC:** fresh assignment 2026-05-13 12:05. First optimizer-hook test on this stack.
- **#2168 thorfinn σ-refine {0.5, 2.0}:** running on β=0.0 stack — will compare vs 66.66 when results land. If meets bar: merge; if 66.66 < val < 70.63: was win on old bar, send back for rebase+rerun on β=0.3.
- **#2170 nezuko nfeatures=32:** same situation as thorfinn — compare vs 66.66 when lands.
- **#2187 tanjiro swa-start-0.6:** same — compare vs 66.66 when lands.
- **#2220 fern LayerScale γ_init=1e-4:** same — compare vs 66.66 when lands.
- **Researcher-agent output:** `/research/RESEARCH_IDEAS_2026-05-13_12:00.md` available for wave-10+ ideas after current wave completes.
