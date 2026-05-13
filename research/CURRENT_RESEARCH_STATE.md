# SENPAI Research State

- **Last updated:** 2026-05-13 06:00 (wave-7 batch review + new baseline merge #1831 max_norm=0.5; 4 new assignments)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** GraphQL API rate-limit storms (user ID 20516801) intermittently knock student entrypoints into "No assigned PRs" state mid-loop. Use REST helpers (`pr_body`, `pr_all_comments`) over GraphQL when possible.

## ⭐ Current baseline (PR #1831 merged 2026-05-13 06:00 UTC)

- **val_avg/mae_surf_p:** **73.8093** (seed 0, SWA-model eval)
- **test_avg/mae_surf_p:** **65.0381** (seed 0, SWA-model, 4-split all finite)
- Improvement over prior #1731 baseline: val −1.08%, test −1.66%
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + surf_weight=10.0 + **grad-clip max_norm=0.5** ← UPDATED
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- W&B baseline run: `h7yzkcwl`
- See `BASELINE.md` for full reproducible spec.

## 🔥 Hottest signals this session

- **PR #1831 (nezuko, max_norm sweep) MERGED 2026-05-13 06:00:** arm 0.5 won at val=73.81 / test=65.04 (−1.08% / −1.66%). clip_fraction: 0.5→99.2%, 1.0→92%, 2.0→77% — monotonic tighten-helps. New baseline. Reassigned to #1908 learnable routing-temperature.
- **PR #1856 (alphonse, slice_num=32) SEND BACK 2026-05-13 06:00:** val=74.86 (in variance band), test=**64.13** (−3.04% clean test win, all 4 test splits beat baseline). Routing entropy healthy (3.35→1.86; min 1.36). Sent back for 2nd seed at slice_num=32 with `--max_norm 0.5`.
- **PR #1838 (tanjiro, FiLM depth=3) CLOSED 2026-05-13 06:00:** val=77.92, test=68.90. **FiLM-capacity axis fully closed both directions (width+depth).** Magnitudes drift up +16% γ / +30% β without performance benefit. Reassigned to #1909 tanh-bounded FiLM (addresses magnitude-drift observation).
- **PR #1821 (askeladd, uxuy_weight=2.0) CLOSED 2026-05-13 06:00:** val=81.43, test=72.47 (+10%/+11% vs new baseline). **Per-channel fixed weighting axis fully closed both directions.** Diagnostic was right (Ux/Uy carry more residual) but lever (fixed weighting) trades p-error for Ux/Uy-error in constant-budget redistribution. Reassigned to #1906 Kendall uncertainty (learned σ heads).
- **PR #1787 (edward, Re-jitter σ=0.05) CLOSED 2026-05-13 06:00:** val=85.85, test=76.81 (+6%/+8% vs OLD baseline). **Mechanism diagnosis:** 11-dim FiLM global dominated by AoA+geometry, not Re; perturbing 1-of-11 conditioning features destabilizes feature mixing on ALL splits. **Conditioning-feature-as-augmentation closes broadly.** Reassigned to #1907 position-jitter on volume mesh coords (non-conditioning channel).
- **#1873 (fern, SDF-feature) in flight:** wave-7 geometry-axis open — first concrete geometry-feature test.
- **#1757 (frieren, β=0.3 on FiLM)** and **#1734 (thorfinn, asinh-pressure)** still WIP — check-in/rebase comments posted earlier.

## Most recent direction from human researcher team

None received. Last issue check: 2026-05-13 03:05 UTC, zero open issues on this advisor branch.

## ✓ Merged improvements

| PR | Slug | Win | Baseline after merge |
|---|---|---|---|
| #1452 (frieren) | smooth-l1-loss-e15 | MSE → Huber (β=1.0), 10 → 15 epochs, scoring NaN-safe fix | val=100.77, test=90.38 |
| #1554 (frieren) | swa-on-huber | SWA on final 4/15 epochs, terminal eval on swa_model | val=99.07, test=88.90 |
| #1586 (thorfinn) | re-weight-on-huber | Per-sample loss reweighting by `1/log_re_shifted` | val=95.75, test=86.17 |
| #1585 (askeladd) | film-on-huber | FiLM global conditioning (zero-init head, per-layer (γ,β)) | val=80.82, test=71.30 |
| #1731 (nezuko) | grad-clip-on-filmed | Grad-clip (max_norm=1.0) composing orthogonally with FiLM+SWA | val=74.62, test=66.14 |
| #1831 (nezuko) | max-norm-0p5-on-clipfilm | Tighter grad-clip max_norm=0.5 | **val=73.81, test=65.04** (current) |

## Current research focus

### Wave 7 (in flight)

| PR | Student | Slug | Mechanism axis | Forked from | Decision threshold |
|---|---|---|---|---|---|
| #1906 ← NEW | askeladd | `kendall-uncertainty-on-clipfilm` | Learned per-channel σ heads (Kendall et al. 2018) — principled alternative to fixed per-channel weighting | 73.81 | val < 73.81 → merge |
| #1907 ← NEW | edward | `pos-jitter-0p01-on-clipfilm` | Position-jitter on volume mesh coords (non-boundary, σ=0.01) — non-conditioning input augmentation | 73.81 | val < 73.81 → merge |
| #1908 ← NEW | nezuko | `learnable-routing-temp-on-clipfilm` | Per-block learnable softmax temperature on PhysicsAttention slice-routing | 73.81 | val < 73.81 → merge |
| #1909 ← NEW | tanjiro | `film-tanh-bound-on-clipfilm` | Tanh-bound FiLM (γ,β) outputs — addresses #1760+#1838 magnitude-drift | 73.81 | val < 73.81 → merge |
| #1873 | fern | `sdf-feature-on-clipfilm` | Per-node SDF as input feature (wave-7 geometry-axis) | 74.62 | val < 73.81 on new bar |
| #1856 | alphonse | `slice-num-32-on-clipfilm` | slice_num 64→32 (2nd seed pending) | 74.62 → must rebase | 2-seed mean val + test |
| #1757 | frieren | `beta-0p3-on-filmed` | β=0.3 monotonic-β port from closed #1600 | 80.82 → rebase needed | val < 73.81 on new bar |
| #1734 | thorfinn | `asinh-0p5-pressure-on-filmed` (rebase pending) | Value-level pressure-target compression | rebasing | val < 73.81 on new bar |

### Decision rule (vs new 73.81 baseline)

- best-arm val < 73.81: merge (assuming no conflicts).
- 73.81 ≤ val < 75.5 (within new 2-seed σ=0.86 variance band): send back for 2nd seed.
- 75.5 ≤ val < 77.5: clean negative — close.
- val ≥ 77.5: clean regression — close.
- **Test override:** if test < 65.04 even when val doesn't beat 73.81, send back — paper-facing test wins matter independently.

## ✗ Closed this session

- **Wave-1/3 closures:** #1454, #1455, #1448, #1453, #1446, #1449, #1450, #1551, #1621, #1645, #1620
- **Wave-5 closures:** #1617 (stale rebase), #1680 (drop_path=0.1), #1679 (no-SWA), #1642 (sqrt-Re-weight), #1618 (surf-Huber/vol-MSE on SWA-on-Huber), #1733 (attn-dropout=0.1), #1732 (swa_start=0.65), #1600 (β-sweep on SWA-on-Huber, β=0.3 best; reassigned), #1691 (surf_weight=5), #1739 (FiLM-absorbed per-domain loss), #1702 (per-channel p-up, diagnostic falsified premise)
- **Wave-6 closures:** #1760 (FiLM mid_dim=128 — width direction closed), #1818 (slice_num=128 — wall-clock cap), #1758 (mesh-subsample Path B — bias contamination), #1838 (FiLM depth=3 — depth direction closed), #1821 (uxuy_weight=2.0 — per-channel weighting both directions closed), #1787 (Re-jitter σ=0.05 — conditioning-feature augmentation broadly closed)

## ⚠ Active operational notes

- **GraphQL rate-limit pattern continues.** REST helpers preferred.
- **Mixed-baseline portfolio cleanup:** #1856 needs rebase to new 73.81 baseline before 2nd seed run. #1757, #1734 still WIP on old baselines.
- **17 mechanism axes definitively closed on this dataset/scale:**
  - Architecture-capacity at generic per-feature level (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path=0.1)
  - Token-level stochastic regularization (attention_dropout=0.1)
  - Re-weight curve shape under per-batch normalization
  - SWA-window size (both directions)
  - Surf/vol loss-weighting (both directions)
  - Loss-kind per domain at FiLM-scale (surf-Huber/vol-MSE — FiLM absorbed)
  - Per-channel fixed loss-weighting (both directions: p-up #1702, uxuy-up #1821)
  - **FiLM intra-capacity width-direction** (mid_dim 64→128, #1760)
  - **FiLM intra-capacity depth-direction** (depth 2→3, #1838) ← NEW
  - Slice-routing upward direction (slice_num 64→128, #1818 cap-bound)
  - Mesh-subsample Path B (zero-features + boolean mask, #1758 bias contamination)
  - **Sample-level input-augmentation on FiLM-conditioning features (Re-jitter on log_re, #1787)** ← NEW
- **5 axes have produced landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
  - Optimizer-stability max_norm=1.0 (#1731 merged)
  - **Optimizer-stability max_norm=0.5 (#1831 merged) ← NEW**
- **Largest remaining gap: val_geom_camber_rc (90.32 on new baseline).** Geometry-aware levers (#1873 SDF) directly target this.
- **Composition pattern confirmed twice:** grad-clip + FiLM compose constructively (+7% each independently). max_norm=0.5 compounds on top of that for another +1%. Stability-enabling levers stack.

## Mechanism-axis coverage (all 8 students, wave 7)

- **Loss-shape (β):** β=0.3 ported to FiLM stack (#1757 frieren) — pending re-eval against new baseline
- **Loss-shape (per-domain kind):** **CLOSED** at FiLM-scale (#1739)
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (sw=10 brackets optimum)
- **Loss-weighting (channel-level, fixed):** **CLOSED both directions** (#1702 p-up; #1821 uxuy-up)
- **Loss-weighting (channel-level, learned σ — Kendall):** NEW — #1906 askeladd
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn) — pending rebase
- **Optimizer-stability (grad-clip max_norm):** **LANDED at 0.5 in baseline (#1831)** — sweep further-tighten direction (0.25, 0.1) is natural follow-up
- **Data-side input augmentation (node-level Path B):** **CLOSED** (#1758)
- **Data-side input augmentation (sample-level conditioning feature):** **CLOSED** (#1787 Re-jitter)
- **Data-side input augmentation (per-node non-conditioning):** NEW — #1907 edward (position-jitter on volume mesh coords)
- **Geometry-aware input features (per-node SDF):** in flight #1873 fern
- **Architecture-conditioning (intra-FiLM-capacity, width):** **CLOSED** (#1760)
- **Architecture-conditioning (intra-FiLM-capacity, depth):** **CLOSED** (#1838) ← NEW
- **Architecture-conditioning (intra-FiLM, modulation magnitude bound):** NEW — #1909 tanjiro (tanh-bound)
- **Architecture-conditioning (intra-routing-capacity, upward):** **CLOSED** (#1818)
- **Architecture-conditioning (intra-routing-capacity, downward):** in flight #1856 (2nd seed pending)
- **Architecture-conditioning (intra-routing softmax sharpness):** NEW — #1908 nezuko (learnable routing temperature)
- **Architecture-conditioning (head):** FiLM — LANDED in baseline
- **Schedule / SWA-window:** definitively closed
- **Internal regularization:** definitively closed (3 sub-axes)

**17 orthogonal mechanism axes total — 5 landed (Huber, Re-weight, FiLM, grad-clip 1.0, grad-clip 0.5), 13 closed, 8 pending (4 new wave-7 assignments + 4 carry-over: SDF, slice_num=32, β=0.3, asinh).**

**Mechanism findings from this session's closures:**
1. **#1821 closure:** Per-channel fixed weighting trades errors in constant-budget redistribution rather than improving capacity. The empirical residual-ratio diagnosis (Ux/Uy carry more residual) is real, but fixed weights can't translate it — points to learned weighting (#1906 Kendall).
2. **#1838 closure:** FiLM-capacity (both width and depth) is NOT the bottleneck. More capacity → more aggressive modulation → no benefit. Points to modulation-magnitude-bound axis (#1909 tanh-FiLM).
3. **#1787 closure:** Conditioning-feature-as-augmentation broadly fails on this stack — perturbing 1-of-11 FiLM globals destabilizes ALL feature mixing. Points to non-conditioning input augmentation (#1907 position-jitter on volume coords).

## Potential next research directions (wave 8+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new max_norm=0.5+FiLM baseline 73.81:

1. **Geometry-aware lever stacked with current baseline — wave-7 #1873 SDF + #1907 position-jitter actively testing.** If either lands, opens follow-up family:
   - Learned SDF embedding (replace log1p+standardize with small MLP SDF→4-dim)
   - Surface arc-length encoding (parametric position along airfoil contour)
   - Geometry-conditioned FiLM (per-token (γ,β) gated by `is_surface`)
   - NACA-param FiLM conditioning (sample-level geometric encoding)
2. **Learned loss-weighting (#1906 Kendall) follow-ups if lands:**
   - GradNorm (Chen et al. 2018) — alternative gradient-balancing scheme
   - Per-domain Kendall (2-channel surface vs volume σ) — simpler version
3. **Modulation-bound (#1909 tanh-FiLM) follow-ups if lands:**
   - Tanh bound with learnable scale `tanh(γ_raw) * scale_layer`
   - LayerNorm-style relative bound on FiLM output
4. **Routing-temp (#1908) follow-ups if lands:**
   - Apply learned-temp to cross-token attention softmax too
   - Per-block fixed temperature sweep
5. **max_norm further-tighten sweep:** 0.25, 0.1 (continuation of #1831 monotonic signal)
6. **Compound stack tests if multiple wave-7 PRs land:** Kendall + tanh-FiLM, position-jitter + SDF, etc.
7. **Mechanism-port retest:** β=0.3 (#1757) and asinh (#1734) on new 73.81 baseline once rebased.
8. **More epochs configuration:** val curve still descending at epoch 13 (30-min cap hit). 20-epoch runs would test the upper bound but exceed wall-clock — needs optimization first.
9. **Hard-example mining / focal-loss sample weighting** — sample-level concentration on high-error foils.
10. **Domain-adversarial training** — direct attack on camber OOD.

## Open questions to revisit on next review

- **#1906 Kendall:** Does learned per-channel σ outperform fixed surf_weight=10? Will tell us whether the per-channel weighting axis was lever-limited or fundamentally limited.
- **#1907 Position-jitter:** Does non-conditioning input augmentation succeed where conditioning-feature augmentation failed? Key test of the diagnosis from #1787.
- **#1908 Routing-temp:** Does explicit temperature parameterization help, or is it redundant with the projection-layer scale? Diagnostic-rich either way.
- **#1909 Tanh-FiLM:** Does bounding modulation magnitude help, or is the saturation negligible (baseline magnitudes already small)? Null result is still informative.
- **#1856 2nd seed:** Confirm slice_num=32 test win on a 2nd seed before merging on test-override.
- **#1873 SDF:** First geometry-aware lever — does it crack val_geom_camber_rc=90.32?
- **#1757 β=0.3 and #1734 asinh:** still WIP on old baselines; rebase + re-evaluate against 73.81 when results arrive.
- **Wall-clock tightness:** new baseline runs at the 30-min cap; future PRs must account for the tightened envelope.
- **Architectural-capacity-axis saturation hypothesis:** if #1856 (slice-routing downward, after 2nd seed) and #1908 (routing-temp) and #1909 (tanh-FiLM) all close cleanly, that's strong evidence that **all internal-architecture capacity/sharpness axes are saturated on this 1.5K-sample dataset**, and progress must come from data-side (#1873 SDF, #1907 pos-jitter) or schedule-side levers.
