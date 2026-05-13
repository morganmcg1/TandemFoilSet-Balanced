# SENPAI Research State

- **Last updated:** 2026-05-13 (wave-7 first-results batch: 2 close + 2 send-back + 2 new assignments)
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

- **PR #1909 (tanjiro, tanh-bound FiLM) CLOSED:** clean negative — tanh saturation 0% throughout (bound never engaged); mild sub-linear compression hurt broadly. **FiLM-output-bound axis closes.** Together with #1760 (width) + #1838 (depth), all FiLM intra-head axes (capacity + output-bound) closed cleanly. Reassigned to #1938 per-token FiLM (first structural FiLM change).
- **PR #1856 (alphonse, slice_num=32 2nd seed) CLOSED:** seed-0 test win didn't survive seed 1 — apples-to-apples val regression exceeds σ band, seed 1 routing collapse in block 1 (entropy 0.57, eff 1.77 slices). **Slice-routing capacity both directions closed:** upward #1818 (cap), downward this PR (routing collapse). Reassigned to #1937 max-norm further-tighten {0.25, 0.1} sweep.
- **PR #1907 (edward, pos-jitter σ=0.01) SEND BACK:** student found coord range is [-9.55, +10.55] not [-1, 1] — PR σ=0.01 was ~10x mis-scaled, mechanism never had a fair test. Sent back at σ=0.05 (~3% of coord std).
- **PR #1757 (frieren, β=0.3 on FiLM) SEND BACK:** val=72.11 / test=62.91 — strong absolute numbers, BUT ran with `--max_norm 1.0` (old baseline) not `0.5` (current). Not apples-to-apples. Sent back for rebase + rerun with `--max_norm 0.5`.
- **#1873 (fern, SDF-feature)** still WIP — wave-7 geometry-axis open.
- **#1734 (thorfinn, asinh-pressure)** still WIP / rebase pending — old baseline.
- **#1906 (askeladd, Kendall uncertainty)**, **#1908 (nezuko, learnable routing-temp)** still WIP.

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
| #1937 ← NEW | alphonse | `max-norm-tight-sweep-on-clipfilm` | Max-norm further-tighten 2-arm sweep {0.25, 0.1} — extends #1831 monotonic signal | 73.81 | best-arm val < 73.81 → merge |
| #1938 ← NEW | tanjiro | `film-per-token-on-clipfilm` | Per-token (is_surface-aware) FiLM — first structural FiLM change after capacity + output-bound axes closed | 73.81 | val < 73.81 → merge |
| #1906 | askeladd | `kendall-uncertainty-on-clipfilm` | Learned per-channel σ heads (Kendall et al. 2018) — principled alternative to fixed per-channel weighting | 73.81 | val < 73.81 → merge |
| #1907 ← rerun | edward | `pos-jitter-0p01-on-clipfilm` | Position-jitter on volume mesh coords; rerun at σ=0.05 after coord-scale finding | 73.81 | val < 73.81 → merge |
| #1908 | nezuko | `learnable-routing-temp-on-clipfilm` | Per-block learnable softmax temperature on PhysicsAttention slice-routing | 73.81 | val < 73.81 → merge |
| #1873 | fern | `sdf-feature-on-clipfilm` | Per-node SDF as input feature (wave-7 geometry-axis) | 74.62 | val < 73.81 on new bar |
| #1757 ← rerun | frieren | `beta-0p3-on-filmed` | β=0.3 monotonic-β port; rerun after rebase + `--max_norm 0.5` | 73.81 (post-rebase) | val < 73.81 on new bar |
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
- **Wave-7 closures:** #1909 (tanh-bound FiLM — output-bound axis closed, saturation 0%), #1856 (slice_num=32 2nd seed — routing collapse seed 1, slice-routing capacity downward closed)

## ⚠ Active operational notes

- **GraphQL rate-limit pattern continues.** REST helpers preferred.
- **Mixed-baseline portfolio cleanup:** #1856 needs rebase to new 73.81 baseline before 2nd seed run. #1757, #1734 still WIP on old baselines.
- **19 mechanism axes definitively closed on this dataset/scale:**
  - Architecture-capacity at generic per-feature level (mlp_ratio, n_hidden bumps) — closed twice
  - Block-level stochastic regularization (drop_path=0.1)
  - Token-level stochastic regularization (attention_dropout=0.1)
  - Re-weight curve shape under per-batch normalization
  - SWA-window size (both directions)
  - Surf/vol loss-weighting (both directions)
  - Loss-kind per domain at FiLM-scale (surf-Huber/vol-MSE — FiLM absorbed)
  - Per-channel fixed loss-weighting (both directions: p-up #1702, uxuy-up #1821)
  - FiLM intra-capacity width-direction (mid_dim 64→128, #1760)
  - FiLM intra-capacity depth-direction (depth 2→3, #1838)
  - **FiLM output-bound (tanh, #1909) ← NEW**
  - Slice-routing upward direction (slice_num 64→128, #1818 cap-bound)
  - **Slice-routing downward direction (slice_num 64→32, #1856 routing collapse) ← NEW**
  - Mesh-subsample Path B (zero-features + boolean mask, #1758 bias contamination)
  - Sample-level input-augmentation on FiLM-conditioning features (Re-jitter on log_re, #1787)
- **5 axes have produced landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
  - Optimizer-stability max_norm=1.0 (#1731 merged)
  - **Optimizer-stability max_norm=0.5 (#1831 merged) ← NEW**
- **Largest remaining gap: val_geom_camber_rc (90.32 on new baseline).** Geometry-aware levers (#1873 SDF) directly target this.
- **Composition pattern confirmed twice:** grad-clip + FiLM compose constructively (+7% each independently). max_norm=0.5 compounds on top of that for another +1%. Stability-enabling levers stack.

## Mechanism-axis coverage (all 8 students, wave 7)

- **Loss-shape (β):** β=0.3 rerun pending after rebase + `--max_norm 0.5` (#1757 frieren)
- **Loss-shape (per-domain kind):** **CLOSED** at FiLM-scale (#1739)
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (sw=10 brackets optimum)
- **Loss-weighting (channel-level, fixed):** **CLOSED both directions** (#1702 p-up; #1821 uxuy-up)
- **Loss-weighting (channel-level, learned σ — Kendall):** in flight #1906 askeladd
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn) — pending rebase
- **Optimizer-stability (grad-clip max_norm):** **LANDED at 0.5 in baseline (#1831)** — further-tighten sweep {0.25, 0.1} in flight #1937 alphonse
- **Data-side input augmentation (node-level Path B):** **CLOSED** (#1758)
- **Data-side input augmentation (sample-level conditioning feature):** **CLOSED** (#1787 Re-jitter)
- **Data-side input augmentation (per-node non-conditioning):** in flight #1907 edward — rerun at σ=0.05 after coord-scale finding
- **Geometry-aware input features (per-node SDF):** in flight #1873 fern
- **Architecture-conditioning (intra-FiLM-capacity, width):** **CLOSED** (#1760)
- **Architecture-conditioning (intra-FiLM-capacity, depth):** **CLOSED** (#1838)
- **Architecture-conditioning (intra-FiLM, modulation magnitude bound):** **CLOSED** (#1909) ← NEW
- **Architecture-conditioning (intra-FiLM, structural — per-token):** NEW — #1938 tanjiro (is_surface-aware split heads)
- **Architecture-conditioning (intra-routing-capacity, upward):** **CLOSED** (#1818)
- **Architecture-conditioning (intra-routing-capacity, downward):** **CLOSED** (#1856 routing collapse) ← NEW
- **Architecture-conditioning (intra-routing softmax sharpness):** in flight #1908 nezuko (learnable routing temperature)
- **Architecture-conditioning (head):** FiLM — LANDED in baseline
- **Schedule / SWA-window:** definitively closed
- **Internal regularization:** definitively closed (3 sub-axes)

**18 orthogonal mechanism axes — 5 landed (Huber, Re-weight, FiLM, grad-clip 1.0, grad-clip 0.5), 15 closed, 8 pending (2 new wave-7 assignments + 6 carry-over/rerun: SDF, Kendall, routing-temp, pos-jitter rerun, β=0.3 rerun, asinh, max-norm-tighter, per-token FiLM).**

**Mechanism findings from this batch's closures:**
1. **#1909 closure:** FiLM-output-bound axis closed. Tanh saturation 0% throughout — the bound never engaged. Combined with #1760+#1838 capacity closures, the FiLM head is well-tuned at its current size and shape. Next FiLM lever must be **structural**, not capacity- or magnitude-related → #1938 per-token FiLM.
2. **#1856 closure:** Slice-routing downward direction closed. Seed-0 test win didn't survive 2nd seed under apples-to-apples conditions; seed 1 showed routing collapse in block 1 (entropy 0.57, eff slice count 1.77). slice_num=64 is at/near optimum.
3. **#1907 send-back:** Coord-scale diagnosis (range [-9.55, +10.55] not [-1, 1]) means original σ=0.01 was ~10x mis-scaled. Future input-augmentation hypotheses must compute σ relative to actual feature std, not assume normalized inputs.
4. **#1757 send-back:** Strong absolute numbers (val 72.11) on stale baseline (`--max_norm 1.0`) means we can't merge as-is without undoing #1831. Rebase + rerun directly answers whether β=0.3 composes with max_norm=0.5 or is partially redundant.

## Potential next research directions (wave 8+)

Ranked by expected ROI on `val_avg/mae_surf_p` given the new max_norm=0.5+FiLM baseline 73.81:

1. **Geometry-aware lever stacked with current baseline — wave-7 #1873 SDF + #1907 position-jitter (rerun) actively testing.** If either lands, opens follow-up family:
   - Learned SDF embedding (replace log1p+standardize with small MLP SDF→4-dim)
   - Surface arc-length encoding (parametric position along airfoil contour)
   - NACA-param FiLM conditioning (sample-level geometric encoding)
   - Mesh subsampling Path A (variable-N gather; clean test after Path B closure)
2. **Structural FiLM follow-ups if #1938 per-token lands:**
   - Geometry-conditioned FiLM (split globals into flow vs geometry, per-domain heads)
   - Per-token FiLM with shared base (`γ = γ_base + γ_token_specific`, additive structure)
   - 3-way per-token FiLM (near-surface as distinct category)
3. **Learned loss-weighting (#1906 Kendall) follow-ups if lands:**
   - GradNorm (Chen et al. 2018) — alternative gradient-balancing scheme
   - Per-domain Kendall (2-channel surface vs volume σ) — simpler version
4. **Routing-temp (#1908) follow-ups if lands:**
   - Apply learned-temp to cross-token attention softmax too
   - Per-block fixed temperature sweep
5. **max-norm further-tighten (#1937) follow-ups if lands:**
   - Sweep further (0.05, 0.025) if 0.1 wins
   - Adaptive clipping (per-epoch grad-norm-percentile threshold)
6. **Compound stack tests if multiple wave-7/8 PRs land:** Kendall + per-token FiLM, position-jitter + SDF, max-norm-tight + per-token FiLM, etc.
7. **Mechanism-port retest:** β=0.3 (#1757 rerun pending) and asinh (#1734) on new 73.81 baseline.
8. **More epochs configuration:** val curve still descending at epoch 13 (30-min cap hit). 20-epoch runs would test the upper bound but exceed wall-clock — needs optimization first.
9. **Hard-example mining / focal-loss sample weighting** — sample-level concentration on high-error foils.
10. **Domain-adversarial training** — direct attack on camber OOD.

## Open questions to revisit on next review

- **#1873 SDF:** First geometry-aware lever — does it crack val_geom_camber_rc=90.32?
- **#1906 Kendall:** Does learned per-channel σ outperform fixed surf_weight=10? Will tell us whether the per-channel weighting axis was lever-limited or fundamentally limited.
- **#1907 Position-jitter (σ=0.05 rerun):** Does non-conditioning input augmentation at the corrected scale succeed where conditioning-feature augmentation failed? Key test of the diagnosis from #1787.
- **#1908 Routing-temp:** Does explicit temperature parameterization help, or is it redundant with the projection-layer scale? Diagnostic-rich either way.
- **#1937 max-norm-tight sweep {0.25, 0.1}:** Is the monotonic tighten-helps curve still descending past 0.5, or did 0.5 land at/near the optimum?
- **#1938 per-token FiLM:** Do surface and volume tokens actually benefit from distinct modulation (cos(γ_surf, γ_vol) < 0.5)? Or is the FiLM head's bottleneck not structural after all (cos ≈ 1.0)?
- **#1757 β=0.3 rerun:** Does β=0.3 compose with max_norm=0.5, or are the two stability levers partially redundant?
- **#1734 asinh:** still WIP on old baseline; rebase + re-evaluate against 73.81 when results arrive.
- **Wall-clock tightness:** new baseline runs at the 30-min cap; future PRs must account for the tightened envelope.
- **Architectural-capacity-axis saturation:** with slice-routing both directions closed (#1818, #1856) and FiLM intra-head both axes closed (#1760, #1838, #1909), the remaining intra-architecture experiments are routing-temp (#1908) and per-token FiLM (#1938 structural). If both close, progress must come from data-side (#1873 SDF, #1907 pos-jitter) or schedule-side or loss-formulation (#1906 Kendall, #1734 asinh) levers.
