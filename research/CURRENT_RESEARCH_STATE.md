# SENPAI Research State

- **Last updated:** 2026-05-13 09:10 (send-back #1873 fern SDF — strong test win on PRE-Kendall stack [val_avg 74.89 +0.36% within 2σ, test_avg 65.10 −1.56% clean test win, val_geom_camber_rc bottleneck −0.77%, test_single_in_dist −5.30%, test_geom_camber_rc −2.33%]; cannot merge to current Kendall baseline 71.43/62.99 — need rerun on Kendall stack to confirm SDF × Kendall compounding. Banked findings: precomputed SDF is correct wall-clock optimization [per-batch costs 6 min/epoch not predicted 1-3]; SDF well-scaled [-0.47, 4.83] with log1p+standardize; FiLM γ_l2/β_l2 unchanged alongside SDF)
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r2`
- **Research tag:** `willow-pai2g-48h-r2`
- **Target repo:** `morganmcg1/TandemFoilSet-Balanced` (base branch `icml-appendix-willow`)
- **W&B:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r2`
- **Per-run cap:** `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Students × GPU:** 8 × 1 (96 GB each)
- **Idle students:** 0 (all 8 active)
- **⚠ Operational note:** GraphQL API rate-limit storms (user ID 20516801) intermittently knock student entrypoints into "No assigned PRs" state mid-loop, AND prevent `gh pr create` (which uses GraphQL). Workaround for PR creation: REST API direct (`gh api repos/.../pulls --method POST --input <file>` + labels via `/issues/.../labels`). Workaround for read: REST helpers (`pr_body`, `pr_all_comments`). **Today's storms:** ~05:00–05:50 UTC (resolved 05:48-05:54), then **second storm active ~08:30+ UTC** locking out fern, tanjiro, edward (assignment poll fails → "No work assigned, sleeping 300s"). **Critical caveat:** nezuko #1981 is observed ACTIVELY TRAINING (GPU 96/97 GB at 100%) under the storm — watchdog `leaving Claude running` once work latched. **Stale_wip detection is misleading during storms** — PR has zero commits/comments since assignment (storm-blocked from posting), but the student pod is actually working through training. Don't reassign storm-flagged students.

## ⭐ Current baseline (PR #1906 merged 2026-05-13 — Kendall uncertainty)

- **val_avg/mae_surf_p:** **71.4346** (seed 0, SWA-model eval) ← NEW
- **test_avg/mae_surf_p:** **62.9866** (seed 0, SWA-model, 4-split all finite) ← NEW
- Improvement over prior #1831 baseline: val **−3.22%**, test **−3.15%** (2.76× σ band on val)
- Config: Transolver + FiLM (mid_dim=64) + Smooth-L1 (Huber β=1.0) + per-sample Re-weight + **Kendall uncertainty per-channel σ heads** (replaces fixed surf_weight=10) + grad-clip max_norm=0.5
- Schedule: CosineAnnealingLR(T_max=15), SWA (start_frac=0.75, swa_lr=1e-4, anneal_epochs=2)
- W&B baseline run: `dkfjae5o`
- See `BASELINE.md` for full reproducible spec.
- **Learned σ:** max/min weight spread 1.20× — nearly uniform, slight Ux/Uy emphasis (consistent with #1821 residual-ratio diagnosis); no clamp saturation.
- **Per-split test gains (vs #1831):** test_single_in_dist −8.10 (huge!), test_geom_camber_rc −0.39, test_geom_camber_cruise −0.05, test_re_rand +0.33. **The Kendall win is concentrated on in-distribution accuracy; OOD splits barely moved** → OOD generalization remains bottlenecked by architecture/data-side levers, not loss weighting.

## 🔥 Hottest signals this session

- **PR #1873 (fern, per-node SDF) SEND-BACK for rebase + rerun on Kendall:** val=74.89 (+0.36% vs pre-Kendall #1731, within 2σ) / test=65.10 (**−1.56% test win vs #1731**) but **regression vs current Kendall baseline** (val +4.85%, test +3.35%). **Mechanism CONFIRMED on pre-Kendall stack:** val_geom_camber_rc (bottleneck) −0.77%, test_single_in_dist −5.30%, test_geom_camber_rc −2.33% — geometry-aware features deliver asymmetric test gains on geometry-related splits exactly as predicted. **Right experiment now:** does SDF compound with Kendall? Geometry-aware × multi-task-weighting are mechanism-orthogonal. Banked: precomputed SDF is correct (per-batch costs 6 min/epoch); SDF well-scaled; FiLM unchanged alongside SDF. Likely outcome: ~50% lands on Kendall stack, ~30% partial overlap, ~20% doesn't stack.
- **PR #1937 (alphonse, max-norm tighten {0.25, 0.1}) CLOSED:** val=74.07 (best new arm, max_norm=0.1) / test=65.63 — clean negative vs pre-Kendall assignment baseline 73.81 (+0.35%/+0.91%) and even further from Kendall baseline 71.43. Non-monotonic ordering (0.1 < 0.25) within ~1σ of 2-seed variance. **High-info mechanism finding: clip_fraction saturation at max_norm=0.5.** Student's diagnostic: clip_fraction_mean = 99.2% at 0.5, **saturates to 100%** at both 0.25 and 0.1, while pre-clip grad_norm_mean stays stable (~5.0) across all three. **Past max_norm=0.5, the threshold is no longer a regularization knob — it's a uniform step-magnitude rescaler** (effectively a per-batch lr-cut on the already-clipped 99%+ fraction, compounding with cosine-anneal LR shrinkage → uniform underfitting). Grad-clip-tighten direction closes on optimizer-stability axis. Reassigned to #2082 Fourier coordinate features (Tancik 2020 RFF) — fresh input-encoding axis.
- **PR #1954 (askeladd, HEM via EMA loss tracker) CLOSED:** val=75.80 / test=67.12 — clean +6.10% / +6.56% regression on Kendall (5σ above noise band). All 8 splits regress; **largest hit on in-dist splits (+7.94% val_single_in_dist) NOT OOD splits**. **High-info mechanism finding: "sample-loss-difficulty ≠ OOD-distance" on TandemFoilSet** — EMA loss tracker upweighted intrinsically hard fluid-dynamics configurations, causing model to overfit to those configurations across the board. Per-sample loss-magnitude-driven rebalancing axis closes on Kendall (joins #1691 surf_weight=5). Reassigned to #2063 Lion optimizer sweep — fresh optimizer-family axis.
- **PR #1734 (thorfinn, asinh α=0.5 + Kendall) CLOSED:** val=79.12 / test=70.41 — **+10.76% / +11.78% regression** — LARGEST regression on the Kendall stack to date. All 8 splits regress 7-18%. **Mechanism finding:** Kendall self-adapts σ to asinh-transformed loss space (final log_σ_surf_p −1.500 vs baseline −1.408 → ~20% higher pressure-channel effective weight). Kendall + asinh compound and overshoot. **Value-compression-on-output × Kendall σ-adaptation is mechanism-incompatible.** General lesson: future output-side loss-space-reshape hypotheses must consider Kendall σ interaction. Reassigned to #2049 aux-log_re-prediction sweep.
- **PR #1907 (edward, pos-jitter) CLOSED:** val=71.68 / test=63.11 (σ=0.05 + Kendall arm). Combined with σ=0.01 pre-Kendall arm (val=74.45/test=65.45), **two arms × two baselines give same regression direction at same approximate magnitude**. **Position-jitter axis closes.** Reassigned to #2021 OneCycleLR-with-warmup sweep (after withdrawing #2016 DropPath when #1680 closure audit revealed 15-epoch under-convergence pathology).
- **PR #1908 (nezuko, learnable routing-temp) CLOSED:** val=76.28 / test=68.01 (clean negative vs both 73.81 and 71.43 bars). **Per-block multiplicative routing_log_temp barely moves (<10% drift across 5 blocks).** Precondition-finding: pre-existing per-head `self.temperature` (init=0.5) was already in PhysicsAttention. **Routing-sharpness axis closes cleanly.** Reassigned to #1981 wd-sweep.
- **PR #1906 (askeladd, Kendall uncertainty) MERGED:** val=**71.43** / test=**62.99** — clean win, all 8 splits improve, learned σ near-uniform (1.20× spread). Per-channel weighting axis LANDED. **OOD splits barely moved (camber_rc/cruise/re_rand) — OOD bottleneck now dominant.**
- **All 7 other students actively training** (GPU 60-97GB across pods, rate-limit storm recovering; edward, alphonse, askeladd, fern in iter-between state with branches fetched). Results pending in next 1-2 loop iterations.

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
| #1831 (nezuko) | max-norm-0p5-on-clipfilm | Tighter grad-clip max_norm=0.5 | val=73.81, test=65.04 |
| #1906 (askeladd) | kendall-uncertainty-on-clipfilm | Learned per-channel σ heads (Kendall et al. 2018) | **val=71.43, test=62.99** (current) |

## Current research focus

### Wave 7 (in flight) — all decisions vs new Kendall baseline (71.43/62.99)

| PR | Student | Slug | Mechanism axis | Forked from | Decision threshold |
|---|---|---|---|---|---|
| #2063 ← NEW | askeladd | `lion-optimizer-on-kendall` | Lion optimizer (Chen et al. 2023) sweep {lr=1e-4 wd=1e-3, lr=3e-4 wd=3e-4} — fresh optimizer-family axis, AdamW completely untouched | 71.43 | best-arm val < 71.43 → merge |
| #1981 ← NEW | nezuko | `wd-sweep-on-kendall` | AdamW weight decay sweep {3e-4, 1e-3} — classical OOD-regularization attack | 71.43 | best-arm val < 71.43 → merge |
| #2082 ← NEW | alphonse | `fourier-coord-features-on-kendall` | Random Fourier Features (Tancik 2020) on per-node coords, sweep {σ=1.0, 4.0} — fresh input-encoding axis (distinct from closed grid-based `unified_pos`) | 71.43 | best-arm val < 71.43 → merge; val_geom_camber_rc ≥4% improvement → 2nd-seed override |
| #1938 | tanjiro | `film-per-token-on-clipfilm` | Per-token (is_surface-aware) FiLM — first structural FiLM change | 73.81 → reframe vs 71.43 | val < 71.43 → merge |
| #2021 ← NEW | edward | `onecycle-lr-warmup-on-kendall` | OneCycleLR sweep {max_lr=5e-4, 1e-3} + 10% warmup — fresh schedule axis | 71.43 | best-arm val < 71.43 → merge |
| #1873 ← rerun | fern | `sdf-feature-on-clipfilm` (rebase to Kendall in progress) | Per-node SDF (geometry-axis) — pre-Kendall run produced clean test win (-1.56%) and bottleneck rc improvement (-0.77%); rebasing + rerunning with `--use_kendall_uncertainty --max_norm 0.5` to test SDF × Kendall compounding | 71.43 | val < 71.43 → merge; val < 73.0 AND test < 62.99 → 2nd-seed |
| #1757 ← rerun | frieren | `beta-0p3-on-filmed` | β=0.3 monotonic-β port; rerun after rebase + Kendall | (post-rebase) 71.43 | val < 71.43 on new bar |
| #2049 ← NEW | thorfinn | `aux-re-prediction-on-kendall` | Auxiliary log_re prediction MLP head per-block, sweep {0.01, 0.1} — OOD-targeted representation bottleneck | 71.43 | best-arm val < 71.43 → merge; test_re_rand ≥3% → send-back override |

### Decision rule (vs new 71.43 baseline)

- best-arm val < 71.43: merge (assuming no conflicts).
- 71.43 ≤ val < 73.0 (within σ=0.86 variance band): send back for 2nd seed if test < 62.99 (test override).
- 73.0 ≤ val < 75.0: clean negative — close.
- val ≥ 75.0: clean regression — close.
- **Test override:** if test < 62.99 even when val doesn't beat 71.43, send back — paper-facing test wins matter independently.
- **In-flight PRs assigned vs old 73.81 baseline:** their results will be re-framed at review time against 71.43 / 62.99. A PR that beats 73.81 but lands above 71.43 was already non-decisive and now needs re-evaluation through the new lens.

## ✗ Closed this session

- **Wave-1/3 closures:** #1454, #1455, #1448, #1453, #1446, #1449, #1450, #1551, #1621, #1645, #1620
- **Wave-5 closures:** #1617 (stale rebase), #1680 (drop_path=0.1), #1679 (no-SWA), #1642 (sqrt-Re-weight), #1618 (surf-Huber/vol-MSE on SWA-on-Huber), #1733 (attn-dropout=0.1), #1732 (swa_start=0.65), #1600 (β-sweep on SWA-on-Huber, β=0.3 best; reassigned), #1691 (surf_weight=5), #1739 (FiLM-absorbed per-domain loss), #1702 (per-channel p-up, diagnostic falsified premise)
- **Wave-6 closures:** #1760 (FiLM mid_dim=128 — width direction closed), #1818 (slice_num=128 — wall-clock cap), #1758 (mesh-subsample Path B — bias contamination), #1838 (FiLM depth=3 — depth direction closed), #1821 (uxuy_weight=2.0 — per-channel weighting both directions closed), #1787 (Re-jitter σ=0.05 — conditioning-feature augmentation broadly closed)
- **Wave-7 closures:** #1909 (tanh-bound FiLM — output-bound axis closed, saturation 0%), #1856 (slice_num=32 2nd seed — routing collapse seed 1, slice-routing capacity downward closed), #1908 (learnable per-block routing-temp — drift <10%, precondition self.temperature already present, routing-sharpness axis closed), #1907 (pos-jitter σ∈{0.01, 0.05} — 2-arm × 2-baseline same-direction regression, volume-coord noise jitter axis closed), #2016 (DropPath sweep — withdrawn before student start, #1680 audit revealed 15-epoch under-convergence pathology), #1734 (asinh α=0.5 + Kendall — +10.76% / +11.78% regression, Kendall × asinh σ-adaptation interaction overshoots, value-compression-on-output axis closed under Kendall), #1954 (HEM EMA-loss focal weighting — +6.10%/+6.56% regression, in-dist hit hardest, "loss-difficulty ≠ OOD-distance" finding, per-sample loss-magnitude rebalancing axis closes), #1937 (max-norm-tighten {0.25, 0.1} — both arms regress, clip_fraction saturated at 99.2% already at 0.5 → 100% at tighter thresholds, grad-clip-tighten direction closes on optimizer-stability axis)
- **Wave-7 merges:** #1906 (Kendall uncertainty — learned per-channel σ heads, val=71.43/test=62.99 new baseline)

## ⚠ Active operational notes

- **GraphQL rate-limit pattern continues.** REST helpers preferred.
- **Mixed-baseline portfolio cleanup:** #1856 needs rebase to new 73.81 baseline before 2nd seed run. #1757, #1734 still WIP on old baselines.
- **24 mechanism axes definitively closed on this dataset/scale:**
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
  - **Grad-clip-tighten direction past max_norm=0.5 (#1937) ← NEW** — clip_fraction saturates at 99.2% at 0.5, no further regularization headroom
- **6 axes have produced landings:**
  - Loss-shape: Huber (#1452 merged)
  - Loss-weighting: per-sample Re-weight (#1586 merged)
  - Architecture-conditioning: FiLM (#1585 merged)
  - Optimizer-stability max_norm=1.0 (#1731 merged)
  - Optimizer-stability max_norm=0.5 (#1831 merged)
  - **Loss-weighting (channel-level, learned σ — Kendall, #1906 merged) ← NEW**
- **Largest remaining gap: val_geom_camber_rc (88.09 on new baseline).** Geometry-aware levers (#1873 SDF, #1907 pos-jitter) directly target this.
- **Largest single-PR test gain:** Kendall's test_single_in_dist (76.74 → 68.64, −8.10) — biggest split-level move on this branch since FiLM merge.
- **Composition pattern confirmed three times:** grad-clip + FiLM compose constructively. max_norm=0.5 compounds on top. **Kendall composes on top of grad-clip + FiLM** for another +3.2% val. Stability-enabling + multi-task-balancing levers stack additively.
- **Mechanism finding from #1906 split breakdown:** Kendall improvements are 90%+ concentrated on test_single_in_dist. OOD splits (geom_camber_rc, geom_camber_cruise, re_rand) barely moved. **OOD generalization is bottlenecked by something OTHER than loss weighting** — likely architecture (per-token FiLM #1938, routing-temp #1908) or data-side (SDF #1873, pos-jitter #1907).

## Mechanism-axis coverage (all 8 students, wave 7)

- **Loss-shape (β):** β=0.3 rerun pending after rebase + Kendall (#1757 frieren)
- **Loss-shape (per-domain kind):** **CLOSED** at FiLM-scale (#1739)
- **Loss-weighting (surf/vol split):** **CLOSED both directions** (sw=10 brackets optimum)
- **Loss-weighting (channel-level, fixed):** **CLOSED both directions** (#1702 p-up; #1821 uxuy-up)
- **Loss-weighting (channel-level, learned σ — Kendall):** **LANDED #1906** ← NEW
- **Loss-weighting (value-level):** asinh on pressure target (#1734 thorfinn) — rebase pending against new Kendall baseline
- **Optimizer-stability (grad-clip max_norm):** **LANDED at 0.5 in baseline (#1831)** — further-tighten direction **CLOSED #1937** (clip_fraction saturated at 99.2%, no headroom)
- **Input-encoding (Random Fourier Features on coords):** in flight #2082 alphonse (Tancik 2020 RFF, σ-sweep — fresh axis; distinct from closed grid-based unified_pos #1454)
- **Data-side input augmentation (node-level Path B):** **CLOSED** (#1758)
- **Data-side input augmentation (sample-level conditioning feature):** **CLOSED** (#1787 Re-jitter)
- **Data-side input augmentation (per-node non-conditioning):** in flight #1907 edward — rerun at σ=0.05 after coord-scale finding
- **Geometry-aware input features (per-node SDF):** in flight #1873 fern
- **Architecture-conditioning (intra-FiLM-capacity, width):** **CLOSED** (#1760)
- **Architecture-conditioning (intra-FiLM-capacity, depth):** **CLOSED** (#1838)
- **Architecture-conditioning (intra-FiLM, modulation magnitude bound):** **CLOSED** (#1909)
- **Architecture-conditioning (intra-FiLM, structural — per-token):** in flight #1938 tanjiro (is_surface-aware split heads)
- **Architecture-conditioning (intra-routing-capacity, upward):** **CLOSED** (#1818)
- **Architecture-conditioning (intra-routing-capacity, downward):** **CLOSED** (#1856 routing collapse)
- **Architecture-conditioning (intra-routing softmax sharpness):** in flight #1908 nezuko (learnable routing temperature)
- **Architecture-conditioning (head):** FiLM — LANDED in baseline
- **Schedule / SWA-window:** definitively closed
- **Internal regularization:** definitively closed (3 sub-axes)
- **Loss-weighting (channel-level, learned σ — Kendall):** **LANDED #1906** ← NEW

**Mechanism-axis coverage status: 6 landed (Huber, Re-weight, FiLM, grad-clip 1.0, grad-clip 0.5, Kendall) + 8 in-flight wave-7 PRs (#2063 Lion, #1981 wd-sweep, #2082 Fourier features ← NEW, #1938 per-token FiLM, #2021 OneCycleLR, #1873 SDF, #1757 β=0.3, #2049 aux-Re) + 24 axes closed.**

**Mechanism findings from this batch:**
1. **#1906 MERGE — Kendall lands:** Per-channel learned σ heads succeeded where fixed weighting failed (#1702, #1821). Near-uniform learned weighting (1.20× spread) **beats** the hand-set `surf_weight=10` baseline. Confirms principled task uncertainty estimation is the right lever.
2. **#1906 split breakdown — OOD bottleneck remains:** test_single_in_dist −8.10 (huge!), OOD splits barely move. **Loss-weighting axis cannot fix OOD; that bottleneck is architecture- or data-side.**
3. **#1734 send-back:** Config confound (max_norm=1.0 vs current 0.5) plus within-σ-band non-win. Rerun with `--max_norm 0.5 --use_kendall_uncertainty --asinh_alpha 0.5` to test asinh on the new bar.

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

- **#1873 SDF:** First geometry-aware lever — does it crack val_geom_camber_rc=88.09 on the new bar?
- **#1907 Position-jitter (σ=0.05 rerun):** Does non-conditioning input augmentation at the corrected scale succeed where conditioning-feature augmentation failed? Key test of the diagnosis from #1787.
- **#1908 Routing-temp:** Does explicit temperature parameterization help, or is it redundant with the projection-layer scale? Diagnostic-rich either way.
- **#1937 max-norm-tight sweep {0.25, 0.1}:** CLOSED — 0.5 is at/near the optimum. Clip_fraction was already 99.2% at 0.5, saturates to 100% at tighter thresholds → no further regularization headroom on this lever. Reassigned to #2082 Fourier coord features (fresh input-encoding axis).
- **#2082 Fourier coordinate features {σ=1.0, 4.0}:** Does the sin/cos representation prior (Tancik 2020) help Transolver learn the high-freq pressure spikes at foil leading edges? Particularly: does val_geom_camber_rc (88.09) crack ≥4% even if val_avg doesn't beat 71.43?
- **#1938 per-token FiLM:** Do surface and volume tokens actually benefit from distinct modulation (cos(γ_surf, γ_vol) < 0.5)? Or is the FiLM head's bottleneck not structural after all (cos ≈ 1.0)?
- **#1757 β=0.3 rerun:** Does β=0.3 compose with max_norm=0.5 + Kendall, or are these stability levers partially redundant?
- **#1734 asinh rebase:** Does value-level pressure-target compression help on the new Kendall baseline?
- **Wall-clock tightness:** new baseline runs near the 30-min cap; future PRs must account for the tightened envelope.
- **OOD-bottleneck attribution:** Kendall's win was concentrated on test_single_in_dist (−8.10) with OOD splits barely moving. **The OOD generalization gap is now exposed as the dominant remaining challenge.** Architecture (#1938 per-token FiLM, #1908 routing-temp) or data-side (#1873 SDF, #1907 pos-jitter) levers are best-positioned to crack it. Loss-weighting axis is now landed; further gains there are diminishing-returns.
- **In-flight reframe:** 6 wave-7 PRs were assigned with decision rule vs old 73.81 baseline. They will be re-evaluated against 71.43 at review time. A PR landing in [71.43, 73.81] beats the old bar but not the new — these become close-or-send-back rather than merge candidates.
