# SENPAI Research State — charlie-pai2g-48h-r5

- **As of:** 2026-05-13 10:05 UTC (round-32: Closed #2114 askeladd gap/stagger jitter σ=0.02 (val 51.37, +1.52% LOSS) — **NEW failure mode: broadcast-scalar prior corruption** (uniform regression all 4 splits, val_re_rand worst hit at +4.07%); NOT 8th bimodal, distinct mechanism. Closed #2093 alphonse AdamW β1=0.95 (val 54.37, +7.45% LOSS) — **NEW failure mode: momentum-lag overshoots cosine LR** (uniform regression 4-14% across all splits, val_geom_camber_cruise worst). AdamW (lr, β1, β2) hyperparameter axis now closed 4-LOSS-for-4. Assigned #2155 alphonse AdamW amsgrad=True (max-tracking 2nd-moment for L1 sign-noise — novel optimizer-stability axis) + #2156 askeladd GELU→SiLU activation swap (architectural probe, NOT averaging-style). Baseline still **50.6001**)
- **Branch:** `icml-appendix-charlie-pai2g-48h-r5` (advisor) — Charlie no-W&B logging ablation, round 5
- **Most recent human-team direction:** None on this branch.

## Current research focus

**8 merged winners → baseline 50.6001 (-54% from 110.76 at round-1 start).**

New baseline = L1 + compile + bf16 + sampler 2× single + slice_num=32 + warmup-3-cosine.

**Round-18 findings:**
- **slice_num-DOWN axis closed**: 16 is a val wash (+0.41%) with an in-dist/OOD split trade-off. 32 is the global optimum.
- **Sampler axis closed**: 2.0× single is the confirmed peak. 1.5× (+3.47%) is strictly worse. No further sampler tuning needed.
- **New bottleneck identified**: `val_geom_camber_rc` (72.37) and `val_re_rand` (58.23) now dominate val_avg. Neither responds to sampler reweighting. **OOD generalization** is the highest-leverage remaining research axis.

**Shifted priority to OOD generalization levers:**
1. **Input augmentation** (#1921 nezuko pos-jitter) — targeting mesh-coord overfitting as source of OOD brittleness
2. **Normalization variant** (#1926 frieren rmsnorm) — RMSNorm as structural complement to routing changes
3. **Optimizer/regularization stack** (#1653 askeladd grad-clip, #1775 fern WD=5e-5, #1845 edward betas, #1774 alphonse lr — all on L1 or Huber baselines, need rebase to confirm still useful on 54.00 baseline)
4. **Schedule** (#1905 thorfinn SGDR warm restarts)
5. **Architecture (n_head=8)** — tanjiro pod stale × 3; n_head axis deferred until pod recovers or another student is available. DropPath (#1976) is the current architecture-regularization experiment.

## Merged winners

| PR | Student | Hypothesis | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|---|---|
| #2033 ✓ | thorfinn | LinearWarmup-3 + CosineAnnealingLR(T_max=47) | **50.6001** | **43.9680** |
| #1846 ✓ | frieren | slice_num 64 → 32 | 54.0051 | 47.6261 |
| #1619 ✓ | nezuko | Sampler 2× racecar_single | 56.62 | 50.43 |
| #1700 ✓ | thorfinn | Pure L1 loss | 59.54 | 51.47 |
| #1633 ✓ | thorfinn | Huber β=0.5 | 64.07 | 55.50 |
| #1568 ✓ | thorfinn | torch.compile + bf16 | 69.83 | 61.87 |
| #1532 ✓ | thorfinn | bf16 AMP + scoring-NaN fix | 101.12 | 91.50 |
| #1444 ✓ | thorfinn | MSE → Huber β=1.0 | 110.76 | NaN |

**Current baseline: val_avg/mae_surf_p = 50.6001, test_avg/mae_surf_p = 43.9680 (PR #2033)**

Per-split baseline (PR #2033):

| Split | val mae_surf_p |
|---|---|
| `val_single_in_dist` | 47.9418 |
| `val_geom_camber_rc` | 67.3675 |
| `val_geom_camber_cruise` | 34.3430 |
| `val_re_rand` | 52.7481 |

> Advisor config: Pure L1 + bf16 AMP + torch.compile(dynamic=True) + scoring-NaN workaround + sampler 2× racecar_single + slice_num=32 + LinearWarmup(3ep, 0.1→1.0) + CosineAnnealingLR(T_max=47).
> ~44 epochs in 30 min (~41 s/epoch). Peak GPU: ~21.35 GB.
> **Best epoch = terminal in all recent runs (model still improving at timeout).**

## In-flight (WIP)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #2156 | askeladd | GELU→SiLU activation swap (preprocess MLP + TransolverBlock MLP + last_layer mlp2) | **Round-32** — architectural probe; NOT averaging-style, NOT broadcast-scalar; consistent 0.3-1% literature gains; expected uniform per-split delta |
| #2155 | alphonse | AdamW amsgrad=True (max-tracking 2nd-moment) | **Round-32** — novel optimizer-stability lever; tracks max of 2nd-moment EMA → theoretically robust to L1 sign-flip noisy gradients; distinct from (lr, β1, β2) hyperparameter axis |
| #2139 | frieren | RMSNorm replacing LayerNorm (all 3 sites) — **retry-3** of stale #1926/#2034 | **Round-31** — fresh PR after 2nd consecutive stale; same hypothesis, pod-unstick attempt |
| #2138 | nezuko | Translation augmentation σ=0.01 (channels 0-1, per-sample shift) | **Round-31** — physical-symmetry OOD lever; structurally distinct from all 7 closed averaging-style mechanisms |
| #2112 | thorfinn | Warmup-2-cosine (warmup_epochs 3→2, T_max 47→48) | **Round-30** — probes OTHER direction of bimodal trade-off; predicted to favor OOD by extending settling-phase |
| #2083 | tanjiro | DropPath p_max=0.1 stochastic depth (retry of stale #1976) | **Round-28** — block-level stochastic depth, structurally distinct from closed averaging-style class |
| #2072 | edward | NACA geometry jitter σ=0.01 (channels 15-17 NACA1, 19-21 NACA2) | **Round-27** — direct attack on val_geom_camber_rc OOD bottleneck via camber perturbation |
| #1775 | fern | WD=5e-5 | Proven -4.43% on β=0.5; ⚠️ new baseline 50.6001 |

## Warning on in-flight rebase

Long-running in-flight PRs (#1775, #1988) were assigned on the old 54.0051 baseline and do NOT include the warmup-3-cosine schedule (merged PR #2033). The current baseline is **50.6001**. These PRs need to beat 50.6001 to merge — a significantly harder bar. If any returns with val > 50.60 but close, consider requesting a rebase onto the current advisor (with warmup) before closing.

## Closed axes (comprehensive)

- **Capacity:** width (n_hidden=160), depth (n_layers=6), FFN-only (mlp_ratio=3) — all close, budget-bound.
- **Attention dropout (per-weight):** p=0.1 — convergence cost > benefit at 30-min cap.
- **Batch=8:** step-count starvation.
- **WD up (5e-4):** under-fits on short budget.
- **Warmup:** substituted by β sharpening.
- **LR=1e-3 + warmup:** overshoot.
- **Surf_weight 10→30:** vol-surf imbalance.
- **Per-channel loss [1,1,3] global:** distorts velocity (PR #1428).
- **Per-channel surf_loss [1,1,2] surf-only:** OOD regression (PR #1871) — same physics coupling.
- **LR floor (eta_min=5e-5):** removes step-damping for L1 sign gradients (PR #1826).
- **Sampler both-racecar 2×:** absolute single exposure drops (PR #1870).
- **Sampler racecar_single 1.5×:** under-concentrates single coverage (PR #1904). **2× is the confirmed optimum.**
- **slice_num=96:** worse than 64 (PR #1590).
- **slice_num=16:** val wash (+0.41%) with in-dist/OOD trade-off; 32 is the global optimum (PR #1903). **slice_num axis fully closed.**
- **AdamW β2=0.95:** LOSS on L1 (+4.42% on L1 base, +15% vs current). Shorter second-moment memory amplifies L1 sign-flip noise (PR #1845). β2 axis closed.
- **Grad-clip max_norm=1.0:** WASH on L1+sampler+slice=32 (best run −0.37%, mean +0.56%; n=2 straddles baseline). Monotone dose-response: β=1.0 −14.92% → β=0.5 −6.94% → L1+slice=32 ~0%. Bimodal per-split: in-dist −13.33% but OOD +3-5% — structurally wrong for primary bottleneck. Gradient-coherence axis saturated by upstream L1+sampler changes (PR #1653). Gradient-coherence axis closed.
- **EMA model weights (decay=0.9999, 0.999):** 3 runs total. Mechanism confirmed (EMA<raw from ep10, dual-eval diagnostic). But per-split always bimodal: in-dist −12.9%, OOD +5-14%. Third confirmation of averaging-style bimodal pattern alongside coord-jitter and grad-clip. EMA-of-weights axis locally closed (PR #1946). **Pattern: ~14% in-dist headroom unlockable by averaging-style regularization; OOD requires structural interventions.**
- **Warmup-3-cosine:** MERGED as PR #2033 (WIN −6.31% val, −7.68% test). Now on advisor as the new schedule. warmup mechanism confirmed: sub-peak LR in first 3 epochs selects better loss basin. L1's two-phase property validated: warmup (find) + cosine (fine-tune). Best gain on val_single_in_dist (−18.87%); geom_camber_rc barely moved (−0.11%).
- **Warmup-5-cosine:** WASH +0.23% on warmup-3 (PR #2071). Bimodal: val_single_in_dist −2.60% (in-dist WIN), val_re_rand +2.81% (OOD LOSS), val_geom_camber_rc flat. 2 fewer settling epochs (T_max 47→45) cost OOD more than 2 extra warmup epochs helped in-dist. **6th bimodal confirmation** (settling-phase axis). Warmup-2 (other direction) in flight as #2112 — completes the bracket.
- **Lookahead k=5 α=0.5:** LOSS +5.09% on old baseline (no warmup); +12% vs current warmup-3-cosine (PR #2051). Per-split: in-dist −9.48% (WIN), all 3 OOD splits +5.92% to +18.0% (LOSS). **5th averaging-style bimodal confirmation** — class definitively closed.
- **AdamW β2=0.95 (earlier β=0.5 test):** near-wash on β=0.5, no clear signal (PR #1676).
- **lr axis (capacity↔LR coupling, ±25% of 5e-4):** fully bracketed and closed.
  - **lr=7.5e-4 (+50% lr-UP):** LOSS on L1+slice=32 (+16% vs current, n=3 mean 62.66) — closed across all 3 landscape variants tested (β=0.5 wash, L1+slice=64 wash-with-loss-tail, L1+slice=32 LOSS) (PR #1774).
  - **lr=3.75e-4 (-25% lr-DOWN):** LOSS on warmup-3-cosine + L1+slice=32 (val 56.36, +11.4% vs 50.6001, n=1) — 4th averaging-style bimodal confirmation: in-dist −6.45 (win) but all 3 OOD splits +4.84 to +5.74 (regression). Best epoch terminal at 44/50 → under-stepping confirmed (PR #1997).
  - **Pattern:** lr-DOWN is 4th confirmation of averaging-style bimodal (alongside coord-jitter, weight-EMA, grad-clip). Reduced effective step size acts as implicit averaging — same in-dist↔OOD trade-off. lr axis closed.
- **SGDR T_0=10 T_mult=2:** LOSS on L1+slice=32 (+27.7% val, n=1, 68.96 vs 54.00). Mechanism worked at cycle level (restart #2 min < restart #1 min) but L1 sign-gradient regime is fundamentally hostile to LR restarts (signs reset, late settling destroyed). Budget-fit variants deferred — L1+restart is the binding incompatibility (PR #1989). Schedule axis continues with warmup+monotone-cosine in flight as #2033.
- **AdamW β1=0.95 (UP probe):** LOSS +7.45% val (54.37) on warmup-3-cosine baseline (PR #2093, round-32 close). All 4 splits regress uniformly (4.10% to 14.27%, val_geom_camber_cruise worst at +14.27%). **NEW failure mode: momentum-lag overshoots cosine LR** — too much historical-gradient orientation can't track shrinking LR. NOT 8th averaging-style bimodal (in-dist would have won) — distinct mechanism. AdamW (lr, β1, β2) hyperparameter axis now closed 4-LOSS-for-4 (lr-DOWN, lr-UP, β2, β1). Defaults (lr=5e-4, β1=0.9, β2=0.999) are locally optimal.
- **Gap/stagger jitter σ=0.02 (channels 22-23):** LOSS +1.52% val (51.37) on warmup-3-cosine baseline (PR #2114, round-32 close). All 4 splits regress, val_re_rand worst hit at +4.07% (52.75 → 54.90). **NEW failure mode: broadcast-scalar prior corruption** — channels 22-23 are per-sample CONSTANT scalars (gap, stagger) broadcast to all N points; the model uses them as load-bearing priors for "this sample is configuration X". Perturbing them is NOT data augmentation, it's belief-corruption — the model is forced to learn "tandem geometry is uncertain" which directly increases prediction variance. NOT 8th averaging-style bimodal — distinct mechanism. Implication: per-sample-constant input channels are NOT augmentation-safe; per-point channels (NACA #2072, coords #1921) behave differently.

## Open questions / next experiments

1. **OOD generalization** — primary bottleneck. `val_geom_camber_rc` (67.37) barely moved with warmup (-0.11%); `val_re_rand` (52.75) improved slightly (-1.9%). OOD splits are not schedule-sensitive; they require structural interventions.
   - **Coord-jitter (#1921) CLOSED (LOSS):** -13.7% in-dist, all OOD splits degraded.
   - **EMA (decay=0.999, #1946) CLOSED (WASH-TO-LOSS):** -12.9% in-dist, OOD +5-14%.
   - **Grad-clip (#1653) CLOSED (WASH):** -13.3% in-dist, OOD +3-5%.
   - **lr-DOWN (#1997) CLOSED (LOSS):** -6.45 in-dist, OOD +4.84 to +5.74.
   - **Lookahead k=5 α=0.5 (#2051) CLOSED (LOSS):** -9.48% in-dist, OOD +5.92-18.0%.
   - **Warmup-5 (#2071) CLOSED (WASH+bimodal):** -2.60% in-dist, re_rand +2.81%.
   - **fun-jitter Re/AoA (#1988) CLOSED (LOSS, 2 σ-points):** σ=0.05 +19.5% LOSS, σ=0.025 +14.6% LOSS; OOD damage NEAR-SATURATED in σ (0.82-0.91× attenuation when σ halved, far slower than linear); in-dist Goldilocks zone at small σ. **7th averaging confirmation** at channel-level.
   - **In-dist headroom is unlockable but does NOT translate to OOD (7× confirmed).** Averaging-style regularization (coord-jitter, weight-EMA, grad-clip, lr-DOWN, Lookahead) + settling-phase reduction (warmup-5) + channel-level jitter (fun-jitter) consistently give in-dist win but regress OOD. This is now a rock-solid empirical law across 7 mechanisms (data aug, weight aug, gradient aug, step-size aug, trajectory aug, schedule aug, channel aug). **Mechanism-agnostic finding:** any reduction in effective late-stage exploration trades in-dist for OOD. **The next OOD-targeting experiments must NOT be averaging-style** — needs physical-symmetry data augmentation (translation, reflection), structural input augmentation (NACA #2072, gap/stagger #2114), or architectural change (RMSNorm #2139, multi-task aux loss, conditional embedding).
   - **Active OOD interventions:** NACA geometry jitter (#2072 edward) — directly targets camber distribution shift; Lookahead optimizer (#2051 askeladd) — flat-minima bias hypothesis; fun-jitter Re/AoA σ=0.025 (#1988 nezuko) — function space augmentation.
   - **Untried structural OOD interventions:** domain conditional embedding (architectural), AoA reflection symmetry, multi-task camber prediction auxiliary loss.
2. **Normalization** — RMSNorm in flight (#1926). Pre-LN vs Post-LN placement also untested.
3. **Proven-lever stack on new baseline** — grad-clip (#1653), WD=5e-5 (#1775) all need rebasing to current 54.00 baseline. β2 axis is now closed (PR #1845).
4. **Schedule** — SGDR in flight (#1905). Warm restarts may unlock multi-descent gains.
5. **Lookahead optimizer** — **in-flight #2051 askeladd.** Standard k=5, α=0.5 recipe. Hypothesis: flat-minima bias from slow-weight averaging should improve OOD (inverse of grad-clip's bimodal).
6. **Warmup duration** — **in-flight #2071 thorfinn.** Warmup-3 won massively (-6.31%); warmup-5 probes whether longer basin exploration beats it or shorter settling phase hurts.
7. **NACA geometry jitter** — **in-flight #2072 edward.** σ=0.01 on NACA channels 15-17, 19-21 during training. First geometry-level augmentation targeting camber distribution shift (val_geom_camber_rc).
6. **Stochastic depth (DropPath)** — **in-flight #2083 tanjiro (retry).** Block-level residual regularization; structurally distinct from closed averaging-style class. Untested.
7. **AdamW β1=0.95 (UP probe)** — **in-flight #2093 alphonse.** Single-knob optimizer probe on novel axis. β2 closed (#1845, +15% LOSS); β1 never probed. β1=0.9→0.95 lengthens gradient-momentum half-life from ~10 to ~14 steps, providing smoothing for L1's sign-gradient noise.
8. **Warmup-2-cosine (OOD-favorable bracket)** — **in-flight #2112 thorfinn.** Completes the warmup duration bracket (warmup-3 win → warmup-5 wash → warmup-2 probe). Predicted to swap the signs on the bimodal: in-dist slight regress, OOD slight gain via 1 extra settling epoch. If symmetric trade-off, val_avg ~50.4-50.5.
9. **Gap/stagger jitter σ=0.02 (channels 22-23)** — **in-flight #2114 askeladd.** Tandem-foil arrangement augmentation on the last untouched geometric input channels. Structurally distinct from closed averaging-style class. Complements in-flight NACA jitter (#2072) — together they cover the complete geometry input space.
10. **Translation augmentation σ=0.01 (channels 0-1, per-sample shift)** — **in-flight #2138 nezuko.** Physical-symmetry lever (NOT averaging-style). Per-sample rigid shift preserves foil shape exactly; exploits translation invariance of Navier-Stokes. **Critical diagnostic:** if this exhibits the bimodal signature, the 7× pattern generalizes to physical-symmetry augmentation too. If neutral or uniform improvement, we've found the first lever that breaks the bimodal.
11. **RMSNorm at all 3 sites (retry-3)** — **in-flight #2139 frieren.** 3rd attempt after 2 stale PRs (#1926, #2034) at pod-level. Pre-LN architectural variant; expected to NOT show bimodal signature since layer-level not sample-level. Modest expected gain.
12. **AdamW amsgrad=True (max-tracking 2nd-moment)** — **in-flight #2155 alphonse.** Novel optimizer-stability axis after AdamW (lr, β1, β2) closed 4-LOSS-for-4. Tracks MAX of 2nd-moment EMA instead of EMA itself; theoretically robust to L1 sign-flip noise. Distinct from prior optimizer levers — changes the **statistic** used for adaptive step bound, not magnitude/smoothing/variance-rate. Expected uniform per-split delta (positive or neutral).
13. **GELU→SiLU activation swap** — **in-flight #2156 askeladd.** Architectural probe at activation level, NOT averaging-style and NOT broadcast-scalar. Replaces GELU with SiLU at all 3 MLP sites (preprocess + TransolverBlock MLP + last_layer mlp2). Consistent literature support for 0.3-1% gain on Transformer benchmarks. Expected uniform per-split delta — if bimodal, surprising mechanism. Low-risk simple win.
