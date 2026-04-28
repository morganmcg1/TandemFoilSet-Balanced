# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-28 23:35
- **Advisor branch:** `icml-appendix-willow-pai2e-r5`
- **Track tag:** `willow-pai2e-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r5`
- **Most recent direction from human team:** none yet (no human GitHub issues open).

## Research target

Beat the Transolver baseline on TandemFoilSet. Primary ranking metric is
`val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the four
validation splits) with `test_avg/mae_surf_p` as the paper-facing decision
metric. Baseline config is `n_hidden=128, n_layers=5, n_head=4, slice_num=64,
mlp_ratio=2`, AdamW + CosineAnnealingLR, `lr=5e-4`, `surf_weight=10.0`,
batch=4, 50 epochs, vol+10·surf MSE in normalized space.

The four validation tracks each probe a different generalization axis:
- `val_single_in_dist` — random holdout from single-foil (sanity).
- `val_geom_camber_rc` — unseen front-foil camber raceCar tandem M=6-8.
- `val_geom_camber_cruise` — unseen front-foil camber cruise tandem M=2-4.
- `val_re_rand` — stratified Re holdout across all tandem domains.

Per programme contract, surface pressure on the held-out camber/Re splits is
where the paper-facing numbers live. Per-sample y std varies by an order of
magnitude even inside one domain, so high-Re samples drive the extremes.

## Wave-1 + Wave-2 status

| Student | PR | Status | Result |
|---------|----|--------|--------|
| alphonse | #732 | Closed | val_avg=154.95 ref; NaN test; 6/50 epochs |
| alphonse | #796 | Closed | FiLM-Re ineffective: val_avg=135.51 vs 135.35 control; test +4.5%; predicted re_rand −5–15% but got −0.9% |
| alphonse | #896 | **WIP** | Per-sample y-normalization (`sigma_per`-batch) |
| askeladd | #733 | Closed | val_avg=151.50; +18.5% regression; throughput cost decisive |
| askeladd | #811 | **Merged** | val_avg=127.402 ← best; test_avg=116.211 (clean); 1.20× speedup; 33 GB VRAM (63 GB free) |
| askeladd | #848 | Closed | bs={8,10}: +11.9% / +15.6% regression; bs=12 OOM; per-sample throughput dropped (`add_derived_features` Python loop bottleneck) |
| askeladd | #885 | **WIP** | Huber delta sweep ∈ {0.3, 0.5, 1.0, 2.0} on BF16 baseline |
| edward | #734 | Closed | sw=10 wins (130.43); sw=50/100 regress 3.9-4.8%. Volume context informs surface — refutes the "more weight = more focus" hypothesis. |
| edward | #850 | WIP | Lower-surf-weight sweep {3, 5, 7} on rebased baseline |
| fern | #737 | **Merged** | val_avg=127.87 ← best; warmup+cosine |
| fern | #809 | WIP | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | **Sent back (rebase)** | Huber d=1.0: **val_avg=103.89 (−18.5%)**, test_avg=102.60 3-split (−20.2%); biggest single-PR win. On pre-merge code; rebase onto BF16 baseline before merge. |
| nezuko | #742 | Closed | dropout=0.1 regresses 12.4% — undertrained model has no overfitting to regularize. OOD-hits-hardest signature. |
| nezuko | #878 | WIP | DropPath/stochastic depth on residual branches (drop_path_max=0.1, linear scaling) |
| tanjiro | #745 | WIP (rebase) | Heads on old code: Opt1=130.82 / Opt2=134.46. Sent back for Option 3 capacity-matched on rebased baseline. |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | WIP (rebase) | EMA d=0.999 missed (+9.07 val_avg) due to warmup contamination; sent back for post-warmup-init + decay sweep on BF16 baseline. Test improved (-4.3). |

**Current best val_avg/mae_surf_p (merged):** 127.402 (askeladd #811, run newqt8dd).
**Current best test_avg/mae_surf_p (merged):** 116.211 (askeladd #811, clean 4-split).

**Pending winner:** frieren #739 reports val_avg=103.89 (−18.5% on pre-merge baseline). Verified on W&B (run `z2a34zbu`). Sent back for rebase onto BF16 baseline; if the win holds the merge will set a new floor by a large margin.

**Key learnings as of 2026-04-28 23:35:**
- **Huber loss (δ=1.0) is the largest single intervention so far.** −18.5% val_avg, consistent across all 4 splits (14–25% improvement each). Mechanism: Huber caps the contribution of high-Re outlier samples that otherwise dominate the MSE gradient with surf_weight=10.
- **Three compounding baseline wins merged:** distance features (#763) + warmup+cosine (#737) + BF16 (#811) → 127.402. **Huber → 103.89** (pre-merge); rebased Huber will likely set the new floor.
- **Batch-size scaling is dead at constant LR** (#848 closed). Per-sample throughput regressed. The unblocking lever is **vectorizing `add_derived_features`** (Python loop with `.item()` CPU sync). This is now the dominant non-matmul cost.
- BF16 merged (#811): 1.20× per-epoch speedup, 17 epochs in 30 min (was 14), 33 GB VRAM. 63 GB headroom now wasted because batch-size scaling didn't pay off without LR scaling.
- `test_geom_camber_cruise` NaN reproduces on under-trained large models; scoring.py NaN-pred gap confirmed (data/ read-only). NaN-safe workaround in train.py merged (#763).
- FiLM-Re (#796) refuted: log(Re) is already in input features and the attention extracts it adequately. Per-sample y-normalization (#896) targets the same problem (Re-regime gradient imbalance) more directly.
- **All 8 GPUs in use:** alphonse #896 (per-sample-y-norm), askeladd #885 (Huber-δ sweep), edward #850 (lower-sw), fern #809 (schedule-budget), frieren #739 (Huber rebase), nezuko #878 (DropPath), tanjiro #745 (heads Option 3 rebase), thorfinn #810 (EMA post-warmup rebase).

## Current research themes

1. **Huber loss is the headline win to verify and stack.** −18.5% on pre-merge code (#739, run `z2a34zbu`). The rebased run + δ-sweep (#885) will tell us whether δ=1.0 is the optimum and how much of the gain survives BF16+warmup. **This is the highest-priority result.**
2. **Three compounding baseline wins merged:** distance features (#763) + warmup+cosine (#737) + BF16 (#811). val_avg=127.402, test_avg=116.211 (clean 4-split). This is the platform; rebased Huber will likely replace it as floor.
3. **Re-regime gradient imbalance is the next axis to attack** (Huber attacks it from the loss side). Per-sample y-normalization (#896) attacks it from the target-scale side — direct test of the same underlying problem from a complementary angle.
4. **Throughput bottleneck has shifted:** the matmul cost is no longer dominant; `add_derived_features` Python loop with per-sample `.item()` calls and chunked pairwise distances is. Vectorizing this would unlock larger batch-size scaling.
5. **In-flight Wave-2 still needs to land:** EMA post-warmup (#810), tanjiro Option 3 heads (#745), schedule-budget (#809), DropPath (#878), lower-sw sweep (#850).

## Potential next research directions (Wave 3+ candidates)

If Huber-rebase wins, focus shifts to **stacking on top of Huber+BF16+features+warmup**. Higher-priority bets given current results:

- **Huber + per-sample y-norm joint test.** If both stack, the gain may be 25–30%.
- **Vectorize `add_derived_features`** (Python loop → torch ops). Pure throughput win, unblocks batch-size scaling. Target a student who can do clean systems work.
- **Scale-aware pressure head** (log-magnitude or per-Re rescale) — if y-norm wins, bake it into the model output formulation.
- **PhysicsAttention with explicit mesh mask.** Padded zero-coordinate nodes still receive softmax mass, contaminating slice tokens. Fixes the cruise-NaN at root and may improve all splits.
- **WeightedRandomSampler over Re bins.** Hard-example mining / Re-stratified curriculum — alternative attack on the same imbalance Huber targets.
- **Linearly-scaled-LR + bs=8.** Now that we know constant-LR fails, the natural retry is `lr=1e-3, bs=8` (linear LR scaling rule) once the feature-loop is vectorized.
- **Gradient accumulation** to test "effective batch size" hypothesis at a fixed step count, decoupled from VRAM cost.
- **Architecture pivot.** If tuning saturates: GINOs/FNOs (spectral), MeshGraphNets (explicit edges), neural-operator transformers, equivariant attention.
- **Multi-scale / hierarchical attention** for the 74K–242K node range and three-zone tandem topology.
- **Boundary-layer-aware auxiliary loss** (heteroscedastic head, gradient reweighting on `is_surface`).
- **Test-time adaptation** for unseen-camber splits.

Full literature bank in `research/RESEARCH_IDEAS_2026-04-28_19:30.md`.

## Open questions

- Will Huber's −18.5% pre-merge gain hold (or grow) on the merged BF16 baseline? Frieren #739 rebase will tell us.
- What is the optimal δ for Huber? Askeladd #885 sweep on {0.3, 0.5, 1.0, 2.0} should give us a maximum.
- Does per-sample y-norm (#896) stack with Huber, or are they hitting the same gradient-imbalance lever from different sides?
- Whether the four val tracks rank interventions consistently (early data suggests cruise/Re-rand improve faster than single_in_dist and rc — may reflect domain difficulty ordering).
- Whether vectorizing `add_derived_features` makes batch-size scaling viable (next throughput target if a student becomes idle and we want to retry batch-size).
