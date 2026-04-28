# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-28 23:48
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
| alphonse | #796 | Closed | FiLM-Re ineffective: val_avg=135.51 vs 135.35 control; test +4.5% |
| alphonse | #896 | **WIP** | Per-sample y-normalization (`sigma_per`-batch) |
| askeladd | #733 | Closed | val_avg=151.50; throughput cost decisive |
| askeladd | #811 | **Merged** | val_avg=127.402; BF16 1.20× speedup |
| askeladd | #848 | Closed | bs={8,10}: regressed; bs=12 OOM; `add_derived_features` loop bottleneck |
| askeladd | #885 | **WIP** | Huber delta sweep ∈ {0.3, 0.5, 1.0, 2.0} on BF16 baseline |
| edward | #734 | Closed | sw=10 wins; sw=50/100 regress |
| edward | #850 | WIP | Lower-surf-weight sweep {3, 5, 7} |
| fern | #737 | **Merged** | val_avg=127.87; warmup+cosine |
| fern | #809 | WIP | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | **Merged** | Huber d=1.0: **val_avg=110.594 (−13.2%)**, test_avg=101.299 (−12.8%); new best. All 4 test splits finite. |
| frieren | #915 | **WIP** | PhysicsAttention padding mask — silence padded nodes before slice softmax |
| nezuko | #742 | Closed | dropout=0.1 regresses 12.4% |
| nezuko | #878 | WIP | DropPath/stochastic depth (drop_path_max=0.1) |
| tanjiro | #745 | WIP (rebase) | Sent back for Option 3 capacity-matched heads on rebased baseline |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | WIP (rebase) | EMA post-warmup-init + decay sweep on BF16 baseline |

**Current best val_avg/mae_surf_p (merged):** 110.594 (frieren #739, run `l95azbnv`).
**Current best test_avg/mae_surf_p (merged):** 101.299 (frieren #739, run `l95azbnv`).

**Four compounding wins stacked:**
1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → **val_avg=110.594**

**All 8 GPUs in use:** alphonse #896 (per-sample-y-norm), askeladd #885 (Huber-δ sweep), edward #850 (lower-sw), fern #809 (schedule-budget), frieren #915 (PhysicsAttention mask), nezuko #878 (DropPath), tanjiro #745 (heads Option 3 rebase), thorfinn #810 (EMA post-warmup rebase).

## Current research themes

1. **Four compounding wins now stacked (val_avg=110.594).** Next frontier: fix the remaining structural sources of error: padding attention contamination (#915), Re-imbalance from the target side (#896), and optimal loss threshold (#885).
2. **PhysicsAttention padding mask (#915) is the highest-priority structural fix.** `val_geom_camber_cruise` barely improved with Huber (+1.2% val / −1.4% test). Frieren diagnosed the root: padded zero-vector nodes contaminate slice tokens via unmasked softmax. Cruise has the most variable mesh sizes → worst padding ratio → most contamination. Fix is exact and contained (5 surgical changes in train.py).
3. **Re-imbalance attacks from two complementary angles in-flight:** per-sample y-norm (#896, alphonse — target-space normalization) and Huber-δ-sweep (#885, askeladd — loss-space). Both should complete soon. If they stack, the combined gain could be substantial.
4. **Throughput bottleneck = `add_derived_features` Python loop.** Per-sample `.item()` CPU-sync + chunked pairwise distance prevent batch-size scaling. Vectorizing this is the highest-leverage systems engineering next step once a suitable student is idle.
5. **Wave-2 still in-flight:** EMA post-warmup (#810), tanjiro Option 3 heads (#745), schedule-budget (#809), DropPath (#878), lower-sw sweep (#850). These should all be evaluated against the new 110.594 floor.

## Potential next research directions (Wave 3+)

Prioritized given Huber's mechanism is now validated:

- **Vectorize `add_derived_features`** — remove Python loop + `.item()` CPU sync. Unblocks batch-size scaling. One student slot when someone becomes idle.
- **Linearly-scaled-LR + bs=8** (after vectorization). With `lr=1e-3 * (bs/4)` = `lr=2e-3, bs=8` and gradient accumulation guard, this retries batch-size scaling correctly.
- **Scale-aware pressure output** — per-Re rescaling or log-magnitude head for the pressure channel specifically. If per-sample y-norm wins, this is the natural follow-on.
- **Capacity scaling to n_hidden=192** — with 17 epochs available in BF16+Huber, ~1.5M params might fit. Particularly interesting for cruise/rc splits that require tandem interactions.
- **Hard negative mining / Re-weighted sampler** — `WeightedRandomSampler` with weights ∝ Re or per-sample loss magnitude. Alternative attack on Re-imbalance.
- **Architecture pivot** if tuning saturates: GINOs/FNOs (spectral), MeshGraphNets (explicit edges), equivariant attention, neural operator transformers.
- **Boundary-layer auxiliary loss** (heteroscedastic head or gradient reweighting on `is_surface` nodes).

Full literature bank in `research/RESEARCH_IDEAS_2026-04-28_19:30.md`.

## Open questions

- Does the PhysicsAttention padding mask (#915) explain the cruise split stagnation? Are the other splits also affected?
- What is the optimal Huber δ? Askeladd #885 sweep {0.3, 0.5, 1.0, 2.0} will answer this.
- Do per-sample y-norm (#896) and Huber δ=1.0 stack (complementary mechanisms) or cancel (same lever)?
- Does EMA (#810) work properly after post-warmup init? Test improvement from #810 pre-fix was already −4.3 points.
- Will the val curve continue descending rapidly post-epoch-17 if we could run longer? (We can't change the timeout, but throughput improvements increase effective epochs.)
