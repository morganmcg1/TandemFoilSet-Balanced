# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-29 01:25
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
| alphonse | #896 | **WIP (rebase)** | Per-sample y-norm won on MSE (val=105.45, −4.7% vs Huber best); merge conflict with Huber #739 → sent back to rebase on Huber baseline and stack both |
| askeladd | #733 | Closed | val_avg=151.50; throughput cost decisive |
| askeladd | #811 | **Merged** | val_avg=127.402; BF16 1.20× speedup |
| askeladd | #848 | Closed | bs={8,10}: regressed; bs=12 OOM; `add_derived_features` loop bottleneck |
| askeladd | #885 | **WIP** | Huber delta sweep ∈ {0.3, 0.5, 1.0, 2.0} on BF16 baseline |
| edward | #734 | Closed | sw=10 wins; sw=50/100 regress |
| edward | #850 | **WIP (rebase)** | sw=3 won internally (val=124.05, −2.6% vs 127.40) but on pre-Huber code. Sent back for single decisive sw=3 + Huber run on rebased baseline. |
| fern | #737 | **Merged** | val_avg=127.87; warmup+cosine |
| fern | #809 | **WIP** | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | **Merged** | Huber d=1.0: **val_avg=110.594 (−13.2%)**, test_avg=101.299 (−12.8%); new best. All 4 test splits finite. |
| frieren | #915 | **Closed** | PhysicsAttention padding mask — cruise improved −14% test (as predicted) but rc regressed +30.8% test; net test +3.3% worse. Binary post-softmax mask disrupts attention on dense-mesh rc geometries. |
| frieren | #943 | **WIP** | Per-channel surface loss weights: p_surf_weight ∈ {3, 20} vs vel_surf_weight=10 |
| nezuko | #742 | Closed | dropout=0.1 regresses 12.4%; undertrained model has no overfitting to regularize |
| nezuko | #878 | Closed | DropPath p=0.1 neutral on val_avg (+0.32, within seed noise) and +3% per-step overhead |
| nezuko | #923 | **WIP** | Vectorize `add_derived_features` — remove per-sample Python loop + `.item()` CPU sync; throughput unblock |
| tanjiro | #745 | **WIP (rebase)** | Sent back for Option 3 capacity-matched heads on rebased baseline |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | **WIP (rebase)** | EMA post-warmup-init + decay sweep on BF16 baseline |

**Current best val_avg/mae_surf_p (merged):** 110.594 (frieren #739, run `l95azbnv`).
**Current best test_avg/mae_surf_p (merged):** 101.299 (frieren #739, run `l95azbnv`).

**Five compounding wins stacked:**
1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → **val_avg=110.594**

**[PENDING MERGE]** Per-sample y-norm (#896, alphonse) — val=105.45, test=93.81 on MSE baseline;
needs rebase onto Huber + rerun to confirm stacking.

**All 8 GPUs in use:** alphonse #896 (per-sample-y-norm rebase), askeladd #885 (Huber-δ sweep),
edward #850 (lower-sw rebase), fern #809 (schedule-budget), frieren #943 (per-channel-surf-weight),
nezuko #923 (vectorize data prep), tanjiro #745 (heads Option 3 rebase), thorfinn #810 (EMA rebase).

## Current research themes

1. **Four compounding wins stacked (val_avg=110.594).** Per-sample y-norm (#896) appears decisive
   (−4.7% even on MSE baseline) but needs rebase to confirm stacking with Huber.
2. **Re-imbalance: two attacks in-flight.** Target-space: per-sample y-norm (#896 rebasing, alphonse).
   Loss-space: Huber-δ sweep (#885, askeladd). Both may stack — they address different aspects of
   the high-Re outlier problem (sigma rescaling vs gradient clipping).
3. **Per-channel surface loss weighting (#943, frieren)** — new hypothesis motivated by:
   (a) edward's observation that low surf_weight helps surf_p via volume-driven inference;
   (b) the physical insight that pressure is determined globally (Poisson equation), not just locally
   at surface nodes. Two competing directions: lower p_surf_weight=3 (volume-driven) vs
   higher p_surf_weight=20 (direct boost). Run p=3 first.
4. **Throughput bottleneck = `add_derived_features` Python loop (#923, nezuko).** Vectorization
   should free 5-15% wall-clock and unblock batch-size scaling.
5. **PhysicsAttention padding mask insight (#915 closed):** Binary post-softmax mask confirmed
   mechanism on cruise (−14% test) but disrupted dense-mesh rc geometries (+31% test). A soft
   learnable gate (sigmoid(MLP(x))) could capture the cruise benefit without the rc regression —
   Wave 3 candidate.

## Open questions

- Does per-sample y-norm (#896) still win when stacked on top of Huber δ=1.0? The MSE-only run
  won (val=105.45), but the rc split regressed (+12.6% test). With Huber already capping rc
  outlier gradients, per-sample-norm + Huber might partially cancel on rc.
- What is the optimal Huber δ? Askeladd #885 sweep {0.3, 0.5, 1.0, 2.0} will answer this.
- Does lower p_surf_weight (→3 volume-driven) outperform direct boost (→20) for surf_p MAE?
- Does EMA (#810) work properly after post-warmup init?
- Will surf_weight=3 + Huber (edward #850) beat the current 110.594 baseline?

## Potential next research directions (Wave 3+)

Prioritized given current insights:

- **Per-sample-norm + Huber confirmed stacking** → then per-channel sigma (normalize each of Ux,
  Uy, p separately, so pressure doesn't dominate sigma_per scalar).
- **Linearly-scaled-LR + bs=8** (after #923 lands). With `lr=2e-3, bs=8` (linear scaling rule)
  and gradient accumulation guard, retries batch-size scaling correctly.
- **Soft attention gate for padding** — `sigmoid(learned_gate(x_node))` multiplied into slice
  weights rather than hard 0/1 mask. Captures cruise benefit without rc regression.
- **`torch.compile` on model forward.** 1.2-1.5× speedup candidate but requires care with
  dynamic mesh sizes.
- **Capacity scaling to n_hidden=192** — with 17 epochs available in BF16+Huber, ~1.5M params
  might fit. Particularly interesting for cruise/rc splits.
- **Hard negative mining / Re-weighted sampler** — `WeightedRandomSampler` with weights ∝ Re or
  per-sample loss magnitude. Direct attack on Re-imbalance (alternative to per-sample-norm).
- **Architecture pivot** if tuning saturates: GINOs/FNOs (spectral), MeshGraphNets (explicit
  edges), equivariant attention, neural operator transformers.

Full literature bank in `research/RESEARCH_IDEAS_2026-04-28_19:30.md`.
