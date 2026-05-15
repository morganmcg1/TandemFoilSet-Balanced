# SENPAI Research State

- **Updated:** 2026-05-15 20:55
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 open for `test_geom_camber_cruise` NaN bug (frieren and fern both independently pinpointed root cause: inf×0=NaN in surf_mask-zeroed pressure sum).

## Current baseline

**`val_avg/mae_surf_p = 98.88`** — CosineAnnealingWarmRestarts(T_0=5, T_mult=2), PR #3320, **merged** (2026-05-15 20:25)

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 116.36 |
| val_geom_camber_rc | 108.40 |
| val_geom_camber_cruise | 77.91 |
| val_re_rand | 92.87 |
| **val_avg** | **98.88** |

## Winner pending replication

**PR #3307 askeladd OneCycleLR right-sized → val_avg=97.44 (single arm, beats current baseline by 1.44).** Sent back for (1) rebase onto warm-restarts baseline (resolve scheduler conflict by keeping OneCycleLR) and (2) 2 replication arms. If 3-arm mean beats 98.88, this becomes the new baseline (replaces warm-restarts on the scheduler axis). 3-split test (94.41) is also slightly better than current best (94.82).

## All merged results (best-first)

| PR | Change | val_avg | Δ vs prior baseline |
|---|---|---|---|
| #3320 nezuko | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | **98.88** | −15.6% ✓ **MERGED** |
| #3157 tanjiro | grad clip max_norm=1.0 | 117.16 | baseline |

## Round 3 portfolio (active experiments)

| Student | PR | Change | Rationale |
|---|---|---|---|
| nezuko | #3431 | EMA weights (decay=0.999) | Smooth out warm-restart oscillations; may improve best checkpoint quality |
| alphonse | #3436 | CosineAnnealingWarmRestarts T_0=3 | More frequent restarts (2→3 cycles in 14-epoch budget); direct T_0 sweep |
| edward | #3434 | L1 surface loss (vs MSE) | Align training objective with MAE metric; L1 minimizer = L1 metric |
| thorfinn | #3416 | Per-channel surf loss: p×3 | Directly weight primary metric channel harder vs Ux/Uy |
| askeladd | #3307 | OneCycleLR right-sized (rebase + replicate) | Winner pending — beat single-arm baseline; replicating |
| fern | #3462 | surf_weight 10 → 5 (multi-arm) | Test fern's own analysis: is the sweet spot below 10? Vol term provides useful structure |
| frieren | #3464 | slice_num 64 → 32 (multi-arm) | Counter-probe to closed #3146: if 64→128 over-partitions, does 64→32 under-partition? Buys ~30% per-epoch speedup |

## Round 2 WIP still completing (multi-arm, stale labels)

| Student | PR | Change | Status |
|---|---|---|---|
| tanjiro | #3360 | grad clip max_norm=0.5 | Stale WIP — ran training (GPU evidence), no comment yet; nudged with new-baseline note |

## Round 2 final results (closed without merge)

| PR | Change | val_avg | Δ | Decision |
|---|---|---|---|---|
| #3146 frieren | slice_num=128 | 143.88 mean (best 133.95) | +45.5% mean | closed — over-partitions, noisier projections |
| #3139 fern | surf_weight=25 | 159.16 mean (best 134.80) | +61.0% mean | closed — loses volume term's inductive structure, seed variance blows up |
| #3381 edward | n_hidden=192 | 126.44 | +7.9% | closed — width not bottleneck (per-epoch identical) |
| #3112 alphonse | bf16 autocast | 114.34 (best), ~122 (mean) | mean +4.3% | closed — speed benefit, accuracy neutral-to-negative |
| #3310 edward | n_layers=6 | 127.23 | +8.6% | closed — depth costs more epochs than it gains |
| #3308 thorfinn | AdamW beta2=0.95 | 134.89 | +15.1% | closed — grad noise grows, not shrinks |
| #3306 tanjiro | grad clip max_norm=100 | 124.31 | +6.1% | closed — confirms tight clip = gradient normalizer |
| #3153 nezuko | Huber vol loss | 127.22 (confounded) | +8.6% | closed — missing grad clip, high variance |
| #3164 thorfinn | dropout=0.05 | 142.51 | +21.7% | closed — no overfit in this budget |
| #3133 edward | n_layers=7 | 146.62 | +25.1% | closed — unstable without clip |
| #3125 askeladd | lr=1e-3 + warmup + cosine | 135.06 | +15.3% | closed — cosine horizon mismatch |

## Key research findings

1. **Schedule is the dominant lever** — warm-restarts gave 15.6% (#3320 merged), OneCycle right-sized may give another ~1.5% (#3307 pending replication). Both improvements come from giving the model multiple LR regimes within the 14-epoch budget, not from changing optimization fundamentals.
2. **Grad clip is gradient normalization** — max_norm=1.0 fires on 100% of steps; loosening regresses. Testing max_norm=0.5 (tanjiro #3360).
3. **Model is NOT capacity-limited** — width (n_hidden=192), depth (n_layers=6,7), and slice count (slice_num=128) all fail because they slow per-epoch time without proportional quality gain. The 30-min budget rewards configurations that finish more epochs, not those that fit more capacity per epoch.
4. **Loss weighting has a sweet spot** — surf_weight=25 regresses badly (loses volume term's structure); testing surf_weight=5 (#3462) to find lower bound.
5. **Slice count: 64 may not be optimal either** — 128 over-partitions (PR #3146 closed). Testing 32 (#3464) to find lower bound; expected to win on compute-saving alone.
6. **Second-moment adaptation** — beta2=0.95 fails (more gradient noise); baseline beta2=0.999 is near-optimal for this regime.
7. **bf16 is speed-neutral** — 29% more epochs in 30 min but accuracy neutral on average; not yet worth merging.
8. **Run variance is real and asymmetric to lever** — base 15-pp spread reduced to 3-pp by warm-restarts; surf_weight=25 blew it back up to 66 pp. Replication matters more for any change that affects loss landscape curvature.

## Potential next research directions (round 4+)

1. **Schedule compounding** — once OneCycle right-sized is confirmed (replication of #3307), test OneCycle + EMA stacking (nezuko #3431 result will inform). The two are orthogonal: OneCycle is LR schedule, EMA is weight averaging.
2. **Lion / SignSGD optimizer** — pure gradient-direction optimizer, often 1–2% over AdamW. Could compound with the schedule wins. Needs different LR (~3× lower).
3. **FiLM conditioning of global features** — Re, AoA, NACA codes are constant across mesh; currently broadcast per-node as input features. FiLM-encode once via small MLP, modulate Transolver block activations multiplicatively (γ⊙x + β). Adds multiplicative interactions that additive concat can't represent. ~few hundred extra params.
4. **Coordinate features** — random Fourier features `[sin(2^k π x), cos(2^k π x)]` for k∈{0..5} to help with high-frequency pressure spikes near foil surface. Cheap; could unblock the val_single_in_dist split (currently the highest at 116.36).
5. **Slice projection regularization** — entropy bonus on slice-assignment softmax to prevent token collapse. Addresses the "smaller node clusters get noisy projections" failure mode that frieren's slice_num=128 result diagnosed.
6. **Larger effective batch via grad accumulation** — current batch=4. Effective batch=16 via 4-step accumulation lowers gradient noise, may allow higher peak LR. Trade-off: 4× fewer optimizer steps per epoch.
7. **Test-time augmentation** — reflect airfoil around y=0, average predictions. Zero training cost; 0.5–1% typical gain on symmetric-physics problems.
8. **Per-domain adaptive slice_num** — the 3 physical domains have very different mesh sizes (85K, 127K, 210K nodes). Fixed slice_num gives very different node/token ratios. `slice_num ∝ sqrt(N)` would equalize granularity (frieren's follow-up #5).
9. **Auxiliary outputs** — predict pressure gradient or vorticity as auxiliary targets, multi-task with the main output head. Provides denser supervision signal.
10. **Log-space pressure loss** — surface pressure has wide dynamic range; MSE weights high-pressure points; log-space distributes weight per decade. May help OOD-camber generalization.
