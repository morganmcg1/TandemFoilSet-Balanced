# SENPAI Research State

- 2026-05-14 00:55 — willow-pai2g-48h-r1, round 3 ongoing. **CURRENT BEST: test=46.2751 (PR #2568 Lion lr=2e-4 + post-LN + T_max=20 + wd=0)**. Cumulative gain from PR #1391: 121.28 → 46.28 = **−61.8%**.
- No directives from human researcher team yet.
- ⚠️ Cycle 13 fleet-wide polling issue (00:54 UTC): W&B confirms NO student has launched a run matching their assigned PR config — neither for Cycle 10 (#2630/#2632/#2634/#2640) nor Cycle 11 (#2663/#2664/#2665/#2591). All 8 pods are alive (kubectl Running) but appear to not have polled for new work. Most recent runs are from BEFORE PR creation. Sent nudge comments to 4 Cycle 10 PRs (most overdue: 2-7.8h since pod activity). Cycle 11 PRs (~75 min old) and #2591 (~3.5h old, already nudged) await pod recovery.
- ⚠️ Cycle 14 root cause identified (01:52 UTC): pod log inspection (kubectl logs nezuko-pod) reveals **GitHub GraphQL API rate limit exhaustion** — student poll-for-work skill fails 6× retries on every heartbeat, falls back to "No assigned PRs or issues" → sleeps 300s → repeats. Fleet-wide silence is NOT a pod bug but an API throttle on user 20516801 shared across all 8 student pods. Pods will resume polling once rate limit clears (~1h). **Stopped sending nudge comments** to avoid adding more API load. Awaiting natural recovery.
- Cycle 11: 2 closed (#2584 lr=2e-4+T_max=22 falsified; #2586 clip=3.0 tied); 3 new assigned (#2663 alphonse clip=3.0+T_max=20 stack; #2664 thorfinn lr=1.75e-4+T_max=20 left-probe; #2665 askeladd Fourier L=12 post-LN re-validation).

## Current baseline (PR #2568 merged — lr=2e-4 + T_max=20 stack)

**test_avg/mae_surf_p = 46.2751** | val = 53.1311 (best epoch 18/18)
Config: bf16 + bs=4 + accum=2 + Lion **lr=2e-4** + β1=0.9 + β2=0.99 + **wd=0** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, **slice_num=24**, mlp_ratio=2 + **grad_clip_max_norm=5.0** + surf_weight=10 + act=gelu + eta_min=0 + dropout=0 + **post-LN** + **t_max=20**. W&B run: lym1yzlo.

Per-split: in_dist=47.8109, rc=58.8461, cruise=32.0006, re_rand=46.4428.

**Reproduce:** `cd target/ && python train.py --batch_size 4 --accumulation_steps 2 --grad_clip_max_norm 5.0 --weight_decay 0.0 --lr 2e-4 --t_max 20`

⚠️ ALL future run commands must include `--weight_decay 0.0 --lr 2e-4 --t_max 20`.
⚠️ Post-LN is now merged into train.py default via PR #2456.
⚠️ Note: lr=2.25e-4 at T_max=18 gave 46.8821 (PR #2572 — also a win, slightly worse). lr=2e-4+T_max=20 (46.27) is now the baseline. frieren #2591 tests lr=2.25e-4+T_max=20 to find the compound optimum.

## Win history (this round)

| PR | Config change | test_avg | Δ | Cumulative |
|---|---|---|---|---|
| #2282 | slice_num=24 | 61.8457 | −1.52% | −49.0% |
| #2343 | wd=0 | 60.7447 | −1.78% | −49.9% |
| #2456 | post-LN swap | 51.5839 | −15.08% | −57.5% |
| #2508 | T_max=20 | 49.3466 | −4.34% | −59.3% |
| #2494 | lr=2e-4 | 47.9076 | −2.92% | −60.5% |
| #2572 | lr=2.25e-4 | 46.8821 | −2.14% | −61.4% |
| **#2568** | **lr=2e-4 + T_max=20** | **46.2751** | **−1.30%** | **−61.8%** |

## Major milestones

### Lion optimizer fully tuned (pre-LN closures revised under post-LN)

| Lever | Status | Optimal | Source |
|---|---|---|---|
| `lr` | **CLOSED** (post-LN — UPDATED) | 2.25e-4 | #2572 / Finding #54 updated |
| `β1` | TESTING (post-LN) | 0.9 (pre-LN) | #2530 alphonse (wip) |
| `β2` | **CLOSED** (post-LN — same as pre-LN) | 0.99 | #2533 closed, Finding #51 |
| `weight_decay` | CLOSED | 0.0 | #2343 / Finding #38 |

### Post-LN as new foundation (Finding #45)
PR #2456 post-LN swap: **−15.08%** (60.74 → 51.58). All 4 splits improved uniformly (13.5–17.3%). Key mechanism: placement-after-residual keeps residual stream stationary — representation-level effect, not IID/OOD redistribution. Sharp contrast with RMSNorm (#2425): **computation type is second-order; placement is first-order**.

### Optimizer scale + schedule shape — both moved under post-LN
- **T_max** moved 18 → 20 (Finding #47, PR #2508)
- **lr** moved 1.5e-4 → 2e-4 (Finding #54, PR #2494)
- **β2** stayed at 0.99 (Finding #51 confirmed under post-LN — β2=0.999 +35.1% catastrophic)

The two moves are individually validated; the updated best LR is 2.25e-4 (Finding #54 update). The stack (lr=2.25e-4 + T_max=20) is now being tested by frieren #2591. The old lr=2e-4 + T_max=20 is askeladd #2568. Predicted additional gain for the stack: −1 to −3%.

### IID/OOD redistribution meta-pattern (Finding #41) — REVISED
Originally: capacity/resolution increases improve IID, harm OOD. Post-LN's uniform IID+OOD improvement BREAKS this pattern. Revised: the pattern holds for capacity/resolution axes; normalization position is an orthogonal axis that can improve both simultaneously. **Under lr=2e-4, the IID split now dominates the gain** (in_dist −6%) — the OOD/IID redistribution may be reasserting itself in the lr-scale axis even though it was broken in the LN-position axis.

### Per-split lever-flip under post-LN (Finding #53, frieren #2426)
The pre-LN sign of the surf_weight–rc relationship **inverted** under post-LN. Pre-LN: sw=5 → rc −0.74 (DOWN helps rc). Post-LN: sw=5 → rc +1.49 (DOWN hurts rc). Cruise mirror-inverted. Mechanism: post-LN's compressed residual stream makes surface/volume token representations more similar, breaking the surf_weight gradient signal that pre-LN had. **Implication:** other per-split findings from pre-LN should be re-validated case-by-case under post-LN.

## Round-3 status (updated 2026-05-13 23:50)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| frieren | #2591 | lr=2.25e-4 + T_max=20 STACK | **wip** |
| fern | #2630 | lr=2.5e-4 + T_max=20 (right-side LR probe) | **wip** |
| edward | #2632 | lr=2.25e-4 + T_max=22 (new LR × T_max extension) | **wip** |
| nezuko | #2634 | surf_weight=15 UP (Finding #53 mirror) | **wip** |
| tanjiro | #2640 | mlp_ratio=2.5 (non-integer capacity compromise) | **wip** |
| alphonse | #2663 | clip=3.0 + T_max=20 stack (compound IID×OOD) | **wip** (NEW) |
| thorfinn | #2664 | lr=1.75e-4 + T_max=20 (left-side LR probe) | **wip** (NEW) |
| askeladd | #2665 | Fourier L=12 under post-LN (Finding #41 flip test) | **wip** (NEW) |
| thorfinn | #2584 | lr=2e-4 + T_max=22 stack | **CLOSED** ✗ +6.17% rel regression. Finding #57 (T_max=22 only helps at lr≤1.5e-4). |
| alphonse | #2586 | grad_clip_max_norm=3.0 under post-LN+lr=2e-4 | **CLOSED** ✗ tied with current best aggregate; per-split lever-flip in_dist −6.74% vs cruise/re_rand +4-5%. Reassigned to stack with T_max=20 (#2663). |
| askeladd | #2568 | lr=2e-4 + T_max=20 STACK | **MERGED** ✓ test=46.2751 (−1.30%). **NEW BEST.** |
| frieren | #2572 | lr=2.25e-4 LR midpoint refinement | **MERGED** ✓ test=46.8821 (−2.14%) |
| fern | #2573 | mlp_ratio=3 capacity width | **CLOSED** ✗ +2.68% worse vs new best. Timeout-bound. |
| edward | #2577 | slice_num=32 physics re-cal | **CLOSED** ✗ +4.75% worse, rc regressed. |
| nezuko | #2466 | SwiGLU MLP | **CLOSED** ✗ stale WIP (tied with old baseline, no response). |
| tanjiro | #2499 | RMSNorm under post-LN | **CLOSED** ✗ stale WIP (crashed, no response). |
| fern | #2573 | mlp_ratio=3 capacity width without depth | **wip** |
| edward | #2577 | slice_num=32 physics-aware re-cal under post-LN | **wip** |
| nezuko | #2466 | SwiGLU MLP (param-matched) | **wip** (status check #2 sent — `d0nsbeam` run finished at test=51.59 on old config, awaiting confirmation arm with lr=2e-4) |
| tanjiro | #2499 | RMSNorm under post-LN | **wip** (status check sent — last run crashed) |
| askeladd | #2494 | Post-LN LR re-calibration: lr=2e-4 wins | **MERGED** ✓ test=47.9076 (−2.92%). **NEW BEST.** |
| thorfinn | #2508 | T_max=20 cosine extension under post-LN | **MERGED** ✓ test=49.3466 (−4.34%) |
| tanjiro | #2456 | Pre-LN → Post-LN swap | **MERGED** ✓ test=51.5839 (−15.08%). |
| alphonse | #2530 | Lion β1 re-cal under post-LN (β1=0.95) | **CLOSED** ✗ +5.44% uniform, OOD-heavy. Finding #55 (β1=0.9 robust). |
| thorfinn | #2527 | T_max=22 under post-LN at lr=1.5e-4 | **CLOSED** ✗ beat old baseline by −0.93% but lost to new 47.91 best. Finding #56 (Finding #44 falsified for decaying cosine). Mechanism clean — reassigned to lr=2e-4 + T_max=22 stack (#2584). |
| fern | #2533 | Lion β2 re-calibration under post-LN | **CLOSED** ✗ β2=0.999 catastrophic +35.1%. Finding #51. |
| edward | #2528 | n_layers=6: depth increase under post-LN | **CLOSED** ✗ +8.78% timeout-bound. Finding #52. |
| frieren | #2426 | surf_weight DOWN (sw=5/7) | **CLOSED** ✗ Flat +0.60%, per-split flip. Finding #53. |
| alphonse | #2485 | Lion gradient noise: LR-scaled Langevin perturbation | **CLOSED** ✗ +3.46% regression. Finding #49. |
| edward | #2473 | Slot routing temperature: fixed T | **CLOSED** ✗ +4.87% regression. Finding #48. |
| fern | #2474 | Coord-noise augmentation | **CLOSED** ✗ Within noise floor. Finding #50. |
| askeladd | #2458 | Lookahead-wrapped Lion | **CLOSED** ✗ +9.76% regression. Finding #45b. |
| thorfinn | #2433 | Per-iter warmup | **CLOSED** ✗ +7.78–9.19% regression. Finding #46. |
| alphonse | #2326 | cosine eta_min=1.5e-5 | **CLOSED** ✗ +6.24%. Finding #44. |
| edward | #2327 | GELU → SiLU activation swap | **CLOSED** ✗ +16% regression. Finding #42. |
| fern | #2117 | EMA decay=0.95/0.99 | **CLOSED** ✗ Finding #43. |
| nezuko | #2393 | Fourier L=12/4 | **CLOSED** ✗ Finding #41. |
| tanjiro | #2425 | LayerNorm → RMSNorm (standalone) | **CLOSED** ✗ Finding #40. |
| askeladd | #2382 | Lion β2=0.999/0.95 (pre-LN) | **CLOSED** ✗ Finding #39. |

## Key research findings (cumulative — latest at bottom)

1–19. [Lion, Fourier, width, schedule, clip, accum=2, etc.]
20. **Lion LR scaling (pre-LN closed)**: lr=1.5e-4 correct for pre-LN. **SUPERSEDED by Finding #54.**
21. **Clip mechanism fully characterized**: clip=5.0 in bulk-rescaling plateau [2,5].
22. **LayerScale CLOSED**: counterproductive with Lion+clip.
23. **slice_num scan COMPLETE**: Floor at slice_num=24 (under pre-LN — re-cal testing as of #2577).
24. **EMA gain is budget-dependent**: SUPERSEDED by Finding #43.
25. **accum=4 PERMANENTLY CLOSED**: accum=2 optimal.
26. **T_max sweep CLOSED (pre-LN)**: T_max=18 optimal pre-LN. **SUPERSEDED by Finding #47.**
27. **Capacity-budget structural bound**: >+5% per-epoch overhead = net loser. Reinforced by Finding #52.
28. **Schedule completion bonus**: slice=24 → full 18/18 schedule.
29. **Per-epoch warmup = step function (CLOSED BUG)**.
30. **Attention dropout CLOSED**: Slot routing needs determinism.
31. **Inter-seed variance noise floor ~1.4% rel**.
32. **Lion β1 lever CLOSED (pre-LN)**: β1=0.9 optimal. Testing under post-LN (#2530).
33. **Locality-prior OOD tradeoff**: slice_num lever trades OOD, not in-dist accuracy (pre-LN).
34. **Capacity-limited regime** (#2344): 1.47M+18 epochs = capacity-limited.
35. **Locality regularization incompatible with stochastic attention** (#2344).
36. **surf_weight has split-asymmetric effects** (#2294): pre-LN.
37. **surf_weight lever CLOSED (pre-LN)** (#2294, #2426 follow-up): sw=15/20 worse on rc pre-LN; sw=5 flat under post-LN with per-split flip (Finding #53). DOWN closed.
38. **Lion wd=0 optimal** (#2343): NEW BASELINE.
39. **Lion β2=0.99 sharp sweet spot (pre-LN)** (#2382). **REINFORCED by Finding #51 under post-LN.**
40. **Normalization-type lever closed** (#2425): LayerNorm optimal; RMSNorm neutral aggregate.
41. **IID/OOD redistribution meta-pattern** (#2393, REVISED): post-LN breaks the capacity-tradeoff; lr-scale axis may reassert it.
42. **SiLU vs GELU convergence-rate asymmetry** (#2327): GELU optimal (pre-LN).
43. **EMA gain is timeout-budget-dependent** (#2117).
44. **LR floor ≥ 1e-5 in cosine tail is harmful** (#2326): applies to FIXED floor; T_max=22 (#2527) probes whether decaying cosine through 1e-5 is harmful too.
45. **Post-LN swap is decisive** (#2456): −15.08% across ALL splits uniformly. Residual-stream stationarity is the load-bearing lever.
45b. **Lookahead incompatible with Lion's binary updates** (#2458).
46. **Warmup incompatible with Lion's sign-update** (#2433): clip_fire stays ~100% in epoch 1 WITH warmup; instability doesn't exist under Lion.
47. **T_max=18 was pre-LN-specific; post-LN requires T_max=20** (#2508): Extended tail LR is load-bearing.
48. **Slot routing temperature T is load-bearing** (#2473): 20 params not dead weight.
49. **Gradient noise contraindicated for Lion** (#2485): Lion's sign-update already a noisy direction estimator.
50. **Coord-noise has narrow per-split-heterogeneous sigma-curve** (#2474): rc anti-correlated with cruise/re_rand on sigma.
51. **β2=0.99 confirmed optimal under post-LN** (#2533 closed): β2=0.999 → +35.1% catastrophic test regression. Post-LN gradient statistics make over-smoothing **worse** not better — Lion's sign-of-momentum needs gradient variance to make adaptive per-coordinate decisions. β2 valley appears NARROWER under post-LN, not broader (asymmetric sensitivity).
52. **n_layers=6 timeout-bound under 30-min cap** (#2528 closed): Post-LN does enable stable depth-6 training (sanity gates pass, gn_mean monotonically descends), but +23% per-epoch cost cuts schedule by 3 epochs → +8.78% regression. Mechanism viable; budget constraint blocks merge under fixed cap. Reinforces Finding #27.
53. **Per-split lever-flip under post-LN** (#2426 closed): surf_weight–rc gradient SIGN flipped pre-LN → post-LN. sw=5 helps rc pre-LN, hurts rc post-LN. Cruise mirror-inverted. Mechanism: post-LN's compressed residual stream makes surface/volume tokens more similar. **Implication for round design**: per-split findings from pre-LN are not transferable — must re-validate case-by-case under each new optimizer config.
54. **Lion lr=2.25e-4 optimal under post-LN** (#2494 + #2572 UPDATED): pr=2494 showed lr=2e-4 (1.33×) best among {1.5e-4, 2e-4, 3e-4}. PR #2572 found the true interior optimum at lr=2.25e-4 (1.50× pre-LN's 1.5e-4). The LR-vs-test curve is sharply concave at 2.25e-4: val at the midpoint is ~10× further from each endpoint than the endpoints from each other. All 4 splits improve; clip-fire at e18=44.1% (below both neighboring LR points), indicating the 2.25e-4 optimization path settles most cleanly. **Updated reproduce:** `--lr 2.25e-4`. The valley's right side (lr=2.5e-4) remains unmapped.
55. **Lion β1=0.9 robust to post-LN representation shift** (#2530 closed): β1=0.95 produces uniform +5.44% test regression under post-LN+T_max=20. OOD penalty (mean +7.10%) ~2.7× IID penalty (+2.66%) — post-LN tail geometry rewards faster adaptation, not more history. Finding #32 NOT stale. β1 lever closed across both LN positions. **Mechanism**: post-LN's deeper minimum demands sharper directional updates in cosine tail; β1=0.95 over-smooths the sign vote precisely in this regime.
56. **Finding #44 falsified for decaying cosine** (#2527 closed): T_max=22 at lr=1.5e-4 produces the **largest single-epoch val improvement of the second half** (−3.74 at ep17→18) precisely as LR decays through 1.83e-5 → 1.19e-5. Finding #44's "LR ≥1e-5 harmful" constraint applies to **fixed floors perturbing converged solutions**, NOT to transient decaying LR during active descent. The mechanism is decisive: rc OOD gain of −2.92% rel at T_max=22 (lr=1.5e-4) cannot survive the lr=2e-4 baseline shift (final test 48.89 vs current best 47.91), but the **tail-LR-is-load-bearing** finding is general and motivates the lr=2e-4 + T_max=22 stack (#2584).
57. **T_max>20 only helps at very low LR (≤1.5e-4)** (#2584 closed): Stacking lr=2e-4 + T_max=22 under post-LN regressed to test=50.86 (+6.17% rel vs PR #2494 baseline 47.91, ~4× noise floor) — more severe than worst-case prediction. Mechanism: Finding #56 was calibrated at lr=1.5e-4 where ep17-18 reaches productive tail LR (1.83e-5 → 1.19e-5). At lr=2e-4 + T_max=22, the same epochs sit at ~4.3e-5 → 2.0e-5 — 2-3× higher than the productive refinement regime. Higher mid-training LR also keeps the model in a noisier regime during the critical mid-late training window (in_dist +16.74% regression). The T_max=20 cap is the right schedule for the lr=2e-4 stack; longer T_max only pays off when the peak LR is low enough that the tail actually reaches sub-1.5e-5 within 18 epochs.

## Active experiments (8 students — all occupied)

### Tier 1: LR×T_max landscape — primary grid mapping (HIGHEST EV)
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2591 | frieren | **lr=2.25e-4 + T_max=20 STACK** (new LR optimum + T_max=20) | −0.5% to −2% vs 46.27 |
| #2630 | fern | **lr=2.5e-4 + T_max=20** (right-side LR probe at T_max=20) | maps valley; −1% if optimum shifts right |
| #2632 | edward | **lr=2.25e-4 + T_max=22** (new LR + further extension) | −1% to −2% vs 46.27 |
| #2664 | thorfinn | **lr=1.75e-4 + T_max=20** (left-side LR probe at T_max=20) | maps left side; −0.5% to −1.5% if optimum shifts left |

### Tier 2: Optimizer × schedule stacks
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2663 | alphonse | **clip=3.0 + T_max=20** (compound IID gain × OOD gain) | −0.5% to −2.0% if mechanisms stack |

### Tier 3: Representation re-validation
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2665 | askeladd | **Fourier L=12** under post-LN (Finding #41 flip test) | −1% on rc if post-LN supports finer encoding |

### Tier 4: Loss formulation + capacity
| PR | Student | Hypothesis | Expected gain |
|---|---|---|---|
| #2634 | nezuko | surf_weight=15 UP (Finding #53 mirror — rc-targeted) | rc-focused; −1% if directional sign holds |
| #2640 | tanjiro | mlp_ratio=2.5 (non-integer capacity compromise, stays under 110s gate) | −0.5% to −1.5% if timeout was the blocker |

## Key open questions

1. **Does lr=2.25e-4 + T_max=20 beat lr=2e-4 + T_max=20?** (frieren #2591) — determines whether the LR optimum (2.25e-4) and T_max (20) compound at BOTH levels. If yes, new best < 46.27.
2. **Does lr=2.5e-4 win at T_max=20?** (fern #2630) — maps right side of LR valley at new T_max baseline. If optimum is at 2.5e-4, we shift the best config further.
3. **Does lr=2.25e-4 + T_max=22 beat 2.25e-4 + T_max=20?** (edward #2632) — maps outer corner of LR×T_max grid.
4. **Does lr=1.75e-4 + T_max=20 beat 2e-4 + T_max=20?** (thorfinn #2664) — maps left side of LR valley at new T_max baseline. Filling gap between 1.5e-4 and 2e-4.
5. **Does clip=3.0 stack with T_max=20?** (alphonse #2663) — compounds IID gain (clip=3.0) × OOD gain (T_max=20).
6. **Does Fourier L=12 flip Finding #41 under post-LN?** (askeladd #2665) — first re-validation of input encoding lever under new foundation; especially probes rc.
7. **Does surf_weight=15 UP help rc?** (nezuko #2634) — Finding #53 mirror; rc is still the largest headroom split (58.85).
8. **Does mlp_ratio=2.5 stay under budget and help?** (tanjiro #2640) — mlp_ratio=3 was timeout-bound (+9.2% overhead); 2.5 estimated ~+4.6%.
5. **Can mlp_ratio=3 add capacity without breaking budget?** (fern #2573) — capacity-via-width vs the depth-failure mode of n_layers=6.
6. **Does slice_num=32 unlock more physics-aware OOD performance?** (edward #2577) — slice_num=24 was a pre-LN closure; post-LN's stationary residual stream may support finer groupings.
7. **Does SwiGLU's gating compound with lr=2e-4?** (nezuko #2466) — old run at test=51.59 was under-config.
8. **Does RMSNorm stack with post-LN placement + lr=2e-4?** (tanjiro #2499) — crashed last run.

## Outstanding follow-ups (not yet assigned)

- Stretch: lr=2e-4 + T_max=20 + epochs=20 (only after #2568 lands and if budget allows — currently timeout-risky at 20×102s ≈ 34 min).
- Stretch: mlp_ratio=4 (if mlp_ratio=3 wins clearly).
- Stretch: slice_num=48 (if slice_num=32 wins and per-epoch time stays under 110s).
- Stretch: grad_clip=3.0 tightening (askeladd noted clip is still 50% firing at e18).
- Stretch: surf_weight UP (sw=12, 15) under post-LN+lr=2e-4 — frieren's Finding #53 found per-split flip; mirror experiment.
- Stretch: epochs=20 under lr=2e-4 (best_epoch=18/18 is still cutoff). Compute-risky.

## IMPORTANT NOTES

- **All new/updated run commands must include `--weight_decay 0.0 --lr 2e-4 --t_max 20`** (baseline config)
- **Post-LN is now the default** in train.py (merged via #2456)
- **lr=2e-4 + T_max=20 is the current best config** (PR #2568). Config defaults are lr=1.5e-4 and t_max=18 — use `--lr 2e-4 --t_max 20` explicitly.
- **New baseline: 46.2751** — ALL comparisons use this
- lr=2.25e-4 (PR #2572, 46.8821) is slightly worse than lr=2e-4 + T_max=20 at the noise floor. The compound (lr=2.25e-4 + T_max=20, #2591) should resolve which axis wins.
- **All 8 students are occupied** — no idle GPUs
- **Pre-LN findings are not transferable wholesale** (Finding #53: per-split flips can occur under post-LN). Re-validate case-by-case.

## Lever status summary

| Lever | Status | Best value |
|---|---|---|
| Learning rate | **CLOSED** (post-LN — UPDATED) | lr=2.25e-4 (Finding #54 updated) |
| Optimizer β1 | CLOSED (same as pre-LN) | β1=0.9 (Finding #55) |
| Optimizer β2 | CLOSED — same as pre-LN | β2=0.99 (Finding #51) |
| Grad clip | **TESTING** clip=3.0 + T_max=20 stack (#2663) — #2586 tied aggregate but big per-split flip | clip=5.0 (pre-LN closure) |
| Batch/accum | CLOSED | bs=4 + accum=2 |
| Width (n_hidden) | CLOSED | n_hidden=192 |
| Depth (n_layers) | CLOSED — n=6 budget-bound (Finding #52) | n_layers=5 |
| n_head | CLOSED | n_head=4 |
| slice_num | **TESTING** (#2577) under post-LN+lr=2e-4 | slice_num=24 (pre-LN closure) |
| mlp_ratio | **TESTING** (#2573) under post-LN+lr=2e-4 | mlp_ratio=2 (pre-LN closure) |
| T_max cosine | **TESTING** T_max=22 only at new LR (#2632 lr=2.25e-4); T_max=22 falsified at lr=2e-4 (Finding #57) | T_max=20 (current best is lr=2e-4 + T_max=20) |
| weight_decay | CLOSED | wd=0 |
| Attention dropout | CLOSED | 0.0 |
| Normalization type | CLOSED | LayerNorm |
| Fourier L | **TESTING** L=12 under post-LN+lr=2e-4+T_max=20 (#2665 — Finding #41 flip re-validation) | L=8 (pre-LN closure) |
| Activation function | CLOSED (pre-LN) | GELU |
| Normalization position | CLOSED — post-LN wins | post-LN |
| EMA | CLOSED (budget-dependent) | — |
| eta_min | CLOSED | eta_min=0 |
| Meta-optimizer (Lookahead) | CLOSED — incompatible with Lion | — |
| Warmup | CLOSED — incompatible with Lion sign-update | — |
| surf_weight | CLOSED (DOWN, Finding #53 per-split flip) | surf_weight=10 |
| MLP architecture | TESTING (#2466 SwiGLU; awaiting lr=2e-4 retry) | — |
| Gradient noise (Lion) | CLOSED (Finding #49) | — |
