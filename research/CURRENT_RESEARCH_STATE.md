# SENPAI Research State

- **Date:** 2026-05-14 13:10
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 34.544** (n_layers=2 + slice_num=16 + **epochs=50**, best_epoch=47, PR #2872)
**`test_avg/mae_surf_p` = 29.916**

> **EPOCH-BUDGET WIN (Round 42).** epochs=50 (T_max=50) vs epochs=46 (T_max=46): the cosine's extended decay captured one more productive descent step at e46→e47 (slope −0.452, largest in final tail). Val −2.02%, test −1.09%. E47-50 is flat plateau [34.54-34.79] — epoch-budget axis saturated at e47 for this stack. BOTH val and test improved; confirmed above seed-variance floor.
> **Previous baseline (PR #2468):** val=35.256, test=30.245 (n_layers=2+slice_num=16+epochs=46).

> **SMOOTHING AXIS FULLY CLOSED** — SWA (#2857 + #2888) AND EMA (#2907) both refuted at this stack. Mechanism: any smoothing window over a still-descending trajectory averages over non-stationary weights and pulls toward worse iterates. SWA neutral (Δ ≤ 0.07 val); EMA at per-epoch decay=0.999 was catastrophic (+33% val) due to math/impl mismatch — coefficient of model_e30 = 0.999^20 ≈ 0.980, EMA frozen near worst-window-iterate. Both schedules (46-epoch and 50-epoch) tested.
>
> **MAJOR INSIGHT — seed variance dominates** (n=4 same-config runs): val_avg = [34.544 (#2872), 35.414, 35.697 (#2888), 34.887 (#2907)], mean 35.135, sample std 0.531. The 34.544 baseline is ~1.1σ above mean (moderately favorable seed). Possibly bimodal: 2 runs at 34.5-34.9, 2 runs at 35.4-35.7 (if real, seeds select between two basins). **Re-interprets:** (a) frieren #2883 (val 35.140) was AT the seed-mean — test wins on all 4 splits are the more robust signal; (b) future experiments need ≥3-seed runs OR ≥3-5% mechanism gains OR variance-robust mechanisms (regularization, augmentation, but NOT smoothing — that axis is closed); (c) Single-seed <2% delta comparisons are below noise floor.

> **Partition axis FULLY CLOSED.** slice_num=16 is narrow local minimum across all neighbors (12, 14, 18, 20, 24, 32). No further partition sweeping needed at n_layers=2.

> **Round 41 frontier signals — ALL HP/training axes CLOSED at this stack; pivoting to architectural and epoch-budget:**
> - **n_layers=2 is the depth-down FLOOR** (PR #2684: n_layers=1 catastrophic +12.7% loss).
> - **CAPACITY AXIS FULLY DEAD**: n_hidden (#2685 +2.53%, #2737 +7.55%), mlp_ratio (#2738 +4.35%), depth (#2684 +12.7%).
> - **SCHEDULE-SHAPE AXIS FULLY DEAD**: TAIL truncated cosine (#2760 +1.63%), HEAD warmup_epochs=3 (#2797 +2.04%). Standard cosine T_max=epochs confirmed optimal SHAPE.
> - **SCHEDULE-FLOOR AXIS DEAD FOR LION** (#2861 eta_min=5e-6 +1.09% val LOSS). Mechanism: Lion's sign-only updates have step magnitude = lr exactly (no gradient-magnitude scaling), so non-zero LR floor → late updates take full-magnitude directional steps, oscillating around basin (best_epoch shifted 46→45). Structurally tied to Lion optimizer, not a calibration issue.
> - **LOSS-WEIGHT AXIS SATURATED**: swp=15 (+0.21% val noise, −1.06% test directional), swp=20 (+4.76% val LOSS — over-pushed, broke optimization). Loss-WEIGHTING is non-monotone; weight amplification is not the path.
> - **LOSS-FORM AXIS CLOSED**: Huber d=5.0 (#2822 +116% catastrophic, raw-scale miscalibration → MSE everywhere), Huber d=0.1 (#2847 +5.54% mild regression, calibrated correctly in normalized space, no divergence). L1 is locally optimal for this stack. Both Huber endpoints tested → axis fully closed.
> - **OPTIMIZER AXIS CLOSED (at 30-min budget)**: AdamW lr=3e-4 (#2824 +29.6% val UNDER-CONVERGED at epoch 46 still descending), AdamW lr=1e-3 (#2850 +39.1% val UNDER-CONVERGED, cut at ep37, WORSE than lr=3e-4). Two structural disadvantages: ~10% per-epoch overhead (9 fewer epochs in budget) + step-magnitude mismatch (no lr scale replicates Lion's sign-update directional bias). Lion + L1 + cosine confirmed optimal for this regime.
> - **POST-HOC WEIGHT-AVERAGING AXIS CLOSED (for 46-epoch schedule)**: SWA swa_start=30 (#2857 SWA +4.16% val, best-epoch +2.90% — SWA strictly worse than best-epoch on every val/test split). Mechanism: trajectory still descending in averaging window (val 42.22 → 36.28 across SWA epochs 31-46 = 14% relative descent), so SWA averages over non-stationary weights pulling backward toward earlier worse-performing iterates. \"Still descending at end\" ≠ \"flat region.\" SWA fails when weights are still moving directionally. Variants requiring epochs>46 don't fit the 30-min cap.
> - **OVERFIT-OOD signature** (in #2738): bottleneck for camber-OOD is NOT capacity.
> - **Round 41 cont. — fresh-axis pivots (active)**:
>   - **AUX-HEAD AXIS CLOSED** — frieren #2871 aux surface head at weight=1.0, hidden=64: +4.73% val (36.925 vs 35.256), REFUTED. Aux head DID learn surface structure (final aux_surf_loss tracks main surf_loss) but did NOT regularize the shared encoder. Root cause: surf_weight=10 already provides 10× gradient signal in the same direction — the aux head was redundant. Adding any signal that duplicates the existing loss direction will fail by the same mechanism.
>   - **SPECIALIZED DECODERS AXIS FULLY CLOSED (#2883 + #2904)**: frieren #2883 at T_max=46 produced val 35.140 (within seed noise of true mean 35.2) with test wins on all 4 splits (29.594, −1.08%); cos(surf,vol)=0.0396 orthogonal. #2904 (compound with epochs=50) refuted the orthogonality assumption: val=39.731 (+15.02% LOSS). **Mechanism interaction via LR schedule** (frieren's diagnosis): specialized decoders need LR fully annealed by best_epoch — they don't tolerate the higher residual LR of T_max=50. At T_max=46 e46, LR is fully annealed and decoders settle into basin (35.14); at T_max=50 e46, LR is still ~2-3% of initial and decoders fail to converge (39.85). The #2883 test wins are schedule-specific, not robust. Closed under any compound.
>   - **EPOCH-BUDGET WIN (#2872 MERGED)** — askeladd epochs=50: val=34.544 (−2.02% vs 35.256), test=29.916 (−1.09%). Best epoch=47, plateau at e47-50 confirmed. The persistent 'still-descending-at-46' signal was genuine. **New baseline: val=34.544, test=29.916.**
>   - **SWA REOPENED at epochs=50** — SWA previously closed because trajectory was still descending into e46 (averaging over non-stationary weights). With epochs=50 and plateau confirmed at e47-50 (3 flat epochs), the standard SWA premise (flat loss landscape → ensemble gains) is now satisfied. Assigning askeladd SWA from e47 as compound on new baseline.
>   - **Future levers if decoder split + SWA stagnate**: physics-informed loss (divergence/curl), seed-averaged baseline confirmation, complete model replacement, quantile loss, focal MAE, camber-aware sample reweighting (target geom-OOD boundaries).
> - **NEW NEGATIVE RESULTS this round**:
>   - MSE-on-everything is decisively worse than L1 in normalized space (#2822 Huber d=5.0 → +116% val). Confirms L1 is the right magnitude shape.
>   - AdamW under-converges at all tested lr scales (×3 and ×10) under 30-min budget. Lion's sign-update advantage is structural at this stack.
>   - Properly-calibrated Huber d=0.1 still net worse than L1 (#2847 → +5.54% val). Lion's sign-only updates make near-zero gradient magnitude immaterial.
>   - SWA fails on still-descending trajectories regardless of swa_start (#2857). Mathematical issue with averaging non-stationary weights.
>   - Schedule-FLOOR eta_min>0 fails for Lion (#2861) due to sign-update oscillation. Lion structurally requires LR→0 endpoint.
> - **All single-axis HP sweeps CLOSED at n_layers=2 stack**: LR, WD, surface_weight, n_head, depth, slice_num, all 3 capacity axes, schedule shape (tail+head), schedule FLOOR, loss-form shape, optimizer, post-hoc averaging.
> - **#2638 split-dependent OOD diagnostic**: geom_camber is INFORMATION-limited or REPRESENTATION-limited — needs mechanism change, not weight tuning.
> - **Key OOD ceiling**: geom_camber_rc (~48 val, ~44 test) dominates val_avg — any future arm should explicitly target this split.
> - **Future levers if aux head + epochs=50 stagnate**: physics-informed loss (divergence/curl), data augmentation (CFD-valid symmetries — limited: AoA asymmetric for raceCar [-10,0]; cruise [-5,+6] OK), seed-averaged baseline confirmation, complete model replacement (e.g., transformer→GNN hybrid), quantile loss, focal MAE, camber-aware sample reweighting (target geom-OOD boundaries), Lion-with-different-betas perturbations.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | **35.113** | 31.646 |
| geom_camber_rc | **48.106** | 44.898 |
| geom_camber_cruise | 18.895 | 15.049 |
| re_rand | **36.060** | 28.072 |
| **avg** | **34.544** | **29.916** |

**Reproduce:** `cd target/ && python train.py --epochs 50 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 2 --slice_num 16`

**~35.2s/epoch × 50 epochs = 29.3 min total. 361K params. Peak memory 13.49 GB. Best epoch=47.**

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769)
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **surf_weight=5**: −9.0% val / −9.8% test (PR #1836)
6. **T_max=12 (cosine aligned to epoch budget)**: −7.9% val / −8.9% test (PR #1793)
7. **mlp_ratio=4**: −5% (PR #1408)
8. **RMSNorm**: −2.9% val / −5.9% test (PR #1837) ← geom_camber_rc −17.2%
9. **bf16 mixed precision**: −0.34% (PR #1724)
10. **T_max=12 + surf_weight=5 compound**: −3.33% val (PR #1956)
11. **n_layers=5 + T_max=14**: −6.98% val / −6.98% test (PR #1995)
12. **slice_num=48 + T_max=15**: −1.33% val / −1.10% test (PR #1996)
13. **n_layers=4 + T_max=17**: −1.07% val / −2.17% test (PR #2080) ← lr=cfg.lr bug fixed
14. **slice_num=32 + T_max=21**: −7.6% val / −7.6% test (PR #2108)
15. **n_head=2 (head_dim=32→64)**: −0.25% val (PR #2149) ← plumbed --n_head CLI arg; n_head=4 is still the train.py default
16. **epochs=24 (3 extra cosine epochs)**: −6.21% val / −5.41% test (PR #2172) ← n_head=4 default at run time
17. **n_layers=3 + slice_num=32 + epochs=27**: −8.58% val / −9.02% test (PR #2107) ← BIGGEST SINGLE STEP; best_epoch=27 STILL DESCENDING
18. **epochs=30 on n_layers=3+slice_num=32**: −2.20% val (PR #2228) ← best_epoch=30 STILL DESCENDING
19. **slice_num=24 + epochs=33 on n_layers=3**: −2.36% val / −3.38% test (PR #2229) ← ~54s/epoch, 33 epochs, STILL DESCENDING
20. **slice_num=12 + epochs=36 on n_layers=3**: −3.74% val / −3.53% test (PR #2351) ← ~50s/epoch, 36 epochs, HIT CAP STILL DESCENDING
21. **slice_num=16 + epochs=36 on n_layers=3**: −1.17% val (PR #2348) ← partition non-monotone; slice_num=16 beats 12
22. **n_layers=2 + slice_num=16 + epochs=46**: −0.82% val / −0.33% test (PR #2468) ← depth-down+epoch-up; ~35s/epoch×46 epochs; OOD gains offset single_in_dist regression

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10**
- **n_layers=3** (PR #2107) ← CLI `--n_layers 3` (default is 5)
- **slice_num=24** (PR #2229) ← CLI `--slice_num 24` (default is 48)
- **n_head=4** — this is the actual train.py default (line 392). PR #2149 plumbed CLI arg but did NOT change the default.
- **epochs=33** (PR #2229) ← CLI `--epochs 33`
- **mlp_ratio=4, GeGLU activation** (PR #1769) — hardcoded in model_config at line 435
- **RMSNorm** (PR #1837)
- n_hidden=128
- Lion optimizer, lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=33** (auto-aligns: `T_max=MAX_EPOCHS=cfg.epochs`)
- bf16 mixed precision
- **~514K params** at n_layers=3+slice_num=24
- **~53.7s/epoch**, 33 epochs = 29.5 min total

## The epoch-count mechanism: trajectory so far

"Make epochs faster → fit more epochs → align T_max → better convergence":

| Step | Change | Per-epoch | Epochs | val Δ |
|---|---|---|---|---|
| PR #1995 | n_layers=6→5 | 116s→94s | 14 | −6.98% |
| PR #1996 | slice_num=64→48 | 94s→~100s | 15 | −1.33% |
| PR #2080 | n_layers=5→4 | ~94s | 17 | −1.07% |
| PR #2108 | slice_num=48→32 | 94s→74s | 21 | **−7.6%** |
| PR #2172 | epochs=21→24 | ~74s | 24 | **−6.21%** |
| PR #2107 | n_layers=4→3 + compound | 74s→57s | 27 | **−8.58%** |

**Gains are NOT diminishing.** The n_layers=3 step was the biggest single gain yet. best_epoch=final has held for 8+ consecutive experiments. The model is always budget-limited, never capacity-saturated.

**Current per-epoch timing at n_layers=3+slice_num=32: ~57s → 30 epochs = 28.5 min (fits in 30-min cap)**

## Systemic pod issue (2026-05-14 03:15)

**6 of 8 student pods rate-limit stuck.** Bot user 20516801 has exhausted GitHub GraphQL budget. Pods affected: alphonse, edward, nezuko, thorfinn (Round 39 followup just created — PRs #2744-2747), fern, tanjiro (Round 38 followup #2695, #2696). Only **askeladd (#2738) and frieren (#2737) are actually running training**.

**Strategic decision: DO NOT close/recreate stale_wip PRs.** Closing and recreating consumes the same shared bot-account budget — that's what's making the problem worse round over round. The Round 38 and Round 39 followup hypotheses are all still valid; we just need the rate limit to clear. Estimated reset window: ~36-60 min cycles.

**Round 39 capacity tests (askeladd #2738, frieren #2737) are the critical experiments** — disambiguating capacity vs information limit. The stuck pods are running lower-EV HP-axis fine probes anyway.

## Active experiments (Round 38 — architectural pivot; 8 PRs in flight)

**Current Baseline: val=35.256 (PR #2468 n_layers=2+epochs=46), test=30.245**

| Student | PR | Hypothesis | Type |
|---------|-----|------------|------|
| fern | **#2695** | **batch_size=2** retry × n_layers=2+slice_num=16+epochs=46 (Round 37 #2636 stale_wip closed) | bs axis recreate |
| tanjiro | **#2696** | **batch_size=8** retry × n_layers=2+slice_num=16+epochs=46 (Round 37 #2637 stale_wip closed) | bs axis recreate |
| alphonse | **#2744** | **lr=8e-5** (3rd attempt; #2608 and #2680 stale_wip'd) | LR axis |
| edward | **#2745** | **slice_num=24+epochs=33** (3rd attempt) | slice axis |
| nezuko | **#2746** | **mlp_ratio=2** (3rd attempt) | mlp_ratio axis |
| thorfinn | **#2747** | **lr=7e-5** PIVOT from lr=5e-5 (3 stale_wip attempts; collecting new axis data) | LR axis (pivot) |
| askeladd | **#2921** | **Input-embedding noise σ=0.05** — data-side regularization probe, mechanism-distinct from frieren's dropout (activation-side). Both target seed-variance/overfitting from different angles. | **Round 42: data-side regularization** |
| frieren | **#2917** | **Dropout p=0.1 on attention + FFN** — canonical transformer regularization (missing from this stack). Attacks seed-variance/overfitting hypothesis from #2888. Mechanism-distinct from all prior attempts. | **Round 42: activation-side regularization** |

**Closed Round 40**:
- #2737 frieren ISO-EPOCH capacity test (n_hidden=160+slice_num=12+epochs=46): +7.55% val LOSS — only 37/46 epochs completed
- #2738 askeladd FFN capacity (mlp_ratio=6+epochs=40): +4.35% val LOSS — OVERFIT-OOD signature (single_in_dist improved, all OOD regressed)
- #2760 askeladd truncated cosine (T_max=60+epochs=46): +1.63% val LOSS — schedule-TAIL hypothesis refuted, polish phase matters
- **All 3 capacity-via-param-count axes REFUTED**: n_hidden (x2), mlp_ratio (x1, x2 across rounds), n_layers (x1)
- **Scheduler TAIL hypothesis REFUTED** (#2760)

**Round 40 cont. result — #2755 frieren swp=15/swuv=10 (sent back 2026-05-14 04:55)**:
- val_avg/mae_surf_p = 35.330 (+0.21% vs baseline 35.256, within noise floor)
- test_avg/mae_surf_p = 29.923 (−1.06% vs baseline 30.245 — DIRECTIONALLY POSITIVE on test, 3/4 splits improved)
- Decision: NOT merged (val primary metric did not beat baseline). Direction confirmed — 1.5× pressure ratio insufficient, sent back for 2× ratio (swp=20/swuv=10).

**Round 38 strategy: ARCHITECTURAL PIVOT after HP axes closed.**

**Closed HP axes at n_layers=2 stack (Rounds 32-37):** LR, WD, surf_weight, n_head, epochs all closed at or near baseline.

**Two bold experiments target architectural levers:**
1. **askeladd n_layers=1**: Continues the proven depth-down + epoch-up trajectory. n_layers 6→5→4→3→2 has been monotonically beneficial. n_layers=1 with epochs=60 tests the extreme — does the mechanism still hold?
2. **frieren n_hidden=160**: Targets the geom_camber OOD capacity bottleneck identified in #2638's split-dependent OOD diagnostic. n_hidden has been fixed at 128 throughout the project — width axis is genuinely unexplored. Requires student to add `--n_hidden` CLI flag.

**Diagnostic from Round 37 (#2638 — KEY INSIGHT):**
- **re_rand OOD is regularization-friendly** (WD=3e-4 improved it -3.97% val)
- **geom_camber OOD is CAPACITY-LIMITED** (WD=3e-4 hurt it +5.9%/+7.3% val on rc/cruise)
- Two different OOD bottlenecks require different treatments. Future architectural moves (aux surface head, per-channel surf_weight, physics-informed features) should target geom_camber specifically.

**LR axis sweep at n_layers=2 stack (3 in-flight + 1 done):** 5e-5 (thorfinn) → 1e-4 (BASELINE 35.256) → 1.2e-4 (alphonse) → **1.5e-4 (#2525 CLOSED, +3.30% LOSS — OOD bottleneck confirmed)**

**Key insight from #2525 result (in-dist vs OOD tradeoff):** Higher LR fixed n_layers=2 in-dist regression (−6.35%) but worsened all 3 OOD splits (+3.4% to +10.5%). The OOD splits dominate the avg metric. Next experiments need to target OOD generalization (wider heads, regularization, architecture priors) — NOT optimization aggressiveness.

**Critical insight from #2523 result (seed variance):** Run-to-run variance is ~±1.0 val units (the same config produced epoch-46 vals of 35.26 vs 36.42 across two runs). Single-seed comparisons of small tweaks (<5% change) are below the noise floor. **Strategy shift: prioritize bigger experimental swings (compounds, 20%+ axis changes) over marginal tuning.**

**Merged:** #2348 (val=35.548), #2468 (val=35.256 NEW BEST)
**Closed Round 39:** #2684 (askeladd n_layers=1 **+12.7% LOSS** — MAJOR FINDING: n_layers=2 is depth-down floor); #2685 (frieren n_hidden=160 **+2.53% loss** — capacity/epoch confound; iso-epoch retest assigned)
**Closed Round 38:** #2638 (frieren wd=3e-4 alone **+1.43% loss** — KEY diagnostic: split-dependent OOD finding); #2639 (askeladd wd=5e-5 **+2.07% within noise** — WD axis fully closed); #2608/2609/2610/2611 (4 Round 36 stale_wip, all recreated in Round 38)
**Closed Round 37:** #2600 (frieren sw=12 **+3.96% LOSS** — sw axis at this stack now closed: optimum bracketed tightly around sw=10); #2601 (askeladd compound lr=1.5e-4+wd=3e-4 **+2.89% LOSS** — WD partially rescues OOD but erases in-dist gain; LR/WD sweep space exhausted at this stack); #2570 (fern sw=8 stale_wip); #2571 (tanjiro mlp_ratio=3 stale_wip)
**Closed Round 36:** #2543 (alphonse stale_wip), #2545 (edward stale_wip), #2547 (nezuko stale_wip), #2549 (thorfinn stale_wip)
**Closed Round 35:** #2523 (frieren epochs=50 +2.30% loss — seed variance insight); #2558 (askeladd n_head=2 +3.16% loss — every split regressed)
**Closed Round 34:** #2492 (fern stale_wip old-stack), #2493 (tanjiro stale_wip old-stack)
**Closed Round 33:** #2525 (askeladd lr=1.5e-4 +3.30% loss — informative on in-dist/OOD tradeoff)
**Closed Round 32:** #2471 (alphonse stale old-stack), #2478 (edward stale old-stack), #2479 (nezuko stale old-stack), #2450 (thorfinn stale old-stack)
**Closed earlier:** #2451 (askeladd slice18 +4.3%), #2447, #2404, #2431, #2402, #2417, #2375, #2383, #2409, #2408

**Complete baseline trajectory:** 40.158 → 39.143 → 38.270 → 37.366 → 35.969 → 35.548 → **35.256** (−12.2% total from round start)

**Variance note (from #2274 student diagnosis):** Inter-run variance on identical configs is ~1.7 val units. Recent ~0.5 val improvements are at or near this noise floor — single-run signals require corroboration.

**Partition sweep ladder (completed + in-flight):**
- slice_num=32 → val=39.143 (PR #2107)
- slice_num=24 → val=37.366 (PR #2229)
- slice_num=20 → val=36.854 (PR #2375, beats old baseline, loses vs new)
- slice_num=18 → askeladd #2451 (IN FLIGHT — filling 20→16 gap)
- **slice_num=16 → val=35.548 (PR #2348, CURRENT BASELINE)**
- slice_num=14 → edward #2447 (IN FLIGHT — lower-side probe)
- slice_num=12 → val=35.969 (PR #2351, WORSE than 16 — non-monotone)
- slice_num=8 → tanjiro #2408 (IN FLIGHT — expected capacity collapse)
- **PARTITION AXIS FULLY CLOSED. Robust local minimum at 16. Both neighbors (14, 12) are worse.**
Remaining in-flight: askeladd slice_num=18 (informative, expected to confirm monotone above 16), tanjiro slice_num=8 (expected capacity collapse).

## Confirmed mechanisms

**Epoch-budget (dominant lever):** Faster per-epoch → more epochs → better convergence. **13+ consecutive best_epoch=final wins.** Partition sweep: 32→24→12 each a large win, but 12 regressed vs 16 (non-monotone). Mechanism shows NO sign of saturating at 16.

**Dead at n_layers=3:**
- mlp_ratio=6 (+5.4%) — attention is the bottleneck, not FFN capacity
- Linear warmup (+1.66%) — compresses cosine tail
- sw=5/vol-gradient (+2.57%) — depth-sensitive, doesn't compound at compact stack

## Next priority hypotheses

**Highest EV (immediate):**
1. **Compound lr=1.5e-4 × slice_num=16**: alphonse #2431 IN FLIGHT — highest EV
2. **n_head=2/1 at slice_num=16**: once edward/thorfinn/nezuko n_head results land, rerun winner at slice_num=16
3. **lr=5e-5 at slice_num=16**: once frieren's lr=5e-5 at slice_num=24 lands, confirm whether it signals
4. **Triple compound lr × n_head × slice_num=16**: after independent axes confirmed

**Medium priority (after stale-stack tests land):**
5. **slice_num=20 result (askeladd #2375)**: informative for 16–24 neighborhood
6. **slice_num=8 result (tanjiro #2408)**: confirms capacity floor at 16

**Partition sweep state (FULLY CLOSED):**
- slice_num=32: val=39.143
- slice_num=24: val=37.366
- slice_num=20: val=36.854 (PR #2375, worse than 16)
- slice_num=18: askeladd #2451 (in flight — expected slightly worse)
- **slice_num=16: val=35.548 (CURRENT BASELINE, PR #2348) ← ROBUST OPTIMUM**
- slice_num=14: val=36.483 (PR #2447, +2.6% LOSS)
- slice_num=12: val=35.969 (PR #2351, WORSE than 16 — non-monotone)
- slice_num=8: CLOSED (PR #2408, stale — capacity collapse expected)

**LR axis state at slice_num=16 (ACTIVE):**
- lr=5e-5 (thorfinn #2450, IN FLIGHT — expected undertrained)
- lr=8e-5 (UNTESTED)
- lr=1e-4 (BASELINE, val=35.548)
- lr=1.2e-4 (alphonse #2471, IN FLIGHT — fine-grained probe above trough)
- lr=1.5e-4 (#2431 CLOSED, +7.3% val — partition-dependent LR ceiling confirmed)
- lr=2e-4 (closed at slice_num=24, +4.4%)

**KEY INSIGHT: LR optimum shifts down with slice_num:**
- slice_num=24: lr=1.5e-4 beats 1e-4 (positive signal from #2353)
- slice_num=16: lr=1e-4 beats 1.5e-4 (this confirmation from #2431)
- Hypothesis: smaller partition → tighter per-slice budget → more conservative LR needed

**LR axis at slice_num=24 FULLY BRACKETED (closed):**
- lr=5e-5: +5.1% (frieren #2402) — cosine decays before convergence
- lr=1e-4: TROUGH
- lr=2e-4: +4.4% (#2367)

**LR axis state at slice_num=12 (CLOSED — informative only):**
- lr=1e-4: val=35.969 (PR #2351)
- lr=1.5e-4: val=35.688 (PR #2409 CLOSED — beats slice12 baseline but loses vs 35.548)

**surf_weight axis at slice_num=16 (ACTIVE — previously only tested at slice_num=24):**
- sw=8: fern #2492 (IN FLIGHT)
- sw=10: val=35.548 (BASELINE)
- sw tested at old stacks: sw=2 lost, sw=5 lost, sw=12/16 lost — ALL at slice_num=24

**weight_decay axis at slice_num=16 (ACTIVE — previously only tested at slice_num=24):**
- wd=5e-5: tanjiro #2493 (IN FLIGHT)
- wd=1e-4: val=35.548 (BASELINE)
- wd=0 at slice_num=24 lost +2.2% (PR #2274)

**Research frontier ideas:**
- PINN-style auxiliary loss (divergence/curl regularization) — physics-informed volume constraint
- PhysicsAttention capacity floor: confirmed at slice_num=16 (both neighbors 14, 12 worse)

## Dead ends

- **Vol-gradient mechanism (surf_weight<10) at n_layers=3**: sw=5 regressed +2.57% vs baseline (PR #2245); mechanism is depth-sensitive and does not compound at compact stack. sw=2/sw=3 still in flight (askeladd #2248 / nezuko #2279).
- **LR axis fully saturated at 1e-4** on n_layers=4: lr=1.5e-4 neutral on n_layers=4+slice_num=32, lr=8e-5 regressed +12.4% vs old baseline. **Retesting lr=1.5e-4 at n_layers=3** (fern #2301 in flight).
- **surf_weight=15**: neutral on n_layers=4 — sw=10 near optimum in high direction
- **n_head=8**: +43% per-epoch, +15.7% worse
- **n_layers=7**: +4.6% (epoch budget dominates; 160s/epoch too slow)
- **mlp_ratio axis at n_layers=3 CLOSED**: mlp_ratio=6 (+5.4%, PR #2278) AND mlp_ratio=2 (+2.3%, PR #2350) both lose. mlp_ratio=4 confirmed optimal at compact stack. FFN width is load-bearing (can't cut) but not the limiting factor (can't expand). The mlp_ratio flag was hardcoded; student added CLI arg `--mlp_ratio`.
- **mlp_ratio axis fully closed**: mlp_ratio=2 +9.95% (old stack), +2.3% (PR #2350 new stack); mlp_ratio=6 +5.4% (PR #2278); mlp_ratio=8 +5.95%. mlp_ratio=4 confirmed optimal across all stacks.
- **Linear epoch warmup (2 ep)**: +1.66% val worse (PR #2273) — compresses cosine T_max by 2 epochs, hurting the high-value late-stage descent; no benefit to Lion+L1 at lr=1e-4
- **mlp_ratio=6 at n_layers=3**: +5.4% val worse (PR #2278) — FFN capacity is not the bottleneck at depth 3
- **weight_decay=0 at compact stack**: +2.2% val worse vs new baseline (PR #2274), BUT confirmed compact stack does NOT overfit at WD=0 → WD axis is effectively FLAT. Future experiments can safely use WD=0 if convenient; not a winning lever.
- **surf_weight axis at n_layers=3 FULLY CLOSED**: sw=2 (+3.13%), sw=3 (+3.87%), sw=5 (+2.57%) all lose vs new baseline. Vol-gradient mechanism confirmed active at all three (mae_vol_p −10 to −19%) but vol→surface coupling too weak to compensate. sw=10 remains optimal at compact stack.
- **lr=2e-4 at compact stack**: +4.4% val worse (#2367). Geometry-OOD splits hurt most. LR ceiling below 2e-4 confirmed.
- **grad-clip**: worse for Lion (sign-update already handles magnitude)
- **DropPath**: needs 100-300 epoch budgets; useless at 20-30 epoch budgets
- **Dropout**: always worse (model is underfitting)
- **n_head axis FULLY CLOSED across {1, 2, 4} at slice_num=24:**
  - n_head=1: val=37.64 (nezuko #2404, 3-seed mean ±0.48). +28% more params — no gain.
  - n_head=2: val=37.63 (edward #2383). +6% more params — slight loss.
  - n_head=4: val=37.37 (BEST). Parallelism wins.
  - **Bottleneck is slice mechanism, not Q/K/V capacity.** dim_head doesn't matter — routing diversity does.
- **n_head=2 on n_layers=4**: marginal win (+0.25%), but current best is n_layers=3+n_head=4 anyway
- See full dead-ends list in older entries above for complete history

## Key constraints

- 30 min / run cap: ~57s/epoch at n_layers=3+slice_num=32 → max ~31 epochs
- slice_num=16: ~49.8s/epoch → 36 epochs = 29.9 min (at cap)
- slice_num=12: ~50.3s/epoch → 36 epochs = 30.2 min (at cap, similar to 16)
- slice_num=24 estimate: ~50s/epoch → max ~33 epochs
- n_head, mlp_ratio do not dramatically change per-epoch time at current config
- Batch increase: step-count limited (always worse)
- EMA: cold-start drag, incompatible with short budgets

## Infrastructure notes

- **lr=cfg.lr bug FIXED** (PR #2080): Lion uses `cfg.lr`; `--lr` flag works correctly.
- **slice_num CLI arg** (PR #2108): `--slice_num` accepted in Config.
- **n_head CLI arg** (PR #2149): `--n_head` accepted in Config. **Default is still 4** (train.py line 392: `n_head: int = 4`).
- **mlp_ratio CLI arg added** (PR #2350): `--mlp_ratio` accepted via Config. Default is still 4 (optimal, axis now closed).
