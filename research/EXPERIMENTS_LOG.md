# SENPAI Research Results

## 2026-05-12 20:11 — PR #1359: LR warmup + 1e-3 peak: cosine aligned to 30-min budget
- Branch: willowpai2g48h1-alphonse/lr-warmup-1e-3
- Hypothesis: 2-epoch linear warmup (lr=1e-5 → 1e-3) then cosine decay over remaining 48 epochs. Peak LR = 2× default 5e-4.
- W&B runs: `o0s1z3aq` (trial-1 reported), `qua872ss`, `x9ntld98` (variance check)

| Metric | trial-1 | mean 3 runs | std |
|---|---|---|---|
| val_avg/mae_surf_p (best) | **138.85** @ ep 12 | 144.06 | 4.05 |
| test_avg/mae_surf_p | NaN (cruise GT bug) | — | — |
| test 3-split partial | 139.20 | — | — |
| Epochs completed | 13 (timeout) | — | — |
| Peak GPU mem | 42.1 GB | — | — |

**Analysis**: LR=1e-3 is at the edge of stability (epoch-7 spike: val jumped 184 → 214, recovered). High run-to-run variance (std=4.05 ≈ 3% of mean) from undertrained regime. Student proactively ran 3 trials for variance characterization — valuable. Test NaN from the known data bug (PR not rebased onto fix). val=138.85 on default batch=4 (no bf16) is comparable to or slightly worse than bf16 baseline val=133.75, but on a different config — not directly comparable. Warmup schedule hypothesis itself is still untested vs the new baseline.

**Action**: Sent back to rebase onto bf16+batch-8 baseline and retest lr=1e-3 warmup on top. New baseline gives ~17 epoch headroom, making cosine decay more meaningful. Target: test_avg < 121.28.

## 2026-05-12 19:00 — PR #1361: Wider model: n_hidden 128→192 for more flow-field capacity
- Branch: willowpai2g48h1-askeladd/wider-hidden-192
- Hypothesis: Increase Transolver hidden width from 128 → 192 (n_head=4, dim_head=48) for more flow-field capacity. Predicted -3% to -7% on val_avg/mae_surf_p.
- W&B run: `86cbe3io` (trial-1)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | **140.728** @ epoch 9 |
| test_avg/mae_surf_p | **NaN** (cruise pred inf) |
| test partial avg (3 finite splits) | 139.36 |
| Params | 1.47M |
| Peak GPU memory | 58.0 GB |
| Epochs completed | 10 (timeout, cosine T_max=50 → undertrained) |
| Avg epoch time | ~186 s |

Per-split val mae_surf_p @ best: in-dist 161.6, geom_rc 149.6, geom_cruise 120.1, re_rand 131.6.
Per-split test mae_surf_p: in-dist 144.1, geom_rc 141.2, geom_cruise **NaN**, re_rand 132.8.

**Analysis**: The wider model trains stably and produces a finite val signal (140.73), but one or more test_geom_camber_cruise samples drive a node prediction to inf in physical pressure space, poisoning the float64 accumulator (accumulate_batch skips non-finite ground truth but not non-finite predictions). Underlying cause is likely undertraining (cosine T_max=50 means LR ≈ 4.5e-4 even at epoch 10) interacting with an OOD camber-cruise extreme. Validation across the matching val split was finite throughout (120.06 at best), so the model isn't globally diverging — this is an OOD edge case.

**Action**: Sent back with a targeted clipping fix inside `evaluate_split` (clip pred to ±50 stds in normalized space) to guarantee a finite test_avg for direct cohort comparison without distorting normal predictions. Rerun expected to land same-magnitude val with a clean test number.

**Trial-2 update (W&B `np5qnkp2`)**: clamp didn't catch NaN (NaN comparisons return False; `torch.clamp(NaN, ...)` returns NaN unchanged). Trial-2 still NaN test_avg, val_avg=148.37. Student proposed `nan_to_num + clamp` fix (handles all three: NaN, ±inf, finite extremes) + updated `n_clipped` counter to include non-finite. Approved; trial-3 pending.

## 2026-05-12 19:58 — PR #1387: Fourier pos encoding: high-freq spatial features for pressure
- Branch: willowpai2g48h1-nezuko/fourier-pos-features
- Hypothesis: Add NeRF-style Fourier features over (x,z) (L=8 frequencies → 34 pos dims) so the preprocess MLP gets direct high-frequency spatial information. Predicted -5% to -10%.
- W&B runs: `twpifp5a` (trial-1), `111nh26k` (trial-2 variance check)

| Metric | trial-1 | trial-2 |
|---|---|---|
| val_avg/mae_surf_p (best) | **119.70** @ ep 12 | 132.96 @ ep 11 |
| test_avg/mae_surf_p | NaN (cruise inf) | NaN (cruise inf) |
| test single_in_dist | 117.91 | 152.54 |
| test geom_camber_rc | 122.03 | 122.82 |
| test geom_camber_cruise | NaN | NaN |
| test re_rand | 111.74 | 120.77 |
| Epochs completed | 14 | 14 |
| Peak GPU memory | 42.5 GB | 42.5 GB |
| Params | ~0.94M | ~0.94M |

**Analysis**: Strong val signal — trial-1 119.70 is the best round-1 number so far (vs askeladd wider-192 at 140.73). Run-to-run variance is high (~11% between trials) consistent with early-training noise (cosine T_max=50 means LR still at ~96% of peak at epoch 12). The model is clearly still descending (val_avg drops from 225.9 → 119.7 in 12 epochs), so this hypothesis is undertrained but already strongest. Both trials hit the same cruise inf, suggesting it's structural to the Fourier-features model on extreme OOD cruise cambers, not a random fluke. Ux/Uy channels remain finite, so the inf is localized to predicted pressure on a few nodes.

**Action**: Sent back with the same `nan_to_num + clamp` fix as #1361 to get a finite cohort-comparable test_avg. Schedule change (T_max=actual epochs) deferred to round 2 — it's a confound vs other round-1 PRs and is being separately tested by alphonse's `lr-warmup-1e-3` PR.

**Correction (2026-05-12 20:10)**: nan_to_num advice was wrong diagnosis — the NaN was caused by corrupted GT y (not pred). Sent back to rebase onto new baseline and retest Fourier with only the Fourier feature changes.

---

## 2026-05-12 20:05 — PR #1362: More physics slices: slice_num 64→128 for richer flow tokens
- Branch: willowpai2g48h1-edward/more-slices-128
- Hypothesis: Double slice_num 64→128 (more physics tokens per head). Predicted -2% to -6%.
- W&B run: `4ka7n8pi` (trial-1, includes scoring bug workaround)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | **142.42** @ epoch 9 |
| test_avg/mae_surf_p | **129.60** (with bug workaround) |
| test single_in_dist | 152.02 | test geom_camber_rc | 141.54 |
| test geom_camber_cruise | 97.40 (199/200 samples) | test re_rand | 127.44 |
| Epochs completed | 11 (timeout) | Epoch time | ~172 s |
| Peak GPU mem | 54.5 GB | Params | 0.67M |

**Analysis**: Edward independently found and fixed the scoring bug (identical root cause as tanjiro). val_avg=142.42 and test_avg=129.60 are both worse than tanjiro's baseline (133.75/121.28), so `slice_num=128` alone is not a win. However the slice doubling costs near-zero compute (+0% epoch time) so it's worth testing as an additive change on top of the bf16 baseline.

**Action**: Sent back to rebase onto new bf16+batch-8 baseline and retest `slice_num=128` addition. Target: test_avg < 121.28.

**Trial-2 result (W&B `tfmvmowl`, rebased on PR #1391)**: val=166.66 (+24.6%), test=155.15 (+27.9%). **Dead end — PR CLOSED.** Root cause: `slice_num=128` at `batch_size=8` quadruples the attention map `[B=8, H=4, N≈150k, slice_num=128]`, pushing peak memory to 94.3 GB (vs 65.9 GB baseline, near the 96 GB OOM cap) and epoch time up +44% to 154.5s — only 12 epochs complete vs 17 for baseline. Model under-trains severely, with a sharp regression at ep 12 (167→202, possible bf16 instability in larger attention maps). The trial-1 "zero overhead" finding was batch=4+fp32; at batch=8+bf16 the cross-attention map is 4× larger. Direction ruled out under the 30-min budget at this batch size.

## 2026-05-12 20:05 — PR #1391: BF16 + batch 8: more epochs within 30-min cap via AMP ⭐ MERGED
- Branch: willowpai2g48h1-tanjiro/bf16-batch-8
- Hypothesis: bf16 autocast + batch_size=8 + lr=7e-4 (√2 scaling) → more epochs in 30-min budget. Predicted -3% to -7%.
- W&B run: `s8kl6dza` (trial-1)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | **133.7491** @ epoch 17 |
| **test_avg/mae_surf_p** | **121.2830** ← NEW BASELINE |
| test single_in_dist | 166.19 | test geom_camber_rc | 136.20 |
| test geom_camber_cruise | 78.57 | test re_rand | 104.17 |
| Epochs completed | 17 (timeout, ~107 s/epoch) |
| Peak GPU mem | 65.9 GB | Params | 0.67M |

**Analysis**: Tanjiro found the scoring bug independently, patched it in evaluate_split, and produced the strongest test_avg in the cohort (121.28 vs edward's 129.60 and partial-fourier estimates). The bf16+batch-8 config doubles effective throughput (17 epochs vs ~10-11 for others), enabling the model to reach a significantly lower loss at timeout. Also: `test_geom_camber_cruise`=78.57 is very low — BF16+higher-batch seems to particularly improve cruise OOD generalization. Val loss still descending at epoch 17 — more training would help further.

**Action**: MERGED as new round-1 baseline. Also merged critical scoring bug workaround. All future PRs must rebase and beat test_avg < 121.28.

## 2026-05-12 21:55 — PR #1591: Cosine schedule aligned to 30-min budget: --epochs 18 ⭐ WINNER (pending code-change commit)
- Branch: willowpai2g48h1-edward/cosine-aligned-epochs
- Hypothesis: Set `--epochs 18` so cosine T_max=18 fully decays within the realistic 30-min budget. Predicted -2% to -5%.
- W&B run: `h7w6skh8`

| Metric | This run | Baseline | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 15) | **125.36** | 133.75 | **−6.27%** |
| **test_avg/mae_surf_p** | **111.98** | 121.28 | **−7.67%** ⭐ |
| test single_in_dist | 148.79 | 166.19 | −10.47% |
| test geom_camber_rc | 117.15 | 136.20 | −13.99% |
| test geom_camber_cruise | 77.85 | 78.57 | −0.92% |
| test re_rand | 104.13 | 104.17 | −0.04% |
| Epochs completed | 17/18 | 17/50 | — |
| Final LR | 5.32e-6 | 6.2e-4 | 2 orders of magnitude lower |
| Peak GPU mem | 82.68 GB | 65.9 GB | +25% (extra allocator activity, fits cleanly) |

**Analysis**: The schedule-alignment hypothesis lands well above predicted range (−7.67% vs predicted −2% to −5%). The mechanism is exactly as theorized: aligning T_max to the realistic epoch budget gives the model the full cosine decay including the low-LR refinement phase (5.32e-6 vs baseline 6.2e-4 final LR). This 2-orders-of-magnitude lower LR phase is what finds a flatter optimum. Best-val shifts from epoch 17 (last) on baseline to epoch 15 (mid-refinement) on this run — concrete evidence that the LR decay finds a flatter minimum rather than just rescaling.

The wins concentrate on geometry-OOD splits (single_in_dist −10.47%, geom_camber_rc −13.99%) where weight-space noise hurts most. Cruise and re_rand barely move (~−1%, ~0%) — already at noise floor for this model size. This pattern strongly suggests the schedule fix unlocks generalization the model already had latent capacity for.

**Action**: Win confirmed via W&B `h7w6skh8`. PR has zero code diff — win achieved via `--epochs 18` CLI flag only. Sent back to commit `epochs: int = 18` to Config dataclass default so the win compounds for the whole cohort. Will merge as new baseline once committed.

## 2026-05-12 21:55 — PR #1361 trial-4: Wider model n_hidden 192 on rebased baseline
- Branch: willowpai2g48h1-askeladd/wider-hidden-192
- Hypothesis: n_hidden 128→192 on top of PR #1391 baseline.
- W&B run: `p3uzgdir` (trial-4)

| Metric | Trial-4 | Baseline | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 15) | **128.31** | 133.75 | **−4.07%** |
| **test_avg/mae_surf_p** | **115.30** | 121.28 | **−4.93%** ✓ |
| test single_in_dist | 125.91 | 166.19 | −24.24% |
| test geom_camber_rc | 123.73 | 136.20 | −9.16% |
| test geom_camber_cruise | 94.46 | 78.57 | +20.22% (worse — extra capacity overfits cruise) |
| test re_rand | 117.11 | 104.17 | +12.42% (worse) |
| Params | 1.47M | 0.67M | +2.2× |
| Batch size | 4 (OOM at 8) | 8 | — |

**Analysis**: Width hypothesis holds on the corrected baseline — wider model wins overall (−4.93%) within predicted range. **However**, the per-split signal is interesting: in_dist (−24%) and rc (−9%) improve while cruise (+20%) and re_rand (+12%) regress. This suggests width helps the splits closer to training distribution but hurts the OOD-extrapolation splits. The wider model with more capacity at the same depth may be overfitting to in-distribution structure. Worth noting for round 2.

Practical constraint: at bs=8+bf16+n_hidden=192 the model OOMs at 94GB (≥96GB cap). Forced fallback to bs=4 which halves throughput. The wider+bs=4 config still wins, so the headline result holds.

**Action**: Holding merge pending edward's #1591 landing first (bigger win). After edward merges, askeladd to rebase + retest on schedule-aligned baseline (which will have epochs=18 default). Expected: width may compound with schedule, or may plateau — that's the test.

## 2026-05-12 22:30 — PR #1578: Polyak EMA at evaluation (decay=0.999, warmup=200 steps)
- Branch: willowpai2g48h1-tanjiro/ema-eval-weights
- Hypothesis: Polyak averaging of model weights during training; evaluate with the EMA copy. Should denoise late-training oscillation and improve generalization.
- W&B run: `nh3l7psd` (trial-1 on PR #1391 baseline, not yet schedule-aligned)
- Status: **CLOSED ✗ (dead end on this config — redirected as PR #1664)**

| Metric | EMA eval | Baseline (#1391) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 148.92 | 133.75 | +11.3% (worse) |
| **test_avg/mae_surf_p** | **141.87** | 121.28 | **+17.0%** (worse) |
| Epochs completed | 17 | 17 | — |

**Analysis** (student's diagnostic was excellent): Classical Polyak EMA failed for two compounding reasons:
1. **EMA lags a still-descending model in undertrained regime.** With only 17 epochs and no schedule alignment, the model is still descending fast at termination — the EMA average sits behind the current weights, not at a flatter optimum.
2. **Random-init contamination.** With ~3000 update steps and decay=0.999, the EMA still carries ~`0.999^3000 ≈ 5%` of the random initialization — substantial pollution. Adam-style bias correction (`ema / (1 - decay^t)`) cancels this analytically.

The hypothesis is correct in principle but tested on the wrong baseline. Polyak works when late-training is in low-LR oscillation around a basin — exactly the regime that schedule-aligned baseline (PR #1591, final LR 5e-6) creates.

**Action**: Closed #1578. Reassigned tanjiro to PR #1664 — bias-corrected EMA on schedule-aligned baseline.

## 2026-05-12 22:45 — PR #1664 (NEW, assigned): Bias-corrected Polyak EMA on schedule-aligned baseline
- Branch: willowpai2g48h1-tanjiro/ema-bias-corrected
- Hypothesis: EMA failure modes from #1578 are config-specific, not fundamental. Schedule-aligned baseline gives Polyak its proper conditions (low-LR oscillation), and Adam-style bias correction (`ema / (1 - decay^t)`) cancels random-init contamination.
- Status: WIP (newly assigned)
- Target: test_avg < 111.98.

**Action**: Closed #1578. Reassigned tanjiro to PR #1664 — bias-corrected EMA on schedule-aligned baseline.

## 2026-05-12 22:45 — PR #1664 (NEW, assigned): Bias-corrected Polyak EMA on schedule-aligned baseline
- Branch: willowpai2g48h1-tanjiro/ema-bias-corrected
- Hypothesis: EMA failure modes from #1578 are config-specific, not fundamental. Schedule-aligned baseline gives Polyak its proper conditions (low-LR oscillation), and Adam-style bias correction (`ema / (1 - decay^t)`) cancels random-init contamination.
- Status: WIP (newly assigned)
- Target: test_avg < 111.98.

## 2026-05-13 00:10 — PR #1380: Higher surface weight surf_weight 10→25 (CLOSED)
- Branch: willowpai2g48h1-frieren/surf-weight-25
- Hypothesis: Reweight training objective toward surface terms to directly compress surface MAE (the primary metric).
- W&B run: `5lmlsuai` — group `surf-weight-25`
- Status: **CLOSED ✗**

| Metric | surf_weight=25 | Baseline (#1591) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 140.27 | 125.36 | +11.9% (worse) |
| **test_avg/mae_surf_p** | **123.41** | 111.98 | **+10.2%** (worse) |
| test_single_in_dist | 173.99 | 148.79 | +17% |
| test_geom_camber_rc | 126.64 | 117.15 | +8.1% |
| test_geom_camber_cruise | 80.13 | 77.85 | +2.9% |
| test_re_rand | 112.90 | 104.13 | +8.4% |

**Analysis**: Regression is uniform across all 4 splits — not a distribution-sensitivity issue, a global optimization effect. The loss is `vol_loss + surf_weight * surf_loss`; raising surf_weight from 10→25 scales the surface gradient 2.5× without rescaling lr, equivalent to a much hotter effective LR on surface-coupled parameters. Under 18-epoch cosine schedule there's no time to settle. Additionally, volume nodes are 97–99% of mesh points; de-emphasizing volume loss starves the shared encoder of the representation signal that surface decoding depends on. Training surf_loss dropped (model overfits surface targets) but val surf MAE rose — classic gradient-imbalance overfitting signature.

The student's diagnostic pointed to the asymmetric question: if 25 hurts both surf and vol, maybe we're already over-weighting at 10. Worth testing the other direction.

**Action**: Closed #1380. Assigned frieren to PR #1710 — surf_weight=5 (the other direction).

## 2026-05-13 00:10 — PR #1710 (NEW, assigned): Surface weight 10→5
- Branch: willowpai2g48h1-frieren/surf-weight-5
- Hypothesis: surf_weight=10 may over-weight surface terms. Reducing to 5 gives the shared encoder richer volume gradient signal → better upstream representations → better surface predictions downstream.
- Status: WIP (newly assigned)
- Target: test_avg < 111.98.

## 2026-05-13 01:10 — PR #1364: Deeper model n_layers 5→7 (CLOSED)
- Branch: willowpai2g48h1-fern/deeper-7-layers
- Hypothesis: More Transolver blocks → richer physics representations → better generalization, especially on OOD splits
- W&B run: `r6vksjdm` — group `deeper-7-layers`
- Status: **CLOSED ✗ — confounded by schedule misalignment, not a fair test**

| Metric | n_layers=7 | Baseline (#1591) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 12) | 146.46 | 125.36 | +16.8% (worse) |
| **test_avg/mae_surf_p** | **132.06** | 111.98 | **+17.9%** (worse) |
| test_single_in_dist | 161.05 | 148.79 | +8.2% |
| test_geom_camber_rc | 150.73 | 117.15 | +28.6% |
| test_geom_camber_cruise | 90.92 | 77.85 | +16.8% |
| test_re_rand | 125.53 | 104.13 | +20.5% |
| Epochs completed | 13/18 | 17/18 | — |
| Epoch time | 146s | ~96s | +52% |
| Peak GPU mem | 89.8 GB | 82.68 GB | — |

**Analysis**: The regression is confounded, not an indictment of depth. At 146s/epoch (52% slower than baseline), the 30-min cap allowed only 13 of 18 scheduled epochs. The cosine schedule at epoch 13/18 has final LR ≈ 1.25e-4 (18% of peak 7e-4) — the low-LR refinement phase that produced baseline's entire win never happened. The model's val curve was still actively descending at the cutoff. This is a wall-clock budget failure, not a depth failure.

Key learning: the schedule-aligned baseline (epochs=18) assumes a fixed per-epoch cost. Architecture changes that increase epoch cost (deeper, wider-at-bs8) violate this assumption and need budget-aware re-alignment.

**Action**: Closed #1364. Assigned fern to PR #1742 — n_layers=6 (~120s/epoch, fits ~15-16/18 epochs). Budget-safe depth test.

## 2026-05-13 01:10 — PR #1742 (NEW, assigned): Depth n_layers 5→6, budget-safe
- Branch: willowpai2g48h1-fern/n-layers-6
- Hypothesis: n_layers=6 is the budget-safe depth increase — ~120s/epoch, fitting ~15-16 of 18 scheduled epochs, reaching the low-LR tail unlike n_layers=7.
- Status: WIP (newly assigned)
- Target: test_avg < 111.98.

## 2026-05-13 02:00 — PR #1361 trial-5: Wider model n_hidden 192 on schedule-aligned baseline — MERGED ⭐ NEW BASELINE
- Branch: willowpai2g48h1-askeladd/wider-hidden-192
- Hypothesis: n_hidden 128→192 on top of PR #1591 schedule-aligned baseline. Width × schedule compounds.
- W&B runs (3 seeds): `jvphwc6p`, `dcfy4v1z`, `9skp8i3k` — group `wider-hidden-192`
- Status: **MERGED ✓ — new baseline test=99.69 (−10.97%)**

| Metric | Mean (n=3) | Std | Best seed | Baseline (#1591) | Δ (mean) |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p | **111.32** | 2.87 | 108.42 | 125.36 | −11.51% |
| **test_avg/mae_surf_p** | **99.69** | 3.16 | **96.19** | 111.98 | **−10.97%** |
| test_single_in_dist | 116.57 | 4.13 | 110.92 | 148.79 | −21.6% |
| test_geom_camber_rc | 108.61 | 2.54 | 105.65 | 117.15 | −7.3% |
| test_geom_camber_cruise | 74.18 | 2.83 | 71.53 | 77.85 | −4.7% |
| test_re_rand | 99.41 | 2.26 | 96.66 | 104.13 | −4.5% |
| Epochs completed | 15-16 | — | — | 17/18 | — |
| Epoch time | ~126s | — | — | ~96s | +31% |

**Analysis**: The 3-seed accidental replication (entrypoint re-invoked 3×) gives free variance characterization: std=3.16 on test (~3% of mean), worst seed still beats baseline by 8.65%. Width × schedule compounded: trial-4 (un-aligned T_max=50) gave −4.93%; trial-5 (schedule-aligned T_max=18) gives −10.97% — the schedule fix acted as a force-multiplier for capacity. All 4 splits improve, with in_dist showing the largest absolute gain (−21.6%). n_hidden=192 + bs=8 + bf16 OOMs at ~94 GB; bs=4 fallback required as structural constraint.

**Key insight**: Schedule alignment unlocks ~70% of the cumulative gain on top of width. Width alone was −4.93%; width + schedule = −10.97%. The lesson: always align T_max to realistic epoch budget first, then add capacity.

**Action**: Merged as new baseline. Assigned askeladd to PR #1771 — schedule realignment for n_hidden=192 at bs=4 budget (epochs=14 aligns cosine T_max to actual 14 eps/30min).

## 2026-05-13 02:00 — PR #1771 (NEW, assigned): Schedule realignment for n_hidden=192 (epochs=14)
- Branch: willowpai2g48h1-askeladd/wider-192-schedule-realigned
- Hypothesis: epochs=18 default was calibrated for n_hidden=128 at bs=8 (17-18 eps/30min). n_hidden=192 at bs=4 runs at ~126s/ep → 14 eps in 30min. At trial-5's cutoff (epoch 15), LR is at 4.7e-5 (6.7% of peak). Setting epochs=14 aligns T_max to actual budget → LR reaches 0 within budget. Same principle as PR #1591 (-7.67%).
- Status: WIP (newly assigned)
- Target: test_avg < 99.69.
