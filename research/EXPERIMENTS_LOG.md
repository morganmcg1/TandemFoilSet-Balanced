# SENPAI Research Results

## 2026-05-13 04:10 — PR #1359: LR warmup + 1e-3 peak — REDIRECTED to lr=3e-4 for Lion
- Branch: willowpai2g48h1-alphonse/lr-warmup-1e-3
- Original: 2-epoch linear warmup + peak lr=1e-3 vs AdamW baseline. After multiple rebase cycles (through bf16, wider-192, Fourier, Lion merges), the original lr=1e-3 became 6.7× the Lion baseline (1.5e-4) — almost certain to diverge.
- Student correctly flagged the scaling ambiguity and held for guidance.

**Advisor direction**: Go with lr=3e-4 (2× Lion baseline) + 2-epoch warmup + cosine. Tests "higher LR with warmup safety" in Lion regime, preserving the spirit of the original hypothesis.

**Status**: Returned to student as wip for rerun with lr=3e-4 on current full stack (Lion+Fourier+wider-192).

---

## 2026-05-12 20:11 — PR #1359: LR warmup + 1e-3 — trial-1 (pre-Lion baseline)
- Branch: willowpai2g48h1-alphonse/lr-warmup-1e-3
- W&B runs: `o0s1z3aq` (trial-1 reported), `qua872ss`, `x9ntld98` (variance check)

| Metric | trial-1 | mean 3 runs | std |
|---|---|---|---|
| val_avg/mae_surf_p (best) | **138.85** @ ep 12 | 144.06 | 4.05 |
| test_avg/mae_surf_p | NaN (cruise GT bug) | — | — |
| test 3-split partial | 139.20 | — | — |
| Epochs completed | 13 (timeout) | — | — |
| Peak GPU mem | 42.1 GB | — | — |

**Analysis**: LR=1e-3 at the edge of stability (epoch-7 spike). High run-to-run variance (std=4.05). Test NaN from data bug (pre-fix). Warmup hypothesis untested vs full stack.

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

## 2026-05-13 04:10 — PR #1710: Surface weight 10→5 — CLOSED ✗
- Branch: willowpai2g48h1-frieren/surf-weight-5
- Hypothesis: surf_weight=10 may over-weight surface terms. Reducing to 5 gives the shared encoder richer volume gradient signal.
- W&B run: `cqjbjme9` — group `surf-weight-sweep`

| Metric | surf_weight=5 | Fourier+wider base (93.29) | Δ | vs Lion base (83.77) | Δ |
|---|---|---|---|---|---|
| val_avg/mae_surf_p | 107.94 | — | — | — | — |
| **test_avg/mae_surf_p** | **94.71** | 93.29 | +1.52% | 83.77 | **+13.1%** |
| test_single_in_dist | 102.47 | 97.57 | +5.0% | 90.07 | +13.8% |
| test_geom_camber_rc | 105.45 | 106.32 | −0.8% | 98.72 | +6.8% |
| test_geom_camber_cruise | 71.93 | 72.25 | −0.4% | 60.96 | +18.0% |
| test_re_rand | 99.01 | 97.04 | +2.0% | 85.32 | +16.0% |

**Analysis**: Tie/slight loss vs Fourier+wider-192 baseline (93.29), clear loss vs current Lion baseline (83.77). Note: by the time this ran, train.py included Fourier, so actual comparison is vs 93.29 not 99.69.  The `test_single_in_dist` regression (+5.0%) is the most interpretable signal: surface supervision strength interacts with Fourier near-foil features — de-emphasizing surface loss specifically hurts the in_dist split where Fourier delivers the largest gain. The volume:surface gradient balance is robust around the default.

**Conclusion**: **Surf-weight lever now closed in BOTH directions** (surf_weight=25 → +10%, surf_weight=5 → +1.5%). Default surf_weight=10 is in a robust local optimum. Do not revisit.

## 2026-05-13 04:10 — PR #1887 (NEW, assigned): Fourier L=8→16 — double frequency resolution
- Branch: willowpai2g48h1-frieren/fourier-L-16
- Hypothesis: Doubling Fourier frequency levels (L=8→16) expands spatial frequency ceiling (space_dim 34→66). NeRF theory predicts monotone gain if pressure field has structure above the L=8 cutoff. Thin trailing-edge boundary layers and sharp suction peaks are plausible sources of L>8 spatial content.
- Single-line change: `fourier_L: int = 16`
- Target: test_avg/mae_surf_p < 83.77 (current Lion provisional baseline)

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

## 2026-05-13 03:00 — PR #1742: Depth n_layers 5→6, budget-safe (CLOSED)
- Branch: willowpai2g48h1-fern/n-layers-6
- W&B run: `uqrafwrn` — group `deeper-model`
- Status: **CLOSED ✗ — fair test of depth at n_hidden=128, rejected**

| Metric | n_layers=6 | Baseline (#1591) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 15) | 144.86 | 125.36 | +15.6% (worse) |
| **test_avg/mae_surf_p** | **127.69** | 111.98 | **+14.0%** (worse) |
| test_single_in_dist | 197.26 | 148.79 | +32.6% |
| test_geom_camber_rc | 131.66 | 117.15 | +12.4% |
| test_geom_camber_cruise | 75.47 | 77.85 | −3.1% (improved!) |
| test_re_rand | 106.37 | 104.13 | +2.2% |
| Epochs completed | 15/18 | 17/18 | — |
| Epoch time | 126.5s | ~96s | +32% |
| Peak GPU mem | 82.97 GB | 82.68 GB | — |

**Analysis**: Unlike n_layers=7 (#1364), this is a fair test — schedule was healthy (15/18 epochs, final LR 4.7e-5, proper cosine coverage). Depth itself hurts at n_hidden=128. Key diagnostic: in_dist regression was worst (+33% on the easiest split), characteristic of training-bottleneck underfitting (not generalization failure). The wider model needs more training signal per epoch than the 18-epoch schedule provides with one extra layer. Cruise actually improved slightly (-3.1%) — depth may help extreme-OOD geometric interactions, but not enough to overcome in_dist and rc losses. **Depth direction dead at n_hidden=128.** Not tested at n_hidden=192 yet.

**Action**: Closed. Assigned fern to PR #1796 — weight_decay 1e-4→1e-3. Wider model (1.47M params) may benefit from stronger regularization.

## 2026-05-13 03:00 — PR #1664: Bias-corrected EMA (CLOSED)
- Branch: willowpai2g48h1-tanjiro/ema-bias-corrected
- W&B run: `xhyqauof` — group `ema-bias-corrected`
- Status: **CLOSED ✗ — lag dominates, wrong regime for Polyak averaging**

| Metric | EMA-BC | Baseline (#1591) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 17) | 139.35 | 125.36 | +11.16% (worse) |
| **test_avg/mae_surf_p** | **125.38** | 111.98 | **+11.97%** (worse) |
| test_single_in_dist | 177.29 | 148.79 | +19.2% |
| test_geom_camber_rc | 124.54 | 117.15 | +6.3% |
| test_geom_camber_cruise | 83.52 | 77.85 | +7.3% |
| test_re_rand | 116.17 | 104.13 | +11.6% |
| Final EMA update_count | 2997 | — | — |
| Final bias_correction | 0.950 | — | — |

**Analysis**: Bias correction worked exactly as designed (val 397→139, -65% from trial-1 uncorrected). The (1-0.999^2997) = 0.95 divisor correctly cancelled random-init contamination. But lag remains the dominant failure mode: EMA val improved 3.3 points in the *final* epoch alone when LR was 5e-6 — EMA was still chasing live-weight gains the live model had already made epochs earlier. The model is still descending at end-of-run (not oscillating in a basin), which is the wrong regime for Polyak averaging. **EMA direction dead in this regime.** All 4 splits regressed 6-19%.

**Action**: Closed. Assigned tanjiro to PR #1798 — gradient norm clipping (max_norm=1.0). Untested stability tool, potentially important for bf16 + n_hidden=192.

## 2026-05-13 03:00 — PR #1796 (NEW): Weight decay 1e-4→1e-3 on wider-192 baseline
- Branch: willowpai2g48h1-fern/weight-decay-1e-3
- Hypothesis: n_hidden=192 (1.47M params, 2.2× more params) may benefit from stronger regularization. wd=1e-3 is 10× current, still modest by transformer standards (BERT: 0.01). Stronger wd → better cross-domain generalization.
- Status: WIP (newly assigned). Target: test_avg < 99.69.

## 2026-05-13 03:00 — PR #1798 (NEW): Gradient norm clipping max_norm=1.0
- Branch: willowpai2g48h1-tanjiro/grad-norm-clip
- Hypothesis: train.py has no grad clipping. Standard in transformer training (GPT, BERT). bf16 + wider model may have gradient spikes disrupting the low-LR refinement phase.
- Status: WIP (newly assigned). Target: test_avg < 99.69.

## 2026-05-13 02:30 — PR #1387: Fourier positional encoding L=8 (MERGED ✓)
- Branch: willowpai2g48h1-nezuko/fourier-pos-features
- Hypothesis: Raw (x,z) coordinates limit the model to low-frequency spatial representations. NeRF-style log-scale Fourier encoding (L=8, space_dim: 2→34) provides explicit high-frequency basis functions, helping the model learn fine-scale pressure gradients near foil leading/trailing edges.
- W&B runs: `nh6alavj` (trial-5, canonical) + earlier trials `twpifp5a`/`111nh26k` (pre-rebase val-only signal)
- Status: **MERGED ✓ — new baseline test=93.29**

| Metric | Trial-5 (final) | Baseline (PR #1361) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 15) | **103.29** | 111.32 ± 2.87 | **−7.21%** |
| **test_avg/mae_surf_p** | **93.29** | 99.69 ± 3.16 | **−6.42%** |
| test_single_in_dist | **97.57** | 116.57 | −16.30% |
| test_geom_camber_rc | **106.32** | 108.61 | −2.11% |
| test_geom_camber_cruise | **72.25** | 74.18 | −2.60% |
| test_re_rand | **97.04** | 99.41 | −2.38% |
| Epochs | 15/18 | 15-16/18 | — |
| Peak GPU memory | 42.5 GB | ~30-40 GB | slight ↑ (space_dim 2→34) |
| Params | ~1.49M | ~1.47M | +0.02M |

**Analysis**: Fourier × width compounds cleanly. Round-1 val signal (trial-1: val=119.70 at n_hidden=128) carried through to the merged baseline. Rebasing onto n_hidden=192 + schedule-aligned cosine amplified the signal. All 4 splits improve — in_dist wins most (−16.3%) suggesting Fourier helps primarily with in-distribution fine-scale pressure patterns near the foil. OOD gains are more modest (−2 to −3%), consistent with the hypothesis that high-frequency basis helps geometry-specific learning more than Reynolds-number generalization. Model beat best single-seed baseline (96.19) by −3.02%. No NaN or inf in any split (cruise GT bug workaround in baseline handles this).

**Action**: Merged as new baseline. Assigned nezuko to PR #1862 — n_layers=6 on Fourier+wider baseline. Prior depth dead ends (n_layers=6 +14%, n_layers=7 +18%) were at n_hidden=128; retesting at n_hidden=192+Fourier where model has richer per-epoch representations.

## 2026-05-13 02:35 — PR #1862 (NEW): n_layers 5→6 on Fourier+wider-192 baseline
- Branch: nezuko/n-layers-6-fourier-wider
- Hypothesis: Depth failed at n_hidden=128 due to training bottleneck (in_dist regressed most). At n_hidden=192 + Fourier (richer residual streams and explicit positional encoding), an extra layer has more structure to compose. Deeper nets extract higher-order feature interactions when input encoding is expressive enough.
- Status: WIP (newly assigned). Target: test_avg < 93.29.

## 2026-05-13 03:20 — PR #1395: Lion optimizer (lr=1.5e-4) (MERGED ✓)
- Branch: willowpai2g48h1-thorfinn/lion-optimizer
- Hypothesis: AdamW's second-moment accumulation is stable but slow; Lion's sign-momentum uses fixed-size steps with momentum for sharper convergence. Lion lr = ~1/5× AdamW lr per paper guidance.
- W&B run: `xhg3h5mi` (trial-3) — group `lion-optimizer`
- Status: **MERGED ✓ — new provisional best test=83.77 (Lion alone, pre-Fourier)**

| Metric | Lion | Fourier baseline (#1387) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best, ep 15) | **92.70** | 103.29 | **−10.26%** |
| **test_avg/mae_surf_p** | **83.77** | 93.29 | **−10.20%** |
| test_single_in_dist | **90.07** | 97.57 | −7.69% |
| test_geom_camber_rc | 98.72 | 106.32 | −7.15% |
| test_geom_camber_cruise | **60.96** | 72.25 | −15.63% |
| test_re_rand | **85.32** | 97.04 | −12.08% |
| Peak GPU memory | ~43 GB | 42.5 GB | similar |
| Optimizer memory | no second-moment buffer | — | ~50% reduction vs AdamW |

**Analysis**: Far exceeded predicted −2% to −5%. Lion's sign-momentum with lr=1.5e-4 outperforms AdamW 7e-4 by a massive margin on all splits. The rc split had previously been the weakest link for all improvements — Lion helps rc too (−7.15%). Cruise gets the biggest boost (−15.6%). The model had 43 GB GPU usage at bs=4 + n_hidden=192 + Lion, vs ~94 GB for AdamW at bs=8. This opens an important budget for bs=8 exploration. CRITICAL: run was on pre-Fourier baseline (space_dim=2). After merge, train.py has Lion + Fourier (space_dim=34) — compound result pending.

**Action**: Merged. New provisional baseline test=83.77. Assigned thorfinn n_head=8 (first test on full stack + attention diversity); assigned askeladd lion-bs-8-sqrt2-lr (bs=8 now fits with Lion's 43 GB budget).

## 2026-05-13 03:25 — PR #1771: Wider-192 schedule realigned epochs=14 (CLOSED ✗)
- Branch: willowpai2g48h1-askeladd/wider-192-schedule-realigned
- Hypothesis: T_max=18 with only 14-15 completable epochs → truncated cosine. Setting T_max=14 aligns schedule to actual budget.
- W&B run: `9ltpngvo` — group `wider-192-schedule-realigned`
- Status: **CLOSED ✗ — over-correction, current T_max=18 already optimal**

| Metric | epochs=14 | Baseline (#1361) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 115.87 | 111.32 ± 2.87 | +4.09% |
| **test_avg/mae_surf_p** | **104.86** | 99.69 ± 3.16 | **+5.19%** |

**Analysis**: Negative result. T_max=14 is worse than T_max=18 despite training 14 complete epochs. The current model completes 14-15 of 18 epochs, which means the last 1-2 epochs have very low LR (~1e-6 range) — but this "slow landing" phase appears beneficial for final weight averaging. Setting T_max=14 makes LR reach exactly 0 at ep 14, which may cause premature over-refinement without the very-low-LR exploration phase. Direction closed.

**Action**: Closed. Assigned askeladd lion-bs-8-sqrt2-lr.

## 2026-05-13 03:40 — PR #1876 (NEW): n_head 4→8 on Lion+Fourier+wider baseline
- Branch: thorfinn/n-head-8-wider-lion
- Hypothesis: n_head=4 at n_hidden=192 gives head_dim=48. Doubling to n_head=8 (head_dim=24) creates 2× more distinct attention patterns, allowing finer-grained physical specialization. Also the first run on fully stacked Lion+Fourier baseline — serves as Lion+Fourier confirmation.
- Status: WIP (newly assigned). Target: test_avg < 83.77.

## 2026-05-13 03:40 — PR #1877 (NEW): Lion bs=8 + sqrt2-lr
- Branch: askeladd/lion-bs-8-sqrt2-lr
- Hypothesis: Lion has no second-moment buffer — only 43 GB vs AdamW's 94 GB at bs=4. This opens the budget for bs=8 (should be ~55-70 GB). Larger batches improve gradient accuracy, especially OOD. lr=2.1e-4 (√2 × 1.5e-4) for √2 batch-size scaling.
- Status: WIP (newly assigned). Target: test_avg < 83.77.
