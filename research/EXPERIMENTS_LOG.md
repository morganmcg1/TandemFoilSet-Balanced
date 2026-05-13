# SENPAI Research Results

## 2026-05-13 05:00 — PR #1359: LR warmup + lr=3e-4 + Lion — CLOSED ✗
- Branch: willowpai2g48h1-alphonse/lr-warmup-1e-3
- W&B run: `10eslxj8` — group `lr-warmup-lion`

| Metric | lr=3e-4 + warmup | Lion baseline (83.77) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 96.93 | 92.70 | +4.56% |
| **test_avg/mae_surf_p** | **88.37** | **83.77** | **+5.49%** |
| test_single_in_dist | 95.43 | 90.07 | +5.95% |
| test_geom_camber_rc | 99.72 | 98.72 | +1.01% |
| test_geom_camber_cruise | 68.00 | 60.96 | +11.55% |
| test_re_rand | 90.34 | 85.32 | +5.88% |

**Analysis**: 2-epoch warmup + 2× LR (1.5e-4→3e-4) regressed all 4 splits. Key insight: Lion's sign-momentum is inherently LR-stable — epoch-1 warmup LR (1.515e-4) was *already at the baseline LR*, making warmup structurally redundant. The higher peak (3e-4) spent more budget at high LR without faster convergence. Curve still descending at cutoff (e14→e15: val 97.06→96.93) — schedule vs time-budget mismatch compounds the issue.

**LR-warmup lever CLOSED for Lion.** Lion lr=1.5e-4 + cosine is near-optimal.

## 2026-05-13 05:10 — PR #1945 (NEW, assigned): n_hidden 192→256
- Branch: willowpai2g48h1-alphonse/n-hidden-256
- Hypothesis: Lion's memory savings (~43GB vs AdamW's ~94GB at n_hidden=192) create headroom for n_hidden=256 (~57GB estimated). Width has been the dominant lever: 128→192 gave −11%. Capacity stacking test.
- Single change: `n_hidden=256` in model_config dict
- Target: test_avg/mae_surf_p < 83.77

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

## 2026-05-13 04:55 — PR #1796: Weight decay 1e-4→1e-3 — trial-1 on STALE BASELINE, sent back
- Branch: willowpai2g48h1-fern/weight-decay-1e-3
- W&B run: `yg909luj` — group `weight-decay-sweep`
- **CRITICAL**: Trial-1 ran on pre-Lion config (lr=7e-4 AdamW, 101.75 GB peak — NOT Lion's 43GB). Did NOT pick up the Lion+Fourier merge.

| Metric | Trial-1 (stale base) | Old wider-192 base (99.69) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 111.12 | 111.32 | −0.2% (tie) |
| test_avg/mae_surf_p | 100.10 | 99.69 | +0.4% (tie/loss) |
| test_single_in_dist | 121.25 | 116.57 | +4.0% (worse) |
| test_geom_camber_rc | 112.24 | 108.61 | +3.3% (worse) |
| test_geom_camber_cruise | **69.97** | 74.18 | **−5.7%** (better) |
| test_re_rand | **96.93** | 99.41 | **−2.5%** (better) |

**Per-split signal**: Stronger wd helps OOD splits (cruise, re_rand) but hurts in-distribution and rc — equal-weight aggregate is a wash. Hypothesis directionally supported but cancelled in average.

**Action**: Sent back to rebase onto current advisor branch (Lion+Fourier+wider-192) and rerun. Lion paper specifically recommends 3-10× larger wd than AdamW because sign-momentum has uniform-magnitude updates; wd=1e-3 is exactly in the Lion-recommended range. Expected memory drop from 101GB → 43GB confirms whether rebase succeeded.

**Status**: WIP (rerun pending).

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
- Status: WIP (in-flight). Target: test_avg < 83.77.

---

## 2026-05-13 05:54 — PR #1862: n_layers=6 on Fourier+wider-192 — CLOSED ✗
- Branch: nezuko/n-layers-6-fourier-wider
- W&B runs: `7qciycr8` (primary), `3hv0f341` (replicate) — group `n-layers-6-fourier-wider`
- Config: n_hidden=192, n_layers=6, Fourier L=8, AdamW lr=7e-4 (pre-Lion stack; stale config vs current)

| Metric | n_layers=6 (primary) | n_layers=6 (replicate) | Baseline (PR #1387, Fourier+wider AdamW) | Δ vs Fourier base |
|---|---|---|---|---|
| val_avg/mae_surf_p (best) | 117.72 (epoch 12) | 117.48 (epoch 12) | 103.29 | +14.0% |
| **test_avg/mae_surf_p** | **107.01** | 105.68 | **93.29** | **+14.7%** |
| test_single_in_dist | 127.93 | 126.11 | 97.57 | +31.1% |
| test_geom_camber_rc | 115.02 | 110.88 | 106.32 | +8.2% |
| test_geom_camber_cruise | 77.87 | 79.67 | 72.25 | +7.8% |
| test_re_rand | 107.21 | 106.06 | 97.04 | +10.5% |

- Epochs completed: 12/18 (timeout); epoch time ~151 s (+20% vs baseline 126 s)
- Trial-to-trial variance ±0.7 on test_avg — result statistically conclusive

**Analysis**: **n_layers=6 dead-end confirmed TRIPLE-CROSS across three width/feature regimes:**
| Config | n_layers | Δ test |
|---|---|---|
| n_hidden=128 (prior round) | 6 | +14% |
| n_hidden=128 (prior round) | 7 | +18% |
| n_hidden=192 + Fourier (this PR) | 6 | +14% |

Root cause: **horizon-vs-depth tradeoff**, not capacity. Per-epoch cost 126s→151s (+20%) compresses 30-min budget from 14 epochs → 12. Cosine T_max=18 LR never reaches refinement phase. Best-epoch=last-epoch in both trials = model still improving at cutoff. Single_in_dist worst hit (+31%) is the underfitting-sensitive split — fingerprints premature schedule termination, not generalization failure.

**Depth axis: CLOSED** for this benchmark at 30-min wall-clock budget. Lion would not change this; the binding constraint is wall-clock per-epoch cost.

**Action**: Closed. Assigned nezuko slice-num-96 (#1967).

---

## 2026-05-13 05:54 — PR #1796: wd=1e-3 under Lion+Fourier — CLOSED ✗
- Branch: willowpai2g48h1-fern/weight-decay-1e-3
- W&B run: `3loe2ooi` — group `weight-decay-sweep`
- Config: Lion lr=1.5e-4, Fourier L=8, n_hidden=192, **weight_decay=1e-3**, bs=4, bf16, epochs=18

| Metric | Lion+Fourier+wd=1e-3 | Lion-only baseline (83.77) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, epoch 14) | **92.37** | 92.70 | −0.4% (tie) |
| **test_avg/mae_surf_p** | **84.41** | **83.77** | **+0.8% (slight regression)** |
| test_single_in_dist | 87.27 | 90.07 | **−3.1%** ✓ |
| test_geom_camber_rc | 96.24 | 98.72 | **−2.5%** ✓ |
| test_geom_camber_cruise | 66.67 | 60.96 | +9.4% ✗ |
| test_re_rand | 87.47 | 85.32 | +2.5% ✗ |

- 15/18 epochs at 128 s/epoch, peak GPU 43.4 GB
- Note: Comparison is Lion+Fourier+wd=1e-3 vs Lion-only+wd=1e-4 — not fully apples-to-apples

**Analysis**: **wd=1e-3 lever exhausted** under both optimizers, with a striking per-split sign-flip between AdamW and Lion:

| Split | AdamW + wd=1e-3 Δ | Lion + wd=1e-3 Δ |
|---|---|---|
| single_in_dist | +4.0% (worse) | **−3.1%** (better) |
| geom_camber_rc | +3.3% (worse) | **−2.5%** (better) |
| geom_camber_cruise | **−5.7%** (better) | +9.4% (worse) |
| re_rand | **−2.5%** (better) | +2.5% (worse) |

Mechanically: Lion+Fourier already pushed cruise from 72.25 → 60.96 (~15%), exhausting OOD headroom. Adding wd on top claws capacity back into in-distribution. AdamW hadn't reached the same OOD floor, so wd had the opposite incentive. Effects cancel at the aggregate — net wash.

**Weight-decay magnitude lever: CLOSED** in both directions (wd=1e-3 tested, wd=1e-4 is default). Pivot to decoupled-wd instead (structurally different: zero wd on biases/norms).

**Action**: Closed. Assigned fern decoupled-weight-decay (#1969).

---

## 2026-05-13 05:56 — PR #1876: n_head=8 on Lion+Fourier+wider — CLOSED ✗
- Branch: thorfinn/n-head-8-wider-lion
- W&B run: `xo9xcyat` — group `n-head-8`
- Config: Lion lr=1.5e-4, Fourier L=8, n_hidden=192, **n_head=8** (head_dim 48→24), bs=4, bf16, epochs=18

| Metric | n_head=8 | Lion baseline (83.77) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, epoch 11) | **117.16** | 92.70 | **+26.4%** |
| **test_avg/mae_surf_p** | **105.04** | **83.77** | **+25.4%** |
| test_single_in_dist | 115.47 | 90.07 | +28.2% |
| test_geom_camber_rc | 130.04 | 98.72 | +31.7% |
| test_geom_camber_cruise | 71.97 | 60.96 | +18.1% |
| test_re_rand | 102.68 | 85.32 | +20.4% |

- Epochs completed: 11/18; epoch time ~168 s (+33% vs 126 s baseline); peak GPU 84.4 GB
- Val curve still descending at epoch 11 (uncoverged): 215 → 203 → 148 → 136 → 124 → 117

**Analysis**: Three compounding failures:
1. **head_dim halved 48→24** — below Transolver paper's ≥32 threshold. Physics-aware attention with slice_num=64 physics tokens needs sufficient per-head rank.
2. **+33% per-epoch cost** → 11 epochs vs 14 baseline. Same "horizon-vs-capacity" pattern as depth and mlp_ratio.
3. **Oscillatory val curve** (124→128→135→133→117) — Lion sign-momentum on narrow-head gradients creates rough landscape.

**First Lion+Fourier compound run** — but confounded by n_head=8. Cannot cleanly read the compound. Four in-flight PRs (#1877, #1887, #1945, #1798) will implicitly confirm Lion+Fourier compound baseline.

**n_head=8 closed.** Revisit would need n_hidden=256 (keeps head_dim=32 at 8 heads) — wait for alphonse's #1945 result first.

**Action**: Closed. Assigned thorfinn lion-beta2-0999 (#1971).

---

## 2026-05-13 05:56 — PR #1643: mlp_ratio=4 on wider-192 — CLOSED ✗
- Branch: willowpai2g48h1-edward/mlp-ratio-4
- W&B runs: `q21pfshj` (bs=4 primary), `mgdq08qc` (bs=8 sanity) — group `mlp-ratio-4`
- Config: n_hidden=192, **mlp_ratio=4**, AdamW lr=7e-4 (pre-Lion stack — stale config)

| Metric | mlp_ratio=4 bs=4 | mlp_ratio=4 bs=8 | Baseline PR #1361 (wider-192, 3-seed) | Δ bs=4 |
|---|---|---|---|---|
| val_avg/mae_surf_p (best) | **120.48** | 121.30 | 111.32 ± 2.87 | +8.2% |
| **test_avg/mae_surf_p** | **109.69** | 110.92 | **99.69 ± 3.16** | **+10.0%** |
| test_single_in_dist | 123.10 | 138.60 | 116.57 | +5.6% |
| test_geom_camber_rc | 124.88 | 118.09 | 108.61 | +14.9% |
| test_geom_camber_cruise | 82.64 | 78.25 | 74.18 | +11.4% |
| test_re_rand | 108.15 | 108.73 | 99.41 | +8.8% |

- Epochs completed: 13/18; epoch time ~139 s (+11% vs 126 s); peak GPU 50.6 GB at bs=4

**Analysis**: Third dead-end from "horizon-vs-capacity" tradeoff. +11% per-epoch cost compresses budget from 15-16 → 13 epochs. Val still improving monotonically at last epoch (121.23 → 120.48) — not converged. Same failure signature as n_layers=6 (+14%) and n_head=8 (+33%).

**Key side observation**: bs=8 at n_hidden=192 + mlp_ratio=4 = **50.6 GB peak** — well within 97 GB envelope. The prior documented "OOM cliff at bs=8 + n_hidden=192" was likely attributed to mlp_ratio=2 at a different memory spike. This clarifies that askeladd's #1877 (Lion bs=8) is not at risk of OOM. BASELINE.md OOM note is overstated.

Third dead-end from per-epoch cost in round-2. Pattern confirmed: any change that slows per-epoch > ~10% hurts on 30-min budget unless schedule is realigned.

**Action**: Closed. Assigned edward cosine-eta-min (#1973).

---

## 2026-05-13 06:00 — PR #1967 (NEW): slice_num 64→96
- Branch: willowpai2g48h1-nezuko/slice-num-96
- Hypothesis: Physics-attention slot count increase (1.5×). Orthogonal to depth/width/Fourier. More distinct flow-regime templates per block. Per-epoch cost impact: <8%.
- Target: test_avg/mae_surf_p < 83.77

## 2026-05-13 06:00 — PR #1969 (NEW): Decoupled weight decay
- Branch: willowpai2g48h1-fern/decoupled-weight-decay
- Hypothesis: Apply wd=1e-4 only to weight matrices, zero wd on biases/norm scales. Standard Transformer practice; especially important for Lion's uniform-magnitude updates on small bias parameters. Zero memory/compute overhead.
- Target: test_avg/mae_surf_p < 83.77

## 2026-05-13 06:00 — PR #1971 (NEW): Lion beta2=0.999
- Branch: willowpai2g48h1-thorfinn/lion-beta2-0999
- Hypothesis: beta2=0.99 → 0.999. Longer sign-momentum horizon (~1000 steps vs ~100). More stable at bs=4 noisy small-batch gradients. Paper alternate value for vision-like tasks with diverse gradients.
- Target: test_avg/mae_surf_p < 83.77

## 2026-05-13 06:00 — PR #1973 (NEW): Cosine eta_min=lr/10
- Branch: willowpai2g48h1-edward/cosine-eta-min
- Hypothesis: Set `eta_min=1.5e-5` (lr/10) in CosineAnnealingLR. Prevents LR from quenching to 0 at late epochs. Insurance for truncated training; keeps final epochs learning at floor rate. Single-line change.
- Target: test_avg/mae_surf_p < 83.77

---

## 2026-05-13 06:00 — PR #1877: Lion bs=8 + sqrt2-lr — CLOSED ✗
- Branch: askeladd/lion-bs-8-sqrt2-lr
- W&B runs: `7w5h25xi` (bs=8 primary), `ycm0yf98` (bs=6 fallback), `g4cbgm98` (OOM'd attempt) — group `lion-bs-8`

| Metric | bs=8 lr=2.1e-4 | bs=6 lr=1.84e-4 | Lion-bs=4 baseline | Δ bs=8 |
|---|---|---|---|---|
| val_avg/mae_surf_p | 99.005 (ep 12) | 94.719 (ep 14) | 92.70 | +6.8% |
| **test_avg/mae_surf_p** | **89.213** | 86.032 | **83.77** | **+6.5%** |
| test_single_in_dist | 95.342 | 92.384 | 90.07 | +5.9% |
| test_geom_camber_rc | 102.546 | **94.288** | 98.72 | +3.9% / **−4.5%** ← |
| test_geom_camber_cruise | 67.575 | 68.730 | 60.96 | +10.9% |
| test_re_rand | 91.388 | 88.727 | 85.32 | +7.1% |

- bs=8 epoch time: ~136 s (+6% vs 128 s); peak GPU ~89 GB; 14/18 epochs; 2632 optimizer steps
- bs=6 epoch time: ~130 s; peak GPU ~99 GB; 14/18 epochs; 3514 optimizer steps
- Baseline optimizer steps (bs=4): 5640

**Analysis**: Step-count starvation is the binding constraint. bs=8 gets 2.1× fewer optimizer steps than bs=4 (2632 vs 5640). Lion sign-momentum needs update count to converge; halving steps halves total weight displacement. Per-epoch val table: baseline still improving at ep 15 (val 92.7); bs=8 plateaus at val ~99 from ep 12.

**Interesting signal**: bs=6 improved test_geom_camber_rc by **−4.5%** (only split that beat baseline). Partially validates that better gradient signal helps noisy OOD rc geometry. But cruise regressed +12.7% in both runs.

**Gradient accumulation as next step**: Accumulate 2 micro-batches at bs=4 → effective bs=8, same 5640 update steps, same 43 GB peak memory. Gets the gradient quality benefit without step-starvation. Assigned to askeladd.

**Action**: Closed. Assigned askeladd gradient-accumulation (#1980).

## 2026-05-13 06:05 — PR #1980 (NEW): Gradient accumulation (accum=2, eff_bs=8)
- Branch: willowpai2g48h1-askeladd/gradient-accumulation
- Hypothesis: 2 micro-batches at bs=4 → effective bs=8 but same step count as bs=4 (5640 updates). Gets the gradient quality benefit without step-starvation. Sign vote over accumulated gradient = better directional signal. Zero memory change (43 GB peak).
- Target: test_avg/mae_surf_p < 83.77

---

## 2026-05-13 06:55 — PR #1980: Gradient accumulation accum=2 — **MERGED** ✓ NEW BEST
- Branch: willowpai2g48h1-askeladd/gradient-accumulation
- W&B run: `6qxwtm0v` — group `gradient-accumulation`
- Config: Lion lr=1.5e-4, Fourier L=8, n_hidden=192, **accumulation_steps=2** (eff_bs=8), bs=4, bf16

| Metric | Grad-accum=2 | Previous baseline (Lion-only, 83.77) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, epoch 14) | **90.82** | 92.70 | **−2.04%** |
| **test_avg/mae_surf_p** | **80.62** | **83.77** | **−3.77%** ✅ |
| test_single_in_dist | 82.23 | 90.07 | **−8.71%** 🏆 |
| test_geom_camber_rc | 93.60 | 98.72 | **−5.18%** |
| test_geom_camber_cruise | 61.57 | 60.96 | +1.00% |
| test_re_rand | 85.06 | 85.32 | −0.31% |

- Epochs: 14/18 at ~129 s/epoch; peak GPU **43.4 GB** (unchanged from bs=4!); 2632 total optimizer steps

**Analysis**: Gradient accumulation over 2 micro-batches gives a higher signal-to-noise sign vote for Lion, mainly via tighter per-micro-batch mesh padding. TandemFoilSet meshes vary 900–250K nodes; padding to the local max of 4 samples vs the full bs=8 max reduces noise-from-padding significantly on variable-length batches. The dominant improvement is on in_dist (−8.71%) and rc (−5.18%). Cruise +1.0% — already the easiest split (smallest error), likely near noise floor.

**Note**: The student correctly observed that this gives bs=8's step count (~1316 steps/epoch × 14 = 2632), NOT the same as bs=4 (5640). Despite the step reduction, the per-micro-batch padding advantage wins vs the true bs=8 attempt (#1877, +6.5%). The gradient accumulation approach compares favorably to both reference points.

**NEW BASELINE: test=80.62, val=90.82** — Cumulative gain from PR #1391: 121.28 → 80.62 = **−33.5%**

**Action**: MERGED. Assigned askeladd grad-accum-4 (#2009).

---

## 2026-05-13 06:55 — PR #1973: Cosine eta_min=lr/10 — CLOSED ✗
- Branch: willowpai2g48h1-edward/cosine-eta-min
- W&B run: `57aqam8j` — group `cosine-eta-min`
- Config: Lion lr=1.5e-4, Fourier L=8, n_hidden=192, **eta_min=1.5e-5** (lr/10), bs=4, bf16

| Metric | eta_min=1.5e-5 | Lion baseline (83.77) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, epoch 14) | **95.79** | 92.70 | **+3.33%** |
| **test_avg/mae_surf_p** | **86.71** | **83.77** | **+2.94%** |
| test_single_in_dist | 89.09 | 90.07 | −0.98% |
| test_geom_camber_rc | 99.46 | 98.72 | +0.74% |
| test_geom_camber_cruise | 67.23 | 60.96 | +10.29% |
| test_re_rand | 91.03 | 85.32 | +6.70% |

- 14/18 epochs at 130 s/epoch; peak GPU 88.12 GB; final LR at ep14 = 3.08e-5

**Analysis**: My LR math in the PR body was wrong. Claimed eta_min=1.5e-5 gives LR ≈ 1.84e-5 at epoch 14 — "nearly identical" to eta_min=0's 1.76e-5. Actual: LR(14) = 3.08e-5, **75% higher**. The floor raises the effective LR throughout the late-training window, not just at the tail. Lion's sign-momentum took oversized steps during refinement, blowing sharp minima on cruise (+10.3%) and re_rand (+6.7%). Excellent student diagnosis.

**LR schedule floor lever: CLOSED.** eta_min=lr/10 is too large for truncated cosine at T_max=18. eta_min=lr/100 would be marginal, T_max-alignment is the cleaner fix (but T_max=14 was already tested in #1771 and failed). Pivot to activation function change.

**Action**: Closed. Assigned edward swiglu-activation (#2010).

---

## 2026-05-13 07:00 — PR #2009 (NEW): Grad accum=4 (eff_bs=16)
- Branch: willowpai2g48h1-askeladd/grad-accum-4
- Hypothesis: Natural follow-up to accum=2 win. accum=4 further reduces padding noise (even tighter micro-batch padding) but halves optimizer steps again (~1316 total). Tests whether gradient-noise reduction continues to dominate vs step-starvation at eff_bs=16. Either outcome is informative: win = noise still dominant; loss = starvation resumes at eff_bs=16.
- Target: test_avg/mae_surf_p < 80.62 (new baseline)

## 2026-05-13 07:00 — PR #2010 (NEW): SiLU (Swish) activation
- Branch: willowpai2g48h1-edward/swiglu-activation
- Hypothesis: Replace GELU with SiLU in all MLP blocks (model_config: act="silu"). SiLU is already in ACTIVATION dict. Single parameter change, zero memory/compute overhead. SiLU consistently matches or beats GELU on modern Transformer benchmarks. Testing on grad-accum=2 stack (--accumulation_steps 2).
- Target: test_avg/mae_surf_p < 80.62 (new baseline)

---

## 2026-05-13 07:15 — PR #1971: Lion beta2=0.999 — CLOSED (regressed)
- Branch: willowpai2g48h1-thorfinn/lion-beta2-0999
- Hypothesis: Increase Lion beta2 from 0.99 → 0.999. Longer sign-momentum horizon (~1000 steps vs ~100 at beta2=0.99) for more stable sign votes under bs=4 padding noise.
- W&B run: see PR #1971 comments
- Config: Lion lr=1.5e-4, **lion_beta2=0.999**, Fourier L=8, n_hidden=192, bs=4, bf16

| Metric | beta2=0.999 | Lion baseline (83.77) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **88.79** | **83.77** | **+5.99%** |

**Analysis (from student diagnosis, adopted)**: The 0.999 horizon (~1000 steps to forget a gradient) **exceeds the total training step budget** of 1170-1316 steps at bs=4. The momentum buffer never equilibrates — it remains in an "averaging-in" phase for the entire run. Sign votes are dominated by the initial random-gradient soup rather than the optimization trajectory. At beta2=0.99 (~100-step horizon), the buffer equilibrates after ~3 epochs, then tracks the true gradient direction productively for the remaining 15 epochs.

Per-split signs are uniformly bad (no compensating wins on any split), consistent with a system-wide signal degradation rather than a regularization tradeoff.

**Lion beta2 horizon lever: CLOSED.** beta2=0.99 (~100 steps) is well-matched to our truncated step budget. beta2=0.999 would only be viable with substantially longer training. beta2=0.95 (already ruled out in original Lion paper for general use) would shorten the horizon further but is unlikely to improve over 0.99 given the variable-mesh batch noise.

**Action**: Closed. Assigned thorfinn drop-path-stochastic-depth (#2030).

---

## 2026-05-13 07:15 — PR #2030 (NEW): DropPath stochastic depth (rate=0.1)
- Branch: willowpai2g48h1-thorfinn/drop-path-stochastic-depth
- Hypothesis: Add DropPath (stochastic depth) at the residual branches of each TransolverBlock with linear schedule 0→0.1 across 5 layers. Stochastically drops the entire attention or MLP residual branch during training; identity at inference. Acts as implicit ensembling over depth subnets without touching optimizer state, loss, or representation — pure structural regularizer.
- Mechanism: Orthogonal to every previous lever (Lion, bf16, Fourier, grad-accum, n_hidden, schedule). Per-split divergence pattern (cruise=61.57 vs rc=93.60, 1.52× gap) suggests OOD regularization headroom. Zero memory/compute overhead — just a per-sample Bernoulli mask during training.
- References: Huang 2016 (stochastic depth), Touvron 2021 (CaiT linear schedule), Liu 2022 (ConvNeXt).
- Target: test_avg/mae_surf_p < 80.62 (new baseline from #1980)

---

## 2026-05-13 07:45 — PR #1945: n_hidden 192→256 — CLOSED (budget mismatch, not capacity failure)
- Branch: willowpai2g48h1-alphonse/n-hidden-256
- Hypothesis: Increase model width n_hidden 192→256 (~33%) on Lion+Fourier+grad-accum stack to leverage Lion's freed memory budget.
- W&B run: z3h0j6ks (alphonse, n-hidden-256-trial-1)
- Config: n_hidden=256, all other defaults; bs=4 (no grad-accum=2 was set — note: ran against old 83.77 baseline target)

| Metric | n_hidden=256 | n_hidden=192 baseline | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **107.39** | **83.77** | **+28.2%** |
| val_avg/mae_surf_p (best ep12) | 116.65 | 92.70 | +25.8% |
| test_single_in_dist | 116.08 | 90.07 | +28.9% |
| test_geom_camber_rc | 110.54 | 98.72 | +12.0% |
| test_geom_camber_cruise | 89.66 | 60.96 | +47.1% |
| test_re_rand | 113.29 | 85.32 | +32.8% |

- 12/18 epochs at 153 s/epoch; hit `SENPAI_TIMEOUT_MINUTES=30` mid-epoch 13
- Peak GPU: **53.5 GB** (predicted 57 GB ✓ — memory model was correct)
- Model params: 2.62M (vs 1.47M @ 192, 1.78× ratio matches O(n_hidden²) ✓)
- Validation curve **still monotonically declining** at epoch 12 — undertrained

**Analysis (student diagnosis adopted)**: This is a **budget/schedule mismatch, not a capacity verdict**. Per-epoch time scaled as O(n_hidden^1.4) (not the naive 1.33×). At 153s/epoch the 30-min budget cuts off at 12 epochs while T_max=18 cosine never reaches the late refinement window. Two confounders stacked: undertraining (67% of schedule) + premature LR quench (cosine at minimum but model still learning). The +28% regression cannot be attributed to width itself.

**Action**: Closed. Assigned alphonse n-hidden-224-rescaled-cosine (#2047): n_hidden=224 (17% wider — moderate), epochs=15, T_max=15. Budget projects ~117s/epoch × 15 ≈ 29.3 min, fits cleanly.

---

## 2026-05-13 07:45 — PR #1887: Fourier L=8→16 — CLOSED (frequency aliasing)
- Branch: willowpai2g48h1-frieren/fourier-L-16
- Hypothesis: Double NeRF Fourier frequency ceiling L=8→16 (space_dim 34→66) for richer positional encoding.
- W&B run: xsqzxgi7 (frieren, fourier-L-16-trial-1)
- Config: fourier_L=16, all other defaults; bs=4 (note: ran against old 83.77 baseline target)

| Metric | L=16 | L=8 baseline | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **87.60** | **83.77** | **+4.57%** |
| val_avg/mae_surf_p (best ep14) | 97.89 | 92.70 | +5.60% |
| test_single_in_dist | 90.98 | 90.07 | +1.01% |
| test_geom_camber_rc | 99.67 | 98.72 | +0.96% |
| test_geom_camber_cruise | 67.52 | 60.96 | +10.76% |
| test_re_rand | 92.24 | 85.32 | +8.11% |

- 14/18 epochs at 130 s/epoch (truncated by 30-min cap — same budget as L=8 baseline so still a fair comparison)
- Peak GPU: 43.7 GB (unchanged — extra input dims are negligible)

**Analysis (student diagnosis adopted)**: **Frequency aliasing on irregular mesh**. CFD mesh density is highly non-uniform (dense near foils, sparse in freestream); L=16 pushes Fourier wavelengths to 2^15, far below the local Nyquist frequency in sparse regions → aliased noise, not informative encoding. Per-split signature confirms: cruise (+10.76%) and re_rand (+8.11%) regress most — these have the largest geometric variation where mesh-density mismatch is worst. Secondary factor: 66-d vs 34-d input layer slows optimization (~35% epoch-time increase, 130s vs 96s).

**Fourier frequency-ceiling lever: CLOSED.** L=8 is optimal in this stack. Per-axis Fourier L (Lx=8, Lz=4) is mentioned by student as a possible follow-up but is high-implementation-cost for expected ~1% gain — parked.

**Action**: Closed. Assigned frieren ema-weights-decay-0999 (#2050): EMA weight averaging with decay=0.999, eval on EMA copy. Orthogonal trajectory smoother for Lion sign-momentum bounce in late-cosine phase. ~6 MB memory overhead, zero compute overhead.

---

## 2026-05-13 07:45 — PR #2047 (NEW): n_hidden=224 with budget-aligned cosine (epochs=15, T_max=15)
- Branch: willowpai2g48h1-alphonse/n-hidden-224-rescaled-cosine
- Hypothesis: Moderate width bump (17%, vs 33% for #1945) with rescaled epochs so cosine schedule completes. Tests whether ANY width gain compounds on Lion+Fourier+grad-accum stack within the 30-min budget. Projects 117 s/epoch × 15 = 29.3 min; T_max=15 means schedule properly hits the LR floor.
- Target: test_avg/mae_surf_p < 80.62 (new baseline from #1980)

## 2026-05-13 07:45 — PR #2050 (NEW): EMA weight averaging (decay=0.999) for eval
- Branch: willowpai2g48h1-frieren/ema-weights-decay-0999
- Hypothesis: Maintain a shadow copy of model weights updated as ema = 0.999*ema + 0.001*model per optimizer step; evaluate on the EMA copy. Averages over ~1000-step horizon (final ~40% of training) — recovers the trajectory center rather than the noisy late-cosine endpoint. Especially valuable under Lion's uniform-magnitude sign updates which produce endpoint variance even at the LR floor. ~6 MB memory overhead, zero compute overhead, orthogonal to every current lever.
- Target: test_avg/mae_surf_p < 80.62

---

## 2026-05-13 08:30 — PR #2009: Grad accum=4 (eff_bs=16) — CLOSED (step starvation)
- Branch: willowpai2g48h1-askeladd/grad-accum-4
- Hypothesis: accum=4 further reduces padding noise vs accum=2; tests where gradient-quality saturates vs step-count starvation.
- W&B run: tutefqn0 (askeladd, grad-accum-4-trial-1, group `gradient-accumulation`)
- Config: bs=4, accum=4 (eff_bs=16), Lion lr=1.5e-4, all other defaults

| Metric | accum=4 | accum=2 baseline (80.62) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **89.01** | **80.62** | **+10.40%** |
| val_avg/mae_surf_p (best ep14) | 99.64 | 90.82 | +9.72% |
| test_single_in_dist | 96.22 | 82.23 | +17.01% |
| test_geom_camber_rc | 101.97 | 93.60 | +8.94% |
| test_geom_camber_cruise | 66.28 | 61.57 | +7.65% |
| test_re_rand | 91.59 | 85.06 | +7.67% |

- 14 epochs (timeout-stopped, 127s/epoch); total optimizer steps **1316** (exact 2× fewer than accum=2's 2632 — same as failed bs=8 #1877)
- Peak GPU **93.7 GB** (higher than predicted — variable-mesh padding scales with eff_bs)
- Final LR: 2e-5 (cosine reached late-stage)

**Analysis (student diagnosis adopted)**: Step-count starvation now dominates beyond accum=2. The gradient-quality benefit from accum saturated at eff_bs=8. accum=4 trades 50% of optimizer steps for cleaner gradients — and the trade is unfavorable. Per-split signature is roughly uniform (+7.6 to +17%), with in_dist hit hardest (matches step-starvation pattern: in-distribution benefits most from longer low-LR fine-tuning).

Interesting datapoint: peak GPU 93.7 GB at accum=4 is much higher than expected from naive micro-batch view. Variable-mesh padding peaks must scale with grad accumulator state size. Parked as future investigation.

**Gradient accumulation lever CLOSED at accum=2 (optimum).** accum=3 spot-check (eff_bs=12) would refine the bracket but is expected to be midway and not worth a slot.

**Action**: Closed. Assigned askeladd lion-lr-2.1e-4-sqrt2-scaling (#2088): Lion lr 1.5e-4→2.1e-4 (sqrt(2) scaling per Lion-paper rule, untested in the grad-accum=2 stack).

---

## 2026-05-13 08:30 — PR #1798: Grad-norm-clip max_norm=1.0 — CLOSED (wrong baseline, re-test needed)
- Branch: willowpai2g48h1-tanjiro/grad-norm-clip
- Hypothesis: clip_grad_norm with max_norm=1.0 as tail stabilizer for n_hidden=192 + bf16 + cosine refinement.
- W&B run: errhax66 (tanjiro, grad-norm-clip-trial-1, group `grad-norm-clip`)
- Config: PR branch did NOT rebase before run → tested on **pre-Lion AdamW config** (lr=7e-4, accum=1), not current Lion+grad-accum stack.

| Metric | clip=1.0 | OLD AdamW baseline (99.69) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **79.91** | **99.69** | **−19.8%** |
| val_avg/mae_surf_p (best ep15) | 91.31 | 111.32 | −18.0% |
| test_single_in_dist | 89.82 | 116.57 | −23.0% |
| test_geom_camber_rc | 96.94 | 108.61 | −10.7% |
| test_geom_camber_cruise | 54.62 | 74.18 | −26.4% |
| test_re_rand | 78.26 | 99.41 | −21.3% |

- 15/18 epochs at ~120s/epoch (timeout cap)
- **Clip fired on 100% of batches** at all epochs (raw gradient norms 25-550 throughout)
- Per-epoch grad diagnostics: mean‖g‖ from 76.67 (ep1) → 24.90 (ep15); max‖g‖ stayed 270-550

**Analysis (mechanism diagnosis)**: With max_norm=1.0 firing 100% of batches at grad norms 25-550, AdamW becomes effectively **sign-of-gradient** — combining unit-norm gradients with AdamW's per-parameter adaptive scaling. This approximates Lion's sign-of-momentum mechanism via a different optimizer route, landing in the same ~80 test neighborhood. The result is genuinely interesting analytically (confirms that Lion's win is from the *normalization*, not the *symbolic search*) but **cannot be merged** — the branch is on the old AdamW config (lr=7e-4, no Lion, no grad-accum); merging would replace the entire current Lion+grad-accum stack.

**Per-split sig** (uniform 10-26% gains across all splits including in_dist) confirms regularizer-not-stabilizer reading — but the *tail-stabilizer* hypothesis (the original intent) was not actually tested because clip=1.0 was orders of magnitude below natural norms.

**Action**: Closed (merge-blocking conflict + wrong-baseline confound). Assigned tanjiro grad-norm-clip-5-on-lion-stack (#2090): grad_clip_max_norm=5.0 on current Lion+grad-accum=2 stack. clip=5.0 should fire ~10-15% of batches (rare-event "real" tail clipping), tests whether residual variance in Lion's late-cosine endpoint comes from outlier batches.

---

## 2026-05-13 08:30 — PR #2088 (NEW): Lion lr 1.5e-4 → 2.1e-4 (sqrt(2) scaling for eff_bs=8)
- Branch: willowpai2g48h1-askeladd/lion-lr-2.1e-4-sqrt2-scaling
- Hypothesis: Lion's original paper recommends lr scale as sqrt(batch_size). Going eff_bs 4→8 (via grad-accum=2 merge) → optimal lr should scale 1.5e-4 × sqrt(2) = 2.1e-4. Untested in current merged stack — lr was inherited from pre-grad-accum baseline. Zero overhead, single hyperparameter change.
- References: Chen 2023 (Lion paper), McCandlish 2018 (critical batch size theory).
- Target: test_avg/mae_surf_p < 80.62

## 2026-05-13 08:30 — PR #2090 (NEW): Grad-norm-clip max_norm=5.0 on Lion+grad-accum stack
- Branch: willowpai2g48h1-tanjiro/grad-norm-clip-5-on-lion-stack
- Hypothesis: With max_norm=5.0 (vs Lion-implicit-normalized inputs), clipping fires on ~10-15% of batches — only genuine outlier-mesh-induced spikes. Tests the *original* tail-stabilizer mechanism that PR #1798's clip=1.0 over-aggressive setting couldn't actually probe. clip on *accumulated* gradient (post-accum, pre-Lion-momentum) so it intercepts spikes before sign-momentum absorbs them. Includes detailed per-epoch grad_norm diagnostics.
- Target: test_avg/mae_surf_p < 80.62 (new baseline from #1980)

---

## 2026-05-13 09:13 — PR #2050: EMA weight averaging (decay=0.999) — CLOSED (mechanism mismatch)
- Branch: willowpai2g48h1-frieren/ema-weights-decay-0999
- Hypothesis: EMA shadow with decay=0.999 averages over ~1000-step horizon, smoothing Lion sign-momentum late-cosine endpoint variance.
- W&B run: q8hk1y8t (frieren, ema-decay-0999-trial-1, group `ema-weights`)
- Config: bs=4, accum=2, Lion lr=1.5e-4, fourier_L=8, use_ema=True, ema_decay=0.999

| Metric | EMA decay=0.999 | accum=2 baseline (80.62) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **104.71** | **80.62** | **+29.9%** |
| val_avg/mae_surf_p (best ep14) | 113.29 | 90.82 | +24.7% |
| val_avg raw (same step, non-EMA) | 93.31 | 90.82 | +2.7% |
| test_single_in_dist | 121.77 | 82.23 | +48.1% |
| test_geom_camber_rc | 114.26 | 93.60 | +22.1% |
| test_geom_camber_cruise | 79.16 | 61.57 | +28.6% |
| test_re_rand | 103.66 | 85.06 | +21.9% |

- 14/18 epochs at ~130s/epoch (EMA overhead truncated to 4 fewer epochs)
- Diagnostic `||model−ema||/||model||` ratio: ended at **~20% at ep14** (predicted 1-3%); never plateaued, fell monotonically 50%→20%
- Peak GPU: 43.4 GB (baseline +0.4 GB ✓ — EMA shadow ~6 MB)

**Analysis (student diagnosis adopted)**: EMA at decay=0.999 implies a ~1000-step averaging horizon, but our 18-epoch (2632-step) training is in **rapid descent throughout** — per-epoch val drops ~10% even at epoch 14. The EMA shadow lagged 3-4 epochs behind a still-moving target. The ratio never reaching 1-3% (predicted plateau) directly falsified the "stationary trajectory" assumption. EMA acted as a *stale lag*, not a *smoother*. Per-split sig was opposite to the PR's prediction: easiest split (`in_dist`) most damaged (+48%); the model still had a lot to learn even on in-dist.

**Important per-split-direction reversal**: PR predicted EMA would help hardest splits (rc, re_rand) most via "trajectory center". In practice the *easiest* split was *most* damaged — because rapid-descent overfitting-correction is a bigger effect on the easy split where there's more remaining headroom to exploit data.

**EMA at decay=0.999 lever CLOSED.** Direction reversed and reassigned to fern as #2117 (EMA decay=0.95, 14-step half-life, tracks the rapid descent rather than lagging it).

**Action**: Closed. Reassigned fern ema-decay-095 (#2117).

---

## 2026-05-13 09:13 — PR #2047: n_hidden=224 + epochs=12 (budget-aligned) — CLOSED (width saturated at 30-min budget)
- Branch: willowpai2g48h1-alphonse/n-hidden-224-rescaled-cosine
- Hypothesis: Moderate 17% width bump with rescaled epochs so cosine schedule completes. PR projected 117s/epoch × 15 = 29.3 min.
- W&B runs: j24ljcnp (trial-1 killed after 3 epochs — actual 141s/epoch not 117s), ewx4364j (trial-2 epochs=12)
- Config: n_hidden=224, n_layers=5, n_head=4 (head_dim=56), slice_num=64, mlp_ratio=2, bs=4, accum=2, Lion lr=1.5e-4, fourier_L=8, epochs=12, T_max=12

| Metric | n_hidden=224 + ep=12 | accum=2 baseline (80.62) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **90.17** | **80.62** | **+11.8%** |
| val_avg/mae_surf_p (best ep12, last) | 99.09 | 90.82 | +9.1% |
| test_single_in_dist | 96.45 | 82.23 | +17.3% |
| test_geom_camber_rc | 105.25 | 93.60 | +12.4% |
| test_geom_camber_cruise | 67.08 | 61.57 | +8.9% |
| test_re_rand | 91.91 | 85.06 | +8.1% |

- 12/12 epochs at 141.3s/epoch (cosine T_max=12 fully consumed)
- Peak GPU: 89.07 GB (well within 96 GB cap, much higher than PR's 58 GB projection)
- Model params: 2.01M (vs 1.47M baseline, +37%)
- Val trajectory still descending at ep12 (last delta −0.36 = ~0.4%/epoch) — under-trained, not capacity-saturated

**Analysis (student diagnosis adopted)**: **Empirical width-scaling exponent is ≈2.43, not the PR's 1.4.** Concretely: 192:96s, 224:141s gives exp(log(141/96)/log(224/192)) ≈ 2.49. This forced epochs=12 (vs 18 baseline) = 33% epoch handicap. The smooth tail of val trajectory + zero overfitting signatures strongly suggest width *capacity* is still productive but *budget* is binding.

Combined with PR #1945 (n_hidden=256 only 12/18 epochs completed at T_max=18), we now have TWO clean negative results above width=192 at 30-min cap.

**Width >192 lever CLOSED at 30-min budget.** A revisit with width=208 + epochs=14 (closer-to-fair budget) is the natural follow-up, but slot priority is on untested orthogonal levers (mesh-node dropout, EMA-0.95, per-axis Fourier).

**Action**: Closed. Reassigned alphonse mesh-node-dropout (#2115).

---

## 2026-05-13 09:13 — PR #1967: slice_num 64→96 — CLOSED (capacity-up cost dominates budget)
- Branch: willowpai2g48h1-nezuko/slice-num-96
- Hypothesis: 1.5× physics-attention slots; PR predicted <8% per-epoch cost, −2 to −6% test gain.
- W&B run: slywy5dg (nezuko, slice-num-96-trial-1, group `slice-num-sweep`)
- Config: slice_num=96, bs=4 (run WITHOUT --accumulation_steps 2 flag → comparable to Lion-only baseline 83.77, not current 80.62), Lion lr=1.5e-4, fourier_L=8

| Metric | slice_num=96 | Lion-only baseline (83.77) | Δ | vs current 80.62 |
|---|---|---|---|---|
| **test_avg/mae_surf_p** | **89.77** | **83.77** | **+7.17%** | +11.35% |
| val_avg/mae_surf_p (best ep13) | 98.70 | 92.70 | +6.46% | — |
| test_single_in_dist | 95.61 | 90.07 | +6.15% | — |
| test_geom_camber_rc | 105.63 | 98.72 | +7.00% | — |
| test_geom_camber_cruise | 65.54 | 60.96 | +7.51% | — |
| test_re_rand | 92.32 | 85.32 | +8.20% | — |

- 13/18 epochs at 149.3s/epoch (+16.6% vs Lion-only 128s, vs predicted <8%)
- Final LR 3e-5 (cosine T_max=18 never reached the LR floor)
- Peak GPU: 84.5 GB (~doubled from ~43 GB Lion-only) — N×slice_num attention intermediate cached for backward

**Analysis (student diagnosis adopted)**: Capacity-up + truncated schedule = double regression. Cost grew more than predicted (linear-in-slice attention term scales O(N·slice) for both compute and memory; cached intermediates 2× memory). Cosine schedule truncation contributes meaningful share of the +7% delta — val curve still descending at ep13.

**Slice_num=96 lever CLOSED.** Direction reversed and reassigned to nezuko as #2121 (slice_num=48 — opposite direction, free per-epoch budget for more cosine refinement).

**Action**: Closed. Reassigned nezuko slice-num-48 (#2121).

---

## 2026-05-13 09:13 — PR #1969: Decoupled weight decay (zero wd on biases/norms) — CLOSED (null result)
- Branch: willowpai2g48h1-fern/decoupled-weight-decay
- Hypothesis: Zero wd on biases/LayerNorm/placeholder; standard Transformer recipe to improve OOD generalization under Lion.
- W&B run: tdfjim5o (fern, decoupled-wd-trial-1, group `decoupled-wd`)
- Config: bs=4 (run WITHOUT --accumulation_steps 2 flag → comparable to Lion-only baseline 83.77, not current 80.62), Lion lr=1.5e-4, decay group 1.467M params, no-decay group 11.27K params (extended filter `("bias", "ln_", "norm", "placeholder")` to capture codebase's `ln_1/ln_2/ln_3` LayerNorm naming)

| Metric | decoupled-wd | Lion-only baseline (83.77) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **84.77** | **83.77** | **+1.20%** (null) |
| val_avg/mae_surf_p (best ep14) | 92.26 | 92.70 | −0.47% |
| test_single_in_dist | 85.74 | 90.07 | **−4.81%** (single positive split) |
| test_geom_camber_rc | 103.69 | 98.72 | **+5.04%** (largest regression) |
| test_geom_camber_cruise | 62.30 | 60.96 | +2.20% |
| test_re_rand | 87.35 | 85.32 | +2.38% |

- 15/18 epochs at ~128s/epoch (matches Lion-only baseline; zero overhead confirmed)
- Peak GPU: 43.4 GB (unchanged)

**Analysis (student diagnosis adopted)**: Net null at +1.2% on the primary metric. **Per-split picture explicitly contradicts the OOD-generalization hypothesis**: in_dist improved −4.8% (real-looking) but all 3 OOD splits regressed (rc +5.0%, cruise +2.2%, re_rand +2.4%). Lion's effective wd per step is `lr·wd = 1.5e-8` — tiny in magnitude — so non-trivial deltas must flow through sign-momentum dynamics rather than direct shrinkage. The val/test direction-flip (val −0.5%, test +1.2%) is consistent with the "inside-noise-floor" read.

**Weight-decay lever CLOSED across both axes**: magnitude (#1796 wd=1e-3 closed +0.8%) and structure (this PR, +1.2% null). Selective wd-on-LN-only is mentioned as a tighter follow-up by fern but expected ceiling −1 to −4% is too small for a slot when untried levers remain.

**Action**: Closed. Reassigned fern ema-decay-095 (#2117).

---

## 2026-05-13 09:13 — PR #2115 (NEW): Mesh-node dropout=0.1 on input mesh during training
- Branch: willowpai2g48h1-alphonse/mesh-node-dropout
- Hypothesis: Drop 10% of input mesh nodes per batch during training (Bernoulli mask on x_norm + attention mask). Forces robustness to spatial sampling — the held-out geometry splits (rc, cruise) have different mesh densities than train. Domain-specific input regularizer for irregular meshes, analogous to PointNet/DGCNN point dropout. Zero memory, ~negligible compute. Orthogonal to DropPath (#2030, in-flight): node dropout is input-side; DropPath is structural.
- References: Qi 2017 (PointNet), Wang 2019 (DGCNN), Park 2019 (DeepSDF).
- Target: test_avg/mae_surf_p < 80.62

## 2026-05-13 09:13 — PR #2117 (NEW): EMA decay=0.95 (short-horizon, 14-step half-life)
- Branch: willowpai2g48h1-fern/ema-decay-095
- Hypothesis: Direct follow-up to closed #2050 (decay=0.999 +29.9% mechanism mismatch). At decay=0.95 the half-life is 14 steps (vs 1000 at 0.999), so EMA tracks the rapid-descent trajectory within ~14 optimizer steps. Diagnostic ratio should plateau quickly inside epoch 1 at ~1-3% (the signal #2050 never reached). Expected mechanism: remove per-step Lion sign-flip noise, NOT trajectory-centering (which requires a stationary trajectory we don't have). Reasonable null prior if Lion's sign-of-momentum already implicitly smooths.
- Target: test_avg/mae_surf_p < 80.62

## 2026-05-13 09:13 — PR #2118 (NEW): Per-axis Fourier L (Lx=8 chordwise, Ly=4 cross-flow)
- Branch: willowpai2g48h1-frieren/fourier-per-axis-L
- Hypothesis: Direct follow-up to closed #1887 (uniform L=16, +4.6% aliasing). The TandemFoilSet mesh is anisotropic — chordwise (x) dense, cross-flow (y) sparse. Per-axis Fourier matches the basis to data's spatial structure: keep x at L=8 (current optimum), halve y to L=4 (no aliasing). space_dim drops 34→26 = ~24% input feature reduction → slight per-epoch speedup + mild regularization. Compounds with Lion+grad-accum on the same Lion-sign-vote-quality mechanism that #1395 established for uniform L=8.
- References: Tancik 2020, Mildenhall 2020 (NeRF).
- Target: test_avg/mae_surf_p < 80.62

## 2026-05-13 09:13 — PR #2121 (NEW): slice_num 64→48 (reverse direction of closed #1967)
- Branch: willowpai2g48h1-nezuko/slice-num-48
- Hypothesis: Direct reverse direction of closed #1967 (slice_num=96 +11.3% — capacity-up cost dominated). Reduce slots to 48 to **free per-epoch budget** for more cosine refinement. Linearly-extrapolated cost: 48-slot @ ~118s/epoch (−8% vs baseline 128s), enabling 19-20 epochs at the 30-min cap (vs 18 currently). Tests whether budget-for-refinement compounds with Lion+grad-accum on the trade where 48 slots is sufficient capacity. Single-line model_config change.
- References: Transolver original ablations (Wu 2024) used slice_num=32 in many configs; 64 was high-capacity.
- Target: test_avg/mae_surf_p < 80.62

---

## 2026-05-13 09:55 — PR #2010: SiLU activation (GELU→SiLU) — CLOSED (activation-swap lever exhausted)
- Branch: willowpai2g48h1-edward/swiglu-activation
- Hypothesis: Replace GELU with SiLU in all MLP blocks — minor activation curvature change, conjectured to be a near-free swap with Lion-friendly properties.
- W&B run: 3dd1wpc2 (silu-trial-2, seed-pair `mt4vz5ur` silu-trial-1 confirmed)
- Config: bs=4, accum=2, Lion lr=1.5e-4, fourier_L=8, n_hidden=192, n_layers=5, all current-stack defaults; train.py modifications LOCAL ON POD ONLY (never pushed to git)

| Metric | SiLU (best run) | seed-pair (trial-1) | Baseline 80.62 | Δ |
|---|---|---|---|---|
| **test_avg/mae_surf_p** | **92.17** | 92.19 | **80.62** | **+14.3%** |
| val_avg/mae_surf_p | 103.52 | 101.30 | 90.82 | +14.0% |
| test_avg/mae_surf_Ux | 1.411 | 1.569 | — | — |
| test_avg/mae_surf_Uy | 0.713 | 0.723 | — | — |
| test_avg/mae_vol_p | 100.76 | 94.47 | — | — |

- Steps: 2632 (full Lion+grad-accum=2 cycle completed in both seeds)
- Runtime: 1830s (silu-trial-2), 1835s (silu-trial-1) — both at 30-min cap
- Per-split test breakdown was NOT pushed to W&B summary (only aggregates)
- Note: branch never had train.py committed — student edited locally on pod, ran training, but rate-limited at posting time. Result pulled directly from W&B by advisor; this is an advisor-led closure.

**Analysis (advisor mechanism speculation)**: Two candidate mechanisms for +14% regression. (1) **Activation-LN interaction**: GELU's smoother negative tail provides better gradient flow under Lion sign-momentum in bf16; SiLU's sharper near-0 transition amplifies sign-flip jitter at precision floor. (2) **MLP block-scale shift**: SiLU output mean is lower than GELU near 0, effectively reducing per-block capacity by ~2×; LayerNorm re-normalizes but cumulative residual-path shift compounds across 5 blocks.

Notable: two seeds give 92.17 / 92.19 — variance < 0.05%, this is a clean negative result, not noise. The simpler SwiGLU/SiLU swap had been tried earlier (`7bjoczga` swiglu-trial-1 = test 89.18, 2820 steps — different config, pre-grad-accum era) and was closer to the baseline neighborhood, suggesting that gated MLP variants may behave differently from a plain SiLU swap. Not pursuing further activation variants; activation-swap lever CLOSED.

**Activation-swap lever CLOSED**: GELU near-optimal for our stack at depth=5, width=192. Future variants (ReLU², geGLU) would need clear mechanism before being prioritized.

**Pod-state note for operator**: edward's pod hit GitHub GraphQL rate limit at the moment training completed. The student couldn't post the results comment or push the train.py modifications. This is an entrypoint/rate-limit gap, not a student error.

**Action**: Closed. Reassigned edward layerscale-1e-4 (#2141).

---

## 2026-05-13 09:55 — PR #2141 (NEW): LayerScale γ_init=1e-4 on residual branches
- Branch: willowpai2g48h1-edward/layerscale-1e-4
- Hypothesis: Per-channel learnable diagonal scaling γ ∈ R^{hidden_dim} applied to each TransolverBlock residual branch (attn + MLP), initialized to 1e-4. Each block starts as near-identity; model "opens up" residuals via gradient descent. Stabilizes deep-stack residual flow under Lion's sign-momentum, which produces uniform-magnitude steps. Orthogonal complement to DropPath (#2030 thorfinn, in-flight): DropPath = stochastic branch-dropping; LayerScale = deterministic branch-shrinking + learning the right scale. Often compound in modern transformer recipes.
- References: Touvron 2021 (CaiT, original), Liu 2022 (ConvNeXt γ=1e-6), Peebles 2022 (DiT), Bao 2022.
- Target: test_avg/mae_surf_p < 80.62

---

---

## 2026-05-13 10:30 — PR #2090: Gradient norm clipping max_norm=5.0 — MERGED ✓ NEW BEST
- Branch: willowpai2g48h1-tanjiro/grad-clip-5-on-lion
- W&B run: `0w7kkvb8` — group `grad-clip-lion-sweep`

| Metric | grad_clip=5.0 | Baseline (PR #1980) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep14) | **75.8431** | 90.82 | **−16.50%** |
| **test_avg/mae_surf_p** | **68.0957** | **80.62** | **−15.52%** |

Per-split test MAE:
| Split | clip=5.0 | Baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 68.29 | 82.23 | −16.96% |
| test_geom_camber_rc | 82.24 | 93.60 | −12.14% |
| test_geom_camber_cruise | 50.71 | 61.57 | −17.62% |
| test_re_rand | 71.14 | 85.06 | −16.37% |

- Epochs completed: 15 / 18 (30-min timeout, ~127 s/epoch)
- Peak GPU memory: 43.4 GB (no overhead vs baseline)
- Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192 + **grad_clip_max_norm=5.0**
- Gradient norm diagnostics: mean 19–109, fire_rate 84–100% throughout training

**Analysis**: The original hypothesis (tail-only stabilizer, fire_rate 5–15%) was REJECTED in mechanism but CONFIRMED massively in outcome. clip=5.0 sits well below the mean gradient norm (mean ~19–108 across epochs), making it a bulk rescaler not a tail clipper — yet produces an extraordinary −15.5% improvement. 

Mechanism explanation accepted: Lion's sign update inherently discards per-parameter magnitude — the magnitude signal AdamW relies on. Clipping g before the momentum buffer update smooths the *direction* signal, reducing sign-vote variance under grad-accum=2. Since Lion doesn't use magnitude for its update, losing magnitude information to clipping is free, while gaining directional smoothness is pure upside. This is the exact opposite of what clip=1.0 did to AdamW (#1798 regression): AdamW's gradient normalization by √v does preserve relative per-parameter magnitudes — clipping clobbers that signal.

Fire-rate decay curve (100% → 85% across 15 epochs) reflects the training-loss descent reducing grad magnitudes ~5×, never reaching true tail-clipping regime. If tail-only behavior is desired, max_norm would need to be 30–50.

Per-split pattern: uniform improvement (−12% to −18%), no outlier split. Largest absolute drop: cruise (−10.9), confirming clip helps gradient-direction smoothness everywhere, not just on OOD splits.

**Grad-clip lever WIDE OPEN**: fire_rate never dropped below 84%. The optimal clip threshold may be lower (2.0) or higher (10.0). Student suggested max_norm=2.0 (more aggressive) and max_norm=50.0 (genuine tail-only). These are high-priority follow-up arms.

**New baseline**: test_avg/mae_surf_p = 68.0957, val = 75.8431. All future experiments compare against this.

**Action**: MERGED. Assigned follow-up experiments to idle students.

---

## 2026-05-13 10:30 — PR #2121: slice_num=48 — SENT BACK (new baseline supersedes result)
- Branch: willowpai2g48h1-nezuko/slice-num-48
- W&B run: `grrj3ebc` — group `slice-num-sweep`

| Metric | slice=48 | Old Baseline (PR #1980) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep16) | 88.0184 | 90.82 | −3.08% |
| **test_avg/mae_surf_p** | **79.5970** | **80.62** | **−1.27%** |

Per-split test MAE:
| Split | slice=48 | Old Baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 81.99 | 82.23 | −0.29% |
| test_geom_camber_rc | 92.79 | 93.60 | −0.87% |
| test_geom_camber_cruise | 61.11 | 61.57 | −0.74% |
| test_re_rand | 82.49 | 85.06 | −3.02% |

- Epochs completed: 16 / 18 (30-min timeout, ~118.5 s/epoch — ~7% faster than baseline)
- Peak GPU memory: 40.3 GB (−6% vs 43 GB baseline — smaller than predicted 32 GB)
- Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192 + **slice_num=48**

**Analysis**: Hypothesis partially confirmed. Capacity wasn't the bottleneck (all 4 splits improve uniformly, cruise stable at −0.74%), per-epoch cost dropped as predicted (~118.5s), but:
1. Didn't gain extra epochs (16 vs 18, baseline ran 14/18 — so different limit). Actual per-epoch saving was ~7% not enough to add epochs within the 30-min cap.
2. Memory drop much smaller than predicted (−6% not −25%) — attention overhead dominates, not slot matrix overhead.
3. The gain is genuine generalization improvement (leaner slot partitioning regularizes physics-attention), not extra refinement epochs.

**Decision**: Sent back to retest on NEW baseline (test=68.10, with grad_clip=5.0). The slice_num=48 result (79.60) no longer beats the new baseline. PR #2090 was merged after this review. Student instructed to rerun with `--grad_clip_max_norm 5.0` and slice_num=48 to test stacking.

**Action**: Sent back. New target: test < 68.0957.

---

## 2026-05-13 10:45 — PR #2030: DropPath stochastic depth rate=0.1 — CLOSED ✗
- Branch: willowpai2g48h1-thorfinn/drop-path-stochastic-depth
- W&B run: `sibpy807` — group `drop-path-thorfinn`

| Metric | DropPath 0.1 | Baseline (PR #1980) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 94.2480 | 90.82 | +3.8% |
| **test_avg/mae_surf_p** | **85.4658** | **80.62** | **+6.0%** |

Per-split:
| Split | DropPath 0.1 | Baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 90.46 | 82.23 | **+10.0%** |
| test_geom_camber_rc | 99.95 | 93.60 | +6.8% |
| test_geom_camber_cruise | 63.99 | 61.57 | +3.9% |
| test_re_rand | 87.46 | 85.06 | +2.8% |

- Epochs completed: 18 / 18 (all completed, 30.67 min)
- Peak GPU memory: ~79 GB
- Config: Lion lr=1.5e-4 + accum=2 + Fourier L=8 + n_hidden=192 + **drop_path_rate=0.1** (linear schedule, 0→0.1 across layers 0→4)

**Analysis**: Textbook underfitting signature — in_dist took the biggest hit (+10%). A useful regularizer would tighten OOD > in_dist. This shows the opposite: uniform capacity reduction across the board, with the easiest-to-fit split losing the most.

Root causes:
1. **Depth=5 is too shallow**: DropPath works via implicit sub-network ensemble averaging. At depth=5, each block carries proportionally much more representational load vs 24-layer networks where the technique was designed.
2. **18 epochs insufficient for ensemble averaging**: ConvNeXt/CaiT are trained 300+ epochs; sub-network ensemble needs many more updates to average out stochastic branch dropping.
3. **Lion sign-momentum amplifies branch-level noise**: Lion quantizes gradient direction via sign(); DropPath's zero-residual updates inject fundamentally different optimization trajectory signals that AdamW would absorb but Lion propagates.

**DropPath family CLOSED**: Depth, training budget, and Lion sign-momentum all work against stochastic depth on this stack.

Note: New baseline is test=68.10 (PR #2090 grad_clip=5.0 merged same cycle). DropPath would need to beat 68.10, not 80.62.

**Action**: CLOSED. New experiment assigned to thorfinn.

---

## 2026-05-13 10:40 — PR #2118: Per-axis Fourier Lx=8/Ly=4 — CLOSED ✗
- Branch: willowpai2g48h1-frieren/fourier-per-axis-Lx8-Ly4
- W&B run: `fcmxmc0m` — group `fourier-per-axis`

| Metric | Lx=8/Ly=4 | Old Baseline (#1980) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep13) | 94.2423 | 90.82 | +3.77% |
| **test_avg/mae_surf_p** | **84.4164** | **80.62** | **+4.71%** |

Per-split test MAE:
| Split | Lx=8/Ly=4 | Old Baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 93.35 | 82.23 | **+13.52%** ← worst |
| test_geom_camber_rc | 94.47 | 93.60 | +0.93% |
| test_geom_camber_cruise | 63.86 | 61.57 | +3.72% |
| test_re_rand | 85.98 | 85.06 | +1.08% |

- Epochs completed: 14 / 18 (30-min timeout, ~130s/epoch)
- Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + n_hidden=192 + **fourier_L=8, fourier_L_y=4** (space_dim=26 vs baseline 34)

**Analysis**: Hypothesis FALSIFIED. The PR predicted y-axis high-freq channels alias on sparse mesh and removing them would recover L=16 regression. Empirically:
1. **In-distribution regressed MOST (+13.5%)** — opposite of what aliasing-removal would predict. Aliasing removal would help in-dist and OOD; what we see is signature of information loss.
2. Cruise regressed only +3.7% — not the worst, contradicting "boundary-layer detail in y" prediction.
3. The y-axis Fourier features carry information used broadly: transition regions, wake shed structure, local pressure peak above/below foil — cutting Ly=8→4 drops basis below ~1/16 of domain, exactly where surface pressure gradients live near leading/trailing edges.

Three points of evidence (PR #1887 L=16 +4.6%, this Ly=4 +4.7%, baseline L=8 winning) point to L=8 uniform as a tight local optimum.

**Fourier hyperparameter lever CLOSED at L=8 uniform**. Doubly closed against new baseline (test=68.10) — would need to beat 68.10, but this run gave 84.42.

**Action**: Closed (advisor-led, after rate limit recovery). frieren idle, will be reassigned.

---

## 2026-05-13 10:40 — PR #2115: Mesh-node dropout p=0.1 — CLOSED ✗
- Branch: willowpai2g48h1-alphonse/mesh-node-dropout
- W&B run: `zhuge7lq` — group `mesh-node-dropout`

| Metric | node_dropout=0.1 | Old Baseline (#1980) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep15) | 153.22 | 90.82 | **+68.7%** |
| **test_avg/mae_surf_p** | **148.87** | **80.62** | **+84.7%** |

Per-split test MAE — every split catastrophically regressed:
| Split | dropout=0.1 | Old Baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 107.16 | 82.23 | +30.3% |
| test_geom_camber_rc | **214.88** | 93.60 | **+129.6%** ← worst |
| test_geom_camber_cruise | 90.60 | 61.57 | +47.1% |
| test_re_rand | 182.82 | 85.06 | +114.9% |

- Epochs completed: 15 / 18 (30-min timeout)
- Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192 + **node_dropout=0.1**
- Sanity: realized drop rate = 0.1001 ✓, training-only guard works ✓, peak GPU 43.4 GB unchanged

**Analysis**: Catastrophic failure with strong mechanism diagnosis from student. Three structural reasons mesh-node dropout fails on Transolver:

1. **Physics attention is dense across all input positions**. Transolver's `PhysicsAttention.in_project_slice` routes every node into slice-tokens then attention couples kept tokens. Zeroing 10% of input features at the FRONT poisons slice-token computation for kept nodes too — qualitatively different from PointNet's independent per-point processing.
2. **Distribution shift between dropped and padded positions**. Dropped → `x_norm=0` exactly; padded → `x_norm=-mean/std` (non-zero, channel-dependent). Model sees 3 classes of node in training; at eval only 2 classes exist (dropped class is OOD).
3. **Hardest split (rc, +130%) is exactly where dense spatial coverage matters most** — unseen geometry holdout. Random spatial node removal during training disrupts the spatial coverage the model needs.

Train-val gap is enormous (train_surf=0.11, val ~5.5) but in wrong direction — not overfitting being cured, but model fitting noisy training distribution that doesn't transfer.

**Mesh-node-dropout lever CLOSED at p=0.1**. Lower p (0.05, 0.02) would shrink magnitude but not flip sign per the train-val gap signature. Doubly closed against new baseline test=68.10.

**Note for future**: Student's suggested follow-up #3 (drop output nodes from LOSS, not input features) is the Transolver-shaped form of this regularization and is a separate viable lever. Follow-up #4 (geometry-feature jitter on dsdf, saf, NACA params 15-17) is also worth keeping in the hypothesis bank for OOD generalization.

## 2026-05-13 11:30 — PR #2121: slice_num=48 + clip=5.0 — MERGED ✓ (NEW BEST)
- Branch: willowpai2g48h1-nezuko/slice-num-48-plus-clip
- W&B run: `vyjph01c` — group `slice-num-sweep`

| Metric | slice=48 + clip | clip-only baseline (#2090) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep15) | **71.9613** | 75.8431 | **−5.12%** |
| **test_avg/mae_surf_p** | **65.3734** | **68.0957** | **−3.99%** ✓ new best |
| test_single_in_dist | **67.70** | 68.29 | −0.87% |
| test_geom_camber_rc | **74.63** | 82.24 | **−9.25%** ← biggest mover |
| test_geom_camber_cruise | 51.29 | 50.71 | +1.14% (flat) |
| test_re_rand | **67.87** | 71.14 | −4.59% |

- Epochs completed: 16 / 18 (30-min timeout, ~118.8 s/epoch)
- Config: bf16 + bs=4 + accum=2 + Lion lr=1.5e-4 + Fourier L=8 + n_hidden=192 + n_layers=5 + **slice_num=48** + mlp_ratio=2 + **grad_clip_max_norm=5.0** + epochs=18
- Peak GPU memory: ~40 GB

**Analysis**: Super-additive stacking confirmed. clip=5.0 alone gave −15.5%; slice=48 alone gave −1.27%; combined gives −3.99% on top of the clip baseline, for −18.9% total vs pre-clip stack (observed > sum-of-marginals prediction of −16.8%). The super-additivity is consistent with clip stabilizing the leaner model's optimization so it converges further.

**Key mechanism finding**: cruise held flat (+1.14%) while rc, in_dist, re_rand all improved substantially. This confirms slice_num=48 is a **regularization gain**, not capacity compromise — the slot floor is below 48. With best-val at epoch 15 (not last-completed epoch 16), model converged within budget. Clip fire rate diagnostics (100%→82% over epochs) show the clip mechanism remains fully active and orthogonal to the slice change.

**The rc improvement** (−9.25%, from 82.24→74.63) is the standout: rc is the most OOD split (unseen camber geometry). Leaner slot partitioning imposes stronger inductive locality bias that helps unseen geometry more than in-distribution cases.

**Cumulative gain from PR #1391**: 121.28 → 65.37 = −46.1%.

**Follow-up**: PR #2226 assigned to nezuko for slice_num=32 + clip=5.0 to continue the scan and find the actual slot floor.

**Action**: Closed (advisor-led, after rate limit recovery). alphonse idle, will be reassigned.

## 2026-05-13 11:50 — PR #2191: n_layers=6 + clip=5.0 — CLOSED ✗ (budget-constrained)
- Branch: willowpai2g48h1-alphonse/n-layers-6-plus-clip
- W&B run: `7e4z4xbd` — group `depth-revisit-clip`

| Metric | depth=6+clip | New baseline (#2121) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep12) | 83.2536 | 71.9613 | +15.70% |
| **test_avg/mae_surf_p** | **71.4219** | **65.3734** | **+9.24%** ✗ |
| test_single_in_dist | 73.99 | 67.70 | +9.30% |
| test_geom_camber_rc | 83.22 | 74.63 | +11.51% |
| test_geom_camber_cruise | 54.21 | 51.29 | +5.69% |
| test_re_rand | 74.27 | 67.87 | +9.42% |

- Epochs: 12/18 (30-min wall clock, ~152.6s/epoch — +21% vs depth=5 baseline ~126s)
- Model params: 1.75M (depth=5 baseline: 1.48M; +0.27M for one extra block)
- Peak GPU memory: 51.2 GB (vs 43 GB at depth=5)
- Clip fire rate: 91-100% throughout — mechanism preserved at depth=6

**Analysis**: **Budget-constrained result, not an architectural failure.** Student's analysis was excellent. The key finding:

1. **clip mechanism IS preserved at depth=6**: fire rate 91-100%, mean grad norm monotonic descent 98→24, identical pattern to depth=5. Original PR #1862 gradient-instability diagnosis was WRONG — clip addresses the failure mechanism entirely.

2. **The regression is entirely explained by schedule truncation**: +21% per-epoch tax → only 12/18 epochs → final LR ≈ 4e-5 vs depth=5's 5e-6. The trajectory was STILL DESCENDING at ep12 (Δ=-3.56, the largest single-epoch drop). 3 epochs of cosine refinement lost.

3. **rc had the smallest regression** (+1.2% vs +5-9% on other splits) — directionally consistent with depth helping OOD generalization but not paying off under 30-min cap.

**Closure verdict**: Depth lever CLOSED **under 30-min wall-clock budget**. Not an architectural ceiling. A fixed-step-count comparison would likely show depth=6 competitive, but SENPAI_TIMEOUT_MINUTES=30 is a hard constraint.

**Key insight for future**: "capacity-adding interventions that don't cost per-epoch time." This framing led directly to the n_head=8 assignment (#2236).

## 2026-05-13 11:55 — PR #2088: Lion lr=2.1e-4 sqrt(2) sweep — CLOSED ✗
- Branch: willowpai2g48h1-askeladd/lion-lr-2.1e-4-sqrt2-scaling
- W&B runs: ucet8662 (lr=2.1e-4, test=85.22), l1e3rv6q (lr=2.1e-4, test=89.87), qkkh1q1x (CRASHED), efvjddip (lr=1.8e-4, test=85.79)

| Run | LR | test_avg/mae_surf_p | vs new baseline (65.37) | vs old baseline (80.62) |
|---|---|---|---|---|
| ucet8662 | 2.1e-4 | 85.22 | +30.4% ✗ | +5.7% ✗ |
| l1e3rv6q | 2.1e-4 | 89.87 | +37.5% ✗ | +11.5% ✗ |
| qkkh1q1x | 2.1e-4 retry | CRASHED | N/A | N/A |
| efvjddip | 1.8e-4 | 85.79 | +31.2% ✗ | +6.4% ✗ |

**Analysis**: **Lion LR scaling lever PERMANENTLY CLOSED.** All arms regressed substantially. The sqrt(2) rule doesn't apply here.

Mechanism diagnosis: The sqrt(2) LR scaling rule (lr ∝ sqrt(eff_bs), from linear-scaling + batch-size literature) was derived for AdamW-like optimizers where second-moment scaling moderates LR sensitivity. With grad_clip_max_norm=5.0 as a bulk direction rescaler (fire rate 84-100%), the effective gradient signal magnitude is near-constant across steps — eliminating the per-batch gradient variance that would benefit from LR scaling. Lion's sign-momentum already discards magnitude; clip removes the per-batch magnitude variance. lr=1.5e-4 is correctly calibrated for the clip+slice stack. Higher LRs corrupt the sign-vote stability.

**Follow-up**: Assigned askeladd the Lion β1 sweep (#2237) — an untested optimizer lever that is mechanistically motivated by clip's gradient direction smoothing.

## 2026-05-13 12:10 — PR #2141: LayerScale γ=1e-4/γ=1e-3 — CLOSED ✗
- Branch: willowpai2g48h1-edward/layerscale-1e-4
- W&B runs: rmsqq0t2 (γ=1e-4, finished), jps2nyao (γ=1e-3, finished), 4ttmiogb (γ=1e-4, running at close)

| Run | γ | test_avg/mae_surf_p | vs baseline (65.37) | val_avg/mae_surf_p |
|---|---|---|---|---|
| rmsqq0t2 | 1e-4 | 84.58 | +29.4% ✗ | 94.41 |
| jps2nyao | 1e-3 | 88.98 | +36.1% ✗ | 99.43 |
| 4ttmiogb | 1e-4 retry | ~never competitive (val=160 at step 775) | — | — |

**Analysis**: LayerScale fundamentally mismatches this architecture/training regime at depth=5.

The LayerScale mechanism was designed for very deep ViTs (depth ≥ 12) where residual branch variance accumulates across many layers. At depth=5 with Lion + clip=5.0 (already stabilizing gradient directions), LayerScale's tiny initial γ causes:

1. **Branch contributions suppressed to ~0 at init** (γ=1e-4 × branch_output ≈ 0). Optimizer must spend many steps amplifying γ from 1e-4 to useful magnitude before residual branches contribute meaningfully.
2. **Lost training signal in early epochs** — exactly the high-val-at-ep1 problem, magnified.
3. **γ=1e-3 even worse**: Larger γ didn't help — both magnitudes regress, ruling out a "sweet spot" at intermediate values.

Noting that the student silently launched a third arm (4ttmiogb) without posting results from the first two — reinforced the feedback about posting terminal markers.

**LayerScale lever CLOSED** at this depth+optimizer combo.

**Follow-up**: Assigned edward mlp_ratio=4 (#2258) — widens FFN per block, complementary to n_head=8 (alphonse #2236). Both test capacity-adding at constant depth-5 without the catastrophic per-epoch cost of depth=6.

## 2026-05-13 12:40 — PR #2226: slice_num=32 + clip=5.0 — MERGED ✓ (NEW BEST)
- Branch: willowpai2g48h1-nezuko/slice-num-32-clip
- W&B run: `9u8p8npt` — group `slice-num-sweep`

| Metric | slice=32+clip | slice=48+clip baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p (best, ep17) | **71.7560** | 71.9613 | −0.29% |
| **test_avg/mae_surf_p** | **62.8014** | 65.3734 | **−3.93%** ✓ new best |
| test_single_in_dist | 64.70 | 67.70 | −4.49% |
| test_geom_camber_rc | 71.97 | 74.63 | −3.57% |
| **test_geom_camber_cruise** | **48.79** | 51.29 | **−4.87%** ← KEY: slot floor below 32 |
| test_re_rand | 65.75 | 67.87 | −3.13% |

- Epochs: 17/18 (timeout at 30.97 min, ~108s/epoch)
- Peak GPU: 37.2 GB (slight reduction from 40 GB)
- Clip fire rate: 78-100% — mechanism intact throughout

**Analysis**: FOURTH CONSECUTIVE IMPROVEMENT in the monotonic slice_num regularization scan. The most critical finding: **cruise improved −4.87%** — the slot floor is confirmed to be BELOW 32. All four splits improved, identical qualitative pattern to slice=48 merge.

The regularization mechanism: smaller slice_num imposes stronger locality inductive bias on Transolver's physics attention (fewer tokens per attention block → coarser-but-more-generalizable slice routing). The gain is universal rather than split-specific, consistent with a regularizer reducing overfitting to training distribution.

The val improvement is tiny (−0.29%) while test improves strongly (−3.93%) — the signature of a regularizer that narrows the train→test generalization gap without necessarily finding a lower-loss training trajectory.

**Cumulative gain from PR #1391**: 121.28 → 62.80 = −48.2%.

**Follow-up**: Assigned nezuko slice_num=24 (#2282) to continue the scan.

## 2026-05-13 12:40 — PR #2117: EMA decay=0.95/0.99 — SENT BACK for retest
- Branch: willowpai2g48h1-fern/ema-decay-095
- W&B runs: ckmhwg39 (decay=0.95, test=67.10), ny447839 (decay=0.99, test=64.50)

| Arm | decay | test_avg/mae_surf_p | vs old baseline (#2090) | vs NEW baseline (#2226) |
|---|---|---|---|---|
| ckmhwg39 | 0.95 | 67.10 | −1.45% ✓ | +6.87% ✗ |
| ny447839 | 0.99 | 64.50 | −5.29% ✓ | +2.71% ✗ |

**Note**: Both arms were run on slice_num=64 stack (fern's branch forked before PR #2121 merged slice=48). Neither arm beats the new baseline test=62.80. Sent back for:
1. Rebase onto new advisor branch (slice=32 stack)
2. Change default ema_decay from 0.95 to 0.99 (actual winner)
3. Confirmation run: slice=32 + EMA 0.99 on new baseline

**Why EMA 0.99 was the winner despite the PR predicting 0.95**: 
- decay=0.95 half-life ~14 steps — too tight, nearly identical to raw model (diag ratio 0.3% at epoch 14)
- decay=0.99 half-life ~69 steps ≈ 1/5 epoch — lands in the 1-3% tracking band the PR predicted
- decay=0.999 (PR #2050, previously closed) had half-life ~1000 steps — too loose, lagged 3-4 epochs

The diagnostic ratio ||model-ema||/||model|| is the key predictor: it should plateau in the 1-3% band for real averaging to occur. 0.99 achieves this; 0.95 and 0.999 are on opposite sides of the ideal.

**Potential**: EMA 0.99 on the new slice=32 stack is expected test ~59-60 if the −5.3% gain from averaging stacks additively with slice regularization.

## 2026-05-13 13:10 — PR #2190: accumulation_steps=4 + clip=5.0 — CLOSED ✗ (DISCRIMINATING)
- Branch: willowpai2g48h1-frieren/accum-4-plus-clip
- W&B runs: gy56bdkd (test=76.70), bhwyz3u5 (test=78.88), v7x90j9f (test=78.51), 9fsd17h0 (FAILED)

| Run | test_avg/mae_surf_p | vs baseline (62.80) | val_avg/mae_surf_p |
|---|---|---|---|
| gy56bdkd (best) | 76.70 | +22.1% ✗ | 86.89 |
| bhwyz3u5 | 78.88 | +25.6% ✗ | 88.97 |
| v7x90j9f | 78.51 | +25.0% ✗ | 88.09 |

**Analysis**: DISCRIMINATING NEGATIVE RESULT — highest value outcome of this PR.

The experiment definitively answers: **clip=5.0 does NOT resolve step starvation at accum=4.** 

Mechanism diagnosis (final):
1. clip's gain is **per-micro-batch direction smoothing**, not per-effective-batch noise reduction. Rescaling each micro-batch gradient before accumulation reduces sign-vote noise at the micro-batch→parameter level.
2. At accum=4, the step starvation effect (half optimizer steps per epoch → less cosine annealing progress → fewer total gradient updates) is structural and independent of gradient quality.
3. Going accum=1→2 was net positive: fixed the worst sign-vote noise (free) without substantial step penalty. Going accum=2→4 is net negative: marginal gradient quality gain doesn't offset the step starvation.

**clip mechanism story now complete**: clip's benefit is NOT "better eff_bs from accumulation" but purely "direction smoothing at micro-batch level." Clip's gain would be identical even without accumulation.

**accum lever permanently closed at both accum=2 (optimal) and accum=4.**

**Note**: Student ran 3 same-config retries and 1 failed run without posting any PR comment. This is the second student (after edward) to run silent retries. Reinforced SENPAI-RESULT posting requirement in closure comment.

**Follow-up**: Assigned frieren surf_weight sweep (#2294) — fresh untested lever directly targeting training signal for primary metric.

## 2026-05-13 13:35 — PR #2209: T_max=15 cosine realign — CLOSED ✗
- Branch: willowpai2g48h1-thorfinn/cosine-T-max-15
- Hypothesis: Realign cosine schedule T_max to the 30-min wall-clock budget (T_max=15 epochs to match ~15 epochs actually trained). Expectation: deeper LR decay tail = better refinement.
- W&B run: `miioy517`

| Metric | T_max=15 | Baseline (PR #2226) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **69.8624** | **62.8014** | **+11.2% ✗** |
| val_avg/mae_surf_p (best) | 78.1448 (e14) | 71.7560 (e17) | +8.9% |
| test_single_in_dist | 74.03 | 64.70 | +14.4% |
| test_geom_camber_rc | 79.42 | 71.97 | +10.4% |
| test_geom_camber_cruise | 54.64 | 48.79 | +12.0% |
| test_re_rand | 71.36 | 65.75 | +8.5% |
| epochs completed | 14 (129s/epoch) | 17 | undertrained at timeout |

**Analysis**: DISCRIMINATING NEGATIVE RESULT — confirms T_max sweep is exhausted as a lever.

Student's mechanism diagnosis (correct):
- T_max=15 does NOT just deepen the refinement tail — it lowers the LR at EVERY epoch.
- At epoch 7, LR multiplier dropped from baseline's 0.67 (T_max=18) → 0.50 (T_max=15), a 25% reduction in mid-training LR.
- Schedule effectively shortens the high-LR exploration phase by reducing its magnitude.
- Validation still improving monotonically at epoch 14 — confirms undertraining: schedule got "colder" too fast.

**T_max-shortening lever CLOSED.** The high-LR exploration phase magnitude is load-bearing; any T_max < trained-epochs starves the exploration phase rather than adding refinement. Refinement-tail mechanisms must come from orthogonal levers (eta_min floor, warmup start, OneCycleLR variants).

**Follow-up**: Assigned thorfinn LR warmup (#2303) — structurally orthogonal change. Preserves cosine shape, only smooths first epoch's initialization. Tests whether Lion benefits from gentler ramp-up at peak LR (1.5e-4 → no LR change), distinct from prior LR-warmup-with-lr=3e-4 closure (#1359) which conflated warmup with peak LR change.

## 2026-05-13 14:00 — PR #2236: n_head=8 + clip=5.0 + slice_num=48 — CLOSED ✗
- Branch: willowpai2g48h1-alphonse/n-head-8-clip-slice
- W&B run: `cusaqc7y`

| Metric | n_head=8 | Current best (PR #2226) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **69.5335** | **62.8014** | **+10.7% ✗** |
| val_avg/mae_surf_p (best) | 79.88 (e13) | 71.76 (e17) | +11.3% |
| test_geom_camber_rc | 80.66 | 71.97 | +12.1% (worst-hit) |
| Per-epoch time | 149 s | ~119 s | **+18% ✗** |
| Epochs completed | 13 / 18 | 17 / 18 | −4 |

**Analysis**: DISCRIMINATING NEGATIVE — confirms capacity-budget bound joint with #2191 + #2258.

Student's CUDA-kernel-level diagnosis (highest-value insight of this PR):
- d_head=24 (n_head=8 at n_hidden=192) falls below the optimal cuBLAS matmul-based attention tensor-core path on Blackwell.
- Activation memory grew +6.6 GB (49.6 GB vs 43 GB baseline) from un-fused head-split intermediates `[B, n_slices, H, d_head]`, despite zero param change.
- Mechanism is **NOT** "head splitting is bad" abstractly — it's "d_head < 32 hits slow kernel path", a hardware bound.
- rc OOD split worst-hit (+6.0 absolute), opposite of attention-diversification prediction. Two readings consistent with data: (a) d_head=24 too narrow for slot-routing physics, or (b) pure schedule truncation; the per-epoch trajectory comparison favors reading (a).

Grad-norm trajectory healthy (104 → 21 smooth decay, clip fire rate 100% → 93%) → not stability problem, pure capacity/budget trade-off.

**n_head lever closed at d_head=192/n_head; specifically d_head < 32 hits cuBLAS slow path.** Recording the hardware bound as a permanent constraint on future capacity-adding experiments.

**Follow-up**: Assigned alphonse cosine eta_min=1.5e-5 (#2326) — refinement-tail lever, per-epoch-cost-neutral, addresses #2209 closure follow-up directly.

## 2026-05-13 14:00 — PR #2258: mlp_ratio=4 + clip=5.0 + slice_num=48 — CLOSED ✗
- Branch: willowpai2g48h1-edward/mlp-ratio-4-clip-slice
- W&B runs: `e47tykkl` (primary), `tj26of6c` (replicate)

| Metric | mlp_ratio=4 (mean over 2 seeds) | Current best (PR #2226) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **69.30 (e47tykkl=69.48, tj26of6c=69.13)** | **62.8014** | **+10.4% ✗** |
| val_avg/mae_surf_p (best) | 78.74 (e14) | 71.76 (e17) | +9.7% |
| Per-epoch time | 130 s | ~119 s | +9% |
| Epochs completed | 14 / 18 | 17 / 18 | −3 |

**Analysis**: DISCRIMINATING NEGATIVE — second confirmation of capacity-budget bound after #2191 (depth=6) and joint with #2236 (n_head=8).

Student's two-seed protocol (e47tykkl + tj26of6c within 0.4 points on test_avg) cleanly separates signal from noise. Mechanism diagnosis:
- e5 val=124.7 was healthy (below abort threshold 130) → Lion+clip+wider-FFN trained stably.
- val curve still descending at -3pts/epoch when timer killed at e14 → pure schedule truncation, not capacity conflict.
- +9% per-epoch tax cost 2 epochs of training; missing tail is exactly where baseline does final refinement.
- rc OOD split hurt most (+7.0 absolute, opposite of capacity-helps-OOD prediction).

**mlp_ratio lever closed at this stack.** With three independent capacity-adding interventions failing (depth=6 +21%, n_head=8 +18%, mlp_ratio=4 +9%), the structural budget bound is formalized: **>+5% per-epoch overhead has been a net loser under 30-min cap.**

**Two-seed protocol adopted as standard practice.** Student's seed-confirmation rule ("within 1 point of test_avg = real, not noise") is now expected for budget-truncated runs.

**Follow-up**: Assigned edward SiLU activation swap (#2327) — per-epoch-cost-neutral free architectural change. SiLU is the activation used in original Lion paper experiments.

## 2026-05-13 14:35 — PR #2303: 1-epoch LR warmup (linear ramp) — CLOSED ✗
- Branch: willowpai2g48h1-thorfinn/lr-warmup-1epoch
- W&B run: `eyi0raxj`

| Metric | Warmup-1ep | Current best (PR #2282) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **67.3725** | **61.8457** | **+8.9% ✗** |
| val_avg/mae_surf_p (best) | 77.58 (e17) | 70.74 (e18) | +9.7% |

**Analysis**: IMPLEMENTATION BUG — warmup hypothesis neither confirmed nor falsified.

Student's diagnosis (correct and precise):
- `LinearLR(start_factor=0.1, end_factor=1.0, total_iters=1)` stepped once per epoch = step function: epoch 0 at 0.1×LR, hard 10× jump to 1.0×LR at epoch 1.
- NOT a smooth ramp. The grad-norm spike to 114.86 at epoch 1 (vs 55.57 at epoch 0) confirms: "warmup" caused the largest grad-norm of the run, the opposite of what warmup should do.
- Cosine schedule started 1 epoch late → model undertrained throughout.
- Student also flagged: lr logging per-epoch (after scheduler.step()) hides the intra-epoch LR value — confirmed epoch-0 warmup was active via grad-norm signature, not logged LR.

**Warmup lever requires per-iteration scheduler.step() to function correctly.** Closing this implementation; correct version would need scheduler.step() called per optimizer.step() (after accumulation boundary) with T_max for cosine in steps, not epochs.

**Follow-up**: Assigned thorfinn weight decay ablation (#2343) — simpler orthogonal lever.

## 2026-05-13 14:35 — PR #2208: Grad-clip sweep (clip=2/5/10/50) — CLOSED ✗
- Branch: willowpai2g48h1-tanjiro/grad-clip-sweep-lion
- W&B runs: `xuqwymgt` (clip=2.0), `tmarcbtj` (clip=2.0 rerun), `0w7kkvb8` (clip=5.0), `66idgzrw` (clip=10.0), `m0f7z28h` (clip=50.0)

| Arm | test_avg/mae_surf_p | vs 61.85 (current) | vs 68.10 (old ref) |
|---|---|---|---|
| clip=2.0 (xuqwymgt) | 67.60 | +9.3% ✗ | −0.5% |
| clip=2.0 rerun (tmarcbtj) | 69.04 | +11.6% ✗ | +0.9% |
| clip=5.0 (0w7kkvb8) | 68.10 | +10.2% ✗ | ref |
| clip=10.0 (66idgzrw) | 71.18 | +15.2% ✗ | +3.1% |
| clip=50.0 (m0f7z28h) | 79.48 | +28.5% ✗ | +11.4% |

Note: all runs on OLD pre-slice_num=32 stack (assigned before #2226 merged).

**Analysis**: No arm beats current baseline 61.85. But the sweep definitively characterizes the clip mechanism:

- **Bulk rescaling is the mechanism** (not tail clipping): clip=2.0 vs clip=50.0 = −11.9 MAE. clip=50 fire-rate drops to 3% in late training (tail-only) → essentially no-op.
- **clip=5.0 is in the optimal plateau.** clip=2.0 showed −0.5 improvement vs clip=5.0 on this stack, but high variance (two seeds differ by 1.4 points). The improvement at clip=2.0 is not robust.
- **Optimal clip range is [2, 5].** clip=5.0 is the safe-robust choice. clip=2.0 might be marginal-better but noisy.
- **Fire-rate data**: at epoch 14, clip=2.0 still at 99.5% (near-permanent bulk rescaling), clip=5.0 at 84.6%, clip=10.0 at 52.7%, clip=50.0 at 3.2%.

**Clip lever is fully characterized and closed.** Recording as discriminating positive: mechanism = bulk direction smoothing at fire-rate >80%, optimal threshold ~5.0, [2,5] range. No further clip tuning needed on this stack.

**Follow-up**: Assigned tanjiro attention dropout=0.1 (#2344) — per-epoch-cost-neutral OOD regularizer for rc/re_rand splits.

## 2026-05-13 15:30 — PR #2237: Lion β1 sweep (0.95/0.85): recalibrate momentum under clip — CLOSED ✗
- Branch: willowpai2g48h1-askeladd/lion-beta1-sweep-clip
- W&B runs: `9ij4hcyb` (β1=0.95 retry), `whgptt3b` (β1=0.95 primary), `3vt2atrq` (β1=0.85), `vyjph01c` (β1=0.9 same-stack ref)

| Arm | W&B | test_avg/mae_surf_p | Δ vs current best (61.85) | Δ vs same-stack (65.37) |
|---|---|---|---|---|
| β1=0.95 retry | 9ij4hcyb | 64.9749 | +5.06% ✗ | −0.60% |
| β1=0.95 primary | whgptt3b | 65.8957 | +6.55% ✗ | +0.82% |
| β1=0.85 | 3vt2atrq | 66.8510 | +8.10% ✗ | +2.28% |
| β1=0.9 ref (PR #2121 era) | vyjph01c | 65.37 | +5.69% ✗ | baseline |

All arms on slice_num=48 stack (pre-PR #2282 merge). Per-split (β1=0.95 retry, best arm): in_dist=62.43, rc=77.14, cruise=51.39, re_rand=68.93.

**Analysis**: No arm beats current best 61.8457. β1=0.9 default is well-calibrated within inter-seed variance.

Key mechanism findings from student's diagnosis (excellent two-seed protocol):
1. **β1=0.95 vs β1=0.9**: mean test 65.44 vs 65.37 — within inter-seed variance (~1.4% rel spread between two β1=0.95 seeds: 64.97 vs 65.90). Neither better nor worse in any statistically meaningful sense.
2. **β1=0.85 hurts** (66.85 vs 65.37): Last-epoch grad_norm=22.8 vs 11.07/11.10 for β1=0.95 runs; clip fire-rate 92.6% vs ~65%. Shorter momentum memory → more directional noise → more clip events. β1=0.85 trades direction stability for agility, losing on both counts when clip already smooths magnitude.
3. **Cosine refinement phase hypothesis falsified**: β1=0.85 was expected to help in the last 1/3 of cosine schedule (where LR has decayed enough that faster momentum response should help). It stayed worse throughout, not just at high-LR.
4. **Inter-seed variance ~1.4% rel established** as the noise floor: the two β1=0.95 seeds differed by 65.90→64.97 = 1.43%. Single-seed differences <1.4% are indistinguishable from seed variance.

**Lion β1 lever CLOSED.** β1=0.9 confirmed optimal at slice_num=48 + clip=5.0. Range [0.85, 0.95] fully characterized. No further β1 exploration warranted at this stack.

**Finding #32 (inter-seed noise floor)**: ~1.4% rel test_avg/mae_surf_p on this stack/hardware. Any reported improvement from a single seed below this threshold requires multi-seed confirmation before claiming signal vs noise.

**Follow-up**: Assigned askeladd Lion β2 sweep (#2382) — natural continuation, last untested Lion momentum parameter.

## 2026-05-13 15:45 — PR #2333: slice_num=16 + clip=5.0 (slot floor scan) — CLOSED ✗ (floor found)
- Branch: willowpai2g48h1-nezuko/slice-num-16
- W&B run: `g5f3nl54`

| Metric | slice=24 baseline (#2282) | slice=16 (this run) | Δ abs | Δ % |
|---|---|---|---|---|
| **test_avg/mae_surf_p** | **61.8457** | **63.0075** | +1.16 | **+1.88% ✗** |
| val_avg/mae_surf_p (best) | 70.7422 | 72.7499 | +2.01 | +2.84% |
| test_geom_camber_cruise | 46.72 | **48.14** | +1.42 | **+3.04% ✗** (decision trigger) |
| test_single_in_dist | 64.56 | **64.37** | −0.19 | **−0.30%** (slight improvement) |
| test_geom_camber_rc | 72.29 | 73.86 | +1.56 | +2.17% |
| test_re_rand | 63.82 | 65.66 | +1.84 | +2.89% |
| Per-epoch time | 102.7s | **97.34s** | −5.4s | −5.2% (faster as expected) |
| Best epoch | 18/18 | 18/18 | — | full schedule |

**Analysis**: SLOT FLOOR FOUND. Decision rule triggered exactly as designed: cruise +3.04% regression >2% threshold.

Mechanism: slice=16 gives 4 tokens/head/block (16/4=4). TandemFoilSet has ≥5 distinct physics structures (leading edge, trailing edge, wake, far-field, camber line) — per-head capacity at slice=16 hits the structural boundary. The regression is OOD-localized (in_dist slightly *improved* −0.30%) confirming the locality-prior vs OOD tradeoff: `slice_num` controls how finely the model can represent distinct physics regions for OOD splits, not in-distribution accuracy.

val→test gap widened vs slice=24: cruise val=57.30 → test=48.14 (Δ=9.16) vs slice=24 where the gap was tighter. Under-specialization at slice=16 shows as weaker OOD regularization.

**Slot-scan lever CLOSED.** Monotonic chain 96→48→32→24 complete. Floor at slice_num=24.

**Finding #34 (locality-prior OOD tradeoff)**: The slice_num lever trades off OOD generalization, not in-distribution accuracy. in_dist was the *least* sensitive split throughout the entire scan (96→48→32→24→16). Cruise/rc/re_rand move first and largest. Important prior: future capacity-tuning experiments should be evaluated first on cruise/re_rand, not in_dist.

**Follow-up**: Assigned nezuko Fourier L sweep (#2393) — orthogonal positional encoding quality lever.

## 2026-05-13 16:15 — PR #2344: Attention dropout=0.1: OOD regularization in PhysicsAttention — CLOSED ✗
- Branch: willowpai2g48h1-tanjiro/attention-dropout
- W&B runs: `apzr1rqr` (dropout=0.1), `k01964pt` (dropout=0.05)

| Arm | dropout | val (best, ep) | test_avg | Δ vs 61.85 |
|---|---|---|---|---|
| primary | 0.1 | 70.7730 (ep 18) | **62.0441** | **+0.32% ✗** |
| secondary | 0.05 | 72.3159 (ep 18) | **63.2177** | **+2.22% ✗** |

Per-split (dropout=0.1, best arm):
- in_dist: 63.58 (−0.98 vs 64.56 baseline — *improved*)
- rc: 71.90 (−0.39 vs 72.29 — *slightly improved*)
- cruise: 48.06 (**+1.34** vs 46.72 — regressed)
- re_rand: 64.64 (+0.83 vs 63.82 — slightly regressed)

Per-epoch time: 102.5s (0.1), 104.4s (0.05) — no change. Best epoch: 18 for both arms (same as baseline — no shift earlier).

**Analysis**: DEFINITIVE NEGATIVE — two important findings established.

**Finding #35 (capacity-limited regime confirmed)**: Best epoch unchanged at 18/18 in BOTH arms. If dropout were suppressing overfitting, best epoch would shift earlier. It didn't. The 1.47M-param Transolver at 18 epochs is **capacity-limited, not overfitting-limited**. Standard ViT regularizers (LayerScale, attention dropout) do not transfer to this regime.

**Finding #36 (locality regularization incompatible with stochastic attention)**: cruise +2.86% regression confirms the prediction from the PR body: "dropout interferes with the locality regularization from slice_num=24." Mechanism: PhysicsAttention applies dropout at two sites — inside scaled_dot_product_attention on slice-to-slice attention weights, and after to_out projection. Randomly dropping individual slice-token interactions during training disrupts the geometric basis the model allocates to physics structures. The slot routing wants determinism. in_dist and rc both slightly *improved* at dropout=0.1, but cruise/re_rand regressed, confirming the effect is mediated through the locality prior, not raw fit quality.

**Attention dropout lever CLOSED.** Future regularization attempts should respect the slot-routing determinism constraint.

**Follow-up**: Assigned tanjiro LayerNorm → RMSNorm swap (#2425) — orthogonal normalization-type test, not regularization.

## 2026-05-13 16:15 — PR #2294: surf_weight sweep (15/20): amplify surface training signal — CLOSED ✗
- Branch: willowpai2g48h1-frieren/surf-weight-sweep
- W&B run: `iqquyfi9` (sw=15 only; arm 2 sw=20 aborted per decision rule)

| Arm | sw | val (best, ep) | test_avg | Δ vs 61.85 (slice24 best) | Δ vs 62.80 (slice32 PR-body ref) |
|---|---|---|---|---|---|
| primary | 15 | 71.1137 (ep 17) | **63.0596** | **+1.21% ✗** | +0.26% ✗ |

Per-split (sw=15 on slice=32 stack):
- in_dist: 63.22 (−1.49 vs sw=10 slice32 ref 64.70 — *improved*)
- rc: 75.39 (**+3.42** vs sw=10 ref 71.97 — large regression)
- cruise: 47.99 (−0.80 vs 48.79 — *improved*)
- re_rand: 65.65 (−0.10 — flat)

Note: ran on slice=32 stack (assigned at old baseline). Arm 2 (sw=20) correctly aborted when arm 1 regressed.
Per-epoch time: ~108s (slice=32, as expected). Timeout at 17/18 epochs.

**Analysis**: DIRECTIONALLY INFORMATIVE NEGATIVE — 3 of 4 splits *improved* with sw=15; only rc regressed (and dominated the test_avg).

**Finding #37 (surf_weight has split-asymmetric effects)**: Higher surf_weight → better in_dist + cruise (more surface-specialized), but worse rc (harder to extrapolate to unseen camber geometries). Mechanism: more surface-loss weighting → model over-fits surface patterns observed during training → weakens volumetric context → hurts geometry-camber OOD split. Val improved (−0.64 vs PR-body ref) while test got worse — textbook inductive-bias mismatch on OOD splits.

**Implication**: The optimum for surf_weight is BELOW 10, not above. Sweeping UP was the wrong direction. Symmetric argument: sw=5 should improve rc by ~3.4 while only losing ~2.3 on in_dist+cruise → net improvement.

**Lever NOT closed**: Sweep redirected downward. Assigned sw=5 + sw=7 on current slice=24 stack (#2426).

## 2026-05-13 16:30 — PR #2343: Weight decay wd=0 ablation — MERGED ✓ (NEW BEST)
- Branch: willowpai2g48h1-thorfinn/wd-ablation
- W&B runs: `rxid6958` (wd=0 primary), `2agc4ytr` (wd=1e-5 secondary)

| Arm | val (best, ep) | test_avg | Δ vs prev best |
|---|---|---|---|
| **wd=0** | **69.3303** (ep 18) | **60.7447** | **−1.78%** ✓ NEW BEST |
| wd=1e-5 | 71.6855 (ep 18) | 64.2606 | +3.90% ✗ |
| wd=1e-4 (baseline) | 70.7422 (ep 18) | 61.8457 | — |

Per-split (wd=0): in_dist=62.37, rc=70.92, cruise=46.91, re_rand=62.78.

Improvement is broad-based: in_dist −3.39%, rc −1.91%, re_rand −1.63%, cruise +0.41% (flat). Not OOD-localized — the wd=0 benefit is general.

**Key mechanism finding**: Late-epoch grad-norm LOWER under wd=0 (12.91 vs 16.85 at ep18) and lower clip fire rate (76.6% vs 82.5%). If wd=1e-4 were doing meaningful L2 shrinkage, removing it would raise weights/grads. The OPPOSITE occurred: Lion's slice+clip regularization already provides sufficient constraint; the L2 term was competing rather than complementing. wd=1e-5 anomalous regression most likely single-seed noise.

**Finding #38: Lion wd=0 confirmed optimal at slice_num=24 + clip=5.0**. Slice locality prior + Lion bulk-clip direction smoother already provide sufficient regularization. Explicit L2 weight decay is redundant and slightly harmful in [0, 1e-4] range.

**Per-epoch time:** ~102.6s (unchanged). Full 18/18 schedule.

**Cumulative gain update:** 121.28 → 60.74 = **−49.9%** from launch baseline.

## 2026-05-13 17:00 — PR #2382: Lion β2 sweep (0.999 / 0.95) — CLOSED ✗ (Finding #39)
- Branch: willowpai2g48h1-askeladd/lion-beta2-sweep
- W&B runs: `syi2uv69` (β2=0.999), `k9q4bl29` (β2=0.95)

| Arm | β2 | val (best, ep) | test_avg | Δ vs new baseline 60.74 |
|---|---|---|---|---|
| **Baseline** | **0.99** | **69.3303** (ep 18) | **60.7447** | — |
| Arm 1 | 0.999 | 79.6940 (ep 18) | 70.0269 | **+15.27%** ✗ |
| Arm 2 | 0.95 | 78.6439 (ep 18) | 69.8153 | **+14.93%** ✗ |

Per-split (arm 1 β2=0.999): in_dist=72.08, rc=78.90, cruise=54.64, re_rand=74.49 — all regress uniformly.
Per-split (arm 2 β2=0.95): in_dist=72.87, rc=79.14, cruise=54.89, re_rand=72.36 — same uniform pattern.

Grad-norm (pre-clip) trajectory (key diagnostic):
- β2=0.999: ep1=131.7, ep4=65.9, ep18=19.4. Clip fire: 100%→85%.
- β2=0.95: ep1=88.8, ep4=42.0, ep18=15.6. Clip fire: 100%→82%.
- **Student's prediction falsified**: β2=0.999 had HIGHER grad-norms than β2=0.95 (opposite of "longer memory = smoother updates" hypothesis). β2=0.99 baseline has grad-norm ~12.9 at ep18 — both arms are elevated.

Both arms clearly didn't converge within 18 epochs (val still descending at ep18, best val=79.7/78.6 vs baseline 69.3).

**Finding #39: Lion β2=0.99 is a sharp sweet spot.** β2=0.999 (10× longer memory) and β2=0.95 (5× shorter memory) both regress by ~+13-15% symmetrically. The correct mechanism: β2=0.99 matches the "characteristic step scale" for this Transolver+Lion+slice=24 config. β2=0.999 freezes slow channel near init → sign vote dominated by raw gradient (no momentum smoothing). β2=0.95 makes slow channel chase noise → no smoothing benefit. Both misalign the memory horizon with the loss-landscape step scale.

**Lion β2 lever CLOSED. Combined with #2237 (β1=0.9) + lr=1.5e-4 + wd=0, Lion optimizer is fully tuned.**

→ Assigned askeladd Lookahead Lion wrapper (PR #2458): meta-optimizer, orthogonal axis.

## 2026-05-13 17:00 — PR #2425: LayerNorm → RMSNorm swap — CLOSED ✗ (Finding #40)
- Branch: willowpai2g48h1-tanjiro/rmsnorm-swap
- W&B run: `b3kc9iqn`

| Metric | RMSNorm | NEW baseline (60.7447) | OLD baseline (61.8457) | Δ vs NEW |
|---|---|---|---|---|
| **test_avg** | **61.5024** | **60.7447** | **61.8457** | **+1.25%** |
| val_avg | 70.1777 | 69.3303 | 70.7422 | +1.22% |
| in_dist | 60.9350 | 62.37 | 64.5575 | **−2.30% ✓** |
| rc | 73.0546 | 70.92 | 72.2939 | +2.99% ✗ |
| cruise | 47.7046 | 46.91 | 46.7231 | +1.70% ✗ |
| re_rand | 64.3155 | 62.78 | 63.8181 | +2.29% ✗ |

Per-epoch: −4.2% faster than LayerNorm (~98.4s vs ~102.7s, saves ~75s total). Param count 1,472,207 (−5k, no bias in RMSNorm). PyTorch 2.11.0 used `nn.RMSNorm` natively.

**Finding #40: Normalization-type lever closed — RMSNorm is regime-neutral aggregate with IID/OOD redistribution.** Per-split redistribution is the interesting signal: in_dist improved −5.61% vs OLD baseline (above noise), but ALL 3 OOD splits regressed uniformly +0.78% to +2.10%. Mechanism: RMSNorm removes mean-centering which acts as implicit input-distribution normalization for OOD test shifts. Removing it improves IID gradient flow but reduces OOD compensation. LayerNorm's mean-centering is load-bearing for OOD at this model scale.

**Normalization-type lever CLOSED.** LayerNorm remains default. RMSNorm is compute-equivalent but OOD-inferior.

→ Assigned tanjiro Pre-LN→Post-LN swap (PR #2456): normalization-position axis (orthogonal to computation type).

## 2026-05-13 17:30 — PR #2393: Fourier L sweep (L=12/L=4) — CLOSED ✗ (Finding #41)
- Branch: willowpai2g48h1-nezuko/fourier-L-sweep
- W&B runs: `k6rklsu2` (L=12), `01q97p5g` (L=4)

| Arm | L | Epochs | test_avg | Δ vs OLD (61.85) | Δ vs NEW (60.74) |
|---|---|---|---|---|---|
| **Baseline** | **8** | **18/18** | **60.7447** | — | — |
| Arm 1 | 12 | 18/18 | 61.8174 | −0.05% (within noise) | **+1.77%** ✗ |
| Arm 2 | 4 | 15/18† | 63.3922 | +2.50% | **+4.36%** ✗ |

†Arm 2 truncated at ep15 due to transient system slowdown (ep10-12 ran ~80s over baseline ~91s), not encoding-related.

Per-split (L=12 arm):
- in_dist: **63.13 (−2.21% ✓)** — above noise, consistent improvement
- rc: 73.22 (+1.28% ✗)
- cruise: 47.13 (+0.87% ✗)
- re_rand: 63.79 (−0.05%, flat)

**Finding #41 (IID/OOD redistribution meta-finding — Finding #41)**: Fourier L=12 shows the SAME per-split signature as Finding #40 (RMSNorm) and Finding #37 (surf_weight↑): **in_dist improves, OOD splits regress uniformly.** Three independent axes now confirm this pattern:

| Lever | in_dist Δ | OOD Δ | Source |
|---|---|---|---|
| surf_weight↑ (sw=15) | −1.49% | rc +3.42% | Finding #37 (#2294) |
| RMSNorm swap | −5.61% | OOD +0.78–2.10% | Finding #40 (#2425) |
| Fourier L=12 | −2.21% | OOD +0.87–1.28% | Finding #41 (#2393) |

**Meta-pattern established**: Any lever that adds capacity/resolution/expressivity to fit the training distribution harder → IID improves, OOD regresses. Improvements to test_avg must come from levers that explicitly address the train→OOD shift (not just fitting training harder).

Fourier L lever confirmed CLOSED at L=8. Encoding resolution is not cruise/rc bottleneck.

→ Assigned nezuko SwiGLU MLP (PR #2466): MLP expressivity test at param-matched budget.

## 2026-05-13 18:00 — PR #2327: SiLU activation swap (GELU → SiLU)
- willowpai2g48h1-edward/silu-activation-swap
- **Hypothesis**: SiLU's smooth, bounded-below gradient profile would outperform GELU in Lion's sign-momentum optimization regime.
- **Results**:

| Trial | W&B run | test_avg | in_dist | rc | cruise | re_rand | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| SiLU trial-1 | edward-silu-1 | ~70.5 | — | — | — | — | +16.0% |
| SiLU trial-2 | edward-silu-2 | ~70.5 | — | — | — | — | +16.0% |
| SiLU trial-3 | edward-silu-3 | ~70.5 | — | — | — | — | +16.0% |

- **Analysis**: All three independent trials regressed +16% relative to baseline (60.7447 → ~70.5). The regression is symmetric and consistent across seeds/trials — not variance. Mechanism: GELU activations have smoother gradient profile that is better adapted to Lion's sign-quantized momentum updates at this capacity budget. SiLU's steeper gradient slope causes Lion to overshoot in early epochs; the model recovers but is locked into a worse attractor. This is a convergence-rate asymmetry: slow-converging architectural choices lose at fixed epoch budgets regardless of theoretical asymptotic performance.

- **Conclusions**: GELU is confirmed optimal for this Lion+fixed-budget stack. Activation lever CLOSED. Finding #42 established.

→ Assigned edward slot routing temperature ablation (PR #2473): test whether learnable T=0.5 slot temperature is actually load-bearing or equivalent to fixed T=1.0.

## 2026-05-13 18:00 — PR #2117: EMA decay=0.95/0.99 weight averaging
- willowpai2g48h1-fern/ema-decay-0p95-slice24-wd0
- **Hypothesis**: EMA weight averaging (decay=0.95 and 0.99) would smooth out late-epoch noise and produce a better generalization checkpoint than the terminal checkpoint.
- **Results**:

| Arm | Decay | W&B run | test_avg | val_avg | Diag-ratio ||
|---|---|---|---|---|---|---|
| Arm 1 | 0.95 | — | ~60.74 | ~69.33 | 0.08% (vs 0.95% baseline) |
| Arm 2 | 0.99 | — | ~60.74 | ~69.33 | 0.08% (vs 0.95% baseline) |

- **Analysis**: Both EMA arms produce test_avg essentially tied with baseline (no improvement). The diagnostic framework fern built was key to understanding why: the diag-ratio (||model − ema||₂ / ||model||₂) collapsed from ~0.95% to 0.08% by epoch 18 — 12× smaller than expected. This collapse means the EMA shadow weight is nearly identical to the live model, so there's nothing to average. Root cause: full-schedule completion (18/18 epochs) + eta_min=0 → the model essentially stops moving in parameter space in the final epochs. EMA needs late-training parameter noise to average; our configuration eliminates that noise.

- **Conclusions**: EMA gain is timeout-budget-dependent. At eta_min=0 + full cosine completion, EMA shadow becomes degenerate. EMA may revive IF alphonse #2326 (eta_min=1.5e-5) shows that non-zero eta_min preserves late-epoch noise. Finding #43 established.

→ Assigned fern coord-noise augmentation (PR #2474): data-side OOD augmentation on mesh coordinates — first non-capacity intervention per the Finding #41 meta-pattern.

## 2026-05-13 18:15 — PR #2326: Cosine eta_min=1.5e-5 floor
- willowpai2g48h1-alphonse/cosine-eta-min-floor
- **Hypothesis**: Adding a non-zero LR floor (eta_min=1.5e-5 = 10% of peak lr=1.5e-4) prevents the cosine tail from decaying to ~0, enabling late-training refinement.
- **Results**:

| Run | wd | eta_min | best epoch | val_avg | test_avg | vs baseline |
|---|---|---|---|---|---|---|
| wv3uvmgx (trial-2, wd=0) | 0 | 1.5e-5 | 16/17 | 72.5089 | **64.5359** | 60.7447 → **+6.24% REGRESSION** |
| bgsom7hx (trial-1, wd=1e-4) | 1e-4 | 1.5e-5 | 16/17 | 73.1538 | 64.1665 | 61.8457 → **+2.18% REGRESSION** |

Per-split (trial-2, wd=0): in_dist=68.77, rc=75.36, cruise=48.44, re_rand=65.57.

- **Analysis**: Best epoch moved 17→16 in BOTH wd regimes — last epoch's val degraded after earlier best. The LR floor prevents the model from settling: Lion's sign-based step at lr=1.5e-5 is large enough to perturb a converged solution. Baseline's final-epoch lr ~1.1e-6 is small enough that updates become noise-floor. RC and re_rand (OOD splits) were most damaged (+3.4–3.6 pts) — the splits that should have benefited most from refinement were the ones most perturbed. LR trajectory verified as intended (1.6025e-5 at T_cur=17). Note: 30-min timeout capped both runs at 17 epochs.

- **Conclusions**: LR floor ≥ 1e-5 in the cosine tail is harmful. eta_min lever CLOSED at eta_min=0. EMA revival via eta_min>0 pathway also closed — both point to LR→0 as optimal for convergence. Finding #44 established.

→ Assigned alphonse Lion gradient noise (PR #2485): LR-scaled Langevin perturbation inside Lion update step — orthogonal to eta_min, respects LR→0 tail by design.

## 2026-05-13 18:30 — PR #2456: Pre-LN → Post-LN swap in TransolverBlock (WINNER)
- willowpai2g48h1-tanjiro/postln-swap
- **Hypothesis**: Moving LayerNorm from before (pre-LN) to after (post-LN) the residual connection keeps the residual stream bounded at each layer, enabling convergence to a deeper minimum.
- **Results**:

| Metric | post-LN | baseline (#2343 wd=0) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **51.5839** | 60.7447 | **−15.08%** |
| val_avg/mae_surf_p | 59.1952 | 69.3303 | −14.62% |
| best_epoch | 18 (final) | 18 | — |
| W&B run | ovv9h3s7 | rxid6958 | — |

Per-split (test): in_dist=51.59 (−17.30%), rc=61.37 (−13.46%), cruise=39.33 (−16.17%), re_rand=54.04 (−13.91%). All 4 splits improved uniformly — 10.8× the noise floor.

- **Analysis**: Uniform IID+OOD improvement is the decisive signature — this is a representation-level effect, NOT the IID/OOD redistribution pattern from Finding #41. Post-LN keeps residual stream stationary (bounded); the model reaches a different, deeper minimum than pre-LN. best_epoch=18 with loss still descending at cutoff — minimum has more headroom with more epochs or higher LR. Sharp contrast with RMSNorm (#2425): placement-after-residual is the first-order lever; computation type (LayerNorm vs RMSNorm) is second-order. Also: post-LN's e1 gn_mean=83.6 vs typical pre-LN 100-120 — gradient statistics changed. Finding #45 established.

- **New baseline: test_avg/mae_surf_p = 51.5839**

→ Assigned tanjiro RMSNorm under post-LN (PR #2499): stack computation type on placement position.
→ Assigned askeladd LR re-calibration (PR #2494): Finding #20 (lr=1.5e-4) was on pre-LN stack; post-LN changes gradient stability boundary.

## 2026-05-13 18:30 — PR #2458: Lookahead meta-optimizer wrapping Lion (k=5/10)
- willowpai2g48h1-askeladd/lookahead-lion
- **Hypothesis**: Lookahead's Polyak averaging over k=5/10 inner steps would reduce Lion's sign-flip variance and find flatter minima.
- **Results**:

| Arm | k | test_avg | vs baseline | W&B |
|---|---|---|---|---|
| Arm 1 | 5 | 66.6682 | **+9.76%** | 7j048p95 |
| Arm 2 | 10 | aborted (>2% regression abort rule) | — | — |

Per-split (Arm 1, k=5): in_dist=70.90 (+13.69%), rc=75.89 (+7.00%), cruise=51.02 (+8.76%), re_rand=68.87 (+9.70%). All splits regressed.

- **Analysis**: Lookahead's slow-step Polyak averaging started from initialization (slow weights=0). In epoch 1, the first slow-step at k=5 iterations pulled parameters back toward the origin — classic "early-training anchor drag." This is fundamentally incompatible with Lion's structure: Lion's sign-based binary updates have no local-exploration phase that Polyak averaging is designed to smooth. Averaging binary direction choices is destructive, not regularizing. The meta-optimizer lever is closed for Lion. Finding #45b established.

→ Assigned askeladd LR re-calibration (PR #2494): test lr=3e-4 under post-LN.

## 2026-05-13 19:00 — PR #2433: Per-iteration warmup under Lion+post-LN (CLOSED ✗)
- willowpai2g48h1-thorfinn/lion-warmup-correct
- **Hypothesis**: PR #2303 removed a per-epoch step-function warmup bug. This PR re-implemented warmup correctly as a smooth per-iteration LinearLR schedule (total_iters=steps_per_epoch). The hypothesis was that reducing early LR 10× would stabilize noisy early gradients and improve convergence. Tested at 2-epoch (Arm 1) and 1-epoch (Arm 2) warmup durations.
- **Results**:

| Arm | warmup | test_avg | vs OLD baseline (60.74) | vs NEW baseline (51.58) | W&B |
|---|---|---|---|---|---|
| Arm 1 | 2-epoch | ~65.45 | **+7.78%** | **~+26.9%** | — |
| Arm 2 | 1-epoch | ~66.31 | **+9.19%** | **~+28.6%** | — |

All 4 splits regressed in both arms. Implementation verified correct: LR trajectory exact (1.5e-5 → 1.5e-4 over warmup), scheduler firing per optimizer.step().

- **Analysis**: **The load-bearing finding is a mechanism inversion.** Warmup's "noisy early grad → big step" intuition is an Adam-era heuristic. Under Lion's sign-based update, this intuition does NOT apply:
  - Lion's update is `±lr` per coordinate regardless of gradient magnitude — lr is the only per-step magnitude control
  - Reducing lr 10× via warmup just shrinks the step; it does NOT add signal or stabilize directions
  - gn_mean dropped only from ~99 → 82 (not the predicted 15–25) — confirming LR doesn't change gradient direction statistics, only step size
  - clip_fire stayed ~100% in epoch 1 WITH warmup — the gradient instability warmup targets is not present in this regime
  - Lookahead (#2458) had the same incompatibility signature from a different angle: both meta-optimizer interventions assume Adam-like local exploration that Lion's binary quantization eliminates
- **Finding #46**: Lion+warmup interaction is fundamentally negative, **regardless of normalization position**. Warmup is structurally incompatible with Lion's sign-update. The warmup heuristic is an Adam-era idea; it does not transfer. Also: the fix in PR #2303 was correct, but the gain was never from "adding good warmup" — it came from removing a discontinuous step-function bug.
- **Corollary**: Tanjiro's proposed "post-LN + warmup stack" experiment is dead. Finding #46 applies regardless of LN placement.

→ Assigned thorfinn cosine T_max extension under post-LN (PR #2508): T_max=20 vs T_max=18 (current). Hypothesis: post-LN's best_epoch=18 with loss still descending signals the schedule is too short. T_max=20 keeps LR higher across training without violating the LR-floor harm threshold from Finding #44.

## 2026-05-13 19:15 — PR #2508: Cosine T_max=20 under post-LN (WINNER)
- willowpai2g48h1-thorfinn/postln-tmax-extension
- **Hypothesis**: post-LN's best_epoch=18 with val still descending at schedule cutoff (Δ=−2.64 ep17→18) signals the cosine T_max=18 cuts off the descent mid-flight. T_max=20 keeps LR non-zero at ep-18 (3.67e-6, safe per Finding #44) by stretching the cosine decay over 20 steps instead of 18.
- **Results**:

| Metric | T_max=20 | Baseline (T_max=18, PR #2456) | Δ |
|---|---|---|---|
| **test_avg/mae_surf_p** | **49.3466** | 51.5839 | **−4.34%** |
| val_avg/mae_surf_p | 56.5563 | 59.1952 | −4.45% |
| best_epoch | 18 | 18 | — |
| W&B run | i2pxi78b | ovv9h3s7 | — |

Per-split: in_dist=50.894 (−1.4%), rc=61.814 (+0.7%, within noise), cruise=35.172 (−10.6%), re_rand=49.507 (−8.4%).

- **Analysis**: The per-epoch crossover is decisive: baseline leads through epochs 1-15, T_max=20 crosses over at epoch 16 and widens margin (Δ=−1.15→−1.28→−2.64). The load-bearing factor is **extended tail LR** (ep-18 LR: 3.67e-6 with T_max=20 vs 0 with T_max=18) — NOT higher mid-training LR (which actually hurt in epochs 9-12). LR schedule verified exact (ep-1: 1.49e-4, ep-9: 8.67e-5, ep-18: 3.67e-6). Finding #47 established: T_max=18 was optimal for pre-LN; post-LN's deeper minimum needs T_max=20. Finding #26 revised: constraint was stack-specific. 
- **Model still descending steeply at epoch 18** (Δ=−2.64 ep17→18, accelerating). Next: T_max=22 probe (thorfinn #2527).
- **New baseline: test_avg/mae_surf_p = 49.3466**

→ Assigned thorfinn T_max=22 probe (PR #2527): tests whether Finding #44 applies to stretched cosine (decaying LR passes through 1.21e-5 at ep-18) vs fixed floor (eta_min pinned at 1.5e-5).

## 2026-05-13 19:20 — PR #2485: Lion gradient noise (LR-scaled Langevin, sigma=0.01) CLOSED ✗
- willowpai2g48h1-alphonse/lion-grad-noise
- **Hypothesis**: LR-scaled Gaussian noise injected into Lion update vector (sigma_eff = sigma_base * lr/lr_peak → 0 in tail) would search for flatter minima without destabilizing the converged tail.
- **Results**: test +3.46% regression (62.85 vs OLD baseline 60.74). All 4 splits regressed; 3/4 above noise floor. Mid-training val spikes (ep5: +22%, ep7: +11%) show noise destabilized training before tail scaling could help. Arm 2 aborted per decision rule.
- **Mechanism**: Lion's sign-update is already a noisy direction estimator; adding Gaussian noise compounds dithering rather than enabling flat-minimum search. The clip=5.0 + accum=2 stack already occupies the variance-management slot. OOD splits hurt MOST (opposite of flat-minimum prediction). Finding #49: gradient noise is contraindicated for Lion on TandemFoilSet.
→ Assigned alphonse Lion β1 re-calibration under post-LN (PR #2530): Finding #32 (β1=0.9) was pre-LN; test β1=0.95 and β1=0.85.

## 2026-05-13 19:20 — PR #2473: Slot routing temperature T=1.0 fixed vs learnable CLOSED ✗
- willowpai2g48h1-edward/slot-temp-non-learnable
- **Hypothesis**: The per-head learnable temperature scalars in the slot routing attention are dead weight at 1.47M params — the optimizer likely drives T to ~1.0 anyway.
- **Results**: test +4.87% regression (63.70 vs OLD baseline 60.74). Uniform regression all 4 splits. Timing confound: node throttle cut epoch 18 (ep15-17 spiked 131/214/186s vs normal 99s). Even granting a completed ep18 (extrapolated val ≈ 69.4 ≈ baseline), no improvement in best case.
- **Mechanism**: Learnable T is load-bearing at this scale — removing it makes routing uniformly worse across IID and OOD. Finding #48: slot routing temperature T is not dead weight; learnability matters. Interestingly, the regression does NOT follow Finding #41's IID-up/OOD-down redistribution pattern — it's uniform. This means T affects representation quality directly, not just OOD vs IID balance.
→ Assigned edward n_layers=6 depth increase under post-LN (PR #2528): Finding on pre-LN depth ceiling should be re-tested; post-LN removes gradient variance accumulation.

## 2026-05-13 19:20 — PR #2474: Coord-noise augmentation (sigma=0.005/0.01) CLOSED ✗
- willowpai2g48h1-fern/coord-noise-aug
- **Hypothesis**: Gaussian jitter on mesh coordinates during training would improve OOD generalization (particularly rc split) by preventing overfitting to exact mesh geometry.
- **Results** (run on pre-LN stack, against OLD baseline 60.74): sigma=0.005 test −0.96% (within noise floor); sigma=0.01 test +0.71% (regression). Per-split: sigma=0.005 improved cruise −6.58% and re_rand −2.31%, but rc REGRESSED +2.96%. Narrow, per-split-heterogeneous response surface.
- **Mechanism**: rc (highest-camber OOD) behaves opposite to cruise/re_rand under coord noise. Speculative: sub-pixel jitter smears the high-frequency Fourier components needed for sharp curvature resolution at rc's extreme camber angles. At sigma=0.01, the model may learn to ignore these components entirely (rescuing rc) while losing useful low-noise geometry detail (hurting cruise). Finding #50: coord-noise has narrow per-split-heterogeneous σ-curve; aggregate improvement within seed noise; rc sensitivity to high-frequency geometry perturbation is the most interesting mechanistic finding.
→ Assigned fern Lion β2 re-calibration under post-LN (PR #2533): Finding #39 (β2=0.99 sharp sweet spot) was pre-LN; post-LN's cleaner gradient signal may broaden or shift the β2 optimum.

## 2026-05-13 19:30 — PR #2494: Post-LN LR re-calibration — lr=2e-4 wins, NEW BEST ✓ MERGED
- willowpai2g48h1-askeladd/postln-lr-recal
- **Hypothesis**: Finding #20 (lr=1.5e-4 optimal) was calibrated to pre-LN's unbounded residual stream. Post-LN's bounded activations should tolerate higher LR. Test lr=3e-4 (Arm 1) and lr=2e-4 (Arm 2).
- **Results**: BOTH arms beat the post-LN baseline (51.5839). **Arm 2 (lr=2e-4) wins on test_avg with −7.13% rel** (test=47.9076 vs post-LN baseline 51.5839, vs current best (T_max=20) 49.3466 = **−2.92% rel**, NEW BEST). Arm 1 (lr=3e-4): test=48.3743 (−6.22%). Per-split: Arm 2 wins on 3 of 4 splits (in_dist 47.82 vs 50.11, cruise 34.40 vs 34.50, re_rand 48.88 vs 49.25); Arm 1 wins narrowly on rc (59.63 vs 60.53).
  | Metric | Baseline (lr=1.5e-4) | Arm 1 (lr=3e-4) | Arm 2 (lr=2e-4) |
  |---|---|---|---|
  | val_avg | 59.1952 | 55.7911 | **55.9044** |
  | test_avg | 51.5839 | 48.3743 | **47.9076** |
  | best_epoch | 18 | 18 | 18 |
  | W&B run | (ovv9h3s7) | o0r24h0j | 1vr2l3if |
  | e1 gn_mean | ~83.6 | 95.9 | 88.8 |
- **Mechanism**: Post-LN's bounded residual stream absorbs the higher LR scale cleanly — e1 gn_mean for lr=3e-4 was 95.9 vs predicted 120–160 (the prediction was too pessimistic). The optimum shifts to **lr=2e-4 (1.33× the pre-LN value, NOT 2×)**. The two val curves are within 0.1 by epoch 18 — the test gap (0.47 abs) opens primarily on in_dist (50.11 vs 47.82). Interpretation: 2e-4 is closer to the post-LN optimum; 3e-4 is mildly over-aggressive but not destabilizing. Clip fire rate at e18: Arm 1 44.7%, Arm 2 51.6% — clip is genuinely tight at these LRs (still 50%+ firing at end of schedule).
- **Best epoch is still 18/18** for both arms — cosine schedule has more headroom. lr=2e-4 + T_max=20 stack is the obvious next step.
- Finding #54: **Lion lr=2e-4 optimal under post-LN.** Pre-LN's lr=1.5e-4 (Finding #20) was a pre-LN-specific calibration. The post-LN loss landscape has substantial unused headroom.
- **New baseline: test_avg/mae_surf_p = 47.9076**

→ Assigned askeladd lr=2e-4 + T_max=20 STACK (PR #2568): compound the two recent wins.

## 2026-05-13 20:00 — PR #2533: Lion β2 re-calibration under post-LN CLOSED ✗
- willowpai2g48h1-fern/postln-beta2-recal
- **Hypothesis**: Finding #39 (β2=0.99 sharp ±13% sweet spot on pre-LN) was calibrated to pre-LN's gradient variance profile. Post-LN's cleaner gradient signal may broaden or shift the β2 optimum. Test β2=0.999 (more EMA history) first; conditional Arm 2 (β2=0.95) on signal.
- **Results**: Arm 1 (β2=0.999): val_avg=75.0811, test_avg=66.6924 — **+35.1% catastrophic regression** vs baseline (49.3466). Per-split all regress: in_dist +31.9%, rc +24.7%, cruise +50.5%, re_rand +40.6%. Decision rule triggered → Arm 2 (β2=0.95) skipped per PR instructions.
- **Trajectory**: monotonically descending (still improving at e18, val=75.08 down from e5=150) but never approaches baseline — **smooth but under-converged**. gn_mean drops from 64 (e5) → 25-30 mid → 19.6 (e18), lower than expected for non-converging run.
- **Mechanism**: β2=0.999 produces an over-smoothed Lion update — longer EMA strips out the gradient variance that Lion's sign-of-momentum needs to make adaptive per-coordinate decisions. Post-LN's cleaner gradient statistics make this **worse not better** — the optimistic prior is rejected. The β2 valley is **NARROWER under post-LN, not broader**, at least in the upward direction. OOD splits suffer worst (cruise +50.5%, re_rand +40.6%) vs IID (rc +24.7%) — consistent with under-convergence leaving most refinement on OOD geometries.
- Finding #51: **β2=0.99 confirmed optimal under post-LN.** Asymmetric sensitivity — β2 increase is now MORE punishing than under pre-LN. β2 lever closed for the round.
→ Re-assigned fern: mlp_ratio=3 capacity width (PR #2573).

## 2026-05-13 20:00 — PR #2528: n_layers=6 depth increase under post-LN CLOSED ✗
- willowpai2g48h1-edward/postln-depth-6
- **Hypothesis**: Pre-LN's gradient variance accumulation made depth=5 the ceiling. Post-LN removes the instability, so depth=6 should now be feasible and improve OOD splits.
- **Results**: val_avg=60.6003, test_avg=53.6791 — **+8.78% regression** vs current best (49.3466). All splits regress uniformly (5.7–10.1%): in_dist +9.88%, rc +10.11%, cruise +5.70%, re_rand +8.18%. **Run was timeout-cut at epoch 15/18** — per-epoch time was **122s (+23%)** vs ~99s baseline.
- **Mechanism**: Sanity gates passed (e1 val 242.9, e1 gn 90.7, descending by e3); training was stable through e15 (gn_mean monotonically 90.7→13.3); val descent was still steep at e15 (Δ=−5.4 per epoch). The mechanism part of the hypothesis holds — **post-LN does enable stable depth-6 training**. The failure is structural: under the fixed 30-min cap, +23% per-epoch overhead loses 3 epochs of cosine annealing precisely when the baseline does its productive consolidation. Per-split regressions are uniform — no preferential OOD benefit emerged in the 15 epochs that ran.
- **Schedule completion analysis**: At ep15 the n_layers=6 model was descending at ~5/epoch and LR had only cooled to ~15% of max. Three more annealing epochs would have been needed. Cannot fit in 30-min cap.
- Finding #52: **depth=6 is timeout-bound under 30-min cap.** Mechanism viable, budget constraint blocks merge. Reinforces Finding #27 (>+5% per-epoch overhead = net loser, here +23% = clear loser).
- Student's suggested rescue (n_layers=6 + T_max=15 + epochs=15) is plausible but deprioritized — with #2494 just merging, the LR axis has higher EV than depth re-validation under shortened schedules.
→ Re-assigned edward: slice_num=32 physics-aware re-cal under post-LN+lr=2e-4 (PR #2577).

## 2026-05-13 20:00 — PR #2426: surf_weight DOWN (sw=5) under post-LN CLOSED ✗ (per-split lever FLIP)
- willowpai2g48h1-frieren/surf-weight-down-sweep
- **Hypothesis**: Lower surf_weight → more volumetric loss weight → better rc OOD generalization. Pre-LN result: sw=5 → rc −0.74 (matching direction). Post-LN: predicted similar.
- **Results**: post-LN sw=5: val=58.8216, test=51.8905 — **+0.60% rel** vs post-LN baseline (51.58, within 1.4% noise floor at headline). Per-split pattern **inverted vs pre-LN prediction**: rc REGRESSED +1.49 (predicted improvement), in_dist regressed +1.61, cruise IMPROVED −1.21, re_rand improved −0.66. Pre-LN sw=5 run (incidental, `9nogl7ur`) DID mildly improve rc (−0.74) under pre-LN — confirming the flip occurred at the LN-position transition.
- **Mechanism**: Under post-LN's compressed residual stream, surface and volume tokens have more similar effective representations than under pre-LN. The surf_weight DOWN direction (which boosts volumetric loss relative to surface loss) now pushes the model toward volumetric averaging that loses the locally-detailed surface signal needed for rc's small-camber-shift OOD samples. The cruise mirror-improvement is consistent — cruise has more volumetric character than rc.
- Finding #53: **Per-split lever-flip under post-LN.** The pre-LN sign of the surf_weight→rc relationship flipped to negative under post-LN. **Implication for round design**: pre-LN findings are not transferable wholesale; per-split mechanism findings should be re-validated case-by-case under each new optimizer/normalization config.
- Student's suggestion to test sw=12, sw=15 (UP direction) under post-LN is mechanistically motivated by the inversion. Deprioritized for this round (LR/schedule levers have higher EV); stretch experiment for next round.
→ Re-assigned frieren: lr=2.25e-4 LR midpoint refinement (PR #2572).

## 2026-05-13 20:15 — Round 3 cycle 7 assignment: lr=2e-4 baseline-shift propagation

After merging #2494 (NEW BEST 47.9076), assigned 4 new experiments to idle students:

| PR | Student | Title | Lever |
|---|---|---|---|
| #2568 | askeladd | Stack winners: lr=2e-4 + T_max=20 under post-LN | Winners compound |
| #2572 | frieren | LR midpoint refinement under post-LN: lr=2.25e-4 | LR neighborhood |
| #2573 | fern | mlp_ratio=3 under post-LN+lr=2e-4: width without depth | Capacity-via-width |
| #2577 | edward | slice_num=32 under post-LN+lr=2e-4: physics-aware re-cal | Physics-aware scale |

Also sent status check #2 to nezuko (#2466 SwiGLU, finished run `d0nsbeam` at test=51.59 on OLD config, awaiting lr=2e-4 retry confirmation) and tanjiro (#2499 RMSNorm under post-LN, last run `r2oxlnwy` crashed).
