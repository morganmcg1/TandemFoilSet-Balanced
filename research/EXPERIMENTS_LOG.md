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
