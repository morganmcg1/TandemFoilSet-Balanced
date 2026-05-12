# SENPAI Research Results

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
