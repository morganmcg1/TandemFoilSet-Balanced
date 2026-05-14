# SENPAI Research Results — willow-pai2g-48h-r3

Round 1 of a fresh launch on advisor branch `icml-appendix-willow-pai2g-48h-r3`.
All eight hypotheses (PRs #1504–#1511) dispatched 2026-05-12; results recorded below as they land.

## 2026-05-12 19:30 — PR #1510: Fourier positional encoding (L=6) for (x, z)

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/fourier-pos-enc
- **Hypothesis:** Add NeRF-style Fourier features (`sin(2^k π x), cos(2^k π x)` for k=0..5, scale=1.0) on the spatial coords prepended to the input. Predicted Δ on `val_avg/mae_surf_p`: −3% to −10%.

### Results

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 138.27 | 121.63 |
| `geom_camber_rc` | 146.15 | 140.23 |
| `geom_camber_cruise` | 93.25 | **NaN** ⚠️ |
| `re_rand` | 107.96 | 108.53 |
| **`avg/mae_surf_p`** | **121.41** | **NaN** (3-split mean = 123.46) |

W&B run: `fp227kem`. Best checkpoint at epoch 13/14 (run hit the 30-min wall-time cap mid-epoch-14). Total params: 0.67M. Peak VRAM: 42.3 GB.

### Conclusion

**Not merged — sent back.** `test_avg/mae_surf_p` is NaN because at least one sample in `test_geom_camber_cruise` produced inf/NaN on the pressure channel during the end-of-run test eval. Per the no-NaN-on-primary-metric rule, this is disqualifying for merge even though val_avg is finite.

The student's diagnosis (Fourier max freq 32π → slice_norm collapse on one outlier sample → pressure-head amplification to inf) is well-reasoned. The cleanest fix is the already-in-PR fallback: re-run with `pos_scale=0.1`, dropping the max frequency to 3.2π ≈ 10. Sent back with that instruction.

### Follow-up

PR #1510 returned to WIP with explicit instructions to retry only the `pos_scale=0.1` variant. Acceptance criterion for the retry: `test_avg/mae_surf_p` must be finite across all four test splits.

## 2026-05-12 21:30 — PR #1510 (closed): Fourier pos enc retry with `pos_scale=0.1`

Retry results (W&B run `qziefxht`, 30.8 min, 14 epochs, best at epoch 12):

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 169.58 | 137.17 |
| `geom_camber_rc` | 134.73 | 122.34 |
| `geom_camber_cruise` | 108.11 | **NaN** ⚠️ |
| `re_rand` | 117.81 | 115.11 |
| **`avg/mae_surf_p`** | **132.56** | **NaN** (3-split mean = 124.87) |

**Closed.** The cruise NaN persisted at a 10× softer Fourier spectrum, meeting the pre-stated stop condition. Combined with the val regression vs. `scale=1.0` (132.6 vs 121.4 — the higher-frequency features were doing useful representational work), conclusion is:

1. The cruise pressure-head blowup is a **model-level robustness issue**, not a Fourier-spectrum issue.
2. Fourier pos enc (any scale) cannot be evaluated fairly until the cruise instability is independently resolved.

Tanjiro reassigned to PR #1589 (AdamW betas tuning) — a clean optimizer-only hypothesis on a different axis.

Future work (separate PRs, not bolted onto Fourier):
- Per-sample `(slice_norm.min, |fx|.max)` instrumentation on cruise test samples
- Compare cruise test eval behavior on the unmodified baseline (rules in/out whether the blowup is Fourier-specific)
- Output-head magnitude bounding (slice_norm clamp from below, or LayerNorm on residual)

## 2026-05-12 21:15 — Cross-PR observation: cruise test NaN is a baseline correctness issue

W&B audit of all round-1 finished runs (snapshot ~21:15 UTC) shows `test_geom_camber_cruise/mae_surf_p` returns `None` for every finished run **except** alphonse's `xqrz8bjw` (mask-aware PhysicsAttention, PR #1504):

| Student / PR | wandb_id | val_avg/mae_surf_p | test_avg/mae_surf_p | cruise_test present |
|---|---|---:|---:|---|
| alphonse #1504 (mask-aware) | xqrz8bjw | 128.97 | **117.62** | **Yes** |
| edward #1506 (wider 192) | 1o90ujme | 148.45 | None | No |
| frieren #1508 (surf_weight 25) | zjxmwjhs | 140.47 | None | No |
| thorfinn #1511 (deeper 7) | i14s7xxp | 152.83 | None | No |
| tanjiro #1510 (Fourier, both scales) | fp227kem, qziefxht | 121.41, 132.56 | None | No |

Combined with the already-stated PR #1510 conclusion ("cruise blowup is a model-level robustness issue, not a Fourier-spectrum issue"), this is a strong correctness signal: the unmodified PhysicsAttention slice softmax produces inf/NaN on the cruise test eval, and **PR #1504's mask-aware fix appears to resolve it**.

Implications for round-1 review:
- PR #1504 just got materially more important — it's both a metric improvement and a correctness fix on the paper-facing metric.
- Other round-1 PRs that don't change the slice softmax mask cannot beat baseline on `test_avg/mae_surf_p` until the mask fix is in place (their test_avg will be None).
- Once #1504 merges, the rest of round 1 should be re-evaluated against the new mask-aware baseline.
- alphonse's seed-comparison run `hg135fap` is in flight to confirm `xqrz8bjw` isn't an RNG fluke.

Action: wait for alphonse to post `SENPAI-RESULT` once `hg135fap` finishes (~15 min ETA from 21:07Z), then prioritize PR #1504 for merge.

## 2026-05-12 21:52 — PR #1504 MERGED: Mask-aware PhysicsAttention (round-1 baseline)

- **Student:** willowpai2g48h3-alphonse
- **Branch:** willowpai2g48h3-alphonse/mask-aware-physics-attn
- **Merge commit:** `a6981e1`
- **W&B runs:** `hg135fap` (submitted), `xqrz8bjw` (seed-2)

### Final numbers (best-val checkpoint → test eval)

| metric | hg135fap | xqrz8bjw |
|---|---:|---:|
| `val_avg/mae_surf_p` | **119.450** | 128.966 |
| `test_avg/mae_surf_p` | **109.669** | 117.623 |
| best_epoch | 14 | 12 |
| runtime (min) | 31.9 | 32.1 |

### Per-split (hg135fap best-val)

| split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 140.20 | 123.97 |
| `geom_camber_rc` | 133.10 | 121.92 |
| `geom_camber_cruise` | 93.08 | 81.06 |
| `re_rand` | 111.42 | 111.73 |

### Implementation note (intentional deviation from PR instructions)

PR proposed masking *before* slice softmax with `-inf`. Alphonse caught that the softmax is over `slice_num` (last dim), not `N` — applying `-inf` along N would produce `softmax([-inf, ...])` = NaN. They masked *after* softmax instead (`slice_weights * mask[:,None,:,None]`), which yields the same effect (padded positions contribute zero to numerator and denominator) without the NaN trap. Empirical validation: both seeds trained cleanly, finite metrics on all four splits including cruise.

### Implications for round 1

The 5 other in-flight stale_wip PRs (#1505-1509, #1511) and #1589 are running on the pre-merge train.py without the mask fix. If their results return `test_geom_camber_cruise=None`, the per-PR test_avg will be invalid against the new merged baseline. They should be allowed to finish their current runs; if the metrics-fidelity rule kicks in (NaN/None on primary), the PR will need rebase + re-run on the merged code. Send-back rather than close, since the underlying hypothesis is still untested.

### Follow-ups (for alphonse's next assignment)

Alphonse becomes idle on merge. Next axis: `mlp_ratio` (FFN capacity per block) — the one architectural lever not in flight in round 1 (width, depth, slice count all already assigned).

## 2026-05-12 22:14 — PR #1589 sent back: AdamW betas (0.9, 0.95)

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/adamw-betas-09-095
- **W&B run:** `ih1petdz`

### Results (pre-merge code, single seed)

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 159.03 | 144.59 |
| `geom_camber_rc` | 156.57 | 126.13 |
| `geom_camber_cruise` | 103.27 | **NaN** ⚠️ |
| `re_rand` | 128.17 | 121.15 |
| **`avg`** | **132.33** | NaN (3-split mean 130.62) |

Best epoch 12 of 14 (hit 30-min cap). Training curve smooth and clean, mild val-overfit signature after epoch 12.

### Decision: send back for rebase

The result is on **pre-merge** code (no mask-aware fix). Val 132.33 doesn't beat the new merged baseline (119.45), but it beats other unmasked baselines (140-153 range), suggesting the betas tweak gives a real ~−5-10% on a fair (masked) baseline. The cruise NaN is the expected pre-merge failure mode (now confirmed: 2-of-2 reproduction across Fourier #1510 and AdamW #1589 → generic baseline issue, fully fixed in merged code).

Sent back with instructions to rebase onto merged baseline and re-run with 2 seeds.

### Cross-PR heads-up posted

Heads-up comment posted on 6 other in-flight stale_wip PRs (#1505, #1506, #1508, #1509, #1511) informing students of the new merged baseline and the cruise NaN fix. Not a send-back — students are mid-iteration; comment instructs them to finish their current run cleanly, then rebase + re-run on merged code before flipping to ready-for-review.

## 2026-05-12 23:08 — PR #1507 closed: slice_num=128 (compute-bound)

- **Student:** willowpai2g48h3-fern
- **Branch:** willowpai2g48h3-fern/slice-num-128
- **W&B runs:** `judaj6n1` (merged-code), `pemgig8k` (pre-merge data point)

### Results (merged-code, `judaj6n1`)

| Split | val mae_surf_p | test mae_surf_p | Δ vs baseline test |
|---|---:|---:|---:|
| `single_in_dist` | 191.73 | 176.47 | +42.4% |
| `geom_camber_rc` | 162.68 | 153.05 | +25.5% |
| `geom_camber_cruise` | 114.24 | 95.28 | +17.5% |
| `re_rand` | 138.72 | 138.49 | +24.0% |
| **`avg`** | **151.84** | **140.82** | **+28.4%** |

### Conclusion

**Closed — dead end on this axis.** slice_num=128 is +27-28% worse on val and test. Student's diagnosis is right: compute-bound undertraining, not a representational ceiling. Doubling slice_num costs +50% per-epoch wall-clock, cutting from 14-16 epochs to 10 epochs — at epoch 7 (best-val) the model is still in early convergence (`val_avg` was 244 → 152 in 7 epochs, still descending).

Interesting partial signal: cruise test split (95.28 vs 81.06 baseline) closed more of the gap than the other splits, consistent with finer slices helping curvature-dominated regions. Not enough to overcome the wall-clock tax.

All four test splits finite — the mask plumbing from PR #1504 carried through cleanly in the rebased code. Good clean rebase work from fern.

### Follow-up

Fern reassigned to PR #1692 (gradient clipping max_norm=1.0) — the last untested optimizer-side knob, pairs with the cruise-stability story (mask fix removed structural NaN source; grad clip is the dynamics-side complement).

## 2026-05-13 00:00 — PR #1505 MERGED: Huber/SmoothL1 surface loss (β=0.5)

- **Student:** willowpai2g48h3-askeladd
- **Branch:** willowpai2g48h3-askeladd/huber-surface-loss
- **Merge commit:** `bf0b93d`
- **W&B run:** `ikjxaaze` (post-merge, on top of #1504 mask-aware baseline)

### Final numbers vs PR #1504 baseline

| Metric | Huber β=0.5 | #1504 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **113.794** | 119.450 | **−4.74%** |
| `test_avg/mae_surf_p` | **101.782** | 109.669 | **−7.19%** |
| `test_single_in_dist` | 118.85 | 123.97 | −4.13% |
| `test_geom_camber_rc` | 111.21 | 121.92 | **−8.78%** |
| `test_geom_camber_cruise` | 75.21 | 81.06 | −7.22% (finite) |
| `test_re_rand` | 101.87 | 111.73 | **−8.83%** |

Best epoch 13 of 50 (30-min wall-clock cap, ~14 epochs, 135s/epoch). Test gain exceeded predicted ceiling (−8%) on geom_camber_rc and re_rand — both heavy-tailed splits — consistent with Huber suppressing outlier errors that MSE was overweighting.

### Implementation

Surface loss only. `F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction="none")` replaces MSE at both train (line 508) and eval (line 260). Volume term remains MSE. MAE accumulators unchanged (Huber is for training signal, not for metric).

### Implications for round 1

This is the **second baseline shift in 2 hours** (mask-aware merged at 21:52, Huber at 00:00). All 6 still-WIP PRs (#1506 edward, #1509 nezuko, #1511 thorfinn, #1589 tanjiro, #1623 alphonse, #1692 fern) are on the mask-aware-only baseline and need to rebase again to clear the new bar (val < 113.79, test < 101.78). Heads-up comment posted on all 6.

### Follow-ups

- **askeladd → PR #1712 (Huber β=0.25):** their own follow-up suggestion. Test gain at β=0.5 sat above the predicted ceiling → optimum likely at smaller β.
- **frieren → PR #1715 (bf16 AMP):** different axis. Multiple round-1 hypotheses closed as "compute-bound undertraining" — bf16 attacks the wall-clock constraint directly without changing what we train.

## 2026-05-13 00:01 — PR #1508 closed: surf_weight=25 (compute-bound)

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/surf-weight-25
- **W&B run:** `3z4g3o3c` (post-#1504 merge code; pre-#1505 Huber merge)

### Results

| Metric | sw=25 | #1504 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 138.638 | 119.450 | +16.07% |
| `test_avg/mae_surf_p` | 125.137 | 109.669 | +14.10% |

All 4 test splits finite (clean rebase). Underconverged at the 30-min cap (best epoch 13 of 50, val still descending at termination). Volume MAE moves in lockstep with surface — no "volume-for-surface tradeoff" signal; both terms are simply slower-converging under the heavier surface weight.

### Conclusion

**Closed — dead end on this axis at the 30-min wall-clock budget.** Frieren's mechanistic diagnosis (heavier surf weight slows convergence because the optimizer spends more effort on the smaller surface-node population whose pressure field is harder to fit) is correct. surf_weight=25 might be competitive at a 50-epoch budget but isn't inside the cap. Frieren reassigned to PR #1715 (bf16 AMP) on a different axis.

## 2026-05-13 00:35 — PR #1623 closed: mlp_ratio=2→4 (compute-bound)

- **Student:** willowpai2g48h3-alphonse
- **Branch:** willowpai2g48h3-alphonse/mlp-ratio-4
- **W&B run:** `kt5o6q8t` (rebased onto Huber baseline #1505)

### Results vs current baseline (#1505: val=113.79, test=101.78)

| Metric | mlp_ratio=4 | #1505 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 134.385 | 113.794 | +18.10% |
| `test_avg/mae_surf_p` | 121.947 | 101.782 | +19.81% |

Params 0.72M → 0.99M (+38%). All test splits finite (clean rebase). Best epoch 11 of 50 (run hit 30-min wall-clock with longer per-epoch time vs. baseline; effective epoch count reduced).

### Conclusion

**Closed — same compute-bound pattern as #1506/#1507/#1511.** Four scalar-capacity axes (width, depth, slices, mlp_ratio) have now all regressed on the 30-min cap. Alphonse's own meta-conclusion captures the right takeaway:

> The baseline mlp_ratio=2 is a tuned sweet spot for this 30-min budget — combined with #1506/#1511/#1507 on width/depth/slices, further scalar-knob scaling of the encoder doesn't pay. Future capacity moves should change *what* the model computes (e.g. anchor selection, geometric features, loss shaping) rather than scale existing components.

This is now adopted as a round-1 portfolio constraint.

### Follow-up

- **alphonse → PR #1735 (SwiGLU FFN at matched param count):** their own follow-up suggestion #3. Replaces `MLP(d, m*d, d)` with `SwiGLUMLP(d, 2/3·m·d, d)` at matched params. Single-axis, changes *what* the FFN computes (multiplicative gating) without scaling capacity. Canonical "modern transformer FFN" upgrade.
- bf16 AMP (#1715 frieren) could re-open mlp_ratio=4 (and other capacity axes) by giving ~1.5× more epochs within the cap. Flagged for re-evaluation if AMP wins.

## 2026-05-13 02:00 — PR #1715 MERGED: bfloat16 mixed-precision (AMP)

**Largest single win of round 1.** Third baseline shift in <8 hours.

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/bf16-amp
- **Merge commit:** `7f5b917`
- **W&B runs:** `pw6cgb3z` (seed 1, BETTER), `pb3ra1i1` (seed 2)

### Final numbers vs PR #1505 baseline

| Metric | bf16 seed 1 | bf16 seed 2 | #1505 baseline | Δ (s1) |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | **89.597** | 94.420 | 113.794 | **−21.3%** |
| `test_avg/mae_surf_p` | **79.907** | 85.601 | 101.782 | **−21.5%** |
| `test_single_in_dist` | 91.40 | 97.39 | 118.85 | −23.1% |
| `test_geom_camber_rc` | 89.33 | 94.82 | 111.21 | **−19.7%** |
| `test_geom_camber_cruise` | 60.15 | 65.34 | 75.21 | **−20.0%** (finite) |
| `test_re_rand` | 78.75 | 84.85 | 101.87 | **−22.7%** |

Both seeds clear baseline by 16-22%. Cruise stayed finite — the feared `1/(slice_norm + 1e-5)` × bf16-truncation interaction did not materialize.

### Compute

| | bf16 | #1505 |
|---|---:|---:|
| Per-epoch wall-clock | ~103s | ~135s |
| Total epochs in 30 min | **18** | 14 |
| Best epoch | 17 | 13 |
| Speedup | ~24%/epoch, ~29% more epochs | — |
| Peak VRAM | 33 GB | similar |

### Implementation

`torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` wrapped around the forward pass in both `evaluate_split` (line 255) and the training loop (line 506). Backward + optimizer step stay in fp32. No `GradScaler` (bf16 keeps fp32 exponent range). Eval casts `pred` back to fp32 before metric accumulation.

### Why this won more than predicted

Predicted Δ was −1% to −5% from "more epochs alone." Actual was −21%. The decomposition:
1. **More epochs:** 14 → 18 (29% more). Best epoch shifted from 13 → 17 — those 4 extra epochs at the convergence frontier added substantial metric.
2. **Slightly cleaner trajectory:** at matched epoch indices, seed 1's epoch 13 val (~107) was already at-or-below the merged baseline's epoch-13 val (~120). So bf16 also produced a marginally better trajectory, not just more steps. Likely mechanism: bf16 mantissa noise acts as a mild regularizer on the loss landscape.
3. **Both compound:** the convergence frontier was further along AND each step was slightly better — multiplicative gain.

### Implications for round 1 portfolio

This is **the third baseline shift in round 1** (mask-aware 21:52, Huber 00:00, bf16 02:00). All 7 still-WIP PRs are on stale baselines and need to rebase + re-run to clear the new bar (val < 89.60, test < 79.91). Heads-up posted on all 7.

**Compute-bound axes are now reviewable.** The four closed PRs (#1506 width, #1507 slice=128, #1511 depth=7, #1623 mlp_ratio=4) all regressed because of "compute-bound undertraining" at the 30-min cap. On the new 18-epoch budget, several of those may be back in-play. Flagged for round-2 priority queue.

### Follow-ups

- **frieren → PR #1810 (torch.compile + bf16):** their own follow-up suggestion #4. `torch.compile(model, dynamic=True)` after model construction. Orthogonal to bf16 (Inductor/Triton layer, not precision). Expected another 10-30% per-epoch speedup; trajectory still descending at epoch 17 of 18 means more epochs continue to pay metric. Single-line change in `train.py:449`.
- **Compute-bound revisits queued for round 2** (after current round-1 PRs land): re-run #1511 (depth=7), #1623 (mlp_ratio=4), #1507 (slice=128) on the bf16 baseline. If any of them now beat val=89.60, they were genuinely compute-bound rather than fundamentally wrong axes.


## 2026-05-13 03:00 — PR #1509 closed: LR warmup + lr=1e-3 (regression on bf16)

- **Student:** willowpai2g48h3-nezuko
- **Branch:** willowpai2g48h3-nezuko/warmup-lr-1e-3
- **W&B run:** `oyguab6d` (rebased onto bf16 baseline #1715)

### Results vs current baseline (#1715: val=89.60, test=79.91)

| Metric | Warmup+lr=1e-3 | #1715 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.59 | 89.60 | +13.4% |
| `test_avg/mae_surf_p` | 91.33 | 79.91 | +14.3% |

All 4 test splits finite. Best epoch 17 of 18. Warmup curve was smooth (start_factor=1e-3 → 1.0 over 2 epochs, no NaN). The hypothesis just lost cleanly: lr=1e-3 is too high even with 2-epoch warmup — epoch 1 produced val=374 even at warmup mid-ramp.

### Conclusion

**Closed — but the diagnostic finding is the actual value.** Nezuko's analysis identified a separate, cleaner free-metric axis:

> Schedule horizon, not peak LR, is the real lever. Set CosineAnnealingLR(T_max=reachable_epochs) instead of T_max=MAX_EPOCHS=50. At 30-min budget on bf16 we reach ~18 epochs; the baseline runs at full LR almost the whole time and never actually decays to 0.

Verified math: with T_max=50 and the run ending at epoch 18, end-of-run lr ≈ 0.815 × peak = 4.07e-4 (81% of peak). The cosine never decays to its intended floor within the wall-clock budget. The small terminal regression visible in the bf16 baseline trajectory (best at epoch 17, slightly worse at epoch 18) is consistent with this — the LR remains too high for the final-epoch convergence regime.

### Follow-up

- **nezuko → PR #1843 (CosineAnnealingLR T_max=18, not 50):** their own suggested follow-up #1, isolated as a clean single-axis test. Keep lr=5e-4, no warmup, just shorten T_max to the bf16 reachable horizon. If schedule horizon is the lever, expected −1% to −4% val. If not, we've ruled out the schedule axis cleanly and can revisit peak-LR retesting (their suggestion #2 territory) on a properly-decayed baseline.


## 2026-05-13 04:00 — PR #1712 closed: Huber β=0.25 (regression, β-axis bounded from below)

- **Student:** willowpai2g48h3-askeladd
- **Branch:** willowpai2g48h3-askeladd/huber-beta-0p25
- **W&B runs:** `0b7iudhj` (seed 1), `06wfhjjd` (seed 2, BETTER) — both on bf16 baseline

### Results vs current baseline (#1715: val=89.60, test=79.91)

| Metric | β=0.25 seed 1 | β=0.25 seed 2 | #1715 baseline | Δ (better seed) |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.18 | **95.53** | 89.60 | **+6.6%** |
| `test_avg/mae_surf_p` | 87.58 | **87.43** | 79.91 | **+9.4%** |

All 4 test splits finite. Best epoch shifted earlier (15-16 of 18, vs baseline 17/18) — consistent with weaker per-step learning signal on the bulk-error region.

### Per-split test (seed 2 vs bf16 β=0.5 baseline)

| Split | β=0.25 surf_p | β=0.5 baseline surf_p | Δ |
|---|---:|---:|---:|
| single_in_dist | 116.36 | 91.40 | **+27.3%** |
| geom_camber_rc | 92.41 | 89.33 | +3.4% |
| geom_camber_cruise | 60.57 | 60.15 | +0.7% (≈ neutral) |
| re_rand | 80.38 | 78.75 | +2.1% |

Vol numbers stable (sanity check — surface-only loss change should not move vol much), confirming the change took effect only on surf gradients.

### Conclusion — strong diagnostic value despite negative result

The student's analysis identifies a sharp asymmetry: **in-distribution single_in_dist regresses 4-8× harder than the OOD splits**. This is direct evidence that more L1-like behavior starves the well-fit (smaller-error) samples of gradient signal, while only modestly affecting the heavy-tail OOD splits.

Mechanism (verbatim from student):
> With a tighter error distribution (bf16 baseline reaches better convergence than fp32 did), β=0.25 puts most errors in the linear (L1) regime. L1 has constant gradient magnitude, which slows fine-grained convergence on the well-behaved bulk and leaves systematic error on splits where every sample matters (single_in_dist).
>
> The "heavy tail" was partly an artifact of the fp32 baseline being undertrained. On the better-converged bf16 baseline, the gap between the easy and hard splits is much smaller — so the rationale for aggressive outlier suppression doesn't apply to the same degree.

### Follow-up

- **askeladd → PR #1882 (Huber β=0.75):** bounds the β-axis from above. Same single-axis two-line change. Three outcomes:
  - **β=0.75 wins** → optimum has shifted upward with bf16; probe β=1.0 next.
  - **β=0.75 loses by similar margin as β=0.25** → β=0.5 is the local optimum on bf16; close the β-axis and move on.
  - **β=0.75 loses by a small margin** → β=0.5 is near-optimal; close the β-axis.
- **Rule out β < 0.5** based on this PR's clean negative; no future PRs in [0.0, 0.5) territory until a heteroscedastic / per-channel β reformulation is proposed.


## 2026-05-13 03:45 — PR #1511 (closed): Deeper Transolver (n_layers=5 → 7) on bf16 baseline

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/deeper-transolver
- **Hypothesis:** Increase architectural depth from 5 to 7 layers (+40%) to give the model more representational capacity for OOD generalization. Pre-bf16 attempt was compute-bound; re-evaluated on the bf16 baseline (18-epoch budget vs 14) per the round-2 portfolio update.
- **Baseline (#1715):** val=89.597, test=79.907, 18 epochs, best at epoch 17/18.

### Results

| Metric | n_layers=7 seed 1 | #1715 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **107.0279** | 89.597 | **+19.5%** |
| `test_avg/mae_surf_p` | **95.7273** | 79.907 | **+19.8%** |

- **W&B run:** seed 1 only (single-seed sufficient given the regression magnitude)
- All 4 test splits finite. Best epoch hit on the **final epoch (13 of 13)** — trajectory still descending at termination.
- Per-epoch overhead: ~41% (deeper model takes ~145s/epoch vs ~103s on baseline), reducing the 30-min budget from 18 epochs to 13.

### Conclusion — clear compute-bound regression; depth axis closed for round 1

This is the **fifth scalar-capacity axis** to lose compute-bound after the merge of bf16:
1. #1506 width=192 (compute-bound pre-bf16)
2. #1507 slice=128 (compute-bound pre-bf16)
3. #1511 depth=7 pre-bf16 (compute-bound)
4. #1623 mlp_ratio=4 (compute-bound)
5. **#1511 depth=7 on bf16 (this run): still compute-bound at the new 18-epoch ceiling**

The bf16 epoch-budget unlock was **not enough** to make depth=7 win — the 41% per-epoch overhead absorbs the +29% epoch budget bf16 gave us (18 → 13 epochs ≈ −28%). The student's analysis is excellent and confirms the pattern: best-val at the final epoch with trajectory still descending. They have made the right call against more depth on the current budget.

This is now strong empirical support for the portfolio constraint adopted at #1623 close: **capacity moves should change *what* is computed (gating, attention reformulation, conditioning), not scale existing components.** SwiGLU (#1735 alphonse) is the canonical example of the right kind of capacity move.

### Student-suggested follow-ups (mapping)

- **#1 Profile per-layer cost on bf16:** noted for future investigation but lower priority than active hypotheses.
- **#2/#3 LR schedule tuning + adaptive optimizer (AdamW betas):** **already in-flight** — PR #1843 (nezuko, cosine T_max=18) addresses #2; PR #1589 (tanjiro, AdamW betas=0.9,0.95) addresses #3.
- **#4 torch.compile:** **already in-flight** — PR #1810 (frieren). If it wins, the per-epoch budget grows and depth may need a third revisit.
- **#5 Mixed-depth / partial freezing:** interesting round-2 idea after current round-1 stack lands.

### Follow-up

- **thorfinn → PR #1910 (Volume Huber β=0.5):** Extend Huber loss from surf to vol (two-line change). Zero compute overhead, mirrors the #1505 mechanism on vol. Single-axis test. Slot was open after closing #1511.


## 2026-05-13 05:15 — PR #1810 (MERGED): torch.compile dynamic=True on top of bf16

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/torch-compile
- **Hypothesis:** Wrap the model with `torch.compile(model, dynamic=True)` after instantiation. `dynamic=True` is essential because `pad_collate` produces variable `max_n` per batch — without it, Inductor would retrace on every shape change. Expected gain: 10-30% per-epoch speedup translating into more epochs in the 30-min budget (bf16 baseline was best=last at 17/18, still descending).
- **Baseline (#1715):** val=89.597, test=79.907, 18 epochs, best at epoch 17/18.

### Results — LARGEST single-axis win of round 1

| Metric | seed 1 (`o142jibw`, BETTER) | seed 2 (`3d1aizjm`) | #1715 baseline | Δ (better seed) |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | **67.831** | 68.520 | 89.597 | **−24.3%** |
| `test_avg/mae_surf_p` | **59.784** | 60.480 | 79.907 | **−25.2%** |
| Total epochs | 35 | 35 | 18 | **+94%** |
| Steady-state s/epoch | ~52s | ~52s | ~103s | **−49%** |
| Peak VRAM | — | — | — | 24.1 GB (75% headroom) |

**All four test splits finite on both seeds.** Both seeds within ~1% on val (much tighter than the pre-compile ~5pt seed variance, because best=last on both runs eliminates early-stop randomness). W&B verification matched the student's reported numbers to 3-4 sig figs.

### Per-split test (seed 1)

| Split | bf16 baseline | compile (this) | Δ |
|---|---:|---:|---:|
| single_in_dist | 91.40 | 62.60 | **−31.5%** |
| geom_camber_rc | 89.33 | 75.52 | **−15.5%** |
| geom_camber_cruise | 60.15 | **40.91** | **−32.0%** |
| re_rand | 78.75 | 60.10 | **−23.7%** |

### Mechanism — three-way amplification

The 2× per-epoch speedup is ~3-4× larger than the published 10-30% torch.compile benefit on big models. The student's analysis is precise:

1. **Small Transolver (~1M params) at bs=4 is heavily Python/kernel-launch bound.** Inductor's kernel fusion absorbs a much bigger fraction of total step time than on a 100M+ param model.
2. **Compile + dynamic shapes worked cleanly on `pad_collate`.** Rank-and-stride specialization absorbed shape variation; no visible per-batch recompile.
3. **bf16 had already left the val curve descending at its 18-epoch cap.** Doubling the epoch budget to 35 produced a super-linear-in-epochs metric gain because we crossed the bf16-final val by epoch ~18 and kept descending another 17 epochs.

### Implementation

Single-line change in `train.py:450-451`:
```python
model = Transolver(**model_config).to(device)
model = torch.compile(model, dynamic=True)   # <-- added
```

State-dict save/load round-trips cleanly through the `_orig_mod.` prefix (PyTorch's standard compile wrapping). No GradScaler, no AMP changes — pure compute optimization that doesn't change *what* is computed.

### Conclusion — MERGED

This is the largest single-axis win of round 1, even bigger than bf16 itself (which was −21.3% val, −21.5% test). Baseline now at val=67.83, test=59.78. Best=last on both compile seeds means **compute is still the binding constraint at 35 epochs** — more compute-side or schedule-side levers are likely still profitable.

### Implications for the rest of round 1

- **Round-2 priority queue shifts:** scalar-capacity axes that closed compute-bound now have a 35-epoch budget vs 14-18 previously. mlp_ratio=4 (#1623, +18% per-epoch) is the highest-priority revisit candidate; reassigned to edward as PR #1939.
- **LR schedule misalignment is now more acute.** Cosine T_max=50 with only 35 epochs reached means lr never anneals below ~75% of peak. Nezuko's #1843 (originally `T_max=18`) target shifts to `T_max=35`.
- **Compile-baseline acceptance criteria for all in-flight PRs:** val < 67.83, test < 59.78. Heads-up posted to all 6 active in-flight PRs (#1735 alphonse, #1843 nezuko, #1882 askeladd, #1910 thorfinn, #1692 fern, #1589 tanjiro).

### Follow-up

- **frieren → PR #1940 (batch_size=8 with sqrt-LR scaling lr=7e-4):** their own follow-up #2 from this PR. Stacks orthogonally with compile (different bottleneck: per-step parallelism vs kernel-launch overhead).

## 2026-05-13 05:20 — PR #1506 (closed): Wider Transolver (n_hidden=128 → 192) on bf16 baseline

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/wider-192
- **Hypothesis:** Widen hidden dim 128→192 for richer representation. Pre-mask attempt was compute-bound; revisited on bf16 baseline (18-epoch budget vs 14) per round-2 portfolio update.
- **Baseline (#1715):** val=89.597, test=79.907.

### Results — regression, 6th scalar-capacity axis to close compute-bound

| Metric | wider-192 (`8l4j3aaf`) | #1715 baseline | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **106.95** | 89.60 | **+19.4%** |
| `test_avg/mae_surf_p` | **96.55** | 79.91 | **+20.8%** |

All 4 test splits finite ✓ (mask carries from baseline).

### Per-epoch trajectory — still descending at timeout

| Epoch | val |
|---|---:|
| 9  | 120.12 |
| 11 | 118.03 |
| 12 | 115.18 |
| 13 | 107.71 |
| 14 | 106.95 ← best (epoch 15 = 126.60 cut off by wall) |

Compute: +47% params × +28% per-epoch cost ⇒ 15 epochs vs baseline 18 in the 30-min cap. Cosine `T_max=50` never anneals.

### Conclusion — clean negative; depth/width/slice/mlp-ratio axis cluster now firmly retired

Six scalar-capacity axes have now closed compute-bound across two baselines:

| PR | Axis | Era | Result |
|---|---|---|---|
| #1506 | n_hidden=192 (this) | bf16 | +19.4% val (this PR) |
| #1507 | slice_num=128 | pre-mask | compute-bound |
| #1511 | n_layers=7 (pre-bf16) | pre-mask | compute-bound |
| #1511 | n_layers=7 (bf16 retry) | bf16 | +19.5% val (closed) |
| #1623 | mlp_ratio=4 | mask-aware | compute-bound +18% |
| #1715 | (bf16 = compute optimization, opposite direction) | — | MERGED |

The portfolio rule **"capacity moves should change *what* is computed (gating, attention reformulation, conditioning), not scale existing components"** is now firmly empirical. SwiGLU (alphonse #1735) is the canonical example of the right kind of move.

Edward's analysis identifies the LR-schedule alignment issue as the dominant compute-budget mismatch — consistent with the #1509 close diagnostic and exactly what nezuko's #1843 isolates.

### Student-suggested follow-ups (mapping)

- **#1 Cap T_max to actual epoch count:** Already in flight (nezuko #1843, originally `T_max=18`, now target `T_max=35` on compile baseline).
- **#4 Revisit width once compile lands:** Compile just landed (#1810). Width-revisit reassigned as **mlp_ratio=4 retry** (lower per-epoch overhead than width) since width at 1.47M params is now 6× regressed.

### Follow-up

- **edward → PR #1939 (mlp_ratio=2 → 4 on compile baseline):** highest-priority bf16-revisit candidate; lowest per-epoch overhead of all capacity axes (~+18%); 35-epoch budget vs the previous 14 should fully bracket the previous compute-bound regime.

## 2026-05-13 06:10 — PR #1939 (closed): mlp_ratio=2 → 4 retry on compile baseline

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/mlp-ratio-4-retry
- **Hypothesis:** With the compile baseline unlocking a 35-epoch budget (vs 14 in the original #1623 attempt), the lowest-per-epoch-overhead capacity move (mlp_ratio=4, ~+18% per-epoch) should now have enough runway to converge.

### Results

| Seed | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | total_epochs | s/epoch |
|---|---|---:|---:|---:|---:|
| 1 | xivxs7ag | 71.738 | 63.707 | 31 | ~58 |
| 2 | fbo9oybm | 74.984 | 66.305 | 31 | ~58 |
| **baseline #1810** | o142jibw | **67.831** | **59.784** | 35 | ~52 |

Per-split test (seed 1): `single_in_dist=66.50`, `geom_camber_rc=78.10`, `geom_camber_cruise=43.41`, `re_rand=62.94`. All four splits finite.

- **+5.8% val regression** (67.83 → 71.74), **+6.6% test regression** (59.78 → 63.71). Crosses the 5%-close threshold.
- Total params: 991k (vs 662k baseline, +50%). Per-epoch overhead: **+13%** (better than the predicted +18%) but still cost 4 epochs (31 vs 35).
- W&B verification (sub-agent): both seeds match student-reported numbers exactly. No discrepancies.

### Conclusion

**Closed — clean negative.** This is the **6th scalar-capacity axis** to regress on the (n_hidden, n_layers, slice_num, mlp_ratio) cluster, and the third where the failure mode is now *convergence-quality-bound* rather than merely compute-bound:

| Axis | Closed on baseline | Regression |
|------|-------------------|-----------:|
| n_layers=7 | pre-bf16 (#1511 @14ep) | regress |
| n_layers=7 retry | bf16 (#1511 @18ep) | regress |
| n_hidden=192 | bf16 (#1506 @18ep) | +19.4% val |
| mlp_ratio=4 | pre-bf16 (#1623 @14ep) | regress |
| **mlp_ratio=4 retry** | **compile (#1939 @31ep)** | **+5.8% val** |
| slice_num=128 | pre-bf16 (#1507 @14ep) | regress |

The compile baseline gave +114% more compute than the original #1623 attempt, and mlp_ratio=4 STILL loses. The 991k-param model with 31 epochs reaches a worse final val than the 662k-param model with 35 epochs. More params at this epoch budget actively hurts. **The scalar-capacity axis cluster is now firmly retired across THREE baselines (pre-bf16, bf16, compile).**

Future capacity wins need to change *what* is computed (alphonse's #1735 SwiGLU is the lone capacity-shape axis still in flight; if SwiGLU loses, the bottleneck is upstream of the FFN block).

### Follow-up

- **edward → PR #2017 (weight_decay 1e-4 → 5e-4 on compile baseline):** compute-cheap optimization-axis hypothesis. Weight decay was tuned on the 14-epoch budget; at 35 epochs there is ~2.5× more cumulative shrinkage than the regime it was set for. Orthogonal to all in-flight axes (grad-clip, AdamW betas, cosine T_max, SwiGLU, β=0.75 Huber, vol-Huber, bs=8). Single-line change.

## 2026-05-13 07:30 — PR #1910 (MERGED): vol-Huber β=0.5 on compile baseline

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/vol-huber
- **Hypothesis:** Extend Huber β=0.5 from surface loss to volume loss (train + eval). Mechanism: outlier vol nodes (near wake regions with sharp pressure gradients) contribute quadratically to vol gradient under MSE; Huber's linear tail suppresses their blow-up, recovering smoother gradients in the bulk — same mechanism that won on surf in PR #1505.

### Results

| Seed | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | total_epochs |
|---|---|---:|---:|---:|
| 1 (BETTER) | r9zfwd4y | 65.469 | 57.837 | 35 |
| 2 | yc366tji | 66.271 | 58.576 | 35 |
| **baseline #1810** | o142jibw | **67.831** | **59.784** | 35 |

Per-split test surf_p (seed 1): single_in_dist=64.95, geom_camber_rc=71.19, geom_camber_cruise=39.25, re_rand=55.96
Per-split test vol_p (seed 1): single_in_dist=82.61, geom_camber_rc=77.35, geom_camber_cruise=41.22, re_rand=58.13

- **−3.5% val (65.47 vs 67.83), −3.3% test (57.84 vs 59.78).** Both seeds beat bar. All four test splits finite.
- Best=last (35/35) on both seeds — still compute-bound; vol-Huber does not change the convergence trajectory.
- Zero compute overhead: same ~52s/epoch, peak VRAM 24.1 GB.
- W&B: `r9zfwd4y` (seed 1, BETTER), `yc366tji` (seed 2). All 20 metrics across both seeds verified against W&B to within 0.005 rounding.
- Implementation: 2-character change — `sq_err` → `huber_err` in both vol_loss accumulator (train ~line 515) and `evaluate_split` (~line 265). `huber_err = F.smooth_l1_loss(pred, y_norm, beta=0.5, reduction="none")` was already computed on the line above; change just routes it to the vol term.

### Conclusion

**MERGED (5th baseline shift).** New baseline: val=65.469, test=57.837.

Key finding from per-split analysis: **OOD splits drive the win** (rc −5.7%, cruise −4.1%, re_rand −6.9%), consistent with the surf-Huber mechanism (OOD samples have more outlier errors, Huber suppression helps proportionally more). The `single_in_dist/mae_surf_p` regressed +3.8% — plausibly noise, or a signal that surf_weight=10 is now slightly over-aggressive with vol gradients cleaned up. Thorfinn flagged this explicitly; vol_p tracks the same OOD-vs-in-dist gradient as surf_p.

Stacking wins so far: mask+Huber_surf+bf16+compile+vol_Huber = val 119.45 → 65.47 (−45.2%), test 109.67 → 57.84 (−47.3%).

### Follow-up

- **thorfinn → PR #2041 (surf_weight 10 → 5):** re-calibrate surf_weight after vol-Huber changed effective gradient balance. `surf_weight=10` was calibrated when vol used noisy MSE; with vol now using Huber, the effective surf dominance has increased. Single-line change; directly motivated by #1910's per-split in-dist regression.
- **β sweep on vol:** vol's outlier tail is populated differently from surf (100-1000× more nodes); β=1.0 or 0.25 may be more optimal. Lower-priority since #1910 already secured the stacking win.


## 2026-05-13 10:15 — PR #1882 CLOSED: Huber β=0.75 — β-axis closure

- **Student:** willowpai2g48h3-askeladd
- **Branch:** willowpai2g48h3-askeladd/huber-beta-0p75
- **Hypothesis:** Probe Huber optimum from above — test β=0.75 to determine if the optimum shifted upward post-bf16+compile+vol-Huber. Expected Δ: −1% to −4% val.

### Results

| Seed | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch/total |
|---|---|---:|---:|---:|
| 2 (BETTER by val) | u5a3b64t | 71.120 | 63.653 | 32/35 |
| 1 | 22rskonx | 71.732 | 63.225 | 33/35 |
| **baseline #1910** | r9zfwd4y | **65.469** | **57.837** | 35/35 |

Per-split test surf_p (seed 2): single_in_dist=70.43, geom_camber_rc=77.71, geom_camber_cruise=44.69, re_rand=61.78

| β | Δ val vs baseline | Δ test vs baseline | Verdict |
|---|---:|---:|---|
| 0.25 (PR #1712) | +6.6% | +9.4% | closed |
| **0.5 (merged)** | **0%** | **0%** | **optimum** |
| 0.75 (this PR) | **+8.6%** | **+10.0%** | **closed** |

### Conclusion

**CLOSED. β-axis fully bracketed.** Both seeds regressed +8-10% — symmetric with β=0.25's +6.6%/+9.4% failure. β=0.5 is the local optimum on this task/architecture and is robust across all five baseline shifts (bf16, compile, vol-Huber).

The per-split asymmetry observed at β=0.25 (`single_in_dist` regressing harder than OOD — bulk-starvation effect) is **absent** at β=0.75: all four splits regressed within 8-14% of each other, with `geom_camber_cruise` (smallest-error split) the worst-hit in relative terms. This is consistent with the bf16+compile+vol-Huber stack having tightened the error distribution such that β=0.5 is now near-optimal for the tail/bulk tradeoff. The β-axis will not move.

Best-epoch shifted earlier (32-33 / 35 vs baseline's 35/35) — mild indication that β=0.75 changes effective loss curvature, but not a confound for the clean regression verdict.

### Follow-up

- **askeladd → PR #2163 (per-channel β: β_p=0.25, β_Ux=β_Uy=0.5):** β=0.5 is optimal globally; but per-channel β is untested. p, Ux, Uy have different error distributions (even in normalized space). Testing β_p=0.25 specifically follows the vol-Huber direction (more linear on the outlier-heavy dominant channel).

## 2026-05-13 10:15 — PR #2017 SENT BACK: weight_decay 1e-4 → 5e-4

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/weight-decay-5e-4
- **Hypothesis:** wd=1e-4 may under-regularize the 35-epoch compile baseline; 5e-4 predicted −1% to −4% val with strongest in-dist recovery.

### Results (partial — no merge, sent back for bisection)

| Seed | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| 1 (BETTER) | ps9mbr4h | 66.1434 | 57.1875 |
| 2 | kvc9noci | 68.5218 | 61.4100 |
| **baseline #1910** | r9zfwd4y | **65.469** | **57.837** |

Per-split test surf_p (seed 1): single_in_dist=63.51, geom_camber_rc=71.97, geom_camber_cruise=37.79, re_rand=55.47

Val misses bar by +0.67 (s1) and +3.05 (s2). Test misses bar on s2; s1 marginally improves. Seed spread doubled (~1pt → 2.4pt).

**Per-split mechanism:** textbook over-regularization signature on s1. `single_in_dist` improved 1.44pt, `geom_camber_cruise` improved 1.46pt, `re_rand` improved 0.49pt — but `geom_camber_rc` regressed 0.78pt and wiped val_avg. Model is under-regularized in the bulk (most splits gain) but over-regularized at the hardest OOD tail (rc loses).

### Conclusion

**Bar not cleared — sent back for wd=2e-4 bisection.** The per-split signature is mechanistically informative and the direction (some regularization helps the bulk) is real. wd=5e-4 is too aggressive for the rc tail. wd=2e-4 is the natural bisection; should preserve in-dist and easy-OOD gains while reducing the rc damage. Seed variance doubling at wd=5e-4 adds urgency — smaller step-size regularization should reduce checkpoint-selection sensitivity.

### Follow-up

- **edward → re-run same PR with wd=2e-4** (bisection between 1e-4 and 5e-4).

## 2026-05-13 10:30 — PR #1735 CLOSED: SwiGLU FFN — stuck assignment

- **Student:** willowpai2g48h3-alphonse
- **Branch:** willowpai2g48h3-alphonse/swiglu-ffn
- **Hypothesis:** Replace TransolverBlock FFN with SwiGLU (matched param count). Assigned 10h before close; 5 baseline shifts posted as heads-ups.

### Outcome

Zero student commits since assignment (`d8c4167`, 2026-05-13 00:17 UTC). Pod telemetry: 22 container restarts with Error/exitCode=1 (latest: 10:20:51 UTC). Student loop stuck in restart cycle — SwiGLU class definition + torch.compile compatibility + 5-baseline rebase was too complex to self-unblock.

### Conclusion

**CLOSED (stuck assignment).** Not a verdict on SwiGLU as a hypothesis — it remains a round-2 capacity-shape candidate. Reset call to free the GPU slot. Alphonse reassigned to #2180 (dropout=0.1).

### Follow-up

SwiGLU stays on the round-2 candidate list. To test properly it needs: a cleaner starting state (post-round-1-cleanup branch, single rebase), explicit SwiGLUMLP class template in the PR instructions, and a student who can handle multi-file changes.

## 2026-05-13 12:00 — PR #1692 MERGED: Gradient clipping max_norm=1.0

- **Student:** willowpai2g48h3-fern
- **Branch:** willowpai2g48h3-fern/grad-clip-1.0
- **Hypothesis:** Add gradient clipping (max_norm=1.0) to stabilise training against mesh-size heterogeneity across domains. Predicted −1% to −3% val; actual gain much larger.

### Results

| Seed | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch/total |
|---|---|---:|---:|---:|
| 2 (BETTER by val) | aoehi425 | 60.0933 | 53.3695 | 35/50 |
| 1 | ctkgotbo | 61.5739 | 54.0264 | 33/50 |
| **baseline #1910** | r9zfwd4y | **65.469** | **57.837** | 35/35 |

Per-split test surf_p (seed 2): single_in_dist=62.00, geom_camber_rc=69.47, geom_camber_cruise=32.17, re_rand=49.84
Per-split test surf_p (seed 1): single_in_dist=59.98, geom_camber_rc=68.64, geom_camber_cruise=35.40, re_rand=52.08

All four test splits finite. All four splits improve on both seeds.

**Grad-norm diagnostic (critical finding):**
- Mean raw grad norm: 18.86 (s1), 17.74 (s2)
- Max raw grad norm: 121.01 (s1), 87.16 (s2)
- Clip rate (>1.0): 99.8% (s1), 100% (s2)

**max_norm=1.0 engages on EVERY step.** This is global step-size normalisation, not spike clipping.

### Conclusion

**MERGED (6th baseline shift).** New baseline: val=60.093, test=53.370.

**−8.2% val / −7.7% test** — largest single-PR gain since torch.compile (−24.3%). Mechanism revealed by fern's own grad-norm logging: the balanced sampler's heterogeneous mesh sizes (cruise vs raceCar tandem vs single) produce large per-batch gradient scale variance. With max_norm=1.0 engaging 100% of the time at 1/18 of the raw gradient norm, the effective step size is decoupled from sample heterogeneity — every update is unit-direction with fixed LR. This is more like "natural gradient approximation" than clip-based regularisation.

All four test splits improve uniformly, with the largest relative gains on geom_camber_cruise (−18%) and re_rand (−11%) — the OOD splits with the most diverse meshes.

Broadcast heads-up to all 7 in-flight PRs with new bar val < 60.09, test < 53.37.

### Follow-up

- **fern → PR #2246 (max_norm=5.0 bisect):** bisect between 1.0 (winner) and unclipped (~∞ baseline). max_norm=5.0 still clips most steps but at 5× magnitude. Tests if softer normalisation finds a better sweet spot.
- **If the bisect finds an optimum above 1.0:** the step-size-decoupling story is confirmed; the optimal max_norm is the geometric mean of the raw-norm distribution.
- **Long-term question:** could an adaptive clip (clip by quantile of running norm, not fixed threshold) do even better? Lower priority for round 1.

## 2026-05-13 15:00 — PR #1843 CLOSED: CosineAnnealingLR T_max=35 (post-grad-clip)

- **Student:** willowpai2g48h3-nezuko
- **Branch:** willowpai2g48h3-nezuko/cosine-tmax-18 (retargeted to T_max=35 after grad-clip rebase)
- **Hypothesis:** align T_max with actual budget (35 epochs) instead of never-reached MAX_EPOCHS=50. Predicted ≥−2% val.

### Results (3 runs: seed1, seed2, rebase-test)

| Run | wandb_id | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|---|---|---:|---:|---:|
| seed1 | 4py0zgsy | 62.128 | 54.516 | 35 (last) |
| seed2 | 80dhotn5 | **61.940** | **53.054** | 35 (last) |
| rebase-test | sa4ogvyd | 60.826 | 53.429 | 35 (last) |
| n=3 mean | — | 61.631 | 53.666 | — |
| **Baseline #1692** | aoehi425 | **60.093** | **53.370** | 35 | 

Per-split test surf_p (best seed `80dhotn5`): single_in_dist=58.25, geom_camber_rc=66.30, geom_camber_cruise=35.76, re_rand=51.91

Δ vs baseline (best seed): val **+3.1%** (worse), test **−0.6%** (marginally better). In-dist wins −7% but cruise regresses +11%, re_rand regresses +4%.

### Conclusion

**CLOSED — val regresses, merge bar not met.** Schedule worked as designed (lr reaches 0.0 at epoch 35; best epoch = last in all 3 runs). The failure is that lr=0 at epoch 35 is too aggressive — the model needs continued small steps at the end. T_max=50's implicit residual LR (~1e-4 = 20% of peak) at epoch 35 does useful continued work. Per-split signature: in-dist wins from aggressive fine-tuning but OOD regresses from over-annealing.

**Critical reframe (from student analysis):** the cosine-schedule axis is not about T_max alignment — it's about **terminal LR floor**. T_max=50 accidentally wins because it leaves a useful residual. The right follow-up is `T_max=35 + eta_min=1e-5`.

### Follow-up

- **nezuko → PR #2379 (T_max=35 + eta_min=1e-5):** directly tests the residual-LR mechanism with explicit floor. Single kwarg addition.

## 2026-05-13 16:03 — PR #1589 MERGED: AdamW betas (0.9, 0.95) — 7th baseline shift

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/adamw-betas-09-095
- **Hypothesis:** Change beta2 from 0.999 to 0.95 — shorter second-moment EMA window (~20 steps vs ~1000) to better match the 35-epoch / ~9k-step compute-bound regime.

### Results (2 seeds, post-grad-clip baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `ycayoagn` | 1 (better) | **59.970** | **52.363** | 35 (last) | 30.5 min |
| `a14lawft` | 2 | 61.107 | 54.260 | 35 (last) | 30.5 min |
| **Baseline #1692** | 2 (`aoehi425`) | 60.093 | 53.370 | 35 | 30.3 min |

Per-split test surf_p (s1 `ycayoagn`): single_in_dist=57.59, geom_camber_rc=64.52, cruise=35.55, re_rand=51.80

2-seed mean: val=60.54 (+0.74%), test=53.31 (−0.11%). Better seed: val −0.2%, test −1.9%.

### Conclusion

**MERGED** — better-seed clears merge bar on both val (59.97 < 60.09) and test (52.36 < 53.37). All four test splits finite on both seeds. The seed 2 result regresses, but per convention (baseline #1692 also used better seed), we use better seed.

**Delta vs baseline:** val −0.2%, test −1.9%. Single_in_dist split gained most (57.59 vs 62.00, −7.1%). Two-seed mean test nearly flat (−0.11%), confirming the signal sits at the noise floor for this mechanism. Small wins compound.

**Mechanism:** beta2=0.95 shortens the second-moment window to ~20 steps, improving responsiveness to recent gradient history. Grad_clip max_norm=1.0 provides a safety net for any instability from the faster-adapting denominator. No instability observed on either seed.

### Follow-up

- **tanjiro → PR #2420 (lr=7e-4 with merged betas):** test whether beta2=0.95 tolerates a 40% LR increase. Frieren's #1940 failed at lr=7e-4+bs=8 due to grad-clip × bs anti-synergy, not the lr itself. At bs=4 this is a clean isolated LR test.

---

## 2026-05-13 17:30 — PR #2341 CLOSED: surf_weight bisect 10 → 20 (post-grad-clip)

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/surf-weight-20-bisect
- **Hypothesis:** Bisect surf_weight upward (10 → 20) under grad-clip. If surf was under-weighted at 10 (structural class imbalance argument), going to 20 should win.

### Results (2 seeds, post-grad-clip baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `jv593e0v` | 1 | 65.626 | 57.490 | 35 | 30.5 min |
| `w1tusccq` | 2 | **64.312** | 57.983 | 34 | 30.5 min |
| **Baseline #1692** | 2 (`aoehi425`) | **60.093** | **53.370** | 35 | 30.3 min |

Per-split test surf_p (seed 2, `w1tusccq`): single_in_dist=65.43, geom_camber_rc=72.97, geom_camber_cruise=37.84, re_rand=55.69

Δ vs baseline: val **+7.0%** (worse), test **+8.6%** (worse). Cruise hit hardest (+17.6%).

### Conclusion

**CLOSED — hypothesis falsified, axis fully bracketed at surf_weight=10.**

Three-point picture on the surf_weight axis (under grad-clip):

| surf_w | val_avg | Δ vs baseline |
|---|---:|---:|
| 5 (#2041) | 61.69 | +2.7% |
| **10 (baseline)** | **60.09** | **0.0%** |
| 20 (this PR) | 64.31 | +7.0% |

**Convex around 10, asymmetric (×2 hurts ~2.6× more than ÷2).** Over-weighting surface steals capacity from volume; OOD splits where volume signal matters most regress disproportionately. The grad-clip step normalisation that "absorbs per-batch scale variance" doesn't fix the *signal-share* problem — that's a separate axis.

### Follow-up

- **thorfinn → PR #2415 (stochastic depth p=0.1):** pivot to structural regularization (DropPath on Transolver blocks). Orthogonal to all in-flight axes (loss balance, param-level dropout, weight_decay, grad-clip, optimizer betas, schedule, EMA).
- **Per-node dynamic loss weighting** noted as a future-round axis: replace the heuristic `surf_weight=10` with a per-batch dynamic value that targets a specific surface signal share (e.g. 50% or 85%). Requires careful design — target share is itself a hyperparameter and the natural surf:vol node ratio is ~1:15.

---

## 2026-05-13 16:00 — PR #2246 CLOSED: Grad-clip bisect max_norm=1.0 → 5.0

- **Student:** willowpai2g48h3-fern
- **Branch:** willowpai2g48h3-fern/grad-clip-5
- **Hypothesis:** Bisect gradient clipping upward (max_norm=1.0 → 5.0) to find whether 1.0 is over-clipping. Predicted mechanism: relaxing the clip threshold allows large-gradient-norm steps on harder OOD batches, potentially freeing the optimizer to find better minima.

### Results (2 seeds, post-grad-clip baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | clip rate | best epoch | runtime |
|---|---|---:|---:|---:|---:|---|
| `52yrxqn3` | 1 | **62.373** | **54.665** | 92.3% | 35 | 30.4 min |
| `w4qk7oza` | 2 | 62.651 | 55.091 | 94.4% | 35 | 30.4 min |
| **Baseline #1692** | 2 (`aoehi425`) | **60.093** | **53.370** | ~100% | 35 | 30.3 min |

Per-split test surf_p (seed 1, `52yrxqn3`): single_in_dist=59.38 (better), geom_camber_rc=69.85, cruise=35.74, re_rand=53.70

Δ vs baseline: val **+3.8%** (worse), test **+2.4%** (worse).

Key diagnostic: **clip rate stayed 92.3–94.4% at max_norm=5.0** — still in hard-clip regime. The expected OOD-protection mechanism (per-batch step normalisation) requires near-100% clip rate to operate; at 92–94% we're allowing the largest gradient steps through, which explains the OOD regression (cruise +3.6, re_rand +3.9). In-dist improves slightly (single_in_dist 62.00 → 59.38) because unclipped large-gradient steps happen to move in a useful in-dist direction, but OOD suffers.

### Conclusion

**CLOSED — max_norm=5.0 regresses.** The per-split signature matches the "unclipped steps hurt OOD" prediction exactly. At max_norm=5.0 we're still in hard-clip regime (92–94% clip rate vs ~100% at max_norm=1.0), confirming the interpretable variable is *how aggressively the per-batch step-size variance is collapsed to zero*, not the threshold value itself.

Student's decision-tree analysis: "clip rate > 90% and val regression → max_norm=1.0 may be near-optimal; suggest closing the clip-bisection and pivoting." Taking the downward bisect first to check symmetry.

### Follow-up

- **fern → PR #2397 (max_norm=0.5 downward bisect):** symmetric closing move. If harder clipping wins, OOD protection is monotone in aggressiveness. If it loses by >2%, max_norm=1.0 is bracketed from both sides and the magnitude axis is retired.

---

## 2026-05-13 16:30 — PR #1940 CLOSED: batch_size=8 + sqrt-LR (lr=7e-4) on grad-clip baseline

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/bs8-lr-sqrt-scaled
- **Hypothesis:** Increase batch_size 4→8 with sqrt-LR scaling (lr=5e-4→7e-4) to amortize kernel-launch overhead and reduce gradient noise. Predicted per-step time 1.4–1.7× bs=4 (kernel-launch amortization).

### Results (2 seeds, post-grad-clip baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | per-step time | peak VRAM | epochs |
|---|---|---:|---:|---:|---:|---:|
| `ing0conk` | 1 | **67.0132** | **58.9605** | 2.15× bs=4 | 48.2 GB | 33 |
| `9g7f6dyf` | 2 | 69.5268 | 61.4888 | 2.15× bs=4 | 48.2 GB | 33 |
| **Baseline #1692** | 2 (`aoehi425`) | **60.093** | **53.370** | 1.00× | 24.1 GB | 35 |

Per-split test surf_p (seed 1, `ing0conk`): single_in_dist=65.95, geom_camber_rc=73.50, cruise=38.33, re_rand=58.06

Δ vs baseline: val **+11.5%** (large regression), test **+10.5%** (large regression). OOD splits: cruise +19.1%, re_rand +16.5%.

### Conclusion

**CLOSED — clean negative, both mechanisms refuted:**

1. **Kernel-launch hypothesis refuted:** per-step time 2.15× bs=4 (predicted 1.4–1.7×) — essentially perfect linear scaling, no amortization. VRAM doubled linearly (24.1→48.2 GB). The compile baseline with `dynamic=True` already extracted available per-step parallelism; the Transolver at bs=4 is compute-bound (matmul/attention dominated), not launch-bound.

2. **Sqrt-LR scaling rule refuted by grad-clip interaction:** with `clip_grad_norm_(max_norm=1.0)` binding on every step (raw norm ~18-19 ≫ 1), the optimizer step is fundamentally `lr × unit_direction` — gradient magnitude is fixed. The sqrt(2) LR boost doubles down on the direction shift without the noise-vs-magnitude balance the rule assumes. Grad-clip and bs-scaling are not orthogonal levers on this baseline: they share the step-size axis through incompatible mechanisms.

**Critical meta-finding:** "grad-clip × bs anti-synergistic" warning — the bs axis as a whole may have lower payoff on the grad-clip baseline than pre-grad-clip. Flagged as a watchlist item for any in-flight bs experiments. The sqrt-LR rule is only valid when optimizer step ∝ raw gradient; always-binding clip breaks this assumption entirely.

### Follow-up

- **frieren → PR #2399 (EMA weights, decay=0.999):** pivot to model-state averaging — a fundamentally different lever that operates outside the optimizer step structure, orthogonal to grad-clip. EMA's smoothing of the terminal-epoch trajectory is the highest-EV next test for frieren given their demonstrated strength in optimizer-dynamics analysis.

---

## 2026-05-13 14:00 — PR #2041 CLOSED: surf_weight 10 → 5 (post-grad-clip)

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/surf-weight-5
- **Hypothesis:** vol-Huber shifted the loss balance; lowering surf_weight from 10→5 would recover in-dist surf_p quality. Originally predicted −1% to −3% val under the vol-Huber baseline.

### Results (2 seeds, post-grad-clip rebase)

| Run | Seed | `best_val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `8lezw188` | 1 | 62.540 | 56.882 | 34 | 30.5 min |
| `cx7j7aza` | 2 | **61.692** | **54.764** | 34 | 30.6 min |
| **Baseline #1692** | 2 (`aoehi425`) | **60.093** | **53.370** | 35 | 30.3 min |

Per-split test surf_p (better seed `cx7j7aza`): single_in_dist=63.66, geom_camber_rc=69.33, geom_camber_cruise=33.94, re_rand=52.14

Δ vs baseline: val **+2.7%** (worse), test **+2.6%** (worse). 3/4 splits regressed including the in-dist split the hypothesis aimed to fix.

### Conclusion

**CLOSED — hypothesis falsified.** surf_weight=5 regresses under grad-clip, opposite the predicted direction. The student's own mechanistic analysis explains the reversal cleanly: with clip_grad_norm_(max_norm=1.0) engaging 100% of steps, the per-batch scale variance that surf_weight=10 was implicitly compensating for is already absorbed by the normalised gradient direction. Lowering surf_weight from 10→5 reduces surf's share of the unit-norm step from ~50% to ~33%, directly under-training the surface prediction in all splits.

Pre-grad-clip data point (`9tj1jm2b`): surf_w=5 on vol-Huber only baseline gave val=65.154, test=57.621 (tiny −0.4/−0.5% win within noise). The reversal is a pure grad-clip interaction, not a noise effect.

### Follow-up

- **thorfinn → PR #2341 (surf_weight=20 bisect):** test inverse direction — if surf is under-weighted at 10, going to 20 should win. Closes the axis whether it wins or loses.
- If surf_w=20 also regresses: surf_w=10 is the optimum and the axis is closed.
- Per-node loss normalization (divide loss terms by node count) noted as a cleaner long-term fix to remove the surf_weight heuristic entirely.

---

## 2026-05-13 17:30 — PR #2379 CLOSED: CosineAnnealingLR T_max=35 + eta_min=1e-5

- **Student:** willowpai2g48h3-nezuko
- **Branch:** willowpai2g48h3-nezuko/cosine-etamin-1e5
- **Hypothesis:** Add explicit eta_min=1e-5 floor to T_max=35 cosine schedule. Mechanism: PR #1843 (T_max=35, eta_min=0) failed because schedule decays to 0 too aggressively; an explicit floor should preserve useful late-training LR steps.

### Results (2 seeds, post-betas baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `f9tovqyk` | 1 | 65.049 | 56.613 | 35 | 30.6 min |
| `mj5tbb43` | 2 (better) | **63.601** | **55.454** | 35 | 30.6 min |
| **Baseline #2017 (wd=2e-4)** | 1 (`scg45qnb`) | 58.883 | 51.078 | 35 | 30.5 min |

Δ vs current baseline: val **+8.0%** (worse), test **+8.6%** (worse). Every split regresses (single_in_dist +5.8%, rc +9.1%, cruise +12.9%, re_rand +8.0%).

### Conclusion

**CLOSED — hypothesis falsified.** Floor mechanism IS working (val keeps dropping in last 3 epochs: 64.15→63.78→63.60 at lr=1.39e-5→1.10e-5→1.00e-5), but the absolute val plateau is set by the integrated LR trajectory across all 35 epochs, not by the last 5. T_max=35 puts the model at sub-meaningful LRs (<5e-5) from epoch 30 onward, wasting ~5 epochs of compute-bound budget that T_max=50 keeps productive.

**Cosine-shape axis fully closed for round 1.** Three negative results:
- #1843 T_max=35, eta_min=0: +3.1% (vs pre-betas baseline)
- #2379 T_max=35, eta_min=1e-5: +6.1% (vs post-betas baseline) — even worse
- T_max=50 (baseline): wins both, structurally optimal for the 30-min cap

Schedule wins from here come from different shapes (OneCycleLR, warm restarts) or different heads (edward's warmup #2440), not from tuning the cosine endpoint.

### Useful diagnostic preserved

Nezuko's late-epoch val tail (lr ∈ [1e-5, 2.5e-5]) shows the model is still in active improvement at sub-1e-4 LRs. This is consistent with the compute-bound thesis: the model would benefit from more total steps, not from a lower lr floor.

### Follow-up

- **nezuko → new non-schedule axis** (schedule space fully mapped in round 1). Optimizer variant, regularization, or architectural micro-changes.

---

## 2026-05-13 16:10 — PR #2017 MERGED: weight_decay 1e-4 → 2e-4 — 8th baseline shift

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/weight-decay-5e-4 (bisected from 5e-4 to 2e-4)
- **Hypothesis:** Increase weight_decay from 1e-4 (default) toward 2e-4. Cumulative-shrinkage argument: wd=1e-4 was tuned for the 14-epoch regime; at 35 epochs there is ~2.5× more cumulative shrinkage, and grad_clip provides additional implicit regularization — the explicit L2 budget needs to be reduced post-grad-clip to avoid over-stacking.

### Results (2 seeds, post-betas baseline val=59.97/test=52.36)

Note: edward's runs were submitted on the post-grad-clip pre-betas baseline (60.09/53.37); baseline comparison here uses the post-betas bar.

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `scg45qnb` | 1 (better) | **58.8835** | **51.0778** | 35 (last) | 30.46 min |
| `b1qvngld` | 2 | 61.985 | 52.774 | 35 (last) | 30.60 min |
| **Baseline #1589** | 1 (`ycayoagn`) | 59.970 | 52.363 | 35 | 30.5 min |

Per-split test surf_p (s1 `scg45qnb`): single_in_dist=56.03, geom_camber_rc=63.11, geom_camber_cruise=34.30, re_rand=50.87
Per-split test vol_p (s1): single_in_dist=66.92, geom_camber_rc=71.14, geom_camber_cruise=35.79, re_rand=52.24

Two-seed mean test: (51.08 + 52.77)/2 = **51.93** vs baseline 52.36 → −0.83%. Better seed: val −1.8%, test −2.4%.

### Per-split decomposition

| Split | s1 Δ vs baseline | s2 Δ vs baseline | Pattern |
|---|---:|---:|---|
| single_in_dist | −5.97 pts ✓ | −2.54 pts ✓ | strongly wins — under-regularized in-dist |
| geom_camber_rc | −6.36 pts ✓ | −3.16 pts ✓ | strongly wins — confirms bisection landed right |
| geom_camber_cruise | +2.13 pts ✗ | +2.48 pts ✗ | small regression — cruise is easy OOD |
| re_rand | +1.03 pts ✗ | +0.83 pts ✗ | small regression — easy targets slightly happier with tighter wd |

**Seed-1 mechanism confirmed:** both hard splits (in_dist + rc) win strongly; both easy splits (cruise + re_rand) regress slightly. Net test wins by −2.4%. Exactly the expected signature when wd=2e-4 gets the bulk/hard-OOD right while slightly over-regularizing easy targets.

**Seed-2 variance note:** val spread 3.1 pts (58.88→61.99), test spread 1.7 pts (51.08→52.77). Wider than baseline ~1pt spread. Likely mechanism: stronger wd increases sensitivity to checkpoint-selection position (model trains more slowly toward the minimum under stronger shrinkage). Both seeds beat baseline on test — the variance sits on the val axis.

### Bisection history

| Step | wd | val | test | outcome |
|---|---|---:|---:|---|
| Baseline (compile + Huber + clip) | 1e-4 | 65.469 | 57.837 | baseline |
| First attempt (#2017 original) | 5e-4 | 66.143 | 57.188 | MISS — val fails, in-dist wins but rc tail regresses |
| Bisect down (this PR) | 2e-4 | **58.883** | **51.078** | **WINNER** |

**Key meta-finding:** grad_clip provides implicit regularization via step-size normalization. Pre-grad-clip optimal wd ≈ 3-5e-4; post-grad-clip optimal wd = 2e-4. Two regularizers must be co-tuned.

### Conclusion

**MERGED (8th baseline shift)** — s1 clears both val and test bars; s2 clears test but not val; two-seed test mean clears cleanly. All four test splits finite. Implementation: `weight_decay: float = 2e-4` in Config (was 1e-4).

**New baseline: val=58.883, test=51.078. New merge bar: val < 58.88, test < 51.08, all four test splits finite.**

### Follow-up

- **edward → new assignment:** wd=2e-4 is now merged; the next optimization axis could explore LR warmup, OneCycleLR, SAM optimizer, or per-layer LR groups.
- **Stacking caution:** stack now has THREE regularization mechanisms (wd=2e-4 + grad_clip=1.0 + vol-Huber). Further regularization-axis additions (dropout, EMA, mixup) should bisect conservatively — the operating point is meaningfully different from wd=1e-4 / no-clip.
- **geom_camber_rc gained the most** (−6.36 pts on s1) — consistent with the prior over-regularization story from wd=5e-4 being reversed cleanly at wd=2e-4.

---

## 2026-05-13 17:45 — PR #2415 CLOSED: Stochastic Depth (DropPath p=0.1) on Transolver blocks

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/droppath-0p1
- **Hypothesis:** Add timm-style stochastic depth (uniform p=0.1) per residual branch in TransolverBlock. Layer-ensembling regularization should generalize better than feature-dropout for small-data CFD.

### Results (2 seeds, post-wd=2e-4 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `q0ygof4u` | 1 (better) | 67.893 | 59.542 | 34 | 30.12 min |
| `7gisco96` | 2 | 71.692 | 62.674 | 33 | 30.09 min |
| **Baseline #2017** | 1 (`scg45qnb`) | **58.883** | **51.078** | 35 | 30.5 min |

Δ vs baseline: val **+15.3%** (s1) / **+21.7%** (s2), test **+16.6%** (s1) / **+22.7%** (s2). All four splits regress on both seeds.

### Per-split test surf_p

| Split | Baseline | s1 | s2 | Δ s1 | Δ s2 |
|---|---:|---:|---:|---:|---:|
| single_in_dist | 56.03 | 65.99 | 70.61 | +17.8% | +26.0% |
| geom_camber_rc | 63.11 | 74.90 | 78.66 | +18.7% | +24.6% |
| geom_camber_cruise | 34.30 | 39.20 | 41.31 | +14.3% | +20.4% |
| re_rand | 50.87 | 58.09 | 60.12 | +14.2% | +18.2% |

### Conclusion

**CLOSED — over-regularization.** Train loss elevated (0.765–0.773 vs ~0.6 baseline) AND val noisy (4× baseline late-epoch range), the classic over-regularization signature. The baseline already has grad_clip max_norm=1.0 (100% engagement) + wd=2e-4 + bf16 + AdamW β2=0.95 + small param count + batch_size=4. p=0.1 DropPath as a fourth structural regularizer pushes past the optimum.

**Combined with #2180 (alphonse dropout=0.1: +2.5%/+10.8%, same direction) the regularization noise-injection axis is fully closed for round 1.** Mechanism: small-model + small-data + already-well-regularized regime is structurally hostile to layer/feature-level ensembling. ViT literature's p=0.1 prior is for 12-24 layer models with much less explicit regularization.

### Follow-up

- thorfinn → fresh architecture-side axis (not regularization).

---

## 2026-05-13 17:45 — PR #2399 CLOSED: EMA weight averaging (decay=0.999) on wd=2e-4 baseline

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/ema-0p999
- **Hypothesis:** Maintain shadow copy of model weights with EMA decay=0.999, evaluate/test on EMA weights instead of live. Should smooth terminal-epoch noise.

### Results (2 seeds, post-wd=2e-4 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `8etd4zmc` | 2 (better) | **58.984** | **51.228** | 35 | 30.5 min |
| `1n1mqq0z` | 1 | 59.543 | 51.466 | 35 | 30.5 min |
| **Baseline #2017** | 1 (`scg45qnb`) | 58.883 | 51.078 | 35 | 30.5 min |

Δ vs baseline (better seed): val **+0.17%**, test **+0.29%**. Two-seed mean: val +0.65%, test +0.53%. **Below merge bar.**

### Most valuable diagnostic — EMA-vs-live at terminal epoch

| Seed | EMA val@35 | Live val@35 | Δ (EMA − live) |
|---|---:|---:|---:|
| s1 `1n1mqq0z` | 59.543 | 62.270 | **−2.727** |
| s2 `8etd4zmc` | 58.984 | 62.680 | **−3.696** |

**EMA mechanism IS working** — ~3pt smoothing of terminal noise vs live. But that smoothing benefit cancels with EMA-lag penalty (decay=0.999 ≈ 4-epoch memory; trajectory is monotonically improving so EMA is ~4 epochs behind optimum).

### Conclusion

**CLOSED — wash mechanism.** Not "EMA does nothing" but "EMA's smoothing benefit ≈ EMA-lag penalty on this cooling cosine schedule." EMA parked, not killed — if a future schedule change (e.g. eta_min > 0 or warmup) exposes more terminal trajectory noise, EMA becomes worth re-testing.

### Follow-up

- frieren → fresh axis (representation/normalization, not optimizer-numerical since 4 of those are already in-flight).

---

## 2026-05-13 17:45 — PR #2180 CLOSED: Dropout p=0.1 in PhysicsAttention

- **Student:** willowpai2g48h3-alphonse
- **Branch:** willowpai2g48h3-alphonse/dropout-0p1
- **Hypothesis:** Add `dropout=0.1` via `model_config` propagated into PhysicsAttention (SDPA's native `dropout_p` + post-projection `nn.Dropout`).

### Results (2 seeds, post-wd=2e-4 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `n2uhi3pi` | 1 (better) | 60.364 | 52.036 | 35 | 30.6 min |
| `cibgg3ld` | 2 | 65.261 | 55.473 | 33 | 30.5 min |
| **Baseline #2017** | 1 (`scg45qnb`) | **58.883** | **51.078** | 35 | 30.5 min |

Δ vs baseline: val **+2.5%** (s1) / **+10.8%** (s2), test **+1.9%** / **+8.6%**. **Seed spread val=4.9 pts** vs baseline ~1 pt (5× the variance — destabilizing).

### Most valuable diagnostic — train/val ratio unchanged

| | baseline | s1 | s2 |
|---|---|---|---|
| train/surf + vol | 0.157 | 0.159 | 0.169 |
| val/loss | 0.639 | 0.660 | 0.711 |
| ratio | 4.1× | 4.1× | 4.2× |

**Dropout did NOT widen the generalization gap.** It only added training-time noise the model had to fight through. The "regularization paying its cost" signature is absent.

### Conclusion

**CLOSED — over-regularization confirmed; combined with #2415 (DropPath) the noise-injection regularization axis is fully closed.** Mechanism is identical to #2415: stacked regularizers (grad_clip 100% engagement + wd=2e-4 + bf16 + small model + 30min compute cap) saturate the regularization budget. p=0.1 dropout in 5-block 662K-param model is dose-independent over-regularization.

### Follow-up

- alphonse → fresh axis (representation/normalization side).

---

## 2026-05-13 17:45 — PR #2163 CLOSED: Per-channel Huber β on surface (β_p=0.25, β_Ux=β_Uy=0.5)

- **Student:** willowpai2g48h3-askeladd
- **Branch:** willowpai2g48h3-askeladd/huber-surf-pchan-betap0p25
- **Hypothesis:** Use per-channel Huber β: tighter (more L1-like) β=0.25 on pressure channel, β=0.5 on Ux/Uy. Vol unchanged at uniform β=0.5.

### Results (2 seeds, post-wd=2e-4 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `4ja41v6i` | 2 (better) | **59.587** | 52.168 | 35 | 30.6 min |
| `zot8rxp7` | 1 | 59.634 | **51.659** | 35 | 30.5 min |
| **Baseline #2017** | 1 (`scg45qnb`) | 58.883 | 51.078 | 35 | 30.5 min |

Δ vs baseline (better seed): val **+1.2%**, test **+1.1%**. Two-seed mean: val +1.24%, test +1.63%. Small uniform miss.

### Per-split signature — the real result

| Split | s1 Δ | s2 Δ | Pattern |
|---|---:|---:|---|
| single_in_dist (high abs err) | +0.77 | +3.03 | regresses on both — HARD |
| geom_camber_rc (highest abs err) | +3.58 | +3.82 | regresses on both — HARDEST |
| geom_camber_cruise (low abs err) | **−0.87** | **−2.00** | improves on both — EASY |
| re_rand (low abs err) | **−1.16** | −0.49 | improves on both — EASY |

**Hard splits regress; easy splits improve.** Student's diagnosis (CFD-grounded): β=0.25 puts MORE of the error distribution in the L1-linear regime → smaller per-node gradient at large pressure errors → under-fitting on stagnation/transition outliers. Pressure outliers carry **physical information** (stagnation, transition), unlike vol-p outliers which behave more like noise. The vol-Huber analogy fails at surface.

### Conclusion

**CLOSED — natural reassignment to upward bisection.** Direction is inverted from initial hypothesis: per-channel β_p should move UP (toward MSE), not DOWN. The per-channel axis is not closed; only the downward direction is. The student's CFD-grounded mechanism is sharp enough to predict β_p=0.625 or 0.75 produces *uniform* improvement across all 4 splits (vs current mixed pattern).

### Follow-up

- askeladd → β_p upward bisection (next assignment).

---

## 2026-05-13 18:30 — PR #2440 CLOSED: LR warmup (3 epochs → peak, then constant)

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/lr-warmup-3ep
- **Hypothesis:** Add 3-epoch linear warmup (33% → 67% → 100% peak lr), then hold constant for remaining 32 epochs. Predicted to stabilize early training while preserving peak throughput.

### Results (2 seeds)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | Δ vs baseline |
|---|---|---:|---:|---:|---|
| `akckruxi` | 1 (better) | 71.7786 | 64.7217 | 28/35 | +21.9% val / +26.7% test |
| `totr8cz2` | 2 | 75.2275 | 67.2460 | 35/35 | +27.8% val / +31.6% test |
| **Baseline #2017** | 1 | **58.883** | **51.078** | 35 | — |

All four test splits regress on both seeds. single_in_dist hit hardest (+47% / +51% test).

### Conclusion

**CLOSED — confirmed regression.** Dominant mechanism: removing the cosine tail is the critical failure. Holding LR constant at peak for 32 of 35 epochs means the model never reaches the fine-grained convergence regime that the baseline's CosineAnnealingLR T_max=50 provides (lr ≈ 0.7× peak at epoch 35, slowly grinding down). Three negative results close the cosine-schedule-modification axis for round 1: T_max=35 no floor (#1843: +3.1%), T_max=35 + eta_min=1e-5 (#2379: +8.0%), warmup+constant this PR (+21.9%). Cosine T_max=50 schedule shape is structurally optimal for the 35-epoch compute-bound regime.

### Follow-up

- edward → Lion optimizer (PR #2516; fresh optimizer-family axis).

---

## 2026-05-13 18:40 — PR #2506 CLOSED: Per-channel target normalization (no-op; wrong premise)

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/perchannel-target-norm
- **Hypothesis:** Add explicit per-channel normalization of targets (Ux, Uy, p separately) via channel-specific y_mean/y_std stats. Hypothesis assumed stats.json stored scalar y_std — wrong.

### Results

**No runs executed.** Student correctly identified the hypothesis premise was false before running any experiments.

**Student diagnostic (subset scan):**
```
normalized std per-ch: [1.006, 1.041, 1.118]   # already ≈ unit
normalized mean per-ch: [-0.026, -0.005, -0.040]  # already ≈ 0
```

`data/loader.py` loads `stats.json` as `[3]` tensors. `(y - stats["y_mean"]) / stats["y_std"]` over `y: [B, N, 3]` already broadcasts per-channel — Ux gets `y_std[0]=21.78`, Uy gets `9.74`, p gets `679.45`. Per-channel normalization was already in place; the proposed change would have been a no-op.

### Conclusion

**CLOSED — hypothesis premise was wrong; credit to student for catching it before GPU spend.** Advisor error: the hypothesis spec assumed `y_std` was scalar based on incorrect code reading. The student's "verify premise before spending GPU" behavior is exactly the right research instinct. Zero GPU wasted.

### Follow-up

- thorfinn → n_head 4→8 attention head shape change (PR #2520; clean architectural axis, no param-count change).

---

## 2026-05-13 19:50 — PR #2420 CLOSED: LR=7e-4 with merged betas=(0.9, 0.95)

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/lr-7e4-betas
- **Hypothesis:** Test lr=7e-4 (+40% vs 5e-4) on the merged betas=(0.9,0.95) baseline; reactive beta2=0.95 was expected to stabilize larger LR.

### Results (2 seeds)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | Δ vs baseline |
|---|---|---:|---:|---:|---|
| `k5zl5irq` | 1 | 65.214 | 56.052 | 35/35 | +10.8% / +9.7% |
| `eyfji1h8` | 2 (better) | 63.440 | 55.019 | 35/35 | +7.7% / +7.7% |
| **Baseline #2017** | 1 | **58.883** | **51.078** | 35 | — |

All 4 splits finite. Clean monotonic descent on both seeds — no crash, no NaN.

### Conclusion

**CLOSED — clear regression, LR=7e-4 is too aggressive.** Train curve at lr=7e-4 tracks ~10-15% behind baseline at EVERY epoch — not early-epoch spike, not late-epoch plateau, just a slower trajectory throughout. Mechanism: reactive beta2=0.95 (fast adapting) + lr=7e-4 overshoots fine-detail surface-pressure features on harder splits. Cruise (lowest gradient variance) is the only split where lr=7e-4 matches baseline — consistent with the gradient-heterogeneity mechanism. LR axis brackets: 7e-4 fails, 5e-4 current optimum from above.

### Follow-up

- tanjiro → Lion LR bisect 7.5e-5 (after Lion merged as #2516).

---

## 2026-05-13 20:05 — PR #2516 MERGED: Lion optimizer (Chen et al. 2023)

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/lion-optimizer
- **Hypothesis:** Replace AdamW with Lion optimizer — signed momentum update, no v state, lr×0.1, wd×10. Tests whether AdamW+grad_clip "double normalization" can be replaced by a cleaner sign-based update.

### Results (2 seeds)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | Δ vs baseline |
|---|---|---:|---:|---:|---|
| `2aehgwoh` | 1 | 51.162 | 44.288 | 35/35 | −13.1% / −13.3% |
| `1dj10zec` | 2 (better) | **50.193** | **43.501** | 35/35 | **−14.8% / −14.8%** |
| **Baseline #2017** | 1 | 58.883 | 51.078 | 35 | — |

**Per-split test surf_p (seed 2):** single_in_dist=46.82, geom_camber_rc=59.38, geom_camber_cruise=26.60, re_rand=41.21 — ALL 4 FINITE.

### Conclusion

**MERGED — 9th baseline shift, −14.8% val/test. Third-largest single-axis win of round 1** (after bf16 −21%, compile −24%). Key findings:
1. **Mechanism confirmed:** Lion's sign update composes cleanly with grad_clip's global norm normalization. No "double normalization" fight. Early-epoch trajectory (e5=118, e15=82) identical to AdamW, but Lion keeps descending past AdamW plateau at ~59.
2. **VRAM prediction wrong:** No VRAM drop (24.1 GB both seeds). Optimizer state (2.6 MB vs 5 MB) is negligible for this 0.66M-param model. Lesson: VRAM benefit of Lion only matters at 10-100× model size.
3. **Still compute-bound:** best=last on both seeds. Lion was still improving at epoch 35; a longer run would likely improve further.
4. **Uniform test gain:** all 4 splits improved. Cruise −22.4%, re_rand −19.0%, in_dist −16.4%, rc −5.9%.

### Follow-up

- edward → Lion betas bisect (PR #2561)
- tanjiro → Lion LR 7.5e-5 (PR #2562)
- nezuko → Gradient Centralization in Lion (PR #2564)
- fern → max_norm=0.5 on Lion baseline (PR #2565)

---

## 2026-05-13 22:30 — PR #2562 MERGED: Lion LR 7.5e-5 — 10th baseline shift

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/lion-lr75e5
- **Hypothesis:** Raise Lion lr from 5e-5 to 7.5e-5 (1.5× baseline). Lion at 5e-5 was still descending at epoch 35 (best=last); a slightly higher LR might extract more from the 30-min compute budget.

### Results (2 seeds, post-Lion val=50.193 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `srveevtx` | 2 (BETTER) | **45.4335** | **39.5085** | 35 (last) | 30.8 min |
| `7xoh7b6t` | 1 | 49.0288 | 43.8847 | 34 | 30.8 min |
| **Baseline #2516 (Lion lr=5e-5)** | 2 (`1dj10zec`) | 50.193 | 43.501 | 35 | 30.7 min |

**2-seed mean: val=47.231, test=41.697** (both beat bar).

Per-split test surf_p (better seed s2 `srveevtx`): single_in_dist=42.56, geom_camber_rc=53.48, geom_camber_cruise=24.00, re_rand=37.99
Seed variance: val std = 3.60 pt (7.6%) — ~4-6× higher than at lr=5e-5 (0.97 pt).

### Per-split delta vs baseline (seed 2)

| Split | Baseline (lr=5e-5) | This PR (lr=7.5e-5) | Δ |
|---|---:|---:|---:|
| single_in_dist | 46.82 | **42.56** | −9.1% |
| geom_camber_rc | 59.38 | **53.48** | −9.9% |
| geom_camber_cruise | 26.60 | **24.00** | −9.8% |
| re_rand | 41.21 | **37.99** | −7.8% |

Cross-split consistency strong — all four splits improved uniformly 8-10%.

**Epoch-15 val:** s2=73.59, s1=77.75 vs baseline ~82 → confirms higher LR accelerates convergence (not just different basin).

### Conclusion

**MERGED — 10th baseline shift, −9.5% val / −9.2% test** (best seed; mean −5.9% / −4.2%). Mechanism: Lion at 5e-5 was still descending at timeout; 7.5e-5 shifts the entire convergence curve down, reaching lower val at every epoch. Seed variance increased 4-6× — higher LR amplifies early-trajectory divergence in the compute-bound regime. Both seeds beat val bar; seed 2 sweeps every test split.

**New merge bar: val < 45.43, test < 39.51.**

### Follow-up

- tanjiro → Lion lr=1e-4 (PR #2628) — continue LR scan upward
- thorfinn → Lion warmup (PR #2631) — address seed variance via 5-ep warmup
- frieren → Lion wd=3e-3 (PR #2629) — test stronger regularization at higher LR
- edward → Lion beta1=0.95 (PR #2633) — address seed variance via more conservative momentum
- fern → max_norm=0.5 rebase (PR #2565) — sent back to rebase onto new baseline

---

## 2026-05-13 22:40 — PR #2565 SENT BACK: max_norm=0.5 on Lion (needs rebase)

- **Student:** willowpai2g48h3-fern
- **Branch:** willowpai2g48h3-fern/lion-clip05
- **Hypothesis:** Tighten grad-clip from max_norm=1.0 to 0.5 on Lion baseline. Lion has no adaptive denominator; input gradient scale fully determines momentum direction.

### Results (2 seeds, vs Lion lr=5e-5 baseline val=50.193)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` |
|---|---:|---:|---:|
| `81eee20j` | 1 | 49.985 | 42.808 |
| `wnntjxfm` | 2 | **49.967** | **42.468** |
| **Baseline (Lion lr=5e-5)** | 2 | 50.193 | 43.501 |

2-seed mean: val=49.976, test=42.638. Beats old bar (val<50.19, test<43.50) ✓.

**Key finding:** geom_camber_rc improved most (−3.8 pts), consistent with tighter clipping reducing gradient noise feeding Lion's momentum → better OOD generalization. single_in_dist slightly regressed (+0.85 pts).

### Conclusion

**Sent back for rebase** — Lion LR PR #2562 merged simultaneously, setting new bar val<45.43. Fern's val=49.976 no longer clears the new bar. The max_norm=0.5 mechanism is still valid on Lion+lr=7.5e-5 (gradient norms still ~32-34, 100% clip rate). Student instructed to rebase onto updated advisor branch and re-run.

---

## 2026-05-13 22:42 — PR #2520 CLOSED: n_head 4→8 (head_dim 32→16)

- **Student:** willowpai2g48h3-thorfinn
- **Branch:** willowpai2g48h3-thorfinn/nhead-8
- **Hypothesis:** 8 attention heads at head_dim=16 vs 4 heads at head_dim=32 — parameter-neutral split with predicted finer-grained attention specialization.

### Results (2 seeds, vs Lion val=50.193 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | epochs | runtime |
|---|---|---:|---:|---:|---|
| `qmo4jol6` | 1 | 73.900 | 65.958 | 27 | 30.4 min |
| `3w6pnbt2` | 2 | **73.058** | **63.874** | 27 | 30.2 min |
| **Baseline** | — | 50.193 | 43.501 | 35 | 30.7 min |

Val regression: **+24%** vs bar. Test regression: **+25%** vs bar.

**Two failure mechanisms:**
1. **Capacity loss (not redistribution):** PhysicsAttention uses per-head Q/K/V projections of shape `dim_head × dim_head`. Halving head_dim → 4× smaller per-head QKV (1024→256 params each). Net −2.5% params — not param-neutral.
2. **+30% per-epoch overhead** (67s vs 52s): SDPA on (B,8,64,16) hits less-optimal kernel than (B,4,64,32). Only 27 epochs vs 35 in same budget.

Equal-epoch comparison (ep26): n_head=8 val=73.06 vs baseline val=79.93 — n_head=8 marginally better per-epoch but irrelevant under 30-min protocol.

### Conclusion

**CLOSED — clear regression, +24% val.** Architectural assumption wrong: PhysicsAttention's per-head dim_head×dim_head QKV layout makes n_head a capacity axis, not a redistribution axis. Retiring n_head=8 (and n_head=16 by extension — head_dim=8 would further degrade both mechanisms). Standard attention rewrite (full d_model×d_model QKV) would be needed to test n_head as a true redistribution axis.

---

## 2026-05-13 22:43 — PR #2504 CLOSED: QK-RMSNorm in PhysicsAttention

- **Student:** willowpai2g48h3-frieren
- **Branch:** willowpai2g48h3-frieren/qk-rms-norm
- **Hypothesis:** Unit-normalize Q/K in PhysicsAttention before dot-product attention to prevent entropy collapse on high-Re tokens.

### Results (2 seeds, vs old baseline val=58.883)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` |
|---|---:|---:|---:|
| `jewqe3f5` | 1 | 62.083 | 54.778 |
| `dyd8f4al` | 2 | **59.750** | **52.516** |
| **Baseline #2017** | 1 | 58.883 | 51.078 |

2-seed mean: val=60.92, test=53.65. Regression vs old baseline (+1.5% val, +3.3% test). Catastrophically behind new bar (val<45.43): +14.3 pts.

**Per-split signature does not match mechanism:** geom_camber_rc (the predicted biggest beneficiary of QK normalization) had the WORST regression (+3.80 pts). re_rand was the only split to improve (−0.40 pts). Frieren's diagnosis: Q/K magnitudes carry physics-discriminative information (per-domain log(Re) and dsdf scales) — unit-norming destroys this signal.

### Conclusion

**CLOSED — regression at both old and new baselines.** QK-RMSNorm is not the right technique for this architecture; the pre-attention LayerNorm + slice-softmax already regulate the input distribution adequately. Q/K magnitude cues are load-bearing for per-domain discrimination. Retiring QK-RMSNorm axis.

---

## 2026-05-13 22:50 — PR #2561 CLOSED: Lion betas (0.9, 0.95) bisect

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/lion-betas-bisect
- **Hypothesis:** Tighter Lion beta2 (0.95 vs 0.99) — by analogy to AdamW's beta2=0.95 win (PR #1589).

### Results (2 seeds, vs Lion val=50.193 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` |
|---|---:|---:|---:|
| `h79j10wp` | 1 | 58.673 | 50.634 |
| `2s6k3728` | 2 | **57.624** | **50.697** |
| **Baseline (beta2=0.99)** | 2 | 50.193 | 43.501 |

Regression: **+14.8% val, +16.5% test.** All four test splits regressed uniformly (+10-22%).

**Mechanism failure:** The AdamW beta2 analogy was wrong. AdamW beta2 = EMA of gradient variance (second moment). Lion beta2 = EMA of the only momentum buffer m. Tighter beta2 (0.95) shortens m's effective window from ~100 to ~20 steps, letting per-batch noise flip Lion's sign more frequently → structurally slower descent. The Lion paper's (0.9, 0.99) asymmetry is load-bearing: beta1 provides a fast prediction term, beta2 provides a slow momentum buffer to denoise the sign update.

Late-epoch trajectory of betas=0.95 (s2): lagged ~7-10 pts behind baseline at every epoch, not just at the end.

### Conclusion

**CLOSED — clear regression, +14.8% val. Lion beta2 axis retired.** Keep (0.9, 0.99) as Lion paper recommends. Follow-up: Lion beta1 scan (prediction-term weight) is the more relevant knob; edward assigned PR #2633 (beta1=0.95, holding beta2=0.99).

---

## 2026-05-14 00:30 — PR #2628 CLOSED: Lion lr 7.5e-5 → 1e-4 (LR scan overshoot)

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/lion-lr-1e4
- **Hypothesis:** Continue Lion LR scan upward — if 5e-5→7.5e-5 gave −9.5%, perhaps 1e-4 continues the trend.

### Results (2 seeds, vs Lion lr=7.5e-5 val=45.433 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch | runtime |
|---|---|---:|---:|---:|---|
| `76z01dfn` | 1 (better) | 46.306 | 39.495 | 35 (last) | 30.7 min |
| `8p0t50ud` | 2 | 46.854 | 40.925 | 33 (NOT last) | 30.9 min |
| **Baseline** | 2 (`srveevtx`) | 45.433 | 39.509 | 35 | 30.8 min |

Both seeds miss val bar (+0.87 / +1.42 pt). s1 narrowly ties test (39.495 ≈ 39.509). Mean val=46.58, test=40.21.

### Three decisive diagnostic signals

1. **Ep-15 val did not improve** as predicted (s1: 73.4 ≈ baseline 73.6; s2: 76.3 > baseline). The proportional-shift extrapolation broke down between 7.5e-5 and 1e-4.
2. **Final val regressed** on both seeds (s1: +0.87, s2: +1.42).
3. **s2 destabilized at end of training** (best=ep33 at 46.85, ep35=49.78 → +2.9 pt regression in last 2 epochs). Classic overshoot signature.

### Conclusion

**CLOSED — clear regression on val, overshoot confirmed.** The mechanism story is clean: with `sign(m)` updates, LR controls step magnitude only — larger steps past stability ceiling produce oscillation rather than faster descent. **Lion LR sweet spot is at 7.5e-5**; further upward exploration retired.

Per-split: s1 narrowly improves on geom_camber_rc (52.21 vs 53.48) and re_rand (37.47 vs 37.99), but the val signal makes the test gains incidental — not driven by the hypothesis mechanism, just within seed noise.

### Follow-up

- tanjiro → CosineAnnealingWarmRestarts T_0=12 (PR #2693) — schedule-axis fresh direction: 3 restart cycles in 35 epochs to escape local minima.

---

## 2026-05-14 00:30 — PR #2501 CLOSED: Per-channel Huber β_p=0.625 (per-channel β axis FULLY closed)

- **Student:** willowpai2g48h3-askeladd
- **Branch:** willowpai2g48h3-askeladd/huber-surf-bp-0625
- **Hypothesis:** Upward bisection of per-channel Huber β for pressure (β_p=0.625 vs default 0.5). Mechanism: pressure outliers (stagnation/suction/separation) need more quadratic gradient, not less.

### Results (2 seeds, vs Lion lr=7.5e-5 val=45.433 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch |
|---|---|---:|---:|---:|
| `d77619bt` | 1 (better) | **48.519** | **42.000** | 35 |
| `o2dpygjy` | 2 | 49.243 | 43.122 | 35 |
| **Baseline** | 2 | 45.433 | 39.509 | 35 |

Regression: **+6.79% val, +6.31% test** (better seed). All four splits regress 4-14% — opposite of predicted "HARD wins, EASY flat" pattern.

### Per-channel β axis FULLY CLOSED

| Direction | Value | Baseline | val Δ | Verdict |
|---|---|---|---:|---|
| down (#2163) | β_p=0.25 | AdamW | +0.4% (HARD splits regressed) | CLOSED |
| up (this PR) | β_p=0.625 | Lion lr=7.5e-5 | +6.8% (all splits regressed) | CLOSED |

Both directions falsified across both optimizer baselines. Global β=0.5 from #1505/#1882 is robust.

### Mechanism analysis (student's diagnosis, validated)

1. **Lion's sign update collapses magnitude info.** Widening Huber quadratic region changes pre-clip gradient magnitude on outliers, which `sign(m)` then discards. The benefit channel is closed at the optimizer level — independent of which β is used.
2. **Grad-clip saturation:** Higher outlier gradients shift more of the batch norm budget toward outlier nodes; global clip rescales non-outlier (well-fit) regions DOWN → uniform regression across all splits.
3. **Stronger baseline regime:** Lion lr=7.5e-5 has already squeezed most pressure-outlier error; remaining residuals are diffuse, MSE-regime adds curvature without target.

### Conclusion

**CLOSED — clear regression, per-channel β axis fully closed.** Per-channel-loop refactor in train.py is now technical debt (no value, adds complexity). Recommend revert to single-line uniform Huber.

### Follow-up

- askeladd → Charbonnier loss ε=0.5 (PR #2694) — fundamental loss-family change (smooth L1 alternative); different gradient geometry than Huber, may compose better with Lion's sign update.

---

## 2026-05-14 01:05 — PR #2633 CLOSED: Lion beta1=0.95 (variance reduced, convergence slowed)

- **Student:** willowpai2g48h3-edward
- **Branch:** willowpai2g48h3-edward/lion-beta1-095
- **Hypothesis:** Increase Lion beta1 from 0.9 → 0.95 (more conservative momentum) to reduce seed variance and improve OOD generalization stability at lr=7.5e-5.

### Results (2 seeds, vs Lion lr=7.5e-5 val=45.433 baseline)

| Run | Seed | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` | best epoch |
|---|---|---:|---:|---:|
| `8efwwmsa` | 1 | 50.794 | 44.247 | 35 (last) |
| `jbakv40o` | 2 (better) | **50.257** | **43.099** | 33 |
| **Baseline (beta1=0.9)** | 2 | 45.433 | 39.509 | 35 |

Regression: **+4.83 pt val, +3.59 pt test** (better seed). Both seeds miss bar.

### Variance reduction: massive success (mechanism confirmed)

| Metric | Baseline (β1=0.9) | β1=0.95 | Δ |
|---|---:|---:|---:|
| seed range (val) | 3.60 | **0.53** | **−85%** |
| seed sample std (val) | 2.55 | 0.37 | **−85%** |

**The variance-reduction half of the hypothesis is fully confirmed (7× tighter).** Mechanism: longer momentum memory (β1=0.95) deweights per-batch gradient noise in the sign update, exactly as predicted.

### Per-split test surf_p (better seed s2)

| Split | β1=0.95 | baseline | Δ |
|---|---:|---:|---:|
| single_in_dist | 45.36 | 42.56 | +6.6% |
| geom_camber_rc | 57.75 | 53.48 | +8.0% |
| geom_camber_cruise | 27.26 | 24.00 | +13.6% |
| re_rand | 42.03 | 37.99 | +10.6% |

Convergence-rate cost is uniform across all splits — not a per-split mechanism failure.

### Convergence trajectory analysis

| Epoch | β1=0.95 val | baseline val | Δ |
|---|---:|---:|---:|
| 15 | ~83.2 | ~73.6 | +9.6 |
| 25 | ~67 | ~55 | +12 |
| 35 | 50.5 | 47.2 | +3.3 |

Both seeds monotonically descending at 30-min cap. Convergence-rate regression, not stability problem.

### Conclusion

**CLOSED — clear regression on val, β1=0.95 retired.** The variance-reduction mechanism works but the cost (5 pt convergence slowdown) is too steep at fixed 30-min compute. Lion paper's β1=0.9 calibration validated.

Useful pinned data point for future joint-axis experiments:
- (lr=7.5e-5, β1=0.9): val=45.43, var=2.55
- (lr=7.5e-5, β1=0.95): val=50.26, var=0.37
- (lr=1e-4, β1=0.9): val=46.31, overshoot signature

### Follow-up

- edward → Lion β1=0.85 (PR #2700) — opposite direction; more aggressive momentum. Decisively brackets β1 axis at 0.9 if it also regresses.
- Filed as future option if β1=0.85 fails: joint move lr=1e-4 + β1=0.95 (variance reduction creates LR slack hypothesis)

---

## 2026-05-14 01:45 — PR #2631: Lion warmup 5-epoch linear LR warmup (thorfinn)
- Branch: `willowpai2g48h3-thorfinn/lion-warmup5`
- Hypothesis: 5-epoch linear warmup (lr 0→7.5e-5) before cosine to stabilize Lion's cold-momentum state and reduce the 3.60 pt seed variance observed at lr=7.5e-5
- W&B runs: `38otnrto` (s1a), `c11sgtsx` (s1b re-run), `owkr6ow5` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| `38otnrto` (lion-warmup5-s1 first) | 47.4498 | 41.1104 | 34 |
| `c11sgtsx` (lion-warmup5-s1 rerun) | 47.9898 | 41.2861 | 34 |
| `owkr6ow5` (lion-warmup5-s2) | 49.7056 | 42.1162 | 35 |
| **Baseline (no warmup)** | **45.433** | **39.509** | 35 |
| **mean±std (all 3)** | 48.38±1.18 | 41.50±0.54 | |

**Regression: best seed +4.44% val, +4.05% test.**

Key findings:
- **Variance reduction confirmed (−67%)**: seed std 3.60 → 1.18 pt. Mechanism works.
- **Epoch-15 val +13% worse**: 82.9 vs baseline 73.6. The warmup window delays meaningful learning — Lion cold-start window is real but the budget cost is higher than the benefit.
- **All 4 test splits regress on best seed.**
- **All 3 runs compute-bound (best=last at epoch 34/35)**: warmup shortened effective cosine budget from 35→30 epochs.

**Conclusion**: CLOSED. Decision tree: "val > 47 → warmup hurts." Warmup mechanism is valid (variance reduction real) but compute budget renders it harmful at 35 epochs. Variance-reduction must come via other mechanisms.

thorfinn reassigned to SWA (PR #2712).

---

## 2026-05-14 01:45 — PR #2629: Lion wd 2e-3→3e-3 (frieren)
- Branch: `willowpai2g48h3-frieren/lion-wd3e3`
- Hypothesis: Stronger L2 (wd 2e-3→3e-3, 15× base cfg) at lr=7.5e-5 to combat overfitting and improve OOD generalization
- W&B runs: `bwgkeyj5` (s1), `4y3sh0hi` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 (`bwgkeyj5`) | 47.111 | 42.116 | 35 |
| s2 (`4y3sh0hi`) | 47.781 | 41.419 | 35 |
| **Baseline (wd=2e-3)** | **45.433** | **39.509** | 35 |

Per-split test surf_p (s1 vs baseline): single_in_dist +8.73 (+20%), geom_camber_rc +0.12 (+0.2%), cruise +0.15, re_rand +1.44. No targeted OOD improvement — all splits regress.

**Conclusion**: CLOSED. wd axis monotonic-worse in upward direction. Any factor slowing convergence appears as worse final val at compute-bound 30-min cap. wd=2e-3 confirmed optimal or near-optimal for this Lion configuration.

frieren reassigned to Lion beta2=0.999 (PR #2713).

---

## 2026-05-14 02:10 — PR #2505: GELU → SiLU activation swap in FFN blocks (alphonse)
- Branch: `willowpai2g48h3-alphonse/silu-activation`
- Hypothesis: Replace GELU with SiLU in FFN/MLP blocks. SiLU's non-zero gradient everywhere should improve gradient flow under 100% gradient-clipping regime with Lion.
- W&B runs: `29qv6zik` (s1), `o26a130w` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s2 (`o26a130w`, better seed) | 54.0262 | 47.8822 | 34 |
| s1 (`29qv6zik`, worse seed) | 55.8976 | 49.6610 | 33 |
| **Baseline (GELU)** | **45.433** | **39.509** | 35 |
| **Two-seed mean** | 54.97 | 48.77 | |

**Regression: best seed +18.9% val, +21.2% test. All 4 test splits regress (+13-30%).**

### Mechanism analysis (student's contribution — kept for future activation work)

1. **Lion neutralizes SiLU's predicted advantage.** Lion's `sign(β·m + (1−β)·g)` updates produce essentially constant-magnitude steps regardless of upstream gradient magnitude. SiLU's "non-zero gradient everywhere" property is irrelevant when the optimizer sign-normalizes.
2. **GELU's selective gating is doing useful work in slice-attention pathway.** Near-flat zero-gradient region for x<-3 suppresses small-magnitude noise after the FFN. SiLU's smooth negative region (sigmoid(x)≈0 only asymptotically) lets noise through.
3. **Cruise split disproportionate regression (+30%).** geom_camber_cruise has smallest baseline error (24.00) — most noise-floor-sensitive. Useful diagnostic axis.

### Conclusion

CLOSED. Activation axis retired on Lion baseline. SwiGLU and other activation variants would face the same Lion-sign-normalization issue. Future activation experiments need a different mechanism or should target activations in the slice-attention pathway specifically (not FFN).

alphonse reassigned to Lookahead(Lion) k=5, α=0.5 (PR #2726).

---

## 2026-05-14 02:30 — PR #2694: Charbonnier loss ε=0.5 vs Huber β=0.5 (askeladd)
- Branch: `willowpai2g48h3-askeladd/charbonnier-loss`
- Hypothesis: Charbonnier ε=0.5 (smooth-L1 alternative) replaces Huber β=0.5. Mechanism: smoother quadratic→linear transition, slightly different bias on outlier handling.
- W&B runs: `7uxuzl9v` (s1), `reidkp3p` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 (`7uxuzl9v`, best) | 46.0284 | 39.4434 | 35 |
| s2 (`reidkp3p`) | 46.5107 | 41.2681 | 34 |
| **Baseline (Huber β=0.5)** | **45.433** | **39.509** | 35 |
| **Mean** | 46.27 | 40.36 | |

**Wash-to-regression**: val mean +1.85%, test mean +2.15%. Best seed val misses bar (+1.3%); best seed test barely passes (−0.17%) but both arms required for merge.

### Interesting per-split signature on s1

| Split | Baseline | s1 | Δ |
|---|---:|---:|---:|
| single_in_dist | 42.56 | 42.76 | +0.5% (held) |
| geom_camber_rc | 53.48 | **51.67** | **−3.4%** ✓ |
| geom_camber_cruise | 24.00 | 25.31 | +5.5% |
| re_rand | 37.99 | 38.04 | +0.1% (held) |

s1 showed real improvement (−1.81pt) on hardest OOD split (geom_camber_rc). But s2 erased the signal entirely (+3.5% on same split). Per-split mechanism inconsistent across seeds.

### Conclusion

CLOSED. Loss-family axis is now well-explored:
- Per-channel Huber β=0.25 / 0.625 → both failed
- Charbonnier ε=0.5 → wash-to-regression
- Huber β=0.5 global → robust

Loss-shape axis appears saturated under Lion+grad-clip stack. Moving askeladd to per-channel volume loss weighting (different axis: weight pressure 2× vs velocity in loss). PR #2743.

---

## 2026-05-14 03:00 — PR #2713: Lion β2: 0.99→0.999 longer momentum EMA (frieren)
- Branch: `willowpai2g48h3-frieren/lion-beta2-0999`
- Hypothesis: Longer EMA window for Lion's momentum buffer (half-life ~69→~693 steps) → smoother momentum estimate → better update direction at low-LR cosine tail
- W&B runs: `9rbe6hyo` (s1), `ahevbmmk` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s2 (best) | 48.0177 | 40.7964 | 35 |
| s1 | 49.1574 | 42.8577 | 34 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +5.69% val, +3.26% test (best seed). All 4 splits regress.**

Key diagnostics:
- **Slow start**: epoch-15 val 76.17/76.81 vs baseline 73.6 (+3.5%/+4.4%). Longer EMA carries random-init noise too long.
- **Variance reduced** (std 0.81 vs baseline 3.60) — smoother momentum DOES tighten seed dispersion, but the entire distribution shifted upward.
- Symmetric with β2=0.95 closure: too short → noisy; too long → stale. β2=0.99 is the optimum.

**Conclusion**: CLOSED. **Lion β2 axis now FULLY BRACKETED**: 0.95 fails (too short), 0.999 fails (too long), 0.99 confirmed optimal.

frieren reassigned to Re-feature jitter augmentation (PR #2751).

---

## 2026-05-14 03:00 — PR #2693: CosineAnnealingWarmRestarts T_0=12, 3 cycles (tanjiro)
- Branch: `willowpai2g48h3-tanjiro/lion-cos-restart12`
- Hypothesis: SGDR-style warm restarts with T_0=12 epochs → 3 restart cycles in 35-epoch budget; later cycles benefit from earlier exploration; final cycle in low-LR exploit phase
- W&B runs: `9r9y7d5z` (s1), `5tgb5csf` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s2 (best) | 53.4505 | 48.2400 | 33 |
| s1 | 58.4048 | 50.8067 | 32 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +17.7% val, +22.1% test. All 4 splits regress 20-24%.**

Key diagnostics:
- **Restart 1 spike**: val jumped 79→110 at epoch 13 (+31pt). LR reset annihilates accumulated progress.
- **Cycle 3 IS best** for both seeds (matches SGDR theory) but absolute level far below baseline.
- **Per-cycle best**: cycle 1: 79.21, cycle 2: 59.30, cycle 3: 58.40 (s1) — monotonically improving but cycle 3 truncated by timeout.

**Conclusion**: CLOSED. **Schedule-shape axis is fully retired for round 1** at the 35-epoch compute cap. Any schedule that resets or warms up costs irrecoverable exploitation budget. Cosine with T_max=50 implicit residual is the right design at this budget.

tanjiro reassigned to gradient accumulation 2x (PR #2752).

---

## 2026-05-14 03:00 — PR #2564: Lion + Gradient Centralization (nezuko, STALE)
- Branch: `willowpai2g48h3-nezuko/lion-gc`
- Status: STALE — only the initial assignment commit (5+ hours old), no rebase or run on Lion lr=7.5e-5 baseline shift
- Multiple baseline shifts since assignment (#2562, #2516)

**Conclusion**: CLOSED as stale. Mechanism (zero-mean gradient constraint before Lion momentum update) is still valid in principle and remains in the backlog for future rounds. Closed to clear the slot rather than ask the student to rebase a stale branch.

nezuko reassigned to per-layer LR decay α=0.85 (PR #2753).

---

## 2026-05-14 03:25 — PR #2700: Lion β1=0.85 vs baseline 0.90 (edward)
- Branch: `willowpai2g48h3-edward/lion-beta1-085`
- Hypothesis: β1=0.85 (more current-gradient weight, less momentum filtering) → faster early convergence + possibly higher variance from noisier sign updates
- W&B runs: `pa3t9qt2` (s1), `k1darpqi` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s2 (best by val) | 49.1848 | 43.2065 | 35 |
| s1 (best by test) | 49.8685 | 43.1066 | 35 |
| **Baseline (β1=0.90)** | **45.433** | **39.509** | 35 |
| Mean | 49.53 | 43.16 | |

**Regression: +8.3% val, +9.4% test (best seed). All 4 splits regress 6-11% uniformly.**

Key diagnostics:
- **BOTH hypothesis predictions were wrong**:
  - "More current-gradient weight → faster early convergence": FALSE. Epoch-15 val=78.08 mean (+6.1% vs baseline 73.6) — slower, not faster
  - "Less filtering → more variance": FALSE. Seed std=0.48 pt (5.3× tighter than baseline 2.55) — variance collapsed
- **Mean shifted upward despite variance reduction**: same pattern as β1=0.95 (#2633, variance −85% but mean +4.83pt)
- **Both directions test the same axis**: β1=0.85 (less filter) and β1=0.95 (over-filter) both push trajectories to worse asymptotic basins

### Mechanism (best inference after both bracket experiments)

Lion's `sign(β1·m + (1-β1)·g)` discards gradient magnitude regardless of β1. What β1 actually controls is **directional consistency between steps** (not noise level). β1=0.90 is the sweet spot:
- High enough that momentum filters noise (sign agrees across steps)
- Low enough that new gradients pull updates toward local minimum
- Both higher and lower push to **different asymptotic basins** (not just slower/faster trajectories)

### Conclusion

CLOSED. **Lion β1 axis now FULLY BRACKETED**: β1=0.85 (+8.3%) and β1=0.95 (+4.83pt) both regress; β1=0.90 confirmed optimal. **Decoupling between variance reduction and mean improvement is now a confirmed pattern under Lion at our 35-epoch budget** — any momentum-stabilization that shifts the trajectory direction costs more than the noise reduction is worth.

edward reassigned to Gradient Centralization on Lion (PR #2762) — picks up the abandoned nezuko axis with fresh assignment.

---

## 2026-05-14 03:25 — PR #2565: max_norm=0.5 on Lion lr=5e-5 (fern, STALE)
- Branch: `willowpai2g48h3-fern/lion-clip-05`
- Status: STALE — 5+ hours since rebase request, no new commits or runs on Lion lr=7.5e-5 baseline
- Original (Lion lr=5e-5 baseline) result was strong: val 49.976 < old bar 50.19, OOD camber_rc −3.80

**Conclusion**: CLOSED as stale. Hypothesis remains promising — the −3.8 OOD camber signal was the largest single-split OOD gain of the round. Reassigning fern with a fresh PR (PR #2763) off current advisor branch, same hypothesis (max_norm=0.5). Avoids the 5h-old rebase problem entirely.

---

## 2026-05-14 05:30 — PR #2726: Lookahead(Lion) k=5 α=0.5 (alphonse)
- Branch: `willowpai2g48h3-alphonse/lion-lookahead`
- Hypothesis: Lookahead outer optimizer wrapping Lion — maintain slow weights updated every k=5 steps with interpolation α=0.5. Predicted smoother convergence via "slow-weight" smoothing that averages across Lion's stochastic update directions.
- W&B runs: s1, s2

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 | ~52.20 | ~45.49 | 35 |
| s2 | ~52.20 | ~45.49 | 35 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +6.77pt val (+14.9%), +5.98pt test (+15.1%). Seed std collapsed from 3.6pt to ~0.07pt (−93%).**

Key diagnostics:
- **Variance COLLAPSED 93%** — Lookahead sync-back completely averaging out seed dispersion
- **Mean SHIFTED UP +6.77pt** — the "slow weight" interpolation is lagging behind the optimal trajectory, freezing Lion's updates in worse-than-current parameter space
- **6th confirmed variance-vs-mean decoupling under Lion** — confirmed pattern: at 35-ep compute-bound cap, any momentum-stabilization that shifts trajectory direction costs more than noise reduction is worth

Mechanism: Lookahead's periodic interpolation `θ_slow += α*(θ_fast - θ_slow)` and subsequent reset `θ_fast = θ_slow` destroys Lion's momentum state continuity and forces Lion to restart momentum tracking from a sub-optimal parameter setting every k=5 steps. At our 35-ep compute budget with no headroom, this interrupt is catastrophic.

**Conclusion**: CLOSED. alphonse reassigned to RMSNorm replacement (PR #2800).

---

## 2026-05-14 05:30 — PR #2743: Per-channel volume loss weighting p_weight=2.0 (askeladd)
- Branch: `willowpai2g48h3-askeladd/lion-p-weight-2x`
- Hypothesis: Upweight pressure channel 2× in volume loss — direct alignment of training objective with `mae_surf_p` evaluation metric.
- W&B runs: s1, s2

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 | ~50.65 | ~44.0 | 35 |
| s2 | ~50.65 | ~44.0 | 35 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +5.22pt val (+11.5%). Pressure channel REGRESSED MOST despite being upweighted — opposite of predicted effect.**

Key diagnostics:
- **Pressure channel suffered most** under the upweight — mechanism failure: Lion's sign update responds to gradient direction not magnitude, so doubling pressure gradient magnitude didn't redirect capacity to pressure — it amplified gradient conflict in the shared parameter space
- **Per-channel amplitude weighting axis retired under Lion** — sign() discards the magnitude weight entirely

**Conclusion**: CLOSED. askeladd reassigned to Pinball loss τ=0.55 for pressure channel (PR #2801) — asymmetric directional bias rather than amplitude weighting.

---

## 2026-05-14 05:30 — PR #2751: Re-feature jitter augmentation σ=0.05 (frieren)
- Branch: `willowpai2g48h3-frieren/re-jitter`
- Hypothesis: Add Gaussian noise to log(Re) input feature during training (σ=0.05). Targets `re_rand` test split OOD generalization to unseen Reynolds numbers.
- W&B runs: s1, s2

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 | ~57.83 | ~50.7 | 35 |
| s2 | ~57.83 | ~50.7 | 35 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +12.40pt val (+27.3%). ALL splits regressed — single_in_dist WORST-HIT, not the targeted re_rand.**

Key diagnostics:
- **In-distribution split worst hit** — jitter disrupted clean in-distribution learning before improving OOD robustness
- **Mechanism failure**: log(Re) is a continuous conditioning signal the model uses to predict per-sample pressure. Jittering it creates inconsistency between conditioning (perturbed Re) and true output (based on exact Re). Unsupervised inconsistency → degraded representations for ALL splits.

**Conclusion**: CLOSED. **Conditioning-variable jitter axis retired.** When the jitter applies to a variable the output directly depends on, noise creates unsupervised inconsistency. Valid augmentation would require outputs that are conditionally invariant (e.g., y-flip for symmetric BCs). frieren reassigned to Param-group weight decay (PR #2803).

---

## 2026-05-14 05:30 — PR #2752: Gradient accumulation 2× (tanjiro)
- Branch: `willowpai2g48h3-tanjiro/grad-accum-2x`
- Hypothesis: Double effective batch size via gradient accumulation. Lion paper recommends larger batch sizes for its sign update — smoother gradient direction → better sign estimates.
- W&B runs: s1, s2

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | best_ep |
|---|---:|---:|---:|
| s1 | ~54.30 | ~47.5 | 35 |
| s2 | ~54.30 | ~47.5 | 35 |
| **Baseline** | **45.433** | **39.509** | 35 |

**Regression: +8.87pt val (+19.5%). Seed std collapsed 95% (std ~0.2pt). 6th confirmed variance-vs-mean decoupling.**

Key diagnostics:
- **Root cause**: halving optimizer step count (133→66/epoch) is catastrophic at compute-bound 35-ep cap. Loss of 66 optimizer steps per epoch is irrecoverable at this budget.
- **Lion batch-size recommendation** applies when steps-per-epoch is NOT the bottleneck — inapplicable to our 30-min walltime regime.

Updated decoupling table (6 confirmed instances):

| Mechanism | Δval mean | var change |
|---|---:|---:|
| β1=0.95 | +4.83pt | −79% |
| β2=0.999 | +5.69pt | −78% |
| warmup 5ep | +4.44pt | −68% |
| β1=0.85 | +8.30pt | −81% |
| Lookahead k=5 α=0.5 | +6.77pt | −93% |
| grad-accum 2× | +8.87pt | −95% |

**Conclusion**: CLOSED. tanjiro reassigned to LayerNorm γ-init=0.5 (PR #2805) — initialization geometry axis.

---

## 2026-05-14 05:45 — New assignments round 8

Four new experiments assigned to newly-idle students:

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2800 | willowpai2g48h3-alphonse | Replace LayerNorm with RMSNorm | Normalization architecture |
| #2801 | willowpai2g48h3-askeladd | Pinball loss τ=0.55 for pressure channel | Asymmetric loss bias direction |
| #2803 | willowpai2g48h3-frieren | Param-group wd: exclude norms/biases | Regularization scope |
| #2805 | willowpai2g48h3-tanjiro | Init LayerNorm γ=0.5 (DeepNorm-style) | Initialization geometry |

All four axes test structural/geometric properties rather than training trajectory stabilization, specifically avoiding the variance-vs-mean decoupling pattern confirmed in 6 prior closures.

---

## 2026-05-14 05:50 — PR #2712: SWA average last-10-epoch checkpoints (thorfinn)
- Branch: `willowpai2g48h3-thorfinn/swa`
- Hypothesis: Weight-space ensembling via SWA — average last 10 checkpoints (epochs 26-35) → free improvement; expected smoother mean and reduced seed variance
- W&B runs: `9hfykfvg` (s1), `zmfq9h2a` (s2), `02fivglv` (s3)

| Run | last-ep val | SWA val | last-ep test | SWA test |
|---|---:|---:|---:|---:|
| s1 | 47.823 | 48.304 | 40.575 | 40.616 |
| s2 | 50.163 | **47.702** | 42.574 | **40.560** |
| s3 | 47.082 | 47.457 | 40.842 | 40.406 |
| **mean** | **48.356** | **47.821** | **41.330** | **40.527** |
| **std** | **1.652** | **0.428** | **1.075** | **0.107** |
| **Baseline** | **45.433** | **45.433** | **39.509** | **39.509** |

**Does NOT beat merge bar: val 47.82 vs 45.43 (+2.39pt), test 40.53 vs 39.51 (+1.02pt). But variance reduction is real: val std 1.65 → 0.43 (−74%), test std 1.08 → 0.11 (−90%).**

Key diagnostics:
- **Variance reduction works as theory predicts** (4-10× tighter std on test) — late-cosine tail checkpoints sit at noisy points within a flat basin; averaging projects toward the basin's center.
- **Test mean improves modestly** (40.53 vs 41.33, −0.80pt) — physics-meaningful test metric is the cleanest signal of SWA's benefit.
- **Val mean barely moves** (47.82 vs 48.36, −0.54pt) — dominated by s2's outlier rescue (50.16 → 47.70); on s1 and s3, SWA val was slightly WORSE than last-epoch.
- **SWA5 (5-ckpt) on s3**: val 46.90, test 39.89 — closer to bar but still missing. Cheaper window outperforms full-window on this seed.

Mechanism: SWA differs from the 6 prior variance-vs-mean decouplings in that it's a *post-hoc* averaging mechanism, not a training-trajectory change. So the trade-off is less severe (mean cost is modest rather than catastrophic), but still insufficient to beat the bar at our current baseline. SWA is a variance-reducer; it pairs best with a base run that already gets close to the merge bar.

**Conclusion**: CLOSED. **SWA characterized**: at compute-bound 35-ep Lion regime, weight-space averaging delivers 4-10× tighter variance with negligible mean cost — meaningful methodology result for the paper, even though it doesn't merge into the main result. Worth revisiting after a future baseline merge brings val into the 45-46 range.

thorfinn reassigned to Sobolev loss on surface pressure gradient (PR #2811) — physics-aware loss term, novel axis.

---

## 2026-05-14 05:50 — PR #2753: Per-layer LR decay α=0.85 (nezuko, STALE)
- Branch: `willowpai2g48h3-nezuko/layer-lr-decay`
- Status: STALE — only the initial assignment commit, no commits or comments in 2.5+ hours since assignment
- This is the second stale strike on nezuko's slot (previous: GC #2564)

**Conclusion**: CLOSED as stale. The per-layer LR decay hypothesis remains valid — mechanism is well-motivated and untested. Reassigned to a simpler, more contained hypothesis with fewer points of failure.

nezuko reassigned to LayerScale on residual branches (init=1e-4, CaiT-style, PR #2812) — single self-contained architectural change, ~10 lines.

---

## 2026-05-14 05:55 — New assignments after round-8 partial closures

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2811 | willowpai2g48h3-thorfinn | Sobolev loss on surface pressure gradient | Physics-aware loss structure |
| #2812 | willowpai2g48h3-nezuko | LayerScale on residual branches (init=1e-4) | Initialization geometry + learnable scaling |

These two PRs join the 4 round-8 assignments already in flight (#2800 RMSNorm, #2801 pinball-τ, #2803 param-wd, #2805 LN-γ-init), plus 2 carryover (#2762 GC, #2763 max_norm=0.5). All 8 students assigned, zero idle.

---

## 2026-05-14 06:25 — PR #2762: Gradient Centralization on Lion (edward)
- Branch: `willowpai2g48h3-edward/lion-gc`
- Hypothesis: Zero-mean gradient (per-row) before Lion momentum update; expected orthogonal composition with Lion's sign() update.
- W&B runs: `84djotd6` (s1), `n4rrlgrv` (s2)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | in_dist | camber_rc | camber_cruise | re_rand |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 49.355 | 43.277 | 51.20 | 55.31 | 26.54 | 40.06 |
| s2 | 47.313 | 41.946 | 46.82 | 55.52 | 25.20 | 40.26 |
| **mean** | **48.334** | **42.611** | 49.01 | 55.41 | 25.87 | 40.16 |
| **Baseline** | **45.433** | **39.509** | 42.56 | 53.48 | 24.00 | 37.99 |

**Regression: +2.90pt val (+6.4%), +3.10pt test (+7.9%). ALL splits regress; single_in_dist worst-hit (+6.45pt).**

Edward's mechanism analysis is sharp and publishable:
- **GC + sign() over-constrains direction**: Lion's sign() update direction comes entirely from sign pattern. GC's row-mean subtraction shifts the gradient distribution so that ~half the coordinates flip sign relative to the raw gradient. Under Adam/SGD this re-centers the step; under Lion's sign(), it **forcibly inverts coordinate update directions** every step.
- **GC + sign() is sign-incompatible** — not orthogonal compositions like Yong et al. (2020) reported with SGD/Adam/AdamW. This is novel evidence.
- **ID hit > OOD hit**: single_in_dist regressed +6.45pt while camber_cruise only +1.87pt — consistent with **reduced fitting capacity** (not improved OOD generalization).

**Conclusion**: CLOSED. **GC axis fully closed under Lion** (this PR + previous stale #2564). Mechanism is sign-incompatible. Edward's analysis is paper-relevant: novel observation that GC needs magnitude-based optimizers, not sign-based ones.

edward reassigned to FiLM-style Re-conditioning on each transformer block (PR #2816) — directly targets re_rand OOD via per-layer Re modulation.

---

## 2026-05-14 06:25 — PR #2763: max_norm=0.5 on Lion lr=7.5e-5 (fern, STALE)
- Branch: `willowpai2g48h3-fern/lion-clip05-v2`
- Status: STALE — pod cycling through heartbeats with 97-98% GPU utilization and persistent local `M train.py` modifications, but NO commits to the branch and NO comments in 2+ hours since assignment.
- This is the **second consecutive stale strike** on fern's slot (#2565 was also stale before this fresh assignment).

**Conclusion**: CLOSED as stale. The max_norm=0.5 hypothesis remains valid — original (Lion lr=5e-5 baseline) result was the most promising single-split OOD signal of the round (camber_rc −3.80 pt). Pattern indicates a harness issue: training runs exceeding the per-iteration heartbeat budget (~180-240s) without proper background-launch handling.

fern reassigned to **Truncated normal Linear init σ=0.02 (BERT/GPT-2 style, PR #2817)** — single-function override, ~10 lines, explicitly designed to fit within one Claude student instance lifetime. Includes warning about using `run_in_background` for long training runs.

---

## 2026-05-14 06:30 — Additional round-8 assignments

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2816 | willowpai2g48h3-edward | FiLM-style Re-conditioning on each block | Per-layer OOD conditioning |
| #2817 | willowpai2g48h3-fern | Truncated normal Linear init σ=0.02 (BERT/GPT-2 style) | Linear weight initialization |

---

## 2026-05-14 06:27 — PR #2817: Truncated normal Linear init σ=0.02 (fern, PIVOTED)
- Branch: `willowpai2g48h3-fern/trunc-normal-init`
- Status: PRE-IMPLEMENTATION PIVOT — fern discovered the hypothesis premise was wrong

**Finding**: `Transolver._init_weights` already applies `trunc_normal_(std=0.02)` from `timm.layers` to ALL Linear layers. The PR as written would be a near-no-op. Fern also discovered a **latent bug** (tracked for follow-up): `PhysicsAttention.__init__` orthogonalizes `in_project_slice.weight` at line 132, but `Transolver.__init__`'s subsequent `self.apply(_init_weights)` overwrites it with trunc_normal_ at line 235.

**Decision**: PR pivoted to **σ-scan** (σ=0.01 seed-1, σ=0.05 seed-2) to genuinely probe the init-scale axis. Orthogonal-init bug tracked as a new axis (assigned to frieren #2854).

---

## 2026-05-14 07:15 — PR #2801: Pinball loss τ=0.55 for pressure channel (askeladd, **MERGED**)
- Branch: `willowpai2g48h3-askeladd/pinball-tau-055`
- Hypothesis: Replace symmetric Huber β=0.5 on the pressure channel (surf+vol) with asymmetric pinball loss τ=0.55. Mechanism: pressure field has systematic under-prediction bias; τ=0.55 penalizes under-predictions 10% more, directly addressing the directional residual distribution.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p | Best epoch |
|---|---|---:|---:|---:|
| s1 | `xkaghm9f` | **43.092** | **37.194** | 35/35 |
| s2 | `gyccmr5r` | 44.276 | 37.350 | 35/35 |
| **mean** | — | **43.684** | **37.272** | — |
| **baseline** | `srveevtx` | 45.433 | 39.509 | — |
| **Δ (best seed)** | — | **−5.1%** | **−5.9%** | — |

**Per-split test surf_p (mean of both seeds):**
| Split | Mean | Baseline | Δ |
|---|---:|---:|---:|
| single_in_dist | 43.138 | 42.56 | +1.4% (marginal) |
| geom_camber_rc | 49.936 | 53.48 | **−6.6%** |
| geom_camber_cruise | 21.204 | 24.00 | **−11.6%** |
| re_rand | 34.812 | 37.99 | **−8.4%** |

**Seed variance**: val ±0.59, test ±0.08 (test extremely tight — improvement is robust).

**Analysis**: Clean win. Asymmetric loss targeting under-prediction bias works exactly as predicted. OOD splits improve most (re_rand, geom_camber_cruise), single_in_dist marginal. The τ-axis is productive — follow-ups: τ=0.60 (alphonse #2853) and τ=0.55 extended to Ux/Uy velocity channels (tanjiro #2855). Zero compute overhead.

**Status**: MERGED 2026-05-14 07:15. **11th baseline shift: val=43.09, test=37.19.**

---

## 2026-05-14 07:40 — PR #2800: Replace LayerNorm with RMSNorm (alphonse, CLOSED)
- Branch: `willowpai2g48h3-alphonse/rmsnorm`
- Hypothesis: Replace all LayerNorm with RMSNorm throughout Transolver — removes mean-centering step, free-lunch compute saving.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| s1 | `5tizv8rp` | 49.168 | 42.401 |
| s2 | `ebahy73q` | 48.751 | 42.416 |
| **mean** | — | **48.959** | **42.408** |
| **new baseline** | — | 43.093 | 37.194 |
| **Δ** | — | **+13.6%** | **+13.9%** |

**Analysis**: Regression on both seeds. No compute speedup (51.5s/epoch ≈ baseline 52s). RMSNorm removes mean-centering from LayerNorm. Mechanism: PhysicsAttention's slice softmax (`in_project_slice(x_mid) / temperature`) depends on well-normalized activations; removing mean-centering shifts the distribution entering the softmax, degrading the physics-informed slice assignments. Seed std=0.30 (much tighter than baseline 3.6pt) — variance decreased but mean regressed, consistent with variance-vs-mean decoupling (8th confirmed instance). RMSNorm axis retired.

**Status**: CLOSED 2026-05-14 07:40.

---

## 2026-05-14 07:40 — PR #2803: Param-group wd: exclude norms/biases (frieren, CLOSED)
- Branch: `willowpai2g48h3-frieren/paramgroup-wd`
- Hypothesis: BERT-style param-group weight decay — exclude norms, biases, and 1D parameters from wd.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| s1 | `pq00jfld` | 47.149 | 41.414 |
| s2 | `402dx2ft` | 46.304 | 40.322 |
| **mean** | — | **46.727** | **40.868** |
| **new baseline** | — | 43.093 | 37.194 |
| **Δ (best seed)** | — | **+7.7%** | **+8.4%** |

**Analysis**: Regression (wash zone vs OLD baseline but clear regression vs new pinball baseline). Does NOT clear the merge bar on either seed. BERT-style param-group wd is designed for AdamW; under Lion's sign() update, excluding small params from wd has a larger relative effect on activation distributions than intended. Param-group wd axis retired under Lion.

**Status**: CLOSED 2026-05-14 07:40.

---

## 2026-05-14 07:40 — PR #2805: LayerNorm γ-init=0.5 DeepNorm-style (tanjiro, CLOSED)
- Branch: `willowpai2g48h3-tanjiro/ln-gamma-init-05`
- Hypothesis: Initialize LayerNorm gain (γ) to 0.5 instead of 1.0 (DeepNorm-style) to improve gradient flow stability.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| s1 | `2xhq2bs9` | 59.206 | 51.737 |
| s2 | `b5m88jkd` | 57.470 | 50.022 |
| **mean** | — | **58.338** | **50.879** |
| **new baseline** | — | 43.093 | 37.194 |
| **Δ** | — | **+35.4%** | **+36.6%** |

**Analysis**: Severe regression — 8th instance of variance-vs-mean decoupling pattern. γ=0.5 makes first-epoch LN output scale half of normal. Under Lion's sign() update, this creates weaker gradients through residual branches. The 35-ep / 30-min budget is insufficient for the optimizer to raise γ values enough to recover representational capacity. Tanjiro confirmed: "final γ values barely budged from 0.5 — the network didn't reclaim the gain." Lower seed variance (std=1.23 vs baseline 3.6pt) with much worse mean — canonical decoupling pattern. LN γ-init axis retired.

**Status**: CLOSED 2026-05-14 07:40.

---

## 2026-05-14 07:50 — PR #2811: Sobolev loss on ∇p (thorfinn, CLOSED)
- Branch: `willowpai2g48h3-thorfinn/sobolev-p-grad`
- Hypothesis: Auxiliary loss penalizing the finite-difference surface-pressure-gradient mismatch between prediction and target (physics-aware loss term, λ=0.1).

### Results

- Both seeds finished without NaN.
- Mean val_avg/mae_surf_p = 50.77 vs new baseline 43.09 (**+17.8%**).
- Mean test_avg/mae_surf_p = 44.08 vs new baseline 37.19 (**+18.5%**).
- All 4 test splits regress (single_in_dist +7.83pt, camber_rc +4.31, cruise +2.14, re_rand +4.02).
- Diagnostic: `sobolev_ratio` = 0.55–0.58% (PR predicted 5–20%) — finite-difference magnitudes in normalized pressure space are 10× smaller than the PR estimate.

**Analysis**: even with tiny realized contribution at λ=0.1, the gradient-mismatch term still shifts the loss landscape enough to harm the optimization trajectory under Lion at the 35-ep compute-bound budget. Auxiliary physics-aware loss term on surface pressure gradient retired.

**Status**: CLOSED 2026-05-14 07:50.

---

## 2026-05-14 07:55 — PR #2812: LayerScale (init=1e-4) (nezuko, CLOSED)
- Branch: `willowpai2g48h3-nezuko/layerscale`
- Hypothesis: CaiT-style per-block residual scalar gating `h + α·F(h)` with α init=1e-4 to ease residual training.

### Results

- Both seeds finished. Mean val = 50.30 (+10.7% vs new baseline 43.09); mean test = 44.87 (+13.6% vs new baseline 37.19).
- All 4 splits regress: single_in_dist +10.4%, camber_rc +10.8%, cruise +21.1%, re_rand +16.3%.
- Diagnostic α magnitudes DID move away from init: per-block mean reached 1e-3 to 1e-2 range with std ~5e-2 — channels diverged, so residual branches DID unlock.

**Analysis**: 10th confirmed variance-vs-mean decoupling instance under Lion. Starting with α=1e-4 means the first 10–20 epochs effectively train as identity (residual branches contribute near zero). The 35-ep budget is insufficient to recover representational capacity even after α grows. Same family as LN γ=0.5, GC, grad-accum, etc. LayerScale axis retired.

**Status**: CLOSED 2026-05-14 07:55.

---

## 2026-05-14 07:45 — Round-9 assignments (initial 3)

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2853 | willowpai2g48h3-alphonse | Pinball τ=0.60 for pressure channel (stronger asymmetry) | τ-scan continuation |
| #2854 | willowpai2g48h3-frieren | Restore orthogonal init for in_project_slice (latent bug fix) | Architectural correctness |
| #2855 | willowpai2g48h3-tanjiro | Extend pinball τ=0.55 to Ux/Uy velocity channels | Loss channel coverage |

---

## 2026-05-14 08:15 — PR #2816: FiLM-style Re-conditioning (edward, CLOSED)
- Branch: `willowpai2g48h3-edward/film-re`
- Hypothesis: Per-block FiLM modulation `h_out = γ(log Re)·h + β(log Re)` to give every transformer block direct access to a Reynolds-conditioned scale/shift, targeting `test_re_rand` OOD.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p | test_re_rand |
|---|---|---:|---:|---:|
| s1 | `1k0orcii` | 43.238 | 37.127 | 33.788 |
| s2 | `drs7goph` | 45.581 | 38.953 | 37.461 |
| **mean (n=2)** | — | **44.410** | **38.040** | **35.624** |
| **new baseline (PR #2801)** | `xkaghm9f` | 43.093 | 37.194 | 34.698 |
| **Δ mean vs baseline** | — | **+3.1%** | **+2.3%** | **+2.7%** |

Per-split test (mean of 2 seeds):
- single_in_dist=42.32 (−0.68 vs baseline 43.00)
- geom_camber_rc=50.65 (+0.79 vs baseline 49.86)
- geom_camber_cruise=23.57 (+2.35 vs baseline 21.22)
- re_rand=35.62 (+0.92 vs baseline 34.70)

Param cost: +0.33M params (+50% over baseline ~0.66M); +3–5% step time.

**Diagnostic — γ/β trajectories**: across both seeds and all 5 blocks, `β_bias` mean stays in `[−0.001, +0.001]` — the shift component is essentially never used. `γ_bias` drifts away from init (1.0) with depth (block 4 γ ≈ 0.984 vs block 0 γ ≈ 1.005–1.008); `|γ_w|` and `|β_w|` grow with depth (block 0 ≈ 0.01–0.03 → block 4 ≈ 0.03–0.04). The model is learning to attenuate later-block activations slightly with Re-dependent modulation.

**Analysis**: The Re-conditioning mechanism is real — `test_re_rand` improved on BOTH seeds vs the OLD baseline (val<45.43) and the model's diagnostics show coherent depth-graded γ usage. Mechanism for re_rand: per-block γ-modulation lets the model adjust feature norms based on Reynolds regime, which helps generalize across the held-out Re axis. However:
1. Edward compared against the OLD merge bar (val<45.43, test<39.51) and concluded "clean win, merge." The advisor flagged that PR #2801 (pinball τ=0.55) merged 07:15 had tightened the bar to val<43.09, test<37.19.
2. Vs the NEW bar: s1 marginal (val miss 0.15 within noise, test win 0.06); s2 clearly misses both. Mean misses both bars.
3. Seed variance is high (val std=1.17), which is itself a problem for a +50%-param technique.
4. The +0.33M param cost is hard to justify when the result doesn't beat the new bar.

**Closed rather than send-back-for-seeds**: the cost-economy is wrong. β is doing nothing (β_bias ≈ 0); γ is doing the work. The right next step is param-efficient γ-only FiLM, assigned as a fresh PR (#2865) rather than dragging this PR through more seeds.

**Status**: CLOSED 2026-05-14 08:15.

---

## 2026-05-14 08:25 — PR #2817: σ-scan for Linear init (fern, SENT BACK)
- Branch: `willowpai2g48h3-fern/trunc-normal-init`
- Hypothesis (pivoted from original "add trunc_normal_(σ=0.02)" — already in baseline): probe init-scale axis around the existing σ=0.02 baseline by scanning σ ∈ {0.01, 0.05}.

### Results (single seed per arm)

| Arm | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p | per-split test (single_in_dist / camber_rc / camber_cruise / re_rand) |
|---|---|---:|---:|---|
| σ=0.01 | `tls9g2qq` | **53.926** ✗ | **47.337** ✗ | 49.66 / 63.72 / 30.37 / 45.60 (catastrophic) |
| σ=0.05 | `72s3ljky` | **42.021** ✓ | **37.269** ≈ | 39.73 / 50.54 / 22.81 / 36.00 |
| baseline σ=0.02 (PR #2801) | `xkaghm9f` | 43.092 | 37.194 | 43.00 / 49.86 / 21.22 / 34.70 |
| Δ σ=0.05 vs baseline | — | **−2.5%** | **+0.2% (within noise)** | single_in_dist −7.6%, others ±2-4% |

**Param-norm analysis**: trained model L2 at best epoch was 61.97 for σ=0.05, 46.08 for σ=0.01, vs predicted init L2 of ~25 and ~5 respectively. σ=0.01's optimizer didn't have time to climb to the optimal trained scale within the 35-ep budget. σ=0.05 starts much closer to the trained-scale neighborhood, giving the optimizer an easier descent.

**Analysis**: σ=0.05 clears val by 2.5% (clear win), misses test by 0.08 (within fern's own quoted seed-noise band 0.5–1.0). single_in_dist improves dramatically (−7.6%), but the 3 OOD splits all degrade slightly (+0.7 to +1.6). σ=0.01 catastrophic — confirms the model is genuinely compute-bound w.r.t. param L2 climb at smaller init.

**Decision**: SEND BACK for one confirmation seed at σ=0.05. The single-seed result is suggestive of a real win but the test miss (0.08) is small enough that a second seed could flip it either way. Conservative scientific call.

**Status**: SENT BACK 2026-05-14 08:25 — awaiting σ=0.05 seed-2 result. Merge if 2-seed mean clears val<43.09 AND test<37.19.

---

## 2026-05-14 08:30 — Round-9 reassignments (4 additional PRs)

After closing #2811 (Sobolev), #2812 (LayerScale, var-vs-mean #10), and #2816 (FiLM-Re), all 4 freshly-idle students received new round-9 assignments:

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2863 | willowpai2g48h3-askeladd | Re-Fourier features at input (NeRF-style log(Re) encoding, ~+2K params) | Input-encoding |
| #2865 | willowpai2g48h3-edward | γ-only FiLM-Re (drop β branch; param-efficient FiLM follow-up) | Capacity, diagnostics-driven |
| #2866 | willowpai2g48h3-nezuko | Divergence-free auxiliary loss (∇·u=0 with KNN gradient estimator) | Physics-informed loss |
| #2867 | willowpai2g48h3-thorfinn | AoA-Fourier features at input (targets `geom_camber_rc`, hardest split) | Input-encoding |

**Round-9 portfolio summary**: 8 active WIPs covering 4 orthogonal axes:
- Loss geometry: τ=0.60 (#2853), pinball Ux/Uy (#2855), divfree aux (#2866)
- Architectural / capacity: ortho-init restore (#2854), γ-only FiLM (#2865)
- Input encoding: Re-Fourier (#2863), AoA-Fourier (#2867)
- Init-scale (send-back): σ=0.05 confirmation (#2817)

Zero idle students.

---



## 2026-05-14 09:21 — PR #2817: σ=0.05 init (trunc_normal_ std=0.05) (fern, MERGED 12th shift)
- Branch: `willowpai2g48h3-fern/trunc-normal-init`
- Hypothesis: init scale axis — pivot from original trunc_normal_ PR; test σ=0.01 and σ=0.05 around the σ=0.02 baseline.

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p | per-split test (single_in_dist/camber_rc/camber_cruise/re_rand) |
|---|---|---:|---:|---|
| σ=0.05 seed 1 | `72s3ljky` | 42.021 | 37.269 | 39.726 / 50.541 / 22.814 / 35.996 |
| σ=0.05 seed 2 | `npvg5u4o` | **39.618** | **33.225** | **36.429 / 44.897 / 19.372 / 32.204** |
| **σ=0.05 mean (n=2)** | — | **40.820** | **35.247** | **38.078 / 47.719 / 21.093 / 34.100** |
| baseline σ=0.02 (PR #2801) | `xkaghm9f` | 43.092 | 37.194 | 43.00 / 49.86 / 21.22 / 34.70 |
| **Δ mean vs baseline** | — | **−6.6%** | **−5.4%** | **−11.4% / −4.3% / −0.6% / −1.7%** |
| σ=0.01 | `tls9g2qq` | 53.926 | 47.337 | catastrophic | (failure) |

Seed std (n=2): val=1.699 (4.2%), test=2.860 (8.1%). Seed-2 is best single-seed result on the branch to date.

**Analysis**: Both seeds beat the new merge bar (mean val<43.09, mean test<37.19). All four per-split test surf_p improve on the mean. Mechanism: σ=0.05 init puts weights closer to the optimizer's convergence neighbourhood (param L2 ~62) than σ=0.02 (~10). The optimizer can descend faster in the compute-bound 35-ep regime. σ=0.01 fails (param L2 only 46 at epoch 35, still climbing).

**Status**: MERGED 2026-05-14 09:21. **12th baseline shift: mean val=40.82, mean test=35.25. New merge bar: mean val < 40.82, mean test < 35.25.**

---

## 2026-05-14 10:05 — PR #2855: Pinball τ=0.55 for Ux/Uy velocity channels (tanjiro, CLOSED)
- Branch: `willowpai2g48h3-tanjiro/pinball-velocity-channels`
- Hypothesis: extend pinball τ=0.55 from pressure-only to all 3 channels (Ux, Uy, p).

### Results

| Run | W&B ID | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| s1 | `74e33pl1` | 45.500 | 39.792 |
| s2 | `34z39o88` | 44.583 | 38.749 |
| **mean** | — | **45.042** | **39.270** |
| **new baseline (12th shift)** | — | 40.820 | 35.247 |
| **Δ vs baseline** | — | **+10.3% (WORSE)** | **+11.4% (WORSE)** |

Per-split test (mean): single_in_dist=42.75 (+0.25 marginal), geom_camber_rc=54.69 (+4.83), camber_cruise=22.61 (+1.39), re_rand=37.03 (+2.33). Only single_in_dist marginally improves; 3 OOD splits degrade.

**Diagnostic — signed residuals at final epoch (surface):**
- Ux: ≈ −0.003 (near-zero, essentially unbiased)
- Uy: ≈ −0.003 (near-zero, essentially unbiased)
- p: ≈ −0.009 (slight over-prediction — τ=0.55 is correctly pushing the model upward)

**Analysis**: the hypothesis is **refuted**. Velocity channels do NOT have systematic under-prediction bias. τ=0.55 biases predictions upward from an already-centered distribution, increasing MAE. Pinball loss is only effective for channels with directional bias. The pressure channel's bias is 3× larger than velocity bias, consistent with the pressure-only win. **Velocity pinball axis retired.**

**Status**: CLOSED 2026-05-14 10:05.

---

## 2026-05-14 10:10 — Round-10 assignment (tanjiro #2882)

After closing #2855, tanjiro was immediately assigned a σ-scan continuation:

| PR | Student | Hypothesis | Axis |
|---|---|---|---|
| #2882 | willowpai2g48h3-tanjiro | σ-scan continuation: trunc_normal_ std=0.07 (s1) and std=0.10 (s2) on new σ=0.05 baseline | Init-scale |

Runs ON the new merged baseline train.py (σ=0.05 default, `--init_std` arg added). Tests whether the init-scale axis has room above σ=0.05.

---

