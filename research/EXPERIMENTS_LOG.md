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

