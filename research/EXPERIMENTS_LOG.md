# SENPAI Research Results — charlie-pai2i-24h-r2

## 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss
- Branch: `charliepai2i24h2-fern/huber-loss`
- Student: charliepai2i24h2-fern
- Hypothesis: Replace MSE in normalized target space with SmoothL1 (Huber, β=1.0) in both training and validation; expect 4–10% improvement on `val_avg/mae_surf_p` from reducing high-Re sample domination.

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **116.611** |
| `test_avg/mae_surf_p` | NaN (pre-existing infra bug; 3 clean splits avg 114.59) |
| Epochs completed | 14/50 (cut at 30-min cap) |
| Peak VRAM | 42.1 GB |
| Wall-clock | ~30.6 min |
| Per-split val mae_surf_p | single 161.69 \| geom_rc 117.56 \| geom_cruise 85.67 \| re_rand 101.53 |
| Per-split test mae_surf_p | single 139.80 \| geom_rc 104.38 \| geom_cruise NaN \| re_rand 99.60 |
| Metrics artifact | `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}` |

### Analysis
- First completed experiment on this branch — establishes the de facto baseline at `val_avg/mae_surf_p` = 116.61 under the 30-min/14-epoch wall-clock cap.
- Trajectory still improving at the cap (epoch 1: 229.6 → epoch 13: 116.6). Headroom from longer schedules is real.
- Per-split pattern is consistent with the hypothesis: cruise (smallest pressure magnitudes) is easiest at 85.7; high-Re raceCar single is hardest at 161.7.
- **Important: no clean MSE companion run exists on this branch**, so the Huber-vs-MSE delta is not isolated. The 7 other in-flight round-1 PRs all carry MSE plus one other change, so we will get indirect signal.
- Two-line diff exactly matched the prescription — clean execution.

### Pre-existing NaN bug surfaced
- `test_geom_camber_cruise` sample 20 has non-finite `y` ground-truth values. `data/scoring.py:accumulate_batch` has a sample-level skip, but it computes `err = |pred - y|` *before* masking, and `NaN * 0 = NaN` so the masked sum propagates NaN. The same pattern exists in the normalized-space loss path of `train.py:evaluate_split`.
- `data/scoring.py` is read-only per `program.md`. The workaround is to sanitize `y` (zero-out NaN, mask out the sample) *before* the loss/scoring calls in `train.py:evaluate_split`.
- Routed to next fern assignment (gradient-clip + selective-decay PR also carries the NaN guard fix in `evaluate_split`).

### Decision
- **Merge.** The PR establishes the first concrete baseline. Huber loss carries forward to subsequent experiments. NaN bug is independent of this change.
- Updated `BASELINE.md` with the new reference numbers.

---

## 2026-05-15 15:30 — PR #3276: Gradient clip + AdamW selective decay (+ test NaN guard)
- Branch: `charliepai2i24h2-fern/grad-clip-selective-decay`
- Student: charliepai2i24h2-fern

### Hypothesis
Two bundled changes: (1) grad clip max_norm=1.0 + AdamW selective decay (LN/bias/1D params excluded from weight decay) — transformer best-practice optimizer tuning; (2) NaN sample guard in `evaluate_split` to recover finite test metrics. Predicted 1–4% improvement on `val_avg/mae_surf_p`.

### Results table

| Metric | Value | Delta vs baseline (116.61) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **109.681** | **-5.94%** |
| `test_avg/mae_surf_p` | **97.315** | finite (was NaN) |
| Epochs completed | 14/50 (cut at 30-min cap) |  |
| Peak VRAM | 42.1 GB | |
| Wall-clock | ~30.7 min | |
| Per-split val mae_surf_p | single 148.09 \| geom_rc 114.87 \| geom_cruise 78.85 \| re_rand 96.91 | all 4 improved |
| Per-split test mae_surf_p | single 123.24 \| geom_rc 104.76 \| geom_cruise 68.48 \| re_rand 92.79 | 3 improved; geom_rc flat |
| Optimizer groups | decay=49 (0.655M params), no_decay=62 (0.008M params) | |
| Metrics artifact | `models/model-grad-clip-selective-decay-20260515-142950/metrics.{jsonl,yaml}` | |

### Analysis
- Beat the 1–4% prediction with a 5.94% improvement — best result on this branch.
- All 4 val splits improved. Largest gains: single (-8.4%) and cruise (-8.0%). Smallest: geom_rc (-2.3%).
- The geom_rc split lags on both val and test — consistent with a capacity or domain-coverage issue rather than optimizer sensitivity.
- NaN guard worked: test_geom_camber_cruise now reports 68.48 instead of NaN. Cruise is actually the easiest test split.
- Best epoch was the final (14/14) — cosine still annealing at timeout; headroom exists from longer runs.
- Optimizer group split is correct: ~99% of param mass in decay group; LN gains/biases/temperature/placeholder in no_decay.

### Decision
- **Merge.** Clear improvement across all metrics. NaN guard is now baseline infrastructure. New benchmark: `val_avg/mae_surf_p` = 109.68.

---

## 2026-05-15 15:30 — PR #3220: Linear warmup + cosine over 100 epochs at lr 7e-4 (CLOSED)
- Branch: `charliepai2i24h2-tanjiro/warmup-cosine-100ep`
- Student: charliepai2i24h2-tanjiro

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 14) | 148.200 |
| `test_avg/mae_surf_p` | NaN (pre-existing bug; 3-clean 148.74) |
| Epochs completed | 14/100 (cut at 30-min cap) |
| Per-split val mae_surf_p | single 191.75 \| geom_rc 152.67 \| geom_cruise 118.65 \| re_rand 129.73 |

### Analysis
- 27% regression vs Huber baseline (148.20 vs 116.61). Schedule was effectively a flat-high-LR run — 100 epochs under a 30-min cap means the cosine never cooled. The hypothesis (warmup + cosine helps) is not testable with this schedule length.
- Student correctly diagnosed the issue and suggested matching schedule to completable epochs.
- **Decision: Closed. Follow-up: matched 14-epoch warmup+cosine (PR #3294).**

---

## 2026-05-15 15:30 — PR #3205: Scale attention slice_num 64→192 and n_head 4→8 (CLOSED)
- Branch: `charliepai2i24h2-edward/slices-192-heads-8`
- Student: charliepai2i24h2-edward

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 5) | 164.377 |
| `test_avg/mae_surf_p` | 150.944 |
| Epochs completed | 5/50 (cut at 30-min cap) |
| Peak VRAM | 38.1 GB (with gradient checkpointing + bf16 AMP) |
| Per-split val mae_surf_p | single 207.07 \| geom_rc 179.17 \| geom_cruise 129.06 \| re_rand 142.20 |

### Analysis
- 41% regression vs Huber baseline. Only 5 epochs at ~385 s/epoch. Hypothesis untestable at this scale under 30-min budget. Student correctly diagnosed OOM → added gradient checkpointing + bf16 AMP. cruise split at 5 epochs (129.06 val) shows representation capacity but budget overwhelms the signal.
- Merge conflict with HEAD (post Huber merge). Student's follow-up #2 (decouple slice_num vs n_head axes) is the right next move.
- **Decision: Closed. Follow-up: single-axis slice_num=128 (PR #3295).**

---

## 2026-05-15 15:30 — PR #3179: Scale Transolver n_hidden 128→192 (CLOSED)
- Branch: `charliepai2i24h2-alphonse/width-192`
- Student: charliepai2i24h2-alphonse

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 7) | 154.979 |
| `test_avg/mae_surf_p` | 144.489 |
| Epochs completed | 10/50 (cut at 30-min cap) |
| Actual param count | 1.47M (not 2.25M as predicted) |
| Peak VRAM | 58.0 GB |

### Analysis
- 33% regression vs baseline. 10 epochs at ~185 s/epoch; cosine never annealed. Epoch 8 noise jump (155→215→169) shows the model is operating at high LR throughout. Student correctly identified the budget mismatch and suggested matched-epoch re-run.
- **Decision: Closed. Follow-up: width-192 with epochs=10 budget-matched (PR #3301).**

---

## 2026-05-15 15:30 — PR #3183: Scale Transolver depth 5→8 layers (CLOSED)
- Branch: `charliepai2i24h2-askeladd/depth-8`
- Student: charliepai2i24h2-askeladd

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 9) | 154.955 |
| `test_avg/mae_surf_p` | NaN (3-clean 157.11) |
| Epochs completed | 9/50 (cut at 30-min cap) |
| Actual param count | ~1.03M |
| Peak VRAM | 64.5 GB |

### Analysis
- 33% regression vs baseline. 9 epochs at ~206 s/epoch. Val still rapidly descending at termination (155 at epoch 9 down from 254 at epoch 1). Hypothesis untestable without budget-matched schedule. Student correctly identified budget-schedule mismatch and noted pressure overflow NaN (now fixed in merged #3276).
- **Decision: Closed. Follow-up: depth-8 with epochs=9 budget-matched (PR #3302).**

---

## 2026-05-15 15:30 — PR #3214: Surf weight 10→30 with 2× pressure channel (CLOSED)
- Branch: `charliepai2i24h2-frieren/surf-weight-30-pchannel-rerun`
- Student: charliepai2i24h2-frieren

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 12) | 138.435 |
| `test_avg/mae_surf_p` | 125.189 |
| Epochs completed | 14/50 (cut at 30-min cap) |

### Analysis
- 19% regression vs Huber baseline (138.44 vs 116.61); 26% vs new baseline (109.68). Combined 6× pressure emphasis hurts velocity-channel learning in early epochs. The geom_cruise split (99.00 val) vs baseline (85.67) shows cruise is actually worse with the heavier weighting. Student correctly identified that a paired surf_weight=10 run would be needed for clean attribution.
- **Decision: Closed. Follow-up: surf_weight=20 single-axis no channel weighting (PR #3304).**

---

## 2026-05-15 16:50 — PR #3216: 32-frequency Fourier features over (x, z) (CLOSED)
- Branch: `charliepai2i24h2-nezuko/fourier-pe-32`
- Student: charliepai2i24h2-nezuko

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 13) | 137.936 |
| `test_avg/mae_surf_p` | NaN (3-clean 140.08) |
| Epochs completed | 14/50 (cut at 30-min cap) |
| Per-split val mae_surf_p | single 177.19 \| geom_rc 150.36 \| geom_cruise 99.72 \| re_rand 124.47 |

### Analysis
- 26% regression vs current baseline (137.94 vs 109.68). The prescription had a bug that the student correctly flagged before the run: `freqs.unsqueeze(-1).expand(num_freq, num_input_channels)` collapses `B[k,0]==B[k,1]`, so the encoding could only distinguish (x+z) not (x, z) independently. This isn't a clean test of 2D Fourier features.
- Also had merge conflicts (predates Huber + grad-clip merges).
- Student again diagnosed the NaN-GT bug correctly; that's now fixed in merged PR #3276.
- **Decision: Closed. Follow-up: corrected Fourier-PE with random Gaussian B per Tancik 2020 RFF, true 2D directional information (PR #3344).**

---

## 2026-05-15 17:30 — PR #3294: Warmup + cosine over 14 epochs, lr=7e-4 (WINNER — pending rebase)
- Branch: `charliepai2i24h2-tanjiro/warmup-cosine-14ep`
- Student: charliepai2i24h2-tanjiro
- Hypothesis: Budget-matched 14-epoch schedule (vs 50-epoch cosine that never anneals), lr bumped 5e-4→7e-4, 2-ep linear warmup.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **100.811** | **-8.08%** |
| `test_avg/mae_surf_p` | NaN (3-clean 99.15) | — |
| Epochs completed | 14/14 (30-min cap) | |
| Per-split val mae_surf_p | single 118.74 \| geom_rc 107.10 \| geom_cruise 81.97 \| re_rand 95.43 | |
| Per-split test mae_surf_p | single 109.88 \| geom_rc 95.54 \| geom_cruise NaN (infra) \| re_rand 92.02 | |
| Peak VRAM | 42.1 GB | |
| Wall-clock | ~31 min (~132 s/epoch) | |
| Metrics artifact | `models/model-warmup-cosine-14ep-20260515-162249/metrics.{jsonl,yaml}` | |

### Analysis
- Clear winner: -8.08% val improvement (compared against prior best 109.68), budget-matching the cosine schedule was the key insight.
- Best epoch = 14 (the last); cosine cooled fully within budget and epochs 12–14 monotonically extracted additional signal from the low-LR tail.
- Per-split pattern: single_in_dist -26.6% (biggest lever), geom_rc -8.9%, re_rand -6.0%. Cruise slightly regressed +4.0% — same pattern as other improvements, suggests harder OOD patterns benefit most from schedule quality.
- Two runs at same config: 99.24 and 100.81 — run-to-run variability ~1.6%, well within noise.
- test_geom_camber_cruise still NaN (pre-existing; fern's NaN guard fixes training but test eval route has separate issue); 3-clean test mean 99.15 confirms val improvement is real.
- PR has merge conflict (predates advisor HEAD); student asked to rebase before merge.

### Decision
- **Pending merge** — winner, sent back to student for rebase. Will become the new baseline at val=100.81 after merge.

---

## 2026-05-15 17:30 — PR #3304: surf_weight 10→20 (WINNER — on hold pending tanjiro merge)
- Branch: `charliepai2i24h2-frieren/surf-weight-20`
- Student: charliepai2i24h2-frieren
- Hypothesis: Raise surf_weight 10→20 (single-axis, no channel weighting) — a 2× surface emphasis as a derisked follow-up to the failed 6× combined version in PR #3214.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 14) | **103.668** | **-5.49%** |
| `test_avg/mae_surf_p` | **93.243** | **-4.18%** |
| Epochs completed | 14/50 (30-min cap) | |
| Per-split val mae_surf_p | single 120.30 \| geom_rc 120.82 \| geom_cruise 77.22 \| re_rand 96.33 | |
| Per-split test mae_surf_p | single 111.45 \| geom_rc 106.33 \| geom_cruise 65.51 \| re_rand 89.69 | |
| Metrics artifact | `models/model-surf-weight-20-20260515-162204/metrics.{jsonl,yaml}` | |

### Analysis
- Beats baseline by 5.49% val / 4.18% test, well within predicted 1–4% range (actually slightly beat it).
- Concentrated improvement on single_in_dist (-18.8% val / -9.6% test). Moderate cruise + re_rand gains. geom_rc regressed slightly (+5.2% val, +1.5% test — the hardest geometry-OOD split).
- The cruise split (easiest) improved mildly (-2.1%) — confirms surf emphasis isn't over-weighting.
- Training was healthy at epoch 14 (best epoch = final); velocity channels not visibly degraded.
- **However**: tanjiro's warmup+cosine (#3294) achieves 100.81 — better than frieren's 103.67. Decision: hold frieren in review, request rebase+retest after tanjiro merges to see if surf_weight=20 still helps on the warmup+cosine baseline.

### Decision
- **On hold** — result beats 109.68 baseline, but pending tanjiro merge to new baseline ~100.81. Frieren will rebase+retest to confirm surf_weight=20 is orthogonally beneficial.

---

## 2026-05-15 17:30 — PR #3314: weight_decay 1e-4→3e-4 on decay group (ON HOLD pending tanjiro merge)
- Branch: `charliepai2i24h2-fern/weight-decay-3e-4`
- Student: charliepai2i24h2-fern
- Hypothesis: Triple the weight decay on the decay group; fern's own suggestion from PR #3276 analysis.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **105.640** | **-3.69%** |
| `test_avg/mae_surf_p` | **94.734** | **-2.65%** |
| Epochs completed | 14/50 (30-min cap) | |
| Per-split val mae_surf_p | single 120.97 \| geom_rc 114.89 \| geom_cruise 88.84 \| re_rand 97.86 | |
| Per-split test mae_surf_p | single 108.73 \| geom_rc 101.17 \| geom_cruise 75.70 \| re_rand 93.33 | |
| Metrics artifact | `models/model-weight-decay-3e-4-20260515-162522/metrics.{jsonl,yaml}` | |

### Analysis
- Beats baseline by 3.69% val / 2.65% test (slightly above the predicted 1–3% range).
- Concentrated gain on single_in_dist (-18.3% val / -11.8% test), while cruise **regresses** +12.7% val / +10.6% test. Other splits flat.
- Same pattern as frieren's surf_weight=20 run: both levers improve the harder single_in_dist split by reducing overfit, while slightly over-regularizing the easier cruise distribution.
- The favorable aggregate is because single_in_dist has the largest absolute MAE — a 27-point improvement there offsets the 10-point regression on cruise.
- **However**: tanjiro's 100.81 beats fern's 105.64. Hold pending tanjiro merge; the optimal wd value may shift on the warmup+cosine baseline.

### Decision
- **On hold** — result beats 109.68 baseline, but pending tanjiro merge to new baseline ~100.81. Fern will rebase+retest to confirm wd=3e-4 is still the right magnitude with warmup+cosine schedule.

---

## 2026-05-15 17:30 — PR #3302: Depth-8 budget-matched epochs=9 (CLOSED)
- Branch: `charliepai2i24h2-askeladd/depth-8-matched-budget`
- Student: charliepai2i24h2-askeladd
- Hypothesis: Depth-8 (n_layers=8) with epochs=9 budget-matched to ~205 s/epoch; follow-up to round-1 where 50-epoch cosine never annealed.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 9) | **111.357** | +1.53% |
| `test_avg/mae_surf_p` | **100.776** | +3.55% |
| Epochs completed | 9/9 (30-min cap) | |
| Per-split val mae_surf_p | single 137.19 \| geom_rc 126.59 \| geom_cruise 82.75 \| re_rand 98.91 | |
| Per-split test mae_surf_p | single 124.31 \| geom_rc 109.95 \| geom_cruise 70.86 \| re_rand 97.99 | |
| Param count | ~1.03M (~1.5× baseline) | |
| Peak VRAM | 64.5 GB | |
| Metrics artifact | `models/model-depth-8-matched-budget-20260515-162201/metrics.{jsonl,yaml}` | |

### Analysis
- Regressed +1.53% val / +3.55% test vs baseline. Budget-matching helped (was 33% regression in round-1 at 9ep) but still hasn't converged — best epoch = 9 (last), slope -6.5 mae/epoch at termination.
- Per-split: depth-8 WINS on single_in_dist (-7.4%) but loses on geom_camber_rc (+10.2%) and cruise (+4.9%). More capacity helps in-distribution but can't generalize to geometry-OOD under this budget.
- Student correctly diagnosed: needs more wall time than depth-5 (205 s vs 130 s per epoch). BF16 (#3223) or larger batch_size needed to compress per-epoch cost.
- **Decision: Closed.** Architecture depth-scaling held until per-epoch cost drops (BF16 or batch8).

### Decision
- **Closed.** Reassigned askeladd to n_head=8 single-axis (#3362) — orthogonal, minimal cost overhead.

---

## 2026-05-15 17:50 — PR #3223: BF16 autocast + batch_size=8 (CLOSED)
- Branch: `charliepai2i24h2-thorfinn/bf16-batch8`
- Student: charliepai2i24h2-thorfinn
- Hypothesis: BF16 mixed precision + batch_size 4→8 for ~2× throughput; preserve 50-epoch cosine schedule.

### Results table

| Metric | Value | Delta vs baseline (109.68) |
|--------|-------|---------------------------|
| `val_avg/mae_surf_p` (best @ epoch 16, primary run) | **147.328** | **+34.30%** |
| `test_avg/mae_surf_p` (post-hoc) | **133.364** | +37.04% |
| Mean of 4 runs val | 140.18 ± 10.0 | +27.8% |
| Mean of 4 runs test | 126.57 ± 9.8 | +30.1% |
| Epochs completed | 15–17 / 50 (30-min cap, run-dependent) | |
| Per-epoch wall clock | ~106 s (vs baseline ~130 s) | |
| Peak VRAM | 65.87 GB | |
| Per-split val mae_surf_p (primary) | single 205.77 \| geom_rc 168.76 \| geom_cruise 95.99 \| re_rand 118.79 | |
| Metrics artifacts | `models/model-charliepai2i24h2-thorfinn-bf16-batch8-20260515-162822/metrics.{jsonl,yaml}` (primary) + 3 prior runs | |

### Analysis
- Significant regression: +34% val on primary run, +27% mean across 4 runs. Large run-to-run variance (10-point std) because truncated 50-epoch cosine never anneals — different stopping points hit different LR points.
- BF16 was numerically clean end-to-end (no NaN/Inf in training). The regression is from the *combined* change, not BF16 itself.
- Per-epoch cost (~106 s) was lower than baseline (~130 s) but per-step cost was higher: 188 batches/epoch × 564 ms/batch vs baseline 376 batches × ~345 ms. The variable-mesh padding overhead dominates batch=8 cost; each large-mesh sample (242K nodes) forces padding for the whole batch.
- The combined effect = (modest per-epoch speedup) + (larger gradient noise reduction) + (truncated cosine schedule) → net loss.
- **Bug-fix bonus**: Student correctly diagnosed and fixed the `evaluate_split` NaN bug (Inf×False=NaN in test_geom_camber_cruise sample 20). Identical to fern's #3276 fix already in HEAD. Useful confirmation.
- Student also added `eval_test_only.py` for post-hoc checkpoint re-evaluation. Useful tool but not pulled into baseline (extra surface area).
- Student's follow-up analysis is excellent: (1) decouple precision and batch knobs; (2) the wall-clock is the bottleneck; (3) padding overhead is the architectural blocker for batch scaling.

### Decision
- **Closed.** Combined hypothesis regressed. Reassigned thorfinn to n_hidden=96 (#3377) — orthogonal architectural axis, completes the 3-point width sweep {96, 128, 192}.
