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
