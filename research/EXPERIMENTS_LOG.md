# SENPAI Research Results — charlie-pai2g-48h-r5

---

## 2026-05-13 02:20 — PR #1700: β=0.25 + L1 sweep — MERGED (L1 wins, new baseline 59.54)

- **Branch:** `charliepai2g48h5-thorfinn/huber-beta-0.25-l1-sweep`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Continue the monotone β trend: β=0.5 → 0.25 → 0 (pure L1). Two arms.

### Results

| Arm | val_avg/mae_surf_p | Δ vs #1633 | test_avg/mae_surf_p | Δ vs #1633 |
|---|---:|---:|---:|---:|
| β=0.5 baseline (#1633) | 64.0705 | — | 55.4961 | — |
| **Arm A: β=0.25** | **60.7558** | **−5.17%** | **52.3312** | **−5.70%** |
| **Arm B: L1 (β→0)** | **59.5354** | **−7.08%** ✓ | **51.4666** | **−7.26%** ✓ |

| Split | β=0.5 | β=0.25 | L1 |
|---|---:|---:|---:|
| `val_single_in_dist` | 72.5692 | 66.4260 | **64.8899** |
| `val_geom_camber_rc` | 78.3209 | 74.3348 | **74.0437** |
| `val_geom_camber_cruise` | 43.3744 | 42.7601 | **39.9687** |
| `val_re_rand` | 62.0174 | 59.5022 | **59.2391** |

- **Best epoch:** 37/37 (both arms; still descending at timeout).
- **Time/epoch:** ~49.6s (unchanged). **Memory:** 23.83 GB.
- **Artifacts:**
  - `models/model-charliepai2g48h5-thorfinn-l1-loss-20260513-005443/metrics.jsonl`
  - `models/model-charliepai2g48h5-thorfinn-huber-beta-0.25-20260513-000538/metrics.jsonl`

### Analysis

**Complete β sweep: β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07) > β=0.25 (60.76) > L1 (59.54).** Monotone with diminishing returns: 8.2% → 5.2% → 2.0% per halving. The gain saturates near L1 — likely the heavy-tailed surface-pressure residual distribution is well-matched to L1's unit-magnitude gradient throughout training.

**Why L1 > β=0.25:** β=0.25 keeps a small quadratic central region (|e|<0.25), down-weighting small but useful gradient signal. Pure L1 uses subgradient ±1 sign everywhere, giving consistent step size across all residual magnitudes. This pays off in the late cosine tail (visible in smoother trajectory: epoch 33-37 less bouncy on L1).

**Mechanism closed:** β axis is fully characterized. No further β sweep needed — L1 is the optimal point under the 30-min budget.

### Conclusions

- **Merged as new advisor baseline.** val_avg = 59.5354, test_avg = 51.4666.
- All in-flight PRs that were on β=0.5 baseline (64.07) need L1 rebase to be comparable.
- L1 + grad-clip and L1 + WD=5e-5 are the next two high-confidence stack candidates.

---

## 2026-05-13 02:20 — PR #1775: WD=5e-5 on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-fern
- **Result:** val_avg=61.2311 (−4.43% vs 64.07); test_avg=53.8792 (−2.91%); wins ALL 4 splits.
- **Why sent back:** L1 merged (new baseline 59.54); 61.23 does not beat it. WD lever likely stacks.
- **Per-split analysis:** Mirror-image prediction failed — model under-regularized everywhere. Monotone: WD lower → better across all splits.
- **Artifacts:** `models/model-charliepai2g48h5-fern-weight-decay-5e-5-20260513-012009/metrics.jsonl`

---

## 2026-05-13 02:20 — PR #1653: grad-clip on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-askeladd
- **Rebase result:** val_avg=59.6214 (−6.94% vs 64.07); test_avg=52.6522 (−5.13%).
- **Why sent back:** L1 merged (new baseline 59.54); 59.62 does not beat it (within noise margin).
- **Key diagnostic:** Pre-clip p50 grew from 28→36 at epoch 1 under β=0.5 (sharper β → larger grads). clip_frac ≈ 1.0 throughout. Lever confirmed real on β=0.5; need to test on L1.
- **Old result (β=1.0):** val_avg=59.42 (−14.92% vs β=1.0 baseline 69.83).
- **Artifacts:** `models/model-charliepai2g48h5-askeladd-grad-clip-1.0-beta-0.5-rebase-20260513-011736/metrics.jsonl`

---

## 2026-05-13 02:20 — PR #1826: cosine eta_min=5e-5 — ASSIGNED (thorfinn)

- **Branch:** `charliepai2g48h5-thorfinn/cosine-eta-min-5e-5`
- **Hypothesis:** Add a non-zero LR floor to CosineAnnealingLR (eta_min=5e-5 = lr/10). Prevents gradient collapse in the low-LR tail; motivated by best_epoch=terminal in all recent runs.
- **Baseline to beat:** val_avg < 59.5354.

---

## 2026-05-13 05:00 — Round 14: PR reviews and new assignments

### PR #1788: attention-dropout=0.1 — CLOSED (slow convergence, budget-bound loss)

- **Student:** charliepai2g48h5-frieren
- **Result:** val_avg=65.8345 (+2.75% vs OLD 64.07 baseline), test_avg=59.3951 (+7.03%).
- **Analysis:** Best epoch = terminal epoch (36/36). All four val splits regressed — no preferential OOD gain from attention dropout. Per-weight activation noise dominated regularization benefit under 36-epoch cap. "Closed on 30-min regime" — slower convergence never caught the baseline.
- **Closed axes:** Attention dropout (per-weight) at p=0.1 closed under wall-clock cap. DropPath (block-level) remains untested — different convergence profile.
- **Artifacts:** `models/model-charliepai2g48h5-frieren-attention-dropout-0.1-20260513-020018/metrics.jsonl`

---

### PR #1741: mlp_ratio=3 — CLOSED (capacity axis triangulated closed)

- **Student:** charliepai2g48h5-edward
- **Result:** val_avg=68.9250 (+7.6% vs OLD 64.07 baseline), test_avg=61.9016 (+11.5%).
- **Param count:** 826K (up from 662K). Per-epoch: +6.5% (within prediction). Epochs: 34 (vs 36 baseline).
- **Analysis:** Plateau-bound trajectory (oscillating tail: ep 32→33→34: 68.92→69.26→72.47). NOT the undertrained-but-converging shape. Combined with #1688 (n_hidden=160 also lost), **both capacity axes (uniform width and asymmetric FFN) are triangulated closed** on 30-min budget. Future gains from regularization, optimizer, schedule, or loss-shape — not model size.
- **Artifacts:** `models/model-charliepai2g48h5-edward-mlp-ratio-3-20260513-020905/metrics.jsonl`

---

### PR #1774: lr=7.5e-4 on β=0.5 — SENT BACK (L1 rebase needed)

- **Student:** charliepai2g48h5-alphonse
- **Result (2 runs, β=0.5):** mean val_avg=63.30 (-1.20% vs 64.07), mean test_avg=56.34 (+1.51%). Run-to-run gap: 1.06 val, 1.55 test — effect size within noise floor.
- **Why sent back:** L1 merged (new baseline 59.54). Student's own analysis: "revisit if loss landscape changes" — L1 IS that change. Unit-bounded gradients (L1 sign) change what optimal LR is. Per-epoch cost: UNCHANGED (~49.5s). Sent back for L1 rebase.
- **Predicted L1 outcome:** 57-59 val if larger step stacks; wash if neutral; >60 if L1 high-variance sign gradients penalize larger steps.
- **Artifacts:** `models/model-charliepai2g48h5-alphonse-lr-7.5e-4-20260513-012003/metrics.jsonl`, `models/model-charliepai2g48h5-alphonse-lr-7.5e-4-20260513-015915/metrics.jsonl`

---

### PR #1845: AdamW betas=(0.9, 0.95) — ASSIGNED (edward)

- **Branch:** `charliepai2g48h5-edward/adamw-betas-0.9-0.95`
- **Hypothesis:** On L1 loss, every gradient is a unit sign (bounded, high-variance). beta2=0.999 (1000-step memory) over-smooths the preconditioner. beta2=0.95 (20-step memory) adapts faster to per-parameter gradient variance. Modern transformer default (GPT/LLaMA). Single line change.
- **Baseline to beat:** val_avg < 59.5354.

---

### PR #1846: slice_num 64 → 32 — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/slice-num-32`
- **Hypothesis:** slice_num=64 may be over-allocated for TandemFoilSet's natural spatial structure (~10-20 canonical CFD regimes: stagnation, transition, separation, wake, inter-foil). Reducing to 32 tightens the attention bottleneck inductive bias, reduces per-epoch time ~3-5%, and forces load-balancing across fewer slices. CFD intuition: fewer coarser slices ≈ more physics-consistent decomposition than 64 fine-grained slices spread thin.
- **Note:** #1590 (slice_num=96 on bf16) was closed at +3.86% regression — but that was LARGER slices. This is the SMALLER direction and a different compute regime.
- **Baseline to beat:** val_avg < 59.5354.

---

## 2026-05-13 01:55 — PR #1652: warmup-500 + cosine (β=0.5 rebase) — CLOSED (substituted by β)

- **Branch:** `charliepai2g48h5-frieren/warmup-500-cosine`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Linear warmup over 500 steps stacked on Huber β=0.5 baseline; predicted additive gain since mechanisms (LR-trajectory vs loss-shape) seemed orthogonal.

### Results

| Metric | Baseline (#1633 β=0.5) | This PR (warmup+β=0.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | 63.9922 | −0.12% (noise floor) |
| `test_avg/mae_surf_p` | 55.4961 | **55.9481** | **+0.81%** ✗ (all 4 test splits worse) |

| Split | warmup+β=0.5 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 71.90 | 72.57 | −0.92% |
| `val_geom_camber_rc` | 77.08 | 78.32 | −1.58% |
| `val_geom_camber_cruise` | 44.36 | 43.37 | +2.28% |
| `val_re_rand` | 62.62 | 62.02 | +0.98% |

- **Epochs:** 36. **Time/epoch:** ~49.65s (unchanged). **Best epoch:** 36/36 (still descending).
- **Artifacts:** `models/model-charliepai2g48h5-frieren-warmup-500-on-huber0.5-20260513-001402/metrics.jsonl`

### Analysis — paper-grade mechanism finding

The student's split-direction analysis is the high-value part. On β=1.0 (her original #1652 run), warmup helped OOD splits (camber_rc −6.4%, camber_cruise −3.5%, re_rand −3.1%) and hurt in-dist (+6.1%) — classical "warmup → flatter minimum" mechanism. On β=0.5 rebase, the pattern *inverts*: warmup helps in-dist and camber_rc, hurts camber_cruise and re_rand. **Warmup and β=0.5 are competing for the same "early-training stabilization" lever**, not stacking additively.

**Three independent students converging on the same axis:**
- Frieren #1652 warmup-500 on β=1.0: −1.62% val
- Askeladd #1653 grad-clip on β=1.0: −14.92% val
- Huber β=0.5 itself (PR #1633): −8% val

All three reduce gradient-signal volatility in early training, but they substitute for each other (warmup on β=0.5 = no gain). Future "stabilization" interventions must come from a different mechanism (data-distribution warmup, schedule-completion alignment, etc.).

### Conclusions

- Closes the LR-trajectory-warmup axis on β=0.5 base.
- The "early-step coherence" cluster is well-explored; further gains must come from capacity, data distribution, late-training schedule, or eval-time techniques.
- Predicts: askeladd's #1653 rebase will show *partial* (not additive) gain — likely a substantial fraction of −14.92% will be captured by β=0.5 already.

---

## 2026-05-13 01:55 — PR #1660: EMA decay=0.999 — CLOSED (pod never started)

- **Branch:** `charliepai2g48h5-tanjiro/ema-eval-decay-0.999-compile`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Per-step EMA of weights for evaluation, decay=0.999.
- **Reason for closure:** Pod stuck in GraphQL rate-limit retry loop for 3+ hours (since ~22:30 UTC 2026-05-12). Only commit on branch is the assignment commit `1f9dfdd`. EMA experiment was never actually started.
- **Reassignment:** PR #1789 surf_weight=15 (simpler 1-line change, smaller failure surface).

---

## 2026-05-13 01:55 — PR #1788: attention dropout=0.1 — ASSIGNED (frieren)

- **Branch:** `charliepai2g48h5-frieren/attention-dropout-0.1`
- **Hypothesis:** Activate PhysicsAttention dropout (existing wired parameter, currently 0.0) at p=0.1. Activation-level regularization, orthogonal to L2 (fern's WD work) and grad-clip (askeladd). Targets slice-attention pathway redundancy.
- **Config change:** Add `dropout=0.1` to `model_config` dict.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:55 — PR #1789: surf_weight 10 → 15 — ASSIGNED (tanjiro)

- **Branch:** `charliepai2g48h5-tanjiro/surf-weight-15`
- **Hypothesis:** Up-weight surface loss term to align training objective with the surf-p-primary validation metric. +50% bump to test the loss-balancing axis on β=0.5 base.
- **Config change:** `cfg.surf_weight` 10.0 → 15.0 (1-line in dataclass).
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:15 — PR #1727: weight_decay 1e-4 → 5e-4 — CLOSED (regression)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-4`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** Stronger L2 regularization targets OOD splits (geom_camber_rc, re_rand) where train/val distribution shift is real.

### Results

| Metric | Baseline (#1633) | This PR (WD=5e-4) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | **66.6723** | **+4.06%** ✗ |
| `test_avg/mae_surf_p` | 55.4961 | **58.6256** | **+5.64%** ✗ |

| Split | WD=5e-4 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 73.9857 | 72.5692 | +1.95% |
| `val_geom_camber_rc` | **77.5818** | 78.3209 | **−0.94%** (only improvement) |
| `val_geom_camber_cruise` | 49.3384 | 43.3744 | **+13.75%** (worst) |
| `val_re_rand` | 65.7833 | 62.0174 | +6.07% |

- **Epochs:** 36 (wall-clock bound). **Time/epoch:** ~49.7s (unchanged). **Best epoch:** 35.
- **Artifacts:** `models/model-charliepai2g48h5-fern-weight-decay-5e-4-20260513-001807/metrics.jsonl`

### Analysis

Under-fit, not over-regularization. Val trajectory lagged baseline by 3-5 equivalent epochs (epoch 18 val_avg=101.98 vs baseline ~92). Still descending at timeout. With 50+ epochs, 5e-4 plausibly catches up; under 30-min cap it's a net loss. One OOD split improved (geom_camber_rc −0.94%) — regularization thesis has *directional* validity but swamped by convergence lag. WD-UP axis closed.

**Follow-up assigned:** weight_decay=5e-5 (PR #1775) — DOWN bracket to close the WD optimum search.

---

## 2026-05-13 01:15 — PR #1701: batch_size 4 → 8 on compile baseline — CLOSED (regression)

- **Branch:** `charliepai2g48h5-alphonse/batch-size-8-compile`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Larger batch → better gradient quality per step, tests quality-vs-quantity trade-off on compile+bf16 baseline.

### Results

| Metric | Baseline (#1633) | batch=8 | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 64.0705 | **74.4033** | **+16.1%** ✗ |
| `test_avg/mae_surf_p` | 55.4961 | **66.8002** | **+20.4%** ✗ |

| Quantity | batch=4 | batch=8 | Δ |
|---|---:|---:|---:|
| Steps/epoch | 375 | 188 | −50% |
| Total grad updates | 13,875 | 6,392 | **−54%** |
| Time/epoch | 49.7s | 53.4s | +7.4% |
| Peak GPU memory | 23.83 GB | 47.63 GB | +2× |

- **Artifacts:** `models/model-charliepai2g48h5-alphonse-batch-size-8-compile-20260513-000307/metrics.jsonl`

### Analysis

Textbook small-dataset step-count starvation. Matched-wall-clock probe (epoch 24): batch=8 (94.95) within 1.8% of batch=4 (93.26) — gradient quality is similar. The loss is entirely in step count: −54% total grad updates means the cosine-LR tail (which drives the last 1.5 MAE points) is never reached. **Batch scaling is fully dead** at TandemFoil scale in all compute regimes (#1439 fp32, #1701 compile).

**Follow-up assigned:** lr=7.5e-4 (PR #1774) — raise per-step magnitude while keeping step count.

---

## 2026-05-13 01:15 — PR #1653: grad-clip max_norm=1.0 — SENT BACK (β=0.5 rebase required)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-1.0-compile`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Gradient norm clipping at max_norm=1.0 to stabilize large-gradient regime.

### Results (on stale β=1.0 baseline)

| Metric | β=1.0 compile baseline (#1568) | grad-clip | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **59.4157** | **−14.92%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **53.0090** | **−14.32%** ✓ |

**Note: askeladd's branch has `beta=1.0` at lines 246, 486 — result measured on OLD β=1.0 compile baseline, not current β=0.5 advisor (val_avg=64.07).** Cannot merge as-is.

### Key grad-norm diagnostics

| Epoch | grad_norm_p50 | clip_frac |
|---|---:|---:|
| 1 | 28.39 | 1.00 |
| 6 | 15.20 | 1.00 |
| 16 | 10.37 | 1.00 |
| 26 | 6.52 | 0.99 |
| 37 | 4.92 | 0.95 |

Pre-clip norms are 10-30× above threshold throughout training. This is not a rare-spike guard — it is near-uniform per-step downscaling, acting as an adaptive LR floor that bounds the AdamW step to `lr × max_norm = 5e-4`.

### Action

Sent back for β=0.5 rebase. When stacked with β=0.5, if orthogonal: val_avg could reach ~55-58. If redundant: grad-clip and β=0.5 share the same "make gradient signal more coherent" mechanism and the stacking will be smaller.

---

## 2026-05-13 01:15 — PR #1774: lr=7.5e-4 — ASSIGNED (alphonse)

- **Branch:** `charliepai2g48h5-alphonse/lr-7.5e-4`
- **Hypothesis:** From #1701's step-count analysis: raise per-step magnitude (lr +50%) while keeping step count constant. β=0.5 reduced outlier gradient magnitude vs β=1.0, leaving room for larger LR. CosineAnnealingLR peak moves 5e-4 → 7.5e-4.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 01:15 — PR #1775: weight_decay=5e-5 — ASSIGNED (fern)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-5`
- **Hypothesis:** From #1727's bracketology: DOWN sweep to complete 3-point WD bracket (5e-4=bad, 1e-4=baseline, 5e-5=this test). Closes the WD axis or identifies weaker-regularization win.
- **Baseline to beat:** val_avg < 64.0705.

---

## 2026-05-13 02:40 — PR #1688: n_hidden 128 → 160 on compile baseline — CLOSED (width ruled out)

- **Branch:** `charliepai2g48h5-edward/wider-hidden-160-compile`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver n_hidden 128→160 on compile + β=0.5 baseline.

### Results

| Metric | Baseline (#1568 compile) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **73.6658** | **+5.49%** ✗ |
| `test_avg/mae_surf_p` | 61.8652 | **64.5826** | **+4.39%** ✗ |

| Split | n_hidden=160 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 89.83 | 77.10 | **+16.5%** |
| `val_geom_camber_rc` | 83.13 | 83.49 | -0.4% |
| `val_geom_camber_cruise` | 52.78 | 50.64 | +4.2% |
| `val_re_rand` | 68.92 | 68.10 | +1.2% |

- **Best epoch:** 30/31. **Time/epoch:** ~58.3 s (+17.7%). **Peak GPU:** 28.4 GB. **Params:** 1.027M.
- **Val trajectory:** Oscillating in 73-85 range (not cleanly descending like baseline).
- **Metric artifacts:** `models/model-charliepai2g48h5-edward-wider-hidden-160-compile-20260512-235416/metrics.jsonl`

### Analysis

**Compute starvation — same mechanism as depth experiments.** Per-epoch cost 49.5→58.3s (+17.7%), epochs 36→31 (-14%), val_single_in_dist regressed +16.5% (in-dist split, should be best if capacity actually helped). Val curve oscillated 73-85 rather than cleanly descending.

**Width axis now fully ruled out under 30-min cap.** Complete lever characterization:
- n_hidden=192+fp32 (#1398): wall-clock bound
- n_hidden=160+bf16 (#1587): pod stall
- n_hidden=160+compile (#1688): +5.49% loss

**Student's valuable insight:** `mlp_ratio=3` is the next-cheapest targeted test — affects FFN only, attention cost unchanged, ~5-8% per-epoch overhead vs 17.7% for uniform widening.

### Conclusions

- Uniform width scaling is dead under the 30-min cap. Do not re-run on β=0.5 baseline.
- `mlp_ratio=3` assigned as PR #1741 — smallest-footprint capacity change.
- If mlp_ratio=3 also loses, all capacity axes are closed and we should focus entirely on regularization and data levers.

---

## 2026-05-13 02:40 — PR #1741: mlp_ratio 2 → 3 — ASSIGNED (edward)

- **Branch:** `charliepai2g48h5-edward/mlp-ratio-3`
- **Student:** charliepai2g48h5-edward (fresh assignment after #1688 closed)
- **Hypothesis:** FFN-only capacity increase. mlp_ratio=3 → FFN hidden 256→384; attention cost unchanged.
- **Mechanism:** Slice-attention does spatial mixing; FFN does per-token non-linear projection. Richer FFN may model more complex physics interactions without the full compute hit of uniform widening.
- **Config change:** `mlp_ratio=2` → `mlp_ratio=3` in model_config dict (single-line diff)
- **Expected per-epoch cost:** ~52-53s (vs 49.5 baseline; 17.7% for n_hidden=160)
- **Expected epochs:** ~33-34 (vs 36 baseline; 31 for n_hidden=160)
- **Baseline to beat:** val_avg/mae_surf_p < 64.0705.

---

## 2026-05-13 02:20 — PR #1676: AdamW β2=0.95 — CLOSED (lever refuted)

- **Branch:** `charliepai2g48h5-fern/adamw-beta2-0.95`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** β2=0.95 (faster second-moment tracking, "transformer recipe") vs 0.999 default.

### Results

| Metric | Baseline (#1568) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **69.9029** | +0.10% (wash) |
| `test_avg/mae_surf_p` | 61.8652 | **62.9973** | +1.83% ✗ |

| Split | β2=0.95 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 79.9420 | 77.10 | +2.84 |
| `val_geom_camber_rc` | 83.7474 | 83.49 | +0.26 |
| `val_geom_camber_cruise` | 48.7266 | 50.64 | -1.91 |
| `val_re_rand` | 67.1955 | 68.10 | -0.90 |

- **Best epoch:** 36 (terminal, still descending). **Time/epoch:** ~49.78 s. **Peak GPU:** 23.83 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-fern-adamw-beta2-0.95-20260512-230647/metrics.jsonl`

### Analysis

Mixed per-split signal (in-dist slightly worse, two OOD splits slightly better, one slightly worse).
Net result is noise — 69.90 vs 69.83 is within random seed variance. The training loss showed
mild spikes (~10-16% bumps) at epochs 20, 28, 32 — consistent with "spikier Adam" from shorter
second-moment averaging window under batch=4 noise. Notably, 69.90 also doesn't beat the new
64.07 baseline (PR #1633, Huber β=0.5 merged same round).

Student diagnosis is correct: β2=0.95 is suited for large-scale LM training where intra-epoch
gradient distribution shifts are real. On 1499-sample TandemFoil with 375 steps/epoch, the
gradient distribution is stationary — β2=0.999 provides better L2 stability for this regime.
The cosine LR schedule already handles "late-training adaptation" that β2=0.95 was supposed to help with.

### Conclusions

- **β2 axis: CLOSED.** Lever does not transfer to small encoder-only Transolver on this dataset scale.
- Do not re-run further β2 variants (0.99, etc.) — the mechanism mismatch is understood.
- Training-loss bump characterization is useful diagnostic prior: if a future design uses
  aggressive β2, pair with grad-clipping.

---

## 2026-05-13 02:20 — PR #1652: Warmup-500-cosine — SENT BACK (needs β=0.5 rebase)

- **Branch:** `charliepai2g48h5-frieren/warmup-500-cosine`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Linear warmup over 500 steps (LR 0.01×→1.0× peak) + cosine T_max=50 decay.

### Results (on OLD 69.83 baseline)

| Metric | Baseline (#1568) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **68.7004** | **-1.62%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **60.7640** | **-1.78%** ✓ |

| Split | Warmup-500 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 81.77 | 77.10 | +6.1% |
| `val_geom_camber_rc` | 78.16 | 83.49 | **-6.4%** |
| `val_geom_camber_cruise` | 48.87 | 50.64 | **-3.5%** |
| `val_re_rand` | 66.01 | 68.10 | **-3.1%** |

- **Best epoch:** 35/36 (one before terminal). **Time/epoch:** ~49.6 s. **Peak GPU:** 23.8 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-frieren-warmup-500-cosine-20260512-225549/metrics.jsonl`
- **LR trajectory verified:** Epoch-1 start at 5e-6 (0.01×), step-500 peak 5e-4, cosine engaged.

### Analysis

Real lever, but the old baseline (69.83) has been superseded. Warmup-500 delivered -1.62% on val_avg,
concentrated on OOD splits (camber_rc -6.4%, camber_cruise -3.5%, re_rand -3.1%) with an in-dist
regression (+6.1%). The mechanism prediction was correct: warmup → flatter minimum → better OOD
generalization, at small in-dist cost.

68.70 does NOT beat the new 64.07 baseline (PR #1633). Warmup is orthogonal to β=0.5 (one changes
LR trajectory shape, the other changes residual sensitivity). Combined, multiplicative stacking
predicts ~62.7-63.5 val_avg — beats 64.07.

**Note on in-dist regression:** Under β=0.5 (cleaner per-sample gradients for medium residuals),
the in-dist regression may shrink or disappear — an informative interaction effect to watch.

### Conclusions

- **SENT BACK** for rebase onto `icml-appendix-charlie-pai2g-48h-r5` (inherits β=0.5).
- Lever is real; expected to compound with β=0.5.
- Post-merge follow-ups queued: warmup-length sweep (250, 1000, 2000 steps); T_max alignment.

---

## 2026-05-13 02:20 — PR #1619: Sampler 2× compile rebase — SENT BACK AGAIN (needs β=0.5 rebase)

- **Branch:** `charliepai2g48h5-nezuko/sampler-boost-single-2x`
- **Student:** charliepai2g48h5-nezuko
- **Second run:** Compile + sampler-2x (rebase onto #1568 compile baseline per prior advisor feedback)

### Results (on 69.83 compile baseline, second rebase)

| Metric | Compile baseline (#1568) | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.8316 | **68.2641** | **-2.25%** ✓ |
| `test_avg/mae_surf_p` | 61.8652 | **61.4236** | **-0.71%** ✓ |

| Split | Sampler+Compile | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | **64.6884** | 77.10 | **-16.10%** |
| `val_geom_camber_rc` | 85.6303 | 83.49 | +2.56% |
| `val_geom_camber_cruise` | 53.6013 | 50.64 | +5.85% |
| `val_re_rand` | 69.1363 | 68.10 | +1.52% |

- **Best epoch:** 39 (terminal — still descending). **Time/epoch:** ~46.2 s. **Peak GPU:** 23.83 GB.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-boost-single-2x-compile-20260512-230452/metrics.jsonl`

### Analysis

Sampler lever amplified under compile: val_single_in_dist -16.10% (vs -10.7% pre-compile), because
more gradient steps per 30-min window amplify the coverage benefit. Best epoch=39=terminal means
the model is still descending — sampler+compile gains are under-saturated.

68.26 does NOT beat the new 64.07 baseline (PR #1633, β=0.5). Sampler is orthogonal to β=0.5 by
construction (batch sampling vs per-sample loss shape). Combined stacking is the highest-confidence
win remaining in the queue. Predicted multiplicative combination: 69.83 × (1-0.082) × (1-0.0225) ≈ 62.65.

Key diagnostic for the third run: watch val_geom_camber_cruise. β=0.5 alone gave -14.4% on cruise;
sampler-2x alone gave +5.85%. Net direction is uncertain — if they cancel, the implication is that
the 2× cruise-mass reduction is a meaningful cost and "boost both racecar domains" is the right fix.

### Conclusions

- **SENT BACK AGAIN** for third rebase — now onto current `icml-appendix-charlie-pai2g-48h-r5`
  (which has β=0.5). This is the final rebase needed.
- If sampler+β=0.5 beats 64.07, follow-ups: boost-factor sweep (1.5×, 3×); "boost both racecar
  domains (single=2, tandem=2, cruise=1)".

---

## 2026-05-13 02:20 — PR #1727: weight_decay 1e-4 → 5e-4 — ASSIGNED (fern)

- **Branch:** `charliepai2g48h5-fern/weight-decay-5e-4`
- **Student:** charliepai2g48h5-fern (fresh assignment after #1676 closed)
- **Hypothesis:** Stronger L2 regularization improves OOD generalization on the 1499-sample dataset.
- **Config change:** `weight_decay: float = 1e-4` → `weight_decay: float = 5e-4` (single-line diff)
- **Mechanism:** 5× L2 penalty trades mild in-dist capacity for better parameter-space flatness
  on OOD splits (camber_rc, re_rand). Independent of all in-flight experiments (different axis).
- **Prediction:** -1% to -3% on val_avg, concentrated on OOD splits. val_avg landing zone: 62-63.5.
- **Baseline to beat:** val_avg/mae_surf_p < 64.0705.

---

## 2026-05-13 01:10 — PR #1560: T_max=36 cosine on compile baseline — CLOSED (lever characterized)

- **Branch:** `charliepai2g48h5-alphonse/tmax-14-cosine`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Match CosineAnnealingLR T_max to the actual epoch budget at the 30-min cap.

### Results (compile-era re-run, T_max=36)

| Metric | Baseline (#1568) | This PR (T_max=36) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 69.832 | **69.598** | -0.234 (-0.34%) |
| `test_avg/mae_surf_p` | 61.865 | **61.729** | -0.136 (-0.22%) |

Per-split (mixed):

| Split | T_max=36 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 76.167 | 77.10 | -0.93 |
| `val_geom_camber_rc` | 81.053 | 83.49 | -2.44 |
| `val_geom_camber_cruise` | 52.196 | 50.64 | +1.56 |
| `val_re_rand` | 68.975 | 68.10 | +0.87 |

- **Best epoch:** 36 (terminal, descending). **Epochs:** 36. **Time/epoch:** 49.8 s.
- **Status:** CLOSED. 69.60 > new baseline 64.07 (Huber β=0.5, PR #1633 merged same round).

### Analysis

**Mechanism confirmed but gain diminished at compile budget.** The T_max=36 win internaly shows -6.2 MAE in epochs 28→36 (exactly as predicted), matching the T_max=14/18 arms' "last few epochs gain ~8 MAE" pattern. But the comparison vs the T_max=50 baseline at 36/50 epochs is tiny (+0.23 MAE) because the T_max=50 compile baseline already captures most of the cosine-decay benefit by epoch 36 (LR decays to ~0.21·lr_max, not zero).

**Lever characterization complete:**

| Arm | T_max | Epochs | Baseline | val_avg | Δ | Mechanism |
|---|---|---|---|---|---|---|
| #1560 A (fp32) | 14 | 14 | #1444 (110.76) | 98.75 | -10.8% | Low-LR tail traversal |
| #1560 B (bf16) | 18 | 18 | #1532 (101.12) | 90.32 | -10.7% | Low-LR tail traversal |
| #1560 C (compile) | 36 | 36 | #1568 (69.83) | 69.60 | -0.34% | Most of arc already covered |

The gain collapses when the baseline already runs most of the cosine arc (36/50 epochs). This is a closed lever.

### Conclusions

- T_max=epoch_budget is a strong win when the baseline epoch count is a small fraction of T_max (e.g. 19/50 epochs with bf16 only). Neutral when the baseline already runs most of the arc.
- Do NOT re-run this hypothesis on the β=0.5 baseline. Same math applies.
- Closed: 69.60 does not beat new baseline 64.07.

---

## 2026-05-13 01:00 — PR #1633: Huber β=0.5 (sharper loss) — MERGED ✓

- **Branch:** `charliepai2g48h5-thorfinn/huber-beta-sweep`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Huber β=0.5 (sharper quadratic-to-linear transition at |e|=0.5) vs β=1.0 baseline. Simultaneously test β=2.0 (smoother). TandemFoil surface pressure has a heavy-tailed residual distribution — smaller β makes loss linear for a wider range of medium residuals, down-weighting outlier gradients.

### Results

| Arm | β | val_avg/mae_surf_p | Δ vs baseline | test_avg |
|---|---|---:|---:|---:|
| **A (winner)** | **0.5** | **64.0705** | **-8.2% ✓** | **55.4961** |
| B | 2.0 | 77.8090 | +11.4% ✗ | 69.2942 |

Per-split — β=0.5 (Arm A):

| Split | β=0.5 | Baseline (#1568) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 72.5692 | 77.10 | -5.9% |
| `val_geom_camber_rc` | 78.3209 | 83.49 | -6.2% |
| `val_geom_camber_cruise` | **43.3744** | 50.64 | **-14.4%** |
| `val_re_rand` | 62.0174 | 68.10 | -8.9% |

- **Best epoch:** 37 (terminal, still descending at timeout). **Time/epoch:** ~49.5 s. **Peak GPU:** 23.83 GB.
- **Status:** MERGED — new baseline 64.0705 / 55.4961.
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-huber-beta-0.5-20260512-221022/metrics.jsonl`

### Analysis

**Clean monotone signal:** β=2.0 (77.81) > β=1.0 (69.83) > β=0.5 (64.07). All four val splits improved. Largest win on val_geom_camber_cruise (-14.4%), the lowest-error split where the bulk of residuals is moderate — sharper β concentrates gradient on the bulk, ignoring outliers.

β=2.0 regression is symmetric: approaching MSE over a wider band overweights tail residuals, hurting all four splits. The heavy-tailed residual distribution of TandemFoil surface pressure is the key underlying mechanism.

Best epoch=37=terminal: model still descending at the wall-clock cap. This is a consistent pattern across all winning PRs — the model has more headroom.

**Key consequence:** β=0.5 is now the advisor baseline. β=0.25 is the natural next step — PR #1700 (thorfinn) queued.

### Conclusions

- Sharper Huber β is a real, zero-cost lever for this dataset.
- Direction is clear: sweep toward β=0.25 and pure L1 to find the optimum.
- Surface-weight interaction possible (surf_weight=10 was tuned with β=1.0; re-tuning with β=0.5 may compound further).

---

## 2026-05-13 00:40 — PR #1587: n_hidden 128 → 160 + bf16 — CLOSED (stale)

- **Branch:** `charliepai2g48h5-edward/wider-hidden-160-bf16`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver n_hidden 128→160 paired with bf16 AMP.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod stalled or
failed to start training; no training trajectory, no metrics.jsonl, no terminal
SENPAI-RESULT. Same pattern as previously stale #1561 (askeladd) and #1535 (tanjiro).

This is the third edward assignment to stall; the hypothesis itself remains valid.
Width capacity has not been refuted — reassigned as PR #1688 on the compile baseline
with explicit n_hidden=160+compile instructions and updated run command.

---

## 2026-05-13 00:10 — PR #1619: RaceCar single sampler boost 2× — SENT BACK (needs compile rebase)

- **Branch:** `charliepai2g48h5-nezuko/sampler-boost-single-2x`
- **Student:** charliepai2g48h5-nezuko
- **Hypothesis:** Boost `racecar_single` sample weights by 2× (→ 50% single / 25% tandem / 25% cruise)
  to close the coverage gap on `val_single_in_dist`, which consistently dominates `val_avg/mae_surf_p`.

### Results

| Metric | Baseline (#1532 bf16) | This PR | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **98.2897** | -2.80% ✓ |
| `test_avg/mae_surf_p` | 91.50 | **88.8539** | -2.89% ✓ |

| Split | This PR | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 107.14 | 120.02 | **-10.73%** |
| `val_geom_camber_rc` | 108.83 | 107.10 | +1.61% |
| `val_geom_camber_cruise` | 82.99 | 82.84 | +0.18% |
| `val_re_rand` | 94.21 | 94.53 | -0.34% |

- **Best epoch:** 18/20 (30-min SENPAI_TIMEOUT_MINUTES cap)
- **Time/epoch:** ~91.5 s (unchanged from bf16 baseline)
- **Peak GPU:** 32.94 GB
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-sampler-boost-single-2x-20260512-215136/metrics.jsonl`
- **Sampler verification:** `racecar_single=2.0, racecar_tandem=1.0, cruise=1.0` — boost applied correctly.

### Status

**SENT BACK** for compile rebase. The 98.29 result beats the bf16 baseline (101.12) by -2.80% but does
NOT beat the current compile baseline (69.83 from PR #1568). The lever is confirmed real —
val_single_in_dist dropped -10.7% — it just needs to be measured on the new advisor baseline.

### Analysis

**Mechanism confirmed.** val_single_in_dist is coverage-bound, not capacity-bound. Doubling the
sampler mass for racecar_single (from 33.3% → 50% of effective mix) gave -10.7% on that split while
all other splits moved ≤1.6% in either direction. Cost: zero (sampler only changes which samples are
drawn, not per-sample compute).

The tiny +1.6% regression on val_geom_camber_rc is expected: racecar_tandem share dropped 33.3% → 25%,
and that split is from RaceCar tandem geometry. The signal is that the per-domain coverage directly
determines per-split performance — a strong signal for sampler as a lever.

**At compile baseline**, the same 2× boost should give a similar relative win: val_single_in_dist
from 77.10 → ~68-69. Net val_avg could reach ~66-68, compounding with the compile gain.

### Conclusions

- Sampler reweighting is a real, orthogonal, zero-cost lever.
- On compile baseline, sampler+compile should compound to give the next round winner.
- Follow-ups queued: 1.5× and 3× boost factor sweep; "both RaceCar domains boosted together."

---

## 2026-05-13 00:10 — PR #1588: n_layers 5 → 6 + bf16 — CLOSED

- **Branch:** `charliepai2g48h5-fern/deeper-6-layers-bf16`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** n_layers=6+bf16 trades 3 epochs of refinement for ~20% more per-step capacity.

### Results

| Metric | Baseline (#1532 bf16, n_layers=5) | This PR (n_layers=6) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **111.058** | **+9.83%** (WORSE) |
| `test_avg/mae_surf_p` | 91.50 | **98.793** | **+7.97%** (WORSE) |

| Split | n_layers=6 | Baseline | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 134.81 | 120.02 | +12.3% |
| `val_geom_camber_rc` | 122.93 | 107.10 | +14.8% |
| `val_geom_camber_cruise` | 86.20 | 82.84 | +4.1% |
| `val_re_rand` | 100.29 | 94.53 | +6.1% |

- **Best epoch:** 14/16 (116.1 s/epoch; 30-min cap hit during epoch 17 validation)
- **Peak GPU:** 38.93 GB

### Analysis

**Depth lever fully ruled out.** Both n_layers=7+fp32 (PR #1413, wall-clock-bound at 10 epochs)
and n_layers=6+bf16 (this PR, 16 epochs) converged on the same qualitative story: deeper model
needs more gradient steps than the 30-min cap allows, and the capacity gain does NOT compensate.

The key falsifying signal: surface metric (mae_surf_p) regressed MORE than volume metric (+8-10%
surface vs +3-5% volume). This is the OPPOSITE of the "extra slice-attention refinement helps near
sharp pressure gradients" mechanism that motivated the hypothesis. The model is under-trained, not
capacity-limited.

Generalisation gap is normal (test < val) and stable across n_layers values — depth doesn't worsen
the gap, it just shifts both metrics worse uniformly.

### Conclusions

- n_layers scaling is the wrong lever for 1499 training samples at 30-min wall-clock cap.
- Two experiments (n_layers=6 and n_layers=7) both lost, and the mechanism analysis confirms this
  is compute starvation, not a data-limited ceiling.
- **Do NOT follow up with n_layers + compile.** Even at 36 epochs, adding a 6th layer would take
  ~139 s/epoch → only ~13 epochs in 30 min. Still worse than the 36-epoch baseline.
- Reassigned fern to AdamW β2=0.95 (transformer fast-adapting recipe, PR #1676).

---

## 2026-05-12 22:45 — PR #1535: EMA model weights for eval (decay=0.999) — CLOSED (stale)

- **Branch:** `charliepai2g48h5-tanjiro/ema-eval-decay-0.999`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Maintain EMA copy of model weights with decay=0.999 and use it for eval —
  typical late-training noise smoothing.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no training trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis itself remains in-play and was reassigned on the compile baseline (decay=0.999,
  `torch.optim.swa_utils.AveragedModel` after compile, eval/test via EMA model).
- No data lost; closing simply frees the student slot.

---

## 2026-05-12 22:45 — PR #1561: Gradient clipping max_norm=1.0 — CLOSED (stale)

- **Branch:** `charliepai2g48h5-askeladd/grad-clip-1.0`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Bound rare large gradient updates via `clip_grad_norm_(.., max_norm=1.0)`;
  also a high-diagnostic-value characterization of training gradient norms.

### Outcome

**CLOSED** with no commits past the original assignment commit. Pod appears to have stalled
or failed to start training; no trajectory, no metrics.jsonl, no terminal SENPAI-RESULT.

### Disposition

- Hypothesis reassigned on the compile baseline with the per-epoch grad-norm aggregation
  (min/p50/mean/max/clip_frac) added so we still get the diagnostic value regardless of
  whether clipping wins on validation.

---

## 2026-05-12 22:45 — PR #1590: slice_num 64 → 96 + bf16 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-96-bf16`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase Transolver slice_num from 64 → 96 paired with bf16 AMP. With
  bf16's ~2× throughput we can afford the slightly more expensive forward and still complete
  full training; more slices = finer flow-field tokenization.

### Results

| Metric | Baseline (#1532 bf16) | This PR (#1590) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.12 | **105.024** | +3.86% (WORSE) |

- **Status:** CLOSED — does not beat bf16 baseline, well behind the new compile baseline (69.83).
- **Metric artifacts:** PR comment; trajectory shows monotone-worse vs slice_num=64 within
  the same epoch budget.

### Analysis

Combined with the round-3 fp32 result (slice_num=128 → val=145.97 at 11 epochs, wall-clock-bound)
and the bf16 result here (slice_num=96 → 105.02 at full budget), the slice_num lever is now
well-characterized:

| slice_num | regime | val_avg/mae_surf_p |
|---:|---|---:|
| 64 | bf16 (baseline) | 101.12 |
| 96 | bf16 | 105.02 (+3.86%) |
| 128 | fp32 (epoch-bound) | 145.97 |

Monotone-worse with slice count. The 64-slice default appears near-optimal for this dataset
size — adding slices adds capacity that overfits or simply costs throughput without finding
useful additional flow-field structure. **slice_num is a dead lever upward**; downward
(slice_num=32 or 48) would be a separate experiment but is a low-priority swing.

### Conclusions

- slice_num=64 is the right value for the current data + architecture.
- Closing this arm; do NOT pair slice_num=96 with compile (no signal it would help).
- Round-6 reassignment: frieren → step-based linear warmup + cosine on compile baseline.

---

## 2026-05-12 22:10 — PR #1568: torch.compile + bf16 AMP for additional throughput — MERGED ✓

- **Branch:** `charliepai2g48h5-thorfinn/torch-compile-bf16`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** `torch.compile(model, dynamic=True)` stacked on top of bf16 AMP doubles
  per-epoch throughput from ~98 s → ~49.5 s, fitting 36 epochs in 30 min vs 19 previously.
  Mechanism: kernel fusion eliminates Python dispatch overhead. `dynamic=True` prevents
  recompilation on variable-length mesh batches (N_max varies per batch).

### Results

| Metric | Baseline (#1532) | This PR (#1568) | Δ |
|---|---|---:|---:|
| `val_avg/mae_surf_p` | 101.1212 | **69.8316** | **-30.9%** |
| `test_avg/mae_surf_p` | 91.5013 | **61.8652** | **-32.4%** |

| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 77.10 | -35.8% |
| `val_geom_camber_rc` | 83.49 | -22.0% |
| `val_geom_camber_cruise` | 50.64 | -38.9% |
| `val_re_rand` | 68.10 | -28.0% |

| Split | test mae_surf_p |
|---|---:|
| `test_single_in_dist` | 67.81 |
| `test_geom_camber_rc` | 77.68 |
| `test_geom_camber_cruise` | 41.98 |
| `test_re_rand` | 59.99 |

- **Status:** MERGED — new baseline 69.8316 / 61.8652.
- **Epochs reached:** 36 (timeout-bound, 29.41 min; best epoch = 36, still descending)
- **Time/epoch:** ~49.5 s (2.0× speedup vs bf16-only ~98 s)
- **Peak GPU:** 23.8 GB (64 GB headroom on 96 GB card)
- **Compile status:** active for all 36 epochs, no recompilation stalls with `dynamic=True`
- **Metric artifacts:** `models/model-charliepai2g48h5-thorfinn-torch-compile-bf16-20260512-205152/metrics.jsonl`

### Analysis

The win is almost entirely explained by epoch count: 36 vs 19 epochs = ~1.9× more gradient
steps. The model was monotonically improving through epoch 36 with no late-training instability.
`dynamic=True` was the correct choice — without it, dynamo would specialize per N_max and
accumulate recompilation costs that outweigh the kernel-fusion gain on variable-mesh batches.

All 4 val splits improved uniformly (+22-39%), including the hardest OOD splits. This is
pure optimization headroom, not overfitting.

**Key consequence:** The new 36-epoch budget changes the arithmetic for every in-flight arm.
- Capacity arms (#1587, #1588, #1590) were targeting n_hidden=160/n_layers=6/slice_num=96
  + bf16 (without compile). With compile now on advisor, those arms now run at compile speed
  IF they rebase — but they were branched before this merge and won't automatically have compile.
- T_max=50 cosine schedule with 36 epochs reaches LR≈0.012 at epoch 36 (not the full
  low-LR tail). Alphonse's T_max=18 result proved the terminal LR decay matters — so
  T_max=36 on top of compile is now the highest-confidence cheap win.

### Conclusions

- torch.compile is a free 2× throughput multiplier with no accuracy cost.
- 23.8 GB peak (batch=4, n_hidden=128) leaves 72 GB headroom for capacity exploration.
- Budget is still binding at 36 epochs — the model is still descending. More compute =
  more improvement. Highest-value follow-up: T_max=36 schedule to exploit the low-LR tail.

---

## 2026-05-12 22:00 — PR #1560: Match cosine T_max to actual epoch budget — SENT BACK

- **Branch:** `charliepai2g48h5-alphonse/tmax-18-cosine`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** `CosineAnnealingLR(T_max=50)` with 19 bf16 epochs never reaches the
  low-LR tail. Setting T_max=epoch_budget (originally T_max=14 for fp32, T_max=18 for
  bf16) lets the schedule complete, adding a meaningful low-LR fine-tuning phase.

### Results (two arms)

**Arm A — T_max=14 (fp32-era budget, pre-bf16 advisor commit 1341b98):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **98.7502** (best epoch = 14, terminal) |
| `test_avg/mae_surf_p` | **88.8030** |
| Epochs reached | 14/14 (complete) |
| Time/epoch | ~132.4 s (fp32) |
| vs #1444 baseline (110.76) | -10.8% |

**Arm B — T_max=18 (bf16-era budget, current advisor commit afd445a):**
| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **90.3237** (best epoch = 18, terminal) |
| `test_avg/mae_surf_p` | **80.1938** |
| Epochs reached | 18/18 (complete) |
| Time/epoch | ~98.0 s (bf16) |
| vs #1532 baseline (101.12) | **-10.7%** |

Per-split Arm B (tmax-18):
| Split | val mae_surf_p | Δ vs #1532 |
|---|---:|---:|
| `val_single_in_dist` | 105.86 | -14.2% |
| `val_geom_camber_rc` | 99.48 | -7.1% |
| `val_geom_camber_cruise` | 70.74 | -14.6% |
| `val_re_rand` | 85.22 | -9.8% |

- **Status:** SENT BACK — baseline moved to 69.83 (PR #1568 merged). T_max=18 (90.32)
  no longer beats new baseline. Reassigned to retest with T_max=36 matching compile budget.
- **Metric artifacts:** `models/model-charliepai2g48h5-alphonse-tmax-18-cosine-20260512-210749/metrics.jsonl`,
  `models/model-charliepai2g48h5-alphonse-tmax-14-cosine-20260512-201325/metrics.jsonl`

### Analysis

**Mechanism confirmed.** Best epoch = terminal epoch in BOTH arms. The cosine schedule's
low-LR tail (final ~20-25% of epochs where LR approaches 0) provides material fine-tuning
benefit. The trajectory is clear:

val_avg at epochs 14→18 in Arm B: 98.34 → 92.62 → 92.34 → 91.44 → **90.32** — the last 4
epochs (T_max=14 to terminal) gained ~8.0 absolute MAE points. This is the "low-LR tail" the
hypothesis predicted.

**At epoch 14, both arms agree** (Arm B epoch 14 = 98.34, Arm A terminal = 98.75) — the LR
trajectory difference up to epoch 14 is negligible. The improvement is purely from completing
the cosine arc.

**Key implication for compile baseline:** With torch.compile reaching 36 epochs, the "natural
budget" has doubled. T_max=36 would complete the cosine arc and provide the same low-LR tail
effect — potentially gaining ~8-12 MAE off the 69.83 baseline.

### Conclusions

- Schedule-completion is a real, cheap, orthogonal lever. Best epoch = terminal epoch = strong
  signal that the low-LR tail does fine-tuning work.
- T_max=18 is obsolete — compile changed the budget to 36 epochs.
- **Follow-up:** alphonse re-running PR #1560 with `--epochs 36` on the updated advisor branch.
  If the same epoch-14→18 proportional gain holds (~8% of the remaining MAE), val_avg could
  drop from ~60 (extrapolating compile curve) to ~55 in the final epochs.

---

## 2026-05-12 21:30 — PR #1428: Per-channel loss weights [1,1,3] favoring pressure — CLOSED

- **Branch:** `charliepai2g48h5-nezuko/pressure-channel-weight`
- **Student:** charliepai2g48h5-nezuko
- **Hypothesis:** Reweight loss channels [1,1,3] so the pressure channel (the
  one we're scored on) carries 3× the gradient signal of Ux/Uy. Expected
  -5% to -12% delta on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **135.5317** (epoch 13 best) |
| `val_single_in_dist` | 167.07 |
| `val_geom_camber_rc` | 143.28 |
| `val_geom_camber_cruise` | 103.33 |
| `val_re_rand` | 128.44 |
| `test_avg/mae_surf_p` | **122.2302** (finite — student applied scoring workaround) |
| Best epoch | 13 |
| Epochs reached | 14 (timeout-bound, ~131 s/epoch, fp32 — pre-bf16) |
| Peak GPU | 42.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch, pre-bf16) |

- **Status:** CLOSED — +34.1% worse than bf16 baseline 101.12.
- **Metric artifacts:** `models/model-charliepai2g48h5-nezuko-pressure-channel-weight-20260512-200303/metrics.jsonl`

### Analysis

Two compounding factors explain the poor result:

1. **Wall-clock disparity.** Branch predates PR #1532 — 14 fp32 epochs at
   ~131 s/epoch vs baseline's ~19 bf16 epochs at ~98 s/epoch. Partially
   accounts for the gap (maybe 50%?).

2. **Channel weighting fundamentally wrong at 3×.** All four val splits
   regressed — including val_geom_camber_cruise (103.33 vs 82.84 at
   baseline). The only mechanistic explanation for regression on ALL splits
   simultaneously is that [1,1,3] distorted the optimization geometry.
   With 3× pressure gradient, the model optimizes pressure at the expense
   of Ux/Uy, but pressure predictions depend on accurate velocity (physical
   coupling), so the interference cascades back to `mae_surf_p`. Even on
   the "easiest" split (`val_geom_camber_cruise`) only reached ~25% above
   the baseline's full-budget performance at epoch 13.

3. **Student's diagnostic insight for `val_single_in_dist`.** Student noted
   this split (RaceCar single random hold-out) is the hardest despite being
   in-distribution — suggesting the WeightedRandomSampler may be
   under-covering that domain. This is the seed for the reassignment below.

### Conclusions

- Per-channel reweighting at [1,1,3] is ruled out — too aggressive, harms Ux/Uy
  via physical coupling, all-split regression.
- Milder weights ([1,1,2] or [1,1,1.5]) might be worth revisiting after
  other improvements are stacked, but the priority is the sampler direction.
- **New assignment for nezuko (PR #1619): domain-aware sampler reweighting** —
  boost RaceCar single sample weights 2× (→ 50% share) to directly attack
  `val_single_in_dist` coverage deficit. Inherits bf16 AMP + scoring fix.

---

## 2026-05-12 20:55 — PR #1422: slice_num 64 → 128 — CLOSED

- **Branch:** `charliepai2g48h5-frieren/slice-num-128`
- **Student:** charliepai2g48h5-frieren
- **Hypothesis:** Increase `slice_num` from 64 to 128 to give Transolver
  more physics-aware slice tokens per attention layer.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **145.9708** (epoch 11 best) |
| `val_single_in_dist` | 184.67 |
| `val_geom_camber_rc` | 154.30 |
| `val_geom_camber_cruise` | 114.72 |
| `val_re_rand` | 130.19 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 11 |
| Epochs reached | 11 (timeout-bound) |
| Time/epoch | ~171 s (vs ~131 s baseline) |
| Peak GPU | 54.5 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +44% worse than baseline 101.12.

### Analysis

Same diagnosis as #1398, #1413: capacity scale-up at fp32 + MSE only fits
11 epochs in the 30-min cap, vs baseline's 19 epochs (bf16) — undertrained.
Val still descending monotonically through epoch 11 (no plateau, no
instability, no OOM at 54.5 GB). The lever itself isn't refuted — the
budget is binding.

### Conclusions

- slice_num=128 untestable under current wall-clock budget without bf16.
- Next assignment for frieren: slice_num=96 + bf16 inheritance (PR #1590) —
  milder slice bump paired with throughput fix for fair test.

---

## 2026-05-12 20:55 — PR #1413: n_layers 5 → 7 — CLOSED

- **Branch:** `charliepai2g48h5-fern/deeper-7-layers`
- **Student:** charliepai2g48h5-fern
- **Hypothesis:** Increase `n_layers` from 5 to 7 (deeper Transolver) to
  give more iterative slice-attention refinement.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **144.9040** (epoch 10 best) |
| `val_single_in_dist` | 171.26 |
| `val_geom_camber_rc` | 177.24 |
| `val_geom_camber_cruise` | 103.29 |
| `val_re_rand` | 127.83 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~181 s |
| Peak GPU | 57.1 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +43% worse than baseline 101.12.

### Analysis

Same diagnosis as the capacity-arms pattern: at n_layers=7 + fp32 + MSE,
only 10 epochs fit in the 30-min cap. Val descended steeply through
epoch 10 with no plateau. No instability, no OOM. Wall-clock is the
binding constraint, not depth.

### Conclusions

- n_layers=7 untestable under current budget without bf16.
- Next assignment for fern: n_layers=6 + bf16 inheritance (PR #1588) —
  milder depth bump paired with throughput fix.

---

## 2026-05-12 20:53 — PR #1398: n_hidden 128 → 192 — CLOSED

- **Branch:** `charliepai2g48h5-edward/wider-hidden-192`
- **Student:** charliepai2g48h5-edward
- **Hypothesis:** Widen Transolver `n_hidden` from 128 to 192 for more
  representational capacity.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **138.1375** (epoch 10 best) |
| `val_single_in_dist` | 187.30 |
| `val_geom_camber_rc` | 141.23 |
| `val_geom_camber_cruise` | 103.21 |
| `val_re_rand` | 120.81 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) |
| Best epoch | 10 |
| Epochs reached | 10 (timeout-bound) |
| Time/epoch | ~186 s |
| Peak GPU | 58.0 GB |
| Loss used | **MSE** (pre-Smooth-L1 branch) |

- **Status:** CLOSED — +37% worse than baseline 101.12.

### Analysis

Trajectory was volatile at epoch 7-10 (167→179→197→138) — clearly still
in early-training oscillation, not converged. Wider model at fp32 trades
epochs for capacity 1-for-1. No instability, no OOM. Pattern matches
fern (#1413) and frieren (#1422) exactly: wall-clock is binding for
capacity scale-ups under MSE+fp32.

### Conclusions

- n_hidden=192 untestable under current budget without bf16.
- Three students (edward, fern, frieren) independently identified the
  same pattern: capacity-scale-up arms get killed by wall-clock cap
  unless paired with throughput recovery (bf16).
- Next assignment for edward: n_hidden=160 + bf16 inheritance (PR #1587) —
  milder width bump paired with throughput fix.

---

## 2026-05-12 20:01 — PR #1532: bf16 AMP for 2x epoch throughput + scoring-NaN fix — MERGED

- **Branch:** `charliepai2g48h5-thorfinn/bf16-amp-scoring-fix`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Enable bf16 mixed-precision training (`torch.autocast("cuda", dtype=torch.bfloat16)`) to increase epoch throughput and reach more training epochs within the 30-min cap. Also includes scoring-NaN workaround: batch-level `y_finite_mask` filter in `evaluate_split` before `accumulate_batch`.

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **101.1212** (epoch 17 best) |
| `val_single_in_dist/mae_surf_p` | 120.0176 |
| `val_geom_camber_rc/mae_surf_p` | 107.0980 |
| `val_geom_camber_cruise/mae_surf_p` | 82.8425 |
| `val_re_rand/mae_surf_p` | 94.5268 |
| `test_avg/mae_surf_p` | **91.5013** (finite — first on this branch) |
| `test_single_in_dist/mae_surf_p` | 105.4434 |
| `test_geom_camber_rc/mae_surf_p` | 99.9931 |
| `test_geom_camber_cruise/mae_surf_p` | 69.2841 |
| `test_re_rand/mae_surf_p` | 91.2844 |
| Best epoch | 17 |
| Epochs reached | 19 (~25% faster at ~98 s/epoch vs ~131 s) |
| Peak GPU | 32.95 GB |

- **Improvement:** -9.64 MAE (-8.7%) vs PR #1444 baseline (110.7608)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-bf16-amp-scoring-fix-20260512-192502/metrics.{jsonl,yaml}`
- **Status:** MERGED → new round-5 baseline floor: **val_avg/mae_surf_p = 101.1212**

### Analysis

1. **bf16 AMP gave a real throughput win**: ~25% faster per epoch (98 vs 131 s), reaching epoch 19 vs baseline's epoch 14 — 5 extra epochs of convergence. The extra epochs drove the primary win: best epoch 17 vs baseline's 14.

2. **Scoring fix unblocked test_avg**: The `y_finite_mask` filter in `evaluate_split` correctly skipped `test_geom_camber_cruise/000020.pt`, giving the first finite `test_avg/mae_surf_p` (91.50) on this branch. This fix is now on the advisor branch for all subsequent PRs.

3. **Throughput under 2×**: At 0.66 M params, the model is small — Python/I/O overhead is a non-trivial fraction of step time. Bigger models would amortize the autocast win more. The `~25%` gain is real but modest.

4. **Still improving at cap**: Val was 102.26 at epoch 19 (final) vs best 101.12 at epoch 17 — slight uptick at the last epoch, still trending overall. More compute budget would likely gain additional MAE points.

5. **`val_geom_camber_cruise` slight regression (+5 MAE pts)**: The only split that worsened. Possibly noise from the different convergence trajectory (more epochs = different phase of the schedule). Worth watching in follow-up runs.

### Conclusions

- bf16 AMP is now the baseline — it's merged and available for all subsequent PRs to inherit.
- The scoring-NaN workaround is now on advisor — new baseline for test_avg is 91.5013.
- New bar: any PR must beat **101.1212** on val_avg/mae_surf_p to merge.
- Next for thorfinn: compound the wins — pair bf16 with the best capacity lever once architecture results settle.

---

## 2026-05-12 20:00 — PR #1388: Linear warmup + lr 5e-4 → 1e-3 with cosine anneal — CLOSED

- **Branch:** `charliepai2g48h5-askeladd/warmup-lr-1e3`
- **Student:** charliepai2g48h5-askeladd
- **Hypothesis:** Add 5-epoch linear warmup and raise peak lr from 5e-4 to 1e-3
  (with cosine anneal afterward). Compensate for small batch and short
  wall-clock budget.

### Results

| Metric | lr=1e-3 (primary) | lr=7.5e-4 (fallback) |
|---|---:|---:|
| `val_avg/mae_surf_p` | **152.0332** | 152.5056 |
| `val_single_in_dist/mae_surf_p` | 184.95 | 177.17 |
| `val_geom_camber_rc/mae_surf_p` | 163.59 | 163.31 |
| `val_geom_camber_cruise/mae_surf_p` | 122.49 | 124.96 |
| `val_re_rand/mae_surf_p` | 137.10 | 144.58 |
| `test_avg/mae_surf_p` | NaN (no scoring workaround) | NaN |
| `test_3of4_avg/mae_surf_p` | 148.47 | 148.80 |
| Best epoch | 12 | 12 |
| Epochs reached | 14 | 14 |
| Time/epoch | 131.4 s | 132.0 s |
| Peak GPU | 42.11 GB | 42.12 GB |
| Loss used | **MSE** (PR predates Smooth-L1) | **MSE** |

- **Artifacts:** `models/model-charliepai2g48h5-askeladd-warmup-lr-1e3-20260512-181136/metrics.{jsonl,yaml}`, `models/model-charliepai2g48h5-askeladd-warmup-lr-7.5e4-20260512-185418/metrics.{jsonl,yaml}`
- **Status:** CLOSED — both arms ~41 MAE worse than baseline.

### Analysis

- ~41 MAE gap is too large to be MSE-vs-Smooth-L1 alone; lr=1e-3 is the
  dominant cause. The 5-epoch warmup + 9 epochs at peak lr=1e-3 + small
  cosine decay integrates LR-area-under-curve comparable to baseline's
  14 epochs at lr=5e-4, but more time at high lr overshoots good basins.
- Not divergence (loss curves were clean) — just a worse local minimum.
- Student independently rediscovered the scoring NaN bug, identical to
  thorfinn/alphonse's findings. Three independent students all found the
  same `0 × Inf = NaN` interaction — high-confidence diagnosis.
- The "step-based warmup over the first ~500 steps" idea is worth queuing
  separately, since 5 epochs = ~36% of the 14 epochs actually fitting in the
  cap.

### Conclusions

- lr=1e-3 with warmup is not productive at this wall-clock budget. The lr
  lever appears to be tuned correctly at baseline (lr=5e-4). Pushing lr
  higher (e.g., lr=1.5e-3, lr=2e-3) is not promising given the 41 MAE gap.
- More promising direction implied: step-based warmup at a *lower* peak.
  Queued for later, not assigned now.
- Next assignment for askeladd: gradient clipping max_norm=1.0 (PR #1561) —
  orthogonal to schedule lever space.

---

## 2026-05-12 19:53 — PR #1375: Raise surf_weight 10 → 30 — CLOSED

- **Branch:** `charliepai2g48h5-alphonse/surf-weight-30`
- **Student:** charliepai2g48h5-alphonse
- **Hypothesis:** Raise `surf_weight` from 10 to 30 to bias gradients more
  toward the ranking quantity (surface pressure MAE).

### Results

| Metric | Value |
|---|---:|
| `val_avg/mae_surf_p` | **120.3944** (epoch 13) |
| `val_single_in_dist/mae_surf_p` | 148.75 |
| `val_geom_camber_rc/mae_surf_p` | 125.45 |
| `val_geom_camber_cruise/mae_surf_p` | 93.73 |
| `val_re_rand/mae_surf_p` | 113.65 |
| `test_avg/mae_surf_p` | **112.6536** (finite — scoring workaround applied) |
| `test_single_in_dist/mae_surf_p` | 133.54 |
| `test_geom_camber_rc/mae_surf_p` | 123.03 |
| `test_geom_camber_cruise/mae_surf_p` | 79.73 |
| `test_re_rand/mae_surf_p` | 114.32 |
| Best epoch | 13 |
| Epochs reached | 14 |
| Time/epoch | 131.9 s |
| Peak GPU | 42.11 GB |
| Loss used | **MSE** (PR predates Smooth-L1) |

- **Artifacts:** `models/model-charliepai2g48h5-alphonse-surf-weight-30-20260512-191201/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (120.39 > 110.76).

### Analysis

- ~10 MAE gap to baseline. Smooth-L1 vs MSE typically buys ~5% in this
  regime — even a full recovery wouldn't close the gap.
- Per-split signal is diagnostic: `val_single_in_dist` got *worse* under
  surf_weight=30 (148.75 vs baseline 135.16) — surface-heavy reweighting
  biased gradients away from the volume manifold, hurting the hardest split.
  This is not an MSE-vs-Smooth-L1 artifact.
- Student independently rediscovered the scoring NaN bug AND wrote a clean
  `train.py:evaluate_split` workaround — exactly the same workaround being
  rolled centrally via PR #1532 (thorfinn). All four test splits finite as
  a result.
- Student also surfaced the recurring "T_max=50 cosine never decays in 14
  epochs" observation that tanjiro/askeladd also raised.

### Conclusions

- `surf_weight=30` is not productive — biases away from volume manifold.
  The baseline at `surf_weight=10` is well-tuned.
- Next assignment for alphonse: T_max=14 cosine schedule matched to actual
  epoch budget (PR #1560) — exactly the lever the student's own analysis
  pointed at, and orthogonal to all in-flight work.

---

## 2026-05-12 19:27 — PR #1439: Double batch_size 4 → 8 — CLOSED

- **Branch:** `charliepai2g48h5-tanjiro/batch-size-8`
- **Student:** charliepai2g48h5-tanjiro
- **Hypothesis:** Raise effective batch size from 4 → 8 to lower gradient
  variance under the 30-min wall-clock cap.

### Results

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 155.504 (epoch 14) |
| `val_single_in_dist/mae_surf_p` | 256.30 |
| `val_geom_camber_rc/mae_surf_p` | 145.07 |
| `val_geom_camber_cruise/mae_surf_p` | 103.11 |
| `val_re_rand/mae_surf_p` | 117.55 |
| `test_avg/mae_surf_p` | NaN (round-5 scoring bug) |
| Mean test_mae_surf_p (3 splits, excl. cruise) | 155.71 |
| Peak GPU | **84.2 GB** of 96 (no OOM) |
| Time/epoch | ~130 s |
| Epochs/30 min | 14 |
| Loss used | **MSE** (PR predates the Smooth-L1 merge) |

- **Artifacts:** `models/model-charliepai2g48h5-tanjiro-batch-size-8-20260512-185115/metrics.{jsonl,yaml}`
- **Status:** CLOSED — does not beat baseline (155.504 > 110.76).

### Analysis

- The comparison is unfair to the hypothesis: tanjiro's branch was created
  before #1444 merged Smooth-L1, so this run is MSE+batch=8 vs the current
  Smooth-L1+batch=4 baseline.
- However, the student's own analysis is decisive: **wall-clock is the binding
  constraint, not gradient noise**. Doubling batch trades step count 2:1 for
  variance reduction, but PR #1444 was monotonically improving at batch=4 —
  variance is not the bottleneck. Batch=8 just means fewer training epochs in
  the same 30-min window.
- batch=8 sits at 84 GB peak — no more headroom on this model size, so
  batch=8 is at its memory ceiling on the default Transolver. The lever is
  fully exercised.
- The student independently rediscovered the scoring NaN bug (same root
  cause as PR #1444) — solid debugging.

### Conclusions

- `batch_size=8` is feasible but does not appear to be a productive lever on
  this dataset + model + wall-clock budget. Closing the arm.
- The student's observation that "T_max=50 cosine never gets used because we
  only reach ~14 epochs" is a separately valuable insight — worth a future PR
  matching `T_max` to expected actual epoch budget.
- Next assignment for tanjiro: EMA model weights for eval (PR #1535) —
  orthogonal to the throughput / schedule lever space.

---

## 2026-05-12 18:58 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Branch:** `charliepai2g48h5-thorfinn/smooth-l1-loss`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Replace squared-error loss with Smooth-L1 (Huber, β=1.0) in
  normalized space for both training and evaluation losses. The ranking metric is
  MAE in original space; MSE in normalized space over-weights extreme high-Re
  samples. Smooth-L1 is linear outside |err|>β, providing bounded gradients.
  Both vol_loss and surf_loss use the same substitution; `surf_weight=10.0` and
  `data/scoring.py` MAE unchanged.

### Results

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p |
|---|---:|---:|---:|---:|
| `single_in_dist` | 135.16 | 1.719 | 0.769 | 120.38 |
| `geom_camber_rc` | 129.08 | 2.104 | 0.988 | 119.47 |
| `geom_camber_cruise` | 77.70 | 1.047 | 0.555 | NaN (bug) |
| `re_rand` | 101.10 | 1.607 | 0.740 | 97.36 |
| **avg** | **110.76** | — | — | NaN / 112.40 (3-split) |

- **Best epoch:** 14 of 50 configured (wall-clock-bound; monotonically improving)
- **Epochs/30-min:** ~14 at default model size (~131 s/epoch)
- **Peak GPU:** 42.1 GB (Blackwell RTX PRO 6000)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.{jsonl,yaml}`
- **Status:** MERGED → round-5 baseline floor

### Analysis

This is the first terminal result on the round-5 branch, so we cannot yet compare
against an MSE baseline on the same branch. The absolute val_avg = 110.76 sets
the floor. Key observations:

1. **Under-convergence.** The run was strictly monotonically improving at epoch 14
   when the 30-min cap hit (~14 epochs in 30 min for n_hidden=128). The floor is
   a loose lower bound on what the model could achieve with more compute.
2. **Split pattern consistent with hypothesis.** `val_geom_camber_cruise` (77.70)
   and `val_re_rand` (101.10) — the two splits the PR predicted would benefit most
   from bounded gradients at high-Re — are the best-performing splits. The raceCar
   splits (`single_in_dist` 135.16, `geom_camber_rc` 129.08) are noisier
   epoch-to-epoch, consistent with the loss being driven by the wide-Re tail.
3. **Scoring NaN bug discovered.** `test_geom_camber_cruise/000020.pt` has ±Inf
   values in the `p` channel. The `data/scoring.py` sample-skip logic misses this
   due to `0 × Inf = NaN` (IEEE-754). This affects all PRs in round 5 that run
   the test step. Round-5 ranking decision: **val_avg/mae_surf_p only**. The fix
   (filter the bad sample in `train.py`'s `evaluate_split` before calling
   `accumulate_batch`) will be rolled into an upcoming student assignment.

### Conclusions

- Smooth-L1 is a viable baseline for round 5. Whether it beats MSE requires the
  other in-flight arms (which use MSE) to finish and post results.
- The binding constraint is wall-clock convergence speed: ~14 epochs in 30 min.
  The highest-leverage next move is anything that increases epochs/wall-clock
  (bf16 AMP, smaller batch, smaller model, compile) rather than per-epoch quality.
- `val_geom_camber_cruise` is the easiest split (lowest MAE). The hardest splits
  are the raceCar ones.
