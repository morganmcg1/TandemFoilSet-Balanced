# SENPAI Research Results — willow-pai2e-r5

---

## 2026-04-29 06:50 — Methodological note: seed variance is large

Discovered via askeladd's #1031 sweep: **δ=0.1 single-seed numbers vary by 3.2pt val / 5.8pt test** between independent runs. Specifically:

- #885 single-seed (run `nffbil1x`): val=96.866 / test=87.348
- #1031 seed2 same config (run `o6liykzm`): val=93.624 / test=81.589

This invalidates many small-margin "win" claims in past PRs. From now on:

- Single-seed wins under **5pt test_avg** are not decisive — require a second seed.
- Larger wins (e.g. #986's 24-pt drop) are robust to seed noise.
- Past close-call merges (#850, #885) likely benefitted from lucky seeds; their true mean is somewhat higher than recorded. Future winners are easier to demonstrate but need to clear a higher bar.
- This affects how we interpret the askeladd #1031 and edward #1019 results below — both showed gains at the seed-noise boundary on the OBSOLETE eager+δ=0.1 baseline, so verification on the new compile baseline is needed.

---

## 2026-04-28 19:51 — PR #732: Scale Transolver to n_hidden=256, n_layers=7, n_head=8

- **Branch:** `willowpai2e5-alphonse/larger-model-capacity` (closed)
- **W&B run:** `pkyat9dy` — group `larger-model-capacity`
- **Hypothesis:** Scaling Transolver from 0.93M to 3.01M params (n_hidden=256, n_layers=6, n_head=8 after n_layers=7 OOM fallback) improves fitting and OOD generalization.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist`      | 184.36 | 169.38 |
| `val_geom_camber_rc`      | 162.75 | 152.78 |
| `val_geom_camber_cruise`  | 130.79 | **NaN** |
| `val_re_rand`             | 141.88 | 147.91 |
| **avg**                   | **154.95** | **NaN** |

Best epoch = 6 (final epoch before 30-min timeout). Training trajectory (val_avg): 245.66 → 210.36 → 202.81 → 184.30 → 184.74 → 154.95.

### Commentary & Conclusions

- **Non-conclusive under our compute budget.** The 30-min SENPAI_TIMEOUT_MINUTES cap allowed only 6 of 50 scheduled epochs. The CosineAnnealingLR (T_max=50) was running at ~94% initial LR — effectively zero LR decay. Trajectory shows rapid, unfinished descent; we don't know where the model would converge.
- **5.2× throughput penalty.** Each epoch took ~5.2 min vs ~1.5 min baseline. A matched-wall-clock baseline (128/5/4) would do ~20 epochs; this 3.01M-param model got 6.
- **NaN on `test_geom_camber_cruise/mae_surf_p`.** The undertrained large model produced non-finite pressure predictions on at least one hard test sample. Val numbers are finite because no val sample was numerically unstable. This is a model-stability issue, not a data issue.
- **Bug identified in `data/scoring.py`.** `accumulate_batch` skips samples with non-finite ground truth but not non-finite predictions — a single NaN prediction node poisons the entire split's channel sum. data/ is read-only for student PRs; flagging for follow-up.
- **Reference data point recorded:** val_avg/mae_surf_p = 154.95 (first run; model ~1/3 trained). Will compare against in-flight Wave-1 runs once they complete.
- **Decision:** Closed (premature — NaN test metric, no matched-budget baseline comparison, 5.2× throughput cost unjustified without fair comparison).

### Next step for this direction

Revisit capacity scaling after (1) in-flight Wave-1 runs give us a baseline-architecture val number, (2) BF16 mixed-precision is available to reduce per-epoch time, and (3) gradient accumulation is verified to allow n_layers=7. Alphonse reassigned to `film-re-conditioning` (PR #796).

---

## 2026-04-28 20:15 — PR #763: Add physics-informed distance features to Transolver input

- **Branch:** `willowpai2e5-thorfinn/feature-engineering` (merged)
- **W&B run:** `072wo9xb` — group `feature-engineering`
- **Hypothesis:** Appending dist_to_surface, log(1+dist_to_surface), and is_tandem as derived node features provides physically grounded inductive bias for boundary-layer gradients and tandem vs. single-foil regime.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 177.0157 | 148.3098 |
| `val_geom_camber_rc` | 157.4591 | 145.5500 |
| `val_geom_camber_cruise` | 105.7864 | 91.0172 |
| `val_re_rand` | 125.4113 | 121.3623 |
| **avg** | **141.4181** | **126.5598** |

Best epoch = 12 of 50 (30-min timeout). AUG_X_DIM=27, +768 params.

### Commentary

- **First PR with clean test_avg across all 4 splits.** Student correctly diagnosed the `test_geom_camber_cruise` NaN as a dataset-side issue (sample 20 has 761 NaN y[:, 2] entries) and implemented a NaN-safe `evaluate_split` workaround in `train.py` since `data/` is read-only. Critical fix — all future runs now report finite test_avg.
- **Decision: Merged.** Beats #732 reference (154.95), produces clean test metrics. Features are physically motivated and low-risk.
- Became new baseline at val_avg=141.42 before fern's #737 landed.

---

## 2026-04-28 20:25 — PR #737: Add 5-epoch linear warmup + peak lr=1e-3 before cosine decay

- **Branch:** `willowpai2e5-fern/lr-warmup-cosine` (merged)
- **W&B run:** `5b22tecz` — group `lr-warmup-cosine`
- **Hypothesis:** Linear LR warmup (1e-4 → 1e-3, 5 epochs) followed by cosine decay to eta_min=1e-6 stabilises early training of slice-attention projections and allows a higher peak LR without instability.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 149.241 | 126.021 |
| `val_geom_camber_rc` | 146.033 | 129.155 |
| `val_geom_camber_cruise` | 96.362 | NaN (pre-fix; ~91 expected) |
| `val_re_rand` | 119.852 | 115.590 |
| **avg** | **127.872** | NaN (3-split avg ≈ 123.59) |

Best epoch = 14 of 50 (30-min timeout). LR at epoch 14 ≈ 8e-4 (barely into cosine decay).

### Commentary

- **Decision: Merged (current best).** val_avg=127.87 beats new baseline of 141.42 after #763 merged. Schedule change is orthogonal to features change — both merged cleanly.
- **Critical gap identified:** T_max=50 with a 30-min (~14 epoch) budget means the LR never decays properly. Model was still at 80% of peak LR when cut off, trajectory still steeply descending (136→128 at epochs 13→14). Budget-matched re-run (epochs=14, warmup=2) assigned to fern as PR #809.
- **test_avg is NaN** because #737 ran BEFORE #763's NaN fix was in the merged branch. Future runs will be clean.

---

## 2026-04-28 20:30 — PR #733: Increase slice_num from 64 to 256 for richer physics decomposition

- **Branch:** `willowpai2e5-askeladd/more-slices` (closed)
- **W&B run:** `8l3pbq6x` — group `more-slices`
- **Hypothesis:** Quadrupling slice tokens (64→256) gives the model finer-grained physics decomposition for boundary layer / wake / freestream separation.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 185.28 | 161.17 |
| `val_geom_camber_rc` | 179.72 | 160.89 |
| `val_geom_camber_cruise` | 111.53 | NaN (pre-fix) |
| `val_re_rand` | 129.46 | 130.14 |
| **avg** | **151.50** | NaN (3-split avg ≈ 150.73) |

Best epoch = 8 of 50 (30-min timeout). slice_num=256 fits in 82.3 GB VRAM. Per-epoch wall-time: 4.2 min vs 2.2 min for slice_num=64.

### Commentary

- **Decision: Closed (regression).** val_avg=151.50 vs current baseline 127.87 = 18.5% regression, well above the 5% close threshold.
- **Throughput penalty is structural, not tunable.** ~2× per-epoch cost meant 8 epochs vs ~14 for baseline-config runs in the same 30-min cap. Val curve still steeply falling at cutoff (161→151 epochs 7→8), so even granting "256 needs more time," it's strictly worse for our compute budget. Would need a budget extension or a separate throughput improvement to revisit.
- Student also flagged the same `test_geom_camber_cruise` NaN issue thorfinn diagnosed in #763 — but that's already fixed in the current merged baseline. Confirms the bug is global (not slice-num-specific).
- Askeladd reassigned to `mixed-precision-bf16` (PR #811) — directly attacks the throughput constraint this PR exposed.

---

## 2026-04-28 20:45 — PR #745: Separate output heads for velocity (Ux/Uy) and pressure (p) — sent back

- **Branch:** `willowpai2e5-tanjiro/separate-pressure-head` (sent back for rebase + Option 3)
- **W&B runs:** `m5ydsa1t` (split_linear), `7aw36w9e` (split_mlp) — group `separate-pressure-head`
- **Hypothesis:** Splitting the final output projection into two specialized MLPs (Ux/Uy and p) gives the model architectural capacity for pressure-specific feature detectors and directly improves `val_avg/mae_surf_p`.

### Results (against PRE-merge code: no features, no warmup)

| Option | head params | val_avg/mae_surf_p ↓ | test_avg/mae_surf_p ↓ |
|--------|-------------|----------------------|-----------------------|
| Option 1 — `split_linear` (two `Linear`) | 387 | **130.82** | **118.99** |
| Option 2 — `split_mlp` (deeper, capacity-matched) | 16,707 | 134.46 | 123.79 |

Per-split surf_p MAE for Option 1 (winner):

| Split | val | test |
|-------|------|------|
| `val_single_in_dist` | 155.53 | 137.41 |
| `val_geom_camber_rc` | 141.41 | 133.08 |
| `val_geom_camber_cruise` | 104.96 | 86.68 |
| `val_re_rand` | 121.37 | 118.81 |

Best epoch = 12 of 50 (30-min timeout at epoch 14). Both runs hit timeout.

### Commentary & Conclusions

- **Comparison NOT fair against current baseline.** Student's run was on pre-merge code (no distance features from #763, no warmup+cosine from #737). Direct val_avg=130.82 looks like a 2.3% regression vs 127.87, but that's apples-to-oranges. Against the no-features/no-warmup reference points (alphonse #732 154.95, askeladd #733 151.50), 130.82 is a substantial improvement — head split appears to be a real signal.
- **Option 1 winning is partially a smaller-head effect, not pure specialization.** Student correctly identified that the baseline's `mlp2` already had ~16.9k params; their "Option 1" actually shrinks the head 44× to 387 params. So Option 1 winning at 14 epochs may be undertraining favoring smaller-capacity heads, not pressure-specialization succeeding.
- **Critical follow-up: Option 3 (capacity-matched split).** Student proposed `Linear(hidden,hidden)→GELU` shared first layer, then forked `Linear(hidden,2)` and `Linear(hidden,1)`. This isolates specialization from capacity. **Sent back to test this on the rebased baseline.**
- **NaN bug.** Student found and patched the same `nan*0=nan` propagation bug thorfinn diagnosed in #763 (data/scoring.py masking via multiplication poisons the running sum when ground truth has NaN). Their patch was in train.py:evaluate_split, same approach as #763 (already merged). Duplicate work but confirms the fix is correct. Student also patched the W&B run summaries via wandb.Api() to retrofit clean test numbers.
- **Decision: Sent back.** Rebase onto current baseline, run Option 3 only. If <127.872, merge.

---

## 2026-04-28 21:29 — PR #811: Enable bf16 mixed precision for 1.5-2x training throughput

- **Branch:** `willowpai2e5-askeladd/mixed-precision-bf16` (merged)
- **W&B run:** `newqt8dd` — group `mixed-precision-bf16`
- **Hypothesis:** BF16 autocast on forward+loss yields 1.5-2× per-epoch speedup → more epochs in the 30-min budget → directly improves val_avg/mae_surf_p. BF16 preferred over FP16 for this dataset's ±29K pressure range (bf16 has the same 8-bit exponent as fp32, no overflow).

### Results

| Metric | Baseline (fp32, #737) | bf16 (#811) | Δ |
|---|---|---|---|
| Wall-clock / epoch | 131.96 s | **110.02 s** | 1.20× speedup |
| Epochs in 30 min | 14 | **17** | +3 (+21%) |
| Peak VRAM | — | **33.1 GB** | 63 GB headroom on 96 GB |
| **val_avg/mae_surf_p** | 127.872 | **127.402** | **−0.47** |
| **test_avg/mae_surf_p** | NaN (pre-fix) | **116.211** | first clean 4-split test |
| NaN/Inf events | — | **0** | numerically stable |

Per-split:

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist`     | 151.791 | 141.142 |
| `val_geom_camber_rc`     | 147.898 | 134.121 |
| `val_geom_camber_cruise` | **93.729** | **79.094** |
| `val_re_rand`            | **116.189** | 110.488 |
| **avg**                  | **127.402** | **116.211** |

Best epoch = 17 of 50. Gradient norm mean=69.6, max=1224.9 — isolated spike, no explosion pattern.

### Commentary & Conclusions

- **Decision: Merged.** val_avg=127.402 beats 127.872 baseline; test_avg=116.211 is a clean 4-split number, 10 points better than the last clean test (#763, 126.56). This is a platform improvement: all subsequent experiments inherit the BF16 speedup and VRAM headroom.
- **Speedup was 1.20× not 1.5-2×.** Gap explained: (1) model is small (663K params), matmul share of total step time is bounded; (2) `add_derived_features` runs a Python for-loop with `.item()` GPU→CPU sync — pure non-matmul cost untouched by autocast; (3) LayerNorm kept in fp32 by PyTorch autocast default. These three factors cap the matmul-side gain.
- **Key structural finding: 33.1 GB VRAM with batch_size=4.** 63 GB headroom unlocked for batch scaling. Batch_size=8–16 now plausible without OOM risk. This is the highest-leverage immediate follow-up.
- **bf16 numerically clean.** Zero NaN/Inf events across 17 epochs / 6,381 steps. bf16's 8-bit exponent handles ±29K pressure range without issue; no GradScaler, no fp32 loss-cast needed.
- **Next bottleneck: `add_derived_features` Python loop.** With matmul now faster, the non-matmul Python distance loop with `.item()` sync is the new dominant cost. Vectorizing it should push speedup closer to 1.5-2×.
- **Askeladd reassigned** to batch_size scaling (leveraging the 63 GB headroom directly).

---

## 2026-04-28 21:30 — PR #734: Increase surf_weight from 10 to 50/100 to directly target surface pressure MAE — closed

- **Branch:** `willowpai2e5-edward/higher-surf-weight` (closed)
- **W&B runs:** `zi70jnyl` (sw=10 control), `srbt5u57` (sw=50), `24dctht8` (sw=100) — group `higher-surf-weight`
- **Hypothesis:** Increasing surf_weight from 10 → 50/100 forces the model to prioritize fitting surface pressure during training, directly improving val_avg/mae_surf_p.

### Results (against PRE-merge code)

| surf_weight | val_avg/mae_surf_p ↓ | best epoch | Δ vs sw=10 |
|-------------|----------------------|------------|------------|
| **10 (control)** | **130.43** | 13 | — |
| 50 | 135.56 | 14 | +3.9% (worse) |
| 100 | 136.72 | 12 | +4.8% (worse) |

Per-split val/mae_surf_p:

| Split | sw=10 | sw=50 | sw=100 |
|-------|-------|-------|--------|
| `val_single_in_dist` | **157.32** | 178.62 (+13.5%) | 158.34 (+0.6%) |
| `val_geom_camber_rc` | **139.59** | 139.07 (-0.4%) | 148.17 (+6.1%) |
| `val_geom_camber_cruise` | **106.21** | 106.37 (+0.2%) | 108.81 (+2.4%) |
| `val_re_rand` | **118.62** | 118.20 (-0.4%) | 131.57 (+10.9%) |

All three runs hit 30-min timeout; peak VRAM 42.1 GB.

### Commentary & Conclusions

- **Decision: Closed (clean negative result).** Both up-direction values regress; no variation in the up direction is likely to flip this.
- **Key mechanistic insight from the student:** "Volume residuals provide spatial context that the Transolver attention uses for surface predictions — they're not just regularization, they're informative." This explains why naively up-weighting surface fails — removing volume context starves the surface prediction of spatial information. The +13.5% single_in_dist regression at sw=50 is the cleanest demonstration: single-foil has highest pressure amplitude and most depends on volume context.
- **The current `surf_weight=10` may already be past the optimum.** Surface nodes are ~1% of total but get 10× weight per node, ≈10% effective contribution to the gradient. Pushing further hits diminishing/negative returns.
- **Edward reassigned** to lower-surf-weight sweep ({3, 5, 7}) on the merged baseline — direct test of the counter-hypothesis the student's analysis raises.
- **Bug observation:** Student also flagged the `test_geom_camber_cruise` NaN issue. Already fixed in merged baseline (#763 NaN-safe eval).

---

## 2026-04-28 22:26 — PR #742: Add dropout=0.1 to MLP sublayers to reduce OOD overfitting — closed

- **Branch:** `willowpai2e5-nezuko/dropout-regularization` (closed)
- **W&B runs:** `55aolphx` (no dropout control), `g81o4brf` (dropout=0.1) — group `dropout-regularization`
- **Hypothesis:** Dropout=0.1 in MLP sublayers reduces overfitting to training geometry/Re combinations and improves OOD generalization on `val_geom_camber_rc/cruise` and `val_re_rand`.

### Results (against PRE-merge code; no BF16, no warmup)

| mlp_dropout | val_avg/mae_surf_p ↓ | best epoch | Δ |
|-------------|----------------------|------------|------|
| **0.0 (control)** | **123.37** | 14 (= last) | — |
| 0.1 | 138.68 | 14 (= last) | +12.4% (worse) |

Per-split val/mae_surf_p:

| Split | dropout=0.0 | dropout=0.1 | Δ |
|-------|-------------|-------------|------|
| `val_single_in_dist` | 148.11 | 162.06 | +9.4% |
| `val_geom_camber_rc` | 123.92 | 149.62 | **+20.7%** |
| `val_geom_camber_cruise` | 99.57 | 113.49 | +14.0% |
| `val_re_rand` | 121.89 | 129.54 | +6.3% |

Both runs hit 30-min timeout at epoch 14/50; peak VRAM 42-43 GB (no BF16 in this run).

### Commentary & Conclusions

- **Decision: Closed (clean negative result with excellent root-cause analysis).**
- **Critical mechanistic insight from the student:** "Both runs stopped at epoch 14/50 and best_epoch=14=last trained epoch in both cases. That's a clear signal of an *under*-trained model — there is no overfitting to regularize. Dropout's only effect is to inject noise that slows convergence."
- **OOD-hits-hardest signature confirms under-training, not overfitting.** If dropout were correctly closing a generalization gap, ID would suffer most and OOD least; we see the opposite (rc +20.7% > id +9.4%). This is the fingerprint of "fewer effective gradient updates per parameter."
- **Implementation verified:** dropout in standard transformer-FFN location (between GELU and linear), model.eval() correctly called for val and test, attention dropout untouched. Negative result is not from a bug.
- **Implication for regularization more broadly:** Until the model demonstrates overfitting (best_epoch < final_epoch by a wide margin), traditional regularizers (dropout, weight decay) have no benefit to provide. Schedule fix (#809) or batch-size scaling (#848) may unlock convergence first; only then does regularization become testable.
- **Nezuko reassigned** to DropPath/stochastic depth — student's suggestion #3, different mechanism (drops entire residual branches, model-averaging interpretation, compute-efficient).

---

## 2026-04-28 22:44 — PR #810: Add EMA weight averaging for lower-variance OOD checkpointing — sent back

- **Branch:** `willowpai2e5-thorfinn/ema-model-checkpoint` (sent back for post-warmup EMA + decay sweep)
- **W&B run:** `g2yfau61` — group `ema-model-checkpoint`
- **Hypothesis:** EMA decay=0.999 of model weights smooths over noisy gradient updates, especially helping OOD val splits. Validate and checkpoint with EMA shadow weights.

### Results (against PRE-merge code; no BF16)

| Split | EMA (this run) | Baseline #737 | Δ |
|-------|----------------|---------------|------|
| `val_single_in_dist` | 170.472 | 149.241 | +21.23 |
| `val_geom_camber_rc` | 149.418 | 146.033 | +3.39 |
| `val_geom_camber_cruise` | 106.842 | 96.362 | +10.48 |
| `val_re_rand` | 121.037 | 119.852 | +1.19 |
| **val_avg/mae_surf_p** | **136.942** | **127.872** | **+9.07 (+7.1%)** |

| Test Split | EMA (this run) | #763 baseline |
|-----------|----------------|---------------|
| `test_single_in_dist` | 147.616 | 148.310 |
| `test_geom_camber_rc` | 132.479 | 145.550 |
| `test_geom_camber_cruise` | 88.305 | 91.017 |
| `test_re_rand` | 120.607 | 121.362 |
| **test_avg/mae_surf_p** | **122.252** | **126.560** |

Best epoch = 13/50 (30-min timeout); peak VRAM 42.2 GB (no BF16 in this run).

### Commentary & Conclusions

- **Decision: Sent back (despite >5% val regression).** Strict close criterion would apply, but the student's diagnostic is so clear and the fix so simple that one more iteration is high-value.
- **The mechanism appears to work — the timing is wrong.** Trajectory was monotonically improving; test_avg actually IMPROVED by 4.3 points relative to the last clean 4-split test (#763 → #810: 126.56 → 122.25). The EMA shadow lags the live model on val, but the smoothing effect on test geometry held up.
- **Critical diagnostic from the student (worth recording):** With decay=0.999 and warmup_epochs=5, the shadow has effective memory ~1000 steps (~2.7 epochs). Of that, ~32% of the shadow mass at epoch 13 still sits on warmup-era weights (lr=1e-4 regime where the model is barely learning). The shadow is contaminated by warmup gibberish.
- **Send-back instructions:** (1) defer EMA initialization until after warmup completes; (2) sweep decay ∈ {0.999, 0.995}; (3) rebase onto BF16 baseline (17 epochs instead of 14 = more post-warmup steps to amortize); (4) log both EMA and live val_avg per epoch to verify the gap closes.
- **Implementation correctness verified:** deepcopy isolation, no live-model overwrite, EMA state correctly checkpointed.

---

## 2026-04-28 22:58 — PR #848: Scale batch_size 4→8/10/12 to use BF16 VRAM headroom — closed

- **Branch:** `askeladd/larger-batch-size` (closed)
- **W&B runs:** `0zt9fppw` (bs=8), `qt40qg9s` (bs=10), `l829jyzw` (bs=12 OOM); baseline `newqt8dd` (bs=4) — group `larger-batch-size`
- **Hypothesis:** With BF16 freeing 63 GB VRAM, larger batches (8/12) increase samples/step + smoother gradients → faster convergence and better OOD generalization.

### Results (against merged BF16 baseline #811)

| batch_size | val_avg/mae_surf_p ↓ | test_avg/mae_surf_p ↓ | best epoch | s/epoch | Peak VRAM | Δ val |
|------------|---------------------:|----------------------:|-----------:|--------:|----------:|------:|
| **4 (baseline)** | **127.40** | **116.21** | 17 | 108.7 | 33.1 GB | — |
| 8 | 142.51 | 131.81 | 12 | 117.1 | 66.1 GB | +11.9% (worse) |
| 10 | 147.23 | 134.43 | 15 | 119.9 | 82.6 GB | +15.6% (worse) |
| 12 | OOM | — | — | — | >95 GB | failed (epoch 1) |

### Commentary & Conclusions

- **Decision: Closed (clean negative result with strong mechanistic analysis).**
- **Both halves of the hypothesis flipped or failed:**
  1. Per-sample throughput *regressed* from 13.79 → 12.41 samples/sec (bs=4→10) — the `add_derived_features` Python loop with per-sample CPU sync (`mask[b].sum().item()`) plus chunked pairwise distance is now the dominant cost at high B. **Vectorizing this is a separate throughput-engineering target.**
  2. Same LR + larger batch → drastically fewer optimizer steps under 30-min wall clock (bs=10 hits only 35% of baseline's 6,375 steps). Linear LR scaling was deferred — and is almost certainly required.
- **Gradient noise smoothing did hold:** max grad norm collapsed from 1225 (bs=4 baseline, one >1000 spike) to 572 (bs=8) to 0 spikes at higher B. The gradient signal *is* richer per step; we just take fewer of those steps.
- **Per-split mixing pattern at bs=10:** `val_geom_camber_rc` improved (147.90 → 136.53) while `val_single_in_dist` collapsed (151.79 → 209.05). Larger batches reshuffle which split is favored — likely a sampler/domain-balance interaction.
- **Real bottleneck identified:** `add_derived_features` Python loop in `train.py:79-98`. Removing the CPU-sync `.item()` calls and vectorizing the chunked pairwise distance is the unblocking lever before batch_size scaling becomes useful.
- **Askeladd reassigned** to Huber-delta sweep (#885) — the validated signal from #739 deserves a clean delta optimum.

---

## 2026-04-28 23:04 — PR #796: FiLM-condition TransolverBlocks on log(Re) — closed

- **Branch:** `willowpai2e5-alphonse/film-re-conditioning` (closed)
- **W&B runs:** `2pqqhqo6` (no FiLM control), `b6qc62sq` (FiLM log-Re); pre-fix runs `2nn5hiyh` and `23tdi1dg` produced NaN test metrics — group `film-re-conditioning`
- **Hypothesis:** FiLM(log Re) per-block conditioning gives the model an explicit Reynolds-regime signal so cross-Re generalization improves on `val_re_rand` (predicted −5 to −15%).

### Results — paired comparison (both runs hit 30-min wall clock at epoch 13/50)

| Split (lower better) | Baseline (no FiLM) | FiLM (log Re) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist`/`mae_surf_p` | 169.39 | 159.84 | −5.6% |
| `val_geom_camber_rc`/`mae_surf_p` | 143.55 | 153.87 | +7.2% |
| `val_geom_camber_cruise`/`mae_surf_p` | 106.06 | 106.97 | +0.9% |
| `val_re_rand`/`mae_surf_p` | 122.42 | 121.38 | **−0.9%** (predicted −5 to −15%) |
| **`val_avg/mae_surf_p`** | **135.35** | **135.51** | **+0.1% (tied)** |
| `test_avg/mae_surf_p` | 119.71 | 125.08 | **+4.5% (worse)** |

### FiLM diagnostics at end of run (FiLM run)

| Metric | Value | Interpretation |
|---|---:|---|
| `film/last_weight_l2` | 18.58 | grew from 0 — net is actively learning |
| `film/gamma_dev_mean` | 0.26 | gamma drifting ~26% off identity |
| `film/beta_abs_mean` | 0.23 | non-trivial bias shift |

### Commentary & Conclusions

- **Decision: Closed (clean negative on the primary diagnostic).**
- **The hypothesis's predicted mechanism didn't materialize.** FiLM is *alive* (non-trivial gamma/beta on log-Re sweep, weights grew from zero) but the val_re_rand improvement is essentially noise (−0.9% vs predicted −5 to −15%). The model is *not* using log-Re modulation to close the cross-Re gap — log-Re is already in the input feature vector and the attention is plausibly extracting it adequately without FiLM-style explicit conditioning.
- **Test_avg regressed by 4.5%.** This combined with the val tie is a strong signal that the conditioning is adding noise without a generalization payoff at this training budget.
- **Implementation deviations from spec — all justified:**
  - Used `x[:, 0, 13:14]` not `x[:, :, 13].mean(...)` because pad_collate right-pads with zeros (mean would mix in padding).
  - Added `--use_film` flag for clean paired comparison.
  - Re-zeroed FiLM final layer after `_init_weights` (otherwise trunc_normal overrides identity init).
- **Bug fix flagged:** Student also identified the `nan*0=nan` issue in `data/scoring.py` (separately fixed in train.py since `data/` is read-only). This is a known issue from #763.
- **Alphonse reassigned** to per-sample y-normalization (#896) — direct attack on the high-Re-dominates-loss issue that FiLM was hoping to fix indirectly.

---

## 2026-04-28 23:30 — PR #739: Replace MSE with Huber loss (delta=1.0) — sent back for rebase

- **Branch:** `willowpai2e5-frieren/huber-loss` (sent back; major win; needs rebase onto BF16 baseline)
- **W&B run:** `z2a34zbu` (`willowpai2e5-frieren/huber-loss-d1.0`) — group `huber-loss`
- **Hypothesis:** Huber loss with delta=1.0 caps influence of high-Re outlier samples (per-sample y_std varies 10×), reducing gradient noise and improving OOD generalization.

### Results (against PRE-merge code — no BF16, no warmup; comparable baseline = #763 features-only val_avg=141.42)

| Split | Huber (best epoch 14, last epoch) | BF16 baseline #811 (best epoch 17) | Δ vs BF16 baseline |
|-------|-----------------------------------:|-----------------------------------:|-------------------:|
| `val_single_in_dist` | 125.13 | 151.79 | **−17.6%** |
| `val_geom_camber_rc` | 110.93 | 147.90 | **−25.0%** |
| `val_geom_camber_cruise` | 79.69 | 93.73 | **−15.0%** |
| `val_re_rand` | 99.79 | 116.19 | **−14.1%** |
| **val_avg/mae_surf_p** | **103.89** | **127.40** | **−18.5%** |

| Test split | Huber (epoch 14) | BF16 baseline #811 | Δ |
|-----------|----------------:|-------------------:|------:|
| `test_single_in_dist` | 108.45 | 141.14 | −23.2% |
| `test_geom_camber_rc` | 102.57 | 134.12 | −23.5% |
| `test_geom_camber_cruise` | NaN (data bug repro on pre-merge code) | 79.09 | — |
| `test_re_rand` | 96.78 | 110.49 | −12.4% |
| `test_avg/mae_surf_p` | NaN (cruise) — 3-split avg = 102.60 | 116.21 | −20.2% on 3-split |

Best epoch 14 = LAST epoch (timeout cut-off); val_avg trajectory `224.75 → 262.81 → 177.70 → 179.66 → 146.75 → 198.43 → 160.12 → 125.46 → 137.59 → 132.87 → 162.95 → 129.82 → 124.26 → 103.89` — **steepest descent in the final 2 epochs (124.26 → 103.89, −16.4% in one epoch). 103.89 is a lower bound on Huber's potential.**

### Commentary & Conclusions

- **Decision: Sent back for rebase onto BF16 baseline.** Result is a major win (−18.5% val_avg) but is on pre-merge code. We need the result on the merged baseline (BF16 + features + warmup) to get the actual stack effect, and the BF16 platform gives 1.20× more epochs in the same wall clock — this should make the result *better*, not worse.
- **The 4-split improvement is consistent and strong.** All splits improve 14–25%. This is the largest single-PR improvement seen in the program so far. OOD splits (rc, cruise, re_rand) all improve by 14–25% — Huber is *not* just a single_in_dist trick.
- **Mechanism likely real:** with `surf_weight=10` and per-sample y_std varying 10×, MSE gradient is dominated by 1–2 high-Re samples per batch. Huber caps that contribution at delta=1.0, letting low-Re samples contribute usefully.
- **Run did not converge.** 14/50 epochs at 30-min cap; val curve was descending fast at cutoff. The post-rebase BF16 run gets ~17 epochs in same wall clock — 3 more steep-descent epochs available.
- **NaN test_geom_camber_cruise:** this is the known IEEE-754 padded-batch issue. Already fixed in BF16 baseline via NaN-safe eval (#763). The rebase will inherit the fix and produce a clean 4-split test_avg.
- **Validated independent investigation by student:** ran `batch_size=1` inference on cruise to confirm the NaN is a padding-related issue in `PhysicsAttention` (no node mask in attention), not a Huber problem.
- **Implication for #885 (askeladd Huber-delta-sweep):** delta=1.0 on pre-merge code already wins by 18.5%; sweep delta ∈ {0.3, 0.5, 1.0, 2.0} on BF16 baseline becomes the natural follow-up (already in flight as #885).
- **Send-back instructions to frieren:** (1) rebase onto current `icml-appendix-willow-pai2e-r5` (BF16 + features + warmup); (2) re-run with same `delta=1.0`; (3) confirm the −18.5% holds on the merged baseline.

---

## 2026-04-28 23:40 — PR #739 (rebased): Huber loss δ=1.0 on BF16 baseline — **MERGED**

- **Branch:** `willowpai2e5-frieren/huber-loss` (merged)
- **W&B run:** `l95azbnv` (`willowpai2e5-frieren/huber-loss-d1.0-rebased`) — group `huber-loss`
- **Hypothesis:** Same as pre-rebase #739 above; now on merged BF16 + features + warmup baseline.

### Results (rebased onto BF16 baseline #811; best epoch 16 of 17 completed)

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 130.87 | 124.544 |
| `val_geom_camber_rc` | 115.14 | 99.385 |
| `val_geom_camber_cruise` | 92.61 | 80.195 |
| `val_re_rand` | 103.76 | 101.070 |
| **avg** | **110.594** | **101.299** |

vs BF16 baseline #811 (val_avg=127.40, test_avg=116.21):
- val_avg: **−13.2%**; test_avg: **−12.8%**
- Per-split improvement: rc −22.1% / sid −13.8% / re_rand −10.7% / cruise −1.2% (val)
- All 4 test splits finite (NaN-safe eval inherited from baseline)
- Peak VRAM 33.1 GB (unchanged), 30.0 min wall clock

### Commentary & Conclusions

- **Decision: Merged. New baseline.** Four compounding wins: distance features (#763) + warmup+cosine (#737) + BF16 (#811) + Huber δ=1.0 (#739). New floor: val_avg=110.594, test_avg=101.299.
- **Mechanism validated.** Huber caps gradient contribution of high-Re outlier samples (per-sample y_std varies 10×, surf_weight=10 amplifies the imbalance). All 4 splits improve, OOD splits (rc −22.1%, re_rand −10.7%) benefit most.
- **Gain is slightly attenuated vs pre-rebase (−13.2% vs −18.5%).** This is consistent with run-to-run variance (~8% across 3 Huber runs). The direction and approximate magnitude robustly replicate. Also, the rebased run started from a harder baseline (127.40 vs 141.42 before warmup).
- **Still timeout-limited at epoch 16/17.** Val curve descending at cutoff (epoch 15→16: 125 → 110). Convergence floor not reached.
- **`val_geom_camber_cruise` stagnated.** Only −1.2% val (vs −15% pre-rebase on older baseline). cruise test slightly worse (+1.4%). Frieren diagnosed the likely cause: `PhysicsAttention` distributes softmax mass over padded positions — cruise has the most geometric diversity (most variable mesh sizes) → worst padding ratio per batch. **Assigned frieren to fix this (#915).**
- **Askeladd #885 now the right follow-up.** Sweep δ ∈ {0.3, 0.5, 1.0, 2.0} on top of this merged baseline.

---

## 2026-04-29 00:05 — PR #878: DropPath/stochastic depth on residual branches — closed

- **Branch:** `willowpai2e5-nezuko/drop-path` (closed)
- **W&B runs:** `zrpxz35j` (control p=0.0), `vixyda0y` (p=0.1) — group `drop-path`
- **Hypothesis:** DropPath at p_max=0.1 (linear schedule across 5 layers) provides implicit ensembling regularization, frees per-step compute (skipped residual branches), and shifts best_epoch later → better generalization on undertrained model.

### Results (against BF16 baseline `newqt8dd`, before Huber merge)

| drop_path | val_avg/mae_surf_p (best) | test_avg/mae_surf_p | best_epoch | s/epoch | epochs in 30 min |
|-----------|--------------------------:|--------------------:|-----------:|--------:|-----------------:|
| **0.0 (control)** | **131.89** | **121.64** | 16 | 110.0 | **17** |
| 0.1 | 132.22 | 122.19 | 12 | 113.2 | 16 |

(both runs worse than current Huber baseline 110.594, but the comparison is internal to this PR)

### Commentary & Conclusions

- **Decision: Closed. Clean negative with excellent mechanistic analysis.** All three pillars of the hypothesis falsified.
- **No regularization benefit:** val_avg delta (+0.32) is well within seed noise (~4–5 units). Per-split redistribution exists (rc −10.9, cruise +14.6) but net-cancels. Implicit-ensembling claim does not survive at one-seed precision.
- **Wall-clock argument falsified:** per-epoch time *increased* by +3% (113.2 vs 110.0 s/epoch). The per-sample mask construction and division cost across 10 residual branches per step exceeds the autograd savings on n_layers=5 × n_hidden=128. DropPath wall-clock benefits show up at ViT scale (12+ layers, 384+ hidden), not here.
- **best_epoch moved earlier (16 → 12):** opposite direction from the predicted later-best-epoch signature. Adding noise to an undertrained model finds the optimum sooner and then plateaus — the opposite of "regularization extending useful training horizon".
- **Joint with #742 (per-activation dropout):** two negative results in a row with the same mechanistic root cause — the model is undertrained, not overfit. Standard transformer regularizers do not pay rent in this regime.
- **Student's diagnosis is correct:** the dominant constraint is wall-clock, not regularization. Their #1 follow-up suggestion (`torch.compile`) and the broader throughput direction are exactly right.
- **Nezuko reassigned** to vectorize `add_derived_features` (#923) — the per-sample CPU-sync Python loop identified by askeladd in #848 as the throughput bottleneck. This is an exact, contained, deterministic optimization that should free 5–15% wall-clock.

---

## 2026-04-29 00:14 — PR #850: Lower surf_weight sweep {3, 5, 7} — sent back for sw=3 + Huber stack

- **Branch:** `willowpai2e5-edward/lower-surf-weight` (sent back; result is internally clean but compared against pre-Huber baseline)
- **W&B runs:** `2sv6lptb` (sw=3), `ge7sjn6i` (sw=5), `rnhf5mmx` (sw=7) — group `lower-surf-weight`. **All three runs used `huber_delta=None` (MSE; pre-#739 merge code).**
- **Hypothesis:** Lowering surf_weight (counter to refuted #734 going up) lets the volume residuals contribute more spatial context, which the Transolver attention propagates back to surface predictions.

### Results (against BF16 baseline #811 val_avg=127.40, BEFORE Huber merged)

| sw | val_avg/mae_surf_p | test_avg/mae_surf_p | best epoch |
|----|--------------------:|--------------------:|-----------:|
| 10 (baseline) | 127.402 | 116.211 | 17 |
| **3** | **124.053 (-2.6%)** | **112.563 (-3.1%)** | 13 |
| 5 | 125.837 (-1.2%) | 115.176 (-0.9%) | 17 (still descending) |
| 7 | 142.777 (+12.0%) | 133.367 (+14.7%) | 17 (plateau) |

### Per-channel diagnostic (val avg)

| sw | surf_p ↓ | surf_Ux | surf_Uy | vol_p ↓ |
|----|---------:|--------:|--------:|--------:|
| 3 | **124.05** | 2.41 | 0.89 | **111.31** |
| 7 | 142.78 | **2.00** | **0.85** | 136.10 |

### Commentary & Conclusions

- **Decision: Sent back for re-run on Huber baseline.** The mechanism is real and partially complementary to Huber — needs a single decisive sw=3 + Huber run.
- **Mechanism validated within sweep:**
  1. Lower sw → substantial improvement in vol_p (sw=3: 111.31 vs sw=7: 136.10, -22%). Volume signal is informative for surface prediction; weakening it hurts surface predictions too.
  2. Counter-intuitive trade: surf_p improves at sw=3 but surf_Ux/Uy degrade. Pressure has fat-tailed magnitudes (high-Re outliers); strong surface emphasis amplifies gradient noise on `p` more than on velocity.
- **Stale baseline issue:** all three runs predate the Huber merge (sw=3 created at 22:25, Huber merged at 23:40). Current best is now 110.594, so sw=3's 124.05 looks worse than baseline. But Huber and sw-lowering attack different mechanisms: Huber caps high-Re gradient contribution, low-sw boosts volume informativeness. They MAY stack.
- **Send-back instructions:** Single re-run with `--surf_weight 3.0` on rebased branch (Huber δ=1.0 is now default). If beats 110.594 → merge as new baseline. If lands 105-115 → marginal. If >115 → Huber already captured this lever.
- **Future PR (per-channel surface weights):** student suggested `surf_weight_p=3, surf_weight_uv=10` to keep velocity accuracy while gaining pressure improvement — clever, assigned to frieren as #943.

---

## 2026-04-29 01:55 — PR #850: Lower surf_weight 10→3 on Huber+BF16 stack — **MERGED** (new best)

- **Branch:** `willowpai2e5-edward/lower-surf-weight` (merged)
- **W&B run:** `6rh7dzkx` — group `lower-surf-weight-huber-stack`
- **Hypothesis:** Lower `surf_weight` from 10 to 3 on the Huber+BF16 baseline. The mechanism: lower surface weight forces more gradient signal through volume residuals; Transolver cross-token attention propagates that volume information back to surface predictions, exploiting the global pressure-Poisson relationship.

### Results vs Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | Huber sw=10 | sw=3 + Huber | Δ val | test sw=10 | test sw=3 | Δ test |
|-------|------------:|-------------:|------:|-----------:|----------:|-------:|
| `single_in_dist`    | 130.87 | **120.51** | −7.92% | 124.544 | **102.846** | −17.4% |
| `geom_camber_rc`    | 115.14 | **107.95** | −6.24% | 99.385  | **94.352**  | −5.1%  |
| `geom_camber_cruise`| 92.61  | **82.16**  | −11.3% | 80.195  | **70.128**  | −12.5% |
| `re_rand`           | 103.76 | **95.64**  | −7.83% | 101.070 | **92.346**  | −8.6%  |
| **avg**             | **110.594** | **101.563** | **−8.17%** | **101.299** | **89.918** | **−11.24%** |

Best epoch = 17/17 (val still descending at 30-min timeout cutoff).

### Commentary & Conclusions

- **Decisive win — all 4 splits improved on both val and test.** Test improvements larger than val (e.g., sid −17.4% test vs −7.9% val), consistent with better generalization rather than noise.
- **Mechanisms stack orthogonally:** Huber caps gradient *magnitude* on high-Re outliers; lower sw rebalances surface vs volume *weight* in the loss. These are independent levers → compounding gains as predicted.
- **val curve was still descending at epoch 17** (trajectory: 230.5 → 165.0 → 147.8 → 128.2 → 124.9 → 143.2 → 110.7 → 101.56). Suggests further improvement possible with more budget or even lower sw.
- **PR diff was initially empty** (CLI-only run; student didn't update default). Sent back for one-line Config change (`surf_weight: float = 3.0`). Merged cleanly on resubmit.
- **Fifth compounding win stacked:** distance-features → warmup+cosine → BF16 → Huber → sw=3.
- **Follow-up: #953 (edward)** — sweep sw ∈ {0.5, 1.0, 2.0} to find floor below sw=3.

---

## 2026-04-29 02:25 — PR #885: Sweep Huber loss delta {0.3, 0.5, 1.0, 2.0} — sent back for rebase + stacking test

- **Branch:** `askeladd/huber-delta-sweep` (sent back; conflicts with merged Huber #739 and stale sw=10)
- **W&B runs:** `3yiixbyg` (δ=0.3), `vr6g2rxa` (δ=0.5), `295hulp0` (δ=1.0), `ki36m2z6` (δ=2.0) — group `huber-delta-sweep`
- **Hypothesis:** Sweep δ ∈ {0.3, 0.5, 1.0, 2.0} on the BF16 baseline to find the optimal Huber transition threshold. Smaller δ should help the heavy-tailed normalized residual distribution (high-Re outliers).

### Results vs pre-Huber MSE baseline (val_avg=127.402, test_avg=116.211, sw=10) — askeladd's sweep

| delta | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p |
|------:|:---------:|--------------------:|--------------------:|
| 2.0 | 17 | 113.804 | 102.353 |
| 1.0 | 17 | 115.386 | 104.471 |
| 0.5 | 16 | 107.271 | 97.437 |
| **0.3** | **16** | **97.963 (−23.1%)** | **87.785 (−24.5%)** |

### Compared to current merged baseline (#850, sw=3 + Huber δ=1.0: val_avg=101.563, test_avg=89.918)

- **δ=0.3 alone (sw=10) BEATS current best (sw=3 + δ=1.0)** by −3.5% val / −2.4% test in absolute terms.
- The δ lever appears stronger than the sw lever (δ sweep gave −17.4% val improvement at fixed sw=10, vs sw=3 → sw=10 giving −8.2% improvement at fixed δ=1.0).

### Commentary & Conclusions

- **Trend is monotone-with-noise:** δ {2.0 → 1.0} flat (within noise), but {1.0 → 0.5 → 0.3} clear monotone descent. Bottom not yet found.
- **Mechanism confirmed:** smaller δ pulls more outlier samples into the L1 regime, where gradient magnitudes are bounded. Most consistent gains are on `val_single_in_dist` and `val_geom_camber_rc` — splits with the heaviest-tailed residual distributions.
- **PR is in conflict with current advisor branch:** askeladd's branch was created before #739 (Huber) merged. Their diff re-adds Huber code that's now on main; combined with #850's sw=3 default change, the merge state is dirty.
- **Stacking question is open:** δ=0.3 (loss-shape lever) and sw=3 (loss-balance lever) attack different mechanisms, so they may stack additively. But both ultimately address outlier-driven instability — partial overlap is plausible.
- **Decision: Send back** for rebase + 2 decisive runs:
  1. `δ=0.3 + sw=3` — test stacking with current baseline.
  2. `δ=0.1 + sw=3` — continue the monotone trend test (askeladd suggested {0.1, 0.2}).
- **If δ=0.3 + sw=3 wins:** merge as new baseline. **If δ=0.1 also wins:** prefer whichever is lower.

---

## 2026-04-29 02:45 — PR #896: Per-sample y-normalization on Huber+sw=3 — closed (redundant with existing baseline)

- **Branch:** `alphonse/per-sample-y-normalization` (closed; ran clean on rebased Huber+sw=3 stack)
- **W&B run:** `ngaailh7` — `per-sample-y-norm-huber-sw3-clip1`
- **Hypothesis:** Normalize each sample's residual by per-sample sigma_per before loss, equalizing Re-regime contributions. This is a TARGET-SPACE fix; Huber is a LOSS-SPACE fix. Hypothesis: they're complementary.

### Results vs current baseline (#850: val_avg=101.563, test_avg=89.918)

| Split | baseline #850 | per-sample-norm + Huber + sw=3 | Δ val | test baseline | test + norm | Δ test |
|-------|-------------:|-------------------------------:|------:|--------------:|------------:|-------:|
| `single_in_dist`    | 120.507 | 127.759 | **+6.0%** | 102.846 | 116.274 | **+13.1%** |
| `geom_camber_rc`    | 107.951 | 120.592 | **+11.7%** | 94.352 | 110.080 | **+16.7%** |
| `geom_camber_cruise`| 82.156  | **63.843** | −22.3% | 70.128 | **53.730** | −23.4% |
| `re_rand`           | 95.636  | **88.545** | −7.4% | 92.346 | **82.658** | −10.5% |
| **avg**             | **101.563** | **100.185** | **−1.36%** | **89.918** | **90.686** | **+0.85%** |

### Commentary & Conclusions

- **Redistributive, not Pareto-improving.** Val/test directions disagree. The per-split flip pattern is exact: low-Re splits (cruise, re_rand) win massively (−22%, −10% test), high-Re splits (sid, rc) regress equally (+13%, +17% test). This confirms per-sample-norm and Huber+sw=3 attack the SAME underlying Re-imbalance problem from different sides, largely substituting for each other.
- **Val avg marginally better (−1.36%) but within seed noise.** Test avg slightly worse (+0.85%) — the paper-facing metric goes the wrong direction.
- **sigma_per stats confirm mechanism:** mean=0.336, min=0.082, max=0.700 in normalized space. Low-Re samples have ~4× smaller sigma than high-Re — confirming the imbalance the normalizer attacks. But Huber already caps high-Re gradient contribution, leaving per-sample-norm with little independent room.
- **Decision: Closed.** The mechanism works but the lever is saturated by the existing baseline. The earlier appearance of a +17%/+19% win (#896 vs #811) was a stale-baseline artifact.
- **Mechanism insight:** Target-space (sigma_per) and loss-space (Huber) approaches to Re-imbalance appear to be nearly equivalent substitutes when both are properly tuned. A hybrid approach (smaller Huber δ AND sigma-rescaling) might still add value if they're not fully equivalent — this is a Wave 3 question.
- **Follow-up assigned: #980 (alphonse)** — boundary-layer-weighted volume loss, a mechanistically distinct lever using dist_to_surface feature.

---

## 2026-04-29 03:00 — PR #923: Vectorize `add_derived_features` — merged (neutral throughput, clean code)

- **Branch:** `willowpai2e5-nezuko/vectorize-add-derived-features` (merged)
- **W&B runs:** A/B runs under group `vectorize-data-prep`; also tested literal B×N×N proposal
- **Hypothesis:** Per-sample CPU `.item()` syncs in `add_derived_features` are the throughput bottleneck. Removing them should give 5-15% wall-clock improvement and unblock batch-size scaling.

### Results

| Metric | A: per-sample loop | B: vectorized | Δ |
|---|---|---|---|
| `epoch_data_prep_ms_mean` | 26.20 ms | 25.78 ms | −1.6% (noise) |
| `epoch_time_s` | 109.53 s | 110.04 s | +0.5% (noise) |
| Total epochs in 30-min budget | 16 | 17 | +1 (likely noise) |
| Max abs numerical diff | — | 0.0 | exact |

Also tested literal B×N×N batched approach: **1.75× slower** (46 ms vs 26 ms per step) due to 50× more pairwise distance computations (N=242K vs s_b≈5K surface nodes).

### Commentary & Conclusions

- **Hypothesis refuted.** GPU pairwise compute dominates (≈22ms/26ms = 85%), not CPU syncs (≈4ms).
- **Full elimination of data_prep would save only ~1.4% of epoch_time** — within measurement noise.
- **Actual bottleneck: model forward+backward = ~91% of epoch time** (~100s/110s).
- **Merged for architectural cleanliness:** bit-exact implementation, removes `.item()` CPU syncs, cleaner code even if not faster currently. Future torch.compile benefits from sync-free data prep.
- **Bottleneck map: model FLOPs dominate.** torch.compile is the correct next throughput attack.
- **Follow-up assigned: #986 (nezuko)** — torch.compile(dynamic=True) targeting 1.2-1.5× model speedup.

---

## 2026-04-29 01:15 — PR #915: Mask padded nodes in PhysicsAttention slice aggregation — closed (mixed result)

- **Branch:** `willowpai2e5-frieren/physics-attention-padding-mask` (closed)
- **W&B run:** `msywsg7o`
- **Hypothesis:** Padded zero-vector nodes contaminate slice tokens via unmasked softmax in PhysicsAttention. Masking them out (post-softmax zero) should improve predictions, especially on cruise geometries with variable mesh sizes / high padding ratio.

### Results vs Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | val/mae_surf_p (base) | val/mae_surf_p (mask) | Δ val | test/mae_surf_p (base) | test/mae_surf_p (mask) | Δ test |
|-------|----------------------:|-----------------------:|------:|------------------------:|------------------------:|-------:|
| `single_in_dist`    | 130.87 | ~130.2 | ~−0.5% | 124.544 | ~127.7 | ~+2.5% |
| `geom_camber_rc`    | 115.14 | ~118.6 | ~+3.0% | 99.385  | ~130.0 | **+30.8%** |
| `geom_camber_cruise`| 92.61  | ~79.6  | **−14.1%** | 80.195  | ~68.7  | **−14.3%** |
| `re_rand`           | 103.76 | ~104.1 | ~+0.3% | 101.070 | ~102.5 | ~+1.4% |
| **avg**             | **110.594** | **~108.1** | **−0.6%** (noise) | **101.299** | **~107.2** | **+3.3%** |

(Approximate per-split numbers reconstructed from PR comment; W&B run `msywsg7o` confirmed.)

### Commentary & Conclusions

- **Mechanism confirmed on cruise** (−14.3% test, −14.1% val) exactly as predicted. Cruise has the most variable mesh sizes and highest padding ratio → most contamination from padded zero-nodes in the slice-softmax.
- **RC split regressed sharply** (+30.8% test). RC geometries (raceCar tandem, M=6-8) have denser, more uniform meshes → lower padding ratio → the hard binary post-softmax mask zeroes real attention weight, disrupting tandem-wake slice tokens.
- **Net aggregate:** val_avg −0.6% (within ~5-unit seed noise), test_avg +3.3% worse. The cruise gain and rc regression approximately cancel; aggregate is negative.
- **Why the binary mask fails on rc:** The fix patches symptom (padded nodes contaminating slices) but breaks mechanism (slice assignment flexibility) on dense-mesh geometries. A soft learnable gate (sigmoid(MLP(x))) would preserve attention on dense meshes while suppressing true padding — this is a Wave 3 idea.
- **Decision: Closed.** Mechanism insight is valuable but the binary mask is not a net improvement. Redirecting frieren to per-channel surface loss weighting (#943).

---

## 2026-04-29 01:20 — PR #896: Per-sample y-normalization — sent back for rebase on Huber baseline

- **Branch:** `willowpai2e5-alphonse/per-sample-y-normalization` (sent back, merge conflict with #739)
- **W&B run:** `5ihd38bk` (winning run: `per-sample-y-norm-clip1`)
- **Hypothesis:** Normalize each sample's residual by its per-sample standard deviation (sigma_per) before computing the loss, equalizing Re-regime contributions. High-Re samples (large sigma_per) are effectively down-weighted; low-Re samples (small sigma_per) get amplified. This is a target-space fix vs Huber's loss-space fix — potentially complementary.

### Results vs current Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | Huber baseline | Per-sample-norm (MSE) | Δ vs Huber |
|-------|---------------:|----------------------:|------------|
| `single_in_dist`    | 130.87 | **131.105** | +0.2% |
| `geom_camber_rc`    | 115.14 | **128.262** | +11.4% |
| `geom_camber_cruise`| 92.61  | **72.292**  | **−21.9%** |
| `re_rand`           | 103.76 | **90.152**  | **−13.1%** |
| **val_avg**         | **110.594** | **105.453** | **−4.7%** |

| Split | Huber baseline | Per-sample-norm (MSE) | Δ vs Huber |
|-------|---------------:|----------------------:|------------|
| `test_single_in_dist`    | 124.544 | **117.075** | −6.0% |
| `test_geom_camber_rc`    | 99.385  | **111.894** | **+12.6%** |
| `test_geom_camber_cruise`| 80.195  | **60.451**  | **−24.6%** |
| `test_re_rand`           | 101.070 | **85.834**  | **−15.1%** |
| **test_avg**             | **101.299** | **93.814** | **−7.4%** |

Note: alphonse's PR compared against the pre-Huber #811 baseline (127.402), not the current Huber best. Even vs current Huber baseline, this is still a clear winner on average.

### Commentary & Conclusions

- **Per-sample-norm is a decisive win on cruise and re_rand** (both OOD Re splits). Cruise −24.6% test, re_rand −15.1% test — the Re-normalization is directly attacking the generalization failure mode.
- **RC regression at +12.6% test** is concerning. RC has dense meshes, tighter Re range — sigma_per may be less variable for RC samples, so per-sample-norm doesn't help and possibly adds noise.
- **Average win is clear** (−4.7% val, −7.4% test) despite the RC regression, because cruise and re_rand dominate by sheer magnitude.
- **Important:** ran WITHOUT Huber loss (huber_delta=None). The per-sample-norm mechanism supersedes Huber for Re-imbalance but may stack with it. Merge conflict with Huber code (different edit points in the loss computation) — sent back for rebase.
- **Grad_clip=1.0** added by student (undocumented in original PR instructions) — correct decision for stability with large per-sample weight variation.
- **Decision: Sent back for rebase on Huber baseline.** When rebased, instruct alphonse to stack both: the sq_err normalization by sigma_per should be applied to the huber_err tensor (not a raw sq_err) — i.e., compute `huber_err = F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)` then divide by `sigma_per.unsqueeze(1)` before weighting by surf/vol masks.

---

## 2026-04-29 03:15 — PR #810: EMA post-warmup-init + decay sweep — sent back for rebase + verification on sw=3 baseline

- **Branch:** `willowpai2e5-thorfinn/ema-model-checkpoint` (sent back, merge state DIRTY)
- **W&B runs:** `n1rv11qt` (d=0.995 winner), `a9v9qwbi` (d=0.999)
- **Hypothesis:** EMA over model weights with deferred init (after warmup_epochs=5 / 1875 steps) reduces checkpoint variance on noisy small-dataset training, with bigger gains on the OOD splits. Dual-flavor checkpointing tracks both live and ema; selects best by val_avg over the ema model after init.

### Results vs sw=10 stale baseline (val_avg=110.594, test_avg=101.299)

| Split | sw=10+Huber baseline | EMA d=0.995 (winner) | EMA d=0.999 |
|-------|---------------------:|---------------------:|------------:|
| `val_single_in_dist`     | 130.87 | **104.52** (-20.1%) | 110.43 |
| `val_geom_camber_rc`     | 115.14 | **97.02** (-15.7%) | 102.17 |
| `val_geom_camber_cruise` | 92.61  | **75.18** (-18.8%) | 80.43 |
| `val_re_rand`            | 103.76 | **82.77** (-19.8%) | 88.79 |
| **val_avg**              | **110.594** | **89.872 (-18.7%)** | **95.457** |
| **test_avg**             | **101.299** | **79.254 (-21.8%)** | **85.851** |

Vs current sw=3+Huber baseline (val_avg=101.563, test_avg=89.918): d=0.995 wins by **−11.5% val / −11.9% test** on stale code. All 4 splits improved.

### Commentary & Conclusions

- **Strong winner on stale baseline.** d=0.995 beats every split. d=0.999 also wins but by less — the higher decay's effective memory window (~10k steps) overshoots the post-warmup descent regime. d=0.995's effective window (~2k steps) tracks the steepest improvement phase.
- **Clean implementation per prior advisor feedback:** deferred init after warmup, dual-flavor checkpointing, best_flavor=ema selected from epoch 6 onward.
- **PR is in CONFLICTING state.** 902 additions / 99 deletions across 4 files. Created before #850 (sw=3 default) and #923 (vectorized data prep) merged. Merge conflicts in train.py loss block + Config + add_derived_features.
- **Decision: Sent back for rebase + decisive verification run on sw=3 baseline.** EMA mechanism is orthogonal to surf_weight, so the gain should hold — but a 902-line PR is too valuable to merge with conflicts. Asked thorfinn to rerun d=0.995 only on rebased code (skip d=0.999 to save GPU time). If rebased d=0.995 lands within ~1% of stale-baseline winning numbers (val ≈ 90, test ≈ 79), this becomes the **6th compounding win**.

---

## 2026-04-29 03:25 — PR #943: Per-channel surface loss weights — sent back for rebase + anchored sweep

- **Branch:** `willowpai2e5-frieren/per-channel-surf-weight` (sent back, merge state DIRTY)
- **W&B runs:** `u2tmgfwk` (Run 1: p=3, vel=10), `3qewws8e` (Run 2: p=20, vel=10)
- **Hypothesis:** Splitting surface loss into per-channel weights for pressure (p_surf_weight) vs velocity (vel_surf_weight) lets us steer pressure supervision independently from velocity, exploiting the fact that paper-facing metric is surface pressure only.

### Results vs stale Huber-only baseline (val_avg=110.594, test_avg=101.299)

| Split | Stale Huber baseline | Run 1 (p=3, vel=10) | Run 2 (p=20, vel=10) |
|-------|---------------------:|--------------------:|---------------------:|
| `val_single_in_dist`     | 130.87 | 155.954 | 140.201 |
| `val_geom_camber_rc`     | 115.14 | 117.384 | 116.203 |
| `val_geom_camber_cruise` |  92.61 |  82.827 |  79.022 |
| `val_re_rand`            | 103.76 |  99.380 |  94.747 |
| **val_avg**              | **110.594** | **113.886** | **107.543** |
| **test_avg**             | **101.299** | **99.866** | **98.422** |

### Vs CURRENT sw=3+Huber baseline (val_avg=101.563, test_avg=89.918) — paper-facing comparison

| Run | val_avg Δ | test_avg Δ | Verdict |
|-----|----------:|-----------:|---------|
| Run 1 (p=3, vel=10)  | **+12.1%** | **+11.1%** | clear regression |
| Run 2 (p=20, vel=10) | **+5.9%**  | **+9.5%**  | regression |

### Commentary & Conclusions

- **Both runs regress vs current baseline** because both raise vel_surf_weight from 3 (current) to 10 (stale code default). The velocity weight change confounds the per-channel mechanism.
- **Pressure boost mechanism may be real, but masked.** Run 2 cruise val=79.022 vs current sw=3 cruise val=82.16 → −3.8% on the most paper-facing OOD split. Could indicate pressure supervision boost helps OOD generalization through better boundary-layer fitting.
- **PR in CONFLICTING state.** Loss block in train.py was changed by #850 (surf_weight default 10→3), conflicting with frieren's per-channel rewrite.
- **Decision: Sent back for rebase + anchored 3-point sweep.** Asked frieren to: (a) rebase, (b) run control p=3/vel=3 (must reproduce 101.5 baseline), (c) sweep p_surf ∈ {10, 20} with vel_surf=3 fixed. This isolates the pressure boost mechanism cleanly. If Run B (p=10) or Run C (p=20) beats 101.563 → real per-channel win.

---

## 2026-04-29 03:40 — PR #953: Sweep surf_weight ∈ {0.5, 1.0, 2.0} — closed (sw lever exhausted)

- **Branch:** `willowpai2e5-edward/surf-weight-below-3-sweep` (closed, branch deleted)
- **W&B runs:** `uy9csp5x` (sw=0.5 winner), `arebv1r0` (sw=1.0), `8u2vzihj` (sw=2.0)
- **Hypothesis:** Lower surf_weight below 3 may continue the volume-driven pressure mechanism from #850. Val curve was still descending at sw=3 timeout — sw=1 or below might be the true floor.

### Results vs current sw=3 baseline (val_avg=101.563, test_avg=89.918)

| surf_weight | val_avg | Δ val | test_avg | Δ test | best_epoch | W&B |
|------------:|--------:|------:|---------:|-------:|-----------:|-----|
| 0.5 | **99.185** | **−2.34%** | 90.293 | +0.42% | 15/17 | uy9csp5x |
| 1.0 | 109.032 | +7.35% | 98.849 | +9.93% | 16/17 | arebv1r0 |
| 2.0 | 114.243 | +12.49% | 105.999 | +17.88% | 11/17 | 8u2vzihj |
| 3.0 (baseline) | 101.563 | — | 89.918 | — | 17/17 | 6rh7dzkx |

### Per-split test (current baseline → sw=0.5 winner)

| Split | Baseline | sw=0.5 | Δ |
|-------|---------:|-------:|--:|
| `single_in_dist` | 102.85 | 112.81 | **+9.96 worse** |
| `geom_camber_rc` | 94.35  | 102.39 | **+8.04 worse** |
| `geom_camber_cruise` | 70.13 | 60.44 | **−9.69 better** |
| `re_rand` | 92.35 | 85.53 | **−6.82 better** |
| **avg** | **89.92** | **90.29** | **+0.42% essentially tied** |

### Commentary & Conclusions

- **Test correctly evaluated at best checkpoint.** Verified train.py line 730 reloads `model_path` (best val checkpoint) before test eval. The +0.42% test result is real.
- **Val improvement does not transfer to test.** Despite val_avg dropping 2.34%, test_avg ties. Per-split test shows split-trade: 2 splits improve (cruise/re_rand by ~7-10%), 2 splits regress (sid/rc by ~8-10%).
- **Non-monotone sweep is a strong red flag.** sw=2 worse than sw=1 worse than sw=0.5 doesn't fit a simple "lower-is-better" or "higher-is-better" story. Suggests sw=0.5 is exploring a different local basin (volume-dominated regime) rather than smoothly extending the sw=3 mechanism.
- **Non-Pareto improvement.** The split-trade indicates sw=0.5 puts the model in a regime that helps smoother fields (cruise/re_rand) but hurts sharper-gradient ones (sid/rc). Not stackable safely.
- **Decision: Closed. sw lever exhausted.** sw=3 is stable and defensible as default. No multi-seed verification needed — the val gain doesn't transfer to test, which is the paper-facing decision metric.

### Reassignment

Edward → #1019 (loss-weighted hard-negative sampling). Mechanistically distinct from sw (per-sample EMA loss → resampling weights, not loss-magnitude scaling).

---

## 2026-04-29 04:10 — PR #885: Huber δ sweep + δ=0.1 stacking on sw=3 — MERGED (6th compounding win)

- **Branch:** `askeladd/huber-delta-sweep` (squash-merged into icml-appendix-willow-pai2e-r5)
- **W&B runs:** `nffbil1x` (δ=0.1 + sw=3 winner), `jul80d41` + `3leo4hv5` (δ=0.3 seeds, both at noise floor)
- **Hypothesis:** Huber δ caps gradient magnitude on heavy-tailed residuals. With sw=3 absorbing surface heavy-tail, the volume residual heavy tail (high-Re outliers) becomes the new noise source. Pushing δ from 1.0 → 0.1 stabilizes volume gradient regimes.

### Sweep on stale sw=10 baseline (askeladd's first run)

| δ | val_avg | test_avg | W&B |
|---|--------:|---------:|-----|
| 2.0 | 113.80 | 102.35 | ki36m2z6 |
| 1.0 | 115.39 | 104.47 | 295hulp0 |
| 0.5 | 107.27 | 97.44 | vr6g2rxa |
| 0.3 | 97.96 | 87.79 | 3yiixbyg |

Monotone trend: smaller δ → better val/test. Original δ=0.3 win on sw=10 was striking but tested against stale baseline.

### Stacking on current sw=3 baseline (post-rebase)

| variant | val_avg | test_avg | best_epoch | W&B |
|---------|--------:|---------:|-----------:|-----|
| **δ=0.1 + sw=3** ✅ | **96.866** | **87.348** | 16 | nffbil1x |
| δ=0.3 + sw=3 (run B) | 100.314 | 90.818 | 15 | jul80d41 |
| δ=0.3 + sw=3 (run A) | 101.880 | 89.874 | 16 | 3leo4hv5 |
| baseline (δ=1.0+sw=3) | 101.563 | 89.918 | 17 | 6rh7dzkx |

### Per-split test on δ=0.1 winner (vs current baseline → δ=0.1)

| Split | Baseline | δ=0.1 | Δ |
|-------|---------:|------:|--:|
| `single_in_dist` | 102.85 | 109.152 | **+6.13% worse** |
| `geom_camber_rc` | 94.35  | 107.290 | **+13.71% worse** |
| `geom_camber_cruise` | 70.13 | 53.250 | **−24.07% better** |
| `re_rand` | 92.35 | 79.700 | **−13.69% better** |
| **avg** | **89.92** | **87.35** | **−2.86% better** |

### Commentary & Conclusions

- **δ=0.1 + sw=3 is the new baseline (val=96.866, test=87.348).** Both metrics improve cleanly:
  val −4.62%, test −2.86% vs prior baseline 101.563/89.918.
- **δ=0.3 stacking with sw=3 collapsed to noise floor.** The δ=0.3 win on sw=10 (97.96 val) was
  effectively absorbed by lowering surf_weight: sw=3 already cures the surface-channel heavy-tail
  volatility that δ=0.3 was capping. The two mechanisms partially overlap.
- **δ=0.1 attacks a different residual.** At sw=3, the loss is volume-dominated (vol_mask >> surf_mask);
  the volume residual heavy tail (high-Re outliers in the bulk flow field) is the new noise source.
  δ=0.1 caps those gradients tightly, allowing the inlier majority to drive learning unobstructed.
- **Split-trade pattern reappears.** sid+rc regress, cruise+re_rand improve — same direction as sw=0.5
  (#953 closed). But δ=0.1 wins on aggregate while sw=0.5 only tied. This suggests a fundamental
  trade-off: smoother-field splits (cruise/re_rand) benefit from outlier-bounded gradients;
  sharp-gradient splits (sid/rc) need the outlier-driven signal. Future work needs a different
  architectural lever to break this trade (multi-resolution, per-split heads, or ensemble).
- **Decision: Merged as 6th compounding win.** New baseline: val=96.866, test=87.348.

### Five → Six compounding wins stacked

1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → val_avg=110.59
5. Lower surf_weight=3 (#850) → val_avg=101.56
6. **Huber δ=0.1 stacked on sw=3 (#885) → val_avg=96.87 / test_avg=87.35**

### Reassignment

Askeladd → δ-floor-below-0.1-sweep (next experiment). Trend not yet bottomed out at δ=0.1 per askeladd's own analysis. Need to find where the floor is.

---

## 2026-04-29 04:55 — PR #980: Boundary-layer-weighted volume loss — CLOSED

- **Branch:** `willowpai2e5-alphonse/boundary-layer-weighted-vol-loss` (closed, no merge)
- **W&B runs:** `vh3pgrtv` (focus=1.0, control), `pp9c3csm` (focus=2.0), `xyk0e9wn` (focus=5.0)
- **Hypothesis:** Reweighting volume-loss residuals by `exp(-focus * dist_to_surface_norm)` (focus=2.0 default) emphasizes near-surface bulk-flow nodes where pressure-Poisson gradients are strongest, restoring sid/rc supervision while preserving cruise/re_rand wins from δ=0.1.

### Results (alphonse's reported numbers, current baseline = val 96.866 / test 87.348)

| variant | val_avg | test_avg | val Δ% | test Δ% | W&B |
|---------|--------:|---------:|------:|------:|-----|
| focus=1.0 | 103.27 | 91.99 | +6.6% | +5.3% | vh3pgrtv |
| focus=2.0 | 100.74 | 89.41 | +4.0% | +2.4% | pp9c3csm |
| focus=5.0 | 93.88 | 88.21 | −3.1% | +1.0% | xyk0e9wn |

### Per-split delta @ focus=5.0 (best variant) vs current baseline

| Split | Baseline | focus=5.0 | Δ |
|-------|---------:|----------:|--:|
| `single_in_dist`    | 109.152 | ~98.7  | **−9.5% better** |
| `geom_camber_rc`    | 107.290 | ~95.5  | **−11.0% better** |
| `geom_camber_cruise`| 53.250  | ~67.2  | **+26.2% worse** |
| `re_rand`           | 79.700  | ~91.4  | **+14.7% worse** |

### Commentary & Conclusions

- **Inverse split-trade pattern.** Boundary-layer weighting helps sid/rc (sharp-gradient splits) by emphasizing near-surface gradients — exactly where #885's δ=0.1 cure regressed. But it hurts cruise/re_rand (smooth-field splits) by underweighting their bulk-flow signal.
- **Net regression.** focus=5.0 is the only variant where val improves (−3.1%), but test ties (+1.0%) — and the per-split pattern is just inverted from #885. This is the same Pareto wall as #953 (sw=0.5): you can shift gain between split categories but not lift the aggregate.
- **focus=2.0 default is too gentle.** With focus=2.0, the weighting decays to ~14% at half-domain — most of the volume loss is still uniform. focus=5.0 (~0.7% at half-domain) is what actually changes behavior.
- **Vol-loss-shape lever exhausted as a same-axis cure.** Both shape-of-loss-on-volume-residuals knobs (δ via #885 and now exp-decay weighting via #980) live on the same trade-off curve.
- **Decision: Closed.** Need a different mechanism to break the split-trade — per-channel sigma normalization (#1045 alphonse), per-split heads, multi-resolution, or attention-level changes are more promising than further loss-shape tweaks.

### Reassignment

Alphonse → #1045 per-channel sigma normalization (huber_err / sigma_per[b, c]). Mechanistically distinct from boundary-layer weighting: targets pressure-vs-velocity supervision balance rather than near-surface-vs-bulk weighting. Tests whether #896's per-sample-y-norm regression on rc was caused by sigma being pressure-dominated.

---

## 2026-04-29 06:20 — PR #986: torch.compile(dynamic=True) — MERGED (7th compounding win, largest single jump)

- **Branch:** `willowpai2e5-nezuko/torch-compile-model-forward` (squash-merged into icml-appendix-willow-pai2e-r5)
- **W&B runs:** `up4t33m5` (compile-on + δ=0.1 verify, winner), `9q56e46a` (compile-off + δ=0.1 control), `c2zkwwnm` (Run A original, eager + δ=1.0), `g9j5w0bq` (Run B original, compile + δ=1.0)
- **Hypothesis:** torch.compile with dynamic=True (CUDA Graph Trees disabled — meshes are 74K-242K nodes, variable per-batch) gives 1.5-2× wall-clock speedup → more epochs in 30-min budget → better val. Throughput-only mechanism — orthogonal to all loss-shape and architecture levers.

### Final results (compile-on + δ=0.1 vs current baseline #885)

| Split | Baseline (#885) | Compile+δ=0.1 (this PR) | Δ val | Δ test |
|-------|----------------:|------------------------:|------:|-------:|
| `single_in_dist`    | val=119.405 / test=109.152 | val=83.20 / test=78.96  | **−30.3%** | **−27.7%** |
| `geom_camber_rc`    | val=116.812 / test=107.290 | val=80.07 / test=72.75  | **−31.5%** | **−32.2%** |
| `geom_camber_cruise`| val=65.983  / test=53.250  | val=46.47 / test=40.41  | **−29.6%** | **−24.1%** |
| `re_rand`           | val=85.265  / test=79.700  | val=64.84 / test=57.98  | **−23.9%** | **−27.3%** |
| **avg**             | **val=96.866 / test=87.348** | **val=68.65 / test=62.53** | **−29.1%** | **−28.4%** |

### Throughput

| Run | Steady-state s/epoch | Total epochs in 30 min | Peak VRAM | Compile fallbacks |
|-----|--------------------:|----------------------:|----------:|------------------:|
| Eager (control)  | 109.5 | 17 | 33.1 GB | n/a |
| **Compile (winner)** | **61.9** | **29** | **23.9 GB** | **0** |
| Speedup | **1.77×** | **+71% epochs** | **−28% memory** | clean |

### Commentary & Conclusions

- **Largest single jump in the programme.** Both metrics drop ~28% — uniform across all 4 splits, breaking the previous split-trade pattern (where one group always paid for the other's gain). With more training, BOTH groups benefit — confirming that the previous trades were budget-induced, not fundamental.
- **Mechanism orthogonality verified super-additively.** Each lever alone:
  - δ=0.1 (#885) val=96.87 (−4.6% vs prior baseline)
  - compile alone (Run B original on δ=1.0) val=82.61 (−18.7% vs δ=1.0 baseline)
  - Stacked: val=68.65 (−32.4% vs δ=1.0 baseline) — **slightly super-additive**
- **Val curve still descending steeply at epoch 29** (last 6 epochs: 79.25 → 79.73 → 70.54 → 72.62 → 68.65). The compile-extended budget is still budget-limited.
- **VRAM headroom freed (9.2 GB)** opens larger-batch and capacity-scaling directions.
- **Defaults flipped:** `use_compile: bool = True`, `compile_mode: str = "default"`. Compile + δ=0.1 are now both default for every future student.
- **Critical implementation choices:**
  - `mode="default"` (not `reduce-overhead`) — CUDA Graph Trees + variable-N is documented failure mode in PyTorch issue #128424.
  - `torch._dynamo.config.cache_size_limit = 64` (vs default 8) — prevents recompile storms on variable-N.
  - `torch._dynamo.config.assume_static_by_default = False` — symbolic N from first trace.
  - Checkpoint plumbing handles `_orig_mod` wrapper correctly.
- **Decision: Merged as 7th compounding win.** New baseline: val=68.646, test=62.526.

### Six → Seven compounding wins stacked

1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → val_avg=110.59
5. Lower surf_weight=3 (#850) → val_avg=101.56
6. Huber δ=0.1 stacked on sw=3 (#885) → val_avg=96.87 / test_avg=87.35
7. **torch.compile(dynamic=True) (#986) → val_avg=68.65 / test_avg=62.53**

### Reassignment

Nezuko → #1072 (larger-batch + linear-LR scaling). Mechanism distinct from compile (gradient quality, not throughput) — exploits VRAM headroom freed (9.2 GB) and tests Goyal et al. 2017 linear-LR-scaling rule. Sweep bs ∈ {4 control, 6, 8} with lr peak scaled linearly.

### Implications for in-flight experiments

- **fern #809 (schedule sized to budget)** — the budget changed from 17 to 29 epochs. fern's "epochs=14, warmup=2" instructions are obsolete. Will need to send back with new instructions sized to the 29-epoch regime when fern resubmits.
- **askeladd #1031 (δ-floor below 0.1)** — should now run on the compile baseline (29 epochs) where the curve still descends. Higher chance of seeing a δ-effect than at 17 epochs.
- **edward #1019 (loss-weighted hard-negative sampling)** — orthogonal mechanism; should still hold.
- **alphonse #1045 (per-channel sigma)** — refines #896's mechanism; orthogonal to compile.
- **frieren #943 (per-channel surf weight rebase)** — anchored sweep, orthogonal to compile.
- **thorfinn #810 (EMA rebase)** — orthogonal; EMA quality lever still applies.
- **tanjiro #745 (separate pressure head rebase)** — orthogonal; capacity-on-output question still open.

---

## 2026-04-29 06:45 — PR #1031: Huber δ floor sweep below 0.1 — SENT BACK FOR COMPILE VERIFY

- **Branch:** `willowpai2e5-askeladd/huber-delta-floor-below-0.1` (rebase pending)
- **W&B runs:** `du4fvvv8` (δ=0.01), `s34n4sa5` (δ=0.03), `x8acbc5e` (δ=0.05), `o6liykzm` (δ=0.1 seed2) — all on eager+δ=0.1 baseline (17 epochs)
- **Hypothesis:** Push Huber δ below 0.1 to find where the trend bottoms out. Three predictions: floor at 0.05–0.1, floor below 0.05, or training instability at extreme δ.

### Results (against single-seed #885 baseline val=96.87 / test=87.35)

| δ | val_avg | test_avg | best_epoch | W&B |
|---|--------:|---------:|-----------:|-----|
| 0.1 (seed2 same config) | **93.62** | **81.59** | 16 | o6liykzm |
| 0.05 | 96.81 | 87.88 | 16 | x8acbc5e |
| 0.03 | **91.96** | **80.91** | 16 | s34n4sa5 |
| 0.01 | 91.20 | 82.25 | 17 | du4fvvv8 |

### Two findings

**Finding 1 (BIG): seed variance is ~3.2pt val / 5.8pt test on δ=0.1.** The δ=0.1 seed2 run hit val=93.62 / test=81.59 vs #885's reported val=96.87 / test=87.35. Same config. This is the most important methodological insight from the sweep — past single-seed wins under 5pt test_avg are at the noise boundary. **Future merge decisions on small-margin wins now require a confirmation seed.**

**Finding 2: δ=0.03 wins on test_avg (80.91)** but only by 3.6pt vs δ=0.1 seed-mean (84.47) — within seed-variance band. δ=0.01 has best val (91.20) but worst sid regression (test_sid 95.26 → 108.27 from δ=0.03 → 0.01); the sweep is non-monotone past δ=0.03 due to under-fitting hard in-distribution shapes. δ=0.05 disappears into seed noise.

### Per-split test pattern (vs δ=0.1 seed2)

- **rc:** keeps winning bigger as δ shrinks (94.78 → 93.71 → 90.60 from δ=0.1 → 0.03 → 0.01) — geometry-OOD samples benefit from L1 outlier-capping.
- **cruise:** modest wins (54.48 → 58.21 → 52.32 — weakly non-monotone, settling on improvement).
- **sid:** monotone improvement to δ=0.03 (99.92 → 100.98 → 95.26), then sharp REVERSAL at δ=0.01 (108.27).
- **re_rand:** modest wins (77.16 → 76.45 → 77.80, essentially noise-tied).

### Decision: SEND BACK for compile verify

The single most important issue: **all runs are on the obsolete pre-#986 baseline (peak GPU 33.1 GB, 17-epoch budget).** Current baseline is now compile + δ=0.1 at val=68.65 / test=62.53 (29 epochs). δ=0.03's val=91.96 is 23pt WORSE than current baseline.

The δ-floor question must be re-asked at the new operating point: with 29-epoch budget and the curve still descending at termination, the loss-shape × budget interaction may yield a different optimum.

Sent back for ONE decisive verification: δ=0.03 + compile vs current baseline. If it beats current by >5pt test (past seed noise), merge as 8th compounding win. If <2pt or negative, close direction with clean conclusion.

### Reassignment

Askeladd remains on #1031 for the compile verification round.

---

## 2026-04-29 06:45 — PR #1019: Loss-weighted hard-negative sampling — SENT BACK FOR COMPILE VERIFY

- **Branch:** `willowpai2e5-edward/loss-weighted-hard-negative-sampling` (rebase pending)
- **W&B runs:** `awapotmi` (α=0.5 winner), `1avytlb3` (α=1.0), `lwv3eytx` (control) — all on eager+#850 baseline (17 epochs)
- **Hypothesis:** WeightedRandomSampler with weights ∝ per-sample EMA loss focuses optimization on samples the model is currently failing on. Mechanistically distinct from sw, per-sample-norm, or boundary-layer weighting.

### Results (against #850 baseline val=101.56 / test=89.92)

| Run | val_avg | test_avg | best_epoch | final_w_max | W&B |
|-----|--------:|---------:|-----------:|------------:|-----|
| Control (no HNS) | 111.81 | 102.61 | 15 | n/a | lwv3eytx |
| α=0.5, floor=0.1 | **97.59** | **88.98** | 17 | 5.85× | awapotmi |
| α=1.0, floor=0.1 | 101.09 | 94.11 | 18 | 23.91× | 1avytlb3 |

### Per-split val/test (α=0.5 vs #850 baseline)

| Split | val Δ | test Δ |
|-------|------:|-------:|
| `single_in_dist` | −6.61 | −0.25 |
| `geom_camber_rc` | −1.02 | **+11.01** |
| `geom_camber_cruise` | **−8.64** | **−8.70** |
| `re_rand` | +0.36 | **−5.81** |
| **avg** | **−3.97** | **−0.94** |

### Mechanism diagnostics

- α=0.5 concentrated max 5.85× on hardest sample; α=1.0 concentrated max 23.91× — α=1.0 is "noise amplification" mode that the PR explicitly warned about. Per-split test shows the regression: `test_re_rand` and `test_cruise` both worsen ~10pt vs Run B at α=1.0 — concrete evidence that α<1 is correct.
- `loss_ema_p95` monotonically decreased in all runs — the sampler IS doing what it should (concentrating on the tail, then the tail shrinks).
- Per-split anomaly: `test_geom_camber_rc` worsens +11pt at α=0.5. The training-data geometry (`P1: M=2-5, P3: M=9 + 5 specials, P2: M=6-8 held out`) is the most adversarial OOD split. Loss-weighted sampling may amplify rc training samples whose geometry interpolates poorly to the held-out test set. Edward's suggestion: per-domain loss EMA — strong follow-up if the mechanism stacks with compile.

### Decision: SEND BACK for compile verify

Same issue as #1031: runs are all on obsolete pre-#986 baseline. Run B's val=97.59 is 29pt worse than current compile baseline (val=68.65). The 4pt val / 0.94pt test win against #850 is also right at the seed-variance boundary (3.2pt val / 5.8pt test on δ=0.1 alone).

Sent back for ONE decisive verification: α=0.5 + compile vs current baseline. Mechanism is throughput-orthogonal (compile changes WHEN you visit samples, HNS changes WHICH samples) so the gain SHOULD stack additively. The 29-epoch budget also gives the per-sample loss EMA more time to settle (24 effective epochs of weighting vs 12 before). Both effects favor a stacked win.

### Reassignment

Edward remains on #1019 for the compile verification round.
